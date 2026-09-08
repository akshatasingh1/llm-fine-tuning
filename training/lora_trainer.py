"""
LoRA fine-tuning wrapper using HuggingFace PEFT + Transformers.
Handles model loading, LoRA injection, training, adapter saving, and inference.
"""
from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                               TrainingArguments, Trainer, DataCollatorForSeq2Seq)
    from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training,
                      TaskType, PeftModel)
    from datasets import Dataset
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

LABEL_IGNORE = -100  # HF convention: positions with this label are excluded from loss


def _dtype_and_device_map():
    """fp16 + device_map='auto' only make sense with a CUDA GPU."""
    if torch.cuda.is_available():
        return torch.float16, "auto"
    return torch.float32, None


def load_model_and_tokenizer(model_name: str, adapter_dir: str | None = None):
    """Load a base causal-LM (optionally with a LoRA adapter) ready for generation."""
    if not PEFT_AVAILABLE:
        raise ImportError("Run: pip install transformers peft datasets torch")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype, device_map = _dtype_and_device_map()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map=device_map
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 128,
    batch_size: int = 8,
    max_prompt_tokens: int = 1024,
) -> list[str]:
    """Greedy-decode completions for a list of prompts. Returns only the new text."""
    tokenizer.padding_side = "left"  # decoder-only models must left-pad for batched gen
    device = getattr(model, "device", None) or next(model.parameters()).device
    outputs: list[str] = []

    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True,
                truncation=True, max_length=max_prompt_tokens,
            ).to(device)
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    return outputs


@dataclass
class LoRAConfig:
    r: int = 16                      # LoRA rank
    lora_alpha: int = 32             # scaling = alpha / r
    target_modules: list = None      # which attention matrices to adapt
    lora_dropout: float = 0.05
    bias: str = "none"


@dataclass
class TrainConfig:
    model_name: str = "microsoft/phi-2"
    output_dir: str = "results/lora_model"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    fp16: bool = True          # only applied when a CUDA GPU is available
    load_in_4bit: bool = False  # QLoRA: 4-bit base weights (needs CUDA + bitsandbytes)
    lora: LoRAConfig = None

    def __post_init__(self):
        if self.lora is None:
            self.lora = LoRAConfig()
        if self.lora.target_modules is None:
            # attention projections common to Llama/Qwen/Mistral-style models
            self.lora.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class LoRAFinetuner:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    @property
    def use_4bit(self) -> bool:
        """QLoRA needs a CUDA GPU (bitsandbytes has no usable CPU path here)."""
        return bool(self.config.load_in_4bit and torch.cuda.is_available())

    @property
    def use_fp16(self) -> bool:
        """fp16 only on a CUDA GPU, and not under QLoRA (which computes in bf16)."""
        return bool(self.config.fp16 and torch.cuda.is_available() and not self.use_4bit)

    def load_model(self):
        if not PEFT_AVAILABLE:
            raise ImportError("Run: pip install transformers peft datasets torch")

        print(f"Loading {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.config.load_in_4bit and not self.use_4bit:
            print("load_in_4bit requested but no CUDA GPU — loading in full precision.")

        quant_cfg = None
        if self.use_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            dtype = torch.bfloat16
        else:
            dtype = torch.float16 if self.use_fp16 else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quant_cfg,
        )
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
        )

        self.model = get_peft_model(self.model, lora_cfg)
        trainable, total = self.model.get_nb_trainable_parameters()
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        return self

    @staticmethod
    def _to_prompt_completion(ex: dict) -> tuple[str, str]:
        """Accept either {'prompt','completion'} or Alpaca {'instruction','input','output'}."""
        if "prompt" in ex and "completion" in ex:
            return ex["prompt"], ex["completion"]
        prompt = f"### Instruction:\n{ex['instruction']}\n"
        if ex.get("input"):
            prompt += f"### Input:\n{ex['input']}\n"
        prompt += "### Response:\n"
        return prompt, ex["output"]

    def _tokenize(self, examples: list[dict]) -> Dataset:
        """
        Tokenize with completion-only loss: prompt tokens get label -100 so the
        model is trained only to produce the answer, not to echo the instruction.
        Rows are variable-length here; DataCollatorForSeq2Seq pads each batch
        (labels padded with -100).
        """
        max_len = self.config.max_seq_length
        eos_id = self.tokenizer.eos_token_id
        rows = {"input_ids": [], "attention_mask": [], "labels": []}

        for ex in examples:
            prompt, completion = self._to_prompt_completion(ex)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = self.tokenizer(completion, add_special_tokens=False)["input_ids"]
            if eos_id is not None:
                completion_ids = completion_ids + [eos_id]

            input_ids = (prompt_ids + completion_ids)[:max_len]
            labels = ([LABEL_IGNORE] * len(prompt_ids) + completion_ids)[:max_len]

            rows["input_ids"].append(input_ids)
            rows["attention_mask"].append([1] * len(input_ids))
            rows["labels"].append(labels)

        return Dataset.from_dict(rows)

    def train(self, train_examples: list[dict], eval_examples: list[dict] = None):
        if self.model is None:
            self.load_model()

        train_dataset = self._tokenize(train_examples)
        eval_dataset = self._tokenize(eval_examples) if eval_examples else None

        # transformers 5.x dropped warmup_ratio; derive warmup_steps from the schedule.
        steps_per_epoch = max(
            1,
            len(train_dataset)
            // (self.config.batch_size * self.config.gradient_accumulation_steps),
        )
        total_steps = steps_per_epoch * self.config.num_epochs
        warmup_steps = int(self.config.warmup_ratio * total_steps)

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.use_fp16,
            bf16=self.use_4bit,
            warmup_steps=warmup_steps,
            lr_scheduler_type=self.config.lr_scheduler,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            logging_steps=10,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer, padding="longest", label_pad_token_id=LABEL_IGNORE,
            ),
        )

        print("Starting training...")
        trainer.train()
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"Saved LoRA adapter to {self.config.output_dir}")
        return trainer.state.log_history