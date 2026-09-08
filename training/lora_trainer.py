"""
LoRA fine-tuning wrapper using HuggingFace PEFT + Transformers.
Handles model loading, LoRA injection, training, and adapter saving.
"""
from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                               TrainingArguments, Trainer, DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from datasets import Dataset
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


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
    fp16: bool = True
    lora: LoRAConfig = None

    def __post_init__(self):
        if self.lora is None:
            self.lora = LoRAConfig()
        if self.lora.target_modules is None:
            self.lora.target_modules = ["q_proj", "v_proj"]


class LoRAFinetuner:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if not PEFT_AVAILABLE:
            raise ImportError("Run: pip install transformers peft datasets torch")

        print(f"Loading {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            device_map="auto",
        )

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

    def _tokenize(self, examples: list[dict]) -> Dataset:
        def fmt(ex):
            text = f"### Instruction:\n{ex['instruction']}\n"
            if ex.get("input"):
                text += f"### Input:\n{ex['input']}\n"
            text += f"### Response:\n{ex['output']}"
            return text

        texts = [fmt(ex) for ex in examples]
        tokenized = self.tokenizer(
            texts, max_length=self.config.max_seq_length,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return Dataset.from_dict({k: v.tolist() for k, v in tokenized.items()})

    def train(self, train_examples: list[dict], eval_examples: list[dict] = None):
        if self.model is None:
            self.load_model()

        train_dataset = self._tokenize(train_examples)
        eval_dataset = self._tokenize(eval_examples) if eval_examples else None

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            logging_steps=10,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        print("Starting training...")
        trainer.train()
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"Saved LoRA adapter to {self.config.output_dir}")
        return trainer.state.log_history