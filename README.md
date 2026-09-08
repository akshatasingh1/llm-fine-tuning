# LLM Fine-Tuning with LoRA / PEFT

[![CI](https://github.com/cerenaaa/llm-fine-tuning/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/llm-fine-tuning/actions)

Parameter-efficient fine-tuning (PEFT) workflow for domain adaptation using LoRA. Covers dataset preparation, training, evaluation, and serving — with practical guidance on when fine-tuning beats RAG and when it doesn't.

## When to fine-tune vs RAG

| Scenario | Recommendation |
|---|---|
| New knowledge / facts | RAG (cheaper, updatable) |
| Style / tone / format | Fine-tune |
| Domain vocabulary / jargon | Fine-tune |
| Reasoning over retrieved docs | RAG + fine-tune |
| Latency-critical, no retrieval step | Fine-tune |

## Approach

- **LoRA**: Low-Rank Adaptation — inserts small trainable rank-decomposition matrices into attention layers. Only ~0.1–1% of params are trained.
- **PEFT**: HuggingFace library managing adapter configs, merging, and saving.
- **Evaluation**: Perplexity, ROUGE-L, task-specific accuracy, and hallucination rate.

## Structure

```
llm-fine-tuning/
├── data/
│   ├── dataset_builder.py       # Instruction-tuning dataset construction
│   └── data_quality.py          # Dataset quality filters and dedup
├── training/
│   ├── lora_trainer.py          # LoRA config + HuggingFace Trainer wrapper
│   └── training_config.py       # Hyperparameter dataclass
├── evaluation/
│   └── eval_suite.py            # Perplexity, ROUGE, task accuracy, hallucination
├── finetune.py                  # Main training entry point
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python finetune.py --model_name microsoft/phi-2 --output_dir results/
```
