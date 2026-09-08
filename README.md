# LLM Fine-Tuning with LoRA / PEFT

[![CI](https://github.com/<your-username>/llm-fine-tuning/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/llm-fine-tuning/actions)

Parameter-efficient fine-tuning (PEFT) with LoRA, end to end: dataset prep →
training with completion-only loss → **before/after evaluation on a held-out
test set**. The worked example is **text-to-SQL** on `b-mc2/sql-create-context`.

## Results

`Qwen/Qwen2.5-0.5B`, LoRA (r=16, q/k/v/o), 4k train examples, 2 epochs,
500-example held-out test split. Run `notebooks/train_sql.ipynb` on a Colab T4
to reproduce (~25 min).

| metric | base | fine-tuned | Δ |
|---|---|---|---|
| exact match (normalized) | _TBD_ | _TBD_ | _TBD_ |
| execution match | _TBD_ | _TBD_ | _TBD_ |
| valid SQL rate | _TBD_ | _TBD_ | _TBD_ |

<!-- paste results/benchmark.json numbers here after the Colab run; sample
     generations land in results/samples.md -->

- **exact match** — normalized string equality with the gold query
- **execution match** — gold and predicted SQL run against an in-memory SQLite DB
  built from the schema and return the same rows (the standard text-to-SQL metric)
- **valid SQL rate** — fraction of predictions that parse and execute at all

## When to fine-tune vs RAG

| Scenario | Recommendation |
|---|---|
| New knowledge / facts | RAG (cheaper, updatable) |
| Style / tone / format | Fine-tune |
| Domain vocabulary / jargon | Fine-tune |
| Reasoning over retrieved docs | RAG + fine-tune |
| Latency-critical, no retrieval step | Fine-tune |

Text-to-SQL is a fine-tuning task: the model must learn an output *format* and a
*schema-grounding behaviour*, not new facts.

## Approach

- **LoRA**: inserts small trainable rank-decomposition matrices into the
  attention projections. Here ~1% of parameters are trained.
- **Completion-only loss**: prompt tokens are masked (`-100`), so the model is
  trained to produce the SQL, not to echo the instruction.
- **Honest evaluation**: the test split is sliced from the same fixed shuffle as
  training and never seen during training; the base model is scored on the exact
  same prompts.

## Structure

```
llm-fine-tuning/
├── data/
│   ├── sql_dataset.py           # text-to-SQL loader + prompt template
│   └── dataset_builder.py       # synthetic instruction-tuning demo data
├── training/
│   └── lora_trainer.py          # LoRA config, Trainer wrapper, generation helpers
├── evaluation/
│   ├── sql_metrics.py           # exact / execution / valid-SQL scoring
│   └── eval_suite.py            # generic perplexity / ROUGE-L / EM
├── finetune.py                  # training entry point  (--task sql | synthetic)
├── benchmark.py                 # base vs fine-tuned on the held-out test set
├── notebooks/train_sql.ipynb    # Colab runner (T4 GPU)
└── .github/workflows/ci.yml     # import + metric + dry-run smoke tests
```

## Setup (local)

Local machine is for editing and dry-runs. **Training needs a CUDA GPU** — on
CPU even a 135M model is ~80 s/step. Use the Colab notebook for real runs.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS / Linux
pip install -r requirements.txt
```

Without a CUDA GPU, `pip` installs the CPU build of `torch` and the trainer
stays in fp32 automatically.

```bash
# dataset build only — no model download, no training
python finetune.py --task sql --dry_run --n_train 100
```

## Training (Colab)

1. Push this repo to GitHub.
2. Open `notebooks/train_sql.ipynb` in Colab → Runtime → **T4 GPU**.
3. Set `REPO_URL` in the config cell, Run all.
4. The notebook trains, benchmarks, prints the table + samples, and downloads
   `sql_lora_run.zip` (adapter + `benchmark.json` + `samples.md`).

Equivalent CLI:

```bash
python finetune.py --task sql --model_name Qwen/Qwen2.5-0.5B \
    --n_train 4000 --epochs 2 --batch_size 16 --output_dir results/sql_lora

python benchmark.py --base_model Qwen/Qwen2.5-0.5B \
    --adapter_dir results/sql_lora --n_test 500
```

### `finetune.py` options

| Flag | Default | Description |
|---|---|---|
| `--task` | `sql` | `sql` (real dataset) or `synthetic` (template demo) |
| `--model_name` | `Qwen/Qwen2.5-0.5B` | base model |
| `--n_train` / `--n_eval` / `--n_test` | 4000 / 200 / 500 | SQL split sizes |
| `--epochs` | `2` | training epochs |
| `--batch_size` | `8` | per-device batch size |
| `--dry_run` | off | build the dataset and exit |

LoRA rank / alpha / target modules / learning rate live in `TrainConfig` and
`LoRAConfig` in [`training/lora_trainer.py`](training/lora_trainer.py).

## Notes

- Execution match uses empty SQLite DBs (the dataset only ships schemas), so it
  cannot separate two valid queries with identical empty output — it does catch
  hallucinated columns, broken syntax, and wrong query shape.
- Requires `transformers >= 5.0` (v5 `TrainingArguments` API).
