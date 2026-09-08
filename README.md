# LLM Fine-Tuning with LoRA / PEFT

[![CI](https://github.com/akshatasingh1/llm-fine-tuning/actions/workflows/ci.yml/badge.svg)](https://github.com/akshatasingh1/llm-fine-tuning/actions)

Parameter-efficient fine-tuning (PEFT) with LoRA, end to end: dataset prep →
training with completion-only loss → **before/after evaluation on a held-out
test set**. The worked example is **text-to-SQL** on `b-mc2/sql-create-context`.

## Results

`Qwen/Qwen2.5-0.5B`, LoRA (r=16, α=32, on q/k/v/o), 4,000 train examples,
2 epochs (~4 min on a Colab T4), scored on a 500-example held-out test split.
Reproduce with `notebooks/train_sql.ipynb`.

| metric | base | fine-tuned | Δ |
|---|---|---|---|
| exact match (normalized) | 10.0% | **72.8%** | **+62.8 pp** |
| execution match | 84.8% | **93.0%** | +8.2 pp |
| valid SQL rate | 93.0% | 95.8% | +2.8 pp |
| LLM-judge mean (n=25) | 0.70 | **0.92** | +0.22 |

- **exact match** — normalized string equality with the gold query
- **execution match** — gold and predicted SQL run against an in-memory SQLite DB
  built from the schema and return the same rows (the standard text-to-SQL metric)
- **valid SQL rate** — fraction of predictions that parse and execute at all
- **LLM-judge** — Gemini grades semantic equivalence (correct=1 / partial=0.5 /
  wrong=0); small sample because the free tier is rate-limited

Full numbers in [`docs/benchmark.json`](docs/benchmark.json) /
[`docs/judge_result.json`](docs/judge_result.json), side-by-side generations in
[`docs/sample_generations.md`](docs/sample_generations.md), and the full account
in [`WRITEUP.md`](WRITEUP.md).

**What the model learned.** The base model already writes plausible SQL
(84.8% execution match), so the large exact-match gain is mostly *conventions*:
it learns to use the dataset's double-quoted, lower-cased string literals, stop
over-selecting columns, and ground column names in the schema instead of
inventing them (e.g. `championship_years__years_` → `championships__years_`).
Execution match — which ignores those cosmetic differences — still improves
8 points, so there is real accuracy gain on top of the formatting.

**Caveats.** Execution match runs on empty tables, so it cannot separate two
valid queries that both return nothing. `b-mc2/sql-create-context` also contains
templated near-duplicate questions, so held-out rows can be structurally similar
to training rows — this is single-table SQL, not Spider-level difficulty.

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

- **LoRA / QLoRA**: inserts small trainable rank-decomposition matrices into the
  attention projections (~1% of parameters trained). `--load_in_4bit` adds 4-bit
  NF4 base weights so a 7B model fits on a free 16 GB T4.
- **Completion-only loss**: prompt tokens are masked (`-100`), so the model is
  trained to produce the SQL, not to echo the instruction.
- **Honest evaluation**: the test split is sliced from the same fixed shuffle as
  training and never seen during training; the base model is scored on the exact
  same prompts. Three cheap metrics (exact / execution / valid) plus an optional
  **LLM-as-judge** pass that catches correct-but-differently-written queries.

## Structure

```
llm-fine-tuning/
├── data/
│   ├── sql_dataset.py           # text-to-SQL loader + prompt template
│   └── dataset_builder.py       # synthetic instruction-tuning demo data
├── training/
│   └── lora_trainer.py          # LoRA/QLoRA config, Trainer wrapper, generation helpers
├── evaluation/
│   ├── sql_metrics.py           # exact / execution / valid-SQL scoring
│   ├── llm_judge.py             # Claude-as-judge: correct / partial / wrong
│   └── eval_suite.py            # generic perplexity / ROUGE-L / EM
├── finetune.py                  # training entry point  (--task sql | synthetic)
├── benchmark.py                 # base vs fine-tuned on the held-out test set
├── inference.py                 # run the model on a question, or merge adapter -> base
├── tests/                       # pytest: metrics, judge parser, dataset, config
├── notebooks/train_sql.ipynb    # Colab runner (T4 GPU)
├── WRITEUP.md                   # the engineering narrative
└── .github/workflows/ci.yml     # pytest + dry-run smoke tests
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
| `--load_in_4bit` | off | QLoRA — 4-bit base weights (needs CUDA + `bitsandbytes`) |
| `--dry_run` | off | build the dataset and exit |

LoRA rank / alpha / target modules / learning rate live in `TrainConfig` and
`LoRAConfig` in [`training/lora_trainer.py`](training/lora_trainer.py).

## Inference

```bash
# one question
python inference.py --adapter_dir results/sql_lora \
    --schema "CREATE TABLE head (age INTEGER, name VARCHAR)" \
    --question "How many heads are older than 56?"

# merge the adapter into the base weights -> a plain model, no PEFT needed to load
python inference.py --adapter_dir results/sql_lora --merge --out results/sql_merged
```

## LLM-as-judge

`benchmark.py --judge` grades a sample of predictions with an LLM
(correct / partial / wrong), which credits queries that are right but written
differently from the gold. Uses Gemini (free tier) — get a key at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey) and
`export GEMINI_API_KEY=...`. `--judge_n` sets how many examples are sent
(default 100; free-tier flash is ~15 req/min so ~200 calls takes a few minutes).

```bash
python benchmark.py --base_model Qwen/Qwen2.5-0.5B --adapter_dir results/sql_lora \
    --n_test 500 --judge --judge_n 100
```

The judge model is set by `JUDGE_MODEL` in
[`evaluation/llm_judge.py`](evaluation/llm_judge.py) (any current Gemini flash model).
`benchmark.py` also writes `results/predictions.json` so the judge can be re-run
later without regenerating.

## Notes

- Execution match uses empty SQLite DBs (the dataset only ships schemas), so it
  cannot separate two valid queries with identical empty output — it does catch
  hallucinated columns, broken syntax, and wrong query shape.
- `bitsandbytes` is not in `requirements.txt` (GPU-only); the Colab notebook
  installs it. Requires `transformers >= 5.0` (v5 `TrainingArguments` API).
