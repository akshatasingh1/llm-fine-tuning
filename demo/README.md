# Demo — Text-to-SQL Space

A Gradio web UI for the fine-tuned model. Hosted free on HuggingFace Spaces
(ZeroGPU). The app loads the **merged** model (adapter baked into the base
weights) so nothing touches CUDA at startup — the GPU is only used per request.

## Deploy (one time)

Free HuggingFace account + a **write** token from
<https://huggingface.co/settings/tokens>, then `hf auth login`.

**1. Merge the adapter and push the model to the Hub** (~1 GB, one time):

```bash
python inference.py --adapter_dir results/sql_lora --merge --out /tmp/sql_merged
hf upload <HF_USERNAME>/qwen2.5-0.5b-sql /tmp/sql_merged . --repo-type=model
```

(`results/sql_lora` is wherever your trained adapter lives — e.g. the Colab zip.)

**2. Edit `demo/app.py`** — set `MODEL_ID` to `<HF_USERNAME>/qwen2.5-0.5b-sql`.

**3. Create the Space:** <https://huggingface.co/new-space> → SDK **Gradio**,
template **Blank**, hardware **ZeroGPU (free)**.

**4. Push the two files to the Space:**

```bash
hf upload <HF_USERNAME>/text-to-sql demo/app.py app.py --repo-type=space
hf upload <HF_USERNAME>/text-to-sql demo/requirements.txt requirements.txt --repo-type=space
```

Builds automatically (~3–5 min) → `https://huggingface.co/spaces/<HF_USERNAME>/text-to-sql`.

## Run locally

```bash
pip install -r demo/requirements.txt
MODEL_ID=/tmp/sql_merged python demo/app.py   # or MODEL_ID=<HF_USERNAME>/qwen2.5-0.5b-sql
```

Opens http://127.0.0.1:7860. Falls back to CPU when there's no GPU
(~5–15 s per query).
