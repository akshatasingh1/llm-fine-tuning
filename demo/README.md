# Demo — Text-to-SQL Space

A Gradio web UI for the fine-tuned model. Hosted free on HuggingFace Spaces.

## Deploy (one time)

You need a free HuggingFace account and a **write** token from
<https://huggingface.co/settings/tokens>.

**1. Push the adapter to the Hub** (~4 MB):

```bash
pip install huggingface_hub
huggingface-cli login          # paste the write token
python demo/upload_adapter.py --adapter_dir results/sql_lora \
    --repo <HF_USERNAME>/qwen2.5-0.5b-sql-lora
```

(`results/sql_lora` is wherever your trained adapter lives — e.g. the folder from
the Colab run's zip.)

**2. Edit `demo/app.py`** — set `ADAPTER_REPO` to `<HF_USERNAME>/qwen2.5-0.5b-sql-lora`.

**3. Create the Space:** <https://huggingface.co/new-space> → SDK **Gradio**,
hardware **CPU basic (free)**.

**4. Push the two files to the Space repo:**

```bash
git clone https://huggingface.co/spaces/<HF_USERNAME>/text-to-sql
cp demo/app.py demo/requirements.txt text-to-sql/
cd text-to-sql && git add -A && git commit -m "add app" && git push
```

The Space builds automatically (~2–3 min) and gives you a public URL like
`https://huggingface.co/spaces/<HF_USERNAME>/text-to-sql`.

## Run locally first

```bash
pip install -r demo/requirements.txt
python demo/app.py      # opens http://127.0.0.1:7860
```

CPU inference of the 0.5B model is ~5–15 s per query — fine for a demo.
