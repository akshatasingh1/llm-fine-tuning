"""
Gradio demo for the text-to-SQL LoRA fine-tune. Runs on a free HF Spaces CPU:
loads base Qwen2.5-0.5B + the ~4 MB LoRA adapter from the Hub.

Set ADAPTER_REPO to your uploaded adapter repo id.
"""
import os
import re

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
# On the Space this is the Hub repo id; locally, set ADAPTER_DIR=path/to/adapter to test.
ADAPTER_REPO = os.environ.get("ADAPTER_DIR", "<HF_USERNAME>/qwen2.5-0.5b-sql-lora")

PROMPT_TEMPLATE = (
    "### Task:\n"
    "Write a single SQL query that answers the question, using only the "
    "tables and columns in the schema. Reply with SQL only.\n"
    "### Schema:\n{schema}\n"
    "### Question:\n{question}\n"
    "### SQL:\n"
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(model, ADAPTER_REPO)
model.eval()


def extract_sql(text: str) -> str:
    text = text.strip()
    fence = re.match(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    text = re.split(r"\n###\s", text)[0].strip()
    if ";" in text:
        text = text.split(";")[0].strip()
    return " ".join(text.split())


@torch.no_grad()
def to_sql(schema: str, question: str) -> str:
    if not schema.strip() or not question.strip():
        return "-- provide both a schema and a question"
    prompt = PROMPT_TEMPLATE.format(schema=schema.strip(), question=question.strip())
    enc = tokenizer(prompt, return_tensors="pt")
    out = model.generate(
        **enc, max_new_tokens=128, do_sample=False, num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    new = out[0][enc["input_ids"].shape[1]:]
    return extract_sql(tokenizer.decode(new, skip_special_tokens=True))


demo = gr.Interface(
    fn=to_sql,
    inputs=[
        gr.Textbox(label="Schema (CREATE TABLE ...)", lines=3,
                   value="CREATE TABLE head (name VARCHAR, age INTEGER, born_state VARCHAR)"),
        gr.Textbox(label="Question", value="How many heads are older than 56?"),
    ],
    outputs=gr.Code(label="SQL", language="sql"),
    title="Text-to-SQL — Qwen2.5-0.5B + LoRA",
    description="Fine-tuned on b-mc2/sql-create-context. Single-table queries.",
    examples=[
        ["CREATE TABLE head (name VARCHAR, age INTEGER)", "List the names of heads ordered by age."],
        ["CREATE TABLE table_name_37 (date_of_vacancy VARCHAR, team VARCHAR)",
         "What is Aberdeen team's date of vacancy?"],
        ["CREATE TABLE table_1 (pos INTEGER, dutch_cup VARCHAR, tier VARCHAR)",
         "Which Pos has a Dutch Cup of winner and a Tier larger than 1?"],
    ],
)

if __name__ == "__main__":
    demo.launch()
