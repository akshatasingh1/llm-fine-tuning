"""
Run the fine-tuned text-to-SQL model, or merge the adapter into a standalone model.

Generate:
    python inference.py --adapter_dir results/sql_lora \
        --schema "CREATE TABLE head (age INTEGER, name VARCHAR)" \
        --question "How many heads are older than 56?"

Read questions from stdin (one per line; schema stays fixed):
    echo "How many rows are there?" | python inference.py --adapter_dir results/sql_lora \
        --schema "CREATE TABLE t (id INT)"

Merge adapter -> base and save a plain model (no PEFT needed to load it):
    python inference.py --adapter_dir results/sql_lora --merge --out results/sql_merged
"""
import argparse
import sys

from data.sql_dataset import build_prompt
from evaluation.sql_metrics import extract_sql
from training.lora_trainer import load_model_and_tokenizer, generate_batch


def merge(base_model: str, adapter_dir: str, out: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Merging {adapter_dir} into {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out)
    print(f"Saved standalone model to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--adapter_dir", default="results/sql_lora")
    ap.add_argument("--schema", help="CREATE TABLE statement(s)")
    ap.add_argument("--question", help="one question; omit to read questions from stdin")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--merge", action="store_true", help="merge adapter into base and exit")
    ap.add_argument("--out", default="results/sql_merged", help="output dir for --merge")
    args = ap.parse_args()

    if args.merge:
        merge(args.base_model, args.adapter_dir, args.out)
        return

    if not args.schema:
        ap.error("--schema is required for generation")

    questions = [args.question] if args.question else [
        line.strip() for line in sys.stdin if line.strip()
    ]
    if not questions:
        ap.error("no question given (pass --question or pipe questions on stdin)")

    model, tok = load_model_and_tokenizer(args.base_model, adapter_dir=args.adapter_dir)
    prompts = [build_prompt(q, args.schema) for q in questions]
    raw = generate_batch(model, tok, prompts, max_new_tokens=args.max_new_tokens)

    for q, out in zip(questions, raw):
        print(f"Q: {q}")
        print(f"SQL: {extract_sql(out)}\n")


if __name__ == "__main__":
    main()
