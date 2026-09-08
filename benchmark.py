"""
Before/after benchmark for the text-to-SQL fine-tune.

Generates SQL on a held-out test split with (1) the base model and (2) the base
model + trained LoRA adapter, scores both, and writes a comparison.

Usage:
    python benchmark.py --base_model Qwen/Qwen2.5-0.5B \
        --adapter_dir results/sql_lora --n_test 500
"""
import argparse
import json
from pathlib import Path

from data.sql_dataset import load_sql_splits
from evaluation.sql_metrics import evaluate_sql, extract_sql
from training.lora_trainer import load_model_and_tokenizer, generate_batch


def _run_model(base_model, adapter_dir, prompts, max_new_tokens, batch_size):
    model, tok = load_model_and_tokenizer(base_model, adapter_dir=adapter_dir)
    outs = generate_batch(model, tok, prompts, max_new_tokens=max_new_tokens,
                          batch_size=batch_size)
    del model
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--adapter_dir", default="results/sql_lora")
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=15, help="qualitative samples to dump")
    ap.add_argument("--judge", action="store_true",
                    help="also grade with an LLM judge (needs GEMINI_API_KEY)")
    ap.add_argument("--judge_n", type=int, default=100, help="examples to send to the judge")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same seed/slicing as training -> this test split was never trained on.
    _, _, test = load_sql_splits(n_test=args.n_test)
    prompts = [ex["prompt"] for ex in test]
    references = [ex["completion"] for ex in test]
    schemas = [ex["schema"] for ex in test]

    print(f"\n=== Base model: {args.base_model} ===")
    base_raw = _run_model(args.base_model, None, prompts, args.max_new_tokens, args.batch_size)
    base_scores = evaluate_sql(base_raw, references, schemas)

    print(f"\n=== Fine-tuned: {args.base_model} + {args.adapter_dir} ===")
    ft_raw = _run_model(args.base_model, args.adapter_dir, prompts, args.max_new_tokens, args.batch_size)
    ft_scores = evaluate_sql(ft_raw, references, schemas)

    # raw generations, so the judge can be re-run later without regenerating
    (out_dir / "predictions.json").write_text(json.dumps([
        {"question": t["question"], "schema": t["schema"], "gold": t["completion"],
         "base": extract_sql(b), "finetuned": extract_sql(f)}
        for t, b, f in zip(test, base_raw, ft_raw)
    ], indent=2))

    report = {
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "n_test": len(test),
        "base": base_scores,
        "finetuned": ft_scores,
        "delta": {k: round(ft_scores[k] - base_scores[k], 4)
                  for k in ("exact_match", "execution_match", "valid_sql_rate")},
    }

    if args.judge:
        from evaluation.llm_judge import judge_batch

        k = min(args.judge_n, len(test))
        print(f"\n=== LLM-as-judge on {k} examples ===")
        jq, js, jg = ([ex[f] for ex in test[:k]] for f in ("question", "schema", "completion"))
        base_j = judge_batch(jq, js, jg, [extract_sql(x) for x in base_raw[:k]])
        ft_j = judge_batch(jq, js, jg, [extract_sql(x) for x in ft_raw[:k]])

        def _slim(j):
            return {"n_scored": j["n_scored"], "errors": j["errors"],
                    "mean_score": j["mean_score"], "counts": j["counts"]}

        report["judge"] = {"n": k, "base": _slim(base_j), "finetuned": _slim(ft_j)}
        if base_j["mean_score"] is not None and ft_j["mean_score"] is not None:
            report["judge"]["delta"] = round(ft_j["mean_score"] - base_j["mean_score"], 4)

        for name, j in (("base", base_j), ("finetuned", ft_j)):
            score = "n/a" if j["mean_score"] is None else f"{j['mean_score']:.3f}"
            err = f"  ({j['errors']} errors)" if j["errors"] else ""
            print(f"  {name:<10} {score}  {j['counts']}{err}")
        if base_j["errors"] or ft_j["errors"]:
            print("  note: errors are usually free-tier quota — reduce --judge_n or use a paid key")

    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=2))

    def pct(x):
        return f"{100 * x:.1f}%"

    print("\n" + "=" * 56)
    print(f"{'metric':<20}{'base':>10}{'finetuned':>14}{'delta':>12}")
    print("-" * 56)
    for k in ("exact_match", "execution_match", "valid_sql_rate"):
        d = report["delta"][k]
        print(f"{k:<20}{pct(base_scores[k]):>10}{pct(ft_scores[k]):>14}"
              f"{('+' if d >= 0 else '') + pct(d):>12}")
    print("=" * 56)

    # qualitative samples
    lines = ["# Sample generations\n",
             f"Base: `{args.base_model}` · Adapter: `{args.adapter_dir}`\n"]
    for i in range(min(args.n_samples, len(test))):
        lines += [
            f"\n## {i + 1}. {test[i]['question']}",
            f"\n**Schema:** `{test[i]['schema']}`\n",
            f"\n| | SQL |",
            f"|---|---|",
            f"| gold | `{references[i]}` |",
            f"| base | `{extract_sql(base_raw[i])}` |",
            f"| finetuned | `{extract_sql(ft_raw[i])}` |",
        ]
    (out_dir / "samples.md").write_text("\n".join(lines))

    print(f"\nWrote {out_dir/'benchmark.json'} and {out_dir/'samples.md'}")


if __name__ == "__main__":
    main()
