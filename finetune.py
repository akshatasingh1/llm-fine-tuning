"""
Main fine-tuning entry point.

Synthetic demo task:
    python finetune.py --task synthetic --dry_run

Text-to-SQL (real dataset: b-mc2/sql-create-context):
    python finetune.py --task sql --model_name Qwen/Qwen2.5-0.5B \
        --n_train 4000 --epochs 2 --output_dir results/sql_lora
"""
import argparse
import json
from pathlib import Path

from training.lora_trainer import LoRAFinetuner, TrainConfig


def build_synthetic(args):
    from data.dataset_builder import generate_synthetic_dataset, quality_filter, save_dataset

    examples = generate_synthetic_dataset(n_examples=args.n_examples)
    examples = quality_filter(examples)
    split = int(len(examples) * 0.9)
    save_dataset(examples[:split], "data/train.json")
    save_dataset(examples[split:], "data/eval.json")
    return [e.to_alpaca() for e in examples[:split]], [e.to_alpaca() for e in examples[split:]]


def build_sql(args):
    from data.sql_dataset import load_sql_splits

    train, eval_, test = load_sql_splits(
        n_train=args.n_train, n_eval=args.n_eval, n_test=args.n_test
    )
    Path("data").mkdir(exist_ok=True)
    for name, rows in [("sql_train", train), ("sql_eval", eval_), ("sql_test", test)]:
        Path(f"data/{name}.json").write_text(json.dumps(rows, indent=2))
    print(f"Saved data/sql_train.json ({len(train)}), sql_eval.json ({len(eval_)}), "
          f"sql_test.json ({len(test)})")
    return train, eval_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["synthetic", "sql"], default="sql")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", default="results/sql_lora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="QLoRA: 4-bit base weights (needs CUDA + bitsandbytes)")
    # synthetic
    parser.add_argument("--n_examples", type=int, default=500)
    # sql
    parser.add_argument("--n_train", type=int, default=4000)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--dry_run", action="store_true", help="Build dataset, skip training")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    print(f"Building {args.task} dataset...")
    train_ex, eval_ex = build_synthetic(args) if args.task == "synthetic" else build_sql(args)

    if args.dry_run:
        print(f"\nDry run complete. {len(train_ex)} train / {len(eval_ex)} eval examples ready.")
        return

    config = TrainConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    trainer = LoRAFinetuner(config)
    history = trainer.train(train_ex, eval_ex)

    with open("results/training_log.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\nFine-tuning complete.")
    if args.task == "sql":
        print(f"\nNext: python benchmark.py --base_model {args.model_name} "
              f"--adapter_dir {args.output_dir} --n_test {args.n_test}")


if __name__ == "__main__":
    main()
