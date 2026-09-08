"""
Main fine-tuning entry point.
Usage: python finetune.py [--model_name microsoft/phi-2] [--n_examples 500]
"""
import argparse
import json
from pathlib import Path
from data.dataset_builder import generate_synthetic_dataset, quality_filter, save_dataset
from training.lora_trainer import LoRAFinetuner, TrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/phi-2")
    parser.add_argument("--n_examples", type=int, default=500)
    parser.add_argument("--output_dir", default="results/lora_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dry_run", action="store_true", help="Just build dataset, skip training")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    print("Building training dataset...")
    examples = generate_synthetic_dataset(n_examples=args.n_examples)
    examples = quality_filter(examples)

    split = int(len(examples) * 0.9)
    train_ex = [e.to_alpaca() for e in examples[:split]]
    eval_ex  = [e.to_alpaca() for e in examples[split:]]

    save_dataset(examples[:split], "data/train.json")
    save_dataset(examples[split:], "data/eval.json")

    if args.dry_run:
        print(f"\nDry run complete. {len(train_ex)} train / {len(eval_ex)} eval examples ready.")
        print("Run without --dry_run to start training (requires GPU + peft/transformers).")
        return

    config = TrainConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
    )

    trainer = LoRAFinetuner(config)
    history = trainer.train(train_ex, eval_ex)

    with open("results/training_log.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\n✓ Fine-tuning complete.")


if __name__ == "__main__":
    main()