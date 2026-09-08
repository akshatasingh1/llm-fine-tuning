"""
Instruction-tuning dataset construction.
Converts raw domain text into (instruction, input, output) triplets
suitable for fine-tuning with chat or completion templates.
"""
from __future__ import annotations
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingExample:
    instruction: str
    input_text: str
    output: str
    source: str = ""

    def to_alpaca(self) -> dict:
        return {
            "instruction": self.instruction,
            "input": self.input_text,
            "output": self.output,
        }

    def to_chatml(self) -> dict:
        user_content = self.instruction
        if self.input_text:
            user_content += f"\n\n{self.input_text}"
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": self.output},
            ]
        }


DOMAIN_TEMPLATES = [
    {
        "instruction": "Explain what {concept} means in the context of machine learning.",
        "concepts": ["gradient descent", "regularization", "cross-validation",
                     "feature importance", "model calibration", "data leakage"],
    },
    {
        "instruction": "Write a Python function that {task}.",
        "tasks": ["computes RMSE between two arrays", "applies min-max normalization",
                  "splits data into train/validation/test sets", "encodes categorical features"],
    },
    {
        "instruction": "What are the key assumptions of {model}?",
        "models": ["linear regression", "logistic regression", "naive Bayes",
                   "the Cox proportional hazards model", "an ARIMA model"],
    },
]


def generate_synthetic_dataset(
    n_examples: int = 500,
    domain: str = "data_science",
    seed: int = 42,
) -> list[TrainingExample]:
    """
    Generates synthetic instruction-tuning examples for a DS domain.
    In production: replace with real domain documents + extraction pipeline.
    """
    random.seed(seed)
    examples = []

    for _ in range(n_examples):
        template = random.choice(DOMAIN_TEMPLATES)
        key = [k for k in template if k != "instruction"][0]
        item = random.choice(template[key])
        instruction = template["instruction"].format(**{key[:-1]: item})

        output = (
            f"This is a {domain} explanation of '{item}'. "
            f"In practice, {item.lower()} involves careful consideration of "
            f"data quality, model assumptions, and evaluation strategy."
        )

        examples.append(TrainingExample(
            instruction=instruction,
            input_text="",
            output=output,
            source=domain,
        ))

    print(f"Generated {len(examples)} training examples")
    return examples


def save_dataset(
    examples: list[TrainingExample],
    path: str,
    fmt: str = "alpaca",
) -> str:
    Path(path).parent.mkdir(exist_ok=True)
    records = [e.to_alpaca() if fmt == "alpaca" else e.to_chatml() for e in examples]
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} examples to {path}")
    return path


def quality_filter(examples: list[TrainingExample], min_output_len: int = 30) -> list[TrainingExample]:
    """Basic quality filters: length, dedup, no empty outputs."""
    seen = set()
    filtered = []
    for ex in examples:
        key = ex.instruction.strip().lower()
        if key in seen:
            continue
        if len(ex.output.strip()) < min_output_len:
            continue
        seen.add(key)
        filtered.append(ex)
    print(f"Quality filter: {len(examples)} → {len(filtered)} examples")
    return filtered