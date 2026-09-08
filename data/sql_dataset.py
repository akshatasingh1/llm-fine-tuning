"""
Text-to-SQL dataset loader.

Source: b-mc2/sql-create-context (HuggingFace) — ~78k examples of
(question, schema, SQL answer), where the schema is one or more
CREATE TABLE statements.

Each split is a list of dicts with:
    question   : natural-language question
    schema     : CREATE TABLE statement(s)
    answer     : gold SQL query
    prompt     : formatted model input (ends right before the SQL)
    completion : gold SQL query (what the model should generate)
"""
from __future__ import annotations

DATASET_ID = "b-mc2/sql-create-context"

PROMPT_TEMPLATE = (
    "### Task:\n"
    "Write a single SQL query that answers the question, using only the "
    "tables and columns in the schema. Reply with SQL only.\n"
    "### Schema:\n{schema}\n"
    "### Question:\n{question}\n"
    "### SQL:\n"
)


def build_prompt(question: str, schema: str) -> str:
    return PROMPT_TEMPLATE.format(schema=schema.strip(), question=question.strip())


def _to_example(row: dict) -> dict:
    return {
        "question": row["question"],
        "schema": row["context"],
        "answer": row["answer"],
        "prompt": build_prompt(row["question"], row["context"]),
        "completion": row["answer"].strip(),
    }


def load_sql_splits(
    n_train: int = 4000,
    n_eval: int = 200,
    n_test: int = 500,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns (train, eval, test). The dataset ships only a `train` split, so we
    shuffle once with a fixed seed and carve out disjoint slices.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train").shuffle(seed=seed)

    total = n_train + n_eval + n_test
    if total > len(ds):
        raise ValueError(f"Requested {total} examples but dataset has {len(ds)}")

    train = [_to_example(ds[i]) for i in range(n_train)]
    eval_ = [_to_example(ds[i]) for i in range(n_train, n_train + n_eval)]
    test = [_to_example(ds[i]) for i in range(n_train + n_eval, total)]

    print(f"SQL dataset: {len(train)} train / {len(eval_)} eval / {len(test)} test")
    return train, eval_, test
