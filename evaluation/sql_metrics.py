"""
Text-to-SQL evaluation metrics.

- exact_match        : normalized string equality (strict, cheap)
- execution_match    : run gold vs predicted SQL against an in-memory SQLite DB
                       built from the schema, compare result sets
- valid_sql_rate     : fraction of predictions that parse + execute without error

Execution match is the standard text-to-SQL metric. The DBs here are empty
(the dataset only provides CREATE TABLE statements), so two queries "match" when
they return the same rows on empty tables *and* both execute — this still catches
hallucinated columns/tables, broken syntax, and wrong query shape, though it
cannot distinguish two syntactically-valid queries with the same empty output.
"""
from __future__ import annotations
import re
import sqlite3


def extract_sql(text: str) -> str:
    """Pull the SQL out of a raw model generation."""
    text = text.strip()
    # strip markdown fences
    fence = re.match(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    # if the model kept generating extra sections, cut at the next header
    text = re.split(r"\n###\s", text)[0].strip()
    # take up to the first statement terminator
    if ";" in text:
        text = text.split(";")[0].strip()
    # collapse to a single line
    return " ".join(text.split())


def normalize_sql(sql: str) -> str:
    sql = " ".join(sql.strip().rstrip(";").split()).lower()
    sql = re.sub(r"\s*([(),])\s*", r"\1", sql)  # tighten punctuation spacing
    sql = re.sub(r"\s*(=|<|>|<=|>=|<>|!=)\s*", r"\1", sql)
    return sql


def exact_match(predictions: list[str], references: list[str]) -> float:
    hits = sum(normalize_sql(p) == normalize_sql(r)
               for p, r in zip(predictions, references))
    return hits / max(1, len(predictions))


def _run(schema: str, query: str):
    """Execute query against a fresh in-memory DB built from schema. Returns rows."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(schema)
        cur = conn.execute(query)
        return sorted(map(str, cur.fetchall()))
    finally:
        conn.close()


def execution_scores(
    predictions: list[str],
    references: list[str],
    schemas: list[str],
) -> dict:
    exec_hits = 0
    valid_preds = 0
    for pred, ref, schema in zip(predictions, references, schemas):
        try:
            gold_rows = _run(schema, ref)
        except Exception:
            continue  # unusable gold row, skip from denominator-of-truth comparisons
        try:
            pred_rows = _run(schema, pred)
            valid_preds += 1
            if pred_rows == gold_rows:
                exec_hits += 1
        except Exception:
            pass

    n = max(1, len(predictions))
    return {
        "execution_match": exec_hits / n,
        "valid_sql_rate": valid_preds / n,
    }


def evaluate_sql(
    raw_predictions: list[str],
    references: list[str],
    schemas: list[str],
) -> dict:
    preds = [extract_sql(p) for p in raw_predictions]
    results = {"n": len(preds), "exact_match": exact_match(preds, references)}
    results.update(execution_scores(preds, references, schemas))
    return results
