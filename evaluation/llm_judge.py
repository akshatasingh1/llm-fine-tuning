"""
LLM-as-judge scoring for text-to-SQL, using the Gemini API.

String and execution metrics miss cases where a query is written differently but
semantically correct (or matches the gold string but is wrong for the question).
This asks a model to grade each prediction against the question, schema, and gold
query, returning correct / partial / wrong.

Setup: get a key at https://aistudio.google.com/apikey and
    export GEMINI_API_KEY=...        # or GOOGLE_API_KEY   (also read from .env)

Free-tier quotas are small and per-model (e.g. some flash models allow only
~20 requests/day). Calls that fail after retries are recorded as "error" and
excluded from the score — check the `errors` count in the result.
"""
from __future__ import annotations
import concurrent.futures as cf
import json
import os
import re
import time

JUDGE_MODEL = "gemini-flash-lite-latest"  # flash-lite has the most generous free tier

_VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
_ERROR = "error"  # judge call/parse failed — not a statement about the SQL

_PROMPT = """You are grading a text-to-SQL model.

Schema:
{schema}

Question:
{question}

Reference query (known correct):
{gold}

Model's query:
{pred}

Is the model's query a correct answer to the question, given the schema? Judge
semantic equivalence to the reference - different formatting, quoting, aliasing,
column order, or equivalent predicates are fine. Use "partial" if it retrieves
the right data but with a wrong aggregate, extra/missing column, or wrong filter
value. Use "wrong" if it would error or answers a different question.

Respond with only JSON: {{"verdict": "correct" | "partial" | "wrong", "reason": "<one sentence>"}}"""


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {"verdict": _ERROR, "reason": f"unparseable judge reply: {(text or '')[:80]}"}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"verdict": _ERROR, "reason": "invalid JSON from judge"}
    obj["verdict"] = str(obj.get("verdict", "")).lower().strip()
    if obj["verdict"] not in _VERDICT_SCORE:
        return {"verdict": _ERROR, "reason": f"unknown verdict {obj['verdict']!r}"}
    return obj


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines, existing env wins."""
    if not os.path.isfile(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def _make_client():
    from google import genai

    _load_dotenv()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY (get one at https://aistudio.google.com/apikey)")
    return genai.Client(api_key=key)


def judge_one(client, question: str, schema: str, gold: str, pred: str,
              model: str = JUDGE_MODEL, max_retries: int = 4) -> dict:
    from google.genai import types
    from google.genai.errors import ClientError, ServerError

    prompt = _PROMPT.format(schema=schema, question=question, gold=gold, pred=pred)
    cfg = types.GenerateContentConfig(
        temperature=0.0, max_output_tokens=512, response_mime_type="application/json",
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    last_err = "unknown"
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            out = _parse(resp.text)
            out["score"] = _VERDICT_SCORE.get(out["verdict"])  # None for errors
            return out
        except (ClientError, ServerError) as e:
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            last_err = f"{status}"
            if status == 429 or (status or 0) >= 500:
                time.sleep(2 ** attempt + 1)  # backoff on rate limit / transient
                continue
            raise
    return {"verdict": _ERROR, "reason": f"judge call failed (HTTP {last_err})", "score": None}


def judge_batch(
    questions: list[str],
    schemas: list[str],
    golds: list[str],
    preds: list[str],
    model: str = JUDGE_MODEL,
    max_workers: int = 2,
) -> dict:
    """
    Grade a set of predictions. Score is the mean over successfully-graded items;
    `errors` counts calls that failed (e.g. quota) and are excluded from the score.
    """
    client = _make_client()
    items = list(zip(questions, schemas, golds, preds))

    def run(i_item):
        i, (q, s, g, p) = i_item
        return i, judge_one(client, q, s, g, p, model=model)

    results: list[dict] = [None] * len(items)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, verdict in ex.map(run, enumerate(items)):
            results[i] = verdict

    scored = [r for r in results if r["verdict"] in _VERDICT_SCORE]
    counts = {v: sum(r["verdict"] == v for r in scored) for v in _VERDICT_SCORE}
    return {
        "n": len(results),
        "n_scored": len(scored),
        "errors": len(results) - len(scored),
        "mean_score": (sum(r["score"] for r in scored) / len(scored)) if scored else None,
        "counts": counts,
        "verdicts": results,
    }
