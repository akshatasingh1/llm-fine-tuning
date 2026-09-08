"""
Evaluation suite for fine-tuned LLMs.
Perplexity, ROUGE-L, task accuracy, and hallucination rate.
"""
from __future__ import annotations
import math
import numpy as np
from rouge_score import rouge_scorer as rs


def perplexity(log_probs: list[float]) -> float:
    """Compute perplexity from token log-probabilities. Lower = better."""
    return math.exp(-np.mean(log_probs))


def rouge_l(predictions: list[str], references: list[str]) -> dict:
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure
              for pred, ref in zip(predictions, references)]
    return {"rouge_l_mean": float(np.mean(scores)), "rouge_l_std": float(np.std(scores))}


def exact_match(predictions: list[str], references: list[str]) -> float:
    matches = sum(p.strip().lower() == r.strip().lower()
                  for p, r in zip(predictions, references))
    return matches / len(predictions)


def hallucination_rate(
    answers: list[str],
    contexts: list[str],
    threshold: float = 0.4,
) -> float:
    """
    Approximate hallucination rate via token overlap:
    if an answer has low overlap with its context, flag as potential hallucination.
    Production version: use NLI model or LLM-as-judge.
    """
    flags = []
    for ans, ctx in zip(answers, contexts):
        ans_tokens = set(ans.lower().split())
        ctx_tokens = set(ctx.lower().split())
        overlap = len(ans_tokens & ctx_tokens) / (len(ans_tokens) + 1e-9)
        flags.append(1 if overlap < threshold else 0)
    return float(np.mean(flags))


def run_eval_suite(
    predictions: list[str],
    references: list[str],
    contexts: list[str] = None,
) -> dict:
    results = rouge_l(predictions, references)
    results["exact_match"] = exact_match(predictions, references)
    if contexts:
        results["hallucination_rate"] = hallucination_rate(predictions, contexts)
    print("Evaluation Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    return results