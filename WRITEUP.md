# Fine-tuning Qwen2.5-0.5B for text-to-SQL — writeup

A short account of what was built, what was measured, and the decisions along the
way. Numbers reproduce with `notebooks/train_sql.ipynb` on a Colab T4.

## 1. Why fine-tune (and not RAG)

The task — natural-language question + `CREATE TABLE` schema → SQL query — needs
the model to learn an **output behaviour**, not new facts:

- produce SQL in a fixed dialect and style, every time
- ground column/table names in the schema it was given, not invent them
- stop after the query

RAG helps when the missing thing is *knowledge*. Here the base model already
knows SQL; it just does not reliably follow the dataset's conventions. That is a
fine-tuning problem.

## 2. Data

`b-mc2/sql-create-context` — 78,577 rows of `(question, schema, answer)`.
Single fixed shuffle (seed 42), then disjoint slices: 4,000 train / 200 eval /
500 test. The test slice is never seen in training, and the base model is scored
on the *same* prompts, so the before/after is honest.

Mostly single-table queries — easier than Spider. Some templated questions
recur across the corpus, so held-out rows can be structurally similar to training
rows. Both caveats are stated in the README.

## 3. Method

- **LoRA**, rank 16, α 32, dropout 0.05, on the attention projections
  (`q_proj, k_proj, v_proj, o_proj`). ~0.9M trainable params (0.7% of the model).
- **Completion-only loss**: the prompt tokens get label `-100`, so cross-entropy
  is computed only over the SQL. Training the model to also reproduce the
  instruction wastes capacity and, on a 0.5B model, measurably hurt output
  discipline in early runs.
- **Dynamic padding** (`DataCollatorForSeq2Seq`) instead of padding every row to
  512 — most prompt+SQL pairs are under 200 tokens, so this cut training compute
  ~3×.
- 2 epochs, LR 2e-4, cosine schedule, batch size 16. ~4 minutes on a T4.
- QLoRA (`--load_in_4bit`, NF4) is wired in so the same script fine-tunes a 7B
  model on the same 16 GB GPU; the headline run did not need it.

Loss: train 0.19 → 0.065, eval 0.097. No sign of overfitting at 2 epochs; a
third epoch moved eval loss <0.005.

## 4. Results

500-example held-out test split (LLM-judge on a 20-example subset).

| metric | base | fine-tuned | Δ |
|---|---|---|---|
| exact match (normalized string) | 10.0% | 72.8% | +62.8 pp |
| execution match (same rows on a SQLite DB from the schema) | 84.8% | 93.0% | +8.2 pp |
| valid SQL rate | 93.0% | 95.8% | +2.8 pp |
| LLM-judge mean (correct=1 / partial=0.5 / wrong=0), n=25 | 0.70 | 0.92 | +0.22 |

Reading these together matters:

- **Exact match** overstates the gain — most of that +63 is the model learning
  cosmetic conventions (double-quoted lowercase string literals, no trailing
  columns). A hand grader would call many of the base model's "misses" acceptable.
- **Execution match** ignores formatting and still improves 8 points — real
  accuracy, not just neatness.
- **The LLM-judge** (Gemini, semantic-equivalence grading) sits between the two
  and confirms the direction independently: base 0.70, fine-tuned 0.92. Sample is
  small (25) because the free Gemini tier is rate-limited; `docs/judge_result.json`
  has both runs.

## 5. What the model actually learned

From `docs/sample_generations.md`, base → fine-tuned:

| behaviour | base | fine-tuned |
|---|---|---|
| string literals | `team = 'Aberdeen'` | `team = "aberdeen"` |
| over-selection | `SELECT name, round FROM ...` | `SELECT name FROM ...` |
| hallucinated columns | `championship_years__years_` | `championships__years_` |
| structure | splits `"tko ... round 1"` into 3 fake columns | keeps the literal |

Two of fifteen sampled queries are still wrong after fine-tuning (`a_` vs
`kickoff_[a_]`; `MAX` vs `AVG`) — kept in the samples file rather than removed.

## 6. Decisions and dead ends

- **Target modules**: started with `q_proj, v_proj` only (the classic LoRA
  choice). Adding `k_proj, o_proj` was a clear win on this task for ~2× the
  adapter size — cheap at 0.5B.
- **Prompt masking**: the first version trained on the full sequence
  (prompt + completion). The model learned to echo `### SQL:` headers into its
  output. Masking the prompt fixed it.
- **fp16 vs bf16**: fp16 on the T4 with LoRA was stable; QLoRA needs bf16 compute
  (NF4 dequant), so the trainer picks the dtype from the quantization setting.
- **Judge model**: `gemini-2.0-flash` is retired; some flash models cap the free
  tier at ~20 requests/day, which silently looked like "the base model scores
  0%" until the quota errors were surfaced separately. Settled on
  `gemini-flash-lite-latest` and made failed calls count as `error`, not `wrong`.

## 7. Limitations

- Single-table SQL; no JOINs, subqueries, or Spider-level schema linking.
- Execution match runs on empty databases (the dataset ships only schemas), so it
  cannot separate two valid queries that both return nothing.
- Possible train/test structural overlap from templated questions.
- 0.5B model — a 1.5–7B base (via the QLoRA path) would raise the ceiling.

## 8. Next

- Run the QLoRA path on Qwen2.5-7B and compare the curve.
- Swap in the Spider dev set for a harder, cleaner benchmark.
- Full 500-example LLM-judge pass on a paid key; add position-swap debiasing.
- Serve the merged model behind a small FastAPI endpoint.
