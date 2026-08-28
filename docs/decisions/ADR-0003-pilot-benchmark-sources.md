# ADR-0003: Pilot benchmark sources and licensing

## Status

Accepted for `data/benchmarks/pilot_v1.jsonl` (200 items). This is a
pilot pool for model-pair screening only — see "Explicitly not decided
here."

## Context

Stage 0/1 built the infrastructure to call models and score responses;
nothing in this repo had real benchmark prompts yet
(`data/example_dataset.jsonl` is 5 illustrative rows, not a benchmark).
This ADR records where the real ~200-item pilot pool's prompts came from,
how each source's license was verified, and how items were selected —
so the dataset is reproducible and its provenance is auditable without
re-deriving it from git history.

## Sources selected

| Category | Source | Repo | License | Verified via |
|---|---|---|---|---|
| `mathematical_reasoning` | GSM8K | `openai/grade-school-math` | MIT | `LICENSE` file at that repo's `master` branch, fetched directly |
| `coding` | HumanEval | `openai/human-eval` | MIT | `LICENSE` file at that repo's `master` branch, fetched directly |
| `logical_reasoning` | BIG-Bench-Hard (BBH) | `suzgunmirac/BIG-Bench-Hard` | MIT | `LICENSE` file at that repo's `main` branch, fetched directly |
| `factual_knowledge` | TruthfulQA (MC1) | `sylinrl/TruthfulQA` | Apache-2.0 | `LICENSE` file at that repo's `master` branch, fetched directly |
| `structured_extraction_constraint_following` | IFEval | `google-research/google-research` (`instruction_following_eval`) | Apache-2.0 | repo-root `LICENSE` file, fetched directly; cross-checked against the "Apache 2.0 license" statement on the `google/IFEval` Hugging Face dataset card |

All five licenses were fetched as raw file content directly from each
repository's canonical branch (not inferred from a README badge or a
third-party mirror) before any item was copied. Exact URLs and the
sha256 of every raw file actually used are recorded in
`data/benchmarks/pilot_v1.manifest.json` → `raw_source_files_sha256`,
reproducible via `scripts/fetch_pilot_raw_sources.py`.

Every one of these five licenses is a permissive OSS license (MIT /
Apache-2.0) that clearly permits redistribution of the underlying data
files. Consequently — per this ADR's own requirement to only copy raw
items where redistribution is clearly permitted — **all selected items
are copied directly into `data/benchmarks/pilot_v1.jsonl`**, not just
referenced. No source in this pilot required the reference-only
fallback (source + retrieval instructions without raw items).

## Category considered and deliberately excluded: summarization/transformation

Summarization was in the desired task-mix list but is not represented in
this pilot. Established summarization benchmarks (CNN/DailyMail, XSum,
...) source their *articles* from copyrighted news outlets; the
"dataset" license usually covers the compilation/splits, not necessarily
clean redistribution rights over the underlying article text itself, and
verifying that distinction authoritatively for each candidate source was
judged not worth the risk for a pilot pool when four other qualitatively
distinct, cleanly-licensed categories were already available. This is
the "do not force categories whose source/evaluation quality is
inadequate" case, not an oversight. A future pass could add
summarization via a source with self-contained licensed text (e.g. a
dataset whose source text was authored for the dataset itself, not
scraped from copyrighted news) if that's judged worth pursuing.

## Evaluator-compatibility filtering

- **GSM8K / BBH / TruthfulQA MC1** → `exact_match`. Original prompts were
  augmented with an explicit answer-format instruction ("answer with only
  the final number" / "...the letter..." / "...the exact option text...")
  because our `ExactMatchEvaluator` does whole-response normalized string
  equality, not extraction from free-form reasoning — without that
  instruction, a well-reasoned but verbose correct answer would score 0.
  The original, un-augmented prompt/question is preserved in
  `metadata.original_question` / `metadata.original_input` on every item.
- **BBH specifically** required detecting *which* of two options-formats a
  task uses (lettered `(A)/(B)/...` vs. a plain dash list like
  `- Yes\n- No`) and phrasing the instruction accordingly — an initial
  version of the curation script asked every BBH item to "answer with a
  letter," which is simply wrong for dash-style tasks like
  `causal_judgement` (caught during review before this was committed —
  see the deterministic sampling / evaluator sanity checks in
  `scripts/validate_pilot_benchmark.py`).
- **HumanEval** → `code_generation`, scored only by
  `HeuristicMockCodeEvalEvaluator` (checks for `def <entry_point>(` and a
  `return` — a non-executing surface heuristic). HumanEval's own
  assert-based `test` field and `canonical_solution` are preserved in
  `reference_answer` for a possible future real executor, but
  `test_cases` (the field `UnsandboxedSubprocessCodeEvalEvaluator` reads)
  is left empty because HumanEval's test format doesn't map onto that
  evaluator's call/expected-value schema without further adaptation.
- **IFEval** → `constraint_checking`, but only for items whose entire
  `instruction_id_list` is drawn from a 9-type subset our
  `ConstraintCheckingEvaluator` can express exactly
  (`keywords:existence`, `keywords:forbidden_words`,
  `startend:end_checker`, `startend:quotation`,
  `detectable_format:title`, `change_case:english_capital`,
  `change_case:english_lowercase`, `punctuation:no_comma`,
  `length_constraints:number_words`) — out of 24 instruction types IFEval
  uses in total. The mapping (including the `num_words` off-by-one fix
  needed because IFEval's "less than" is strict `<` but our `max_words`
  check is `<=`) was verified against the authoritative
  `instructions.py` / `instructions_registry.py` source in
  `google-research/google-research`, not guessed from example prompts.
  387 of 541 IFEval items were excluded for using an unsupported
  instruction type; see `pilot_v1.manifest.json` → `exclusions`.

## Selection procedure

Fixed seed **1337**. Per source: load all items → apply the
source-specific eligibility filter above → merge into one pool per task
category → one global de-duplication pass, in a fixed
category-then-original-order, over whitespace/case-normalized prompt text
(first occurrence wins; e.g. two near-duplicate BBH `causal_judgement`
items were caught this way) → per category, `random.Random(1337).shuffle()`
the de-duplicated pool and take the first 40 → final file sorted by
`prompt_id`. Implemented in `scripts/build_pilot_benchmark.py`; every
excluded item (ineligible or duplicate) and its reason is in
`pilot_v1.manifest.json` → `exclusions` (392 total across the run this
produced).

No item was labeled "easy" or "hard" — difficulty is intentionally not a
field anywhere in this pipeline; it would only ever be derived later from
actual measured model performance.

## Explicitly not decided here

- Which two models form the final "strong"/"cheap" pair (ADR-0002 still
  applies — nothing here touches `configs/models.yaml`).
- Final `k` or `δ` (ADR-0001 still applies).
- Anything about routers, thresholds, or calibration.
- **This pool must not be used to tune any of the above.** It exists for
  model-pair screening, quality-separation/stochasticity/non-monotonic-case
  understanding, and cost/quality headroom estimation. The eventual
  held-out test set is a separate, later artifact and must not overlap
  with or be derived from this pool's selection process.

## Consequences

- `data/benchmarks/pilot_v1.jsonl` + `pilot_v1.manifest.json` are the
  first real (non-synthetic) data this project has committed, and are
  ready to be pointed at from an `ExperimentConfig`'s `dataset_path` once
  real model calls are authorized.
- `scripts/fetch_pilot_raw_sources.py` + `scripts/build_pilot_benchmark.py`
  make regeneration fully reproducible (same raw bytes, verified by
  sha256, plus the same fixed seed ⇒ byte-identical output); rerunning
  them makes no network calls to any LLM provider.
