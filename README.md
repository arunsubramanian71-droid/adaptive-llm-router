# Adaptive LLM Cost Router

A provider-agnostic LLM cost router: given a prompt, predict whether a
cheap or a strong model is good enough, and route accordingly. The core
research question is the measured quality gap between the two:

```
Δ(x) = q_strong(x) - q_cheap(x)
```

Routing labels are derived offline from response-level evaluation data —
never guessed, never re-derived by calling a model again once the data is
collected.

## Status

**The full software system is implemented and tested. No real experiment
has been run yet.** Every module below is real, working code exercised by
its own tests — but every test uses synthetic/mocked prompts, responses,
and scores, never a real model call. See "What still needs real model
responses" below for exactly what's blocked on spending API money, and
`docs/decisions/` for what's deliberately still an open decision (final
model pair, final k, final δ).

## Pipeline overview

```
dataset (JSONL)
   │
   ▼
router.models  ──── AnthropicModelClient.complete() ────▶ ResponseRecord   (Stage 0)
   │                                                             │
   ▼                                                             ▼
router.evaluation  ── objective evaluators / judge ──▶ EvalResult / JudgeVerdict
   │
   ▼
router.aggregation  ── q_hat, Δ_hat, label(δ) ──▶ PromptAggregate
   │
   ├──▶ router.policies.oracle    (reference upper bounds — NOT deployable)
   ├──▶ router.policies.baselines (deployable, no learning)
   └──▶ router.routers + calibration ──▶ router.policies.RouterPolicy (deployable, learned)
   │
   ▼
router.analysis  ── thresholds, frontier, metrics, bootstrap CIs, error analysis, ablation
   │
   ▼
router.experiment  ── ExperimentConfig ties the above into one run; reporting.py renders
                       CSV/markdown tables and PNG figures from the results
```

`router.api` is a thin FastAPI demo of the request/response shape
(routing decision, probability, selected role, estimated cost) — it fits
a router on a handful of synthetic bundled prompts at startup and is
explicitly not a research artifact (see its `DEMO_DISCLAIMER`).

## Repository layout

```
configs/
  models.yaml, pricing.yaml    Provider/model/pricing config (Stage 0)
  experiments/                 ExperimentConfig YAML (example_pilot.yaml)
data/
  example_dataset.jsonl        5-row illustrative dataset (all task types)
src/router/
  config.py, hashing.py, git_utils.py        Stage 0 shared infra
  models/                Provider-agnostic model client + Anthropic adapter
  cost/                  Cost calculation from provider-reported usage
  storage/               Response records, content-addressed cache, run metadata
  logging/                Structured (JSON) logging
  dataset/                DatasetItem schema + JSONL loader/validator
  evaluation/
    evaluators/            exact_match, structured_extraction, constraint_checking,
                           code_eval (non-executing mock by default — see below)
    judge/                 JudgeClient interface + mock judges + pipeline
    pipeline.py             Ties dataset + ResponseRecords + evaluators together
  aggregation/             q_hat / Δ_hat / label(δ) + offline k/δ sweep
  policies/                Policy/OraclePolicy interfaces, baselines, oracles,
                           RouterPolicy (wraps a fitted Router + tau)
  routers/                 Router interface, TF-IDF+LogReg, handcrafted-feature,
                           gradient-boosting routers, calibration (Platt/isotonic,
                           Brier, ECE)
  analysis/                metrics, threshold sweep, cost-quality/Pareto frontier,
                           paired bootstrap CIs, error analysis, ablation framework
  experiment/              ExperimentConfig (YAML-backed) + CSV/markdown/figure reporting
  api/                     FastAPI demo app (optional `api` extra)
scripts/
  verify_stage0.py         Tiny, explicitly-gated LIVE verification run
tests/                     176 tests — no test requires a paid API call
runs/                      One directory per experiment run (git-ignored)
docs/decisions/            ADRs
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv\Scripts\activate on Windows cmd
pip install -e ".[dev]"         # core + numpy/scikit-learn + dev tooling
pip install -e ".[viz]"         # optional: matplotlib, for reporting figures
pip install -e ".[api]"         # optional: fastapi/uvicorn, for the demo app
cp .env.example .env            # then fill in ANTHROPIC_API_KEY (only needed for --live)
```

## Running tests

```bash
pytest          # 176 tests, all mocked/synthetic — no test spends money
ruff check src tests scripts
mypy src
```

## Verifying the Stage 0 pipeline end-to-end (the only thing that can spend money)

```bash
python scripts/verify_stage0.py                                              # free dry run
python scripts/verify_stage0.py --live --model claude-haiku-4-5 --num-prompts 2   # real calls
```

See [`docs/decisions/`](docs/decisions/) and the Stage 0 section of this
file's history for the full description of run artifacts, caching, and
cost-calculation guarantees — those are unchanged by everything added
since.

## Running the demo API (optional, not a research artifact)

```bash
pip install -e ".[api]"
uvicorn router.api.app:app --reload
curl -s localhost:8000/info | python -m json.tool
curl -s -X POST localhost:8000/route -H 'content-type: application/json' \
  -d '{"prompt": "Prove that this algorithm terminates for every input."}'
```

Every response includes a `disclaimer` field stating the router behind it
is fit on synthetic bundled prompts, not real data.

## What's real vs. mock in this repo, explicitly

| Component | Status |
|---|---|
| Anthropic adapter, cost calc, storage, cache, run metadata (Stage 0) | Real — makes real calls when you run `verify_stage0.py --live` |
| Dataset schema/loader | Real — loads and validates real JSONL you provide |
| Objective evaluators (exact_match, structured_extraction, constraint_checking) | Real — score real response text against real ground truth |
| Code evaluator | Two implementations: `HeuristicMockCodeEvalEvaluator` (default, non-executing, surface heuristic) and `UnsandboxedSubprocessCodeEvalEvaluator` (real execution, opt-in only via `confirm_unsandboxed_execution=True`, **not sandboxed** — see its docstring before ever using it on real model output) |
| Judge | Interface + pipeline are real; both bundled judges (`FixtureJudgeClient`, `HeuristicMockJudgeClient`) are mocks — no real LLM-judge adapter exists yet |
| Aggregation, k/δ sweep, oracles, baselines, routers, calibration, analysis, ablation, experiment config, reporting | Real, fully-implemented mechanisms — tested against synthetic fixtures, never run against real experiment data yet |
| FastAPI demo | Real code, explicitly-labeled demo router fit on synthetic prompts |

## What still needs real model responses

Nothing more needs to be *built* before a real experiment — every
mechanism above is implemented and tested. What's still blocked on
spending real API money:

1. **A real benchmark dataset** — `data/example_dataset.jsonl` is 5
   illustrative rows, not a benchmark. Someone has to write real prompts
   with real ground truth per task type.
2. **The actual pilot run** — calling both candidate models (currently
   placeholder `claude-sonnet-5` / `claude-haiku-4-5`, see ADR-0002) for
   every benchmark prompt, up to k=6 samples each, and persisting the
   results via Stage 0's `ResponseRecord` pipeline.
3. **Real judge scoring**, if any dataset rows are `judge_scored` — needs
   an actual LLM-judge adapter built on `router.models.anthropic_adapter`
   (or another provider), which does not exist yet; only mocks do.
4. **Fitting routers on real labels** — `router.aggregation` needs real
   `EvalResult`/`JudgeVerdict` scores from step 2/3 before `q_hat`, `Δ_hat`,
   and labels mean anything; only then do `router.routers` and
   `router.routers.calibration` have real data to fit and calibrate on.
5. **Choosing k and δ** — `router.aggregation.kdelta_analysis.sweep_k_delta`
   is ready to run this sweep the moment real per-sample scores exist; k
   and δ stay unfrozen (per ADR-0001) until that sweep is actually run.
6. **Freezing the model pair** — `role_hint: candidate_strong/candidate_cheap`
   in `configs/models.yaml` are placeholders (per ADR-0002) until a real
   pilot's cost-quality frontier justifies a choice.

## What the real-experiment phase needs to execute

In order, once real spend is authorized:

1. Write/curate a real benchmark dataset as JSONL matching
   `router.dataset.schemas.DatasetItem` (see `data/example_dataset.jsonl`
   for the shape of each `task_type`).
2. Populate `configs/experiments/<name>.yaml` (copy
   `configs/experiments/example_pilot.yaml`) with the real dataset path,
   model pair, and starting k/δ.
3. Run Stage 0's adapter (`AnthropicModelClient`, via the same pattern as
   `scripts/verify_stage0.py`) over every dataset prompt for both models,
   k samples each — a new orchestration script, not yet written, since
   Stage 0 explicitly deferred "run the pilot."
4. Run `router.evaluation.pipeline.run_evaluation_pipeline` (objective
   tasks) and `router.evaluation.judge.pipeline.run_judge_pipeline` with a
   real judge adapter (judge-scored tasks) against the collected records.
5. Run `router.aggregation.kdelta_analysis.sweep_k_delta` to pick k and δ
   from real data.
6. Fit each `router.routers.*` implementation on the resulting labels,
   calibrate with `router.routers.calibration`, and sweep operating
   thresholds with `router.analysis.thresholds` /
   `router.analysis.frontier`.
7. Compare against `router.policies.oracle` (upper bounds) and
   `router.policies.baselines` (floors) using `router.analysis.metrics`,
   `router.analysis.bootstrap` for confidence intervals, and
   `router.analysis.error_analysis` to understand failure modes.
8. Render tables/figures with `router.experiment.reporting` and write up
   results — only at this point does the project have anything to claim
   about real cost savings or quality retention.

## Configuration

- **Secrets** (`ANTHROPIC_API_KEY`, etc.) come only from the environment /
  `.env` — never from YAML or Python literals.
- **Pricing and generation parameters** come only from
  `configs/models.yaml` / `configs/pricing.yaml`. Nothing in
  `src/router/cost/` hardcodes a price.
- **Experiment parameterization** (dataset, k, δ, router choice,
  calibration method, thresholds, bootstrap settings) is a separate,
  eagerly-validated YAML schema — `router.experiment.config.ExperimentConfig`.
- All configuration is validated eagerly at load time — a bad config fails
  fast with a clear error, never mid-run.

Model IDs, generation parameters, and pricing in the shipped configs are
placeholders for exercising the pipeline — see
[`docs/decisions/ADR-0002-stage0-model-assumptions.md`](docs/decisions/ADR-0002-stage0-model-assumptions.md)
for what's verified vs. assumed, and
[`docs/decisions/ADR-0001-pilot-sampling.md`](docs/decisions/ADR-0001-pilot-sampling.md)
for why nothing here freezes a model pair, k, or δ yet.
