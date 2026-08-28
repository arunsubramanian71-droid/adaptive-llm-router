# ADR-0001: Pilot sampling range (k and δ)

## Status

Accepted (design phase). Final values NOT frozen.

## Context

The router's routing labels come from offline evaluation of response-level
data, not from a fixed formula decided in advance. We need to be able to
try several candidate values of the per-model sample count `k` and the
quality-gap threshold `δ` without regenerating any model responses, because
regeneration costs money and introduces sampling variance between analysis
attempts.

## Decision

The pilot collects **up to k = 6 samples per model** per prompt. This is
large enough to evaluate any `k ∈ {1, 3, 5, 6}` offline by subsetting the
stored samples (`sample_index < k`), and any `δ ∈ {0.2, 0.4, 0.6}` offline
by recomputing labels from stored per-sample quality scores once a scoring
stage exists.

Consequently, `router.storage.records.ResponseRecord` stores one row per
`(prompt, provider, model, generation_config, sample_index)` — never an
aggregate — and reserves `q_hat` / `label` / `score_status` as placeholders
for a scoring stage that does not exist yet (Stage 0 does not implement
evaluation).

## Explicitly not decided here

- The final value of `k` used for routing-label generation.
- The final value of `δ`.
- The final model pair (`configs/models.yaml` currently lists Anthropic
  models as adapter-testing placeholders only — see ADR-0002).

## Consequences

- Stage 0's persistence schema must support per-sample storage up to
  `k=6`; it must not aggregate scores or responses at write time.
- Any later evaluation/labeling stage reads directly from `records.jsonl`
  and never needs to re-call a provider API to try a different `k` or `δ`.
