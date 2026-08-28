# ADR-0002: Stage 0 provider/model assumptions

## Status

Accepted for Stage 0 infrastructure only. Not a model-pair decision.

## Context

Stage 0 needs at least one real provider adapter to exercise end-to-end
(request building, usage parsing, cost calculation, persistence). Anthropic
was named as the first experimental provider. Before writing the adapter,
the current Anthropic Messages API contract was checked against live
documentation (via the bundled `claude-api` skill, cached 2026-06-24,
cross-checked 2026-08-25) rather than assumed from training data, because
several API shapes (extended thinking, effort, usage fields) changed in
2025–2026.

## What was verified

- **Model IDs** in `configs/models.yaml` (`claude-sonnet-5`,
  `claude-haiku-4-5`, `claude-opus-5`) are real, current model ID
  strings — not constructed or date-suffixed. `claude-opus-5`'s
  `ModelEntry` was added 2026-08-25 during model-pair pilot preparation,
  mirroring `claude-sonnet-5`'s generation settings (adaptive thinking,
  `effort: high`) since both are thinking-on-by-default models.
- **Usage fields**: `response.usage` exposes `input_tokens`,
  `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`.
  There is **no separate billed reasoning/thinking-token field** — thinking
  tokens are counted inside `output_tokens`. `TokenUsage.reasoning_tokens`
  is therefore always `None` for this provider by design, not an oversight.
- **Request id**: `response._request_id` (public despite the leading
  underscore).
- **Extended thinking**: current form is
  `thinking={"type": "adaptive", "display": ...}`; the older
  `budget_tokens` fixed-budget parameter is rejected (400) on
  current-generation models, so the adapter never sends it unless a model
  entry explicitly sets `thinking.budget_tokens` (kept only for a possible
  future older-model adapter path).
- **Effort**: `output_config={"effort": ...}`, one of
  `low|medium|high|xhigh|max`.

## Pricing in `configs/pricing.yaml` — verification history

**2026-08-25 correction:** an earlier version of this ADR and of
`pricing.yaml` modeled Claude Sonnet 5's $2/$10 per-MTok rate as a
temporary introductory period expiring 2026-08-31, after which the
config switched to a modeled "standard" rate of $3/$15. That was wrong.
Re-checking Anthropic's official pricing docs
(https://platform.claude.com/docs/en/about-claude/pricing) directly
found an explicit statement superseding it: "The $2/$10 per million
input/output token pricing for Claude Sonnet 5, announced at launch as
introductory pricing through August 31, 2026, is now the standard
price. The previously scheduled increase to $3/$15 per million
input/output tokens on September 1, 2026 will not occur." `pricing.yaml`
now models Sonnet 5 as a single open-ended rate period at $2/$10 — the
same structure Haiku 4.5 and Opus 5 already used — with
`pricing_config_version: "2026-08-25.2"` recording the correction.

- Cache write/read price multipliers (5-minute write 1.25x base input
  price, 1-hour write 2x, cache read 0.1x) were previously flagged here
  as an unverified assumption carried forward from Anthropic's general
  caching docs. The same 2026-08-25 pricing-docs check **confirmed**
  these exactly, per-model — the official table lists each model's
  5-minute-write / 1-hour-write / cache-read rate explicitly, and every
  value in `pricing.yaml` matches it. This is no longer an open
  assumption for Sonnet 5, Opus 5, or Haiku 4.5.
- Claude Haiku 4.5 ($1/$5) and Claude Opus 5 ($5/$25) were unchanged by
  this correction — both were already accurate.

## Explicitly not decided here

- Which model plays "strong" and which plays "cheap" — `role_hint` values
  in `configs/models.yaml` (`candidate_strong` / `candidate_cheap`) are
  provisional labels for adapter testing, not a frozen pair (see ADR-0001).
  `claude-opus-5` was given `role_hint: null` when its `ModelEntry` was
  added — it now has a working config but no assigned role.
- Whether Claude Opus 5 or another tier belongs in the final pair — having
  a complete, callable config for Opus 5 is a prerequisite for piloting
  it, not a decision to use it.

## Consequences

- `router.models.anthropic_adapter` is the only module allowed to import
  `anthropic` or assume any of the shapes above — if the API changes again,
  that's the only file that needs to change.
- Cost figures produced by Stage 0's verification script, and by any
  pilot built on `configs/pricing.yaml`, can now be treated as accurate
  for Sonnet 5, Opus 5, and Haiku 4.5's token pricing (base and cache) —
  all independently confirmed against official docs as of 2026-08-25.
  They remain estimates in the sense that real thinking-token consumption
  is only knowable by actually calling the API.
