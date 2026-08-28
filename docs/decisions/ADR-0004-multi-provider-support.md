# ADR-0004: Multi-provider support (Anthropic, OpenAI, Google Gemini)

## Status

Accepted for the provider abstraction layer only. Not a research-pair
decision — no cross-provider pair is frozen by this ADR.

## Context

Through ADR-0001–0003 this project had one working adapter (Anthropic)
behind a `ModelClient` interface that was *designed* to be provider-agnostic
(see `router.models.base`), but nothing exercised that design with a second
provider. In practice the project was "an Anthropic router with a provider
abstraction around it," not a genuinely multi-provider one. This ADR adds
two more adapters — OpenAI and Google Gemini — and records the official
documentation used to configure them, so the abstraction is proven rather
than assumed.

## What was verified, and how

Every model ID, price, SDK method name, and field name below was checked
against current official documentation (fetched 2026-08-25), then
cross-checked directly against the *installed* SDK package's actual types
and exception classes — not assumed from training-data familiarity, which
is exactly the kind of thing that goes stale (see ADR-0002's Sonnet 5
pricing correction for a concrete example of why this matters).

### OpenAI

- **Docs used**: `developers.openai.com/api/docs/{quickstart,pricing,
  guides/reasoning,guides/error-codes}` (the current canonical OpenAI API
  docs; `platform.openai.com/docs/*` 301-redirects here now).
- **Pricing table** was fetched twice independently with a "quote
  verbatim" instruction and returned identical rows both times, including
  a `gpt-5.6-{sol,terra,luna}` naming tier not present in this project's
  training-era knowledge — treated as real rather than dismissed, given
  the consistent verbatim reproduction alongside many independently
  recognizable rows (`gpt-5`, `gpt-5-mini`, `o3`, `gpt-4o`, ...).
- **Configured models**: `gpt-5.6-luna` ($0.20 / $1.20 per MTok, cheap
  tier) and `gpt-5.6-sol` ($4.00 / $20.00 per MTok, flagship tier) — same
  generation, cheap/strong pair, deliberately not mixed with an older
  generation's mini/nano naming.
- **SDK / endpoint**: `openai` Python package, `client.responses.create(
  model=..., input=..., instructions=..., reasoning={"effort": ...})`,
  output via `response.output_text`. Confirmed live against the installed
  package (`openai==3.3.1` at verification time): `client.responses.create`
  exists; `openai.APIStatusError`/`RateLimitError`/`APIConnectionError`/
  `APITimeoutError` exist with the same shape as Anthropic's (both SDKs
  are Stainless-generated).
- **Usage fields**: `response.usage.{input_tokens, output_tokens,
  total_tokens, input_tokens_details.cached_tokens,
  output_tokens_details.reasoning_tokens}`. Confirmed via a documented
  example where `input_tokens + output_tokens == total_tokens` exactly —
  i.e. `reasoning_tokens` is a **breakdown within** `output_tokens`, not
  additional to it. `configs/pricing.yaml` deliberately does not set
  `reasoning_per_mtok` for OpenAI models as a result (see below).
- **Not verified this pass**: cache write/read pricing, `stop`-sequence
  parameter support on the Responses API, `top_k` support (not sent by
  this adapter — no confirmed equivalent).

### Google Gemini

- **Docs used**: `ai.google.dev/gemini-api/docs/{quickstart,
  text-generation,thinking,pricing,api-key,models}`, cross-checked against
  the installed `google-genai==2.19.0` package's actual types
  (`google.genai.interactions.Interaction`, `...usage.Usage`,
  `google.genai.errors`).
- **API surface**: `ai.google.dev`'s product docs state "The Interactions
  API is now generally available. We recommend using this API for access
  to all the latest features and models" — used here
  (`client.interactions.create`) over the older `client.models
  .generate_content`, which the installed SDK still exposes (confirmed
  both exist on the client) but which current docs no longer lead with.
  This is a genuine judgment call between two live, non-deprecated
  surfaces on the same installed SDK; revisit if it turns out to be wrong.
- **Configured models**: `gemini-2.5-flash-lite` ($0.10 / $0.40 per MTok)
  and `gemini-3.1-pro-preview` ($2.00 / $12.00 per MTok, for prompts
  ≤200k tokens) — both model ID *and* price confirmed together from the
  same official pricing-page fetch.
- **Auth**: `Client(api_key=...)`. Both `GEMINI_API_KEY` and
  `GOOGLE_API_KEY` are supported; if both are set, `GOOGLE_API_KEY` wins —
  confirmed identically from two independent official sources
  (`ai.google.dev/gemini-api/docs/api-key` and the `googleapis/python-genai`
  README) and directly matches `Secrets.resolved_google_api_key`'s
  behavior in `router/config.py`.
- **System prompt**: top-level `system_instruction=` (not nested, not
  `instructions=` like OpenAI, not `system=` like Anthropic).
  `temperature`/`max_output_tokens`/`top_p`/`top_k`/`thinking_level` nest
  under a `generation_config={}` dict.
- **Reasoning control**: `generation_config={"thinking_level":
  "low"|"medium"|"high"}` — 3 levels, vs. `router.config.EffortLevel`'s 5
  (`low|medium|high|xhigh|max`). `xhigh`/`max` clamp to `"high"`.
- **Usage fields**: `response.usage.{total_input_tokens,
  total_output_tokens, total_thought_tokens, total_cached_tokens,
  total_tokens}` — confirmed directly against the installed SDK's `Usage`
  Pydantic model fields (not just docs). Google's own cost guidance states
  a response's bill includes visible output *plus* thinking tokens
  combined — i.e. `total_thought_tokens` is additive to
  `total_output_tokens`, the opposite of OpenAI's breakdown relationship.
  `configs/pricing.yaml` sets `reasoning_per_mtok` equal to
  `output_per_mtok` for both Gemini models as a result.
- **Response shape**: the `Interaction` object has its own `status` field
  (`completed|failed|cancelled|incomplete|budget_exceeded|...`, confirmed
  as a `Literal` on the installed type) that can indicate failure
  *without the SDK raising a Python exception* — confirmed by inspecting
  the `interactions.create` return type, which is just `Interaction`
  (non-streaming) with no separate failure-signaling wrapper. The adapter
  treats `status in ("failed", "cancelled")` as `CompletionStatus.ERROR`
  even with no exception raised.
- **Error hierarchy**: `google.genai.errors.APIError` (base, `.code` is
  the HTTP status int) with `ClientError`/`ServerError` subclasses —
  confirmed directly against the installed package. No dedicated
  `RateLimitError` class the way Anthropic/OpenAI have one; a 429 is just
  a `ClientError` with `.code == 429`.
- **Not verified this pass**: cache pricing, `stop_sequences` equivalent
  (not sent by this adapter — no confirmed parameter name).

## What did NOT change

`router.models.schemas.{TokenUsage, GenerationConfig, NormalizedCompletion,
CompletionStatus}` and `router.models.base.ModelClient` — the normalized
schema and the interface both already supported this without modification.
`router.config.ModelEntry` — already provider-neutral (id, display_name,
role_hint, generation caps, `thinking`/`effort` controls); zero fields
added. `router.cost.calculator`, `router.storage.*`,
`router.evaluation.*`, `router.aggregation.*`, `router.policies.*`,
`router.routers.*`, `router.analysis.*` — none of these reference a
provider name; nothing there needed to change. This is what "the smallest
correct extension" looked like in practice: two new adapter modules, one
new (optional) registry helper (`router.models.registry.build_model_client`,
resolving the previously-unused `ProviderModelsConfig.adapter` dotted path
that's existed since Stage 0), a handful of new `Secrets` fields, a
provider-aware `AppConfig.require_api_key(provider=...)` (defaulting to
`"anthropic"` so every existing zero-arg call site is unaffected), and
config/pricing YAML entries.

## Explicitly not decided here

- The final cross-provider research pair — none of `gpt-5.6-luna`,
  `gpt-5.6-sol`, `gemini-2.5-flash-lite`, `gemini-3.1-pro-preview` has a
  `role_hint` set.
- Whether the eventual pilot even uses more than one provider — this ADR
  makes it *possible*, not decided.
- k, δ, or anything about routers/thresholds/calibration.
- Fable 5 was explicitly excluded from this pass (not required to
  demonstrate the abstraction).

## Consequences

- Cost figures for the four new models should be treated with the same
  "directionally correct, not reconciled" caveat ADR-0002 established for
  Anthropic, and are *more* provisional where noted above (cache pricing
  unverified for both new providers).
- `router.models.registry.build_model_client(config, provider, api_key)`
  is now the recommended way for higher-level code (pilot runners,
  experiment scripts) to get a client without hardcoding an adapter
  import — existing Anthropic-only call sites (`scripts/verify_stage0.py`,
  `scripts/run_model_pair_pilot.py`) still import `AnthropicModelClient`
  directly and continue to work unchanged; migrating them to the registry
  is optional follow-up, not required by this ADR.
- Adding a fourth provider means: one new adapter module implementing
  `ModelClient`, new `Secrets` fields if its auth differs, and new
  `configs/models.yaml` / `configs/pricing.yaml` entries — no changes to
  any research-logic layer.
