# Pre-dispatch run budgets

Budgets are opt-in admission control for one invocation of `run` or `harness`.
One ledger covers every supported subject and judge model, probe, retry and
worker in that invocation. An unaffordable request is rejected before SDK
dispatch, and execution stops with partial results and a nonzero CLI exit.
Ordinary configurations without `budget` retain their existing behavior.

## What the bound means

**The input bound and prices are caller-supplied assumptions.**
`maximum_input_tokens` must cover the provider's maximum **billable input
context per attempt**, including message framing, system content and any other
billed input. It is not an expected prompt size, a character-count estimate, or
a limit that insideLLMs can send to the provider. The caller must establish that
the chosen model/provider enforces this bound and that the supplied rates cover
the complete supported request. insideLLMs does not fetch or certify prices or
provider context limits.

Every attempt reserves the whole supplied input-context bound, plus its enforced
output cap, at the fixed rates. This is deliberately conservative for short
prompts. The provider must honor the configured output parameter for **all billed
output**, including reasoning when applicable; otherwise that model/parameter
combination cannot be used under this policy.

```text
quote = (maximum_input_tokens × input_cost_per_million
         + outgoing_output_cap × output_cost_per_million) / 1,000,000

admit only if settled + uncertain + reserved + quote <= allowance
```

The guarantee is admission under these explicit, fixed assumptions. It is **not
an invoice cap**, a provider-account quota, or a sandbox for arbitrary Python
code. Price changes, false bounds, charges outside supported request categories,
and calls outside this invocation are outside that guarantee.

## Configuration

Add a top-level `budget` block to the canonical configuration. The following is
a template: replace every placeholder with a verified policy value before use.
The allowance shown is illustrative and does not authorize any provider call.

```yaml
budget:
  currency: USD
  scope: invocation
  allowance: "5.00"
  prices:
    - provider: openai
      model: "<exact requested model identifier>"
      endpoint: https://api.openai.com/v1
      pricing_id: "<dated source or identifier of your fixed pricing policy>"
      input_cost_per_million: "<input price upper bound>"
      output_cost_per_million: "<output price upper bound>"
      maximum_input_tokens: <maximum billable input context per attempt>
      maximum_output_tokens: <enforced maximum billable output per attempt>
      output_token_parameter: max_completion_tokens
```

Prices must uniquely identify each provider/model pair. Monetary values must be
finite and nonnegative, with at most 24 digits and 12 decimal places; token bounds
are positive integers no larger than one billion. Zero prices are allowed only
when supplied explicitly. Decimal strings survive resolved YAML/JSON snapshots.
`pricing_id` records provenance supplied by the caller; it is not independently
verified. There is no unknown-model or estimated-price fallback.

For Anthropic, the exact endpoint is `https://api.anthropic.com` and the output
parameter is `max_tokens`. OpenAI supports the explicitly selected `max_tokens`
or `max_completion_tokens` parameter. A smaller positive per-call cap is allowed;
larger, absent-value, conflicting or unsupported cap overrides are rejected.

SDK `max_retries` is set to zero when the factory creates budgeted models.
An explicitly configured nonzero value is rejected. Runtime checks also inspect
the client's actual endpoint and retry setting immediately before dispatch.

## Initial support

Supported paid operations are standard OpenAI chat-completions and Anthropic
messages, using plain text `generate` and `chat` calls with one completion.
Async runners and async pipeline methods can execute these guarded synchronous
adapters in worker threads. Model/probe batches reserve separately for each
actual call. Cache hits make no reservation. Builtin DummyModel is offline and
needs no price entry, so `prices: []` is valid for a completely offline run.

Strict mode rejects streaming, remote batch jobs, embeddings, reranking, tools,
multimodal payloads, multiple completions, custom gateways/headers, extra request
bodies and unsupported request parameters before dispatch. OpenRouter, Gemini,
Cohere and local-serving adapters are not yet supported for paid admission.
There is no silent fallback to ordinary execution.

Canonical builtin model/probe factories are checked before construction. Replaced
registry entries, registration defaults, custom model/probe objects, and callback
validators are rejected. Programmatic runner plumbing must carry the same ledger
already bound by the model factory; passing a ledger alongside an unrelated
model cannot enable assurance. This validates the supported application path,
not arbitrary code executing in the same Python process.

Builtin judge probes accept a nested model configuration and share the subject's
ledger:

```yaml
probe:
  type: judge
  args:
    judge_model:
      type: openai
      args:
        model_name: "<exact model also present in budget.prices>"
    judge_kwargs:
      max_completion_tokens: 100
```

All subjects, judges and their configured generation bounds are preflighted
before any subject call. The usual top-level and per-probe generation precedence
still applies. Unsupported arbitrary agent/plugin callouts require separate
integration before they can participate in a budgeted invocation.

## Settlement and failures

Complete, matching provider usage settles the cost at the supplied rates and
releases the verified unused amount. Missing/incomplete usage or unmatched model
identity retains the full reservation as uncertain cost. Cache-specific
Anthropic usage also retains the full reservation; this version does not claim
to reconcile cache billing tiers. The original string-returning model API is
unchanged; ledger accounting consumes the SDK response before text extraction.

Every transport exception, disconnect or cancellation after dispatch retains the
full possible liability. Cancellation of an async waiter does not stop its
synchronous worker. While that worker runs, its reservation remains held; only
its completed, trusted response can settle it. A missing response remains
uncertain. An outer retry must reserve another complete attempt. Budget denials
are nonretryable. Duplicate settlement never changes the totals twice.

Reported costs or token counts beyond their reserved bounds are recorded and
raise a visible budget breach; no more requests are admitted. Accounting never
clamps the reported cost to pretend an excess was prevented.

The ledger is invocation-scoped and in memory. Budgeted `--resume` is refused
until durable reservation recovery exists. A new invocation has a new allowance;
multiple processes do not share a cap. Partial-run budget snapshots distinguish
`settled`, `uncertain`, and still-`reserved` liability. They are diagnostic
evidence, not a durable billing ledger or provider receipt.
