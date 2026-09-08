# Matched-compute assurance

Matched-compute reports separate declared limits, observed usage, and verified
request-cap assurance. These are different claims.

`matching.declared_limits_match` reports whether the two `ComputeProfile`
declarations match. The runner rejects unequal declared call ceilings or unequal
per-call output-token limits before execution.

`matching.output_limit_assurance` is `"verified"` only when every variant has
complete evidence from the instrumented request dispatch boundary: its observed
call count equals the evidence count, every dispatch carries the declared cap,
and every call completed with known output usage within its individual cap.
Equal declarations, equal observed usage, a shared model, or favorable custom
`Spend` and provenance values do not prove which cap was sent. Missing bindings,
unsupported usage, incomplete calls, and incomplete coverage yield `"unknown"`.
`matching.max_output_tokens` and `MatchedComputeReport.output_tokens_matched`
are true only under verified assurance.

Bind a supported flat generation keyword explicitly:

```python
from insideLLMs.inference import InferenceClient, OutputLimitBinding

client = InferenceClient(
    model,
    generation_kwargs={"max_tokens": 16},
    output_limit=OutputLimitBinding("max_tokens", 16),
)
```

Only `max_tokens` and `max_completion_tokens` are supported. Exactly one of these
aliases may appear on a bound request. Each dispatch copies and revalidates its
kwargs against the binding and enclosing profile; mutations cannot silently
change the cap, and the runner never clamps requests. `one_shot_baseline` checks
the binding both at construction and before execution. Missing bindings preserve
ordinary execution but cannot verify caps. `from_model_config` accepts the same
optional binding.

Observed output-token totals remain under `compute.output_tokens` and need not be
equal when the declared limits are equal. A reported zero is serialized as `0` only
when canonical response usage contains an explicit zero. Omitted provider usage
is serialized as `null`; it is not treated as known zero. Positive custom-runner
totals remain visible as reported observations, but do not increase assurance.

The runner rejects invalid spend before summing results, including negative or
non-finite counters and token usage with zero calls. It also rejects an aggregate
whose observed output tokens exceed `observed calls * declared per-call cap`.
Instrumented dispatch also rejects a bound-cap mismatch or a call beyond the
declared ceiling before invoking the model, and rejects a response exceeding its
individual cap. Catching these errors inside a callback does not erase them from
the scope. A response violation is detected after generation; this is not a
guarantee that the provider obeys the sent limit.

## Scope and trust boundary

Each variant invocation gets a separate context-local collector. Child tasks and
instrumented clients, including model-backed judges and verifiers, share its call
ceiling. Exceptions and cancellation leave unsettled reservations; calls are never
refunded. A timed-out synchronous worker may continue running, but its late result
cannot settle into a subsequent scope. The scope is reset in `finally`.

When the evidence count matches reported calls and all response output counts are
known, the report uses their total, including judge/verifier output. Incomplete
scoped usage is serialized as `null`, including per-case usage and ratios. This does
not establish complete input-token, latency, cost, or billing accounting. Custom
unobserved results retain Task 8's reported usage semantics and unknown assurance.

Arbitrary Python callbacks are trusted code. They can call models, tools, or foreign
clients outside instrumentation and can lie about observed calls; the collector is
not a sandbox or proof against malicious callbacks. Strict callers must restrict
execution to instrumented clients and report every model-backed call, including
judges and verifiers. Unobserved calls cannot contribute evidence; honest coverage
gaps make assurance unknown. Candidate metadata is presentation/accounting data,
never an authority for request-cap assurance. There is no USD or billing enforcement.

## Migration note

In older report artifacts, `matching.max_output_tokens: true` meant only that the
two profile declarations were equal. Consumers must not interpret such historical
values as verified request-cap evidence. New artifacts expose that declaration result
separately and reserve the legacy boolean for verified assurance.
