# Matched-compute evaluation

`insideLLMs.analysis.matched_compute` compares an inference strategy with a simple
baseline under an executable compute contract. It is the Phase 3 evidence path;
it does not make a strategy a default by itself.

## Comparison contract

Every variant declares a `ComputeProfile`. A comparison is accepted only when:

- both variants share one model executor (`executor_id`, normally `id(client)`);
- both declare the same number of expensive model calls per example;
- both declare the same per-call output-token cap;
- every variant's *observed* `Spend.calls` stays within its declared ceiling on
  every example and trial (a judge legitimately skipped on some inputs may
  observe fewer calls than declared; observing more always fails);
- both variants report model provenance, and the provenance matches.

The observed-versus-declared check stops a strategy from hiding model-backed
judge calls counted in `Spend.calls`: if a Best-of-N run makes two generations
plus two judge calls, it must declare at least four. Exceeding the declared
ceiling raises `ComputeMismatchError`; nothing is silently labeled matched.
Caveat: verifier callback invocations are NOT part of `Spend.calls` (the
harness cannot know which verifiers are model-backed), so model-backed
verifiers must be declared in `generated_calls` by the caller and are surfaced
for audit via the strategy's `provenance["verifier_invocations"]` count rather
than enforced by this check.

Realized token counts are recorded rather than forced equal, because stochastic
outputs differ in length. Fairness comes from the declared cap; the realized
ratios are diagnostics.

For an N-call strategy, the standard baseline is N independent one-shot results.
The report includes the mean one-shot score, pass@N, and the oracle upper bound.
The oracle is diagnostic only and is not a deployable selector.

Repeated trials alternate execution order AB/BA. Ordering is balanced only when
`trials` is even; with an odd count the report sets `order_balanced` to false and
the artifact carries that flag, so residual ordering bias is visible rather than
assumed away.

## Timing semantics

Two timing families are reported and they are **not** interchangeable:

- **Wall seconds** — measured by the runner around each variant invocation. This
  is the comparable statistic, reported as total/p50/p95 plus call throughput.
- **Provider seconds** — the sum of provider-reported latencies inside `Spend`.
  Concurrent baseline samples sum their individual latencies while a concurrent
  strategy reports a single wall-clock-style figure, so these are not the same
  statistic across variants. The artifact reports them with
  `"comparable": false` and they must not be used for a latency verdict.

## Quick smoke test

```bash
python -m examples.matched_compute_evaluation > matched-compute.json
```

This deterministic example uses `InferenceClient`, `ModelResponse` metadata, an
N-sample one-shot baseline, Best-of-N, and exact-match evaluation. It proves the
execution lifecycle, accounting, matching checks, and JSON artifact shape only.
It is explicitly marked `offline-smoke-only` and is not evidence of model quality.

## Live-model use

Reuse the existing model configuration path; do not construct another provider
client:

```python
from insideLLMs.analysis import (
    one_shot_baseline,
    run_matched_compute,
    single_result_variant,
)
from insideLLMs.analysis.evaluation import ExactMatchEvaluator
from insideLLMs.inference import InferenceClient

client = InferenceClient.from_model_config(existing_model_config)
report = await run_matched_compute(
    held_out_examples,
    baseline=one_shot_baseline(
        client,
        samples=4,
        max_output_tokens_per_call=256,
        cost_available=True,
    ),
    strategy=single_result_variant(
        "best-of-4",
        lambda request: client.best_of_n(
            request,
            n=4,
            verifiers=objective_verifiers,
        ),
        client=client,
        generated_calls=4,
        max_output_tokens_per_call=256,
        cost_available=True,
    ),
    evaluator=ExactMatchEvaluator(),
    trials=6,
)
artifact = report.to_dict()
```

Use an even `trials` count for balanced AB/BA ordering. Set `cost_available=True`
only when the provider actually reports cost; otherwise the artifact emits `null`
cost and `comparable: false` rather than presenting a defaulted zero as data.

Verifiers used during selection must not read held-out references. References are
used only by the post-generation evaluator. Use repeated trials and representative
subsets; a single prompt or deterministic model is not a quality benchmark.

## Artifact contents

The JSON-ready report records:

- mean baseline and strategy score plus score delta;
- subset regression counts;
- one-shot pass@N, strategy pass rate, and oracle score;
- model calls (observed and declared) and declared output-token budgets;
- realized input/output token totals and ratios;
- cost totals when reported, else `null` with `comparable: false`;
- wall-time total and p50/p95, plus call throughput;
- provider-reported time, explicitly marked non-comparable;
- matching flags for calls, models, output-token budget, and order balance;
- per-example/per-trial scores, pass vectors, spend, model identity, wall time,
  and regression status.

Evaluator scores are validated as finite before aggregation, so the artifact
serializes under `json.dumps(..., allow_nan=False)`.

## Known gaps

The current canonical `ModelResponse`/`InferenceResult` envelope does not expose
streaming TTFT, cached-token counts, or cache-hit rate. The report emits TTFT as
`null` rather than fabricating a value. Context reduction, confidence calibration,
and escalation-rate experiments remain later Phase 3 slices. Per-call cost
attribution is coarser than per-call availability tracking.

No optional strategy should become a default until reproducible live-model runs on
held-out datasets show a useful quality improvement without unacceptable subset,
latency, token, or cost regressions.
