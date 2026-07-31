# insideLLMs productization roadmap

## Product position

insideLLMs is the model-neutral reliability and evaluation harness for detecting,
explaining, and improving LLM behavior under auditable budgets. It orchestrates
provider APIs and optimized serving engines; it does not compete with vLLM or
SGLang's kernels and schedulers.

## Architectural rule

New online inference behavior enters through `InferenceClient` and returns
`InferenceResult`. Existing provider models, registries, and middleware remain the
single construction, retry, cache, rate-limit, and cost-accounting path. Strategies
must not introduce another provider abstraction.

## Phase 1 — blessed inference path

Status: implemented in the current worktree.

- Adapt canonical `Model`/`AsyncModel` objects through `ModelProposer`.
- Expose one-shot, self-consistency, and verifier-first Best-of-N through
  `InferenceClient`.
- Reuse model registry and middleware configuration with
  `InferenceClient.from_model_config`.
- Preserve model, latency, token, candidate, trace, stop, and provenance records.
- Ship an executable offline example.

Acceptance gate:

- Public imports work from `insideLLMs` and `insideLLMs.inference`.
- Sync and async models behave identically at the strategy boundary.
- The full test suite, Ruff, format, and mypy pass.

## Phase 2 — consolidate overlapping execution surfaces

1. Inventory retry, exact/semantic cache, routing, ensemble, and orchestration APIs.
2. Name one canonical implementation for each concern.
3. Adapt compatible old APIs to the canonical implementation.
4. Deprecate duplicate entry points with migration examples; do not break them
   silently.
5. Keep provider construction in `models`/registry and cross-cutting execution in
   runtime middleware.

Acceptance gate:

- One ownership table names the canonical module for every execution concern.
- Duplicate APIs either delegate or carry a dated deprecation path.
- Existing public behavior remains regression-tested.

## Phase 3 — matched-compute evidence suite

Build task datasets and compare one-shot against each optional strategy with the
same model and normalized calls/tokens. Record:

- task score and subset regressions;
- calls, input/output/cached tokens, cost;
- TTFT and wall-clock p50/p95;
- pass@N, deterministic-verifier pass rate, and oracle bound;
- cache-hit rate and context reduction;
- confidence calibration and escalation rate.

No strategy becomes a default from paper evidence or deterministic smoke tests
alone. Promotion requires a held-out improvement without unacceptable subset,
cost, or latency regressions.

## Phase 4 — CLI and configuration

Add `insidellms infer` only after the Python contract stabilizes. Support:

- one-shot and named strategy selection;
- existing model/pipeline YAML blocks;
- JSONL result and trace output;
- explicit budgets;
- dry-run/config validation;
- non-zero exit for verifier or budget policy failure.

Acceptance gate: the CLI output round-trips through `InferenceResult` without a
second result schema.

## Phase 5 — stable release contract

- Publish the canonical API map and migration guide.
- Mark experimental versus stable strategies explicitly.
- Add compatibility tests for public imports and serialized result records.
- Document security boundaries for tools, retries, timeouts, and tenant cache keys.
- Cut a release only after real-model matched-compute runs are reproducible in CI
  or a documented benchmark environment.

## Non-goals

- Multi-agent debate by default.
- Hidden LLM judges presented as objective verification.
- Per-request evolutionary search.
- Reimplementing serving-engine batching, KV management, or speculative decoding.
- Adding strategy-specific provider clients, retry loops, or result envelopes.
