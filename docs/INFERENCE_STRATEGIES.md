# Inference-time harness strategies

`insideLLMs.inference` is an async-first, provider-neutral package for eleven
composable harness strategies. It keeps candidate, verification, trace, budget,
spend, provenance, and result records consistent across strategies while leaving
the existing `contrib` APIs backward compatible.

## Strategies

1. Deterministic validation with one bounded structured repair.
2. Stable-prefix composition with canonical tool/schema ordering and tenant-scoped cache keys.
3. Broad retrieval, query-aware reranking, deduplication, and boundary-aware assembly.
4. Self-consistency with a mathematically safe sequential stop.
5. Deterministic-verifier-first Best-of-N with pass@N, oracle, top-k, and optional order-debiased judging.
6. Confidence-driven escalation with risk, threshold, stop, call, and cost traces.
7. Typed, registered, allowlisted tools with transport limits and objective postconditions.
8. Evidence-grounded repair that retains the original and accepts validity before score improvements.
9. Validated plan DAGs with concurrent ready nodes, deterministic reduction, and checkpoint input.
10. Budgeted beam search with objective values and a transposition cache.
11. Seeded population search over prompts, playbooks, and handoff templates.

## Recommended model-backed API

Use `InferenceClient` for normal application code. It adapts the canonical
`Model` and `AsyncModel` interfaces, preserves provider metadata, and returns an
`InferenceResult` for every supported online strategy.

```python
from insideLLMs import InferenceClient, OpenAIModel

client = InferenceClient(
    OpenAIModel(model_name="gpt-4o-mini"),
    generation_kwargs={"temperature": 0},
)

one_shot = await client.generate("What is the capital of France?")
self_consistent = await client.self_consistency(
    "What is 37 * 41?",
    max_samples=5,
)
```

For objective selection, make verification explicit rather than hiding an LLM
judge behind a preset:

```python
from insideLLMs import Verification
from insideLLMs.inference import VerifierSpec

def valid_integer(candidate):
    passed = candidate.output.strip().isdigit()
    return Verification("integer", score=float(passed), passed=passed)

selected = await client.best_of_n(
    "What is 37 * 41? Return only the integer.",
    n=4,
    verifiers=(VerifierSpec("integer", valid_integer, hard=True),),
)
```

`InferenceClient.from_model_config(...)` reuses the existing registry and
middleware pipeline configuration; it does not introduce another provider or
retry system. Run `python -m examples.inference_client` for an offline example.

All callbacks may be plain functions or async functions. Use the async APIs in
applications. `insideLLMs.inference.run_sync` is safe for scripts and deliberately
rejects use inside an already-running event loop.

Tool retries are opt-in (`ToolAction.idempotent=True`). A synchronous callback
cannot be safely killed once a worker thread has started, so any strategy with an
active hard time budget requires asynchronous callbacks and fails before execution
otherwise. Evolution has no time limit by default; set `max_seconds` only with
async evaluators/selectors/mutators. DAG failures cancel and await sibling tasks
before returning. Call budgets include the final DAG reduction and every search
proposal, transition, and value callback.

## Evolution is offline only

`evolve_artifacts` is a meta-optimization workflow, not a per-request routing
default. Selection uses only `fitness`. An evaluator may return `Fitness` with a
separate `validation_fitness`, or callers may supply `final_validation`; neither
validation value participates in parent selection. Candidate IDs, text, fitness,
parents, lineage, generation, metadata, seed, history, and budget stop reason are
preserved for auditability.

Use a held-out evaluator, hidden tests where possible, and a final untouched test
set. Weak fitness functions invite reward hacking and benchmark overfitting.

## Offline example

```bash
/Users/k/.local/share/mise/installs/python/3.12.12/bin/python benchmarks/inference_strategies.py
```

The example uses deterministic plain callbacks and reports calls, a safe early
stop, exact-prefix cache reuse, simulated diffable latency fields, zero cost, and
the evolutionary evaluation budget. Real adoption runs must additionally compare
one-shot and matched-compute task score, calls/tokens, wall-clock p50/p95, TTFT,
cost, pass@N, deterministic-verifier bounds, and subset regressions.
