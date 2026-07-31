# Top harness-level inference techniques for insideLLMs (11 selected)

Research date: 2026-07-31

Scope: inference-time manipulations implementable primarily in a Python LLM harness, without training the target model. Reported effects are authors' results, not locally reproduced gains. Every strategy must be evaluated against one-shot and matched-compute baselines.

## Decision rule

Rank combines: empirical strength, benefit breadth, compatibility with insideLLMs, implementation risk, and whether the method improves the cost/latency/quality Pareto frontier rather than accuracy at any price.

## Selected top 10

| Rank | Method | Primary benefit | Evidence and limits | insideLLMs implementation |
|---:|---|---|---|---|
| 1 | Structured outputs + deterministic validation/repair | Reliability; fewer wasted retries | Strong engineering pattern; deterministic schema/runtime checks dominate ungrounded self-judgment. Repair adds a call only on invalid output. | Typed action/result schemas, validator chain, one bounded schema-focused repair. Extend `structured.py`, model adapters, and traces. |
| 2 | Stable-prefix prompt composition + exact provider/KV cache support | Lower TTFT/input cost with no quality loss | SGLang reports 74.1% cache hit and 1.7x average TTFT reduction in a production workload; vLLM states exact prefix caching does not alter outputs. Provider benefits depend on repeated exact token prefixes. | Canonical stable/dynamic prompt sections, deterministic tool/schema ordering, cache-control metadata, cached-token telemetry, tenant isolation. |
| 3 | Retrieve broadly, rerank narrowly, then lost-in-middle-aware assembly | Better grounding and lower prefill cost | RankRAG reranking ablation averaged 52.6 vs 49.8 across nine datasets. Lost-in-the-Middle observed >20-point drops when evidence moved to middle. Requires good first-stage recall. | Query-aware reranker protocol, dedupe/diversity policy, boundary ordering, protected source IDs, top-k budget. Reuse `contrib/retrieval.py`. |
| 4 | Self-consistency with sequential early stopping | Broad reasoning gain with conditional cost | Original paper: +17.9 GSM8K, +11.0 SVAMP, +12.2 AQuA, +6.4 StrategyQA over greedy CoT. Returns diminish on modern strong/easy tasks. | Parallel/sequential candidate sampling, answer normalizers, modal selection, agreement confidence, mathematically safe early stop. Reuse async model APIs and ensemble aggregation. |
| 5 | Deterministic-verifier-first Best-of-N | Quality for checkable tasks | GSM8K verifier reranking produced gains comparable to a 30x generator-size increase; too-large N can exploit verifier errors. Deterministic checks are safer than same-model judges. | Candidate generation + ordered verifier stack: schema/tests/exact constraints, custom scorer, blinded order-debiased judge fallback. Retain top-k vote and pass@N oracle metrics. |
| 6 | Confidence/disagreement-driven adaptive escalation | Better quality/cost frontier | Semantic entropy incorrect-answer AUROC 0.790 vs 0.691 naive entropy; FrugalGPT reports up to 98% cost reduction at matched best-model performance in its setups. Calibration drifts by model/domain. | Start cheap; escalate N, tools, verifier, or model only on low agreement/confidence. Log risk-coverage, thresholds, cost, and stop reason. Extend routing rather than replace it. |
| 7 | Program/tool execution with typed actions and objective feedback | Large gains on mechanically checkable work | PAL: 72.0 vs 65.6 GSM8K for same model; PoT reports about +12 points average across numerical tasks. Wrong formalization and unsafe execution remain risks. | Typed tool calls, allowlists, resource budgets, sandbox interface, policy checks, deterministic postconditions, separate transport retry from semantic repair. Upgrade current text-parsed ReAct path. |
| 8 | Tool-grounded critique-and-repair retaining the original | Correct verified failures without corrupting correct answers | CRITIC raised ChatGPT GSM8K PoT 72.5→78.2 and TabMWP 75.0→89.0 with tools. Intrinsic self-correction can degrade GPT-4 GSM8K 95.5→91.5→89.0. | Repair only after verifier/tool evidence; retain original; accept revision only if score/postconditions improve; bounded rounds and right→wrong telemetry. |
| 9 | Plan/execute DAG with parallel independent branches | Lower wall time and better decomposition | ReWOO-style separation reduces repeated observations/context; multi-agent evidence is strongest for genuinely separable parallel work, not generic chat/debate. | Typed plan DAG, dependency validation, parallel ready-node execution, compact observations, deterministic reducer, checkpoint/resume. Reuse chains/orchestration/async runners. |
| 10 | Budgeted tree/beam search over objective states | High upside on search-shaped tasks | ToT: 74% vs 4% on Game of 24, but bespoke/toy-task evidence; LATS improves code/web tasks when tests/environment reward are real. Weak LLM value functions can prune correct paths or reward-hack. | Generic state/propose/transition/value protocols, beam first, MCTS optional, node/token/time caps, transposition cache; require objective feedback by default. |
| 11 | Population-based evolutionary search over textual harness artifacts | Offline discovery of better prompts, playbooks, and handoffs | AlphaEvolve-style systems show that evolutionary search can improve reusable artifacts when fitness is executable or otherwise grounded. Promptbreeder and ShinkaEvolve support mutation/evaluation loops, but gains can overfit benchmarks or exploit weak evaluators. | Generic text candidate population, seeded parent selection, sync/async mutation, fitness evaluator, elitism, deduplication, lineage, generation/evaluation/time budgets, validation callback, best/history result. Offline/meta-optimization only—not a per-request default. |

## Explicit non-defaults

- Homogeneous multi-agent debate: at matched responses on GSM8K, 9-response debate scored 83.0 versus 88.2 for 9-answer self-consistency in a controlled reproduction. Keep only as an experimental strategy.
- Intrinsic self-reflection: revisions are not accepted without external evidence or an independently validated selector.
- Generic lossy prompt compression: selection/reranking comes first. Never prune code, schemas, numbers, citations, quotes, negation, or user constraints. Compressor latency must be included in end-to-end p95.
- Semantic answer cache: keep opt-in and provenance-aware because it can return stale/different answers. It is not equivalent to exact prefix/KV caching.
- PagedAttention, continuous batching, speculative decoding: important serving-engine controls, but not core insideLLMs algorithms. Expose configuration/telemetry adapters rather than reimplementing GPU kernels.

## Shared architecture

All eleven strategies operate on a common trace DAG:

```text
InferenceRequest
  → PromptParts(stable, dynamic, evidence)
  → Candidate(parent_id, output, normalized_answer, metadata)
  → Verification(score, passed, evidence, verifier_id)
  → TraceEvent(calls, tokens, latency, cost, tool state)
  → BudgetPolicy(next action or stop reason)
  → InferenceResult(answer, confidence, provenance, trace, spend)
```

Core protocols:

```python
class Proposer: async def sample(request, n) -> list[Candidate]: ...
class Normalizer: def key(candidate) -> str | None: ...
class Verifier: async def verify(request, candidate) -> Verification: ...
class ToolEnvironment: async def execute(action, limits) -> Observation: ...
class BudgetPolicy: def decide(trace, uncertainty, budget) -> Decision: ...
```

## Evaluation gates

1. Report task score against calls, input/output tokens, wall-clock p50/p95, TTFT, and dollars.
2. Compare matched calls/tokens; include one-shot, self-consistency, pass@N oracle, and deterministic-verifier upper bounds.
3. Keep calibration/tuning separate from test data; use paired bootstrap confidence intervals.
4. Measure wrong→right and right→wrong transitions for every repair method.
5. For judges: blind model identity, shuffle order, score both permutations, and validate against deterministic/human labels.
6. For tools/code: use hidden tests, resource-limited execution, and prompt-injection cases.
7. Adoption requires a Pareto improvement or an explicit high-quality mode; no average-only claims that hide regressions in code/math/exact-detail subsets.

## Primary sources

- Structured/tool agents and security: ReAct https://arxiv.org/abs/2210.03629 ; PAL https://proceedings.mlr.press/v202/gao23f.html ; PoT https://arxiv.org/abs/2211.12588 ; AgentDojo https://arxiv.org/abs/2406.13352
- Exact prefix caching/serving: SGLang https://arxiv.org/abs/2312.07104 ; vLLM prefix cache design https://docs.vllm.ai/en/stable/design/prefix_caching/
- Retrieval/context: RankRAG https://arxiv.org/abs/2407.02485 ; Lost in the Middle https://aclanthology.org/2024.tacl-1.9/ ; LongLLMLingua https://arxiv.org/abs/2310.06839
- Self-consistency: https://arxiv.org/abs/2203.11171
- Verifiers/process rewards: https://arxiv.org/abs/2110.14168 ; https://arxiv.org/abs/2305.20050
- Adaptive compute/confidence: https://arxiv.org/abs/2408.03314 ; https://www.nature.com/articles/s41586-024-07421-0 ; https://arxiv.org/abs/2305.05176
- Grounded repair: CRITIC https://arxiv.org/abs/2305.11738 ; Reflexion https://arxiv.org/abs/2303.11366 ; self-correction counterevidence https://arxiv.org/abs/2310.01798
- Search: ToT https://arxiv.org/abs/2305.10601 ; LATS https://proceedings.mlr.press/v235/zhou24r.html
- Evolutionary artifact search: AlphaEvolve https://arxiv.org/abs/2506.13131 ; Promptbreeder https://arxiv.org/abs/2309.16797 ; ShinkaEvolve https://arxiv.org/abs/2509.19349
- Debate counterevidence: https://arxiv.org/abs/2310.01798 ; https://proceedings.mlr.press/v235/smit24a.html

## Research artifacts

The source research notes and generated handbook for this document live outside
the repository and are not reproducible from a clone. The tracked, runnable
artifacts are:

- [`docs/INFERENCE_STRATEGIES.md`](../INFERENCE_STRATEGIES.md) — the implemented strategy reference
- [`benchmarks/inference_strategies.py`](../../benchmarks/inference_strategies.py) — the deterministic offline benchmark
