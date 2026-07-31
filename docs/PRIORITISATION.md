# insideLLMs prioritisation checklist

Audit-derived prioritisation of work across the repository. Baseline facts, ranked
work items (P0–P3), split/extract candidates, and the execution-order checklist.

## 0. Repo shape (baseline facts)

| Layer                      | Files     | Total lines | Docstring % | Real code lines | Real code %          |
|----------------------------|-----------|-------------|-------------|-----------------|----------------------|
| contrib/                   | 45        | 104,710     | 71%         | 23,571          | 23%                  |
| root modules               | 39        | 56,436      | 70%         | 12,889          | 23%                  |
| runtime/                   | 20        | 15,239      | 50%         | 6,373           | 42%                  |
| analysis/                  | 8         | 15,659      | 57%         | 5,400           | 34%                  |
| cli/                       | 31        | 6,258       | 3%          | 5,204           | 83%                  |
| inference/                 | 18        | 2,156       | 3%          | 1,799           | 83%                  |
| nlp/                       | 16        | 11,134      | 80%         | 1,679           | 15%                  |
| probes/                    | 10        | 11,596      | 82%         | 1,638           | 14%                  |
| models/                    | 9         | 8,661       | 79%         | 1,528           | 18%                  |
| supply-chain pkgs combined | ~25       | ~1,670      | low         | ~900            | high                 |
| insideLLMs total           | 236       | ~248k       | 67.4%       | ~63k            | 25.5%                |
| tests                      | 226 files | ~96k        | —           | —               | 6826 pass / 161 skip |

Product position (from docs/PRODUCTIZATION_ROADMAP.md): model-neutral
reliability/evaluation harness; not a serving engine; new online behaviour via
InferenceClient.

Killer path in README is real: `insidellms diff` / harness artefacts /
deterministic runs (runtime/diffing.py ~1011 lines, CLI diff, receipt).

Base install is sane: only pyyaml. Providers/nlp/viz/signing are extras.

## P0 — Fix or stop claiming (correctness / trust)

These either misrepresent security or inflate product claims.

### P0.1 TUF dataset verification is fake even when tuf imports

Evidence: `insideLLMs/datasets/tuf_client.py:43-75` imports
`tuf.ngclient.Updater` only as F401, never constructs it, always writes a temp
mock JSON, and if import succeeded sets `"status": "verified"` /
`"method": "tuf.ngclient"`.

Work: Implement real Updater fetch+verify, or refuse with error until
implemented; never emit verified for mock path. Tests must not treat
import-success as proof.

Why first: This is a silent integrity bypass labelled as supply-chain
verification.

### P0.2 SCITT receipt "verify" is field-presence, not crypto

Evidence: `transparency/scitt_client.py:97-111` checks status==success, digest
string equality, and non-empty dict. No signature/Merkle/receipt crypto.

Work: Either wire real SCITT receipt verification or rename to
`receipt_looks_well_formed` and stop CLI/docs language of "verify".

Split? Yes — with the rest of trust stack (below).

### P0.3 DSSE envelopes can be unsigned forever

Evidence: `attestations/dsse.py` builds envelopes with `signatures: []` by
default; comment says verification is "signing layer". Cosign wrapper is real
CLI (`signing/cosign.py`) but optional and not required by envelope build.

Work: Fail closed on verify paths when signatures empty; document unsigned
envelopes as draft-only.

### P0.4 Name inflation on "semantic" / "attention" / "safety"

| Claim surface                | What it actually is                              | Evidence                                    |
|------------------------------|--------------------------------------------------|---------------------------------------------|
| `SemanticSimilarityEvaluator`| Weighted Jaccard + BoW cosine + token F1         | analysis/evaluation.py class docstring      |
| `AttentionAnalyzer`          | Heuristic simulation of 4×2 heads from token overlap | contrib/introspection.py ~1030+         |
| `safety.py`                  | Regex/keyword heavy                              | 41 regex/keyword hits, 0 embedding          |
| HF streaming                 | Simulated single-chunk stream                    | models/huggingface.py comments ~248, 615-696|
| `ParallelStep`               | Sequential loop labelled parallel                | contrib/chains.py execute runs `for step in self.steps` |

Work: Rename or hard-label "heuristic/approx" in public API and README feature
list; do not present as model-internals analysis.

### P0.5 Builtin benchmarks are toy (87 examples total)

Evidence: Live call of all 13 `create_*_dataset` helpers → 5–10 examples each,
87 total.

Work: Either mark builtin-smoke-only everywhere (registry, CLI benchmark, docs)
or ship loaders for real public sets (GSM8K/MMLU/etc.) with license/cache.
Matched-compute already admits offline-smoke-only; datasets should match that
honesty.

## P1 — Product core (align with stated mission)

Do these before more contrib surface.

### P1.1 One evaluation story: probes × evaluators × inference × matched-compute

Evidence of fragmentation:

- Probe hierarchy: Probe / ScoredProbe / ComparativeProbe / AgentProbe /
  JudgeScoredProbe (probes/base.py, judge.py)
- Separate analysis.evaluation.Evaluator family + JudgeModel
  (analysis/evaluation.py ~3497 lines)
- New run_matched_compute + InferenceClient (Phase 3 tracer; 8 importers only)
- Classic ProbeRunner / ModelPipeline still the common path (22 / 10 importers
  vs InferenceClient 8)

Work:

1. Table mapping "when to use probe vs Evaluator vs matched_compute" (docs +
   CLI doctor).
2. Single score envelope: finite score, pass flag, details, spend — probes and
   Evaluators both emit it.
3. Wire matched-compute into CLI (`insidellms compare-compute` or under
   benchmark) on held-out live runs.
4. Keep Phase 3 live-provider datasets + TTFT/cache fields as next evidence
   slice (roadmap already says incomplete).

### P1.2 Finish dual model stacks consolidation (Phase 2 incomplete)

Evidence:

- Canonical: models.Model + runtime.pipeline middleware + InferenceClient
  (ownership doc is good).
- Parallel stack: contrib.adapters (OpenAIAdapter/AnthropicAdapter own SDK
  clients, separate ModelRegistry/FallbackChain) — only tests import it, but
  public via package size.
- Lazy exports from root init pull large contrib agent/HITL/routing/deployment
  surfaces (160 lazy names; hitl 21, deployment 15, agents 12).

Work: Deprecate contrib.adapters toward models.* + pipeline; stop expanding
adapter API. Enforce ownership tests beyond inference (already have
tests/inference/test_architecture.py).

### P1.3 Harness/diff CI path is the crown jewel — harden, don't dilute

Evidence: README centres on `insidellms diff --fail-on-changes`;
runtime/diffing.py is substantial real code; CLI has harness, diff, compare,
attest, sign, verify.

Work: Golden-path integration test that is not coverage-padding: real
DummyModel run → artefact dir → diff exit codes; document stability matrix
against STABILITY.md. Prefer this over new probes.

### P1.4 Cost/token/latency accounting must be one path

Evidence: Spend lives on inference results; pipeline has cost middleware;
matched-compute now honest about null cost; provider latency ≠ wall time
(fixed in Phase 3 docs).

Work: Ensure classic ProbeRunner results expose the same Spend fields or an
adapter; otherwise A/B via probes vs matched-compute is incomparable.

### P1.5 Provider completeness gaps

Evidence: OpenAI/Anthropic/Gemini/Cohere/Local(Ollama/vLLM/llama.cpp)/
OpenRouter(wrapper)/HF real SDK usage present; HF streaming simulated;
OpenRouter is thin subclass (good).

Work: Document capability matrix (stream, tools, async, cost, tokens) per
provider; fix HF stream contract or expose supports_streaming=False.

## P2 — High-value product work (after P0/P1)

### P2.1 Live matched-compute evidence (Phase 3 remainder)

- Held-out tasks, even trials, live cost when provider gives it
- Self-consistency call-matching (early_stop issue already noted in prior audit)
- Do not promote Best-of-N/self-consistency to defaults without this

### P2.2 Judge path integrity

- JudgeScoredProbe / JudgeEvaluator / best_of_n judge calls must all hit Spend
  + model provenance (Phase 3 partially addresses for matched_compute only)
- Blind references: verifiers must not see gold labels (doc already says this;
  enforce in API)

### P2.3 Statistics module is real but hand-rolled

Evidence: Welch t, bootstrap CI, Bonferroni/Holm/FDR, Cohen's d, power
analysis in analysis/statistics.py (~1873 lines); normal CDF via math.erf (OK).

Work: Property tests against scipy on a grid for t/FDR/bootstrap; or optional
scipy backend. Don't expand custom stats further without that.

### P2.4 CLI surface is large (23 commands) vs thin golden path

Commands include: attest, benchmark, compare, diff, doctor, export,
generate_suite, harness, info, init, interactive, list, optimize_prompt,
quicktest, report, run, schema, sign, trend, validate, verify, welcome…

Work: Promote 5: quicktest, init, run/harness, diff, doctor. Mark rest
experimental in --help and STABILITY_MATRIX.

### P2.5 Docstring mass is a maintainability tax

Evidence: 167k docstring lines (67%); many 1k–3k line files are essay+examples
with thin code (resources.py 1185 lines / ~69 code; several contrib files 80%+
doc).

Work: Policy: move long tutorials to docs/; keep API docstrings short; stop
generating coverage tests against docstring-only branches. Not user-facing
priority but blocks every review.

## P3 — Test suite and packaging quality

### P3.1 Coverage-driven test inflation

Evidence: 70 coverage / branch_coverage files; 2026 tests inside them; file
headers say "Supplemental tests to increase coverage". Full suite 6826 pass in
~80s — density is high relative to ~63k code lines (~9 tests per 100 code
lines) but much is branch-tickling.

Work:

- Gate CI on a curated marker (`@pytest.mark.core`) for
  harness/diff/models/inference/matched_compute
- Stop adding `*_coverage.py` files; replace with behavioural tests
- Keep coverage metric but don't let it drive API design

### P3.2 Skips hide optional capability truth (161)

Almost all skips: NLTK/spaCy/sklearn/gensim not installed; OpenTelemetry;
anthropic; Redis; Plotly.

Work: CI job matrix: base, nlp, providers, signing. Doctor should report the
same. README already says opt-in — CI should not pretend nlp is green when
skipped.

### P3.3 Stale audit docs

docs/AUDIT_FINDINGS.md still discusses caching_unified.py / three-file cache
layout; tree now has single caching.py. Treat AUDIT_* as historical or refresh.

## Split / extract candidates (hypothetical other frameworks)

Justification rule used: belongs outside if (a) not required for
harness/diff/eval mission, (b) little/no import from core, (c) could version
independently, (d) different user persona.

### SPLIT A — insideLLMs-contrib (or drop from default wheel)

What: Nearly all of insideLLMs/contrib/ (~104k lines, ~24k code).

Why: Static imports from core/examples almost only touch synthesis (CLI
generate_suite), security.injection_engine (via injection.py),
benchmark/prompt_utils (examples). Rest is test-only or lazy-exported public
candy (agents, HITL, routing, deployment, orchestration, distributed,
steering, …).

How: Extra package `pip install insidellms[contrib]`; keep lazy exports with
DeprecationWarning pointing to new distro; do not import contrib from core
(already mostly true).

Keep in core from contrib-ish behaviour: injection testing if marketed for
jailbreak probes (or move probes/attack to own thin module); synthesis only if
CLI generate_suite stays.

### SPLIT B — insideLLMs-nlp

What: insideLLMs/nlp/* (~11k lines, ~1.7k code, 80% docs).

Why: Optional deps (nltk/spacy/sklearn/gensim); 161 skips dominated by these;
used mainly as word-overlap helpers inside contrib heuristics and one
evaluation import. Reimplements standard NLP poorly relative to upstream libs.

How: Already behind extra nlp; physically move package; core keeps tiny
normalize_text / token F1 needed for evaluators.

### SPLIT C — insideLLMs-trust (supply chain)

What: attestations/, crypto/, signing/, transparency/, publish/, policy/,
datasets/tuf_client, privacy/, contrib evalbom/openvex, CLI attest/sign/verify.

Why: Different standards (in-toto, SCITT, TUF, ORAS, cosign), different
release cadence, currently partial (P0.1–P0.3). Evaluation harness users don't
need OCI push.

How: Separate package; core emits digests/manifest only; trust package
consumes run dirs. Cosign shell-out is the only production-grade piece today —
keep that thin.

### SPLIT D — insideLLMs-viz / reporting

What: analysis/visualization.py (4113 lines) + export format zoo + HTML report
builders.

Why: Optional matplotlib/seaborn/plotly; product core needs JSON artefacts +
maybe one HTML template; interactive viz is a different product.

How: Already visualization extra; extract module; CLI report becomes plugin.

### SPLIT E — Agent / orchestration lab (subset of contrib)

What: contrib/agents, chains, orchestration, hitl, deployment, distributed,
LangChain integration.

Why: Explicitly "research" in ownership doc; duplicates LangGraph/Temporal/
etc.; ParallelStep not even parallel; deployment is FastAPI serving — outside
"we don't compete with vLLM" positioning if it grows.

How: Separate research repo; root lazy imports warn.

### SPLIT F — Do not split (load-bearing core)

Keep together:

- models/, registry/, runtime/ (pipeline, diffing, receipt, runners,
  reproducibility)
- types, schemas (artefact contracts), results, resources (atomic writes)
- probes/ (behaviour tests) + thin eval metrics
- analysis/evaluation.py + statistics.py + matched_compute.py + comparison.py
- inference/ (strategies on top of Model)
- cli golden path + caching/retry/rate_limiting/cost_tracking
- trace/tracing as needed for artefact provenance

## Prioritised work checklist (execution order)

### Now (blockers / honesty)

- [x] P0.1 Fix or disable TUF mock-as-verified — `fetch_dataset` now refuses
      without `allow_mock=True` and always labels proofs `status="mock"` /
      `verified=False`
- [x] P0.2 Downgrade SCITT verify claims / implement crypto — renamed to
      `receipt_looks_well_formed` (deprecated `verify_receipt` alias);
      docstrings/policy language now say structural check only
- [x] P0.3 Fail closed on unsigned DSSE in verify CLI — `verify-signatures`
      errors on zero attestations; dsse.py documents unsigned envelopes as
      draft-only (signing is detached via cosign bundles)
- [x] P0.4 Relabel heuristic "semantic/attention/safety" APIs — heuristic
      labels on SemanticSimilarityEvaluator, analyze_attention, safety.py
      detectors, ParallelStep; `HuggingFaceModel.supports_streaming=False`
- [ ] P0.5 Mark builtin 87-example datasets as smoke-only; start one real
      dataset loader — smoke-only labelling done (descriptions, docstrings,
      CLI warnings, `scale="smoke"` metadata); real dataset loader still to do

### Next (product spine)

- [ ] P1.1 Unify score/spend envelope across Probe, Evaluator, matched_compute
- [ ] P1.3 Golden-path CI: harness → artefacts → diff gate (core marker)
- [ ] P1.4 Spend parity on ProbeRunner path
- [ ] P1.2 Deprecate contrib.adapters; freeze dual provider stacks
- [ ] P1.5 Provider capability matrix + HF streaming truth

### Then (evidence & quality)

- [ ] P2.1 Live matched-compute runs (even trials, live cost)
- [ ] P2.2 Judge/verifier accounting everywhere
- [ ] P2.4 CLI help: stable vs experimental
- [ ] P2.3 Stats vs scipy property tests
- [ ] P3.1/P3.2 Core test marker + CI extras matrix; stop coverage-file factory

### Extract when spine is stable

- [ ] SPLIT A contrib package (biggest LOC win)
- [ ] SPLIT C trust package (after P0 trust fixes)
- [ ] SPLIT B nlp package
- [ ] SPLIT D viz package
- [ ] SPLIT E agents/orchestration lab

## What is already in good shape (do not thrash)

1. Dependency direction — core does not generally import contrib; inference
   architecture tests enforce no contrib/provider leakage
   (docs/EXECUTION_API_OWNERSHIP.md matches code).
2. Base install minimal — pyyaml only; extras are well factored in
   pyproject.toml.
3. Inference + matched-compute tracer — dense real code (83% code ratio),
   honest limitations, ComputeProfile contract.
4. Diff/harness determinism story — real implementation, not docstring
   theatre; matches README pitch.
5. Provider models — real SDK calls for major vendors; OpenRouter correctly
   reuses OpenAI client.
6. Cosign wrapper — thin, real, injection-constrained identity flags.
7. Function bodies — only ~2.2% trivial pass/ellipsis/NotImplemented; the
   disease is heuristic capability + docstring bulk + test inflation, not
   empty stubs.

## One-line strategy

Shrink the story to harness → artefacts → diff → matched-compute evidence on
real datasets; quarantine contrib/nlp/trust/viz; kill security theatre and
coverage theatre before adding features.

That ranking is justified by: (1) product position text, (2) import graph
(contrib almost unused by core), (3) measured 87-example builtins, (4)
TUF/SCITT code paths that return "verified" without crypto, (5) 70 coverage
test files vs ~63k real code, (6) dual evaluation/provider stacks that Phase 2
already admitted are unfinished.
