# Documentation Audit Report

**Date:** 2026-01-26  
**Scope:** Complete codebase vs documentation coverage analysis

---

## Executive Summary

**Current State:** Documentation covers core workflow (run → records → diff) comprehensively. **Critical gaps** exist for advanced features that differentiate insideLLMs from competitors.

**Recommendation:** Add 3-5 focused pages covering undocumented power features. Restructure navigation to surface these differentiators.

---

## Codebase Feature Inventory

### Core Features (Well Documented)

| Feature | Code | Documentation | Status |
|---------|------|---------------|--------|
| ProbeRunner/AsyncProbeRunner | `runtime/runner.py` | Getting Started, Concepts/Runners | Complete |
| Model providers | `models/*.py` | Models Catalog, Providers-and-Models | Complete |
| Built-in probes | `probes/*.py` | Probes Catalog | Complete |
| Deterministic artefacts | `runtime/runner.py` | Determinism, Understanding Outputs | Complete |
| CLI commands | `cli.py` | CLI Reference | Complete |
| Dataset loading | `dataset_utils.py` | Datasets, Configuration | Complete |
| Registry system | `registry.py` | Concepts, API Reference | Complete |

### Advanced Features (Poorly/Not Documented)

| Feature | Code | Documentation | Gap |
|---------|------|---------------|-----|
| **Pipeline middleware** | `pipeline.py` (291 lines) | Performance-and-Caching (brief mention) | **CRITICAL** |
| **Cost tracking** | `cost_tracking.py` (1653 lines) | None | **CRITICAL** |
| **Structured output** | `structured.py` (2892 lines) | None | **CRITICAL** |
| **Input validation** | `validation.py` (1302 lines) | None | **HIGH** |
| **Retry logic** | `retry.py` (2200+ lines) | Performance-and-Caching (brief) | **HIGH** |
| **AgentProbe** | `probes/agent_probe.py` (69KB) | None | **HIGH** |
| **Observability** | `observability.py`, `runtime/observability.py` | None | **MEDIUM** |
| **Distributed execution** | `distributed.py` (104KB) | None | **MEDIUM** |
| **Ensemble methods** | `ensemble.py` (92KB) | None | **MEDIUM** |
| **Hallucination detection** | `hallucination.py` (102KB) | None | **MEDIUM** |
| **Knowledge probing** | `knowledge.py` (114KB) | None | **MEDIUM** |
| **Reasoning analysis** | `reasoning.py` (125KB) | None | **MEDIUM** |
| **Sensitivity analysis** | `sensitivity.py` (91KB) | None | **MEDIUM** |
| **Template versioning** | `template_versioning.py` (94KB) | None | **LOW** |
| **Semantic caching** | `semantic_cache.py` (58KB) | None | **LOW** |

### Massive Undocumented Modules

| Module | Size | Purpose (from code) | Documentation |
|--------|------|---------------------|---------------|
| `agents.py` | 121KB | Tool-using agent evaluation | None |
| `adversarial.py` | 115KB | Adversarial attack testing | None |
| `adapters.py` | 107KB | Model adapters and wrappers | None |
| `context_window.py` | 107KB | Context window management | None |
| `conversation.py` | 109KB | Multi-turn conversation handling | None |
| `hitl.py` | 144KB | Human-in-the-loop evaluation | None |
| `steering.py` | 122KB | Model steering/control | None |
| `synthesis.py` | 111KB | Response synthesis | None |

---

## Critical Documentation Gaps

### 1. Pipeline Middleware (CRITICAL)

**Why it matters:** This is a **major differentiator**. Composable middleware for caching, rate limiting, retry, cost tracking, tracing.

**Current coverage:** Brief mention in Performance-and-Caching.md  
**Needed:** Dedicated "Pipeline Architecture" guide

**Example from code:**
```python
from insideLLMs.pipeline import ModelPipeline, CacheMiddleware, RetryMiddleware
pipeline = ModelPipeline(model)
pipeline.add_middleware(CacheMiddleware())
pipeline.add_middleware(RetryMiddleware(max_attempts=3))
```

**Impact:** Teams can't discover this without reading source code.

### 2. Cost Tracking (CRITICAL)

**Why it matters:** Production teams need budget management. This exists but is invisible.

**Current coverage:** None  
**Needed:** "Cost Management" guide

**Features in code:**
- `TokenCostCalculator` - Calculate costs before/after requests
- `BudgetManager` - Set spending limits with alerts
- `CostForecaster` - Project future costs
- `UsageTracker` - Track usage by model/time period

**Impact:** Users don't know they can track/limit costs.

### 3. Structured Output Parsing (CRITICAL)

**Why it matters:** Extracting structured data from LLM outputs is a common need.

**Current coverage:** None  
**Needed:** "Structured Outputs" guide

**Features in code:**
- Pydantic model extraction
- JSON schema generation
- Robust parsing (handles markdown, code blocks)
- Batch processing
- Export to JSON/HTML/DataFrame

**Impact:** Users resort to manual JSON parsing when this exists.

### 4. AgentProbe (HIGH)

**Why it matters:** Testing tool-using agents is increasingly important.

**Current coverage:** None  
**Needed:** "Agent Evaluation" tutorial

**Features in code:**
- Tool definition and execution
- Trace integration
- Multi-step agent workflows

### 5. Input Validation (HIGH)

**Why it matters:** Clear error messages improve DX.

**Current coverage:** None  
**Needed:** Mention in Troubleshooting or Developer Guide

---

## Documentation Structure Issues

### Current Navigation

```
1. Home
2. Getting Started (has_children)
3. Tutorials (has_children)
4. Concepts (has_children)
5. Reference (has_children)
6. Guides (has_children)
7. FAQ
8. Philosophy
20-32. Legacy pages
```

### Problems

1. **Philosophy buried at nav_order 8** - Should be 2 or 3 (sell the vision early)
2. **No "Advanced Features" section** - Power features hidden
3. **FAQ at 7** - Should be last (after you've learned)
4. **Legacy pages clutter** - Should be archived or integrated

### Proposed Structure

```
1. Home
2. Philosophy (sell the vision)
3. Getting Started
4. Tutorials
5. Concepts
6. Advanced Features (NEW)
   - Pipeline Architecture
   - Cost Management
   - Structured Outputs
   - Agent Evaluation
7. Reference
8. Guides
9. FAQ
```

---

## Missing Content Analysis

### What's Missing from Probes Catalog

**Documented:** LogicProbe, BiasProbe, AttackProbe, FactualityProbe, Code probes, Instruction probes

**Not documented but exist:**
- AgentProbe (tool-using agents)
- ScoredProbe (base class for scored evaluation)
- ComparativeProbe (base class for comparisons)

### What's Missing from Models Catalog

**Documented:** OpenAI, Anthropic, Gemini, Cohere, HuggingFace, Ollama, vLLM, llama.cpp, DummyModel

**Not documented:**
- ModelPipeline (middleware wrapper)
- AsyncModel (async base class)
- ModelWrapper (decorator pattern)

### What's Missing from Guides

**Exist:** Caching, Rate Limiting, Experiment Tracking, Local Models, Troubleshooting

**Should exist:**
- Pipeline Architecture
- Cost Management
- Structured Outputs
- Retry and Circuit Breakers
- Agent Evaluation
- Observability and Tracing

---

## Specific Recommendations

### Priority 1: Add Missing Power Feature Pages

**Create these pages:**

1. **`wiki/advanced/Pipeline-Architecture.md`**
   - Middleware pattern explanation
   - Available middleware (Cache, RateLimit, Retry, CostTracking, Trace)
   - Composition examples
   - Custom middleware guide

2. **`wiki/advanced/Cost-Management.md`**
   - TokenCostCalculator usage
   - BudgetManager setup
   - Cost forecasting
   - Usage tracking and reporting

3. **`wiki/advanced/Structured-Outputs.md`**
   - Pydantic model extraction
   - JSON schema generation
   - Batch processing
   - Export formats

4. **`wiki/advanced/Agent-Evaluation.md`**
   - AgentProbe overview
   - Tool definition
   - Multi-step agent testing
   - Trace integration

5. **`wiki/advanced/Retry-Strategies.md`**
   - RetryMiddleware configuration
   - Circuit breaker pattern
   - Exponential backoff
   - Error handling

### Priority 2: Restructure Navigation

**Move Philosophy to nav_order: 2** (right after Home)
- Sell the vision before diving into tutorials

**Create Advanced Features section at nav_order: 6**
- Surface power features that differentiate from competitors

**Move FAQ to nav_order: 9** (last)
- Reference material, not learning material

**Archive or delete legacy pages**
- Harness.md, Probes-and-Models.md, etc. are redundant

### Priority 3: Strengthen Value Messaging

**Add to homepage:**
- "Advanced Features" section highlighting pipeline, cost tracking, structured outputs
- Comparison table should mention these differentiators

**Add to Philosophy:**
- Section on "What You Get That Others Don't"
- Specifically call out pipeline architecture, cost management

**Add to README:**
- Feature list should include advanced capabilities
- Not just "compare models" but "with cost tracking, retry logic, and structured output parsing"

---

## Content Quality Issues

### Inconsistent Depth

- Getting Started: Now very tight (good)
- Tutorials: Streamlined (good)
- Concepts: Mixed (Models/Probes tight, others verbose)
- Guides: Inconsistent (Caching tight, others verbose)

**Fix:** Apply same tightening to all Guides pages.

### Missing Cross-Links

Many pages don't link to related advanced features:
- Caching.md should link to Pipeline Architecture
- Rate-Limiting.md should link to Retry Strategies
- Models Catalog should mention ModelPipeline

**Fix:** Add "See Also: Advanced Features" sections.

### No "When to Use" Guidance

Pages explain "what" and "how" but not "when":
- When should I use caching vs semantic caching?
- When should I use retry vs circuit breaker?
- When should I use structured outputs vs manual parsing?

**Fix:** Add decision matrices to advanced feature pages.

---

## Proposed File Additions

### New Pages Needed

```
wiki/advanced/
├── index.md (Advanced Features overview)
├── Pipeline-Architecture.md
├── Cost-Management.md
├── Structured-Outputs.md
├── Agent-Evaluation.md
└── Retry-Strategies.md
```

### Pages to Update

```
wiki/index.md
  + Add "Advanced Features" section
  + Strengthen differentiators

wiki/Philosophy.md
  + Add "What You Get That Others Don't" section

README.md
  + Expand feature list with advanced capabilities

wiki/reference/Probes-Catalog.md
  + Add AgentProbe, ScoredProbe, ComparativeProbe

wiki/reference/Models-Catalog.md
  + Add ModelPipeline, AsyncModel, ModelWrapper
```

### Pages to Archive

Move to `wiki/archive/` or delete:
- Harness.md (covered by Getting Started/First-Harness)
- Probes-and-Models.md (covered by Concepts)
- Providers-and-Models.md (covered by Models Catalog)
- CLI.md (covered by reference/CLI.md)
- Configuration.md (covered by reference/Configuration.md)

---

## Navigation Restructure

### Current (Problematic)

```
1. Home
2. Getting Started
3. Tutorials
4. Concepts
5. Reference
6. Guides
7. FAQ
8. Philosophy
20+. Legacy clutter
```

### Proposed (Logical Flow)

```
1. Home (overview + quick start paths)
2. Philosophy (sell the vision)
3. Getting Started (install → first run → first harness)
4. Tutorials (bias, comparison, CI, custom probe)
5. Concepts (models, probes, runners, datasets, determinism, artefacts)
6. Advanced Features (NEW)
   - Pipeline Architecture
   - Cost Management
   - Structured Outputs
   - Agent Evaluation
   - Retry Strategies
7. Reference (CLI, config, probes catalog, models catalog)
8. Guides (caching, rate limiting, tracking, local models, troubleshooting)
9. FAQ (last - reference material)
```

**Rationale:**
- Philosophy early (position the product)
- Advanced Features before Reference (showcase power)
- FAQ last (it's reference, not learning)

---

## Missing "Why Choose insideLLMs" Content

### Competitor Comparison Needs Expansion

**Current:** Brief table in Philosophy.md

**Should be:** Dedicated comparison showing:

| Capability | Eleuther | HELM | OpenAI Evals | insideLLMs |
|------------|----------|------|--------------|------------|
| CI diff-gating | No | No | No | **Yes** |
| Deterministic artefacts | No | No | No | **Yes** |
| Response-level granularity | No | Partial | No | **Yes** |
| Pipeline middleware | No | No | No | **Yes** |
| Cost tracking | No | No | No | **Yes** |
| Structured output parsing | No | No | No | **Yes** |
| Agent evaluation | No | No | No | **Yes** |
| Local model support | Yes | Partial | No | **Yes** |

### Missing Use Case Stories

**Current:** Generic use case table

**Should add:** Real-world scenarios:
- "How Acme Corp caught a bias regression before launch"
- "How Beta Inc reduced eval costs by 80% with cost tracking"
- "How Gamma Labs tests tool-using agents with AgentProbe"

---

## Polish Recommendations

### Immediate (This Session)

1. **Create Advanced Features section** (5 new pages)
2. **Restructure navigation** (move Philosophy to 2, FAQ to 9)
3. **Update homepage** with Advanced Features highlight
4. **Archive legacy pages** (clean up nav_order 20+)

### Short-Term (Next Session)

5. **Add comparison matrix** to Philosophy page
6. **Add use case stories** to homepage or separate page
7. **Cross-link advanced features** from basic guides
8. **Add "When to Use" decision matrices** to advanced pages

### Medium-Term

9. **Video walkthrough** (5-min quickstart)
10. **Interactive examples** (CodeSandbox/Replit)
11. **API reference** on docs site (not just local markdown)
12. **Search optimisation** (keywords, meta descriptions)

---

## Bottom Line

**The documentation tells 60% of the story.**

The core workflow (run → records → diff) is well documented. But the **differentiating features** that make insideLLMs more powerful than competitors are hidden:

- Pipeline middleware (composable cross-cutting concerns)
- Cost tracking (budget management, forecasting)
- Structured outputs (Pydantic integration)
- Agent evaluation (tool-using LLMs)

**These aren't nice-to-haves. They're why teams would choose insideLLMs over Eleuther or HELM.**

Without documenting them, you're selling a bicycle when you've built a car.

---

## Proposed Action Plan

**Phase 1: Advanced Features (2-3 hours)**
- Create `wiki/advanced/` section with 5 pages
- Update navigation structure
- Add to homepage

**Phase 2: Integration (1 hour)**
- Cross-link from basic guides
- Update Probes/Models catalogs
- Add to Philosophy page

**Phase 3: Polish (1 hour)**
- Archive legacy pages
- Final consistency pass
- Verify all links work

**Total time:** 4-5 hours for complete coverage of all major features.
