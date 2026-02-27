# API tiers

This document defines compatibility tiers for Python imports, CLI behavior, and extension points.
Use it together with [docs/STABILITY.md](../STABILITY.md), [docs/STABILITY_MATRIX.md](../STABILITY_MATRIX.md), and [docs/IMPORT_PATHS.md](../IMPORT_PATHS.md).

## Why this exists

insideLLMs has grown into a platform-scale project. Explicit API tiering helps contributors make safe changes and helps downstream users choose stable integration points.

## Tier definitions

| Tier | Meaning | Change policy |
|---|---|---|
| Tier 1 (stable) | User-facing contract surface. | Backward compatibility expected across minor releases. Breaking changes require deprecation and migration notes. |
| Tier 2 (evolving) | Public but still maturing. | Best-effort compatibility. Minor-release changes allowed with release notes. |
| Tier 3 (internal) | Implementation detail. | No compatibility guarantees. Can change without notice. |

## Tier map (current)

| Surface | Tier | Notes |
|---|---|---|
| CLI deterministic spine: `insidellms run`, `harness`, `report`, `diff`, `schema validate` | Tier 1 | Primary CI diff-gating workflow. |
| Canonical run artifacts (`records.jsonl`, `manifest.json`, `summary.json`, `diff.json`) and schema names | Tier 1 | Governed by versioned schema contracts. |
| Core runtime entry points: `insideLLMs.runtime.runner` (`ProbeRunner`, `AsyncProbeRunner`, config-driven execution helpers) | Tier 1 | Canonical programmatic execution interface. |
| Runtime diffing API: `insideLLMs.runtime.diffing` (`build_diff_computation`, `compute_diff_exit_code`) | Tier 1 | Canonical programmatic diff + gate surface shared by CLI/integrations. |
| Public diff facade: `insideLLMs.diffing` | Tier 1 | Narrow import surface for diff computation, gate policy, and interactive snapshot helpers. |
| Public injection facade: `insideLLMs.injection` | Tier 1 | Stable import path for injection detection/sanitization/prompt-defense APIs. |
| Registry extension points: `model_registry`, `probe_registry`, `dataset_registry` | Tier 1 | Plugin and custom extension integration surface. |
| Public model/probe base interfaces (`insideLLMs.models.base`, `insideLLMs.probes.base`) | Tier 1 | Foundation for custom adapters and probes. |
| Analysis/export/report helper modules (`insideLLMs.analysis.*`) | Tier 2 | Public and documented, but still evolving. |
| Broader convenience exports from `insideLLMs.__init__` | Tier 2 | User-friendly surface; may be reorganized with migration notes. |
| Advanced optional subsystems (for example observability, HITL, steering) | Tier 2 | Feature-rich and still in active architectural evolution. |
| Underscored modules (`insideLLMs.runtime._*`, `insideLLMs.cli._*`) | Tier 3 | Internal-only implementation details. |
| Private helper functions not documented in API docs | Tier 3 | Subject to change at any time. |

## Contributor rules

1. Default new user-facing APIs to Tier 2 unless they are part of the deterministic spine.
2. Promote a Tier 2 surface to Tier 1 only after:
   - clear docs,
   - compatibility tests,
   - at least one release cycle without breaking changes.
3. Never place external integrations on Tier 3 paths.
4. When touching Tier 1:
   - update changelog and docs,
   - add/adjust compatibility tests,
   - provide migration guidance for any behavior changes.

## Practical import guidance

- Prefer canonical imports documented in [docs/IMPORT_PATHS.md](../IMPORT_PATHS.md).
- Avoid importing from underscored modules outside the insideLLMs codebase.
- If a symbol is only available via a broad re-export and not a canonical module path, treat it as Tier 2 unless otherwise documented.
