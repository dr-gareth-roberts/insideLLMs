# Stability matrix and contract policy

This document defines the compatibility contract for insideLLMs surfaces.
Use it with `docs/STABILITY.md` when reviewing changes.

## Why this exists

insideLLMs is intended for CI diff-gating workflows. Users need clear guarantees
about what can change safely and what requires migration planning.

## Stability levels

- **Stable**: Backward compatibility is expected across minor releases.
- **Experimental**: May change in minor releases; migration notes are still expected.
- **Internal**: No compatibility guarantees.

## Contract matrix

| Surface | Status | Compatibility promise | Change requirements |
|---|---|---|---|
| CLI command names (`insidellms run`, `harness`, `diff`, `report`, etc.) | Stable | Commands should remain available with consistent behavior. | Breaking change requires deprecation phase + changelog + migration note. |
| CLI flag semantics for core deterministic flow (`run`, `harness`, `diff`, `report`, `schema validate`) | Stable | Existing flags should preserve meaning. | Semantic changes require deprecation alias and docs updates before removal. |
| Canonical run artifacts (`records.jsonl`, `manifest.json`, `summary.json`, `diff.json`) | Stable | Schema-governed and deterministic for identical inputs/config. | Any field addition/removal/meaning change requires schema version update and docs. |
| Schema validation contracts (`SchemaRegistry` names and versions) | Stable | Published schema names remain valid; versioning follows SemVer principles. | Schema bumps + compatibility notes required. |
| Deterministic controls (`strict_serialization`, `deterministic_artifacts`) | Stable | Flag/config semantics remain consistent. | Behavior changes require explicit docs and regression tests. |
| Registry extension points (model/probe/dataset registration APIs) | Stable | Existing registration and lookup patterns remain supported. | API changes require migration path and updated plugin docs. |
| Python import paths in public docs (`insideLLMs.*` user-facing modules) | Experimental | Best-effort compatibility while architecture settles. | Any move/rename must include release notes and import aliases where feasible. |
| Private/underscored modules (`insideLLMs.runtime._*`, `insideLLMs.cli._*`) | Internal | No compatibility guarantee. | Can change without notice; avoid external use. |

## Contract change policy

When changing a **Stable** surface:

1. Add an item to `CHANGELOG.md` under **Unreleased**.
2. Update `docs/STABILITY.md` and/or this matrix.
3. Add or update compatibility tests (CLI/schema/determinism).
4. If breaking, provide:
   - deprecation window,
   - migration instructions,
   - explicit removal version.

## Practical review checklist for PRs

- Does this modify a Stable surface?
- If yes, is there a compatibility path?
- Were schema versions and docs updated where needed?
- Were deterministic artifact guarantees preserved?
- Is the changelog updated?

## Recommended CI enforcement (policy target)

For world-class contract discipline, enforce these checks in CI:

- CLI smoke tests for stable command/flag compatibility.
- Schema contract tests for canonical artifact files.
- Determinism golden-path job (`run -> records -> report -> diff`).

This file defines policy and expected behavior; CI wiring may evolve separately.
