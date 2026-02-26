# plugin sdk RFC (draft)

Status: draft  
Audience: maintainers and external plugin authors  
Scope: entry-point plugin contract for models, probes, datasets, and deterministic compatibility metadata

## Problem statement

insideLLMs already supports entry-point plugins via `insidellms.plugins`, but current contracts focus on registration mechanics and do not define a formal metadata contract for:

- compatibility expectations,
- capability declaration,
- deterministic-behavior declaration.

This makes ecosystem growth possible, but operational governance (CI policy, preflight diagnostics, and upgrade planning) harder than necessary.

## Goals

1. Keep existing plugin loading behavior working.
2. Add a lightweight metadata contract that tools can consume.
3. Make deterministic compatibility explicit so CI users can make policy decisions.
4. Keep plugin authoring simple and Python packaging-native.

## Non-goals

- Replacing Python entry points with a custom plugin runtime.
- Forcing immediate migration for existing plugins.
- Hard-blocking plugin execution by default in this phase.

## Current baseline (already implemented)

- Entry-point group: `insidellms.plugins`
- Registration callable signatures:
  - `def register() -> None`
  - `def register(*, model_registry, probe_registry, dataset_registry) -> None`
- Deterministic plugin load order by sorted entry-point metadata.

## Proposal

### 1) Keep the existing registration contract

Existing entry points continue to work with no changes.

### 2) Add optional manifest function

Plugins may expose:

```python
def plugin_manifest() -> dict:
    return {...}
```

This function is optional in phase 1 and never required for loading.

### 3) Proposed manifest fields

```json
{
  "name": "acme-evals",
  "version": "0.3.0",
  "sdk_version": "1",
  "insideLLMs": ">=0.2.0,<0.3.0",
  "capabilities": {
    "models": ["acme_chat_v1"],
    "probes": ["acme_reasoning_v2"],
    "datasets": ["acme_evalset_small"]
  },
  "requires": {
    "python": ">=3.10",
    "extras": ["nlp"],
    "env": ["ACME_API_KEY"]
  },
  "determinism": {
    "class": "best_effort",
    "notes": "Provider output is stochastic unless seed is pinned."
  }
}
```

### 4) Determinism class vocabulary

- `deterministic`: plugin behavior is deterministic for fixed inputs/configuration.
- `best_effort`: mostly deterministic with known external/stochastic caveats.
- `non_deterministic`: plugin behavior is intentionally or inherently non-deterministic.

### 5) Validation posture

- Phase 1: warn-only if manifest is missing or incomplete.
- Phase 2: optional strict mode (`insidellms doctor --capabilities --strict-plugins`) can fail on policy violations.
- Phase 3: CI presets may enforce determinism class requirements for selected run profiles.

## Compatibility and migration

- Existing plugins without `plugin_manifest()` remain supported.
- New metadata fields are additive.
- If future manifest schema evolves, version via `sdk_version` and maintain migration notes.

## CLI and observability integration targets

1. `insidellms doctor --capabilities` should expose discovered plugin entry points and plugin capability summaries.
2. `insidellms list` can optionally show plugin provenance (`builtin` vs `plugin`).
3. Run manifests can include plugin provenance fingerprints in a deterministic-safe field (future phase).

## Security and reliability considerations

- Plugin loading remains code execution; only install trusted distributions.
- Deterministic declarations are attestations by plugin authors; consumers should verify behavior in CI.
- In strict CI environments, pin plugin package versions and hashes.

## Open questions

1. Should manifest validation live in core runtime or only in `doctor`/preflight flows?
2. Should determinism classes be declarative only, or verified by optional conformance tests?
3. How should conflicts be resolved when multiple plugins register the same key?

## Next steps

1. Publish this RFC and gather maintainer feedback.
2. Add optional manifest extraction utilities in the registry layer.
3. Expose manifest fields in `doctor --capabilities` JSON.
4. Add docs examples for a full plugin package template and compatibility testing checklist.

