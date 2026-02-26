# CI Harness

This directory contains the minimal harness used for CI diff-gating. It runs fully offline with no API keys.

## Contents

- `harness.yaml` — Harness config (DummyModel + probes that accept dict-or-string inputs)
- `harness_dataset.jsonl` — Small deterministic dataset (`question`, `prompt`, `task` fields)

## Extending the CI Harness

To add probes while preserving offline determinism:

1. **Use DummyModel only** — The harness uses `type: dummy`; do not add real model providers.
2. **Use compatible probes** — Probes must accept dict-or-string inputs. The current set (`logic`, `attack`, `instruction_following`, `code_generation`) all work with the dataset fields (`question`, `prompt`, `task`).
3. **Add to `harness.yaml`** — Add a probe entry under `probes:` with `type` and optional `args`:

   ```yaml
   probes:
     - type: logic
       args: {}
     - type: bias
       args: {}
   ```

4. **Verify dataset compatibility** — Ensure `harness_dataset.jsonl` includes fields your probe expects. Logic uses `question`; attack uses `prompt`; instruction_following and code_generation use `task`. If adding a probe that needs a new field, extend the dataset accordingly.
5. **Keep `max_examples` low** — The harness uses `max_examples: 3` for fast CI. Avoid increasing it unless necessary.

## Running Locally

```bash
insidellms harness ci/harness.yaml --output-dir .tmp/ci_harness
```

## Related Docs

- `docs/DETERMINISM.md` — Determinism guarantees
- `docs/ARTIFACT_CONTRACT.md` — Artifact field contract
- `wiki/Determinism-and-CI.md` — CI diff-gating workflows
