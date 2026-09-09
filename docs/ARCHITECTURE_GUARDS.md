# Executable architecture guards

The package boundaries are enforced from static evidence rather than by importing
`insideLLMs`. This matters because importing the public facade can trigger registry
initialization and would hide coupling to optional dependencies.

## Generated evidence

`python3 scripts/architecture_evidence.py --write` generates:

- `architecture/api_manifest.json`: every symbol exposed by the root `__all__` or
  lazy facade, its status, provider module, and repository-owned usage evidence;
- `architecture/import_graph.json`: module LOC, internal fan-in/fan-out, layer
  edges, optional dependency imports, and static test references by subtree;
- `docs/API_STATUS.md`: the reviewable root-facade summary.

The scanner samples root documentation, `examples/`, `docs/`, `wiki/`, `ci/`,
`action.yml`, `scripts/`, CLI handlers, and tests. It captures regular Python imports
plus literal dynamic-import calls and lazy-import maps. Usage is evidence of coupling,
not an automatic stability promise. `docs/STABILITY_MATRIX.md` remains the
compatibility policy.

## Layer matrix

`architecture/layers.json` defines the current target graph:

```text
contracts -> contracts only
core      -> contracts + core infrastructure + runtime protocols
runtime   -> contracts + core infrastructure
inference -> contracts + core + runtime protocols
providers -> contracts + core + runtime protocols
evals     -> contracts + core
analysis  -> contracts + core + artifacts/analysis
cli       -> leaf integration layer
labs      -> may consume stable product layers; never the reverse
```

The root facade remains in `core` while it is reduced, so its legacy lazy edges are
recorded as explicit debt rather than permitted by the matrix. Provider SDK imports
are separately restricted to provider and labs modules.

## Existing debt and exception policy

The first snapshot deliberately records existing violations instead of weakening
the target matrix. Every entry in `architecture/import_exceptions.json` is exact
(source module plus target module), owned, justified, and expiring. Wildcards are
not supported. Tests fail when an exception expires or becomes stale.

New forbidden edges are rejected. Remove an exception in the same change that
removes its import. Renewing an expiry requires owner review and a new migration
rationale; it must not be an automatic date bump.

## Commands

```bash
make architecture-update  # regenerate evidence for review
make architecture         # check drift, layers, SDK placement, and expiry
make clean-install-golden-path
```

The clean-install command copies the working tree to a disposable build context,
builds a wheel, and installs it with only its mandatory dependency closure into a
fresh virtual environment. Checks run from a neutral directory with source-path
environment variables removed, and assert that `insideLLMs` resolves from the venv.
It then verifies installed CLI help, two deterministic harness runs, and a no-change
diff gate.
