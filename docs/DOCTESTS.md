# Docstring examples

67% of this package by line count is docstrings (167,461 of 248,754 lines).
Until now nothing executed the examples inside them: `--doctest-modules` appears
in no pytest config, no CI job, and no Makefile target.

This is the direct cause of the `docs_drift` backlog category. Those items are a
hand-sampled fraction of a much larger population, found one at a time by
reading rather than by running.

## Current baseline

Measured 2026-08-20, `[all]` install, CPython 3.12.12, `insideLLMs/` excluding
`contrib/`:

| Result | Count |
|---|---|
| failed | 587 |
| passed | 679 |
| skipped | 36 |
| collection errors | 4 |

A 46% failure rate on the documented examples, before `contrib/` is even
considered.

Worst modules by failing examples:

| Module | Failures |
|---|---|
| `runtime/_async_runner.py` | 303 |
| `probes/factuality.py` | 113 |
| `runtime/timeout_wrapper.py` | 100 |
| `models/local.py` | 91 |
| `rate_limiting.py` | 82 |
| `exceptions.py` | 69 |

The four collection errors are `analysis/evaluation.py`, `probes/attack.py`,
`registry.py`, and `visualization.py` — the last because the shim replaces
itself in `sys.modules`, so pytest sees two modules with the same basename.

## Why `make doctest` is not part of `make check`

Two reasons, both of which have to be fixed before it can gate.

**1. The examples have side effects.** 458 example lines call something with an
external effect — opening files for write, network calls, `sleep`, starting
threads. Several write into the current working directory, and four such files
were committed to the repo root by accident before being removed:
`fingerprint.json`, `trace.json`, `trace_export.json`, `log.txt`. Their sources
are `contrib/debugging.py:624`, `contrib/debugging.py:1564`,
`contrib/fingerprinting.py:1212`, and `runtime/pipeline.py:1269`.

`make doctest` runs from a temporary directory for this reason. That contains
the damage but does not fix it — the examples should be marked
`# doctest: +SKIP` or rewritten to use `tmp_path`.

**2. Some examples block forever.** `contrib/deployment.py` documented
`quick_deploy(model, port=8080)` — annotated "Blocking call" in prose but not
marked `+SKIP` — so running the suite booted a Uvicorn server on port 8080 and
hung. Both occurrences are now `# doctest: +SKIP`. `contrib/` remains excluded
from the target until the rest of its examples are audited the same way.

## Burning it down

Run `make doctest` and work top-down through the table above. Each fix is
independently verifiable: the example either produces its documented output or
it does not.

Once the count reaches zero for a module, that module's examples are executable
documentation and cannot silently rot again. Once it reaches zero overall, add
`doctest` to the `check` target and to CI, and the whole `docs_drift` category
stops being something a human has to find by reading.
