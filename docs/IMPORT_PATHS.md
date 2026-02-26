# Import Path Migration Matrix

Quick reference for canonical import paths. Use these when writing new code or updating existing imports.

## Preferred Import Style

For new code, prefer:

1. **Direct canonical imports** — `from insideLLMs.analysis.statistics import generate_summary_report` rather than `from insideLLMs.statistics import ...`
2. **One symbol per import line** when importing many items — improves readability and reduces merge conflicts
3. **Avoid `import *`** — explicit imports make dependencies clear and help static analysis
4. **Runtime/CLI internals** — import from `insideLLMs.runtime.runner` for public APIs; avoid `insideLLMs.runtime._*` in external code

## Canonical Paths (Preferred)

| Symbol / Capability | Canonical import |
|---------------------|------------------|
| Runner, run_experiment_from_config | `from insideLLMs.runtime.runner import ProbeRunner, AsyncProbeRunner, run_experiment_from_config` |
| Statistics, summary reports | `from insideLLMs.analysis.statistics import generate_summary_report, ...` |
| Comparison utilities | `from insideLLMs.analysis.comparison import compare_experiments, ...` |
| Trace config | `from insideLLMs.trace.trace_config import TraceConfig, load_trace_config, ...` |
| Caching | `from insideLLMs.caching import InMemoryCache, StrategyCache, ...` |
| Export | `from insideLLMs.analysis.export import export_to_csv, ...` |
| Visualization | `from insideLLMs.analysis.visualization import create_html_report, ...` |
| Config (YAML) | `from insideLLMs.config import load_config, ExperimentConfig, ...` |
| Run config (programmatic) | `from insideLLMs.config_types import RunConfig, RunConfigBuilder, ...` |

## Deprecated / Removed Paths

| Old path | Status | Use instead |
|----------|--------|-------------|
| `insideLLMs.runner` | Removed (0.2.0) | `insideLLMs.runtime.runner` |
| `insideLLMs.statistics` | Removed (0.2.0) | `insideLLMs.analysis.statistics` |
| `insideLLMs.comparison` | Removed (0.2.0) | `insideLLMs.analysis.comparison` |
| `insideLLMs.trace_config` | Removed (0.2.0) | `insideLLMs.trace.trace_config` |
| `insideLLMs.cache` | Removed (0.2.0) | `insideLLMs.caching` |
| `insideLLMs.caching_unified` | Renamed (0.2.0) | `insideLLMs.caching` |

## Compatibility Shims (Still Supported)

These re-export from canonical modules for backward compatibility. Prefer canonical paths in new code.

| Shim | Canonical target | Status |
|------|------------------|--------|
| `insideLLMs.export` | `insideLLMs.analysis.export` | Indefinite support |
| `insideLLMs.visualization` | `insideLLMs.analysis.visualization` | Indefinite support |
| `insideLLMs.pipeline` | `insideLLMs.runtime.pipeline` | Indefinite support |

## Internal Modules (Avoid External Use)

Modules prefixed with `_` have no compatibility guarantee:

- `insideLLMs.runtime._*` (e.g. `_sync_runner`, `_async_runner`, `_artifact_utils`)
- `insideLLMs.cli._*` (e.g. `_parsing`, `_report_builder`, `_record_utils`)

See `docs/STABILITY_MATRIX.md` for the full contract.
