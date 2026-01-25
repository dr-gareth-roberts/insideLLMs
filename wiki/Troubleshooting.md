# Troubleshooting

## Quick Diagnostics

Start here:

```bash
insidellms doctor
```

If you’re running from source, ensure you have the optional extras you need:

```bash
pip install -e ".[dev,nlp,visualization]"
```

## Missing API Keys

If you see model initialization failures for hosted providers, check env vars:

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google/Gemini: `GOOGLE_API_KEY`
- Cohere: `CO_API_KEY` (or `COHERE_API_KEY`)

See [Providers and Models](Providers-and-Models).

## “Run directory is not empty”

Both `run` and `harness` use a safety guard to prevent clobbering existing artifacts.

Fixes:

- Choose a fresh `--run-dir`, or
- Use `--overwrite` (dangerous), or
- Use `--run-root` and let insideLLMs select a deterministic directory name from the config.

## Dataset Not Found / Wrong Paths

Relative dataset paths are resolved relative to the config file’s directory.
If you run from `examples/`, `../data/...` is often what you want.

See [Configuration](Configuration) and [Datasets and Harness](Datasets-and-Harness).

## Schema Validation Errors

If you pass `--validate-output` you need `pydantic` installed.

You can also choose warn mode:

```bash
insidellms run config.yaml --validate-output --validation-mode warn
insidellms harness harness.yaml --validate-output --validation-mode warn
```

## Tracking Backends Not Installed

Tracking is best-effort. If a backend dependency is missing, insideLLMs continues the run and
prints a warning.

Use `--track local` for a no-dependency option.

See [Experiment Tracking](Experiment-Tracking).

## Report Generation Issues

If interactive HTML report generation fails due to missing visualization dependencies, install:

```bash
pip install -e ".[visualization]"
```

The harness also has a basic HTML fallback when optional visualization components aren’t present.

## CI Failures (Lint / Types / Tests)

Local mirrors of CI:

```bash
ruff check .
ruff format --check .
mypy insideLLMs
pytest
```

## See Also

- [CLI](CLI)
- [Getting Started](Getting-Started)
- [Determinism and CI](Determinism-and-CI)
