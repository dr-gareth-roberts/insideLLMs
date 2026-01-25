# CLI

This page documents the `insidellms` command-line interface and its subcommands.
For configs and end-to-end workflows, see (Getting-Started), (Configuration), (Harness), and
(Results-and-Reports).

Tip: when running from source without installing the console script, you can use:

```bash
python -m insideLLMs.cli --help
python -m insideLLMs.cli harness ci/harness.yaml
```

## Command Summary

- `run`: Run a single experiment from a config file.
- `harness`: Run a cross-model probe harness from a harness config.
- `report`: Rebuild `summary.json` and `report.html` from an existing run directory.
- `diff`: Compare two run directories for regressions or behavioural changes.
- `schema`: Inspect and validate versioned output schemas.
- `doctor`: Diagnose environment and optional dependencies.
- `quicktest`: Run a single prompt against a model (fast sanity check).
- `benchmark`: Run benchmark suites across models and probes.
- `compare`: Compare multiple models on shared inputs.
- `list`: List available models, probes, datasets, or trackers.
- `init`: Generate a sample configuration file.
- `info`: Show details for a model, probe, or dataset.
- `interactive`: Start an interactive exploration session.
- `validate`: Validate a config file or a run directory.
- `export`: Export results JSON to other formats.

## Run Directory Control (`run` + `harness`)

Both `run` and `harness` write deterministic run artifacts (including `manifest.json` and
`records.jsonl`) under a run directory.

- `--run-dir`: write artifacts exactly to this directory (overrides `--run-root`/`--run-id`).
- `--run-root`: base directory for runs (default: `~/.insidellms/runs`, or `INSIDELLMS_RUN_ROOT`).
- `--run-id`: run identifier recorded in the manifest and used as the directory name under
  `--run-root`. If omitted, a deterministic run ID is derived from the resolved config snapshot.
- `--overwrite`: allow overwriting an existing non-empty run directory (dangerous).

See (Determinism-and-CI) for the byte-for-byte determinism contract.

## Command Reference

### `run`

Run an experiment from a configuration file (YAML or JSON).

```bash
insidellms run configs/experiment.yaml
insidellms run configs/experiment.yaml --run-root .tmp/runs --run-id my-run --overwrite
insidellms run configs/experiment.yaml --async --concurrency 5
```

Common flags:

- `--resume`: resume from an existing run directory if `records.jsonl` is present.
- `--async` + `--concurrency`: enable async execution and set concurrency.
- `--track` + `--track-project`: enable experiment tracking (see (Experiment-Tracking)).
- `--validate-output`: validate serialized outputs against a versioned schema (requires pydantic).
- `--schema-version` and `--validation-mode`: schema version selection and strict vs warn behavior.

### `harness`

Run a cross-model probe harness.

```bash
insidellms harness examples/harness.yaml
insidellms harness examples/harness.yaml --run-dir .tmp/harness --report-title "Weekly Harness"
insidellms harness ci/harness.yaml --run-dir .tmp/runs/base --overwrite --skip-report
```

### `report`

Rebuild `summary.json` and `report.html` from an existing run directory.

```bash
insidellms report .tmp/runs/my-run
insidellms report .tmp/runs/my-run --report-title "Experiment Report"
```

### `diff`

Compare two run directories.

```bash
insidellms diff ./baseline ./candidate --fail-on-regressions
insidellms diff ./baseline ./candidate --format json --output diff.json
```

Useful flags:

- `--fail-on-regressions`: fail only on score/status regressions
- `--fail-on-changes`: fail on any difference (including additions/removals)
- `--fail-on-trace-drift`: fail if trace fingerprints drift (when present)
- `--fail-on-trace-violations`: fail if contract violations increase (when present)
- `--output-fingerprint-ignore`: ignore volatile keys when fingerprinting structured outputs

See (Tracing-and-Fingerprinting) for details.

### `schema`

Inspect and validate versioned output schemas.

```bash
insidellms schema list
insidellms schema ResultRecord
insidellms schema validate --name ResultRecord --input ./run_dir/records.jsonl
```

### `doctor`

Diagnose environment and optional dependencies.

```bash
insidellms doctor
insidellms doctor --format json --fail-on-warn
```

### `quicktest`

Run a single prompt against a model.

```bash
insidellms quicktest "What is 2 + 2?" --model dummy
insidellms quicktest "Summarize this paragraph." --model openai --temperature 0.2
```

### `benchmark`

Run benchmark suites across models and probes.

```bash
insidellms benchmark --models openai,anthropic --probes logic,bias
insidellms benchmark --models openai --datasets reasoning,math --max-examples 5 --html-report
```

### `compare`

Compare multiple models on shared inputs.

```bash
insidellms compare --models openai,anthropic --input "Explain gradient descent"
insidellms compare --models openai,anthropic --input-file inputs.txt --format markdown
```

### `list`

List available models, probes, datasets, trackers, or all.

```bash
insidellms list models
insidellms list all --detailed --filter openai
insidellms list trackers
```

### `init`

Generate a sample configuration file.

```bash
insidellms init
insidellms init --template tracking --model openai --probe logic config.yaml
```

### `info`

Show details about a model, probe, or dataset.

```bash
insidellms info model openai
insidellms info probe logic
insidellms info dataset reasoning
```

### `interactive`

Start an interactive exploration session.

```bash
insidellms interactive
insidellms interactive --model openai --history-file .tmp/insidellms_history
```

### `validate`

Validate a configuration file or a run directory.

```bash
insidellms validate configs/experiment.yaml
insidellms validate .tmp/runs/my-run --schema-version 1.0.1 --mode warn
```

### `export`

Export a results JSON file to another format.

```bash
insidellms export results.json --format csv --output results.csv
insidellms export results.json --format markdown
insidellms export results.json --format latex
```

Note: `--format html` is reserved; use `insidellms report <run_dir>` to produce `report.html`.

## Global Flags

- `--help`: show help and exit
- `--version`: show version and exit
- `--no-color`: disable coloured output
- `--quiet` / `-q`: minimal output (errors only)
