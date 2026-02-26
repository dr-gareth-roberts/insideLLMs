---
title: CLI Reference
parent: Reference
nav_order: 1
---

# CLI Reference

Complete reference for the `insidellms` command-line interface.

## Synopsis

```bash
insidellms <command> [options]
```

## Commands

| Command | Description |
|---------|-------------|
| [`run`](#run) | Run probes from a config file |
| [`harness`](#harness) | Run multi-model comparison harness |
| [`quicktest`](#quicktest) | Quick single-prompt test |
| [`diff`](#diff) | Compare two run directories |
| [`report`](#report) | Generate HTML report from records |
| [`validate`](#validate) | Validate run artifacts |
| [`schema`](#schema) | Schema utilities |
| [`doctor`](#doctor) | Check environment and dependencies |
| [`attest`](#attest) | Generate DSSE attestations for a run directory |
| [`sign`](#sign) | Sign attestations with Sigstore |
| [`verify-signatures`](#verify-signatures) | Verify attestation signature bundles |

---

## run

Run probes from a YAML/JSON configuration file.

```bash
insidellms run <config> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `config` | Path to YAML/JSON config file |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--run-dir DIR` | Output directory for artifacts | Auto-generated |
| `--run-root DIR` | Root for run directories | `~/.insidellms/runs` |
| `--run-id ID` | Explicit run ID | Computed from config |
| `--overwrite` | Overwrite existing run directory | `false` |
| `--resume` | Resume from existing records | `false` |
| `--async` | Enable async execution | `false` |
| `--concurrency N` | Max concurrent requests | `1` |
| `--max-examples N` | Limit dataset size | All |
| `--validate` | Validate outputs against schema | `false` |
| `--skip-report` | Skip HTML report generation | `false` |
| `--verbose` | Verbose output | `false` |

### Examples

```bash
# Basic run
insidellms run config.yaml

# With explicit output directory
insidellms run config.yaml --run-dir ./my_run

# Async with concurrency
insidellms run config.yaml --async --concurrency 10

# Resume interrupted run
insidellms run config.yaml --run-dir ./my_run --resume

# Overwrite existing run
insidellms run config.yaml --run-dir ./my_run --overwrite
```

---

## harness

Run a multi-model comparison harness.

```bash
insidellms harness <config> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `config` | Path to harness YAML/JSON config |

### Options

Same as [`run`](#run), plus:

| Option | Description | Default |
|--------|-------------|---------|
| `--model-filter NAME` | Run only specified model | All |
| `--probe-filter NAME` | Run only specified probe | All |

### Examples

```bash
# Basic harness
insidellms harness harness.yaml

# Filter to specific model
insidellms harness harness.yaml --model-filter gpt-4o

# Async with limited examples
insidellms harness harness.yaml --async --concurrency 10 --max-examples 100
```

---

## quicktest

Quick single-prompt test.

```bash
insidellms quicktest <prompt> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `prompt` | The prompt to send to the model |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model TYPE` | Model type (openai, anthropic, dummy) | `dummy` |
| `--model-name NAME` | Specific model name | Provider default |
| `--probe TYPE` | Probe type | `logic` |
| `--temperature T` | Sampling temperature | `1.0` |
| `--max-tokens N` | Max response tokens | Provider default |

### Examples

```bash
# Quick test with dummy model
insidellms quicktest "What is 2 + 2?" --model dummy

# Test with OpenAI
insidellms quicktest "Explain gravity" --model openai --model-name gpt-4o

# With specific parameters
insidellms quicktest "Be creative" --model openai --temperature 1.5
```

---

## diff

Compare two run directories.

```bash
insidellms diff <baseline> <candidate> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `baseline` | Path to baseline run directory |
| `candidate` | Path to candidate run directory |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output FILE` | Write diff report to file | stdout |
| `--fail-on-changes` | Exit code 1 if changes detected | `false` |
| `--fail-on-trace-violations` | Fail on trace drift | `false` |
| `--ignore-fields FIELDS` | Comma-separated fields to ignore | None |
| `--trace-aware` | Enable trace-aware diffing | `false` |
| `--format FORMAT` | Output format (json, text) | `text` |

### Examples

```bash
# Basic diff
insidellms diff ./baseline ./candidate

# CI gating (fail on changes)
insidellms diff ./baseline ./candidate --fail-on-changes

# Output to file
insidellms diff ./baseline ./candidate --output diff.json --format json

# Ignore volatile fields
insidellms diff ./baseline ./candidate --ignore-fields latency_ms,timestamps
```

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | No changes detected |
| `1` | Changes detected (with `--fail-on-changes`) |
| `2` | Error (missing files, invalid format) |

---

## report

Generate HTML report from records.

```bash
insidellms report <run-dir> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `run-dir` | Path to run directory with records.jsonl |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output FILE` | Output HTML file | `report.html` in run-dir |
| `--template FILE` | Custom HTML template | Built-in |

### Examples

```bash
# Generate report
insidellms report ./my_run

# Custom output path
insidellms report ./my_run --output ./reports/comparison.html
```

---

## validate

Validate run artifacts against schemas.

```bash
insidellms validate <run-dir> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `run-dir` | Path to run directory |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--schema-version VER` | Expected schema version | Auto-detect |
| `--strict` | Fail on any validation warning | `false` |

### Examples

```bash
# Validate a run
insidellms validate ./my_run

# Strict validation
insidellms validate ./my_run --strict
```

---

## schema

Schema utilities.

```bash
insidellms schema [op] [options]
```

### Operations

| Operation | Description |
|-----------|-------------|
| `list` (default) | List available schemas and versions |
| `dump` | Print/write a JSON Schema document |
| `validate` | Validate `.json` or `.jsonl` input payloads |
| `<SchemaName>` | Shortcut for `dump --name <SchemaName>` |

### Examples

```bash
# List schemas
insidellms schema list

# Dump a schema to stdout
insidellms schema dump --name ResultRecord

# Shortcut dump form
insidellms schema ResultRecord

# Validate a JSON object (manifest)
insidellms schema validate --name RunManifest --input ./baseline/manifest.json

# Validate a JSONL stream (records)
insidellms schema validate --name ResultRecord --input ./baseline/records.jsonl

# Warn-only mode
insidellms schema validate --name ResultRecord --input ./baseline/records.jsonl --mode warn
```

---

## doctor

Check environment and dependencies.

```bash
insidellms doctor [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--fix` | Attempt to fix issues | `false` |
| `--verbose` | Show detailed information | `false` |

### Checks Performed

- Python version
- Required dependencies
- Optional dependencies (nlp, visualisation)
- API key environment variables
- Write permissions for run root

### Examples

```bash
# Check environment
insidellms doctor

# Verbose check
insidellms doctor --verbose
```

---

## attest

Generate attestation artifacts for an existing run directory.

```bash
insidellms attest <run-dir>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `run-dir` | Path to run directory (must contain `manifest.json`) |

### Examples

```bash
insidellms attest ./baseline
```

---

## sign

Sign attestation envelopes in a run directory using Sigstore (`cosign`).

```bash
insidellms sign <run-dir>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `run-dir` | Path to run directory (must contain `attestations/`) |

### Examples

```bash
insidellms sign ./baseline
```

---

## verify-signatures

Verify attestation signatures against Sigstore bundles.

```bash
insidellms verify-signatures <run-dir> [--identity ...]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `run-dir` | Path to run directory (must contain `attestations/` and `signing/`) |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--identity CONSTRAINTS` | Identity constraints passed to verifier | None |

### Examples

```bash
insidellms verify-signatures ./baseline
insidellms verify-signatures ./baseline --identity "issuer=https://token.actions.githubusercontent.com"
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google/Gemini API key |
| `CO_API_KEY` | Cohere API key |
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace token |
| `INSIDELLMS_RUN_ROOT` | Default run root directory |
| `NO_COLOR` | Disable coloured output |

---

## Global Options

Available for all commands:

| Option | Description |
|--------|-------------|
| `--help` | Show help message |
| `--version` | Show version number |
| `--quiet` | Suppress non-error output |
| `--debug` | Enable debug logging |
