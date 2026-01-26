---
title: Troubleshooting
parent: Guides
nav_order: 5
---

# Troubleshooting

**Quick fixes for common issues.**

## Installation

**"command not found: insidellms"**
```bash
source .venv/bin/activate
# or
python -m insideLLMs.cli --version
```

**pip fails**
```bash
pip install --upgrade pip
python --version  # Must be 3.10+
```

**Missing dependencies**
```bash
pip install -e ".[all]"
```

---

## API Key Issues

### "OPENAI_API_KEY not set"

**Cause:** Environment variable not configured.

**Solutions:**
```bash
# Set for current session
export OPENAI_API_KEY="sk-..."

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc

# Verify
echo $OPENAI_API_KEY
```

### "Invalid API key"

**Cause:** Incorrect key format or expired key.

**Solutions:**
1. Verify key format (OpenAI: `sk-...`, Anthropic: `sk-ant-...`)
2. Check key in provider dashboard
3. Regenerate if expired

### Rate limiting errors

**Cause:** Too many requests.

**Solutions:**
```yaml
# Reduce concurrency
async: true
concurrency: 3  # Lower number

# Add rate limiting
rate_limit:
  requests_per_minute: 60
```

---

## Configuration Issues

### "Dataset file not found"

**Cause:** Relative path resolved incorrectly.

**Important:** Paths are relative to the **config file's directory**, not the current working directory.

**Solution:**
```yaml
# If config is at /project/configs/harness.yaml
dataset:
  path: ../data/test.jsonl  # Resolves to /project/data/test.jsonl
```

### YAML syntax errors

**Cause:** Indentation or formatting issues.

**Solutions:**
```bash
# Validate YAML
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

Common fixes:
- Use spaces, not tabs
- Ensure consistent indentation (2 spaces)
- Quote strings with special characters

### "Unknown model type"

**Cause:** Registry not initialized or typo.

**Solutions:**
```python
from insideLLMs.registry import ensure_builtins_registered
ensure_builtins_registered()

# Check available types
from insideLLMs.registry import model_registry
print(model_registry.list())
```

---

## Runtime Issues

### "Refusing to overwrite directory"

**Cause:** Safety guard preventing overwrite of non-run directories.

**Solutions:**
```bash
# Use --overwrite flag
insidellms run config.yaml --run-dir ./my_run --overwrite

# Or use a new directory
insidellms run config.yaml --run-dir ./my_run_v2

# Or delete manually
rm -rf ./my_run
```

### Memory errors with large datasets

**Cause:** Loading entire dataset into memory.

**Solutions:**
```yaml
# Limit examples
max_examples: 1000

# Use streaming (if supported)
dataset:
  format: jsonl
  path: large_file.jsonl
  streaming: true
```

### Timeout errors

**Cause:** Slow API responses.

**Solutions:**
```yaml
model:
  type: openai
  args:
    model_name: gpt-4o
    timeout: 120  # Increase timeout
```

### "Resume validation failed"

**Cause:** Prompt set changed between runs.

**Solutions:**
1. Use same config for resume
2. Or start fresh: delete run directory and re-run

---

## Diff Issues

### "Unexpected changes in diff"

**Cause:** Non-deterministic model outputs.

**Solutions:**
```yaml
# Use DummyModel for determinism testing
models:
  - type: dummy
    args:
      response: "Fixed response"
```

### "Diff fails on latency"

**Cause:** Comparing volatile fields.

**Solutions:**
```bash
insidellms diff baseline candidate --ignore-fields latency_ms
```

---

## Output Issues

### "Invalid JSONL record"

**Cause:** Corrupted or incomplete write.

**Solutions:**
```bash
# Validate file
python -c "
import json
with open('records.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except:
            print(f'Error on line {i}')
"

# Resume from incomplete run
insidellms run config.yaml --run-dir ./my_run --resume
```

### Empty report.html

**Cause:** No records to report.

**Solutions:**
1. Check `records.jsonl` exists and has content
2. Run harness with `--skip-report false`
3. Generate manually: `insidellms report ./run_dir`

---

## Environment Check

Run diagnostics:

```bash
insidellms doctor --verbose
```

This checks:
- Python version
- Required dependencies
- Optional dependencies
- API key environment variables
- Write permissions

---

## Getting Help

### Check logs

```bash
insidellms run config.yaml --verbose
```

### Debug mode

```bash
insidellms run config.yaml --debug
```

### Report an issue

Include:
1. insideLLMs version: `insidellms --version`
2. Python version: `python --version`
3. OS: `uname -a`
4. Full error message
5. Minimal reproducible config

[Open an issue on GitHub](https://github.com/dr-gareth-roberts/insideLLMs/issues)
