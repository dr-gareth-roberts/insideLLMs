---
title: Quick Install
parent: Getting Started
nav_order: 1
---

# Quick Install

**2 minutes to a working, offline installation.**

## Install

```bash
git clone https://github.com/dr-gareth-roberts/insideLLMs.git
cd insideLLMs
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the checkout (release publication is not yet established)
python3 -m pip install .
```

**Using uv?** Replace the last command with `uv pip install .`.
PyYAML and Pydantic are included; validation works without development extras.

## Verify

```bash
insidellms --version
# insidellms 0.2.0

# Offline smoke test: no API key or provider SDK required
insidellms quicktest "What is 2 + 2?" --model dummy
```

## Create an Offline Harness

Run this from the directory where you want to keep the example. The command
creates both `harness.yaml` and `data/harness_dataset.jsonl`; keep the config in
that directory so its relative dataset path resolves correctly.

```bash
insidellms init harness.yaml --template harness
insidellms harness harness.yaml --dry-run
```

This generated DummyModel harness is the portable starting point whether you
installed from a built wheel or from a source checkout.

## Extras (Optional)

```bash
python3 -m pip install ".[openai]"       # OpenAI provider
python3 -m pip install ".[anthropic]"    # Anthropic provider
python3 -m pip install ".[nlp]"          # Text processing
python3 -m pip install ".[visualization]" # Report dependencies
```

For development from a clone, use `python3 -m pip install -e ".[dev]"` from
the repository root. The `providers` extra installs OpenAI, Anthropic, and
Hugging Face support; other providers may require their SDK separately.

## Troubleshooting

**"command not found: insidellms"**
- Ensure your virtual environment is activated
- Try: `python3 -m insideLLMs.cli --version`

**pip installation fails**
- Upgrade pip: `python3 -m pip install --upgrade pip`
- Check Python version: `python3 --version` (must be 3.10+)

## Next

[First Run →](First-Run.md)
