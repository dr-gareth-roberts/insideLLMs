---
title: Quick Install
parent: Getting Started
nav_order: 1
---

# Quick Install

**2 minutes to working installation.**

## Install

```bash
# Clone the repository
git clone https://github.com/dr-gareth-roberts/insideLLMs.git
cd insideLLMs

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with all extras
pip install -e ".[all]"
```

**Using uv?** Replace `pip install` with `uv pip install` above.

## Verify

```bash
insidellms --version
# insidellms 0.1.0
```

## Extras (Optional)

```bash
pip install -e ".[nlp]"           # Text processing
pip install -e ".[visualisation]" # Charts
pip install -e ".[dev]"           # Development tools
```

## Troubleshooting

**"command not found: insidellms"**
- Ensure your virtual environment is activated
- Try: `python -m insideLLMs.cli --version`

**pip installation fails**
- Upgrade pip: `pip install --upgrade pip`
- Check Python version: `python --version` (must be 3.10+)

## Next

[First Run →](First-Run.md)
