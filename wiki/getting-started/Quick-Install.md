---
title: Quick Install
parent: Getting Started
nav_order: 1
---

# Quick Install

Install insideLLMs in under 2 minutes.

## Requirements

- Python 3.10+
- Git

## Standard Installation

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

## Using uv (Faster)

If you use [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/dr-gareth-roberts/insideLLMs.git
cd insideLLMs
uv venv
source .venv/bin/activate
uv pip install -e ".[all]"
```

## Verify Installation

```bash
insidellms --version
```

You should see output like:

```
insidellms 0.1.0
```

## Optional Extras

Install only what you need:

| Extra | What it includes | Install command |
|-------|------------------|-----------------|
| `nlp` | Text processing (nltk, spacy, scikit-learn) | `pip install -e ".[nlp]"` |
| `visualization` | Charts and reports (matplotlib, seaborn) | `pip install -e ".[visualization]"` |
| `langchain` | LangChain/LangGraph integration | `pip install -e ".[langchain]"` |
| `dev` | Testing and linting tools | `pip install -e ".[dev]"` |
| `all` | Everything above | `pip install -e ".[all]"` |

## Troubleshooting

**"command not found: insidellms"**
- Ensure your virtual environment is activated
- Try: `python -m insideLLMs.cli --version`

**pip installation fails**
- Upgrade pip: `pip install --upgrade pip`
- Check Python version: `python --version` (must be 3.10+)

## Next

[First Run →](First-Run.md)
