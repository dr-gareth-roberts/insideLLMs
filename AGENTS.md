# Repository Guidelines

## Project Structure & Module Organization
The core Python package lives in `insideLLMs/`, with focused modules such as `models/`, `probes/`, and `nlp/`, plus utilities like `benchmark.py`, `runner.py`, `results.py`, and `visualization.py`. Tests are in `tests/`, runnable example scripts are in `examples/`, and sample assets or datasets live in `data/`. Packaging and dependency metadata are in `pyproject.toml`, with development dependencies listed in `requirements-dev.txt`.

## Build, Test, and Development Commands
- `pip install .` installs the base library.
- `pip install .[nlp]`, `pip install .[visualization]`, or `pip install .[nlp,visualization]` enable optional extras.
- `pip install -r requirements-dev.txt` sets up a local dev environment.
- `pytest` runs the test suite.
- `python examples/example_models.py` (or other scripts in `examples/`) runs a local demo.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and keep modules small and focused. Use `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_CASE` for constants. Keep docstrings short and consistent with the existing files, and prefer explicit method names in `models/` and `probes/`.

## Testing Guidelines
Tests use `pytest`. Place new tests in `tests/` with filenames like `test_*.py` and functions named `test_*`. Add or update tests whenever you change public APIs or add new modules; lightweight import tests are common for new package surfaces.

## Commit & Pull Request Guidelines
Recent history uses concise, imperative commit subjects with optional prefixes (for example, `Refactor: improve packaging` or `Add NLP helpers`). For pull requests, include a brief summary, the tests run (for example, `pytest`), and note any API or environment changes; link issues when applicable.

## Configuration & Secrets
API-backed models require environment variables: `OPENAI_API_KEY` for `OpenAIModel` and `ANTHROPIC_API_KEY` for `AnthropicModel`. HuggingFace-based models download weights on first run, so ensure network access and enough local disk space.
