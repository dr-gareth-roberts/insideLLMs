### Project Guidelines

This document provides project-specific information for developers working on `insideLLMs`.

### Build and Configuration

The project uses `pyproject.toml` for dependency management and can be installed via `pip`.

#### Installation
- **Base package**: `pip install .`
- **With NLP utilities**: `pip install .[nlp]` (Installs `nltk`, `spacy`, `scikit-learn`, `gensim`)
- **With Visualization**: `pip install .[visualization]` (Installs `matplotlib`, `pandas`, `seaborn`)
- **All optional dependencies**: `pip install .[all]`

#### Development Environment
For development, it is recommended to install the package in editable mode with all dependencies:
```bash
pip install -e .[all]
```
Alternatively, you can install the development dependencies from `requirements-dev.txt`:
```bash
pip install -r requirements-dev.txt
```

### Testing Information

The project uses `pytest` for testing.

#### Running Tests
To run all tests:
```bash
pytest
```
To run tests with output:
```bash
pytest -s
```
If you haven't installed the package in editable mode, you may need to set `PYTHONPATH` to the current directory:
```bash
PYTHONPATH=. pytest
```

#### Adding New Tests
When adding new tests for probes or benchmarks, it is recommended to use `DummyModel` from `insideLLMs.models` to avoid external API calls and ensure tests are fast and deterministic.

#### Test Example
The following is a simple test case demonstrating how to test a probe using the `DummyModel`:

```python
import pytest
from insideLLMs.models import DummyModel
from insideLLMs.probes import LogicProbe

def test_logic_probe_with_dummy_model():
    # Initialize a dummy model that echoes input
    model = DummyModel()
    probe = LogicProbe()
    
    problem = "If A then B. A is true. What is B?"
    result = probe.run(model, problem)
    
    # Verify the output contains the expected dummy model prefix and the problem text
    assert "[DummyModel]" in result
    # Note: LogicProbe wraps the problem text in its own template
    assert "If A then B. A is true. What is B?" in result
```

### Additional Development Information

#### Code Style
- **Naming**: Follow PEP 8 (snake_case for functions/variables, PascalCase for classes).
- **Type Hints**: Use type hints for all new function signatures to improve maintainability and support static analysis.
- **Docstrings**: Provide Google-style docstrings for all public classes and methods.
- **Linting & Formatting**: The project uses `ruff`, `black`, `isort`, and `mypy`. Configuration for these tools is maintained in `pyproject.toml`.

#### Registries
The project uses a registry pattern for models, probes, and datasets (see `insideLLMs/registry.py`).
- **Registration**: New components can be registered using decorators (e.g., `@model_registry.register_decorator("name")`) or by adding them to `register_builtins()` in `insideLLMs/registry.py`.
- **Lookup**: Components can be retrieved by name using `model_registry.get("name")` or `probe_registry.get("name")`.

#### NLP Utilities
Many NLP utilities in `insideLLMs.nlp` have optional dependencies. Always use the provided check functions (e.g., `check_nltk()`) or handle `ImportError` when implementing new features that depend on external NLP libraries.

#### Model Implementation
All new models should inherit from `insideLLMs.models.base.Model` and implement the `generate` and `chat` methods. For asynchronous support, inherit from `insideLLMs.models.base.AsyncModel` and implement `agenerate` and `achat`.
