### Project Guidelines

This document provides project-specific information for developers working on `insideLLMs`.

### Build and Configuration

The project uses `pyproject.toml` for dependency management and can be installed via `pip`.

#### Installation
- **Base package**: `pip install .`
- **With NLP utilities**: `pip install .[nlp]` (Installs `nltk`, `spacy`, `scikit-learn`, `gensim`)
- **With Visualization**: `pip install .[visualization]` (Installs `matplotlib`, `pandas`, `seaborn`)
- **All optional dependencies**: `pip install .[nlp,visualization]`

#### Development Environment
For development, install the dependencies from `requirements-dev.txt`:
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
    assert "Solve this logic problem: If A then B. A is true. What is B?" in result
```

### Additional Development Information

#### Code Style
- **Naming**: Follow PEP 8 (snake_case for functions/variables, PascalCase for classes).
- **Type Hints**: Use type hints for all new function signatures to improve maintainability.
- **Docstrings**: Provide docstrings for all public classes and methods.
- **Registries**: The project uses a registry pattern for models, probes, and datasets in `insideLLMs/__init__.py`. When adding new core components, consider registering them if applicable.

#### NLP Utilities
Many NLP utilities in `insideLLMs.nlp` (like `nltk_tokenize` or `spacy_tokenize`) have optional dependencies. Always use the provided check functions (e.g., `check_nltk()`) or handle `ImportError` when implementing new features that depend on these libraries.

#### Model Implementation
All new models should inherit from `insideLLMs.models.base.Model` and implement the `generate` and `chat` methods.
