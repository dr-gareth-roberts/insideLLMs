# Contributing to insideLLMs

Thank you for your interest in contributing to insideLLMs! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager
- Git

### Initial Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/dr-gareth-roberts/insideLLMs.git
   cd insideLLMs
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install in development mode with all dependencies:
   ```bash
   pip install -e ".[all]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Code Style

We use the following tools for code quality:

- **Ruff**: Linting and formatting
- **MyPy**: Type checking

Run all checks locally:
```bash
ruff check .
ruff format --check .
mypy insideLLMs
```

### Testing

Run the test suite:
```bash
# Basic test run
pytest

# With coverage
pytest --cov=insideLLMs --cov-report=term

# Run specific test file
pytest tests/test_models.py

# Skip slow/integration tests
pytest -m "not slow and not integration"
```

### Documentation

Build the documentation:
```bash
pip install -e ".[docs]"
cd docs
make html
```

View locally at `docs/_build/html/index.html`.

## Making Changes

### Branching Strategy

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Make your changes with clear, atomic commits

3. Push to your fork and open a Pull Request

### Commit Messages

Follow conventional commit format:
```
type(scope): description

[optional body]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(models): add support for Gemini models
fix(registry): handle missing default kwargs
docs(readme): update installation instructions
```

### Pull Request Guidelines

1. **One feature per PR**: Keep changes focused
2. **Include tests**: Add tests for new functionality
3. **Update documentation**: Update docs if needed
4. **Pass CI**: Ensure all checks pass
5. **Describe changes**: Write clear PR descriptions

## Architecture Overview

### Package Structure

```
insideLLMs/
├── models/        # Model implementations
│   ├── base.py    # Base classes (Model, AsyncModel)
│   ├── openai.py  # OpenAI implementation
│   ├── anthropic.py
│   └── ...
├── probes/        # Probe implementations
│   ├── base.py    # Base classes (Probe, ScoredProbe)
│   ├── logic.py
│   ├── bias.py
│   └── ...
├── registry.py    # Plugin registry system
├── runner.py      # Probe execution engine
├── types.py       # Type definitions
├── exceptions.py  # Exception hierarchy
└── ...
```

### Key Design Patterns

1. **Registry Pattern**: Plugin system for models, probes, datasets
2. **Protocol-based Design**: Duck typing with `@runtime_checkable`
3. **Generic Types**: Type-safe containers (e.g., `Probe[T]`)
4. **Lazy Loading**: Heavy imports deferred until needed

### Adding a New Model

1. Create `insideLLMs/models/mymodel.py`:
   ```python
   from .base import Model, ChatMessage

   class MyModel(Model):
       def generate(self, prompt: str, **kwargs) -> str:
           ...

       def chat(self, messages: List[ChatMessage], **kwargs) -> str:
           ...

       def stream(self, prompt: str, **kwargs) -> Iterator[str]:
           ...
   ```

2. Export in `insideLLMs/models/__init__.py`

3. Register in `insideLLMs/registry.py`

4. Add tests in `tests/test_models.py`

### Adding a New Probe

1. Create `insideLLMs/probes/myprobe.py`:
   ```python
   from .base import Probe
   from insideLLMs.types import ProbeCategory

   class MyProbe(Probe[str]):
       def __init__(self, name="MyProbe"):
           super().__init__(name=name, category=ProbeCategory.CUSTOM)

       def run(self, model, data, **kwargs) -> str:
           ...
   ```

2. Export in `insideLLMs/probes/__init__.py`

3. Register in `insideLLMs/registry.py`

4. Add tests in `tests/test_probes.py`

## Questions?

- Open a [GitHub Issue](https://github.com/dr-gareth-roberts/insideLLMs/issues)
- Check existing issues for similar questions

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming environment for all contributors.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
