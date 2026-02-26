# Migration Guide

This guide helps you migrate between major versions of insideLLMs.

## Version 2.0.0 (Upcoming)

### Breaking Changes

#### 1. Visualization Module Relocation

**What changed:**
The `insideLLMs.visualization` module has been moved to `insideLLMs.analysis.visualization` for better organization.

**Migration:**

```python
# Before (v1.x)
from insideLLMs.visualization import TraceVisualizer, create_visualizer

# After (v2.0+)
from insideLLMs.analysis.visualization import TraceVisualizer, create_visualizer
```

**Automated migration:**

```bash
# Find all deprecated imports
grep -rn "from insideLLMs.visualization import" . --include="*.py"

# Replace automatically (GNU sed)
find . -name "*.py" -exec sed -i \
  's/from insideLLMs\.visualization import/from insideLLMs.analysis.visualization import/g' {} +

# Replace automatically (macOS sed)
find . -name "*.py" -exec sed -i '' \
  's/from insideLLMs\.visualization import/from insideLLMs.analysis.visualization import/g' {} +
```

**Timeline:**
- **v1.1.0** (Current): Deprecation warnings issued, old imports still work
- **v1.2.0**: Continued warnings, compatibility layer maintained
- **v2.0.0**: Old import path removed, must use new location

**Suppressing warnings during migration:**

If you need to suppress deprecation warnings temporarily while migrating a large codebase:

```python
# Option 1: Suppress specific deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="insideLLMs.visualization")

from insideLLMs.visualization import TraceVisualizer  # No warning

# Option 2: Context manager for temporary suppression
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from insideLLMs.visualization import TraceVisualizer

# Option 3: Command-line suppression for testing
# python -W ignore::DeprecationWarning your_script.py
```

**Testing your migration:**

```python
# Verify new imports work
import sys
import warnings

# Enable all warnings
warnings.simplefilter("always", DeprecationWarning)

# Test import
try:
    from insideLLMs.analysis.visualization import TraceVisualizer
    print("✓ New import path works")
except ImportError as e:
    print(f"✗ New import failed: {e}")

# Verify old import triggers warning
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from insideLLMs.visualization import TraceVisualizer as OldViz
    
    if w and issubclass(w[0].category, DeprecationWarning):
        print("✓ Old import triggers deprecation warning")
    else:
        print("✗ Old import does not trigger warning")
```

#### 2. Other Planned Changes

No other breaking changes are currently planned for v2.0.0.

## Version 1.1.0 (Current)

### New Features

- International PII detection (EU, UK regions)
- Rate limiting integration via `RunConfig`
- Enhanced prompt injection defenses

### Deprecations

- `insideLLMs.visualization` → `insideLLMs.analysis.visualization`

### Migration Checklist

- [ ] Update visualization imports to new location
- [ ] Run tests with deprecation warnings enabled
- [ ] Update CI/CD to fail on deprecation warnings (optional)
- [ ] Review CHANGELOG for other changes

## Version 1.0.0

Initial stable release. No migrations needed.

## General Migration Tips

### 1. Enable Deprecation Warnings in Tests

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
filterwarnings = [
    "error::DeprecationWarning",  # Treat as errors in tests
]
```

### 2. Use Version Pinning During Migration

```bash
# Pin to v1.x during migration
pip install "insideLLMs>=1.1.0,<2.0.0"
```

### 3. Gradual Migration Strategy

```python
# Create compatibility layer in your codebase
# my_project/compat.py

import warnings
import sys

if sys.version_info >= (2, 0):
    # v2.0+ imports
    from insideLLMs.analysis.visualization import TraceVisualizer
else:
    # v1.x imports (with warning suppression)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from insideLLMs.visualization import TraceVisualizer

# Use in your code
from my_project.compat import TraceVisualizer
```

### 4. CI/CD Integration

```yaml
# .github/workflows/deprecation-check.yml
name: Check Deprecations

on: [push, pull_request]

jobs:
  check-deprecations:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run tests with deprecation warnings as errors
        run: |
          pytest -W error::DeprecationWarning
```

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bug Reports**: File a GitHub Issue
- **Migration Issues**: Tag with `migration` label

## Version Support Policy

- **Current major version**: Full support (bug fixes, features, security)
- **Previous major version**: Security fixes only for 12 months
- **Older versions**: No support

Example:
- v2.x (current): Full support
- v1.x: Security fixes until 12 months after v2.0.0 release
- v0.x: No support
