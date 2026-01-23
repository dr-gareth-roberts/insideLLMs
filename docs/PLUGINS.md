## Plugins (Third-party models/probes/datasets)

insideLLMs can auto-discover third-party extensions via Python entry points.

### Entry point group

Use the entry point group:

- `insidellms.plugins`

Each entry point must resolve to a callable that registers into the provided registries.

Supported signatures:

- `def register() -> None: ...`
- `def register(*, model_registry, probe_registry, dataset_registry) -> None: ...`

Plugins are enabled by default. Disable auto-loading with:

```bash
export INSIDELLMS_DISABLE_PLUGINS=1
```

### Minimal plugin example

`pyproject.toml`:

```toml
[project.entry-points."insidellms.plugins"]
my_plugin = "my_pkg.insidellms_plugin:register"
```

`my_pkg/insidellms_plugin.py`:

```python
from __future__ import annotations

from insideLLMs.models.base import Model


class MyModel(Model):
    ...


def register(*, model_registry, probe_registry, dataset_registry) -> None:
    model_registry.register("my_model", MyModel)
```

