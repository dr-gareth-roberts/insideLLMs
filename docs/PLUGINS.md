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

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Plugins not discovered | `INSIDELLMS_DISABLE_PLUGINS=1` set | Unset or remove the env var |
| Import errors when loading plugin | Missing dependency or wrong module path in entry point | Check `pyproject.toml` entry point value; ensure package is installed (`pip install -e .`) |
| Registration collisions | Two plugins register the same name | Use unique names; check `model_registry`, `probe_registry`, `dataset_registry` for existing keys |
| Plugin loads but model/probe not found | Entry point runs but doesn't call `registry.register()` | Ensure `register()` is invoked with the correct registry and key |

