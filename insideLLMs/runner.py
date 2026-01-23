"""Compatibility shim for insideLLMs.runtime.runner."""

from insideLLMs.runtime import runner as _runner

# Re-export all public + private symbols to preserve legacy imports.
for _name in dir(_runner):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_runner, _name)

del _name, _runner
