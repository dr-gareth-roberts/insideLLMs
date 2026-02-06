"""Tests for insideLLMs.runner compatibility shim."""

import warnings


def test_runner_shim_emits_deprecation_warning():
    """Importing insideLLMs.runner should emit a DeprecationWarning."""
    import importlib
    import sys

    # Remove cached module so we can re-import
    mod_name = "insideLLMs.runner"
    saved = sys.modules.pop(mod_name, None)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            importlib.import_module(mod_name)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()
    finally:
        if saved is not None:
            sys.modules[mod_name] = saved


def test_runner_shim_reexports_symbols():
    """The shim should re-export key symbols from insideLLMs.runtime.runner."""
    import sys

    saved = sys.modules.pop("insideLLMs.runner", None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import insideLLMs.runner as shim

        from insideLLMs.runtime import runner as canonical

        # Check that key symbols are available
        for name in ["ProbeRunner", "AsyncProbeRunner", "run_probe"]:
            if hasattr(canonical, name):
                assert hasattr(shim, name), f"{name} not re-exported"
                assert getattr(shim, name) is getattr(canonical, name)
    finally:
        if saved is not None:
            sys.modules["insideLLMs.runner"] = saved
