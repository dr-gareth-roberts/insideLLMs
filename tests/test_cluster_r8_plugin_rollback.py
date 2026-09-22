"""R8: plugin discovery must not deepcopy registered factories/kwargs."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class _NoDeepCopy:
    """Object that refuses deepcopy — models a lock-like default kwarg."""

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:  # noqa: ARG002
        raise TypeError("cannot deepcopy _NoDeepCopy")


def _make_eps(entry_points: list[Any]) -> Any:
    class _Eps:
        def select(self, group: str) -> list[Any]:  # noqa: ARG002
            return list(entry_points)

    return _Eps()


def _ep(name: str, value: str, fn: Any) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.value = value
    ep.load.return_value = fn
    return ep


class TestPluginRollbackStructuralSnapshot:
    def test_non_deepcopyable_preloaded_kwargs_do_not_abort_discovery(self) -> None:
        from insideLLMs import registry as reg_mod

        model_registry = reg_mod.model_registry
        probe_registry = reg_mod.probe_registry

        pre_name = "_r8_preloaded_lock_model"
        bad_partial = "_r8_failing_plugin_partial"
        good_name = "_r8_good_plugin_probe"
        lock = threading.Lock()

        for reg, name in (
            (model_registry, pre_name),
            (model_registry, bad_partial),
            (probe_registry, good_name),
        ):
            if name in reg:
                reg.unregister(name)

        model_registry.register(pre_name, lambda **_kw: "pre", lock=lock)
        pre_entry_before = model_registry._registry[pre_name]

        def bad_plugin(**_kwargs: Any) -> None:
            model_registry.register(bad_partial, lambda: "partial")
            raise RuntimeError("plugin boom")

        def good_plugin(**_kwargs: Any) -> None:
            probe_registry.register(good_name, lambda: "good")

        # Sorted by name: bad before good, so a snapshot crash would skip good.
        eps = [
            _ep("bad_r8", "bad_r8:register", bad_plugin),
            _ep("good_r8", "good_r8:register", good_plugin),
        ]

        try:
            with patch.object(reg_mod.metadata, "entry_points", return_value=_make_eps(eps)):
                with pytest.warns(RuntimeWarning, match="Failed to load plugin 'bad_r8'"):
                    loaded = reg_mod.load_entrypoint_plugins(enabled=True)

            assert loaded == {"good_r8": "good_r8:register"}
            assert bad_partial not in model_registry
            assert good_name in probe_registry
            assert pre_name in model_registry
            # Pre-existing entry restored by reference (same object, same lock).
            assert model_registry._registry[pre_name] is pre_entry_before
            assert model_registry._registry[pre_name]["default_kwargs"]["lock"] is lock
        finally:
            for reg, name in (
                (model_registry, pre_name),
                (model_registry, bad_partial),
                (probe_registry, good_name),
            ):
                if name in reg:
                    reg.unregister(name)

    def test_failing_plugin_partial_registrations_roll_back(self) -> None:
        from insideLLMs import registry as reg_mod

        model_registry = reg_mod.model_registry
        probe_registry = reg_mod.probe_registry
        dataset_registry = reg_mod.dataset_registry

        names = {
            "model": "_r8_partial_model",
            "probe": "_r8_partial_probe",
            "dataset": "_r8_partial_dataset",
        }
        registries = {
            "model": model_registry,
            "probe": probe_registry,
            "dataset": dataset_registry,
        }
        for key, name in names.items():
            if name in registries[key]:
                registries[key].unregister(name)

        # Also pre-register a non-deepcopyable value so rollback path is exercised
        # under the same structural snapshot used for multi-registry residue.
        pre_name = "_r8_pre_nodc"
        if pre_name in model_registry:
            model_registry.unregister(pre_name)
        model_registry.register(pre_name, lambda **_kw: "x", sentinel=_NoDeepCopy())

        def bad_plugin(**_kwargs: Any) -> None:
            model_registry.register(names["model"], lambda: "m")
            probe_registry.register(names["probe"], lambda: "p")
            dataset_registry.register(names["dataset"], lambda: "d")
            raise RuntimeError("multi residue boom")

        ep = _ep("multi_bad", "multi_bad:register", bad_plugin)

        try:
            with patch.object(reg_mod.metadata, "entry_points", return_value=_make_eps([ep])):
                with pytest.warns(RuntimeWarning, match="Failed to load plugin 'multi_bad'"):
                    loaded = reg_mod.load_entrypoint_plugins(enabled=True)

            assert loaded == {}
            for key, name in names.items():
                assert name not in registries[key]
            assert pre_name in model_registry
        finally:
            if pre_name in model_registry:
                model_registry.unregister(pre_name)
            for key, name in names.items():
                if name in registries[key]:
                    registries[key].unregister(name)

    def test_snapshot_does_not_deepcopy_entry_values(self) -> None:
        """Structural snapshot keeps factories/kwargs by reference."""
        from insideLLMs import registry as reg_mod

        model_registry = reg_mod.model_registry
        name = "_r8_snapshot_ref_model"
        if name in model_registry:
            model_registry.unregister(name)

        sentinel = _NoDeepCopy()
        model_registry.register(name, lambda **_kw: "y", obj=sentinel)
        try:
            snap = reg_mod._snapshot_registry_state()
            assert name in snap["model"]
            assert snap["model"][name] is model_registry._registry[name]
            assert snap["model"][name]["default_kwargs"]["obj"] is sentinel
        finally:
            model_registry.unregister(name)
