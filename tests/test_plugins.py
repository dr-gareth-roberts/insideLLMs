"""Tests for plugin entry point loading."""

from __future__ import annotations

from insideLLMs.registry import load_entrypoint_plugins


def test_load_entrypoint_plugins_empty_group_returns_empty():
    loaded = load_entrypoint_plugins(group="insidellms.plugins.__test_nonexistent__", enabled=True)
    assert loaded == {}


def test_load_entrypoint_plugins_disabled_returns_empty():
    loaded = load_entrypoint_plugins(enabled=False)
    assert loaded == {}
