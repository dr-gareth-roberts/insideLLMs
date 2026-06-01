"""Tests for CLI parsing helper utilities."""

from __future__ import annotations

import builtins
import importlib.util


def test_check_nltk_resource_returns_false_when_nltk_missing(monkeypatch) -> None:
    """_check_nltk_resource should gracefully handle missing nltk imports."""
    from insideLLMs.cli._parsing import _check_nltk_resource

    original_import = builtins.__import__

    def _patched_import(name, *args, **kwargs):
        if name == "nltk":
            raise ModuleNotFoundError("No module named 'nltk'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _patched_import)
    assert _check_nltk_resource("tokenizers/punkt") is False


def test_has_module_returns_false_when_find_spec_raises(monkeypatch) -> None:
    """_has_module should not crash on ModuleNotFoundError from dotted imports."""
    from insideLLMs.cli._parsing import _has_module

    def _raising_find_spec(_name: str):
        raise ModuleNotFoundError("No module named 'google'")

    monkeypatch.setattr(importlib.util, "find_spec", _raising_find_spec)
    assert _has_module("google.generativeai") is False
