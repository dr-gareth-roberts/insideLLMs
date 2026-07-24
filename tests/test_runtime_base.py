"""Focused tests for runtime base helpers."""

import pytest

from insideLLMs.runtime._base import _invoke_progress_callback, _normalize_validation_mode


def test_unexpected_progress_style_cache_error_propagates() -> None:
    """Only unsupported attribute assignment is treated as best-effort."""

    class ExplodingCallback:
        def __call__(self, current: int, total: int) -> None:
            return None

        def __setattr__(self, name: str, value: object) -> None:
            raise RuntimeError("unexpected cache failure")

    with pytest.raises(RuntimeError, match="unexpected cache failure"):
        _invoke_progress_callback(ExplodingCallback(), current=1, total=2, start_time=0.0)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (None, "strict"),
        ("strict", "strict"),
        (" STRICT ", "strict"),
        ("lenient", "warn"),
        ("warn", "warn"),
    ],
)
def test_normalize_validation_mode(mode: str | None, expected: str) -> None:
    """Supported runner modes map to validator literals."""
    assert _normalize_validation_mode(mode) == expected


def test_normalize_validation_mode_rejects_unknown_value() -> None:
    """Unknown modes cannot silently fall through to strict validation."""
    with pytest.raises(ValueError, match="validation_mode"):
        _normalize_validation_mode("permissive")
