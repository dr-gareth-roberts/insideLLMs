"""Focused tests for runtime base helpers."""

import pytest

from insideLLMs.runtime._base import _invoke_progress_callback


def test_unexpected_progress_style_cache_error_propagates() -> None:
    """Only unsupported attribute assignment is treated as best-effort."""

    class ExplodingCallback:
        def __call__(self, current: int, total: int) -> None:
            return None

        def __setattr__(self, name: str, value: object) -> None:
            raise RuntimeError("unexpected cache failure")

    with pytest.raises(RuntimeError, match="unexpected cache failure"):
        _invoke_progress_callback(ExplodingCallback(), current=1, total=2, start_time=0.0)
