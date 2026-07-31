"""Cover contrib.diffing re-export facade (omit-shrink candidate)."""

from __future__ import annotations

import insideLLMs.contrib.diffing as diffing


def test_contrib_diffing_reexports() -> None:
    assert "build_diff_computation" in diffing.__all__
    assert callable(diffing.build_diff_computation)
    assert callable(diffing.compute_diff_exit_code)
    assert diffing.SNAPSHOT_CANDIDATE_FILES
