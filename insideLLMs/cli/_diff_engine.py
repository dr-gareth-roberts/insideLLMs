"""Backward-compatible import shim for diff computation.

Use :mod:`insideLLMs.runtime.diffing` for the supported public API.
"""

from insideLLMs.runtime.diffing import (
    DiffComputation,
    DiffGatePolicy,
    DiffJudgeComputation,
    JudgePolicy,
    build_diff_computation,
    compute_diff_exit_code,
    judge_diff_report,
)

__all__ = [
    "compute_diff_exit_code",
    "DiffComputation",
    "DiffGatePolicy",
    "DiffJudgeComputation",
    "JudgePolicy",
    "build_diff_computation",
    "judge_diff_report",
]
