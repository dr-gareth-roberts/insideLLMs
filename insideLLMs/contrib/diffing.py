"""Public diffing facade.

This module provides a narrow import surface for deterministic behavioral diff
computation and snapshot-interactive helpers without requiring broad
``insideLLMs.runtime`` imports.
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
from insideLLMs.runtime.diffing_interactive import (
    SNAPSHOT_CANDIDATE_FILES,
    build_interactive_review_lines,
    copy_candidate_artifacts_to_baseline,
    print_interactive_review,
    prompt_accept_snapshot,
)

__all__ = [
    "SNAPSHOT_CANDIDATE_FILES",
    "build_diff_computation",
    "build_interactive_review_lines",
    "compute_diff_exit_code",
    "copy_candidate_artifacts_to_baseline",
    "DiffComputation",
    "DiffGatePolicy",
    "DiffJudgeComputation",
    "judge_diff_report",
    "JudgePolicy",
    "print_interactive_review",
    "prompt_accept_snapshot",
]
