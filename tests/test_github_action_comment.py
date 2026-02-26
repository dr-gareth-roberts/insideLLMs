from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_comment_builder_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "github_action_build_comment.py"
    spec = importlib.util.spec_from_file_location("github_action_build_comment", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_comment_contains_marker_and_counts() -> None:
    module = _load_comment_builder_module()
    payload = {
        "baseline": "/tmp/base",
        "candidate": "/tmp/head",
        "counts": {
            "common": 4,
            "regressions": 1,
            "improvements": 2,
            "other_changes": 1,
            "only_baseline": 0,
            "only_candidate": 1,
            "trace_drifts": 0,
            "trace_violation_increases": 0,
        },
        "regressions": [
            {
                "label": {"model": "dummy", "probe": "logic", "example": "e1"},
                "detail": "status success -> error",
            }
        ],
        "improvements": [],
        "changes": [],
        "only_baseline": [],
        "only_candidate": [],
        "trace_drifts": [],
        "trace_violation_increases": [],
    }

    comment = module.build_comment(payload, diff_exit_code=2)
    assert "<!-- insidellms-diff-comment -->" in comment
    assert "| Regressions | 1 |" in comment
    assert "`dummy | logic | example e1`: status success -> error" in comment


def test_build_comment_handles_error_payload() -> None:
    module = _load_comment_builder_module()
    comment = module.build_comment({"error": "boom"}, diff_exit_code=1)
    assert "Diff command error" in comment
    assert "boom" in comment
