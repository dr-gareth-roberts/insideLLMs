from __future__ import annotations

from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_action_manifest_has_fork_safe_comment_controls() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest = _load_yaml(repo_root / "action.yml")

    inputs = manifest["inputs"]
    outputs = manifest["outputs"]
    steps = manifest["runs"]["steps"]

    assert "comment-on-forks" in inputs
    assert inputs["comment-on-forks"]["default"] == "false"
    assert "baseline-commit" in outputs
    assert "is-fork-pr" in outputs
    assert "comment-status" in outputs

    step_ids = {step.get("id") for step in steps if isinstance(step, dict)}
    assert "run_diff" in step_ids
    assert "comment_gate" in step_ids
    assert "upsert_comment" in step_ids


def test_example_workflow_uses_local_action_and_permissions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = _load_yaml(repo_root / ".github" / "workflows" / "behavioural-diff-example.yml")

    assert workflow["name"] == "Behavioural Diff (Example)"
    assert workflow["permissions"]["contents"] == "read"
    assert workflow["permissions"]["pull-requests"] == "write"

    job = workflow["jobs"]["behavioural-diff"]
    uses_steps = [
        step for step in job["steps"] if isinstance(step, dict) and "uses" in step
    ]
    assert any(step["uses"] == "./" for step in uses_steps)
