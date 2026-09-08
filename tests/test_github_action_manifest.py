from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_action_evaluation_does_not_use_comment_credentials() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest = _load_yaml(repo_root / "action.yml")

    inputs = manifest["inputs"]
    outputs = manifest["outputs"]
    steps = manifest["runs"]["steps"]

    assert "github-token" not in inputs
    assert inputs["post-pr-comment"]["default"] == "false"
    assert "Deprecated" in inputs["post-pr-comment"]["description"]
    assert "baseline-commit" in outputs
    assert "is-fork-pr" in outputs
    assert "comment-status" in outputs

    step_ids = {step.get("id") for step in steps if isinstance(step, dict)}
    assert "run_diff" in step_ids
    assert "comment_gate" not in step_ids
    assert "upsert_comment" not in step_ids
    assert not any("github-script" in step.get("uses", "") for step in steps)
    assert "pr-report-json" in outputs


def test_legacy_action_only_invokes_retirement_script() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest = _load_yaml(repo_root / ".github/actions/diff-gate/action.yml")

    assert set(manifest["inputs"]) == {
        "config",
        "baseline-ref",
        "fail-on",
        "python-version",
        "extra-pip-args",
        "harness-args",
        "comment",
    }
    assert manifest["runs"]["steps"] == [
        {
            "name": "Explain legacy action retirement",
            "shell": "bash",
            "run": 'bash "${{ github.action_path }}/diff-gate.sh"',
        }
    ]
    manifest_text = json.dumps(manifest["runs"]["steps"])
    assert "GITHUB_TOKEN" not in manifest_text
    assert "pip install" not in manifest_text
    assert "actions/setup-python" not in manifest_text


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _compatible_bash() -> Path:
    candidates = [
        Path("/opt/homebrew/bin/bash"),
        Path("/usr/local/bin/bash"),
    ]
    path_bash = shutil.which("bash")
    if path_bash is not None:
        candidates.append(Path(path_bash))

    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        version = subprocess.run(
            [str(candidate), "-c", "printf '%s' \"${BASH_VERSINFO[0]}\""],
            capture_output=True,
            text=True,
            check=False,
        )
        if version.returncode == 0 and version.stdout.isdigit() and int(version.stdout) >= 4:
            return candidate
    pytest.skip("maintained action fixture requires an installed Bash 4 or newer")


def _make_action_history(tmp_path: Path) -> tuple[Path, str, str, str]:
    origin = tmp_path / "origin.git"
    workspace = tmp_path / "workspace"
    _git(tmp_path, "init", "--bare", str(origin))
    _git(tmp_path, "clone", str(origin), str(workspace))
    _git(workspace, "config", "user.name", "Fixture Author")
    _git(workspace, "config", "user.email", "fixture@example.invalid")
    (workspace / "ci").mkdir()
    source_ci = Path(__file__).resolve().parents[1] / "ci"
    shutil.copy(source_ci / "harness.yaml", workspace / "ci/harness.yaml")
    shutil.copy(source_ci / "harness_dataset.jsonl", workspace / "ci/harness_dataset.jsonl")
    _git(workspace, "add", "ci")
    _git(workspace, "commit", "-m", "baseline fixture")
    base_sha = _git(workspace, "rev-parse", "HEAD")
    _git(workspace, "branch", "-M", "main")
    _git(workspace, "push", "-u", "origin", "main")

    config = workspace / "ci/harness.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "args: {}", "args:\n      canned_response: OVERRIDE", 1
        ),
        encoding="utf-8",
    )
    _git(workspace, "add", "ci/harness.yaml")
    _git(workspace, "commit", "-m", "override fixture")
    override_sha = _git(workspace, "rev-parse", "HEAD")
    _git(workspace, "tag", "approved-baseline")
    _git(workspace, "push", "origin", "HEAD:main", "approved-baseline")

    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "canned_response: OVERRIDE", "canned_response: CANDIDATE"
        ),
        encoding="utf-8",
    )
    _git(workspace, "add", "ci/harness.yaml")
    _git(workspace, "commit", "-m", "candidate fixture")
    head_sha = _git(workspace, "rev-parse", "HEAD")
    return workspace, base_sha, override_sha, head_sha


def _run_maintained_action(
    tmp_path: Path, workspace: Path, *, event_base: str = "", baseline_ref: str = ""
) -> tuple[subprocess.CompletedProcess[str], dict[str, str], list[str]]:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    revision_log = tmp_path / "harness-revisions.log"
    python_wrapper = bin_dir / "python"
    python_wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == '-m' && \"$2\" == 'pip' ]]; then exit 0; fi\n"
        "if [[ \"$1\" == '-m' && \"$2\" == 'insideLLMs.cli' && \"$3\" == 'harness' ]]; then\n"
        '  /usr/bin/git rev-parse HEAD >> "$REVISION_LOG"\n'
        "fi\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n',
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)
    event_path = tmp_path / "event.json"
    if event_base:
        event_path.write_text(
            json.dumps(
                {
                    "pull_request": {
                        "number": 17,
                        "base": {"sha": event_base},
                        "head": {
                            "sha": _git(workspace, "rev-parse", "HEAD"),
                            "repo": {"fork": False},
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
    output_path = tmp_path / "github-output"
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir(exist_ok=True)
    bash_path = _compatible_bash()
    env = {
        "HOME": str(tmp_path),
        "PATH": f"{bin_dir}:{Path(bash_path).parent}:/usr/bin:/bin",
        "REVISION_LOG": str(revision_log),
        "GITHUB_WORKSPACE": str(workspace),
        "GITHUB_OUTPUT": str(output_path),
        "RUNNER_TEMP": str(runner_temp),
        "INSIDELLMS_ACTION_PATH": str(repo_root),
        "INPUT_BASELINE_REF": baseline_ref,
        "INPUT_FAIL_ON_CHANGES": "true",
    }
    if event_base:
        env["GITHUB_EVENT_PATH"] = str(event_path)
    result = subprocess.run(
        [str(bash_path), str(repo_root / "scripts/github_action_run.sh")],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    outputs = {}
    if output_path.exists():
        outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    revisions = revision_log.read_text().splitlines() if revision_log.exists() else []
    return result, outputs, revisions


@pytest.mark.parametrize("selector", ["event", "override"])
def test_maintained_action_resolves_baseline_and_compares_artifacts(
    tmp_path: Path, selector: str
) -> None:
    workspace, base_sha, override_sha, head_sha = _make_action_history(tmp_path)
    event_base = base_sha
    baseline_ref = "approved-baseline" if selector == "override" else ""

    result, outputs, revisions = _run_maintained_action(
        tmp_path, workspace, event_base=event_base, baseline_ref=baseline_ref
    )

    expected_baseline = override_sha if selector == "override" else base_sha
    assert result.returncode == 0, result.stderr
    assert outputs["diff_exit_code"] == "2"
    assert outputs["baseline_commit"] == expected_baseline
    assert revisions == [expected_baseline, head_sha]
    baseline_records = Path(outputs["baseline_run_dir"], "records.jsonl").read_text()
    candidate_records = Path(outputs["candidate_run_dir"], "records.jsonl").read_text()
    assert baseline_records != candidate_records
    assert json.loads(Path(outputs["diff_json"]).read_text())["counts"]["other_changes"] > 0


def test_maintained_action_noop_pr_still_records_event_base(tmp_path: Path) -> None:
    workspace, _, _, head_sha = _make_action_history(tmp_path)

    result, outputs, revisions = _run_maintained_action(tmp_path, workspace, event_base=head_sha)

    assert result.returncode == 0, result.stderr
    assert outputs["diff_exit_code"] == "0"
    assert outputs["baseline_commit"] == head_sha
    assert revisions == [head_sha, head_sha]
    baseline_records = Path(outputs["baseline_run_dir"], "records.jsonl").read_text()
    candidate_records = Path(outputs["candidate_run_dir"], "records.jsonl").read_text()
    assert baseline_records == candidate_records


def test_maintained_action_unknown_baseline_ref_fails_without_candidate_fallback(
    tmp_path: Path,
) -> None:
    workspace, _, _, _ = _make_action_history(tmp_path)

    result, outputs, revisions = _run_maintained_action(
        tmp_path, workspace, baseline_ref="missing-baseline"
    )

    assert result.returncode == 1
    assert "Unable to resolve baseline ref 'missing-baseline'" in result.stderr
    assert outputs["baseline_commit"] == ""
    assert revisions == []


def test_example_workflow_uses_local_action_and_permissions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = _load_yaml(repo_root / ".github" / "workflows" / "behavioural-diff-example.yml")

    assert workflow["name"] == "Behavioural Diff (Example)"
    assert workflow["permissions"]["contents"] == "read"
    assert workflow["permissions"] == {"contents": "read"}

    job = workflow["jobs"]["behavioural-diff"]
    uses_steps = [step for step in job["steps"] if isinstance(step, dict) and "uses" in step]
    assert any(step["uses"] == "./" for step in uses_steps)
    checkout = next(step for step in uses_steps if "actions/checkout@" in step["uses"])
    assert checkout["with"]["persist-credentials"] is False


def test_pr_evaluation_has_no_write_token_and_uses_strict_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = _load_yaml(root / ".github/workflows/diff-gate.yml")
    assert workflow["permissions"] == {"contents": "read"}
    steps = workflow["jobs"]["diff-gate"]["steps"]
    checkout = next(step for step in steps if "actions/checkout@" in step.get("uses", ""))
    assert checkout["with"]["persist-credentials"] is False
    action = next(step for step in steps if step.get("uses") == "./")
    assert action["with"]["fail-on-changes"] == "true"


def test_trusted_comment_workflow_never_checks_out_or_executes_candidate_code() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = _load_yaml(root / ".github/workflows/diff-gate-comment.yml")
    trigger = workflow.get("on", workflow.get(True))
    assert trigger == {"workflow_run": {"workflows": ["Diff Gate"], "types": ["completed"]}}
    assert workflow["permissions"] == {"actions": "read", "pull-requests": "write"}
    steps = workflow["jobs"]["comment"]["steps"]
    assert all(
        step.get("uses", "").startswith("actions/github-script@") or "run" in step for step in steps
    )
    text = (root / ".github/workflows/diff-gate-comment.yml").read_text()
    assert "archive.extract" not in text
    assert "pr.head.sha !== report.head_sha" in text
    assert "associated.number !== report.pr_number" in text
    assert "c.user.login === 'github-actions[bot]'" in text


@pytest.mark.parametrize("case", ["valid", "path_traversal", "extra_text", "boolean", "oversized"])
def test_trusted_workflow_validates_untrusted_archive_without_extracting(tmp_path, case):
    root = Path(__file__).resolve().parents[1]
    workflow = _load_yaml(root / ".github/workflows/diff-gate-comment.yml")
    step = next(s for s in workflow["jobs"]["comment"]["steps"] if s.get("id") == "validate")
    counts = dict.fromkeys(
        (
            "common",
            "regressions",
            "improvements",
            "other_changes",
            "only_baseline",
            "only_candidate",
            "trace_drifts",
            "trace_violation_increases",
            "trajectory_drifts",
        ),
        0,
    )
    report = {"pr_number": 42, "head_sha": "a" * 40, "exit_code": 0, "counts": counts}
    if case == "extra_text":
        report["output"] = "@everyone injected comment"
    if case == "boolean":
        counts["regressions"] = True
    payload = json.dumps(report) if case != "oversized" else "x" * 20000
    filename = "../escaped.json" if case == "path_traversal" else "pr-report.json"
    with zipfile.ZipFile(tmp_path / "insidellms-report.zip", "w") as archive:
        archive.writestr(filename, payload)
    output = tmp_path / "outputs"
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        env={**os.environ, "RUNNER_TEMP": str(tmp_path), "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is (case == "valid"), result.stderr
    if case == "valid":
        assert json.loads(output.read_text().removeprefix("report=")) == report
    else:
        assert not output.exists()
    assert not (tmp_path.parent / "escaped.json").exists()
