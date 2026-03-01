from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from insideLLMs import cli
from insideLLMs.runtime.runner import derive_run_id_from_config_path

pytestmark = pytest.mark.determinism


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_python(code: str, *, cwd: Path, extra_env: dict[str, str]) -> str:
    env = os.environ.copy()
    env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _assert_tree_bytes_equal(left: Path, right: Path, *, filenames: list[str]) -> None:
    for name in filenames:
        left_path = left / name
        right_path = right / name
        assert left_path.exists(), f"Missing file in baseline: {left_path}"
        assert right_path.exists(), f"Missing file in candidate: {right_path}"
        assert left_path.read_bytes() == right_path.read_bytes(), f"File differs: {name}"


def test_ci_harness_is_behaviorally_deterministic(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "ci" / "harness.yaml"
    dataset_path = repo_root / "ci" / "harness_dataset.jsonl"

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"

    assert cli.main(["--no-color", "harness", str(config_path), "--run-dir", str(run_a)]) == 0
    assert cli.main(["--no-color", "harness", str(config_path), "--run-dir", str(run_b)]) == 0

    expected_run_id = derive_run_id_from_config_path(config_path)
    manifest = json.loads((run_a / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == expected_run_id
    assert manifest.get("command") is None
    expected_dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert manifest.get("dataset", {}).get("dataset_hash") == f"sha256:{expected_dataset_hash}"

    assert (
        cli.main(
            [
                "--no-color",
                "diff",
                str(run_a),
                str(run_b),
                "--fail-on-changes",
            ]
        )
        == 0
    )
    _assert_tree_bytes_equal(
        run_a,
        run_b,
        filenames=[
            ".insidellms_run",
            "config.resolved.yaml",
            "manifest.json",
            "records.jsonl",
            "results.jsonl",
            "summary.json",
            "report.html",
        ],
    )


def test_run_id_is_portable_across_path_representations(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_abs = repo_root / "ci" / "harness.yaml"

    monkeypatch.chdir(repo_root)
    config_rel = Path("ci/harness.yaml")

    assert derive_run_id_from_config_path(config_abs) == derive_run_id_from_config_path(config_rel)


def test_run_id_changes_when_dataset_content_changes(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps({"example_id": "0", "question": "2+2?"}) + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  type: dummy",
                "  args: {}",
                "probe:",
                "  type: logic",
                "  args: {}",
                "dataset:",
                "  format: jsonl",
                f"  path: {dataset_path.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_id_a = derive_run_id_from_config_path(config_path)
    dataset_path.write_text(
        json.dumps({"example_id": "0", "question": "2+3?"}) + "\n",
        encoding="utf-8",
    )
    run_id_b = derive_run_id_from_config_path(config_path)

    assert run_id_a != run_id_b


def test_simple_embedding_is_independent_of_python_hash_seed() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    code = "\n".join(
        [
            "import hashlib, json",
            "from insideLLMs.rag.retrieval import SimpleEmbedding",
            "vec = SimpleEmbedding(dimension=64).embed('hello world hello')",
            "payload = json.dumps(vec, separators=(',', ':'))",
            "print(hashlib.sha256(payload.encode('utf-8')).hexdigest())",
        ]
    )

    digest_a = _run_python(code, cwd=repo_root, extra_env={"PYTHONHASHSEED": "0"})
    digest_b = _run_python(code, cwd=repo_root, extra_env={"PYTHONHASHSEED": "123"})
    assert digest_a == digest_b


def test_prompt_variant_hash_is_independent_of_python_hash_seed() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    code = "\n".join(
        [
            "from insideLLMs.prompts.prompt_testing import PromptVariant",
            "v = PromptVariant(id='variant-a', content='hi')",
            "print(hash(v))",
        ]
    )

    value_a = _run_python(code, cwd=repo_root, extra_env={"PYTHONHASHSEED": "0"})
    value_b = _run_python(code, cwd=repo_root, extra_env={"PYTHONHASHSEED": "999"})
    assert value_a == value_b


def test_tracker_run_listing_is_sorted(tmp_path: Path) -> None:
    from insideLLMs.experiment_tracking import LocalFileTracker

    tracker = LocalFileTracker(output_dir=str(tmp_path))
    project_dir = tmp_path / tracker.config.project
    (project_dir / "run_b").mkdir(parents=True)
    (project_dir / "run_a").mkdir(parents=True)

    assert tracker.list_runs() == ["run_a", "run_b"]


def test_checkpoint_listing_is_sorted(tmp_path: Path) -> None:
    from insideLLMs.system.distributed import DistributedCheckpointManager

    manager = DistributedCheckpointManager(str(tmp_path))
    manager.save("cp_b", [], [])
    manager.save("cp_a", [], [])

    assert manager.list_checkpoints() == ["cp_a", "cp_b"]


def test_report_is_idempotent(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "ci" / "harness.yaml"

    run_dir = tmp_path / "run"
    assert (
        cli.main(
            [
                "--no-color",
                "harness",
                str(config_path),
                "--run-dir",
                str(run_dir),
                "--skip-report",
            ]
        )
        == 0
    )

    assert cli.main(["--no-color", "report", str(run_dir), "--report-title", "CI Report"]) == 0
    summary_hash_1 = _sha256_bytes((run_dir / "summary.json").read_bytes())
    report_hash_1 = _sha256_bytes((run_dir / "report.html").read_bytes())

    assert cli.main(["--no-color", "report", str(run_dir), "--report-title", "CI Report"]) == 0
    summary_hash_2 = _sha256_bytes((run_dir / "summary.json").read_bytes())
    report_hash_2 = _sha256_bytes((run_dir / "report.html").read_bytes())

    assert summary_hash_1 == summary_hash_2
    assert report_hash_1 == report_hash_2


def test_async_run_matches_sync_run(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"example_id": "0", "question": "If A > B and B > C, is A > C?"}),
                json.dumps(
                    {
                        "example_id": "1",
                        "question": "All roses are flowers. All flowers need water. Do roses need water?",
                    }
                ),
                json.dumps({"example_id": "2", "question": "What is 12 * 8?"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  type: dummy",
                "  args: {}",
                "probe:",
                "  type: logic",
                "  args: {}",
                "dataset:",
                "  format: jsonl",
                f"  path: {dataset_path.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_sync = tmp_path / "run_sync"
    run_async = tmp_path / "run_async"

    assert (
        cli.main(
            [
                "--no-color",
                "run",
                str(config_path),
                "--run-dir",
                str(run_sync),
                "--format",
                "json",
            ]
        )
        == 0
    )

    assert (
        cli.main(
            [
                "--no-color",
                "run",
                str(config_path),
                "--run-dir",
                str(run_async),
                "--async",
                "--concurrency",
                "3",
                "--format",
                "json",
            ]
        )
        == 0
    )

    assert (
        cli.main(
            [
                "--no-color",
                "diff",
                str(run_sync),
                str(run_async),
                "--fail-on-changes",
            ]
        )
        == 0
    )

    _assert_tree_bytes_equal(
        run_sync,
        run_async,
        filenames=[
            ".insidellms_run",
            "config.resolved.yaml",
            "manifest.json",
            "records.jsonl",
        ],
    )
