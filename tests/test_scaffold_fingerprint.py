"""Tests for scaffold identity fingerprinting.

Verify that scaffold.id and scaffold.version affect run_id deterministically
and that custom.scaffold appears in manifest.json and records.jsonl when set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from insideLLMs import cli
from insideLLMs.runtime.runner import derive_run_id_from_config_path

pytestmark = pytest.mark.determinism


def _write_config(path: Path, scaffold: dict[str, str] | None = None) -> None:
    """Write a minimal config with optional scaffold section."""
    lines = [
        "model:",
        "  type: dummy",
        "  args: {}",
        "probe:",
        "  type: logic",
        "  args: {}",
        "dataset:",
        "  format: inline",
        "  data:",
        '    - {example_id: "0", question: "If A > B and B > C, is A > C?"}',
    ]
    if scaffold is not None:
        lines.append("scaffold:")
        if scaffold.get("id"):
            lines.append(f'  id: "{scaffold["id"]}"')
        if scaffold.get("version"):
            lines.append(f'  version: "{scaffold["version"]}"')
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_different_scaffold_id_produces_different_run_id(tmp_path: Path) -> None:
    """Runs with different scaffold.id values must have different run_ids."""
    config_a = tmp_path / "config_a.yaml"
    config_b = tmp_path / "config_b.yaml"

    _write_config(config_a, scaffold={"id": "scaffold-alpha", "version": "1.0.0"})
    _write_config(config_b, scaffold={"id": "scaffold-beta", "version": "1.0.0"})

    run_id_a = derive_run_id_from_config_path(config_a)
    run_id_b = derive_run_id_from_config_path(config_b)

    assert run_id_a != run_id_b, "Different scaffold.id should produce different run_id"


def test_different_scaffold_version_produces_different_run_id(tmp_path: Path) -> None:
    """Runs with different scaffold.version values must have different run_ids."""
    config_a = tmp_path / "config_a.yaml"
    config_b = tmp_path / "config_b.yaml"

    _write_config(config_a, scaffold={"id": "my-scaffold", "version": "1.0.0"})
    _write_config(config_b, scaffold={"id": "my-scaffold", "version": "2.0.0"})

    run_id_a = derive_run_id_from_config_path(config_a)
    run_id_b = derive_run_id_from_config_path(config_b)

    assert run_id_a != run_id_b, "Different scaffold.version should produce different run_id"


def test_same_scaffold_produces_same_run_id(tmp_path: Path) -> None:
    """Runs with identical scaffold values must have the same run_id."""
    config_a = tmp_path / "config_a.yaml"
    config_b = tmp_path / "config_b.yaml"

    _write_config(config_a, scaffold={"id": "my-scaffold", "version": "1.0.0"})
    _write_config(config_b, scaffold={"id": "my-scaffold", "version": "1.0.0"})

    run_id_a = derive_run_id_from_config_path(config_a)
    run_id_b = derive_run_id_from_config_path(config_b)

    assert run_id_a == run_id_b, "Same scaffold should produce same run_id"


def test_omitted_scaffold_matches_baseline(tmp_path: Path) -> None:
    """When scaffold is omitted, behavior matches current (no custom.scaffold)."""
    config_path = tmp_path / "config.yaml"
    run_dir = tmp_path / "run"

    _write_config(config_path, scaffold=None)

    exit_code = cli.main(
        [
            "--no-color",
            "run",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--format",
            "json",
        ]
    )
    assert exit_code == 0

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "scaffold" not in manifest.get("custom", {}), (
        "custom.scaffold should not appear when scaffold is omitted from config"
    )

    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) > 0
    for record in records:
        assert "scaffold" not in record.get("custom", {}), (
            "custom.scaffold should not appear in records when scaffold is omitted"
        )


def test_scaffold_emitted_in_manifest_and_records(tmp_path: Path) -> None:
    """When scaffold is present, custom.scaffold appears in manifest and records."""
    config_path = tmp_path / "config.yaml"
    run_dir = tmp_path / "run"

    _write_config(config_path, scaffold={"id": "test-scaffold", "version": "1.2.3"})

    exit_code = cli.main(
        [
            "--no-color",
            "run",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--format",
            "json",
        ]
    )
    assert exit_code == 0

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "scaffold" in manifest.get("custom", {}), (
        "custom.scaffold should appear in manifest when scaffold is set"
    )
    assert manifest["custom"]["scaffold"]["id"] == "test-scaffold"
    assert manifest["custom"]["scaffold"]["version"] == "1.2.3"

    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) > 0
    for record in records:
        assert "scaffold" in record.get("custom", {}), (
            "custom.scaffold should appear in records when scaffold is set"
        )
        assert record["custom"]["scaffold"]["id"] == "test-scaffold"
        assert record["custom"]["scaffold"]["version"] == "1.2.3"


def test_scaffold_in_resolved_config(tmp_path: Path) -> None:
    """Scaffold should appear in config.resolved.yaml."""
    config_path = tmp_path / "config.yaml"
    run_dir = tmp_path / "run"

    _write_config(config_path, scaffold={"id": "resolved-test", "version": "0.1.0"})

    exit_code = cli.main(
        [
            "--no-color",
            "run",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--format",
            "json",
        ]
    )
    assert exit_code == 0

    import yaml

    resolved_config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert "scaffold" in resolved_config
    assert resolved_config["scaffold"]["id"] == "resolved-test"
    assert resolved_config["scaffold"]["version"] == "0.1.0"


def test_omitted_scaffold_same_run_id_as_before(tmp_path: Path) -> None:
    """Two configs without scaffold should have the same run_id (baseline stability)."""
    config_a = tmp_path / "config_a.yaml"
    config_b = tmp_path / "config_b.yaml"

    _write_config(config_a, scaffold=None)
    _write_config(config_b, scaffold=None)

    run_id_a = derive_run_id_from_config_path(config_a)
    run_id_b = derive_run_id_from_config_path(config_b)

    assert run_id_a == run_id_b, "Two configs without scaffold should produce identical run_ids"


def test_scaffold_with_vs_without_changes_run_id(tmp_path: Path) -> None:
    """Adding scaffold to a config must change the run_id."""
    config_without = tmp_path / "config_without.yaml"
    config_with = tmp_path / "config_with.yaml"

    _write_config(config_without, scaffold=None)
    _write_config(config_with, scaffold={"id": "new-scaffold", "version": "1.0.0"})

    run_id_without = derive_run_id_from_config_path(config_without)
    run_id_with = derive_run_id_from_config_path(config_with)

    assert run_id_without != run_id_with, "Adding scaffold should change the run_id"
