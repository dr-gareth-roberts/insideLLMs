from __future__ import annotations

from pathlib import Path

from insideLLMs.cli._parsing import create_parser
from insideLLMs.schemas.registry import SchemaRegistry


def _subcommand_names() -> set[str]:
    parser = create_parser()
    subparsers_action = next(
        action
        for action in parser._actions
        if action.__class__.__name__ == "_SubParsersAction"
    )
    return set(subparsers_action.choices.keys())


def test_stable_cli_commands_exist() -> None:
    commands = _subcommand_names()
    assert {"run", "harness", "diff", "report", "schema"}.issubset(commands)


def test_stable_diff_flag_parse_contract() -> None:
    parser = create_parser()
    args = parser.parse_args(["diff", "run_a", "run_b", "--fail-on-changes"])
    assert args.fail_on_changes is True


def test_stable_run_determinism_flags_parse_contract() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "config.yaml",
            "--strict-serialization",
            "--deterministic-artifacts",
            "--run-dir",
            "./tmp/run",
        ]
    )
    assert args.strict_serialization is True
    assert args.deterministic_artifacts is True
    assert args.run_dir == "./tmp/run"


def test_stable_schema_registry_contract_names() -> None:
    registry = SchemaRegistry()
    for schema_name in [
        SchemaRegistry.RUNNER_ITEM,
        SchemaRegistry.RESULT_RECORD,
        SchemaRegistry.RUN_MANIFEST,
        SchemaRegistry.DIFF_REPORT,
    ]:
        assert registry.available_versions(schema_name)


def test_stability_docs_linkage_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    stability_doc = (repo_root / "docs" / "STABILITY.md").read_text(encoding="utf-8")
    matrix_doc = (repo_root / "docs" / "STABILITY_MATRIX.md").read_text(encoding="utf-8")

    assert "docs/STABILITY_MATRIX.md" in stability_doc
    assert "Stable" in matrix_doc and "Experimental" in matrix_doc and "Internal" in matrix_doc
