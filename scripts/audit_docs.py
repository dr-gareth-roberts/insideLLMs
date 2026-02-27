#!/usr/bin/env python3
"""Audit high-impact docs coverage against current CLI/runtime surfaces."""

from __future__ import annotations

import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_section(markdown: str, heading: str) -> str | None:
    pattern = re.compile(
        rf"(?ms)^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s+|\Z)",
    )
    match = pattern.search(markdown)
    return match.group(1) if match else None


def _cli_options_for_command(command: str) -> set[str]:
    parser = _create_parser()
    subparsers_action = next(
        action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
    )
    subparser = subparsers_action.choices[command]
    options: set[str] = set()
    for action in subparser._actions:
        for option in action.option_strings:
            if option.startswith("--"):
                options.add(option)
    return options


def _missing_tokens(text: str, tokens: Iterable[str]) -> list[str]:
    return [token for token in tokens if token not in text]


def _create_parser() -> ArgumentParser:
    from insideLLMs.cli._parsing import create_parser

    return create_parser()


def _smoke_parse(parser: ArgumentParser, argv: list[str]) -> str | None:
    try:
        parser.parse_args(argv)
    except SystemExit as exc:
        if int(exc.code or 0) != 0:
            return f"CLI parser smoke check failed: {' '.join(argv)}"
    except Exception as exc:  # pragma: no cover - defensive
        return f"CLI parser smoke check errored for {' '.join(argv)}: {exc}"
    return None


def main() -> int:
    repo_root = REPO_ROOT
    failures: list[str] = []
    parser = _create_parser()

    cli_reference_path = repo_root / "wiki" / "reference" / "CLI.md"
    cli_reference = _read(cli_reference_path)
    harness_section = _extract_section(cli_reference, "harness")
    diff_section = _extract_section(cli_reference, "diff")

    if harness_section is None:
        failures.append("wiki/reference/CLI.md: missing '## harness' section")
    if diff_section is None:
        failures.append("wiki/reference/CLI.md: missing '## diff' section")

    harness_options = _cli_options_for_command("harness")
    diff_options = _cli_options_for_command("diff")

    harness_expected = [
        "--profile",
        "--explain",
        "--active-red-team",
        "--red-team-rounds",
        "--red-team-attempts-per-round",
        "--red-team-target-system-prompt",
    ]
    diff_expected = [
        "--interactive",
        "--judge",
        "--judge-policy",
        "--judge-limit",
        "--fail-on-trajectory-drift",
    ]

    for token in harness_expected:
        if token not in harness_options:
            failures.append(f"CLI harness parser missing expected option: {token}")
    for token in diff_expected:
        if token not in diff_options:
            failures.append(f"CLI diff parser missing expected option: {token}")

    if harness_section is not None:
        for token in _missing_tokens(harness_section, harness_expected):
            failures.append(f"wiki/reference/CLI.md harness section missing token: {token}")

    if diff_section is not None:
        for token in _missing_tokens(diff_section, diff_expected):
            failures.append(f"wiki/reference/CLI.md diff section missing token: {token}")
        if "--fail-on-trajectory-drift" in diff_options and "| `5` |" not in diff_section:
            failures.append(
                "wiki/reference/CLI.md diff section missing exit-code table row for code 5"
            )

    readme = _read(repo_root / "README.md")
    for token in [
        "--active-red-team",
        "--fail-on-trajectory-drift",
        "shadow.fastapi",
        "dr-gareth-roberts/insideLLMs@v1",
    ]:
        if token not in readme:
            failures.append(f"README.md missing expected token: {token}")

    documentation_index = _read(repo_root / "DOCUMENTATION_INDEX.md")
    for token in [
        "docs/GITHUB_ACTION.md",
        "extensions/vscode-insidellms/README.md",
        "wiki/guides/Production-Shadow-Capture.md",
        "wiki/guides/IDE-Extension-Workflow.md",
    ]:
        if token not in documentation_index:
            failures.append(f"DOCUMENTATION_INDEX.md missing expected entry: {token}")

    api_reference = _read(repo_root / "API_REFERENCE.md")
    for token in [
        "insideLLMs.diffing",
        "DiffGatePolicy",
        "insideLLMs.shadow",
        "shadow.fastapi",
        "insidellms harness harness.yaml --profile healthcare-hipaa --explain",
        "insidellms diff ./baseline ./candidate --judge --judge-policy balanced",
        "insidellms diff ./baseline ./candidate --interactive --fail-on-changes",
        "insidellms doctor --format json --capabilities",
    ]:
        if token not in api_reference:
            failures.append(f"API_REFERENCE.md missing expected token: {token}")

    stale_token_files = [
        repo_root / "wiki" / "tutorials" / "CI-Integration.md",
        repo_root / "wiki" / "guides" / "Troubleshooting.md",
        repo_root / "wiki" / "reference" / "CLI.md",
        repo_root / "wiki" / "guides" / "index.md",
        repo_root / "wiki" / "reference" / "Configuration.md",
        repo_root / "wiki" / "concepts" / "Artifacts.md",
        repo_root / "wiki" / "concepts" / "Datasets.md",
        repo_root / "wiki" / "getting-started" / "First-Harness.md",
    ]
    stale_tokens = [
        "--ignore-fields",
        "--trace-aware",
        "insidellms doctor --verbose",
        "insidellms run config.yaml --debug",
        "--dry-run",
        "--summary-only",
        "--model-override",
        "--max-examples",
    ]
    for path in stale_token_files:
        if not path.exists():
            failures.append(f"docs audit target missing: {path.relative_to(repo_root)}")
            continue
        text = _read(path)
        for token in stale_tokens:
            if token in text:
                failures.append(f"{path.relative_to(repo_root)} contains stale token: {token}")

    smoke_cases = [
        [
            "harness",
            "harness.yaml",
            "--profile",
            "healthcare-hipaa",
            "--explain",
            "--active-red-team",
            "--red-team-rounds",
            "2",
            "--red-team-attempts-per-round",
            "10",
        ],
        [
            "diff",
            "base",
            "head",
            "--fail-on-changes",
            "--fail-on-trajectory-drift",
            "--judge",
            "--judge-policy",
            "balanced",
            "--judge-limit",
            "50",
            "--interactive",
        ],
        ["doctor", "--format", "json", "--capabilities"],
        ["init", "experiment.yaml", "--model", "dummy", "--probe", "logic"],
    ]
    for argv in smoke_cases:
        error = _smoke_parse(parser, argv)
        if error:
            failures.append(error)

    extension_readme_path = repo_root / "extensions" / "vscode-insidellms" / "README.md"
    if not extension_readme_path.exists():
        failures.append("extensions/vscode-insidellms/README.md missing")
    else:
        extension_readme = _read(extension_readme_path)
        if "Run insideLLMs probes" not in extension_readme:
            failures.append("extensions/vscode-insidellms/README.md missing 'Run insideLLMs probes'")

    if failures:
        print("Documentation audit issues detected:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Documentation audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
