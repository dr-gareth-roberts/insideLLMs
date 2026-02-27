"""Coverage tests for generate-suite and optimize-prompt CLI commands."""

from __future__ import annotations

import argparse
import json

from insideLLMs.cli.commands.generate_suite import cmd_generate_suite
from insideLLMs.cli.commands.optimize_prompt import cmd_optimize_prompt


def test_generate_suite_writes_jsonl(tmp_path) -> None:
    output_path = tmp_path / "suite.jsonl"
    args = argparse.Namespace(
        target="customer support bot",
        num_cases=12,
        output=str(output_path),
        format="jsonl",
        include_adversarial=True,
        model="dummy",
        model_args="{}",
        seed_example=[],
        quiet=False,
        no_color=True,
    )

    exit_code = cmd_generate_suite(args)
    assert exit_code == 0
    assert output_path.exists()

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12
    first = json.loads(lines[0])
    assert first["target_domain"] == "customer support bot"
    assert "prompt" in first


def test_generate_suite_invalid_model_args_returns_error(tmp_path) -> None:
    output_path = tmp_path / "suite.jsonl"
    args = argparse.Namespace(
        target="customer support bot",
        num_cases=5,
        output=str(output_path),
        format="jsonl",
        include_adversarial=True,
        model="dummy",
        model_args="{invalid",
        seed_example=[],
        quiet=False,
        no_color=True,
    )

    assert cmd_generate_suite(args) == 1


def test_optimize_prompt_json_report_output(tmp_path) -> None:
    output_path = tmp_path / "optimize_report.json"
    args = argparse.Namespace(
        prompt="In order to try to write a good answer basically.",
        input_file=None,
        strategies="compression,clarity",
        format="json",
        show_diff=False,
        output=str(output_path),
        quiet=False,
        no_color=True,
    )

    exit_code = cmd_optimize_prompt(args)
    assert exit_code == 0
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "optimized_prompt" in payload
    assert "token_reduction" in payload


def test_optimize_prompt_invalid_strategy_returns_error() -> None:
    args = argparse.Namespace(
        prompt="Test prompt",
        input_file=None,
        strategies="not-a-strategy",
        format="text",
        show_diff=False,
        output=None,
        quiet=False,
        no_color=True,
    )
    assert cmd_optimize_prompt(args) == 1
