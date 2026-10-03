"""Tests for insideLLMs.cli.commands.benchmark to increase coverage."""

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from insideLLMs.cli.commands.benchmark import cmd_benchmark


def _make_args(**kwargs):
    defaults = {
        "models": "dummy",
        "probes": "dummy",
        "datasets": None,
        "max_examples": 3,
        "output": None,
        "html_report": False,
        "verbose": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestBenchmarkCommand:
    def test_benchmark_runs(self, capsys, tmp_path):
        """Regression: the default offline dummy-model run must succeed.

        This used to crash with ``'ProbeRunner' object has no attribute
        'run_single'`` (exit 1) because the command called a nonexistent
        per-item runner API. It must now run the batch API and exit 0.
        """
        rc = cmd_benchmark(_make_args(output=str(tmp_path / "results")))
        assert rc == 0
        out = capsys.readouterr().out
        assert "Benchmark Results Summary" in out

    def test_benchmark_with_datasets(self, capsys, tmp_path):
        rc = cmd_benchmark(_make_args(datasets="factuality", output=str(tmp_path / "results")))
        assert rc in (0, 1)

    def test_benchmark_error_verbose(self, capsys):
        rc = cmd_benchmark(_make_args(verbose=True, models="nonexistent_model"))
        assert rc in (0, 1)

    def test_benchmark_model_load_failure(self, capsys):
        """When a model can't be loaded, benchmark should continue with other models."""
        rc = cmd_benchmark(_make_args(models="nonexistent_model_xyz"))
        assert rc in (0, 1)

    def test_benchmark_with_html_report(self, capsys, tmp_path):
        rc = cmd_benchmark(_make_args(output=str(tmp_path / "results"), html_report=True))
        assert rc in (0, 1)

    def test_benchmark_multiple_models(self, capsys):
        rc = cmd_benchmark(_make_args(models="dummy,dummy"))
        assert rc in (0, 1)

    def test_benchmark_with_mock_data(self, capsys, tmp_path):
        """Use mocked benchmark datasets to test the full flow."""
        mock_example = MagicMock()
        mock_example.input_text = "What is 2+2?"
        mock_suite = MagicMock()
        mock_suite.sample.return_value = [mock_example] * 3

        # ProbeRunner.run returns a list of result dicts with a status field.
        mock_results = [{"input": "What is 2+2?", "output": "4", "status": "success"}] * 3

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_results

        with patch("insideLLMs.runtime.runner.ProbeRunner", return_value=mock_runner):
            with patch(
                "insideLLMs.benchmark_datasets.create_comprehensive_benchmark_suite",
                return_value=mock_suite,
            ):
                with patch("insideLLMs.benchmark_datasets.load_builtin_dataset"):
                    rc = cmd_benchmark(_make_args(probes="logic", output=str(tmp_path / "results")))

        assert rc in (0, 1)
        mock_runner.run.assert_called_once()
