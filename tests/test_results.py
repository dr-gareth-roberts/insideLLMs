"""Tests for results and reporting utilities."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

from insideLLMs.results import (
    _escape_markdown_cell,
    _format_number,
    comparison_to_markdown,
    experiment_to_html,
    experiment_to_markdown,
    generate_statistical_report,
    load_results_json,
    results_to_csv,
    results_to_markdown,
    save_results_csv,
    save_results_json,
    save_results_markdown,
)
from insideLLMs.types import (
    BenchmarkComparison,
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_escape_markdown_cell_pipe(self):
        """Test escaping pipe characters."""
        assert _escape_markdown_cell("a|b") == "a\\|b"

    def test_escape_markdown_cell_newline(self):
        """Test escaping newlines."""
        assert _escape_markdown_cell("a\nb") == "a<br>b"

    def test_escape_markdown_cell_combined(self):
        """Test escaping both pipe and newline."""
        assert _escape_markdown_cell("a|b\nc") == "a\\|b<br>c"

    def test_format_number_basic(self):
        """Test basic number formatting."""
        assert _format_number(0.12345, 2) == "0.12"
        assert _format_number(0.12345, 4) == "0.1235"

    def test_format_number_none(self):
        """Test formatting None."""
        assert _format_number(None) == "N/A"


class TestResultsToMarkdown:
    """Tests for results_to_markdown function."""

    def test_empty_results(self):
        """Test with empty results list."""
        result = results_to_markdown([])
        assert "_No results_" in result

    def test_basic_results(self):
        """Test with basic results."""
        results = [
            {"input": "test1", "output": "out1", "error": ""},
            {"input": "test2", "output": "out2", "error": "err2"},
        ]
        md = results_to_markdown(results)

        assert "| Input | Output | Error |" in md
        assert "| test1 | out1 |" in md
        assert "| test2 | out2 | err2 |" in md


class TestResultsToCSV:
    """Tests for CSV export functions."""

    def test_results_to_csv_basic(self):
        """Test basic CSV conversion."""
        results = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]
        csv = results_to_csv(results)

        assert "a,b" in csv
        assert "1,2" in csv
        assert "3,4" in csv

    def test_results_to_csv_empty(self):
        """Test CSV with empty results."""
        assert results_to_csv([]) == ""

    def test_results_to_csv_custom_fields(self):
        """Test CSV with custom fields."""
        results = [
            {"a": 1, "b": 2, "c": 3},
        ]
        csv = results_to_csv(results, fields=["a", "c"])

        assert "a,c" in csv
        assert "1,3" in csv
        assert "b" not in csv


class TestExperimentToMarkdown:
    """Tests for experiment_to_markdown function."""

    def _create_experiment(self) -> ExperimentResult:
        """Create a test experiment result."""
        return ExperimentResult(
            experiment_id="test-exp-001",
            model_info=ModelInfo(
                name="TestModel",
                provider="TestProvider",
                model_id="test-v1",
            ),
            probe_name="test_probe",
            probe_category=ProbeCategory.LOGIC,
            results=[
                ProbeResult(
                    input="test input 1",
                    output="test output 1",
                    status=ResultStatus.SUCCESS,
                    latency_ms=100.5,
                ),
                ProbeResult(
                    input="test input 2",
                    output=None,
                    status=ResultStatus.ERROR,
                    error="Test error",
                    latency_ms=50.0,
                ),
            ],
            score=ProbeScore(
                accuracy=0.8,
                precision=0.75,
                recall=0.85,
                f1_score=0.79,
                mean_latency_ms=75.25,
                error_rate=0.1,
            ),
            started_at=datetime(2024, 1, 1, 10, 0, 0),
            completed_at=datetime(2024, 1, 1, 10, 0, 30),
        )

    def test_contains_experiment_id(self):
        """Test that markdown contains experiment ID."""
        exp = self._create_experiment()
        md = experiment_to_markdown(exp)

        assert "test-exp-001" in md

    def test_contains_model_info(self):
        """Test that markdown contains model information."""
        exp = self._create_experiment()
        md = experiment_to_markdown(exp)

        assert "TestModel" in md
        assert "TestProvider" in md
        assert "test-v1" in md

    def test_contains_scores(self):
        """Test that markdown contains scores."""
        exp = self._create_experiment()
        md = experiment_to_markdown(exp)

        assert "80.0%" in md  # Accuracy
        assert "Precision" in md
        assert "Recall" in md

    def test_contains_results_table(self):
        """Test that markdown contains results table."""
        exp = self._create_experiment()
        md = experiment_to_markdown(exp)

        assert "| # | Input | Output | Status | Latency (ms) |" in md
        assert "test input 1" in md


class TestExperimentToHTML:
    """Tests for experiment_to_html function."""

    def _create_experiment(self) -> ExperimentResult:
        """Create a test experiment result."""
        return ExperimentResult(
            experiment_id="test-exp-001",
            model_info=ModelInfo(
                name="TestModel",
                provider="TestProvider",
                model_id="test-v1",
            ),
            probe_name="test_probe",
            probe_category=ProbeCategory.LOGIC,
            results=[
                ProbeResult(
                    input="test input",
                    output="test output",
                    status=ResultStatus.SUCCESS,
                    latency_ms=100.0,
                ),
            ],
            score=ProbeScore(accuracy=0.9),
        )

    def test_valid_html(self):
        """Test that output is valid HTML."""
        exp = self._create_experiment()
        html = experiment_to_html(exp)

        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "</html>" in html

    def test_contains_styles(self):
        """Test that HTML contains styles."""
        exp = self._create_experiment()
        html = experiment_to_html(exp)

        assert "<style>" in html

    def test_contains_experiment_data(self):
        """Test that HTML contains experiment data."""
        exp = self._create_experiment()
        html = experiment_to_html(exp)

        assert "TestModel" in html
        assert "90.0%" in html  # Accuracy

    def test_escapes_html_content(self):
        """Model/user content should not be interpreted as HTML in reports."""
        exp = self._create_experiment()
        exp.results[0].input = "<script>alert(1)</script>"
        exp.results[0].output = "<b>hi</b>"
        html = experiment_to_html(exp)

        assert "<script>" not in html
        assert "<b>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
        assert "&lt;b&gt;hi&lt;/b&gt;" in html


class TestComparisonToMarkdown:
    """Tests for comparison_to_markdown function."""

    def _create_comparison(self) -> BenchmarkComparison:
        """Create a test comparison."""
        experiments = [
            ExperimentResult(
                experiment_id="exp1",
                model_info=ModelInfo(name="Model1", provider="P1", model_id="m1"),
                probe_name="probe1",
                probe_category=ProbeCategory.LOGIC,
                results=[ProbeResult(input="i", output="o", status=ResultStatus.SUCCESS)],
                score=ProbeScore(accuracy=0.9),
            ),
            ExperimentResult(
                experiment_id="exp2",
                model_info=ModelInfo(name="Model2", provider="P2", model_id="m2"),
                probe_name="probe1",
                probe_category=ProbeCategory.LOGIC,
                results=[ProbeResult(input="i", output="o", status=ResultStatus.SUCCESS)],
                score=ProbeScore(accuracy=0.8),
            ),
        ]

        return BenchmarkComparison(
            name="Test Comparison",
            experiments=experiments,
            rankings={"accuracy": ["Model1", "Model2"]},
            summary={"best_model": "Model1"},
        )

    def test_contains_title(self):
        """Test that markdown contains title."""
        comp = self._create_comparison()
        md = comparison_to_markdown(comp)

        assert "# Benchmark Comparison: Test Comparison" in md

    def test_contains_experiments_table(self):
        """Test that markdown contains experiments table."""
        comp = self._create_comparison()
        md = comparison_to_markdown(comp)

        assert "Model1" in md
        assert "Model2" in md
        assert "probe1" in md

    def test_contains_rankings(self):
        """Test that markdown contains rankings."""
        comp = self._create_comparison()
        md = comparison_to_markdown(comp)

        assert "## Rankings" in md
        assert "1. Model1" in md


class TestSaveAndLoadFunctions:
    """Tests for file I/O functions."""

    def test_save_and_load_json(self):
        """Test saving and loading JSON results."""
        results = [{"input": "test", "output": "result"}]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_results_json(results, path)
            loaded = load_results_json(path)

            assert loaded == results
        finally:
            Path(path).unlink()

    def test_save_results_json_serializes_enums_and_datetimes(self, tmp_path: Path):
        """Enums and datetimes should serialize to stable JSON values."""
        experiment = ExperimentResult(
            experiment_id="exp-001",
            model_info=ModelInfo(name="Model", provider="Provider", model_id="model-1"),
            probe_name="probe",
            probe_category=ProbeCategory.LOGIC,
            results=[
                ProbeResult(
                    input="q",
                    output="a",
                    status=ResultStatus.SUCCESS,
                )
            ],
            started_at=datetime(2024, 1, 1, 0, 0, 0),
            completed_at=datetime(2024, 1, 1, 0, 0, 1),
        )

        path = tmp_path / "experiment.json"
        save_results_json(experiment, str(path))

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["started_at"] == "2024-01-01T00:00:00"
        assert payload["results"][0]["status"] == "success"

    def test_save_markdown(self):
        """Test saving markdown results."""
        results = [{"input": "test", "output": "result", "error": ""}]

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name

        try:
            save_results_markdown(results, path)
            content = Path(path).read_text()

            assert "| test | result |" in content
        finally:
            Path(path).unlink()

    def test_save_csv(self):
        """Test saving CSV results."""
        results = [{"a": 1, "b": 2}]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            save_results_csv(results, path)
            content = Path(path).read_text()

            assert "a,b" in content
            assert "1,2" in content
        finally:
            Path(path).unlink()


class TestStatisticalReport:
    """Tests for statistical report generation."""

    def _create_experiments(self, n: int = 5) -> list:
        """Create test experiments."""
        experiments = []
        for i in range(n):
            experiments.append(
                ExperimentResult(
                    experiment_id=f"exp-{i}",
                    model_info=ModelInfo(
                        name="TestModel",
                        provider="TestProvider",
                        model_id="test-v1",
                    ),
                    probe_name="test_probe",
                    probe_category=ProbeCategory.LOGIC,
                    results=[
                        ProbeResult(
                            input=f"input-{j}",
                            output=f"output-{j}",
                            status=ResultStatus.SUCCESS,
                            latency_ms=100.0 + j,
                        )
                        for j in range(10)
                    ],
                    score=ProbeScore(accuracy=0.8 + i * 0.02),
                )
            )
        return experiments

    def test_markdown_report(self):
        """Test markdown statistical report."""
        experiments = self._create_experiments()
        report = generate_statistical_report(experiments, format="markdown")

        assert "# Statistical Analysis Report" in report
        assert "TestModel" in report

    def test_html_report(self):
        """Test HTML statistical report."""
        experiments = self._create_experiments()
        report = generate_statistical_report(experiments, format="html")

        assert "<!DOCTYPE html>" in report
        assert "Statistical Analysis Report" in report

    def test_json_report(self):
        """Test JSON statistical report."""
        experiments = self._create_experiments()
        report = generate_statistical_report(experiments, format="json")

        data = json.loads(report)
        assert "total_experiments" in data
        assert data["total_experiments"] == 5

    def test_empty_experiments(self):
        """Test with empty experiments list."""
        report = generate_statistical_report([])
        assert "No experiments" in report

    def test_save_to_file(self):
        """Test saving report to file."""
        experiments = self._create_experiments()

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name

        try:
            generate_statistical_report(experiments, output_path=path)
            content = Path(path).read_text()

            assert "# Statistical Analysis Report" in content
        finally:
            Path(path).unlink()
