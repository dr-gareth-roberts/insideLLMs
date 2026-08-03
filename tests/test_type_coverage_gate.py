"""Tests for the non-regressing mypy expression-coverage gate."""

from scripts.check_type_coverage import (
    DEFAULT_TYPE_COVERAGE_THRESHOLD,
    check_type_coverage,
)


def _write_report(tmp_path, coverage: float) -> str:
    report_dir = tmp_path / "mypy-report"
    report_dir.mkdir()
    (report_dir / "any-exprs.txt").write_text(
        f"Total   100   1000   {coverage:.2f}%\n",
        encoding="utf-8",
    )
    return str(report_dir)


def test_default_threshold_tracks_current_baseline(tmp_path) -> None:
    assert DEFAULT_TYPE_COVERAGE_THRESHOLD == 0.92
    assert check_type_coverage(_write_report(tmp_path, 92.00))


def test_default_threshold_rejects_regression(tmp_path) -> None:
    assert not check_type_coverage(_write_report(tmp_path, 91.99))


def test_explicit_threshold_supports_future_increases(tmp_path) -> None:
    assert not check_type_coverage(_write_report(tmp_path, 92.50), threshold=0.93)
