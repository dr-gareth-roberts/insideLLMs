"""Tests for logging utilities."""

import logging
import tempfile
import time
from pathlib import Path

import pytest

from insideLLMs.logging_utils import (
    ColoredFormatter,
    ErrorRecord,
    ErrorTracker,
    ExperimentLog,
    ExperimentLogger,
    LogEntry,
    LogLevel,
    LoggerAdapter,
    ProgressLogger,
    StructuredFormatter,
    configure_logging,
    get_error_tracker,
    get_experiment_logger,
    get_logger,
    log_call,
    log_timing,
    log_with_context,
    progress_context,
    track_error,
)


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_all_levels_exist(self):
        """Test all log levels exist."""
        assert LogLevel.DEBUG.value == logging.DEBUG
        assert LogLevel.INFO.value == logging.INFO
        assert LogLevel.WARNING.value == logging.WARNING
        assert LogLevel.ERROR.value == logging.ERROR
        assert LogLevel.CRITICAL.value == logging.CRITICAL


class TestLogEntry:
    """Tests for LogEntry dataclass."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        from datetime import datetime

        entry = LogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level="INFO",
            message="Test message",
            module="test_module",
            function="test_func",
            extra={"key": "value"},
        )

        result = entry.to_dict()
        assert result["level"] == "INFO"
        assert result["message"] == "Test message"
        assert result["key"] == "value"


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_basic_format(self):
        """Test basic log formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert "INFO" in result
        assert "Test message" in result


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_with_level(self):
        """Test configuring with different levels."""
        configure_logging(level=LogLevel.DEBUG)
        logger = get_logger()
        assert logger.level == logging.DEBUG

    def test_configure_with_string_level(self):
        """Test configuring with string level."""
        configure_logging(level="WARNING")
        logger = get_logger()
        assert logger.level == logging.WARNING

    def test_configure_with_file(self):
        """Test configuring with file output."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name

        try:
            configure_logging(log_file=path)
            logger = get_logger()
            logger.info("Test message")

            with open(path) as f:
                content = f.read()
            assert "Test message" in content
        finally:
            Path(path).unlink()


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_root_logger(self):
        """Test getting root library logger."""
        logger = get_logger()
        assert logger.name == "insideLLMs"

    def test_get_named_logger(self):
        """Test getting named logger."""
        logger = get_logger("mymodule")
        assert logger.name == "insideLLMs.mymodule"


class TestLogTiming:
    """Tests for log_timing context manager."""

    def test_logs_timing(self):
        """Test that timing is logged."""
        configure_logging(level=LogLevel.DEBUG)

        with log_timing("test operation"):
            time.sleep(0.01)

        # Should complete without error


class TestLogCall:
    """Tests for log_call decorator."""

    def test_logs_function_call(self):
        """Test that function calls are logged."""
        configure_logging(level=LogLevel.DEBUG)

        @log_call()
        def test_func(x: int) -> int:
            return x * 2

        result = test_func(5)
        assert result == 10

    def test_logs_exceptions(self):
        """Test that exceptions are logged."""
        configure_logging(level=LogLevel.DEBUG)

        @log_call()
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_func()


class TestErrorRecord:
    """Tests for ErrorRecord dataclass."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        from datetime import datetime

        record = ErrorRecord(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            error_type="ValueError",
            message="Test error",
            traceback="...",
            context={"key": "value"},
        )

        result = record.to_dict()
        assert result["error_type"] == "ValueError"
        assert result["message"] == "Test error"


class TestErrorTracker:
    """Tests for ErrorTracker."""

    def test_record_error(self):
        """Test recording an error."""
        tracker = ErrorTracker()
        error = ValueError("Test error")
        record = tracker.record(error)

        assert record.error_type == "ValueError"
        assert record.message == "Test error"

    def test_get_recent(self):
        """Test getting recent errors."""
        tracker = ErrorTracker()
        for i in range(5):
            tracker.record(ValueError(f"Error {i}"))

        recent = tracker.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].message == "Error 4"

    def test_get_by_type(self):
        """Test getting errors by type."""
        tracker = ErrorTracker()
        tracker.record(ValueError("Error 1"))
        tracker.record(TypeError("Error 2"))
        tracker.record(ValueError("Error 3"))

        value_errors = tracker.get_by_type("ValueError")
        assert len(value_errors) == 2

    def test_error_counts(self):
        """Test error count tracking."""
        tracker = ErrorTracker()
        tracker.record(ValueError("Error 1"))
        tracker.record(ValueError("Error 2"))
        tracker.record(TypeError("Error 3"))

        counts = tracker.error_counts
        assert counts["ValueError"] == 2
        assert counts["TypeError"] == 1

    def test_clear(self):
        """Test clearing errors."""
        tracker = ErrorTracker()
        tracker.record(ValueError("Error"))
        tracker.clear()

        assert len(tracker.errors) == 0
        assert len(tracker.error_counts) == 0

    def test_summary(self):
        """Test summary generation."""
        tracker = ErrorTracker()
        tracker.record(ValueError("Error 1"))
        tracker.record(ValueError("Error 2"))

        summary = tracker.summary()
        assert summary["total_errors"] == 2
        assert summary["unique_types"] == 1


class TestTrackError:
    """Tests for track_error function."""

    def test_tracks_with_global_tracker(self):
        """Test tracking with global tracker."""
        error = ValueError("Test error")
        record = track_error(error, context={"key": "value"})

        assert record.error_type == "ValueError"
        assert record.context["key"] == "value"


class TestProgressLogger:
    """Tests for ProgressLogger."""

    def test_basic_progress(self):
        """Test basic progress logging."""
        configure_logging(level=LogLevel.INFO)
        progress = ProgressLogger(total=100, description="Test")

        progress.update(10)
        assert progress.current == 10

        progress.update(20)
        assert progress.current == 30

    def test_finish(self):
        """Test finishing progress."""
        configure_logging(level=LogLevel.INFO)
        progress = ProgressLogger(total=10, description="Test")

        for _ in range(10):
            progress.update(1)

        progress.finish()
        assert progress.current == 10


class TestProgressContext:
    """Tests for progress_context manager."""

    def test_context_manager(self):
        """Test progress context manager."""
        configure_logging(level=LogLevel.INFO)

        with progress_context(total=5, description="Test") as progress:
            for _ in range(5):
                progress.update(1)

        assert progress.current == 5


class TestExperimentLog:
    """Tests for ExperimentLog dataclass."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        from datetime import datetime

        log = ExperimentLog(
            experiment_id="exp1",
            model_id="model1",
            probe_type="logic",
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            n_samples=100,
        )

        result = log.to_dict()
        assert result["experiment_id"] == "exp1"
        assert result["model_id"] == "model1"


class TestExperimentLogger:
    """Tests for ExperimentLogger."""

    def test_start_experiment(self):
        """Test starting an experiment."""
        logger = ExperimentLogger()
        log = logger.start_experiment(
            experiment_id="exp1",
            model_id="model1",
            probe_type="logic",
            n_samples=100,
        )

        assert log.experiment_id == "exp1"
        assert log.status == "running"

    def test_log_sample_result(self):
        """Test logging sample results."""
        logger = ExperimentLogger()
        logger.start_experiment(
            experiment_id="exp1",
            model_id="model1",
            probe_type="logic",
            n_samples=100,
        )

        logger.log_sample_result("exp1", success=True)
        logger.log_sample_result("exp1", success=True)
        logger.log_sample_result("exp1", success=False, error=ValueError("Error"))

        log = logger.get_experiment("exp1")
        assert log.n_completed == 2
        assert log.n_failed == 1

    def test_finish_experiment(self):
        """Test finishing an experiment."""
        logger = ExperimentLogger()
        logger.start_experiment(
            experiment_id="exp1",
            model_id="model1",
            probe_type="logic",
            n_samples=10,
        )

        for _ in range(10):
            logger.log_sample_result("exp1", success=True)

        log = logger.finish_experiment(
            "exp1",
            status="completed",
            metrics={"accuracy": 0.95},
        )

        assert log.status == "completed"
        assert log.metrics["accuracy"] == 0.95

    def test_get_all_experiments(self):
        """Test getting all experiments."""
        logger = ExperimentLogger()
        logger.start_experiment("exp1", "model1", "logic", 10)
        logger.start_experiment("exp2", "model2", "bias", 20)

        experiments = logger.get_all_experiments()
        assert len(experiments) == 2


class TestGetExperimentLogger:
    """Tests for get_experiment_logger function."""

    def test_returns_global_logger(self):
        """Test getting global experiment logger."""
        logger1 = get_experiment_logger()
        logger2 = get_experiment_logger()
        assert logger1 is logger2
