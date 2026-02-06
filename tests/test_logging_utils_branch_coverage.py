"""Additional branch coverage for logging utilities."""

from __future__ import annotations

import logging
import time
from unittest.mock import Mock

import pytest

import insideLLMs.logging_utils as logging_utils


def _record(level: int = logging.INFO, msg: str = "hello") -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=level,
        pathname="test_file.py",
        lineno=12,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_structured_formatter_extra_data_truthy_and_empty_paths():
    formatter = logging_utils.StructuredFormatter(include_extra=True)
    record = _record()
    record.extra_data = {"user_id": "u1", "action": "run"}
    formatted = formatter.format(record)
    assert "user_id=u1" in formatted
    assert "action=run" in formatted

    empty_record = _record(msg="no extra")
    empty_record.extra_data = {}
    empty_formatted = formatter.format(empty_record)
    assert "no extra" in empty_formatted
    assert "user_id=u1" not in empty_formatted


def test_colored_formatter_wraps_output_with_color_codes():
    formatter = logging_utils.ColoredFormatter()
    record = _record(level=logging.ERROR, msg="boom")
    formatted = formatter.format(record)
    assert formatted.startswith(logging_utils.ColoredFormatter.COLORS["ERROR"])
    assert formatted.endswith(logging_utils.ColoredFormatter.RESET)


def test_configure_logging_colored_tty_branch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(logging_utils.sys.stderr, "isatty", lambda: True)
    logging_utils.configure_logging(level="INFO", colored=True, include_extra=True)
    handler = logging_utils.get_logger().handlers[0]
    assert isinstance(handler.formatter, logging_utils.ColoredFormatter)


def test_get_module_logger_prefix_and_root_paths():
    assert logging_utils.get_module_logger("insideLLMs") is logging_utils.get_logger()
    assert logging_utils.get_module_logger("insideLLMs.trace").name == "insideLLMs.trace"
    assert logging_utils.get_module_logger("custom.module").name == "insideLLMs.custom.module"


def test_logger_adapter_process_merges_extra_data():
    adapter = logging_utils.LoggerAdapter(logging.getLogger("adapter_test"), {"request_id": "r1"})
    msg, kwargs = adapter.process("message", {"extra": {"event": "start"}})
    assert msg == "message"
    assert kwargs["extra"]["event"] == "start"
    assert kwargs["extra"]["request_id"] == "r1"
    assert kwargs["extra"]["extra_data"]["event"] == "start"


def test_log_with_context_returns_logger_adapter():
    adapter = logging_utils.log_with_context(logging.getLogger("ctx_logger"), run_id="abc")
    assert isinstance(adapter, logging_utils.LoggerAdapter)
    assert adapter.extra["run_id"] == "abc"


def test_log_call_no_args_and_log_result_branch(monkeypatch: pytest.MonkeyPatch):
    logger_mock = Mock()
    monkeypatch.setattr(logging_utils, "get_logger", lambda _name=None: logger_mock)

    @logging_utils.log_call(log_args=False, log_result=True, log_timing_flag=False)
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    logged_messages = [call.args[1] for call in logger_mock.log.call_args_list]
    assert any(msg == "Calling add" for msg in logged_messages)
    assert any("add returned 5" in msg for msg in logged_messages)


def test_log_call_timing_without_result_branch(monkeypatch: pytest.MonkeyPatch):
    logger_mock = Mock()
    monkeypatch.setattr(logging_utils, "get_logger", lambda _name=None: logger_mock)

    @logging_utils.log_call(log_args=False, log_result=False, log_timing_flag=True)
    def identity(x: str) -> str:
        return x

    assert identity("ok") == "ok"
    logged_messages = [call.args[1] for call in logger_mock.log.call_args_list]
    assert any(msg == "Calling identity" for msg in logged_messages)
    assert any("identity completed in" in msg for msg in logged_messages)


def test_error_tracker_trims_to_max_errors():
    tracker = logging_utils.ErrorTracker(max_errors=2)
    for i in range(3):
        tracker.record(ValueError(f"err-{i}"))
    assert len(tracker.errors) == 2
    assert tracker.errors[0].message == "err-1"
    assert tracker.errors[1].message == "err-2"


def test_progress_logger_interval_and_zero_total_paths():
    logger_mock = Mock()
    progress = logging_utils.ProgressLogger(
        total=0,
        description="ZeroTotal",
        logger_instance=logger_mock,
        log_interval=0,
    )
    progress.update(1)
    assert logger_mock.info.called

    # Force elapsed <= 0 path; this should still log without error.
    progress.start_time = time.time() + 10
    progress.current = 0
    progress._log_progress()
    latest_message = logger_mock.info.call_args[0][0]
    assert "ZeroTotal" in latest_message
    assert "(0.0%)" in latest_message


def test_experiment_logger_unknown_and_optional_metrics_paths():
    exp_logger = logging_utils.ExperimentLogger(logger_instance=Mock())
    # Unknown experiment id branches
    exp_logger.log_sample_result("missing", success=True)
    assert exp_logger.finish_experiment("missing") is None

    exp_logger.start_experiment("exp-branch", "m1", "probe", 2)
    exp_logger.log_sample_result("exp-branch", success=False, error=None)
    result = exp_logger.finish_experiment("exp-branch", status="completed", metrics=None)
    assert result is not None
    assert result.status == "completed"
    assert result.n_failed == 1
