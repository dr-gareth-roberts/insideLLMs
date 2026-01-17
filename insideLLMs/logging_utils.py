"""Logging utilities for insideLLMs.

This module provides structured logging capabilities for:
- Experiment tracking and progress
- Model API call logging
- Error tracking and diagnostics
- Performance monitoring
"""

import functools
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union


# Create library logger
logger = logging.getLogger("insideLLMs")


class LogLevel(Enum):
    """Log levels for the library."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogEntry:
    """Structured log entry."""

    timestamp: datetime
    level: str
    message: str
    module: str = ""
    function: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "module": self.module,
            "function": self.function,
            **self.extra,
        }


class StructuredFormatter(logging.Formatter):
    """Formatter that produces structured log output."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        include_extra: bool = True,
    ):
        super().__init__(fmt, datefmt)
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record."""
        # Get basic info
        timestamp = datetime.fromtimestamp(record.created)
        level = record.levelname

        # Build message
        message = record.getMessage()

        # Add location info
        location = f"{record.module}.{record.funcName}:{record.lineno}"

        # Format base message
        base = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {level:8s} | {location:40s} | {message}"

        # Add extra fields if present
        if self.include_extra and hasattr(record, "extra_data"):
            extra = record.extra_data
            if extra:
                extra_str = " | ".join(f"{k}={v}" for k, v in extra.items())
                base = f"{base} | {extra_str}"

        return base


class ColoredFormatter(StructuredFormatter):
    """Formatter with colored output for terminal."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format with colors."""
        base = super().format(record)
        color = self.COLORS.get(record.levelname, "")
        return f"{color}{base}{self.RESET}"


def configure_logging(
    level: Union[int, str, LogLevel] = LogLevel.INFO,
    log_file: Optional[Union[str, Path]] = None,
    colored: bool = True,
    include_extra: bool = True,
) -> None:
    """Configure library logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file to write logs to.
        colored: Use colored output for console.
        include_extra: Include extra fields in output.
    """
    # Convert level
    if isinstance(level, LogLevel):
        level = level.value
    elif isinstance(level, str):
        level = getattr(logging, level.upper())

    # Configure root logger
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    if colored and sys.stderr.isatty():
        console_handler.setFormatter(ColoredFormatter(include_extra=include_extra))
    else:
        console_handler.setFormatter(StructuredFormatter(include_extra=include_extra))
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(StructuredFormatter(include_extra=include_extra))
        logger.addHandler(file_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (will be prefixed with 'insideLLMs.').

    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"insideLLMs.{name}")
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that supports extra context."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process the log message."""
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        extra["extra_data"] = extra.copy()
        kwargs["extra"] = extra
        return msg, kwargs


def log_with_context(
    logger_instance: logging.Logger,
    **context: Any,
) -> LoggerAdapter:
    """Create a logger adapter with additional context.

    Args:
        logger_instance: Base logger.
        **context: Context to include in all log messages.

    Returns:
        Logger adapter with context.
    """
    return LoggerAdapter(logger_instance, context)


# Timing and performance logging

@contextmanager
def log_timing(
    name: str,
    logger_instance: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
):
    """Context manager to log execution time.

    Args:
        name: Name of the operation being timed.
        logger_instance: Logger to use (default: library logger).
        level: Log level for timing message.

    Yields:
        None
    """
    log = logger_instance or logger
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.log(level, f"{name} completed in {elapsed:.3f}s")


T = TypeVar("T")


def log_call(
    level: int = logging.DEBUG,
    log_args: bool = True,
    log_result: bool = False,
    log_timing_flag: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to log function calls.

    Args:
        level: Log level.
        log_args: Whether to log arguments.
        log_result: Whether to log return value.
        log_timing_flag: Whether to log execution time.

    Returns:
        Decorator function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            func_logger = get_logger(func.__module__)

            # Log call
            if log_args:
                args_str = ", ".join(
                    [repr(a)[:50] for a in args] +
                    [f"{k}={repr(v)[:50]}" for k, v in kwargs.items()]
                )
                func_logger.log(level, f"Calling {func.__name__}({args_str})")
            else:
                func_logger.log(level, f"Calling {func.__name__}")

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)

                elapsed = time.perf_counter() - start
                if log_result:
                    func_logger.log(
                        level,
                        f"{func.__name__} returned {repr(result)[:100]} in {elapsed:.3f}s"
                    )
                elif log_timing_flag:
                    func_logger.log(level, f"{func.__name__} completed in {elapsed:.3f}s")

                return result

            except Exception as e:
                elapsed = time.perf_counter() - start
                func_logger.error(
                    f"{func.__name__} failed after {elapsed:.3f}s: {type(e).__name__}: {e}"
                )
                raise

        return wrapper
    return decorator


# Error tracking

@dataclass
class ErrorRecord:
    """Record of an error occurrence."""

    timestamp: datetime
    error_type: str
    message: str
    traceback: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "message": self.message,
            "traceback": self.traceback,
            "context": self.context,
        }


class ErrorTracker:
    """Tracks errors for analysis and debugging."""

    def __init__(self, max_errors: int = 1000):
        self._errors: List[ErrorRecord] = []
        self._max_errors = max_errors
        self._counts: Dict[str, int] = {}

    def record(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorRecord:
        """Record an error.

        Args:
            error: The exception that occurred.
            context: Additional context information.

        Returns:
            The error record.
        """
        record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            context=context or {},
        )

        self._errors.append(record)
        self._counts[record.error_type] = self._counts.get(record.error_type, 0) + 1

        # Trim if needed
        if len(self._errors) > self._max_errors:
            self._errors = self._errors[-self._max_errors:]

        return record

    @property
    def errors(self) -> List[ErrorRecord]:
        """Get all recorded errors."""
        return self._errors.copy()

    @property
    def error_counts(self) -> Dict[str, int]:
        """Get error counts by type."""
        return self._counts.copy()

    def get_recent(self, n: int = 10) -> List[ErrorRecord]:
        """Get the most recent errors.

        Args:
            n: Number of errors to return.

        Returns:
            List of recent error records.
        """
        return self._errors[-n:]

    def get_by_type(self, error_type: str) -> List[ErrorRecord]:
        """Get errors of a specific type.

        Args:
            error_type: Error type name.

        Returns:
            List of matching error records.
        """
        return [e for e in self._errors if e.error_type == error_type]

    def clear(self) -> None:
        """Clear all recorded errors."""
        self._errors.clear()
        self._counts.clear()

    def summary(self) -> Dict[str, Any]:
        """Get a summary of recorded errors.

        Returns:
            Summary dictionary.
        """
        return {
            "total_errors": len(self._errors),
            "unique_types": len(self._counts),
            "counts_by_type": self._counts.copy(),
            "most_recent": (
                self._errors[-1].to_dict() if self._errors else None
            ),
        }


# Global error tracker
_error_tracker: Optional[ErrorTracker] = None


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker."""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker()
    return _error_tracker


def track_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
) -> ErrorRecord:
    """Track an error using the global tracker.

    Args:
        error: The exception.
        context: Additional context.

    Returns:
        The error record.
    """
    return get_error_tracker().record(error, context)


# Progress logging

class ProgressLogger:
    """Logger for tracking progress of long-running operations."""

    def __init__(
        self,
        total: int,
        description: str = "Processing",
        logger_instance: Optional[logging.Logger] = None,
        log_interval: int = 10,
    ):
        self.total = total
        self.description = description
        self.logger = logger_instance or logger
        self.log_interval = log_interval

        self.current = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def update(self, n: int = 1) -> None:
        """Update progress.

        Args:
            n: Number of items completed.
        """
        self.current += n
        current_time = time.time()

        # Log at intervals
        if current_time - self.last_log_time >= self.log_interval:
            self._log_progress()
            self.last_log_time = current_time

    def _log_progress(self) -> None:
        """Log current progress."""
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0

        pct = 100 * self.current / self.total if self.total > 0 else 0

        self.logger.info(
            f"{self.description}: {self.current}/{self.total} ({pct:.1f}%) "
            f"| Rate: {rate:.1f}/s | ETA: {remaining:.0f}s"
        )

    def finish(self) -> None:
        """Mark progress as complete."""
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0

        self.logger.info(
            f"{self.description}: Completed {self.current} items "
            f"in {elapsed:.1f}s ({rate:.1f}/s)"
        )


@contextmanager
def progress_context(
    total: int,
    description: str = "Processing",
    logger_instance: Optional[logging.Logger] = None,
):
    """Context manager for progress tracking.

    Args:
        total: Total number of items.
        description: Description of the operation.
        logger_instance: Logger to use.

    Yields:
        ProgressLogger instance.
    """
    progress = ProgressLogger(total, description, logger_instance)
    try:
        yield progress
    finally:
        progress.finish()


# Experiment logging

@dataclass
class ExperimentLog:
    """Log entry for an experiment."""

    experiment_id: str
    model_id: str
    probe_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    n_samples: int = 0
    n_completed: int = 0
    n_failed: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "probe_type": self.probe_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "n_samples": self.n_samples,
            "n_completed": self.n_completed,
            "n_failed": self.n_failed,
            "metrics": self.metrics,
            "errors": self.errors,
        }


class ExperimentLogger:
    """Logger specifically for tracking experiments."""

    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        self.logger = logger_instance or get_logger("experiment")
        self._experiments: Dict[str, ExperimentLog] = {}

    def start_experiment(
        self,
        experiment_id: str,
        model_id: str,
        probe_type: str,
        n_samples: int,
    ) -> ExperimentLog:
        """Start logging an experiment.

        Args:
            experiment_id: Unique experiment identifier.
            model_id: Model being evaluated.
            probe_type: Type of probe.
            n_samples: Number of samples.

        Returns:
            Experiment log entry.
        """
        log = ExperimentLog(
            experiment_id=experiment_id,
            model_id=model_id,
            probe_type=probe_type,
            start_time=datetime.now(),
            n_samples=n_samples,
        )
        self._experiments[experiment_id] = log

        self.logger.info(
            f"Starting experiment {experiment_id}: "
            f"model={model_id}, probe={probe_type}, samples={n_samples}"
        )

        return log

    def log_sample_result(
        self,
        experiment_id: str,
        success: bool,
        error: Optional[Exception] = None,
    ) -> None:
        """Log a sample result.

        Args:
            experiment_id: Experiment identifier.
            success: Whether the sample succeeded.
            error: Error if failed.
        """
        if experiment_id not in self._experiments:
            return

        log = self._experiments[experiment_id]
        if success:
            log.n_completed += 1
        else:
            log.n_failed += 1
            if error:
                log.errors.append({
                    "type": type(error).__name__,
                    "message": str(error),
                })

    def finish_experiment(
        self,
        experiment_id: str,
        status: str = "completed",
        metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[ExperimentLog]:
        """Finish logging an experiment.

        Args:
            experiment_id: Experiment identifier.
            status: Final status.
            metrics: Final metrics.

        Returns:
            Final experiment log.
        """
        if experiment_id not in self._experiments:
            return None

        log = self._experiments[experiment_id]
        log.end_time = datetime.now()
        log.status = status
        if metrics:
            log.metrics.update(metrics)

        duration = (log.end_time - log.start_time).total_seconds()
        self.logger.info(
            f"Experiment {experiment_id} {status}: "
            f"completed={log.n_completed}, failed={log.n_failed}, "
            f"duration={duration:.1f}s"
        )

        if metrics:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            self.logger.info(f"Metrics: {metrics_str}")

        return log

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentLog]:
        """Get an experiment log."""
        return self._experiments.get(experiment_id)

    def get_all_experiments(self) -> List[ExperimentLog]:
        """Get all experiment logs."""
        return list(self._experiments.values())


# Global experiment logger
_experiment_logger: Optional[ExperimentLogger] = None


def get_experiment_logger() -> ExperimentLogger:
    """Get the global experiment logger."""
    global _experiment_logger
    if _experiment_logger is None:
        _experiment_logger = ExperimentLogger()
    return _experiment_logger
