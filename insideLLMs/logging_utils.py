"""Logging utilities for insideLLMs.

This module provides a comprehensive structured logging framework designed
specifically for LLM experimentation and evaluation workflows. It extends
Python's standard logging capabilities with features tailored for:

- **Experiment tracking**: Log experiment lifecycle, sample results, and metrics
- **Model API call logging**: Track API calls with timing, arguments, and results
- **Error tracking and diagnostics**: Record, categorize, and analyze errors
- **Performance monitoring**: Time operations and track progress of long-running tasks

The module follows a layered architecture:
    1. Core logging configuration (configure_logging, get_logger)
    2. Structured formatters (StructuredFormatter, ColoredFormatter)
    3. Timing utilities (log_timing, log_call decorator)
    4. Error tracking (ErrorTracker, track_error)
    5. Progress logging (ProgressLogger, progress_context)
    6. Experiment logging (ExperimentLogger, ExperimentLog)

Examples
--------
Basic logging configuration:

>>> from insideLLMs.logging_utils import configure_logging, get_logger, LogLevel
>>> configure_logging(level=LogLevel.DEBUG, colored=True)
>>> logger = get_logger("my_module")
>>> logger.info("Starting experiment")

Timing a code block:

>>> from insideLLMs.logging_utils import log_timing
>>> with log_timing("data_preprocessing"):
...     # expensive operation
...     data = load_and_preprocess_data()

Tracking errors during batch processing:

>>> from insideLLMs.logging_utils import get_error_tracker, track_error
>>> tracker = get_error_tracker()
>>> try:
...     result = model.generate(prompt)
... except Exception as e:
...     track_error(e, context={"prompt": prompt, "model": "gpt-4"})

Logging experiment progress:

>>> from insideLLMs.logging_utils import get_experiment_logger
>>> exp_logger = get_experiment_logger()
>>> log = exp_logger.start_experiment(
...     experiment_id="exp_001",
...     model_id="gpt-4",
...     probe_type="consistency",
...     n_samples=1000
... )
>>> for sample in samples:
...     try:
...         result = process(sample)
...         exp_logger.log_sample_result("exp_001", success=True)
...     except Exception as e:
...         exp_logger.log_sample_result("exp_001", success=False, error=e)
>>> exp_logger.finish_experiment("exp_001", metrics={"accuracy": 0.95})

Using the progress context manager:

>>> from insideLLMs.logging_utils import progress_context
>>> with progress_context(total=1000, description="Processing prompts") as progress:
...     for item in items:
...         process(item)
...         progress.update()

Notes
-----
- All loggers in this module are children of the "insideLLMs" root logger
- The module provides thread-safe global singletons for ErrorTracker and ExperimentLogger
- Log formatting includes timestamps, log levels, module/function locations, and optional extra fields
- Colored output is automatically disabled when stderr is not a TTY

See Also
--------
logging : Python's standard logging module
dataclasses : Used for structured log entries

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
from typing import Any, Callable, Optional, TypeVar, Union

# Create library logger
logger = logging.getLogger("insideLLMs")


class LogLevel(Enum):
    """Enumeration of log levels for the insideLLMs library.

    This enum wraps the standard Python logging levels, providing a type-safe
    way to specify log levels in configuration functions.

    Parameters
    ----------
    value : int
        The numeric logging level value from the logging module.

    Attributes
    ----------
    DEBUG : int
        Detailed information, typically of interest only when diagnosing
        problems. Value: 10.
    INFO : int
        Confirmation that things are working as expected. Value: 20.
    WARNING : int
        An indication that something unexpected happened, or indicative
        of some problem in the near future. Value: 30.
    ERROR : int
        Due to a more serious problem, the software has not been able to
        perform some function. Value: 40.
    CRITICAL : int
        A serious error, indicating that the program itself may be unable
        to continue running. Value: 50.

    Examples
    --------
    Using LogLevel with configure_logging:

    >>> from insideLLMs.logging_utils import configure_logging, LogLevel
    >>> configure_logging(level=LogLevel.DEBUG)
    >>> configure_logging(level=LogLevel.WARNING)

    Comparing LogLevel values:

    >>> LogLevel.DEBUG.value < LogLevel.INFO.value
    True
    >>> LogLevel.ERROR.value
    40

    Converting from string representation:

    >>> level = LogLevel["INFO"]
    >>> level.value
    20

    See Also
    --------
    configure_logging : Function that accepts LogLevel for configuration
    logging : Python's standard logging module levels

    """

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogEntry:
    """A structured log entry containing metadata and context.

    LogEntry provides a dataclass-based representation of log records that
    can be easily serialized to JSON or other formats. This is useful for
    structured logging pipelines, log aggregation systems, and post-hoc
    analysis of experiment logs.

    Parameters
    ----------
    timestamp : datetime
        When the log entry was created.
    level : str
        The log level as a string (e.g., "INFO", "ERROR").
    message : str
        The log message content.
    module : str, optional
        The module where the log was generated. Default is "".
    function : str, optional
        The function where the log was generated. Default is "".
    extra : dict[str, Any], optional
        Additional context fields. Default is an empty dict.

    Attributes
    ----------
    timestamp : datetime
        The timestamp when this entry was created.
    level : str
        The severity level of this log entry.
    message : str
        The main log message.
    module : str
        Source module name.
    function : str
        Source function name.
    extra : dict[str, Any]
        Additional key-value pairs providing context.

    Examples
    --------
    Creating a basic log entry:

    >>> from datetime import datetime
    >>> from insideLLMs.logging_utils import LogEntry
    >>> entry = LogEntry(
    ...     timestamp=datetime.now(),
    ...     level="INFO",
    ...     message="Model evaluation started"
    ... )
    >>> print(entry.level)
    INFO

    Creating an entry with full context:

    >>> entry = LogEntry(
    ...     timestamp=datetime.now(),
    ...     level="ERROR",
    ...     message="API rate limit exceeded",
    ...     module="model_client",
    ...     function="generate",
    ...     extra={"model_id": "gpt-4", "retry_count": 3}
    ... )
    >>> entry.extra["model_id"]
    'gpt-4'

    Converting to dictionary for JSON serialization:

    >>> entry = LogEntry(
    ...     timestamp=datetime(2024, 1, 15, 10, 30, 0),
    ...     level="DEBUG",
    ...     message="Cache hit",
    ...     extra={"cache_key": "prompt_12345"}
    ... )
    >>> d = entry.to_dict()
    >>> d["timestamp"]
    '2024-01-15T10:30:00'
    >>> d["cache_key"]
    'prompt_12345'

    See Also
    --------
    ErrorRecord : Similar dataclass for error-specific records
    ExperimentLog : Dataclass for experiment-level logging

    """

    timestamp: datetime
    level: str
    message: str
    module: str = ""
    function: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the log entry to a dictionary.

        Creates a flat dictionary representation suitable for JSON
        serialization or insertion into databases. The timestamp is
        converted to ISO 8601 format, and extra fields are merged
        into the top level of the dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: timestamp, level, message, module,
            function, and all keys from extra merged at the top level.

        Examples
        --------
        Basic conversion:

        >>> from datetime import datetime
        >>> from insideLLMs.logging_utils import LogEntry
        >>> entry = LogEntry(
        ...     timestamp=datetime(2024, 1, 15, 10, 30, 0),
        ...     level="INFO",
        ...     message="Test message"
        ... )
        >>> d = entry.to_dict()
        >>> d["timestamp"]
        '2024-01-15T10:30:00'
        >>> d["level"]
        'INFO'

        With extra fields (note: extra fields are merged to top level):

        >>> entry = LogEntry(
        ...     timestamp=datetime(2024, 1, 15, 10, 30, 0),
        ...     level="DEBUG",
        ...     message="API call",
        ...     extra={"latency_ms": 150, "model": "gpt-4"}
        ... )
        >>> d = entry.to_dict()
        >>> d["latency_ms"]
        150
        >>> d["model"]
        'gpt-4'

        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "module": self.module,
            "function": self.function,
            **self.extra,
        }


class StructuredFormatter(logging.Formatter):
    """A logging formatter that produces structured, parseable output.

    This formatter extends the standard logging.Formatter to produce
    log lines with a consistent structure that includes:
    - ISO-formatted timestamp
    - Padded log level
    - Module, function, and line number location
    - The log message
    - Optional extra fields in key=value format

    The output format is designed to be both human-readable and
    machine-parseable, making it suitable for log aggregation systems.

    Parameters
    ----------
    fmt : str, optional
        Format string (passed to parent but not used directly).
    datefmt : str, optional
        Date format string (passed to parent but not used directly).
    include_extra : bool, optional
        Whether to include extra fields in the output. Default is True.

    Attributes
    ----------
    include_extra : bool
        Flag controlling whether extra fields are appended to log lines.

    Examples
    --------
    Basic usage with a logger:

    >>> import logging
    >>> from insideLLMs.logging_utils import StructuredFormatter
    >>> handler = logging.StreamHandler()
    >>> handler.setFormatter(StructuredFormatter())
    >>> logger = logging.getLogger("test")
    >>> logger.addHandler(handler)
    >>> logger.setLevel(logging.DEBUG)
    >>> logger.info("Processing started")  # doctest: +SKIP
    [2024-01-15 10:30:00] INFO     | test.module:42                           | Processing started

    Without extra fields:

    >>> handler = logging.StreamHandler()
    >>> handler.setFormatter(StructuredFormatter(include_extra=False))

    The output format is:
        [YYYY-MM-DD HH:MM:SS] LEVEL    | module.function:line                     | message | extra=fields

    See Also
    --------
    ColoredFormatter : Subclass that adds ANSI color codes
    configure_logging : Applies formatters to the library logger

    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        include_extra: bool = True,
    ):
        """Initialize the StructuredFormatter.

        Parameters
        ----------
        fmt : str, optional
            Format string passed to the parent Formatter. Not directly
            used by StructuredFormatter but maintained for compatibility.
        datefmt : str, optional
            Date format string passed to the parent Formatter. Not
            directly used as this formatter has its own date formatting.
        include_extra : bool, optional
            If True, extra fields attached to log records will be
            appended to the formatted output. Default is True.

        Examples
        --------
        Create formatter with default settings:

        >>> from insideLLMs.logging_utils import StructuredFormatter
        >>> formatter = StructuredFormatter()
        >>> formatter.include_extra
        True

        Create formatter without extra field output:

        >>> formatter = StructuredFormatter(include_extra=False)
        >>> formatter.include_extra
        False

        """
        super().__init__(fmt, datefmt)
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record into a structured string.

        Produces a log line with the following structure:
        ``[timestamp] LEVEL    | module.function:line | message | extra_fields``

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format. This is provided by the logging
            framework when a log message is emitted.

        Returns
        -------
        str
            The formatted log line as a string.

        Examples
        --------
        The formatter is typically used via a logger, not called directly:

        >>> import logging
        >>> from insideLLMs.logging_utils import StructuredFormatter
        >>> # Create a record manually for demonstration
        >>> record = logging.LogRecord(
        ...     name="test",
        ...     level=logging.INFO,
        ...     pathname="test.py",
        ...     lineno=42,
        ...     msg="Test message",
        ...     args=(),
        ...     exc_info=None
        ... )
        >>> formatter = StructuredFormatter()
        >>> formatted = formatter.format(record)
        >>> "INFO" in formatted
        True
        >>> "Test message" in formatted
        True

        With extra data attached to the record:

        >>> record.extra_data = {"user_id": "123", "action": "login"}
        >>> formatted = formatter.format(record)
        >>> "user_id=123" in formatted
        True

        """
        # Get basic info
        timestamp = datetime.fromtimestamp(record.created)
        level = record.levelname

        # Build message
        message = record.getMessage()

        # Add location info
        location = f"{record.module}.{record.funcName}:{record.lineno}"

        # Format base message
        base = (
            f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {level:8s} | {location:40s} | {message}"
        )

        # Add extra fields if present
        if self.include_extra and hasattr(record, "extra_data"):
            extra = record.extra_data
            if extra:
                extra_str = " | ".join(f"{k}={v}" for k, v in extra.items())
                base = f"{base} | {extra_str}"

        return base


class ColoredFormatter(StructuredFormatter):
    """A structured formatter with ANSI color-coded log levels.

    This formatter extends StructuredFormatter to add color coding to
    log output when displayed in terminals that support ANSI escape
    codes. Each log level is assigned a distinct color for improved
    visual scanning of log output.

    Color assignments:
        - DEBUG: Cyan
        - INFO: Green
        - WARNING: Yellow
        - ERROR: Red
        - CRITICAL: Magenta

    Parameters
    ----------
    fmt : str, optional
        Format string (passed to parent).
    datefmt : str, optional
        Date format string (passed to parent).
    include_extra : bool, optional
        Whether to include extra fields in output. Default is True.

    Attributes
    ----------
    COLORS : dict[str, str]
        Mapping of log level names to ANSI color escape codes.
    RESET : str
        ANSI escape code to reset terminal colors.

    Examples
    --------
    Using ColoredFormatter with a console handler:

    >>> import logging
    >>> import sys
    >>> from insideLLMs.logging_utils import ColoredFormatter
    >>> handler = logging.StreamHandler(sys.stderr)
    >>> handler.setFormatter(ColoredFormatter())
    >>> logger = logging.getLogger("colored_test")
    >>> logger.addHandler(handler)
    >>> logger.setLevel(logging.DEBUG)
    >>> logger.error("This will appear in red")  # doctest: +SKIP

    The configure_logging function automatically uses ColoredFormatter
    when outputting to a TTY:

    >>> from insideLLMs.logging_utils import configure_logging
    >>> configure_logging(colored=True)  # Uses ColoredFormatter if TTY

    Notes
    -----
    Colors are only visible in terminals that support ANSI escape codes.
    The configure_logging function automatically checks sys.stderr.isatty()
    and falls back to StructuredFormatter if not a TTY.

    See Also
    --------
    StructuredFormatter : Parent class without color support
    configure_logging : Applies colored formatting when appropriate

    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with ANSI color codes.

        Wraps the formatted log line with the appropriate color
        escape code based on the log level, then resets the color
        at the end of the line.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            The formatted log line wrapped in ANSI color codes.

        Examples
        --------
        >>> import logging
        >>> from insideLLMs.logging_utils import ColoredFormatter
        >>> formatter = ColoredFormatter()
        >>> record = logging.LogRecord(
        ...     name="test", level=logging.ERROR,
        ...     pathname="test.py", lineno=1,
        ...     msg="Error occurred", args=(), exc_info=None
        ... )
        >>> formatted = formatter.format(record)
        >>> formatted.startswith('\\033[31m')  # Red color code
        True
        >>> formatted.endswith('\\033[0m')  # Reset code
        True

        """
        base = super().format(record)
        color = self.COLORS.get(record.levelname, "")
        return f"{color}{base}{self.RESET}"


def configure_logging(
    level: Union[int, str, LogLevel] = LogLevel.INFO,
    log_file: Optional[Union[str, Path]] = None,
    colored: bool = True,
    include_extra: bool = True,
) -> None:
    """Configure the insideLLMs library logging system.

    This is the primary function for setting up logging in the insideLLMs
    library. It configures the root "insideLLMs" logger with appropriate
    handlers, formatters, and log levels.

    Parameters
    ----------
    level : int or str or LogLevel, optional
        The minimum log level to capture. Can be specified as:
        - A LogLevel enum value (e.g., LogLevel.DEBUG)
        - An integer (e.g., logging.DEBUG, 10)
        - A string (e.g., "DEBUG", "info")
        Default is LogLevel.INFO.
    log_file : str or Path, optional
        Path to a file for writing logs. If provided, logs are written
        to both the console and this file. The file handler always uses
        StructuredFormatter (no colors). Default is None (console only).
    colored : bool, optional
        Whether to use colored output for console logging. Colors are
        only applied if stderr is a TTY. Default is True.
    include_extra : bool, optional
        Whether to include extra context fields in formatted output.
        Default is True.

    Returns
    -------
    None

    Examples
    --------
    Basic configuration with default INFO level:

    >>> from insideLLMs.logging_utils import configure_logging
    >>> configure_logging()

    Debug level with colored output:

    >>> from insideLLMs.logging_utils import configure_logging, LogLevel
    >>> configure_logging(level=LogLevel.DEBUG, colored=True)

    Using a string level (case-insensitive):

    >>> configure_logging(level="warning")

    Writing logs to a file:

    >>> configure_logging(
    ...     level=LogLevel.INFO,
    ...     log_file="/tmp/experiment.log"
    ... )

    Full configuration for production:

    >>> from pathlib import Path
    >>> configure_logging(
    ...     level=LogLevel.WARNING,
    ...     log_file=Path("/var/log/insidellms/app.log"),
    ...     colored=False,  # No colors in production
    ...     include_extra=True
    ... )

    Configuring for CI/CD environments (no TTY):

    >>> configure_logging(
    ...     level="DEBUG",
    ...     colored=False,  # CI runners often don't support ANSI
    ...     include_extra=True
    ... )

    Notes
    -----
    - Calling this function clears any existing handlers on the
      insideLLMs logger before adding new ones.
    - The file handler always uses StructuredFormatter regardless
      of the `colored` parameter.
    - Color detection for TTY is automatic when `colored=True`.

    See Also
    --------
    get_logger : Get a child logger after configuration
    LogLevel : Enum of available log levels
    StructuredFormatter : The formatter used for file output
    ColoredFormatter : The formatter used for colored console output

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
    """Get a logger instance within the insideLLMs namespace.

    Returns a logger that is a child of the main "insideLLMs" logger.
    This ensures that all loggers in the library share the same
    configuration (handlers, formatters, levels) set by configure_logging.

    Parameters
    ----------
    name : str, optional
        The name to append to "insideLLMs." for the logger. If None,
        returns the root "insideLLMs" logger. Default is None.

    Returns
    -------
    logging.Logger
        A logger instance. If name is provided, returns a logger named
        "insideLLMs.{name}". Otherwise, returns the root "insideLLMs" logger.

    Examples
    --------
    Get the root library logger:

    >>> from insideLLMs.logging_utils import get_logger
    >>> logger = get_logger()
    >>> logger.name
    'insideLLMs'

    Get a module-specific logger:

    >>> logger = get_logger("model_client")
    >>> logger.name
    'insideLLMs.model_client'
    >>> logger.info("Client initialized")  # doctest: +SKIP

    Get a logger for a specific component:

    >>> logger = get_logger("probes.consistency")
    >>> logger.name
    'insideLLMs.probes.consistency'

    Typical usage in a module:

    >>> # At the top of your module:
    >>> from insideLLMs.logging_utils import get_logger
    >>> logger = get_logger(__name__)  # Uses module's __name__
    >>> # Then use throughout the module:
    >>> logger.debug("Detailed information")  # doctest: +SKIP
    >>> logger.info("General information")  # doctest: +SKIP
    >>> logger.warning("Warning message")  # doctest: +SKIP
    >>> logger.error("Error occurred")  # doctest: +SKIP

    See Also
    --------
    configure_logging : Configure the logging system before using loggers
    log_with_context : Create a logger adapter with persistent context

    """
    if name:
        return logging.getLogger(f"insideLLMs.{name}")
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """A logger adapter that adds persistent context to log messages.

    This adapter wraps a logger and automatically includes additional
    context fields in every log message. The context is stored in the
    adapter and merged into the 'extra' dict of each log call.

    This is particularly useful for adding consistent metadata across
    multiple log calls, such as request IDs, user IDs, or experiment
    identifiers.

    Parameters
    ----------
    logger : logging.Logger
        The underlying logger to wrap.
    extra : dict[str, Any]
        Context to include in all log messages from this adapter.

    Attributes
    ----------
    logger : logging.Logger
        The wrapped logger instance.
    extra : dict[str, Any]
        The persistent context dictionary.

    Examples
    --------
    Creating an adapter with request context:

    >>> import logging
    >>> from insideLLMs.logging_utils import LoggerAdapter, get_logger
    >>> base_logger = get_logger("api")
    >>> adapter = LoggerAdapter(base_logger, {"request_id": "req-12345"})
    >>> adapter.info("Processing request")  # doctest: +SKIP
    # Output includes: request_id=req-12345

    Using log_with_context helper (preferred method):

    >>> from insideLLMs.logging_utils import log_with_context, get_logger
    >>> logger = get_logger("experiment")
    >>> ctx_logger = log_with_context(logger, experiment_id="exp-001", model="gpt-4")
    >>> ctx_logger.info("Starting evaluation")  # doctest: +SKIP
    # Output includes: experiment_id=exp-001 | model=gpt-4

    Combining adapter context with per-call extra:

    >>> adapter = LoggerAdapter(get_logger(), {"session": "abc"})
    >>> adapter.info("Event", extra={"event_type": "click"})  # doctest: +SKIP
    # Output includes both: session=abc | event_type=click

    See Also
    --------
    log_with_context : Convenience function to create LoggerAdapter instances
    StructuredFormatter : Formatter that displays extra_data fields

    """

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple:
        """Process the log message and kwargs before logging.

        Merges the adapter's persistent context with any extra fields
        provided in the log call, and sets up the extra_data attribute
        that StructuredFormatter uses for output.

        Parameters
        ----------
        msg : str
            The log message.
        kwargs : dict[str, Any]
            Keyword arguments passed to the log call, which may include
            an 'extra' dict.

        Returns
        -------
        tuple
            A tuple of (msg, kwargs) where kwargs['extra'] has been
            updated with the merged context.

        Examples
        --------
        This method is called internally by the logging framework:

        >>> from insideLLMs.logging_utils import LoggerAdapter, get_logger
        >>> adapter = LoggerAdapter(get_logger(), {"user_id": "u123"})
        >>> # When you call adapter.info("message", extra={"action": "login"})
        >>> # process() merges: {"user_id": "u123", "action": "login"}

        """
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        extra["extra_data"] = extra.copy()
        kwargs["extra"] = extra
        return msg, kwargs


def log_with_context(
    logger_instance: logging.Logger,
    **context: Any,
) -> LoggerAdapter:
    """Create a logger adapter with additional context fields.

    This is a convenience function for creating LoggerAdapter instances
    with persistent context. The returned adapter will include all
    specified context fields in every log message.

    Parameters
    ----------
    logger_instance : logging.Logger
        The base logger to wrap.
    **context : Any
        Keyword arguments that become persistent context fields.
        These will be included in every log message from the adapter.

    Returns
    -------
    LoggerAdapter
        A logger adapter that includes the specified context in all logs.

    Examples
    --------
    Adding experiment context to all log messages:

    >>> from insideLLMs.logging_utils import log_with_context, get_logger
    >>> logger = get_logger("evaluation")
    >>> ctx_logger = log_with_context(
    ...     logger,
    ...     experiment_id="exp-2024-001",
    ...     model="gpt-4",
    ...     probe_type="consistency"
    ... )
    >>> ctx_logger.info("Evaluation started")  # doctest: +SKIP
    >>> ctx_logger.info("Sample 1 complete")  # doctest: +SKIP
    >>> ctx_logger.info("Evaluation finished")  # doctest: +SKIP
    # All messages include experiment_id, model, and probe_type

    Creating request-scoped logging:

    >>> def handle_request(request_id: str, user_id: str):
    ...     req_logger = log_with_context(
    ...         get_logger("api"),
    ...         request_id=request_id,
    ...         user_id=user_id
    ...     )
    ...     req_logger.info("Request received")  # doctest: +SKIP
    ...     # ... process request ...
    ...     req_logger.info("Request completed")  # doctest: +SKIP

    Using with batch processing:

    >>> def process_batch(batch_id: str, items: list):
    ...     batch_logger = log_with_context(
    ...         get_logger("batch"),
    ...         batch_id=batch_id,
    ...         batch_size=len(items)
    ...     )
    ...     batch_logger.info("Starting batch")  # doctest: +SKIP
    ...     for i, item in enumerate(items):
    ...         batch_logger.debug(f"Processing item {i}")  # doctest: +SKIP

    See Also
    --------
    LoggerAdapter : The class returned by this function
    get_logger : Get a base logger to wrap

    """
    return LoggerAdapter(logger_instance, context)


# Timing and performance logging


@contextmanager
def log_timing(
    name: str,
    logger_instance: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
):
    """Context manager to log execution time of a code block.

    Measures the wall-clock time taken to execute the code within the
    context and logs it upon completion. Uses time.perf_counter() for
    high-precision timing.

    Parameters
    ----------
    name : str
        A descriptive name for the operation being timed. This appears
        in the log message.
    logger_instance : logging.Logger, optional
        The logger to use for output. If None, uses the root insideLLMs
        logger. Default is None.
    level : int, optional
        The log level for the timing message. Default is logging.DEBUG.

    Yields
    ------
    None
        This context manager does not yield a value.

    Examples
    --------
    Basic timing of an operation:

    >>> from insideLLMs.logging_utils import log_timing, configure_logging, LogLevel
    >>> configure_logging(level=LogLevel.DEBUG)
    >>> with log_timing("data_loading"):
    ...     data = [i**2 for i in range(10000)]  # doctest: +SKIP
    # Logs: "data_loading completed in 0.005s"

    Timing with a specific logger:

    >>> from insideLLMs.logging_utils import log_timing, get_logger
    >>> logger = get_logger("preprocessing")
    >>> with log_timing("tokenization", logger_instance=logger):
    ...     tokens = text.split()  # doctest: +SKIP

    Using INFO level for important timing:

    >>> import logging
    >>> with log_timing("model_inference", level=logging.INFO):
    ...     result = model.generate(prompt)  # doctest: +SKIP

    Nested timing blocks:

    >>> with log_timing("full_pipeline"):
    ...     with log_timing("step_1_load"):
    ...         data = load_data()  # doctest: +SKIP
    ...     with log_timing("step_2_process"):
    ...         processed = process(data)  # doctest: +SKIP
    ...     with log_timing("step_3_save"):
    ...         save(processed)  # doctest: +SKIP

    Timing is logged even if an exception occurs:

    >>> try:
    ...     with log_timing("risky_operation"):
    ...         raise ValueError("Something went wrong")
    ... except ValueError:
    ...     pass  # Timing is still logged before exception propagates
    # doctest: +SKIP

    See Also
    --------
    log_call : Decorator for timing function calls
    ProgressLogger : For tracking progress of iterative operations

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
    """Decorator to log function calls with optional arguments, results, and timing.

    This decorator wraps a function to automatically log:
    - When the function is called (optionally with arguments)
    - Execution time (optional)
    - Return value (optional)
    - Any exceptions that occur

    Parameters
    ----------
    level : int, optional
        The log level for normal messages. Errors are always logged at
        ERROR level. Default is logging.DEBUG.
    log_args : bool, optional
        Whether to include function arguments in the log message.
        Arguments are truncated to 50 characters each. Default is True.
    log_result : bool, optional
        Whether to log the return value. Results are truncated to 100
        characters. Default is False.
    log_timing_flag : bool, optional
        Whether to log execution time. If log_result is True, timing
        is included in the result message regardless of this flag.
        Default is True.

    Returns
    -------
    Callable[[Callable[..., T]], Callable[..., T]]
        A decorator function that wraps the target function.

    Examples
    --------
    Basic function call logging:

    >>> from insideLLMs.logging_utils import log_call, configure_logging, LogLevel
    >>> configure_logging(level=LogLevel.DEBUG)
    >>> @log_call()
    ... def process_data(data: list) -> int:
    ...     return len(data)
    >>> result = process_data([1, 2, 3])  # doctest: +SKIP
    # Logs: "Calling process_data([1, 2, 3])"
    # Logs: "process_data completed in 0.000s"

    Logging with return value:

    >>> @log_call(log_result=True)
    ... def calculate_score(x: float, y: float) -> float:
    ...     return x * y + 10
    >>> score = calculate_score(5.0, 3.0)  # doctest: +SKIP
    # Logs: "Calling calculate_score(5.0, 3.0)"
    # Logs: "calculate_score returned 25.0 in 0.000s"

    Without arguments (for privacy):

    >>> @log_call(log_args=False)
    ... def authenticate(password: str) -> bool:
    ...     return password == "secret"
    >>> authenticate("secret")  # doctest: +SKIP
    # Logs: "Calling authenticate"
    # Logs: "authenticate completed in 0.000s"

    Using INFO level for important operations:

    >>> import logging
    >>> @log_call(level=logging.INFO, log_result=True)
    ... def train_model(epochs: int) -> dict:
    ...     return {"accuracy": 0.95}

    Error logging (always at ERROR level):

    >>> @log_call()
    ... def risky_function():
    ...     raise ValueError("Something failed")
    >>> try:
    ...     risky_function()  # doctest: +SKIP
    ... except ValueError:
    ...     pass
    # Logs at ERROR: "risky_function failed after 0.000s: ValueError: Something failed"

    Class method decoration:

    >>> class ModelClient:
    ...     @log_call(log_args=True, log_timing_flag=True)
    ...     def generate(self, prompt: str, max_tokens: int = 100):
    ...         return "Generated text"

    Notes
    -----
    - Arguments and results are repr'd and truncated for safety
    - The decorator preserves the function's name, docstring, etc.
    - Exceptions are re-raised after logging

    See Also
    --------
    log_timing : Context manager for timing code blocks
    ErrorTracker : For more detailed error tracking

    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            func_logger = get_logger(func.__module__)

            # Log call
            if log_args:
                args_str = ", ".join(
                    [repr(a)[:50] for a in args]
                    + [f"{k}={repr(v)[:50]}" for k, v in kwargs.items()]
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
                        level, f"{func.__name__} returned {repr(result)[:100]} in {elapsed:.3f}s"
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
    """A structured record of an error occurrence.

    ErrorRecord captures detailed information about an exception,
    including the full traceback and optional context. This is useful
    for post-hoc analysis of errors, pattern detection, and debugging.

    Parameters
    ----------
    timestamp : datetime
        When the error occurred.
    error_type : str
        The exception class name (e.g., "ValueError", "APIError").
    message : str
        The exception message (str(exception)).
    traceback : str
        The full formatted traceback.
    context : dict[str, Any], optional
        Additional context about what was happening when the error
        occurred. Default is an empty dict.

    Attributes
    ----------
    timestamp : datetime
        When this error was recorded.
    error_type : str
        The type/class name of the exception.
    message : str
        The error message.
    traceback : str
        Full traceback as a string.
    context : dict[str, Any]
        Additional context information.

    Examples
    --------
    Creating an error record manually:

    >>> from datetime import datetime
    >>> from insideLLMs.logging_utils import ErrorRecord
    >>> record = ErrorRecord(
    ...     timestamp=datetime.now(),
    ...     error_type="ValueError",
    ...     message="Invalid input: expected positive number",
    ...     traceback="Traceback (most recent call last):...",
    ...     context={"input_value": -5, "function": "validate_input"}
    ... )
    >>> record.error_type
    'ValueError'

    Typically created via ErrorTracker.record():

    >>> from insideLLMs.logging_utils import ErrorTracker
    >>> tracker = ErrorTracker()
    >>> try:
    ...     raise ValueError("test error")
    ... except ValueError as e:
    ...     record = tracker.record(e, context={"step": "validation"})
    >>> record.error_type
    'ValueError'
    >>> record.context["step"]
    'validation'

    Converting to dictionary for storage:

    >>> d = record.to_dict()
    >>> d["error_type"]
    'ValueError'
    >>> "timestamp" in d
    True

    See Also
    --------
    ErrorTracker : Class that creates and manages ErrorRecord instances
    track_error : Global function to record errors

    """

    timestamp: datetime
    error_type: str
    message: str
    traceback: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the error record to a dictionary.

        Creates a dictionary representation suitable for JSON
        serialization, database storage, or log aggregation systems.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: timestamp (ISO format), error_type,
            message, traceback, and context.

        Examples
        --------
        Basic conversion:

        >>> from datetime import datetime
        >>> from insideLLMs.logging_utils import ErrorRecord
        >>> record = ErrorRecord(
        ...     timestamp=datetime(2024, 1, 15, 10, 30, 0),
        ...     error_type="APIError",
        ...     message="Rate limit exceeded",
        ...     traceback="Traceback...",
        ...     context={"endpoint": "/v1/chat", "retry_count": 3}
        ... )
        >>> d = record.to_dict()
        >>> d["timestamp"]
        '2024-01-15T10:30:00'
        >>> d["error_type"]
        'APIError'
        >>> d["context"]["retry_count"]
        3

        Serializing to JSON:

        >>> import json
        >>> json_str = json.dumps(record.to_dict())
        >>> "APIError" in json_str
        True

        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "message": self.message,
            "traceback": self.traceback,
            "context": self.context,
        }


class ErrorTracker:
    """Tracks and analyzes errors for debugging and monitoring.

    ErrorTracker maintains a bounded list of error records and provides
    methods for querying, analyzing, and summarizing errors. It's
    designed for use in long-running processes where understanding
    error patterns is important.

    The tracker automatically manages memory by limiting the number
    of stored errors, keeping only the most recent ones.

    Parameters
    ----------
    max_errors : int, optional
        Maximum number of error records to retain. When exceeded,
        oldest records are removed. Default is 1000.

    Attributes
    ----------
    errors : list[ErrorRecord]
        Property returning a copy of all stored error records.
    error_counts : dict[str, int]
        Property returning error counts by type.

    Examples
    --------
    Basic error tracking:

    >>> from insideLLMs.logging_utils import ErrorTracker
    >>> tracker = ErrorTracker(max_errors=100)
    >>> try:
    ...     raise ValueError("invalid value")
    ... except Exception as e:
    ...     record = tracker.record(e)
    >>> tracker.error_counts
    {'ValueError': 1}

    Tracking with context:

    >>> try:
    ...     result = api_call(prompt)  # doctest: +SKIP
    ... except Exception as e:
    ...     tracker.record(e, context={
    ...         "prompt": prompt,
    ...         "model": "gpt-4",
    ...         "retry_attempt": 2
    ...     })  # doctest: +SKIP

    Analyzing errors:

    >>> tracker = ErrorTracker()
    >>> # After many operations...
    >>> summary = tracker.summary()
    >>> print(f"Total errors: {summary['total_errors']}")  # doctest: +SKIP
    >>> print(f"By type: {summary['counts_by_type']}")  # doctest: +SKIP

    Getting recent errors for debugging:

    >>> recent = tracker.get_recent(5)
    >>> for err in recent:
    ...     print(f"{err.error_type}: {err.message}")  # doctest: +SKIP

    Filtering by error type:

    >>> api_errors = tracker.get_by_type("APIError")
    >>> print(f"Found {len(api_errors)} API errors")  # doctest: +SKIP

    Notes
    -----
    - The tracker is not thread-safe. For concurrent access, external
      synchronization is required.
    - Error counts persist even when old records are trimmed.

    See Also
    --------
    ErrorRecord : The record type stored by this tracker
    track_error : Global convenience function
    get_error_tracker : Get the global ErrorTracker instance

    """

    def __init__(self, max_errors: int = 1000):
        """Initialize the ErrorTracker.

        Parameters
        ----------
        max_errors : int, optional
            Maximum number of error records to store. Older records
            are removed when this limit is exceeded. Default is 1000.

        Examples
        --------
        Create a tracker with default capacity:

        >>> from insideLLMs.logging_utils import ErrorTracker
        >>> tracker = ErrorTracker()
        >>> tracker._max_errors
        1000

        Create a tracker with limited capacity (for memory-constrained environments):

        >>> tracker = ErrorTracker(max_errors=50)
        >>> tracker._max_errors
        50

        """
        self._errors: list[ErrorRecord] = []
        self._max_errors = max_errors
        self._counts: dict[str, int] = {}

    def record(
        self,
        error: Exception,
        context: Optional[dict[str, Any]] = None,
    ) -> ErrorRecord:
        """Record an error occurrence.

        Creates an ErrorRecord from the exception, captures the current
        traceback, and stores it. Also increments the count for this
        error type.

        Parameters
        ----------
        error : Exception
            The exception that occurred.
        context : dict[str, Any], optional
            Additional context about what was happening when the error
            occurred. Default is None (empty context).

        Returns
        -------
        ErrorRecord
            The created error record.

        Examples
        --------
        Recording a simple error:

        >>> from insideLLMs.logging_utils import ErrorTracker
        >>> tracker = ErrorTracker()
        >>> try:
        ...     x = 1 / 0
        ... except ZeroDivisionError as e:
        ...     record = tracker.record(e)
        >>> record.error_type
        'ZeroDivisionError'

        Recording with context:

        >>> try:
        ...     response = call_api(prompt="test")  # doctest: +SKIP
        ... except Exception as e:
        ...     record = tracker.record(e, context={
        ...         "prompt": "test",
        ...         "endpoint": "/v1/completions",
        ...         "attempt": 3
        ...     })  # doctest: +SKIP

        The error count is updated:

        >>> tracker = ErrorTracker()
        >>> for i in range(5):
        ...     try:
        ...         raise ValueError(f"error {i}")
        ...     except ValueError as e:
        ...         tracker.record(e)
        >>> tracker.error_counts["ValueError"]
        5

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
            self._errors = self._errors[-self._max_errors :]

        return record

    @property
    def errors(self) -> list[ErrorRecord]:
        """Get a copy of all recorded errors.

        Returns
        -------
        list[ErrorRecord]
            A copy of the list of all error records, in chronological
            order (oldest first).

        Examples
        --------
        >>> from insideLLMs.logging_utils import ErrorTracker
        >>> tracker = ErrorTracker()
        >>> try:
        ...     raise ValueError("test")
        ... except ValueError as e:
        ...     tracker.record(e)
        >>> all_errors = tracker.errors
        >>> len(all_errors)
        1
        >>> all_errors[0].error_type
        'ValueError'

        The returned list is a copy (modifications don't affect tracker):

        >>> errors = tracker.errors
        >>> errors.clear()
        >>> len(tracker.errors)  # Original unchanged
        1

        """
        return self._errors.copy()

    @property
    def error_counts(self) -> dict[str, int]:
        """Get error counts by type.

        Returns
        -------
        dict[str, int]
            Dictionary mapping error type names to occurrence counts.
            This includes all errors ever recorded, not just those
            currently in the buffer.

        Examples
        --------
        >>> from insideLLMs.logging_utils import ErrorTracker
        >>> tracker = ErrorTracker()
        >>> for _ in range(3):
        ...     try:
        ...         raise ValueError("v")
        ...     except ValueError as e:
        ...         tracker.record(e)
        >>> for _ in range(2):
        ...     try:
        ...         raise TypeError("t")
        ...     except TypeError as e:
        ...         tracker.record(e)
        >>> counts = tracker.error_counts
        >>> counts["ValueError"]
        3
        >>> counts["TypeError"]
        2

        """
        return self._counts.copy()

    def get_recent(self, n: int = 10) -> list[ErrorRecord]:
        """Get the most recent error records.

        Parameters
        ----------
        n : int, optional
            Maximum number of recent errors to return. Default is 10.

        Returns
        -------
        list[ErrorRecord]
            The n most recent error records, in chronological order.
            May return fewer than n if fewer errors have been recorded.

        Examples
        --------
        Get the last 5 errors:

        >>> from insideLLMs.logging_utils import ErrorTracker
        >>> tracker = ErrorTracker()
        >>> for i in range(20):
        ...     try:
        ...         raise ValueError(f"error {i}")
        ...     except ValueError as e:
        ...         tracker.record(e)
        >>> recent = tracker.get_recent(5)
        >>> len(recent)
        5
        >>> recent[-1].message
        'error 19'

        When fewer errors exist than requested:

        >>> tracker = ErrorTracker()
        >>> try:
        ...     raise ValueError("only one")
        ... except ValueError as e:
        ...     tracker.record(e)
        >>> len(tracker.get_recent(10))
        1

        """
        return self._errors[-n:]

    def get_by_type(self, error_type: str) -> list[ErrorRecord]:
        """Get all errors of a specific type.

        Parameters
        ----------
        error_type : str
            The error type name to filter by (e.g., "ValueError",
            "APIError").

        Returns
        -------
        list[ErrorRecord]
            All error records matching the specified type, in
            chronological order.

        Examples
        --------
        Filter errors by type:

        >>> from insideLLMs.logging_utils import ErrorTracker
        >>> tracker = ErrorTracker()
        >>> try:
        ...     raise ValueError("v1")
        ... except ValueError as e:
        ...     tracker.record(e)
        >>> try:
        ...     raise TypeError("t1")
        ... except TypeError as e:
        ...     tracker.record(e)
        >>> try:
        ...     raise ValueError("v2")
        ... except ValueError as e:
        ...     tracker.record(e)
        >>> value_errors = tracker.get_by_type("ValueError")
        >>> len(value_errors)
        2
        >>> [e.message for e in value_errors]
        ['v1', 'v2']

        Non-existent type returns empty list:

        >>> tracker.get_by_type("NonExistentError")
        []

        """
        return [e for e in self._errors if e.error_type == error_type]

    def clear(self) -> None:
        """Clear all recorded errors and reset counts.

        Removes all error records and resets all error type counts
        to zero. Use this to start fresh, e.g., between experiments.

        Returns
        -------
        None

        Examples
        --------
        >>> from insideLLMs.logging_utils import ErrorTracker
        >>> tracker = ErrorTracker()
        >>> for _ in range(10):
        ...     try:
        ...         raise ValueError("test")
        ...     except ValueError as e:
        ...         tracker.record(e)
        >>> len(tracker.errors)
        10
        >>> tracker.error_counts["ValueError"]
        10
        >>> tracker.clear()
        >>> len(tracker.errors)
        0
        >>> tracker.error_counts
        {}

        """
        self._errors.clear()
        self._counts.clear()

    def summary(self) -> dict[str, Any]:
        """Get a summary of recorded errors.

        Provides an overview of error statistics useful for monitoring
        dashboards or quick diagnostics.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - total_errors: Total number of errors currently stored
            - unique_types: Number of distinct error types
            - counts_by_type: Dict mapping error type to count
            - most_recent: Dict representation of the most recent
              error, or None if no errors recorded

        Examples
        --------
        >>> from insideLLMs.logging_utils import ErrorTracker
        >>> tracker = ErrorTracker()
        >>> for i in range(5):
        ...     try:
        ...         raise ValueError(f"err {i}")
        ...     except ValueError as e:
        ...         tracker.record(e)
        >>> try:
        ...     raise TypeError("type err")
        ... except TypeError as e:
        ...     tracker.record(e)
        >>> summary = tracker.summary()
        >>> summary["total_errors"]
        6
        >>> summary["unique_types"]
        2
        >>> summary["counts_by_type"]["ValueError"]
        5
        >>> summary["most_recent"]["error_type"]
        'TypeError'

        Empty tracker summary:

        >>> tracker = ErrorTracker()
        >>> summary = tracker.summary()
        >>> summary["total_errors"]
        0
        >>> summary["most_recent"] is None
        True

        """
        return {
            "total_errors": len(self._errors),
            "unique_types": len(self._counts),
            "counts_by_type": self._counts.copy(),
            "most_recent": (self._errors[-1].to_dict() if self._errors else None),
        }


# Global error tracker
_error_tracker: Optional[ErrorTracker] = None


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker instance.

    Returns the singleton ErrorTracker instance, creating it if it
    doesn't exist. This provides a convenient way to track errors
    across the entire application without passing tracker instances.

    Returns
    -------
    ErrorTracker
        The global ErrorTracker singleton.

    Examples
    --------
    Getting the global tracker:

    >>> from insideLLMs.logging_utils import get_error_tracker
    >>> tracker = get_error_tracker()
    >>> isinstance(tracker, ErrorTracker)
    True

    Multiple calls return the same instance:

    >>> tracker1 = get_error_tracker()
    >>> tracker2 = get_error_tracker()
    >>> tracker1 is tracker2
    True

    Using the global tracker:

    >>> tracker = get_error_tracker()
    >>> try:
    ...     raise RuntimeError("test")
    ... except RuntimeError as e:
    ...     tracker.record(e, context={"operation": "test"})
    >>> tracker.error_counts.get("RuntimeError", 0) >= 1
    True

    See Also
    --------
    track_error : Convenience function that uses the global tracker
    ErrorTracker : The class of the returned instance

    """
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker()
    return _error_tracker


def track_error(
    error: Exception,
    context: Optional[dict[str, Any]] = None,
) -> ErrorRecord:
    """Track an error using the global error tracker.

    Convenience function that records an error to the global
    ErrorTracker singleton. Equivalent to calling
    ``get_error_tracker().record(error, context)``.

    Parameters
    ----------
    error : Exception
        The exception to record.
    context : dict[str, Any], optional
        Additional context about the error. Default is None.

    Returns
    -------
    ErrorRecord
        The created error record.

    Examples
    --------
    Basic error tracking:

    >>> from insideLLMs.logging_utils import track_error
    >>> try:
    ...     x = int("not a number")
    ... except ValueError as e:
    ...     record = track_error(e)
    >>> record.error_type
    'ValueError'

    Tracking with context:

    >>> try:
    ...     response = api.call(endpoint="/v1/chat")  # doctest: +SKIP
    ... except Exception as e:
    ...     record = track_error(e, context={
    ...         "endpoint": "/v1/chat",
    ...         "method": "POST",
    ...         "request_id": "req-12345"
    ...     })  # doctest: +SKIP

    Using in a batch processing loop:

    >>> from insideLLMs.logging_utils import track_error, get_error_tracker
    >>> for item in items:  # doctest: +SKIP
    ...     try:
    ...         process(item)
    ...     except Exception as e:
    ...         track_error(e, context={"item_id": item.id})
    >>> # Later, analyze errors:
    >>> tracker = get_error_tracker()  # doctest: +SKIP
    >>> print(tracker.summary())  # doctest: +SKIP

    See Also
    --------
    get_error_tracker : Get the underlying global tracker
    ErrorTracker.record : The method this function wraps

    """
    return get_error_tracker().record(error, context)


# Progress logging


class ProgressLogger:
    """Logger for tracking progress of long-running iterative operations.

    ProgressLogger provides periodic progress updates for batch processing,
    including completion percentage, processing rate, and estimated time
    remaining. It logs at configurable intervals to avoid flooding logs.

    Parameters
    ----------
    total : int
        Total number of items to process.
    description : str, optional
        Description of the operation for log messages. Default is "Processing".
    logger_instance : logging.Logger, optional
        Logger to use for output. If None, uses the root insideLLMs logger.
        Default is None.
    log_interval : int, optional
        Minimum seconds between progress log messages. Default is 10.

    Attributes
    ----------
    total : int
        Total items to process.
    description : str
        Operation description.
    logger : logging.Logger
        The logger instance being used.
    log_interval : int
        Seconds between progress logs.
    current : int
        Number of items completed so far.
    start_time : float
        Unix timestamp when processing started.
    last_log_time : float
        Unix timestamp of the last progress log.

    Examples
    --------
    Basic progress tracking:

    >>> from insideLLMs.logging_utils import ProgressLogger
    >>> progress = ProgressLogger(total=1000, description="Processing samples")
    >>> for item in items:  # doctest: +SKIP
    ...     process(item)
    ...     progress.update()
    >>> progress.finish()  # doctest: +SKIP
    # Logs: "Processing samples: Completed 1000 items in 45.2s (22.1/s)"

    With custom log interval:

    >>> progress = ProgressLogger(
    ...     total=10000,
    ...     description="Generating embeddings",
    ...     log_interval=30  # Log every 30 seconds
    ... )  # doctest: +SKIP

    Updating by batch:

    >>> progress = ProgressLogger(total=1000)
    >>> for batch in batches:  # doctest: +SKIP
    ...     results = process_batch(batch)
    ...     progress.update(n=len(batch))  # Update by batch size
    >>> progress.finish()  # doctest: +SKIP

    Using with a specific logger:

    >>> from insideLLMs.logging_utils import get_logger, ProgressLogger
    >>> custom_logger = get_logger("batch_processor")
    >>> progress = ProgressLogger(
    ...     total=500,
    ...     description="Evaluating prompts",
    ...     logger_instance=custom_logger
    ... )

    Notes
    -----
    - Progress is logged at INFO level
    - The finish() method should always be called to log final statistics
    - Consider using progress_context for automatic finish() handling

    See Also
    --------
    progress_context : Context manager wrapper for ProgressLogger

    """

    def __init__(
        self,
        total: int,
        description: str = "Processing",
        logger_instance: Optional[logging.Logger] = None,
        log_interval: int = 10,
    ):
        """Initialize the ProgressLogger.

        Parameters
        ----------
        total : int
            Total number of items to process.
        description : str, optional
            Description prefix for log messages. Default is "Processing".
        logger_instance : logging.Logger, optional
            Logger for output. Default is the root insideLLMs logger.
        log_interval : int, optional
            Minimum seconds between automatic progress logs. Default is 10.

        Examples
        --------
        >>> from insideLLMs.logging_utils import ProgressLogger
        >>> progress = ProgressLogger(total=100)
        >>> progress.total
        100
        >>> progress.description
        'Processing'

        >>> progress = ProgressLogger(
        ...     total=5000,
        ...     description="Tokenizing documents",
        ...     log_interval=5
        ... )
        >>> progress.log_interval
        5

        """
        self.total = total
        self.description = description
        self.logger = logger_instance or logger
        self.log_interval = log_interval

        self.current = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def update(self, n: int = 1) -> None:
        """Update progress by n items.

        Increments the completion counter and logs progress if enough
        time has passed since the last log (controlled by log_interval).

        Parameters
        ----------
        n : int, optional
            Number of items completed. Default is 1.

        Returns
        -------
        None

        Examples
        --------
        Single item update:

        >>> from insideLLMs.logging_utils import ProgressLogger
        >>> progress = ProgressLogger(total=100)
        >>> progress.update()
        >>> progress.current
        1

        Batch update:

        >>> progress = ProgressLogger(total=1000)
        >>> progress.update(n=50)  # Completed a batch of 50
        >>> progress.current
        50

        In a processing loop:

        >>> progress = ProgressLogger(total=len(items))  # doctest: +SKIP
        >>> for item in items:  # doctest: +SKIP
        ...     result = process(item)
        ...     progress.update()
        ...     # Progress is logged automatically at log_interval

        """
        self.current += n
        current_time = time.time()

        # Log at intervals
        if current_time - self.last_log_time >= self.log_interval:
            self._log_progress()
            self.last_log_time = current_time

    def _log_progress(self) -> None:
        """Log current progress with statistics.

        Internal method that calculates and logs:
        - Current/total items and percentage
        - Processing rate (items/second)
        - Estimated time remaining

        Returns
        -------
        None

        Examples
        --------
        This method is called internally by update() and finish().
        Direct use is generally not needed:

        >>> from insideLLMs.logging_utils import ProgressLogger
        >>> progress = ProgressLogger(total=100, description="Testing")
        >>> progress.current = 50
        >>> progress._log_progress()  # doctest: +SKIP
        # Logs: "Testing: 50/100 (50.0%) | Rate: X.X/s | ETA: Xs"

        """
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0

        pct = 100 * self.current / self.total if self.total > 0 else 0

        self.logger.info(
            f"{self.description}: {self.current}/{self.total} ({pct:.1f}%) "
            f"| Rate: {rate:.1f}/s | ETA: {remaining:.0f}s"
        )

    def finish(self) -> None:
        """Mark progress as complete and log final statistics.

        Should be called when all processing is complete. Logs a
        summary with total items, elapsed time, and average rate.

        Returns
        -------
        None

        Examples
        --------
        >>> from insideLLMs.logging_utils import ProgressLogger
        >>> progress = ProgressLogger(total=100, description="Test run")
        >>> progress.current = 100  # Simulate completion
        >>> progress.finish()  # doctest: +SKIP
        # Logs: "Test run: Completed 100 items in X.Xs (X.X/s)"

        Always call finish() even if processing stops early:

        >>> progress = ProgressLogger(total=1000)
        >>> # Processing stops at 500 due to error
        >>> progress.current = 500
        >>> progress.finish()  # doctest: +SKIP
        # Logs: "Processing: Completed 500 items in X.Xs (X.X/s)"

        """
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0

        self.logger.info(
            f"{self.description}: Completed {self.current} items in {elapsed:.1f}s ({rate:.1f}/s)"
        )


@contextmanager
def progress_context(
    total: int,
    description: str = "Processing",
    logger_instance: Optional[logging.Logger] = None,
):
    """Context manager for progress tracking with automatic finish.

    Creates a ProgressLogger and ensures finish() is called when the
    context exits, even if an exception occurs.

    Parameters
    ----------
    total : int
        Total number of items to process.
    description : str, optional
        Description for log messages. Default is "Processing".
    logger_instance : logging.Logger, optional
        Logger to use. Default is the root insideLLMs logger.

    Yields
    ------
    ProgressLogger
        The progress logger instance for calling update().

    Examples
    --------
    Basic usage:

    >>> from insideLLMs.logging_utils import progress_context
    >>> with progress_context(total=100, description="Processing files") as progress:
    ...     for item in items:  # doctest: +SKIP
    ...         process(item)
    ...         progress.update()
    # Automatically logs completion when context exits

    With error handling (finish still called):

    >>> try:
    ...     with progress_context(total=1000) as progress:
    ...         for i, item in enumerate(items):  # doctest: +SKIP
    ...             if should_stop(item):
    ...                 break
    ...             process(item)
    ...             progress.update()
    ... except Exception as e:
    ...     handle_error(e)  # doctest: +SKIP
    # finish() is called regardless of how context exits

    With custom logger:

    >>> from insideLLMs.logging_utils import get_logger, progress_context
    >>> custom_logger = get_logger("data_pipeline")
    >>> with progress_context(
    ...     total=5000,
    ...     description="Loading records",
    ...     logger_instance=custom_logger
    ... ) as progress:
    ...     for record in records:  # doctest: +SKIP
    ...         load(record)
    ...         progress.update()

    See Also
    --------
    ProgressLogger : The underlying progress tracking class

    """
    progress = ProgressLogger(total, description, logger_instance)
    try:
        yield progress
    finally:
        progress.finish()


# Experiment logging


@dataclass
class ExperimentLog:
    """Log entry containing complete information about an experiment.

    ExperimentLog captures the full lifecycle of an experiment, from
    start to finish, including sample counts, timing, metrics, and
    any errors that occurred. This is the primary data structure for
    experiment tracking in insideLLMs.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for this experiment.
    model_id : str
        Identifier of the model being evaluated.
    probe_type : str
        Type of probe/evaluation being run.
    start_time : datetime
        When the experiment started.
    end_time : datetime, optional
        When the experiment finished. None if still running.
    status : str, optional
        Current status: "running", "completed", "failed", etc.
        Default is "running".
    n_samples : int, optional
        Total number of samples in the experiment. Default is 0.
    n_completed : int, optional
        Number of samples successfully completed. Default is 0.
    n_failed : int, optional
        Number of samples that failed. Default is 0.
    metrics : dict[str, float], optional
        Final metrics from the experiment. Default is empty dict.
    errors : list[dict[str, Any]], optional
        List of error summaries that occurred. Default is empty list.

    Attributes
    ----------
    experiment_id : str
        Unique experiment identifier.
    model_id : str
        Model being evaluated.
    probe_type : str
        Type of evaluation probe.
    start_time : datetime
        Experiment start timestamp.
    end_time : datetime or None
        Experiment end timestamp (None if running).
    status : str
        Current experiment status.
    n_samples : int
        Total sample count.
    n_completed : int
        Successfully completed samples.
    n_failed : int
        Failed samples.
    metrics : dict[str, float]
        Experiment metrics.
    errors : list[dict[str, Any]]
        Error summaries.

    Examples
    --------
    Creating an experiment log:

    >>> from datetime import datetime
    >>> from insideLLMs.logging_utils import ExperimentLog
    >>> log = ExperimentLog(
    ...     experiment_id="exp-2024-001",
    ...     model_id="gpt-4",
    ...     probe_type="consistency",
    ...     start_time=datetime.now(),
    ...     n_samples=1000
    ... )
    >>> log.status
    'running'

    Updating during experiment:

    >>> log.n_completed = 500
    >>> log.n_failed = 5

    Finishing an experiment:

    >>> from datetime import datetime
    >>> log.end_time = datetime.now()
    >>> log.status = "completed"
    >>> log.metrics = {"accuracy": 0.95, "f1_score": 0.93}

    Converting to dictionary:

    >>> d = log.to_dict()
    >>> d["experiment_id"]
    'exp-2024-001'
    >>> d["metrics"]["accuracy"]
    0.95

    See Also
    --------
    ExperimentLogger : Class that creates and manages ExperimentLog instances

    """

    experiment_id: str
    model_id: str
    probe_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    n_samples: int = 0
    n_completed: int = 0
    n_failed: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the experiment log to a dictionary.

        Creates a dictionary representation suitable for JSON
        serialization, database storage, or sending to experiment
        tracking services.

        Returns
        -------
        dict[str, Any]
            Dictionary with all experiment fields. Timestamps are
            converted to ISO 8601 format.

        Examples
        --------
        Basic conversion:

        >>> from datetime import datetime
        >>> from insideLLMs.logging_utils import ExperimentLog
        >>> log = ExperimentLog(
        ...     experiment_id="exp-001",
        ...     model_id="gpt-4",
        ...     probe_type="consistency",
        ...     start_time=datetime(2024, 1, 15, 10, 0, 0),
        ...     end_time=datetime(2024, 1, 15, 11, 30, 0),
        ...     status="completed",
        ...     n_samples=1000,
        ...     n_completed=995,
        ...     n_failed=5,
        ...     metrics={"accuracy": 0.95}
        ... )
        >>> d = log.to_dict()
        >>> d["experiment_id"]
        'exp-001'
        >>> d["start_time"]
        '2024-01-15T10:00:00'
        >>> d["end_time"]
        '2024-01-15T11:30:00'
        >>> d["metrics"]
        {'accuracy': 0.95}

        Serializing to JSON:

        >>> import json
        >>> json_str = json.dumps(log.to_dict())
        >>> "exp-001" in json_str
        True

        """
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
    """Logger for tracking experiment lifecycle and results.

    ExperimentLogger provides a high-level interface for logging
    experiments, including starting experiments, recording sample
    results, and finishing with final metrics. It maintains a
    registry of all experiments and provides query methods.

    Parameters
    ----------
    logger_instance : logging.Logger, optional
        Logger for output messages. If None, creates a logger named
        "insideLLMs.experiment". Default is None.

    Attributes
    ----------
    logger : logging.Logger
        The logger instance used for output.

    Examples
    --------
    Basic experiment logging:

    >>> from insideLLMs.logging_utils import ExperimentLogger
    >>> exp_logger = ExperimentLogger()
    >>> log = exp_logger.start_experiment(
    ...     experiment_id="exp-001",
    ...     model_id="gpt-4",
    ...     probe_type="consistency",
    ...     n_samples=100
    ... )
    >>> # Process samples...
    >>> for sample in samples:  # doctest: +SKIP
    ...     try:
    ...         result = evaluate(sample)
    ...         exp_logger.log_sample_result("exp-001", success=True)
    ...     except Exception as e:
    ...         exp_logger.log_sample_result("exp-001", success=False, error=e)
    >>> # Finish the experiment
    >>> exp_logger.finish_experiment(
    ...     "exp-001",
    ...     status="completed",
    ...     metrics={"accuracy": 0.95, "f1": 0.93}
    ... )  # doctest: +SKIP

    Using the global experiment logger:

    >>> from insideLLMs.logging_utils import get_experiment_logger
    >>> exp_logger = get_experiment_logger()
    >>> # Same instance across the application

    Querying experiment history:

    >>> exp_logger = ExperimentLogger()
    >>> # After running several experiments...
    >>> all_exps = exp_logger.get_all_experiments()
    >>> for exp in all_exps:
    ...     print(f"{exp.experiment_id}: {exp.status}")  # doctest: +SKIP

    See Also
    --------
    ExperimentLog : The data structure for experiment records
    get_experiment_logger : Get the global ExperimentLogger instance

    """

    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """Initialize the ExperimentLogger.

        Parameters
        ----------
        logger_instance : logging.Logger, optional
            Logger for output. If None, uses get_logger("experiment").
            Default is None.

        Examples
        --------
        Create with default logger:

        >>> from insideLLMs.logging_utils import ExperimentLogger
        >>> exp_logger = ExperimentLogger()
        >>> exp_logger.logger.name
        'insideLLMs.experiment'

        Create with custom logger:

        >>> from insideLLMs.logging_utils import get_logger, ExperimentLogger
        >>> custom_logger = get_logger("custom_experiments")
        >>> exp_logger = ExperimentLogger(logger_instance=custom_logger)
        >>> exp_logger.logger.name
        'insideLLMs.custom_experiments'

        """
        self.logger = logger_instance or get_logger("experiment")
        self._experiments: dict[str, ExperimentLog] = {}

    def start_experiment(
        self,
        experiment_id: str,
        model_id: str,
        probe_type: str,
        n_samples: int,
    ) -> ExperimentLog:
        """Start logging a new experiment.

        Creates an ExperimentLog entry, registers it, and logs the
        experiment start.

        Parameters
        ----------
        experiment_id : str
            Unique identifier for this experiment. Should be unique
            across all experiments in this logger.
        model_id : str
            Identifier of the model being evaluated.
        probe_type : str
            Type of probe/evaluation being performed.
        n_samples : int
            Total number of samples to be processed.

        Returns
        -------
        ExperimentLog
            The created experiment log entry.

        Examples
        --------
        Starting a basic experiment:

        >>> from insideLLMs.logging_utils import ExperimentLogger
        >>> exp_logger = ExperimentLogger()
        >>> log = exp_logger.start_experiment(
        ...     experiment_id="exp-2024-001",
        ...     model_id="gpt-4",
        ...     probe_type="consistency",
        ...     n_samples=1000
        ... )
        >>> log.status
        'running'
        >>> log.n_samples
        1000

        With descriptive experiment ID:

        >>> log = exp_logger.start_experiment(
        ...     experiment_id="gpt4_consistency_v2_20240115",
        ...     model_id="gpt-4-turbo",
        ...     probe_type="self-consistency",
        ...     n_samples=5000
        ... )

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
        """Log the result of processing a single sample.

        Updates the experiment's completion/failure counters and
        optionally records error information.

        Parameters
        ----------
        experiment_id : str
            The experiment to update.
        success : bool
            Whether the sample was processed successfully.
        error : Exception, optional
            The exception if the sample failed. Only used if
            success=False. Default is None.

        Returns
        -------
        None

        Examples
        --------
        Logging a successful sample:

        >>> from insideLLMs.logging_utils import ExperimentLogger
        >>> exp_logger = ExperimentLogger()
        >>> log = exp_logger.start_experiment(
        ...     experiment_id="exp-001",
        ...     model_id="gpt-4",
        ...     probe_type="test",
        ...     n_samples=100
        ... )
        >>> exp_logger.log_sample_result("exp-001", success=True)
        >>> exp_logger.get_experiment("exp-001").n_completed
        1

        Logging a failed sample:

        >>> try:
        ...     raise ValueError("Invalid response format")
        ... except ValueError as e:
        ...     exp_logger.log_sample_result("exp-001", success=False, error=e)
        >>> exp_logger.get_experiment("exp-001").n_failed
        1

        In a processing loop:

        >>> for sample in samples:  # doctest: +SKIP
        ...     try:
        ...         result = process_sample(sample)
        ...         exp_logger.log_sample_result("exp-001", success=True)
        ...     except Exception as e:
        ...         exp_logger.log_sample_result("exp-001", success=False, error=e)

        """
        if experiment_id not in self._experiments:
            return

        log = self._experiments[experiment_id]
        if success:
            log.n_completed += 1
        else:
            log.n_failed += 1
            if error:
                log.errors.append(
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )

    def finish_experiment(
        self,
        experiment_id: str,
        status: str = "completed",
        metrics: Optional[dict[str, float]] = None,
    ) -> Optional[ExperimentLog]:
        """Finish logging an experiment with final status and metrics.

        Marks the experiment as complete, records the end time and
        final metrics, and logs a summary.

        Parameters
        ----------
        experiment_id : str
            The experiment to finish.
        status : str, optional
            Final status (e.g., "completed", "failed", "cancelled").
            Default is "completed".
        metrics : dict[str, float], optional
            Final metrics from the experiment. These are merged with
            any existing metrics. Default is None.

        Returns
        -------
        ExperimentLog or None
            The final experiment log, or None if the experiment_id
            was not found.

        Examples
        --------
        Finishing a successful experiment:

        >>> from insideLLMs.logging_utils import ExperimentLogger
        >>> exp_logger = ExperimentLogger()
        >>> log = exp_logger.start_experiment(
        ...     experiment_id="exp-001",
        ...     model_id="gpt-4",
        ...     probe_type="test",
        ...     n_samples=100
        ... )
        >>> # ... process samples ...
        >>> final_log = exp_logger.finish_experiment(
        ...     "exp-001",
        ...     status="completed",
        ...     metrics={"accuracy": 0.95, "f1_score": 0.93}
        ... )
        >>> final_log.status
        'completed'
        >>> final_log.metrics["accuracy"]
        0.95

        Finishing a failed experiment:

        >>> final_log = exp_logger.finish_experiment(
        ...     "exp-001",
        ...     status="failed",
        ...     metrics={"partial_accuracy": 0.82}
        ... )  # doctest: +SKIP

        Finishing without metrics:

        >>> final_log = exp_logger.finish_experiment("exp-001")  # doctest: +SKIP

        Non-existent experiment returns None:

        >>> result = exp_logger.finish_experiment("non-existent")
        >>> result is None
        True

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
        """Get an experiment log by ID.

        Parameters
        ----------
        experiment_id : str
            The experiment identifier to look up.

        Returns
        -------
        ExperimentLog or None
            The experiment log if found, None otherwise.

        Examples
        --------
        >>> from insideLLMs.logging_utils import ExperimentLogger
        >>> exp_logger = ExperimentLogger()
        >>> log = exp_logger.start_experiment(
        ...     experiment_id="exp-001",
        ...     model_id="gpt-4",
        ...     probe_type="test",
        ...     n_samples=100
        ... )
        >>> retrieved = exp_logger.get_experiment("exp-001")
        >>> retrieved.model_id
        'gpt-4'

        Non-existent experiment:

        >>> exp_logger.get_experiment("non-existent") is None
        True

        """
        return self._experiments.get(experiment_id)

    def get_all_experiments(self) -> list[ExperimentLog]:
        """Get all experiment logs.

        Returns
        -------
        list[ExperimentLog]
            List of all experiment logs registered with this logger,
            in insertion order.

        Examples
        --------
        >>> from insideLLMs.logging_utils import ExperimentLogger
        >>> exp_logger = ExperimentLogger()
        >>> exp_logger.start_experiment("exp-001", "gpt-4", "test", 100)
        ExperimentLog(...)
        >>> exp_logger.start_experiment("exp-002", "gpt-3.5", "test", 200)
        ExperimentLog(...)
        >>> all_exps = exp_logger.get_all_experiments()
        >>> len(all_exps)
        2
        >>> [e.experiment_id for e in all_exps]
        ['exp-001', 'exp-002']

        """
        return list(self._experiments.values())


# Global experiment logger
_experiment_logger: Optional[ExperimentLogger] = None


def get_experiment_logger() -> ExperimentLogger:
    """Get the global experiment logger instance.

    Returns the singleton ExperimentLogger instance, creating it if
    it doesn't exist. This provides a convenient way to track
    experiments across the entire application without passing logger
    instances.

    Returns
    -------
    ExperimentLogger
        The global ExperimentLogger singleton.

    Examples
    --------
    Getting the global logger:

    >>> from insideLLMs.logging_utils import get_experiment_logger
    >>> exp_logger = get_experiment_logger()
    >>> isinstance(exp_logger, ExperimentLogger)
    True

    Multiple calls return the same instance:

    >>> logger1 = get_experiment_logger()
    >>> logger2 = get_experiment_logger()
    >>> logger1 is logger2
    True

    Using in different parts of an application:

    >>> # In experiment_runner.py:
    >>> exp_logger = get_experiment_logger()
    >>> exp_logger.start_experiment("exp-001", "gpt-4", "test", 1000)
    ExperimentLog(...)

    >>> # In sample_processor.py:
    >>> exp_logger = get_experiment_logger()  # Same instance
    >>> exp_logger.log_sample_result("exp-001", success=True)

    >>> # In results_reporter.py:
    >>> exp_logger = get_experiment_logger()  # Same instance
    >>> all_experiments = exp_logger.get_all_experiments()

    See Also
    --------
    ExperimentLogger : The class of the returned instance

    """
    global _experiment_logger
    if _experiment_logger is None:
        _experiment_logger = ExperimentLogger()
    return _experiment_logger
