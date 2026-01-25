"""Command-line interface for insideLLMs.

This module provides a comprehensive command-line interface for interacting with
the insideLLMs framework. It supports running experiments, benchmarking models,
comparing responses, interactive exploration, and managing configurations.

Overview
--------
The CLI is organized into subcommands, each handling a specific workflow:

- ``run``: Execute experiments from YAML/JSON configuration files
- ``harness``: Run cross-model behavioral probe harnesses
- ``benchmark``: Run comprehensive benchmark suites across models and probes
- ``compare``: Compare multiple models on identical inputs
- ``diff``: Compare two run directories for behavioral regressions
- ``list``: Display available models, probes, datasets, and trackers
- ``init``: Generate sample configuration files
- ``info``: Show detailed information about a specific resource
- ``quicktest``: Quickly test a single prompt against a model
- ``interactive``: Start an interactive exploration session
- ``validate``: Validate configuration files or run directories
- ``export``: Export results to CSV, Markdown, or LaTeX formats
- ``schema``: Inspect and validate versioned output schemas
- ``doctor``: Diagnose environment and optional dependencies
- ``report``: Rebuild summary and HTML reports from records

Examples
--------
Running a quick test with a model:

    $ insidellms quicktest "What is 2+2?" --model openai

Running an experiment from a configuration file:

    $ insidellms run experiment.yaml --verbose --format table

Running a behavioral harness across multiple models:

    $ insidellms harness harness_config.yaml --output-dir ./results

Comparing models on the same input:

    $ insidellms compare --models gpt-4,claude-3 --input "Explain gravity"

Listing available resources:

    $ insidellms list all --detailed

Starting interactive mode:

    $ insidellms interactive --model openai

Validating a run directory:

    $ insidellms validate ./results/my-run

Comparing two runs for regressions:

    $ insidellms diff ./baseline ./candidate --fail-on-regressions

Environment Variables
---------------------
INSIDELLMS_RUN_ROOT : str, optional
    Override the default run artifacts root directory (~/.insidellms/runs).
NO_COLOR : str, optional
    Disable colored terminal output when set to any value.
FORCE_COLOR : str, optional
    Force colored output even when stdout is not a TTY.
OPENAI_API_KEY : str, optional
    API key for OpenAI model providers.
ANTHROPIC_API_KEY : str, optional
    API key for Anthropic model providers.

Notes
-----
The CLI supports both synchronous and asynchronous execution modes. For large
experiments, using ``--async`` with ``--concurrency`` can significantly improve
throughput by running multiple model queries in parallel.

All run artifacts are written to a structured directory containing:
- ``manifest.json``: Run metadata and configuration
- ``records.jsonl``: Individual result records in JSON Lines format
- ``config.resolved.yaml``: Resolved configuration snapshot
- ``summary.json``: Aggregated statistics
- ``report.html``: Visual HTML report (if visualization extras installed)

See Also
--------
insideLLMs.runner : Core experiment execution logic
insideLLMs.registry : Model and probe registration
insideLLMs.schemas : Output schema validation
insideLLMs.visualization : Report generation utilities
"""

import argparse
import asyncio
import html
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from insideLLMs._serialization import (
    fingerprint_value as _fingerprint_value,
    serialize_value as _serialize_value,
    stable_json_dumps as _stable_json_dumps,
)
from insideLLMs.registry import (
    ensure_builtins_registered,
    model_registry,
    probe_registry,
)
from insideLLMs.results import results_to_markdown, save_results_json
from insideLLMs.runner import (
    derive_run_id_from_config_path,
    load_config,
    run_experiment_from_config,
    run_experiment_from_config_async,
    run_harness_from_config,
)
from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION
from insideLLMs.statistics import generate_summary_report
from insideLLMs.types import (
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ResultStatus,
)

__all__ = [
    "Colors",
    "ProgressBar",
    "Spinner",
    "colorize",
    "create_parser",
    "main",
    "print_error",
    "print_header",
    "print_info",
    "print_key_value",
    "print_subheader",
    "print_success",
    "print_warning",
]

# CLI output routing:
# - "status" output (headers, warnings, progress) is routed via _CLI_STATUS_STREAM
# - machine-readable payloads (e.g., --format json) should go to stdout only
_CLI_QUIET = False
_CLI_STATUS_STREAM = sys.stdout


def _cli_version_string() -> str:
    """Return the CLI version string.

    Prefer the installed package metadata when available (editable installs, wheels),
    but fall back to the source-tree `insideLLMs.__version__` for local runs.
    """

    try:
        return importlib.metadata.version("insideLLMs")
    except Exception:
        try:
            import insideLLMs

            return str(getattr(insideLLMs, "__version__", "unknown"))
        except Exception:
            return "unknown"


def _add_output_schema_args(parser: argparse.ArgumentParser) -> None:
    """Add common output schema validation arguments to a subcommand parser.

    This helper function adds three standard arguments related to output schema
    validation to any subcommand parser. These arguments control whether outputs
    are validated, which schema version to use, and how to handle validation failures.

    Args:
        parser: The argparse ArgumentParser instance to add arguments to.
            This is typically a subparser for a specific CLI command.

    Returns:
        None. The parser is modified in-place.

    Examples:
        Adding schema arguments to a custom subcommand:

            >>> subparser = subparsers.add_parser("mycommand", help="My command")
            >>> _add_output_schema_args(subparser)
            >>> args = parser.parse_args(["mycommand", "--validate-output"])
            >>> args.validate_output
            True

        Using the added arguments in a command handler:

            >>> if args.validate_output:
            ...     validator.validate(
            ...         schema_name,
            ...         data,
            ...         schema_version=args.schema_version,
            ...         mode=args.validation_mode
            ...     )

    Notes:
        The three arguments added are:
        - ``--validate-output``: Boolean flag to enable schema validation
        - ``--schema-version``: String specifying the schema version (default from DEFAULT_SCHEMA_VERSION)
        - ``--validation-mode``: Choice of "strict" or "warn" for error handling

    See Also:
        insideLLMs.schemas.OutputValidator : The validator that uses these arguments
        insideLLMs.schemas.SchemaRegistry : Registry of available schema versions
    """
    parser.add_argument(
        "--validate-output",
        action="store_true",
        help="Validate serialized outputs against a versioned schema (requires pydantic)",
    )
    parser.add_argument(
        "--schema-version",
        type=str,
        default=DEFAULT_SCHEMA_VERSION,
        help=f"Output schema version to emit/validate (default: {DEFAULT_SCHEMA_VERSION})",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["strict", "warn"],
        default="strict",
        help="On schema mismatch: strict=error, warn=continue (default: strict)",
    )


def _json_default(obj: Any) -> Any:
    """JSON default handler for CLI writes.

    Custom JSON serialization handler that converts Python objects to
    JSON-serializable representations. Used as the ``default`` parameter
    for ``json.dumps()`` and ``json.dump()`` calls throughout the CLI.

    This is a thin wrapper around the shared ``_serialize_value`` function
    from ``insideLLMs._serialization``.

    Args:
        obj: Any Python object that the standard JSON encoder cannot handle.

    Returns:
        A JSON-serializable representation of the object.
    """
    return _serialize_value(obj)


# ============================================================================
# Console Output Utilities (works without external dependencies)
# ============================================================================


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output.

    A collection of ANSI escape sequences for styling terminal output with
    colors, bold text, and other formatting. These codes work on most Unix
    terminals and Windows terminals with ANSI support enabled.

    Attributes:
        RESET: Reset all formatting to terminal default.
        BOLD: Make text bold/bright.
        DIM: Make text dimmer (reduced intensity).
        UNDERLINE: Underline text.

        BLACK: Black foreground color.
        RED: Red foreground color.
        GREEN: Green foreground color.
        YELLOW: Yellow foreground color.
        BLUE: Blue foreground color.
        MAGENTA: Magenta foreground color.
        CYAN: Cyan foreground color.
        WHITE: White foreground color.

        BRIGHT_RED: Bright/light red foreground.
        BRIGHT_GREEN: Bright/light green foreground.
        BRIGHT_YELLOW: Bright/light yellow foreground.
        BRIGHT_BLUE: Bright/light blue foreground.
        BRIGHT_MAGENTA: Bright/light magenta foreground.
        BRIGHT_CYAN: Bright/light cyan foreground.

        BG_RED: Red background color.
        BG_GREEN: Green background color.
        BG_YELLOW: Yellow background color.
        BG_BLUE: Blue background color.

    Examples:
        Basic colored output:

            >>> print(Colors.RED + "Error message" + Colors.RESET)
            Error message  # Displayed in red

        Combining multiple styles:

            >>> print(Colors.BOLD + Colors.CYAN + "Header" + Colors.RESET)
            Header  # Displayed bold and cyan

        Using with the colorize function:

            >>> colorize("Success!", Colors.BRIGHT_GREEN, Colors.BOLD)
            '\x1b[92m\x1b[1mSuccess!\x1b[0m'

    Notes:
        These codes are only applied when the terminal supports ANSI colors.
        Use the ``_supports_color()`` function or the global ``USE_COLOR``
        flag to check support before applying colors.

    See Also:
        _supports_color : Function to detect terminal color support
        colorize : Helper function to apply colors conditionally
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


def _supports_color() -> bool:
    """Check if the terminal supports color output.

    Determines whether the current terminal environment supports ANSI color
    escape sequences. This function checks multiple environment variables
    and terminal capabilities to make an informed decision.

    Returns:
        True if the terminal supports color output, False otherwise.

    Examples:
        Basic usage:

            >>> if _supports_color():
            ...     print("\\033[32mGreen text\\033[0m")
            ... else:
            ...     print("Green text")

        With the global USE_COLOR flag:

            >>> USE_COLOR = _supports_color()
            >>> def print_status(msg):
            ...     if USE_COLOR:
            ...         print(f"\\033[32mOK\\033[0m {msg}")
            ...     else:
            ...         print(f"OK {msg}")

    Notes:
        The detection logic follows this priority:

        1. If NO_COLOR environment variable is set (any value), returns False
           (follows the no-color.org standard)
        2. If FORCE_COLOR environment variable is set, returns True
        3. If stdout is not a TTY (e.g., piped to a file), returns False
        4. On Windows, attempts to enable ANSI support via SetConsoleMode
        5. On Unix-like systems, assumes color support if stdout is a TTY

        On Windows, the function attempts to enable virtual terminal processing
        mode. If this fails, it checks for the ANSICON environment variable
        which indicates a third-party ANSI emulator is installed.

    See Also:
        Colors : Class containing ANSI color codes
        colorize : Function that uses this detection
    """
    # Check for NO_COLOR environment variable (standard)
    if os.environ.get("NO_COLOR"):
        return False

    # Check for FORCE_COLOR environment variable
    if os.environ.get("FORCE_COLOR"):
        return True

    # Check if stdout is a TTY
    if not hasattr(sys.stdout, "isatty"):
        return False

    if not sys.stdout.isatty():
        return False

    # Windows terminal detection
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return os.environ.get("ANSICON") is not None

    return True


# Global flag for color support
USE_COLOR = _supports_color()


def colorize(text: str, *codes: str) -> str:
    """Apply color codes to text if terminal supports colors.

    Wraps text with ANSI escape sequences for terminal styling. If color
    support is disabled (via NO_COLOR env var, non-TTY stdout, etc.),
    returns the text unchanged.

    Args:
        text: The text string to colorize.
        *codes: Variable number of ANSI color codes from the Colors class.
            Multiple codes can be combined (e.g., Colors.BOLD, Colors.RED).

    Returns:
        The text wrapped with ANSI escape sequences if color is supported,
        otherwise returns the original text unchanged.

    Examples:
        Single color:

            >>> colorize("Error occurred", Colors.RED)
            '\x1b[31mError occurred\x1b[0m'  # When color supported
            'Error occurred'  # When color not supported

        Multiple styles combined:

            >>> colorize("Important!", Colors.BOLD, Colors.BRIGHT_YELLOW)
            '\x1b[1m\x1b[93mImportant!\x1b[0m'

        Using in print statements:

            >>> print(colorize("Success", Colors.GREEN) + ": Operation completed")
            Success: Operation completed  # "Success" in green

        Conditional styling:

            >>> status = "PASS" if success else "FAIL"
            >>> color = Colors.GREEN if success else Colors.RED
            >>> print(colorize(status, color))

    Notes:
        The function checks the global USE_COLOR flag which is set at module
        load time based on terminal capabilities. This ensures consistent
        behavior throughout the application.

        The reset code (Colors.RESET) is automatically appended to prevent
        color bleeding into subsequent text.

    See Also:
        Colors : Class containing available ANSI codes
        _supports_color : Function that determines USE_COLOR value
    """
    if not USE_COLOR:
        return text
    return "".join(codes) + text + Colors.RESET


def print_header(title: str) -> None:
    """Print a styled header with decorative borders.

    Prints a prominent header suitable for section titles and command
    introductions. The header is enclosed in double-line box characters
    with the title centered.

    Args:
        title: The text to display in the header. Will be centered
            within a 70-character wide box.

    Returns:
        None. Output is printed directly to stdout.

    Examples:
        Basic header:

            >>> print_header("Running Experiment")
            # Output (with colors):
            # ══════════════════════════════════════════════════════════════════════
            #                           Running Experiment
            # ══════════════════════════════════════════════════════════════════════

        In a command function:

            >>> def cmd_run(args):
            ...     print_header("Running Experiment")
            ...     print_key_value("Config", args.config)
            ...     # ... rest of command

    Notes:
        - A blank line is printed before the header for visual separation
        - The header uses BRIGHT_CYAN color with BOLD title text
        - Fixed width of 70 characters for consistent appearance
    """
    if _CLI_QUIET:
        return
    width = 70
    line = "═" * width
    print(file=_CLI_STATUS_STREAM)
    print(colorize(line, Colors.BRIGHT_CYAN), file=_CLI_STATUS_STREAM)
    print(colorize(f"  {title}".center(width), Colors.BOLD, Colors.BRIGHT_CYAN), file=_CLI_STATUS_STREAM)
    print(colorize(line, Colors.BRIGHT_CYAN), file=_CLI_STATUS_STREAM)


def print_subheader(title: str) -> None:
    """Print a styled subheader with a horizontal rule.

    Prints a secondary heading suitable for subsections within a command's
    output. Uses a lighter style than print_header() to indicate hierarchy.

    Args:
        title: The text to display in the subheader. The horizontal line
            extends to fill remaining space up to 50 characters.

    Returns:
        None. Output is printed directly to stdout.

    Examples:
        Basic subheader:

            >>> print_subheader("Results Summary")
            # Output (with colors):
            # ── Results Summary ─────────────────────────────

        Multiple sections:

            >>> print_header("Benchmark Report")
            >>> print_subheader("Model Performance")
            >>> print_key_value("Accuracy", "95.2%")
            >>> print_subheader("Latency Metrics")
            >>> print_key_value("Average", "123ms")

    Notes:
        - A blank line is printed before the subheader for visual separation
        - Uses CYAN for the title and DIM for the trailing rule
        - The horizontal rule extends based on title length
    """
    if _CLI_QUIET:
        return
    print(file=_CLI_STATUS_STREAM)
    print(
        colorize(f"── {title} ", Colors.CYAN) + colorize("─" * (50 - len(title)), Colors.DIM),
        file=_CLI_STATUS_STREAM,
    )


def print_success(message: str) -> None:
    """Print a success message with green OK prefix.

    Displays a positive status message prefixed with a green "OK" indicator.
    Use for confirming successful operations, file writes, or completions.

    Args:
        message: The success message to display after the OK prefix.

    Returns:
        None. Output is printed directly to stdout.

    Examples:
        Basic success message:

            >>> print_success("Configuration is valid!")
            OK Configuration is valid!  # "OK" in bright green

        After file operations:

            >>> save_results(data, "output.json")
            >>> print_success(f"Results saved to: output.json")
            OK Results saved to: output.json

        In command handlers:

            >>> def cmd_validate(args):
            ...     if validate_config(args.config):
            ...         print_success("Validation passed!")
            ...         return 0
            ...     return 1

    See Also:
        print_error : For error messages with red prefix
        print_warning : For warning messages with yellow prefix
        print_info : For informational messages with blue prefix
    """
    if _CLI_QUIET:
        return
    print(colorize("OK ", Colors.BRIGHT_GREEN) + message, file=_CLI_STATUS_STREAM)


def print_error(message: str) -> None:
    """Print an error message with red ERROR prefix.

    Displays a negative status message prefixed with a red "ERROR" indicator.
    Output is written to stderr rather than stdout for proper stream handling.

    Args:
        message: The error message to display after the ERROR prefix.

    Returns:
        None. Output is printed directly to stderr.

    Examples:
        Basic error message:

            >>> print_error("Config file not found")
            ERROR Config file not found  # Both prefix and message in red

        With file paths:

            >>> if not config_path.exists():
            ...     print_error(f"Config file not found: {config_path}")
            ...     return 1

        Exception handling:

            >>> try:
            ...     run_experiment(config)
            ... except Exception as e:
            ...     print_error(f"Experiment failed: {e}")
            ...     if verbose:
            ...         traceback.print_exc()
            ...     return 1

    Notes:
        - Output goes to stderr, not stdout, to separate errors from normal output
        - Both the "ERROR" prefix and the message text are colored red
        - Use this for fatal errors that prevent operation completion

    See Also:
        print_success : For success messages with green prefix
        print_warning : For warning messages with yellow prefix
        print_info : For informational messages with blue prefix
    """
    print(colorize("ERROR ", Colors.BRIGHT_RED) + colorize(message, Colors.RED), file=sys.stderr)


def print_warning(message: str) -> None:
    """Print a warning message with yellow WARN prefix.

    Displays a cautionary message prefixed with a yellow "WARN" indicator.
    Use for non-fatal issues that may affect results or require attention.

    Args:
        message: The warning message to display after the WARN prefix.

    Returns:
        None. Output is printed directly to stdout.

    Examples:
        Basic warning:

            >>> print_warning("Dataset file not found, using defaults")
            WARN Dataset file not found, using defaults  # In yellow

        Optional dependency missing:

            >>> if not has_plotly:
            ...     print_warning("plotly not installed, skipping visualization")

        Data quality issues:

            >>> if duplicate_count > 0:
            ...     print_warning(f"Found {duplicate_count} duplicate records")

        Deprecation notices:

            >>> print_warning("--output-dir is deprecated, use --run-dir instead")

    Notes:
        - Unlike print_error(), output goes to stdout
        - Both the "WARN" prefix and the message text are colored yellow
        - Use for issues that don't prevent execution but may affect results

    See Also:
        print_error : For fatal error messages
        print_success : For success messages
        print_info : For informational messages
    """
    if _CLI_QUIET:
        return
    print(
        colorize("WARN ", Colors.BRIGHT_YELLOW) + colorize(message, Colors.YELLOW),
        file=_CLI_STATUS_STREAM,
    )


def print_info(message: str) -> None:
    """Print an informational message with blue INFO prefix.

    Displays a neutral status message prefixed with a blue "INFO" indicator.
    Use for progress updates, configuration details, or helpful tips.

    Args:
        message: The informational message to display after the INFO prefix.

    Returns:
        None. Output is printed directly to stdout.

    Examples:
        Progress updates:

            >>> print_info("Using async execution with concurrency=5")
            INFO Using async execution with concurrency=5  # "INFO" in blue

        Configuration details:

            >>> print_info(f"Model: {args.model}")
            >>> print_info(f"Probe: {args.probe}")

        Helpful tips:

            >>> print_info("Tip: Use --verbose for detailed progress")

        Mode indicators:

            >>> if args.dry_run:
            ...     print_info("Dry run mode - no changes will be made")

    Notes:
        - Output goes to stdout
        - Only the "INFO" prefix is colored blue; message text is default
        - Use for neutral information that isn't success, warning, or error

    See Also:
        print_success : For success messages
        print_warning : For warning messages
        print_error : For error messages
    """
    if _CLI_QUIET:
        return
    print(colorize("INFO ", Colors.BRIGHT_BLUE) + message, file=_CLI_STATUS_STREAM)


def print_key_value(key: str, value: Any, indent: int = 2) -> None:
    """Print a formatted key-value pair.

    Displays a label and its value in a consistent format, with the key
    dimmed and followed by a colon. Useful for displaying configuration
    settings, statistics, and metadata.

    Args:
        key: The label or property name to display.
        value: The value to display. Will be converted to string via str().
        indent: Number of spaces to indent from the left margin.
            Defaults to 2 for visual hierarchy under headers.

    Returns:
        None. Output is printed directly to stdout.

    Examples:
        Basic key-value:

            >>> print_key_value("Config", "/path/to/config.yaml")
              Config: /path/to/config.yaml  # "Config:" in dim

        Multiple settings:

            >>> print_key_value("Model", "gpt-4")
            >>> print_key_value("Temperature", 0.7)
            >>> print_key_value("Max tokens", 1000)
              Model: gpt-4
              Temperature: 0.7
              Max tokens: 1000

        With custom indentation:

            >>> print_key_value("Nested value", "data", indent=4)
                Nested value: data

        Under a subheader:

            >>> print_subheader("Configuration")
            >>> print_key_value("Input", args.input)
            >>> print_key_value("Output", args.output)

    Notes:
        - The key is displayed in DIM color to visually separate it from the value
        - A colon is automatically appended to the key
        - Values are converted to strings, so any type is accepted
    """
    if _CLI_QUIET:
        return
    spaces = " " * indent
    print(f"{spaces}{colorize(key + ':', Colors.DIM)} {value}", file=_CLI_STATUS_STREAM)


def _write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write a list of dictionaries to a JSON Lines file.

    Serializes each dictionary as a single JSON line and writes to the
    specified file. Uses the custom _json_default handler for non-standard
    types like datetime, Enum, Path, etc.

    Args:
        records: List of dictionaries to write. Each dictionary becomes
            one line in the output file.
        output_path: Path object specifying where to write the file.
            Parent directories must already exist.

    Returns:
        None. File is written to disk.

    Raises:
        OSError: If the file cannot be opened for writing.
        TypeError: If a record contains a non-serializable type that
            _json_default cannot handle.

    Examples:
        Writing experiment records:

            >>> records = [
            ...     {"input": "What is 2+2?", "output": "4", "status": "success"},
            ...     {"input": "What is 3+3?", "output": "6", "status": "success"},
            ... ]
            >>> _write_jsonl(records, Path("results.jsonl"))

        With complex types:

            >>> from datetime import datetime
            >>> records = [{"timestamp": datetime.now(), "data": {"key": "value"}}]
            >>> _write_jsonl(records, Path("log.jsonl"))

    Notes:
        - Uses UTF-8 encoding for Unicode support
        - Each record is written on its own line with a trailing newline
        - The _json_default function handles datetime, Enum, Path, set, and dataclass types

    See Also:
        _read_jsonl_records : The inverse operation for reading JSONL files
        _json_default : Custom JSON serialization handler
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    default=_json_default,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _format_percent(value: Optional[float]) -> str:
    """Format a decimal value as a percentage string.

    Converts a decimal fraction (0.0 to 1.0) to a human-readable percentage
    string with one decimal place.

    Args:
        value: A decimal value between 0 and 1, or None.

    Returns:
        A percentage string like "95.5%", or "-" if value is None.

    Examples:
        >>> _format_percent(0.955)
        '95.5%'
        >>> _format_percent(1.0)
        '100.0%'
        >>> _format_percent(None)
        '-'
    """
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _format_float(value: Optional[float]) -> str:
    """Format a float value with three decimal places.

    Formats a numeric value to three decimal places for consistent
    display of metrics and scores.

    Args:
        value: A numeric value, or None.

    Returns:
        A string with three decimal places like "0.952", or "-" if value is None.

    Examples:
        >>> _format_float(0.9523)
        '0.952'
        >>> _format_float(123.45678)
        '123.457'
        >>> _format_float(None)
        '-'
    """
    if value is None:
        return "-"
    return f"{value:.3f}"


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Parse a value into a datetime object.

    Attempts to convert various input types into a datetime object.
    Handles datetime instances directly, and parses ISO 8601 format strings.

    Args:
        value: A datetime object, an ISO 8601 format string, or any other value.

    Returns:
        A datetime object if parsing succeeds, None otherwise.

    Examples:
        >>> from datetime import datetime
        >>> _parse_datetime(datetime(2024, 1, 15))
        datetime.datetime(2024, 1, 15, 0, 0)
        >>> _parse_datetime("2024-01-15T12:30:00")
        datetime.datetime(2024, 1, 15, 12, 30)
        >>> _parse_datetime("invalid")
        None
        >>> _parse_datetime(12345)
        None
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _status_from_record(value: Any) -> ResultStatus:
    """Convert a value to a ResultStatus enum.

    Safely converts various input types to a ResultStatus enum member.
    Returns ERROR as the default if conversion fails.

    Args:
        value: A ResultStatus instance, a string like "success" or "error",
            or any other value.

    Returns:
        The corresponding ResultStatus enum member, or ResultStatus.ERROR
        if the value cannot be converted.

    Examples:
        >>> _status_from_record(ResultStatus.SUCCESS)
        <ResultStatus.SUCCESS: 'success'>
        >>> _status_from_record("success")
        <ResultStatus.SUCCESS: 'success'>
        >>> _status_from_record("unknown_status")
        <ResultStatus.ERROR: 'error'>
    """
    if isinstance(value, ResultStatus):
        return value
    try:
        return ResultStatus(str(value))
    except Exception:
        return ResultStatus.ERROR


def _probe_category_from_value(value: Any) -> ProbeCategory:
    """Convert a value to a ProbeCategory enum.

    Safely converts various input types to a ProbeCategory enum member.
    Returns CUSTOM as the default if conversion fails.

    Args:
        value: A ProbeCategory instance, a string like "logic" or "bias",
            or any other value.

    Returns:
        The corresponding ProbeCategory enum member, or ProbeCategory.CUSTOM
        if the value cannot be converted.

    Examples:
        >>> _probe_category_from_value(ProbeCategory.LOGIC)
        <ProbeCategory.LOGIC: 'logic'>
        >>> _probe_category_from_value("bias")
        <ProbeCategory.BIAS: 'bias'>
        >>> _probe_category_from_value("unknown_category")
        <ProbeCategory.CUSTOM: 'custom'>
    """
    if isinstance(value, ProbeCategory):
        return value
    try:
        return ProbeCategory(str(value))
    except Exception:
        return ProbeCategory.CUSTOM


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read records from a JSON Lines file.

    Parses a JSONL file where each line is a JSON object. Empty lines
    are skipped and only dictionary objects are included in the result.

    Args:
        path: Path to the JSONL file to read.

    Returns:
        List of dictionaries, one for each valid JSON object line.

    Raises:
        ValueError: If a non-empty line contains invalid JSON.
        OSError: If the file cannot be opened.

    Examples:
        Reading experiment records:

            >>> records = _read_jsonl_records(Path("results.jsonl"))
            >>> len(records)
            100
            >>> records[0]["status"]
            'success'

        Handling errors:

            >>> try:
            ...     records = _read_jsonl_records(Path("bad.jsonl"))
            ... except ValueError as e:
            ...     print(f"Parse error: {e}")

    Notes:
        - Uses UTF-8 encoding
        - Non-dict JSON values (arrays, strings, etc.) are silently skipped
        - Line numbers in error messages are 1-indexed

    See Also:
        _write_jsonl : The inverse operation for writing JSONL files
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    """Extract a unique key tuple from a result record.

    Generates a composite key (model_id, probe_id, item_id) that uniquely
    identifies a record for comparison operations like diff. The key is
    extracted from various possible locations in the record structure.

    Args:
        record: A result record dictionary from records.jsonl.

    Returns:
        A tuple of (model_id, probe_id, item_id) strings. Falls back to
        default values like "model" or "0" if fields are missing.

    Examples:
        >>> record = {
        ...     "custom": {"harness": {"model_id": "gpt-4", "probe_type": "logic"}},
        ...     "example_id": "ex_001"
        ... }
        >>> _record_key(record)
        ('gpt-4', 'logic', 'ex_001')

        >>> record = {"model": {"model_id": "claude"}, "input": "test"}
        >>> key = _record_key(record)
        >>> key[0]
        'claude'

    Notes:
        The item_id is resolved in priority order:
        1. custom.replicate_key
        2. messages_hash or input fingerprint
        3. example_id or harness.example_index
        4. "0" as fallback

    See Also:
        _record_label : Similar but returns human-readable labels
    """
    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    harness = custom.get("harness") if isinstance(custom.get("harness"), dict) else {}
    model_spec = record.get("model") if isinstance(record.get("model"), dict) else {}
    probe_spec = record.get("probe") if isinstance(record.get("probe"), dict) else {}

    model_id = (
        harness.get("model_id")
        or model_spec.get("model_id")
        or harness.get("model_name")
        or "model"
    )
    probe_id = (
        harness.get("probe_type")
        or probe_spec.get("probe_id")
        or harness.get("probe_name")
        or "probe"
    )
    replicate_key = custom.get("replicate_key")
    example_id = record.get("example_id") or harness.get("example_index")
    stable_id = record.get("messages_hash") or _fingerprint_value(record.get("input"))
    chosen_id = replicate_key or stable_id or example_id or "0"
    return (str(model_id), str(probe_id), str(chosen_id))


def _record_label(record: dict[str, Any]) -> tuple[str, str, str]:
    """Extract human-readable labels from a result record.

    Generates display-friendly labels (model_label, probe_name, example_id)
    for presenting records in CLI output. Unlike _record_key, this may include
    additional context like "(model_id)" suffixes for clarity.

    Args:
        record: A result record dictionary from records.jsonl.

    Returns:
        A tuple of (model_label, probe_name, example_id) strings suitable
        for display. Model label may include ID in parentheses if different
        from the name.

    Examples:
        >>> record = {
        ...     "custom": {"harness": {"model_name": "GPT-4", "model_id": "gpt-4-0125"}}
        ... }
        >>> _record_label(record)
        ('GPT-4 (gpt-4-0125)', 'probe', '0')

        >>> record = {"custom": {"harness": {"model_name": "claude"}}}
        >>> _record_label(record)
        ('claude', 'probe', '0')

    Notes:
        The model label format is:
        - "ModelName (model_id)" if name and id differ
        - "ModelName" if name and id are the same or id is missing

    See Also:
        _record_key : Similar but returns unique keys for comparison
    """
    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    harness = custom.get("harness") if isinstance(custom.get("harness"), dict) else {}
    model_spec = record.get("model") if isinstance(record.get("model"), dict) else {}
    probe_spec = record.get("probe") if isinstance(record.get("probe"), dict) else {}

    model_name = harness.get("model_name") or model_spec.get("model_id") or "model"
    model_id = harness.get("model_id") or model_spec.get("model_id")
    if model_id and model_id != model_name:
        model_label = f"{model_name} ({model_id})"
    else:
        model_label = str(model_name)

    probe_name = harness.get("probe_name") or probe_spec.get("probe_id") or "probe"
    example_id = record.get("example_id") or harness.get("example_index") or "0"
    return (str(model_label), str(probe_name), str(example_id))


def _status_string(record: dict[str, Any]) -> str:
    status = record.get("status")
    return str(status) if status is not None else "unknown"


def _output_text(record: dict[str, Any]) -> Optional[str]:
    output_text = record.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output = record.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        value = output.get("output_text") or output.get("text")
        if isinstance(value, str):
            return value
    return None


def _strip_volatile_keys(value: Any, ignore_keys: set[str]) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key).lower()
            if key_str in ignore_keys:
                continue
            cleaned[key] = _strip_volatile_keys(item, ignore_keys)
        return cleaned
    if isinstance(value, list):
        return [_strip_volatile_keys(item, ignore_keys) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_volatile_keys(item, ignore_keys) for item in value)
    return value


def _trim_text(text: str, limit: int = 200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _output_summary(
    record: dict[str, Any], ignore_keys: Optional[set[str]]
) -> Optional[dict[str, Any]]:
    output_text = _output_text(record)
    if isinstance(output_text, str):
        return {"type": "text", "preview": _trim_text(output_text), "length": len(output_text)}
    output = record.get("output")
    if output is None:
        return None
    return {
        "type": "structured",
        "fingerprint": _output_fingerprint(record, ignore_keys=ignore_keys),
    }


def _output_fingerprint(
    record: dict[str, Any], ignore_keys: Optional[set[str]] = None
) -> Optional[str]:
    output = record.get("output")
    if ignore_keys:
        custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
        override = custom.get("output_fingerprint")
        if output is None and isinstance(override, str):
            return override
        if output is None:
            return None
        sanitized = _strip_volatile_keys(output, ignore_keys)
        return _fingerprint_value(sanitized)

    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    override = custom.get("output_fingerprint")
    if isinstance(override, str):
        return override
    if output is None:
        return None
    return _fingerprint_value(output)


def _trace_fingerprint(record: dict[str, Any]) -> Optional[str]:
    """Extract trace fingerprint from ResultRecord.custom.

    Supports both the legacy flat fields:
      - record.custom.trace_fingerprint
    and the structured trace bundle:
      - record.custom.trace.fingerprint.value
    """
    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    fp = custom.get("trace_fingerprint")
    if isinstance(fp, str):
        return fp

    trace = custom.get("trace") if isinstance(custom.get("trace"), dict) else {}
    fingerprint = trace.get("fingerprint") if isinstance(trace.get("fingerprint"), dict) else {}
    value = fingerprint.get("value")
    if not isinstance(value, str) or not value:
        return None
    # If the bundle stores raw 64-hex without prefix, normalize to the legacy "sha256:<hex>" form.
    if ":" not in value and len(value) == 64:
        return f"sha256:{value}"
    return value


def _trace_violations(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract trace violations from ResultRecord.custom.

    Supports both the legacy flat fields:
      - record.custom.trace_violations
    and the structured trace bundle:
      - record.custom.trace.violations
    """
    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    violations = custom.get("trace_violations")
    if isinstance(violations, list):
        return violations

    trace = custom.get("trace") if isinstance(custom.get("trace"), dict) else {}
    violations = trace.get("violations")
    return violations if isinstance(violations, list) else []


def _trace_violation_count(record: dict[str, Any]) -> int:
    """Count trace violations in a record."""
    return len(_trace_violations(record))


def _primary_score(record: dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    scores = record.get("scores") if isinstance(record.get("scores"), dict) else {}
    metric = record.get("primary_metric")
    if metric and metric in scores and isinstance(scores[metric], (int, float)):
        return str(metric), float(scores[metric])
    if "score" in scores and isinstance(scores["score"], (int, float)):
        return "score", float(scores["score"])
    return None, None


def _metric_mismatch_reason(record_a: dict[str, Any], record_b: dict[str, Any]) -> Optional[str]:
    scores_a = record_a.get("scores")
    scores_b = record_b.get("scores")
    if not isinstance(scores_a, dict) or not isinstance(scores_b, dict):
        return "type_mismatch"

    keys_a = sorted(scores_a.keys())
    keys_b = sorted(scores_b.keys())
    if not keys_a or not keys_b:
        return "missing_scores"

    primary_a = record_a.get("primary_metric")
    primary_b = record_b.get("primary_metric")
    if primary_a and primary_b and primary_a != primary_b:
        return "primary_metric_mismatch"
    if not primary_a or not primary_b:
        return "missing_primary_metric"

    if primary_a not in scores_a or primary_b not in scores_b:
        return "metric_key_missing"

    value_a = scores_a.get(primary_a)
    value_b = scores_b.get(primary_b)
    if not isinstance(value_a, (int, float)) or not isinstance(value_b, (int, float)):
        return "non_numeric_metric"

    return "type_mismatch"


def _metric_mismatch_context(record_a: dict[str, Any], record_b: dict[str, Any]) -> dict[str, Any]:
    scores_a = record_a.get("scores")
    scores_b = record_b.get("scores")
    primary_a = record_a.get("primary_metric")
    primary_b = record_b.get("primary_metric")

    keys_a = sorted(scores_a.keys()) if isinstance(scores_a, dict) else None
    keys_b = sorted(scores_b.keys()) if isinstance(scores_b, dict) else None

    context = {
        "baseline": {
            "primary_metric": primary_a,
            "scores_keys": keys_a,
            "metric_value": scores_a.get(primary_a) if isinstance(scores_a, dict) else None,
        },
        "candidate": {
            "primary_metric": primary_b,
            "scores_keys": keys_b,
            "metric_value": scores_b.get(primary_b) if isinstance(scores_b, dict) else None,
        },
    }

    custom = record_a.get("custom") if isinstance(record_a.get("custom"), dict) else {}
    replicate_key = custom.get("replicate_key")
    if replicate_key:
        context["replicate_key"] = replicate_key

    return context


def _metric_mismatch_details(record_a: dict[str, Any], record_b: dict[str, Any]) -> str:
    context = _metric_mismatch_context(record_a, record_b)

    baseline = context.get("baseline", {})
    candidate = context.get("candidate", {})
    details = [
        f"baseline.primary_metric={baseline.get('primary_metric')!r}",
        f"candidate.primary_metric={candidate.get('primary_metric')!r}",
    ]

    if baseline.get("scores_keys") is not None:
        details.append(f"baseline.scores keys={baseline.get('scores_keys')!r}")
    else:
        details.append("baseline.scores type=None")

    if candidate.get("scores_keys") is not None:
        details.append(f"candidate.scores keys={candidate.get('scores_keys')!r}")
    else:
        details.append("candidate.scores type=None")

    if baseline.get("primary_metric") is not None:
        details.append(
            f"baseline.{baseline.get('primary_metric')}={baseline.get('metric_value')!r}"
        )
    if candidate.get("primary_metric") is not None:
        details.append(
            f"candidate.{candidate.get('primary_metric')}={candidate.get('metric_value')!r}"
        )

    if "replicate_key" in context:
        details.append(f"replicate_key={context['replicate_key']}")

    return "; ".join(details)


def _build_experiments_from_records(
    records: list[dict[str, Any]],
) -> tuple[list[ExperimentResult], dict[str, Any], str]:
    if not records:
        return [], {"derived_from_records": True}, DEFAULT_SCHEMA_VERSION

    schema_version = records[0].get("schema_version") or DEFAULT_SCHEMA_VERSION

    harness_records = [
        r
        for r in records
        if isinstance(r.get("custom"), dict) and isinstance(r["custom"].get("harness"), dict)
    ]

    experiments: list[ExperimentResult] = []
    derived_config: dict[str, Any] = {"derived_from_records": True}

    if harness_records:
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in harness_records:
            harness = record.get("custom", {}).get("harness", {})
            experiment_id = harness.get("experiment_id") or "unknown"
            groups.setdefault(str(experiment_id), []).append(record)

        models: dict[str, dict[str, Any]] = {}
        probes: dict[str, dict[str, Any]] = {}
        dataset_summary: dict[str, Any] = {}

        for experiment_id, group_records in groups.items():
            first = group_records[0]
            harness = first.get("custom", {}).get("harness", {})

            model_name = (
                harness.get("model_name") or first.get("model", {}).get("model_id") or "model"
            )
            model_id = (
                harness.get("model_id") or first.get("model", {}).get("model_id") or model_name
            )
            provider = (
                first.get("model", {}).get("provider") or harness.get("model_type") or "unknown"
            )
            extra = first.get("model", {}).get("params") or {}

            probe_name = (
                harness.get("probe_name") or first.get("probe", {}).get("probe_id") or "probe"
            )
            probe_category = _probe_category_from_value(harness.get("probe_category"))

            model_info = ModelInfo(
                name=str(model_name),
                provider=str(provider),
                model_id=str(model_id),
                extra=extra,
            )

            def _sort_key(item: dict[str, Any]) -> int:
                harness_item = item.get("custom", {}).get("harness", {})
                return int(harness_item.get("example_index", 0))

            sorted_records = sorted(group_records, key=_sort_key)
            probe_results = [
                ProbeResult(
                    input=record.get("input"),
                    output=record.get("output"),
                    status=_status_from_record(record.get("status")),
                    error=record.get("error"),
                    latency_ms=record.get("latency_ms"),
                    metadata=record.get("custom") or {},
                )
                for record in sorted_records
            ]

            started_at = min(
                (dt for dt in (_parse_datetime(r.get("started_at")) for r in group_records) if dt),
                default=None,
            )
            completed_at = max(
                (
                    dt
                    for dt in (_parse_datetime(r.get("completed_at")) for r in group_records)
                    if dt
                ),
                default=None,
            )

            experiments.append(
                ExperimentResult(
                    experiment_id=experiment_id,
                    model_info=model_info,
                    probe_name=str(probe_name),
                    probe_category=probe_category,
                    results=probe_results,
                    score=None,
                    started_at=started_at,
                    completed_at=completed_at,
                    config={
                        "model": {"type": harness.get("model_type")},
                        "probe": {"type": harness.get("probe_type")},
                        "dataset": {
                            "name": harness.get("dataset"),
                            "format": harness.get("dataset_format"),
                        },
                    },
                )
            )

            if harness.get("model_type"):
                models.setdefault(
                    str(harness.get("model_type")), {"type": harness.get("model_type")}
                )
            if harness.get("probe_type"):
                probes.setdefault(
                    str(harness.get("probe_type")), {"type": harness.get("probe_type")}
                )
            if harness.get("dataset") and not dataset_summary:
                dataset_summary = {
                    "name": harness.get("dataset"),
                    "format": harness.get("dataset_format"),
                }

        derived_config.update(
            {
                "models": list(models.values()),
                "probes": list(probes.values()),
                "dataset": dataset_summary,
            }
        )
        return experiments, derived_config, schema_version

    run_groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        run_id = record.get("run_id") or "run"
        run_groups.setdefault(str(run_id), []).append(record)

    for run_id, group_records in run_groups.items():
        first = group_records[0]
        model_spec = first.get("model", {}) if isinstance(first.get("model"), dict) else {}
        probe_spec = first.get("probe", {}) if isinstance(first.get("probe"), dict) else {}

        model_id = model_spec.get("model_id") or "model"
        model_name = model_spec.get("params", {}).get("name") or model_id
        provider = model_spec.get("provider") or "unknown"
        extra = model_spec.get("params") or {}

        model_info = ModelInfo(
            name=str(model_name),
            provider=str(provider),
            model_id=str(model_id),
            extra=extra,
        )

        probe_name = probe_spec.get("probe_id") or "probe"
        probe_category = ProbeCategory.CUSTOM

        probe_results = [
            ProbeResult(
                input=record.get("input"),
                output=record.get("output"),
                status=_status_from_record(record.get("status")),
                error=record.get("error"),
                latency_ms=record.get("latency_ms"),
                metadata=record.get("custom") or {},
            )
            for record in group_records
        ]

        started_at = min(
            (dt for dt in (_parse_datetime(r.get("started_at")) for r in group_records) if dt),
            default=None,
        )
        completed_at = max(
            (dt for dt in (_parse_datetime(r.get("completed_at")) for r in group_records) if dt),
            default=None,
        )

        experiments.append(
            ExperimentResult(
                experiment_id=run_id,
                model_info=model_info,
                probe_name=str(probe_name),
                probe_category=probe_category,
                results=probe_results,
                score=None,
                started_at=started_at,
                completed_at=completed_at,
                config={},
            )
        )

    return experiments, derived_config, schema_version


def _build_basic_harness_report(
    experiments: list[ExperimentResult],
    summary: dict[str, Any],
    title: str,
    generated_at: Optional[datetime] = None,
) -> str:
    rows = []
    for experiment in experiments:
        latencies = [r.latency_ms for r in experiment.results if r.latency_ms is not None]
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        accuracy = experiment.score.accuracy if experiment.score else None
        rows.append(
            (
                html.escape(experiment.model_info.name),
                html.escape(experiment.probe_name),
                _format_percent(experiment.success_rate),
                _format_float(accuracy),
                _format_float(avg_latency),
            )
        )

    rows.sort(key=lambda row: (row[0], row[1]))

    rows_html = "\n".join(
        f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td>"
        f"<td>{row[3]}</td><td>{row[4]}</td></tr>"
        for row in rows
    )

    def summary_table(section: str) -> str:
        items = summary.get(section, {})
        lines = []
        for name in sorted(items):
            stats = items[name]
            success = stats.get("success_rate", {}).get("mean")
            ci = stats.get("success_rate_ci", {})
            ci_text = "-"
            if ci and ci.get("lower") is not None and ci.get("upper") is not None:
                ci_text = f"{ci.get('lower'):.3f}..{ci.get('upper'):.3f}"
            lines.append(
                f"<tr><td>{html.escape(name)}</td>"
                f"<td>{_format_percent(success)}</td><td>{ci_text}</td></tr>"
            )
        return "\n".join(lines)

    by_model_rows = summary_table("by_model")
    by_probe_rows = summary_table("by_probe")

    meta_line = ""
    if generated_at is not None:
        meta_line = f'<div class="meta">Generated {generated_at.isoformat()}</div>'

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #666; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .section {{ margin-top: 24px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  {meta_line}

  <div class="section">
    <h2>Model x Probe Summary</h2>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Probe</th>
          <th>Success Rate</th>
          <th>Accuracy</th>
          <th>Avg Latency (ms)</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>By Model</h2>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Success Rate</th>
          <th>Success Rate CI</th>
        </tr>
      </thead>
      <tbody>
        {by_model_rows}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>By Probe</h2>
    <table>
      <thead>
        <tr>
          <th>Probe</th>
          <th>Success Rate</th>
          <th>Success Rate CI</th>
        </tr>
      </thead>
      <tbody>
        {by_probe_rows}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


class ProgressBar:
    """Simple progress bar for CLI output with ETA estimation.

    A terminal-based progress indicator that displays completion percentage,
    a visual bar, item counts, and estimated time remaining. The bar updates
    in place using carriage returns.

    Parameters
    ----------
    total : int
        The total number of items to process. Used to calculate percentage
        and estimate remaining time.
    width : int, optional
        The width of the progress bar in characters. Default is 40.
    prefix : str, optional
        Text label displayed before the progress bar. Default is "Progress".
    show_eta : bool, optional
        Whether to display estimated time remaining. Default is True.

    Attributes
    ----------
    total : int
        Total number of items to process.
    width : int
        Width of the progress bar in characters.
    prefix : str
        Label displayed before the bar.
    show_eta : bool
        Whether ETA is displayed.
    current : int
        Current progress count (0 to total).
    start_time : float
        Unix timestamp when the progress bar was created.

    Examples
    --------
    Basic usage with explicit updates:

        >>> progress = ProgressBar(100, prefix="Processing")
        >>> for i in range(100):
        ...     do_work(i)
        ...     progress.update(i + 1)
        >>> progress.finish()
        Processing: ████████████████████████████████████████ 100.0% (100/100) Done in 5.23s

    Using increment for simpler iteration:

        >>> progress = ProgressBar(len(items), prefix="Downloading")
        >>> for item in items:
        ...     download(item)
        ...     progress.increment()
        >>> progress.finish()

    Without ETA display:

        >>> progress = ProgressBar(50, show_eta=False, width=20)
        >>> # ... process items ...

    As a progress callback:

        >>> def callback(current, total):
        ...     if progress_bar is None:
        ...         progress_bar = ProgressBar(total)
        ...     progress_bar.update(current)

    Notes
    -----
    - Uses filled blocks (filled) and empty blocks (empty) for the visual bar
    - ETA is calculated using elapsed time and items processed
    - The bar renders on a single line using carriage return
    - Call finish() when complete to print final time and newline

    See Also
    --------
    Spinner : For indeterminate progress (unknown duration)
    """

    def __init__(
        self,
        total: int,
        width: int = 40,
        prefix: str = "Progress",
        show_eta: bool = True,
    ):
        """Initialize a new progress bar.

        Args:
            total: Total number of items to process.
            width: Width of the bar in characters. Default is 40.
            prefix: Label text before the bar. Default is "Progress".
            show_eta: Whether to show time estimate. Default is True.
        """
        self.total = total
        self.width = width
        self.prefix = prefix
        self.show_eta = show_eta
        self.current = 0
        self.start_time = time.time()

    def update(self, current: int) -> None:
        """Update the progress bar to a specific position.

        Sets the current progress to the given value and re-renders the bar.
        Use this when you know the absolute position rather than incrementing.

        Args:
            current: The new progress value (0 to total).

        Examples:
            >>> progress = ProgressBar(100)
            >>> progress.update(25)   # Jump to 25%
            >>> progress.update(50)   # Jump to 50%
            >>> progress.update(100)  # Complete
        """
        self.current = current
        self._render()

    def increment(self, amount: int = 1) -> None:
        """Increment the progress by a given amount.

        Adds to the current progress and re-renders. Useful in simple loops
        where each iteration represents one unit of progress.

        Args:
            amount: How much to add to current progress. Default is 1.

        Examples:
            >>> progress = ProgressBar(10)
            >>> for item in items:
            ...     process(item)
            ...     progress.increment()  # Adds 1 each time

            >>> # Skip by larger amounts
            >>> progress.increment(5)  # Add 5 at once
        """
        self.current += amount
        self._render()

    def _render(self) -> None:
        """Render the progress bar to the terminal.

        Internal method that draws the current state of the progress bar.
        Uses carriage return to update in place without scrolling.
        """
        if _CLI_QUIET:
            return
        pct = 100 if self.total == 0 else min(100, self.current / self.total * 100)

        filled = int(self.width * self.current / max(1, self.total))
        bar = "█" * filled + "░" * (self.width - filled)

        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.current > 0 and self.show_eta:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f" ETA: {eta:.1f}s" if eta > 0 else ""
        else:
            eta_str = ""

        line = (
            f"\r{self.prefix}: {colorize(bar, Colors.CYAN)} "
            f"{pct:5.1f}% ({self.current}/{self.total}){eta_str}"
        )
        print(line, end="", flush=True, file=_CLI_STATUS_STREAM)

    def finish(self) -> None:
        """Complete the progress bar and print final status.

        Sets progress to 100%, renders the final state, and prints the
        total elapsed time with a newline. Call this when processing is done.

        Examples:
            >>> progress = ProgressBar(100, prefix="Processing")
            >>> for i in range(100):
            ...     process(i)
            ...     progress.update(i + 1)
            >>> progress.finish()  # Prints "Done in X.XXs" and newline
        """
        if _CLI_QUIET:
            return
        self.current = self.total
        self._render()
        elapsed = time.time() - self.start_time
        print(
            f" {colorize(f'Done in {elapsed:.2f}s', Colors.GREEN)}",
            file=_CLI_STATUS_STREAM,
        )


class Spinner:
    """Simple spinner for indeterminate progress."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Loading"):
        self.message = message
        self.frame_idx = 0
        self.running = False

    def spin(self) -> None:
        """Render a single spinner frame."""
        if _CLI_QUIET:
            return
        frame = self.FRAMES[self.frame_idx % len(self.FRAMES)]
        print(
            f"\r{colorize(frame, Colors.CYAN)} {self.message}...",
            end="",
            flush=True,
            file=_CLI_STATUS_STREAM,
        )
        self.frame_idx += 1

    def stop(self, success: bool = True) -> None:
        """Stop the spinner with a final status."""
        if _CLI_QUIET:
            return
        if success:
            print(
                f"\r{colorize('OK', Colors.GREEN)} {self.message}... done",
                file=_CLI_STATUS_STREAM,
            )
        else:
            print(
                f"\r{colorize('FAIL', Colors.RED)} {self.message}... failed",
                file=_CLI_STATUS_STREAM,
            )


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    # Custom formatter that preserves formatting in epilog
    class CustomFormatter(argparse.RawDescriptionHelpFormatter):
        def __init__(self, prog):
            super().__init__(prog, max_help_position=40, width=100)

    parser = argparse.ArgumentParser(
        prog="insidellms",
        description=colorize("insideLLMs", Colors.BOLD, Colors.CYAN)
        + " - A world-class toolkit for probing, evaluating, and testing large language models",
        formatter_class=CustomFormatter,
        epilog=f"""
{colorize("Examples:", Colors.BOLD)}

  {colorize("# Quick evaluation", Colors.DIM)}
  insidellms quicktest "What is 2+2?" --model openai

  {colorize("# Run a full experiment from config", Colors.DIM)}
  insidellms run config.yaml --verbose

  {colorize("# Run benchmark suite", Colors.DIM)}
  insidellms benchmark --models openai,anthropic --probes logic,bias

  {colorize("# List available resources", Colors.DIM)}
  insidellms list all

  {colorize("# Interactive exploration", Colors.DIM)}
  insidellms interactive

  {colorize("# Compare models", Colors.DIM)}
  insidellms compare --models gpt-4,claude-3 --input "Explain quantum computing"

{colorize("Documentation:", Colors.BOLD)} https://github.com/dr-gareth-roberts/insideLLMs
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_cli_version_string()}",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output (errors only)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # =========================================================================
    # Run command
    # =========================================================================
    run_parser = subparsers.add_parser(
        "run",
        help="Run an experiment from a configuration file",
        formatter_class=CustomFormatter,
    )
    run_parser.add_argument(
        "config",
        type=str,
        help="Path to the experiment configuration file (YAML or JSON)",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for results (JSON format)",
    )
    run_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "markdown", "table", "summary"],
        default="table",
        help="Output format (default: table)",
    )

    # Deterministic harness artifact location controls
    run_parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Write run artifacts (manifest.json + records.jsonl) exactly to this directory. "
            "(Final directory; overrides --run-root/--run-id.)"
        ),
    )
    run_parser.add_argument(
        "--run-root",
        type=str,
        default=None,
        help=(
            "Base directory for runs when --run-dir is not set (default: ~/.insidellms/runs). "
            "The final directory is <run_root>/<run_id>."
        ),
    )
    run_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Set the run_id recorded in the manifest. Also used as the directory name when using --run-root. "
            "If omitted, a deterministic run_id is derived from the resolved config."
        ),
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing non-empty run directory (DANGEROUS).",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing run directory if records.jsonl is present.",
    )
    run_parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async execution for parallel processing",
    )
    run_parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=5,
        help="Number of concurrent requests (only with --async)",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress information",
    )
    run_parser.add_argument(
        "--track",
        type=str,
        choices=["local", "wandb", "mlflow", "tensorboard"],
        help="Enable experiment tracking with specified backend",
    )
    run_parser.add_argument(
        "--track-project",
        type=str,
        default="insidellms",
        help="Project name for experiment tracking",
    )

    _add_output_schema_args(run_parser)

    # =========================================================================
    # Harness command
    # =========================================================================
    harness_parser = subparsers.add_parser(
        "harness",
        help="Run a cross-model probe harness",
        formatter_class=CustomFormatter,
    )
    harness_parser.add_argument(
        "config",
        type=str,
        help="Path to the harness configuration file",
    )
    harness_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for harness artifacts (alias for --run-dir; deprecated)",
    )

    harness_parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Write harness run artifacts (manifest.json + records.jsonl) exactly to this directory. "
            "(Final directory; overrides --run-root/--run-id.)"
        ),
    )
    harness_parser.add_argument(
        "--run-root",
        type=str,
        default=None,
        help=(
            "Base directory for harness runs when --run-dir is not set. "
            "The final directory is <run_root>/<run_id>."
        ),
    )
    harness_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Set the run_id recorded in the manifest. Also used as the directory name when using --run-root. "
            "If omitted, a deterministic run_id is derived from the resolved config."
        ),
    )
    harness_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing non-empty run directory (DANGEROUS).",
    )
    harness_parser.add_argument(
        "--report-title",
        type=str,
        help="Title for the HTML report",
    )
    harness_parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip generating the HTML report",
    )
    harness_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress and tracebacks",
    )
    harness_parser.add_argument(
        "--track",
        type=str,
        choices=["local", "wandb", "mlflow", "tensorboard"],
        help="Enable experiment tracking with specified backend",
    )
    harness_parser.add_argument(
        "--track-project",
        type=str,
        default="insidellms",
        help="Project name for experiment tracking",
    )

    _add_output_schema_args(harness_parser)

    # =========================================================================
    # Report command
    # =========================================================================
    report_parser = subparsers.add_parser(
        "report",
        help="Rebuild summary.json and report.html from records.jsonl",
        formatter_class=CustomFormatter,
    )
    report_parser.add_argument(
        "run_dir",
        type=str,
        help="Run directory containing records.jsonl",
    )
    report_parser.add_argument(
        "--report-title",
        type=str,
        help="Title for the HTML report",
    )

    # =========================================================================
    # Diff command
    # =========================================================================
    diff_parser = subparsers.add_parser(
        "diff",
        help="Compare two run directories and show behavioural regressions",
        formatter_class=CustomFormatter,
    )
    diff_parser.add_argument(
        "run_dir_a",
        type=str,
        help="Baseline run directory",
    )
    diff_parser.add_argument(
        "run_dir_b",
        type=str,
        help="Comparison run directory",
    )
    diff_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    diff_parser.add_argument(
        "--output",
        type=str,
        help="Write JSON output to a file (json format only)",
    )
    diff_parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of items to show per section (default: 25)",
    )
    diff_parser.add_argument(
        "--fail-on-regressions",
        action="store_true",
        help="Exit with non-zero status if regressions are detected",
    )
    diff_parser.add_argument(
        "--fail-on-changes",
        action="store_true",
        help=(
            "Exit with non-zero status if any differences are detected "
            "(regressions, changes, or missing/extra records)"
        ),
    )
    diff_parser.add_argument(
        "--output-fingerprint-ignore",
        action="append",
        default=[],
        help=(
            "Comma-separated output keys to ignore when fingerprinting structured outputs "
            "(repeatable)."
        ),
    )
    diff_parser.add_argument(
        "--fail-on-trace-violations",
        action="store_true",
        help=(
            "Exit with non-zero status if trace violations increase "
            "(candidate has more violations than baseline for any record)"
        ),
    )
    diff_parser.add_argument(
        "--fail-on-trace-drift",
        action="store_true",
        help=(
            "Exit with non-zero status if trace fingerprints differ between baseline and candidate "
            "(even if output text stays the same)"
        ),
    )

    # =========================================================================
    # Schema command
    # =========================================================================
    schema_parser = subparsers.add_parser(
        "schema",
        help="Inspect and validate versioned output schemas",
        formatter_class=CustomFormatter,
    )
    schema_parser.add_argument(
        "op",
        nargs="?",
        default="list",
        help=(
            "Operation: list | dump | validate | <SchemaName>. "
            "Shortcut: `insidellms schema ProbeResult` dumps JSON Schema."
        ),
    )
    schema_parser.add_argument(
        "--name",
        help="Schema name (optional when using shortcut form: `insidellms schema <SchemaName>`)",
    )
    schema_parser.add_argument(
        "--version",
        default=DEFAULT_SCHEMA_VERSION,
        help=f"Schema version (default: {DEFAULT_SCHEMA_VERSION})",
    )
    schema_parser.add_argument(
        "--input",
        "-i",
        help="Input file to validate (.json or .jsonl) (for op=validate)",
    )
    schema_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Write JSON Schema to a file (for op=dump; otherwise prints to stdout)",
    )
    schema_parser.add_argument(
        "--mode",
        choices=["strict", "warn"],
        default="strict",
        help="On validation error: strict=exit non-zero, warn=continue",
    )

    # =========================================================================
    # Doctor command
    # =========================================================================
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Diagnose environment and optional dependencies",
        formatter_class=CustomFormatter,
    )
    doctor_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    doctor_parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit non-zero if any recommended dependency checks fail",
    )

    # =========================================================================
    # Quick test command
    # =========================================================================
    quicktest_parser = subparsers.add_parser(
        "quicktest",
        help="Quickly test a prompt against a model",
        formatter_class=CustomFormatter,
    )
    quicktest_parser.add_argument(
        "prompt",
        type=str,
        help="The prompt to test",
    )
    quicktest_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="dummy",
        help="Model to use (default: dummy)",
    )
    quicktest_parser.add_argument(
        "--model-args",
        type=str,
        default="{}",
        help="JSON string of model arguments",
    )
    quicktest_parser.add_argument(
        "--probe",
        "-p",
        type=str,
        help="Optional probe to apply to the response",
    )
    quicktest_parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    quicktest_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens in response",
    )

    # =========================================================================
    # Benchmark command
    # =========================================================================
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run comprehensive benchmarks across models and probes",
        formatter_class=CustomFormatter,
    )
    benchmark_parser.add_argument(
        "--models",
        "-m",
        type=str,
        default="dummy",
        help="Comma-separated list of models to benchmark",
    )
    benchmark_parser.add_argument(
        "--probes",
        "-p",
        type=str,
        default="logic",
        help="Comma-separated list of probes to run",
    )
    benchmark_parser.add_argument(
        "--datasets",
        "-d",
        type=str,
        help="Comma-separated list of benchmark datasets (e.g., reasoning,math,coding)",
    )
    benchmark_parser.add_argument(
        "--max-examples",
        "-n",
        type=int,
        default=10,
        help="Maximum examples per dataset (default: 10)",
    )
    benchmark_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory for benchmark results",
    )
    benchmark_parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate an HTML report with visualizations",
    )
    benchmark_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress",
    )

    # =========================================================================
    # Compare command
    # =========================================================================
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple models on the same inputs",
        formatter_class=CustomFormatter,
    )
    compare_parser.add_argument(
        "--models",
        "-m",
        type=str,
        required=True,
        help="Comma-separated list of models to compare",
    )
    compare_parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Single input prompt to compare",
    )
    compare_parser.add_argument(
        "--input-file",
        type=str,
        help="File with inputs (one per line or JSON/JSONL)",
    )
    compare_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for comparison results",
    )
    compare_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json", "markdown"],
        default="table",
        help="Output format",
    )

    # =========================================================================
    # List command
    # =========================================================================
    list_parser = subparsers.add_parser(
        "list",
        help="List available models, probes, or datasets",
        formatter_class=CustomFormatter,
    )
    list_parser.add_argument(
        "type",
        choices=["models", "probes", "datasets", "trackers", "all"],
        help="What to list",
    )
    list_parser.add_argument(
        "--filter",
        type=str,
        help="Filter results by name (substring match)",
    )
    list_parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show detailed information",
    )

    # =========================================================================
    # Init command
    # =========================================================================
    init_parser = subparsers.add_parser(
        "init",
        help="Generate a sample configuration file",
        formatter_class=CustomFormatter,
    )
    init_parser.add_argument(
        "output",
        type=str,
        nargs="?",
        default="experiment.yaml",
        help="Output file path (default: experiment.yaml)",
    )
    init_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="dummy",
        help="Model type for the sample config",
    )
    init_parser.add_argument(
        "--probe",
        "-p",
        type=str,
        default="logic",
        help="Probe type for the sample config",
    )
    init_parser.add_argument(
        "--template",
        type=str,
        choices=["basic", "benchmark", "tracking", "full"],
        default="basic",
        help="Configuration template to use",
    )

    # =========================================================================
    # Info command
    # =========================================================================
    info_parser = subparsers.add_parser(
        "info",
        help="Show detailed information about a model, probe, or dataset",
        formatter_class=CustomFormatter,
    )
    info_parser.add_argument(
        "type",
        choices=["model", "probe", "dataset"],
        help="Type of item to show info for",
    )
    info_parser.add_argument(
        "name",
        type=str,
        help="Name of the model, probe, or dataset",
    )

    # =========================================================================
    # Interactive command
    # =========================================================================
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Start an interactive exploration session",
        formatter_class=CustomFormatter,
    )
    interactive_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="dummy",
        help="Model to use in interactive mode",
    )
    interactive_parser.add_argument(
        "--history-file",
        type=str,
        default=".insidellms_history",
        help="File to store command history",
    )

    # =========================================================================
    # Validate command
    # =========================================================================
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a configuration file or a run directory (manifest + records.jsonl)",
        formatter_class=CustomFormatter,
    )
    validate_parser.add_argument(
        "config",
        type=str,
        help="Path to a config file (.yaml/.json) OR a run_dir containing manifest.json",
    )
    validate_parser.add_argument(
        "--mode",
        choices=["strict", "warn"],
        default="strict",
        help="On schema mismatch for run_dir validation: strict=exit non-zero, warn=continue",
    )
    validate_parser.add_argument(
        "--schema-version",
        type=str,
        default=None,
        help="Override schema version when validating a run_dir (defaults to manifest schema_version)",
    )

    # =========================================================================
    # Export command
    # =========================================================================
    export_parser = subparsers.add_parser(
        "export",
        help="Export results to various formats",
        formatter_class=CustomFormatter,
    )
    export_parser.add_argument(
        "input",
        type=str,
        help="Input results file (JSON)",
    )
    export_parser.add_argument(
        "--format",
        "-f",
        choices=["csv", "markdown", "html", "latex"],
        default="csv",
        help="Export format",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path",
    )

    return parser


def _module_version(dist_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist_name)
    except Exception:
        return None


def _has_module(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _check_nltk_resource(path: str) -> bool:
    try:
        import nltk

        nltk.data.find(path)
        return True
    except Exception:
        return False


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the run command."""
    config_path = Path(args.config)

    if not config_path.exists():
        print_error(f"Config file not found: {config_path}")
        return 1

    print_header("Running Experiment")
    print_key_value("Config", config_path)

    # Create progress bar if verbose
    progress_bar: Optional[ProgressBar] = None

    def progress_callback(current: int, total: int) -> None:
        nonlocal progress_bar
        if args.verbose and not args.quiet:
            if progress_bar is None:
                progress_bar = ProgressBar(total, prefix="Evaluating")
            progress_bar.update(current)

    try:
        start_time = time.time()
        tracker = None

        # Resolve deterministic run artifact location (used for both runner and UX hints)
        env_run_root = os.environ.get("INSIDELLMS_RUN_ROOT")
        default_run_root = (
            Path(env_run_root).expanduser().absolute()
            if env_run_root
            else Path.home() / ".insidellms" / "runs"
        )
        if args.run_id:
            resolved_run_id = args.run_id
        else:
            resolved_run_id = derive_run_id_from_config_path(
                config_path,
                schema_version=args.schema_version,
            )
        # Use absolute paths to reduce surprise when users provide relative paths.
        effective_run_root = (
            Path(args.run_root).expanduser().absolute() if args.run_root else default_run_root
        )
        effective_run_dir = (
            Path(args.run_dir).expanduser().absolute()
            if args.run_dir
            else (effective_run_root / resolved_run_id)
        )

        if args.track:
            try:
                from insideLLMs.experiment_tracking import TrackingConfig, create_tracker

                tracking_root = effective_run_dir.parent / "tracking"

                tracker_kwargs: dict[str, Any] = {}
                if args.track == "local":
                    tracker_kwargs["output_dir"] = str(tracking_root)
                    tracker_kwargs["config"] = TrackingConfig(project=args.track_project)
                elif args.track == "wandb":
                    tracker_kwargs["project"] = args.track_project
                elif args.track == "mlflow":
                    tracker_kwargs["experiment_name"] = args.track_project
                elif args.track == "tensorboard":
                    tracker_kwargs["log_dir"] = str(
                        tracking_root / "tensorboard" / args.track_project
                    )
                    tracker_kwargs["config"] = TrackingConfig(project=args.track_project)

                tracker = create_tracker(args.track, **tracker_kwargs)
                tracker.start_run(run_name=resolved_run_id, run_id=resolved_run_id)
                tracker.log_params(
                    {
                        "run_id": resolved_run_id,
                        "run_dir": str(effective_run_dir),
                        "config_path": str(config_path),
                        "schema_version": args.schema_version,
                    }
                )
            except Exception as e:
                print_warning(f"Tracking disabled: {e}")
                tracker = None

        if args.use_async:
            print_info(f"Using async execution with concurrency={args.concurrency}")
            results = asyncio.run(
                run_experiment_from_config_async(
                    config_path,
                    concurrency=args.concurrency,
                    progress_callback=progress_callback if args.verbose else None,
                    validate_output=args.validate_output,
                    schema_version=args.schema_version,
                    validation_mode=args.validation_mode,
                    emit_run_artifacts=True,
                    run_dir=str(effective_run_dir) if args.run_dir else None,
                    run_root=str(effective_run_root),
                    run_id=resolved_run_id,
                    overwrite=bool(args.overwrite),
                    resume=bool(args.resume),
                )
            )
        else:
            results = run_experiment_from_config(
                config_path,
                progress_callback=progress_callback if args.verbose else None,
                validate_output=args.validate_output,
                schema_version=args.schema_version,
                validation_mode=args.validation_mode,
                emit_run_artifacts=True,
                run_dir=str(effective_run_dir) if args.run_dir else None,
                run_root=str(effective_run_root),
                run_id=resolved_run_id,
                overwrite=bool(args.overwrite),
                resume=bool(args.resume),
            )

        elapsed = time.time() - start_time

        if progress_bar:
            progress_bar.finish()

        experiment_result = results if isinstance(results, ExperimentResult) else None
        if experiment_result is not None:
            raw_results = [
                {
                    "input": r.input,
                    "output": r.output,
                    "error": r.error,
                    "status": r.status.value if isinstance(r.status, Enum) else str(r.status),
                    "latency_ms": r.latency_ms,
                    "metadata": r.metadata,
                }
                for r in experiment_result.results
            ]
            success_count = experiment_result.success_count
            error_count = experiment_result.error_count
            total = experiment_result.total_count
        else:
            raw_results = results
            success_count = sum(1 for r in results if r.get("status") == "success")
            error_count = sum(1 for r in results if r.get("status") == "error")
            total = len(results)

        # Output results
        if args.output:
            save_results_json(
                results,
                args.output,
                validate_output=args.validate_output,
                schema_version=args.schema_version,
                validation_mode=args.validation_mode,
            )
            print_success(f"Results saved to: {args.output}")

        if args.format == "json":
            print(json.dumps(results, indent=2, default=_json_default))
        elif args.format == "markdown":
            print(results_to_markdown(raw_results))
        elif args.format == "summary":
            # Minimal summary output
            print(
                f"OK {success_count}/{total} successful ({success_count / max(1, total) * 100:.1f}%)"
            )
        else:  # table format
            print_subheader("Results Summary")
            print_key_value("Total items", total)
            print_key_value(
                "Successful", f"{success_count} ({success_count / max(1, total) * 100:.1f}%)"
            )
            print_key_value("Errors", f"{error_count} ({error_count / max(1, total) * 100:.1f}%)")
            print_key_value("Duration", f"{elapsed:.2f}s")

            if raw_results:
                latencies = [
                    r.get("latency_ms", 0) for r in raw_results if r.get("status") == "success"
                ]
                if latencies:
                    avg_latency = sum(latencies) / len(latencies)
                    min_latency = min(latencies)
                    max_latency = max(latencies)
                    print_key_value("Avg latency", f"{avg_latency:.1f}ms")
                    print_key_value("Min/Max", f"{min_latency:.1f}ms / {max_latency:.1f}ms")

            # Show first few results
            print_subheader("Sample Results")
            for i, r in enumerate(raw_results[:5]):
                status_icon = (
                    colorize("OK", Colors.GREEN)
                    if r.get("status") == "success"
                    else colorize("FAIL", Colors.RED)
                )
                inp = str(r.get("input", ""))[:50]
                if len(str(r.get("input", ""))) > 50:
                    inp += "..."
                print(f"  {status_icon} [{i + 1}] {inp}")

            if len(raw_results) > 5:
                print(colorize(f"  ... and {len(raw_results) - 5} more", Colors.DIM))

        if tracker is not None:
            try:
                if experiment_result is not None:
                    tracker.log_experiment_result(experiment_result)

                metrics: dict[str, float] = {
                    "wall_time_seconds": float(elapsed),
                    "total_count": float(total),
                    "success_count": float(success_count),
                    "error_count": float(error_count),
                }

                latency_values = [
                    float(latency_ms)
                    for latency_ms in (
                        r.get("latency_ms") for r in raw_results if r.get("status") == "success"
                    )
                    if isinstance(latency_ms, (int, float))
                ]
                if latency_values:
                    metrics["avg_latency_ms"] = float(sum(latency_values) / len(latency_values))
                    metrics["min_latency_ms"] = float(min(latency_values))
                    metrics["max_latency_ms"] = float(max(latency_values))

                tracker.log_metrics(metrics)

                for artifact in (
                    effective_run_dir / "manifest.json",
                    effective_run_dir / "records.jsonl",
                    effective_run_dir / "config.resolved.yaml",
                    effective_run_dir / "summary.json",
                    effective_run_dir / "report.html",
                ):
                    if artifact.exists():
                        tracker.log_artifact(str(artifact), artifact_name=artifact.name)

                if args.output and Path(args.output).exists():
                    tracker.log_artifact(
                        str(Path(args.output)), artifact_name=Path(args.output).name
                    )

                tracker.end_run(status="finished")
                tracker = None
            except Exception as e:
                print_warning(f"Tracking error: {e}")

        # UX sugar: make it obvious where artifacts landed and how to validate.
        # Keep stdout JSON clean when --format json.
        if not args.quiet:
            hint_stream = sys.stderr if args.format == "json" else sys.stdout
            print(f"\nRun written to: {effective_run_dir}", file=hint_stream)
            print(f"Validate with: insidellms validate {effective_run_dir}", file=hint_stream)

        return 0

    except Exception as e:
        if "tracker" in locals() and tracker is not None:
            try:
                tracker.end_run(status="failed")
            except Exception:
                pass
        print_error(f"Error running experiment: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_harness(args: argparse.Namespace) -> int:
    """Execute the harness command."""
    config_path = Path(args.config)

    if not config_path.exists():
        print_error(f"Config file not found: {config_path}")
        return 1

    print_header("Running Behavioural Harness")
    print_key_value("Config", config_path)

    progress_bar: Optional[ProgressBar] = None

    def progress_callback(current: int, total: int) -> None:
        nonlocal progress_bar
        if not args.verbose or args.quiet:
            return
        if progress_bar is None:
            progress_bar = ProgressBar(total, prefix="Evaluating")
        progress_bar.update(current)

    tracker = None

    try:
        start_time = time.time()
        resolved_run_id = args.run_id or derive_run_id_from_config_path(
            config_path, schema_version=args.schema_version
        )

        # Precompute output_dir for tracking. Must match the precedence used later when emitting
        # standardized run artifacts.
        config = load_config(config_path)
        effective_run_root = Path(args.run_root).expanduser().absolute() if args.run_root else None
        config_default_dir = (
            Path(config.get("output_dir", "results")).expanduser().absolute()
            if isinstance(config, dict)
            else Path("results").expanduser().absolute()
        )
        if args.run_dir:
            output_dir = Path(args.run_dir).expanduser().absolute()
        elif args.output_dir:
            output_dir = Path(args.output_dir).expanduser().absolute()
        elif effective_run_root is not None:
            output_dir = effective_run_root / resolved_run_id
        else:
            output_dir = config_default_dir

        if args.track:
            try:
                from insideLLMs.experiment_tracking import TrackingConfig, create_tracker

                tracking_root = output_dir.parent / "tracking"

                tracker_kwargs: dict[str, Any] = {}
                if args.track == "local":
                    tracker_kwargs["output_dir"] = str(tracking_root)
                    tracker_kwargs["config"] = TrackingConfig(project=args.track_project)
                elif args.track == "wandb":
                    tracker_kwargs["project"] = args.track_project
                elif args.track == "mlflow":
                    tracker_kwargs["experiment_name"] = args.track_project
                elif args.track == "tensorboard":
                    tracker_kwargs["log_dir"] = str(
                        tracking_root / "tensorboard" / args.track_project
                    )
                    tracker_kwargs["config"] = TrackingConfig(project=args.track_project)

                tracker = create_tracker(args.track, **tracker_kwargs)
                tracker.start_run(run_name=resolved_run_id, run_id=resolved_run_id)
                tracker.log_params(
                    {
                        "run_id": resolved_run_id,
                        "run_dir": str(output_dir),
                        "config_path": str(config_path),
                        "schema_version": args.schema_version,
                    }
                )
            except Exception as e:
                print_warning(f"Tracking disabled: {e}")
                tracker = None

        result = run_harness_from_config(
            config_path,
            progress_callback=progress_callback if args.verbose else None,
            validate_output=args.validate_output,
            schema_version=args.schema_version,
            validation_mode=args.validation_mode,
        )
        elapsed = time.time() - start_time

        if progress_bar:
            progress_bar.finish()

        # -----------------------------------------------------------------
        # Determine harness run directory and emit standardized run artifacts
        # (manifest.json + records.jsonl + config.resolved.yaml + .insidellms_run)
        # -----------------------------------------------------------------
        from insideLLMs.runner import (
            _build_resolved_config_snapshot,
            _deterministic_base_time,
            _deterministic_run_id_from_config_snapshot,
            _deterministic_run_times,
            _prepare_run_dir,
            _serialize_value,
        )

        config_snapshot = _build_resolved_config_snapshot(result["config"], config_path.parent)

        if args.run_id:
            resolved_run_id = args.run_id
        else:
            resolved_run_id = result.get("run_id") or resolved_run_id
            if not resolved_run_id:
                resolved_run_id = _deterministic_run_id_from_config_snapshot(
                    config_snapshot,
                    schema_version=args.schema_version,
                )

        # Absolute paths reduce surprise when users pass relative paths.
        effective_run_root = Path(args.run_root).expanduser().absolute() if args.run_root else None
        config_default_dir = (
            Path(result["config"].get("output_dir", "results")).expanduser().absolute()
        )

        # Precedence: --run-dir > --output-dir (legacy alias) > --run-root/run-id > config output_dir
        if args.run_dir:
            output_dir = Path(args.run_dir).expanduser().absolute()
        elif args.output_dir:
            output_dir = Path(args.output_dir).expanduser().absolute()
        elif effective_run_root is not None:
            output_dir = effective_run_root / resolved_run_id
        else:
            output_dir = config_default_dir

        def _semver_tuple(version: str) -> tuple[int, int, int]:
            from insideLLMs.schemas.registry import semver_tuple

            return semver_tuple(version)

        def _atomic_write_text(path: Path, text: str) -> None:
            from insideLLMs.resources import atomic_write_text

            atomic_write_text(path, text)

        def _atomic_write_yaml(path: Path, data: Any) -> None:
            import yaml

            content = yaml.safe_dump(
                _serialize_value(data),
                sort_keys=True,
                default_flow_style=False,
                allow_unicode=True,
            )
            _atomic_write_text(path, content)

        def _ensure_run_sentinel(run_dir_path: Path) -> None:
            marker = run_dir_path / ".insidellms_run"
            if not marker.exists():
                try:
                    marker.write_text("insideLLMs run directory\n", encoding="utf-8")
                except Exception:
                    pass

        # Prepare run dir with the same safety policy as `insidellms run`.
        _prepare_run_dir(output_dir, overwrite=bool(args.overwrite), run_root=effective_run_root)
        _ensure_run_sentinel(output_dir)

        # Write resolved config snapshot for reproducibility.
        _atomic_write_yaml(output_dir / "config.resolved.yaml", config_snapshot)

        # Canonical record stream for validate/run-dir tooling.
        records_path = output_dir / "records.jsonl"

        # Ensure records' run_id matches the directory's run_id.
        for record in result.get("records", []):
            if isinstance(record, dict):
                record["run_id"] = resolved_run_id

        _write_jsonl(result["records"], records_path)
        print_success(f"Records written to: {records_path}")

        # Backward compatibility: keep results.jsonl alongside records.jsonl.
        legacy_results_path = output_dir / "results.jsonl"
        try:
            os.symlink("records.jsonl", legacy_results_path)
        except Exception:
            try:
                os.link(records_path, legacy_results_path)
            except Exception:
                shutil.copyfile(records_path, legacy_results_path)

        ds_cfg = result.get("config", {}).get("dataset")
        ds_cfg = ds_cfg if isinstance(ds_cfg, dict) else {}
        resolved_ds_cfg = config_snapshot.get("dataset")
        resolved_ds_cfg = resolved_ds_cfg if isinstance(resolved_ds_cfg, dict) else ds_cfg
        dataset_spec = {
            "dataset_id": resolved_ds_cfg.get("name")
            or resolved_ds_cfg.get("dataset")
            or resolved_ds_cfg.get("path")
            or resolved_ds_cfg.get("dataset_id"),
            "dataset_version": resolved_ds_cfg.get("version")
            or resolved_ds_cfg.get("split")
            or resolved_ds_cfg.get("dataset_version"),
            "dataset_hash": resolved_ds_cfg.get("hash") or resolved_ds_cfg.get("dataset_hash"),
            "provenance": resolved_ds_cfg.get("provenance")
            or resolved_ds_cfg.get("source")
            or resolved_ds_cfg.get("format"),
            "params": resolved_ds_cfg,
        }

        models_cfg = result.get("config", {}).get("models")
        models_cfg = models_cfg if isinstance(models_cfg, list) else []
        model_types = [m.get("type") for m in models_cfg if isinstance(m, dict) and m.get("type")]

        probes_cfg = result.get("config", {}).get("probes")
        probes_cfg = probes_cfg if isinstance(probes_cfg, list) else []
        probe_types = [p.get("type") for p in probes_cfg if isinstance(p, dict) and p.get("type")]

        record_count = len(result.get("records", []))
        success_count = sum(1 for r in result.get("records", []) if r.get("status") == "success")
        error_count = sum(1 for r in result.get("records", []) if r.get("status") == "error")

        run_base_time = _deterministic_base_time(resolved_run_id)
        started_at, completed_at = _deterministic_run_times(run_base_time, record_count)
        created_at = started_at

        manifest: dict[str, Any] = {
            "schema_version": args.schema_version,
            "run_id": resolved_run_id,
            "created_at": created_at,
            "started_at": started_at,
            "completed_at": completed_at,
            "library_version": None,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "command": None,
            "model": {
                "model_id": "harness",
                "provider": "insideLLMs",
                "params": {"model_count": len(model_types), "models": model_types},
            },
            "probe": {
                "probe_id": "harness",
                "probe_version": None,
                "params": {"probe_count": len(probe_types), "probes": probe_types},
            },
            "dataset": dataset_spec,
            "record_count": record_count,
            "success_count": success_count,
            "error_count": error_count,
            "records_file": "records.jsonl",
            "schemas": {"RunManifest": args.schema_version, "ResultRecord": args.schema_version},
            "custom": {
                "harness": {
                    "models": models_cfg,
                    "probes": probes_cfg,
                    "dataset": resolved_ds_cfg,
                    "max_examples": result.get("config", {}).get("max_examples"),
                    "experiment_count": len(result.get("experiments", [])),
                    "legacy_results_file": "results.jsonl",
                }
            },
        }

        if _semver_tuple(args.schema_version) >= (1, 0, 1):
            manifest["run_completed"] = True

        try:
            import insideLLMs

            manifest["library_version"] = getattr(insideLLMs, "__version__", None)
        except Exception:
            pass

        if args.validate_output:
            from insideLLMs.schemas import OutputValidator, SchemaRegistry

            validator = OutputValidator(SchemaRegistry())
            validator.validate(
                SchemaRegistry.RUN_MANIFEST,
                manifest,
                schema_version=args.schema_version,
                mode=args.validation_mode,
            )

        manifest_path = output_dir / "manifest.json"
        _atomic_write_text(
            manifest_path,
            json.dumps(
                _serialize_value(manifest),
                sort_keys=True,
                indent=2,
                default=_serialize_value,
            ),
        )
        print_success(f"Manifest written to: {manifest_path}")

        summary_path = output_dir / "summary.json"
        report_path = output_dir / "report.html"

        summary_payload = {
            "schema_version": args.schema_version,
            "generated_at": created_at,
            "summary": result["summary"],
            "config": result["config"],
        }

        if args.validate_output:
            from insideLLMs.schemas import OutputValidator, SchemaRegistry

            validator = OutputValidator(SchemaRegistry())
            validator.validate(
                SchemaRegistry.HARNESS_SUMMARY,
                summary_payload,
                schema_version=args.schema_version,
                mode=args.validation_mode,
            )
        with open(summary_path, "w") as f:
            json.dump(summary_payload, f, indent=2, default=_json_default, sort_keys=True)
        print_success(f"Summary written to: {summary_path}")

        if not args.skip_report:
            report_title = args.report_title or result["config"].get(
                "report_title", "Behavioural Probe Report"
            )
            try:
                from insideLLMs.visualization import create_interactive_html_report

                create_interactive_html_report(
                    result["experiments"],
                    title=report_title,
                    save_path=str(report_path),
                    generated_at=created_at,
                )
            except ImportError:
                report_html = _build_basic_harness_report(
                    result["experiments"],
                    result["summary"],
                    report_title,
                    generated_at=created_at,
                )
                with open(report_path, "w") as f:
                    f.write(report_html)

            print_success(f"Report written to: {report_path}")

        print_key_value("Elapsed", f"{elapsed:.2f}s")

        if tracker is not None:
            try:
                metrics: dict[str, float] = {
                    "wall_time_seconds": float(elapsed),
                    "record_count": float(record_count),
                    "success_count": float(success_count),
                    "error_count": float(error_count),
                    "experiment_count": float(len(result.get("experiments", []))),
                }
                tracker.log_metrics(metrics)

                tracker.log_params(
                    {
                        "model_count": len(model_types),
                        "probe_count": len(probe_types),
                        "dataset_id": dataset_spec.get("dataset_id"),
                        "dataset_version": dataset_spec.get("dataset_version"),
                        "dataset_hash": dataset_spec.get("dataset_hash"),
                        "dataset_provenance": dataset_spec.get("provenance"),
                    }
                )

                for artifact in (
                    output_dir / "manifest.json",
                    output_dir / "records.jsonl",
                    output_dir / "config.resolved.yaml",
                    output_dir / "summary.json",
                    output_dir / "report.html",
                ):
                    if artifact.exists():
                        tracker.log_artifact(str(artifact), artifact_name=artifact.name)

                tracker.end_run(status="finished")
                tracker = None
            except Exception as e:
                print_warning(f"Tracking error: {e}")

        if not args.quiet:
            print(f"\nRun written to: {output_dir}")
            print(f"Validate with: insidellms validate {output_dir}")
        return 0

    except Exception as e:
        if tracker is not None:
            try:
                tracker.end_run(status="failed")
            except Exception:
                pass
        print_error(f"Error running harness: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_schema(args: argparse.Namespace) -> int:
    """Inspect/dump/validate versioned output schemas."""

    from insideLLMs.schemas import OutputValidationError, OutputValidator, SchemaRegistry

    registry = SchemaRegistry()

    op = getattr(args, "op", None) or "list"
    name = getattr(args, "name", None)
    version = getattr(args, "version", DEFAULT_SCHEMA_VERSION)

    # Shortcut UX: `insidellms schema <SchemaName>` -> dump schema
    if op not in {"list", "dump", "validate"}:
        name = name or op
        op = "dump"

    if op == "list":
        print_header("Available Output Schemas")
        schema_names = [
            registry.RUNNER_ITEM,
            registry.RUNNER_OUTPUT,
            registry.RESULT_RECORD,
            registry.RUN_MANIFEST,
            registry.HARNESS_RECORD,
            registry.HARNESS_SUMMARY,
            registry.BENCHMARK_SUMMARY,
            registry.COMPARISON_REPORT,
            registry.DIFF_REPORT,
            registry.EXPORT_METADATA,
            registry.CUSTOM_TRACE,
        ]
        for name in schema_names:
            versions = registry.available_versions(name)
            if not versions:
                continue
            print_key_value(name, ", ".join(versions))
        return 0

    if op == "dump":
        if not name:
            print_error(
                "Missing schema name. Use: insidellms schema dump --name <SchemaName> [--version X]"
            )
            return 1
        try:
            schema = registry.get_json_schema(name, version)
        except Exception as e:
            print_error(f"Could not dump schema {name}@{version}: {e}")
            return 2

        payload = json.dumps(schema, indent=2, default=_json_default)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(payload)
            print_success(f"Schema written to: {out_path}")
        else:
            print(payload)
        return 0

    if op == "validate":
        if not name:
            print_error(
                "Missing schema name. Use: insidellms schema validate --name <SchemaName> -i <file>"
            )
            return 1
        if not getattr(args, "input", None):
            print_error("Missing --input for schema validate")
            return 1

        in_path = Path(args.input)
        if not in_path.exists():
            print_error(f"Input file not found: {in_path}")
            return 1

        validator = OutputValidator(registry)
        errors = 0

        def validate_one(obj: Any) -> None:
            nonlocal errors
            try:
                validator.validate(
                    name,
                    obj,
                    schema_version=version,
                    mode="strict",
                )
            except OutputValidationError as e:
                errors += 1
                if args.mode == "warn":
                    print_warning(str(e))
                else:
                    raise

        try:
            if in_path.suffix.lower() == ".jsonl":
                with open(in_path) as f:
                    for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError as e:
                            errors += 1
                            if args.mode == "warn":
                                print_warning(f"Invalid JSON on line {line_no}: {e}")
                                continue
                            raise
                        validate_one(obj)
            else:
                obj = json.loads(in_path.read_text())
                if isinstance(obj, list):
                    for item in obj:
                        validate_one(item)
                else:
                    validate_one(obj)
        except Exception as e:
            if args.mode == "warn":
                print_warning(f"Validation completed with errors: {e}")
            else:
                print_error(f"Validation failed: {e}")
                return 1

        if errors:
            if args.mode == "warn":
                print_warning(f"Validated with {errors} error(s) (warn mode)")
                return 0
            print_error(f"Validation failed with {errors} error(s)")
            return 1

        print_success("Validation OK")
        return 0

    print_error(f"Unknown schema op: {op}")
    return 1


def cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnose environment and optional dependencies."""
    checks: list[dict[str, Any]] = []

    def add_check(*, name: str, ok: bool, hint: Optional[str] = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "hint": hint})

    add_check(name="python", ok=True, hint=sys.version.split()[0])
    add_check(name="platform", ok=True, hint=platform.platform())
    add_check(name="insideLLMs", ok=True, hint=_module_version("insideLLMs"))

    # Optional validation/schema tooling
    add_check(name="pydantic", ok=_has_module("pydantic"), hint='pip install ".[dev]"')

    # NLP extras
    add_check(name="nltk", ok=_has_module("nltk"), hint='pip install ".[nlp]"')
    add_check(name="sklearn", ok=_has_module("sklearn"), hint='pip install ".[nlp]"')
    add_check(name="spacy", ok=_has_module("spacy"), hint='pip install ".[nlp]"')
    add_check(name="gensim", ok=_has_module("gensim"), hint='pip install ".[nlp]"')
    add_check(
        name="nltk:punkt",
        ok=_check_nltk_resource("tokenizers/punkt"),
        hint="python -m nltk.downloader punkt",
    )
    add_check(
        name="nltk:vader_lexicon",
        ok=_check_nltk_resource("sentiment/vader_lexicon.zip")
        or _check_nltk_resource("sentiment/vader_lexicon"),
        hint="python -m nltk.downloader vader_lexicon",
    )
    add_check(
        name="spacy:en_core_web_sm",
        ok=_has_module("en_core_web_sm"),
        hint="python -m spacy download en_core_web_sm",
    )

    # Visualization extras
    add_check(
        name="matplotlib", ok=_has_module("matplotlib"), hint='pip install ".[visualization]"'
    )
    add_check(name="pandas", ok=_has_module("pandas"), hint='pip install ".[visualization]"')
    add_check(name="seaborn", ok=_has_module("seaborn"), hint='pip install ".[visualization]"')
    add_check(name="plotly", ok=_has_module("plotly"), hint='pip install ".[visualization]"')
    add_check(name="ipywidgets", ok=_has_module("ipywidgets"), hint="pip install ipywidgets")

    # Optional integrations
    add_check(name="redis", ok=_has_module("redis"), hint="pip install redis")
    add_check(name="datasets", ok=_has_module("datasets"), hint="pip install datasets")

    # API keys (informational)
    add_check(name="OPENAI_API_KEY", ok=bool(os.environ.get("OPENAI_API_KEY")), hint="set env var")
    add_check(
        name="ANTHROPIC_API_KEY", ok=bool(os.environ.get("ANTHROPIC_API_KEY")), hint="set env var"
    )

    run_root = os.environ.get("INSIDELLMS_RUN_ROOT")
    add_check(
        name="INSIDELLMS_RUN_ROOT",
        ok=bool(run_root),
        hint="(optional) override run artifacts root",
    )

    warn_checks = [
        c
        for c in checks
        if c["name"]
        not in {
            "python",
            "platform",
            "insideLLMs",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "INSIDELLMS_RUN_ROOT",
        }
        and not c["ok"]
    ]

    if args.format == "json":
        payload = {"checks": checks, "warnings": warn_checks}
        print(json.dumps(payload, indent=2, default=_json_default))
        return 1 if (args.fail_on_warn and warn_checks) else 0

    print_header("insideLLMs Doctor")
    print_subheader("Environment")
    for item in checks[:3]:
        print_key_value(item["name"], item["hint"] or "-")

    print_subheader("Diagnostics")
    for item in checks[3:]:
        if item["ok"]:
            print_success(item["name"])
        else:
            hint = f" ({item['hint']})" if item.get("hint") else ""
            print_warning(f"{item['name']}{hint}")

    if warn_checks:
        print()
        print_warning(f"{len(warn_checks)} recommended checks failed (optional deps missing).")
    else:
        print()
        print_success("All recommended checks passed.")

    return 1 if (args.fail_on_warn and warn_checks) else 0


def cmd_report(args: argparse.Namespace) -> int:
    """Rebuild summary.json and report.html from records.jsonl."""
    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        print_error(f"Run directory not found: {run_dir}")
        return 1

    records_path = run_dir / "records.jsonl"
    if not records_path.exists():
        print_error(f"records.jsonl not found in: {run_dir}")
        return 1

    try:
        records = _read_jsonl_records(records_path)
    except Exception as e:
        print_error(f"Could not read records.jsonl: {e}")
        return 1

    if not records:
        print_error("No records found in records.jsonl")
        return 1

    experiments, derived_config, schema_version = _build_experiments_from_records(records)
    if not experiments:
        print_error("No experiments could be reconstructed from records")
        return 1

    run_ids = {record.get("run_id") for record in records if record.get("run_id")}
    run_id = sorted(run_ids)[0] if run_ids else None
    if run_id and len(run_ids) > 1:
        print_warning(f"Multiple run_ids found; using {run_id}")

    generated_at = None
    if run_id:
        try:
            from insideLLMs.runner import _deterministic_base_time, _deterministic_run_times

            base_time = _deterministic_base_time(str(run_id))
            _, generated_at = _deterministic_run_times(base_time, len(records))
        except Exception:
            generated_at = None

    if generated_at is None:
        generated_at = max(
            (dt for dt in (_parse_datetime(r.get("completed_at")) for r in records) if dt),
            default=None,
        )

    summary = generate_summary_report(experiments, include_ci=True)
    summary_payload = {
        "schema_version": schema_version,
        "generated_at": generated_at,
        "summary": summary,
        "config": derived_config,
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2, default=_json_default, sort_keys=True)
    print_success(f"Summary written to: {summary_path}")

    report_title = args.report_title or "Behavioural Probe Report"
    report_path = run_dir / "report.html"
    try:
        from insideLLMs.visualization import create_interactive_html_report

        create_interactive_html_report(
            experiments,
            title=report_title,
            save_path=str(report_path),
            generated_at=generated_at,
        )
    except ImportError:
        report_html = _build_basic_harness_report(
            experiments,
            summary,
            report_title,
            generated_at=generated_at,
        )
        with open(report_path, "w") as f:
            f.write(report_html)

    print_success(f"Report written to: {report_path}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Compare two run directories and report behavioural regressions."""
    run_dir_a = Path(args.run_dir_a)
    run_dir_b = Path(args.run_dir_b)

    if not run_dir_a.exists() or not run_dir_a.is_dir():
        print_error(f"Run directory not found: {run_dir_a}")
        return 1
    if not run_dir_b.exists() or not run_dir_b.is_dir():
        print_error(f"Run directory not found: {run_dir_b}")
        return 1

    records_path_a = run_dir_a / "records.jsonl"
    records_path_b = run_dir_b / "records.jsonl"

    if not records_path_a.exists():
        print_error(f"records.jsonl not found in: {run_dir_a}")
        return 1
    if not records_path_b.exists():
        print_error(f"records.jsonl not found in: {run_dir_b}")
        return 1

    try:
        records_a = _read_jsonl_records(records_path_a)
        records_b = _read_jsonl_records(records_path_b)
    except Exception as e:
        print_error(f"Could not read records.jsonl: {e}")
        return 1

    if not records_a or not records_b:
        print_error("Both run directories must contain records to compare")
        return 1

    output_format = getattr(args, "format", "text")
    output_path = getattr(args, "output", None)
    if output_path and output_format != "json":
        print_warning("--output is only used with --format json")

    ignore_keys: set[str] = set()
    for entry in args.output_fingerprint_ignore or []:
        for item in str(entry).split(","):
            item = item.strip()
            if item:
                ignore_keys.add(item.lower())

    ignore_keys_set = ignore_keys if ignore_keys else None

    def build_index(
        records: list[dict[str, Any]],
    ) -> tuple[dict[tuple[str, str, str], dict[str, Any]], int]:
        index: dict[tuple[str, str, str], dict[str, Any]] = {}
        duplicates = 0
        for record in records:
            key = _record_key(record)
            if key in index:
                duplicates += 1
                continue
            index[key] = record
        return index, duplicates

    def record_identity(record: dict[str, Any]) -> dict[str, Any]:
        custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
        label = _record_label(record)
        key = _record_key(record)
        return {
            "record_key": {"model_id": key[0], "probe_id": key[1], "item_id": key[2]},
            "model_id": key[0],
            "probe_id": key[1],
            "example_id": label[2],
            "replicate_key": custom.get("replicate_key"),
            "label": {"model": label[0], "probe": label[1], "example": label[2]},
        }

    def record_summary(record: dict[str, Any]) -> dict[str, Any]:
        scores = record.get("scores") if isinstance(record.get("scores"), dict) else None
        _metric_name, metric_value = _primary_score(record)
        return {
            "status": _status_string(record),
            "primary_metric": record.get("primary_metric"),
            "primary_score": metric_value,
            "scores_keys": sorted(scores.keys()) if isinstance(scores, dict) else None,
            "output": _output_summary(record, ignore_keys_set),
        }

    index_a, dup_a = build_index(records_a)
    index_b, dup_b = build_index(records_b)

    if dup_a:
        print_warning(f"Baseline has {dup_a} duplicate key(s); first occurrence used")
    if dup_b:
        print_warning(f"Comparison has {dup_b} duplicate key(s); first occurrence used")

    all_keys = set(index_a) | set(index_b)
    regressions: list[tuple[str, str, str, str]] = []
    improvements: list[tuple[str, str, str, str]] = []
    changes: list[tuple[str, str, str, str]] = []
    only_a: list[tuple[str, str, str]] = []
    only_b: list[tuple[str, str, str]] = []

    regressions_json: list[dict[str, Any]] = []
    improvements_json: list[dict[str, Any]] = []
    changes_json: list[dict[str, Any]] = []
    only_a_json: list[dict[str, Any]] = []
    only_b_json: list[dict[str, Any]] = []
    # Trace tracking
    trace_drifts: list[tuple[str, str, str, str]] = []
    trace_drifts_json: list[dict[str, Any]] = []
    trace_violation_increases: list[tuple[str, str, str, str]] = []
    trace_violation_increases_json: list[dict[str, Any]] = []

    for key in sorted(all_keys):
        record_a = index_a.get(key)
        record_b = index_b.get(key)

        if record_a is None:
            only_b.append(_record_label(record_b))  # type: ignore[arg-type]
            only_b_json.append(record_identity(record_b))  # type: ignore[arg-type]
            continue
        if record_b is None:
            only_a.append(_record_label(record_a))
            only_a_json.append(record_identity(record_a))
            continue

        label = _record_label(record_a)
        status_a = _status_string(record_a)
        status_b = _status_string(record_b)
        score_name_a, score_a = _primary_score(record_a)
        score_name_b, score_b = _primary_score(record_b)
        identity = record_identity(record_a)
        summary_a = record_summary(record_a)
        summary_b = record_summary(record_b)

        if status_a == "success" and status_b != "success":
            regressions.append((*label, f"status {status_a} -> {status_b}"))
            regressions_json.append(
                {
                    **identity,
                    "kind": "status_regression",
                    "detail": f"status {status_a} -> {status_b}",
                    "baseline": summary_a,
                    "candidate": summary_b,
                }
            )
            continue
        if status_a != "success" and status_b == "success":
            improvements.append((*label, f"status {status_a} -> {status_b}"))
            improvements_json.append(
                {
                    **identity,
                    "kind": "status_improvement",
                    "detail": f"status {status_a} -> {status_b}",
                    "baseline": summary_a,
                    "candidate": summary_b,
                }
            )
            continue

        metrics_compared = False
        if score_a is not None and score_b is not None and score_name_a == score_name_b:
            delta = score_b - score_a
            metrics_compared = True
            if delta < 0:
                regressions.append(
                    (*label, f"{score_name_a} {score_a:.4f} -> {score_b:.4f} (delta {delta:.4f})")
                )
                regressions_json.append(
                    {
                        **identity,
                        "kind": "metric_regression",
                        "metric": score_name_a,
                        "baseline": summary_a,
                        "candidate": summary_b,
                        "delta": delta,
                    }
                )
                continue
            if delta > 0:
                improvements.append(
                    (*label, f"{score_name_a} {score_a:.4f} -> {score_b:.4f} (delta +{delta:.4f})")
                )
                improvements_json.append(
                    {
                        **identity,
                        "kind": "metric_improvement",
                        "metric": score_name_a,
                        "baseline": summary_a,
                        "candidate": summary_b,
                        "delta": delta,
                    }
                )

        if not metrics_compared and (score_a is not None or score_b is not None):
            reason = _metric_mismatch_reason(record_a, record_b) or "type_mismatch"
            detail = _metric_mismatch_details(record_a, record_b)
            changes.append((*label, f"metrics not comparable:{reason}; {detail}"))
            changes_json.append(
                {
                    **identity,
                    "kind": "metrics_not_comparable",
                    "reason": reason,
                    "baseline": summary_a,
                    "candidate": summary_b,
                    "details": _metric_mismatch_context(record_a, record_b),
                }
            )

        if metrics_compared and status_a == "success" and status_b == "success":
            scores_a = record_a.get("scores") if isinstance(record_a.get("scores"), dict) else None
            scores_b = record_b.get("scores") if isinstance(record_b.get("scores"), dict) else None
            if isinstance(scores_a, dict) and isinstance(scores_b, dict):
                keys_a = sorted(scores_a.keys())
                keys_b = sorted(scores_b.keys())
                missing_in_b = sorted(set(keys_a) - set(keys_b))
                missing_in_a = sorted(set(keys_b) - set(keys_a))
                if missing_in_a or missing_in_b:
                    changes.append(
                        (
                            *label,
                            "metric_key_missing: "
                            f"baseline_missing={missing_in_a}, candidate_missing={missing_in_b}",
                        )
                    )
                    changes_json.append(
                        {
                            **identity,
                            "kind": "metric_key_missing",
                            "baseline_missing": missing_in_a,
                            "candidate_missing": missing_in_b,
                            "baseline": summary_a,
                            "candidate": summary_b,
                        }
                    )

        output_a = _output_text(record_a)
        output_b = _output_text(record_b)
        if output_a is not None or output_b is not None:
            if output_a != output_b:
                changes.append((*label, "output changed"))
                changes_json.append(
                    {
                        **identity,
                        "kind": "output_changed",
                        "baseline": summary_a,
                        "candidate": summary_b,
                    }
                )
        else:
            fingerprint_a = _output_fingerprint(record_a, ignore_keys=ignore_keys_set)
            fingerprint_b = _output_fingerprint(record_b, ignore_keys=ignore_keys_set)
            if fingerprint_a != fingerprint_b:
                if fingerprint_a and fingerprint_b:
                    changes.append(
                        (*label, f"output fingerprint {fingerprint_a} -> {fingerprint_b}")
                    )
                else:
                    changes.append((*label, "output changed (structured)"))
                changes_json.append(
                    {
                        **identity,
                        "kind": "output_changed",
                        "baseline": summary_a,
                        "candidate": summary_b,
                        "baseline_fingerprint": fingerprint_a,
                        "candidate_fingerprint": fingerprint_b,
                    }
                )
        if status_a != status_b:
            changes.append((*label, f"status {status_a} -> {status_b}"))
            changes_json.append(
                {
                    **identity,
                    "kind": "status_changed",
                    "detail": f"status {status_a} -> {status_b}",
                    "baseline": summary_a,
                    "candidate": summary_b,
                }
            )

        # Trace drift detection
        trace_fp_a = _trace_fingerprint(record_a)
        trace_fp_b = _trace_fingerprint(record_b)
        if trace_fp_a and trace_fp_b and trace_fp_a != trace_fp_b:
            trace_drifts.append((*label, f"trace {trace_fp_a[:12]} -> {trace_fp_b[:12]}"))
            trace_drifts_json.append(
                {
                    **identity,
                    "kind": "trace_drift",
                    "baseline_trace_fingerprint": trace_fp_a,
                    "candidate_trace_fingerprint": trace_fp_b,
                }
            )

        # Trace violation comparison
        violations_a = _trace_violation_count(record_a)
        violations_b = _trace_violation_count(record_b)
        if violations_b > violations_a:
            trace_violation_increases.append(
                (*label, f"violations {violations_a} -> {violations_b}")
            )
            trace_violation_increases_json.append(
                {
                    **identity,
                    "kind": "trace_violations_increased",
                    "baseline_violations": violations_a,
                    "candidate_violations": violations_b,
                    "candidate_violation_details": _trace_violations(record_b),
                }
            )

    diff_report = {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "baseline": str(run_dir_a),
        "candidate": str(run_dir_b),
        "run_ids": {
            "baseline": sorted(
                {record.get("run_id") for record in records_a if record.get("run_id")}
            ),
            "candidate": sorted(
                {record.get("run_id") for record in records_b if record.get("run_id")}
            ),
        },
        "counts": {
            "common": len(all_keys) - len(only_a) - len(only_b),
            "only_baseline": len(only_a),
            "only_candidate": len(only_b),
            "regressions": len(regressions),
            "improvements": len(improvements),
            "other_changes": len(changes),
            "trace_drifts": len(trace_drifts),
            "trace_violation_increases": len(trace_violation_increases),
        },
        "duplicates": {"baseline": dup_a, "candidate": dup_b},
        "regressions": regressions_json,
        "improvements": improvements_json,
        "changes": changes_json,
        "only_baseline": only_a_json,
        "only_candidate": only_b_json,
        "trace_drifts": trace_drifts_json,
        "trace_violation_increases": trace_violation_increases_json,
    }

    if output_format == "json":
        payload = json.dumps(diff_report, indent=2, default=_json_default)
        if output_path:
            Path(output_path).write_text(payload, encoding="utf-8")
        else:
            print(payload)
        if args.fail_on_regressions and regressions:
            return 2
        if args.fail_on_changes and (regressions or changes or only_a or only_b):
            return 2
        if args.fail_on_trace_violations and trace_violation_increases:
            return 3
        if args.fail_on_trace_drift and trace_drifts:
            return 4
        return 0

    print_header("Behavioural Diff")
    print_key_value("Baseline", run_dir_a)
    print_key_value("Comparison", run_dir_b)
    print_key_value("Common keys", diff_report["counts"]["common"])
    print_key_value("Only in baseline", diff_report["counts"]["only_baseline"])
    print_key_value("Only in comparison", diff_report["counts"]["only_candidate"])
    print_key_value("Regressions", diff_report["counts"]["regressions"])
    print_key_value("Improvements", diff_report["counts"]["improvements"])
    print_key_value("Other changes", diff_report["counts"]["other_changes"])
    if trace_drifts:
        print_key_value("Trace drifts", diff_report["counts"]["trace_drifts"])
    if trace_violation_increases:
        print_key_value(
            "Trace violation increases", diff_report["counts"]["trace_violation_increases"]
        )

    def print_section(title: str, items: list[tuple[str, str, str, str]]) -> None:
        if not items:
            return
        print_subheader(title)
        for model_label, probe_label, example_id, detail in items[: args.limit]:
            print(f"  {model_label} | {probe_label} | example {example_id}: {detail}")
        if len(items) > args.limit:
            print(colorize(f"  ... and {len(items) - args.limit} more", Colors.DIM))

    print_section("Regressions", regressions)
    print_section("Improvements", improvements)
    print_section("Other Changes", changes)
    print_section("Trace Drifts", trace_drifts)
    print_section("Trace Violation Increases", trace_violation_increases)

    if only_a:
        print_subheader("Missing in Comparison")
        for model_label, probe_label, example_id in only_a[: args.limit]:
            print(f"  {model_label} | {probe_label} | example {example_id}")
        if len(only_a) > args.limit:
            print(colorize(f"  ... and {len(only_a) - args.limit} more", Colors.DIM))

    if only_b:
        print_subheader("New in Comparison")
        for model_label, probe_label, example_id in only_b[: args.limit]:
            print(f"  {model_label} | {probe_label} | example {example_id}")
        if len(only_b) > args.limit:
            print(colorize(f"  ... and {len(only_b) - args.limit} more", Colors.DIM))

    if args.fail_on_regressions and regressions:
        return 2
    if args.fail_on_changes and (regressions or changes or only_a or only_b):
        return 2
    if args.fail_on_trace_violations and trace_violation_increases:
        return 3
    if args.fail_on_trace_drift and trace_drifts:
        return 4
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """Execute the list command."""
    ensure_builtins_registered()

    filter_str = args.filter.lower() if args.filter else None

    if args.type in ("models", "all"):
        print_subheader("Available Models")
        models = model_registry.list()
        if filter_str:
            models = [m for m in models if filter_str in m.lower()]

        for name in sorted(models):
            info = model_registry.info(name)
            doc = info.get("doc", "").split("\n")[0] if info.get("doc") else ""
            if args.detailed:
                print(f"\n  {colorize(name, Colors.BOLD, Colors.CYAN)}")
                print(f"    {colorize('Description:', Colors.DIM)} {doc[:70]}")
                if info.get("default_kwargs"):
                    print(f"    {colorize('Defaults:', Colors.DIM)} {info['default_kwargs']}")
            else:
                print(f"  {colorize(name, Colors.CYAN):25} {doc[:50]}")

        print(f"\n  {colorize(f'Total: {len(models)} models', Colors.DIM)}")

    if args.type in ("probes", "all"):
        print_subheader("Available Probes")
        probes = probe_registry.list()
        if filter_str:
            probes = [p for p in probes if filter_str in p.lower()]

        for name in sorted(probes):
            info = probe_registry.info(name)
            doc = info.get("doc", "").split("\n")[0] if info.get("doc") else ""
            if args.detailed:
                print(f"\n  {colorize(name, Colors.BOLD, Colors.GREEN)}")
                print(f"    {colorize('Description:', Colors.DIM)} {doc[:70]}")
            else:
                print(f"  {colorize(name, Colors.GREEN):25} {doc[:50]}")

        print(f"\n  {colorize(f'Total: {len(probes)} probes', Colors.DIM)}")

    if args.type in ("datasets", "all"):
        print_subheader("Built-in Benchmark Datasets")
        try:
            from insideLLMs.benchmark_datasets import list_builtin_datasets

            datasets = list_builtin_datasets()

            if filter_str:
                datasets = [d for d in datasets if filter_str in d["name"].lower()]

            for ds in datasets:
                if args.detailed:
                    print(f"\n  {colorize(ds['name'], Colors.BOLD, Colors.YELLOW)}")
                    print(f"    {colorize('Category:', Colors.DIM)} {ds['category']}")
                    print(f"    {colorize('Examples:', Colors.DIM)} {ds['num_examples']}")
                    print(f"    {colorize('Description:', Colors.DIM)} {ds['description'][:60]}")
                    print(
                        f"    {colorize('Difficulties:', Colors.DIM)} {', '.join(ds['difficulties'])}"
                    )
                else:
                    print(
                        f"  {colorize(ds['name'], Colors.YELLOW):15} {ds['num_examples']:4} examples  [{ds['category']}]"
                    )

            print(f"\n  {colorize(f'Total: {len(datasets)} datasets', Colors.DIM)}")
        except ImportError:
            print_warning("Benchmark datasets module not available")

    if args.type in ("trackers", "all"):
        print_subheader("Experiment Tracking Backends")
        trackers = [
            ("local", "Local file-based tracking (always available)"),
            ("wandb", "Weights & Biases (requires: pip install wandb)"),
            ("mlflow", "MLflow tracking (requires: pip install mlflow)"),
            ("tensorboard", "TensorBoard (requires: pip install tensorboard)"),
        ]
        for name, desc in trackers:
            print(f"  {colorize(name, Colors.MAGENTA):15} {desc}")

    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Execute the init command."""
    import yaml

    print_header("Initialize Experiment Configuration")

    # Base configuration
    config: dict[str, Any] = {
        "model": {
            "type": args.model,
            "args": {},
        },
        "probe": {
            "type": args.probe,
            "args": {},
        },
        "dataset": {
            "format": "jsonl",
            "path": "data/questions.jsonl",
        },
    }

    # Add model-specific args hints
    model_hints = {
        "openai": {"model_name": "gpt-4"},
        "anthropic": {"model_name": "claude-3-opus-20240229"},
        "cohere": {"model_name": "command-r-plus"},
        "gemini": {"model_name": "gemini-pro"},
        "huggingface": {"model_name": "gpt2"},
        "ollama": {"model_name": "llama2"},
    }
    if args.model in model_hints:
        config["model"]["args"] = model_hints[args.model]

    # Apply template enhancements
    if args.template == "benchmark":
        config["benchmark"] = {
            "datasets": ["reasoning", "math", "coding"],
            "max_examples_per_dataset": 10,
        }
    elif args.template == "tracking":
        config["tracking"] = {
            "backend": "local",
            "project": "my-experiment",
            "log_dir": "./experiments",
        }
    elif args.template == "full":
        config["benchmark"] = {
            "datasets": ["reasoning", "math", "coding", "safety"],
            "max_examples_per_dataset": 20,
        }
        config["tracking"] = {
            "backend": "local",
            "project": "my-experiment",
            "log_dir": "./experiments",
        }
        config["async"] = {
            "enabled": True,
            "concurrency": 5,
        }
        config["output"] = {
            "format": "json",
            "path": "results/experiment_results.json",
            "html_report": True,
        }

    output_path = Path(args.output)

    if output_path.suffix in (".yaml", ".yml"):
        content = yaml.dump(config, default_flow_style=False, sort_keys=False)
    else:
        content = json.dumps(config, indent=2)

    output_path.write_text(content)
    print_success(f"Created config: {output_path}")
    print_key_value("Template", args.template)
    print_key_value("Model", args.model)
    print_key_value("Probe", args.probe)

    # Create sample data directory and file
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    sample_data_path = data_dir / "questions.jsonl"
    if not sample_data_path.exists():
        sample_data = [
            {"question": "What is 2+2?", "reference_answer": "4"},
            {"question": "What is the capital of France?", "reference_answer": "Paris"},
            {"question": "Who wrote Romeo and Juliet?", "reference_answer": "William Shakespeare"},
            {
                "question": "If all cats are mammals, and all mammals are animals, are all cats animals?",
                "reference_answer": "Yes",
            },
            {"question": "What is the chemical symbol for water?", "reference_answer": "H2O"},
        ]
        with open(sample_data_path, "w") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")
        print_success(f"Created sample data: {sample_data_path}")

    print()
    print_info("Next steps:")
    print(f"  1. Edit {colorize(str(output_path), Colors.CYAN)} to customize your experiment")
    print(f"  2. Run: {colorize(f'insidellms run {output_path}', Colors.GREEN)}")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command."""
    ensure_builtins_registered()

    try:
        if args.type == "model":
            info = model_registry.info(args.name)
        elif args.type == "probe":
            info = probe_registry.info(args.name)
        else:  # dataset
            from insideLLMs.benchmark_datasets import load_builtin_dataset

            ds = load_builtin_dataset(args.name)
            stats = ds.get_stats()
            print_header(f"Dataset: {args.name}")
            print_key_value("Description", ds.description)
            print_key_value(
                "Category", ds.category.value if hasattr(ds.category, "value") else ds.category
            )
            print_key_value("Total examples", stats.total_count)
            print_key_value(
                "Categories", ", ".join(stats.categories) if stats.categories else "N/A"
            )
            print_key_value(
                "Difficulties", ", ".join(stats.difficulties) if stats.difficulties else "N/A"
            )

            print_subheader("Sample Examples")
            for i, ex in enumerate(ds.sample(3, seed=42)):
                print(f"\n  {colorize(f'Example {i + 1}', Colors.BOLD)}")
                print(f"    {colorize('Input:', Colors.DIM)} {ex.input_text[:80]}...")
                if ex.expected_output:
                    print(f"    {colorize('Expected:', Colors.DIM)} {ex.expected_output[:50]}")
                print(f"    {colorize('Difficulty:', Colors.DIM)} {ex.difficulty}")

            return 0

        print_header(f"{args.type.capitalize()}: {args.name}")
        print_key_value("Factory", info["factory"])

        if info.get("default_kwargs"):
            print_key_value("Default args", json.dumps(info["default_kwargs"], indent=2))

        if info.get("doc"):
            print_subheader("Description")
            print(f"  {info['doc']}")

        return 0

    except KeyError:
        print_error(f"{args.type.capitalize()} '{args.name}' not found")
        return 1
    except Exception as e:
        print_error(f"Error: {e}")
        return 1


def cmd_quicktest(args: argparse.Namespace) -> int:
    """Execute the quicktest command."""
    ensure_builtins_registered()

    print_header("Quick Test")
    print_key_value("Model", args.model)
    print_key_value("Prompt", args.prompt[:50] + "..." if len(args.prompt) > 50 else args.prompt)

    try:
        # Parse model args
        model_args = json.loads(args.model_args)
        model_args["temperature"] = args.temperature
        model_args["max_tokens"] = args.max_tokens

        # Get model from registry - it may be a class or instance
        model_or_factory = model_registry.get(args.model)

        # If it's a class, instantiate it; if it's already an instance, use it
        if isinstance(model_or_factory, type):
            model = model_or_factory(**model_args)
        else:
            model = model_or_factory

        # Generate response
        spinner = Spinner("Generating response")
        start_time = time.time()
        spinner.spin()

        response = model.generate(args.prompt)
        elapsed = time.time() - start_time
        spinner.stop(success=True)

        print_subheader("Response")
        print(f"  {response}")

        print_subheader("Stats")
        print_key_value("Latency", f"{elapsed * 1000:.1f}ms")
        print_key_value("Response length", f"{len(response)} characters")

        # Apply probe if specified
        if args.probe:
            print_subheader(f"Probe: {args.probe}")
            probe_factory = probe_registry.get(args.probe)
            probe_factory()
            # Note: Probe evaluation would go here
            print_info(
                f"Probe '{args.probe}' applied (detailed scoring available in full experiments)"
            )

        return 0

    except KeyError as e:
        print_error(f"Unknown model or probe: {e}")
        return 1
    except Exception as e:
        print_error(f"Error: {e}")
        return 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Execute the benchmark command."""
    ensure_builtins_registered()

    print_header("Running Benchmark Suite")

    models = [m.strip() for m in args.models.split(",")]
    probes = [p.strip() for p in args.probes.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else None

    print_key_value("Models", ", ".join(models))
    print_key_value("Probes", ", ".join(probes))
    if datasets:
        print_key_value("Datasets", ", ".join(datasets))
    print_key_value("Max examples", args.max_examples)

    try:
        from insideLLMs.benchmark_datasets import (
            create_comprehensive_benchmark_suite,
            load_builtin_dataset,
        )

        # Load datasets
        if datasets:
            suite_examples = []
            for ds_name in datasets:
                ds = load_builtin_dataset(ds_name)
                for ex in ds.sample(args.max_examples, seed=42):
                    suite_examples.append(ex)
        else:
            suite = create_comprehensive_benchmark_suite(
                max_examples_per_dataset=args.max_examples, seed=42
            )
            suite_examples = list(suite.sample(args.max_examples * 5, seed=42))

        print_info(f"Loaded {len(suite_examples)} benchmark examples")

        results_all: list[dict[str, Any]] = []

        for model_name in models:
            print_subheader(f"Model: {model_name}")

            try:
                model_or_factory = model_registry.get(model_name)
                model = (
                    model_or_factory() if isinstance(model_or_factory, type) else model_or_factory
                )
            except Exception as e:
                print_warning(f"Could not load model {model_name}: {e}")
                continue

            for probe_name in probes:
                print(f"  Running probe: {colorize(probe_name, Colors.GREEN)}")

                try:
                    probe_or_factory = probe_registry.get(probe_name)
                    probe = (
                        probe_or_factory()
                        if isinstance(probe_or_factory, type)
                        else probe_or_factory
                    )
                except Exception as e:
                    print_warning(f"  Could not load probe {probe_name}: {e}")
                    continue

                # Create runner and run
                from insideLLMs.runner import ProbeRunner

                runner = ProbeRunner(model, probe)

                inputs = [ex.input_text for ex in suite_examples[: args.max_examples]]
                progress = ProgressBar(len(inputs), prefix=f"  {probe_name}")

                probe_results = []
                for i, inp in enumerate(inputs):
                    try:
                        result = runner.run_single(inp)
                        probe_results.append(result)
                    except Exception:
                        pass
                    progress.update(i + 1)

                progress.finish()

                success_count = sum(
                    1
                    for r in probe_results
                    if getattr(r, "status", None) == "success"
                    or (isinstance(r, dict) and r.get("status") == "success")
                )
                results_all.append(
                    {
                        "model": model_name,
                        "probe": probe_name,
                        "total": len(inputs),
                        "success": success_count,
                        "accuracy": success_count / max(1, len(inputs)),
                    }
                )

        # Summary
        print_subheader("Benchmark Results Summary")
        print(f"\n  {'Model':<15} {'Probe':<15} {'Accuracy':>10} {'Success':>10}")
        print("  " + "-" * 55)
        for r in results_all:
            print(
                f"  {r['model']:<15} {r['probe']:<15} {r['accuracy'] * 100:>9.1f}% {r['success']:>5}/{r['total']}"
            )

        # Save results if output specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            results_file = output_dir / "benchmark_results.json"
            with open(results_file, "w") as f:
                json.dump(results_all, f, indent=2)
            print_success(f"Results saved to: {results_file}")

            if args.html_report:
                print_info(
                    "HTML report generation for `benchmark` requires ExperimentResult format; "
                    "use `insidellms harness` + `insidellms report`."
                )

        return 0

    except Exception as e:
        print_error(f"Benchmark error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """Execute the compare command."""
    ensure_builtins_registered()

    try:
        output_format = getattr(args, "format", "table") or "table"

        print_header("Model Comparison")

        models = [m.strip() for m in str(args.models).split(",") if m.strip()]
        if not models:
            print_error("No models provided via --models")
            return 1
        print_key_value("Models", ", ".join(models))

        inputs: list[str] = []
        if args.input:
            inputs = [str(args.input)]
        elif args.input_file:
            input_path = Path(args.input_file)
            if not input_path.exists():
                print_error(f"Input file not found: {input_path}")
                return 1

            try:
                if input_path.suffix == ".json":
                    with open(input_path, encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        inputs = [
                            d.get("input", d.get("question", str(d))) if isinstance(d, dict) else str(d)
                            for d in data
                        ]
                    elif isinstance(data, dict):
                        inputs = [data.get("input", data.get("question", str(data)))]
                    else:
                        inputs = [str(data)]
                elif input_path.suffix == ".jsonl":
                    with open(input_path, encoding="utf-8") as f:
                        for line_no, line in enumerate(f, start=1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError as e:
                                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
                            inputs.append(
                                data.get("input", data.get("question", str(data)))
                                if isinstance(data, dict)
                                else str(data)
                            )
                else:
                    with open(input_path, encoding="utf-8") as f:
                        inputs = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print_error(f"Could not read inputs from {input_path}: {e}")
                return 1
        else:
            print_error("Please provide --input or --input-file")
            return 1

        print_key_value("Inputs", len(inputs))

        # Instantiate models once (best-effort).
        model_instances: dict[str, Any] = {}
        for model_name in models:
            try:
                model_or_factory = model_registry.get(model_name)
                model_instances[model_name] = (
                    model_or_factory() if isinstance(model_or_factory, type) else model_or_factory
                )
            except Exception as e:
                model_instances[model_name] = None
                print_warning(f"Could not load model '{model_name}': {e}")

        results: list[dict[str, Any]] = []
        for inp in inputs:
            inp_display = inp[:50] + "..." if len(inp) > 50 else inp
            if output_format == "table":
                print_subheader(f"Input: {inp_display}")

            item: dict[str, Any] = {"input": inp, "models": []}
            for model_name in models:
                model = model_instances.get(model_name)
                if model is None:
                    item["models"].append(
                        {
                            "model": model_name,
                            "response": None,
                            "latency_ms": None,
                            "error": "model_init_failed",
                        }
                    )
                    continue

                try:
                    start = time.time()
                    response = model.generate(inp)
                    elapsed = time.time() - start
                    item["models"].append(
                        {
                            "model": model_name,
                            "response": response,
                            "latency_ms": elapsed * 1000,
                            "error": None,
                        }
                    )

                    if output_format == "table":
                        print(f"\n  {colorize(model_name, Colors.CYAN)}:")
                        preview = str(response)
                        print(
                            f"    {preview[:100]}{'...' if len(preview) > 100 else ''}"
                        )
                        print(
                            f"    {colorize(f'({elapsed * 1000:.1f}ms)', Colors.DIM)}"
                        )
                except Exception as e:
                    item["models"].append(
                        {
                            "model": model_name,
                            "response": None,
                            "latency_ms": None,
                            "error": str(e),
                        }
                    )
                    if output_format == "table":
                        print(f"\n  {colorize(model_name, Colors.RED)}: Error - {e}")

            results.append(item)

        if output_format == "json":
            payload = json.dumps(results, indent=2, default=_json_default)
            if args.output:
                Path(args.output).write_text(payload, encoding="utf-8")
                print_success(f"Results saved to: {args.output}")
            else:
                print(payload)
            return 0

        if output_format == "markdown":
            rows: list[dict[str, Any]] = []
            for item in results:
                for model_item in item["models"]:
                    rows.append(
                        {
                            "input": item["input"],
                            "model": model_item["model"],
                            "response": model_item.get("response") or "",
                            "error": model_item.get("error") or "",
                            "latency_ms": model_item.get("latency_ms"),
                        }
                    )

            def escape_cell(value: Any) -> str:
                text = "" if value is None else str(value)
                return text.replace("|", "\\|").replace("\n", "<br>")

            header = "| Input | Model | Response | Error | Latency (ms) |"
            sep = "|---|---|---|---|---|"
            lines = [header, sep]
            for row in rows:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            escape_cell(row["input"]),
                            escape_cell(row["model"]),
                            escape_cell(row["response"]),
                            escape_cell(row["error"]),
                            escape_cell(
                                f"{float(row['latency_ms']):.1f}"
                                if isinstance(row.get("latency_ms"), (int, float))
                                else ""
                            ),
                        ]
                    )
                    + " |"
                )
            md = "\n".join(lines)

            if args.output:
                Path(args.output).write_text(md, encoding="utf-8")
                print_success(f"Results saved to: {args.output}")
            else:
                print(md)
            return 0

        # table output already printed; optionally write JSON to file.
        if args.output:
            payload = json.dumps(results, indent=2, default=_json_default)
            Path(args.output).write_text(payload, encoding="utf-8")
            print_success(f"Results saved to: {args.output}")

        return 0

    except Exception as e:
        print_error(f"Comparison error: {e}")
        return 1


def cmd_interactive(args: argparse.Namespace) -> int:
    """Execute the interactive command."""
    ensure_builtins_registered()

    print_header("Interactive Mode")
    print_info(f"Model: {args.model}")
    print_info("Type 'help' for commands, 'quit' to exit")

    try:
        model_or_factory = model_registry.get(args.model)
        model = model_or_factory() if isinstance(model_or_factory, type) else model_or_factory
    except Exception as e:
        print_error(f"Could not load model: {e}")
        return 1

    # Command history
    history: list[str] = []

    # Load history file
    history_path = Path(args.history_file)
    if history_path.exists():
        with open(history_path) as f:
            history = [line.strip() for line in f.readlines()]

    print()

    while True:
        try:
            prompt = input(colorize(">>> ", Colors.BRIGHT_CYAN))
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        prompt = prompt.strip()
        if not prompt:
            continue

        # Save to history
        history.append(prompt)
        with open(history_path, "a") as f:
            f.write(prompt + "\n")

        # Handle commands
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        elif prompt.lower() == "help":
            print(f"""
{colorize("Available Commands:", Colors.BOLD)}
  help          - Show this help message
  quit/exit/q   - Exit interactive mode
  history       - Show command history
  clear         - Clear the screen
  model <name>  - Switch to a different model
  probe <name>  - Run a probe on the last response

{colorize("Usage:", Colors.BOLD)}
  Just type your prompt and press Enter to get a response.
""")
        elif prompt.lower() == "history":
            print_subheader("Command History")
            for i, h in enumerate(history[-20:], 1):
                print(f"  {i:3}. {h[:60]}")
        elif prompt.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
        elif prompt.lower().startswith("model "):
            new_model = prompt[6:].strip()
            try:
                model_or_factory = model_registry.get(new_model)
                model = (
                    model_or_factory() if isinstance(model_or_factory, type) else model_or_factory
                )
                print_success(f"Switched to model: {new_model}")
            except Exception as e:
                print_error(f"Could not load model: {e}")
        else:
            # Regular prompt - generate response
            spinner = Spinner("Thinking")
            start = time.time()
            spinner.spin()

            try:
                response = model.generate(prompt)
                elapsed = time.time() - start
                spinner.stop(success=True)

                print(f"\n{response}\n")
                print(colorize(f"[{elapsed * 1000:.0f}ms]", Colors.DIM))
                print()
            except Exception as e:
                spinner.stop(success=False)
                print_error(f"Error: {e}")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the validate command."""
    target_path = Path(args.config)
    if not target_path.exists():
        print_error(f"Path not found: {target_path}")
        return 1

    # ---------------------------------------------------------------------
    # Run directory validation (manifest.json + records.jsonl)
    # ---------------------------------------------------------------------
    if target_path.is_dir() or target_path.name == "manifest.json":
        run_dir = target_path if target_path.is_dir() else target_path.parent
        manifest_path = (
            target_path if target_path.name == "manifest.json" else run_dir / "manifest.json"
        )

        print_header("Validate Run Directory")
        print_key_value("Run dir", run_dir)
        print_key_value("Manifest", manifest_path)

        if not manifest_path.exists():
            print_error(f"manifest.json not found: {manifest_path}")
            return 1

        from insideLLMs.schemas import OutputValidationError, OutputValidator, SchemaRegistry

        registry = SchemaRegistry()
        validator = OutputValidator(registry)

        errors = 0

        def _handle_error(msg: str) -> None:
            nonlocal errors
            errors += 1
            if args.mode == "warn":
                print_warning(msg)
            else:
                print_error(msg)

        try:
            manifest_obj = json.loads(manifest_path.read_text())
        except Exception as e:
            _handle_error(f"Could not read manifest JSON: {e}")
            return 0 if args.mode == "warn" else 1

        # Determine schema version: CLI override > manifest.schema_version > manifest.schemas[name]
        schema_version = (
            args.schema_version
            or manifest_obj.get("schema_version")
            or manifest_obj.get("schemas", {}).get(registry.RUN_MANIFEST)
            or DEFAULT_SCHEMA_VERSION
        )

        print_key_value("Schema version", schema_version)

        # Validate manifest
        try:
            validator.validate(
                registry.RUN_MANIFEST,
                manifest_obj,
                schema_version=schema_version,
                mode="strict",
            )
        except OutputValidationError as e:
            _handle_error(f"Manifest schema mismatch: {e}")
            if args.mode != "warn":
                return 1

        # Validate records
        records_file = manifest_obj.get("records_file") or "records.jsonl"
        records_path = run_dir / records_file
        print_key_value("Records", records_path)

        if not records_path.exists():
            _handle_error(f"records file not found: {records_path}")
            return 0 if args.mode == "warn" else 1

        try:
            with open(records_path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        _handle_error(f"Invalid JSON on line {line_no}: {e}")
                        if args.mode != "warn":
                            return 1
                        continue
                    try:
                        validator.validate(
                            registry.RESULT_RECORD,
                            obj,
                            schema_version=schema_version,
                            mode="strict",
                        )
                    except OutputValidationError as e:
                        _handle_error(f"Record line {line_no} schema mismatch: {e}")
                        if args.mode != "warn":
                            return 1
        except Exception as e:
            _handle_error(f"Error reading records: {e}")
            return 0 if args.mode == "warn" else 1

        if errors:
            if args.mode == "warn":
                print_warning(f"Validation completed with {errors} error(s) (warn mode)")
                return 0
            print_error(f"Validation failed with {errors} error(s)")
            return 1

        print_success("Validation OK")
        return 0

    # ---------------------------------------------------------------------
    # Config validation (legacy)
    # ---------------------------------------------------------------------
    print_header("Validate Configuration")
    config_path = target_path
    print_key_value("Config", config_path)

    try:
        import yaml

        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f) if config_path.suffix in (".yaml", ".yml") else json.load(f)

        config_errors: list[str] = []
        warnings: list[str] = []

        # Validate model
        if "model" not in config:
            config_errors.append("Missing required field: model")
        else:
            model_config = config["model"]
            if "type" not in model_config:
                config_errors.append("Missing model.type")
            else:
                ensure_builtins_registered()
                if model_config["type"] not in model_registry.list():
                    config_errors.append(f"Unknown model type: {model_config['type']}")

        # Validate probe
        if "probe" not in config:
            config_errors.append("Missing required field: probe")
        else:
            probe_config = config["probe"]
            if "type" not in probe_config:
                config_errors.append("Missing probe.type")
            else:
                ensure_builtins_registered()
                if probe_config["type"] not in probe_registry.list():
                    config_errors.append(f"Unknown probe type: {probe_config['type']}")

        # Validate dataset
        if "dataset" not in config:
            warnings.append("No dataset specified (will use builtin)")
        else:
            ds_config = config["dataset"]
            if "path" in ds_config:
                ds_path = Path(ds_config["path"])
                if not ds_path.exists():
                    warnings.append(f"Dataset file not found: {ds_path}")

        # Report results
        if config_errors:
            print_subheader("Errors")
            for e in config_errors:
                print(f"  {colorize('ERROR', Colors.RED)} {e}")

        if warnings:
            print_subheader("Warnings")
            for w in warnings:
                print(f"  {colorize('WARN', Colors.YELLOW)} {w}")

        if not config_errors:
            print()
            print_success("Configuration is valid!")
            return 0
        else:
            print()
            print_error(f"Configuration has {len(config_errors)} error(s)")
            return 1

    except Exception as e:
        print_error(f"Validation error: {e}")
        return 1


def cmd_export(args: argparse.Namespace) -> int:
    """Execute the export command."""
    print_header("Export Results")

    input_path = Path(args.input)
    if not input_path.exists():
        print_error(f"Input file not found: {input_path}")
        return 1

    print_key_value("Input", input_path)
    print_key_value("Format", args.format)

    try:
        with open(input_path) as f:
            results = json.load(f)

        if not isinstance(results, list):
            results = [results]

        output_path = args.output
        if not output_path:
            output_path = input_path.stem + f".{args.format}"

        if args.format == "csv":
            import csv

            if results:
                keys = results[0].keys()
                with open(output_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(results)

        elif args.format == "markdown":
            content = results_to_markdown(results)
            with open(output_path, "w") as f:
                f.write(content)

        elif args.format == "html":
            print_error(
                "HTML export requires plotly and ExperimentResult format; "
                "use `insidellms report <run_dir>` to produce report.html."
            )
            return 1

        elif args.format == "latex":
            # Generate LaTeX table
            if results:
                keys = list(results[0].keys())
                lines = [
                    "\\begin{table}[h]",
                    "\\centering",
                    "\\begin{tabular}{" + "l" * len(keys) + "}",
                    "\\hline",
                    " & ".join(keys) + " \\\\",
                    "\\hline",
                ]
                for r in results[:20]:  # Limit rows
                    values = [str(r.get(k, ""))[:30] for k in keys]
                    lines.append(" & ".join(values) + " \\\\")
                lines.extend(
                    [
                        "\\hline",
                        "\\end{tabular}",
                        "\\caption{Experiment Results}",
                        "\\end{table}",
                    ]
                )
                with open(output_path, "w") as f:
                    f.write("\n".join(lines))

        print_success(f"Exported to: {output_path}")
        return 0

    except Exception as e:
        print_error(f"Export error: {e}")
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI."""
    global USE_COLOR, _CLI_QUIET, _CLI_STATUS_STREAM

    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle global flags
    if hasattr(args, "no_color") and args.no_color:
        USE_COLOR = False

    prev_quiet = _CLI_QUIET
    prev_stream = _CLI_STATUS_STREAM
    try:
        _CLI_QUIET = bool(getattr(args, "quiet", False))
        _CLI_STATUS_STREAM = sys.stderr if getattr(args, "format", None) == "json" else sys.stdout

        if not args.command:
            # Show a nice welcome message instead of just help
            print_header("insideLLMs")
            print(colorize("  A world-class toolkit for LLM evaluation and exploration", Colors.DIM))
            print()
            parser.print_help()
            print()
            print(
                colorize("Quick start: ", Colors.BOLD)
                + 'insidellms quicktest "Hello world" --model dummy'
            )
            return 0

        commands = {
            "run": cmd_run,
            "harness": cmd_harness,
            "report": cmd_report,
            "diff": cmd_diff,
            "schema": cmd_schema,
            "doctor": cmd_doctor,
            "list": cmd_list,
            "init": cmd_init,
            "info": cmd_info,
            "quicktest": cmd_quicktest,
            "benchmark": cmd_benchmark,
            "compare": cmd_compare,
            "interactive": cmd_interactive,
            "validate": cmd_validate,
            "export": cmd_export,
        }

        handler = commands.get(args.command)
        if handler:
            return handler(args)

        print_error(f"Unknown command: {args.command}")
        return 1
    finally:
        _CLI_QUIET = prev_quiet
        _CLI_STATUS_STREAM = prev_stream


if __name__ == "__main__":
    sys.exit(main())
