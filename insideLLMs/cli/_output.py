"""Console output utilities for the insideLLMs CLI.

Provides colored terminal output, progress bars, spinners, and formatting
helpers. Works without external dependencies beyond the standard library.
"""

import importlib.metadata
import os
import sys
import time
from typing import Any, Optional

# CLI output routing:
# - "status" output (headers, warnings, progress) is routed via _CLI_STATUS_TO_STDERR
# - machine-readable payloads (e.g., --format json) should go to stdout only
_CLI_QUIET = False
_CLI_STATUS_TO_STDERR = False


def _status_stream():
    # Avoid capturing a fixed sys.stdout/sys.stderr at import time. This ensures
    # pytest's capsys and other stdout/stderr redirection works as expected.
    return sys.stderr if _CLI_STATUS_TO_STDERR else sys.stdout


def _cli_version_string() -> str:
    """Return the CLI version string.

    Prefer the installed package metadata when available (editable installs, wheels),
    but fall back to the source-tree ``insideLLMs.__version__`` for local runs.
    """

    try:
        return importlib.metadata.version("insideLLMs")
    except (ImportError, AttributeError):
        try:
            import insideLLMs

            return str(getattr(insideLLMs, "__version__", "unknown"))
        except (ImportError, AttributeError):
            return "unknown"


def _supports_color() -> bool:
    """Check if the terminal supports color output."""
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
        except (OSError, AttributeError):
            return os.environ.get("ANSICON") is not None

    return True


# Global flag for color support
USE_COLOR = _supports_color()


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

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


def colorize(text: str, *codes: str) -> str:
    """Apply color codes to text if terminal supports colors."""
    if not USE_COLOR:
        return text
    return "".join(codes) + text + Colors.RESET


def print_header(title: str) -> None:
    """Print a styled header with decorative borders."""
    if _CLI_QUIET:
        return
    width = 70
    line = "\u2550" * width
    print(file=_status_stream())
    print(colorize(line, Colors.BRIGHT_CYAN), file=_status_stream())
    print(
        colorize(f"  {title}".center(width), Colors.BOLD, Colors.BRIGHT_CYAN),
        file=_status_stream(),
    )
    print(colorize(line, Colors.BRIGHT_CYAN), file=_status_stream())


def print_subheader(title: str) -> None:
    """Print a styled subheader with a horizontal rule."""
    if _CLI_QUIET:
        return
    print(file=_status_stream())
    print(
        colorize(f"\u2500\u2500 {title} ", Colors.CYAN)
        + colorize("\u2500" * (50 - len(title)), Colors.DIM),
        file=_status_stream(),
    )


def print_success(message: str) -> None:
    """Print a success message with green OK prefix."""
    if _CLI_QUIET:
        return
    print(colorize("OK ", Colors.BRIGHT_GREEN) + message, file=_status_stream())


def print_error(message: str) -> None:
    """Print an error message with red ERROR prefix."""
    print(colorize("ERROR ", Colors.BRIGHT_RED) + colorize(message, Colors.RED), file=sys.stderr)


def print_warning(message: str) -> None:
    """Print a warning message with yellow WARN prefix."""
    if _CLI_QUIET:
        return
    print(
        colorize("WARN ", Colors.BRIGHT_YELLOW) + colorize(message, Colors.YELLOW),
        file=_status_stream(),
    )


def print_info(message: str) -> None:
    """Print an informational message with blue INFO prefix."""
    if _CLI_QUIET:
        return
    print(colorize("INFO ", Colors.BRIGHT_BLUE) + message, file=_status_stream())


def print_key_value(key: str, value: Any, indent: int = 2) -> None:
    """Print a formatted key-value pair."""
    if _CLI_QUIET:
        return
    spaces = " " * indent
    print(f"{spaces}{colorize(key + ':', Colors.DIM)} {value}", file=_status_stream())


def _format_percent(value: Optional[float]) -> str:
    """Format a decimal value as a percentage string."""
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _format_float(value: Optional[float]) -> str:
    """Format a float value with three decimal places."""
    if value is None:
        return "-"
    return f"{value:.3f}"


def _trim_text(text: str, limit: int = 200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


class ProgressBar:
    """Simple progress bar for CLI output with ETA estimation."""

    def __init__(
        self,
        total: int,
        width: int = 40,
        prefix: str = "Progress",
        show_eta: bool = True,
    ):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.show_eta = show_eta
        self.current = 0
        self.start_time = time.time()

    def update(self, current: int) -> None:
        """Update the progress bar to a specific position."""
        self.current = current
        self._render()

    def increment(self, amount: int = 1) -> None:
        """Increment the progress by a given amount."""
        self.current += amount
        self._render()

    def _render(self) -> None:
        """Render the progress bar to the terminal."""
        if _CLI_QUIET:
            return
        pct = 100 if self.total == 0 else min(100, self.current / self.total * 100)

        filled = int(self.width * self.current / max(1, self.total))
        bar = "\u2588" * filled + "\u2591" * (self.width - filled)

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
        print(line, end="", flush=True, file=_status_stream())

    def finish(self) -> None:
        """Complete the progress bar and print final status."""
        if _CLI_QUIET:
            return
        self.current = self.total
        self._render()
        elapsed = time.time() - self.start_time
        print(
            f" {colorize(f'Done in {elapsed:.2f}s', Colors.GREEN)}",
            file=_status_stream(),
        )


class Spinner:
    """Simple spinner for indeterminate progress."""

    FRAMES = [
        "\u280b",
        "\u2819",
        "\u2839",
        "\u2838",
        "\u283c",
        "\u2834",
        "\u2826",
        "\u2827",
        "\u2807",
        "\u280f",
    ]

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
            file=_status_stream(),
        )
        self.frame_idx += 1

    def stop(self, success: bool = True) -> None:
        """Stop the spinner with a final status."""
        if _CLI_QUIET:
            return
        if success:
            print(
                f"\r{colorize('OK', Colors.GREEN)} {self.message}... done",
                file=_status_stream(),
            )
        else:
            print(
                f"\r{colorize('FAIL', Colors.RED)} {self.message}... failed",
                file=_status_stream(),
            )
