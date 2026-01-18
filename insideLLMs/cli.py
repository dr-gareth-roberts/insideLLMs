"""Command-line interface for insideLLMs.

Provides commands for running experiments, listing available models and probes,
benchmarking, interactive exploration, and managing configurations.
"""

import argparse
import asyncio
import html
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from insideLLMs.registry import (
    ensure_builtins_registered,
    model_registry,
    probe_registry,
)
from insideLLMs.results import results_to_markdown, save_results_json
from insideLLMs.runner import (
    run_experiment_from_config,
    run_experiment_from_config_async,
    run_harness_from_config,
)
from insideLLMs.types import ExperimentResult

# ============================================================================
# Console Output Utilities (works without external dependencies)
# ============================================================================


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
        except Exception:
            return os.environ.get("ANSICON") is not None

    return True


# Global flag for color support
USE_COLOR = _supports_color()


def colorize(text: str, *codes: str) -> str:
    """Apply color codes to text if terminal supports colors."""
    if not USE_COLOR:
        return text
    return "".join(codes) + text + Colors.RESET


def print_header(title: str) -> None:
    """Print a styled header."""
    width = 70
    line = "═" * width
    print()
    print(colorize(line, Colors.BRIGHT_CYAN))
    print(colorize(f"  {title}".center(width), Colors.BOLD, Colors.BRIGHT_CYAN))
    print(colorize(line, Colors.BRIGHT_CYAN))


def print_subheader(title: str) -> None:
    """Print a styled subheader."""
    print()
    print(colorize(f"── {title} ", Colors.CYAN) + colorize("─" * (50 - len(title)), Colors.DIM))


def print_success(message: str) -> None:
    """Print a success message."""
    print(colorize("OK ", Colors.BRIGHT_GREEN) + message)


def print_error(message: str) -> None:
    """Print an error message."""
    print(colorize("ERROR ", Colors.BRIGHT_RED) + colorize(message, Colors.RED), file=sys.stderr)


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(colorize("WARN ", Colors.BRIGHT_YELLOW) + colorize(message, Colors.YELLOW))


def print_info(message: str) -> None:
    """Print an info message."""
    print(colorize("INFO ", Colors.BRIGHT_BLUE) + message)


def print_key_value(key: str, value: Any, indent: int = 2) -> None:
    """Print a key-value pair."""
    spaces = " " * indent
    print(f"{spaces}{colorize(key + ':', Colors.DIM)} {value}")


def _write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _build_basic_harness_report(
    experiments: list[ExperimentResult],
    summary: dict[str, Any],
    title: str,
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

    rows_html = "\n".join(
        f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td>"
        f"<td>{row[3]}</td><td>{row[4]}</td></tr>"
        for row in rows
    )

    def summary_table(section: str) -> str:
        items = summary.get(section, {})
        lines = []
        for name, stats in items.items():
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
  <div class="meta">Generated {datetime.now(timezone.utc).isoformat()}</div>

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
    """Simple progress bar for CLI output."""

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
        """Update the progress bar."""
        self.current = current
        self._render()

    def increment(self, amount: int = 1) -> None:
        """Increment the progress."""
        self.current += amount
        self._render()

    def _render(self) -> None:
        """Render the progress bar to the terminal."""
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

        line = f"\r{self.prefix}: {colorize(bar, Colors.CYAN)} {pct:5.1f}% ({self.current}/{self.total}){eta_str}"
        print(line, end="", flush=True)

    def finish(self) -> None:
        """Complete the progress bar."""
        self.current = self.total
        self._render()
        elapsed = time.time() - self.start_time
        print(f" {colorize(f'Done in {elapsed:.2f}s', Colors.GREEN)}")


class Spinner:
    """Simple spinner for indeterminate progress."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Loading"):
        self.message = message
        self.frame_idx = 0
        self.running = False

    def spin(self) -> None:
        """Render a single spinner frame."""
        frame = self.FRAMES[self.frame_idx % len(self.FRAMES)]
        print(f"\r{colorize(frame, Colors.CYAN)} {self.message}...", end="", flush=True)
        self.frame_idx += 1

    def stop(self, success: bool = True) -> None:
        """Stop the spinner with a final status."""
        if success:
            print(f"\r{colorize('OK', Colors.GREEN)} {self.message}... done")
        else:
            print(f"\r{colorize('FAIL', Colors.RED)} {self.message}... failed")


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
        version="%(prog)s 0.2.0",
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
        help="Output directory for results, summary, and report",
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
        help="Validate a configuration file without running",
        formatter_class=CustomFormatter,
    )
    validate_parser.add_argument(
        "config",
        type=str,
        help="Path to the configuration file to validate",
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
        if args.verbose:
            if progress_bar is None:
                progress_bar = ProgressBar(total, prefix="Evaluating")
            progress_bar.update(current)

    try:
        start_time = time.time()

        if args.use_async:
            print_info(f"Using async execution with concurrency={args.concurrency}")
            results = asyncio.run(
                run_experiment_from_config_async(
                    config_path,
                    concurrency=args.concurrency,
                    progress_callback=progress_callback if args.verbose else None,
                )
            )
        else:
            results = run_experiment_from_config(
                config_path,
                progress_callback=progress_callback if args.verbose else None,
            )

        elapsed = time.time() - start_time

        if progress_bar:
            progress_bar.finish()

        # Calculate summary
        success_count = sum(1 for r in results if r.get("status") == "success")
        error_count = sum(1 for r in results if r.get("status") == "error")
        total = len(results)

        # Output results
        if args.output:
            save_results_json(results, args.output)
            print_success(f"Results saved to: {args.output}")

        if args.format == "json":
            print(json.dumps(results, indent=2, default=str))
        elif args.format == "markdown":
            print(results_to_markdown(results))
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

            if results:
                latencies = [
                    r.get("latency_ms", 0) for r in results if r.get("status") == "success"
                ]
                if latencies:
                    avg_latency = sum(latencies) / len(latencies)
                    min_latency = min(latencies)
                    max_latency = max(latencies)
                    print_key_value("Avg latency", f"{avg_latency:.1f}ms")
                    print_key_value("Min/Max", f"{min_latency:.1f}ms / {max_latency:.1f}ms")

            # Show first few results
            print_subheader("Sample Results")
            for i, r in enumerate(results[:5]):
                status_icon = (
                    colorize("OK", Colors.GREEN)
                    if r.get("status") == "success"
                    else colorize("FAIL", Colors.RED)
                )
                inp = str(r.get("input", ""))[:50]
                if len(str(r.get("input", ""))) > 50:
                    inp += "..."
                print(f"  {status_icon} [{i + 1}] {inp}")

            if len(results) > 5:
                print(colorize(f"  ... and {len(results) - 5} more", Colors.DIM))

        return 0

    except Exception as e:
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
        if not args.verbose:
            return
        if progress_bar is None:
            progress_bar = ProgressBar(total, prefix="Evaluating")
        progress_bar.update(current)

    try:
        start_time = time.time()
        result = run_harness_from_config(
            config_path,
            progress_callback=progress_callback if args.verbose else None,
        )
        elapsed = time.time() - start_time

        if progress_bar:
            progress_bar.finish()

        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else Path(result["config"].get("output_dir", "results"))
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "results.jsonl"
        summary_path = output_dir / "summary.json"
        report_path = output_dir / "report.html"

        _write_jsonl(result["records"], results_path)
        print_success(f"Results written to: {results_path}")

        summary_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": result["summary"],
            "config": result["config"],
        }
        with open(summary_path, "w") as f:
            json.dump(summary_payload, f, indent=2, default=str)
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
                )
            except ImportError:
                report_html = _build_basic_harness_report(
                    result["experiments"],
                    result["summary"],
                    report_title,
                )
                with open(report_path, "w") as f:
                    f.write(report_html)

            print_success(f"Report written to: {report_path}")

        print_key_value("Elapsed", f"{elapsed:.2f}s")
        return 0

    except Exception as e:
        print_error(f"Error running harness: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


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
                try:
                    from insideLLMs.visualization import create_interactive_html_report

                    output_dir / "benchmark_report.html"
                    # Note: Would need to convert results to ExperimentResult objects
                    print_info("HTML report generation requires ExperimentResult format")
                except ImportError:
                    print_warning("HTML report generation requires plotly")

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

    print_header("Model Comparison")

    models = [m.strip() for m in args.models.split(",")]
    print_key_value("Models", ", ".join(models))

    # Get inputs
    inputs: list[str] = []
    if args.input:
        inputs = [args.input]
    elif args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            print_error(f"Input file not found: {input_path}")
            return 1

        if input_path.suffix == ".json":
            with open(input_path) as f:
                data = json.load(f)
                inputs = [d.get("input", d.get("question", str(d))) for d in data]
        elif input_path.suffix == ".jsonl":
            with open(input_path) as f:
                for line in f:
                    data = json.loads(line)
                    inputs.append(data.get("input", data.get("question", str(data))))
        else:
            with open(input_path) as f:
                inputs = [line.strip() for line in f if line.strip()]
    else:
        print_error("Please provide --input or --input-file")
        return 1

    print_key_value("Inputs", len(inputs))

    try:
        results: list[dict[str, Any]] = []

        for inp in inputs:
            inp_display = inp[:50] + "..." if len(inp) > 50 else inp
            print_subheader(f"Input: {inp_display}")

            row = {"input": inp}

            for model_name in models:
                try:
                    model_or_factory = model_registry.get(model_name)
                    model = (
                        model_or_factory()
                        if isinstance(model_or_factory, type)
                        else model_or_factory
                    )

                    start = time.time()
                    response = model.generate(inp)
                    elapsed = time.time() - start

                    row[f"{model_name}_response"] = response
                    row[f"{model_name}_latency_ms"] = elapsed * 1000

                    print(f"\n  {colorize(model_name, Colors.CYAN)}:")
                    print(f"    {response[:100]}{'...' if len(response) > 100 else ''}")
                    print(f"    {colorize(f'({elapsed * 1000:.1f}ms)', Colors.DIM)}")

                except Exception as e:
                    row[f"{model_name}_response"] = f"ERROR: {e}"
                    row[f"{model_name}_latency_ms"] = None
                    print(f"\n  {colorize(model_name, Colors.RED)}: Error - {e}")

            results.append(row)

        # Output results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
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
    print_header("Validate Configuration")

    config_path = Path(args.config)
    if not config_path.exists():
        print_error(f"Config file not found: {config_path}")
        return 1

    print_key_value("Config", config_path)

    try:
        import yaml

        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f) if config_path.suffix in (".yaml", ".yml") else json.load(f)

        errors: list[str] = []
        warnings: list[str] = []

        # Validate model
        if "model" not in config:
            errors.append("Missing required field: model")
        else:
            model_config = config["model"]
            if "type" not in model_config:
                errors.append("Missing model.type")
            else:
                ensure_builtins_registered()
                if model_config["type"] not in model_registry.list():
                    errors.append(f"Unknown model type: {model_config['type']}")

        # Validate probe
        if "probe" not in config:
            errors.append("Missing required field: probe")
        else:
            probe_config = config["probe"]
            if "type" not in probe_config:
                errors.append("Missing probe.type")
            else:
                ensure_builtins_registered()
                if probe_config["type"] not in probe_registry.list():
                    errors.append(f"Unknown probe type: {probe_config['type']}")

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
        if errors:
            print_subheader("Errors")
            for e in errors:
                print(f"  {colorize('ERROR', Colors.RED)} {e}")

        if warnings:
            print_subheader("Warnings")
            for w in warnings:
                print(f"  {colorize('WARN', Colors.YELLOW)} {w}")

        if not errors:
            print()
            print_success("Configuration is valid!")
            return 0
        else:
            print()
            print_error(f"Configuration has {len(errors)} error(s)")
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
            try:
                from insideLLMs.visualization import create_interactive_html_report

                # Would need ExperimentResult conversion
                print_warning("HTML export requires ExperimentResult format")
                return 1
            except ImportError:
                print_error("HTML export requires plotly")
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
    global USE_COLOR

    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle global flags
    if hasattr(args, "no_color") and args.no_color:
        USE_COLOR = False

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
    else:
        print_error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
