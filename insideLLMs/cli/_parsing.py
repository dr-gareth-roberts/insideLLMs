"""Argument parser construction for the insideLLMs CLI."""

import argparse
import importlib.metadata
import importlib.util

from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION

from ._output import Colors, _cli_version_string, colorize


def _add_output_schema_args(parser: argparse.ArgumentParser) -> None:
    """Add common output schema validation arguments to a subcommand parser."""
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


def _module_version(dist_name: str):
    try:
        return importlib.metadata.version(dist_name)
    except (ImportError, AttributeError):
        return None


def _has_module(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _check_nltk_resource(path: str) -> bool:
    try:
        import nltk

        nltk.data.find(path)
        return True
    except (ImportError, LookupError, OSError):
        return False


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    # Custom formatter that preserves formatting in epilog
    class CustomFormatter(argparse.RawDescriptionHelpFormatter):
        def __init__(self, prog):
            super().__init__(prog, max_help_position=40, width=100)

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    common_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output (errors only)",
    )

    parser = argparse.ArgumentParser(
        prog="insidellms",
        description=colorize("insideLLMs", Colors.BOLD, Colors.CYAN)
        + " - A world-class toolkit for probing, evaluating, and testing large language models",
        formatter_class=CustomFormatter,
        parents=[common_parser],
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

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # =========================================================================
    # Run command
    # =========================================================================
    run_parser = subparsers.add_parser(
        "run",
        help="Run an experiment from a configuration file",
        formatter_class=CustomFormatter,
        parents=[common_parser],
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
        "--strict-serialization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail fast on non-deterministic values during hashing/fingerprinting.",
    )
    run_parser.add_argument(
        "--deterministic-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Omit host-dependent manifest fields (platform/python version).",
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
        parents=[common_parser],
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
        "--strict-serialization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail fast on non-deterministic values during hashing/fingerprinting.",
    )
    harness_parser.add_argument(
        "--deterministic-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Omit host-dependent manifest fields (platform/python version).",
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
        parents=[common_parser],
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
        parents=[common_parser],
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
        parents=[common_parser],
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
        parents=[common_parser],
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
        parents=[common_parser],
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
        parents=[common_parser],
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
        parents=[common_parser],
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
        parents=[common_parser],
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
        parents=[common_parser],
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
        help=(
            "Configuration template: basic (model+probe+dataset), "
            "benchmark (+ dataset suites), tracking (+ experiment logging), "
            "full (all features with async and HTML reports)"
        ),
    )

    # =========================================================================
    # Info command
    # =========================================================================
    info_parser = subparsers.add_parser(
        "info",
        help="Show detailed information about a model, probe, or dataset",
        formatter_class=CustomFormatter,
        parents=[common_parser],
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
        parents=[common_parser],
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
        parents=[common_parser],
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
        parents=[common_parser],
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
