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
"""

import sys
from typing import Optional

from insideLLMs._serialization import StrictSerializationError as StrictSerializationError
from insideLLMs._serialization import fingerprint_value as _fingerprint_value
from insideLLMs._serialization import serialize_value as _serialize_value
from insideLLMs._serialization import stable_json_dumps as _stable_json_dumps

from . import _output
from ._output import USE_COLOR as USE_COLOR
from ._output import Colors as Colors
from ._output import ProgressBar as ProgressBar
from ._output import Spinner as Spinner
from ._output import _cli_version_string as _cli_version_string
from ._output import _format_float as _format_float
from ._output import _format_percent as _format_percent
from ._output import _status_stream as _status_stream
from ._output import _supports_color as _supports_color
from ._output import _trim_text as _trim_text
from ._output import colorize as colorize
from ._output import print_error as print_error
from ._output import print_header as print_header
from ._output import print_info as print_info
from ._output import print_key_value as print_key_value
from ._output import print_subheader as print_subheader
from ._output import print_success as print_success
from ._output import print_warning as print_warning
from ._parsing import _add_output_schema_args as _add_output_schema_args
from ._parsing import _check_nltk_resource as _check_nltk_resource
from ._parsing import _has_module as _has_module
from ._parsing import _module_version as _module_version
from ._parsing import create_parser as create_parser
from ._record_utils import _json_default as _json_default
from ._record_utils import _metric_mismatch_context as _metric_mismatch_context
from ._record_utils import _metric_mismatch_details as _metric_mismatch_details
from ._record_utils import _metric_mismatch_reason as _metric_mismatch_reason
from ._record_utils import _output_fingerprint as _output_fingerprint
from ._record_utils import _output_summary as _output_summary
from ._record_utils import _output_text as _output_text
from ._record_utils import _parse_datetime as _parse_datetime
from ._record_utils import _primary_score as _primary_score
from ._record_utils import _probe_category_from_value as _probe_category_from_value
from ._record_utils import _read_jsonl_records as _read_jsonl_records
from ._record_utils import _record_key as _record_key
from ._record_utils import _record_label as _record_label
from ._record_utils import _status_from_record as _status_from_record
from ._record_utils import _status_string as _status_string
from ._record_utils import _strip_volatile_keys as _strip_volatile_keys
from ._record_utils import _trace_fingerprint as _trace_fingerprint
from ._record_utils import _trace_violation_count as _trace_violation_count
from ._record_utils import _trace_violations as _trace_violations
from ._record_utils import _write_jsonl as _write_jsonl
from ._report_builder import _build_basic_harness_report as _build_basic_harness_report
from ._report_builder import _build_experiments_from_records as _build_experiments_from_records
from .commands.attest import cmd_attest as cmd_attest
from .commands.benchmark import cmd_benchmark as cmd_benchmark
from .commands.compare import cmd_compare as cmd_compare
from .commands.diff import cmd_diff as cmd_diff
from .commands.doctor import cmd_doctor as cmd_doctor
from .commands.export import cmd_export as cmd_export
from .commands.harness import cmd_harness as cmd_harness
from .commands.info import cmd_info as cmd_info
from .commands.init_cmd import cmd_init as cmd_init
from .commands.interactive import cmd_interactive as cmd_interactive
from .commands.list_cmd import cmd_list as cmd_list
from .commands.quicktest import cmd_quicktest as cmd_quicktest
from .commands.report import cmd_report as cmd_report
from .commands.run import cmd_run as cmd_run
from .commands.schema import cmd_schema as cmd_schema
from .commands.sign import cmd_sign as cmd_sign
from .commands.validate import cmd_validate as cmd_validate
from .commands.verify import cmd_verify_signatures as cmd_verify_signatures

# Make _CLI_QUIET and _CLI_STATUS_TO_STDERR accessible at package level
_CLI_QUIET = _output._CLI_QUIET
_CLI_STATUS_TO_STDERR = _output._CLI_STATUS_TO_STDERR

__all__ = [
    "Colors",
    "ProgressBar",
    "Spinner",
    "_fingerprint_value",
    "_serialize_value",
    "_stable_json_dumps",
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


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI."""
    # Modify the _output module's globals directly so all functions see the changes.
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle global flags
    if hasattr(args, "no_color") and args.no_color:
        _output.USE_COLOR = False

    prev_quiet = _output._CLI_QUIET
    prev_to_stderr = _output._CLI_STATUS_TO_STDERR
    try:
        _output._CLI_QUIET = bool(getattr(args, "quiet", False))
        _output._CLI_STATUS_TO_STDERR = bool(getattr(args, "format", None) == "json")

        if not args.command:
            # Show a nice welcome message instead of just help
            print_header("insideLLMs")
            print(
                colorize("  A world-class toolkit for LLM evaluation and exploration", Colors.DIM)
            )
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
            "attest": cmd_attest,
            "sign": cmd_sign,
            "verify-signatures": cmd_verify_signatures,
        }

        handler = commands.get(args.command)
        if handler:
            return handler(args)

        print_error(f"Unknown command: {args.command}")
        return 1
    finally:
        _output._CLI_QUIET = prev_quiet
        _output._CLI_STATUS_TO_STDERR = prev_to_stderr


if __name__ == "__main__":
    sys.exit(main())
