"""Attest command: generate attestations for a run directory (Ultimate mode)."""

import argparse
from pathlib import Path

from insideLLMs.runtime._ultimate import run_ultimate_post_artifact

from .._output import print_error, print_header, print_success


def cmd_attest(args: argparse.Namespace) -> int:
    """Generate attestations for a run directory if not already present."""
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print_error(f"Run directory not found: {run_dir}")
        return 1
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print_error(f"manifest.json not found in {run_dir}. Run a probe first.")
        return 1
    print_header("Generate attestations")
    try:
        run_ultimate_post_artifact(run_dir)
        print_success(f"Attestations written to {run_dir / 'attestations'}")
        return 0
    except Exception as e:
        print_error(str(e))
        return 1
