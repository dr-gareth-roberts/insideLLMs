"""Sign command: sign attestations in a run directory with Sigstore (cosign)."""

import argparse
from pathlib import Path

from .._output import print_error, print_header, print_success

from insideLLMs.signing import sign_blob


def cmd_sign(args: argparse.Namespace) -> int:
    """Sign each attestation in run_dir/attestations/ and write to run_dir/signing/."""
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print_error(f"Run directory not found: {run_dir}")
        return 1
    attestations_dir = run_dir / "attestations"
    if not attestations_dir.exists():
        print_error(f"attestations/ not found in {run_dir}. Run 'insidellms attest' first.")
        return 1
    signing_dir = run_dir / "signing"
    signing_dir.mkdir(parents=True, exist_ok=True)
    print_header("Sign attestations")
    dsse_files = sorted(attestations_dir.glob("*.dsse.json"))
    if not dsse_files:
        print_error("No *.dsse.json files in attestations/")
        return 1
    failed = 0
    for path in dsse_files:
        bundle_path = signing_dir / f"{path.stem}.sigstore.bundle.json"
        try:
            sign_blob(path, bundle_path)
            print_success(f"Signed {path.name}")
        except Exception as e:
            print_error(f"{path.name}: {e}")
            failed += 1
    if failed:
        print_error(f"{failed} attestation(s) failed to sign.")
        return 1
    print_success(f"All {len(dsse_files)} attestations signed in {signing_dir}")
    return 0
