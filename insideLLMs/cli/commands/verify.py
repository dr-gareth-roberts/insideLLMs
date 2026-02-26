"""Verify-signatures command: verify attestation signatures in a run directory."""

import argparse
from pathlib import Path

from insideLLMs.signing import verify_bundle

from .._output import print_error, print_header, print_success


def cmd_verify_signatures(args: argparse.Namespace) -> int:
    """Verify each attestation has a valid signature bundle."""
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print_error(f"Run directory not found: {run_dir}")
        return 1
    attestations_dir = run_dir / "attestations"
    signing_dir = run_dir / "signing"
    if not attestations_dir.exists():
        print_error(f"attestations/ not found in {run_dir}")
        return 1
    if not signing_dir.exists():
        print_error(f"signing/ not found in {run_dir}. Run 'insidellms sign' first.")
        return 1
    print_header("Verify attestation signatures")
    identity = getattr(args, "identity", None)
    failed = 0
    for dsse_path in sorted(attestations_dir.glob("*.dsse.json")):
        bundle_path = signing_dir / f"{dsse_path.stem}.sigstore.bundle.json"
        if not bundle_path.exists():
            print_error(f"No bundle for {dsse_path.name}")
            failed += 1
            continue
        try:
            if verify_bundle(dsse_path, bundle_path, identity_constraints=identity):
                print_success(f"Verified {dsse_path.name}")
            else:
                print_error(f"Verification failed: {dsse_path.name}")
                failed += 1
        except Exception as e:
            print_error(f"{dsse_path.name}: {e}")
            failed += 1
    if failed:
        print_error(f"{failed} attestation(s) failed verification.")
        return 1
    print_success("All attestation signatures verified.")
    return 0
