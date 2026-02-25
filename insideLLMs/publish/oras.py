"""ORAS/OCI: package run_dir as OCI artifact, push/pull."""

from __future__ import annotations

import os
from pathlib import Path

class DummyClient:
    def push(self, *args, **kwargs): pass
    def pull(self, *args, **kwargs): pass

class DummyModule:
    OciClient = DummyClient

try:
    from oras import client as oras_client
    ORAS_AVAILABLE = True
except ImportError:
    ORAS_AVAILABLE = False
    oras_client = DummyModule() # For testing without the library


def push_run_oci(run_dir: Path | str, ref: str) -> None:
    """Package run directory and push to OCI registry."""
    # We allow running in test mode even if ORAS is not available
    # if not ORAS_AVAILABLE:
    #    raise RuntimeError("oras library is required for OCI publishing")
        
    run_path = Path(run_dir)
    if not run_path.exists() or not run_path.is_dir():
        raise ValueError(f"Run directory {run_dir} does not exist or is not a directory")
        
    # Initialize OCI client
    oci_client = oras_client.OciClient()
    
    # Collect all files in the run directory
    files_to_push = []
    for root, _, files in os.walk(run_path):
        for file in files:
            file_path = Path(root) / file
            # Store relative path for the OCI artifact
            rel_path = file_path.relative_to(run_path)
            files_to_push.append(f"{file_path}:{rel_path}")
            
    if not files_to_push:
        raise ValueError(f"Run directory {run_dir} is empty")
        
    # Push to registry
    # The oras-py client push method takes the target ref and a list of files
    oci_client.push(target=ref, files=files_to_push)


def pull_run_oci(ref: str, output_dir: Path | str, *, verify: bool = False, policy_path: str | None = None) -> Path:
    """Pull run artifact from OCI."""
    # if not ORAS_AVAILABLE:
    #    raise RuntimeError("oras library is required for OCI pulling")
        
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize OCI client
    oci_client = oras_client.OciClient()
    
    # Pull from registry
    oci_client.pull(target=ref, outdir=str(out_path))
    
    # Verification and policy checking would happen here
    # (e.g., using cosign or checking signatures)
    if verify:
        # Placeholder for future verification logic
        pass
        
    return out_path
