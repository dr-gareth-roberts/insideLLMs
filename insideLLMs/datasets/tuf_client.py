"""TUF client stub for dataset fetch.

Real TUF verification is NOT implemented. :func:`fetch_dataset` refuses to
run unless the caller explicitly opts into the offline mock implementation
with ``allow_mock=True`` (intended for tests only), and the returned proof is
always labelled ``status="mock"`` / ``verified=False`` — it must never be
treated as supply-chain verification. Installing the ``tuf`` package does not
change this: until a real ``tuf.ngclient.Updater`` fetch+verify flow is wired
in, this module fails closed rather than reporting unverified data as
verified.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def fetch_dataset(
    name: str,
    version: str,
    *,
    base_url: str = "",
    allow_mock: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Fetch a dataset via the explicit offline mock; refuse otherwise.

    Args:
        name: Dataset identifier.
        version: Dataset version to retrieve.
        base_url: Optional remote base URL, recorded in the proof for
            provenance only (nothing is fetched from it).
        allow_mock: Must be True to run. Explicitly opts into the offline
            mock implementation. Intended for tests only.

    Returns:
        Tuple of local path and proof metadata. The proof always has
        ``status="mock"``, ``method="mock"``, and ``verified=False`` — no
        code path in this function performs TUF verification.

    Raises:
        RuntimeError: If ``allow_mock`` is False. Real TUF verification is
            not implemented, so there is no production-safe path through
            this function yet.
    """
    if not allow_mock:
        raise RuntimeError(
            "Real TUF verification is not implemented; refusing mock verification "
            "in production path. Pass allow_mock=True to use the offline mock "
            "implementation in tests."
        )

    logger.warning(
        "TUF verification not implemented; returning explicit mock dataset (allow_mock=True)"
    )

    # Create a local temp file to simulate the fetched dataset
    cache_dir = Path(tempfile.mkdtemp(prefix="insidellms_tuf_"))
    target_path = cache_dir / f"{name}-{version}.json"
    target_path.write_text(json.dumps({"dataset": name, "version": version}))

    proof = {
        "status": "mock",
        "method": "mock",
        "verified": False,
        "name": name,
        "version": version,
        "base_url": base_url,
    }

    return target_path, proof
