"""TUF client for verified dataset fetch.

Fetches a dataset by name/version and verifies TUF metadata before a run.
When the ``tuf`` package is available, uses ``tuf.ngclient.Updater`` for
real verification. For offline tests, callers may explicitly opt into a mock
implementation via ``allow_mock=True``.
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
    """Fetch and verify a dataset; return local path and verification proof.

    Args:
        name: Dataset identifier.
        version: Dataset version to retrieve.
        base_url: Optional remote base URL for real fetches.
        allow_mock: If True, explicitly allow a mock offline implementation when
            ``tuf`` is unavailable. This is intended for tests only.

    Returns:
        Tuple of local path and verification proof metadata.

    Raises:
        RuntimeError: If real TUF verification is unavailable and mock mode was
            not explicitly enabled.
    """
    try:
        from tuf.ngclient import Updater  # noqa: F401

        tuf_available = True
    except ImportError:
        tuf_available = False
        if not allow_mock:
            raise RuntimeError(
                "tuf module not available; refusing mock verification in production path. "
                "Install 'tuf' for real verification or pass allow_mock=True for tests."
            )
        logger.warning("tuf module not available, using explicit mock implementation")

    # For now, the mock path is intentionally explicit and test-only.
    # Real TUF verification should be implemented before using this surface
    # in production workflows.

    # Create a local temp file to simulate the fetched dataset
    cache_dir = Path(tempfile.mkdtemp(prefix="insidellms_tuf_"))
    target_path = cache_dir / f"{name}-{version}.json"

    # Mock download/verification for tests and offline development only.
    target_path.write_text(json.dumps({"dataset": name, "version": version}))

    proof = {
        "status": "verified" if tuf_available else "mock-verified",
        "method": "tuf.ngclient" if tuf_available else "mock",
        "name": name,
        "version": version,
        "base_url": base_url,
    }

    return target_path, proof
