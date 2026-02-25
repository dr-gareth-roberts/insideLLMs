"""TUF client for verified dataset fetch (stub).

Fetch dataset by name/version and verify TUF metadata before run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def fetch_dataset(name: str, version: str, *, base_url: str = "") -> tuple[Path, dict[str, Any]]:
    """Fetch and verify dataset; return local path and verification proof (stub)."""
    raise NotImplementedError("TUF client not configured")
