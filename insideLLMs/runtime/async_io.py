"""Async I/O utilities for non-blocking file operations.

This module provides async wrappers for file I/O operations to prevent
blocking the event loop during record writing.
"""

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def async_write_text(
    filepath: Path,
    content: str,
    mode: str = "a",
    encoding: str = "utf-8",
) -> None:
    """Write text to file asynchronously using executor.

    Args:
        filepath: Path to file
        content: Content to write
        mode: File mode ('a' for append, 'w' for write, 'x' for exclusive)
        encoding: Text encoding
    """
    loop = asyncio.get_running_loop()

    def _write():
        with open(filepath, mode, encoding=encoding) as f:
            f.write(content)
            f.flush()

    try:
        await loop.run_in_executor(None, _write)
    except (IOError, OSError) as e:
        logger.error(f"Failed to write to {filepath}: {e}", exc_info=True)
        raise


async def async_write_lines(
    filepath: Path,
    lines: list[str],
    mode: str = "a",
    encoding: str = "utf-8",
) -> None:
    """Write multiple lines to file asynchronously.

    Args:
        filepath: Path to file
        lines: Lines to write (will add newlines)
        mode: File mode
        encoding: Text encoding
    """
    loop = asyncio.get_running_loop()

    def _write():
        with open(filepath, mode, encoding=encoding) as f:
            for line in lines:
                f.write(line)
                if not line.endswith("\n"):
                    f.write("\n")
            f.flush()

    try:
        await loop.run_in_executor(None, _write)
    except (IOError, OSError) as e:
        logger.error(f"Failed to write lines to {filepath}: {e}", exc_info=True)
        raise
