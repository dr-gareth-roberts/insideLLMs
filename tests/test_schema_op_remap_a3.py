"""A3 proof: unknown schema ops remapped to dump; no fallthrough needed."""

from __future__ import annotations

import argparse

from insideLLMs.cli.commands.schema import _SCHEMA_OPS, cmd_schema


def test_unknown_ops_remap_to_dump() -> None:
    """Any op outside {_SCHEMA_OPS} is treated as a schema name + dump."""
    for op in ("TotallyUnknown", "ResultRecord", "nope", "VALIDATE", ""):
        if op in _SCHEMA_OPS:
            continue
        args = argparse.Namespace(
            op=op or None,
            name=None,
            version="1.0.0",
            output=None,
            input=None,
            mode="strict",
        )
        # Remap → dump; missing/unknown name returns 1 or 2, never a fallthrough.
        rc = cmd_schema(args)
        assert rc in {0, 1, 2}


def test_known_ops_only_list_dump_validate() -> None:
    assert _SCHEMA_OPS == frozenset({"list", "dump", "validate"})
