"""Constants for output schema versioning."""

from __future__ import annotations

# SemVer for the output contract(s) shipped by the library.
#
# NOTE: Schemas are intentionally strict (extra fields forbidden). Any additions
# to serialized outputs must bump this version and be reflected in the registry.
DEFAULT_SCHEMA_VERSION = "1.0.1"
