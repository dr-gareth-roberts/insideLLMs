"""Tests for insideLLMs.trace_config module."""

import pytest

from insideLLMs.trace_config import (
    TraceConfig,
    TraceStoreConfig,
    TraceContractsConfig,
    TraceRedactConfig,
    OnViolationMode,
    StoreMode,
)


class TestTraceConfigStructure:
    """Tests for TraceConfig dataclass structure."""

    def test_default_config(self):
        """Test TraceConfig with defaults."""
        config = TraceConfig()
        assert config.enabled is True
        assert config.store.mode == StoreMode.FULL
        assert config.contracts.enabled is True
        assert config.on_violation.mode == OnViolationMode.RECORD

    def test_store_config_defaults(self):
        """Test TraceStoreConfig defaults."""
        store = TraceStoreConfig()
        assert store.mode == StoreMode.FULL
        assert store.max_events == 5000
        assert store.include_payloads is True
        assert store.redact.enabled is False

    def test_redact_config(self):
        """Test TraceRedactConfig."""
        redact = TraceRedactConfig(
            enabled=True,
            json_pointers=["/payload/headers/authorization"],
            replacement="<redacted>",
        )
        assert redact.enabled is True
        assert len(redact.json_pointers) == 1
