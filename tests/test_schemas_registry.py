"""Tests for insideLLMs/schemas/registry.py module."""

import importlib.util

import pytest

from insideLLMs.schemas.registry import (
    SchemaRegistry,
    normalize_semver,
)


def _has_pydantic_v2() -> bool:
    if importlib.util.find_spec("pydantic") is None:
        return False
    import pydantic

    return hasattr(pydantic.BaseModel, "model_validate")


class TestNormalizeSemver:
    """Tests for normalize_semver function."""

    def test_full_semver_unchanged(self):
        """Test full semver is unchanged."""
        assert normalize_semver("1.0.0") == "1.0.0"
        assert normalize_semver("2.3.4") == "2.3.4"

    def test_two_part_version_padded(self):
        """Test two-part versions get .0 appended."""
        assert normalize_semver("1.0") == "1.0.0"
        assert normalize_semver("2.3") == "2.3.0"

    def test_invalid_version_unchanged(self):
        """Test invalid versions pass through unchanged."""
        assert normalize_semver("invalid") == "invalid"
        assert normalize_semver("v1.0") == "v1.0"


class TestSchemaRegistry:
    """Tests for SchemaRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry instance."""
        return SchemaRegistry()

    def test_available_versions_known_schema(self, registry):
        """Test available_versions for known schemas."""
        versions = registry.available_versions(SchemaRegistry.RESULT_RECORD)
        assert "1.0.0" in versions
        assert "1.0.1" in versions
        explain_versions = registry.available_versions(SchemaRegistry.HARNESS_EXPLAIN)
        assert "1.0.0" in explain_versions
        assert "1.0.1" in explain_versions
        trace_versions = registry.available_versions(SchemaRegistry.CUSTOM_TRACE)
        assert trace_versions == ["insideLLMs.custom.trace@1"]

    def test_available_versions_unknown_schema(self, registry):
        """Test available_versions for unknown schema returns empty."""
        versions = registry.available_versions("UnknownSchema")
        assert versions == []

    @pytest.mark.skipif(
        importlib.util.find_spec("pydantic") is None, reason="pydantic not installed"
    )
    def test_get_model_caching(self, registry):
        """Test that get_model caches models."""
        model1 = registry.get_model(SchemaRegistry.RESULT_RECORD, "1.0.0")
        model2 = registry.get_model(SchemaRegistry.RESULT_RECORD, "1.0.0")
        assert model1 is model2

    @pytest.mark.skipif(
        importlib.util.find_spec("pydantic") is None, reason="pydantic not installed"
    )
    def test_get_model_version_normalization(self, registry):
        """Test that get_model normalizes versions."""
        model1 = registry.get_model(SchemaRegistry.RESULT_RECORD, "1.0")
        model2 = registry.get_model(SchemaRegistry.RESULT_RECORD, "1.0.0")
        assert model1 is model2

    @pytest.mark.skipif(not _has_pydantic_v2(), reason="pydantic v2 not installed")
    def test_get_model_custom_trace(self, registry):
        """Test that get_model resolves the custom trace bundle."""
        model = registry.get_model(
            SchemaRegistry.CUSTOM_TRACE,
            "insideLLMs.custom.trace@1",
        )
        assert model.__name__ == "TraceBundleV1"

    @pytest.mark.skipif(
        importlib.util.find_spec("pydantic") is None, reason="pydantic not installed"
    )
    def test_get_model_unknown_version_raises(self, registry):
        """Test that get_model raises for unknown versions."""
        with pytest.raises(KeyError, match="Unknown schema version"):
            registry.get_model(SchemaRegistry.RESULT_RECORD, "9.9.9")

    @pytest.mark.skipif(
        importlib.util.find_spec("pydantic") is None, reason="pydantic not installed"
    )
    def test_get_json_schema(self, registry):
        """Test get_json_schema returns valid schema."""
        schema = registry.get_json_schema(SchemaRegistry.RESULT_RECORD, "1.0.0")
        assert "title" in schema
        assert SchemaRegistry.RESULT_RECORD in schema["title"]

    def test_migrate_same_version(self, registry):
        """Test migrate with same version returns data unchanged."""
        data = {"key": "value"}
        result = registry.migrate(
            SchemaRegistry.RESULT_RECORD, data, from_version="1.0.0", to_version="1.0.0"
        )
        assert result == data

    def test_migrate_normalized_same_version(self, registry):
        """Test migrate normalizes versions before comparing."""
        data = {"key": "value"}
        result = registry.migrate(
            SchemaRegistry.RESULT_RECORD, data, from_version="1.0", to_version="1.0.0"
        )
        assert result == data

    def test_migrate_with_custom_function(self, registry):
        """Test migrate applies custom_migration function."""
        data = {"key": "value"}

        def custom_migrate(d):
            return {**d, "migrated": True}

        result = registry.migrate(
            SchemaRegistry.RESULT_RECORD,
            data,
            from_version="1.0.0",
            to_version="1.0.0",
            custom_migration=custom_migrate,
        )
        assert result == {"key": "value", "migrated": True}

    def test_migrate_different_versions_raises(self, registry):
        """Test migrate raises for unsupported version transitions."""
        data = {"key": "value"}
        with pytest.raises(NotImplementedError, match="No migration registered"):
            registry.migrate(
                SchemaRegistry.RESULT_RECORD, data, from_version="1.0.0", to_version="2.0.0"
            )


class TestSchemaRegistryConstants:
    """Tests for SchemaRegistry constants."""

    def test_canonical_names_defined(self):
        """Test all canonical schema names are defined."""
        assert SchemaRegistry.RUNNER_ITEM == "ProbeResult"
        assert SchemaRegistry.RUNNER_OUTPUT == "RunnerOutput"
        assert SchemaRegistry.RESULT_RECORD == "ResultRecord"
        assert SchemaRegistry.RUN_MANIFEST == "RunManifest"
        assert SchemaRegistry.HARNESS_RECORD == "HarnessRecord"
        assert SchemaRegistry.HARNESS_SUMMARY == "HarnessSummary"
        assert SchemaRegistry.HARNESS_EXPLAIN == "HarnessExplain"
        assert SchemaRegistry.BENCHMARK_SUMMARY == "BenchmarkSummary"
        assert SchemaRegistry.COMPARISON_REPORT == "ComparisonReport"
        assert SchemaRegistry.DIFF_REPORT == "DiffReport"
        assert SchemaRegistry.EXPORT_METADATA == "ExportMetadata"
        assert SchemaRegistry.CUSTOM_TRACE == "CustomTrace"
