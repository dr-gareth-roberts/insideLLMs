"""Tests for insideLLMs.config_types module.

This module provides test coverage for configuration types and dataclasses.
"""

import pytest

from insideLLMs.config_types import (
    ProgressInfo,
    RunConfig,
    RunConfigBuilder,
    RunContext,
)


class TestRunConfig:
    """Tests for RunConfig dataclass."""

    def test_default_instantiation(self):
        config = RunConfig()
        assert config is not None

    def test_default_stop_on_error(self):
        config = RunConfig()
        assert config.stop_on_error is False

    def test_default_validate_output(self):
        config = RunConfig()
        assert config.validate_output is False

    def test_default_emit_run_artifacts(self):
        config = RunConfig()
        assert config.emit_run_artifacts is True

    def test_default_overwrite(self):
        config = RunConfig()
        assert config.overwrite is False

    def test_default_store_messages(self):
        config = RunConfig()
        assert config.store_messages is True

    def test_default_strict_serialization(self):
        config = RunConfig()
        assert config.strict_serialization is True

    def test_default_deterministic_artifacts(self):
        config = RunConfig()
        assert config.deterministic_artifacts is None

    def test_default_concurrency(self):
        config = RunConfig()
        assert config.concurrency == 5

    def test_default_resume(self):
        config = RunConfig()
        assert config.resume is False

    def test_custom_stop_on_error(self):
        config = RunConfig(stop_on_error=True)
        assert config.stop_on_error is True

    def test_custom_concurrency(self):
        config = RunConfig(concurrency=10)
        assert config.concurrency == 10

    def test_custom_run_id(self):
        config = RunConfig(run_id="test-run-123")
        assert config.run_id == "test-run-123"

    def test_validation_mode_strict(self):
        config = RunConfig(validation_mode="strict")
        assert config.validation_mode == "strict"

    def test_validation_mode_lenient(self):
        config = RunConfig(validation_mode="lenient")
        assert config.validation_mode == "lenient"

    def test_validation_mode_warn_alias(self):
        config = RunConfig(validation_mode="warn")
        assert config.validation_mode == "lenient"

    def test_invalid_strict_serialization_type(self):
        with pytest.raises(ValueError, match="strict_serialization must be a bool"):
            RunConfig(strict_serialization="yes")  # type: ignore[arg-type]

    def test_invalid_deterministic_artifacts_type(self):
        with pytest.raises(ValueError, match="deterministic_artifacts must be a bool or None"):
            RunConfig(deterministic_artifacts="no")  # type: ignore[arg-type]


class TestRunConfigBuilder:
    """Tests for RunConfigBuilder fluent API."""

    def test_builder_instantiation(self):
        builder = RunConfigBuilder()
        assert builder is not None

    def test_builder_builds_config(self):
        config = RunConfigBuilder().build()
        assert isinstance(config, RunConfig)

    def test_builder_with_concurrency(self):
        config = RunConfigBuilder().with_concurrency(20).build()
        assert config.concurrency == 20

    def test_builder_with_stop_on_error(self):
        config = RunConfigBuilder().with_stop_on_error(True).build()
        assert config.stop_on_error is True

    def test_builder_chaining(self):
        config = (
            RunConfigBuilder()
            .with_concurrency(15)
            .with_stop_on_error(True)
            .with_validation(True)
            .build()
        )
        assert config.concurrency == 15
        assert config.stop_on_error is True
        assert config.validate_output is True

    def test_builder_with_determinism(self):
        config = (
            RunConfigBuilder()
            .with_determinism(
                strict_serialization=True,
                deterministic_artifacts=True,
            )
            .build()
        )
        assert config.strict_serialization is True
        assert config.deterministic_artifacts is True


class TestRunContext:
    """Tests for RunContext dataclass."""

    def test_instantiation(self):
        context = RunContext(run_id="test-123")
        assert context is not None
        assert context.run_id == "test-123"


class TestProgressInfo:
    """Tests for ProgressInfo dataclass."""

    def test_create_factory(self):
        info = ProgressInfo.create(current=5, total=10, start_time=0.0)
        assert info is not None
        assert info.current == 5
        assert info.total == 10

    def test_percentage(self):
        info = ProgressInfo.create(current=5, total=10, start_time=0.0)
        assert info.percentage == 50.0

    def test_percentage_zero_total(self):
        info = ProgressInfo.create(current=0, total=0, start_time=0.0)
        assert info.percentage == 0.0
