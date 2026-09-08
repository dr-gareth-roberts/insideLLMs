"""Version 1 configuration shared by the CLI and execution runtime.

Provider and probe constructor arguments intentionally remain open mappings:
registries can supply third-party implementations. Execution options are closed
so a misspelled or unsupported setting cannot silently change the experiment.
"""

from __future__ import annotations

import copy
import os
import warnings
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BudgetRequestPolicy(BaseModel):
    """Caller-supplied upper bounds and fixed prices for one provider/model.

    maximum_input_tokens is the provider's maximum BILLABLE input context, not
    a prompt estimate. Its correctness and the supplied rates are assumptions
    of admission control; insideLLMs does not attest provider billing policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    provider: Literal["openai", "anthropic"]
    model: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    pricing_id: str = Field(min_length=1)
    input_cost_per_million: Decimal = Field(
        ge=0, allow_inf_nan=False, max_digits=24, decimal_places=12
    )
    output_cost_per_million: Decimal = Field(
        ge=0, allow_inf_nan=False, max_digits=24, decimal_places=12
    )
    maximum_input_tokens: int = Field(gt=0, le=1000000000, strict=True)
    maximum_output_tokens: int = Field(gt=0, le=1000000000, strict=True)
    output_token_parameter: Literal["max_tokens", "max_completion_tokens"]

    @model_validator(mode="after")
    def validate_endpoint_and_cap(self) -> BudgetRequestPolicy:
        expected = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com",
        }[self.provider]
        if self.endpoint != expected:
            raise ValueError("Budget admission supports only the exact standard provider endpoint")
        if self.provider == "anthropic" and self.output_token_parameter != "max_tokens":
            raise ValueError("Anthropic requires the max_tokens output bound")
        return self


class BudgetPolicy(BaseModel):
    """An opt-in, immutable allowance for one process invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    currency: Literal["USD"]
    allowance: Decimal = Field(ge=0, allow_inf_nan=False, max_digits=24, decimal_places=12)
    scope: Literal["invocation"] = "invocation"
    prices: tuple[BudgetRequestPolicy, ...]

    @model_validator(mode="after")
    def require_unique_prices(self) -> BudgetPolicy:
        identities = [(price.provider, price.model) for price in self.prices]
        if len(identities) != len(set(identities)):
            raise ValueError("Budget prices must uniquely identify each provider and model")
        return self


class ComponentConfig(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True, extra="forbid")
    type: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    pipeline: dict[str, Any] | None = None
    generation: dict[str, Any] | None = None
    probe_kwargs: dict[str, Any] | None = None
    run_kwargs: dict[str, Any] | None = None
    id: str | None = None


class RuntimeDatasetConfig(BaseModel):
    # Only HuggingFace forwards arbitrary provider-specific loader options.
    model_config = ConfigDict(hide_input_in_errors=True, extra="allow")
    format: Literal["csv", "jsonl", "hf", "inline"]
    path: str | None = None
    name: str | None = None
    data: list[Any] | None = None

    @model_validator(mode="after")
    def require_source(self) -> RuntimeDatasetConfig:
        if self.format != "hf":
            metadata_fields = {
                "dataset",
                "hash",
                "dataset_hash",
                "version",
                "dataset_version",
                "provenance",
            }
            unsupported = set(self.model_extra or {}) - metadata_fields
            if unsupported:
                raise ValueError(
                    f"Unsupported {self.format} dataset settings: {sorted(unsupported)}; "
                    "use max_examples to limit execution or prepare the dataset explicitly"
                )
        if self.format in {"csv", "jsonl"} and not self.path:
            raise ValueError("dataset.path is required for file datasets")
        if self.format == "hf" and not self.name:
            raise ValueError("dataset.name is required for HuggingFace datasets")
        if self.format == "inline" and self.data is None:
            raise ValueError("dataset.data is required for inline datasets")
        return self


class RuntimeRunnerConfig(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True, extra="forbid")
    concurrency: int = Field(default=5, ge=1)
    timeout: float | None = Field(default=None, gt=0)
    stop_on_error: bool = False
    use_probe_batch: bool = False
    batch_workers: int | None = Field(default=None, ge=1)


class DeterminismConfig(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True, extra="forbid", strict=True)
    strict_serialization: bool | None = True
    deterministic_artifacts: bool | None = None


class RuntimeConfiguration(BaseModel):
    """Canonical experiment or harness configuration; use one component shape."""

    model_config = ConfigDict(hide_input_in_errors=True, extra="forbid", protected_namespaces=())
    config_version: Literal["1"] = "1"
    model: ComponentConfig | None = None
    probe: ComponentConfig | None = None
    models: list[ComponentConfig] | None = None
    probes: list[ComponentConfig] | None = None
    dataset: RuntimeDatasetConfig
    runner: RuntimeRunnerConfig | None = None
    budget: BudgetPolicy | None = None
    determinism: DeterminismConfig | None = None
    generation: dict[str, Any] | None = None
    probe_kwargs: dict[str, Any] | None = None
    run_kwargs: dict[str, Any] | None = None
    max_examples: int | None = Field(default=None, gt=0)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    output_dir: str | None = None
    report_title: str | None = None
    name: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    compliance_profile: str | dict[str, Any] | None = None

    @model_validator(mode="after")
    def require_components(self) -> RuntimeConfiguration:
        for model in [self.model] if self.model is not None else self.models or []:
            unsupported = model.model_fields_set & {
                "generation",
                "probe_kwargs",
                "run_kwargs",
                "id",
            }
            if unsupported:
                raise ValueError(
                    f"Unsupported model settings: {sorted(unsupported)}; "
                    "put generation/probe_kwargs/run_kwargs at the top level or on a probe"
                )
        for probe in [self.probe] if self.probe is not None else self.probes or []:
            unsupported = probe.model_fields_set & {"pipeline", "id"}
            if unsupported:
                raise ValueError(
                    f"Unsupported probe settings: {sorted(unsupported)}; "
                    "configure pipeline on the model and name in probe.args"
                )
        single = self.model is not None or self.probe is not None
        harness = self.models is not None or self.probes is not None
        if single and harness:
            raise ValueError("Use model/probe or models/probes, not both")
        if single and self.model is not None and self.probe is not None:
            return self
        if harness and self.models and self.probes:
            return self
        if harness and not self.models:
            raise ValueError("Harness config requires at least one model in 'models'.")
        if harness and not self.probes:
            raise ValueError("Harness config requires at least one probe in 'probes'.")
        raise ValueError("Config requires model and probe, or nonempty models and probes")


def resolve_dataset_path(path: str, base_dir: Path) -> Path:
    """Resolve dataset paths consistently relative to their configuration file."""
    expanded = Path(os.path.expandvars(os.path.expanduser(path)))
    return expanded if expanded.is_absolute() else base_dir / expanded


def _convert_legacy_config(data: dict[str, Any]) -> dict[str, Any]:
    """Translate the old public builder shape without resolving credentials."""
    model = data.get("model")
    if not isinstance(model, dict) or "provider" not in model:
        return data
    warnings.warn(
        "provider/model_id/source configuration is deprecated; use "
        "config_version: '1', model.type/args and dataset.format instead",
        DeprecationWarning,
        stacklevel=3,
    )
    provider = model["provider"]
    unknown_model = set(model) - {
        "provider",
        "model_id",
        "name",
        "api_key_env",
        "api_base",
        "temperature",
        "max_tokens",
        "timeout",
        "max_retries",
        "extra_params",
    }
    if unknown_model:
        raise ValueError(f"Unsupported legacy model settings: {sorted(unknown_model)}")
    args = {
        key: model[key]
        for key in ("name", "api_key_env", "timeout", "max_retries")
        if model.get(key) is not None
    }
    if provider == "dummy":
        args = {key: value for key, value in args.items() if key == "name"}
        args.setdefault("name", model.get("model_id", "DummyModel"))
    else:
        args["model_name"] = model.get("model_id")
    if model.get("api_base") is not None:
        args["base_url"] = model["api_base"]
    data["model"] = {"type": provider, "args": args}
    generation = dict(model.get("extra_params") or {})
    generation.update(
        {key: model[key] for key in ("temperature", "max_tokens") if model.get(key) is not None}
    )
    if generation:
        data["generation"] = {**generation, **data.get("generation", {})}
    probe = data.get("probe", {})
    unknown_probe = set(probe) - {
        "type",
        "name",
        "description",
        "params",
        "timeout_per_item",
        "stop_on_error",
    }
    if unknown_probe:
        raise ValueError(f"Unsupported legacy probe settings: {sorted(unknown_probe)}")
    probe_args = dict(probe.get("params") or {})
    if probe.get("name") is not None:
        probe_args["name"] = probe["name"]
    data["probe"] = {"type": probe.get("type"), "args": probe_args}
    dataset = data.get("dataset", {})
    source = dataset.get("source")
    if dataset.get("shuffle") or dataset.get("sample_size") or dataset.get("seed"):
        raise ValueError(
            "Legacy dataset sampling/shuffling is unsupported; prepare the dataset explicitly"
        )
    ds = {
        key: value
        for key, value in dataset.items()
        if key not in {"source", "shuffle", "sample_size", "seed"} and value is not None
    }
    ds["format"] = Path(ds.get("path", "")).suffix.lstrip(".") if source == "file" else source
    if ds["format"] != "hf" and ds.get("split") == "test":
        # The legacy DatasetConfig includes the HF default on every source.
        ds.pop("split")
    data["dataset"] = ds
    runner = data.get("runner", {})
    defaults = {
        "output_dir": "output",
        "output_formats": ["json", "markdown"],
        "save_intermediate": False,
        "progress_bar": True,
        "verbose": False,
        "cache_responses": False,
    }
    for key, default in defaults.items():
        value = runner.pop(key, default)
        if value != default:
            raise ValueError(
                f"Legacy runner.{key} is unsupported; use the corresponding CLI option"
            )
    if "stop_on_error" in probe:
        runner["stop_on_error"] = probe["stop_on_error"]
    if probe.get("timeout_per_item") not in (None, 30.0):
        raise ValueError(
            "Legacy probe.timeout_per_item is unsupported; use runner.timeout with run --async"
        )
    if runner:
        data["runner"] = runner
    return data


def normalize_runtime_config(
    data: Mapping[str, Any], *, check_registry: bool = False
) -> dict[str, Any]:
    """Validate and normalize a config, preserving omitted defaults and paths.

    Does not initialize models, import provider SDKs, resolve keys, or read data.
    Runtime loading and ``validate`` use the same validation boundary.
    """
    if not isinstance(data, Mapping):
        raise ValueError("Configuration must be a mapping")
    normalized = _convert_legacy_config(copy.deepcopy(dict(data)))
    parsed = RuntimeConfiguration.model_validate(normalized)
    result = parsed.model_dump(exclude_unset=True)
    if parsed.budget is not None:
        # Fixed prices remain decimal strings in YAML/JSON and deterministic IDs.
        result["budget"] = parsed.budget.model_dump(mode="json")
    if check_registry:
        from insideLLMs.registry import (
            ensure_builtins_registered,
            model_registry,
            probe_registry,
        )

        ensure_builtins_registered()
        for singular, plural, registry in (
            ("model", "models", model_registry),
            ("probe", "probes", probe_registry),
        ):
            entries = result.get(plural) or [result[singular]]
            for entry in entries:
                if entry["type"] not in registry.list():
                    raise ValueError(f"Unknown {singular} type: {entry['type']}")
    return result
