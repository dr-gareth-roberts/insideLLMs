"""Configuration loading and object creation utilities.

This module provides utilities for loading experiment configurations from
YAML/JSON files and creating model, probe, and dataset instances from
configuration dictionaries.
"""

import hashlib
import json
import logging
import os
import posixpath
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from insideLLMs.models.base import Model
from insideLLMs.probes.base import Probe
from insideLLMs.registry import (
    NotFoundError,
    dataset_registry,
    ensure_builtins_registered,
    model_registry,
    probe_registry,
)
from insideLLMs.types import ConfigDict

logger = logging.getLogger(__name__)


def load_config(path: Union[str, Path]) -> ConfigDict:
    """Load an experiment configuration file.

    Reads and parses a YAML or JSON configuration file that defines model,
    probe, and dataset settings for an experiment.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the configuration file. Must have .yaml, .yml, or .json extension.

    Returns
    -------
    ConfigDict
        The parsed configuration dictionary containing model, probe, and
        dataset definitions.

    Raises
    ------
    ValueError
        If the file extension is not supported (.yaml, .yml, or .json).
    FileNotFoundError
        If the configuration file doesn't exist.

    Examples
    --------
    Load a YAML configuration:

        >>> from insideLLMs.runtime.runner import load_config
        >>>
        >>> config = load_config("experiment.yaml")
        >>> print(config["model"]["type"])
        'openai'

    Load a JSON configuration:

        >>> config = load_config("experiment.json")
        >>> print(config["probe"]["type"])
        'factuality'

    Check configuration contents:

        >>> config = load_config("config.yaml")
        >>> for key in config:
        ...     print(f"{key}: {type(config[key])}")
        model: <class 'dict'>
        probe: <class 'dict'>
        dataset: <class 'dict'>

    See Also
    --------
    run_experiment_from_config : Load and run an experiment in one step.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix in (".yaml", ".yml"):
        with open(path) as f:
            data = yaml.safe_load(f)
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {path}: expected a mapping at top level.")
    return data


def _resolve_path(path: str, base_dir: Path) -> Path:
    """Resolve a path relative to a base directory.

    Parameters
    ----------
    path : str
        The path to resolve. Can contain environment variables and ~.
    base_dir : Path
        The base directory for relative paths.

    Returns
    -------
    Path
        The resolved absolute path.
    """
    expanded = os.path.expandvars(os.path.expanduser(path))
    p = Path(expanded)
    if p.is_absolute():
        return p
    return base_dir / p


def _build_resolved_config_snapshot(config: ConfigDict, base_dir: Path) -> dict[str, Any]:
    """Build a reproducibility snapshot of the config actually used for the run.

    This is intended for writing to `config.resolved.yaml` in a run directory.
    We keep the structure of the original config, but resolve any relative
    filesystem paths that insideLLMs resolves at runtime (e.g. CSV/JSONL dataset
    paths).

    Parameters
    ----------
    config : ConfigDict
        The original configuration dictionary.
    base_dir : Path
        The base directory for resolving relative paths.

    Returns
    -------
    dict[str, Any]
        A deep copy of the config with paths resolved and dataset hashes computed.
    """
    import copy

    snapshot: dict[str, Any] = copy.deepcopy(config)  # type: ignore[arg-type]  # Config can be dict-like or Mapping

    dataset = snapshot.get("dataset") if isinstance(snapshot, dict) else None
    if isinstance(dataset, dict):
        fmt = dataset.get("format")
        if fmt in ("csv", "jsonl") and isinstance(dataset.get("path"), str):
            # Keep paths deterministic and portable: avoid baking absolute checkout paths
            # into the snapshot (and therefore the run_id). Only normalize relative paths
            # to a canonical posix form unless content hashing succeeds.
            raw_path = dataset["path"]
            normalized = posixpath.normpath(str(raw_path).replace("\\", "/"))
            resolved_path = _resolve_path(normalized, base_dir)

            # Prefer content-addressing over path-addressing for run_id stability.
            # If the user didn't provide an explicit hash, compute a deterministic
            # SHA-256 over the dataset file bytes.
            if dataset.get("hash") is None and dataset.get("dataset_hash") is None:
                try:
                    hasher = hashlib.sha256()
                    with open(resolved_path, "rb") as fp:
                        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                            hasher.update(chunk)
                    dataset["dataset_hash"] = f"sha256:{hasher.hexdigest()}"
                except (IOError, OSError) as e:
                    logger.warning(
                        f"Failed to hash dataset file '{dataset.get('path')}': {e}. "
                        "Run ID will not include dataset content hash."
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error hashing dataset '{dataset.get('path')}': {e}",
                        exc_info=True,
                    )

            dataset_hash = dataset.get("hash") or dataset.get("dataset_hash")
            path_value = normalized
            if Path(normalized).is_absolute():
                try:
                    rel = resolved_path.resolve().relative_to(base_dir.resolve())
                    path_value = rel.as_posix()
                except Exception:
                    if dataset_hash:
                        path_value = Path(normalized).name
            dataset["path"] = posixpath.normpath(path_value)
        elif fmt == "hf":
            if (
                dataset.get("dataset_hash") is None
                and dataset.get("hash") is None
                and dataset.get("revision") is None
            ):
                logger.warning(
                    "HuggingFace dataset config missing revision or dataset_hash; "
                    "run_id may not be stable if the dataset changes."
                )

    return snapshot


def _resolve_determinism_options(
    config: Any,
    *,
    strict_override: Optional[bool],
    deterministic_artifacts_override: Optional[bool],
) -> tuple[bool, bool]:
    """Resolve determinism controls from config plus explicit overrides.

    Parameters
    ----------
    config : Any
        The configuration dictionary.
    strict_override : Optional[bool]
        Explicit override for strict_serialization.
    deterministic_artifacts_override : Optional[bool]
        Explicit override for deterministic_artifacts.

    Returns
    -------
    tuple[bool, bool]
        Tuple of (strict_serialization, deterministic_artifacts).
    """
    cfg_strict = True
    cfg_artifacts: Optional[bool] = None

    if isinstance(config, dict):
        det = config.get("determinism")
        if isinstance(det, dict):
            if "strict_serialization" in det:
                raw_strict = det.get("strict_serialization")
                if raw_strict is not None and not isinstance(raw_strict, bool):
                    raise ValueError(
                        "determinism.strict_serialization must be a bool or null/None, "
                        f"got {type(raw_strict).__name__}"
                    )
                if isinstance(raw_strict, bool):
                    cfg_strict = raw_strict
            if "deterministic_artifacts" in det:
                raw_artifacts = det.get("deterministic_artifacts")
                if raw_artifacts is not None and not isinstance(raw_artifacts, bool):
                    raise ValueError(
                        "determinism.deterministic_artifacts must be a bool or null/None, "
                        f"got {type(raw_artifacts).__name__}"
                    )
                cfg_artifacts = raw_artifacts

    strict_serialization = strict_override if strict_override is not None else cfg_strict
    deterministic_artifacts = (
        deterministic_artifacts_override
        if deterministic_artifacts_override is not None
        else cfg_artifacts
    )
    if deterministic_artifacts is None:
        deterministic_artifacts = strict_serialization
    return strict_serialization, deterministic_artifacts


def _extract_probe_kwargs_from_config(config: Any) -> dict[str, Any]:
    """Extract `probe.run(...)` kwargs from a config dict.

    We support a few synonymous keys so users can express generation settings in a
    more natural way:

    - `generation`: preferred name for model generation parameters
    - `probe_kwargs`: explicit name matching ProbeRunner.run(**probe_kwargs)
    - `run_kwargs`: legacy/alternate name

    If multiple keys are present, later keys override earlier ones.

    Parameters
    ----------
    config : Any
        The configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Merged probe kwargs from all supported keys.
    """
    if not isinstance(config, dict):
        return {}

    merged: dict[str, Any] = {}
    for key in ("generation", "probe_kwargs", "run_kwargs"):
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be a mapping/dict, got {type(value).__name__}")
        merged.update(value)
    return merged


def _create_middlewares_from_config(items: Any) -> list[Any]:
    """Create middleware instances from configuration.

    Parameters
    ----------
    items : Any
        List of middleware configurations. Each item can be a string (middleware type)
        or a dict with 'type' and optional 'args'.

    Returns
    -------
    list[Any]
        List of instantiated middleware objects.

    Raises
    ------
    ValueError
        If the items is not a list or contains invalid middleware configs.
    """
    if not items:
        return []
    if not isinstance(items, list):
        raise ValueError("pipeline.middlewares must be a list")

    from insideLLMs.pipeline import (
        CacheMiddleware,
        CostTrackingMiddleware,
        PassthroughMiddleware,
        RateLimitMiddleware,
        RetryMiddleware,
        TraceMiddleware,
    )

    def normalize(name: str) -> str:
        return name.strip().lower().replace("-", "_")

    from insideLLMs.runtime.receipt import ReceiptMiddleware

    middleware_map = {
        "receipt": ReceiptMiddleware,
        "receiptmiddleware": ReceiptMiddleware,
        "cache": CacheMiddleware,
        "cachemiddleware": CacheMiddleware,
        "rate_limit": RateLimitMiddleware,
        "ratelimit": RateLimitMiddleware,
        "ratelimitmiddleware": RateLimitMiddleware,
        "retry": RetryMiddleware,
        "retrymiddleware": RetryMiddleware,
        "cost": CostTrackingMiddleware,
        "cost_tracking": CostTrackingMiddleware,
        "costtracking": CostTrackingMiddleware,
        "costtrackingmiddleware": CostTrackingMiddleware,
        "trace": TraceMiddleware,
        "tracemiddleware": TraceMiddleware,
        "passthrough": PassthroughMiddleware,
        "passthroughmiddleware": PassthroughMiddleware,
    }

    middlewares: list[Any] = []
    for item in items:
        if isinstance(item, str):
            mw_type = normalize(item)
            mw_args: dict[str, Any] = {}
        elif isinstance(item, dict):
            raw_type = item.get("type") or item.get("name")
            if not raw_type:
                raise ValueError("Middleware config missing 'type'")
            mw_type = normalize(str(raw_type))
            mw_args = item.get("args") or {}
        else:
            raise ValueError(f"Unsupported middleware config entry: {item!r}")

        mw_cls = middleware_map.get(mw_type)
        if mw_cls is None:
            raise ValueError(f"Unknown middleware type: {mw_type}")
        middlewares.append(mw_cls(**mw_args))

    return middlewares


def _create_model_from_config(
    config: ConfigDict,
    *,
    prefer_async_pipeline: bool = False,
) -> Model:
    """Create a model instance from configuration.

    Uses the registry if available, falls back to direct imports.

    Parameters
    ----------
    config : ConfigDict
        Model configuration with 'type' and optional 'args'.
    prefer_async_pipeline : bool, default False
        If True, prefer AsyncModelPipeline when wrapping with middlewares.

    Returns
    -------
    Model
        The instantiated model, possibly wrapped in a pipeline.

    Raises
    ------
    ValueError
        If the model type is unknown.
    """
    ensure_builtins_registered()

    model_type = config["type"]
    model_args = config.get("args", {})

    try:
        base_model = model_registry.get(model_type, **model_args)
    except NotFoundError:
        # Fallback to direct import for backwards compatibility
        from insideLLMs.models import (
            AnthropicModel,
            CohereModel,
            DummyModel,
            GeminiModel,
            HuggingFaceModel,
            LlamaCppModel,
            OllamaModel,
            OpenAIModel,
            VLLMModel,
        )

        model_map = {
            "dummy": DummyModel,
            "openai": OpenAIModel,
            "huggingface": HuggingFaceModel,
            "anthropic": AnthropicModel,
            "gemini": GeminiModel,
            "cohere": CohereModel,
            "llamacpp": LlamaCppModel,
            "ollama": OllamaModel,
            "vllm": VLLMModel,
        }

        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")

        base_model = model_map[model_type](**model_args)

    pipeline_cfg = config.get("pipeline") if isinstance(config, dict) else None
    if isinstance(pipeline_cfg, dict):
        from insideLLMs.pipeline import AsyncModelPipeline, ModelPipeline

        middlewares_cfg = pipeline_cfg.get("middlewares")
        if middlewares_cfg is None:
            middlewares_cfg = pipeline_cfg.get("middleware", [])
        middlewares = _create_middlewares_from_config(middlewares_cfg)
        if middlewares:
            pipeline_async = pipeline_cfg.get("async")
            if pipeline_async is None:
                pipeline_async = prefer_async_pipeline
            pipeline_name = pipeline_cfg.get("name")
            if pipeline_async:
                return AsyncModelPipeline(base_model, middlewares=middlewares, name=pipeline_name)
            return ModelPipeline(base_model, middlewares=middlewares, name=pipeline_name)

    return base_model


def _create_probe_from_config(config: ConfigDict) -> Probe:
    """Create a probe instance from configuration.

    Uses the registry if available, falls back to direct imports.

    Parameters
    ----------
    config : ConfigDict
        Probe configuration with 'type' and optional 'args'.

    Returns
    -------
    Probe
        The instantiated probe.

    Raises
    ------
    ValueError
        If the probe type is unknown.
    """
    ensure_builtins_registered()

    probe_type = config["type"]
    probe_args = config.get("args", {})

    try:
        return probe_registry.get(probe_type, **probe_args)
    except NotFoundError:
        # Fallback to direct import for backwards compatibility
        from insideLLMs.probes import (
            AttackProbe,
            BiasProbe,
            CodeDebugProbe,
            CodeExplanationProbe,
            CodeGenerationProbe,
            ConstraintComplianceProbe,
            FactualityProbe,
            InstructionFollowingProbe,
            JailbreakProbe,
            LogicProbe,
            MultiStepTaskProbe,
            PromptInjectionProbe,
        )

        probe_map = {
            "logic": LogicProbe,
            "bias": BiasProbe,
            "attack": AttackProbe,
            "factuality": FactualityProbe,
            "prompt_injection": PromptInjectionProbe,
            "jailbreak": JailbreakProbe,
            "code_generation": CodeGenerationProbe,
            "code_explanation": CodeExplanationProbe,
            "code_debug": CodeDebugProbe,
            "instruction_following": InstructionFollowingProbe,
            "multi_step_task": MultiStepTaskProbe,
            "constraint_compliance": ConstraintComplianceProbe,
        }

        if probe_type not in probe_map:
            raise ValueError(f"Unknown probe type: {probe_type}") from None

        return probe_map[probe_type](**probe_args)


def _load_dataset_from_config(config: ConfigDict, base_dir: Path) -> list[Any]:
    """Load a dataset from configuration.

    Parameters
    ----------
    config : ConfigDict
        Dataset configuration with 'format' and format-specific options.
    base_dir : Path
        Base directory for resolving relative paths.

    Returns
    -------
    list[Any]
        The loaded dataset as a list of examples.

    Raises
    ------
    ValueError
        If the dataset format is unknown.
    """
    ensure_builtins_registered()

    format_type = config["format"]

    if format_type in ("csv", "jsonl"):
        path = _resolve_path(config["path"], base_dir)
        try:
            loader = dataset_registry.get_factory(format_type)
            return loader(str(path))
        except NotFoundError:
            from insideLLMs.dataset_utils import load_csv_dataset, load_jsonl_dataset

            if format_type == "csv":
                return load_csv_dataset(str(path))
            else:
                return load_jsonl_dataset(str(path))

    elif format_type == "hf":
        excluded_keys = {
            "format",
            "name",
            "split",
            "path",
            "dataset",
            "hash",
            "dataset_hash",
            "version",
            "dataset_version",
            "provenance",
            "source",
        }
        extra_kwargs = {k: v for k, v in config.items() if k not in excluded_keys}
        try:
            loader = dataset_registry.get_factory("hf")
            return loader(config["name"], split=config.get("split", "test"), **extra_kwargs)
        except NotFoundError:
            from insideLLMs.dataset_utils import load_hf_dataset

            dataset = load_hf_dataset(
                config["name"], split=config.get("split", "test"), **extra_kwargs
            )
            if dataset is None:
                raise ValueError(f"Failed to load HuggingFace dataset: {config['name']}")
            return dataset

    else:
        raise ValueError(f"Unknown dataset format: {format_type}")


__all__ = [
    "load_config",
    "_resolve_path",
    "_build_resolved_config_snapshot",
    "_resolve_determinism_options",
    "_extract_probe_kwargs_from_config",
    "_create_middlewares_from_config",
    "_create_model_from_config",
    "_create_probe_from_config",
    "_load_dataset_from_config",
]
