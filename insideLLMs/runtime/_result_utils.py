"""Result conversion and specification building utilities.

This module provides utilities for converting between different result
representations (ProbeResult, dict, JSONL records) and building specification
dictionaries for models, probes, and datasets.
"""

import hashlib
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from insideLLMs._serialization import (
    StrictSerializationError,
)
from insideLLMs._serialization import (
    fingerprint_value as _fingerprint_value,
)
from insideLLMs._serialization import (
    stable_json_dumps as _stable_json_dumps,
)
from insideLLMs.runtime._determinism import _replicate_key
from insideLLMs.types import (
    ModelInfo,
    ProbeResult,
    ResultStatus,
)


def _normalize_status(status: Any) -> str:
    """Normalize a status value to a string representation.

    Converts various status representations (ResultStatus enum, other Enums,
    None, or raw strings) into a consistent string format for storage.

    Parameters
    ----------
    status : Any
        The status value to normalize. Can be ResultStatus, other Enum,
        None, or string.

    Returns
    -------
    str
        Normalized status string. Returns "error" for None values,
        the enum value for Enum types, or str(status) otherwise.

    Examples
    --------
    Normalizing ResultStatus enum:

        >>> from insideLLMs.types import ResultStatus
        >>> _normalize_status(ResultStatus.SUCCESS)
        'success'
        >>> _normalize_status(ResultStatus.ERROR)
        'error'

    Handling None (treated as error):

        >>> _normalize_status(None)
        'error'

    String passthrough:

        >>> _normalize_status("success")
        'success'
        >>> _normalize_status("custom_status")
        'custom_status'
    """
    if isinstance(status, ResultStatus):
        return status.value
    if isinstance(status, Enum):
        return str(status.value)
    if status is None:
        return "error"
    return str(status)


def _normalize_info_obj_to_dict(info_obj: Any) -> dict[str, Any]:
    """Normalize model/probe info object to a dictionary.

    Converts various info object formats (dict, dataclass, pydantic model)
    to a standard dictionary representation. Used for extracting model and
    probe metadata in a format-agnostic way.

    Parameters
    ----------
    info_obj : Any
        Info object to normalize. Can be dict, dataclass, pydantic v1/v2
        model, or any other type.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of the info object. Returns empty dict
        for None or unrecognized types.

    Examples
    --------
    Normalizing a dataclass:

        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class ModelInfo:
        ...     name: str
        ...     version: str
        >>> info = ModelInfo(name="gpt-4", version="1.0")
        >>> _normalize_info_obj_to_dict(info)
        {'name': 'gpt-4', 'version': '1.0'}

    Passthrough for dict:

        >>> _normalize_info_obj_to_dict({"key": "value"})
        {'key': 'value'}

    Handling None:

        >>> _normalize_info_obj_to_dict(None)
        {}

    Notes
    -----
    Supports both pydantic v1 (.dict()) and pydantic v2 (.model_dump())
    for backward compatibility across different pydantic versions.

    See Also
    --------
    _build_model_spec : Uses this to extract model info.
    _build_probe_spec : Uses this to extract probe info.
    """
    if info_obj is None:
        return {}
    if isinstance(info_obj, dict):
        return info_obj
    if is_dataclass(info_obj) and not isinstance(info_obj, type):
        return asdict(info_obj)
    if hasattr(info_obj, "dict") and callable(getattr(info_obj, "dict")):
        # pydantic v1 compatibility
        try:
            return info_obj.dict()
        except Exception:
            return {}
    if hasattr(info_obj, "model_dump") and callable(getattr(info_obj, "model_dump")):
        # pydantic v2 compatibility
        try:
            return info_obj.model_dump()
        except Exception:
            return {}
    return {}


def _build_model_spec(model: Any) -> dict[str, Any]:
    """Build a model specification dictionary from a model object.

    Extracts key identifying information from a model to create a standardized
    specification dictionary for manifest generation and result records.

    Parameters
    ----------
    model : Any
        Model object with optional info() method or attributes.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - model_id: Identifier for the model
        - provider: Model provider/type (e.g., "openai", "anthropic")
        - params: Full info dictionary as parameters

    Examples
    --------
    Building spec from a model:

        >>> class MockModel:
        ...     def info(self):
        ...         return {"model_id": "gpt-4", "provider": "openai"}
        >>> spec = _build_model_spec(MockModel())
        >>> spec["model_id"]
        'gpt-4'
        >>> spec["provider"]
        'openai'

    Fallback to class name:

        >>> class CustomModel:
        ...     pass
        >>> spec = _build_model_spec(CustomModel())
        >>> spec["model_id"]
        'CustomModel'

    See Also
    --------
    _build_probe_spec : Build probe specification.
    _normalize_info_obj_to_dict : Normalize info objects.
    """
    info_obj: Any = {}
    try:
        info_obj = model.info() if hasattr(model, "info") else {}
    except (AttributeError, TypeError):
        info_obj = {}

    info = _normalize_info_obj_to_dict(info_obj)

    model_id = (
        info.get("model_id")
        or info.get("id")
        or info.get("name")
        or info.get("model_name")
        or getattr(model, "model_id", None)
        or getattr(model, "name", None)
        or model.__class__.__name__
    )

    provider = info.get("provider") or info.get("type") or info.get("model_type")
    provider = str(provider) if provider is not None else None
    return {"model_id": str(model_id), "provider": provider, "params": info}


def _build_probe_spec(probe: Any) -> dict[str, Any]:
    """Build a probe specification dictionary from a probe object.

    Extracts key identifying information from a probe to create a standardized
    specification dictionary for manifest generation and result records.

    Parameters
    ----------
    probe : Any
        Probe object with optional name, probe_id, version attributes.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - probe_id: Identifier for the probe
        - probe_version: Version string if available
        - params: Empty dict (reserved for future use)

    Examples
    --------
    Building spec from a probe:

        >>> class MockProbe:
        ...     name = "factuality"
        ...     version = "1.0.0"
        >>> spec = _build_probe_spec(MockProbe())
        >>> spec["probe_id"]
        'factuality'
        >>> spec["probe_version"]
        '1.0.0'

    Fallback to class name:

        >>> class CustomProbe:
        ...     pass
        >>> spec = _build_probe_spec(CustomProbe())
        >>> spec["probe_id"]
        'CustomProbe'

    See Also
    --------
    _build_model_spec : Build model specification.
    _build_dataset_spec : Build dataset specification.
    """
    probe_id = (
        getattr(probe, "name", None) or getattr(probe, "probe_id", None) or probe.__class__.__name__
    )
    probe_version = getattr(probe, "version", None) or getattr(probe, "probe_version", None)
    return {"probe_id": str(probe_id), "probe_version": probe_version, "params": {}}


def _build_dataset_spec(dataset_info: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Build a dataset specification dictionary from dataset info.

    Extracts key identifying information from dataset configuration to create
    a standardized specification dictionary for manifest generation and
    result records.

    Parameters
    ----------
    dataset_info : Optional[dict[str, Any]]
        Dataset configuration dictionary. Can contain name, path, version,
        hash, provenance, format, etc.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - dataset_id: Identifier for the dataset (from name, dataset, or path)
        - dataset_version: Version string if available
        - dataset_hash: Content hash if available
        - provenance: Source information (from provenance, source, or format)

    Examples
    --------
    Building spec from HuggingFace dataset config:

        >>> config = {"name": "cais/mmlu", "version": "1.0", "format": "hf"}
        >>> spec = _build_dataset_spec(config)
        >>> spec["dataset_id"]
        'cais/mmlu'
        >>> spec["provenance"]
        'hf'

    Building spec from local file config:

        >>> config = {"path": "/data/test.jsonl", "format": "jsonl"}
        >>> spec = _build_dataset_spec(config)
        >>> spec["dataset_id"]
        '/data/test.jsonl'

    Handling None:

        >>> spec = _build_dataset_spec(None)
        >>> spec["dataset_id"] is None
        True

    See Also
    --------
    _build_model_spec : Build model specification.
    _build_probe_spec : Build probe specification.
    """
    di = dataset_info or {}
    dataset_id = di.get("name") or di.get("dataset") or di.get("path")
    dataset_version = di.get("version") or di.get("dataset_version")
    if di.get("revision"):
        if dataset_version:
            dataset_version = f"{dataset_version}@{di.get('revision')}"
        else:
            dataset_version = str(di.get("revision"))
    if di.get("split"):
        if dataset_version:
            dataset_version = f"{dataset_version}::{di.get('split')}"
        else:
            dataset_version = str(di.get("split"))
    dataset_hash = di.get("hash") or di.get("dataset_hash")
    provenance = di.get("provenance") or di.get("source") or di.get("format")
    return {
        "dataset_id": str(dataset_id) if dataset_id is not None else None,
        "dataset_version": str(dataset_version) if dataset_version is not None else None,
        "dataset_hash": str(dataset_hash) if dataset_hash is not None else None,
        "provenance": str(provenance) if provenance is not None else None,
    }


def _result_dict_from_probe_result(
    result: ProbeResult,
    *,
    schema_version: str,
    error_type: Optional[str] = None,
) -> dict[str, Any]:
    """Convert a ProbeResult into a dictionary for storage and serialization.

    Transforms a ProbeResult dataclass into a flat dictionary suitable for
    JSONL emission and result aggregation. Handles status normalization
    and optional error type extraction from metadata.

    Parameters
    ----------
    result : ProbeResult
        The probe result to convert. Contains input, output, status,
        latency_ms, error, and metadata fields.
    schema_version : str
        Schema version string to include in the output dictionary.
    error_type : Optional[str], default None
        Explicit error type to include. If None and result has an error,
        attempts to extract error_type from metadata.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - schema_version: The provided schema version
        - input: The original input from the result
        - output: The probe output
        - latency_ms: Execution time in milliseconds
        - status: Normalized status string
        - metadata: Additional metadata dict
        - error: Error message (if present)
        - error_type: Error type string (if present)

    Examples
    --------
    Converting a successful result:

        >>> from insideLLMs.types import ProbeResult, ResultStatus
        >>> result = ProbeResult(
        ...     input={"messages": [{"role": "user", "content": "Hi"}]},
        ...     output="Hello!",
        ...     status=ResultStatus.SUCCESS,
        ...     latency_ms=150.5,
        ...     metadata={"model": "gpt-4"},
        ... )
        >>> d = _result_dict_from_probe_result(result, schema_version="1.0.0")
        >>> d["status"]
        'success'
        >>> d["latency_ms"]
        150.5

    Converting an error result:

        >>> result = ProbeResult(
        ...     input="test",
        ...     status=ResultStatus.ERROR,
        ...     error="API timeout",
        ...     metadata={"error_type": "TimeoutError"},
        ... )
        >>> d = _result_dict_from_probe_result(result, schema_version="1.0.0")
        >>> d["error"]
        'API timeout'
        >>> d["error_type"]
        'TimeoutError'

    See Also
    --------
    _result_dict_from_record : Convert a stored record back to result dict.
    _build_result_record : Build a full JSONL record with additional context.
    """
    status = _normalize_status(result.status)
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if error_type is None:
        error_type = metadata.get("error_type")

    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "input": result.input,
        "output": result.output,
        "latency_ms": result.latency_ms,
        "status": status,
        "metadata": metadata,
    }
    if result.error is not None:
        payload["error"] = result.error
    if error_type:
        payload["error_type"] = error_type
    return payload


def _record_index_from_record(
    record: dict[str, Any], default: Optional[int] = None
) -> Optional[int]:
    """Extract the record index from a stored JSONL record.

    Retrieves the record_index from a record's custom metadata. This is used
    during resume operations to verify that existing records form a contiguous
    prefix of the original prompt set.

    Parameters
    ----------
    record : dict[str, Any]
        A JSONL record dictionary, typically read from records.jsonl.
        Expected to have a "custom" key containing a dict with "record_index".
    default : Optional[int], default None
        Value to return if record_index cannot be extracted or parsed.

    Returns
    -------
    Optional[int]
        The record index as an integer, or the default value if not found
        or not parseable.

    Examples
    --------
    Extracting index from a valid record:

        >>> record = {"custom": {"record_index": 5, "replicate_key": "abc"}}
        >>> _record_index_from_record(record)
        5

    Handling missing custom field:

        >>> record = {"input": "test", "output": "result"}
        >>> _record_index_from_record(record, default=0)
        0

    Handling invalid index value:

        >>> record = {"custom": {"record_index": "not_a_number"}}
        >>> _record_index_from_record(record, default=-1)
        -1

    See Also
    --------
    _result_dict_from_record : Extract result data from a stored record.
    """
    custom = record.get("custom")
    if isinstance(custom, dict):
        index = custom.get("record_index")
        if index is not None:
            try:
                return int(index)
            except (TypeError, ValueError):
                return default
    return default


def _result_dict_from_record(
    record: dict[str, Any],
    *,
    schema_version: str,
) -> dict[str, Any]:
    """Convert a stored JSONL record back into a result dictionary.

    Extracts the core result fields from a full JSONL record, producing a
    result dictionary compatible with the runner's internal result format.
    Used during resume operations to reconstruct results from existing records.

    Parameters
    ----------
    record : dict[str, Any]
        A JSONL record dictionary read from records.jsonl. Expected to
        contain input, output, latency_ms, status, and optionally error fields.
    schema_version : str
        Fallback schema version if not present in the record.

    Returns
    -------
    dict[str, Any]
        A result dictionary containing:

        - schema_version: From record or fallback
        - input: Original input data
        - output: Probe output
        - latency_ms: Execution time
        - status: Status string
        - metadata: Empty dict (not preserved in records)
        - error: Error message (if present)
        - error_type: Error type (if present)

    Examples
    --------
    Converting a success record:

        >>> record = {
        ...     "schema_version": "1.0.0",
        ...     "input": {"messages": [{"role": "user", "content": "Hi"}]},
        ...     "output": "Hello!",
        ...     "latency_ms": 150.0,
        ...     "status": "success",
        ... }
        >>> result = _result_dict_from_record(record, schema_version="1.0.0")
        >>> result["status"]
        'success'

    Converting an error record:

        >>> record = {
        ...     "input": "test",
        ...     "status": "error",
        ...     "error": "Connection failed",
        ...     "error_type": "ConnectionError",
        ... }
        >>> result = _result_dict_from_record(record, schema_version="1.0.0")
        >>> result["error"]
        'Connection failed'

    See Also
    --------
    _result_dict_from_probe_result : Convert ProbeResult to result dict.
    _record_index_from_record : Extract index from a record.
    """
    payload: dict[str, Any] = {
        "schema_version": record.get("schema_version", schema_version),
        "input": record.get("input"),
        "output": record.get("output"),
        "latency_ms": record.get("latency_ms"),
        "status": record.get("status"),
        "metadata": {},
    }
    if record.get("error") is not None:
        payload["error"] = record.get("error")
    if record.get("error_type"):
        payload["error_type"] = record.get("error_type")
    return payload


def _coerce_model_info(model: Any) -> ModelInfo:
    """Coerce model info to a standardized ModelInfo dataclass.

    Converts various model info formats into the canonical ModelInfo type.
    Handles models that return dicts, dataclasses, pydantic models, or
    already-canonical ModelInfo instances.

    Parameters
    ----------
    model : Any
        Model object with optional info() method.

    Returns
    -------
    ModelInfo
        Standardized ModelInfo instance with all fields populated.

    Examples
    --------
    Coercing from a model with info():

        >>> class MockModel:
        ...     def info(self):
        ...         return {"name": "gpt-4", "provider": "openai"}
        >>> info = _coerce_model_info(MockModel())
        >>> info.name
        'gpt-4'
        >>> info.provider
        'openai'

    Fallback for models without info():

        >>> class SimpleModel:
        ...     pass
        >>> info = _coerce_model_info(SimpleModel())
        >>> info.name
        'SimpleModel'

    See Also
    --------
    ModelInfo : The target dataclass type.
    _normalize_info_obj_to_dict : Convert info objects to dicts.
    """

    info_obj: Any = {}
    try:
        info_obj = model.info() or {}
    except Exception:
        info_obj = {}

    # If it's already the canonical type, keep it.
    if isinstance(info_obj, ModelInfo):
        return info_obj

    info = _normalize_info_obj_to_dict(info_obj)

    name = info.get("name") or getattr(model, "name", None) or model.__class__.__name__
    provider = (
        info.get("provider")
        or info.get("type")
        or info.get("model_type")
        or model.__class__.__name__.replace("Model", "")
    )
    model_id = (
        info.get("model_id")
        or info.get("id")
        or info.get("model_name")
        or getattr(model, "model_id", None)
        or name
    )

    extra: dict[str, Any] = {}
    if isinstance(info.get("extra"), dict):
        extra.update(info.get("extra") or {})
    for k, v in info.items():
        if k in {
            "name",
            "provider",
            "model_id",
            "max_tokens",
            "supports_streaming",
            "supports_chat",
            "extra",
        }:
            continue
        extra[k] = v

    return ModelInfo(
        name=str(name),
        provider=str(provider),
        model_id=str(model_id),
        max_tokens=info.get("max_tokens"),
        supports_streaming=bool(info.get("supports_streaming", False)),
        supports_chat=bool(info.get("supports_chat", True)),
        extra=extra,
    )


def _build_result_record(
    *,
    schema_version: str,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
    model: dict[str, Any],
    probe: dict[str, Any],
    dataset: dict[str, Any],
    item: Any,
    output: Any,
    latency_ms: Optional[float],
    store_messages: Optional[bool],
    index: int,
    status: str,
    error: Optional[Union[BaseException, str]],
    error_type: Optional[str] = None,
    strict_serialization: bool = False,
) -> dict[str, Any]:
    """Build a ResultRecord-shaped dict for JSONL emission.

    This is intentionally best-effort and tolerant of arbitrary input/output
    structures so the runner can act as a harness.

    Parameters
    ----------
    schema_version : str
        Schema version for the record.
    run_id : str
        Unique run identifier.
    started_at : datetime
        When processing of this item started.
    completed_at : datetime
        When processing of this item completed.
    model : dict[str, Any]
        Model specification dictionary.
    probe : dict[str, Any]
        Probe specification dictionary.
    dataset : dict[str, Any]
        Dataset specification dictionary.
    item : Any
        The input item being processed.
    output : Any
        The probe output.
    latency_ms : Optional[float]
        Processing time in milliseconds.
    store_messages : Optional[bool]
        Whether to include normalized messages in the record.
    index : int
        Index of this item in the prompt set.
    status : str
        Result status string.
    error : Optional[Union[BaseException, str]]
        Error information if the item failed.
    error_type : Optional[str]
        Type name of the error.
    strict_serialization : bool, default False
        If True, fail on non-JSON-serializable values.

    Returns
    -------
    dict[str, Any]
        Complete result record suitable for JSONL emission.
    """
    # Identify example
    example_id = None
    if isinstance(item, dict):
        example_id = item.get("example_id") or item.get("id")
    example_id = str(example_id) if example_id is not None else str(index)

    # Normalize messages for storage/hashing
    messages_raw = None
    if isinstance(item, dict) and isinstance(item.get("messages"), list):
        messages_raw = item.get("messages")

    normalized_messages: Optional[list[dict[str, Any]]] = None
    if messages_raw is not None:
        normalized_messages = []
        for m in messages_raw:
            if isinstance(m, dict):
                role = m.get("role") or "user"
                normalized_messages.append({"role": str(role), "content": m.get("content")})
            else:
                normalized_messages.append({"role": "user", "content": m})

    messages_hash = None
    if normalized_messages is not None:
        try:
            messages_hash = hashlib.sha256(
                _stable_json_dumps(normalized_messages, strict=strict_serialization).encode("utf-8")
            ).hexdigest()
        except StrictSerializationError as exc:
            if strict_serialization:
                raise ValueError(
                    "strict_serialization requires JSON-stable message content."
                ) from exc
            messages_hash = None
        except (ValueError, TypeError):
            messages_hash = None

    try:
        input_fingerprint = _fingerprint_value(item, strict=strict_serialization)
    except StrictSerializationError as exc:
        raise ValueError(
            "strict_serialization requires JSON-stable inputs for fingerprinting."
        ) from exc
    replicate_key_value = _replicate_key(
        model_spec=model,
        probe_spec=probe,
        dataset_spec=dataset,
        example_id=example_id,
        record_index=index,
        input_hash=messages_hash or input_fingerprint,
        strict_serialization=strict_serialization,
    )

    # Derive output_text / scoring fields (best-effort)
    output_text = None
    if isinstance(output, str):
        output_text = output
    elif isinstance(output, dict):
        output_text = output.get("output_text") or output.get("text")

    output_fingerprint = None
    if output is not None and not isinstance(output, str):
        try:
            output_fingerprint = _fingerprint_value(output, strict=strict_serialization)
        except StrictSerializationError as exc:
            raise ValueError(
                "strict_serialization requires JSON-stable structured outputs."
            ) from exc

    scores: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    primary_metric = None
    if isinstance(output, dict):
        if isinstance(output.get("scores"), dict):
            scores = output.get("scores")
        elif output.get("score") is not None:
            scores = {"score": output.get("score")}
        if isinstance(output.get("usage"), dict):
            usage = output.get("usage")
        primary_metric = output.get("primary_metric")

    err_str: Optional[str] = None
    err_type: Optional[str] = None
    if isinstance(error, BaseException):
        err_str = str(error)
        err_type = type(error).__name__
    elif isinstance(error, str):
        err_str = error
        err_type = error_type
    else:
        err_type = error_type

    return {
        "schema_version": schema_version,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "model": model,
        "probe": probe,
        "dataset": dataset,
        "example_id": example_id,
        "input": item,
        "messages": normalized_messages
        if (store_messages and normalized_messages is not None)
        else None,
        "messages_hash": messages_hash,
        "messages_storage": None,
        "output": output,
        "output_text": output_text,
        "scores": scores,
        "primary_metric": primary_metric,
        "usage": usage,
        "latency_ms": latency_ms,
        "status": status,
        "error": err_str,
        "error_type": err_type,
        "custom": {
            "replicate_key": replicate_key_value,
            "record_index": index,
            "output_fingerprint": output_fingerprint,
        },
    }


__all__ = [
    "_normalize_status",
    "_normalize_info_obj_to_dict",
    "_build_model_spec",
    "_build_probe_spec",
    "_build_dataset_spec",
    "_result_dict_from_probe_result",
    "_record_index_from_record",
    "_result_dict_from_record",
    "_coerce_model_info",
    "_build_result_record",
]
