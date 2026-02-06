"""Deterministic run ID generation and timestamp utilities.

This module provides deterministic hashing, run ID generation, and timestamp
utilities that ensure reproducible experiment results. The same configuration
always produces the same run ID and timestamps.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from insideLLMs._serialization import (
    StrictSerializationError,
)
from insideLLMs._serialization import (
    stable_json_dumps as _stable_json_dumps,
)

# Module-level constants for deterministic timestamp generation.
# These are used to produce reproducible timestamps from run IDs,
# ensuring that the same configuration always produces the same
# artifact timestamps for testing and caching purposes.
_DETERMINISTIC_TIME_BASE = datetime(2000, 1, 1, tzinfo=timezone.utc)
_DETERMINISTIC_TIME_RANGE_SECONDS = 10 * 365 * 24 * 60 * 60  # 10 years in seconds


def _deterministic_hash(payload: Any, *, strict: bool = False) -> str:
    """Generate a deterministic SHA-256 hash of any payload.

    Produces a consistent hash for equivalent data structures by using
    stable JSON serialization. Used for generating run IDs, experiment IDs,
    and replicate keys.

    Parameters
    ----------
    payload : Any
        Data to hash. Can be any JSON-serializable structure.
    strict : bool, default False
        If True, fail on non-JSON-serializable values.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 hash.

    Examples
    --------
    Hashing a configuration dict:

        >>> config = {"model": "gpt-4", "probe": "factuality"}
        >>> hash1 = _deterministic_hash(config)
        >>> hash2 = _deterministic_hash(config)
        >>> hash1 == hash2
        True
        >>> len(hash1)
        64

    Same data, different order, same hash:

        >>> _deterministic_hash({"a": 1, "b": 2}) == _deterministic_hash({"b": 2, "a": 1})
        True

    See Also
    --------
    _stable_json_dumps : Stable JSON serialization.
    _deterministic_run_id_from_inputs : Generate run IDs using this hash.
    """
    return hashlib.sha256(_stable_json_dumps(payload, strict=strict).encode("utf-8")).hexdigest()


def _hash_prompt_set(prompt_set: list[Any], *, strict: bool = False) -> str:
    """Generate a deterministic hash of a prompt set.

    Creates a content-based hash of all prompts in a set, used for
    generating run IDs and detecting duplicate datasets.

    Parameters
    ----------
    prompt_set : list[Any]
        List of prompts to hash. Each prompt is serialized to stable JSON.
    strict : bool, default False
        If True, fail on non-JSON-serializable values.

    Returns
    -------
    str
        64-character hexadecimal SHA-256 hash of the concatenated prompts.

    Examples
    --------
    Hashing a prompt set:

        >>> prompts = [
        ...     {"messages": [{"role": "user", "content": "Hello"}]},
        ...     {"messages": [{"role": "user", "content": "World"}]},
        ... ]
        >>> hash1 = _hash_prompt_set(prompts)
        >>> hash2 = _hash_prompt_set(prompts)
        >>> hash1 == hash2
        True
        >>> len(hash1)
        64

    Order matters:

        >>> prompts_reversed = list(reversed(prompts))
        >>> _hash_prompt_set(prompts) != _hash_prompt_set(prompts_reversed)
        True

    See Also
    --------
    _deterministic_hash : General-purpose hashing.
    _deterministic_run_id_from_inputs : Uses this for run ID generation.
    """
    hasher = hashlib.sha256()
    for index, item in enumerate(prompt_set):
        try:
            hasher.update(_stable_json_dumps(item, strict=strict).encode("utf-8"))
        except StrictSerializationError as exc:
            raise StrictSerializationError(
                f"Non-deterministic prompt at index {index}: {exc}"
            ) from exc
        hasher.update(b"\n")
    return hasher.hexdigest()


def _replicate_key(
    *,
    model_spec: dict[str, Any],
    probe_spec: dict[str, Any],
    dataset_spec: dict[str, Any],
    example_id: str,
    record_index: int,
    input_hash: Optional[str],
    strict_serialization: bool = False,
) -> str:
    """Generate a unique replicate key for a result record.

    Creates a deterministic key that uniquely identifies a specific example
    within a specific run configuration. Used for deduplication and result
    tracking across multiple runs.

    Parameters
    ----------
    model_spec : dict[str, Any]
        Model specification dictionary.
    probe_spec : dict[str, Any]
        Probe specification dictionary.
    dataset_spec : dict[str, Any]
        Dataset specification dictionary.
    example_id : str
        Unique identifier for the example within the dataset.
    record_index : int
        Index position of this record in the result set.
    input_hash : Optional[str]
        Hash of the input data (messages hash or input fingerprint).
    strict_serialization : bool, default False
        If True, fail on non-JSON-serializable values.

    Returns
    -------
    str
        16-character hexadecimal replicate key.

    Examples
    --------
    Generating a replicate key:

        >>> model_spec = {"model_id": "gpt-4", "provider": "openai"}
        >>> probe_spec = {"probe_id": "factuality"}
        >>> dataset_spec = {"dataset_id": "test_data"}
        >>> key = _replicate_key(
        ...     model_spec=model_spec,
        ...     probe_spec=probe_spec,
        ...     dataset_spec=dataset_spec,
        ...     example_id="ex_001",
        ...     record_index=0,
        ...     input_hash="abc123def456",
        ... )
        >>> len(key)
        16

    See Also
    --------
    _deterministic_hash : Hash generation.
    _build_result_record : Uses replicate keys in records.
    """
    payload = {
        "model_id": model_spec.get("model_id"),
        "probe_id": probe_spec.get("probe_id"),
        "dataset_id": dataset_spec.get("dataset_id"),
        "dataset_version": dataset_spec.get("dataset_version"),
        "dataset_hash": dataset_spec.get("dataset_hash"),
        "example_id": example_id,
        "record_index": record_index,
        "input_hash": input_hash,
    }
    return _deterministic_hash(payload, strict=strict_serialization)[:16]


def _deterministic_run_id_from_config_snapshot(
    config_snapshot: dict[str, Any],
    *,
    schema_version: str,
    strict_serialization: bool = False,
) -> str:
    """Generate a deterministic run ID from a configuration snapshot.

    Creates a unique run ID based on the full configuration, ensuring that
    identical configurations produce identical run IDs. This enables
    reproducibility and caching of results.

    Parameters
    ----------
    config_snapshot : dict[str, Any]
        Full configuration dictionary including model, probe, and dataset.
    schema_version : str
        Schema version to include in the hash.
    strict_serialization : bool, default False
        If True, fail on non-JSON-serializable values.

    Returns
    -------
    str
        32-character hexadecimal run ID.

    Examples
    --------
    Generating a run ID:

        >>> config = {
        ...     "model": {"type": "openai", "args": {"model_name": "gpt-4"}},
        ...     "probe": {"type": "factuality"},
        ...     "dataset": {"format": "jsonl", "path": "test.jsonl"},
        ... }
        >>> run_id = _deterministic_run_id_from_config_snapshot(
        ...     config, schema_version="1.0.0"
        ... )
        >>> len(run_id)
        32

    Same config produces same ID:

        >>> run_id2 = _deterministic_run_id_from_config_snapshot(
        ...     config, schema_version="1.0.0"
        ... )
        >>> run_id == run_id2
        True

    See Also
    --------
    _deterministic_run_id_from_inputs : Generate from individual components.
    _deterministic_hash : Hash generation.
    """
    payload = {"schema_version": schema_version, "config": config_snapshot}
    return _deterministic_hash(payload, strict=strict_serialization)[:32]


def _deterministic_run_id_from_inputs(
    *,
    schema_version: str,
    model_spec: dict[str, Any],
    probe_spec: dict[str, Any],
    dataset_spec: dict[str, Any],
    prompt_set: list[Any],
    probe_kwargs: dict[str, Any],
    strict_serialization: bool = False,
) -> str:
    """Generate a deterministic run ID from individual input components.

    Creates a unique run ID based on model, probe, dataset, prompts, and
    additional arguments. Used when running without a config file.

    Parameters
    ----------
    schema_version : str
        Schema version to include in the hash.
    model_spec : dict[str, Any]
        Model specification dictionary.
    probe_spec : dict[str, Any]
        Probe specification dictionary.
    dataset_spec : dict[str, Any]
        Dataset specification dictionary.
    prompt_set : list[Any]
        List of prompts to be processed.
    probe_kwargs : dict[str, Any]
        Additional keyword arguments passed to the probe.
    strict_serialization : bool, default False
        If True, fail on non-JSON-serializable values.

    Returns
    -------
    str
        32-character hexadecimal run ID.

    Examples
    --------
    Generating a run ID from components:

        >>> run_id = _deterministic_run_id_from_inputs(
        ...     schema_version="1.0.0",
        ...     model_spec={"model_id": "gpt-4"},
        ...     probe_spec={"probe_id": "factuality"},
        ...     dataset_spec={"dataset_id": "test"},
        ...     prompt_set=[{"messages": [{"role": "user", "content": "Hi"}]}],
        ...     probe_kwargs={},
        ... )
        >>> len(run_id)
        32

    See Also
    --------
    _deterministic_run_id_from_config_snapshot : Generate from config dict.
    _hash_prompt_set : Hash prompt set for inclusion.
    """
    payload = {
        "schema_version": schema_version,
        "model": model_spec,
        "probe": probe_spec,
        "dataset": dataset_spec,
        "prompt_set_hash": _hash_prompt_set(prompt_set, strict=strict_serialization),
        "probe_kwargs": probe_kwargs,
    }
    return _deterministic_hash(payload, strict=strict_serialization)[:32]


def _deterministic_base_time(run_id: str) -> datetime:
    """Generate a deterministic base timestamp from a run ID.

    Creates a reproducible timestamp derived from the run ID hash. This ensures
    that result records have consistent timestamps across repeated runs with
    the same configuration, enabling deterministic testing.

    Parameters
    ----------
    run_id : str
        The run identifier to derive the timestamp from.

    Returns
    -------
    datetime
        A UTC datetime within a 10-year range starting from 2000-01-01.

    Examples
    --------
    Generating deterministic timestamps:

        >>> from datetime import timezone
        >>> base = _deterministic_base_time("abc123")
        >>> base.tzinfo == timezone.utc
        True
        >>> base.year >= 2000 and base.year <= 2010
        True

    Same run ID produces same timestamp:

        >>> _deterministic_base_time("abc123") == _deterministic_base_time("abc123")
        True

    See Also
    --------
    _deterministic_item_times : Generate timestamps for individual items.
    _deterministic_run_times : Generate start/end timestamps for runs.
    """
    digest = hashlib.sha256(run_id.encode("utf-8")).digest()
    seconds = int.from_bytes(digest[:8], "big") % _DETERMINISTIC_TIME_RANGE_SECONDS
    return _DETERMINISTIC_TIME_BASE + timedelta(seconds=seconds)


def _deterministic_item_times(base_time: datetime, index: int) -> tuple[datetime, datetime]:
    """Generate deterministic timestamps for an individual result item.

    Creates reproducible start and completion timestamps for a single item
    based on its index. Each item gets a unique timestamp pair offset by
    microseconds from the base time.

    Parameters
    ----------
    base_time : datetime
        The base timestamp for the run (from _deterministic_base_time).
    index : int
        Zero-based index of the item in the result set.

    Returns
    -------
    tuple[datetime, datetime]
        Tuple of (started_at, completed_at) timestamps.

    Examples
    --------
    Generating item timestamps:

        >>> from datetime import datetime, timezone
        >>> base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        >>> started, completed = _deterministic_item_times(base, 0)
        >>> started == base
        True
        >>> (completed - started).microseconds
        1

    Sequential items have sequential timestamps:

        >>> t0_start, t0_end = _deterministic_item_times(base, 0)
        >>> t1_start, t1_end = _deterministic_item_times(base, 1)
        >>> t1_start > t0_end
        True

    See Also
    --------
    _deterministic_base_time : Generate the base timestamp.
    _deterministic_run_times : Generate run-level timestamps.
    """
    offset = index * 2
    started_at = base_time + timedelta(microseconds=offset)
    completed_at = base_time + timedelta(microseconds=offset + 1)
    return started_at, completed_at


def _deterministic_run_times(base_time: datetime, total: int) -> tuple[datetime, datetime]:
    """Generate deterministic start and end timestamps for a run.

    Creates reproducible run-level timestamps based on the base time and
    total number of items. The run end time is set to encompass all item
    timestamps.

    Parameters
    ----------
    base_time : datetime
        The base timestamp for the run (from _deterministic_base_time).
    total : int
        Total number of items in the run.

    Returns
    -------
    tuple[datetime, datetime]
        Tuple of (started_at, completed_at) timestamps for the run.

    Examples
    --------
    Generating run timestamps:

        >>> from datetime import datetime, timezone
        >>> base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        >>> started, completed = _deterministic_run_times(base, 100)
        >>> started == base
        True
        >>> completed > started
        True

    Empty run:

        >>> started, completed = _deterministic_run_times(base, 0)
        >>> (completed - started).microseconds
        1

    See Also
    --------
    _deterministic_base_time : Generate the base timestamp.
    _deterministic_item_times : Generate item-level timestamps.
    """
    if total < 0:
        total = 0
    completed_offset = total * 2 + 1
    started_at = base_time
    completed_at = base_time + timedelta(microseconds=completed_offset)
    return started_at, completed_at


def _deterministic_harness_experiment_id(
    *,
    model_config: dict[str, Any],
    probe_config: dict[str, Any],
    dataset_config: dict[str, Any],
    model_index: int,
    probe_index: int,
    max_examples: Optional[int],
    schema_version: str,
    harness_run_id: str,
    strict_serialization: bool = False,
) -> str:
    """Generate a deterministic experiment ID for harness runs.

    Creates a unique experiment ID for a specific model-probe combination
    within a harness run.

    Parameters
    ----------
    model_config : dict[str, Any]
        Model configuration dictionary.
    probe_config : dict[str, Any]
        Probe configuration dictionary.
    dataset_config : dict[str, Any]
        Dataset configuration dictionary.
    model_index : int
        Index of the model in the harness model list.
    probe_index : int
        Index of the probe in the harness probe list.
    max_examples : Optional[int]
        Maximum number of examples to process.
    schema_version : str
        Schema version for the run.
    harness_run_id : str
        Parent harness run ID.
    strict_serialization : bool, default False
        If True, fail on non-JSON-serializable values.

    Returns
    -------
    str
        16-character hexadecimal experiment ID.
    """
    payload = {
        "schema_version": schema_version,
        "harness_run_id": harness_run_id,
        "model": model_config,
        "probe": probe_config,
        "dataset": dataset_config,
        "max_examples": max_examples,
        "model_index": model_index,
        "probe_index": probe_index,
    }
    return _deterministic_hash(payload, strict=strict_serialization)[:16]


__all__ = [
    "_deterministic_hash",
    "_hash_prompt_set",
    "_replicate_key",
    "_deterministic_run_id_from_config_snapshot",
    "_deterministic_run_id_from_inputs",
    "_deterministic_base_time",
    "_deterministic_item_times",
    "_deterministic_run_times",
    "_deterministic_harness_experiment_id",
    "_DETERMINISTIC_TIME_BASE",
    "_DETERMINISTIC_TIME_RANGE_SECONDS",
]
