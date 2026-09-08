"""Provider-independent contracts for request-bound matched-compute evidence."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock


class ComputeMismatchError(ValueError):
    """Raised when paired variants did not consume comparable model compute."""


@dataclass(frozen=True)
class OutputLimitBinding:
    parameter: str
    maximum: int

    def __post_init__(self) -> None:
        if self.parameter not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("output limit requires max_tokens or max_completion_tokens")
        if isinstance(self.maximum, bool) or not isinstance(self.maximum, int) or self.maximum <= 0:
            raise ValueError("output limit maximum must be a positive integer")


@dataclass(frozen=True)
class DispatchEvidence:
    output_cap: int
    output_tokens: int | None
    completed: bool


def validate_output_limit(
    binding: OutputLimitBinding | None,
    kwargs: Mapping[str, object],
    declared_cap: int | None = None,
) -> int | None:
    """Validate explicit flat bindings; absence remains an unsupported path."""
    if binding is None:
        return None
    aliases = {"max_tokens", "max_completion_tokens"}.intersection(kwargs)
    if aliases != {binding.parameter}:
        raise ComputeMismatchError("bound output cap requires exactly one matching request alias")
    value = kwargs[binding.parameter]
    if isinstance(value, bool) or not isinstance(value, int) or value != binding.maximum:
        raise ComputeMismatchError("request output cap differs from bound output cap")
    if declared_cap is not None and binding.maximum != declared_cap:
        raise ComputeMismatchError("bound output cap differs from declared per-call cap")
    return binding.maximum


class DispatchCollector:
    """One variant's reservations, shared by child tasks but never reused.

    A zero output_cap marks an unbound dispatch, which can never verify a cap.
    The lock protects only bookkeeping; no user/provider code runs under it.
    """

    def __init__(self, maximum_calls: int, output_cap: int) -> None:
        self.maximum_calls = maximum_calls
        self.output_cap = output_cap
        self._records: list[DispatchEvidence] = []
        self._lock = Lock()
        self._closed = False
        self._violation: str | None = None

    def reserve(self, binding: OutputLimitBinding | None, kwargs: Mapping[str, object]) -> int:
        with self._lock:
            try:
                if self._closed:
                    raise ComputeMismatchError("dispatch scope has already closed")
                cap = validate_output_limit(binding, kwargs, self.output_cap)
                if len(self._records) >= self.maximum_calls:
                    raise ComputeMismatchError("dispatch would exceed declared model-call ceiling")
            except ComputeMismatchError as error:
                self._violation = str(error)
                raise
            self._records.append(DispatchEvidence(cap or 0, None, False))
            return len(self._records) - 1

    def complete(self, index: int, output_tokens: int | None) -> None:
        with self._lock:
            if self._closed:
                return
            cap = self._records[index].output_cap
            self._records[index] = DispatchEvidence(cap, output_tokens, True)
            if output_tokens is not None and output_tokens > (cap or self.output_cap):
                self._violation = "observed dispatch output tokens exceed per-call cap"
                raise ComputeMismatchError(self._violation)

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def record_violation(self, error: ComputeMismatchError) -> None:
        """Retain a rejected preflight without reserving a model call."""
        with self._lock:
            self._violation = str(error)

    def evidence(self) -> tuple[DispatchEvidence, ...]:
        with self._lock:
            if self._violation is not None:
                raise ComputeMismatchError(self._violation)
            return tuple(self._records)


_active_collector: ContextVar[DispatchCollector | None] = ContextVar(
    "matched_compute_dispatch_collector", default=None
)


def current_collector() -> DispatchCollector | None:
    return _active_collector.get()


def validate_output_limit_preflight(
    binding: OutputLimitBinding | None,
    kwargs: Mapping[str, object],
    declared_cap: int,
) -> None:
    """Validate a baseline before execution and retain any active-scope failure."""
    try:
        validate_output_limit(binding, kwargs, declared_cap)
    except ComputeMismatchError as error:
        collector = current_collector()
        if collector is not None:
            collector.record_violation(error)
        raise


@contextmanager
def dispatch_scope(maximum_calls: int, output_cap: int) -> Iterator[DispatchCollector]:
    collector = DispatchCollector(maximum_calls, output_cap)
    token = _active_collector.set(collector)
    try:
        yield collector
    finally:
        collector.close()
        _active_collector.reset(token)
