"""Base runner class and progress callback utilities.

This module provides the base class for ProbeRunner and AsyncProbeRunner,
along with progress callback handling utilities.
"""

import inspect
from pathlib import Path
from typing import Any, Callable, Optional, Union, cast

from insideLLMs.config_types import ProgressInfo
from insideLLMs.models.base import Model
from insideLLMs.probes.base import Probe
from insideLLMs.types import ExperimentResult

# Type aliases for progress callbacks - supports both simple (current, total) and rich ProgressInfo
LegacyProgressCallback = Callable[[int, int], None]
RichProgressCallback = Callable[["ProgressInfo"], None]
ProgressCallback = Union[LegacyProgressCallback, RichProgressCallback]


def _invoke_progress_callback(
    callback: Optional[ProgressCallback],
    current: int,
    total: int,
    start_time: float,
    current_item: Optional[Any] = None,
    current_index: Optional[int] = None,
    status: Optional[str] = None,
) -> None:
    """Invoke a progress callback with the appropriate signature.

    Supports both legacy (current, total) callbacks and new ProgressInfo callbacks.
    Detects the callback signature and invokes accordingly.

    Parameters
    ----------
    callback : Optional[ProgressCallback]
        The progress callback to invoke.
    current : int
        Number of items completed.
    total : int
        Total number of items.
    start_time : float
        time.perf_counter() value at start.
    current_item : Optional[Any], default None
        The item currently being processed.
    current_index : Optional[int], default None
        Index of current item.
    status : Optional[str], default None
        Current status message.

    Examples
    --------
    Using with a legacy callback:

        >>> def legacy_progress(current, total):
        ...     print(f"{current}/{total}")
        >>> _invoke_progress_callback(legacy_progress, 5, 10, 0.0)
        5/10

    Using with a rich callback:

        >>> from insideLLMs.config_types import ProgressInfo
        >>> def rich_progress(info: ProgressInfo):
        ...     print(f"{info.current}/{info.total} - {info.percent:.1f}%")
        >>> _invoke_progress_callback(rich_progress, 5, 10, 0.0)
        5/10 - 50.0%
    """
    if callback is None:
        return

    callback_style = getattr(callback, "_insidellms_progress_style", None)
    if callback_style not in {"legacy", "info"}:
        try:
            sig = inspect.signature(callback)
            params = list(sig.parameters.values())
            callback_style = "legacy" if len(params) == 2 else "info"
        except (TypeError, ValueError):
            callback_style = "legacy"
        try:
            setattr(callback, "_insidellms_progress_style", callback_style)
        except Exception:
            pass

    if callback_style == "legacy":
        # Runtime signature detection confirmed this is a (current, total) callback
        cast(LegacyProgressCallback, callback)(current, total)
        return

    # Use new ProgressInfo style
    info = ProgressInfo.create(
        current=current,
        total=total,
        start_time=start_time,
        current_item=current_item,
        current_index=current_index,
        status=status,
    )
    # Runtime signature detection confirmed this is a ProgressInfo callback
    cast(RichProgressCallback, callback)(info)


def _normalize_validation_mode(mode: Optional[str]) -> str:
    """Normalize validation mode to the validator's accepted values.

    The schema validator accepts "strict" or "warn". Historically, runner
    configuration exposes "lenient" to mean "warn". This helper preserves
    backward compatibility while keeping validator behavior explicit.

    Parameters
    ----------
    mode : Optional[str]
        The validation mode to normalize.

    Returns
    -------
    str
        Normalized mode: "strict", "warn", or the original value.

    Examples
    --------
    Normalizing modes:

        >>> _normalize_validation_mode(None)
        'strict'
        >>> _normalize_validation_mode("lenient")
        'warn'
        >>> _normalize_validation_mode("strict")
        'strict'
    """
    if mode is None:
        return "strict"
    normalized = str(mode).strip().lower()
    if normalized in {"lenient", "warn"}:
        return "warn"
    return normalized


class _RunnerBase:
    """Base class for ProbeRunner and AsyncProbeRunner with shared functionality.

    This class provides common initialization, properties, and helper methods
    used by both synchronous and asynchronous runners. It should not be
    instantiated directly; use :class:`ProbeRunner` or :class:`AsyncProbeRunner`
    instead.

    Attributes
    ----------
    model : Model
        The model instance to test.
    probe : Probe
        The probe instance to run.
    last_run_id : Optional[str]
        The unique identifier of the last completed run.
    last_run_dir : Optional[Path]
        The directory where the last run's artifacts were stored.
    last_experiment : Optional[ExperimentResult]
        The ExperimentResult from the last completed run.

    Examples
    --------
    Accessing run statistics after execution:

        >>> runner = ProbeRunner(model, probe)
        >>> results = runner.run(prompts)
        >>> print(f"Success rate: {runner.success_rate:.1%}")
        >>> print(f"Error count: {runner.error_count}")
        >>> print(f"Run ID: {runner.last_run_id}")

    Accessing the last experiment result:

        >>> if runner.last_experiment:
        ...     print(f"Score: {runner.last_experiment.score}")
        ...     print(f"Started: {runner.last_experiment.started_at}")
    """

    def __init__(self, model: Model, probe: Probe):
        """Initialize the runner with a model and probe.

        Parameters
        ----------
        model : Model
            The model to test. Must implement the Model interface.
        probe : Probe
            The probe to run. Must implement the Probe interface.

        Examples
        --------
        Creating a runner for testing:

            >>> from insideLLMs.models import OpenAIModel
            >>> from insideLLMs.probes import FactualityProbe
            >>>
            >>> model = OpenAIModel(model_name="gpt-4")
            >>> probe = FactualityProbe()
            >>> runner = ProbeRunner(model, probe)

        Using a custom model and probe:

            >>> from insideLLMs.models.base import Model
            >>> from insideLLMs.probes.base import Probe
            >>>
            >>> class MyModel(Model):
            ...     def generate(self, messages):
            ...         return "response"
            >>>
            >>> class MyProbe(Probe):
            ...     name = "my_probe"
            ...     def run(self, model, input_data):
            ...         return model.generate(input_data)
            >>>
            >>> runner = ProbeRunner(MyModel(), MyProbe())
        """
        self.model = model
        self.probe = probe
        self._results: list[dict[str, Any]] = []
        self.last_run_id: Optional[str] = None
        self.last_run_dir: Optional[Path] = None
        self.last_experiment: Optional[ExperimentResult] = None

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the last run.

        Returns the proportion of successful results from the most recent
        execution. Returns 0.0 if no results are available.

        Returns
        -------
        float
            Success rate as a decimal between 0.0 and 1.0.

        Examples
        --------
        Checking success rate after a run:

            >>> runner = ProbeRunner(model, probe)
            >>> results = runner.run(prompts)
            >>> print(f"Success rate: {runner.success_rate:.1%}")
            Success rate: 95.0%

        Handling empty results:

            >>> runner = ProbeRunner(model, probe)
            >>> # Before running any prompts
            >>> print(runner.success_rate)
            0.0
        """
        if not self._results:
            return 0.0
        successes = sum(1 for r in self._results if r.get("status") == "success")
        return successes / len(self._results)

    @property
    def error_count(self) -> int:
        """Count errors from the last run.

        Returns the total number of results with "error" status from the
        most recent execution.

        Returns
        -------
        int
            Number of errors encountered during the last run.

        Examples
        --------
        Checking error count:

            >>> runner = ProbeRunner(model, probe)
            >>> results = runner.run(prompts)
            >>> if runner.error_count > 0:
            ...     print(f"Encountered {runner.error_count} errors")
        """
        return sum(1 for r in self._results if r.get("status") == "error")


__all__ = [
    "LegacyProgressCallback",
    "RichProgressCallback",
    "ProgressCallback",
    "_invoke_progress_callback",
    "_normalize_validation_mode",
    "_RunnerBase",
]
