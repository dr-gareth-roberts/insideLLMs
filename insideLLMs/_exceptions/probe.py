"""Probe and runner exception types."""

from typing import Any, Optional

from insideLLMs._exceptions.base import InsideLLMsError


class ProbeError(InsideLLMsError):
    """Base exception for probe-related errors.

    This is the parent class for all exceptions that occur during probe
    operations, including probe lookup, validation, and execution.
    Catching this exception will handle any probe-related failure.

    Parameters
    ----------
    message : str
        Human-readable error message.
    details : dict[str, Any], optional
        Additional context about the error.

    Examples
    --------
    Catching any probe-related error:

    >>> try:
    ...     probe = ProbeFactory.create("attention_probe")
    ...     results = probe.run(model, dataset)
    ... except ProbeError as e:
    ...     print(f"Probe operation failed: {e}")
    ...     # Handle any probe error uniformly

    Distinguishing between probe error types:

    >>> try:
    ...     results = probe.run(model, dataset)
    ... except ProbeValidationError as e:
    ...     print(f"Invalid input: {e.details['reason']}")
    ... except ProbeExecutionError as e:
    ...     print(f"Execution failed at sample {e.details.get('sample_index')}")
    ... except ProbeError as e:
    ...     print(f"Other probe error: {e}")

    Notes
    -----
    Subclasses include: ProbeNotFoundError, ProbeValidationError,
    ProbeExecutionError, RunnerExecutionError.

    See Also
    --------
    InsideLLMsError : Parent class for all library errors.
    ProbeExecutionError : When probe execution fails.
    RunnerExecutionError : For detailed runner failure context.
    """

    pass


class ProbeNotFoundError(ProbeError):
    """Raised when a requested probe is not found.

    This exception is raised when attempting to use a probe type that
    does not exist in the registry. The error includes information
    about available probe types to help users correct their input.

    Parameters
    ----------
    probe_type : str
        The identifier of the probe that was not found.
    available : list, optional
        List of available probe type identifiers.

    Attributes
    ----------
    details : dict
        Contains 'probe_type' and optionally 'available_probes' list.

    Examples
    --------
    Handling probe not found with suggestions:

    >>> try:
    ...     probe = ProbeFactory.create("atention_probe")  # typo
    ... except ProbeNotFoundError as e:
    ...     print(f"Unknown probe: {e.details['probe_type']}")
    ...     if 'available_probes' in e.details:
    ...         print("Available probes:")
    ...         for p in e.details['available_probes']:
    ...             print(f"  - {p}")

    Implementing probe selection with fallback:

    >>> def get_probe(probe_type: str, fallback: str = "default_probe"):
    ...     try:
    ...         return ProbeFactory.create(probe_type)
    ...     except ProbeNotFoundError:
    ...         print(f"Probe '{probe_type}' not found, using '{fallback}'")
    ...         return ProbeFactory.create(fallback)

    Validating probe type before use:

    >>> def validate_probe_config(config: dict):
    ...     probe_type = config.get('probe_type')
    ...     try:
    ...         probe = ProbeFactory.create(probe_type)
    ...         return True
    ...     except ProbeNotFoundError as e:
    ...         print(f"Invalid probe in config: {e}")
    ...         return False

    See Also
    --------
    ProbeValidationError : When probe input is invalid.
    ProbeExecutionError : When probe execution fails.
    """

    def __init__(self, probe_type: str, available: Optional[list] = None):
        details = {"probe_type": probe_type}
        if available:
            details["available_probes"] = available
        super().__init__(f"Probe not found: {probe_type}", details)


class ProbeValidationError(ProbeError):
    """Raised when probe input validation fails.

    This exception is raised when input data provided to a probe does
    not meet the required format, type, or constraints. This includes
    invalid prompt formats, missing required fields, or data type
    mismatches.

    Parameters
    ----------
    probe_type : str
        The type of probe that performed validation.
    reason : str
        A description of why validation failed.
    invalid_input : Any, optional
        The input that failed validation (truncated to 100 chars).

    Attributes
    ----------
    details : dict
        Contains 'probe_type', 'reason', and optionally 'invalid_input'.

    Examples
    --------
    Handling validation errors with user feedback:

    >>> try:
    ...     results = probe.run(model, dataset)
    ... except ProbeValidationError as e:
    ...     print(f"Invalid input for {e.details['probe_type']}:")
    ...     print(f"  Reason: {e.details['reason']}")
    ...     if 'invalid_input' in e.details:
    ...         print(f"  Input: {e.details['invalid_input']}")

    Pre-validating data before running probe:

    >>> def safe_run_probe(probe, model, data):
    ...     try:
    ...         return probe.run(model, data)
    ...     except ProbeValidationError as e:
    ...         # Log validation failure and skip
    ...         logging.warning(f"Skipping invalid sample: {e}")
    ...         return None

    Collecting validation errors for batch processing:

    >>> validation_errors = []
    >>> for sample in samples:
    ...     try:
    ...         probe.validate(sample)
    ...     except ProbeValidationError as e:
    ...         validation_errors.append({
    ...             'sample': sample,
    ...             'error': e.details['reason']
    ...         })
    >>> if validation_errors:
    ...     print(f"Found {len(validation_errors)} invalid samples")

    See Also
    --------
    ProbeExecutionError : When execution fails after validation.
    DatasetValidationError : For dataset-level validation failures.
    """

    def __init__(self, probe_type: str, reason: str, invalid_input: Any = None):
        details = {"probe_type": probe_type, "reason": reason}
        if invalid_input is not None:
            details["invalid_input"] = str(invalid_input)[:100]
        super().__init__(f"Probe validation failed: {reason}", details)


class ProbeExecutionError(ProbeError):
    """Raised when probe execution fails.

    This exception is raised when a probe fails during execution after
    passing validation. This can occur due to model errors, computation
    failures, or unexpected data conditions during runtime.

    Parameters
    ----------
    probe_type : str
        The type of probe that failed.
    reason : str
        A description of why execution failed.
    sample_index : int, optional
        The index of the sample that caused the failure, useful for
        resuming batch processing.
    original_error : Exception, optional
        The underlying exception that caused this error.

    Attributes
    ----------
    original_error : Exception or None
        The underlying exception, if any, for debugging.
    details : dict
        Contains 'probe_type', 'reason', and optionally 'sample_index'
        and 'original_error'.

    Examples
    --------
    Handling execution errors with sample tracking:

    >>> try:
    ...     results = probe.run(model, dataset)
    ... except ProbeExecutionError as e:
    ...     failed_index = e.details.get('sample_index')
    ...     if failed_index is not None:
    ...         print(f"Failed at sample {failed_index}")
    ...         # Resume from failed sample
    ...         remaining = dataset[failed_index + 1:]
    ...         results = probe.run(model, remaining)

    Accessing the original error for debugging:

    >>> try:
    ...     results = probe.run(model, dataset)
    ... except ProbeExecutionError as e:
    ...     if e.original_error:
    ...         print(f"Root cause: {type(e.original_error).__name__}")
    ...         import traceback
    ...         traceback.print_exception(type(e.original_error),
    ...                                   e.original_error,
    ...                                   e.original_error.__traceback__)

    Implementing skip-on-failure behavior:

    >>> results = []
    >>> for idx, sample in enumerate(dataset):
    ...     try:
    ...         result = probe.run_single(model, sample)
    ...         results.append(result)
    ...     except ProbeExecutionError as e:
    ...         print(f"Skipping sample {idx}: {e.details['reason']}")
    ...         results.append(None)

    See Also
    --------
    ProbeValidationError : When input fails validation.
    RunnerExecutionError : For detailed runner failure context.
    ModelGenerationError : When the underlying model fails.
    """

    def __init__(
        self,
        probe_type: str,
        reason: str,
        sample_index: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {"probe_type": probe_type, "reason": reason}
        if sample_index is not None:
            details["sample_index"] = sample_index
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(f"Probe execution failed: {reason}", details)
        self.original_error = original_error


class RunnerExecutionError(ProbeError):
    """Raised when runner execution fails with rich context.

    This exception captures the full execution context including model,
    probe, prompt, and timing information for easier debugging. It provides
    detailed information for troubleshooting complex pipeline failures.

    Parameters
    ----------
    reason : str
        A description of why execution failed.
    model_id : str, optional
        The identifier of the model being used.
    probe_id : str, optional
        The identifier of the probe being run.
    prompt : str, optional
        The prompt that caused the error (truncated in output).
    prompt_index : int, optional
        Index of the prompt in the dataset.
    run_id : str, optional
        The unique run identifier for tracking.
    elapsed_seconds : float, optional
        Time elapsed before the error occurred.
    original_error : Exception, optional
        The underlying exception that caused this error.
    suggestions : list[str], optional
        List of suggestions for resolving the error.

    Attributes
    ----------
    model_id : str or None
        The model identifier.
    probe_id : str or None
        The probe identifier.
    prompt : str or None
        The prompt that caused the error.
    prompt_index : int or None
        The index of the problematic prompt.
    run_id : str or None
        The unique run identifier.
    elapsed_seconds : float or None
        Elapsed time before failure.
    original_error : Exception or None
        The underlying exception.
    suggestions : list[str]
        Suggestions for resolving the error.
    details : dict
        Rich context dictionary with all available information.

    Examples
    --------
    Handling runner errors with full context:

    >>> try:
    ...     runner.execute(model, probe, dataset)
    ... except RunnerExecutionError as e:
    ...     print(f"Runner failed: {e.message}")
    ...     if e.model_id:
    ...         print(f"  Model: {e.model_id}")
    ...     if e.probe_id:
    ...         print(f"  Probe: {e.probe_id}")
    ...     if e.prompt_index is not None:
    ...         print(f"  Failed at prompt #{e.prompt_index}")
    ...     if e.elapsed_seconds:
    ...         print(f"  Elapsed time: {e.elapsed_seconds:.2f}s")

    Using suggestions for error resolution:

    >>> try:
    ...     runner.execute(model, probe, dataset)
    ... except RunnerExecutionError as e:
    ...     if e.suggestions:
    ...         print("Suggestions:")
    ...         for suggestion in e.suggestions:
    ...             print(f"  - {suggestion}")

    Logging detailed error information:

    >>> import logging
    >>> try:
    ...     runner.execute(model, probe, dataset)
    ... except RunnerExecutionError as e:
    ...     logging.error(
    ...         "Runner execution failed",
    ...         extra={
    ...             "run_id": e.run_id,
    ...             "model": e.model_id,
    ...             "probe": e.probe_id,
    ...             "prompt_index": e.prompt_index,
    ...             "elapsed": e.elapsed_seconds,
    ...             "error_type": type(e.original_error).__name__ if e.original_error else None,
    ...         }
    ...     )

    Notes
    -----
    The string representation of this exception is specially formatted
    to provide a multi-line detailed error report, including context,
    prompt preview, cause, and suggestions.

    See Also
    --------
    ProbeExecutionError : Simpler execution error without full context.
    ModelGenerationError : When the underlying model fails.
    """

    def __init__(
        self,
        reason: str,
        *,
        model_id: Optional[str] = None,
        probe_id: Optional[str] = None,
        prompt: Optional[str] = None,
        prompt_index: Optional[int] = None,
        run_id: Optional[str] = None,
        elapsed_seconds: Optional[float] = None,
        original_error: Optional[Exception] = None,
        suggestions: Optional[list[str]] = None,
    ):
        self.model_id = model_id
        self.probe_id = probe_id
        self.prompt = prompt
        self.prompt_index = prompt_index
        self.run_id = run_id
        self.elapsed_seconds = elapsed_seconds
        self.original_error = original_error
        self.suggestions = suggestions or []

        details: dict[str, Any] = {"reason": reason}
        if model_id:
            details["model_id"] = model_id
        if probe_id:
            details["probe_id"] = probe_id
        if prompt:
            details["prompt_preview"] = prompt[:100] + "..." if len(prompt) > 100 else prompt
        if prompt_index is not None:
            details["prompt_index"] = prompt_index
        if run_id:
            details["run_id"] = run_id
        if elapsed_seconds is not None:
            details["elapsed_seconds"] = round(elapsed_seconds, 3)
        if original_error:
            details["original_error_type"] = type(original_error).__name__
            details["original_error_message"] = str(original_error)

        super().__init__(f"Runner execution failed: {reason}", details)

    def __str__(self) -> str:
        parts = [self.message]

        context_parts = []
        if self.model_id:
            context_parts.append(f"model={self.model_id}")
        if self.probe_id:
            context_parts.append(f"probe={self.probe_id}")
        if self.prompt_index is not None:
            context_parts.append(f"index={self.prompt_index}")
        if self.run_id:
            context_parts.append(f"run_id={self.run_id}")

        if context_parts:
            parts.append(f"Context: [{', '.join(context_parts)}]")

        if self.prompt:
            preview = self.prompt[:80] + "..." if len(self.prompt) > 80 else self.prompt
            parts.append(f"Prompt: {preview!r}")

        if self.original_error:
            parts.append(f"Caused by: {type(self.original_error).__name__}: {self.original_error}")

        if self.suggestions:
            parts.append("Suggestions:")
            for suggestion in self.suggestions:
                parts.append(f"  - {suggestion}")

        return "\n".join(parts)
