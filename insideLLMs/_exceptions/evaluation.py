"""Evaluation exception types."""

from typing import Optional

from insideLLMs._exceptions.base import InsideLLMsError


class EvaluationError(InsideLLMsError):
    """Base exception for evaluation errors.

    This is the parent class for all exceptions that occur during
    evaluation operations, including evaluator lookup and metric
    computation. Catching this exception will handle any evaluation-related
    failure.

    Parameters
    ----------
    message : str
        Human-readable error message.
    details : dict[str, Any], optional
        Additional context about the error.

    Examples
    --------
    Catching any evaluation-related error:

    >>> try:
    ...     scores = evaluator.evaluate(predictions, references)
    ... except EvaluationError as e:
    ...     print(f"Evaluation failed: {e}")
    ...     # Skip evaluation and continue
    ...     scores = None

    Distinguishing between evaluation error types:

    >>> try:
    ...     evaluator = EvaluatorFactory.create("bleu")
    ...     scores = evaluator.evaluate(predictions, references)
    ... except EvaluatorNotFoundError:
    ...     print("Evaluator not available")
    ... except EvaluationFailedError as e:
    ...     print(f"Evaluation computation failed: {e.details['reason']}")

    Notes
    -----
    Subclasses include: EvaluatorNotFoundError, EvaluationFailedError.

    See Also
    --------
    InsideLLMsError : Parent class for all library errors.
    EvaluationFailedError : When evaluation computation fails.
    """

    pass


class EvaluatorNotFoundError(EvaluationError):
    """Raised when an evaluator type is not found.

    This exception is raised when attempting to use an evaluator that
    does not exist in the registry. The error includes information
    about available evaluators to help users correct their input.

    Parameters
    ----------
    evaluator_type : str
        The identifier of the evaluator that was not found.
    available : list, optional
        List of available evaluator type identifiers.

    Attributes
    ----------
    details : dict
        Contains 'evaluator_type' and optionally 'available_evaluators'.

    Examples
    --------
    Handling evaluator not found with suggestions:

    >>> try:
    ...     evaluator = EvaluatorFactory.create("bleu_score")
    ... except EvaluatorNotFoundError as e:
    ...     print(f"Unknown evaluator: {e.details['evaluator_type']}")
    ...     if 'available_evaluators' in e.details:
    ...         print("Available evaluators:")
    ...         for ev in e.details['available_evaluators']:
    ...             print(f"  - {ev}")

    Implementing evaluator selection with fallback:

    >>> def get_evaluator(eval_type: str, fallback: str = "exact_match"):
    ...     try:
    ...         return EvaluatorFactory.create(eval_type)
    ...     except EvaluatorNotFoundError:
    ...         print(f"'{eval_type}' not found, using '{fallback}'")
    ...         return EvaluatorFactory.create(fallback)

    Validating evaluator configuration:

    >>> def validate_eval_config(config: dict) -> list:
    ...     errors = []
    ...     for metric in config.get('metrics', []):
    ...         try:
    ...             EvaluatorFactory.create(metric)
    ...         except EvaluatorNotFoundError as e:
    ...             errors.append(f"Unknown metric: {metric}")
    ...     return errors

    See Also
    --------
    EvaluationFailedError : When evaluation computation fails.
    ProbeNotFoundError : Similar error for probes.
    """

    def __init__(self, evaluator_type: str, available: Optional[list] = None):
        details = {"evaluator_type": evaluator_type}
        if available:
            details["available_evaluators"] = available
        super().__init__(f"Evaluator not found: {evaluator_type}", details)


class EvaluationFailedError(EvaluationError):
    """Raised when evaluation fails.

    This exception is raised when an evaluation computation fails due
    to incompatible inputs, numerical errors, or other runtime issues
    during metric calculation.

    Parameters
    ----------
    reason : str
        A description of why evaluation failed.
    prediction : str, optional
        The model prediction that caused the failure (truncated to 100 chars).
    reference : str, optional
        The reference/ground truth (truncated to 100 chars).

    Attributes
    ----------
    details : dict
        Contains 'reason', and optionally 'prediction_preview' and
        'reference_preview'.

    Examples
    --------
    Handling evaluation failures with context:

    >>> try:
    ...     score = evaluator.evaluate(prediction, reference)
    ... except EvaluationFailedError as e:
    ...     print(f"Evaluation failed: {e.details['reason']}")
    ...     if 'prediction_preview' in e.details:
    ...         print(f"  Prediction: {e.details['prediction_preview']}")
    ...     if 'reference_preview' in e.details:
    ...         print(f"  Reference: {e.details['reference_preview']}")

    Implementing skip-on-failure for batch evaluation:

    >>> scores = []
    >>> for pred, ref in zip(predictions, references):
    ...     try:
    ...         score = evaluator.evaluate(pred, ref)
    ...         scores.append(score)
    ...     except EvaluationFailedError as e:
    ...         logging.warning(f"Skipping sample: {e.details['reason']}")
    ...         scores.append(None)

    Aggregating evaluation results with error handling:

    >>> def safe_evaluate(evaluator, predictions, references):
    ...     results = {'scores': [], 'errors': []}
    ...     for i, (pred, ref) in enumerate(zip(predictions, references)):
    ...         try:
    ...             results['scores'].append(evaluator.evaluate(pred, ref))
    ...         except EvaluationFailedError as e:
    ...             results['errors'].append({'index': i, 'reason': str(e)})
    ...             results['scores'].append(None)
    ...     return results

    See Also
    --------
    EvaluatorNotFoundError : When evaluator doesn't exist.
    ProbeExecutionError : Similar error for probe execution.
    """

    def __init__(
        self,
        reason: str,
        prediction: Optional[str] = None,
        reference: Optional[str] = None,
    ):
        details = {"reason": reason}
        if prediction:
            details["prediction_preview"] = prediction[:100]
        if reference:
            details["reference_preview"] = reference[:100]
        super().__init__(f"Evaluation failed: {reason}", details)
