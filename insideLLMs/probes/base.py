"""Base class and protocols for LLM probes.

This module defines the abstract interface for probes - components that test
specific aspects of LLM behavior such as logic, factuality, bias, and safety.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar, runtime_checkable

from insideLLMs.types import (
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
)

# Type variable for probe output types
T = TypeVar("T")


@runtime_checkable
class ProbeProtocol(Protocol):
    """Protocol defining the interface for probes.

    Use this for type hints when you want to accept any probe-like object.
    """

    name: str
    category: ProbeCategory

    def run(self, model: Any, data: Any, **kwargs: Any) -> Any:
        """Run the probe on the given model with the provided data."""
        ...


class Probe(ABC, Generic[T]):
    """Base class for all probes.

    A probe tests a specific aspect of LLM behavior by presenting the model
    with carefully designed inputs and analyzing the outputs.

    Type Parameters:
        T: The output type for this probe's results.

    Attributes:
        name: Human-readable name for this probe.
        category: The category this probe belongs to (logic, bias, etc.)
        description: A brief description of what this probe tests.

    Example:
        >>> class MyProbe(Probe[str]):
        ...     def run(self, model, data, **kwargs):
        ...         return model.generate(data)
        ...
        ...     def score(self, results):
        ...         return ProbeScore(accuracy=0.85)
    """

    # Class-level defaults that can be overridden by subclasses
    default_category: ProbeCategory = ProbeCategory.CUSTOM

    def __init__(
        self,
        name: str,
        category: Optional[ProbeCategory] = None,
        description: Optional[str] = None,
    ):
        """Initialize the probe.

        Args:
            name: Human-readable name for this probe.
            category: The category this probe belongs to. If not provided,
                     uses the class-level default_category.
            description: A brief description of what this probe tests.
        """
        self.name = name
        self.category = category or self.default_category
        self.description = description or self.__class__.__doc__ or ""

    @abstractmethod
    def run(self, model: Any, data: Any, **kwargs: Any) -> T:
        """Run the probe on the given model with the provided data.

        Args:
            model: The model to test (should implement ModelProtocol).
            data: The input data for the probe. Format varies by probe type.
            **kwargs: Additional arguments specific to the probe.

        Returns:
            The probe output of type T.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def run_batch(
        self,
        model: Any,
        dataset: List[Any],

        max_workers: int = 1,
        **kwargs: Any,
    ) -> List[ProbeResult[T]]:
        """Run the probe on a batch of inputs.

        This method provides consistent error handling and result formatting
        across all probes.

        Args:
            model: The model to test.
            dataset: A list of input items to test.
            max_workers: Number of concurrent workers (default: 1).
            **kwargs: Additional arguments passed to run().

        Returns:
            A list of ProbeResult objects containing outputs or errors.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor

        def process_item(item: Any) -> ProbeResult[T]:
            start = time.perf_counter()
            try:
                output = self.run(model, item, **kwargs)
                latency = (time.perf_counter() - start) * 1000
                return ProbeResult(
                    input=item,
                    output=output,
                    status=ResultStatus.SUCCESS,
                    latency_ms=latency,
                )
            except TimeoutError:
                return ProbeResult(
                    input=item,
                    status=ResultStatus.TIMEOUT,
                    error="Request timed out",
                )
            except Exception as e:
                error_type = type(e).__name__
                # Check for rate limiting
                if "rate" in str(e).lower() or "429" in str(e):
                    status = ResultStatus.RATE_LIMITED
                else:
                    status = ResultStatus.ERROR
                return ProbeResult(
                    input=item,
                    status=status,
                    error=f"{error_type}: {str(e)}",
                )

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_item, dataset))
        else:
            results = [process_item(item) for item in dataset]

        return results

    def score(self, results: List[ProbeResult[T]]) -> ProbeScore:
        """Calculate aggregate scores from probe results.

        Override this method in subclasses to provide probe-specific scoring.

        Args:
            results: The list of probe results to score.

        Returns:
            A ProbeScore with aggregate metrics.
        """
        if not results:
            return ProbeScore()

        success_count = sum(1 for r in results if r.status == ResultStatus.SUCCESS)
        error_count = sum(1 for r in results if r.status == ResultStatus.ERROR)
        total = len(results)

        # Calculate mean latency for successful results
        latencies = [r.latency_ms for r in results if r.latency_ms is not None]
        mean_latency = sum(latencies) / len(latencies) if latencies else None

        return ProbeScore(
            accuracy=success_count / total if total > 0 else None,
            error_rate=error_count / total if total > 0 else 0.0,
            mean_latency_ms=mean_latency,
        )

    def validate_input(self, data: Any) -> bool:
        """Validate that the input data is in the expected format.

        Override this method in subclasses for specific validation logic.

        Args:
            data: The input data to validate.

        Returns:
            True if valid, False otherwise.
        """
        return True

    def info(self) -> Dict[str, Any]:
        """Return probe metadata.

        Returns:
            A dictionary containing probe information.
        """
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "type": self.__class__.__name__,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, category={self.category.value!r})"


class ScoredProbe(Probe[T]):
    """Base class for probes that evaluate correctness.

    Extends Probe with support for reference answers and automatic scoring.
    Use this for probes where you have expected outputs to compare against.
    """

    @abstractmethod
    def evaluate_single(
        self,
        model_output: T,
        reference: Any,
        input_data: Any,
    ) -> Dict[str, Any]:
        """Evaluate a single model output against a reference.

        Args:
            model_output: The output from the model.
            reference: The expected/reference answer.
            input_data: The original input.

        Returns:
            A dictionary with evaluation metrics (e.g., is_correct, score).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def score(self, results: List[ProbeResult[T]]) -> ProbeScore:
        """Calculate scores including correctness metrics.

        Args:
            results: The list of probe results to score.

        Returns:
            A ProbeScore with accuracy and other metrics.
        """
        base_score = super().score(results)

        # Count correct answers from results with evaluation data
        correct_count = 0
        evaluated_count = 0

        for result in results:
            if result.status == ResultStatus.SUCCESS and result.metadata:
                evaluated_count += 1
                if result.metadata.get("is_correct", False):
                    correct_count += 1

        if evaluated_count > 0:
            base_score.accuracy = correct_count / evaluated_count

        return base_score


class ComparativeProbe(Probe[T]):
    """Base class for probes that compare responses.

    Use this for bias detection, A/B testing, or any probe that compares
    multiple model responses.
    """

    def run_comparison(
        self,
        model: Any,
        input_a: Any,
        input_b: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run the probe comparing two inputs.

        Args:
            model: The model to test.
            input_a: The first input variant.
            input_b: The second input variant.
            **kwargs: Additional arguments.

        Returns:
            A dictionary containing both responses and comparison metrics.
        """
        response_a = model.generate(input_a, **kwargs)
        response_b = model.generate(input_b, **kwargs)

        comparison = self.compare_responses(response_a, response_b, input_a, input_b)

        return {
            "input_a": input_a,
            "input_b": input_b,
            "response_a": response_a,
            "response_b": response_b,
            "comparison": comparison,
        }

    def compare_responses(
        self,
        response_a: str,
        response_b: str,
        input_a: Any,
        input_b: Any,
    ) -> Dict[str, Any]:
        """Compare two model responses.

        Override this method in subclasses for specific comparison logic.

        Args:
            response_a: The first model response.
            response_b: The second model response.
            input_a: The first input.
            input_b: The second input.

        Returns:
            A dictionary with comparison metrics.
        """
        return {
            "length_diff": len(response_a) - len(response_b),
            "response_identical": response_a == response_b,
        }
