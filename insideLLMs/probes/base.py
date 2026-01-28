"""Base class and protocols for LLM probes.

This module defines the abstract interface for probes - components that test
specific aspects of LLM behavior such as logic, factuality, bias, and safety.

A probe is the fundamental unit of evaluation in the insideLLMs framework.
Each probe is designed to test a specific aspect of an LLM's capabilities
or behavior by presenting carefully designed inputs and analyzing the outputs.

Classes:
    ProbeProtocol: Protocol defining the interface any probe-like object must satisfy.
    Probe: Abstract base class for all probes, providing core functionality.
    ScoredProbe: Extended base class for probes that evaluate correctness against references.
    ComparativeProbe: Extended base class for probes that compare multiple responses.

Type Variables:
    T: Generic type parameter representing the probe's output type.

Example: Creating a simple custom probe
    >>> from insideLLMs.probes.base import Probe
    >>> from insideLLMs.types import ProbeCategory, ProbeResult
    >>>
    >>> class EchoProbe(Probe[str]):
    ...     \"\"\"A simple probe that tests if the model echoes input.\"\"\"
    ...
    ...     default_category = ProbeCategory.CUSTOM
    ...
    ...     def run(self, model, data, **kwargs):
    ...         return model.generate(f"Repeat exactly: {data}")
    ...
    >>> probe = EchoProbe(name="echo_test", description="Tests echo capability")
    >>> print(probe.info())
    {'name': 'echo_test', 'category': 'custom', 'description': 'Tests echo capability', 'type': 'EchoProbe'}

Example: Running a batch evaluation with progress tracking
    >>> def on_progress(current, total):
    ...     print(f"Progress: {current}/{total}")
    ...
    >>> results = probe.run_batch(
    ...     model=my_model,
    ...     dataset=["hello", "world", "test"],
    ...     max_workers=2,
    ...     progress_callback=on_progress
    ... )
    Progress: 1/3
    Progress: 2/3
    Progress: 3/3

Example: Creating a scored probe for QA evaluation
    >>> from insideLLMs.probes.base import ScoredProbe
    >>>
    >>> class QAProbe(ScoredProbe[str]):
    ...     \"\"\"Probe for evaluating question-answering accuracy.\"\"\"
    ...
    ...     default_category = ProbeCategory.FACTUALITY
    ...
    ...     def run(self, model, data, **kwargs):
    ...         question = data["question"]
    ...         return model.generate(question)
    ...
    ...     def evaluate_single(self, model_output, reference, input_data):
    ...         is_correct = reference.lower() in model_output.lower()
    ...         return {"is_correct": is_correct, "confidence": 0.9 if is_correct else 0.1}
    ...
    >>> qa_probe = QAProbe(name="qa_eval")

Example: Creating a comparative probe for bias detection
    >>> from insideLLMs.probes.base import ComparativeProbe
    >>>
    >>> class GenderBiasProbe(ComparativeProbe[str]):
    ...     \"\"\"Probe for detecting gender bias in responses.\"\"\"
    ...
    ...     default_category = ProbeCategory.BIAS
    ...
    ...     def run(self, model, data, **kwargs):
    ...         return model.generate(data)
    ...
    ...     def compare_responses(self, response_a, response_b, input_a, input_b):
    ...         # Analyze sentiment and content differences
    ...         sentiment_diff = analyze_sentiment(response_a) - analyze_sentiment(response_b)
    ...         return {
    ...             "sentiment_diff": sentiment_diff,
    ...             "potential_bias": abs(sentiment_diff) > 0.3,
    ...             "response_identical": response_a == response_b,
    ...         }

See Also:
    - insideLLMs.types: Type definitions for ProbeResult, ProbeScore, etc.
    - insideLLMs.probes.logic: Logic-specific probe implementations.
    - insideLLMs.probes.bias: Bias detection probe implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable

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

    This protocol establishes the minimal interface that any probe-like object
    must satisfy. It is marked as runtime_checkable, allowing isinstance()
    checks against it.

    Use this protocol for type hints when you want to accept any probe-like
    object, including custom implementations that don't inherit from Probe.

    Attributes:
        name (str): Human-readable identifier for the probe.
        category (ProbeCategory): The evaluation category this probe belongs to.

    Methods:
        run: Execute the probe on a model with given data.

    Example: Type hinting with ProbeProtocol
        >>> from insideLLMs.probes.base import ProbeProtocol
        >>>
        >>> def execute_probes(probes: list[ProbeProtocol], model, data):
        ...     \"\"\"Execute a list of any probe-like objects.\"\"\"
        ...     results = []
        ...     for probe in probes:
        ...         result = probe.run(model, data)
        ...         results.append((probe.name, result))
        ...     return results

    Example: Runtime type checking
        >>> class CustomProbe:
        ...     name = "custom"
        ...     category = ProbeCategory.CUSTOM
        ...     def run(self, model, data, **kwargs):
        ...         return model.generate(data)
        ...
        >>> probe = CustomProbe()
        >>> isinstance(probe, ProbeProtocol)
        True

    Example: Duck typing without inheritance
        >>> class ThirdPartyEvaluator:
        ...     \"\"\"External evaluator that satisfies ProbeProtocol.\"\"\"
        ...     def __init__(self):
        ...         self.name = "external_eval"
        ...         self.category = ProbeCategory.CUSTOM
        ...
        ...     def run(self, model, data, **kwargs):
        ...         # Third-party evaluation logic
        ...         return {"score": 0.95, "details": "Evaluation passed"}
        ...
        >>> evaluator = ThirdPartyEvaluator()
        >>> isinstance(evaluator, ProbeProtocol)  # Works due to structural typing
        True

    Note:
        Objects satisfying this protocol must have both the `name` and
        `category` attributes as well as the `run` method. Missing any
        of these will cause isinstance() checks to return False.
    """

    name: str
    category: ProbeCategory

    def run(self, model: Any, data: Any, **kwargs: Any) -> Any:
        """Run the probe on the given model with the provided data.

        This method defines the core execution interface for all probes.
        Implementations should present the model with test inputs and
        capture the outputs for analysis.

        Args:
            model (Any): The model to evaluate. Should implement a generate()
                method or similar interface for producing outputs.
            data (Any): Input data for the probe. The format is probe-specific
                and may be a string, dict, list, or any other structure.
            **kwargs (Any): Additional keyword arguments for customizing
                the probe execution (e.g., temperature, max_tokens).

        Returns:
            Any: The probe output. Type and structure depend on the specific
                probe implementation.

        Example:
            >>> class SimpleProbe:
            ...     name = "simple"
            ...     category = ProbeCategory.CUSTOM
            ...
            ...     def run(self, model, data, **kwargs):
            ...         prompt = f"Analyze: {data}"
            ...         return model.generate(prompt, **kwargs)
        """
        ...


class Probe(ABC, Generic[T]):
    """Base class for all probes.

    A probe tests a specific aspect of LLM behavior by presenting the model
    with carefully designed inputs and analyzing the outputs. This abstract
    base class provides the foundation for all probe implementations, including
    batch execution, scoring, and metadata management.

    Subclasses must implement the abstract `run` method and may override
    `score`, `validate_input`, and other methods for custom behavior.

    Type Parameters:
        T: The output type for this probe's results. Common types include
            str (for text outputs), dict (for structured outputs), or
            custom dataclasses for complex results.

    Attributes:
        name (str): Human-readable name for this probe. Used for identification
            in reports and logging.
        category (ProbeCategory): The evaluation category this probe belongs to.
            Examples: LOGIC, BIAS, SAFETY, FACTUALITY, CUSTOM.
        description (str): A brief description of what this probe tests. Defaults
            to the class docstring if not provided.
        default_category (ProbeCategory): Class-level default category. Subclasses
            can override this to set a default category for all instances.

    Example: Minimal probe implementation
        >>> from insideLLMs.probes.base import Probe
        >>> from insideLLMs.types import ProbeCategory
        >>>
        >>> class SimpleProbe(Probe[str]):
        ...     \"\"\"A minimal probe that passes input directly to the model.\"\"\"
        ...
        ...     def run(self, model, data, **kwargs):
        ...         return model.generate(data, **kwargs)
        ...
        >>> probe = SimpleProbe(name="simple", category=ProbeCategory.CUSTOM)
        >>> print(probe)
        SimpleProbe(name='simple', category='custom')

    Example: Probe with custom default category
        >>> class LogicProbe(Probe[dict]):
        ...     \"\"\"Base class for all logic probes.\"\"\"
        ...
        ...     default_category = ProbeCategory.LOGIC
        ...
        ...     def run(self, model, data, **kwargs):
        ...         prompt = f"Solve this logic problem: {data}"
        ...         response = model.generate(prompt, **kwargs)
        ...         return {"problem": data, "solution": response}
        ...
        >>> probe = LogicProbe(name="deduction_test")  # category defaults to LOGIC
        >>> probe.category
        <ProbeCategory.LOGIC: 'logic'>

    Example: Probe with input validation
        >>> class MathProbe(Probe[float]):
        ...     \"\"\"Probe for mathematical reasoning with validated input.\"\"\"
        ...
        ...     default_category = ProbeCategory.LOGIC
        ...
        ...     def validate_input(self, data):
        ...         # Ensure data is a dict with required fields
        ...         if not isinstance(data, dict):
        ...             return False
        ...         return "expression" in data and "expected" in data
        ...
        ...     def run(self, model, data, **kwargs):
        ...         if not self.validate_input(data):
        ...             raise ValueError("Invalid input format")
        ...         prompt = f"Calculate: {data['expression']}"
        ...         result = model.generate(prompt, **kwargs)
        ...         return float(result.strip())
        ...
        >>> probe = MathProbe(name="arithmetic_eval")
        >>> probe.validate_input({"expression": "2+2", "expected": 4})
        True
        >>> probe.validate_input("invalid")
        False

    Example: Probe with custom scoring logic
        >>> class SentimentProbe(Probe[dict]):
        ...     \"\"\"Probe that scores based on sentiment accuracy.\"\"\"
        ...
        ...     default_category = ProbeCategory.CUSTOM
        ...
        ...     def run(self, model, data, **kwargs):
        ...         prompt = f"Analyze the sentiment of: {data['text']}"
        ...         return {"predicted": model.generate(prompt), "expected": data["sentiment"]}
        ...
        ...     def score(self, results):
        ...         base_score = super().score(results)
        ...         correct = sum(
        ...             1 for r in results
        ...             if r.status.value == "success"
        ...             and r.output["predicted"].lower() == r.output["expected"].lower()
        ...         )
        ...         total = len([r for r in results if r.status.value == "success"])
        ...         base_score.accuracy = correct / total if total > 0 else 0.0
        ...         return base_score

    See Also:
        ScoredProbe: For probes that evaluate correctness against reference answers.
        ComparativeProbe: For probes that compare multiple responses.
        ProbeProtocol: For duck-typed probe compatibility.
    """

    # Class-level defaults that can be overridden by subclasses
    default_category: ProbeCategory = ProbeCategory.CUSTOM

    def __init__(
        self,
        name: str,
        category: Optional[ProbeCategory] = None,
        description: Optional[str] = None,
    ):
        """Initialize the probe with name, category, and description.

        Creates a new probe instance with the specified configuration. The
        category defaults to the class-level `default_category` if not provided,
        and the description defaults to the class docstring.

        Args:
            name (str): Human-readable name for this probe. This should be
                unique within a probe suite and is used for identification
                in reports, logs, and result aggregation.
            category (Optional[ProbeCategory]): The evaluation category this
                probe belongs to. If not provided, uses the class-level
                `default_category` attribute. Categories help organize probes
                and enable category-based filtering in evaluation runs.
            description (Optional[str]): A brief description of what this probe
                tests. If not provided, defaults to the class docstring. This
                is included in probe metadata and reports.

        Raises:
            No explicit exceptions, but subclasses may add validation.

        Example: Basic initialization
            >>> probe = SimpleProbe(name="basic_test")
            >>> probe.name
            'basic_test'
            >>> probe.category  # Uses class default
            <ProbeCategory.CUSTOM: 'custom'>

        Example: With explicit category
            >>> probe = SimpleProbe(
            ...     name="logic_test",
            ...     category=ProbeCategory.LOGIC,
            ...     description="Tests basic logical reasoning"
            ... )
            >>> probe.category
            <ProbeCategory.LOGIC: 'logic'>
            >>> probe.description
            'Tests basic logical reasoning'

        Example: Description from class docstring
            >>> class DocumentedProbe(Probe[str]):
            ...     \"\"\"This probe tests something important.\"\"\"
            ...     def run(self, model, data, **kwargs):
            ...         return model.generate(data)
            ...
            >>> probe = DocumentedProbe(name="documented")
            >>> probe.description
            'This probe tests something important.'

        Note:
            The `name` parameter is required and should be descriptive enough
            to identify the probe's purpose in reports and logs.
        """
        self.name = name
        self.category = category or self.default_category
        self.description = description or self.__class__.__doc__ or ""

    @abstractmethod
    def run(self, model: Any, data: Any, **kwargs: Any) -> T:
        """Run the probe on the given model with the provided data.

        This is the core abstract method that all probe subclasses must implement.
        It defines how the probe interacts with the model and what output it produces.

        The implementation should:
        1. Prepare the input data into a format suitable for the model
        2. Call the model's generate method (or equivalent)
        3. Process the model's output as needed
        4. Return the result in the probe's output type T

        Args:
            model (Any): The model to test. Should implement a generate() method
                or similar interface. Typically an instance of a class implementing
                ModelProtocol, but any compatible object works.
            data (Any): The input data for the probe. Format varies by probe type
                and may include:
                - str: Simple text prompts
                - dict: Structured inputs with multiple fields
                - list: Multiple items for batch-like single runs
                - Custom objects: Probe-specific data structures
            **kwargs (Any): Additional arguments specific to the probe or passed
                through to the model. Common kwargs include:
                - temperature (float): Sampling temperature
                - max_tokens (int): Maximum output length
                - timeout (float): Request timeout in seconds

        Returns:
            T: The probe output. The type is determined by the probe's generic
                parameter and should be consistent across all calls.

        Raises:
            NotImplementedError: Always raised if the subclass does not override
                this method.
            ValueError: May be raised by implementations for invalid input data.
            TimeoutError: May be raised if the model request times out.
            Exception: Implementation-specific exceptions from model errors.

        Example: Simple text probe
            >>> class TextProbe(Probe[str]):
            ...     def run(self, model, data, **kwargs):
            ...         return model.generate(data, **kwargs)
            ...
            >>> probe = TextProbe(name="text_probe")
            >>> result = probe.run(model, "What is 2+2?")
            >>> print(result)  # Model's response as a string
            '4'

        Example: Structured output probe
            >>> class AnalysisProbe(Probe[dict]):
            ...     def run(self, model, data, **kwargs):
            ...         response = model.generate(f"Analyze: {data['text']}", **kwargs)
            ...         return {
            ...             "input": data["text"],
            ...             "analysis": response,
            ...             "word_count": len(response.split()),
            ...         }

        Example: Probe with validation
            >>> class ValidatedProbe(Probe[str]):
            ...     def run(self, model, data, **kwargs):
            ...         if not self.validate_input(data):
            ...             raise ValueError(f"Invalid input: {data}")
            ...         return model.generate(data["prompt"], **kwargs)

        Example: Probe with custom model interaction
            >>> class MultiTurnProbe(Probe[list]):
            ...     def run(self, model, data, **kwargs):
            ...         responses = []
            ...         for turn in data["turns"]:
            ...             response = model.generate(turn, **kwargs)
            ...             responses.append(response)
            ...         return responses

        Note:
            This method is called by `run_batch` for each item in the dataset.
            Any exceptions raised here are caught and converted to error results.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def run_batch(
        self,
        model: Any,
        dataset: list[Any],
        max_workers: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs: Any,
    ) -> list[ProbeResult[T]]:
        """Run the probe on a batch of inputs with consistent error handling.

        This method iterates over the dataset, calling `run` for each item,
        and wraps results in ProbeResult objects with status tracking, timing,
        and error capture. It supports concurrent execution via ThreadPoolExecutor
        and optional progress reporting.

        The method handles three types of errors automatically:
        - TimeoutError: Marked as TIMEOUT status
        - Rate limiting (429 errors): Marked as RATE_LIMITED status
        - Other exceptions: Marked as ERROR status

        Args:
            model (Any): The model to test. Passed to each `run` call.
            dataset (list[Any]): A list of input items to test. Each item is
                passed individually to the `run` method.
            max_workers (int): Number of concurrent workers for parallel execution.
                Default is 1 (sequential). Higher values enable concurrent
                execution via ThreadPoolExecutor. Note: Be mindful of API rate
                limits when increasing this value.
            progress_callback (Optional[Callable[[int, int], None]]): Optional
                callback function called after each item completes. Receives
                two arguments: current count and total count. Useful for
                progress bars and logging.
            **kwargs (Any): Additional arguments passed through to each `run` call.
                Common kwargs include temperature, max_tokens, etc.

        Returns:
            list[ProbeResult[T]]: A list of ProbeResult objects, one per input
                item, in the same order as the input dataset. Each result contains:
                - input: The original input item
                - output: The probe output (if successful)
                - status: SUCCESS, ERROR, TIMEOUT, or RATE_LIMITED
                - latency_ms: Execution time in milliseconds (if successful)
                - error: Error message (if failed)
                - metadata: Additional error context (if failed)

        Example: Basic batch execution
            >>> probe = SimpleProbe(name="test")
            >>> dataset = ["input1", "input2", "input3"]
            >>> results = probe.run_batch(model, dataset)
            >>> for r in results:
            ...     print(f"{r.input}: {r.status.value}")
            input1: success
            input2: success
            input3: success

        Example: With progress tracking
            >>> from tqdm import tqdm
            >>> pbar = tqdm(total=len(dataset))
            >>>
            >>> def update_progress(current, total):
            ...     pbar.update(1)
            ...
            >>> results = probe.run_batch(
            ...     model=model,
            ...     dataset=dataset,
            ...     progress_callback=update_progress
            ... )
            >>> pbar.close()

        Example: Parallel execution
            >>> # Run with 4 concurrent workers
            >>> results = probe.run_batch(
            ...     model=model,
            ...     dataset=large_dataset,
            ...     max_workers=4,
            ...     temperature=0.7
            ... )

        Example: Processing results
            >>> results = probe.run_batch(model, dataset)
            >>> successful = [r for r in results if r.status == ResultStatus.SUCCESS]
            >>> failed = [r for r in results if r.status != ResultStatus.SUCCESS]
            >>> print(f"Success: {len(successful)}, Failed: {len(failed)}")
            >>> for r in failed:
            ...     print(f"  {r.input}: {r.error}")

        Example: Custom progress logging
            >>> import logging
            >>> logger = logging.getLogger(__name__)
            >>>
            >>> def log_progress(current, total):
            ...     if current % 10 == 0 or current == total:
            ...         logger.info(f"Progress: {current}/{total} ({100*current/total:.1f}%)")
            ...
            >>> results = probe.run_batch(model, dataset, progress_callback=log_progress)

        Note:
            When using max_workers > 1, results are still returned in the same
            order as the input dataset, not in completion order. The order is
            preserved internally using index tracking.

        Warning:
            High values of max_workers may trigger API rate limits. Consider
            adding appropriate delays or using the RATE_LIMITED status to
            implement retry logic.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

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
                    metadata={"error_type": "TimeoutError"},
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
                    metadata={"error_type": error_type},
                )

        completed = 0
        total = len(dataset)

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results: list[ProbeResult[T]] = [None] * len(dataset)  # type: ignore
                futures = {
                    executor.submit(process_item, item): index for index, item in enumerate(dataset)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        error_type = type(e).__name__
                        if "rate" in str(e).lower() or "429" in str(e):
                            status = ResultStatus.RATE_LIMITED
                        elif isinstance(e, TimeoutError):
                            status = ResultStatus.TIMEOUT
                        else:
                            status = ResultStatus.ERROR
                        results[index] = ProbeResult(
                            input=dataset[index],
                            status=status,
                            error=f"{error_type}: {str(e)}",
                            metadata={"error_type": error_type},
                        )
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
        else:
            results = []
            for item in dataset:
                results.append(process_item(item))
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        return results

    def score(self, results: list[ProbeResult[T]]) -> ProbeScore:
        """Calculate aggregate scores from probe results.

        Computes summary statistics from a list of probe results, including
        accuracy (success rate), error rate, and mean latency. Subclasses
        should override this method to add probe-specific scoring logic.

        The base implementation calculates:
        - accuracy: Proportion of successful results (SUCCESS / total)
        - error_rate: Proportion of error results (ERROR / total)
        - mean_latency_ms: Average latency of successful results

        Args:
            results (list[ProbeResult[T]]): The list of probe results to score.
                Typically obtained from `run_batch`. Empty lists return a
                default ProbeScore with no metrics.

        Returns:
            ProbeScore: An aggregate score object containing:
                - accuracy (Optional[float]): Success rate, 0.0 to 1.0
                - error_rate (float): Error rate, 0.0 to 1.0
                - mean_latency_ms (Optional[float]): Average latency in ms

        Example: Basic scoring
            >>> results = probe.run_batch(model, dataset)
            >>> score = probe.score(results)
            >>> print(f"Accuracy: {score.accuracy:.2%}")
            Accuracy: 85.00%
            >>> print(f"Error rate: {score.error_rate:.2%}")
            Error rate: 5.00%
            >>> print(f"Mean latency: {score.mean_latency_ms:.1f}ms")
            Mean latency: 245.3ms

        Example: Empty results handling
            >>> score = probe.score([])
            >>> score.accuracy is None
            True

        Example: Custom scoring in subclass
            >>> class AccuracyProbe(Probe[dict]):
            ...     def run(self, model, data, **kwargs):
            ...         return {"answer": model.generate(data["question"])}
            ...
            ...     def score(self, results):
            ...         base = super().score(results)
            ...         # Add custom accuracy based on correct answers
            ...         correct = sum(
            ...             1 for r in results
            ...             if r.status == ResultStatus.SUCCESS
            ...             and r.output["answer"] == r.input["expected"]
            ...         )
            ...         base.accuracy = correct / len(results) if results else 0.0
            ...         return base

        Example: Filtering by status before scoring
            >>> all_results = probe.run_batch(model, dataset)
            >>> # Only score successful results
            >>> successful = [r for r in all_results if r.status == ResultStatus.SUCCESS]
            >>> score = probe.score(successful)

        Note:
            This method does not modify the input results. Subclasses that
            override this method should call super().score() to get base
            metrics before adding custom scoring logic.
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

        This method provides a hook for subclasses to implement input validation
        before running the probe. The base implementation accepts all inputs.
        Subclasses should override this to enforce specific input requirements.

        Validation is not automatically called by `run` or `run_batch`. Probe
        implementations should explicitly call this method if validation is
        needed.

        Args:
            data (Any): The input data to validate. The expected format depends
                on the specific probe implementation.

        Returns:
            bool: True if the data is valid and can be processed by `run`,
                False otherwise.

        Example: Base implementation accepts everything
            >>> probe = SimpleProbe(name="test")
            >>> probe.validate_input("any string")
            True
            >>> probe.validate_input({"any": "dict"})
            True
            >>> probe.validate_input(None)
            True

        Example: Validating required dict fields
            >>> class QAProbe(Probe[str]):
            ...     def validate_input(self, data):
            ...         if not isinstance(data, dict):
            ...             return False
            ...         required = {"question", "context"}
            ...         return required.issubset(data.keys())
            ...
            ...     def run(self, model, data, **kwargs):
            ...         if not self.validate_input(data):
            ...             raise ValueError("Invalid input format")
            ...         return model.generate(f"{data['context']}\\n{data['question']}")
            ...
            >>> probe = QAProbe(name="qa")
            >>> probe.validate_input({"question": "What?", "context": "Some text"})
            True
            >>> probe.validate_input({"question": "What?"})  # Missing context
            False

        Example: Validating input types
            >>> class NumericProbe(Probe[float]):
            ...     def validate_input(self, data):
            ...         return isinstance(data, (int, float)) and not isinstance(data, bool)
            ...
            >>> probe = NumericProbe(name="numeric")
            >>> probe.validate_input(42)
            True
            >>> probe.validate_input("42")
            False

        Example: Validating with schema
            >>> class SchemaProbe(Probe[dict]):
            ...     SCHEMA = {
            ...         "text": str,
            ...         "max_length": int,
            ...         "temperature": float,
            ...     }
            ...
            ...     def validate_input(self, data):
            ...         if not isinstance(data, dict):
            ...             return False
            ...         for key, expected_type in self.SCHEMA.items():
            ...             if key in data and not isinstance(data[key], expected_type):
            ...                 return False
            ...         return "text" in data  # text is required

        Note:
            Consider raising ValueError with a descriptive message in `run`
            rather than returning False here, if you need to communicate
            why validation failed.
        """
        return True

    def info(self) -> dict[str, Any]:
        """Return probe metadata as a dictionary.

        Provides a serializable representation of the probe's configuration
        and identity. Useful for logging, reporting, and serialization.

        Returns:
            dict[str, Any]: A dictionary containing:
                - name (str): The probe's human-readable name
                - category (str): The category value (e.g., "logic", "bias")
                - description (str): The probe's description
                - type (str): The probe class name

        Example: Basic usage
            >>> probe = SimpleProbe(
            ...     name="my_probe",
            ...     category=ProbeCategory.LOGIC,
            ...     description="Tests logical reasoning"
            ... )
            >>> info = probe.info()
            >>> print(info)
            {'name': 'my_probe', 'category': 'logic', 'description': 'Tests logical reasoning', 'type': 'SimpleProbe'}

        Example: Serializing to JSON
            >>> import json
            >>> probe = SimpleProbe(name="test")
            >>> json_str = json.dumps(probe.info())
            >>> print(json_str)
            '{"name": "test", "category": "custom", "description": "...", "type": "SimpleProbe"}'

        Example: Logging probe information
            >>> import logging
            >>> logger = logging.getLogger(__name__)
            >>> for probe in probe_suite:
            ...     logger.info(f"Running probe: {probe.info()['name']} ({probe.info()['category']})")

        Example: Extending info in subclass
            >>> class ExtendedProbe(Probe[str]):
            ...     def __init__(self, name, version="1.0", **kwargs):
            ...         super().__init__(name, **kwargs)
            ...         self.version = version
            ...
            ...     def info(self):
            ...         base = super().info()
            ...         base["version"] = self.version
            ...         return base
            ...
            >>> probe = ExtendedProbe(name="versioned", version="2.1")
            >>> probe.info()
            {'name': 'versioned', 'category': 'custom', 'description': '...', 'type': 'ExtendedProbe', 'version': '2.1'}

        Note:
            The returned dictionary is a new object; modifying it does not
            affect the probe's internal state.
        """
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "type": self.__class__.__name__,
        }

    def __repr__(self) -> str:
        """Return a string representation of the probe.

        Provides a concise, unambiguous string representation suitable for
        debugging and logging. Includes the class name, probe name, and category.

        Returns:
            str: A string in the format "ClassName(name='probe_name', category='category')".

        Example:
            >>> probe = SimpleProbe(name="test", category=ProbeCategory.LOGIC)
            >>> repr(probe)
            "SimpleProbe(name='test', category='logic')"
            >>> print(probe)  # Uses __repr__ when __str__ not defined
            SimpleProbe(name='test', category='logic')

        Example: In collections
            >>> probes = [SimpleProbe(name="a"), SimpleProbe(name="b")]
            >>> print(probes)
            [SimpleProbe(name='a', category='custom'), SimpleProbe(name='b', category='custom')]
        """
        return f"{self.__class__.__name__}(name={self.name!r}, category={self.category.value!r})"


class ScoredProbe(Probe[T]):
    """Base class for probes that evaluate correctness against reference answers.

    Extends the Probe base class with support for reference answers and automatic
    scoring based on correctness evaluation. Use this class when you have expected
    outputs to compare against, such as question-answering, classification, or
    any task with ground truth labels.

    Subclasses must implement both `run` (inherited from Probe) and `evaluate_single`
    to define how individual outputs are compared to references.

    Type Parameters:
        T: The output type for this probe's results. Should match the type
            of data returned by `run` and passed to `evaluate_single`.

    Attributes:
        Inherits all attributes from Probe (name, category, description).

    Key Methods:
        evaluate_single: Compare a single model output to a reference answer.
        score: Calculate accuracy based on correctness evaluations.

    Example: Simple QA probe
        >>> class SimpleQAProbe(ScoredProbe[str]):
        ...     \"\"\"Probe for simple question-answering with exact match.\"\"\"
        ...
        ...     default_category = ProbeCategory.FACTUALITY
        ...
        ...     def run(self, model, data, **kwargs):
        ...         return model.generate(data["question"], **kwargs)
        ...
        ...     def evaluate_single(self, model_output, reference, input_data):
        ...         is_correct = model_output.strip().lower() == reference.lower()
        ...         return {"is_correct": is_correct}
        ...
        >>> probe = SimpleQAProbe(name="qa_exact_match")

    Example: Classification probe with confidence scoring
        >>> class ClassificationProbe(ScoredProbe[dict]):
        ...     \"\"\"Probe for text classification tasks.\"\"\"
        ...
        ...     default_category = ProbeCategory.CUSTOM
        ...
        ...     def run(self, model, data, **kwargs):
        ...         response = model.generate(f"Classify: {data['text']}", **kwargs)
        ...         # Parse response to get label and confidence
        ...         return {"label": response.split()[0], "raw": response}
        ...
        ...     def evaluate_single(self, model_output, reference, input_data):
        ...         is_correct = model_output["label"].lower() == reference.lower()
        ...         return {
        ...             "is_correct": is_correct,
        ...             "predicted": model_output["label"],
        ...             "expected": reference,
        ...         }

    Example: Fuzzy matching probe
        >>> from difflib import SequenceMatcher
        >>>
        >>> class FuzzyMatchProbe(ScoredProbe[str]):
        ...     \"\"\"Probe with fuzzy string matching for scoring.\"\"\"
        ...
        ...     def __init__(self, name, threshold=0.8, **kwargs):
        ...         super().__init__(name, **kwargs)
        ...         self.threshold = threshold
        ...
        ...     def run(self, model, data, **kwargs):
        ...         return model.generate(data["prompt"], **kwargs)
        ...
        ...     def evaluate_single(self, model_output, reference, input_data):
        ...         similarity = SequenceMatcher(None, model_output, reference).ratio()
        ...         return {
        ...             "is_correct": similarity >= self.threshold,
        ...             "similarity": similarity,
        ...         }

    Example: Multi-label probe
        >>> class MultiLabelProbe(ScoredProbe[list]):
        ...     \"\"\"Probe for multi-label classification.\"\"\"
        ...
        ...     def run(self, model, data, **kwargs):
        ...         response = model.generate(f"Labels for: {data['text']}")
        ...         return [label.strip() for label in response.split(",")]
        ...
        ...     def evaluate_single(self, model_output, reference, input_data):
        ...         predicted = set(model_output)
        ...         expected = set(reference)
        ...         intersection = predicted & expected
        ...         precision = len(intersection) / len(predicted) if predicted else 0
        ...         recall = len(intersection) / len(expected) if expected else 0
        ...         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        ...         return {
        ...             "is_correct": f1 >= 0.5,  # Threshold for "correct"
        ...             "precision": precision,
        ...             "recall": recall,
        ...             "f1": f1,
        ...         }

    See Also:
        Probe: The base class for all probes.
        ComparativeProbe: For probes that compare multiple responses.
    """

    @abstractmethod
    def evaluate_single(
        self,
        model_output: T,
        reference: Any,
        input_data: Any,
    ) -> dict[str, Any]:
        """Evaluate a single model output against a reference answer.

        This abstract method must be implemented by subclasses to define how
        the probe determines correctness. The returned dictionary should include
        at minimum an "is_correct" boolean field, which is used by the `score`
        method to calculate accuracy.

        Args:
            model_output (T): The output from the model, as returned by `run`.
                The type matches the probe's generic parameter T.
            reference (Any): The expected/reference answer to compare against.
                The format depends on the specific probe and dataset.
            input_data (Any): The original input data that was passed to `run`.
                Useful for context-dependent evaluation logic.

        Returns:
            dict[str, Any]: A dictionary with evaluation metrics. Should include:
                - is_correct (bool): Whether the output matches the reference.
                    This field is required for the `score` method to work.
                - Additional probe-specific metrics (optional):
                    - score (float): Numeric score, e.g., 0.0 to 1.0
                    - confidence (float): Model's confidence in the answer
                    - similarity (float): Similarity measure to reference
                    - error_type (str): Type of error if incorrect

        Raises:
            NotImplementedError: Always raised if the subclass does not override
                this method.

        Example: Exact match evaluation
            >>> def evaluate_single(self, model_output, reference, input_data):
            ...     is_correct = model_output.strip() == reference.strip()
            ...     return {"is_correct": is_correct}

        Example: Case-insensitive comparison
            >>> def evaluate_single(self, model_output, reference, input_data):
            ...     predicted = model_output.strip().lower()
            ...     expected = reference.strip().lower()
            ...     return {
            ...         "is_correct": predicted == expected,
            ...         "predicted": model_output,
            ...         "expected": reference,
            ...     }

        Example: Numeric tolerance
            >>> def evaluate_single(self, model_output, reference, input_data):
            ...     try:
            ...         predicted = float(model_output)
            ...         expected = float(reference)
            ...         is_correct = abs(predicted - expected) < 0.01
            ...         return {
            ...             "is_correct": is_correct,
            ...             "difference": abs(predicted - expected),
            ...         }
            ...     except ValueError:
            ...         return {"is_correct": False, "error": "Could not parse number"}

        Example: Substring match with scoring
            >>> def evaluate_single(self, model_output, reference, input_data):
            ...     # Check if reference appears in output
            ...     contains = reference.lower() in model_output.lower()
            ...     # Calculate a relevance score
            ...     if contains:
            ...         score = len(reference) / len(model_output)
            ...     else:
            ...         score = 0.0
            ...     return {
            ...         "is_correct": contains,
            ...         "score": score,
            ...         "output_length": len(model_output),
            ...     }

        Note:
            The "is_correct" field is used by the overridden `score` method
            to calculate accuracy. If this field is missing, the result
            will not contribute to the accuracy calculation.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def score(self, results: list[ProbeResult[T]]) -> ProbeScore:
        """Calculate scores including correctness-based accuracy.

        Overrides the base `score` method to calculate accuracy based on the
        "is_correct" field in result metadata, rather than just success rate.
        This provides a more meaningful accuracy metric for scored probes.

        The method:
        1. Calls the parent `score` to get base metrics (latency, error_rate)
        2. Counts results with is_correct=True in their metadata
        3. Updates accuracy to reflect correctness, not just success

        Args:
            results (list[ProbeResult[T]]): The list of probe results to score.
                Results should have metadata containing "is_correct" field
                from `evaluate_single`.

        Returns:
            ProbeScore: An aggregate score with:
                - accuracy (float): Proportion of correct answers among evaluated
                    results (correct_count / evaluated_count)
                - error_rate (float): Proportion of errors (from parent)
                - mean_latency_ms (Optional[float]): Average latency (from parent)

        Example: Basic usage
            >>> probe = MyQAProbe(name="qa")
            >>> results = probe.run_batch(model, dataset)
            >>> # Assume evaluate_single was called and metadata contains is_correct
            >>> score = probe.score(results)
            >>> print(f"Accuracy: {score.accuracy:.2%}")  # Correctness-based
            Accuracy: 75.00%

        Example: Results with evaluation data
            >>> from insideLLMs.types import ProbeResult, ResultStatus
            >>> results = [
            ...     ProbeResult(
            ...         input={"q": "2+2?"},
            ...         output="4",
            ...         status=ResultStatus.SUCCESS,
            ...         metadata={"is_correct": True}
            ...     ),
            ...     ProbeResult(
            ...         input={"q": "3+3?"},
            ...         output="5",
            ...         status=ResultStatus.SUCCESS,
            ...         metadata={"is_correct": False}
            ...     ),
            ... ]
            >>> score = probe.score(results)
            >>> score.accuracy
            0.5

        Example: Mixed results with errors
            >>> results = [
            ...     ProbeResult(input="a", output="x", status=ResultStatus.SUCCESS,
            ...                 metadata={"is_correct": True}),
            ...     ProbeResult(input="b", output="y", status=ResultStatus.SUCCESS,
            ...                 metadata={"is_correct": False}),
            ...     ProbeResult(input="c", status=ResultStatus.ERROR, error="timeout"),
            ... ]
            >>> score = probe.score(results)
            >>> score.accuracy  # 1 correct out of 2 evaluated (ignores error)
            0.5
            >>> score.error_rate  # 1 error out of 3 total
            0.3333...

        Note:
            Only results with status=SUCCESS and metadata containing "is_correct"
            are counted toward accuracy. Failed results are excluded from
            the accuracy calculation but included in error_rate.

        See Also:
            Probe.score: The parent implementation for base metrics.
            evaluate_single: Method that produces the "is_correct" metadata.
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
    """Base class for probes that compare multiple model responses.

    Extends the Probe base class with methods for running comparisons between
    two inputs and analyzing differences in the model's responses. This is
    particularly useful for:

    - Bias detection: Compare responses to similar prompts with different
      demographic attributes (e.g., names, genders, races)
    - A/B testing: Compare model behavior with different prompt formulations
    - Consistency testing: Check if the model gives consistent answers to
      rephrased questions
    - Fairness evaluation: Measure differential treatment across groups

    Subclasses must implement `run` (inherited from Probe) and may override
    `compare_responses` for custom comparison logic.

    Type Parameters:
        T: The output type for individual probe results. Typically str for
            text responses or dict for structured outputs.

    Attributes:
        Inherits all attributes from Probe (name, category, description).

    Key Methods:
        run_comparison: Run the probe on two inputs and compare responses.
        compare_responses: Analyze differences between two responses.

    Example: Gender bias detection probe
        >>> class GenderBiasProbe(ComparativeProbe[str]):
        ...     \"\"\"Detect gender bias in career-related responses.\"\"\"
        ...
        ...     default_category = ProbeCategory.BIAS
        ...
        ...     def run(self, model, data, **kwargs):
        ...         return model.generate(data, **kwargs)
        ...
        ...     def compare_responses(self, response_a, response_b, input_a, input_b):
        ...         # Analyze differences in career suggestions
        ...         leadership_a = "leader" in response_a.lower()
        ...         leadership_b = "leader" in response_b.lower()
        ...         return {
        ...             "leadership_mentioned": {"a": leadership_a, "b": leadership_b},
        ...             "potential_bias": leadership_a != leadership_b,
        ...             "response_identical": response_a == response_b,
        ...             "length_diff": len(response_a) - len(response_b),
        ...         }
        ...
        >>> probe = GenderBiasProbe(name="career_gender_bias")
        >>> result = probe.run_comparison(
        ...     model,
        ...     input_a="What career advice would you give John?",
        ...     input_b="What career advice would you give Jane?",
        ... )

    Example: Consistency probe for rephrased questions
        >>> class ConsistencyProbe(ComparativeProbe[str]):
        ...     \"\"\"Check if model gives consistent answers to rephrased questions.\"\"\"
        ...
        ...     default_category = ProbeCategory.LOGIC
        ...
        ...     def run(self, model, data, **kwargs):
        ...         return model.generate(data, **kwargs)
        ...
        ...     def compare_responses(self, response_a, response_b, input_a, input_b):
        ...         # Simple semantic similarity check
        ...         words_a = set(response_a.lower().split())
        ...         words_b = set(response_b.lower().split())
        ...         overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
        ...         return {
        ...             "word_overlap": overlap,
        ...             "consistent": overlap > 0.5,
        ...             "response_identical": response_a == response_b,
        ...         }

    Example: Sentiment analysis comparison
        >>> class SentimentComparisonProbe(ComparativeProbe[dict]):
        ...     \"\"\"Compare sentiment in responses to different demographic prompts.\"\"\"
        ...
        ...     def run(self, model, data, **kwargs):
        ...         response = model.generate(data, **kwargs)
        ...         # Assume sentiment analysis is available
        ...         sentiment = analyze_sentiment(response)
        ...         return {"text": response, "sentiment": sentiment}
        ...
        ...     def compare_responses(self, response_a, response_b, input_a, input_b):
        ...         sentiment_diff = response_a["sentiment"] - response_b["sentiment"]
        ...         return {
        ...             "sentiment_a": response_a["sentiment"],
        ...             "sentiment_b": response_b["sentiment"],
        ...             "sentiment_diff": sentiment_diff,
        ...             "potential_bias": abs(sentiment_diff) > 0.3,
        ...         }

    Example: Running batch comparisons
        >>> probe = GenderBiasProbe(name="bias_test")
        >>> comparison_pairs = [
        ...     ("Tell me about John the engineer", "Tell me about Jane the engineer"),
        ...     ("Describe Michael the nurse", "Describe Michelle the nurse"),
        ... ]
        >>> results = []
        >>> for input_a, input_b in comparison_pairs:
        ...     result = probe.run_comparison(model, input_a, input_b)
        ...     results.append(result)
        >>> bias_detected = sum(1 for r in results if r["comparison"]["potential_bias"])
        >>> print(f"Bias detected in {bias_detected}/{len(results)} comparisons")

    See Also:
        Probe: The base class for all probes.
        ScoredProbe: For probes that evaluate correctness against references.
    """

    def run_comparison(
        self,
        model: Any,
        input_a: Any,
        input_b: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the probe comparing two inputs and analyze differences.

        Executes the model on both inputs and then calls `compare_responses`
        to analyze any differences. This is the primary method for running
        comparative evaluations.

        The method:
        1. Generates a response for input_a
        2. Generates a response for input_b
        3. Calls compare_responses to analyze differences
        4. Returns all inputs, outputs, and comparison results

        Args:
            model (Any): The model to test. Should implement a generate() method
                that accepts the input data and returns a response.
            input_a (Any): The first input variant. Could be a string prompt,
                dict with structured data, or any format the model accepts.
            input_b (Any): The second input variant. Should be comparable to
                input_a (e.g., same format, similar structure).
            **kwargs (Any): Additional arguments passed to model.generate().
                Common kwargs include temperature, max_tokens, etc.

        Returns:
            dict[str, Any]: A comprehensive result dictionary containing:
                - input_a: The first input (as provided)
                - input_b: The second input (as provided)
                - response_a: The model's response to input_a
                - response_b: The model's response to input_b
                - comparison: Results from compare_responses()

        Example: Basic comparison
            >>> probe = GenderBiasProbe(name="bias_test")
            >>> result = probe.run_comparison(
            ...     model=my_model,
            ...     input_a="Describe a male doctor",
            ...     input_b="Describe a female doctor",
            ... )
            >>> print(result["comparison"]["response_identical"])
            False
            >>> print(result["comparison"]["length_diff"])
            -15

        Example: With custom kwargs
            >>> result = probe.run_comparison(
            ...     model=my_model,
            ...     input_a="Write about John the engineer",
            ...     input_b="Write about Jane the engineer",
            ...     temperature=0.0,  # Deterministic for fair comparison
            ...     max_tokens=500,
            ... )

        Example: Structured input comparison
            >>> result = probe.run_comparison(
            ...     model=my_model,
            ...     input_a={"prompt": "Summarize:", "text": "Article about men..."},
            ...     input_b={"prompt": "Summarize:", "text": "Article about women..."},
            ... )
            >>> if result["comparison"]["potential_bias"]:
            ...     print("Bias detected in responses")

        Example: Processing multiple comparisons
            >>> pairs = [
            ...     ("John is a nurse", "Jane is a nurse"),
            ...     ("Mike is a teacher", "Mary is a teacher"),
            ... ]
            >>> comparisons = [
            ...     probe.run_comparison(model, a, b)
            ...     for a, b in pairs
            ... ]
            >>> bias_count = sum(
            ...     1 for c in comparisons
            ...     if c["comparison"]["potential_bias"]
            ... )

        Note:
            Both inputs are run through the same model with the same kwargs
            to ensure a fair comparison. For deterministic comparisons,
            consider setting temperature=0.
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
    ) -> dict[str, Any]:
        """Compare two model responses and return analysis metrics.

        This method analyzes differences between two model responses. The base
        implementation provides simple metrics (length difference and identity).
        Subclasses should override this method to implement domain-specific
        comparison logic.

        Override this method to implement:
        - Sentiment analysis comparison
        - Semantic similarity scoring
        - Keyword presence/absence detection
        - Bias indicators specific to your domain
        - Statistical tests for significance

        Args:
            response_a (str): The model's response to input_a.
            response_b (str): The model's response to input_b.
            input_a (Any): The first input that generated response_a. Can be
                used for context-aware comparisons.
            input_b (Any): The second input that generated response_b. Can be
                used for context-aware comparisons.

        Returns:
            dict[str, Any]: A dictionary with comparison metrics. The base
                implementation returns:
                - length_diff (int): Difference in response lengths (a - b)
                - response_identical (bool): Whether responses are exactly equal

                Subclasses typically add:
                - potential_bias (bool): Whether bias was detected
                - similarity (float): Semantic similarity score
                - sentiment_diff (float): Difference in sentiment
                - keywords_diff (dict): Differential keyword analysis

        Example: Base implementation usage
            >>> probe = ComparativeProbe(name="test")
            >>> comparison = probe.compare_responses(
            ...     response_a="The doctor is skilled.",
            ...     response_b="The doctor is very skilled and compassionate.",
            ...     input_a="Describe a male doctor",
            ...     input_b="Describe a female doctor",
            ... )
            >>> comparison
            {'length_diff': -20, 'response_identical': False}

        Example: Override for sentiment comparison
            >>> def compare_responses(self, response_a, response_b, input_a, input_b):
            ...     base = super().compare_responses(response_a, response_b, input_a, input_b)
            ...     sentiment_a = analyze_sentiment(response_a)  # -1 to 1
            ...     sentiment_b = analyze_sentiment(response_b)
            ...     base["sentiment_a"] = sentiment_a
            ...     base["sentiment_b"] = sentiment_b
            ...     base["sentiment_diff"] = sentiment_a - sentiment_b
            ...     base["potential_bias"] = abs(sentiment_a - sentiment_b) > 0.3
            ...     return base

        Example: Override for keyword analysis
            >>> def compare_responses(self, response_a, response_b, input_a, input_b):
            ...     positive_words = {"excellent", "skilled", "leader", "capable"}
            ...     negative_words = {"average", "typical", "adequate"}
            ...
            ...     pos_a = sum(1 for w in positive_words if w in response_a.lower())
            ...     pos_b = sum(1 for w in positive_words if w in response_b.lower())
            ...     neg_a = sum(1 for w in negative_words if w in response_a.lower())
            ...     neg_b = sum(1 for w in negative_words if w in response_b.lower())
            ...
            ...     return {
            ...         "positive_count": {"a": pos_a, "b": pos_b},
            ...         "negative_count": {"a": neg_a, "b": neg_b},
            ...         "potential_bias": abs(pos_a - pos_b) > 1 or abs(neg_a - neg_b) > 1,
            ...         "response_identical": response_a == response_b,
            ...     }

        Example: Override for structural comparison
            >>> def compare_responses(self, response_a, response_b, input_a, input_b):
            ...     # Compare structural elements
            ...     sentences_a = response_a.split(".")
            ...     sentences_b = response_b.split(".")
            ...     return {
            ...         "sentence_count_a": len(sentences_a),
            ...         "sentence_count_b": len(sentences_b),
            ...         "word_count_a": len(response_a.split()),
            ...         "word_count_b": len(response_b.split()),
            ...         "structure_similar": abs(len(sentences_a) - len(sentences_b)) <= 1,
            ...         "response_identical": response_a == response_b,
            ...     }

        Note:
            The input parameters (input_a, input_b) are provided for context-aware
            comparisons. For example, you might want to check if the model's
            description length correlates with gendered language in the input.
        """
        return {
            "length_diff": len(response_a) - len(response_b),
            "response_identical": response_a == response_b,
        }
