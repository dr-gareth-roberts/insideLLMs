"""Output validation wrapper for validating insideLLMs data against versioned schemas.

This module provides the :class:`OutputValidator` class, which wraps Pydantic-based
schema validation with configurable error handling modes. It serves as the primary
interface for validating output data from insideLLMs probe runs, harness results,
and other structured outputs against versioned schemas.

The validator supports two modes of operation:
    - **strict mode**: Raises :class:`~insideLLMs.schemas.exceptions.OutputValidationError`
      when validation fails. Use this mode in production pipelines where invalid data
      should halt processing.
    - **warn mode**: Issues a :class:`RuntimeWarning` and returns the original unvalidated
      data. Use this mode during development or when processing legacy data that may
      not fully conform to current schemas.

The validator integrates with the :class:`~insideLLMs.schemas.registry.SchemaRegistry`
for schema lookup and supports both Pydantic v1 and v2 model validation APIs.

Key Features:
    - Versioned schema validation with SemVer support
    - Configurable error handling (strict vs warn)
    - Automatic dataclass-to-dict conversion for validation
    - Support for both Pydantic v1 (parse_obj) and v2 (model_validate) APIs
    - Lazy schema loading for performance
    - Thread-safe schema caching via the registry

Architecture:
    The validation flow follows these steps:
        1. Schema lookup via SchemaRegistry.get_model()
        2. Version normalization (e.g., "1.0" -> "1.0.0")
        3. Dataclass-to-dict conversion if needed via _to_plain()
        4. Pydantic model validation (v1 or v2 API)
        5. Error handling based on mode (strict: raise, warn: return original)

Examples:
    Basic validation in strict mode (default):

        >>> from insideLLMs.schemas.validator import OutputValidator
        >>> validator = OutputValidator()
        >>> data = {"input": "What is 2+2?", "output": "4", "status": "success"}
        >>> result = validator.validate("ProbeResult", data)
        >>> result.input
        'What is 2+2?'
        >>> result.status
        'success'

    Validation with explicit version:

        >>> from insideLLMs.schemas.validator import OutputValidator
        >>> validator = OutputValidator()
        >>> data = {
        ...     "input": "Hello, world",
        ...     "output": "Hi there!",
        ...     "status": "success",
        ...     "metadata": {"tokens": 50}
        ... }
        >>> result = validator.validate(
        ...     "ProbeResult",
        ...     data,
        ...     schema_version="1.0.1"
        ... )
        >>> hasattr(result, 'input')
        True

    Using warn mode for lenient validation:

        >>> import warnings
        >>> from insideLLMs.schemas.validator import OutputValidator
        >>> validator = OutputValidator()
        >>> invalid_data = {"incomplete": "data"}
        >>> with warnings.catch_warnings(record=True) as w:
        ...     warnings.simplefilter("always")
        ...     result = validator.validate(
        ...         "ProbeResult",
        ...         invalid_data,
        ...         mode="warn"
        ...     )
        ...     # Result is the original data, warning was issued
        ...     assert result == invalid_data
        ...     assert len(w) == 1
        ...     assert "Output validation failed" in str(w[0].message)

    Catching validation errors:

        >>> from insideLLMs.schemas.validator import OutputValidator
        >>> from insideLLMs.schemas.exceptions import OutputValidationError
        >>> validator = OutputValidator()
        >>> try:
        ...     validator.validate(
        ...         "ProbeResult",
        ...         {"missing": "required fields"},
        ...         mode="strict"
        ...     )
        ... except OutputValidationError as e:
        ...     print(f"Schema: {e.schema_name}")
        ...     print(f"Version: {e.schema_version}")
        Schema: ProbeResult
        Version: 1.0.0

    Validating dataclass instances:

        >>> from dataclasses import dataclass
        >>> from insideLLMs.schemas.validator import OutputValidator
        >>> @dataclass
        ... class MyResult:
        ...     input: str
        ...     output: str
        ...     status: str
        >>> validator = OutputValidator()
        >>> my_data = MyResult(input="test", output="response", status="success")
        >>> validated = validator.validate("ProbeResult", my_data)
        >>> validated.input
        'test'

    Batch validation with error collection:

        >>> from insideLLMs.schemas.validator import OutputValidator
        >>> from insideLLMs.schemas.exceptions import OutputValidationError
        >>> validator = OutputValidator()
        >>> records = [
        ...     {"input": "q1", "status": "success"},
        ...     {"invalid": "record"},
        ...     {"input": "q2", "status": "error", "error_message": "timeout"}
        ... ]
        >>> valid_results, errors = [], []
        >>> for record in records:
        ...     try:
        ...         valid_results.append(validator.validate("ProbeResult", record))
        ...     except OutputValidationError as e:
        ...         errors.append({"data": record, "error": str(e)})
        >>> len(valid_results)
        2
        >>> len(errors)
        1

    Using a custom registry:

        >>> from insideLLMs.schemas.validator import OutputValidator
        >>> from insideLLMs.schemas.registry import SchemaRegistry
        >>> custom_registry = SchemaRegistry()
        >>> validator = OutputValidator(registry=custom_registry)
        >>> # Validator now uses the custom registry for lookups

    Validating run manifests with timestamps:

        >>> from insideLLMs.schemas.validator import OutputValidator
        >>> from datetime import datetime
        >>> validator = OutputValidator()
        >>> manifest_data = {
        ...     "run_id": "run-abc123",
        ...     "created_at": datetime.now().isoformat(),
        ...     "started_at": datetime.now().isoformat(),
        ...     "completed_at": datetime.now().isoformat(),
        ...     "run_completed": True,
        ...     "model": {"model_id": "gpt-4"},
        ...     "probe": {"probe_id": "test-probe"},
        ... }
        >>> manifest = validator.validate(
        ...     "RunManifest",
        ...     manifest_data,
        ...     schema_version="1.0.1"
        ... )
        >>> manifest.run_id
        'run-abc123'

    Processing JSONL records:

        >>> from insideLLMs.schemas.validator import OutputValidator
        >>> validator = OutputValidator()
        >>> # Simulate reading JSONL records
        >>> jsonl_records = [
        ...     {"input": "Question 1?", "status": "success", "output": "Answer 1"},
        ...     {"input": "Question 2?", "status": "success", "output": "Answer 2"},
        ... ]
        >>> validated_records = [
        ...     validator.validate("ProbeResult", r) for r in jsonl_records
        ... ]
        >>> len(validated_records)
        2

Attributes:
    ValidationMode: Type alias for validation modes, either "strict" or "warn".
        Use "strict" for production pipelines and "warn" for development or
        legacy data processing.

Note:
    The validator converts dataclass instances to dictionaries before validation
    using :func:`dataclasses.asdict`. This ensures consistent validation behavior
    regardless of whether the input is a dict or a dataclass instance.

Note:
    Schema versions are normalized to full SemVer format (e.g., "1.0" becomes
    "1.0.0") before lookup. This ensures consistent behavior when specifying
    versions in either short or full format.

Note:
    The module catches generic Exception during validation to handle both
    Pydantic v1 and v2 ValidationError types without explicit imports. This
    keeps Pydantic as an optional dependency at the module level.

See Also:
    :class:`~insideLLMs.schemas.registry.SchemaRegistry`: For schema lookup and
        registration.
    :class:`~insideLLMs.schemas.exceptions.OutputValidationError`: The exception
        raised when validation fails in strict mode.
    :func:`~insideLLMs.schemas.registry.normalize_semver`: For version string
        normalization.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, is_dataclass
from typing import Any, Literal, Optional

from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION
from insideLLMs.schemas.exceptions import OutputValidationError
from insideLLMs.schemas.registry import SchemaRegistry, normalize_semver

ValidationMode = Literal["strict", "warn"]


def _to_plain(obj: Any) -> Any:
    """Convert an object to a plain dictionary if it's a dataclass instance.

    This helper function ensures that dataclass instances are converted to
    dictionaries before being passed to Pydantic for validation. Non-dataclass
    objects are returned unchanged.

    This conversion is necessary because Pydantic's validation methods expect
    either dict-like objects or raw values, and dataclass instances need to
    be serialized to dictionaries for proper field-by-field validation.

    Args:
        obj: Any object to potentially convert. Can be a dataclass instance,
            a dictionary, a Pydantic model, or any other Python object.

    Returns:
        If ``obj`` is a dataclass instance (not a dataclass type), returns
        a dictionary representation via :func:`dataclasses.asdict`.
        Otherwise, returns the original object unchanged.

    Examples:
        Converting a dataclass instance to dictionary:

            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Person:
            ...     name: str
            ...     age: int
            >>> person = Person(name="Alice", age=30)
            >>> result = _to_plain(person)
            >>> result
            {'name': 'Alice', 'age': 30}
            >>> isinstance(result, dict)
            True

        Dictionary objects pass through unchanged:

            >>> _to_plain({"key": "value"})
            {'key': 'value'}
            >>> original = {"nested": {"data": [1, 2, 3]}}
            >>> _to_plain(original) is original
            True

        Primitive types pass through unchanged:

            >>> _to_plain("string")
            'string'
            >>> _to_plain(42)
            42
            >>> _to_plain(3.14)
            3.14
            >>> _to_plain(None) is None
            True

        Lists and other collections pass through:

            >>> _to_plain([1, 2, 3])
            [1, 2, 3]
            >>> _to_plain({"a", "b"})
            {'a', 'b'}

        Dataclass types (not instances) pass through:

            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class MyClass:
            ...     value: int
            >>> _to_plain(MyClass) is MyClass  # The class itself, not an instance
            True

        Nested dataclasses are fully converted recursively:

            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Address:
            ...     street: str
            ...     city: str
            >>> @dataclass
            ... class Contact:
            ...     name: str
            ...     address: Address
            >>> contact = Contact(
            ...     name="Bob",
            ...     address=Address(street="123 Main St", city="Springfield")
            ... )
            >>> result = _to_plain(contact)
            >>> result["address"]["city"]
            'Springfield'
            >>> isinstance(result["address"], dict)
            True

        Dataclass with list of nested dataclasses:

            >>> from dataclasses import dataclass
            >>> from typing import List
            >>> @dataclass
            ... class Item:
            ...     id: int
            ...     name: str
            >>> @dataclass
            ... class Order:
            ...     items: List[Item]
            >>> order = Order(items=[Item(1, "A"), Item(2, "B")])
            >>> result = _to_plain(order)
            >>> result["items"][0]["name"]
            'A'

    Note:
        This function uses :func:`dataclasses.is_dataclass` with an additional
        check to ensure the object is an instance (not a type). This prevents
        accidental conversion of dataclass classes themselves.

    Note:
        The conversion uses :func:`dataclasses.asdict` which recursively converts
        nested dataclass instances. This may have performance implications for
        deeply nested structures.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


class OutputValidator:
    """Validate insideLLMs outputs against versioned schemas.

    The OutputValidator is the primary interface for validating structured output
    data from insideLLMs against versioned Pydantic schemas. It provides a simple
    API for schema validation with configurable error handling behavior.

    The validator integrates with :class:`~insideLLMs.schemas.registry.SchemaRegistry`
    for schema lookup and supports both Pydantic v1 and v2 validation APIs. It
    automatically converts dataclass instances to dictionaries before validation.

    The validator supports two modes of operation:
        - ``strict`` (default): Raises :class:`OutputValidationError` when validation
          fails. Use this mode in production pipelines where invalid data should
          halt processing.
        - ``warn``: Issues a :class:`RuntimeWarning` and returns the original data
          unchanged. Use this mode during development or when processing legacy data.

    Parameters:
        registry: Optional :class:`~insideLLMs.schemas.registry.SchemaRegistry`
            instance for schema lookups. If not provided, a new default registry
            is created. Use a custom registry to inject test schemas or to share
            a registry across multiple validators.

    Attributes:
        registry (SchemaRegistry): The schema registry used for looking up
            Pydantic models by name and version. This registry is used for all
            ``validate()`` calls and caches schema lookups for performance.

    Examples:
        Creating a validator with default registry:

            >>> from insideLLMs.schemas.validator import OutputValidator
            >>> validator = OutputValidator()
            >>> isinstance(validator.registry, SchemaRegistry)
            True

        Creating a validator with a custom registry:

            >>> from insideLLMs.schemas.validator import OutputValidator
            >>> from insideLLMs.schemas.registry import SchemaRegistry
            >>> custom_registry = SchemaRegistry()
            >>> validator = OutputValidator(registry=custom_registry)
            >>> validator.registry is custom_registry
            True

        Validating probe results:

            >>> from insideLLMs.schemas.validator import OutputValidator
            >>> validator = OutputValidator()
            >>> data = {
            ...     "input": "What is the capital of France?",
            ...     "output": "Paris",
            ...     "status": "success"
            ... }
            >>> result = validator.validate("ProbeResult", data)
            >>> result.input
            'What is the capital of France?'
            >>> result.output
            'Paris'

        Validating with optional fields:

            >>> from insideLLMs.schemas.validator import OutputValidator
            >>> validator = OutputValidator()
            >>> data = {
            ...     "input": "Complex question?",
            ...     "output": "Detailed answer.",
            ...     "status": "success",
            ...     "latency_ms": 245.7,
            ...     "metadata": {"model": "gpt-4", "temperature": 0.7}
            ... }
            >>> result = validator.validate("ProbeResult", data)
            >>> result.latency_ms
            245.7
            >>> result.metadata["temperature"]
            0.7

        Validating run manifests:

            >>> from insideLLMs.schemas.validator import OutputValidator
            >>> from datetime import datetime
            >>> validator = OutputValidator()
            >>> manifest_data = {
            ...     "run_id": "run-abc123",
            ...     "created_at": datetime.now().isoformat(),
            ...     "started_at": datetime.now().isoformat(),
            ...     "completed_at": datetime.now().isoformat(),
            ...     "run_completed": True,
            ...     "model": {"model_id": "gpt-4"},
            ...     "probe": {"probe_id": "test-probe"},
            ... }
            >>> manifest = validator.validate(
            ...     "RunManifest",
            ...     manifest_data,
            ...     schema_version="1.0.1"
            ... )
            >>> manifest.run_id
            'run-abc123'

        Using in a validation pipeline:

            >>> from insideLLMs.schemas.validator import OutputValidator
            >>> from insideLLMs.schemas.exceptions import OutputValidationError
            >>> class ValidationPipeline:
            ...     def __init__(self):
            ...         self.validator = OutputValidator()
            ...         self.errors = []
            ...
            ...     def process(self, records, schema_name):
            ...         validated = []
            ...         for record in records:
            ...             try:
            ...                 validated.append(
            ...                     self.validator.validate(schema_name, record)
            ...                 )
            ...             except OutputValidationError as e:
            ...                 self.errors.append(e)
            ...         return validated
            >>> pipeline = ValidationPipeline()
            >>> results = pipeline.process(
            ...     [{"input": "q1", "status": "success"}],
            ...     "ProbeResult"
            ... )
            >>> len(results)
            1

        Batch validation with mixed valid and invalid records:

            >>> from insideLLMs.schemas.validator import OutputValidator
            >>> from insideLLMs.schemas.exceptions import OutputValidationError
            >>> validator = OutputValidator()
            >>> batch = [
            ...     {"input": "Valid 1", "status": "success"},
            ...     {"invalid": "missing required fields"},
            ...     {"input": "Valid 2", "status": "error", "error_message": "timeout"},
            ... ]
            >>> validated, failed = [], []
            >>> for record in batch:
            ...     try:
            ...         validated.append(validator.validate("ProbeResult", record))
            ...     except OutputValidationError:
            ...         failed.append(record)
            >>> len(validated)
            2
            >>> len(failed)
            1

        Sharing a registry across multiple validators:

            >>> from insideLLMs.schemas.validator import OutputValidator
            >>> from insideLLMs.schemas.registry import SchemaRegistry
            >>> shared_registry = SchemaRegistry()
            >>> validator1 = OutputValidator(registry=shared_registry)
            >>> validator2 = OutputValidator(registry=shared_registry)
            >>> validator1.registry is validator2.registry
            True

    See Also:
        :class:`~insideLLMs.schemas.registry.SchemaRegistry`: For direct schema
            lookup and JSON Schema generation.
        :class:`~insideLLMs.schemas.exceptions.OutputValidationError`: Exception
            raised when strict validation fails.
        :data:`ValidationMode`: Type alias for the validation mode parameter.
    """

    def __init__(self, registry: Optional[SchemaRegistry] = None) -> None:
        """Initialize an OutputValidator with an optional schema registry.

        Creates a new validator instance configured with the specified schema
        registry. If no registry is provided, a new default registry is created
        automatically.

        Args:
            registry: Optional :class:`~insideLLMs.schemas.registry.SchemaRegistry`
                instance to use for schema lookups. If ``None``, a new registry
                is instantiated. Passing a custom registry allows sharing cached
                schemas across validators or injecting test schemas.

        Examples:
            Default initialization (creates new registry):

                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> validator = OutputValidator()
                >>> validator.registry is not None
                True
                >>> isinstance(validator.registry, SchemaRegistry)
                True

            With explicit registry for shared caching:

                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> from insideLLMs.schemas.registry import SchemaRegistry
                >>> registry = SchemaRegistry()
                >>> validator = OutputValidator(registry=registry)
                >>> validator.registry is registry
                True

            Sharing registry between multiple validators:

                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> from insideLLMs.schemas.registry import SchemaRegistry
                >>> shared_registry = SchemaRegistry()
                >>> validator1 = OutputValidator(registry=shared_registry)
                >>> validator2 = OutputValidator(registry=shared_registry)
                >>> validator1.registry is validator2.registry
                True
                >>> # Schema cache is shared, improving performance

            Creating validator in test setup:

                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> # In test setup, create validator with fresh registry
                >>> test_validator = OutputValidator()
                >>> # Validate test data
                >>> result = test_validator.validate(
                ...     "ProbeResult",
                ...     {"input": "test input", "status": "success"}
                ... )
                >>> result.status
                'success'

            Multiple independent validators (separate caches):

                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> validator_a = OutputValidator()  # Has its own registry
                >>> validator_b = OutputValidator()  # Has its own separate registry
                >>> validator_a.registry is validator_b.registry
                False
        """
        self.registry = registry or SchemaRegistry()

    def validate(
        self,
        schema_name: str,
        data: Any,
        *,
        schema_version: str = DEFAULT_SCHEMA_VERSION,
        mode: ValidationMode = "strict",
    ) -> Any:
        """Validate data against a versioned schema and return the parsed model.

        This method validates the provided data against the specified schema and
        returns a Pydantic model instance if validation succeeds. The behavior
        on validation failure depends on the ``mode`` parameter.

        The method automatically handles:
            - Dataclass-to-dict conversion via :func:`_to_plain`
            - Schema version normalization (e.g., "1.0" -> "1.0.0")
            - Pydantic v1 (``parse_obj``) and v2 (``model_validate``) API differences

        Args:
            schema_name: The canonical name of the schema to validate against.
                Common values include:

                - ``"ProbeResult"``: Individual probe execution results
                - ``"ResultRecord"``: JSONL record format for probe runs
                - ``"RunManifest"``: Run directory manifest metadata
                - ``"HarnessRecord"``: Harness evaluation records
                - ``"HarnessSummary"``: Harness run summary data
                - ``"BenchmarkSummary"``: Benchmark run output
                - ``"ComparisonReport"``: Comparison output
                - ``"DiffReport"``: Diff output
                - ``"ExportMetadata"``: Export bundle metadata

                Use :class:`~insideLLMs.schemas.registry.SchemaRegistry` class
                constants (e.g., ``SchemaRegistry.RUNNER_ITEM``) for type safety.

            data: The data to validate. Can be:

                - A dictionary with keys matching schema fields
                - A dataclass instance (converted to dict automatically)
                - A Pydantic model instance (converted to dict for re-validation)
                - Any object that can be converted to the schema format

            schema_version: The semantic version of the schema to validate against.
                Defaults to :data:`~insideLLMs.schemas.constants.DEFAULT_SCHEMA_VERSION`. Short-form
                versions (e.g., ``"1.0"``) are
                automatically normalized to full SemVer format (e.g., ``"1.0.0"``).
                Available versions can be queried via
                :meth:`~insideLLMs.schemas.registry.SchemaRegistry.available_versions`.

            mode: The validation mode controlling error handling behavior:

                - ``"strict"`` (default): Raises :class:`OutputValidationError` on
                  validation failure. Use in production pipelines.
                - ``"warn"``: Issues a :class:`RuntimeWarning` and returns the
                  original ``data`` unchanged. Use during development or for
                  legacy data processing.

        Returns:
            On successful validation:
                A Pydantic model instance populated with the validated data.
                The model type corresponds to the requested schema and version.

            On validation failure with ``mode="warn"``:
                The original ``data`` object, unchanged. A :class:`RuntimeWarning`
                is issued with details about the validation failure.

        Raises:
            OutputValidationError: When ``mode="strict"`` and validation fails.
                The exception contains:

                - ``schema_name``: The schema that validation was attempted against
                - ``schema_version``: The normalized version string
                - ``errors``: List of validation error messages

            KeyError: If the schema name or version is not found in the registry.
                This is raised by the underlying registry lookup.

            ImportError: If Pydantic is not installed. The schemas module requires
                Pydantic for model validation.

        Examples:
            Basic strict validation (success):

                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> validator = OutputValidator()
                >>> data = {
                ...     "input": "What is 2+2?",
                ...     "output": "4",
                ...     "status": "success"
                ... }
                >>> result = validator.validate("ProbeResult", data)
                >>> result.input
                'What is 2+2?'
                >>> result.status
                'success'
                >>> type(result).__name__
                'ProbeResult'

            Validation with all optional fields:

                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> validator = OutputValidator()
                >>> data = {
                ...     "input": "Complex query",
                ...     "output": "Detailed response",
                ...     "status": "success",
                ...     "latency_ms": 150.5,
                ...     "metadata": {"model": "gpt-4", "tokens": 100}
                ... }
                >>> result = validator.validate("ProbeResult", data)
                >>> result.latency_ms
                150.5
                >>> result.metadata["tokens"]
                100

            Strict validation with error (raises exception):

                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> from insideLLMs.schemas.exceptions import OutputValidationError
                >>> validator = OutputValidator()
                >>> try:
                ...     validator.validate("ProbeResult", {"bad": "data"})
                ... except OutputValidationError as e:
                ...     print(f"Failed: {e.schema_name}@{e.schema_version}")
                ...     print(f"Errors: {len(e.errors)} error(s)")
                Failed: ProbeResult@1.0.0
                Errors: 1 error(s)

            Warn mode validation (returns original data on failure):

                >>> import warnings
                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> validator = OutputValidator()
                >>> invalid_data = {"not": "valid"}
                >>> with warnings.catch_warnings(record=True) as w:
                ...     warnings.simplefilter("always")
                ...     result = validator.validate(
                ...         "ProbeResult",
                ...         invalid_data,
                ...         mode="warn"
                ...     )
                ...     # Original data returned
                ...     assert result is invalid_data
                ...     # Warning was issued
                ...     assert len(w) == 1
                ...     assert issubclass(w[0].category, RuntimeWarning)

            Warn mode with valid data (no warning):

                >>> import warnings
                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> validator = OutputValidator()
                >>> valid_data = {"input": "test", "status": "success"}
                >>> with warnings.catch_warnings(record=True) as w:
                ...     warnings.simplefilter("always")
                ...     result = validator.validate(
                ...         "ProbeResult",
                ...         valid_data,
                ...         mode="warn"
                ...     )
                ...     # No warnings for valid data
                ...     assert len(w) == 0
                ...     # Returns Pydantic model, not original dict
                ...     assert type(result).__name__ == "ProbeResult"

            Specifying schema version:

                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> validator = OutputValidator()
                >>> data = {"input": "test", "status": "success"}
                >>> # Validate against specific version
                >>> result_v1 = validator.validate(
                ...     "ProbeResult",
                ...     data,
                ...     schema_version="1.0.0"
                ... )
                >>> # Short-form versions are normalized
                >>> result_v1_short = validator.validate(
                ...     "ProbeResult",
                ...     data,
                ...     schema_version="1.0"
                ... )
                >>> type(result_v1) == type(result_v1_short)
                True

            Validating dataclass instances:

                >>> from dataclasses import dataclass
                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> @dataclass
                ... class ProbeData:
                ...     input: str
                ...     output: str
                ...     status: str
                >>> validator = OutputValidator()
                >>> probe_data = ProbeData(
                ...     input="question",
                ...     output="answer",
                ...     status="success"
                ... )
                >>> result = validator.validate("ProbeResult", probe_data)
                >>> result.input
                'question'
                >>> type(result).__name__
                'ProbeResult'

            Validating nested dataclasses:

                >>> from dataclasses import dataclass
                >>> from typing import Dict, Any
                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> @dataclass
                ... class ProbeWithMeta:
                ...     input: str
                ...     status: str
                ...     metadata: Dict[str, Any]
                >>> validator = OutputValidator()
                >>> data = ProbeWithMeta(
                ...     input="test",
                ...     status="success",
                ...     metadata={"key": "value"}
                ... )
                >>> result = validator.validate("ProbeResult", data)
                >>> result.metadata["key"]
                'value'

            Validating RunManifest with timestamps:

                >>> from datetime import datetime
                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> validator = OutputValidator()
                >>> manifest_data = {
                ...     "run_id": "run-123",
                ...     "created_at": datetime.now().isoformat(),
                ...     "started_at": datetime.now().isoformat(),
                ...     "completed_at": datetime.now().isoformat(),
                ...     "run_completed": True,
                ...     "model": {"model_id": "gpt-4"},
                ...     "probe": {"probe_id": "test-probe"},
                ... }
                >>> manifest = validator.validate(
                ...     "RunManifest",
                ...     manifest_data,
                ...     schema_version="1.0.1"
                ... )
                >>> manifest.run_id
                'run-123'
                >>> manifest.run_completed
                True

            Using with registry schema constants:

                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> from insideLLMs.schemas.registry import SchemaRegistry
                >>> validator = OutputValidator()
                >>> data = {"input": "test", "status": "success"}
                >>> result = validator.validate(
                ...     SchemaRegistry.RUNNER_ITEM,  # "ProbeResult"
                ...     data
                ... )
                >>> result.status
                'success'

            Processing a batch with mixed valid/invalid records:

                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> from insideLLMs.schemas.exceptions import OutputValidationError
                >>> validator = OutputValidator()
                >>> batch = [
                ...     {"input": "q1", "status": "success"},
                ...     {"missing": "fields"},
                ...     {"input": "q2", "status": "error", "error_message": "fail"}
                ... ]
                >>> validated, failed = [], []
                >>> for record in batch:
                ...     try:
                ...         validated.append(validator.validate("ProbeResult", record))
                ...     except OutputValidationError:
                ...         failed.append(record)
                >>> len(validated)
                2
                >>> len(failed)
                1

            Error handling with detailed error inspection:

                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> from insideLLMs.schemas.exceptions import OutputValidationError
                >>> validator = OutputValidator()
                >>> try:
                ...     validator.validate("ProbeResult", {"bad": "data"})
                ... except OutputValidationError as e:
                ...     # Access structured error information
                ...     print(f"Schema: {e.schema_name}")
                ...     print(f"Version: {e.schema_version}")
                ...     print(f"Error count: {len(e.errors)}")
                Schema: ProbeResult
                Version: 1.0.0
                Error count: 1

            Validation in a context manager pattern:

                >>> from insideLLMs.schemas.validator import OutputValidator
                >>> from insideLLMs.schemas.exceptions import OutputValidationError
                >>> validator = OutputValidator()
                >>> def safe_validate(data, schema="ProbeResult"):
                ...     try:
                ...         return validator.validate(schema, data), None
                ...     except OutputValidationError as e:
                ...         return None, e
                >>> result, error = safe_validate({"input": "test", "status": "success"})
                >>> result is not None
                True
                >>> error is None
                True

        Note:
            The validation mode affects only error handling behavior. Both modes
            perform the same validation logic; they differ only in how validation
            failures are reported.

        Note:
            When using ``mode="warn"``, the warning is issued with ``stacklevel=2``,
            so the warning appears to originate from the caller's code rather than
            from within this method.

        Note:
            The method catches generic Exception during validation to handle both
            Pydantic v1 ValidationError and Pydantic v2 ValidationError without
            explicit type imports. This allows the schemas module to function
            with either Pydantic version.

        See Also:
            :class:`~insideLLMs.schemas.exceptions.OutputValidationError`: The
                exception type raised in strict mode.
            :meth:`~insideLLMs.schemas.registry.SchemaRegistry.get_model`: The
                underlying method used for schema lookup.
            :func:`~insideLLMs.schemas.registry.normalize_semver`: Version
                normalization function.
        """

        model = self.registry.get_model(schema_name, schema_version)
        version = normalize_semver(schema_version)
        try:
            if hasattr(model, "model_validate"):
                return model.model_validate(_to_plain(data))
            return model.parse_obj(_to_plain(data))
        except Exception as e:  # ValidationError (v1/v2), keep generic
            errors = [str(e)]
            if mode == "warn":
                warnings.warn(
                    f"Output validation failed for {schema_name}@{version}: {errors[0]}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return data
            raise OutputValidationError(
                schema_name=schema_name, schema_version=version, errors=errors
            ) from e
