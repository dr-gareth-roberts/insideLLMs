"""Dataset exception types."""

from typing import Optional

from insideLLMs._exceptions.base import InsideLLMsError


class DatasetError(InsideLLMsError):
    """Base exception for dataset-related errors.

    This is the parent class for all exceptions that occur during dataset
    operations, including loading, parsing, and validation. Catching this
    exception will handle any dataset-related failure.

    Parameters
    ----------
    message : str
        Human-readable error message.
    details : dict[str, Any], optional
        Additional context about the error.

    Examples
    --------
    Catching any dataset-related error:

    >>> try:
    ...     dataset = DatasetLoader.load("path/to/data.csv")
    ...     dataset.validate()
    ... except DatasetError as e:
    ...     print(f"Dataset operation failed: {e}")

    Distinguishing between dataset error types:

    >>> try:
    ...     dataset = DatasetLoader.load(path)
    ... except DatasetNotFoundError:
    ...     print("File not found")
    ... except DatasetFormatError as e:
    ...     print(f"Invalid format: {e.details['reason']}")
    ... except DatasetError as e:
    ...     print(f"Other dataset error: {e}")

    Notes
    -----
    Subclasses include: DatasetNotFoundError, DatasetFormatError,
    DatasetValidationError.

    See Also
    --------
    InsideLLMsError : Parent class for all library errors.
    DatasetValidationError : For row-level validation failures.
    """

    pass


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset is not found.

    This exception is raised when attempting to load a dataset from a
    path that does not exist or is not accessible.

    Parameters
    ----------
    path : str
        The path to the dataset that was not found.

    Attributes
    ----------
    details : dict
        Contains 'path' key with the attempted path.

    Examples
    --------
    Handling missing dataset with fallback:

    >>> try:
    ...     dataset = DatasetLoader.load("data/custom_dataset.csv")
    ... except DatasetNotFoundError as e:
    ...     print(f"Dataset not found: {e.details['path']}")
    ...     dataset = DatasetLoader.load("data/default_dataset.csv")

    Validating dataset path before loading:

    >>> import os
    >>> def load_dataset_safe(path: str):
    ...     if not os.path.exists(path):
    ...         raise DatasetNotFoundError(path)
    ...     return DatasetLoader.load(path)

    Creating helpful error messages:

    >>> try:
    ...     dataset = DatasetLoader.load(user_provided_path)
    ... except DatasetNotFoundError as e:
    ...     print(f"Could not find dataset at: {e.details['path']}")
    ...     print("Please check the path and try again.")

    See Also
    --------
    DatasetFormatError : When dataset exists but has invalid format.
    ConfigNotFoundError : For missing configuration files.
    """

    def __init__(self, path: str):
        super().__init__(
            f"Dataset not found: {path}",
            {"path": path},
        )


class DatasetFormatError(DatasetError):
    """Raised when dataset format is invalid.

    This exception is raised when a dataset file exists but cannot be
    parsed due to format issues. This includes unsupported file types,
    malformed content, or missing required structure.

    Parameters
    ----------
    reason : str
        A description of why the format is invalid.
    expected_format : str, optional
        Description of the expected format for guidance.

    Attributes
    ----------
    details : dict
        Contains 'reason' and optionally 'expected_format'.

    Examples
    --------
    Handling format errors with expected format info:

    >>> try:
    ...     dataset = DatasetLoader.load("data.json")
    ... except DatasetFormatError as e:
    ...     print(f"Format error: {e.details['reason']}")
    ...     if 'expected_format' in e.details:
    ...         print(f"Expected: {e.details['expected_format']}")

    Attempting multiple formats:

    >>> def load_any_format(base_path: str):
    ...     for ext in ['.csv', '.json', '.parquet']:
    ...         try:
    ...             return DatasetLoader.load(base_path + ext)
    ...         except DatasetNotFoundError:
    ...             continue
    ...         except DatasetFormatError:
    ...             continue
    ...     raise DatasetError("No valid dataset found", {"path": base_path})

    Providing format conversion guidance:

    >>> try:
    ...     dataset = DatasetLoader.load("data.xlsx")
    ... except DatasetFormatError as e:
    ...     print(f"Error: {e}")
    ...     print("Tip: Convert Excel to CSV using pandas:")
    ...     print("  pd.read_excel('data.xlsx').to_csv('data.csv')")

    See Also
    --------
    DatasetNotFoundError : When dataset file doesn't exist.
    DatasetValidationError : When content fails validation.
    """

    def __init__(self, reason: str, expected_format: Optional[str] = None):
        details = {"reason": reason}
        if expected_format:
            details["expected_format"] = expected_format
        super().__init__(f"Invalid dataset format: {reason}", details)


class DatasetValidationError(DatasetError):
    """Raised when dataset validation fails.

    This exception is raised when dataset content fails validation checks.
    This includes type mismatches, missing required fields, constraint
    violations, or data quality issues at specific rows.

    Parameters
    ----------
    reason : str
        A description of why validation failed.
    row_index : int, optional
        The index of the row that failed validation (0-based).
    field : str, optional
        The name of the field that failed validation.

    Attributes
    ----------
    details : dict
        Contains 'reason', and optionally 'row_index' and 'field'.

    Examples
    --------
    Handling validation errors with row information:

    >>> try:
    ...     dataset.validate()
    ... except DatasetValidationError as e:
    ...     print(f"Validation failed: {e.details['reason']}")
    ...     if 'row_index' in e.details:
    ...         print(f"  At row: {e.details['row_index']}")
    ...     if 'field' in e.details:
    ...         print(f"  Field: {e.details['field']}")

    Collecting all validation errors:

    >>> errors = []
    >>> for idx, row in enumerate(dataset):
    ...     try:
    ...         validate_row(row)
    ...     except DatasetValidationError as e:
    ...         errors.append({
    ...             'row': idx,
    ...             'reason': e.details['reason'],
    ...             'field': e.details.get('field')
    ...         })
    >>> if errors:
    ...     print(f"Found {len(errors)} validation errors")

    Skipping invalid rows with logging:

    >>> valid_rows = []
    >>> for idx, row in enumerate(dataset):
    ...     try:
    ...         validate_row(row)
    ...         valid_rows.append(row)
    ...     except DatasetValidationError as e:
    ...         logging.warning(f"Skipping row {idx}: {e.details['reason']}")

    See Also
    --------
    DatasetFormatError : When overall format is invalid.
    ProbeValidationError : For probe-specific validation.
    ConfigValidationError : For configuration validation.
    """

    def __init__(
        self,
        reason: str,
        row_index: Optional[int] = None,
        field: Optional[str] = None,
    ):
        details = {"reason": reason}
        if row_index is not None:
            details["row_index"] = row_index
        if field:
            details["field"] = field
        super().__init__(f"Dataset validation failed: {reason}", details)
