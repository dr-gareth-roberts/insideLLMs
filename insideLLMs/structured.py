"""Structured Output Parsing with Pydantic Integration.

This module provides utilities for extracting structured data from LLM outputs:
- Pydantic model integration for type-safe outputs
- JSON schema generation and prompting
- Automatic parsing with validation
- Retry logic for malformed outputs
- Support for nested models and complex types

Example:
    >>> from pydantic import BaseModel
    >>> from insideLLMs.structured import generate_structured
    >>> from insideLLMs.models import OpenAIModel
    >>>
    >>> class Person(BaseModel):
    ...     name: str
    ...     age: int
    ...     occupation: str
    >>>
    >>> model = OpenAIModel(model_name="gpt-4")
    >>> person = generate_structured(
    ...     model,
    ...     "Extract: John is a 30-year-old software engineer",
    ...     Person
    ... )
    >>> print(person.name, person.age)
    John 30
"""

import json
import re
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

if TYPE_CHECKING:
    from insideLLMs.models.base import Model

# Try to import Pydantic
try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
    ValidationError = None

T = TypeVar("T")


# =============================================================================
# Exceptions
# =============================================================================


class StructuredOutputError(Exception):
    """Base exception for structured output errors."""
    pass


class SchemaGenerationError(StructuredOutputError):
    """Error generating JSON schema from model."""
    pass


class ParsingError(StructuredOutputError):
    """Error parsing LLM output into structured format."""

    def __init__(self, message: str, raw_output: str, attempts: int = 1):
        super().__init__(message)
        self.raw_output = raw_output
        self.attempts = attempts


class ValidationErrorWrapper(StructuredOutputError):
    """Wrapper for Pydantic validation errors."""

    def __init__(self, message: str, validation_error: Any, raw_output: str):
        super().__init__(message)
        self.validation_error = validation_error
        self.raw_output = raw_output


# =============================================================================
# JSON Schema Generation
# =============================================================================


def _python_type_to_json_schema(python_type: Any) -> Dict[str, Any]:
    """Convert Python type annotation to JSON schema.

    Args:
        python_type: Python type annotation.

    Returns:
        JSON schema dictionary.
    """
    # Handle None type
    if python_type is type(None):
        return {"type": "null"}

    # Handle basic types
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        bytes: {"type": "string", "format": "byte"},
    }

    if python_type in type_mapping:
        return type_mapping[python_type]

    # Handle generic types
    origin = get_origin(python_type)
    args = get_args(python_type)

    if origin is list or origin is List:
        item_type = args[0] if args else Any
        return {
            "type": "array",
            "items": _python_type_to_json_schema(item_type),
        }

    if origin is dict or origin is Dict:
        value_type = args[1] if len(args) > 1 else Any
        return {
            "type": "object",
            "additionalProperties": _python_type_to_json_schema(value_type),
        }

    if origin is Union:
        # Handle Optional (Union[X, None])
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1 and type(None) in args:
            # This is Optional[X]
            schema = _python_type_to_json_schema(non_none_args[0])
            return {"anyOf": [schema, {"type": "null"}]}
        else:
            return {"anyOf": [_python_type_to_json_schema(a) for a in args]}

    # Handle Pydantic models
    if PYDANTIC_AVAILABLE and isinstance(python_type, type) and issubclass(python_type, BaseModel):
        return pydantic_to_json_schema(python_type)

    # Default to any
    return {}


def pydantic_to_json_schema(model: Type["BaseModel"]) -> Dict[str, Any]:
    """Convert Pydantic model to JSON schema.

    Args:
        model: Pydantic model class.

    Returns:
        JSON schema dictionary.
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic is required for structured outputs. Install with: pip install pydantic")

    # Use Pydantic's built-in schema generation
    try:
        # Pydantic v2
        if hasattr(model, "model_json_schema"):
            return model.model_json_schema()
        # Pydantic v1
        elif hasattr(model, "schema"):
            return model.schema()
        else:
            raise SchemaGenerationError(f"Cannot generate schema for {model}")
    except Exception as e:
        raise SchemaGenerationError(f"Failed to generate schema for {model}: {e}")


def dataclass_to_json_schema(dc: type) -> Dict[str, Any]:
    """Convert dataclass to JSON schema.

    Args:
        dc: Dataclass type.

    Returns:
        JSON schema dictionary.
    """
    import dataclasses

    if not dataclasses.is_dataclass(dc):
        raise SchemaGenerationError(f"{dc} is not a dataclass")

    properties = {}
    required = []

    for field in dataclasses.fields(dc):
        properties[field.name] = _python_type_to_json_schema(field.type)

        # Check if field has default
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            required.append(field.name)

    schema = {
        "type": "object",
        "properties": properties,
    }

    if required:
        schema["required"] = required

    return schema


def get_json_schema(output_type: type) -> Dict[str, Any]:
    """Get JSON schema for any supported type.

    Args:
        output_type: Type to get schema for.

    Returns:
        JSON schema dictionary.
    """
    import dataclasses

    if PYDANTIC_AVAILABLE and isinstance(output_type, type) and issubclass(output_type, BaseModel):
        return pydantic_to_json_schema(output_type)
    elif dataclasses.is_dataclass(output_type):
        return dataclass_to_json_schema(output_type)
    else:
        return _python_type_to_json_schema(output_type)


# =============================================================================
# JSON Extraction
# =============================================================================


def extract_json(text: str) -> str:
    """Extract JSON from text that may contain markdown or other content.

    Args:
        text: Text potentially containing JSON.

    Returns:
        Extracted JSON string.

    Raises:
        ParsingError: If no valid JSON found.
    """
    # Try to find JSON in code blocks
    code_block_patterns = [
        r"```json\s*([\s\S]*?)```",
        r"```\s*([\s\S]*?)```",
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, text)
        if match:
            json_str = match.group(1).strip()
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                continue

    # Try to find raw JSON object or array
    json_patterns = [
        r"(\{[\s\S]*\})",  # Object
        r"(\[[\s\S]*\])",  # Array
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue

    # Try the entire text
    try:
        json.loads(text.strip())
        return text.strip()
    except json.JSONDecodeError:
        pass

    raise ParsingError(f"Could not extract valid JSON from output", text)


def parse_json(text: str) -> Any:
    """Parse JSON from text.

    Args:
        text: Text containing JSON.

    Returns:
        Parsed JSON value.
    """
    json_str = extract_json(text)
    return json.loads(json_str)


# =============================================================================
# Output Parsing
# =============================================================================


def parse_to_type(data: Any, output_type: Type[T]) -> T:
    """Parse data into the specified type.

    Args:
        data: Data to parse (usually dict from JSON).
        output_type: Target type.

    Returns:
        Instance of output_type.
    """
    import dataclasses

    if PYDANTIC_AVAILABLE and isinstance(output_type, type) and issubclass(output_type, BaseModel):
        # Pydantic model
        try:
            # Pydantic v2
            if hasattr(output_type, "model_validate"):
                return output_type.model_validate(data)
            # Pydantic v1
            else:
                return output_type.parse_obj(data)
        except ValidationError as e:
            raise ValidationErrorWrapper(
                f"Validation failed for {output_type.__name__}",
                e,
                str(data)
            )

    elif dataclasses.is_dataclass(output_type):
        # Dataclass
        try:
            return output_type(**data)
        except TypeError as e:
            raise ParsingError(f"Failed to create {output_type.__name__}: {e}", str(data))

    elif output_type in (dict, Dict):
        return data

    elif output_type in (list, List):
        return data

    elif output_type in (str, int, float, bool):
        return output_type(data)

    else:
        # Try direct instantiation
        try:
            return output_type(**data) if isinstance(data, dict) else output_type(data)
        except Exception as e:
            raise ParsingError(f"Failed to create {output_type}: {e}", str(data))


# =============================================================================
# Prompt Templates
# =============================================================================


STRUCTURED_OUTPUT_SYSTEM_PROMPT = """You are a helpful assistant that extracts structured information from text.
Always respond with valid JSON that matches the specified schema exactly.
Do not include any explanation or text outside the JSON."""

STRUCTURED_OUTPUT_TEMPLATE = """Extract the requested information and return it as JSON.

{instructions}

Output Schema:
```json
{schema}
```

{input_section}

Respond with ONLY the JSON object, no other text."""

STRUCTURED_OUTPUT_TEMPLATE_WITH_EXAMPLES = """Extract the requested information and return it as JSON.

{instructions}

Output Schema:
```json
{schema}
```

Examples:
{examples}

{input_section}

Respond with ONLY the JSON object, no other text."""


# =============================================================================
# Structured Output Generator
# =============================================================================


@dataclass
class StructuredOutputConfig:
    """Configuration for structured output generation.

    Attributes:
        max_retries: Maximum number of retry attempts on parsing failure.
        temperature: Temperature for generation (0 for deterministic).
        include_schema_in_prompt: Whether to include JSON schema in prompt.
        instructions: Additional instructions for extraction.
        examples: Example input/output pairs for few-shot prompting.
    """

    max_retries: int = 3
    temperature: float = 0.0
    include_schema_in_prompt: bool = True
    instructions: str = ""
    examples: Optional[List[Dict[str, Any]]] = None


class StructuredOutputGenerator(Generic[T]):
    """Generator for structured outputs from LLM responses.

    Handles schema generation, prompting, parsing, and validation.
    """

    def __init__(
        self,
        model: "Model",
        output_type: Type[T],
        config: Optional[StructuredOutputConfig] = None,
    ):
        """Initialize the generator.

        Args:
            model: LLM model to use.
            output_type: Target Pydantic model or type.
            config: Generation configuration.
        """
        self.model = model
        self.output_type = output_type
        self.config = config or StructuredOutputConfig()
        self._schema = get_json_schema(output_type)

    def _build_prompt(self, input_text: str) -> str:
        """Build the extraction prompt.

        Args:
            input_text: Input text to extract from.

        Returns:
            Complete prompt string.
        """
        schema_str = json.dumps(self._schema, indent=2)

        if self.config.examples:
            examples_str = "\n\n".join(
                f"Input: {ex.get('input', '')}\nOutput: {json.dumps(ex.get('output', {}))}"
                for ex in self.config.examples
            )
            return STRUCTURED_OUTPUT_TEMPLATE_WITH_EXAMPLES.format(
                instructions=self.config.instructions or "Extract the information from the text.",
                schema=schema_str if self.config.include_schema_in_prompt else "See schema above",
                examples=examples_str,
                input_section=f"Input:\n{input_text}",
            )
        else:
            return STRUCTURED_OUTPUT_TEMPLATE.format(
                instructions=self.config.instructions or "Extract the information from the text.",
                schema=schema_str if self.config.include_schema_in_prompt else "See schema above",
                input_section=f"Input:\n{input_text}",
            )

    def generate(self, input_text: str, **model_kwargs: Any) -> T:
        """Generate structured output from input text.

        Args:
            input_text: Text to extract information from.
            **model_kwargs: Additional arguments for the model.

        Returns:
            Parsed instance of output_type.

        Raises:
            ParsingError: If parsing fails after all retries.
        """
        prompt = self._build_prompt(input_text)
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                # Generate response
                if hasattr(self.model, "chat"):
                    messages = [
                        {"role": "system", "content": STRUCTURED_OUTPUT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                    response = self.model.chat(
                        messages,
                        temperature=self.config.temperature,
                        **model_kwargs
                    )
                else:
                    full_prompt = f"{STRUCTURED_OUTPUT_SYSTEM_PROMPT}\n\n{prompt}"
                    response = self.model.generate(
                        full_prompt,
                        temperature=self.config.temperature,
                        **model_kwargs
                    )

                # Parse response
                data = parse_json(response)
                return parse_to_type(data, self.output_type)

            except (ParsingError, ValidationErrorWrapper) as e:
                last_error = e
                # Add context for retry
                if attempt < self.config.max_retries - 1:
                    prompt = self._build_prompt(input_text)
                    prompt += f"\n\nPrevious attempt failed with error: {str(e)}\nPlease try again with valid JSON."

        raise ParsingError(
            f"Failed to generate valid structured output after {self.config.max_retries} attempts: {last_error}",
            getattr(last_error, "raw_output", ""),
            self.config.max_retries,
        )

    def generate_batch(
        self,
        inputs: List[str],
        **model_kwargs: Any,
    ) -> List[T]:
        """Generate structured outputs for multiple inputs.

        Args:
            inputs: List of input texts.
            **model_kwargs: Additional arguments for the model.

        Returns:
            List of parsed outputs.
        """
        return [self.generate(text, **model_kwargs) for text in inputs]


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_structured(
    model: "Model",
    input_text: str,
    output_type: Type[T],
    instructions: str = "",
    max_retries: int = 3,
    examples: Optional[List[Dict[str, Any]]] = None,
    **model_kwargs: Any,
) -> T:
    """Generate structured output from a model.

    This is a convenience function for one-off structured generation.

    Args:
        model: LLM model to use.
        input_text: Text to extract information from.
        output_type: Target Pydantic model or type.
        instructions: Additional extraction instructions.
        max_retries: Maximum retry attempts.
        examples: Optional few-shot examples.
        **model_kwargs: Additional model arguments.

    Returns:
        Parsed instance of output_type.

    Example:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> person = generate_structured(model, "John is 30 years old", Person)
    """
    config = StructuredOutputConfig(
        max_retries=max_retries,
        instructions=instructions,
        examples=examples,
    )
    generator = StructuredOutputGenerator(model, output_type, config)
    return generator.generate(input_text, **model_kwargs)


def create_structured_generator(
    model: "Model",
    output_type: Type[T],
    instructions: str = "",
    examples: Optional[List[Dict[str, Any]]] = None,
    max_retries: int = 3,
) -> StructuredOutputGenerator[T]:
    """Create a reusable structured output generator.

    Args:
        model: LLM model to use.
        output_type: Target Pydantic model or type.
        instructions: Additional extraction instructions.
        examples: Optional few-shot examples.
        max_retries: Maximum retry attempts.

    Returns:
        Configured StructuredOutputGenerator.

    Example:
        >>> generator = create_structured_generator(model, Person)
        >>> person1 = generator.generate("John is 30")
        >>> person2 = generator.generate("Jane is 25")
    """
    config = StructuredOutputConfig(
        max_retries=max_retries,
        instructions=instructions,
        examples=examples,
    )
    return StructuredOutputGenerator(model, output_type, config)


# =============================================================================
# Model Extension
# =============================================================================


def add_structured_method(model: "Model") -> "Model":
    """Add generate_structured method to a model instance.

    This modifies the model in-place to add structured generation capability.

    Args:
        model: Model to extend.

    Returns:
        The same model with added method.

    Example:
        >>> model = add_structured_method(OpenAIModel())
        >>> person = model.generate_structured("John is 30", Person)
    """
    def generate_structured_method(
        input_text: str,
        output_type: Type[T],
        instructions: str = "",
        max_retries: int = 3,
        **kwargs: Any,
    ) -> T:
        return generate_structured(
            model,
            input_text,
            output_type,
            instructions=instructions,
            max_retries=max_retries,
            **kwargs,
        )

    model.generate_structured = generate_structured_method
    return model


# =============================================================================
# Result Viewer and Export Utilities
# =============================================================================


@dataclass
class StructuredResult(Generic[T]):
    """Container for structured output with metadata and export capabilities.

    Attributes:
        data: The parsed structured data.
        raw_response: Original LLM response.
        schema: JSON schema used for extraction.
        prompt: The prompt sent to the model.
        model_name: Name of the model used.
    """

    data: T
    raw_response: str
    schema: Dict[str, Any]
    prompt: str
    model_name: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result data to dictionary.

        Returns:
            Dictionary representation of the data.
        """
        if hasattr(self.data, "model_dump"):  # Pydantic v2
            return self.data.model_dump()
        elif hasattr(self.data, "dict"):  # Pydantic v1
            return self.data.dict()
        elif hasattr(self.data, "__dict__"):
            return {k: v for k, v in self.data.__dict__.items() if not k.startswith("_")}
        elif isinstance(self.data, dict):
            return self.data
        else:
            return {"value": self.data}

    def to_json(self, indent: int = 2) -> str:
        """Convert result to JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_html(self, title: str = "Structured Output") -> str:
        """Convert result to HTML for viewing.

        Args:
            title: Title for the HTML page.

        Returns:
            HTML string.
        """
        data_dict = self.to_dict()

        def dict_to_html_table(d: Dict, level: int = 0) -> str:
            rows = []
            indent = "  " * level
            for key, value in d.items():
                if isinstance(value, dict):
                    nested = dict_to_html_table(value, level + 1)
                    rows.append(f"{indent}<tr><td><strong>{key}</strong></td><td>{nested}</td></tr>")
                elif isinstance(value, list):
                    items = "<ul>" + "".join(f"<li>{v}</li>" for v in value) + "</ul>"
                    rows.append(f"{indent}<tr><td><strong>{key}</strong></td><td>{items}</td></tr>")
                else:
                    rows.append(f"{indent}<tr><td><strong>{key}</strong></td><td>{value}</td></tr>")
            return f"<table border='1' cellpadding='5'>\n{''.join(rows)}\n</table>"

        table = dict_to_html_table(data_dict)

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; margin: 10px 0; }}
        td {{ padding: 8px; vertical-align: top; }}
        .metadata {{ color: #666; font-size: 0.9em; margin-top: 20px; }}
        .json-view {{ background: #f5f5f5; padding: 10px; border-radius: 5px; }}
        pre {{ margin: 0; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <h2>Extracted Data</h2>
    {table}
    <h2>JSON View</h2>
    <div class="json-view">
        <pre>{self.to_json()}</pre>
    </div>
    <div class="metadata">
        <p><strong>Model:</strong> {self.model_name}</p>
        <p><strong>Schema Fields:</strong> {', '.join(self.schema.get('properties', {}).keys())}</p>
    </div>
</body>
</html>"""

    def to_markdown(self) -> str:
        """Convert result to Markdown format.

        Returns:
            Markdown string.
        """
        data_dict = self.to_dict()

        def dict_to_md(d: Dict, level: int = 0) -> str:
            lines = []
            indent = "  " * level
            for key, value in d.items():
                if isinstance(value, dict):
                    lines.append(f"{indent}- **{key}**:")
                    lines.append(dict_to_md(value, level + 1))
                elif isinstance(value, list):
                    lines.append(f"{indent}- **{key}**: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"{indent}- **{key}**: {value}")
            return "\n".join(lines)

        md = f"# Structured Output\n\n{dict_to_md(data_dict)}\n\n"
        md += f"## JSON\n\n```json\n{self.to_json()}\n```\n"
        return md

    def save_json(self, filepath: str) -> None:
        """Save result as JSON file.

        Args:
            filepath: Path to save to.
        """
        with open(filepath, "w") as f:
            f.write(self.to_json())

    def save_html(self, filepath: str, title: str = "Structured Output") -> None:
        """Save result as HTML file.

        Args:
            filepath: Path to save to.
            title: HTML page title.
        """
        with open(filepath, "w") as f:
            f.write(self.to_html(title))

    def __repr__(self) -> str:
        return f"StructuredResult({self.to_dict()})"


# =============================================================================
# Quick Setup Functions
# =============================================================================


def quick_extract(
    text: str,
    output_type: Type[T],
    model_name: str = "gpt-4",
    provider: str = "openai",
    api_key: Optional[str] = None,
    instructions: str = "",
    **model_kwargs: Any,
) -> StructuredResult[T]:
    """Quick one-liner for structured extraction with automatic model setup.

    This is the easiest way to extract structured data - just provide text
    and the expected output type.

    Args:
        text: Text to extract information from.
        output_type: Pydantic model, dataclass, or type for output.
        model_name: Model name (e.g., "gpt-4", "claude-3-opus", "llama-3").
        provider: Model provider ("openai", "anthropic", "huggingface").
        api_key: Optional API key (uses env var if not provided).
        instructions: Additional extraction instructions.
        **model_kwargs: Additional model configuration.

    Returns:
        StructuredResult containing parsed data and export methods.

    Example:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>>
        >>> result = quick_extract(
        ...     "John is a 30-year-old engineer",
        ...     Person,
        ...     model_name="gpt-4"
        ... )
        >>> print(result.data.name)  # John
        >>> result.save_html("person.html")  # Export to HTML
        >>> print(result.to_json())  # Get JSON
    """
    # Auto-detect and create model
    model = _create_model_from_name(model_name, provider, api_key, **model_kwargs)

    # Build prompt
    schema = get_json_schema(output_type)
    config = StructuredOutputConfig(instructions=instructions)
    generator = StructuredOutputGenerator(model, output_type, config)

    prompt = generator._build_prompt(text)

    # Generate
    if hasattr(model, "chat"):
        messages = [
            {"role": "system", "content": STRUCTURED_OUTPUT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw_response = model.chat(messages, temperature=0.0)
    else:
        raw_response = model.generate(
            f"{STRUCTURED_OUTPUT_SYSTEM_PROMPT}\n\n{prompt}",
            temperature=0.0
        )

    # Parse
    data = parse_json(raw_response)
    parsed = parse_to_type(data, output_type)

    return StructuredResult(
        data=parsed,
        raw_response=raw_response,
        schema=schema,
        prompt=prompt,
        model_name=model_name,
    )


def _create_model_from_name(
    model_name: str,
    provider: str,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> "Model":
    """Create a model instance from name and provider.

    Args:
        model_name: Name of the model.
        provider: Provider name.
        api_key: Optional API key.
        **kwargs: Additional model config.

    Returns:
        Configured model instance.
    """
    provider = provider.lower()

    if provider == "openai":
        from insideLLMs.models import OpenAIModel
        return OpenAIModel(model_name=model_name, api_key=api_key, **kwargs)

    elif provider == "anthropic":
        from insideLLMs.models import AnthropicModel
        return AnthropicModel(model_name=model_name, api_key=api_key, **kwargs)

    elif provider == "huggingface":
        from insideLLMs.models import HuggingFaceModel
        return HuggingFaceModel(model_name=model_name, **kwargs)

    elif provider == "dummy":
        from insideLLMs.models import DummyModel
        return DummyModel(name=model_name)

    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, anthropic, huggingface, dummy")


# =============================================================================
# Batch Processing with Results
# =============================================================================


def batch_extract(
    texts: List[str],
    output_type: Type[T],
    model: "Model",
    instructions: str = "",
    show_progress: bool = True,
) -> List[StructuredResult[T]]:
    """Extract structured data from multiple texts.

    Args:
        texts: List of texts to process.
        output_type: Target type for extraction.
        model: Model to use.
        instructions: Extraction instructions.
        show_progress: Whether to print progress.

    Returns:
        List of StructuredResult objects.

    Example:
        >>> texts = ["John is 30", "Jane is 25", "Bob is 40"]
        >>> results = batch_extract(texts, Person, model)
        >>> for r in results:
        ...     print(r.data.name, r.data.age)
    """
    results = []
    schema = get_json_schema(output_type)
    config = StructuredOutputConfig(instructions=instructions)
    generator = StructuredOutputGenerator(model, output_type, config)

    for i, text in enumerate(texts):
        if show_progress:
            print(f"Processing {i + 1}/{len(texts)}...", end="\r")

        try:
            prompt = generator._build_prompt(text)

            if hasattr(model, "chat"):
                messages = [
                    {"role": "system", "content": STRUCTURED_OUTPUT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                raw_response = model.chat(messages, temperature=0.0)
            else:
                raw_response = model.generate(
                    f"{STRUCTURED_OUTPUT_SYSTEM_PROMPT}\n\n{prompt}",
                    temperature=0.0
                )

            data = parse_json(raw_response)
            parsed = parse_to_type(data, output_type)

            results.append(StructuredResult(
                data=parsed,
                raw_response=raw_response,
                schema=schema,
                prompt=prompt,
                model_name=getattr(model, "name", "unknown"),
            ))

        except Exception as e:
            # Create error result
            results.append(StructuredResult(
                data=None,
                raw_response=str(e),
                schema=schema,
                prompt=text,
                model_name=getattr(model, "name", "unknown"),
            ))

    if show_progress:
        print(f"Processed {len(texts)} texts.       ")

    return results


def results_to_dataframe(results: List[StructuredResult]) -> Any:
    """Convert list of StructuredResults to pandas DataFrame.

    Args:
        results: List of StructuredResult objects.

    Returns:
        pandas DataFrame.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for DataFrame export. Install with: pip install pandas")

    rows = []
    for r in results:
        if r.data is not None:
            row = r.to_dict()
            rows.append(row)

    return pd.DataFrame(rows)


def results_to_html_report(
    results: List[StructuredResult],
    title: str = "Extraction Results",
) -> str:
    """Generate HTML report from multiple results.

    Args:
        results: List of StructuredResult objects.
        title: Report title.

    Returns:
        HTML string.
    """
    rows = []
    for i, r in enumerate(results, 1):
        if r.data is not None:
            data_dict = r.to_dict()
            cells = "".join(f"<td>{v}</td>" for v in data_dict.values())
            rows.append(f"<tr><td>{i}</td>{cells}</tr>")

    # Get headers from first valid result
    headers = ["#"]
    for r in results:
        if r.data is not None:
            headers.extend(r.to_dict().keys())
            break

    header_row = "".join(f"<th>{h}</th>" for h in headers)

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #ddd; }}
        .summary {{ color: #666; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="summary">Total results: {len(results)} | Successful: {sum(1 for r in results if r.data is not None)}</p>
    <table>
        <tr>{header_row}</tr>
        {"".join(rows)}
    </table>
</body>
</html>"""


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Exceptions
    "StructuredOutputError",
    "SchemaGenerationError",
    "ParsingError",
    "ValidationErrorWrapper",
    # Schema generation
    "pydantic_to_json_schema",
    "dataclass_to_json_schema",
    "get_json_schema",
    # Parsing
    "extract_json",
    "parse_json",
    "parse_to_type",
    # Configuration
    "StructuredOutputConfig",
    # Generator
    "StructuredOutputGenerator",
    # Result container
    "StructuredResult",
    # Convenience functions
    "generate_structured",
    "create_structured_generator",
    "add_structured_method",
    # Quick setup
    "quick_extract",
    # Batch processing
    "batch_extract",
    "results_to_dataframe",
    "results_to_html_report",
    # Constants
    "PYDANTIC_AVAILABLE",
]
