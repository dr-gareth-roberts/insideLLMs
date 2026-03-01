"""Advanced prompt templating utilities for LLM applications.

This module provides a comprehensive toolkit for creating, formatting, validating,
and managing prompts for use with Large Language Model (LLM) APIs. It supports
various prompt engineering patterns including few-shot learning, chat-style
conversations, template composition, and prompt versioning.

Overview
--------
The module is organized around several core abstractions:

**Templates and Formatting**
    - `PromptTemplate`: Simple {placeholder} style templates with Python string formatting
    - `render_prompt`: Quick template rendering function
    - `render_jinja_template`: Advanced Jinja2 template support for complex logic

**Message Building**
    - `PromptMessage`: Dataclass representing a single conversation message
    - `PromptRole`: Enum defining standard chat roles (system, user, assistant, etc.)
    - `PromptBuilder`: Fluent API for constructing multi-turn conversations
    - `format_chat_messages`: Convert message lists to formatted strings

**Prompt Management**
    - `PromptLibrary`: Named template storage with versioning and tagging
    - `PromptVersion`: Versioned prompt metadata with content hashing
    - `PromptComposer`: Compose prompts from reusable sections

**Validation and Utilities**
    - `PromptValidator`: Validate prompts against custom rules
    - `extract_variables`: Extract placeholder names from templates
    - `validate_template_variables`: Check for missing variables
    - `escape_template`: Escape braces to prevent substitution
    - `truncate_prompt`: Truncate prompts to length limits
    - `estimate_tokens`: Estimate token count (delegates to tokens module)
    - `split_prompt_by_tokens`: Split prompts for chunked processing

**Few-Shot Learning**
    - `create_few_shot_prompt`: Build few-shot prompts with examples

Key Features
------------
- **Type Safety**: Full type hints for IDE support and static analysis
- **Fluent APIs**: Chainable methods for readable prompt construction
- **Versioning**: Track prompt versions with content hashing
- **Validation**: Custom validation rules for prompt quality
- **Jinja2 Support**: Optional advanced templating with conditionals and loops
- **Token Estimation**: Integrate with token counting for context management
- **Serialization**: Export/import prompt libraries as dictionaries

Examples
--------
Basic template usage:

>>> from insideLLMs.prompts.prompt_utils import PromptTemplate
>>> template = PromptTemplate("Translate the following {source_lang} to {target_lang}: {text}")
>>> prompt = template.format(source_lang="English", target_lang="French", text="Hello, world!")
>>> print(prompt)
Translate the following English to French: Hello, world!

Building chat-style prompts with PromptBuilder:

>>> from insideLLMs.prompts.prompt_utils import PromptBuilder
>>> builder = PromptBuilder()
>>> messages = (builder
...     .system("You are a helpful coding assistant specializing in Python.")
...     .user("How do I read a JSON file?")
...     .assistant("You can use the `json` module...")
...     .user("Can you show me an example?")
...     .build())
>>> print(messages[0])
{'role': 'system', 'content': 'You are a helpful coding assistant specializing in Python.'}

Creating few-shot prompts:

>>> from insideLLMs.prompts.prompt_utils import create_few_shot_prompt
>>> examples = [
...     {"input": "The food was great", "output": "positive"},
...     {"input": "Terrible service", "output": "negative"},
...     {"input": "It was okay", "output": "neutral"}
... ]
>>> prompt = create_few_shot_prompt(
...     instruction="Classify the sentiment of the text as positive, negative, or neutral.",
...     examples=examples,
...     query="The movie was absolutely fantastic!"
... )

Using the prompt library for version management:

>>> from insideLLMs.prompts.prompt_utils import PromptLibrary
>>> library = PromptLibrary()
>>> library.register(
...     name="summarize",
...     content="Summarize the following text in {num_sentences} sentences:\\n\\n{text}",
...     version="1.0",
...     description="Text summarization prompt",
...     tags=["summarization", "text-processing"]
... )
>>> template = library.get("summarize")
>>> prompt = template.format(num_sentences=3, text="Your long text here...")

Composing prompts from reusable sections:

>>> from insideLLMs.prompts.prompt_utils import PromptComposer
>>> composer = PromptComposer()
>>> composer.add_section("persona", "You are a senior software engineer with 10 years of experience.")
>>> composer.add_section("task", "Review the following code for bugs, security issues, and style.")
>>> composer.add_section("output_format", "Provide your feedback in bullet points.")
>>> prompt = composer.compose(["persona", "task", "output_format"])

Validating prompts:

>>> from insideLLMs.prompts.prompt_utils import PromptValidator
>>> validator = PromptValidator()
>>> validator.require_variables("name", "task")
>>> validator.add_rule("max_length", lambda p: len(p) <= 2000, "Prompt exceeds 2000 characters")
>>> is_valid, errors = validator.validate("Hello {name}, please {task}.")
>>> print(is_valid)
True

Notes
-----
- The module delegates token estimation to `insideLLMs.tokens.estimate_tokens` for
  consistency across the codebase.
- Jinja2 is an optional dependency; install with `pip install jinja2` for advanced
  template features.
- Prompt hashes use SHA-256 truncated to 12 characters for quick identification.
- The default prompt library is a module-level singleton accessed via
  `get_default_library()` and `set_default_library()`.

See Also
--------
insideLLMs.tokens : Token counting and analysis utilities
insideLLMs.models : Model interaction utilities that consume prompts

References
----------
.. [1] OpenAI Chat Completions API: https://platform.openai.com/docs/guides/chat
.. [2] Anthropic Messages API: https://docs.anthropic.com/claude/reference/messages
.. [3] Prompt Engineering Guide: https://www.promptingguide.ai/
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
    Union,
)

from insideLLMs.tokens import estimate_tokens as _canonical_estimate_tokens

# Type variable for generic prompt content
T = TypeVar("T")


class PromptRole(Enum):
    """Standard roles for chat-style prompts in LLM conversations.

    This enum defines the standard message roles used in chat-completion APIs
    such as OpenAI's Chat API and Anthropic's Messages API. Each role represents
    a different participant or component in a conversation.

    Attributes
    ----------
    SYSTEM : str
        The system role, used for setting context, instructions, and behavior
        guidelines for the assistant. System messages typically appear first
        and are not shown to end users.
    USER : str
        The user role, representing messages from the human user interacting
        with the assistant.
    ASSISTANT : str
        The assistant role, representing responses generated by the LLM.
    TOOL : str
        The tool role, used for tool/function call results in newer API formats.
        Contains the output from external tool executions.
    FUNCTION : str
        The function role, used for function call results in older API formats.
        Deprecated in favor of TOOL in most modern APIs.

    Examples
    --------
    Using PromptRole with PromptMessage:

    >>> from insideLLMs.prompts.prompt_utils import PromptRole, PromptMessage
    >>> system_msg = PromptMessage(role=PromptRole.SYSTEM, content="You are helpful.")
    >>> user_msg = PromptMessage(role=PromptRole.USER, content="Hello!")
    >>> print(system_msg.role_str)
    system
    >>> print(user_msg.role_str)
    user

    Checking role values:

    >>> PromptRole.SYSTEM.value
    'system'
    >>> PromptRole.USER.value
    'user'
    >>> PromptRole.ASSISTANT.value
    'assistant'

    Iterating over all roles:

    >>> for role in PromptRole:
    ...     print(f"{role.name}: {role.value}")
    SYSTEM: system
    USER: user
    ASSISTANT: assistant
    TOOL: tool
    FUNCTION: function

    See Also
    --------
    PromptMessage : Dataclass that uses PromptRole
    PromptBuilder : Fluent API with role-specific methods
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


@dataclass
class PromptMessage:
    """A single message in a conversation-style prompt.

    This dataclass represents one message in a multi-turn conversation with an LLM.
    It encapsulates the role (who is speaking), the content (what they said), and
    optional metadata for tracking and debugging purposes.

    Parameters
    ----------
    role : Union[PromptRole, str]
        The role of the message sender. Can be a PromptRole enum value or a string.
        Common values: "system", "user", "assistant", "tool", "function".
    content : str
        The text content of the message.
    name : Optional[str], default=None
        Optional name identifier for the sender. Useful when multiple users or
        tools are involved in a conversation.
    metadata : dict[str, Any], default={}
        Optional metadata dictionary for storing additional information such as
        timestamps, token counts, or custom tracking data.

    Attributes
    ----------
    role : Union[PromptRole, str]
        The message sender's role.
    content : str
        The message text content.
    name : Optional[str]
        The sender's name, if provided.
    metadata : dict[str, Any]
        Additional metadata associated with the message.

    Examples
    --------
    Creating a simple user message:

    >>> from insideLLMs.prompts.prompt_utils import PromptMessage, PromptRole
    >>> msg = PromptMessage(role=PromptRole.USER, content="What is Python?")
    >>> print(msg.content)
    What is Python?
    >>> print(msg.role_str)
    user

    Creating a system message with string role:

    >>> msg = PromptMessage(role="system", content="You are a helpful assistant.")
    >>> print(msg.to_dict())
    {'role': 'system', 'content': 'You are a helpful assistant.'}

    Creating a message with name and metadata:

    >>> msg = PromptMessage(
    ...     role=PromptRole.USER,
    ...     content="Hello!",
    ...     name="alice",
    ...     metadata={"timestamp": "2024-01-15T10:30:00Z", "token_count": 2}
    ... )
    >>> print(msg.to_dict())
    {'role': 'user', 'content': 'Hello!', 'name': 'alice'}
    >>> print(msg.metadata["timestamp"])
    2024-01-15T10:30:00Z

    Using with tool/function responses:

    >>> tool_result = PromptMessage(
    ...     role=PromptRole.TOOL,
    ...     content='{"temperature": 72, "unit": "fahrenheit"}',
    ...     name="get_weather",
    ...     metadata={"tool_call_id": "call_abc123"}
    ... )
    >>> print(tool_result.role_str)
    tool

    See Also
    --------
    PromptRole : Enum of standard message roles
    PromptBuilder : Fluent API for building message lists
    """

    role: Union[PromptRole, str]
    content: str
    name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def role_str(self) -> str:
        """Get the role as a string value.

        Converts PromptRole enum values to their string representation, or
        returns the role directly if it's already a string.

        Returns
        -------
        str
            The role as a lowercase string (e.g., "system", "user", "assistant").

        Examples
        --------
        >>> msg = PromptMessage(role=PromptRole.SYSTEM, content="Be helpful.")
        >>> msg.role_str
        'system'

        >>> msg = PromptMessage(role="custom_role", content="Hello")
        >>> msg.role_str
        'custom_role'
        """
        return self.role.value if isinstance(self.role, PromptRole) else self.role

    def to_dict(self) -> dict[str, Any]:
        """Convert the message to a dictionary format suitable for API calls.

        Creates a dictionary with 'role' and 'content' keys, and optionally
        includes 'name' if it was provided. This format is compatible with
        most LLM chat completion APIs (OpenAI, Anthropic, etc.).

        Returns
        -------
        dict[str, Any]
            A dictionary with keys:
            - "role": The role string
            - "content": The message content
            - "name": (optional) The sender name, if provided

        Notes
        -----
        The metadata field is intentionally not included in the output as it
        is meant for internal tracking and is not part of the API payload.

        Examples
        --------
        Basic conversion:

        >>> msg = PromptMessage(role=PromptRole.USER, content="Hello!")
        >>> msg.to_dict()
        {'role': 'user', 'content': 'Hello!'}

        Conversion with name:

        >>> msg = PromptMessage(
        ...     role=PromptRole.ASSISTANT,
        ...     content="Hi there!",
        ...     name="claude"
        ... )
        >>> msg.to_dict()
        {'role': 'assistant', 'content': 'Hi there!', 'name': 'claude'}

        Using the result with an API:

        >>> messages = [
        ...     PromptMessage(role=PromptRole.SYSTEM, content="Be concise."),
        ...     PromptMessage(role=PromptRole.USER, content="What is 2+2?")
        ... ]
        >>> api_messages = [m.to_dict() for m in messages]
        >>> # api_messages is now ready for use with chat completion APIs
        """
        result: dict[str, Any] = {
            "role": self.role_str,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class PromptVersion:
    """A versioned prompt with metadata for tracking and management.

    This dataclass represents a single version of a prompt template, including
    metadata for version control, categorization, and content integrity. It is
    primarily used by PromptLibrary for managing multiple versions of prompts.

    Parameters
    ----------
    content : str
        The prompt template content, which may include {placeholder} variables.
    version : str, default="1.0"
        A version identifier string. Typically follows semantic versioning
        (e.g., "1.0", "1.1", "2.0.0-beta").
    created_at : datetime, default=datetime.now()
        Timestamp when this version was created.
    description : Optional[str], default=None
        Human-readable description of this prompt version, including its
        purpose, changes from previous versions, or usage notes.
    tags : list[str], default=[]
        List of tags for categorization and searchability. Useful for
        organizing prompts by domain, task type, or status.
    hash : str, default=""
        SHA-256 hash (first 12 characters) of the content for integrity
        verification. Automatically computed if not provided.

    Attributes
    ----------
    content : str
        The prompt template content.
    version : str
        The version identifier.
    created_at : datetime
        When this version was created.
    description : Optional[str]
        Description of this version.
    tags : list[str]
        Categorization tags.
    hash : str
        Content hash for integrity checking.

    Examples
    --------
    Creating a basic prompt version:

    >>> from insideLLMs.prompts.prompt_utils import PromptVersion
    >>> pv = PromptVersion(
    ...     content="Summarize the following text: {text}",
    ...     version="1.0",
    ...     description="Basic summarization prompt"
    ... )
    >>> print(pv.version)
    1.0
    >>> print(len(pv.hash))
    12

    Creating a versioned prompt with tags:

    >>> pv = PromptVersion(
    ...     content="You are a {expertise} expert. Answer: {question}",
    ...     version="2.0",
    ...     description="Added expertise customization",
    ...     tags=["qa", "customizable", "production"]
    ... )
    >>> "production" in pv.tags
    True

    Verifying content integrity:

    >>> pv1 = PromptVersion(content="Hello {name}!", version="1.0")
    >>> pv2 = PromptVersion(content="Hello {name}!", version="1.1")
    >>> pv1.hash == pv2.hash  # Same content = same hash
    True
    >>> pv3 = PromptVersion(content="Hi {name}!", version="1.0")
    >>> pv1.hash == pv3.hash  # Different content = different hash
    False

    Using with PromptLibrary:

    >>> from insideLLMs.prompts.prompt_utils import PromptLibrary
    >>> library = PromptLibrary()
    >>> pv = library.register(
    ...     name="greeting",
    ...     content="Hello, {name}!",
    ...     version="1.0",
    ...     tags=["simple", "greeting"]
    ... )
    >>> print(type(pv))
    <class 'insideLLMs.prompt_utils.PromptVersion'>

    See Also
    --------
    PromptLibrary : Storage and retrieval of named prompt versions
    PromptTemplate : Simple template for rendering prompts
    """

    content: str
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    hash: str = field(default="")

    def __post_init__(self):
        """Compute content hash if not provided.

        This method is automatically called after dataclass initialization.
        It computes a SHA-256 hash of the content, truncated to 12 characters,
        which can be used for quick content identification and integrity checks.

        Notes
        -----
        The hash is only computed if the hash field is empty. This allows
        for explicit hash values to be provided during deserialization.

        Examples
        --------
        >>> pv = PromptVersion(content="Test prompt")
        >>> len(pv.hash)
        12
        >>> pv.hash.isalnum()
        True
        """
        if not self.hash:
            self.hash = hashlib.sha256(self.content.encode()).hexdigest()[:12]


class PromptTemplate:
    """A simple prompt template using Python's string formatting syntax.

    This class wraps a template string containing {placeholder} style variables
    and provides methods to format it with actual values. It uses Python's
    built-in str.format() method under the hood, supporting all standard
    format specifications.

    Parameters
    ----------
    template : str
        A string containing {placeholder} style variables. Variables can include
        format specifications like {name:>20} for right-aligned padding.

    Attributes
    ----------
    template : str
        The original template string.

    Examples
    --------
    Basic variable substitution:

    >>> from insideLLMs.prompts.prompt_utils import PromptTemplate
    >>> template = PromptTemplate("Hello, {name}! You are a {role}.")
    >>> result = template.format(name="Alice", role="developer")
    >>> print(result)
    Hello, Alice! You are a developer.

    Building a translation prompt:

    >>> template = PromptTemplate(
    ...     "Translate the following {source_lang} text to {target_lang}:\\n\\n{text}"
    ... )
    >>> prompt = template.format(
    ...     source_lang="English",
    ...     target_lang="Spanish",
    ...     text="The weather is beautiful today."
    ... )
    >>> print(prompt)
    Translate the following English text to Spanish:
    <BLANKLINE>
    The weather is beautiful today.

    Using format specifications:

    >>> template = PromptTemplate("Score: {score:>5.2f} | Status: {status:<10}")
    >>> template.format(score=95.5, status="PASS")
    'Score: 95.50 | Status: PASS      '

    Creating a code review prompt:

    >>> template = PromptTemplate('''
    ... Review the following {language} code for:
    ... 1. Bugs and errors
    ... 2. Security vulnerabilities
    ... 3. Performance issues
    ...
    ... Code:
    ... ```{language}
    ... {code}
    ... ```
    ... ''')
    >>> prompt = template.format(
    ...     language="python",
    ...     code="def greet(name):\\n    print('Hello ' + name)"
    ... )

    Reusing templates:

    >>> greeting = PromptTemplate("Dear {title} {name},")
    >>> print(greeting.format(title="Dr.", name="Smith"))
    Dear Dr. Smith,
    >>> print(greeting.format(title="Prof.", name="Jones"))
    Dear Prof. Jones,

    Notes
    -----
    - Use double braces {{ and }} to include literal braces in the output.
    - For more complex templating needs (conditionals, loops), use
      `render_jinja_template()` instead.
    - Variable names must be valid Python identifiers.

    See Also
    --------
    render_prompt : Function-based template rendering
    render_jinja_template : Advanced Jinja2 template rendering
    extract_variables : Extract variable names from templates
    PromptLibrary : Store and manage named templates
    """

    def __init__(self, template: str):
        """Initialize the template with a format string.

        Parameters
        ----------
        template : str
            A string with {placeholder} style variables for substitution.

        Examples
        --------
        >>> t = PromptTemplate("Hello, {name}!")
        >>> t.template
        'Hello, {name}!'

        >>> t = PromptTemplate("User {user_id} requested {action}")
        >>> t.template
        'User {user_id} requested {action}'
        """
        self.template = template

    def format(self, **kwargs: Any) -> str:
        """Format the template by substituting placeholder variables.

        Replaces all {placeholder} patterns in the template with the
        corresponding values from kwargs.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments where keys are variable names and values
            are the substitution values. All placeholders in the template
            must have corresponding keys.

        Returns
        -------
        str
            The formatted string with all placeholders replaced.

        Raises
        ------
        KeyError
            If a placeholder variable is not provided in kwargs.
        ValueError
            If a format specification is invalid for the provided value.

        Examples
        --------
        Simple substitution:

        >>> template = PromptTemplate("Hello, {name}!")
        >>> template.format(name="World")
        'Hello, World!'

        Multiple variables:

        >>> template = PromptTemplate("{greeting}, {name}! Welcome to {place}.")
        >>> template.format(greeting="Hi", name="Alice", place="Python")
        'Hi, Alice! Welcome to Python.'

        With numeric formatting:

        >>> template = PromptTemplate("Temperature: {temp:.1f}C")
        >>> template.format(temp=23.456)
        'Temperature: 23.5C'

        Building an API request prompt:

        >>> template = PromptTemplate(
        ...     "Using the {api_name} API, {action} the resource at endpoint {endpoint}."
        ... )
        >>> template.format(api_name="REST", action="GET", endpoint="/users/123")
        'Using the REST API, GET the resource at endpoint /users/123.'
        """
        return self.template.format(**kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the template.

        Returns
        -------
        str
            A repr string showing the first 50 characters of the template.

        Examples
        --------
        >>> t = PromptTemplate("Short template")
        >>> repr(t)
        "PromptTemplate('Short template'...)"

        >>> t = PromptTemplate("A" * 100)
        >>> "..." in repr(t)
        True
        """
        return f"PromptTemplate({self.template[:50]!r}...)"


def chain_prompts(prompts: list[str], separator: str = "\n") -> str:
    """Chain multiple prompt strings together into a single combined prompt.

    This utility function concatenates a list of prompt strings using a
    specified separator. It's useful for combining system instructions,
    context, and user queries into a single prompt string.

    Parameters
    ----------
    prompts : list[str]
        List of prompt strings to chain together. Empty strings are
        included in the output.
    separator : str, default="\\n"
        The string to insert between each prompt. Common choices include
        "\\n" (newline), "\\n\\n" (paragraph break), or " " (space).

    Returns
    -------
    str
        A single string with all prompts joined by the separator.

    Examples
    --------
    Basic chaining with default newline separator:

    >>> prompts = ["System: You are helpful.", "User: Hello!"]
    >>> chain_prompts(prompts)
    'System: You are helpful.\\nUser: Hello!'

    Using a double newline for paragraph separation:

    >>> sections = [
    ...     "You are an expert Python developer.",
    ...     "Review the following code for bugs.",
    ...     "Code: def add(a, b): return a + b"
    ... ]
    >>> result = chain_prompts(sections, separator="\\n\\n")
    >>> print(result)
    You are an expert Python developer.
    <BLANKLINE>
    Review the following code for bugs.
    <BLANKLINE>
    Code: def add(a, b): return a + b

    Building a multi-part instruction:

    >>> parts = [
    ...     "Step 1: Analyze the problem",
    ...     "Step 2: Break it down into subtasks",
    ...     "Step 3: Solve each subtask",
    ...     "Step 4: Combine the solutions"
    ... ]
    >>> chain_prompts(parts, separator=" -> ")
    'Step 1: Analyze the problem -> Step 2: Break it down into subtasks -> Step 3: Solve each subtask -> Step 4: Combine the solutions'

    See Also
    --------
    PromptComposer : For named, reusable prompt sections
    PromptBuilder : For building chat-style message lists
    """
    return separator.join(prompts)


def render_prompt(template: str, variables: dict[str, Any]) -> str:
    """Render a template string by substituting variables.

    A convenience function that applies Python's str.format() to a template
    string using a dictionary of variables. This is equivalent to calling
    template.format(**variables) but provides a more explicit function-based
    interface.

    Parameters
    ----------
    template : str
        A string containing {placeholder} style variables to be replaced.
    variables : dict[str, Any]
        Dictionary mapping variable names to their substitution values.
        Keys must match placeholder names in the template.

    Returns
    -------
    str
        The rendered string with all placeholders replaced by their values.

    Raises
    ------
    KeyError
        If a placeholder in the template has no corresponding key in variables.

    Examples
    --------
    Basic rendering:

    >>> render_prompt("Hello {name}!", {"name": "World"})
    'Hello World!'

    Multiple variables:

    >>> template = "The {animal} jumped over the {obstacle}."
    >>> variables = {"animal": "fox", "obstacle": "fence"}
    >>> render_prompt(template, variables)
    'The fox jumped over the fence.'

    Building a structured prompt:

    >>> template = '''
    ... Role: {role}
    ... Task: {task}
    ... Context: {context}
    ... '''
    >>> variables = {
    ...     "role": "Data Analyst",
    ...     "task": "Analyze sales trends",
    ...     "context": "Q4 2024 data"
    ... }
    >>> result = render_prompt(template, variables)

    Using with dynamic variable dictionaries:

    >>> base_vars = {"model": "GPT-4", "temperature": 0.7}
    >>> user_vars = {"query": "Explain quantum computing"}
    >>> all_vars = {**base_vars, **user_vars}
    >>> render_prompt("Using {model} at temp {temperature}: {query}", all_vars)
    'Using GPT-4 at temp 0.7: Explain quantum computing'

    See Also
    --------
    PromptTemplate : Object-oriented template wrapper
    render_jinja_template : Advanced templating with Jinja2
    validate_template_variables : Check for missing variables before rendering
    """
    return template.format(**variables)


def render_jinja_template(template_str: str, variables: dict[str, Any]) -> str:
    """Render a Jinja2 template string.

    Requires jinja2 to be installed. Supports advanced features like
    loops, conditionals, and filters.

    Args:
        template_str: A Jinja2 template string.
        variables: Dictionary of variables for the template.

    Returns:
        The rendered string.

    Raises:
        ImportError: If jinja2 is not installed.

    Example:
        >>> render_jinja_template("Hello {{ name }}!", {"name": "World"})
        "Hello World!"
    """
    try:
        from jinja2 import Template
    except ImportError:
        raise ImportError(
            "jinja2 is required for Jinja template rendering. Install it with: pip install jinja2"
        )

    template = Template(template_str)
    return template.render(**variables)


def format_chat_messages(
    messages: list[dict[str, str]],
    system_prefix: str = "System: ",
    user_prefix: str = "User: ",
    assistant_prefix: str = "Assistant: ",
) -> str:
    """Format chat messages into a single prompt string.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        system_prefix: Prefix for system messages.
        user_prefix: Prefix for user messages.
        assistant_prefix: Prefix for assistant messages.

    Returns:
        A formatted string with all messages.

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> format_chat_messages(messages)
        "System: You are helpful.\\nUser: Hello!"
    """
    prefixes = {
        "system": system_prefix,
        "user": user_prefix,
        "assistant": assistant_prefix,
    }

    formatted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prefix = prefixes.get(role, "")
        formatted.append(f"{prefix}{content}")

    return "\n".join(formatted)


def create_few_shot_prompt(
    instruction: str,
    examples: list[dict[str, str]],
    query: str,
    input_key: str = "input",
    output_key: str = "output",
) -> str:
    """Create a few-shot learning prompt with examples.

    Args:
        instruction: The task instruction.
        examples: List of example dicts with input/output pairs.
        query: The actual query to answer.
        input_key: Key for input in example dicts.
        output_key: Key for output in example dicts.

    Returns:
        A formatted few-shot prompt.

    Example:
        >>> examples = [
        ...     {"input": "2+2", "output": "4"},
        ...     {"input": "3+3", "output": "6"}
        ... ]
        >>> create_few_shot_prompt("Calculate:", examples, "4+4")
    """
    parts = [instruction, ""]

    for i, example in enumerate(examples, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Input: {example[input_key]}")
        parts.append(f"Output: {example[output_key]}")
        parts.append("")

    parts.append("Now solve:")
    parts.append(f"Input: {query}")
    parts.append("Output:")

    return "\n".join(parts)


class PromptBuilder:
    """Fluent API for building complex prompts.

    Example:
        >>> prompt = (PromptBuilder()
        ...     .system("You are a helpful assistant.")
        ...     .user("Hello!")
        ...     .assistant("Hi there!")
        ...     .user("What's 2+2?")
        ...     .build())
    """

    def __init__(self):
        """Initialize an empty prompt builder."""
        self._messages: list[PromptMessage] = []
        self._context: dict[str, Any] = {}

    def system(self, content: str, **kwargs: Any) -> "PromptBuilder":
        """Add a system message."""
        self._messages.append(PromptMessage(role=PromptRole.SYSTEM, content=content, **kwargs))
        return self

    def user(self, content: str, **kwargs: Any) -> "PromptBuilder":
        """Add a user message."""
        self._messages.append(PromptMessage(role=PromptRole.USER, content=content, **kwargs))
        return self

    def assistant(self, content: str, **kwargs: Any) -> "PromptBuilder":
        """Add an assistant message."""
        self._messages.append(PromptMessage(role=PromptRole.ASSISTANT, content=content, **kwargs))
        return self

    def message(self, role: Union[PromptRole, str], content: str, **kwargs: Any) -> "PromptBuilder":
        """Add a message with any role."""
        self._messages.append(PromptMessage(role=role, content=content, **kwargs))
        return self

    def context(self, **kwargs: Any) -> "PromptBuilder":
        """Add context variables for template rendering."""
        self._context.update(kwargs)
        return self

    def render(self) -> "PromptBuilder":
        """Render all message templates with context variables."""
        for msg in self._messages:
            if self._context:
                msg.content = msg.content.format(**self._context)
        return self

    def build(self) -> list[dict[str, Any]]:
        """Build the final list of message dicts."""
        return [msg.to_dict() for msg in self._messages]

    def build_messages(self) -> list[PromptMessage]:
        """Build the final list of PromptMessage objects."""
        return list(self._messages)

    def to_string(
        self,
        system_prefix: str = "System: ",
        user_prefix: str = "User: ",
        assistant_prefix: str = "Assistant: ",
        separator: str = "\n",
    ) -> str:
        """Convert to a single string prompt."""
        prefixes = {
            PromptRole.SYSTEM: system_prefix,
            PromptRole.USER: user_prefix,
            PromptRole.ASSISTANT: assistant_prefix,
            "system": system_prefix,
            "user": user_prefix,
            "assistant": assistant_prefix,
        }
        parts = []
        for msg in self._messages:
            prefix = prefixes.get(msg.role, "")
            parts.append(f"{prefix}{msg.content}")
        return separator.join(parts)

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)


class PromptLibrary:
    """A library for storing and managing named prompt templates.

    Supports versioning, tagging, and retrieval of prompts by name.

    Example:
        >>> library = PromptLibrary()
        >>> library.register("greeting", "Hello, {name}!")
        >>> library.get("greeting").format(name="World")
        "Hello, World!"
    """

    def __init__(self):
        """Initialize an empty prompt library."""
        self._prompts: dict[str, list[PromptVersion]] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        content: str,
        version: str = "1.0",
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> PromptVersion:
        """Register a new prompt template.

        Args:
            name: Unique name for the prompt.
            content: The prompt content/template.
            version: Version identifier.
            description: Optional description.
            tags: Optional tags for categorization.

        Returns:
            The created PromptVersion.
        """
        prompt_version = PromptVersion(
            content=content,
            version=version,
            description=description,
            tags=tags or [],
        )

        if name not in self._prompts:
            self._prompts[name] = []
        self._prompts[name].append(prompt_version)

        return prompt_version

    def get(self, name: str, version: Optional[str] = None) -> PromptTemplate:
        """Get a prompt template by name.

        Args:
            name: The prompt name (or alias).
            version: Optional specific version (default: latest).

        Returns:
            A PromptTemplate for the prompt.

        Raises:
            KeyError: If prompt not found.
        """
        # Resolve alias
        resolved_name = self._aliases.get(name, name)

        if resolved_name not in self._prompts:
            raise KeyError(f"Prompt not found: {name}")

        versions = self._prompts[resolved_name]

        if version:
            for v in versions:
                if v.version == version:
                    return PromptTemplate(v.content)
            raise KeyError(f"Version {version} not found for prompt: {name}")

        # Return latest version
        return PromptTemplate(versions[-1].content)

    def get_version(self, name: str, version: Optional[str] = None) -> PromptVersion:
        """Get a PromptVersion object by name.

        Args:
            name: The prompt name.
            version: Optional specific version.

        Returns:
            The PromptVersion object.
        """
        resolved_name = self._aliases.get(name, name)

        if resolved_name not in self._prompts:
            raise KeyError(f"Prompt not found: {name}")

        versions = self._prompts[resolved_name]

        if version:
            for v in versions:
                if v.version == version:
                    return v
            raise KeyError(f"Version {version} not found for prompt: {name}")

        return versions[-1]

    def alias(self, alias_name: str, prompt_name: str) -> None:
        """Create an alias for a prompt.

        Args:
            alias_name: The alias to create.
            prompt_name: The prompt to alias.
        """
        if prompt_name not in self._prompts:
            raise KeyError(f"Prompt not found: {prompt_name}")
        self._aliases[alias_name] = prompt_name

    def list_prompts(self) -> list[str]:
        """List all prompt names."""
        return list(self._prompts.keys())

    def list_versions(self, name: str) -> list[str]:
        """List all versions for a prompt.

        Args:
            name: The prompt name.

        Returns:
            List of version strings.
        """
        if name not in self._prompts:
            raise KeyError(f"Prompt not found: {name}")
        return [v.version for v in self._prompts[name]]

    def search_by_tag(self, tag: str) -> list[str]:
        """Find prompts by tag.

        Args:
            tag: The tag to search for.

        Returns:
            List of prompt names with the tag.
        """
        results = []
        for name, versions in self._prompts.items():
            for v in versions:
                if tag in v.tags:
                    results.append(name)
                    break
        return results

    def export(self) -> dict[str, Any]:
        """Export library to a dictionary."""
        return {
            "prompts": {
                name: [
                    {
                        "content": v.content,
                        "version": v.version,
                        "description": v.description,
                        "tags": v.tags,
                        "hash": v.hash,
                    }
                    for v in versions
                ]
                for name, versions in self._prompts.items()
            },
            "aliases": dict(self._aliases),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptLibrary":
        """Create library from exported dictionary."""
        library = cls()
        for name, versions in data.get("prompts", {}).items():
            for v in versions:
                library.register(
                    name=name,
                    content=v["content"],
                    version=v.get("version", "1.0"),
                    description=v.get("description"),
                    tags=v.get("tags", []),
                )
        for alias, prompt_name in data.get("aliases", {}).items():
            library.alias(alias, prompt_name)
        return library

    def __contains__(self, name: str) -> bool:
        resolved = self._aliases.get(name, name)
        return resolved in self._prompts

    def __len__(self) -> int:
        return len(self._prompts)


class PromptComposer:
    """Compose prompts from reusable components.

    Allows defining prompt sections that can be mixed and matched.

    Example:
        >>> composer = PromptComposer()
        >>> composer.add_section("intro", "You are a helpful assistant.")
        >>> composer.add_section("rules", "Always be polite.")
        >>> composer.compose(["intro", "rules"])
        "You are a helpful assistant.\\n\\nAlways be polite."
    """

    def __init__(self, separator: str = "\n\n"):
        """Initialize the composer.

        Args:
            separator: Default separator between sections.
        """
        self._sections: dict[str, str] = {}
        self._separator = separator

    def add_section(
        self, name: str, content: str, variables: Optional[dict[str, Any]] = None
    ) -> "PromptComposer":
        """Add a named section.

        Args:
            name: Section name.
            content: Section content (can include {placeholders}).
            variables: Optional variables to render immediately.

        Returns:
            Self for chaining.
        """
        if variables:
            content = content.format(**variables)
        self._sections[name] = content
        return self

    def get_section(self, name: str) -> str:
        """Get a section by name.

        Args:
            name: Section name.

        Returns:
            The section content.

        Raises:
            KeyError: If section not found.
        """
        if name not in self._sections:
            raise KeyError(f"Section not found: {name}")
        return self._sections[name]

    def compose(
        self,
        sections: list[str],
        separator: Optional[str] = None,
        variables: Optional[dict[str, Any]] = None,
    ) -> str:
        """Compose a prompt from named sections.

        Args:
            sections: List of section names to include in order.
            separator: Custom separator (default: instance separator).
            variables: Variables to render in all sections.

        Returns:
            The composed prompt string.
        """
        sep = separator if separator is not None else self._separator
        parts = []

        for name in sections:
            content = self.get_section(name)
            if variables:
                content = content.format(**variables)
            parts.append(content)

        return sep.join(parts)

    def list_sections(self) -> list[str]:
        """List all section names."""
        return list(self._sections.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._sections


class PromptValidator:
    """Validate prompt templates and content.

    Example:
        >>> validator = PromptValidator()
        >>> validator.add_rule("max_length", lambda p: len(p) <= 1000)
        >>> validator.validate("Hello, {name}!")
        (True, [])
    """

    def __init__(self):
        """Initialize with default rules."""
        self._rules: dict[str, Callable[[str], bool]] = {}
        self._required_vars: set[str] = set()

    def add_rule(
        self, name: str, rule: Callable[[str], bool], error_msg: Optional[str] = None
    ) -> "PromptValidator":
        """Add a validation rule.

        Args:
            name: Rule name.
            rule: Function that takes prompt string, returns True if valid.
            error_msg: Optional custom error message.

        Returns:
            Self for chaining.
        """
        self._rules[name] = (rule, error_msg)  # type: ignore[assignment]
        return self

    def require_variables(self, *variables: str) -> "PromptValidator":
        """Require certain placeholder variables to be present.

        Args:
            *variables: Variable names that must exist.

        Returns:
            Self for chaining.
        """
        self._required_vars.update(variables)
        return self

    def validate(self, prompt: str) -> tuple[bool, list[str]]:
        """Validate a prompt against all rules.

        Args:
            prompt: The prompt string to validate.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        # Check required variables
        for var in self._required_vars:
            if f"{{{var}}}" not in prompt:
                errors.append(f"Missing required variable: {{{var}}}")

        # Run custom rules
        for name, (rule, error_msg) in self._rules.items():
            try:
                if not rule(prompt):
                    errors.append(error_msg or f"Failed rule: {name}")
            except Exception as e:
                errors.append(f"Rule {name} error: {e}")

        return len(errors) == 0, errors


def extract_variables(template: str) -> list[str]:
    """Extract placeholder variable names from a template.

    Args:
        template: A template string with {placeholder} syntax.

    Returns:
        List of variable names found.

    Example:
        >>> extract_variables("Hello, {name}! Your {item} is ready.")
        ["name", "item"]
    """
    # Match {variable} but not {{ or }}
    pattern = r"\{([^{}]+)\}"
    matches = re.findall(pattern, template)
    # Filter out format specs like {name:>10}
    return [m.split(":")[0].split("!")[0] for m in matches]


def validate_template_variables(template: str, variables: dict[str, Any]) -> tuple[bool, list[str]]:
    """Check if all template variables are provided.

    Args:
        template: The template string.
        variables: Dictionary of provided variables.

    Returns:
        Tuple of (all_provided, list of missing variables).
    """
    required = set(extract_variables(template))
    provided = set(variables.keys())
    missing = required - provided
    return len(missing) == 0, list(missing)


def escape_template(text: str) -> str:
    """Escape curly braces in text to prevent template substitution.

    Args:
        text: Text to escape.

    Returns:
        Escaped text with {{ and }} instead of { and }.

    Example:
        >>> escape_template("Use {name} in JSON: {}")
        "Use {name} in JSON: {{}}"
    """
    # Only escape standalone braces, not {variable} patterns
    result = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            # Check if this is a variable pattern
            end = text.find("}", i)
            if end == -1 or not text[i + 1 : end].replace("_", "").isalnum():
                result.append("{{")
            else:
                result.append("{")
        elif text[i] == "}":
            # Check if previous was a variable end
            if result and result[-1] != "}":
                start = len(result) - 1
                while start >= 0 and result[start] != "{":
                    start -= 1
                if start < 0:
                    result.append("}}")
                else:
                    result.append("}")
            else:
                result.append("}}")
        else:
            result.append(text[i])
        i += 1
    return "".join(result)


def truncate_prompt(
    prompt: str, max_length: int, suffix: str = "...", preserve_end: bool = False
) -> str:
    """Truncate a prompt to a maximum length.

    Args:
        prompt: The prompt to truncate.
        max_length: Maximum length in characters.
        suffix: String to append when truncated.
        preserve_end: If True, preserve the end instead of the beginning.

    Returns:
        Truncated prompt.
    """
    if len(prompt) <= max_length:
        return prompt

    if preserve_end:
        cut_length = max_length - len(suffix)
        return suffix + prompt[-cut_length:]
    else:
        cut_length = max_length - len(suffix)
        return prompt[:cut_length] + suffix


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate the number of tokens in text.

    This is a rough estimate based on average characters per token.
    For accurate counts, use a proper tokenizer.

    Args:
        text: The text to estimate.
        chars_per_token: Average characters per token (default: 4.0).

    Returns:
        Estimated token count.

    Note:
        The chars_per_token parameter is kept for backward compatibility
        but is ignored. Uses the canonical estimate_tokens from tokens.py.
    """
    _ = chars_per_token  # kept for backward compatibility
    return _canonical_estimate_tokens(text)


def split_prompt_by_tokens(
    prompt: str,
    max_tokens: int,
    chars_per_token: float = 4.0,
    overlap: int = 0,
) -> list[str]:
    """Split a prompt into chunks by estimated token count.

    Args:
        prompt: The prompt to split.
        max_tokens: Maximum tokens per chunk.
        chars_per_token: Average characters per token.
        overlap: Number of tokens to overlap between chunks.

    Returns:
        List of prompt chunks.
    """
    max_chars = int(max_tokens * chars_per_token)
    overlap_chars = int(overlap * chars_per_token)

    if len(prompt) <= max_chars:
        return [prompt]

    chunks = []
    start = 0
    while start < len(prompt):
        end = min(start + max_chars, len(prompt))

        # Try to break at a sentence or word boundary
        if end < len(prompt):
            # Look for sentence end
            for delim in [". ", "! ", "? ", "\n\n", "\n", " "]:
                pos = prompt.rfind(delim, start, end)
                if pos > start:
                    end = pos + len(delim)
                    break

        chunks.append(prompt[start:end])
        start = end - overlap_chars if overlap_chars > 0 else end

    return chunks


# Default global library instance
_default_library: Optional[PromptLibrary] = None


def get_default_library() -> PromptLibrary:
    """Get or create the default prompt library."""
    global _default_library
    if _default_library is None:
        _default_library = PromptLibrary()
    return _default_library


def set_default_library(library: PromptLibrary) -> None:
    """Set the default prompt library."""
    global _default_library
    _default_library = library


def register_prompt(
    name: str,
    content: str,
    version: str = "1.0",
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> PromptVersion:
    """Register a prompt in the default library.

    Convenience function for get_default_library().register().
    """
    return get_default_library().register(
        name=name,
        content=content,
        version=version,
        description=description,
        tags=tags,
    )


def get_prompt(name: str, version: Optional[str] = None) -> PromptTemplate:
    """Get a prompt from the default library.

    Convenience function for get_default_library().get().
    """
    return get_default_library().get(name, version)
