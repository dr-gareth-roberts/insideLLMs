"""Advanced prompt templating utilities.

This module provides tools for creating, formatting, and chaining prompts
for use with LLM models.

Key features:
- PromptTemplate: Simple {placeholder} style templates
- PromptBuilder: Fluent API for building complex prompts
- PromptLibrary: Named template storage and versioning
- PromptComposer: Compose prompts from reusable components
- Validation and parsing utilities
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)


# Type variable for generic prompt content
T = TypeVar("T")


class PromptRole(Enum):
    """Standard roles for chat-style prompts."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


@dataclass
class PromptMessage:
    """A single message in a conversation-style prompt.

    Attributes:
        role: The role of the message sender.
        content: The message content.
        name: Optional name for the sender.
        metadata: Optional metadata dict.
    """

    role: Union[PromptRole, str]
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def role_str(self) -> str:
        """Get role as string."""
        return self.role.value if isinstance(self.role, PromptRole) else self.role

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API calls."""
        result: Dict[str, Any] = {
            "role": self.role_str,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class PromptVersion:
    """A versioned prompt with metadata.

    Attributes:
        content: The prompt content.
        version: Version identifier.
        created_at: Creation timestamp.
        description: Optional description.
        tags: Optional tags for categorization.
        hash: Content hash for integrity.
    """

    content: str
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    hash: str = field(default="")

    def __post_init__(self):
        """Compute hash if not provided."""
        if not self.hash:
            self.hash = hashlib.sha256(self.content.encode()).hexdigest()[:12]


class PromptTemplate:
    """A simple prompt template using Python string formatting.

    Example:
        >>> template = PromptTemplate("Hello, {name}! You are a {role}.")
        >>> template.format(name="Alice", role="developer")
        "Hello, Alice! You are a developer."
    """

    def __init__(self, template: str):
        """Initialize the template.

        Args:
            template: A string with {placeholder} style variables.
        """
        self.template = template

    def format(self, **kwargs: Any) -> str:
        """Format the template with the given variables.

        Args:
            **kwargs: Variables to substitute into the template.

        Returns:
            The formatted string.
        """
        return self.template.format(**kwargs)

    def __repr__(self) -> str:
        return f"PromptTemplate({self.template[:50]!r}...)"


def chain_prompts(prompts: List[str], separator: str = "\n") -> str:
    """Chain multiple prompts together into a single prompt.

    Args:
        prompts: List of prompt strings to chain.
        separator: String to use between prompts (default: newline).

    Returns:
        A single string with all prompts joined.

    Example:
        >>> chain_prompts(["System: You are helpful.", "User: Hello!"])
        "System: You are helpful.\\nUser: Hello!"
    """
    return separator.join(prompts)


def render_prompt(template: str, variables: Dict[str, Any]) -> str:
    """Render a template string with variables.

    Uses Python's str.format() for simple templates. For more advanced
    templating, use render_jinja_template if jinja2 is installed.

    Args:
        template: A string with {placeholder} style variables.
        variables: Dictionary of variables to substitute.

    Returns:
        The rendered string.

    Example:
        >>> render_prompt("Hello {name}!", {"name": "World"})
        "Hello World!"
    """
    return template.format(**variables)


def render_jinja_template(template_str: str, variables: Dict[str, Any]) -> str:
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
            "jinja2 is required for Jinja template rendering. "
            "Install it with: pip install jinja2"
        )

    template = Template(template_str)
    return template.render(**variables)


def format_chat_messages(
    messages: List[Dict[str, str]],
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
    examples: List[Dict[str, str]],
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
        self._messages: List[PromptMessage] = []
        self._context: Dict[str, Any] = {}

    def system(self, content: str, **kwargs: Any) -> "PromptBuilder":
        """Add a system message."""
        self._messages.append(
            PromptMessage(role=PromptRole.SYSTEM, content=content, **kwargs)
        )
        return self

    def user(self, content: str, **kwargs: Any) -> "PromptBuilder":
        """Add a user message."""
        self._messages.append(
            PromptMessage(role=PromptRole.USER, content=content, **kwargs)
        )
        return self

    def assistant(self, content: str, **kwargs: Any) -> "PromptBuilder":
        """Add an assistant message."""
        self._messages.append(
            PromptMessage(role=PromptRole.ASSISTANT, content=content, **kwargs)
        )
        return self

    def message(
        self, role: Union[PromptRole, str], content: str, **kwargs: Any
    ) -> "PromptBuilder":
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

    def build(self) -> List[Dict[str, Any]]:
        """Build the final list of message dicts."""
        return [msg.to_dict() for msg in self._messages]

    def build_messages(self) -> List[PromptMessage]:
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
        self._prompts: Dict[str, List[PromptVersion]] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self,
        name: str,
        content: str,
        version: str = "1.0",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
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

    def get(
        self, name: str, version: Optional[str] = None
    ) -> PromptTemplate:
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

    def get_version(
        self, name: str, version: Optional[str] = None
    ) -> PromptVersion:
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

    def list_prompts(self) -> List[str]:
        """List all prompt names."""
        return list(self._prompts.keys())

    def list_versions(self, name: str) -> List[str]:
        """List all versions for a prompt.

        Args:
            name: The prompt name.

        Returns:
            List of version strings.
        """
        if name not in self._prompts:
            raise KeyError(f"Prompt not found: {name}")
        return [v.version for v in self._prompts[name]]

    def search_by_tag(self, tag: str) -> List[str]:
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

    def export(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "PromptLibrary":
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
        self._sections: Dict[str, str] = {}
        self._separator = separator

    def add_section(
        self, name: str, content: str, variables: Optional[Dict[str, Any]] = None
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
        sections: List[str],
        separator: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
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

    def list_sections(self) -> List[str]:
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
        self._rules: Dict[str, Callable[[str], bool]] = {}
        self._required_vars: Set[str] = set()

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
        self._rules[name] = (rule, error_msg)
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

    def validate(self, prompt: str) -> Tuple[bool, List[str]]:
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


def extract_variables(template: str) -> List[str]:
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


def validate_template_variables(
    template: str, variables: Dict[str, Any]
) -> Tuple[bool, List[str]]:
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
    """
    return int(len(text) / chars_per_token)


def split_prompt_by_tokens(
    prompt: str,
    max_tokens: int,
    chars_per_token: float = 4.0,
    overlap: int = 0,
) -> List[str]:
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
    tags: Optional[List[str]] = None,
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
