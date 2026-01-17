"""Tests for prompt utilities."""

import pytest

from insideLLMs.prompt_utils import (
    PromptBuilder,
    PromptComposer,
    PromptLibrary,
    PromptMessage,
    PromptRole,
    PromptTemplate,
    PromptValidator,
    PromptVersion,
    chain_prompts,
    create_few_shot_prompt,
    estimate_tokens,
    extract_variables,
    format_chat_messages,
    get_default_library,
    get_prompt,
    register_prompt,
    render_prompt,
    set_default_library,
    split_prompt_by_tokens,
    truncate_prompt,
    validate_template_variables,
)


class TestPromptTemplate:
    """Tests for PromptTemplate."""

    def test_basic_format(self):
        """Test basic string formatting."""
        template = PromptTemplate("Hello, {name}!")
        result = template.format(name="World")
        assert result == "Hello, World!"

    def test_multiple_variables(self):
        """Test multiple variable substitution."""
        template = PromptTemplate("{greeting}, {name}! Welcome to {place}.")
        result = template.format(greeting="Hi", name="Alice", place="Wonderland")
        assert result == "Hi, Alice! Welcome to Wonderland."

    def test_repr(self):
        """Test string representation."""
        template = PromptTemplate("Hello, {name}!")
        assert "PromptTemplate" in repr(template)


class TestPromptMessage:
    """Tests for PromptMessage."""

    def test_basic_message(self):
        """Test basic message creation."""
        msg = PromptMessage(role=PromptRole.USER, content="Hello!")
        assert msg.role == PromptRole.USER
        assert msg.content == "Hello!"
        assert msg.role_str == "user"

    def test_message_with_name(self):
        """Test message with name."""
        msg = PromptMessage(role="assistant", content="Hi!", name="Claude")
        result = msg.to_dict()
        assert result["role"] == "assistant"
        assert result["content"] == "Hi!"
        assert result["name"] == "Claude"

    def test_role_str_with_enum(self):
        """Test role_str with enum value."""
        msg = PromptMessage(role=PromptRole.SYSTEM, content="Be helpful.")
        assert msg.role_str == "system"

    def test_role_str_with_string(self):
        """Test role_str with string value."""
        msg = PromptMessage(role="custom", content="Custom message")
        assert msg.role_str == "custom"


class TestPromptVersion:
    """Tests for PromptVersion."""

    def test_basic_version(self):
        """Test basic version creation."""
        version = PromptVersion(content="Hello, {name}!", version="1.0")
        assert version.content == "Hello, {name}!"
        assert version.version == "1.0"
        assert version.hash  # Should have computed hash

    def test_hash_consistency(self):
        """Test hash is consistent for same content."""
        v1 = PromptVersion(content="Hello!", version="1.0")
        v2 = PromptVersion(content="Hello!", version="2.0")
        assert v1.hash == v2.hash  # Same content, same hash

    def test_hash_difference(self):
        """Test hash differs for different content."""
        v1 = PromptVersion(content="Hello!", version="1.0")
        v2 = PromptVersion(content="Hi!", version="1.0")
        assert v1.hash != v2.hash


class TestPromptBuilder:
    """Tests for PromptBuilder."""

    def test_basic_builder(self):
        """Test basic message building."""
        builder = PromptBuilder()
        builder.system("You are helpful.")
        builder.user("Hello!")

        messages = builder.build()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_fluent_api(self):
        """Test fluent API chaining."""
        messages = (
            PromptBuilder()
            .system("Be helpful.")
            .user("Hi!")
            .assistant("Hello!")
            .build()
        )
        assert len(messages) == 3

    def test_with_context(self):
        """Test context variable rendering."""
        builder = (
            PromptBuilder()
            .system("You are {role}.")
            .user("Hello, my name is {name}.")
            .context(role="helpful", name="Alice")
            .render()
        )
        messages = builder.build()
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["content"] == "Hello, my name is Alice."

    def test_to_string(self):
        """Test string conversion."""
        result = (
            PromptBuilder()
            .system("Be helpful.")
            .user("Hello!")
            .to_string()
        )
        assert "System: Be helpful." in result
        assert "User: Hello!" in result

    def test_message_with_any_role(self):
        """Test adding message with custom role."""
        builder = PromptBuilder()
        builder.message("custom_role", "Custom content")
        messages = builder.build()
        assert messages[0]["role"] == "custom_role"

    def test_iteration(self):
        """Test iterating over messages."""
        builder = PromptBuilder().system("A").user("B")
        contents = [msg.content for msg in builder]
        assert contents == ["A", "B"]

    def test_len(self):
        """Test length of builder."""
        builder = PromptBuilder().system("A").user("B").assistant("C")
        assert len(builder) == 3


class TestPromptLibrary:
    """Tests for PromptLibrary."""

    def test_register_and_get(self):
        """Test registering and retrieving prompts."""
        library = PromptLibrary()
        library.register("greeting", "Hello, {name}!")

        template = library.get("greeting")
        result = template.format(name="World")
        assert result == "Hello, World!"

    def test_versioning(self):
        """Test multiple versions of a prompt."""
        library = PromptLibrary()
        library.register("greeting", "Hi, {name}!", version="1.0")
        library.register("greeting", "Hello, {name}!", version="2.0")

        # Latest version
        template = library.get("greeting")
        assert "Hello" in template.format(name="World")

        # Specific version
        template_v1 = library.get("greeting", version="1.0")
        assert "Hi" in template_v1.format(name="World")

    def test_aliases(self):
        """Test prompt aliases."""
        library = PromptLibrary()
        library.register("greeting", "Hello, {name}!")
        library.alias("hello", "greeting")

        template = library.get("hello")
        assert template.format(name="World") == "Hello, World!"

    def test_tags(self):
        """Test searching by tags."""
        library = PromptLibrary()
        library.register("greeting", "Hello!", tags=["basic", "friendly"])
        library.register("farewell", "Goodbye!", tags=["basic"])
        library.register("complex", "...", tags=["advanced"])

        basic = library.search_by_tag("basic")
        assert "greeting" in basic
        assert "farewell" in basic
        assert "complex" not in basic

    def test_list_prompts(self):
        """Test listing all prompts."""
        library = PromptLibrary()
        library.register("a", "A")
        library.register("b", "B")

        prompts = library.list_prompts()
        assert set(prompts) == {"a", "b"}

    def test_list_versions(self):
        """Test listing versions."""
        library = PromptLibrary()
        library.register("prompt", "v1", version="1.0")
        library.register("prompt", "v2", version="2.0")

        versions = library.list_versions("prompt")
        assert versions == ["1.0", "2.0"]

    def test_export_import(self):
        """Test exporting and importing library."""
        library = PromptLibrary()
        library.register("greeting", "Hello!", version="1.0", tags=["basic"])
        library.alias("hi", "greeting")

        exported = library.export()
        imported = PromptLibrary.from_dict(exported)

        assert "greeting" in imported
        assert "hi" in imported
        assert imported.get("greeting").format() == "Hello!"

    def test_contains(self):
        """Test contains operator."""
        library = PromptLibrary()
        library.register("exists", "content")

        assert "exists" in library
        assert "missing" not in library

    def test_len(self):
        """Test length."""
        library = PromptLibrary()
        library.register("a", "A")
        library.register("b", "B")

        assert len(library) == 2

    def test_prompt_not_found(self):
        """Test error on missing prompt."""
        library = PromptLibrary()

        with pytest.raises(KeyError):
            library.get("missing")

    def test_version_not_found(self):
        """Test error on missing version."""
        library = PromptLibrary()
        library.register("prompt", "content", version="1.0")

        with pytest.raises(KeyError):
            library.get("prompt", version="999.0")


class TestPromptComposer:
    """Tests for PromptComposer."""

    def test_basic_compose(self):
        """Test basic composition."""
        composer = PromptComposer()
        composer.add_section("intro", "You are helpful.")
        composer.add_section("rules", "Be polite.")

        result = composer.compose(["intro", "rules"])
        assert "You are helpful." in result
        assert "Be polite." in result

    def test_custom_separator(self):
        """Test custom separator."""
        composer = PromptComposer(separator=" | ")
        composer.add_section("a", "A")
        composer.add_section("b", "B")

        result = composer.compose(["a", "b"])
        assert result == "A | B"

    def test_compose_with_variables(self):
        """Test composition with variables."""
        composer = PromptComposer()
        composer.add_section("greeting", "Hello, {name}!")

        result = composer.compose(["greeting"], variables={"name": "World"})
        assert result == "Hello, World!"

    def test_section_with_variables(self):
        """Test adding section with immediate variable rendering."""
        composer = PromptComposer()
        composer.add_section("greeting", "Hello, {name}!", variables={"name": "Alice"})

        result = composer.compose(["greeting"])
        assert result == "Hello, Alice!"

    def test_list_sections(self):
        """Test listing sections."""
        composer = PromptComposer()
        composer.add_section("a", "A")
        composer.add_section("b", "B")

        sections = composer.list_sections()
        assert set(sections) == {"a", "b"}

    def test_contains(self):
        """Test contains operator."""
        composer = PromptComposer()
        composer.add_section("exists", "content")

        assert "exists" in composer
        assert "missing" not in composer

    def test_missing_section(self):
        """Test error on missing section."""
        composer = PromptComposer()

        with pytest.raises(KeyError):
            composer.compose(["missing"])


class TestPromptValidator:
    """Tests for PromptValidator."""

    def test_basic_validation(self):
        """Test basic validation passes."""
        validator = PromptValidator()
        valid, errors = validator.validate("Hello!")

        assert valid
        assert len(errors) == 0

    def test_required_variables(self):
        """Test required variable validation."""
        validator = PromptValidator()
        validator.require_variables("name", "role")

        # Missing variables
        valid, errors = validator.validate("Hello!")
        assert not valid
        assert len(errors) == 2

        # Has variables
        valid, errors = validator.validate("Hello, {name}! You are {role}.")
        assert valid

    def test_custom_rule(self):
        """Test custom validation rule."""
        validator = PromptValidator()
        validator.add_rule(
            "max_length",
            lambda p: len(p) <= 10,
            "Prompt too long"
        )

        valid, errors = validator.validate("Short")
        assert valid

        valid, errors = validator.validate("This is way too long")
        assert not valid
        assert "Prompt too long" in errors[0]

    def test_rule_chaining(self):
        """Test chaining validation rules."""
        validator = (
            PromptValidator()
            .add_rule("not_empty", lambda p: len(p) > 0)
            .require_variables("name")
        )

        valid, errors = validator.validate("")
        assert not valid


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_chain_prompts(self):
        """Test chaining prompts."""
        result = chain_prompts(["A", "B", "C"])
        assert result == "A\nB\nC"

        result = chain_prompts(["A", "B"], separator=" | ")
        assert result == "A | B"

    def test_render_prompt(self):
        """Test rendering prompt."""
        result = render_prompt("Hello, {name}!", {"name": "World"})
        assert result == "Hello, World!"

    def test_format_chat_messages(self):
        """Test formatting chat messages."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        result = format_chat_messages(messages)
        assert "System: Be helpful." in result
        assert "User: Hello!" in result

    def test_create_few_shot_prompt(self):
        """Test few-shot prompt creation."""
        examples = [
            {"input": "2+2", "output": "4"},
            {"input": "3+3", "output": "6"},
        ]
        result = create_few_shot_prompt("Calculate:", examples, "4+4")

        assert "Calculate:" in result
        assert "2+2" in result
        assert "4" in result
        assert "4+4" in result

    def test_extract_variables(self):
        """Test variable extraction."""
        vars = extract_variables("Hello, {name}! Welcome to {place}.")
        assert set(vars) == {"name", "place"}

    def test_extract_variables_with_format_spec(self):
        """Test extraction with format specifiers."""
        vars = extract_variables("Value: {value:>10}")
        assert vars == ["value"]

    def test_validate_template_variables(self):
        """Test template variable validation."""
        template = "Hello, {name}! Welcome to {place}."

        valid, missing = validate_template_variables(
            template, {"name": "Alice", "place": "Wonderland"}
        )
        assert valid
        assert len(missing) == 0

        valid, missing = validate_template_variables(
            template, {"name": "Alice"}
        )
        assert not valid
        assert "place" in missing

    def test_truncate_prompt(self):
        """Test prompt truncation."""
        prompt = "This is a long prompt that needs truncation."

        result = truncate_prompt(prompt, 20)
        assert len(result) <= 20
        assert result.endswith("...")

        # Preserve end
        result = truncate_prompt(prompt, 20, preserve_end=True)
        assert result.startswith("...")

    def test_truncate_prompt_no_truncation_needed(self):
        """Test no truncation when short enough."""
        prompt = "Short"
        result = truncate_prompt(prompt, 100)
        assert result == "Short"

    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "Hello, World!"  # 13 characters
        tokens = estimate_tokens(text)
        assert tokens == 3  # 13 / 4 = 3.25 -> 3

    def test_split_prompt_by_tokens(self):
        """Test splitting by tokens."""
        # Create a long prompt
        prompt = "This is sentence one. This is sentence two. This is sentence three."

        chunks = split_prompt_by_tokens(prompt, max_tokens=5, chars_per_token=4)
        assert len(chunks) > 1

        # All content should be preserved
        combined = "".join(chunks)
        # May have some overlap at boundaries
        assert "sentence one" in combined

    def test_split_prompt_no_split_needed(self):
        """Test no split when short enough."""
        prompt = "Short"
        chunks = split_prompt_by_tokens(prompt, max_tokens=100)
        assert chunks == ["Short"]


class TestDefaultLibrary:
    """Tests for default library functions."""

    def test_register_and_get_prompt(self):
        """Test using default library."""
        # Reset default library
        set_default_library(PromptLibrary())

        register_prompt("test", "Hello, {name}!")
        template = get_prompt("test")

        assert template.format(name="World") == "Hello, World!"

    def test_get_default_library(self):
        """Test getting default library."""
        set_default_library(PromptLibrary())

        lib1 = get_default_library()
        lib2 = get_default_library()

        assert lib1 is lib2  # Same instance


class TestPromptRole:
    """Tests for PromptRole enum."""

    def test_role_values(self):
        """Test role enum values."""
        assert PromptRole.SYSTEM.value == "system"
        assert PromptRole.USER.value == "user"
        assert PromptRole.ASSISTANT.value == "assistant"
        assert PromptRole.TOOL.value == "tool"
        assert PromptRole.FUNCTION.value == "function"
