"""Tests for prompt templates library."""

import pytest

from insideLLMs.templates import (
    BINARY_CLASSIFICATION,
    CHAIN_OF_THOUGHT,
    CODE_GENERATION,
    ENTITY_EXTRACTION,
    PromptTemplateInfo,
    TemplateCategory,
    TemplateLibrary,
    get_library,
    get_template,
    list_templates,
    register_template,
    render_template,
    search_templates,
)


class TestTemplateCategory:
    """Tests for TemplateCategory enum."""

    def test_all_categories_exist(self):
        """Test that all categories are defined."""
        assert TemplateCategory.REASONING.value == "reasoning"
        assert TemplateCategory.EXTRACTION.value == "extraction"
        assert TemplateCategory.CLASSIFICATION.value == "classification"
        assert TemplateCategory.GENERATION.value == "generation"
        assert TemplateCategory.SUMMARIZATION.value == "summarization"
        assert TemplateCategory.CODING.value == "coding"
        assert TemplateCategory.ANALYSIS.value == "analysis"


class TestPromptTemplateInfo:
    """Tests for PromptTemplateInfo."""

    def test_basic_creation(self):
        """Test basic template creation."""
        template = PromptTemplateInfo(
            name="test",
            category=TemplateCategory.REASONING,
            description="A test template",
            template="Hello {name}!",
            variables=["name"],
        )

        assert template.name == "test"
        assert template.category == TemplateCategory.REASONING
        assert "name" in template.variables

    def test_render(self):
        """Test template rendering."""
        template = PromptTemplateInfo(
            name="test",
            category=TemplateCategory.REASONING,
            description="Test",
            template="Hello {name}, you are {age} years old.",
            variables=["name", "age"],
        )

        rendered = template.render(name="Alice", age=30)
        assert "Hello Alice" in rendered
        assert "30 years old" in rendered

    def test_render_partial(self):
        """Test partial rendering."""
        template = PromptTemplateInfo(
            name="test",
            category=TemplateCategory.REASONING,
            description="Test",
            template="Hello {name}, {greeting}!",
            variables=["name", "greeting"],
        )

        rendered = template.render(name="Bob")
        assert "Hello Bob" in rendered
        assert "{greeting}" in rendered

    def test_get_missing_vars(self):
        """Test getting missing variables."""
        template = PromptTemplateInfo(
            name="test",
            category=TemplateCategory.REASONING,
            description="Test",
            template="Hello {a} {b} {c}",
            variables=["a", "b", "c"],
        )

        missing = template.get_missing_vars({"a", "c"})
        assert missing == {"b"}

    def test_with_examples(self):
        """Test template with examples."""
        template = PromptTemplateInfo(
            name="test",
            category=TemplateCategory.REASONING,
            description="Test",
            template="Q: {question}",
            variables=["question"],
            examples=[
                {"question": "What is 2+2?", "output": "4"},
            ],
            best_for=["math"],
        )

        assert len(template.examples) == 1
        assert "math" in template.best_for


class TestBuiltInTemplates:
    """Tests for built-in templates."""

    def test_chain_of_thought(self):
        """Test chain of thought template."""
        rendered = CHAIN_OF_THOUGHT.render(question="What is 2+2?")

        assert "What is 2+2?" in rendered
        assert "step by step" in rendered.lower()

    def test_entity_extraction(self):
        """Test entity extraction template."""
        rendered = ENTITY_EXTRACTION.render(
            entity_types="names and places", text="John went to Paris."
        )

        assert "names and places" in rendered
        assert "John went to Paris" in rendered
        assert "JSON" in rendered

    def test_binary_classification(self):
        """Test binary classification template."""
        rendered = BINARY_CLASSIFICATION.render(
            category_a="positive", category_b="negative", text="I love this!"
        )

        assert "positive" in rendered
        assert "negative" in rendered
        assert "I love this!" in rendered

    def test_code_generation(self):
        """Test code generation template."""
        rendered = CODE_GENERATION.render(
            language="python", requirements="- Calculate factorial\n- Handle edge cases"
        )

        assert "python" in rendered.lower()
        assert "factorial" in rendered.lower()


class TestTemplateLibrary:
    """Tests for TemplateLibrary."""

    def test_get_template(self):
        """Test getting template by name."""
        library = TemplateLibrary()
        template = library.get("chain_of_thought")

        assert template is not None
        assert template.name == "chain_of_thought"

    def test_get_nonexistent(self):
        """Test getting non-existent template."""
        library = TemplateLibrary()
        template = library.get("nonexistent")

        assert template is None

    def test_list_templates(self):
        """Test listing all templates."""
        library = TemplateLibrary()
        templates = library.list_templates()

        assert len(templates) > 0
        assert all(isinstance(t, PromptTemplateInfo) for t in templates)

    def test_list_by_category(self):
        """Test listing templates by category."""
        library = TemplateLibrary()
        reasoning = library.list_templates(TemplateCategory.REASONING)

        assert len(reasoning) > 0
        assert all(t.category == TemplateCategory.REASONING for t in reasoning)

    def test_list_names(self):
        """Test listing template names."""
        library = TemplateLibrary()
        names = library.list_names()

        assert "chain_of_thought" in names
        assert "binary_classification" in names

    def test_register_custom(self):
        """Test registering custom template."""
        library = TemplateLibrary()
        custom = PromptTemplateInfo(
            name="my_custom",
            category=TemplateCategory.GENERATION,
            description="Custom template",
            template="Custom: {input}",
            variables=["input"],
        )

        library.register(custom)
        retrieved = library.get("my_custom")

        assert retrieved is not None
        assert retrieved.name == "my_custom"

    def test_render(self):
        """Test rendering by name."""
        library = TemplateLibrary()
        rendered = library.render(
            "binary_classification", category_a="yes", category_b="no", text="Test input"
        )

        assert "yes" in rendered
        assert "no" in rendered

    def test_render_not_found(self):
        """Test rendering non-existent template."""
        library = TemplateLibrary()

        with pytest.raises(KeyError):
            library.render("nonexistent", param="value")

    def test_search(self):
        """Test searching templates."""
        library = TemplateLibrary()

        # Search by name
        results = library.search("chain")
        assert len(results) > 0

        # Search by description
        results = library.search("extract")
        assert len(results) > 0

        # Search by use case
        results = library.search("sentiment")
        assert len(results) > 0

    def test_search_no_results(self):
        """Test search with no results."""
        library = TemplateLibrary()
        results = library.search("xyznonexistent")

        assert len(results) == 0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_template(self):
        """Test get_template function."""
        template = get_template("chain_of_thought")
        assert template is not None

    def test_list_templates(self):
        """Test list_templates function."""
        templates = list_templates()
        assert len(templates) > 0

        # Filter by category
        reasoning = list_templates(TemplateCategory.REASONING)
        assert all(t.category == TemplateCategory.REASONING for t in reasoning)

    def test_render_template(self):
        """Test render_template function."""
        rendered = render_template(
            "concise_summary", max_sentences="3", text="This is the text to summarize."
        )

        assert "3 sentences" in rendered
        assert "summarize" in rendered.lower()

    def test_register_template(self):
        """Test register_template function."""
        custom = PromptTemplateInfo(
            name="test_custom_func",
            category=TemplateCategory.GENERATION,
            description="Test",
            template="Test: {x}",
            variables=["x"],
        )

        register_template(custom)
        retrieved = get_template("test_custom_func")

        assert retrieved is not None

    def test_search_templates(self):
        """Test search_templates function."""
        results = search_templates("code")
        assert len(results) > 0

    def test_get_library(self):
        """Test get_library function."""
        library = get_library()
        assert isinstance(library, TemplateLibrary)


class TestTemplateCategories:
    """Tests for different template categories."""

    def test_reasoning_templates(self):
        """Test reasoning templates exist."""
        templates = list_templates(TemplateCategory.REASONING)
        names = [t.name for t in templates]

        assert "chain_of_thought" in names
        assert "step_back" in names

    def test_extraction_templates(self):
        """Test extraction templates exist."""
        templates = list_templates(TemplateCategory.EXTRACTION)
        names = [t.name for t in templates]

        assert "entity_extraction" in names
        assert "structured_extraction" in names

    def test_classification_templates(self):
        """Test classification templates exist."""
        templates = list_templates(TemplateCategory.CLASSIFICATION)
        names = [t.name for t in templates]

        assert "binary_classification" in names
        assert "multi_class_classification" in names

    def test_summarization_templates(self):
        """Test summarization templates exist."""
        templates = list_templates(TemplateCategory.SUMMARIZATION)
        names = [t.name for t in templates]

        assert "concise_summary" in names
        assert "bullet_summary" in names

    def test_coding_templates(self):
        """Test coding templates exist."""
        templates = list_templates(TemplateCategory.CODING)
        names = [t.name for t in templates]

        assert "code_generation" in names
        assert "code_explanation" in names
        assert "code_review" in names

    def test_analysis_templates(self):
        """Test analysis templates exist."""
        templates = list_templates(TemplateCategory.ANALYSIS)
        names = [t.name for t in templates]

        assert "sentiment_analysis" in names
        assert "swot_analysis" in names


class TestTemplateRendering:
    """Tests for template rendering scenarios."""

    def test_render_all_variables(self):
        """Test rendering with all variables provided."""
        template = get_template("code_generation")
        rendered = template.render(
            language="javascript", requirements="- Create a function\n- Add error handling"
        )

        assert "javascript" in rendered
        assert "function" in rendered.lower()

    def test_render_multiline(self):
        """Test rendering with multiline content."""
        template = get_template("structured_extraction")
        rendered = template.render(
            fields="- name: string\n- age: number\n- email: string",
            text="John is 30 years old. Email: john@example.com",
        )

        assert "name:" in rendered
        assert "john@example.com" in rendered

    def test_render_special_characters(self):
        """Test rendering with special characters."""
        template = get_template("binary_classification")
        rendered = template.render(
            category_a="<positive>",
            category_b="<negative>",
            text="Test with 'quotes' and \"double quotes\"",
        )

        assert "<positive>" in rendered
        assert "quotes" in rendered


class TestTemplateMetadata:
    """Tests for template metadata."""

    def test_has_examples(self):
        """Test templates have examples."""
        template = get_template("chain_of_thought")
        assert len(template.examples) > 0

    def test_has_best_for(self):
        """Test templates have best_for list."""
        template = get_template("chain_of_thought")
        assert len(template.best_for) > 0

    def test_has_tips(self):
        """Test templates have tips."""
        template = get_template("chain_of_thought")
        assert len(template.tips) > 0

    def test_has_description(self):
        """Test templates have descriptions."""
        templates = list_templates()
        for template in templates:
            assert len(template.description) > 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_variable_value(self):
        """Test rendering with empty variable value."""
        template = PromptTemplateInfo(
            name="test",
            category=TemplateCategory.GENERATION,
            description="Test",
            template="Hello {name}!",
            variables=["name"],
        )

        rendered = template.render(name="")
        assert "Hello !" in rendered

    def test_variable_not_in_template(self):
        """Test providing variable not in template."""
        template = PromptTemplateInfo(
            name="test",
            category=TemplateCategory.GENERATION,
            description="Test",
            template="Hello World!",
            variables=[],
        )

        rendered = template.render(extra="ignored")
        assert rendered == "Hello World!"

    def test_unicode_in_template(self):
        """Test unicode in template."""
        template = PromptTemplateInfo(
            name="test",
            category=TemplateCategory.GENERATION,
            description="Test",
            template="Hello {name}! 🌍",
            variables=["name"],
        )

        rendered = template.render(name="世界")
        assert "世界" in rendered
        assert "🌍" in rendered
