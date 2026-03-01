"""
Prompt templates library for common LLM interaction patterns.

This module provides a comprehensive library of reusable prompt templates for
various LLM interaction patterns. It includes pre-built templates organized by
category, a flexible template rendering system, and utilities for template
management and discovery.

Key Components:
    - TemplateCategory: Enum defining categories like REASONING, EXTRACTION, etc.
    - PromptTemplateInfo: Dataclass representing a prompt template with metadata.
    - TemplateLibrary: Registry for managing and accessing templates.
    - Module-level functions: Convenience functions for the default library.

Features:
    - Pre-built templates for common use cases (chain-of-thought, extraction, etc.)
    - Template categories (reasoning, extraction, classification, generation, etc.)
    - Template variants and customization through variable substitution
    - Best practices and tips built into each template
    - Extensible design allowing custom template registration

Examples:
    Basic template rendering:

        >>> from insideLLMs.contrib.templates import render_template
        >>> prompt = render_template(
        ...     "chain_of_thought",
        ...     question="What is 15% of 80?"
        ... )
        >>> print(prompt)
        Let's approach this step by step.

        Question: What is 15% of 80?

        Think through this carefully, showing your reasoning at each step:

    Listing templates by category:

        >>> from insideLLMs.contrib.templates import list_templates, TemplateCategory
        >>> reasoning_templates = list_templates(TemplateCategory.REASONING)
        >>> for t in reasoning_templates:
        ...     print(f"{t.name}: {t.description}")
        chain_of_thought: Encourages step-by-step reasoning
        chain_of_thought_zero_shot: Zero-shot chain of thought with simple trigger
        step_back: Abstracts to higher-level concepts before answering

    Searching for templates:

        >>> from insideLLMs.contrib.templates import search_templates
        >>> code_templates = search_templates("code")
        >>> print([t.name for t in code_templates])
        ['code_generation', 'code_explanation', 'code_review', 'code_refactor']

    Creating and registering custom templates:

        >>> from insideLLMs.contrib.templates import (
        ...     PromptTemplateInfo, TemplateCategory, register_template
        ... )
        >>> custom = PromptTemplateInfo(
        ...     name="my_template",
        ...     category=TemplateCategory.GENERATION,
        ...     description="Generate haikus",
        ...     template="Write a haiku about {topic}:\\n\\n",
        ...     variables=["topic"],
        ...     best_for=["poetry", "creative writing"],
        ... )
        >>> register_template(custom)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TemplateCategory(Enum):
    """Categories of prompt templates for organizing and filtering templates.

    This enum defines the primary categories used to classify prompt templates.
    Each category groups templates that serve similar purposes, making it easier
    to discover and select appropriate templates for specific tasks.

    Attributes:
        REASONING: Templates for logical reasoning and step-by-step thinking.
            Use for math problems, logical puzzles, and complex decisions.
        EXTRACTION: Templates for extracting structured data from text.
            Use for NER, data mining, and information extraction.
        CLASSIFICATION: Templates for categorizing or labeling text.
            Use for sentiment analysis, topic classification, and tagging.
        GENERATION: Templates for creating new content.
            Use for creative writing, content expansion, and rewriting.
        SUMMARIZATION: Templates for condensing text.
            Use for creating summaries, bullet points, and abstracts.
        TRANSLATION: Templates for language translation tasks.
            Use for translating between languages.
        CODING: Templates for programming-related tasks.
            Use for code generation, review, explanation, and refactoring.
        ANALYSIS: Templates for analyzing content.
            Use for sentiment analysis, comparisons, and SWOT analysis.
        CONVERSATION: Templates for dialogue and persona-based interactions.
            Use for roleplay, tutoring, and specialized assistance.
        INSTRUCTION: Templates for generating instructions and guides.
            Use for tutorials, task decomposition, and how-to content.

    Examples:
        Filtering templates by category:

            >>> from insideLLMs.contrib.templates import list_templates, TemplateCategory
            >>> coding_templates = list_templates(TemplateCategory.CODING)
            >>> print([t.name for t in coding_templates])
            ['code_generation', 'code_explanation', 'code_review', 'code_refactor']

        Using category in template info:

            >>> from insideLLMs.contrib.templates import PromptTemplateInfo, TemplateCategory
            >>> template = PromptTemplateInfo(
            ...     name="debug_helper",
            ...     category=TemplateCategory.CODING,
            ...     description="Help debug code issues",
            ...     template="Debug this {language} code: {code}",
            ...     variables=["language", "code"],
            ... )
            >>> print(template.category.value)
            'coding'

        Iterating over categories:

            >>> from insideLLMs.contrib.templates import TemplateCategory
            >>> for category in TemplateCategory:
            ...     print(f"{category.name}: {category.value}")
            REASONING: reasoning
            EXTRACTION: extraction
            CLASSIFICATION: classification
            ...

        Checking if a template matches a category:

            >>> from insideLLMs.contrib.templates import get_template, TemplateCategory
            >>> template = get_template("chain_of_thought")
            >>> is_reasoning = template.category == TemplateCategory.REASONING
            >>> print(is_reasoning)
            True
    """

    REASONING = "reasoning"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODING = "coding"
    ANALYSIS = "analysis"
    CONVERSATION = "conversation"
    INSTRUCTION = "instruction"


@dataclass
class PromptTemplateInfo:
    """Information about a prompt template with metadata and rendering capabilities.

    This dataclass represents a complete prompt template, including the template
    string, required variables, usage examples, and best practice tips. It provides
    methods for rendering templates with variable substitution and validation.

    Attributes:
        name: Unique identifier for the template (e.g., "chain_of_thought").
            Used for retrieval from the template library.
        category: The TemplateCategory this template belongs to.
            Used for filtering and organization.
        description: Brief human-readable description of what the template does.
        template: The actual prompt template string with {variable} placeholders.
            Variables are enclosed in curly braces and replaced during rendering.
        variables: List of variable names required by the template.
            These must be provided when rendering.
        examples: List of example usage dictionaries showing input/output pairs.
            Each dict maps variable names to example values.
        best_for: List of use cases where this template excels.
            Helps users select the right template.
        tips: List of tips for using the template effectively.
            Includes best practices and caveats.

    Examples:
        Creating a simple template:

            >>> from insideLLMs.contrib.templates import PromptTemplateInfo, TemplateCategory
            >>> template = PromptTemplateInfo(
            ...     name="greeting",
            ...     category=TemplateCategory.GENERATION,
            ...     description="Generate a greeting message",
            ...     template="Hello {name}, welcome to {place}!",
            ...     variables=["name", "place"],
            ...     best_for=["welcome messages", "onboarding"],
            ...     tips=["Keep the place specific for better personalization"],
            ... )
            >>> print(template.render(name="Alice", place="Wonderland"))
            Hello Alice, welcome to Wonderland!

        Checking for missing variables:

            >>> template = PromptTemplateInfo(
            ...     name="email",
            ...     category=TemplateCategory.GENERATION,
            ...     description="Email template",
            ...     template="Dear {recipient},\\n\\n{body}\\n\\nBest,\\n{sender}",
            ...     variables=["recipient", "body", "sender"],
            ... )
            >>> missing = template.get_missing_vars({"recipient", "body"})
            >>> print(missing)
            {'sender'}

        Creating a template with examples:

            >>> template = PromptTemplateInfo(
            ...     name="translate",
            ...     category=TemplateCategory.TRANSLATION,
            ...     description="Translate text between languages",
            ...     template="Translate to {target_lang}: {text}",
            ...     variables=["target_lang", "text"],
            ...     examples=[
            ...         {"target_lang": "Spanish", "text": "Hello world"},
            ...         {"target_lang": "French", "text": "Good morning"},
            ...     ],
            ... )
            >>> print(template.examples[0])
            {'target_lang': 'Spanish', 'text': 'Hello world'}

        Using template metadata for documentation:

            >>> template = PromptTemplateInfo(
            ...     name="code_review",
            ...     category=TemplateCategory.CODING,
            ...     description="Review code for issues",
            ...     template="Review this {language} code:\\n{code}",
            ...     variables=["language", "code"],
            ...     best_for=["quality assurance", "learning"],
            ...     tips=["Specify the language for better context"],
            ... )
            >>> print(f"Best for: {', '.join(template.best_for)}")
            Best for: quality assurance, learning
            >>> print(f"Tips: {template.tips[0]}")
            Tips: Specify the language for better context
    """

    name: str
    category: TemplateCategory
    description: str
    template: str
    variables: list[str]
    examples: list[dict[str, str]] = field(default_factory=list)
    best_for: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)

    def render(self, **kwargs) -> str:
        """Render template by substituting variable placeholders with provided values.

        This method replaces all {variable} placeholders in the template string
        with the corresponding values from kwargs. Variables not provided in kwargs
        are left unchanged in the output.

        Args:
            **kwargs: Variable name-value pairs for substitution. Keys should match
                the variable names defined in self.variables. Values are converted
                to strings using str() before substitution.

        Returns:
            The rendered template string with variables replaced by their values.
            Unprovided variables remain as {variable} placeholders.

        Examples:
            Basic rendering with all variables:

                >>> from insideLLMs.contrib.templates import get_template
                >>> template = get_template("chain_of_thought")
                >>> prompt = template.render(question="What is 2 + 2?")
                >>> print(prompt)
                Let's approach this step by step.

                Question: What is 2 + 2?

                Think through this carefully, showing your reasoning at each step:

            Rendering with multiple variables:

                >>> from insideLLMs.contrib.templates import get_template
                >>> template = get_template("binary_classification")
                >>> prompt = template.render(
                ...     category_a="spam",
                ...     category_b="not spam",
                ...     text="Buy now! Limited offer!"
                ... )
                >>> "spam" in prompt and "not spam" in prompt
                True

            Partial rendering (missing variables stay as placeholders):

                >>> from insideLLMs.contrib.templates import PromptTemplateInfo, TemplateCategory
                >>> template = PromptTemplateInfo(
                ...     name="test",
                ...     category=TemplateCategory.GENERATION,
                ...     description="Test",
                ...     template="Hello {name}, your {item} is ready",
                ...     variables=["name", "item"],
                ... )
                >>> print(template.render(name="Bob"))
                Hello Bob, your {item} is ready

            Values are converted to strings:

                >>> from insideLLMs.contrib.templates import PromptTemplateInfo, TemplateCategory
                >>> template = PromptTemplateInfo(
                ...     name="math",
                ...     category=TemplateCategory.REASONING,
                ...     description="Math problem",
                ...     template="Calculate {num1} + {num2}",
                ...     variables=["num1", "num2"],
                ... )
                >>> print(template.render(num1=42, num2=58))
                Calculate 42 + 58
        """
        result = self.template
        for var in self.variables:
            if var in kwargs:
                result = result.replace(f"{{{var}}}", str(kwargs[var]))
        return result

    def get_missing_vars(self, provided: set[str]) -> set[str]:
        """Get missing required variables by comparing provided names against required.

        This method helps validate that all required variables have been provided
        before rendering a template. It computes the set difference between the
        template's required variables and the provided variable names.

        Args:
            provided: Set of variable names that have been provided. This should
                contain the keys that will be passed to render().

        Returns:
            Set of variable names that are required by the template but were not
            provided. Returns an empty set if all variables are provided.

        Examples:
            Checking for missing variables before rendering:

                >>> from insideLLMs.contrib.templates import get_template
                >>> template = get_template("binary_classification")
                >>> provided = {"category_a", "text"}
                >>> missing = template.get_missing_vars(provided)
                >>> print(missing)
                {'category_b'}

            All variables provided (returns empty set):

                >>> from insideLLMs.contrib.templates import get_template
                >>> template = get_template("chain_of_thought")
                >>> provided = {"question"}
                >>> missing = template.get_missing_vars(provided)
                >>> print(missing)
                set()
                >>> print(len(missing) == 0)
                True

            Validating user input before rendering:

                >>> from insideLLMs.contrib.templates import get_template
                >>> template = get_template("code_generation")
                >>> user_vars = {"language": "Python"}
                >>> missing = template.get_missing_vars(set(user_vars.keys()))
                >>> if missing:
                ...     print(f"Please provide: {', '.join(missing)}")
                Please provide: requirements

            Using with kwargs from function call:

                >>> from insideLLMs.contrib.templates import get_template
                >>> template = get_template("creative_writing")
                >>> kwargs = {"content_type": "poem", "topic": "nature"}
                >>> missing = template.get_missing_vars(set(kwargs.keys()))
                >>> # Returns {'style', 'tone', 'length'}
                >>> len(missing) == 3
                True
        """
        return set(self.variables) - provided


# Chain of Thought Templates
CHAIN_OF_THOUGHT = PromptTemplateInfo(
    name="chain_of_thought",
    category=TemplateCategory.REASONING,
    description="Encourages step-by-step reasoning",
    template="""Let's approach this step by step.

Question: {question}

Think through this carefully, showing your reasoning at each step:""",
    variables=["question"],
    examples=[
        {
            "question": "If a train travels 120 miles in 2 hours, what is its average speed?",
            "output": "Step 1: Identify what we know...",
        }
    ],
    best_for=["math problems", "logical reasoning", "complex decisions"],
    tips=[
        "Works best for multi-step problems",
        "Helps reduce errors in complex reasoning",
        "May increase output length",
    ],
)

CHAIN_OF_THOUGHT_ZERO_SHOT = PromptTemplateInfo(
    name="chain_of_thought_zero_shot",
    category=TemplateCategory.REASONING,
    description="Zero-shot chain of thought with simple trigger",
    template="""{question}

Let's think step by step.""",
    variables=["question"],
    examples=[],
    best_for=["quick reasoning tasks", "when examples aren't available"],
    tips=["Simple but effective", "Add 'Let's think step by step' triggers CoT"],
)

STEP_BACK_PROMPTING = PromptTemplateInfo(
    name="step_back",
    category=TemplateCategory.REASONING,
    description="Abstracts to higher-level concepts before answering",
    template="""Before answering the specific question, let's first consider the broader concepts involved.

Question: {question}

First, what general principles or concepts are relevant here?
Then, apply those to answer the specific question.""",
    variables=["question"],
    best_for=["physics problems", "complex technical questions", "abstract reasoning"],
    tips=["Helps with questions requiring domain knowledge"],
)

# Extraction Templates
ENTITY_EXTRACTION = PromptTemplateInfo(
    name="entity_extraction",
    category=TemplateCategory.EXTRACTION,
    description="Extract named entities from text",
    template="""Extract all {entity_types} from the following text.

Text: {text}

Return the entities as a JSON list with the format:
[{{"type": "entity_type", "value": "extracted_value", "context": "surrounding text"}}]

Entities:""",
    variables=["entity_types", "text"],
    examples=[
        {
            "entity_types": "person names and organizations",
            "text": "John Smith works at Google in Mountain View.",
        }
    ],
    best_for=["NER tasks", "information extraction", "data mining"],
    tips=["Specify entity types clearly", "Request structured output format"],
)

STRUCTURED_DATA_EXTRACTION = PromptTemplateInfo(
    name="structured_extraction",
    category=TemplateCategory.EXTRACTION,
    description="Extract structured data from unstructured text",
    template="""Extract the following information from the text and return it as JSON:

Fields to extract:
{fields}

Text:
{text}

Return only valid JSON with the extracted fields:""",
    variables=["fields", "text"],
    best_for=["form filling", "document parsing", "data structuring"],
    tips=["Define expected fields clearly", "Specify data types if needed"],
)

KEY_VALUE_EXTRACTION = PromptTemplateInfo(
    name="key_value_extraction",
    category=TemplateCategory.EXTRACTION,
    description="Extract key-value pairs from text",
    template="""Extract all key-value pairs from the following text.

Text: {text}

Return as a JSON object where keys are the attribute names and values are their corresponding values:""",
    variables=["text"],
    best_for=["parsing specifications", "extracting metadata"],
    tips=["Works well with semi-structured text"],
)

# Classification Templates
BINARY_CLASSIFICATION = PromptTemplateInfo(
    name="binary_classification",
    category=TemplateCategory.CLASSIFICATION,
    description="Classify into one of two categories",
    template="""Classify the following text as either "{category_a}" or "{category_b}".

Text: {text}

Classification (respond with only "{category_a}" or "{category_b}"):""",
    variables=["category_a", "category_b", "text"],
    examples=[
        {
            "category_a": "positive",
            "category_b": "negative",
            "text": "I love this product!",
        }
    ],
    best_for=["sentiment analysis", "spam detection", "yes/no decisions"],
    tips=["Keep categories mutually exclusive", "Consider adding confidence request"],
)

MULTI_CLASS_CLASSIFICATION = PromptTemplateInfo(
    name="multi_class_classification",
    category=TemplateCategory.CLASSIFICATION,
    description="Classify into multiple categories",
    template="""Classify the following text into one of these categories: {categories}

Text: {text}

Provide your classification in the following format:
Category: [chosen category]
Confidence: [high/medium/low]
Reasoning: [brief explanation]""",
    variables=["categories", "text"],
    best_for=["topic classification", "intent detection", "document categorization"],
    tips=["List categories clearly", "Include confidence for better analysis"],
)

MULTI_LABEL_CLASSIFICATION = PromptTemplateInfo(
    name="multi_label_classification",
    category=TemplateCategory.CLASSIFICATION,
    description="Assign multiple labels to text",
    template="""Assign all applicable labels from the following list to the text.

Available labels: {labels}

Text: {text}

Return the applicable labels as a JSON array. Only include labels that clearly apply:""",
    variables=["labels", "text"],
    best_for=["tagging", "multi-aspect categorization"],
    tips=["Labels are not mutually exclusive", "Specify minimum confidence threshold"],
)

# Summarization Templates
CONCISE_SUMMARY = PromptTemplateInfo(
    name="concise_summary",
    category=TemplateCategory.SUMMARIZATION,
    description="Create a brief summary",
    template="""Summarize the following text in {max_sentences} sentences or fewer.

Text:
{text}

Summary:""",
    variables=["max_sentences", "text"],
    best_for=["quick overviews", "abstract generation"],
    tips=["Specify length constraints", "Use for non-technical content"],
)

BULLET_POINT_SUMMARY = PromptTemplateInfo(
    name="bullet_summary",
    category=TemplateCategory.SUMMARIZATION,
    description="Summarize as bullet points",
    template="""Extract the {num_points} most important points from the following text as bullet points.

Text:
{text}

Key points:
•""",
    variables=["num_points", "text"],
    best_for=["meeting notes", "article highlights", "quick reference"],
    tips=["Specify number of points", "Good for scannable output"],
)

STRUCTURED_SUMMARY = PromptTemplateInfo(
    name="structured_summary",
    category=TemplateCategory.SUMMARIZATION,
    description="Summary with structured sections",
    template="""Create a structured summary of the following text with these sections:
- Main Topic
- Key Points (3-5 bullet points)
- Conclusions
- Action Items (if any)

Text:
{text}

Structured Summary:""",
    variables=["text"],
    best_for=["reports", "meeting summaries", "research papers"],
    tips=["Customize sections as needed"],
)

# Generation Templates
CREATIVE_WRITING = PromptTemplateInfo(
    name="creative_writing",
    category=TemplateCategory.GENERATION,
    description="Generate creative content",
    template="""Write a {content_type} about {topic}.

Style: {style}
Tone: {tone}
Length: {length}

Begin:""",
    variables=["content_type", "topic", "style", "tone", "length"],
    best_for=["stories", "poems", "creative content"],
    tips=["Specify style and tone clearly", "Give length guidance"],
)

CONTENT_EXPANSION = PromptTemplateInfo(
    name="content_expansion",
    category=TemplateCategory.GENERATION,
    description="Expand on given content",
    template="""Expand on the following {content_type}, adding more detail, examples, and explanation.

Original:
{content}

Expanded version:""",
    variables=["content_type", "content"],
    best_for=["elaboration", "content enhancement"],
    tips=["Specify what aspects to expand on"],
)

REWRITE = PromptTemplateInfo(
    name="rewrite",
    category=TemplateCategory.GENERATION,
    description="Rewrite content with different style/tone",
    template="""Rewrite the following text to be more {target_style}.

Original:
{text}

Rewritten version:""",
    variables=["target_style", "text"],
    best_for=["tone adjustment", "formality changes", "simplification"],
    tips=["Be specific about target style"],
)

# Coding Templates
CODE_GENERATION = PromptTemplateInfo(
    name="code_generation",
    category=TemplateCategory.CODING,
    description="Generate code from description",
    template="""Write {language} code that:
{requirements}

Include:
- Clear comments explaining the logic
- Error handling where appropriate
- Example usage

```{language}""",
    variables=["language", "requirements"],
    best_for=["new function creation", "algorithm implementation"],
    tips=["Specify language and requirements clearly", "Request tests if needed"],
)

CODE_EXPLANATION = PromptTemplateInfo(
    name="code_explanation",
    category=TemplateCategory.CODING,
    description="Explain what code does",
    template="""Explain what the following {language} code does, step by step:

```{language}
{code}
```

Provide:
1. Overall purpose
2. Step-by-step explanation
3. Key concepts used
4. Potential improvements (if any)""",
    variables=["language", "code"],
    best_for=["code review", "learning", "documentation"],
    tips=["Specify language for better context"],
)

CODE_REVIEW = PromptTemplateInfo(
    name="code_review",
    category=TemplateCategory.CODING,
    description="Review code for issues and improvements",
    template="""Review the following {language} code for:
- Bugs and potential issues
- Performance concerns
- Code style and readability
- Security vulnerabilities
- Best practices

```{language}
{code}
```

Code Review:""",
    variables=["language", "code"],
    best_for=["quality assurance", "learning", "pre-commit review"],
    tips=["Specify what aspects to focus on"],
)

CODE_REFACTOR = PromptTemplateInfo(
    name="code_refactor",
    category=TemplateCategory.CODING,
    description="Refactor code for improvements",
    template="""Refactor the following {language} code to improve {improvement_focus}.

Original:
```{language}
{code}
```

Refactored version with explanation of changes:""",
    variables=["language", "code", "improvement_focus"],
    best_for=["code cleanup", "performance optimization", "modernization"],
    tips=["Specify what to improve: readability, performance, etc."],
)

# Analysis Templates
SENTIMENT_ANALYSIS = PromptTemplateInfo(
    name="sentiment_analysis",
    category=TemplateCategory.ANALYSIS,
    description="Analyze sentiment of text",
    template="""Analyze the sentiment of the following text.

Text: {text}

Provide:
1. Overall sentiment (positive/negative/neutral/mixed)
2. Sentiment score (-1 to 1)
3. Key phrases indicating sentiment
4. Confidence level (high/medium/low)

Analysis:""",
    variables=["text"],
    best_for=["review analysis", "social media monitoring", "feedback analysis"],
    tips=["Works best with opinion-based text"],
)

COMPARATIVE_ANALYSIS = PromptTemplateInfo(
    name="comparative_analysis",
    category=TemplateCategory.ANALYSIS,
    description="Compare two or more items",
    template="""Compare and contrast the following {item_type}:

{items}

Provide a structured comparison covering:
1. Key similarities
2. Key differences
3. Strengths and weaknesses of each
4. Recommendation (if applicable)""",
    variables=["item_type", "items"],
    best_for=["product comparison", "decision making", "research"],
    tips=["Structure items clearly"],
)

SWOT_ANALYSIS = PromptTemplateInfo(
    name="swot_analysis",
    category=TemplateCategory.ANALYSIS,
    description="Perform SWOT analysis",
    template="""Perform a SWOT analysis for: {subject}

Context: {context}

Provide analysis in the following format:

**Strengths:**
-

**Weaknesses:**
-

**Opportunities:**
-

**Threats:**
-

**Summary and Recommendations:**""",
    variables=["subject", "context"],
    best_for=["business analysis", "strategic planning", "evaluation"],
    tips=["Provide sufficient context about the subject"],
)

# Conversation Templates
PERSONA = PromptTemplateInfo(
    name="persona",
    category=TemplateCategory.CONVERSATION,
    description="Adopt a specific persona for conversation",
    template="""You are {persona_name}, {persona_description}.

Your communication style:
{communication_style}

Your expertise includes:
{expertise}

Respond to all messages in character. Begin by introducing yourself briefly.

User: {user_message}""",
    variables=[
        "persona_name",
        "persona_description",
        "communication_style",
        "expertise",
        "user_message",
    ],
    best_for=["roleplay", "specialized assistance", "educational scenarios"],
    tips=["Define persona clearly", "Include communication style"],
)

SOCRATIC_DIALOGUE = PromptTemplateInfo(
    name="socratic",
    category=TemplateCategory.CONVERSATION,
    description="Guide learning through questions",
    template="""Act as a Socratic tutor helping someone understand {topic}.

Instead of giving direct answers, guide learning through thoughtful questions that:
- Build on what they already know
- Challenge assumptions
- Lead to deeper understanding

The learner says: {learner_statement}

Respond with a guiding question:""",
    variables=["topic", "learner_statement"],
    best_for=["teaching", "critical thinking", "self-discovery"],
    tips=["Encourages active learning"],
)

# Instruction Following Templates
STEP_BY_STEP_INSTRUCTIONS = PromptTemplateInfo(
    name="step_by_step",
    category=TemplateCategory.INSTRUCTION,
    description="Generate clear step-by-step instructions",
    template="""Create clear, numbered step-by-step instructions for: {task}

Target audience: {audience}
Assumed knowledge level: {knowledge_level}

Format each step as:
[Step N]: [Action verb] [specific instruction]
- [Additional details if needed]

Instructions:""",
    variables=["task", "audience", "knowledge_level"],
    best_for=["tutorials", "documentation", "how-to guides"],
    tips=["Specify audience clearly", "Include any prerequisites"],
)

TASK_DECOMPOSITION = PromptTemplateInfo(
    name="task_decomposition",
    category=TemplateCategory.INSTRUCTION,
    description="Break complex task into subtasks",
    template="""Break down the following complex task into smaller, manageable subtasks:

Task: {task}

For each subtask, provide:
1. Clear description
2. Expected output/deliverable
3. Dependencies on other subtasks
4. Estimated effort (if applicable)

Subtasks:""",
    variables=["task"],
    best_for=["project planning", "complex problem solving", "delegation"],
    tips=["Good for overwhelming tasks"],
)


# Template Registry
class TemplateLibrary:
    """Library of prompt templates with registration, retrieval, and search capabilities.

    This class serves as a registry for prompt templates, providing methods to
    register, retrieve, list, and search templates. It comes pre-populated with
    a comprehensive set of built-in templates covering common LLM use cases.

    The library supports:
        - Template retrieval by name
        - Listing templates with optional category filtering
        - Keyword-based template search
        - Custom template registration
        - Direct template rendering by name

    Attributes:
        _templates: Internal dictionary mapping template names to PromptTemplateInfo
            instances.

    Examples:
        Creating and using a template library:

            >>> from insideLLMs.contrib.templates import TemplateLibrary
            >>> library = TemplateLibrary()
            >>> template = library.get("chain_of_thought")
            >>> print(template.description)
            Encourages step-by-step reasoning

        Listing templates by category:

            >>> from insideLLMs.contrib.templates import TemplateLibrary, TemplateCategory
            >>> library = TemplateLibrary()
            >>> coding_templates = library.list_templates(TemplateCategory.CODING)
            >>> print([t.name for t in coding_templates])
            ['code_generation', 'code_explanation', 'code_review', 'code_refactor']

        Searching for templates:

            >>> from insideLLMs.contrib.templates import TemplateLibrary
            >>> library = TemplateLibrary()
            >>> results = library.search("sentiment")
            >>> print([t.name for t in results])
            ['sentiment_analysis']

        Registering custom templates:

            >>> from insideLLMs.contrib.templates import (
            ...     TemplateLibrary, PromptTemplateInfo, TemplateCategory
            ... )
            >>> library = TemplateLibrary()
            >>> custom = PromptTemplateInfo(
            ...     name="my_custom_template",
            ...     category=TemplateCategory.GENERATION,
            ...     description="Custom generation template",
            ...     template="Generate {output_type} about {topic}",
            ...     variables=["output_type", "topic"],
            ... )
            >>> library.register(custom)
            >>> print(library.get("my_custom_template").description)
            Custom generation template
    """

    def __init__(self):
        """Initialize a new TemplateLibrary with built-in templates.

        Creates an empty template registry and populates it with the default
        set of built-in templates covering reasoning, extraction, classification,
        generation, summarization, coding, analysis, conversation, and instruction
        categories.

        Examples:
            Creating a new library instance:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> print(len(library.list_templates()) > 0)
                True

            Built-in templates are immediately available:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> template = library.get("chain_of_thought")
                >>> print(template is not None)
                True

            Each instance has its own template registry:

                >>> from insideLLMs.contrib.templates import TemplateLibrary, PromptTemplateInfo, TemplateCategory
                >>> lib1 = TemplateLibrary()
                >>> lib2 = TemplateLibrary()
                >>> custom = PromptTemplateInfo(
                ...     name="unique", category=TemplateCategory.GENERATION,
                ...     description="test", template="test", variables=[]
                ... )
                >>> lib1.register(custom)
                >>> print(lib1.get("unique") is not None)
                True
                >>> print(lib2.get("unique") is None)
                True
        """
        self._templates: dict[str, PromptTemplateInfo] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register the built-in templates to the library.

        This internal method populates the library with the default set of
        prompt templates. It is called automatically during initialization.

        The built-in templates include:
            - Reasoning: chain_of_thought, chain_of_thought_zero_shot, step_back
            - Extraction: entity_extraction, structured_extraction, key_value_extraction
            - Classification: binary, multi_class, multi_label classification
            - Summarization: concise_summary, bullet_summary, structured_summary
            - Generation: creative_writing, content_expansion, rewrite
            - Coding: code_generation, code_explanation, code_review, code_refactor
            - Analysis: sentiment_analysis, comparative_analysis, swot_analysis
            - Conversation: persona, socratic
            - Instruction: step_by_step, task_decomposition
        """
        builtins = [
            CHAIN_OF_THOUGHT,
            CHAIN_OF_THOUGHT_ZERO_SHOT,
            STEP_BACK_PROMPTING,
            ENTITY_EXTRACTION,
            STRUCTURED_DATA_EXTRACTION,
            KEY_VALUE_EXTRACTION,
            BINARY_CLASSIFICATION,
            MULTI_CLASS_CLASSIFICATION,
            MULTI_LABEL_CLASSIFICATION,
            CONCISE_SUMMARY,
            BULLET_POINT_SUMMARY,
            STRUCTURED_SUMMARY,
            CREATIVE_WRITING,
            CONTENT_EXPANSION,
            REWRITE,
            CODE_GENERATION,
            CODE_EXPLANATION,
            CODE_REVIEW,
            CODE_REFACTOR,
            SENTIMENT_ANALYSIS,
            COMPARATIVE_ANALYSIS,
            SWOT_ANALYSIS,
            PERSONA,
            SOCRATIC_DIALOGUE,
            STEP_BY_STEP_INSTRUCTIONS,
            TASK_DECOMPOSITION,
        ]
        for template in builtins:
            self._templates[template.name] = template

    def get(self, name: str) -> Optional[PromptTemplateInfo]:
        """Get a template by its unique name.

        Retrieves a PromptTemplateInfo instance from the library by its name.
        This is the primary method for accessing specific templates when you
        know the template name.

        Args:
            name: The unique name of the template to retrieve (e.g., "chain_of_thought",
                "code_review"). Names are case-sensitive.

        Returns:
            The PromptTemplateInfo instance if found, or None if no template
            with the given name exists in the library.

        Examples:
            Getting a built-in template:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> template = library.get("chain_of_thought")
                >>> print(template.name)
                chain_of_thought
                >>> print(template.category.value)
                reasoning

            Handling non-existent templates:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> template = library.get("nonexistent_template")
                >>> print(template is None)
                True

            Using the returned template:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> template = library.get("binary_classification")
                >>> if template:
                ...     prompt = template.render(
                ...         category_a="positive",
                ...         category_b="negative",
                ...         text="Great product!"
                ...     )
                ...     print("positive" in prompt)
                True

            Checking template metadata:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> template = library.get("code_generation")
                >>> print(template.variables)
                ['language', 'requirements']
                >>> print(template.best_for)
                ['new function creation', 'algorithm implementation']
        """
        return self._templates.get(name)

    def list_templates(
        self, category: Optional[TemplateCategory] = None
    ) -> list[PromptTemplateInfo]:
        """List all available templates, optionally filtered by category.

        Returns a list of all PromptTemplateInfo instances in the library.
        When a category is specified, only templates belonging to that category
        are returned.

        Args:
            category: Optional TemplateCategory to filter by. If None, all
                templates are returned. If specified, only templates with
                matching category are returned.

        Returns:
            List of PromptTemplateInfo instances. Returns all templates if
            no category filter is specified, or only matching templates
            if a category is provided.

        Examples:
            Listing all templates:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> all_templates = library.list_templates()
                >>> print(len(all_templates) >= 20)  # Many built-in templates
                True

            Filtering by category:

                >>> from insideLLMs.contrib.templates import TemplateLibrary, TemplateCategory
                >>> library = TemplateLibrary()
                >>> reasoning = library.list_templates(TemplateCategory.REASONING)
                >>> print([t.name for t in reasoning])
                ['chain_of_thought', 'chain_of_thought_zero_shot', 'step_back']

            Iterating over templates in a category:

                >>> from insideLLMs.contrib.templates import TemplateLibrary, TemplateCategory
                >>> library = TemplateLibrary()
                >>> for template in library.list_templates(TemplateCategory.CODING):
                ...     print(f"{template.name}: {template.description}")
                code_generation: Generate code from description
                code_explanation: Explain what code does
                code_review: Review code for issues and improvements
                code_refactor: Refactor code for improvements

            Building a template catalog:

                >>> from insideLLMs.contrib.templates import TemplateLibrary, TemplateCategory
                >>> library = TemplateLibrary()
                >>> catalog = {}
                >>> for cat in TemplateCategory:
                ...     templates = library.list_templates(cat)
                ...     if templates:
                ...         catalog[cat.value] = [t.name for t in templates]
                >>> print("reasoning" in catalog)
                True
        """
        templates = list(self._templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates

    def list_names(self, category: Optional[TemplateCategory] = None) -> list[str]:
        """List template names, optionally filtered by category.

        Returns a list of template names (strings) rather than full template
        objects. This is useful for quick lookups, displaying available options,
        or building selection menus.

        Args:
            category: Optional TemplateCategory to filter by. If None, all
                template names are returned. If specified, only names of
                templates with matching category are returned.

        Returns:
            List of template name strings. Returns all names if no category
            filter is specified.

        Examples:
            Getting all template names:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> names = library.list_names()
                >>> print("chain_of_thought" in names)
                True
                >>> print("code_review" in names)
                True

            Getting names by category:

                >>> from insideLLMs.contrib.templates import TemplateLibrary, TemplateCategory
                >>> library = TemplateLibrary()
                >>> extraction_names = library.list_names(TemplateCategory.EXTRACTION)
                >>> print(extraction_names)
                ['entity_extraction', 'structured_extraction', 'key_value_extraction']

            Building a selection menu:

                >>> from insideLLMs.contrib.templates import TemplateLibrary, TemplateCategory
                >>> library = TemplateLibrary()
                >>> print("Available reasoning templates:")
                Available reasoning templates:
                >>> for i, name in enumerate(library.list_names(TemplateCategory.REASONING), 1):
                ...     print(f"  {i}. {name}")
                  1. chain_of_thought
                  2. chain_of_thought_zero_shot
                  3. step_back

            Checking if a template exists:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> available = library.list_names()
                >>> print("binary_classification" in available)
                True
                >>> print("nonexistent" in available)
                False
        """
        return [t.name for t in self.list_templates(category)]

    def register(self, template: PromptTemplateInfo) -> None:
        """Register a custom template to the library.

        Adds a new template to the library, making it available for retrieval,
        listing, and searching. If a template with the same name already exists,
        it will be overwritten.

        Args:
            template: The PromptTemplateInfo instance to register. The template's
                name attribute will be used as the key for retrieval.

        Examples:
            Registering a simple custom template:

                >>> from insideLLMs.contrib.templates import (
                ...     TemplateLibrary, PromptTemplateInfo, TemplateCategory
                ... )
                >>> library = TemplateLibrary()
                >>> template = PromptTemplateInfo(
                ...     name="greeting",
                ...     category=TemplateCategory.GENERATION,
                ...     description="Generate a greeting",
                ...     template="Hello {name}, welcome to {place}!",
                ...     variables=["name", "place"],
                ... )
                >>> library.register(template)
                >>> print(library.get("greeting") is not None)
                True

            Overwriting an existing template:

                >>> from insideLLMs.contrib.templates import (
                ...     TemplateLibrary, PromptTemplateInfo, TemplateCategory
                ... )
                >>> library = TemplateLibrary()
                >>> original = library.get("chain_of_thought")
                >>> custom_cot = PromptTemplateInfo(
                ...     name="chain_of_thought",  # Same name, will overwrite
                ...     category=TemplateCategory.REASONING,
                ...     description="Custom chain of thought",
                ...     template="Think step by step about: {question}",
                ...     variables=["question"],
                ... )
                >>> library.register(custom_cot)
                >>> print(library.get("chain_of_thought").description)
                Custom chain of thought

            Registering with full metadata:

                >>> from insideLLMs.contrib.templates import (
                ...     TemplateLibrary, PromptTemplateInfo, TemplateCategory
                ... )
                >>> library = TemplateLibrary()
                >>> template = PromptTemplateInfo(
                ...     name="email_draft",
                ...     category=TemplateCategory.GENERATION,
                ...     description="Draft a professional email",
                ...     template="Write an email to {recipient} about {subject}.",
                ...     variables=["recipient", "subject"],
                ...     examples=[{"recipient": "team", "subject": "meeting"}],
                ...     best_for=["business communication", "professional emails"],
                ...     tips=["Keep it concise", "Include clear action items"],
                ... )
                >>> library.register(template)
                >>> registered = library.get("email_draft")
                >>> print(registered.best_for)
                ['business communication', 'professional emails']

            Registering multiple templates:

                >>> from insideLLMs.contrib.templates import (
                ...     TemplateLibrary, PromptTemplateInfo, TemplateCategory
                ... )
                >>> library = TemplateLibrary()
                >>> initial_count = len(library.list_templates())
                >>> templates = [
                ...     PromptTemplateInfo(
                ...         name=f"custom_{i}",
                ...         category=TemplateCategory.GENERATION,
                ...         description=f"Custom template {i}",
                ...         template=f"Template {i}: {{input}}",
                ...         variables=["input"],
                ...     )
                ...     for i in range(3)
                ... ]
                >>> for t in templates:
                ...     library.register(t)
                >>> print(len(library.list_templates()) == initial_count + 3)
                True
        """
        self._templates[template.name] = template

    def render(self, name: str, **kwargs) -> str:
        """Render a template by name with variable substitution.

        Retrieves a template by name and renders it with the provided variable
        values. This is a convenience method that combines get() and render()
        in a single call.

        Args:
            name: The name of the template to render. Must match a registered
                template name exactly (case-sensitive).
            **kwargs: Variable name-value pairs for substitution. Keys should
                match the variable names defined in the template. Values are
                converted to strings during substitution.

        Returns:
            The rendered template string with variables replaced by their values.

        Raises:
            KeyError: If no template with the given name exists in the library.

        Examples:
            Basic rendering:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> prompt = library.render(
                ...     "chain_of_thought",
                ...     question="What is the capital of France?"
                ... )
                >>> print("What is the capital of France?" in prompt)
                True

            Rendering with multiple variables:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> prompt = library.render(
                ...     "code_generation",
                ...     language="Python",
                ...     requirements="a function that calculates factorial"
                ... )
                >>> print("Python" in prompt)
                True
                >>> print("factorial" in prompt)
                True

            Handling missing templates:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> try:
                ...     library.render("nonexistent", var="value")
                ... except KeyError as e:
                ...     print("Template not found")
                Template not found

            Rendering classification template:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> prompt = library.render(
                ...     "binary_classification",
                ...     category_a="spam",
                ...     category_b="not spam",
                ...     text="Win a free iPhone now!"
                ... )
                >>> print("spam" in prompt and "not spam" in prompt)
                True
        """
        template = self._templates.get(name)
        if not template:
            raise KeyError(f"Template '{name}' not found")
        return template.render(**kwargs)

    def search(self, query: str) -> list[PromptTemplateInfo]:
        """Search templates by keyword across names, descriptions, and use cases.

        Performs a case-insensitive search across template names, descriptions,
        and best_for lists. This is useful for discovering templates when you
        know what you want to accomplish but not the exact template name.

        Args:
            query: The search keyword or phrase to look for. The search is
                case-insensitive and matches partial strings.

        Returns:
            List of PromptTemplateInfo instances that match the query. Returns
            an empty list if no matches are found.

        Examples:
            Searching by functionality:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> results = library.search("code")
                >>> print([t.name for t in results])
                ['code_generation', 'code_explanation', 'code_review', 'code_refactor']

            Searching by use case:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> results = library.search("sentiment")
                >>> print(len(results) >= 1)
                True
                >>> print(results[0].name)
                sentiment_analysis

            Case-insensitive search:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> lower = library.search("chain")
                >>> upper = library.search("CHAIN")
                >>> print(len(lower) == len(upper))
                True

            Searching for task types:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> results = library.search("classification")
                >>> names = [t.name for t in results]
                >>> print("binary_classification" in names)
                True
                >>> print("multi_class_classification" in names)
                True

            Handling no results:

                >>> from insideLLMs.contrib.templates import TemplateLibrary
                >>> library = TemplateLibrary()
                >>> results = library.search("xyznonexistent")
                >>> print(len(results) == 0)
                True
        """
        query = query.lower()
        results = []
        for template in self._templates.values():
            if (
                query in template.name.lower()
                or query in template.description.lower()
                or any(query in use.lower() for use in template.best_for)
            ):
                results.append(template)
        return results


# Default library instance
_default_library = TemplateLibrary()


def get_template(name: str) -> Optional[PromptTemplateInfo]:
    """Get a template by name from the default library.

    This is a convenience function that retrieves a template from the global
    default library instance. Use this when you need quick access to built-in
    or registered templates without managing your own library instance.

    Args:
        name: The unique name of the template to retrieve (e.g., "chain_of_thought",
            "code_review"). Names are case-sensitive.

    Returns:
        The PromptTemplateInfo instance if found, or None if no template
        with the given name exists.

    Examples:
        Getting a built-in template:

            >>> from insideLLMs.contrib.templates import get_template
            >>> template = get_template("chain_of_thought")
            >>> print(template.name)
            chain_of_thought
            >>> print(template.description)
            Encourages step-by-step reasoning

        Checking if template exists:

            >>> from insideLLMs.contrib.templates import get_template
            >>> template = get_template("nonexistent")
            >>> print(template is None)
            True

        Using the template:

            >>> from insideLLMs.contrib.templates import get_template
            >>> template = get_template("binary_classification")
            >>> prompt = template.render(
            ...     category_a="positive",
            ...     category_b="negative",
            ...     text="I love this!"
            ... )
            >>> print("positive" in prompt)
            True

        Accessing template metadata:

            >>> from insideLLMs.contrib.templates import get_template
            >>> template = get_template("code_generation")
            >>> print(template.variables)
            ['language', 'requirements']
            >>> print(template.category.value)
            coding
    """
    return _default_library.get(name)


def list_templates(category: Optional[TemplateCategory] = None) -> list[PromptTemplateInfo]:
    """List all templates from the default library, optionally filtered by category.

    This is a convenience function that lists templates from the global default
    library instance. Use this to discover available templates or to filter
    templates by category.

    Args:
        category: Optional TemplateCategory to filter by. If None, all templates
            are returned. If specified, only templates with matching category
            are returned.

    Returns:
        List of PromptTemplateInfo instances. Returns all templates if no
        category filter is specified.

    Examples:
        Listing all templates:

            >>> from insideLLMs.contrib.templates import list_templates
            >>> templates = list_templates()
            >>> print(len(templates) >= 20)
            True

        Filtering by category:

            >>> from insideLLMs.contrib.templates import list_templates, TemplateCategory
            >>> reasoning = list_templates(TemplateCategory.REASONING)
            >>> print([t.name for t in reasoning])
            ['chain_of_thought', 'chain_of_thought_zero_shot', 'step_back']

        Iterating over templates:

            >>> from insideLLMs.contrib.templates import list_templates, TemplateCategory
            >>> for template in list_templates(TemplateCategory.EXTRACTION):
            ...     print(f"{template.name}: {len(template.variables)} vars")
            entity_extraction: 2 vars
            structured_extraction: 2 vars
            key_value_extraction: 1 vars

        Counting templates by category:

            >>> from insideLLMs.contrib.templates import list_templates, TemplateCategory
            >>> for cat in [TemplateCategory.CODING, TemplateCategory.ANALYSIS]:
            ...     count = len(list_templates(cat))
            ...     print(f"{cat.value}: {count} templates")
            coding: 4 templates
            analysis: 3 templates
    """
    return _default_library.list_templates(category)


def render_template(name: str, **kwargs) -> str:
    """Render a template from the default library with variable substitution.

    This is a convenience function that retrieves and renders a template from
    the global default library instance in a single call. This is the most
    common way to use templates.

    Args:
        name: The name of the template to render. Must match a registered
            template name exactly (case-sensitive).
        **kwargs: Variable name-value pairs for substitution. Keys should
            match the variable names defined in the template.

    Returns:
        The rendered template string with variables replaced by their values.

    Raises:
        KeyError: If no template with the given name exists in the library.

    Examples:
        Rendering a chain-of-thought prompt:

            >>> from insideLLMs.contrib.templates import render_template
            >>> prompt = render_template(
            ...     "chain_of_thought",
            ...     question="What is 15% of 80?"
            ... )
            >>> print("15% of 80" in prompt)
            True

        Rendering a code generation prompt:

            >>> from insideLLMs.contrib.templates import render_template
            >>> prompt = render_template(
            ...     "code_generation",
            ...     language="Python",
            ...     requirements="a function to reverse a string"
            ... )
            >>> print("Python" in prompt)
            True

        Rendering a classification prompt:

            >>> from insideLLMs.contrib.templates import render_template
            >>> prompt = render_template(
            ...     "multi_class_classification",
            ...     categories="positive, negative, neutral",
            ...     text="The product works as expected."
            ... )
            >>> print("positive, negative, neutral" in prompt)
            True

        Error handling for missing templates:

            >>> from insideLLMs.contrib.templates import render_template
            >>> try:
            ...     render_template("nonexistent", var="value")
            ... except KeyError:
            ...     print("Template not found")
            Template not found
    """
    return _default_library.render(name, **kwargs)


def register_template(template: PromptTemplateInfo) -> None:
    """Register a custom template to the default library.

    This is a convenience function that registers a template to the global
    default library instance. Use this to add custom templates that will be
    available throughout your application via the module-level functions.

    Note: Templates registered here will be available globally. If you need
    isolated template registries, create your own TemplateLibrary instance.

    Args:
        template: The PromptTemplateInfo instance to register. The template's
            name attribute will be used as the key for retrieval.

    Examples:
        Registering a simple template:

            >>> from insideLLMs.contrib.templates import (
            ...     register_template, get_template,
            ...     PromptTemplateInfo, TemplateCategory
            ... )
            >>> template = PromptTemplateInfo(
            ...     name="my_greeting",
            ...     category=TemplateCategory.GENERATION,
            ...     description="Generate a personalized greeting",
            ...     template="Hello {name}! Welcome to {service}.",
            ...     variables=["name", "service"],
            ... )
            >>> register_template(template)
            >>> retrieved = get_template("my_greeting")
            >>> print(retrieved is not None)
            True

        Registering with full metadata:

            >>> from insideLLMs.contrib.templates import (
            ...     register_template, render_template,
            ...     PromptTemplateInfo, TemplateCategory
            ... )
            >>> template = PromptTemplateInfo(
            ...     name="api_doc",
            ...     category=TemplateCategory.CODING,
            ...     description="Generate API documentation",
            ...     template="Document this {language} API:\\n{code}",
            ...     variables=["language", "code"],
            ...     best_for=["API docs", "code documentation"],
            ...     tips=["Include examples in the code"],
            ... )
            >>> register_template(template)
            >>> prompt = render_template("api_doc", language="Python", code="def foo(): pass")
            >>> print("Python" in prompt)
            True

        Overwriting an existing template:

            >>> from insideLLMs.contrib.templates import (
            ...     register_template, get_template,
            ...     PromptTemplateInfo, TemplateCategory
            ... )
            >>> custom = PromptTemplateInfo(
            ...     name="chain_of_thought",  # Overwrites built-in
            ...     category=TemplateCategory.REASONING,
            ...     description="My custom CoT",
            ...     template="Think carefully: {question}",
            ...     variables=["question"],
            ... )
            >>> register_template(custom)
            >>> print(get_template("chain_of_thought").description)
            My custom CoT

        Registering domain-specific templates:

            >>> from insideLLMs.contrib.templates import (
            ...     register_template, search_templates,
            ...     PromptTemplateInfo, TemplateCategory
            ... )
            >>> medical_template = PromptTemplateInfo(
            ...     name="medical_summary",
            ...     category=TemplateCategory.SUMMARIZATION,
            ...     description="Summarize medical documents",
            ...     template="Summarize this medical report:\\n{report}",
            ...     variables=["report"],
            ...     best_for=["medical summaries", "clinical notes"],
            ... )
            >>> register_template(medical_template)
            >>> results = search_templates("medical")
            >>> print(len(results) >= 1)
            True
    """
    _default_library.register(template)


def search_templates(query: str) -> list[PromptTemplateInfo]:
    """Search templates in the default library by keyword.

    This is a convenience function that searches templates in the global
    default library instance. The search is case-insensitive and matches
    against template names, descriptions, and best_for lists.

    Args:
        query: The search keyword or phrase to look for. The search is
            case-insensitive and matches partial strings.

    Returns:
        List of PromptTemplateInfo instances that match the query. Returns
        an empty list if no matches are found.

    Examples:
        Searching for code-related templates:

            >>> from insideLLMs.contrib.templates import search_templates
            >>> results = search_templates("code")
            >>> print([t.name for t in results])
            ['code_generation', 'code_explanation', 'code_review', 'code_refactor']

        Searching for summarization templates:

            >>> from insideLLMs.contrib.templates import search_templates
            >>> results = search_templates("summary")
            >>> names = [t.name for t in results]
            >>> print("concise_summary" in names)
            True
            >>> print("bullet_summary" in names)
            True

        Case-insensitive search:

            >>> from insideLLMs.contrib.templates import search_templates
            >>> lower = search_templates("reasoning")
            >>> upper = search_templates("REASONING")
            >>> mixed = search_templates("ReAsOnInG")
            >>> print(len(lower) == len(upper) == len(mixed))
            True

        Searching by use case:

            >>> from insideLLMs.contrib.templates import search_templates
            >>> results = search_templates("spam")
            >>> # Finds binary_classification which has 'spam detection' in best_for
            >>> print(any("classification" in t.name for t in results))
            True

        Handling no results:

            >>> from insideLLMs.contrib.templates import search_templates
            >>> results = search_templates("xyznonexistent123")
            >>> print(results)
            []
    """
    return _default_library.search(query)


def get_library() -> TemplateLibrary:
    """Get the default template library instance.

    Returns the global TemplateLibrary instance that is used by all the
    module-level convenience functions. This is useful when you need direct
    access to the library object for advanced operations or to share it
    with other components.

    Returns:
        The default TemplateLibrary instance used by this module. This is
        the same instance used by get_template(), list_templates(),
        render_template(), register_template(), and search_templates().

    Examples:
        Getting the library instance:

            >>> from insideLLMs.contrib.templates import get_library
            >>> library = get_library()
            >>> print(type(library).__name__)
            TemplateLibrary

        Using library methods directly:

            >>> from insideLLMs.contrib.templates import get_library
            >>> library = get_library()
            >>> template = library.get("chain_of_thought")
            >>> print(template.name)
            chain_of_thought

        Passing library to other functions:

            >>> from insideLLMs.contrib.templates import get_library
            >>> library = get_library()
            >>> def count_templates(lib):
            ...     return len(lib.list_templates())
            >>> print(count_templates(library) >= 20)
            True

        Using list_names method (not exposed at module level):

            >>> from insideLLMs.contrib.templates import get_library, TemplateCategory
            >>> library = get_library()
            >>> names = library.list_names(TemplateCategory.CODING)
            >>> print(names)
            ['code_generation', 'code_explanation', 'code_review', 'code_refactor']

        Verifying it is the same instance:

            >>> from insideLLMs.contrib.templates import get_library, register_template
            >>> from insideLLMs.contrib.templates import PromptTemplateInfo, TemplateCategory
            >>> library = get_library()
            >>> custom = PromptTemplateInfo(
            ...     name="verify_same_instance",
            ...     category=TemplateCategory.GENERATION,
            ...     description="Test",
            ...     template="Test",
            ...     variables=[],
            ... )
            >>> register_template(custom)  # Uses default library
            >>> print(library.get("verify_same_instance") is not None)
            True
    """
    return _default_library
