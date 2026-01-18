"""
Prompt templates library for common LLM interaction patterns.

Provides:
- Pre-built templates for common use cases
- Template categories (reasoning, extraction, classification, etc.)
- Template variants and customization
- Best practices built-in
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TemplateCategory(Enum):
    """Categories of prompt templates."""

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
    """Information about a prompt template."""

    name: str
    category: TemplateCategory
    description: str
    template: str
    variables: list[str]
    examples: list[dict[str, str]] = field(default_factory=list)
    best_for: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)

    def render(self, **kwargs) -> str:
        """Render template with variables.

        Args:
            **kwargs: Variable values.

        Returns:
            Rendered template.
        """
        result = self.template
        for var in self.variables:
            if var in kwargs:
                result = result.replace(f"{{{var}}}", str(kwargs[var]))
        return result

    def get_missing_vars(self, provided: set[str]) -> set[str]:
        """Get missing required variables.

        Args:
            provided: Set of provided variable names.

        Returns:
            Set of missing variable names.
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
    """Library of prompt templates."""

    def __init__(self):
        """Initialize with built-in templates."""
        self._templates: dict[str, PromptTemplateInfo] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in templates."""
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
        """Get template by name.

        Args:
            name: Template name.

        Returns:
            Template or None if not found.
        """
        return self._templates.get(name)

    def list_templates(
        self, category: Optional[TemplateCategory] = None
    ) -> list[PromptTemplateInfo]:
        """List available templates.

        Args:
            category: Optional category filter.

        Returns:
            List of templates.
        """
        templates = list(self._templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates

    def list_names(self, category: Optional[TemplateCategory] = None) -> list[str]:
        """List template names.

        Args:
            category: Optional category filter.

        Returns:
            List of template names.
        """
        return [t.name for t in self.list_templates(category)]

    def register(self, template: PromptTemplateInfo) -> None:
        """Register a custom template.

        Args:
            template: Template to register.
        """
        self._templates[template.name] = template

    def render(self, name: str, **kwargs) -> str:
        """Render a template by name.

        Args:
            name: Template name.
            **kwargs: Template variables.

        Returns:
            Rendered template.

        Raises:
            KeyError: If template not found.
        """
        template = self._templates.get(name)
        if not template:
            raise KeyError(f"Template '{name}' not found")
        return template.render(**kwargs)

    def search(self, query: str) -> list[PromptTemplateInfo]:
        """Search templates by keyword.

        Args:
            query: Search query.

        Returns:
            Matching templates.
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
    """Get template from default library.

    Args:
        name: Template name.

    Returns:
        Template or None.
    """
    return _default_library.get(name)


def list_templates(category: Optional[TemplateCategory] = None) -> list[PromptTemplateInfo]:
    """List templates from default library.

    Args:
        category: Optional category filter.

    Returns:
        List of templates.
    """
    return _default_library.list_templates(category)


def render_template(name: str, **kwargs) -> str:
    """Render template from default library.

    Args:
        name: Template name.
        **kwargs: Template variables.

    Returns:
        Rendered template.
    """
    return _default_library.render(name, **kwargs)


def register_template(template: PromptTemplateInfo) -> None:
    """Register template to default library.

    Args:
        template: Template to register.
    """
    _default_library.register(template)


def search_templates(query: str) -> list[PromptTemplateInfo]:
    """Search templates in default library.

    Args:
        query: Search query.

    Returns:
        Matching templates.
    """
    return _default_library.search(query)


def get_library() -> TemplateLibrary:
    """Get the default template library.

    Returns:
        Default template library.
    """
    return _default_library
