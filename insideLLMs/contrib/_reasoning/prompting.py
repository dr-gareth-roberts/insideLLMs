"""Chain-of-thought prompt generation."""

from typing import Optional


class CoTPromptGenerator:
    """
    Generates Chain-of-Thought prompts in various styles.

    Provides templates and utilities for creating prompts that encourage
    step-by-step reasoning in language model responses.

    Attributes
    ----------
    TEMPLATES : dict[str, str]
        Built-in prompt templates with {question} placeholder:
        - standard: Simple "Let's think step by step" format
        - structured: Numbered step format
        - detailed: Comprehensive breakdown format
        - math: Mathematical problem format with "Given:"
        - logical: Formal logic format with premises

    Examples
    --------
    Generating prompts with different styles:

        >>> from insideLLMs.contrib.reasoning import CoTPromptGenerator
        >>> generator = CoTPromptGenerator()
        >>> prompt = generator.generate("What is 5+7?", style="standard")
        >>> print(prompt)
        Let's think step by step.
        <BLANKLINE>
        What is 5+7?

    Using math style:

        >>> prompt = generator.generate("Calculate the area of a 5x3 rectangle", style="math")
        >>> print(prompt[:50])
        Let's solve this mathematical problem step by ste

    Generating variations:

        >>> variations = generator.generate_variations("Solve: 2x + 5 = 11", num_variations=3)
        >>> print(f"Generated {len(variations)} variations")
        Generated 3 variations

    Adding custom templates:

        >>> generator.add_template("science", "Scientific analysis:\\n\\n{question}\\n\\nHypothesis:")
        >>> prompt = generator.generate("Why is the sky blue?", style="science")
        >>> print(prompt[:30])
        Scientific analysis:

    See Also
    --------
    generate_cot_prompt : Convenience function
    CoTEvaluator : For evaluating CoT responses
    """

    TEMPLATES = {
        "standard": "Let's think step by step.\n\n{question}",
        "structured": "Please solve this problem step by step:\n\n{question}\n\nStep 1:",
        "detailed": "I need to solve this problem. Let me break it down:\n\n{question}\n\nFirst, I'll identify what we know:\n",
        "math": "Let's solve this mathematical problem step by step:\n\n{question}\n\nGiven:\n",
        "logical": "Let me reason through this logically:\n\n{question}\n\nPremise 1:",
    }

    def generate(
        self,
        question: str,
        style: str = "standard",
        custom_template: Optional[str] = None,
    ) -> str:
        """
        Generate a Chain-of-Thought prompt.

        Creates a prompt that encourages step-by-step reasoning by wrapping
        the question in an appropriate template.

        Parameters
        ----------
        question : str
            The question or problem to wrap in a CoT prompt.
        style : str
            The template style to use. Options: "standard", "structured",
            "detailed", "math", "logical". Defaults to "standard".
        custom_template : Optional[str]
            A custom template string with {question} placeholder.
            If provided, overrides the style parameter.

        Returns
        -------
        str
            The formatted CoT prompt.

        Examples
        --------
        Standard style:

            >>> from insideLLMs.contrib.reasoning import CoTPromptGenerator
            >>> generator = CoTPromptGenerator()
            >>> prompt = generator.generate("What is 2+2?")
            >>> print(prompt)
            Let's think step by step.
            <BLANKLINE>
            What is 2+2?

        Structured style:

            >>> prompt = generator.generate("Solve for x: 3x = 9", style="structured")
            >>> "Step 1:" in prompt
            True

        Custom template:

            >>> prompt = generator.generate(
            ...     "Why do birds fly?",
            ...     custom_template="Analyze this: {question}\\n\\nReason:"
            ... )
            >>> print(prompt)
            Analyze this: Why do birds fly?
            <BLANKLINE>
            Reason:

        Unknown style falls back to standard:

            >>> prompt = generator.generate("Question", style="unknown")
            >>> "step by step" in prompt
            True
        """
        template = custom_template or self.TEMPLATES.get(style, self.TEMPLATES["standard"])

        return template.format(question=question)

    def generate_variations(
        self,
        question: str,
        num_variations: int = 3,
    ) -> list[str]:
        """
        Generate multiple CoT prompt variations for a question.

        Creates prompts using different template styles to explore
        how different prompting approaches affect model responses.

        Parameters
        ----------
        question : str
            The question to generate variations for.
        num_variations : int
            Maximum number of variations to generate. Capped at the
            number of available templates. Defaults to 3.

        Returns
        -------
        list[str]
            List of generated prompts, each using a different style.

        Examples
        --------
        Generate default variations:

            >>> from insideLLMs.contrib.reasoning import CoTPromptGenerator
            >>> generator = CoTPromptGenerator()
            >>> variations = generator.generate_variations("What is 10/2?")
            >>> len(variations)
            3

        Generate all variations:

            >>> variations = generator.generate_variations("Problem", num_variations=10)
            >>> len(variations)  # Capped at number of templates
            5

        Check variation uniqueness:

            >>> variations = generator.generate_variations("Test question")
            >>> len(set(variations)) == len(variations)
            True
        """
        variations = []
        styles = list(self.TEMPLATES.keys())

        for i in range(min(num_variations, len(styles))):
            prompt = self.generate(question, style=styles[i])
            variations.append(prompt)

        return variations

    def add_template(self, name: str, template: str) -> None:
        """
        Add a custom template to the generator.

        Registers a new template that can be used with the generate method.
        Template must contain {question} placeholder.

        Parameters
        ----------
        name : str
            Name for the template, used as the style parameter.
        template : str
            Template string with {question} placeholder.

        Examples
        --------
        Adding a simple template:

            >>> from insideLLMs.contrib.reasoning import CoTPromptGenerator
            >>> generator = CoTPromptGenerator()
            >>> generator.add_template("brief", "Think: {question}")
            >>> prompt = generator.generate("2+2?", style="brief")
            >>> print(prompt)
            Think: 2+2?

        Adding a multi-line template:

            >>> template = '''Analyze this problem carefully:
            ...
            ... {question}
            ...
            ... Step-by-step solution:'''
            >>> generator.add_template("careful", template)
            >>> "carefully" in generator.generate("X", style="careful")
            True

        Overwriting existing template:

            >>> generator.add_template("standard", "New standard: {question}")
            >>> "New standard" in generator.generate("Q", style="standard")
            True
        """
        self.TEMPLATES[name] = template
