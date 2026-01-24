"""Instruction following probes for evaluating LLM instruction compliance.

This module provides probes to systematically test and evaluate a language model's
ability to follow instructions precisely. It includes three main probe classes:

1. **InstructionFollowingProbe**: Tests general instruction following with various
   constraints like format requirements, word limits, and keyword inclusion/exclusion.

2. **MultiStepTaskProbe**: Evaluates the model's ability to complete multi-step tasks
   while maintaining context and producing coherent results across all steps.

3. **ConstraintComplianceProbe**: Focuses on specific constraint types such as
   character limits, word limits, and custom validation rules.

Key capabilities tested:
    - Format compliance (JSON, lists, specific structures)
    - Length constraints (word counts, character limits)
    - Content restrictions (required/forbidden keywords)
    - Multi-step task completion and context maintenance
    - Custom validation through user-defined functions

Module-level Examples
---------------------
Basic instruction following test:

    >>> from insideLLMs.probes.instruction import InstructionFollowingProbe
    >>> probe = InstructionFollowingProbe()
    >>> instructions = {
    ...     "task": "List 3 programming languages",
    ...     "constraints": {
    ...         "format": "numbered_list",
    ...         "max_items": 3,
    ...         "include_keywords": ["Python"]
    ...     }
    ... }
    >>> result = probe.run(model, instructions)
    >>> evaluation = probe.evaluate_single(result, instructions)
    >>> print(evaluation.metadata["score"])

Multi-step task evaluation:

    >>> from insideLLMs.probes.instruction import MultiStepTaskProbe
    >>> probe = MultiStepTaskProbe()
    >>> task = {
    ...     "steps": [
    ...         "Define machine learning",
    ...         "List 3 types of ML algorithms",
    ...         "Explain when to use each type"
    ...     ],
    ...     "expected": {
    ...         "step_1": ["machine learning", "data"],
    ...         "step_2": ["supervised", "unsupervised"],
    ...         "step_3": ["classification", "clustering"]
    ...     }
    ... }
    >>> result = probe.run(model, task)

Strict constraint checking:

    >>> from insideLLMs.probes.instruction import ConstraintComplianceProbe
    >>> probe = ConstraintComplianceProbe(
    ...     constraint_type="word_limit",
    ...     limit=50
    ... )
    >>> result = probe.run(model, "Explain quantum computing briefly")
    >>> evaluation = probe.evaluate_single(result, reference=50)
    >>> print(f"Word count: {evaluation.metadata['word_count']}")

Custom validation:

    >>> def contains_code(output: str) -> bool:
    ...     return "def " in output or "function " in output
    >>> probe = ConstraintComplianceProbe(
    ...     constraint_type="custom",
    ...     custom_constraint="Include a code example",
    ...     validator=contains_code
    ... )
    >>> result = probe.run(model, "Show how to sort a list in Python")

See Also
--------
insideLLMs.probes.base.ScoredProbe : Base class for all scored probes.
insideLLMs.types.ProbeResult : Result container for probe evaluations.
"""

import re
from typing import Any, Callable, Optional

from insideLLMs.probes.base import ScoredProbe
from insideLLMs.types import ProbeCategory, ProbeResult, ResultStatus


class InstructionFollowingProbe(ScoredProbe[str]):
    """Probe to test LLMs' ability to follow instructions precisely.

    This probe evaluates how well a language model can adhere to explicit
    instructions and constraints. It supports a wide variety of constraint
    types and provides detailed scoring on compliance.

    Tests various aspects of instruction following:
        - Format compliance (JSON, numbered lists, bullet lists, code, etc.)
        - Length constraints (word counts, item counts)
        - Content restrictions (required keywords, forbidden words)
        - Tone and language requirements

    Attributes
    ----------
    strict_mode : bool
        If True, any single constraint violation results in a score of 0.0.
        If False, the score is averaged across all constraint checks.
    default_category : ProbeCategory
        The default category for this probe (ProbeCategory.CUSTOM).

    Supported Constraints
    ---------------------
    The following constraint keys are supported in the constraints dict:

    - ``format``: Expected output format. Supported values:
        - "json": Valid JSON output
        - "numbered_list": Numbered list (1. 2. 3.)
        - "bullet_list": Bullet points (-, *, or bullet)
        - "single_word": Single word response
        - "single_sentence": Exactly one sentence
        - "paragraph": Paragraph form
        - "code": Code output (detected via patterns)

    - ``max_words``: Maximum word count allowed
    - ``min_words``: Minimum word count required
    - ``max_items``: Maximum list items allowed
    - ``min_items``: Minimum list items required
    - ``include_keywords``: List of keywords that must appear
    - ``exclude_keywords``: List of keywords that must not appear
    - ``language``: Required response language
    - ``tone``: Required tone (e.g., "formal", "casual")

    Examples
    --------
    Example 1: Basic format compliance test

        >>> from insideLLMs.probes.instruction import InstructionFollowingProbe
        >>> probe = InstructionFollowingProbe()
        >>> instructions = {
        ...     "task": "List 3 fruits",
        ...     "constraints": {"format": "numbered_list", "max_items": 3}
        ... }
        >>> result = probe.run(model, instructions)
        >>> # Model might respond: "1. Apple\\n2. Banana\\n3. Orange"
        >>> evaluation = probe.evaluate_single(result, instructions)
        >>> print(evaluation.metadata["format_compliance"])  # 1.0 if correct

    Example 2: JSON output with keyword requirements

        >>> probe = InstructionFollowingProbe(strict_mode=True)
        >>> instructions = {
        ...     "task": "Describe a car as JSON",
        ...     "constraints": {
        ...         "format": "json",
        ...         "include_keywords": ["make", "model", "year"]
        ...     }
        ... }
        >>> result = probe.run(model, instructions)
        >>> # Model should respond with valid JSON containing required fields
        >>> evaluation = probe.evaluate_single(result, instructions)
        >>> if evaluation.metadata["score"] < 1.0:
        ...     print("Some constraints were violated")

    Example 3: Word count constraints

        >>> probe = InstructionFollowingProbe()
        >>> instructions = {
        ...     "task": "Explain photosynthesis",
        ...     "constraints": {
        ...         "min_words": 50,
        ...         "max_words": 100,
        ...         "include_keywords": ["sunlight", "chlorophyll", "glucose"]
        ...     }
        ... }
        >>> result = probe.run(model, instructions)
        >>> evaluation = probe.evaluate_single(result, instructions)
        >>> print(f"Word count: {evaluation.metadata['word_count']}")
        >>> print(f"Within limits: {evaluation.metadata.get('within_max_words', True)}")

    Example 4: Strict mode vs. normal mode

        >>> # Normal mode: scores are averaged
        >>> normal_probe = InstructionFollowingProbe(strict_mode=False)
        >>> # Strict mode: any violation = 0.0
        >>> strict_probe = InstructionFollowingProbe(strict_mode=True)
        >>>
        >>> instructions = {
        ...     "task": "List colors",
        ...     "constraints": {"format": "bullet_list", "max_items": 3}
        ... }
        >>> # If model lists 4 items but uses bullet format:
        >>> # - Normal mode: score might be 0.5 (format ok, items exceeded)
        >>> # - Strict mode: score = 0.0 (any failure = total failure)

    See Also
    --------
    MultiStepTaskProbe : For testing multi-step task completion.
    ConstraintComplianceProbe : For focused single-constraint testing.
    """

    default_category = ProbeCategory.CUSTOM

    def __init__(
        self,
        name: str = "InstructionFollowingProbe",
        strict_mode: bool = False,
    ):
        """Initialize the instruction following probe.

        Creates a new InstructionFollowingProbe instance configured for testing
        instruction compliance with optional strict evaluation mode.

        Parameters
        ----------
        name : str, optional
            Name identifier for this probe instance. Useful when running multiple
            probes and needing to distinguish results. Default is "InstructionFollowingProbe".
        strict_mode : bool, optional
            Controls how constraint violations affect the overall score.
            - If False (default): The score is the average of all constraint checks,
              allowing partial compliance to be recognized.
            - If True: Any single constraint violation results in an overall score
              of 0.0, requiring perfect compliance to pass.

        Examples
        --------
        Example 1: Default initialization

            >>> probe = InstructionFollowingProbe()
            >>> print(probe.name)
            'InstructionFollowingProbe'
            >>> print(probe.strict_mode)
            False

        Example 2: Named probe for batch testing

            >>> json_probe = InstructionFollowingProbe(
            ...     name="JSONFormatProbe",
            ...     strict_mode=False
            ... )
            >>> list_probe = InstructionFollowingProbe(
            ...     name="ListFormatProbe",
            ...     strict_mode=True
            ... )
            >>> # Use different probes for different test categories

        Example 3: Strict mode for production validation

            >>> strict_probe = InstructionFollowingProbe(strict_mode=True)
            >>> # Any constraint violation will result in score = 0.0
            >>> # Useful for production systems requiring exact compliance

        Example 4: Lenient mode for development/exploration

            >>> lenient_probe = InstructionFollowingProbe(strict_mode=False)
            >>> # Partial compliance is scored proportionally
            >>> # Useful for understanding model capabilities during development
        """
        super().__init__(name=name, category=ProbeCategory.CUSTOM)
        self.strict_mode = strict_mode

    def run(self, model: Any, data: Any, **kwargs: Any) -> str:
        """Run the instruction following probe against a model.

        Executes the probe by sending the task and constraints to the model
        and collecting its response. The constraints are formatted as explicit
        instructions appended to the task prompt.

        Parameters
        ----------
        model : Any
            The language model to test. Must have a ``generate(prompt, **kwargs)``
            method that accepts a string prompt and returns a string response.
        data : dict or str
            The instruction data. Can be:
            - A dict with keys:
                - "task" or "instruction": The main task description (str)
                - "constraints": Optional dict of constraint specifications
            - A plain string (treated as the task with no constraints)
        **kwargs : Any
            Additional keyword arguments passed directly to ``model.generate()``.
            Common options include ``temperature``, ``max_tokens``, etc.

        Returns
        -------
        str
            The model's generated response to the instruction.

        Examples
        --------
        Example 1: Simple task with format constraint

            >>> probe = InstructionFollowingProbe()
            >>> data = {
            ...     "task": "List the primary colors",
            ...     "constraints": {"format": "bullet_list"}
            ... }
            >>> response = probe.run(model, data)
            >>> print(response)
            - Red
            - Blue
            - Yellow

        Example 2: Task with multiple constraints

            >>> probe = InstructionFollowingProbe()
            >>> data = {
            ...     "task": "Describe the water cycle",
            ...     "constraints": {
            ...         "max_words": 50,
            ...         "include_keywords": ["evaporation", "condensation", "precipitation"],
            ...         "format": "paragraph"
            ...     }
            ... }
            >>> response = probe.run(model, data, temperature=0.7)

        Example 3: Plain string task (no constraints)

            >>> probe = InstructionFollowingProbe()
            >>> response = probe.run(model, "What is the capital of France?")
            >>> # No constraints applied, just the raw task

        Example 4: Using 'instruction' key instead of 'task'

            >>> probe = InstructionFollowingProbe()
            >>> data = {
            ...     "instruction": "Convert 100 Celsius to Fahrenheit",
            ...     "constraints": {"format": "single_sentence"}
            ... }
            >>> response = probe.run(model, data)

        Notes
        -----
        The method automatically formats constraints into explicit instructions
        that are appended to the task prompt. This helps ensure the model
        understands what is expected of it.
        """
        if isinstance(data, dict):
            task = data.get("task", data.get("instruction", ""))
            constraints = data.get("constraints", {})
        else:
            task = str(data)
            constraints = {}

        # Build prompt with explicit constraints
        prompt_parts = [task]

        if constraints:
            constraint_text = self._format_constraints(constraints)
            if constraint_text:
                prompt_parts.append(f"\n\nConstraints:\n{constraint_text}")

        prompt = "".join(prompt_parts)
        return model.generate(prompt, **kwargs)

    def _format_constraints(self, constraints: dict[str, Any]) -> str:
        """Format constraints dict as explicit human-readable instructions.

        Converts a constraints dictionary into a formatted string that can be
        appended to the task prompt. Each constraint is converted into a clear,
        actionable instruction for the model.

        Parameters
        ----------
        constraints : dict[str, Any]
            Dictionary of constraint specifications. Supported keys:
            - "format": Output format type (str)
            - "max_words": Maximum word count (int)
            - "min_words": Minimum word count (int)
            - "max_items": Maximum list items (int)
            - "min_items": Minimum list items (int)
            - "include_keywords": Required keywords (list[str])
            - "exclude_keywords": Forbidden keywords (list[str])
            - "language": Response language (str)
            - "tone": Required tone (str)

        Returns
        -------
        str
            Formatted constraint string with each constraint on its own line,
            prefixed with a dash for readability. Returns empty string if no
            recognized constraints are present.

        Examples
        --------
        Example 1: Format constraint only

            >>> probe = InstructionFollowingProbe()
            >>> result = probe._format_constraints({"format": "json"})
            >>> print(result)
            - Respond ONLY with valid JSON

        Example 2: Multiple constraints

            >>> probe = InstructionFollowingProbe()
            >>> constraints = {
            ...     "format": "numbered_list",
            ...     "max_items": 5,
            ...     "include_keywords": ["python", "java"]
            ... }
            >>> result = probe._format_constraints(constraints)
            >>> print(result)
            - Use a numbered list (1. 2. 3. etc.)
            - Include no more than 5 items
            - Must include these words: python, java

        Example 3: Word count constraints with tone

            >>> probe = InstructionFollowingProbe()
            >>> constraints = {
            ...     "min_words": 50,
            ...     "max_words": 100,
            ...     "tone": "professional",
            ...     "language": "English"
            ... }
            >>> result = probe._format_constraints(constraints)
            >>> print(result)
            - Use no more than 100 words
            - Use at least 50 words
            - Respond in English
            - Use a professional tone

        Example 4: Custom format type

            >>> probe = InstructionFollowingProbe()
            >>> result = probe._format_constraints({"format": "haiku"})
            >>> print(result)
            - Use haiku format

        Notes
        -----
        Unknown format types are handled gracefully by generating a generic
        "Use {format} format" instruction.
        """
        parts = []

        if "format" in constraints:
            fmt = constraints["format"]
            format_instructions = {
                "json": "- Respond ONLY with valid JSON",
                "numbered_list": "- Use a numbered list (1. 2. 3. etc.)",
                "bullet_list": "- Use bullet points (- or * for each item)",
                "single_word": "- Respond with a single word only",
                "single_sentence": "- Respond with exactly one sentence",
                "paragraph": "- Respond in paragraph form",
                "code": "- Respond with code only (no explanations)",
            }
            if fmt in format_instructions:
                parts.append(format_instructions[fmt])
            else:
                parts.append(f"- Use {fmt} format")

        if "max_words" in constraints:
            parts.append(f"- Use no more than {constraints['max_words']} words")

        if "min_words" in constraints:
            parts.append(f"- Use at least {constraints['min_words']} words")

        if "max_items" in constraints:
            parts.append(f"- Include no more than {constraints['max_items']} items")

        if "min_items" in constraints:
            parts.append(f"- Include at least {constraints['min_items']} items")

        if "include_keywords" in constraints:
            keywords = constraints["include_keywords"]
            parts.append(f"- Must include these words: {', '.join(keywords)}")

        if "exclude_keywords" in constraints:
            keywords = constraints["exclude_keywords"]
            parts.append(f"- Must NOT include these words: {', '.join(keywords)}")

        if "language" in constraints:
            parts.append(f"- Respond in {constraints['language']}")

        if "tone" in constraints:
            parts.append(f"- Use a {constraints['tone']} tone")

        return "\n".join(parts)

    def evaluate_single(
        self,
        model_output: str,
        reference: Any,
        **kwargs: Any,
    ) -> ProbeResult[str]:
        """Evaluate how well the model followed the given instructions.

        Analyzes the model's output against the specified constraints and
        produces a detailed compliance report with scores for each constraint
        type that was checked.

        Parameters
        ----------
        model_output : str
            The model's generated response to evaluate.
        reference : dict or Any
            The reference constraints to evaluate against. Can be:
            - A dict with a "constraints" key containing the constraint dict
            - A dict that is itself the constraints
            - Any other value (treated as no constraints)
        **kwargs : Any
            Additional parameters (currently unused, reserved for future use).

        Returns
        -------
        ProbeResult[str]
            A ProbeResult containing:
            - input: Truncated string representation of constraints
            - output: The model's response
            - status: ResultStatus.SUCCESS if score >= 0.7, else ERROR
            - metadata: Dict with detailed compliance information including:
                - score: Overall compliance score (0.0 to 1.0)
                - label: "compliant" or "non_compliant"
                - format_compliance: Score for format check (if applicable)
                - word_count: Number of words in output
                - within_max_words/within_min_words: Boolean flags
                - item_count: Number of list items detected
                - within_max_items/within_min_items: Boolean flags
                - included_keywords: List of required keywords found
                - violated_exclusions: List of forbidden keywords found

        Examples
        --------
        Example 1: Evaluate numbered list compliance

            >>> probe = InstructionFollowingProbe()
            >>> model_output = "1. Apple\\n2. Banana\\n3. Orange"
            >>> reference = {
            ...     "constraints": {"format": "numbered_list", "max_items": 3}
            ... }
            >>> result = probe.evaluate_single(model_output, reference)
            >>> print(result.metadata["score"])  # 1.0 (fully compliant)
            >>> print(result.metadata["format_compliance"])  # 1.0
            >>> print(result.status)  # ResultStatus.SUCCESS

        Example 2: Evaluate with keyword requirements

            >>> probe = InstructionFollowingProbe()
            >>> model_output = "Python is a versatile language for data science."
            >>> reference = {
            ...     "constraints": {
            ...         "include_keywords": ["Python", "data", "machine learning"],
            ...         "max_words": 20
            ...     }
            ... }
            >>> result = probe.evaluate_single(model_output, reference)
            >>> print(result.metadata["included_keywords"])  # ["Python", "data"]
            >>> # "machine learning" not found, so keyword score = 2/3

        Example 3: Strict mode evaluation

            >>> probe = InstructionFollowingProbe(strict_mode=True)
            >>> model_output = "1. Red\\n2. Blue\\n3. Green\\n4. Yellow"
            >>> reference = {"constraints": {"format": "numbered_list", "max_items": 3}}
            >>> result = probe.evaluate_single(model_output, reference)
            >>> # Format is correct but 4 items > max 3
            >>> print(result.metadata["score"])  # 0.0 in strict mode

        Example 4: Evaluate word count constraints

            >>> probe = InstructionFollowingProbe()
            >>> model_output = "This is a short response with only ten words total."
            >>> reference = {"constraints": {"min_words": 5, "max_words": 15}}
            >>> result = probe.evaluate_single(model_output, reference)
            >>> print(result.metadata["word_count"])  # 10
            >>> print(result.metadata["within_min_words"])  # True
            >>> print(result.metadata["within_max_words"])  # True

        Notes
        -----
        The overall score calculation depends on strict_mode:
        - strict_mode=False: Average of all individual constraint scores
        - strict_mode=True: 1.0 only if all constraints pass, else 0.0

        A response is considered "compliant" if the overall score is >= 0.7.
        """
        constraints = reference.get("constraints", reference) if isinstance(reference, dict) else {}

        checks = []
        details = {}

        # Format check
        if "format" in constraints:
            fmt_result = self._check_format(model_output, constraints["format"])
            checks.append(fmt_result)
            details["format_compliance"] = fmt_result

        # Word count checks
        word_count = len(model_output.split())
        details["word_count"] = word_count

        if "max_words" in constraints:
            within_max = word_count <= constraints["max_words"]
            checks.append(1.0 if within_max else 0.0)
            details["within_max_words"] = within_max

        if "min_words" in constraints:
            within_min = word_count >= constraints["min_words"]
            checks.append(1.0 if within_min else 0.0)
            details["within_min_words"] = within_min

        # Item count checks (for lists)
        item_count = self._count_items(model_output)
        if item_count > 0:
            details["item_count"] = item_count

            if "max_items" in constraints:
                within_max = item_count <= constraints["max_items"]
                checks.append(1.0 if within_max else 0.0)
                details["within_max_items"] = within_max

            if "min_items" in constraints:
                within_min = item_count >= constraints["min_items"]
                checks.append(1.0 if within_min else 0.0)
                details["within_min_items"] = within_min

        # Keyword checks
        output_lower = model_output.lower()

        if "include_keywords" in constraints:
            keywords = constraints["include_keywords"]
            found = [kw for kw in keywords if kw.lower() in output_lower]
            keyword_score = len(found) / len(keywords) if keywords else 1.0
            checks.append(keyword_score)
            details["included_keywords"] = found

        if "exclude_keywords" in constraints:
            keywords = constraints["exclude_keywords"]
            found = [kw for kw in keywords if kw.lower() in output_lower]
            exclude_score = 1.0 if not found else 0.0
            checks.append(exclude_score)
            details["violated_exclusions"] = found

        # Calculate overall score
        if checks:
            if self.strict_mode:
                overall_score = 1.0 if all(c >= 1.0 for c in checks) else 0.0
            else:
                overall_score = sum(checks) / len(checks)
        else:
            overall_score = 1.0  # No constraints to check

        passed = overall_score >= 0.7

        details["score"] = overall_score
        details["label"] = "compliant" if passed else "non_compliant"

        return ProbeResult(
            input=str(constraints)[:100],
            output=model_output,
            status=ResultStatus.SUCCESS if passed else ResultStatus.ERROR,
            metadata=details,
        )

    def _check_format(self, output: str, expected_format: str) -> float:
        """Check if the model output matches the expected format.

        Evaluates the structural format of the model's response against the
        expected format type. Returns a score from 0.0 to 1.0 indicating
        how well the output conforms to the expected format.

        Parameters
        ----------
        output : str
            The model's response to check.
        expected_format : str
            The expected format type. Supported values:
            - "json": Valid JSON structure
            - "numbered_list": Lines starting with numbers (1. 2. 3.)
            - "bullet_list": Lines starting with -, *, or bullet character
            - "single_word": Response contains exactly one word
            - "single_sentence": Response contains exactly one sentence
            - "code": Response contains code patterns (def, function, class, import)

        Returns
        -------
        float
            Score from 0.0 to 1.0 where:
            - 1.0: Perfect format compliance
            - 0.5: Partial compliance (e.g., JSON-like but invalid)
            - 0.0-0.3: Poor compliance
            For unknown format types, returns 1.0 (no penalty).

        Examples
        --------
        Example 1: Valid JSON detection

            >>> probe = InstructionFollowingProbe()
            >>> output = '{"name": "Alice", "age": 30}'
            >>> score = probe._check_format(output, "json")
            >>> print(score)  # 1.0 (valid JSON)

        Example 2: Invalid but JSON-like structure

            >>> probe = InstructionFollowingProbe()
            >>> output = '{name: "Alice", age: 30}'  # Missing quotes on keys
            >>> score = probe._check_format(output, "json")
            >>> print(score)  # 0.5 (looks like JSON but invalid)

        Example 3: Numbered list check

            >>> probe = InstructionFollowingProbe()
            >>> output = "1. First item\\n2. Second item\\n3. Third item"
            >>> score = probe._check_format(output, "numbered_list")
            >>> print(score)  # 1.0 (all lines are numbered)

        Example 4: Mixed format (partial compliance)

            >>> probe = InstructionFollowingProbe()
            >>> output = "1. First item\\nSome text\\n2. Second item"
            >>> score = probe._check_format(output, "numbered_list")
            >>> # 2 of 3 non-empty lines are numbered = 0.67

        Notes
        -----
        - The single_word and single_sentence formats apply gradual penalties
          for additional words/sentences rather than binary pass/fail.
        - Code format detection uses pattern matching for common programming
          constructs but may not catch all code formats.
        """
        output = output.strip()

        if expected_format == "json":
            try:
                import json

                json.loads(output)
                return 1.0
            except json.JSONDecodeError:
                # Check for JSON-like structure
                if output.startswith("{") and output.endswith("}"):
                    return 0.5
                if output.startswith("[") and output.endswith("]"):
                    return 0.5
                return 0.0

        elif expected_format == "numbered_list":
            pattern = r"^\s*\d+[\.\)]\s+"
            lines = [line for line in output.split("\n") if line.strip()]
            numbered_lines = sum(1 for line in lines if re.match(pattern, line))
            return numbered_lines / len(lines) if lines else 0.0

        elif expected_format == "bullet_list":
            pattern = r"^\s*[-*•]\s+"
            lines = [line for line in output.split("\n") if line.strip()]
            bullet_lines = sum(1 for line in lines if re.match(pattern, line))
            return bullet_lines / len(lines) if lines else 0.0

        elif expected_format == "single_word":
            words = output.split()
            return 1.0 if len(words) == 1 else max(0.0, 1.0 - (len(words) - 1) * 0.3)

        elif expected_format == "single_sentence":
            # Count sentence endings
            sentences = re.split(r"[.!?]+", output)
            sentences = [s for s in sentences if s.strip()]
            return 1.0 if len(sentences) == 1 else max(0.0, 1.0 - (len(sentences) - 1) * 0.3)

        elif expected_format == "code":
            # Check for code indicators
            code_patterns = [
                r"```",
                r"def\s+\w+",
                r"function\s+\w+",
                r"class\s+\w+",
                r"import\s+\w+",
            ]
            has_code = any(re.search(p, output) for p in code_patterns)
            return 1.0 if has_code else 0.3

        return 1.0  # Unknown format, don't penalize

    def _count_items(self, output: str) -> int:
        """Count the number of list items in the model's output.

        Detects and counts list items in the output, supporting both numbered
        lists (1. 2. 3.) and bulleted lists (-, *, bullet). Numbered items
        take precedence if both types are detected.

        Parameters
        ----------
        output : str
            The model's response to analyze.

        Returns
        -------
        int
            The number of list items detected. Returns 0 if no list format
            is detected.

        Examples
        --------
        Example 1: Count numbered list items

            >>> probe = InstructionFollowingProbe()
            >>> output = "1. Apple\\n2. Banana\\n3. Cherry\\n4. Date"
            >>> count = probe._count_items(output)
            >>> print(count)  # 4

        Example 2: Count bullet list items

            >>> probe = InstructionFollowingProbe()
            >>> output = "- Red\\n- Green\\n- Blue"
            >>> count = probe._count_items(output)
            >>> print(count)  # 3

        Example 3: Numbered list with parentheses

            >>> probe = InstructionFollowingProbe()
            >>> output = "1) First\\n2) Second\\n3) Third"
            >>> count = probe._count_items(output)
            >>> print(count)  # 3

        Example 4: No list format detected

            >>> probe = InstructionFollowingProbe()
            >>> output = "This is just a regular paragraph without any list."
            >>> count = probe._count_items(output)
            >>> print(count)  # 0

        Notes
        -----
        - Supports numbered formats: "1.", "1)", with optional leading whitespace
        - Supports bullet formats: "-", "*", "bullet" with optional leading whitespace
        - If both numbered and bullet items are found, only numbered count is returned
        """
        # Count numbered items
        numbered = len(re.findall(r"^\s*\d+[\.\)]\s+", output, re.MULTILINE))
        if numbered > 0:
            return numbered

        # Count bullet items
        bulleted = len(re.findall(r"^\s*[-*•]\s+", output, re.MULTILINE))
        if bulleted > 0:
            return bulleted

        return 0


class MultiStepTaskProbe(ScoredProbe[str]):
    """Probe to test LLMs' ability to complete multi-step tasks.

    This probe evaluates whether a language model can successfully execute
    a sequence of related steps while maintaining context and coherence
    throughout the task. It is particularly useful for testing complex
    reasoning and instruction-following capabilities.

    Evaluates whether the model can:
        - Understand task decomposition into sequential steps
        - Complete each step correctly and thoroughly
        - Maintain context and coherence across steps
        - Reference information from earlier steps
        - Produce a coherent, comprehensive final result

    Attributes
    ----------
    default_category : ProbeCategory
        The default category for this probe (ProbeCategory.CUSTOM).

    Task Data Format
    ----------------
    The probe accepts task data in several formats:

    - **Dict with 'steps' key**: List of step descriptions
        ```python
        {
            "steps": ["Step 1", "Step 2", "Step 3"],
            "preamble": "Optional context",  # Optional
            "expected": {  # Optional patterns for evaluation
                "step_1": ["keyword1", "keyword2"],
                "step_2": ["keyword3"]
            }
        }
        ```
    - **List**: Treated as list of steps
    - **String**: Treated as single step

    Examples
    --------
    Example 1: Basic multi-step task

        >>> from insideLLMs.probes.instruction import MultiStepTaskProbe
        >>> probe = MultiStepTaskProbe()
        >>> task = {
        ...     "steps": [
        ...         "List 3 programming languages",
        ...         "Rank them by popularity",
        ...         "Explain why the top one is popular"
        ...     ]
        ... }
        >>> result = probe.run(model, task)
        >>> # Model should address all three steps in order

    Example 2: Task with expected patterns for evaluation

        >>> probe = MultiStepTaskProbe()
        >>> task = {
        ...     "steps": [
        ...         "Define machine learning",
        ...         "List 3 types of ML algorithms",
        ...         "Give an example use case for each"
        ...     ],
        ...     "expected": {
        ...         "step_1": ["machine learning", "data", "algorithms"],
        ...         "step_2": ["supervised", "unsupervised", "reinforcement"],
        ...         "step_3": ["classification", "clustering", "game"]
        ...     }
        ... }
        >>> result = probe.run(model, task)
        >>> evaluation = probe.evaluate_single(result, task)
        >>> print(f"Step 1 score: {evaluation.metadata['step_1']}")
        >>> print(f"Step 2 score: {evaluation.metadata['step_2']}")

    Example 3: Task with preamble context

        >>> probe = MultiStepTaskProbe()
        >>> task = {
        ...     "preamble": "You are a data science instructor explaining concepts to beginners.",
        ...     "steps": [
        ...         "Explain what a neural network is",
        ...         "Describe the key components",
        ...         "Give a simple analogy"
        ...     ]
        ... }
        >>> result = probe.run(model, task)

    Example 4: Using a list directly as steps

        >>> probe = MultiStepTaskProbe()
        >>> steps = [
        ...     "Calculate 15% of 200",
        ...     "Add 50 to the result",
        ...     "Divide by 4"
        ... ]
        >>> result = probe.run(model, steps)
        >>> # Model should show: 30, 80, 20

    See Also
    --------
    InstructionFollowingProbe : For testing single-step instruction following.
    ConstraintComplianceProbe : For focused single-constraint testing.
    """

    default_category = ProbeCategory.CUSTOM

    def __init__(
        self,
        name: str = "MultiStepTaskProbe",
    ):
        """Initialize the multi-step task probe.

        Creates a new MultiStepTaskProbe instance for evaluating a model's
        ability to complete sequential, multi-step tasks.

        Parameters
        ----------
        name : str, optional
            Name identifier for this probe instance. Useful when running
            multiple probes and needing to distinguish results in logs or
            reports. Default is "MultiStepTaskProbe".

        Examples
        --------
        Example 1: Default initialization

            >>> probe = MultiStepTaskProbe()
            >>> print(probe.name)
            'MultiStepTaskProbe'

        Example 2: Named probe for specific test suite

            >>> math_probe = MultiStepTaskProbe(name="MathReasoningProbe")
            >>> writing_probe = MultiStepTaskProbe(name="WritingTaskProbe")
            >>> # Use different probes for different test categories

        Example 3: Multiple probes in batch testing

            >>> probes = [
            ...     MultiStepTaskProbe(name="easy_tasks"),
            ...     MultiStepTaskProbe(name="medium_tasks"),
            ...     MultiStepTaskProbe(name="hard_tasks")
            ... ]
            >>> for probe in probes:
            ...     results = probe.run(model, get_tasks(probe.name))
        """
        super().__init__(name=name, category=ProbeCategory.CUSTOM)

    def run(self, model: Any, data: Any, **kwargs: Any) -> str:
        """Run the multi-step task probe against a model.

        Executes the probe by constructing a prompt with numbered steps and
        sending it to the model. The model is explicitly instructed to complete
        all steps in order and provide a comprehensive response.

        Parameters
        ----------
        model : Any
            The language model to test. Must have a ``generate(prompt, **kwargs)``
            method that accepts a string prompt and returns a string response.
        data : dict, list, or str
            The task data in one of these formats:
            - dict: With keys "steps" (list), optional "preamble" (str)
            - list: Treated directly as the list of steps
            - str: Treated as a single step
        **kwargs : Any
            Additional keyword arguments passed directly to ``model.generate()``.
            Common options include ``temperature``, ``max_tokens``, etc.

        Returns
        -------
        str
            The model's generated response addressing all steps.

        Examples
        --------
        Example 1: Dict with steps

            >>> probe = MultiStepTaskProbe()
            >>> data = {
            ...     "steps": [
            ...         "Name 3 European countries",
            ...         "List their capitals",
            ...         "State the population of each capital"
            ...     ]
            ... }
            >>> response = probe.run(model, data)
            >>> # Model response will address all three steps

        Example 2: Dict with preamble context

            >>> probe = MultiStepTaskProbe()
            >>> data = {
            ...     "preamble": "You are a helpful cooking assistant.",
            ...     "steps": [
            ...         "List ingredients for chocolate chip cookies",
            ...         "Describe the mixing process",
            ...         "Explain baking time and temperature"
            ...     ]
            ... }
            >>> response = probe.run(model, data, temperature=0.7)

        Example 3: List of steps directly

            >>> probe = MultiStepTaskProbe()
            >>> steps = [
            ...     "Explain what recursion is",
            ...     "Give a simple code example",
            ...     "Describe a real-world use case"
            ... ]
            >>> response = probe.run(model, steps)

        Example 4: Single step as string

            >>> probe = MultiStepTaskProbe()
            >>> response = probe.run(model, "Explain the water cycle in detail")
            >>> # Treated as a single-step task

        Notes
        -----
        The generated prompt follows this format:
        ```
        [preamble if provided]
        Complete the following steps in order:
        Step 1: [first step]
        Step 2: [second step]
        ...
        Provide your complete response, clearly addressing each step.
        ```
        """
        if isinstance(data, dict):
            steps = data.get("steps", [])
            preamble = data.get("preamble", "")
        elif isinstance(data, list):
            steps = data
            preamble = ""
        else:
            steps = [str(data)]
            preamble = ""

        # Build prompt
        prompt_parts = []

        if preamble:
            prompt_parts.append(preamble)

        prompt_parts.append("Complete the following steps in order:\n")

        for i, step in enumerate(steps, 1):
            prompt_parts.append(f"Step {i}: {step}")

        prompt_parts.append("\nProvide your complete response, clearly addressing each step.")

        prompt = "\n".join(prompt_parts)
        return model.generate(prompt, **kwargs)

    def evaluate_single(
        self,
        model_output: str,
        reference: Any,
        **kwargs: Any,
    ) -> ProbeResult[str]:
        """Evaluate how well the model completed the multi-step task.

        Analyzes the model's response to determine whether all steps were
        addressed, whether expected patterns/keywords appear, and whether
        the response has sufficient length to adequately cover all steps.

        Parameters
        ----------
        model_output : str
            The model's generated response to evaluate.
        reference : dict or Any
            The reference data for evaluation. Should be a dict with:
            - "steps": List of step descriptions (used to count expected steps)
            - "expected": Optional dict mapping "step_N" to list of expected
              keywords/patterns for that step

        **kwargs : Any
            Additional parameters (currently unused, reserved for future use).

        Returns
        -------
        ProbeResult[str]
            A ProbeResult containing:
            - input: String indicating number of steps
            - output: The model's response
            - status: ResultStatus.SUCCESS if score >= 0.5, else ERROR
            - metadata: Dict with detailed evaluation including:
                - score: Overall completion score (0.0 to 1.0)
                - label: "completed" if score >= 0.7, else "partial"
                - step_indicators_found: Count of "Step N" mentions detected
                - step_N: Score for each step based on expected patterns
                - word_count: Total words in response
                - length_score: Score based on response length adequacy

        Examples
        --------
        Example 1: Evaluate with expected patterns

            >>> probe = MultiStepTaskProbe()
            >>> model_output = '''
            ... Step 1: Machine learning is a subset of AI that enables
            ... computers to learn from data without explicit programming.
            ...
            ... Step 2: The main types are:
            ... - Supervised learning
            ... - Unsupervised learning
            ... - Reinforcement learning
            ...
            ... Step 3: Supervised learning is used for classification tasks
            ... like spam detection. Unsupervised is used for clustering.
            ... '''
            >>> reference = {
            ...     "steps": ["Define ML", "List types", "Give examples"],
            ...     "expected": {
            ...         "step_1": ["machine learning", "data"],
            ...         "step_2": ["supervised", "unsupervised"],
            ...         "step_3": ["classification", "clustering"]
            ...     }
            ... }
            >>> result = probe.evaluate_single(model_output, reference)
            >>> print(result.metadata["step_1"])  # Score for step 1
            >>> print(result.metadata["step_indicators_found"])  # 3

        Example 2: Evaluate without expected patterns

            >>> probe = MultiStepTaskProbe()
            >>> model_output = "Step 1: Here's the first part. Step 2: And the second."
            >>> reference = {"steps": ["First task", "Second task"]}
            >>> result = probe.evaluate_single(model_output, reference)
            >>> # Without expected patterns, step scores default to 0.5
            >>> print(result.metadata["step_1"])  # 0.5
            >>> print(result.metadata["length_score"])  # Based on word count

        Example 3: Evaluate response length adequacy

            >>> probe = MultiStepTaskProbe()
            >>> # Short response for 3 steps (expects ~60 words minimum)
            >>> model_output = "Step 1: Done. Step 2: Done. Step 3: Done."
            >>> reference = {"steps": ["A", "B", "C"]}
            >>> result = probe.evaluate_single(model_output, reference)
            >>> print(result.metadata["word_count"])  # ~9 words
            >>> print(result.metadata["length_score"])  # 9/60 = 0.15

        Example 4: High-scoring complete response

            >>> probe = MultiStepTaskProbe()
            >>> # Comprehensive response with all expected keywords
            >>> model_output = '''
            ... Step 1: Python is a high-level programming language known
            ... for its readability and versatility.
            ...
            ... Step 2: Python is widely used for web development, data
            ... analysis, machine learning, and automation.
            ...
            ... Step 3: Compared to Java, Python has simpler syntax and is
            ... often faster to write, though Java may perform better at runtime.
            ... '''
            >>> reference = {
            ...     "steps": ["Define Python", "List uses", "Compare to Java"],
            ...     "expected": {
            ...         "step_1": ["python", "programming"],
            ...         "step_2": ["web", "data", "machine learning"],
            ...         "step_3": ["java", "syntax"]
            ...     }
            ... }
            >>> result = probe.evaluate_single(model_output, reference)
            >>> print(result.metadata["label"])  # "completed"
            >>> print(result.status)  # ResultStatus.SUCCESS

        Notes
        -----
        - The length score expects approximately 20 words per step as minimum
        - Step scores without expected patterns default to 0.5 (neutral)
        - Overall score is the average of all step scores plus length score
        """
        if isinstance(reference, dict):
            steps = reference.get("steps", [])
            expected_patterns = reference.get("expected", {})
        else:
            steps = []
            expected_patterns = {}

        details = {}
        step_scores = []

        # Check for step indicators
        output_lower = model_output.lower()
        step_mentions = sum(
            1
            for i in range(1, len(steps) + 1)
            if f"step {i}" in output_lower or f"{i}." in model_output
        )
        details["step_indicators_found"] = step_mentions

        # Check expected patterns per step
        for i, _step in enumerate(steps, 1):
            step_key = f"step_{i}"
            if step_key in expected_patterns:
                patterns = expected_patterns[step_key]
                if isinstance(patterns, str):
                    patterns = [patterns]

                found = sum(1 for p in patterns if p.lower() in output_lower)
                step_score = found / len(patterns) if patterns else 1.0
            else:
                # No specific pattern, check for reasonable content
                step_score = 0.5  # Neutral

            step_scores.append(step_score)
            details[step_key] = step_score

        # Length reasonability (multi-step should have substantial response)
        word_count = len(model_output.split())
        expected_min_words = len(steps) * 20  # ~20 words per step minimum
        length_score = min(1.0, word_count / expected_min_words) if expected_min_words > 0 else 1.0
        details["word_count"] = word_count
        details["length_score"] = length_score

        # Overall score
        all_scores = step_scores + [length_score]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.5

        details["score"] = overall_score
        details["label"] = "completed" if overall_score >= 0.7 else "partial"

        return ProbeResult(
            input=f"{len(steps)} steps",
            output=model_output,
            status=ResultStatus.SUCCESS if overall_score >= 0.5 else ResultStatus.ERROR,
            metadata=details,
        )


class ConstraintComplianceProbe(ScoredProbe[str]):
    """Probe to test specific constraint compliance in LLM outputs.

    This probe focuses on testing whether language models can respect specific,
    well-defined constraints. Unlike the more general InstructionFollowingProbe,
    this probe is designed for focused testing of individual constraint types
    with optional custom validation functions.

    Tests whether models can respect specific constraints like:
        - Character limits (exact character count restrictions)
        - Word limits (word count restrictions)
        - Sentence limits (sentence count restrictions)
        - Custom constraints (user-defined validation rules)

    Attributes
    ----------
    constraint_type : str
        The type of constraint being tested. One of:
        - "word_limit": Limit on number of words
        - "character_limit": Limit on number of characters
        - "sentence_limit": Limit on number of sentences
        - "custom": User-defined constraint with validator function
    limit : int or None
        The numeric limit for word/char/sentence constraints.
    custom_constraint : str or None
        Description of custom constraint (shown to model).
    validator : Callable[[str], bool] or None
        Custom validation function for "custom" constraint type.
    default_category : ProbeCategory
        The default category for this probe (ProbeCategory.CUSTOM).

    Examples
    --------
    Example 1: Word limit constraint

        >>> from insideLLMs.probes.instruction import ConstraintComplianceProbe
        >>> probe = ConstraintComplianceProbe(
        ...     constraint_type="word_limit",
        ...     limit=50
        ... )
        >>> result = probe.run(model, "Explain quantum computing")
        >>> evaluation = probe.evaluate_single(result, reference=50)
        >>> print(f"Word count: {evaluation.metadata['word_count']}")
        >>> print(f"Compliant: {evaluation.metadata['compliant']}")

    Example 2: Character limit for tweets

        >>> probe = ConstraintComplianceProbe(
        ...     constraint_type="character_limit",
        ...     limit=280
        ... )
        >>> result = probe.run(model, "Write a tweet about climate change")
        >>> evaluation = probe.evaluate_single(result, reference=280)
        >>> print(f"Characters: {evaluation.metadata['character_count']}")

    Example 3: Sentence limit constraint

        >>> probe = ConstraintComplianceProbe(
        ...     constraint_type="sentence_limit",
        ...     limit=3
        ... )
        >>> result = probe.run(model, "Summarize World War II")
        >>> evaluation = probe.evaluate_single(result, reference=3)
        >>> print(f"Sentences: {evaluation.metadata['sentence_count']}")

    Example 4: Custom constraint with validator

        >>> def contains_code(output: str) -> bool:
        ...     return "def " in output or "function " in output or "```" in output
        >>>
        >>> probe = ConstraintComplianceProbe(
        ...     constraint_type="custom",
        ...     custom_constraint="Include a working code example",
        ...     validator=contains_code
        ... )
        >>> result = probe.run(model, "Show how to reverse a string in Python")
        >>> evaluation = probe.evaluate_single(result, reference=None)
        >>> print(f"Has code: {evaluation.metadata['custom_validation']}")

    Example 5: Named probe for specific use case

        >>> probe = ConstraintComplianceProbe(
        ...     name="TweetLengthValidator",
        ...     constraint_type="character_limit",
        ...     limit=280
        ... )
        >>> # Use for validating social media content length

    See Also
    --------
    InstructionFollowingProbe : For testing multiple constraints simultaneously.
    MultiStepTaskProbe : For testing multi-step task completion.
    """

    default_category = ProbeCategory.CUSTOM

    def __init__(
        self,
        name: str = "ConstraintComplianceProbe",
        constraint_type: str = "word_limit",
        limit: Optional[int] = None,
        custom_constraint: Optional[str] = None,
        validator: Optional[Callable[[str], bool]] = None,
    ):
        """Initialize the constraint compliance probe.

        Creates a new ConstraintComplianceProbe configured to test a specific
        type of constraint. Supports built-in constraint types (word, character,
        sentence limits) as well as custom constraints with user-defined
        validation functions.

        Parameters
        ----------
        name : str, optional
            Name identifier for this probe instance. Useful for distinguishing
            probes in logs and reports. Default is "ConstraintComplianceProbe".
        constraint_type : str, optional
            The type of constraint to test. Supported values:
            - "word_limit": Test word count limits (default)
            - "character_limit": Test character count limits
            - "sentence_limit": Test sentence count limits
            - "custom": Test custom constraint with validator function
        limit : int or None, optional
            The numeric limit for word/character/sentence constraints.
            Required for built-in constraint types. Default is None.
        custom_constraint : str or None, optional
            Human-readable description of the custom constraint. This text
            is included in the prompt to inform the model of the requirement.
            Only used when constraint_type is "custom". Default is None.
        validator : Callable[[str], bool] or None, optional
            Custom validation function that takes the model output and returns
            True if the constraint is satisfied, False otherwise. Only used
            when constraint_type is "custom". Default is None.

        Raises
        ------
        Note: No explicit validation is performed in __init__. Invalid
        combinations (e.g., word_limit without a limit) will result in
        no constraint being applied during evaluation.

        Examples
        --------
        Example 1: Word limit constraint

            >>> probe = ConstraintComplianceProbe(
            ...     constraint_type="word_limit",
            ...     limit=100
            ... )
            >>> print(probe.constraint_type)
            'word_limit'
            >>> print(probe.limit)
            100

        Example 2: Character limit for short-form content

            >>> tweet_probe = ConstraintComplianceProbe(
            ...     name="TweetValidator",
            ...     constraint_type="character_limit",
            ...     limit=280
            ... )

        Example 3: Sentence limit for summaries

            >>> summary_probe = ConstraintComplianceProbe(
            ...     name="ThreeSentenceSummary",
            ...     constraint_type="sentence_limit",
            ...     limit=3
            ... )

        Example 4: Custom constraint with validator

            >>> def has_bullet_points(output: str) -> bool:
            ...     return any(line.strip().startswith(('-', '*', 'bullet'))
            ...                for line in output.split('\\n'))
            >>>
            >>> probe = ConstraintComplianceProbe(
            ...     name="BulletPointChecker",
            ...     constraint_type="custom",
            ...     custom_constraint="Response must use bullet points",
            ...     validator=has_bullet_points
            ... )

        Example 5: Custom constraint checking for specific content

            >>> def mentions_source(output: str) -> bool:
            ...     keywords = ["according to", "source:", "reference:", "cited"]
            ...     return any(kw in output.lower() for kw in keywords)
            >>>
            >>> probe = ConstraintComplianceProbe(
            ...     constraint_type="custom",
            ...     custom_constraint="Must cite at least one source",
            ...     validator=mentions_source
            ... )
        """
        super().__init__(name=name, category=ProbeCategory.CUSTOM)
        self.constraint_type = constraint_type
        self.limit = limit
        self.custom_constraint = custom_constraint
        self.validator = validator

    def run(self, model: Any, data: Any, **kwargs: Any) -> str:
        """Run the constraint compliance probe against a model.

        Executes the probe by appending the constraint instruction to the task
        and sending the combined prompt to the model. The constraint is stated
        as an important requirement to maximize compliance.

        Parameters
        ----------
        model : Any
            The language model to test. Must have a ``generate(prompt, **kwargs)``
            method that accepts a string prompt and returns a string response.
        data : dict or str
            The task data. Can be:
            - A dict with a "task" key containing the task description
            - A plain string (used directly as the task)
        **kwargs : Any
            Additional keyword arguments passed directly to ``model.generate()``.
            Common options include ``temperature``, ``max_tokens``, etc.

        Returns
        -------
        str
            The model's generated response.

        Examples
        --------
        Example 1: Word limit with dict input

            >>> probe = ConstraintComplianceProbe(
            ...     constraint_type="word_limit",
            ...     limit=50
            ... )
            >>> data = {"task": "Explain the theory of relativity"}
            >>> response = probe.run(model, data)
            >>> # Prompt includes: "IMPORTANT: Your response must be 50 words or fewer."

        Example 2: Character limit with string input

            >>> probe = ConstraintComplianceProbe(
            ...     constraint_type="character_limit",
            ...     limit=140
            ... )
            >>> response = probe.run(model, "Write a short bio for a software engineer")
            >>> # Response should be 140 characters or fewer

        Example 3: Custom constraint

            >>> def has_examples(output: str) -> bool:
            ...     return "example" in output.lower() or "e.g." in output.lower()
            >>>
            >>> probe = ConstraintComplianceProbe(
            ...     constraint_type="custom",
            ...     custom_constraint="Include at least one concrete example",
            ...     validator=has_examples
            ... )
            >>> response = probe.run(model, "Explain polymorphism in programming")
            >>> # Prompt includes: "IMPORTANT: Include at least one concrete example"

        Example 4: Sentence limit with temperature

            >>> probe = ConstraintComplianceProbe(
            ...     constraint_type="sentence_limit",
            ...     limit=2
            ... )
            >>> response = probe.run(
            ...     model,
            ...     "What is machine learning?",
            ...     temperature=0.3
            ... )

        Notes
        -----
        The constraint instruction is formatted as:
        "IMPORTANT: Your response must be [X] [words/characters/sentences] or fewer."
        For custom constraints:
        "IMPORTANT: [custom_constraint text]"
        """
        task = data.get("task", str(data)) if isinstance(data, dict) else str(data)

        # Build constraint instruction
        constraint_text = self._get_constraint_instruction()

        prompt = f"{task}\n\n{constraint_text}"
        return model.generate(prompt, **kwargs)

    def _get_constraint_instruction(self) -> str:
        """Generate the constraint instruction text to append to prompts.

        Converts the probe's constraint configuration into a human-readable
        instruction that clearly communicates the requirement to the model.
        The instruction is prefixed with "IMPORTANT:" to emphasize its
        significance.

        Returns
        -------
        str
            The formatted constraint instruction. Returns empty string if
            no valid constraint is configured (e.g., word_limit without
            a limit value).

        Examples
        --------
        Example 1: Word limit instruction

            >>> probe = ConstraintComplianceProbe(
            ...     constraint_type="word_limit",
            ...     limit=50
            ... )
            >>> instruction = probe._get_constraint_instruction()
            >>> print(instruction)
            IMPORTANT: Your response must be 50 words or fewer.

        Example 2: Character limit instruction

            >>> probe = ConstraintComplianceProbe(
            ...     constraint_type="character_limit",
            ...     limit=280
            ... )
            >>> instruction = probe._get_constraint_instruction()
            >>> print(instruction)
            IMPORTANT: Your response must be 280 characters or fewer.

        Example 3: Sentence limit instruction

            >>> probe = ConstraintComplianceProbe(
            ...     constraint_type="sentence_limit",
            ...     limit=3
            ... )
            >>> instruction = probe._get_constraint_instruction()
            >>> print(instruction)
            IMPORTANT: Your response must be 3 sentence(s) or fewer.

        Example 4: Custom constraint instruction

            >>> probe = ConstraintComplianceProbe(
            ...     constraint_type="custom",
            ...     custom_constraint="Include exactly 3 bullet points"
            ... )
            >>> instruction = probe._get_constraint_instruction()
            >>> print(instruction)
            IMPORTANT: Include exactly 3 bullet points

        Notes
        -----
        If constraint_type is set but limit is None (for built-in types),
        or if constraint_type is "custom" but custom_constraint is None,
        an empty string is returned.
        """
        if self.constraint_type == "word_limit" and self.limit:
            return f"IMPORTANT: Your response must be {self.limit} words or fewer."
        elif self.constraint_type == "character_limit" and self.limit:
            return f"IMPORTANT: Your response must be {self.limit} characters or fewer."
        elif self.constraint_type == "sentence_limit" and self.limit:
            return f"IMPORTANT: Your response must be {self.limit} sentence(s) or fewer."
        elif self.custom_constraint:
            return f"IMPORTANT: {self.custom_constraint}"
        else:
            return ""

    def evaluate_single(
        self,
        model_output: str,
        reference: Any,
        **kwargs: Any,
    ) -> ProbeResult[str]:
        """Evaluate whether the model output complies with the constraint.

        Checks the model's response against the configured constraint type
        and calculates a compliance score. For numeric constraints (word,
        character, sentence limits), the score gradually decreases based
        on how much the limit is exceeded.

        Parameters
        ----------
        model_output : str
            The model's generated response to evaluate.
        reference : int or Any, optional
            Optional override for the limit value. If an int is provided,
            it replaces the probe's configured limit for this evaluation.
            Useful for testing the same model against different thresholds.
        **kwargs : Any
            Additional parameters (currently unused, reserved for future use).

        Returns
        -------
        ProbeResult[str]
            A ProbeResult containing:
            - input: The constraint type being tested
            - output: The model's response
            - status: ResultStatus.SUCCESS if compliant, else ERROR
            - metadata: Dict with detailed evaluation including:
                - compliant: Boolean indicating if constraint was met
                - score: Compliance score (0.0 to 1.0)
                - label: "compliant" or "violation"
                - For word_limit: word_count, limit
                - For character_limit: character_count, limit
                - For sentence_limit: sentence_count, limit
                - For custom: custom_validation (bool)

        Examples
        --------
        Example 1: Evaluate word limit compliance

            >>> probe = ConstraintComplianceProbe(
            ...     constraint_type="word_limit",
            ...     limit=50
            ... )
            >>> # Model generates 45 words
            >>> model_output = " ".join(["word"] * 45)
            >>> result = probe.evaluate_single(model_output, reference=50)
            >>> print(result.metadata["word_count"])  # 45
            >>> print(result.metadata["compliant"])  # True
            >>> print(result.metadata["score"])  # 1.0

        Example 2: Evaluate with exceeded limit

            >>> probe = ConstraintComplianceProbe(
            ...     constraint_type="word_limit",
            ...     limit=50
            ... )
            >>> # Model generates 75 words (50% over limit)
            >>> model_output = " ".join(["word"] * 75)
            >>> result = probe.evaluate_single(model_output, reference=50)
            >>> print(result.metadata["compliant"])  # False
            >>> print(result.metadata["score"])  # 0.5 (gradual decrease)

        Example 3: Override limit at evaluation time

            >>> probe = ConstraintComplianceProbe(
            ...     constraint_type="word_limit",
            ...     limit=100
            ... )
            >>> model_output = " ".join(["word"] * 75)
            >>> # Test against stricter limit
            >>> result = probe.evaluate_single(model_output, reference=50)
            >>> print(result.metadata["limit"])  # 50 (override applied)
            >>> print(result.metadata["compliant"])  # False

        Example 4: Custom validator evaluation

            >>> def has_greeting(output: str) -> bool:
            ...     greetings = ["hello", "hi", "hey", "greetings"]
            ...     return any(g in output.lower() for g in greetings)
            >>>
            >>> probe = ConstraintComplianceProbe(
            ...     constraint_type="custom",
            ...     custom_constraint="Start with a greeting",
            ...     validator=has_greeting
            ... )
            >>> result = probe.evaluate_single("Hello! How are you?", reference=None)
            >>> print(result.metadata["custom_validation"])  # True
            >>> print(result.metadata["score"])  # 1.0

        Notes
        -----
        Score calculation for limit violations:
        - If compliant: score = 1.0
        - If exceeded: score = max(0.0, 1.0 - (overage / limit))

        For example, exceeding a 50-word limit by 25 words results in:
        score = max(0.0, 1.0 - (25 / 50)) = 0.5
        """
        limit = reference if isinstance(reference, int) else self.limit
        details = {}
        compliant = True

        if self.constraint_type == "word_limit":
            word_count = len(model_output.split())
            details["word_count"] = word_count
            details["limit"] = limit
            if limit:
                compliant = word_count <= limit
                # Calculate how close to limit
                if compliant:
                    score = 1.0
                else:
                    overage = word_count - limit
                    score = max(0.0, 1.0 - (overage / limit))

        elif self.constraint_type == "character_limit":
            char_count = len(model_output)
            details["character_count"] = char_count
            details["limit"] = limit
            if limit:
                compliant = char_count <= limit
                if compliant:
                    score = 1.0
                else:
                    overage = char_count - limit
                    score = max(0.0, 1.0 - (overage / limit))

        elif self.constraint_type == "sentence_limit":
            sentences = re.split(r"[.!?]+", model_output)
            sentence_count = len([s for s in sentences if s.strip()])
            details["sentence_count"] = sentence_count
            details["limit"] = limit
            if limit:
                compliant = sentence_count <= limit
                if compliant:
                    score = 1.0
                else:
                    overage = sentence_count - limit
                    score = max(0.0, 1.0 - (overage / limit))

        elif self.validator:
            compliant = self.validator(model_output)
            score = 1.0 if compliant else 0.0
            details["custom_validation"] = compliant

        else:
            score = 1.0  # No constraint to check

        details["compliant"] = compliant
        details["score"] = score
        details["label"] = "compliant" if compliant else "violation"

        return ProbeResult(
            input=self.constraint_type,
            output=model_output,
            status=ResultStatus.SUCCESS if compliant else ResultStatus.ERROR,
            metadata=details,
        )
