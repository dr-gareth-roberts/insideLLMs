"""Instruction following probes.

Tests the model's ability to:
- Follow explicit instructions precisely
- Handle multi-step tasks
- Respect constraints and formatting requirements
- Maintain consistency with given guidelines
"""

import re
from typing import Any, Callable, Optional

from insideLLMs.probes.base import ScoredProbe
from insideLLMs.types import ProbeCategory, ProbeResult, ResultStatus


class InstructionFollowingProbe(ScoredProbe[str]):
    """Probe to test LLMs' ability to follow instructions precisely.

    Tests various aspects of instruction following:
    - Format compliance (JSON, lists, specific structures)
    - Length constraints
    - Content restrictions
    - Multi-step task completion

    Example:
        >>> probe = InstructionFollowingProbe()
        >>> instructions = {
        ...     "task": "List 3 fruits",
        ...     "constraints": {"format": "numbered_list", "max_items": 3}
        ... }
        >>> result = probe.run(model, instructions)
    """

    default_category = ProbeCategory.CUSTOM

    def __init__(
        self,
        name: str = "InstructionFollowingProbe",
        strict_mode: bool = False,
    ):
        """Initialize the instruction following probe.

        Args:
            name: Name for this probe instance.
            strict_mode: If True, any constraint violation fails the probe.
        """
        super().__init__(name=name, category=ProbeCategory.CUSTOM)
        self.strict_mode = strict_mode

    def run(self, model: Any, data: Any, **kwargs: Any) -> str:
        """Run the instruction following probe.

        Args:
            model: The model to test.
            data: Dict with 'task' and optional 'constraints'.
            **kwargs: Additional arguments passed to the model.

        Returns:
            The model's response.
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
        """Format constraints as explicit instructions.

        Args:
            constraints: Dict of constraint specifications.

        Returns:
            Formatted constraint string.
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
        """Evaluate instruction following.

        Args:
            model_output: The model's response.
            reference: Dict with constraints to check against.
            **kwargs: Additional parameters.

        Returns:
            ProbeResult with constraint compliance details.
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
        """Check if output matches expected format.

        Args:
            output: The model's response.
            expected_format: Expected format type.

        Returns:
            Score from 0 to 1.
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
        """Count list items in output.

        Args:
            output: The model's response.

        Returns:
            Number of list items detected.
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

    Evaluates whether the model can:
    - Understand task decomposition
    - Complete each step correctly
    - Maintain context across steps
    - Produce a coherent final result

    Example:
        >>> probe = MultiStepTaskProbe()
        >>> task = {
        ...     "steps": [
        ...         "List 3 programming languages",
        ...         "Rank them by popularity",
        ...         "Explain why the top one is popular"
        ...     ]
        ... }
        >>> result = probe.run(model, task)
    """

    default_category = ProbeCategory.CUSTOM

    def __init__(
        self,
        name: str = "MultiStepTaskProbe",
    ):
        """Initialize the multi-step task probe.

        Args:
            name: Name for this probe instance.
        """
        super().__init__(name=name, category=ProbeCategory.CUSTOM)

    def run(self, model: Any, data: Any, **kwargs: Any) -> str:
        """Run the multi-step task probe.

        Args:
            model: The model to test.
            data: Dict with 'steps' list or 'task' string.
            **kwargs: Additional arguments passed to the model.

        Returns:
            The model's response.
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
        """Evaluate multi-step task completion.

        Args:
            model_output: The model's response.
            reference: Dict with 'steps' and optional 'expected' patterns.
            **kwargs: Additional parameters.

        Returns:
            ProbeResult with step completion details.
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
    """Probe to test specific constraint compliance.

    Tests whether models can respect specific constraints like:
    - Character limits
    - Word limits
    - Specific formatting requirements
    - Role/persona constraints

    Example:
        >>> probe = ConstraintComplianceProbe(
        ...     constraint_type="character_limit",
        ...     limit=100
        ... )
        >>> result = probe.run(model, "Summarize machine learning")
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

        Args:
            name: Name for this probe instance.
            constraint_type: Type of constraint (word_limit, char_limit, custom).
            limit: Numeric limit for word/char constraints.
            custom_constraint: Description of custom constraint.
            validator: Custom validation function.
        """
        super().__init__(name=name, category=ProbeCategory.CUSTOM)
        self.constraint_type = constraint_type
        self.limit = limit
        self.custom_constraint = custom_constraint
        self.validator = validator

    def run(self, model: Any, data: Any, **kwargs: Any) -> str:
        """Run the constraint compliance probe.

        Args:
            model: The model to test.
            data: The task or prompt.
            **kwargs: Additional arguments passed to the model.

        Returns:
            The model's response.
        """
        task = data.get("task", str(data)) if isinstance(data, dict) else str(data)

        # Build constraint instruction
        constraint_text = self._get_constraint_instruction()

        prompt = f"{task}\n\n{constraint_text}"
        return model.generate(prompt, **kwargs)

    def _get_constraint_instruction(self) -> str:
        """Get the constraint instruction text."""
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
        """Evaluate constraint compliance.

        Args:
            model_output: The model's response.
            reference: Optional override for limit.
            **kwargs: Additional parameters.

        Returns:
            ProbeResult with compliance details.
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
