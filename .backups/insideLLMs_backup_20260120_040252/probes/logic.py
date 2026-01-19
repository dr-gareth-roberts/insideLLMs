"""Logic probe for testing LLM reasoning capabilities.

Tests the model's ability to solve logic problems, including:
- Deductive reasoning
- Mathematical logic
- Syllogisms
- Puzzles and riddles
"""

import re
from typing import Any, Optional

from insideLLMs.probes.base import ScoredProbe
from insideLLMs.types import ProbeCategory, ProbeResult, ProbeScore, ResultStatus


class LogicProbe(ScoredProbe[str]):
    """Probe to test LLMs' zero-shot ability at logic problems.

    This probe presents logic problems to the model and evaluates
    whether it can reason through them correctly.

    Attributes:
        name: Name of the probe.
        category: Always LOGIC for this probe.
        prompt_template: Template for formatting the problem.

    Example:
        >>> probe = LogicProbe()
        >>> result = probe.run(model, "If A > B and B > C, what is the relationship between A and C?")
        >>> print(result)  # Model's reasoning
    """

    default_category = ProbeCategory.LOGIC

    def __init__(
        self,
        name: str = "LogicProbe",
        prompt_template: Optional[str] = None,
        extract_answer: bool = True,
    ):
        """Initialize the logic probe.

        Args:
            name: Name for this probe instance.
            prompt_template: Custom template for prompts. Use {problem} as placeholder.
            extract_answer: Whether to extract a final answer from the response.
        """
        super().__init__(name=name, category=ProbeCategory.LOGIC)
        self.prompt_template = prompt_template or (
            "Solve this logic problem step by step. "
            "Show your reasoning, then state your final answer clearly.\n\n"
            "Problem: {problem}"
        )
        self.extract_answer = extract_answer

    def run(self, model: Any, logic_problem: Any, **kwargs: Any) -> str:
        """Run the logic probe on the given model with a logic problem.

        Args:
            model: The model to test.
            logic_problem: Either a string problem or a dict with 'problem' key.
            **kwargs: Additional arguments passed to the model.

        Returns:
            The model's response to the logic problem.
        """
        # Handle both string and dict inputs
        if isinstance(logic_problem, dict):
            problem_text = logic_problem.get("problem", logic_problem.get("question", ""))
        else:
            problem_text = str(logic_problem)

        prompt = self.prompt_template.format(problem=problem_text)
        return model.generate(prompt, **kwargs)

    def evaluate_single(
        self,
        model_output: str,
        reference: Any,
        input_data: Any,
    ) -> dict[str, Any]:
        """Evaluate a single logic problem response.

        Args:
            model_output: The model's response.
            reference: The expected answer.
            input_data: The original problem.

        Returns:
            Evaluation metrics including correctness assessment.
        """
        if reference is None:
            return {"evaluated": False}

        ref_answer = str(reference).lower().strip()
        extracted = self._extract_final_answer(model_output).lower().strip()

        # Check for exact match or containment
        is_correct = ref_answer == extracted or ref_answer in extracted or extracted in ref_answer

        return {
            "is_correct": is_correct,
            "extracted_answer": extracted,
            "reference_answer": ref_answer,
            "response_length": len(model_output),
            "has_reasoning": self._has_reasoning(model_output),
        }

    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from a response.

        Looks for patterns like:
        - "The answer is X"
        - "Therefore, X"
        - "Final answer: X"
        """
        patterns = [
            r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)",
            r"therefore[,:\s]+(.+?)(?:\.|$)",
            r"(?:final\s+)?answer[:\s]+(.+?)(?:\.|$)",
            r"(?:conclusion|result)[:\s]+(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no pattern matches, return last sentence
        sentences = response.split(".")
        if sentences:
            return sentences[-1].strip() or (sentences[-2].strip() if len(sentences) > 1 else "")
        return response[:100]

    def _has_reasoning(self, response: str) -> bool:
        """Check if the response contains step-by-step reasoning."""
        reasoning_indicators = [
            "step",
            "first",
            "second",
            "then",
            "therefore",
            "because",
            "since",
            "thus",
            "if",
            "let",
            "given",
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in reasoning_indicators)

    def score(self, results: list[ProbeResult[str]]) -> ProbeScore:
        """Calculate scores for logic probe results.

        Args:
            results: List of probe results.

        Returns:
            ProbeScore with accuracy and custom metrics.
        """
        base_score = super().score(results)

        # Calculate additional metrics
        reasoning_count = 0
        total_length = 0

        for result in results:
            if result.status == ResultStatus.SUCCESS and result.output:
                if self._has_reasoning(result.output):
                    reasoning_count += 1
                total_length += len(result.output)

        success_count = sum(1 for r in results if r.status == ResultStatus.SUCCESS)

        base_score.custom_metrics = {
            "reasoning_rate": reasoning_count / success_count if success_count > 0 else 0,
            "avg_response_length": total_length / success_count if success_count > 0 else 0,
        }

        return base_score
