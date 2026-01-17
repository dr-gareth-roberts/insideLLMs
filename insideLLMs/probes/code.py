"""Code generation and evaluation probes.

Tests the model's ability to:
- Generate correct code from descriptions
- Understand and explain code
- Debug and fix code issues
- Follow coding best practices
"""

import re
from typing import Any, Dict, List, Optional

from insideLLMs.probes.base import ScoredProbe
from insideLLMs.types import ProbeCategory, ProbeResult, ProbeScore, ResultStatus


class CodeGenerationProbe(ScoredProbe[str]):
    """Probe to test LLMs' ability to generate correct code.

    Evaluates code generation quality based on:
    - Syntactic correctness
    - Functional correctness (if test cases provided)
    - Code style and best practices

    Example:
        >>> probe = CodeGenerationProbe(language="python")
        >>> result = probe.run(model, "Write a function to check if a number is prime")
        >>> score = probe.score(result, expected_output="def is_prime")
    """

    default_category = ProbeCategory.CUSTOM

    def __init__(
        self,
        name: str = "CodeGenerationProbe",
        language: str = "python",
        require_docstrings: bool = False,
        require_type_hints: bool = False,
    ):
        """Initialize the code generation probe.

        Args:
            name: Name for this probe instance.
            language: Target programming language.
            require_docstrings: Whether to penalize missing docstrings.
            require_type_hints: Whether to penalize missing type hints.
        """
        super().__init__(name=name, category=ProbeCategory.CUSTOM)
        self.language = language.lower()
        self.require_docstrings = require_docstrings
        self.require_type_hints = require_type_hints

        self.prompt_template = (
            f"Write {language} code to solve the following task. "
            "Provide only the code, no explanations.\n\n"
            "Task: {task}"
        )

    def run(self, model: Any, task: Any, **kwargs: Any) -> str:
        """Run the code generation probe.

        Args:
            model: The model to test.
            task: Either a string task description or dict with 'task' key.
            **kwargs: Additional arguments passed to the model.

        Returns:
            The generated code.
        """
        if isinstance(task, dict):
            task_text = task.get("task", task.get("description", str(task)))
        else:
            task_text = str(task)

        prompt = self.prompt_template.format(task=task_text)
        return model.generate(prompt, **kwargs)

    def extract_code(self, response: str) -> str:
        """Extract code from a response that may contain markdown.

        Args:
            response: The model's response.

        Returns:
            Extracted code.
        """
        # Try to extract from code blocks
        code_block_pattern = rf"```(?:{self.language})?\n?(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            return matches[0].strip()

        # If no code blocks, try to find code-like content
        lines = response.split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            stripped = line.strip()
            # Heuristics for code detection
            is_code_line = (
                stripped.startswith(("def ", "class ", "import ", "from ", "#"))
                or stripped.startswith(("function ", "const ", "let ", "var "))
                or stripped.startswith(("public ", "private ", "static "))
                or "=" in stripped
                or stripped.endswith(":")
                or stripped.startswith("}")
                or in_code
            )

            if is_code_line:
                in_code = True
                code_lines.append(line)
            elif code_lines and not stripped:
                code_lines.append(line)  # Keep blank lines in code
            elif code_lines:
                break

        return "\n".join(code_lines).strip() if code_lines else response

    def evaluate_single(
        self,
        model_output: str,
        reference: Any,
        **kwargs: Any,
    ) -> ProbeResult[str]:
        """Evaluate a single code generation result.

        Args:
            model_output: The generated code.
            reference: Expected code patterns or test cases.
            **kwargs: Additional evaluation parameters.

        Returns:
            ProbeResult with score and details.
        """
        code = self.extract_code(model_output)
        score_components = []
        details = {}

        # Check for syntax validity (basic)
        syntax_valid = self._check_syntax(code)
        score_components.append(1.0 if syntax_valid else 0.0)
        details["syntax_valid"] = syntax_valid

        # Check for expected patterns
        if isinstance(reference, dict):
            expected_patterns = reference.get("patterns", [])
            expected_output = reference.get("expected_output")
        elif isinstance(reference, str):
            expected_patterns = [reference]
            expected_output = None
        else:
            expected_patterns = []
            expected_output = None

        if expected_patterns:
            pattern_matches = sum(
                1 for p in expected_patterns if p.lower() in code.lower()
            )
            pattern_score = pattern_matches / len(expected_patterns)
            score_components.append(pattern_score)
            details["pattern_match_score"] = pattern_score

        # Check docstrings (Python)
        if self.require_docstrings and self.language == "python":
            has_docstring = '"""' in code or "'''" in code
            score_components.append(1.0 if has_docstring else 0.5)
            details["has_docstring"] = has_docstring

        # Check type hints (Python)
        if self.require_type_hints and self.language == "python":
            has_hints = re.search(r"def \w+\([^)]*:\s*\w+", code) is not None
            score_components.append(1.0 if has_hints else 0.5)
            details["has_type_hints"] = has_hints

        # Calculate overall score
        overall_score = sum(score_components) / len(score_components) if score_components else 0.5

        details["score"] = overall_score
        details["label"] = self._score_to_label(overall_score)

        return ProbeResult(
            input=model_output[:100] + "..." if len(model_output) > 100 else model_output,
            output=code,
            status=ResultStatus.SUCCESS if overall_score >= 0.5 else ResultStatus.ERROR,
            metadata=details,
        )

    def _check_syntax(self, code: str) -> bool:
        """Basic syntax validation.

        Args:
            code: The code to validate.

        Returns:
            True if syntax appears valid.
        """
        if self.language == "python":
            try:
                compile(code, "<string>", "exec")
                return True
            except SyntaxError:
                return False
        else:
            # For other languages, do basic bracket matching
            brackets = {"(": ")", "[": "]", "{": "}"}
            stack = []
            for char in code:
                if char in brackets:
                    stack.append(brackets[char])
                elif char in brackets.values():
                    if not stack or stack.pop() != char:
                        return False
            return len(stack) == 0

    @staticmethod
    def _score_to_label(score: float) -> str:
        """Convert score to human-readable label."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "acceptable"
        elif score >= 0.3:
            return "poor"
        else:
            return "failing"


class CodeExplanationProbe(ScoredProbe[str]):
    """Probe to test LLMs' ability to explain code.

    Evaluates explanation quality based on:
    - Coverage of key concepts
    - Accuracy of descriptions
    - Clarity and structure

    Example:
        >>> probe = CodeExplanationProbe()
        >>> code = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
        >>> result = probe.run(model, code)
    """

    default_category = ProbeCategory.CUSTOM

    def __init__(
        self,
        name: str = "CodeExplanationProbe",
        detail_level: str = "medium",
    ):
        """Initialize the code explanation probe.

        Args:
            name: Name for this probe instance.
            detail_level: Level of explanation detail (brief, medium, detailed).
        """
        super().__init__(name=name, category=ProbeCategory.CUSTOM)
        self.detail_level = detail_level

        detail_instructions = {
            "brief": "Provide a brief one-line explanation.",
            "medium": "Explain what this code does and how it works.",
            "detailed": "Provide a detailed explanation covering the purpose, "
                       "logic, and any important concepts used.",
        }

        self.prompt_template = (
            f"Explain the following code.\n"
            f"{detail_instructions.get(detail_level, detail_instructions['medium'])}\n\n"
            "Code:\n```\n{code}\n```"
        )

    def run(self, model: Any, code: Any, **kwargs: Any) -> str:
        """Run the code explanation probe.

        Args:
            model: The model to test.
            code: The code to explain (string or dict with 'code' key).
            **kwargs: Additional arguments passed to the model.

        Returns:
            The code explanation.
        """
        if isinstance(code, dict):
            code_text = code.get("code", str(code))
        else:
            code_text = str(code)

        prompt = self.prompt_template.format(code=code_text)
        return model.generate(prompt, **kwargs)

    def evaluate_single(
        self,
        model_output: str,
        reference: Any,
        **kwargs: Any,
    ) -> ProbeResult[str]:
        """Evaluate a code explanation.

        Args:
            model_output: The explanation.
            reference: Expected concepts or keywords that should be mentioned.
            **kwargs: Additional parameters.

        Returns:
            ProbeResult with evaluation.
        """
        details = {}

        # Length check
        word_count = len(model_output.split())
        expected_min_words = {"brief": 5, "medium": 20, "detailed": 50}
        min_words = expected_min_words.get(self.detail_level, 20)
        length_score = min(1.0, word_count / min_words)
        details["word_count"] = word_count

        # Check for expected concepts
        if isinstance(reference, dict):
            expected_concepts = reference.get("concepts", [])
        elif isinstance(reference, list):
            expected_concepts = reference
        else:
            expected_concepts = [str(reference)] if reference else []

        if expected_concepts:
            output_lower = model_output.lower()
            matches = sum(1 for c in expected_concepts if c.lower() in output_lower)
            concept_score = matches / len(expected_concepts)
            details["concepts_covered"] = matches
            details["total_concepts"] = len(expected_concepts)
        else:
            concept_score = 0.5  # Neutral if no reference

        # Structure check (has sections, bullet points, etc.)
        has_structure = any(
            indicator in model_output
            for indicator in ["1.", "-", "*", ":", "\n\n"]
        )
        structure_score = 1.0 if has_structure else 0.7
        details["has_structure"] = has_structure

        # Overall score
        overall_score = (length_score + concept_score + structure_score) / 3

        details["score"] = overall_score
        details["label"] = "clear" if overall_score >= 0.7 else "unclear"

        return ProbeResult(
            input=model_output[:100] + "...",
            output=model_output,
            status=ResultStatus.SUCCESS,
            metadata=details,
        )


class CodeDebugProbe(ScoredProbe[str]):
    """Probe to test LLMs' ability to debug and fix code.

    Presents buggy code and evaluates whether the model can:
    - Identify the bug(s)
    - Explain the issue
    - Provide a working fix

    Example:
        >>> probe = CodeDebugProbe()
        >>> buggy_code = "def divide(a, b): return a / b"
        >>> result = probe.run(model, {"code": buggy_code, "error": "ZeroDivisionError"})
    """

    default_category = ProbeCategory.CUSTOM

    def __init__(
        self,
        name: str = "CodeDebugProbe",
        language: str = "python",
    ):
        """Initialize the code debug probe.

        Args:
            name: Name for this probe instance.
            language: Programming language of the code.
        """
        super().__init__(name=name, category=ProbeCategory.CUSTOM)
        self.language = language

        self.prompt_template = (
            "The following {language} code has a bug. "
            "Identify the issue, explain it, and provide a corrected version.\n\n"
            "Buggy code:\n```{language}\n{code}\n```\n"
            "{error_info}"
        )

    def run(self, model: Any, data: Any, **kwargs: Any) -> str:
        """Run the code debug probe.

        Args:
            model: The model to test.
            data: Dict with 'code' and optionally 'error' keys.
            **kwargs: Additional arguments passed to the model.

        Returns:
            The model's debugging response.
        """
        if isinstance(data, dict):
            code = data.get("code", "")
            error = data.get("error", "")
        else:
            code = str(data)
            error = ""

        error_info = f"\nError message: {error}" if error else ""

        prompt = self.prompt_template.format(
            language=self.language,
            code=code,
            error_info=error_info,
        )
        return model.generate(prompt, **kwargs)

    def evaluate_single(
        self,
        model_output: str,
        reference: Any,
        **kwargs: Any,
    ) -> ProbeResult[str]:
        """Evaluate a debugging response.

        Args:
            model_output: The debugging response.
            reference: Expected fix patterns or correct code.
            **kwargs: Additional parameters.

        Returns:
            ProbeResult with evaluation.
        """
        details = {}

        # Check for explanation
        explanation_indicators = [
            "bug", "issue", "problem", "error", "fix", "because",
            "the reason", "should be", "instead of", "causes"
        ]
        has_explanation = any(
            ind in model_output.lower() for ind in explanation_indicators
        )
        details["has_explanation"] = has_explanation

        # Check for code fix
        has_code_fix = "```" in model_output or "def " in model_output
        details["has_code_fix"] = has_code_fix

        # Check for expected fix patterns
        if isinstance(reference, dict):
            fix_patterns = reference.get("fix_patterns", [])
        elif isinstance(reference, list):
            fix_patterns = reference
        else:
            fix_patterns = [str(reference)] if reference else []

        if fix_patterns:
            output_lower = model_output.lower()
            pattern_matches = sum(
                1 for p in fix_patterns if p.lower() in output_lower
            )
            fix_score = pattern_matches / len(fix_patterns)
            details["fix_patterns_found"] = pattern_matches
        else:
            fix_score = 0.5

        # Overall score
        overall_score = (
            (1.0 if has_explanation else 0.3) * 0.3 +
            (1.0 if has_code_fix else 0.3) * 0.3 +
            fix_score * 0.4
        )

        details["score"] = overall_score
        details["label"] = "fixed" if overall_score >= 0.7 else "partially_fixed"

        return ProbeResult(
            input=model_output[:100] + "...",
            output=model_output,
            status=ResultStatus.SUCCESS if overall_score >= 0.5 else ResultStatus.ERROR,
            metadata=details,
        )
