"""Code generation and evaluation probes for testing LLM programming capabilities.

This module provides specialized probes for evaluating a language model's ability
to work with source code. It includes three main probe types:

- **CodeGenerationProbe**: Tests code generation from natural language descriptions
- **CodeExplanationProbe**: Tests comprehension and explanation of existing code
- **CodeDebugProbe**: Tests debugging and bug-fixing capabilities

These probes are designed to provide quantitative scores based on syntax validity,
pattern matching, concept coverage, and other code-quality metrics.

Module Overview
---------------
The probes in this module inherit from `ScoredProbe` and implement the evaluation
pattern where:

1. A prompt is constructed from the input (task description, code, or buggy code)
2. The model generates a response
3. The response is evaluated against reference patterns or expected concepts
4. A score and detailed metadata are returned

Example: Basic Code Generation Testing
--------------------------------------
>>> from insideLLMs.probes.code import CodeGenerationProbe
>>> from insideLLMs.models import OpenAIModel
>>>
>>> # Initialize the probe and model
>>> probe = CodeGenerationProbe(language="python")
>>> model = OpenAIModel(model_name="gpt-4")
>>>
>>> # Generate code for a task
>>> task = "Write a function to calculate the factorial of a number"
>>> response = probe.run(model, task)
>>>
>>> # Evaluate the generated code
>>> result = probe.evaluate_single(
...     response,
...     reference={"patterns": ["def factorial", "return"]}
... )
>>> print(f"Score: {result.metadata['score']:.2f}")

Example: Code Explanation Testing
---------------------------------
>>> from insideLLMs.probes.code import CodeExplanationProbe
>>>
>>> probe = CodeExplanationProbe(detail_level="detailed")
>>> code = '''
... def binary_search(arr, target):
...     left, right = 0, len(arr) - 1
...     while left <= right:
...         mid = (left + right) // 2
...         if arr[mid] == target:
...             return mid
...         elif arr[mid] < target:
...             left = mid + 1
...         else:
...             right = mid - 1
...     return -1
... '''
>>> explanation = probe.run(model, code)
>>> result = probe.evaluate_single(
...     explanation,
...     reference={"concepts": ["binary", "search", "divide", "sorted"]}
... )

Example: Code Debugging Testing
-------------------------------
>>> from insideLLMs.probes.code import CodeDebugProbe
>>>
>>> probe = CodeDebugProbe(language="python")
>>> buggy_code = '''
... def find_max(numbers):
...     max_val = 0  # Bug: assumes positive numbers
...     for num in numbers:
...         if num > max_val:
...             max_val = num
...     return max_val
... '''
>>> debug_response = probe.run(
...     model,
...     {"code": buggy_code, "error": "Returns 0 for list of negative numbers"}
... )
>>> result = probe.evaluate_single(
...     debug_response,
...     reference={"fix_patterns": ["float('-inf')", "None", "numbers[0]"]}
... )

Example: Batch Evaluation with Custom Requirements
--------------------------------------------------
>>> from insideLLMs.probes.code import CodeGenerationProbe
>>>
>>> # Create a probe that enforces docstrings and type hints
>>> strict_probe = CodeGenerationProbe(
...     language="python",
...     require_docstrings=True,
...     require_type_hints=True,
... )
>>>
>>> tasks = [
...     {"task": "Write a function to reverse a string"},
...     {"task": "Write a function to check if a list is sorted"},
...     {"task": "Write a function to merge two sorted lists"},
... ]
>>>
>>> for task in tasks:
...     response = strict_probe.run(model, task)
...     result = strict_probe.evaluate_single(response, reference=None)
...     print(f"Task: {task['task'][:30]}...")
...     print(f"  Syntax valid: {result.metadata['syntax_valid']}")
...     print(f"  Has docstring: {result.metadata.get('has_docstring', 'N/A')}")
...     print(f"  Score: {result.metadata['score']:.2f}")
"""

import re
from typing import Any

from insideLLMs.probes.base import ScoredProbe
from insideLLMs.types import ProbeCategory, ProbeResult, ResultStatus


class CodeGenerationProbe(ScoredProbe[str]):
    """Probe to test LLMs' ability to generate correct code from natural language.

    This probe evaluates a model's code generation capabilities by:

    1. Constructing a prompt from a task description
    2. Asking the model to generate code in a specified language
    3. Extracting code from the response (handling markdown code blocks)
    4. Evaluating the code based on syntax, pattern matching, and style

    Evaluation Criteria
    -------------------
    The probe scores generated code based on:

    - **Syntax validity**: Whether the code compiles/parses correctly
    - **Pattern matching**: Whether expected patterns appear in the code
    - **Docstrings** (optional): Presence of documentation strings (Python)
    - **Type hints** (optional): Presence of type annotations (Python)

    Attributes
    ----------
    language : str
        The target programming language (default: "python").
    require_docstrings : bool
        Whether to penalize missing docstrings in the score.
    require_type_hints : bool
        Whether to penalize missing type hints in the score.
    prompt_template : str
        The template used to construct the code generation prompt.

    Example: Basic Usage
    --------------------
    >>> from insideLLMs.probes.code import CodeGenerationProbe
    >>> from insideLLMs.models import MockModel
    >>>
    >>> probe = CodeGenerationProbe(language="python")
    >>> model = MockModel()
    >>>
    >>> # Run the probe with a simple task
    >>> response = probe.run(model, "Write a function to add two numbers")
    >>> result = probe.evaluate_single(response, reference="def add")
    >>> print(result.metadata["syntax_valid"])
    True

    Example: Testing with Expected Patterns
    ---------------------------------------
    >>> probe = CodeGenerationProbe(language="python")
    >>>
    >>> task = "Write a function to check if a string is a palindrome"
    >>> response = probe.run(model, task)
    >>>
    >>> # Evaluate against expected patterns
    >>> result = probe.evaluate_single(
    ...     response,
    ...     reference={
    ...         "patterns": ["def", "palindrome", "return", "[::-1]"]
    ...     }
    ... )
    >>> print(f"Pattern match score: {result.metadata['pattern_match_score']:.2f}")

    Example: Enforcing Code Quality Standards
    -----------------------------------------
    >>> # Create a strict probe that requires docstrings and type hints
    >>> strict_probe = CodeGenerationProbe(
    ...     language="python",
    ...     require_docstrings=True,
    ...     require_type_hints=True,
    ... )
    >>>
    >>> response = strict_probe.run(
    ...     model,
    ...     "Write a function to calculate the GCD of two numbers"
    ... )
    >>> result = strict_probe.evaluate_single(response, reference=None)
    >>>
    >>> # Check if code quality requirements are met
    >>> print(f"Has docstring: {result.metadata.get('has_docstring', False)}")
    >>> print(f"Has type hints: {result.metadata.get('has_type_hints', False)}")
    >>> print(f"Overall score: {result.metadata['score']:.2f}")

    Example: Multi-Language Support
    -------------------------------
    >>> # Test JavaScript code generation
    >>> js_probe = CodeGenerationProbe(language="javascript")
    >>> response = js_probe.run(
    ...     model,
    ...     "Write a function to filter even numbers from an array"
    ... )
    >>> result = js_probe.evaluate_single(
    ...     response,
    ...     reference={"patterns": ["function", "filter", "return"]}
    ... )
    >>> print(f"Syntax valid: {result.metadata['syntax_valid']}")

    See Also
    --------
    CodeExplanationProbe : For testing code comprehension
    CodeDebugProbe : For testing debugging capabilities
    ScoredProbe : The base class providing scoring infrastructure
    """

    default_category = ProbeCategory.CUSTOM

    def __init__(
        self,
        name: str = "CodeGenerationProbe",
        language: str = "python",
        require_docstrings: bool = False,
        require_type_hints: bool = False,
    ):
        """Initialize the code generation probe with language and quality settings.

        Creates a new CodeGenerationProbe configured for a specific programming
        language with optional code quality requirements.

        Parameters
        ----------
        name : str, optional
            A descriptive name for this probe instance. Useful for identifying
            the probe in logs and reports. Default is "CodeGenerationProbe".
        language : str, optional
            The target programming language for code generation. This affects
            both the prompt template and syntax validation logic. Supported
            languages include "python", "javascript", "java", "go", "rust",
            and others. Default is "python".
        require_docstrings : bool, optional
            When True, the evaluation will check for docstrings in Python code
            and penalize their absence (score 0.5 instead of 1.0 for this
            component). Has no effect for non-Python languages. Default is False.
        require_type_hints : bool, optional
            When True, the evaluation will check for type hints in Python code
            and penalize their absence (score 0.5 instead of 1.0 for this
            component). Has no effect for non-Python languages. Default is False.

        Raises
        ------
        None
            This method does not raise exceptions under normal use.

        Example: Default Initialization
        -------------------------------
        >>> probe = CodeGenerationProbe()
        >>> print(probe.language)
        python
        >>> print(probe.require_docstrings)
        False

        Example: Custom Language Configuration
        --------------------------------------
        >>> probe = CodeGenerationProbe(
        ...     name="JavaScriptProbe",
        ...     language="javascript"
        ... )
        >>> print(probe.language)
        javascript

        Example: Strict Python Configuration
        ------------------------------------
        >>> probe = CodeGenerationProbe(
        ...     name="StrictPythonProbe",
        ...     language="python",
        ...     require_docstrings=True,
        ...     require_type_hints=True,
        ... )
        >>> # This probe will score code lower if missing docs or type hints

        Example: Using the Prompt Template
        ----------------------------------
        >>> probe = CodeGenerationProbe(language="rust")
        >>> print(probe.prompt_template)
        Write rust code to solve the following task. Provide only the code...
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
        """Run the code generation probe against a model.

        Constructs a code generation prompt from the task description and sends
        it to the model. The response typically contains code, possibly wrapped
        in markdown code blocks.

        Parameters
        ----------
        model : Any
            The language model to test. Must implement a `generate(prompt, **kwargs)`
            method that returns a string response.
        task : str or dict
            The code generation task. Can be:
            - A string containing the task description directly
            - A dict with a 'task' or 'description' key containing the description
        **kwargs : Any
            Additional keyword arguments passed directly to `model.generate()`.
            Common options include `temperature`, `max_tokens`, etc.

        Returns
        -------
        str
            The raw model response containing the generated code. This may include
            markdown formatting, explanatory text, or just the code itself.

        Raises
        ------
        AttributeError
            If the model does not have a `generate` method.
        Exception
            Any exceptions raised by the model's `generate` method are propagated.

        Example: Simple String Task
        ---------------------------
        >>> probe = CodeGenerationProbe(language="python")
        >>> response = probe.run(model, "Write a function to reverse a list")
        >>> print(response)
        def reverse_list(lst):
            return lst[::-1]

        Example: Task as Dictionary
        ---------------------------
        >>> probe = CodeGenerationProbe(language="python")
        >>> task = {"task": "Write a function to compute Fibonacci numbers"}
        >>> response = probe.run(model, task)

        Example: With Model Parameters
        ------------------------------
        >>> probe = CodeGenerationProbe(language="python")
        >>> response = probe.run(
        ...     model,
        ...     "Write a recursive function to flatten a nested list",
        ...     temperature=0.2,
        ...     max_tokens=500,
        ... )

        Example: Complex Task Description
        ---------------------------------
        >>> probe = CodeGenerationProbe(language="python")
        >>> task = {
        ...     "task": "Write a class that implements a binary search tree",
        ...     "description": "Include insert, search, and delete methods"
        ... }
        >>> response = probe.run(model, task)
        """
        if isinstance(task, dict):
            task_text = task.get("task", task.get("description", str(task)))
        else:
            task_text = str(task)

        prompt = self.prompt_template.format(task=task_text)
        return model.generate(prompt, **kwargs)

    def extract_code(self, response: str) -> str:
        """Extract code from a response that may contain markdown formatting.

        Intelligently parses the model's response to extract just the code,
        handling various formats including:

        - Markdown code blocks with language tags (```python ... ```)
        - Markdown code blocks without language tags (``` ... ```)
        - Raw code without any markdown formatting
        - Code mixed with explanatory text

        The extraction uses a two-stage approach:
        1. First attempts to find markdown code blocks
        2. Falls back to heuristic-based detection for raw code

        Parameters
        ----------
        response : str
            The raw model response, which may contain markdown formatting,
            explanatory text, and/or code.

        Returns
        -------
        str
            The extracted code with leading/trailing whitespace removed.
            If no code is detected, returns the original response.

        Notes
        -----
        The heuristic detection looks for common code indicators like:
        - Python: `def `, `class `, `import `, `from `, `#`
        - JavaScript: `function `, `const `, `let `, `var `
        - Java/C++: `public `, `private `, `static `
        - General: `=`, lines ending with `:`, `}`

        Example: Extract from Markdown Code Block
        -----------------------------------------
        >>> probe = CodeGenerationProbe(language="python")
        >>> response = '''Here's the function:
        ... ```python
        ... def add(a, b):
        ...     return a + b
        ... ```
        ... This function adds two numbers.'''
        >>> code = probe.extract_code(response)
        >>> print(code)
        def add(a, b):
            return a + b

        Example: Extract from Plain Response
        ------------------------------------
        >>> probe = CodeGenerationProbe(language="python")
        >>> response = '''def multiply(a, b):
        ...     return a * b'''
        >>> code = probe.extract_code(response)
        >>> print(code)
        def multiply(a, b):
            return a * b

        Example: Handle Mixed Content
        -----------------------------
        >>> probe = CodeGenerationProbe(language="javascript")
        >>> response = '''
        ... def calculate(x):
        ...     result = x * 2
        ...     return result
        ...
        ... The above function doubles the input.
        ... '''
        >>> code = probe.extract_code(response)
        >>> print(code)
        def calculate(x):
            result = x * 2
            return result

        Example: No Code Block Markers
        ------------------------------
        >>> probe = CodeGenerationProbe(language="python")
        >>> response = "import math\\ndef sqrt(x):\\n    return math.sqrt(x)"
        >>> code = probe.extract_code(response)
        >>> print("import" in code)
        True
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
        """Evaluate a single code generation result against reference criteria.

        Performs multi-criteria evaluation of generated code including:

        1. **Syntax validation**: Checks if the code compiles/parses correctly
        2. **Pattern matching**: Checks if expected patterns are present
        3. **Docstring check**: If enabled, verifies documentation exists
        4. **Type hint check**: If enabled, verifies type annotations exist

        The overall score is the average of all applicable criteria scores.

        Parameters
        ----------
        model_output : str
            The raw model response containing generated code. The code will
            be extracted from markdown formatting if present.
        reference : str, dict, or None
            The expected patterns or test cases to check against. Can be:
            - A string: treated as a single pattern to search for
            - A dict with 'patterns' key: list of patterns to check
            - None: only syntax and style checks are performed
        **kwargs : Any
            Additional evaluation parameters (currently unused but available
            for future extensibility).

        Returns
        -------
        ProbeResult[str]
            A result object containing:
            - input: Truncated model output (first 100 chars)
            - output: The extracted code
            - status: SUCCESS if score >= 0.5, ERROR otherwise
            - metadata: Dict with detailed scoring breakdown including:
                - syntax_valid: bool
                - pattern_match_score: float (if patterns provided)
                - has_docstring: bool (if require_docstrings=True)
                - has_type_hints: bool (if require_type_hints=True)
                - score: float (overall score 0.0-1.0)
                - label: str (excellent/good/acceptable/poor/failing)

        Example: Evaluate with String Pattern
        -------------------------------------
        >>> probe = CodeGenerationProbe()
        >>> code = "def factorial(n):\\n    return 1 if n <= 1 else n * factorial(n-1)"
        >>> result = probe.evaluate_single(code, reference="def factorial")
        >>> print(result.metadata["syntax_valid"])
        True
        >>> print(result.metadata["pattern_match_score"])
        1.0

        Example: Evaluate with Multiple Patterns
        ----------------------------------------
        >>> probe = CodeGenerationProbe()
        >>> code = '''
        ... def is_prime(n):
        ...     if n < 2:
        ...         return False
        ...     for i in range(2, int(n**0.5) + 1):
        ...         if n % i == 0:
        ...             return False
        ...     return True
        ... '''
        >>> result = probe.evaluate_single(
        ...     code,
        ...     reference={"patterns": ["def is_prime", "return True", "return False"]}
        ... )
        >>> print(f"Pattern score: {result.metadata['pattern_match_score']:.2f}")
        Pattern score: 1.00

        Example: Evaluate with Style Requirements
        -----------------------------------------
        >>> probe = CodeGenerationProbe(require_docstrings=True)
        >>> code_without_doc = "def add(a, b):\\n    return a + b"
        >>> result = probe.evaluate_single(code_without_doc, reference=None)
        >>> print(f"Has docstring: {result.metadata['has_docstring']}")
        Has docstring: False
        >>> print(f"Score: {result.metadata['score']:.2f}")  # Penalized score

        Example: Check Result Status
        ----------------------------
        >>> probe = CodeGenerationProbe()
        >>> invalid_code = "def broken(:\\n    pass"
        >>> result = probe.evaluate_single(invalid_code, reference=None)
        >>> print(result.status)
        ResultStatus.ERROR
        >>> print(result.metadata["syntax_valid"])
        False
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
            # expected_output could be used for exact match checking if needed
            # expected_output = reference.get("expected_output")
        elif isinstance(reference, str):
            expected_patterns = [reference]
        else:
            expected_patterns = []

        if expected_patterns:
            pattern_matches = sum(1 for p in expected_patterns if p.lower() in code.lower())
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
        """Perform basic syntax validation on the generated code.

        Validates that the code is syntactically correct using language-specific
        checks:

        - **Python**: Uses Python's `compile()` function to check for syntax errors
        - **Other languages**: Performs basic bracket/parenthesis matching

        Parameters
        ----------
        code : str
            The code string to validate.

        Returns
        -------
        bool
            True if the code appears syntactically valid, False otherwise.

        Notes
        -----
        For Python, this performs a true syntax check using the compiler.
        For other languages, this only checks bracket matching, which is a
        basic heuristic that may not catch all syntax errors.

        Example: Valid Python Syntax
        ----------------------------
        >>> probe = CodeGenerationProbe(language="python")
        >>> code = "def add(a, b):\\n    return a + b"
        >>> print(probe._check_syntax(code))
        True

        Example: Invalid Python Syntax
        ------------------------------
        >>> probe = CodeGenerationProbe(language="python")
        >>> code = "def add(a, b:\\n    return a + b"
        >>> print(probe._check_syntax(code))
        False

        Example: JavaScript Bracket Matching
        ------------------------------------
        >>> probe = CodeGenerationProbe(language="javascript")
        >>> code = "function add(a, b) { return a + b; }"
        >>> print(probe._check_syntax(code))
        True

        Example: Unbalanced Brackets
        ----------------------------
        >>> probe = CodeGenerationProbe(language="javascript")
        >>> code = "function add(a, b { return a + b; }"
        >>> print(probe._check_syntax(code))
        False
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
        """Convert a numeric score to a human-readable quality label.

        Maps continuous scores to discrete quality categories for easier
        interpretation of results.

        Parameters
        ----------
        score : float
            A numeric score in the range [0.0, 1.0].

        Returns
        -------
        str
            A quality label based on the score:
            - "excellent" for scores >= 0.9
            - "good" for scores >= 0.7
            - "acceptable" for scores >= 0.5
            - "poor" for scores >= 0.3
            - "failing" for scores < 0.3

        Example: Various Score Levels
        -----------------------------
        >>> CodeGenerationProbe._score_to_label(0.95)
        'excellent'
        >>> CodeGenerationProbe._score_to_label(0.75)
        'good'
        >>> CodeGenerationProbe._score_to_label(0.55)
        'acceptable'
        >>> CodeGenerationProbe._score_to_label(0.35)
        'poor'
        >>> CodeGenerationProbe._score_to_label(0.15)
        'failing'
        """
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
    """Probe to test LLMs' ability to understand and explain code.

    This probe evaluates a model's code comprehension by presenting code
    snippets and asking for explanations at various detail levels. The
    quality of explanations is scored based on concept coverage, length
    appropriateness, and structural organization.

    Evaluation Criteria
    -------------------
    The probe scores explanations based on:

    - **Length adequacy**: Whether the explanation meets minimum word counts
      for the specified detail level
    - **Concept coverage**: How many expected concepts/keywords are mentioned
    - **Structure**: Whether the explanation uses organizational elements
      (bullet points, numbered lists, sections)

    Attributes
    ----------
    detail_level : str
        The expected level of explanation detail. One of:
        - "brief": Minimal explanation (5+ words expected)
        - "medium": Standard explanation (20+ words expected)
        - "detailed": Comprehensive explanation (50+ words expected)
    prompt_template : str
        The template used to construct the explanation prompt.

    Example: Basic Usage
    --------------------
    >>> from insideLLMs.probes.code import CodeExplanationProbe
    >>> from insideLLMs.models import MockModel
    >>>
    >>> probe = CodeExplanationProbe(detail_level="medium")
    >>> model = MockModel()
    >>>
    >>> code = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
    >>> explanation = probe.run(model, code)
    >>> result = probe.evaluate_single(
    ...     explanation,
    ...     reference={"concepts": ["recursive", "factorial", "base case"]}
    ... )

    Example: Brief Explanations
    ---------------------------
    >>> probe = CodeExplanationProbe(detail_level="brief")
    >>> code = "sorted(items, key=lambda x: x[1])"
    >>> explanation = probe.run(model, code)
    >>> result = probe.evaluate_single(
    ...     explanation,
    ...     reference=["sort", "second element"]
    ... )
    >>> print(f"Word count: {result.metadata['word_count']}")

    Example: Detailed Explanations
    ------------------------------
    >>> probe = CodeExplanationProbe(detail_level="detailed")
    >>> complex_code = '''
    ... @lru_cache(maxsize=128)
    ... def fibonacci(n):
    ...     if n < 2:
    ...         return n
    ...     return fibonacci(n-1) + fibonacci(n-2)
    ... '''
    >>> explanation = probe.run(model, complex_code)
    >>> result = probe.evaluate_single(
    ...     explanation,
    ...     reference={
    ...         "concepts": ["memoization", "cache", "recursive", "fibonacci", "dynamic"]
    ...     }
    ... )
    >>> print(f"Concepts covered: {result.metadata['concepts_covered']}/{result.metadata['total_concepts']}")

    Example: Evaluating Structure
    -----------------------------
    >>> probe = CodeExplanationProbe(detail_level="detailed")
    >>> explanation = '''
    ... This function implements a binary search algorithm:
    ...
    ... 1. Start with the full array range
    ... 2. Calculate the middle index
    ... 3. Compare target with middle element
    ... 4. Narrow search range based on comparison
    ...
    ... Time complexity: O(log n)
    ... '''
    >>> result = probe.evaluate_single(explanation, reference=None)
    >>> print(f"Has structure: {result.metadata['has_structure']}")
    True

    See Also
    --------
    CodeGenerationProbe : For testing code generation
    CodeDebugProbe : For testing debugging capabilities
    """

    default_category = ProbeCategory.CUSTOM

    def __init__(
        self,
        name: str = "CodeExplanationProbe",
        detail_level: str = "medium",
    ):
        """Initialize the code explanation probe with detail level settings.

        Creates a new CodeExplanationProbe configured for a specific level
        of explanation detail.

        Parameters
        ----------
        name : str, optional
            A descriptive name for this probe instance. Useful for identifying
            the probe in logs and reports. Default is "CodeExplanationProbe".
        detail_level : str, optional
            The expected level of explanation detail. Determines both the
            prompt instructions and the minimum word count for scoring.
            Options are:
            - "brief": Request a one-line explanation (5+ words expected)
            - "medium": Request a standard explanation (20+ words expected)
            - "detailed": Request a comprehensive explanation (50+ words expected)
            Default is "medium".

        Raises
        ------
        None
            This method does not raise exceptions. Invalid detail levels
            fall back to "medium" behavior.

        Example: Default Initialization
        -------------------------------
        >>> probe = CodeExplanationProbe()
        >>> print(probe.detail_level)
        medium

        Example: Brief Explanations
        ---------------------------
        >>> probe = CodeExplanationProbe(detail_level="brief")
        >>> # This probe will expect shorter, more concise explanations

        Example: Detailed Explanations
        ------------------------------
        >>> probe = CodeExplanationProbe(
        ...     name="DetailedCodeExplainer",
        ...     detail_level="detailed"
        ... )
        >>> # This probe expects comprehensive explanations with 50+ words

        Example: Custom Named Probe
        ---------------------------
        >>> probe = CodeExplanationProbe(
        ...     name="AlgorithmExplainer",
        ...     detail_level="detailed"
        ... )
        >>> print(probe.name)
        AlgorithmExplainer
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
        """Run the code explanation probe against a model.

        Constructs an explanation prompt with the provided code and sends
        it to the model. The prompt includes detail-level-specific instructions.

        Parameters
        ----------
        model : Any
            The language model to test. Must implement a `generate(prompt, **kwargs)`
            method that returns a string response.
        code : str or dict
            The code to explain. Can be:
            - A string containing the code directly
            - A dict with a 'code' key containing the code
        **kwargs : Any
            Additional keyword arguments passed directly to `model.generate()`.
            Common options include `temperature`, `max_tokens`, etc.

        Returns
        -------
        str
            The model's explanation of the code.

        Raises
        ------
        AttributeError
            If the model does not have a `generate` method.
        Exception
            Any exceptions raised by the model's `generate` method are propagated.

        Example: Simple Code Explanation
        --------------------------------
        >>> probe = CodeExplanationProbe(detail_level="medium")
        >>> code = "lambda x: x * 2"
        >>> explanation = probe.run(model, code)
        >>> print("This lambda" in explanation or "function" in explanation)
        True

        Example: Code as Dictionary
        ---------------------------
        >>> probe = CodeExplanationProbe()
        >>> data = {"code": "def greet(name): return f'Hello, {name}!'"}
        >>> explanation = probe.run(model, data)

        Example: With Model Parameters
        ------------------------------
        >>> probe = CodeExplanationProbe(detail_level="detailed")
        >>> complex_code = '''
        ... async def fetch_all(urls):
        ...     async with aiohttp.ClientSession() as session:
        ...         tasks = [fetch(session, url) for url in urls]
        ...         return await asyncio.gather(*tasks)
        ... '''
        >>> explanation = probe.run(
        ...     model,
        ...     complex_code,
        ...     temperature=0.3,
        ...     max_tokens=1000,
        ... )

        Example: Brief One-Line Explanation
        -----------------------------------
        >>> probe = CodeExplanationProbe(detail_level="brief")
        >>> code = "[x**2 for x in range(10)]"
        >>> explanation = probe.run(model, code)
        >>> # Expect a short, one-line response
        """
        code_text = code.get("code", str(code)) if isinstance(code, dict) else str(code)

        prompt = self.prompt_template.format(code=code_text)
        return model.generate(prompt, **kwargs)

    def evaluate_single(
        self,
        model_output: str,
        reference: Any,
        **kwargs: Any,
    ) -> ProbeResult[str]:
        """Evaluate a code explanation against reference criteria.

        Performs multi-criteria evaluation of the explanation including:

        1. **Length check**: Verifies the explanation meets minimum word counts
        2. **Concept coverage**: Checks how many expected concepts are mentioned
        3. **Structure check**: Looks for organizational elements

        The overall score is the average of these three components.

        Parameters
        ----------
        model_output : str
            The explanation text generated by the model.
        reference : dict, list, str, or None
            The expected concepts to check for. Can be:
            - A dict with 'concepts' key: list of expected concept keywords
            - A list: treated as a list of expected concepts
            - A string: treated as a single expected concept
            - None: only length and structure are evaluated (concept score = 0.5)
        **kwargs : Any
            Additional evaluation parameters (currently unused).

        Returns
        -------
        ProbeResult[str]
            A result object containing:
            - input: Truncated explanation (first 100 chars)
            - output: The full explanation
            - status: Always SUCCESS for explanations
            - metadata: Dict with detailed scoring breakdown including:
                - word_count: int
                - concepts_covered: int (if concepts provided)
                - total_concepts: int (if concepts provided)
                - has_structure: bool
                - score: float (overall score 0.0-1.0)
                - label: "clear" or "unclear"

        Example: Evaluate with Concept List
        -----------------------------------
        >>> probe = CodeExplanationProbe()
        >>> explanation = "This function recursively calculates the factorial..."
        >>> result = probe.evaluate_single(
        ...     explanation,
        ...     reference={"concepts": ["recursive", "factorial", "multiply"]}
        ... )
        >>> print(f"Covered: {result.metadata['concepts_covered']}/{result.metadata['total_concepts']}")

        Example: Evaluate with Simple List
        ----------------------------------
        >>> probe = CodeExplanationProbe()
        >>> explanation = "Sorts the list using a lambda function as the key."
        >>> result = probe.evaluate_single(
        ...     explanation,
        ...     reference=["sort", "lambda", "key"]
        ... )
        >>> print(result.metadata["score"])

        Example: Evaluate Structure Only
        --------------------------------
        >>> probe = CodeExplanationProbe(detail_level="detailed")
        >>> explanation = '''
        ... This code does the following:
        ... 1. Initializes variables
        ... 2. Loops through the data
        ... 3. Returns the result
        ... '''
        >>> result = probe.evaluate_single(explanation, reference=None)
        >>> print(f"Has structure: {result.metadata['has_structure']}")
        True

        Example: Check Explanation Quality Label
        ----------------------------------------
        >>> probe = CodeExplanationProbe()
        >>> good_explanation = "This function implements a binary search algorithm..."
        >>> result = probe.evaluate_single(
        ...     good_explanation,
        ...     reference=["binary", "search"]
        ... )
        >>> print(result.metadata["label"])
        clear
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
            indicator in model_output for indicator in ["1.", "-", "*", ":", "\n\n"]
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
    """Probe to test LLMs' ability to debug, diagnose, and fix code issues.

    This probe evaluates a model's debugging capabilities by presenting
    buggy code (optionally with error messages) and asking the model to:

    1. Identify the bug(s) in the code
    2. Explain why the bug causes problems
    3. Provide a corrected version of the code

    The response is scored based on the presence of explanations, code fixes,
    and whether expected fix patterns appear in the output.

    Evaluation Criteria
    -------------------
    The probe scores debugging responses based on:

    - **Explanation presence**: Whether the response explains the bug (30% weight)
    - **Code fix presence**: Whether corrected code is provided (30% weight)
    - **Fix pattern matching**: Whether expected fixes are present (40% weight)

    Attributes
    ----------
    language : str
        The programming language of the code being debugged.
    prompt_template : str
        The template used to construct the debugging prompt.

    Example: Basic Usage
    --------------------
    >>> from insideLLMs.probes.code import CodeDebugProbe
    >>> from insideLLMs.models import MockModel
    >>>
    >>> probe = CodeDebugProbe(language="python")
    >>> model = MockModel()
    >>>
    >>> buggy_code = "def divide(a, b): return a / b"
    >>> response = probe.run(
    ...     model,
    ...     {"code": buggy_code, "error": "ZeroDivisionError"}
    ... )
    >>> result = probe.evaluate_single(
    ...     response,
    ...     reference={"fix_patterns": ["if b == 0", "ZeroDivisionError"]}
    ... )

    Example: Debugging Without Error Message
    -----------------------------------------
    >>> probe = CodeDebugProbe()
    >>> buggy_code = '''
    ... def find_max(numbers):
    ...     max_val = 0
    ...     for num in numbers:
    ...         if num > max_val:
    ...             max_val = num
    ...     return max_val
    ... '''
    >>> response = probe.run(model, {"code": buggy_code})
    >>> result = probe.evaluate_single(
    ...     response,
    ...     reference=["float('-inf')", "numbers[0]"]
    ... )

    Example: Debugging with Specific Error
    ---------------------------------------
    >>> probe = CodeDebugProbe(language="python")
    >>> buggy_code = '''
    ... def get_item(lst, index):
    ...     return lst[index]
    ... '''
    >>> response = probe.run(
    ...     model,
    ...     {
    ...         "code": buggy_code,
    ...         "error": "IndexError: list index out of range"
    ...     }
    ... )
    >>> result = probe.evaluate_single(
    ...     response,
    ...     reference={"fix_patterns": ["len(lst)", "try", "except", "IndexError"]}
    ... )
    >>> print(f"Has explanation: {result.metadata['has_explanation']}")
    >>> print(f"Has code fix: {result.metadata['has_code_fix']}")

    Example: Evaluating Fix Quality
    --------------------------------
    >>> probe = CodeDebugProbe()
    >>> # Model response with explanation and fix
    >>> response = '''
    ... The bug is that the function doesn't handle negative numbers.
    ... The issue is that max_val starts at 0, so negative numbers are never
    ... considered as the maximum.
    ...
    ... Here's the fix:
    ... ```python
    ... def find_max(numbers):
    ...     if not numbers:
    ...         return None
    ...     max_val = numbers[0]  # Start with first element
    ...     for num in numbers[1:]:
    ...         if num > max_val:
    ...             max_val = num
    ...     return max_val
    ... ```
    ... '''
    >>> result = probe.evaluate_single(
    ...     response,
    ...     reference={"fix_patterns": ["numbers[0]", "negative"]}
    ... )
    >>> print(f"Score: {result.metadata['score']:.2f}")
    >>> print(f"Label: {result.metadata['label']}")

    See Also
    --------
    CodeGenerationProbe : For testing code generation
    CodeExplanationProbe : For testing code comprehension
    """

    default_category = ProbeCategory.CUSTOM

    def __init__(
        self,
        name: str = "CodeDebugProbe",
        language: str = "python",
    ):
        """Initialize the code debug probe with language settings.

        Creates a new CodeDebugProbe configured for a specific programming
        language. The language affects the prompt template formatting.

        Parameters
        ----------
        name : str, optional
            A descriptive name for this probe instance. Useful for identifying
            the probe in logs and reports. Default is "CodeDebugProbe".
        language : str, optional
            The programming language of the code to be debugged. This is
            included in the prompt to help the model understand the context.
            Common values: "python", "javascript", "java", "go", "rust", etc.
            Default is "python".

        Raises
        ------
        None
            This method does not raise exceptions under normal use.

        Example: Default Initialization
        -------------------------------
        >>> probe = CodeDebugProbe()
        >>> print(probe.language)
        python

        Example: JavaScript Debugging
        -----------------------------
        >>> probe = CodeDebugProbe(
        ...     name="JSDebugger",
        ...     language="javascript"
        ... )
        >>> print(probe.language)
        javascript

        Example: Custom Named Probe
        ---------------------------
        >>> probe = CodeDebugProbe(
        ...     name="PythonBugFinder",
        ...     language="python"
        ... )
        >>> # This probe is ready to debug Python code
        >>> print("python" in probe.prompt_template)
        True

        Example: Multiple Language Probes
        ----------------------------------
        >>> probes = {
        ...     "python": CodeDebugProbe(language="python"),
        ...     "javascript": CodeDebugProbe(language="javascript"),
        ...     "go": CodeDebugProbe(language="go"),
        ... }
        >>> # Use the appropriate probe based on file extension
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
        """Run the code debug probe against a model.

        Constructs a debugging prompt with the buggy code and optional error
        message, then sends it to the model for analysis and correction.

        Parameters
        ----------
        model : Any
            The language model to test. Must implement a `generate(prompt, **kwargs)`
            method that returns a string response.
        data : dict or str
            The buggy code and optional context. Can be:
            - A dict with 'code' key (required) and 'error' key (optional)
            - A string containing the buggy code directly (no error message)
        **kwargs : Any
            Additional keyword arguments passed directly to `model.generate()`.
            Common options include `temperature`, `max_tokens`, etc.

        Returns
        -------
        str
            The model's debugging response, which should include an explanation
            of the bug and a corrected version of the code.

        Raises
        ------
        AttributeError
            If the model does not have a `generate` method.
        Exception
            Any exceptions raised by the model's `generate` method are propagated.

        Example: Debug with Error Message
        ----------------------------------
        >>> probe = CodeDebugProbe(language="python")
        >>> buggy = {"code": "def divide(a, b): return a / b", "error": "ZeroDivisionError"}
        >>> response = probe.run(model, buggy)
        >>> print("b == 0" in response or "zero" in response.lower())
        True

        Example: Debug Code String Directly
        ------------------------------------
        >>> probe = CodeDebugProbe()
        >>> buggy_code = "def greet(name): print('Hello, ' + name"
        >>> response = probe.run(model, buggy_code)
        >>> # Model should identify the missing closing parenthesis

        Example: Debug with Custom Parameters
        -------------------------------------
        >>> probe = CodeDebugProbe(language="python")
        >>> data = {
        ...     "code": '''
        ...     def fibonacci(n):
        ...         if n <= 1:
        ...             return n
        ...         return fibonacci(n-1) + fibonacci(n-2)
        ...     ''',
        ...     "error": "RecursionError: maximum recursion depth exceeded"
        ... }
        >>> response = probe.run(
        ...     model,
        ...     data,
        ...     temperature=0.2,
        ...     max_tokens=800,
        ... )

        Example: Debug Without Error Context
        -------------------------------------
        >>> probe = CodeDebugProbe()
        >>> data = {
        ...     "code": '''
        ...     def reverse_list(lst):
        ...         for i in range(len(lst)):
        ...             lst[i], lst[len(lst)-i] = lst[len(lst)-i], lst[i]
        ...         return lst
        ...     '''
        ... }
        >>> response = probe.run(model, data)
        >>> # Model should identify the off-by-one error
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
        """Evaluate a debugging response against reference criteria.

        Performs multi-criteria evaluation of the debugging response including:

        1. **Explanation check** (30% weight): Whether bug explanation keywords
           are present (e.g., "bug", "issue", "fix", "because")
        2. **Code fix check** (30% weight): Whether corrected code is provided
           (detected via code blocks or function definitions)
        3. **Fix pattern matching** (40% weight): Whether expected fix patterns
           appear in the response

        Parameters
        ----------
        model_output : str
            The debugging response generated by the model.
        reference : dict, list, str, or None
            The expected fix patterns to check for. Can be:
            - A dict with 'fix_patterns' key: list of expected patterns
            - A list: treated as a list of expected fix patterns
            - A string: treated as a single expected pattern
            - None: only explanation and code fix presence are evaluated

        **kwargs : Any
            Additional evaluation parameters (currently unused).

        Returns
        -------
        ProbeResult[str]
            A result object containing:
            - input: Truncated response (first 100 chars)
            - output: The full debugging response
            - status: SUCCESS if score >= 0.5, ERROR otherwise
            - metadata: Dict with detailed scoring breakdown including:
                - has_explanation: bool
                - has_code_fix: bool
                - fix_patterns_found: int (if patterns provided)
                - score: float (overall score 0.0-1.0)
                - label: "fixed" or "partially_fixed"

        Notes
        -----
        Explanation indicators checked: "bug", "issue", "problem", "error",
        "fix", "because", "the reason", "should be", "instead of", "causes"

        Code fix indicators: presence of markdown code blocks (```) or
        function definitions (e.g., "def ")

        Example: Evaluate with Fix Patterns
        ------------------------------------
        >>> probe = CodeDebugProbe()
        >>> response = '''
        ... The bug is that division by zero is not handled.
        ... Here's the fix:
        ... ```python
        ... def divide(a, b):
        ...     if b == 0:
        ...         raise ValueError("Cannot divide by zero")
        ...     return a / b
        ... ```
        ... '''
        >>> result = probe.evaluate_single(
        ...     response,
        ...     reference={"fix_patterns": ["if b == 0", "ValueError"]}
        ... )
        >>> print(f"Has explanation: {result.metadata['has_explanation']}")
        True
        >>> print(f"Fix patterns found: {result.metadata['fix_patterns_found']}")
        2

        Example: Evaluate with Pattern List
        ------------------------------------
        >>> probe = CodeDebugProbe()
        >>> response = "The issue is the loop goes out of bounds. Use len(arr)-1."
        >>> result = probe.evaluate_single(
        ...     response,
        ...     reference=["len(arr)", "bounds"]
        ... )
        >>> print(result.metadata["score"])

        Example: No Reference Patterns
        -------------------------------
        >>> probe = CodeDebugProbe()
        >>> response = '''
        ... The problem is that the variable is not initialized.
        ... Fix: Initialize x = 0 before the loop.
        ... '''
        >>> result = probe.evaluate_single(response, reference=None)
        >>> print(f"Has explanation: {result.metadata['has_explanation']}")
        True
        >>> # Score based only on explanation and code fix presence

        Example: Check Result Status
        ----------------------------
        >>> probe = CodeDebugProbe()
        >>> poor_response = "I don't know what's wrong."
        >>> result = probe.evaluate_single(poor_response, reference=["fix"])
        >>> print(result.status)
        ResultStatus.ERROR
        >>> print(result.metadata["label"])
        partially_fixed
        """
        details = {}

        # Check for explanation
        explanation_indicators = [
            "bug",
            "issue",
            "problem",
            "error",
            "fix",
            "because",
            "the reason",
            "should be",
            "instead of",
            "causes",
        ]
        has_explanation = any(ind in model_output.lower() for ind in explanation_indicators)
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
            pattern_matches = sum(1 for p in fix_patterns if p.lower() in output_lower)
            fix_score = pattern_matches / len(fix_patterns)
            details["fix_patterns_found"] = pattern_matches
        else:
            fix_score = 0.5

        # Overall score
        overall_score = (
            (1.0 if has_explanation else 0.3) * 0.3
            + (1.0 if has_code_fix else 0.3) * 0.3
            + fix_score * 0.4
        )

        details["score"] = overall_score
        details["label"] = "fixed" if overall_score >= 0.7 else "partially_fixed"

        return ProbeResult(
            input=model_output[:100] + "...",
            output=model_output,
            status=ResultStatus.SUCCESS if overall_score >= 0.5 else ResultStatus.ERROR,
            metadata=details,
        )
