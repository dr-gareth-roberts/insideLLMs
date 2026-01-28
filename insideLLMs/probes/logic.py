"""Logic probe for testing LLM reasoning capabilities.

This module provides the LogicProbe class for evaluating language models'
ability to solve various types of logic problems. It tests zero-shot
reasoning capabilities without providing examples or training.

Overview
--------
The LogicProbe is designed to assess a model's logical reasoning abilities
across multiple dimensions. Unlike few-shot evaluations, this probe tests
the model's inherent reasoning capabilities without providing worked examples.

The probe supports testing:

- **Deductive reasoning**: "If A implies B, and A is true, what can we conclude?"
- **Mathematical logic**: Set theory, propositional logic, predicate logic
- **Syllogisms**: "All men are mortal. Socrates is a man. Is Socrates mortal?"
- **Puzzles and riddles**: River crossing, Knights and Knaves, logic grids
- **Transitive relations**: "A > B, B > C, therefore A > C"
- **Conditional reasoning**: Modus ponens, modus tollens, contraposition
- **Quantifier logic**: Universal and existential quantification

Architecture
------------
The LogicProbe inherits from ScoredProbe, providing:

1. **Input formatting**: Converts logic problems to model prompts via templates
2. **Answer extraction**: Parses model responses to extract final answers
3. **Correctness evaluation**: Compares extracted answers to references
4. **Reasoning detection**: Identifies if the model showed its work
5. **Aggregate scoring**: Computes accuracy and reasoning metrics

Attributes
----------
default_category : ProbeCategory
    Class-level default category (ProbeCategory.LOGIC) for all instances.

Examples
--------
Basic usage with a simple logic problem:

>>> from insideLLMs.probes.logic import LogicProbe
>>> probe = LogicProbe()
>>> response = probe.run(model, "If all cats are mammals, and Whiskers is a cat, is Whiskers a mammal?")
>>> print(response)
Step 1: We know that all cats are mammals.
Step 2: We know that Whiskers is a cat.
Step 3: Since Whiskers is a cat, and all cats are mammals,
        Whiskers must be a mammal.
The answer is: Yes, Whiskers is a mammal.

Using a custom prompt template for mathematical proofs:

>>> probe = LogicProbe(
...     name="MathLogicProbe",
...     prompt_template="Solve this mathematical logic problem: {problem}\\nShow all steps."
... )
>>> response = probe.run(model, "If P implies Q, and Q implies R, does P imply R?")

Evaluating model responses against expected answers:

>>> probe = LogicProbe()
>>> eval_result = probe.evaluate_single(
...     model_output="The answer is: A is greater than C",
...     reference="A > C",
...     input_data="If A > B and B > C, what is the relation between A and C?"
... )
>>> print(eval_result['is_correct'])
True

Running on a batch of problems and scoring:

>>> problems = [
...     {"problem": "If A > B and B > C, is A > C?", "answer": "yes"},
...     {"problem": "All birds can fly. Penguins are birds. Can penguins fly?", "answer": "yes"},
... ]
>>> results = probe.run_batch(model, problems)
>>> score = probe.score(results)
>>> print(f"Accuracy: {score.accuracy:.2%}")
>>> print(f"Reasoning rate: {score.custom_metrics['reasoning_rate']:.2%}")

Testing syllogistic reasoning with different prompt styles:

>>> probe = LogicProbe(
...     name="SyllogismProbe",
...     prompt_template=(
...         "Analyze this syllogism:\\n{problem}\\n\\n"
...         "Is the conclusion valid? Answer with 'valid' or 'invalid'."
...     )
... )
>>> problems = [
...     {"problem": "All A are B. All B are C. Therefore all A are C.", "answer": "valid"},
...     {"problem": "Some A are B. Some B are C. Therefore some A are C.", "answer": "invalid"},
... ]
>>> results = probe.run_batch(model, problems)

Comparing reasoning capabilities across models:

>>> models = [gpt4_model, claude_model, llama_model]
>>> probe = LogicProbe()
>>> for model in models:
...     results = probe.run_batch(model, test_problems)
...     score = probe.score(results)
...     print(f"{model.name}: {score.accuracy:.2%} accuracy, "
...           f"{score.custom_metrics['reasoning_rate']:.2%} reasoning rate")

Notes
-----
- The probe uses case-insensitive comparison for answer matching.
- Answer extraction supports multiple patterns including "The answer is...",
  "Therefore...", "Conclusion:", and falls back to the last sentence.
- Reasoning detection is heuristic-based, looking for keywords like "step",
  "because", "therefore", etc.
- The default prompt template encourages step-by-step reasoning, but models
  may still provide direct answers without explanation.

Warnings
--------
- Some logic problems (especially those involving common misconceptions)
  may have counterintuitive correct answers. Ensure your reference answers
  account for this.
- The answer extraction regex may not handle all response formats. Consider
  subclassing and overriding `_extract_final_answer` for domain-specific needs.
- Reasoning detection may have false positives for responses that use
  reasoning keywords in non-reasoning contexts.

See Also
--------
insideLLMs.probes.base.ScoredProbe : The parent class providing scoring infrastructure.
insideLLMs.probes.base.Probe : The base class for all probes.
insideLLMs.types.ProbeCategory : Enum defining probe categories.
insideLLMs.types.ProbeResult : Container for individual probe results.
insideLLMs.types.ProbeScore : Container for aggregate scoring metrics.
"""

import re
from typing import Any, Optional

from insideLLMs.probes.base import ScoredProbe
from insideLLMs.types import ProbeCategory, ProbeResult, ProbeScore, ResultStatus


class LogicProbe(ScoredProbe[str]):
    """Probe to test LLMs' zero-shot ability at logic problems.

    This probe presents logic problems to the model and evaluates
    whether it can reason through them correctly. It is designed for
    testing deductive reasoning, mathematical logic, syllogisms, and
    various puzzle-solving capabilities.

    The probe uses a customizable prompt template to format problems
    and includes built-in answer extraction to parse final answers
    from verbose model responses. It tracks both correctness and
    whether the model demonstrated step-by-step reasoning.

    Parameters
    ----------
    name : str, optional
        Human-readable name for this probe instance. Used in logs, reports,
        and when running multiple probe types. Default is "LogicProbe".
    prompt_template : str, optional
        Template string for formatting prompts. Must contain a ``{problem}``
        placeholder. If None, uses a default template that encourages
        step-by-step reasoning. Default is None.
    extract_answer : bool, optional
        Whether to extract the final answer from model responses using
        pattern matching. When True, looks for patterns like "The answer is...",
        "Therefore...", etc. When False, uses the entire response.
        Default is True.

    Attributes
    ----------
    name : str
        Name of the probe instance. Defaults to "LogicProbe".
    category : ProbeCategory
        Always ``ProbeCategory.LOGIC`` for this probe.
    prompt_template : str
        Template for formatting the problem. Contains a ``{problem}``
        placeholder that will be replaced with the actual problem text.
    extract_answer : bool
        Whether to extract a final answer from responses using pattern matching.
    default_category : ProbeCategory
        Class-level attribute set to ``ProbeCategory.LOGIC``.

    Notes
    -----
    **Answer Matching Logic:**

    The probe evaluates answers using a flexible matching strategy:

    1. Exact match (case-insensitive, whitespace-stripped)
    2. Reference answer contained within extracted answer
    3. Extracted answer contained within reference answer

    This flexibility handles cases where the model provides additional
    context around the answer (e.g., "The answer is definitely yes" matches "yes").

    **Reasoning Detection:**

    The probe detects reasoning by looking for indicator keywords:
    "step", "first", "second", "then", "therefore", "because", "since",
    "thus", "if", "let", "given". This is a heuristic and may have
    false positives/negatives.

    **Default Prompt Template:**

    The default template is::

        Solve this logic problem step by step.
        Show your reasoning, then state your final answer clearly.

        Problem: {problem}

    Examples
    --------
    **Basic instantiation and usage:**

    >>> probe = LogicProbe()
    >>> response = probe.run(model, "If A > B and B > C, what is A's relation to C?")
    >>> print(response)
    Let me solve this step by step.
    Given: A > B and B > C
    Since A is greater than B, and B is greater than C,
    by transitivity, A must be greater than C.
    The answer is: A > C

    **Using a dictionary input with problem and metadata:**

    >>> problem = {
    ...     "problem": "All roses are flowers. All flowers need water. Do roses need water?",
    ...     "category": "syllogism",
    ...     "difficulty": "easy"
    ... }
    >>> response = probe.run(model, problem)

    **Custom prompt template for specific problem types:**

    >>> probe = LogicProbe(
    ...     name="SyllogismProbe",
    ...     prompt_template=(
    ...         "Analyze this syllogism and determine if the conclusion is valid.\\n"
    ...         "Syllogism: {problem}\\n"
    ...         "Is the conclusion logically valid? Explain your reasoning."
    ...     )
    ... )
    >>> response = probe.run(model, "All A are B. All B are C. Therefore, all A are C.")

    **Evaluating correctness against a reference answer:**

    >>> probe = LogicProbe()
    >>> metrics = probe.evaluate_single(
    ...     model_output="Working through this: First, we know... Therefore, yes.",
    ...     reference="yes",
    ...     input_data="Does P follow from the premises?"
    ... )
    >>> print(metrics)
    {'is_correct': True, 'extracted_answer': 'yes', 'reference_answer': 'yes',
     'response_length': 52, 'has_reasoning': True}

    **Creating a probe for mathematical logic:**

    >>> probe = LogicProbe(
    ...     name="PropositionalLogic",
    ...     prompt_template=(
    ...         "Using propositional logic, solve:\\n{problem}\\n\\n"
    ...         "Use symbols P, Q, R for propositions. Show truth table if needed."
    ...     )
    ... )
    >>> response = probe.run(model, "Is (P AND Q) equivalent to (Q AND P)?")

    **Testing Knights and Knaves puzzles:**

    >>> probe = LogicProbe(name="KnightsAndKnaves")
    >>> puzzle = '''
    ... On an island, knights always tell the truth and knaves always lie.
    ... You meet two inhabitants A and B.
    ... A says "At least one of us is a knave."
    ... What are A and B?
    ... '''
    >>> response = probe.run(model, puzzle)
    >>> eval_result = probe.evaluate_single(
    ...     model_output=response,
    ...     reference="A is a knight, B is a knave",
    ...     input_data=puzzle
    ... )

    **Running a complete evaluation pipeline:**

    >>> from insideLLMs.types import ResultStatus
    >>> probe = LogicProbe()
    >>> dataset = [
    ...     {"problem": "If all dogs bark, and Max is a dog, does Max bark?", "answer": "yes"},
    ...     {"problem": "If it rains, the ground is wet. The ground is wet. Did it rain?", "answer": "uncertain"},
    ...     {"problem": "No fish are mammals. All dolphins are mammals. Are dolphins fish?", "answer": "no"},
    ... ]
    >>> results = probe.run_batch(model, dataset)
    >>> # Evaluate each result
    >>> for result, item in zip(results, dataset):
    ...     if result.status == ResultStatus.SUCCESS:
    ...         eval_metrics = probe.evaluate_single(
    ...             model_output=result.output,
    ...             reference=item["answer"],
    ...             input_data=item["problem"]
    ...         )
    ...         result.metadata.update(eval_metrics)
    >>> score = probe.score(results)
    >>> print(f"Accuracy: {score.accuracy:.2%}")
    >>> print(f"Reasoning rate: {score.custom_metrics['reasoning_rate']:.2%}")
    >>> print(f"Avg response length: {score.custom_metrics['avg_response_length']:.0f} chars")

    **Subclassing for domain-specific logic:**

    >>> class MathProofProbe(LogicProbe):
    ...     '''Probe specialized for mathematical proofs.'''
    ...
    ...     def __init__(self, name: str = "MathProofProbe"):
    ...         super().__init__(
    ...             name=name,
    ...             prompt_template=(
    ...                 "Prove or disprove the following mathematical statement.\\n"
    ...                 "Statement: {problem}\\n\\n"
    ...                 "Provide a rigorous proof with clear steps."
    ...             )
    ...         )
    ...
    ...     def _has_reasoning(self, response: str) -> bool:
    ...         '''Override to check for proof-specific indicators.'''
    ...         proof_indicators = ["proof", "assume", "suppose", "qed", "therefore", "hence"]
    ...         return any(ind in response.lower() for ind in proof_indicators)

    See Also
    --------
    insideLLMs.probes.base.ScoredProbe :
        The base class providing scoring infrastructure and evaluate_single interface.
    insideLLMs.probes.base.Probe :
        The fundamental base class for all probes.
    insideLLMs.types.ProbeCategory :
        Enum defining probe categories including LOGIC.
    insideLLMs.types.ProbeResult :
        Container for individual probe results.
    insideLLMs.types.ProbeScore :
        Container for aggregate scoring metrics.
    """

    default_category = ProbeCategory.LOGIC

    def __init__(
        self,
        name: str = "LogicProbe",
        prompt_template: Optional[str] = None,
        extract_answer: bool = True,
    ):
        """Initialize the logic probe with optional customization.

        Creates a new LogicProbe instance configured for testing logic
        and reasoning capabilities. The probe can be customized with
        different prompt templates for specific problem domains.

        Parameters
        ----------
        name : str, optional
            Name for this probe instance. Used for identification in logs,
            reports, and when running multiple probe types. Should be unique
            within a probe suite for clear identification. Default is "LogicProbe".
        prompt_template : str or None, optional
            Custom template for formatting prompts sent to the model. Must
            contain a ``{problem}`` placeholder that will be replaced with
            the actual logic problem text. If None, uses the default template
            that encourages step-by-step reasoning. Default is None.
        extract_answer : bool, optional
            Whether to extract a final answer from the model's response using
            pattern matching. When True, the ``_extract_final_answer`` method
            looks for patterns like "The answer is...", "Therefore...",
            "Final answer:", "Conclusion:". When False, uses the entire
            response for evaluation. Default is True.

        Raises
        ------
        KeyError
            If prompt_template is provided but doesn't contain the ``{problem}``
            placeholder (raised during ``run()`` when formatting fails).

        Notes
        -----
        The default prompt template is designed to elicit step-by-step reasoning::

            Solve this logic problem step by step.
            Show your reasoning, then state your final answer clearly.

            Problem: {problem}

        When creating custom templates, consider:

        - Including instructions for step-by-step reasoning
        - Specifying the expected answer format
        - Adding domain-specific guidance (e.g., "Use formal logic notation")

        Examples
        --------
        **Default initialization:**

        >>> probe = LogicProbe()
        >>> print(probe.name)
        LogicProbe
        >>> print("{problem}" in probe.prompt_template)
        True
        >>> print(probe.extract_answer)
        True

        **Custom name for multiple probe instances:**

        >>> deductive_probe = LogicProbe(name="DeductiveLogicProbe")
        >>> inductive_probe = LogicProbe(name="InductiveLogicProbe")
        >>> print(deductive_probe.name, inductive_probe.name)
        DeductiveLogicProbe InductiveLogicProbe

        **Custom prompt template for mathematical proofs:**

        >>> probe = LogicProbe(
        ...     name="ProofProbe",
        ...     prompt_template=(
        ...         "Prove or disprove the following statement.\\n"
        ...         "Statement: {problem}\\n"
        ...         "Provide a formal proof with clear logical steps."
        ...     )
        ... )
        >>> print("{problem}" in probe.prompt_template)
        True

        **Disabling answer extraction for open-ended problems:**

        >>> probe = LogicProbe(
        ...     name="OpenEndedLogic",
        ...     extract_answer=False
        ... )
        >>> # The full response will be used for evaluation
        >>> print(probe.extract_answer)
        False

        **Template for binary yes/no questions:**

        >>> probe = LogicProbe(
        ...     name="BinaryLogic",
        ...     prompt_template=(
        ...         "Answer the following logic question with 'yes' or 'no'.\\n"
        ...         "Question: {problem}\\n"
        ...         "Think step by step, then give your final answer."
        ...     )
        ... )

        **Template for multiple choice logic problems:**

        >>> probe = LogicProbe(
        ...     name="MultipleChoiceLogic",
        ...     prompt_template=(
        ...         "Solve this logic problem and select the correct answer.\\n\\n"
        ...         "{problem}\\n\\n"
        ...         "Explain your reasoning, then state your answer as: "
        ...         "The answer is: [A/B/C/D]"
        ...     )
        ... )

        **Probe for testing logical fallacies:**

        >>> probe = LogicProbe(
        ...     name="FallacyDetector",
        ...     prompt_template=(
        ...         "Analyze the following argument for logical fallacies.\\n\\n"
        ...         "Argument: {problem}\\n\\n"
        ...         "Identify any fallacies present and explain why they are fallacious. "
        ...         "If the argument is valid, state 'No fallacy detected'."
        ...     )
        ... )

        See Also
        --------
        run : Execute the probe on a model with a logic problem.
        _extract_final_answer : Method used when extract_answer is True.
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

        Formats the logic problem using the prompt template and sends it
        to the model for generation. Supports both string inputs and
        dictionary inputs containing problem metadata.

        Parameters
        ----------
        model : Any
            The language model to test. Must have a ``generate()`` method
            that accepts a prompt string and returns a response string.
            Typically an instance of a class implementing ModelProtocol,
            but any object with a compatible generate method works.
        logic_problem : str or dict or Any
            The logic problem to present to the model. Supported formats:

            - ``str``: A plain text logic problem that will be inserted
              directly into the prompt template.
            - ``dict``: A dictionary with 'problem' or 'question' key
              containing the problem text. Additional keys (e.g., 'category',
              'difficulty', 'answer') are ignored during generation but
              can be used for metadata tracking and evaluation.
            - Other types: Converted to string via ``str()``.

        **kwargs : Any
            Additional keyword arguments passed directly to the model's
            ``generate()`` method. Common options include:

            - ``max_tokens`` (int): Maximum tokens in the response.
            - ``temperature`` (float): Sampling temperature (0.0 = deterministic).
            - ``top_p`` (float): Nucleus sampling parameter.
            - ``stop`` (list[str]): Stop sequences to end generation.
            - ``timeout`` (float): Request timeout in seconds.

        Returns
        -------
        str
            The model's complete response to the logic problem, including
            any reasoning steps and the final answer. The format depends
            on the model and the prompt template used.

        Raises
        ------
        KeyError
            If the prompt_template contains a placeholder other than ``{problem}``
            that cannot be filled.
        AttributeError
            If the model doesn't have a ``generate()`` method.
        Exception
            Any exception raised by the model's generate method (e.g.,
            API errors, rate limits, timeouts).

        Notes
        -----
        The method handles input in the following priority order:

        1. If input is a dict, looks for 'problem' key first, then 'question'
        2. If neither key exists, falls back to empty string
        3. For non-dict inputs, converts to string directly

        The generated prompt follows the format::

            {prompt_template with {problem} replaced by actual problem text}

        Examples
        --------
        **Running with a simple string problem:**

        >>> probe = LogicProbe()
        >>> response = probe.run(
        ...     model,
        ...     "If it's raining, the ground is wet. The ground is wet. Is it raining?"
        ... )
        >>> print("therefore" in response.lower())
        True

        **Running with a dictionary containing problem and metadata:**

        >>> problem = {
        ...     "problem": "A is B's father. B is C's father. What is A to C?",
        ...     "difficulty": "medium",
        ...     "type": "family_relations"
        ... }
        >>> response = probe.run(model, problem)
        >>> print(response)
        Let me work through this step by step.
        Given: A is B's father, and B is C's father.
        This means A is the father of B, who is the father of C.
        Therefore, A is C's grandfather.
        The answer is: A is C's grandfather.

        **Passing additional generation parameters:**

        >>> response = probe.run(
        ...     model,
        ...     "If P then Q. If Q then R. P is true. What can we conclude?",
        ...     max_tokens=500,
        ...     temperature=0.0
        ... )

        **Using the 'question' key instead of 'problem':**

        >>> problem = {"question": "Is the set of even numbers infinite?"}
        >>> response = probe.run(model, problem)

        **Running with deterministic settings for reproducibility:**

        >>> response = probe.run(
        ...     model,
        ...     "All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?",
        ...     temperature=0.0,
        ...     max_tokens=200
        ... )

        **Testing with a complex multi-step problem:**

        >>> problem = '''
        ... There are three boxes: A, B, and C.
        ... One contains gold, one contains silver, one contains nothing.
        ... Box A says: "The gold is in box B."
        ... Box B says: "The gold is not in this box."
        ... Box C says: "The gold is in this box."
        ... Only one statement is true. Where is the gold?
        ... '''
        >>> response = probe.run(model, problem.strip())

        **Using with a model wrapper that has custom parameters:**

        >>> response = probe.run(
        ...     model,
        ...     "If A > B and B > C and C > D, is A > D?",
        ...     model_specific_param="value",
        ...     another_param=42
        ... )

        See Also
        --------
        run_batch : Run the probe on multiple problems.
        evaluate_single : Evaluate the response against a reference answer.
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
        """Evaluate a single logic problem response against a reference answer.

        Extracts the final answer from the model's response and compares it
        to the reference answer. Also analyzes whether the response contains
        step-by-step reasoning.

        The comparison is case-insensitive and supports three matching modes:

        1. Exact match (after lowercasing and stripping whitespace)
        2. Reference contained in extracted answer (word-boundary match)
        3. Extracted answer contained in reference (word-boundary match)

        Parameters
        ----------
        model_output : str
            The model's complete response to the logic problem, typically
            including reasoning steps and a final answer. The method will
            attempt to extract the final answer using pattern matching.
        reference : Any
            The expected/correct answer. Will be converted to string via
            ``str()`` and compared case-insensitively. Can be None if no
            reference is available, in which case evaluation is skipped.
        input_data : Any
            The original logic problem that was presented to the model.
            Currently not used in the base evaluation but available for
            context in subclasses that may need problem-specific evaluation.

        Returns
        -------
        dict[str, Any]
            A dictionary containing evaluation metrics. The contents depend
            on whether a reference was provided:

            **When reference is None:**

            - ``evaluated`` (bool): Always False, indicating no evaluation.

            **When reference is provided:**

            - ``is_correct`` (bool): True if the extracted answer matches
              the reference using the flexible matching strategy.
            - ``extracted_answer`` (str): The answer extracted from the model
              output using pattern matching, lowercased and stripped.
            - ``reference_answer`` (str): The reference answer, lowercased
              and stripped.
            - ``response_length`` (int): Character count of the full model output.
            - ``has_reasoning`` (bool): True if the response contains
              reasoning indicator keywords.

        Notes
        -----
        **Matching Strategy:**

        The flexible matching handles common variations in answer format:

        - "yes" matches "Yes", "YES", "yes"
        - "yes" matches "The answer is yes"
        - "A > C" matches "A is greater than C" (via containment)

        Word-boundary containment helps avoid substring false positives
        (e.g., "no" no longer matches "know"), but it is still intentionally
        permissive. Consider subclassing and overriding for domain-specific
        matching.

        **Answer Extraction:**

        The ``_extract_final_answer`` method looks for patterns like:

        - "The answer is X" / "The final answer is X"
        - "Therefore, X" / "Therefore: X"
        - "Answer: X" / "Final answer: X"
        - "Conclusion: X" / "Result: X"

        If no pattern matches, returns the last sentence.

        Examples
        --------
        **Evaluating a correct response with clear answer:**

        >>> probe = LogicProbe()
        >>> result = probe.evaluate_single(
        ...     model_output="Step 1: Given A > B and B > C. "
        ...                  "Step 2: By transitivity, A > C. "
        ...                  "The answer is: A is greater than C.",
        ...     reference="A > C",
        ...     input_data="If A > B and B > C, compare A and C."
        ... )
        >>> print(result['is_correct'])
        True
        >>> print(result['has_reasoning'])
        True

        **Evaluating when no reference is provided:**

        >>> result = probe.evaluate_single(
        ...     model_output="The answer is 42.",
        ...     reference=None,
        ...     input_data="What is the meaning of life?"
        ... )
        >>> print(result)
        {'evaluated': False}

        **Evaluating a response where answer is contained:**

        >>> result = probe.evaluate_single(
        ...     model_output="Therefore, the answer is definitely yes, it is true.",
        ...     reference="yes",
        ...     input_data="Is P true?"
        ... )
        >>> print(result['is_correct'])
        True
        >>> print("yes" in result['extracted_answer'])
        True

        **Evaluating an incorrect response:**

        >>> result = probe.evaluate_single(
        ...     model_output="I think the answer is no.",
        ...     reference="yes",
        ...     input_data="Is the sky blue?"
        ... )
        >>> print(result['is_correct'])
        False

        **Evaluating response without explicit answer pattern:**

        >>> result = probe.evaluate_single(
        ...     model_output="After careful analysis, the conclusion must be affirmative.",
        ...     reference="yes",
        ...     input_data="Is the statement true?"
        ... )
        >>> # Falls back to last sentence, may not match
        >>> print(result['is_correct'])
        False

        **Using evaluation results for analysis:**

        >>> results = []
        >>> for problem in test_problems:
        ...     response = probe.run(model, problem)
        ...     eval_result = probe.evaluate_single(
        ...         model_output=response,
        ...         reference=problem['answer'],
        ...         input_data=problem['problem']
        ...     )
        ...     results.append(eval_result)
        >>> accuracy = sum(r['is_correct'] for r in results) / len(results)
        >>> reasoning_rate = sum(r['has_reasoning'] for r in results) / len(results)
        >>> avg_length = sum(r['response_length'] for r in results) / len(results)

        **Handling edge cases:**

        >>> # Empty response
        >>> result = probe.evaluate_single(
        ...     model_output="",
        ...     reference="yes",
        ...     input_data="Question?"
        ... )
        >>> print(result['is_correct'])
        False

        >>> # Response matches reference exactly
        >>> result = probe.evaluate_single(
        ...     model_output="yes",
        ...     reference="yes",
        ...     input_data="Simple question?"
        ... )
        >>> print(result['is_correct'])
        True

        See Also
        --------
        _extract_final_answer : Method that extracts the answer from responses.
        _has_reasoning : Method that detects reasoning in responses.
        score : Aggregates evaluation results into metrics.
        """
        if reference is None:
            return {"evaluated": False}

        ref_answer = str(reference).lower().strip()
        extracted = self._extract_final_answer(model_output).lower().strip()

        def _contains_at_word_boundary(needle: str, haystack: str) -> bool:
            """Return True if needle appears in haystack at word boundaries.

            This avoids substring false positives like "no" matching "know".
            """
            if not needle or not haystack:
                return False
            pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
            return re.search(pattern, haystack) is not None

        # Check for exact match or word-boundary containment.
        is_correct = (
            ref_answer == extracted
            or _contains_at_word_boundary(ref_answer, extracted)
            or _contains_at_word_boundary(extracted, ref_answer)
        )

        return {
            "is_correct": is_correct,
            "extracted_answer": extracted,
            "reference_answer": ref_answer,
            "response_length": len(model_output),
            "has_reasoning": self._has_reasoning(model_output),
        }

    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from a model's response.

        Uses regex pattern matching to identify and extract the final answer
        from verbose model responses that include reasoning steps. Falls back
        to returning the last sentence if no standard answer pattern is found.

        The method searches for common answer-indicating patterns in order:

        1. "The answer is X" / "The final answer is X"
        2. "Therefore, X" / "Therefore: X"
        3. "Answer: X" / "Final answer: X"
        4. "Conclusion: X" / "Result: X"

        If none of these patterns match, returns the last non-empty sentence.

        Parameters
        ----------
        response : str
            The complete model response to parse, typically containing
            reasoning steps followed by a final answer. Can be empty or
            contain no sentences.

        Returns
        -------
        str
            The extracted answer text, stripped of surrounding whitespace.
            The extraction follows this priority:

            1. If a pattern matches, returns the captured group (text after
               the pattern marker, up to the first period or end of string).
            2. If no pattern matches but response has sentences (split by "."),
               returns the last non-empty sentence.
            3. If the last sentence is empty but there are more sentences,
               returns the second-to-last sentence.
            4. If response has no periods, returns up to the first 100 characters.

        Notes
        -----
        **Pattern Details:**

        - All patterns are case-insensitive (``re.IGNORECASE``).
        - Patterns capture text up to the first period or end of string.
        - The "answer is" pattern handles both "The answer is" and
          "The final answer is".
        - Colons and commas after keywords are handled flexibly.

        **Regex Patterns Used:**

        .. code-block:: python

            patterns = [
                r"(?:the\\s+)?(?:final\\s+)?answer\\s+is[:\\s]+(.+?)(?:\\.|$)",
                r"therefore[,:\\s]+(.+?)(?:\\.|$)",
                r"(?:final\\s+)?answer[:\\s]+(.+?)(?:\\.|$)",
                r"(?:conclusion|result)[:\\s]+(.+?)(?:\\.|$)",
            ]

        **Limitations:**

        - Cannot handle answers that span multiple sentences.
        - May capture too much if the answer contains a period (e.g.,
          "The answer is 3.14" would extract only "3").
        - The last-sentence fallback may return irrelevant text if the
          model's answer is not at the end.

        Examples
        --------
        **Extracting from "The answer is" pattern:**

        >>> probe = LogicProbe()
        >>> answer = probe._extract_final_answer(
        ...     "Let me think... Given the premises... The answer is yes."
        ... )
        >>> print(answer)
        yes

        **Extracting from "Therefore" pattern:**

        >>> answer = probe._extract_final_answer(
        ...     "We know A implies B. A is true. Therefore, B is true."
        ... )
        >>> print(answer)
        B is true

        **Extracting from "Final answer:" pattern:**

        >>> answer = probe._extract_final_answer(
        ...     "Step 1: ... Step 2: ... Final answer: 42"
        ... )
        >>> print(answer)
        42

        **Extracting from "Conclusion:" pattern:**

        >>> answer = probe._extract_final_answer(
        ...     "After analyzing... Conclusion: the hypothesis is false"
        ... )
        >>> print(answer)
        the hypothesis is false

        **Fallback to last sentence when no pattern matches:**

        >>> answer = probe._extract_final_answer(
        ...     "This is complex. Let me analyze. The result must be X."
        ... )
        >>> print(answer)
        The result must be X

        **Handling empty last sentence (trailing period):**

        >>> answer = probe._extract_final_answer(
        ...     "First point. Second point. The answer is clearly Z."
        ... )
        >>> print(answer)
        Z

        **Handling response with no periods:**

        >>> answer = probe._extract_final_answer(
        ...     "Yes, that is correct"
        ... )
        >>> # Returns entire response (up to 100 chars) as fallback
        >>> print(answer)
        Yes, that is correct

        **Handling empty response:**

        >>> answer = probe._extract_final_answer("")
        >>> print(answer)
        <empty string>

        **Case insensitivity:**

        >>> answer = probe._extract_final_answer(
        ...     "THE ANSWER IS: definitely yes"
        ... )
        >>> print(answer)
        definitely yes

        **Multiple answer patterns (first wins):**

        >>> answer = probe._extract_final_answer(
        ...     "The answer is A. Therefore, B. Conclusion: C."
        ... )
        >>> # "The answer is" pattern matches first
        >>> print(answer)
        A

        See Also
        --------
        evaluate_single : Uses this method to extract answers for evaluation.
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
        """Check if the response contains step-by-step reasoning indicators.

        Analyzes the model's response to determine if it demonstrates
        explicit reasoning rather than just providing a direct answer.
        This is useful for evaluating whether a model "shows its work"
        when solving logic problems.

        The method looks for common reasoning indicator words that suggest
        the model is walking through logical steps or explaining its
        thought process.

        Parameters
        ----------
        response : str
            The model's response to analyze for reasoning patterns.
            Can be empty, in which case False is returned.

        Returns
        -------
        bool
            True if any reasoning indicator is found in the response
            (case-insensitive), False otherwise.

        Notes
        -----
        **Reasoning Indicators:**

        The following keywords are checked (case-insensitive):

        - **Sequencing words**: "step", "first", "second"
        - **Logical connectives**: "then", "therefore", "thus"
        - **Causal words**: "because", "since"
        - **Conditional/setup words**: "if", "let", "given"

        **Heuristic Limitations:**

        This is a simple keyword-based heuristic with known limitations:

        - **False positives**: Responses containing these words in non-reasoning
          contexts will incorrectly return True. Examples:

          - "If you have questions, let me know" (contains "if", "let")
          - "Since 2020, the policy has changed" (contains "since")
          - "The first item on the list is..." (contains "first")

        - **False negatives**: Complex reasoning that doesn't use these
          specific words will incorrectly return False. Examples:

          - "This follows from the premises by modus ponens."
          - "Applying the rule, we get X."
          - "The logical conclusion is Y."

        **Alternative Approaches:**

        For more accurate reasoning detection, consider:

        - Using sentence structure analysis
        - Training a classifier on reasoning vs. non-reasoning text
        - Requiring multiple indicators for positive detection
        - Using LLM-based reasoning detection

        Examples
        --------
        **Response with clear step-by-step reasoning:**

        >>> probe = LogicProbe()
        >>> probe._has_reasoning(
        ...     "First, let's identify the premises. "
        ...     "Since A implies B, and A is true, "
        ...     "therefore B must be true."
        ... )
        True

        **Response with direct answer only:**

        >>> probe._has_reasoning("The answer is yes.")
        False

        **Response with causal reasoning:**

        >>> probe._has_reasoning(
        ...     "Because all mammals are warm-blooded, and dogs are mammals, "
        ...     "dogs are warm-blooded."
        ... )
        True

        **Response with conditional setup:**

        >>> probe._has_reasoning(
        ...     "Let x be the number of apples. Given x > 5, we can conclude..."
        ... )
        True

        **Empty response:**

        >>> probe._has_reasoning("")
        False

        **Response with "therefore" in conclusion:**

        >>> probe._has_reasoning("Therefore, the statement is false.")
        True

        **Response with "thus" as logical connector:**

        >>> probe._has_reasoning("A implies B. A is true. Thus, B is true.")
        True

        **False positive example (non-reasoning "if"):**

        >>> probe._has_reasoning("If you need more help, just ask.")
        True  # False positive - not actual reasoning

        **Response using numbered steps:**

        >>> probe._has_reasoning(
        ...     "Step 1: Identify the premises. "
        ...     "Step 2: Apply the rule. "
        ...     "Step 3: Draw the conclusion."
        ... )
        True

        **Response with mathematical notation but no indicators:**

        >>> probe._has_reasoning("P -> Q. P. Therefore Q. QED.")
        True  # "therefore" is detected

        **Complex reasoning without indicators:**

        >>> probe._has_reasoning(
        ...     "Applying modus ponens to the premises yields the result."
        ... )
        False  # No indicator keywords present

        See Also
        --------
        evaluate_single : Uses this method to check for reasoning.
        score : Aggregates reasoning rate across multiple results.
        """
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
        """Calculate aggregate scores for a batch of logic probe results.

        Computes accuracy and additional metrics from a collection of probe
        results. Extends the base class scoring with logic-specific metrics
        like reasoning rate and average response length.

        Parameters
        ----------
        results : list[ProbeResult[str]]
            A list of ProbeResult objects from running the logic probe on
            multiple problems. Each result should have:

            - ``status``: ResultStatus indicating success/failure
            - ``output``: The model's response string (for successful results)
            - ``metadata``: Optional dict with evaluation data (e.g., is_correct)

        Returns
        -------
        ProbeScore
            A score object containing:

            - ``accuracy`` (float): The base accuracy from the parent class,
              representing the proportion of correct answers (based on
              metadata['is_correct'] if present, otherwise success rate).
            - ``error_rate`` (float): Proportion of results with ERROR status.
            - ``mean_latency_ms`` (float or None): Average latency of successful
              results.
            - ``custom_metrics`` (dict): Additional metrics specific to logic
              probing:

              - ``reasoning_rate`` (float): Proportion of successful responses
                that contain reasoning indicators (0.0 to 1.0). Calculated by
                checking each response with ``_has_reasoning()``.
              - ``avg_response_length`` (float): Average character count of
                successful responses. Useful for comparing verbosity across
                models or configurations.

        Notes
        -----
        **Metric Calculation Details:**

        - **reasoning_rate**: ``(responses with reasoning) / (successful responses)``
        - **avg_response_length**: ``(total chars in successful responses) / (successful responses)``

        **Handling Edge Cases:**

        - If there are no successful results, both custom metrics will be 0.
        - Results with ``status != ResultStatus.SUCCESS`` are excluded from
          reasoning rate and average length calculations.
        - The base accuracy is computed by the parent ``ScoredProbe`` class,
          which uses ``metadata['is_correct']`` if available.

        **Metric Interpretation:**

        - High reasoning_rate (>0.8): Model consistently shows its work
        - Low reasoning_rate (<0.3): Model often gives direct answers
        - High avg_response_length: Model provides verbose explanations
        - Low avg_response_length: Model is concise (may lack reasoning)

        Examples
        --------
        **Scoring a batch of results:**

        >>> from insideLLMs.types import ProbeResult, ResultStatus
        >>> probe = LogicProbe()
        >>> results = [
        ...     ProbeResult(
        ...         status=ResultStatus.SUCCESS,
        ...         output="First, we note that... Therefore, yes.",
        ...         input="Is A > C?",
        ...         metadata={"is_correct": True}
        ...     ),
        ...     ProbeResult(
        ...         status=ResultStatus.SUCCESS,
        ...         output="The answer is no.",
        ...         input="Is B > A?",
        ...         metadata={"is_correct": False}
        ...     ),
        ...     ProbeResult(
        ...         status=ResultStatus.ERROR,
        ...         output=None,
        ...         input="Complex problem",
        ...         error="Timeout"
        ...     ),
        ... ]
        >>> score = probe.score(results)
        >>> print(f"Reasoning rate: {score.custom_metrics['reasoning_rate']:.1%}")
        Reasoning rate: 50.0%

        **Interpreting the metrics:**

        >>> score = probe.score(results)
        >>> if score.custom_metrics['reasoning_rate'] < 0.5:
        ...     print("Model often skips reasoning steps")
        ... else:
        ...     print("Model shows reasoning in most responses")
        Model shows reasoning in most responses

        **Comparing response verbosity across models:**

        >>> score_model_a = probe.score(results_a)
        >>> score_model_b = probe.score(results_b)
        >>> print(f"Model A avg length: {score_model_a.custom_metrics['avg_response_length']:.0f}")
        >>> print(f"Model B avg length: {score_model_b.custom_metrics['avg_response_length']:.0f}")

        **Handling empty results:**

        >>> score = probe.score([])
        >>> print(score.custom_metrics)
        {'reasoning_rate': 0, 'avg_response_length': 0}

        **Complete evaluation workflow:**

        >>> probe = LogicProbe()
        >>> dataset = load_logic_problems()  # Your dataset
        >>> results = probe.run_batch(model, dataset)
        >>>
        >>> # Evaluate each result
        >>> for result, item in zip(results, dataset):
        ...     if result.status == ResultStatus.SUCCESS:
        ...         eval_metrics = probe.evaluate_single(
        ...             model_output=result.output,
        ...             reference=item.get("answer"),
        ...             input_data=item
        ...         )
        ...         result.metadata.update(eval_metrics)
        >>>
        >>> # Calculate scores
        >>> score = probe.score(results)
        >>> print(f"Accuracy: {score.accuracy:.2%}")
        >>> print(f"Error rate: {score.error_rate:.2%}")
        >>> print(f"Reasoning rate: {score.custom_metrics['reasoning_rate']:.2%}")
        >>> print(f"Avg response length: {score.custom_metrics['avg_response_length']:.0f}")

        **Analyzing results by reasoning presence:**

        >>> score = probe.score(results)
        >>> if score.custom_metrics['reasoning_rate'] > 0.7:
        ...     print("Model shows strong chain-of-thought reasoning")
        >>> elif score.custom_metrics['reasoning_rate'] > 0.4:
        ...     print("Model sometimes shows reasoning")
        >>> else:
        ...     print("Model rarely shows explicit reasoning")

        **Correlation between reasoning and correctness:**

        >>> correct_with_reasoning = sum(
        ...     1 for r in results
        ...     if r.status == ResultStatus.SUCCESS
        ...     and r.metadata.get('is_correct')
        ...     and r.metadata.get('has_reasoning')
        ... )
        >>> correct_without_reasoning = sum(
        ...     1 for r in results
        ...     if r.status == ResultStatus.SUCCESS
        ...     and r.metadata.get('is_correct')
        ...     and not r.metadata.get('has_reasoning')
        ... )
        >>> print(f"Correct with reasoning: {correct_with_reasoning}")
        >>> print(f"Correct without reasoning: {correct_without_reasoning}")

        See Also
        --------
        insideLLMs.probes.base.ScoredProbe.score :
            The parent method that computes base accuracy and error rate.
        _has_reasoning : Method used to detect reasoning in responses.
        evaluate_single : Method that populates metadata with is_correct.
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
