"""Factuality Probe Module for Testing LLM Factual Accuracy.

This module provides the FactualityProbe class for systematically testing
Large Language Models (LLMs) on their ability to provide accurate factual
information. It supports categorized questions, reference answer comparison,
and automatic answer extraction from model responses.

The probe is designed to help researchers and developers:
- Assess a model's factual knowledge across different domains
- Compare model responses against verified reference answers
- Categorize and analyze factual accuracy by topic area
- Extract direct answers from verbose model responses

Examples
--------
Basic usage with a simple factual question:

    >>> from insideLLMs.probes.factuality import FactualityProbe
    >>> probe = FactualityProbe()
    >>> questions = [
    ...     {
    ...         "question": "What is the capital of France?",
    ...         "reference_answer": "Paris",
    ...         "category": "geography"
    ...     }
    ... ]
    >>> results = probe.run(model, questions)
    >>> print(results[0]["model_answer"])
    "Paris"

Testing multiple categories of factual knowledge:

    >>> questions = [
    ...     {
    ...         "question": "Who wrote Romeo and Juliet?",
    ...         "reference_answer": "William Shakespeare",
    ...         "category": "literature"
    ...     },
    ...     {
    ...         "question": "What is the chemical symbol for gold?",
    ...         "reference_answer": "Au",
    ...         "category": "science"
    ...     },
    ...     {
    ...         "question": "In what year did World War II end?",
    ...         "reference_answer": "1945",
    ...         "category": "history"
    ...     }
    ... ]
    >>> results = probe.run(model, questions)
    >>> for r in results:
    ...     print(f"{r['category']}: {r['extracted_answer']}")
    literature: William Shakespeare
    science: Au
    history: 1945

Using the probe with custom model parameters:

    >>> probe = FactualityProbe(name="CustomFactualityProbe")
    >>> results = probe.run(
    ...     model,
    ...     questions,
    ...     temperature=0.0,  # More deterministic responses
    ...     max_tokens=100    # Limit response length
    ... )

Analyzing results by category:

    >>> from collections import defaultdict
    >>> category_results = defaultdict(list)
    >>> for result in results:
    ...     category_results[result["category"]].append(result)
    >>> for category, items in category_results.items():
    ...     print(f"{category}: {len(items)} questions tested")

Notes
-----
The probe uses simple pattern matching for answer extraction. For more
sophisticated evaluation (e.g., semantic similarity, LLM-as-judge),
consider using an external evaluator.

See Also
--------
Probe : The base class for all probes
ConsistencyProbe : For testing response consistency
"""

import re
from typing import Any

from .base import Probe


class FactualityProbe(Probe):
    """Probe for testing and evaluating LLM factual accuracy.

    The FactualityProbe systematically tests a language model's ability to
    provide accurate factual information. It accepts a list of questions with
    reference answers, queries the model, and returns structured results that
    can be used for analysis and evaluation.

    This probe is particularly useful for:
    - Benchmarking model knowledge across different domains
    - Identifying knowledge gaps or areas of factual weakness
    - Comparing factual accuracy between different models
    - Tracking factual accuracy changes across model versions

    Parameters
    ----------
    name : str, optional
        A descriptive name for this probe instance.
        Default is "FactualityProbe".

    Attributes
    ----------
    name : str
        The name of this probe instance.

    Examples
    --------
    Create a basic factuality probe:

        >>> probe = FactualityProbe()
        >>> probe.name
        'FactualityProbe'

    Create a named probe for a specific test suite:

        >>> probe = FactualityProbe(name="HistoryFactsProbe")
        >>> probe.name
        'HistoryFactsProbe'

    Run a complete factuality test:

        >>> probe = FactualityProbe()
        >>> questions = [
        ...     {
        ...         "question": "What planet is known as the Red Planet?",
        ...         "reference_answer": "Mars",
        ...         "category": "astronomy"
        ...     },
        ...     {
        ...         "question": "What is the speed of light in km/s?",
        ...         "reference_answer": "299,792",
        ...         "category": "physics"
        ...     }
        ... ]
        >>> results = probe.run(model, questions)
        >>> len(results)
        2

    Analyze probe results:

        >>> for result in results:
        ...     correct = result["reference_answer"].lower() in result["model_answer"].lower()
        ...     status = "PASS" if correct else "FAIL"
        ...     print(f"[{status}] {result['question']}")
        [PASS] What planet is known as the Red Planet?
        [PASS] What is the speed of light in km/s?

    See Also
    --------
    Probe : Base class providing common probe functionality
    """

    def __init__(self, name="FactualityProbe"):
        """Initialize a new FactualityProbe instance.

        Creates a probe configured to test factual accuracy of language models.
        The probe inherits core functionality from the base Probe class and
        adds specialized methods for factual question handling and answer
        extraction.

        Parameters
        ----------
        name : str, optional
            A descriptive name for this probe instance. Useful for
            distinguishing between multiple probe configurations in
            test suites or logging output. Default is "FactualityProbe".

        Examples
        --------
        Create a default factuality probe:

            >>> probe = FactualityProbe()
            >>> probe.name
            'FactualityProbe'

        Create a probe with a custom name for specific testing:

            >>> geography_probe = FactualityProbe(name="GeographyFactsProbe")
            >>> geography_probe.name
            'GeographyFactsProbe'

        Create multiple probes for different test categories:

            >>> probes = {
            ...     "science": FactualityProbe(name="ScienceProbe"),
            ...     "history": FactualityProbe(name="HistoryProbe"),
            ...     "literature": FactualityProbe(name="LiteratureProbe")
            ... }
            >>> for domain, probe in probes.items():
            ...     print(f"{domain}: {probe.name}")
            science: ScienceProbe
            history: HistoryProbe
            literature: LiteratureProbe

        Use in a test harness:

            >>> class FactualityTestSuite:
            ...     def __init__(self, model):
            ...         self.probe = FactualityProbe(name="TestSuiteProbe")
            ...         self.model = model
            ...     def run_tests(self, questions):
            ...         return self.probe.run(self.model, questions)
        """
        super().__init__(name)

    def run(self, model, factual_questions: Any, **kwargs):
        """Execute the factuality probe on a model with the given questions.

        This method iterates through a list of factual questions, queries the
        model for each question, and collects the responses along with metadata
        for later evaluation. Each question is formatted to encourage concise,
        factual responses from the model.

        The method automatically extracts direct answers from potentially
        verbose model responses using pattern matching, making it easier to
        compare model outputs against reference answers.

        Parameters
        ----------
        model : object
            The language model to test. Must implement a `generate(prompt, **kwargs)`
            method that accepts a string prompt and returns a string response.
        factual_questions : list of dict
            A list of question dictionaries, where each dictionary contains:

            - ``'question'`` : str
                The factual question to ask the model.
            - ``'reference_answer'`` : str
                The correct/expected answer for evaluation purposes.
            - ``'category'`` : str, optional
                The category or domain of the question (e.g., 'history',
                'science', 'geography'). Defaults to 'general' if not provided.

        **kwargs : dict, optional
            Additional keyword arguments passed directly to the model's
            `generate` method. Common options include:

            - ``temperature`` : float
                Controls randomness in generation (0.0 = deterministic).
            - ``max_tokens`` : int
                Maximum number of tokens in the response.
            - ``top_p`` : float
                Nucleus sampling parameter.

        Returns
        -------
        list of dict
            A list of result dictionaries, one for each input question.
            Each dictionary contains:

            - ``'question'`` : str
                The original question that was asked.
            - ``'reference_answer'`` : str
                The expected/correct answer provided in the input.
            - ``'model_answer'`` : str
                The full response from the model.
            - ``'extracted_answer'`` : str
                A shortened version of the answer extracted using pattern
                matching, suitable for quick comparison.
            - ``'category'`` : str
                The category of the question (defaults to 'general').

        Raises
        ------
        KeyError
            If a question dictionary is missing required keys ('question'
            or 'reference_answer').
        AttributeError
            If the model does not implement a `generate` method.

        Examples
        --------
        Basic usage with a single question:

            >>> probe = FactualityProbe()
            >>> questions = [
            ...     {
            ...         "question": "What is the capital of Japan?",
            ...         "reference_answer": "Tokyo"
            ...     }
            ... ]
            >>> results = probe.run(model, questions)
            >>> print(results[0]["extracted_answer"])
            'Tokyo'

        Testing multiple questions across categories:

            >>> questions = [
            ...     {
            ...         "question": "Who painted the Mona Lisa?",
            ...         "reference_answer": "Leonardo da Vinci",
            ...         "category": "art"
            ...     },
            ...     {
            ...         "question": "What is the atomic number of carbon?",
            ...         "reference_answer": "6",
            ...         "category": "chemistry"
            ...     },
            ...     {
            ...         "question": "What year was the Declaration of Independence signed?",
            ...         "reference_answer": "1776",
            ...         "category": "history"
            ...     }
            ... ]
            >>> results = probe.run(model, questions)
            >>> for r in results:
            ...     print(f"[{r['category']}] Q: {r['question']}")
            ...     print(f"   A: {r['extracted_answer']}")
            ...     print(f"   Expected: {r['reference_answer']}")
            [art] Q: Who painted the Mona Lisa?
               A: Leonardo da Vinci
               Expected: Leonardo da Vinci
            [chemistry] Q: What is the atomic number of carbon?
               A: 6
               Expected: 6
            [history] Q: What year was the Declaration of Independence signed?
               A: 1776
               Expected: 1776

        Using model generation parameters for more deterministic output:

            >>> results = probe.run(
            ...     model,
            ...     questions,
            ...     temperature=0.0,  # Fully deterministic
            ...     max_tokens=50     # Keep answers concise
            ... )

        Evaluating results programmatically:

            >>> def evaluate_accuracy(results):
            ...     correct = 0
            ...     for r in results:
            ...         ref = r["reference_answer"].lower().strip()
            ...         extracted = r["extracted_answer"].lower().strip()
            ...         if ref in extracted or extracted in ref:
            ...             correct += 1
            ...     return correct / len(results) if results else 0.0
            ...
            >>> accuracy = evaluate_accuracy(results)
            >>> print(f"Accuracy: {accuracy:.1%}")
            Accuracy: 100.0%

        Notes
        -----
        The prompt sent to the model is formatted as:
        "Please answer this factual question accurately and concisely: {question}"

        This formatting encourages the model to provide direct, factual answers
        rather than verbose explanations or hedged responses.

        See Also
        --------
        _extract_direct_answer : Method used to extract concise answers from responses
        """
        questions: Any = factual_questions
        if isinstance(factual_questions, dict):
            if factual_questions.get("factual_questions") is not None:
                questions = factual_questions["factual_questions"]
            elif factual_questions.get("questions") is not None:
                questions = factual_questions["questions"]
            elif (
                factual_questions.get("question") is not None
                and factual_questions.get("reference_answer") is not None
            ):
                questions = [factual_questions]
            else:
                raise ValueError(
                    "FactualityProbe expects a list of question dicts, or a dict containing "
                    "'factual_questions'/'questions', or a single question dict with "
                    "'question' + 'reference_answer'."
                )

        if not isinstance(questions, list):
            raise ValueError(
                "FactualityProbe expects a list of question dicts, or a dict containing "
                "'factual_questions'/'questions', or a single question dict with "
                "'question' + 'reference_answer'."
            )

        results = []

        for item in questions:
            question = item["question"]
            reference = item["reference_answer"]
            category = item.get("category", "general")

            # Format the prompt to encourage factual responses
            prompt = f"Please answer this factual question accurately and concisely: {question}"

            # Get model response
            model_answer = model.generate(prompt, **kwargs)

            # Basic evaluation - extract the direct answer if possible
            # This is a simple implementation; more sophisticated evaluation could be added
            extracted_answer = self._extract_direct_answer(model_answer)

            result = {
                "question": question,
                "reference_answer": reference,
                "model_answer": model_answer,
                "extracted_answer": extracted_answer,
                "category": category,
            }

            results.append(result)

        return results

    def _extract_direct_answer(self, response):
        """Extract a concise, direct answer from a verbose model response.

        This method attempts to identify and extract the core answer from
        a potentially longer model response. It uses pattern matching to
        find common answer formats like "The answer is X" or "X is the answer",
        and falls back to returning the first sentence or a truncated version
        of the response if no patterns match.

        This extraction is useful for automated evaluation, as it provides
        a more normalized representation of the model's answer that can be
        more easily compared against reference answers.

        Parameters
        ----------
        response : str
            The full response text from the language model. This may contain
            explanatory text, hedging language, or multiple sentences in
            addition to the direct answer.

        Returns
        -------
        str
            The extracted answer. This will be one of:

            - The text following "the answer is" or similar patterns
            - The first sentence or fragment of the response
            - The first 100 characters followed by "..." if no pattern matches
              and the response is longer than 100 characters
            - The full response if it's 100 characters or shorter and no
              patterns match

        Examples
        --------
        Extract answer from "The answer is X" format:

            >>> probe = FactualityProbe()
            >>> response = "Based on historical records, the answer is 1776."
            >>> probe._extract_direct_answer(response)
            '1776'

        Extract answer from "The correct answer is X" format:

            >>> response = "The correct answer is Leonardo da Vinci, who painted it in the early 16th century."
            >>> probe._extract_direct_answer(response)
            'Leonardo da Vinci, who painted it in the early 16th century'

        Extract answer from "Answer:" format:

            >>> response = "Answer: Paris is the capital of France."
            >>> probe._extract_direct_answer(response)
            'Paris is the capital of France'

        Extract first sentence when no pattern matches:

            >>> response = "Tokyo is the capital of Japan. It has been the capital since 1868."
            >>> probe._extract_direct_answer(response)
            'Tokyo is the capital of Japan'

        Handle short, direct responses:

            >>> response = "42"
            >>> probe._extract_direct_answer(response)
            '42'

        Truncate long responses without patterns:

            >>> response = "This is a very long response that goes on and on " * 5
            >>> extracted = probe._extract_direct_answer(response)
            >>> len(extracted) <= 103  # 100 chars + "..."
            True

        Notes
        -----
        The patterns used for extraction (in order of priority) are:

        1. ``(?:the answer is|the correct answer is|answer:)\\s*(.*?)(?:\\.|$)``
           Matches responses containing "the answer is", "the correct answer is",
           or "answer:" followed by the answer text.

        2. ``(?:^|\\n)([^\\.]+)(?:\\.|$)``
           Matches the first sentence or line fragment.

        Pattern matching is case-insensitive, so "The Answer Is" and
        "the answer is" are treated identically.

        See Also
        --------
        run : The main method that calls this extraction function
        """
        # Look for patterns like "The answer is X" or "X is the answer"
        patterns = [
            r"(?:the answer is|the correct answer is|answer:)\s*(.*?)(?:\.|$)",
            r"(?:^|\n)([^\.]+)(?:\.|$)",  # First sentence or fragment
        ]

        for pattern in patterns:
            matches = re.search(pattern, response, re.IGNORECASE)
            if matches:
                return matches.group(1).strip()

        # If no pattern matches, return the first 100 characters
        return response[:100] + ("..." if len(response) > 100 else "")
