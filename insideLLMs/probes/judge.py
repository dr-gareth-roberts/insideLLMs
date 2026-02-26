"""LLM-as-judge scoring for probe evaluation.

This module provides components for using a language model as a judge to
evaluate another model's outputs. Instead of relying on regex matching or
exact string comparison, the judge model scores responses against a rubric
using chain-of-thought reasoning.

Classes:
    JudgeScorer: Reusable scorer that prompts a judge model to evaluate outputs.
    JudgeScoredProbe: A ScoredProbe that uses JudgeScorer in evaluate_single().

Example: Using JudgeScorer standalone
    >>> from insideLLMs.probes.judge import JudgeScorer
    >>> scorer = JudgeScorer(judge_model=judge, rubric="Is the answer factually correct?")
    >>> result = scorer.score_output(
    ...     model_output="Paris is the capital of France.",
    ...     reference="Paris",
    ...     input_data="What is the capital of France?",
    ... )
    >>> print(result["score"])  # 0-5 scale
    5
    >>> print(result["is_correct"])
    True

Example: Using JudgeScoredProbe for end-to-end evaluation
    >>> from insideLLMs.probes.judge import JudgeScoredProbe
    >>> probe = JudgeScoredProbe(
    ...     name="factuality_judge",
    ...     judge_model=judge,
    ...     rubric="Is the answer factually correct and complete?",
    ... )
    >>> result = probe.run(subject_model, "What is the capital of France?")
    >>> evaluation = probe.evaluate_single(result, "Paris", "What is the capital of France?")
    >>> print(evaluation["is_correct"])
    True

See Also:
    ScoredProbe: Base class for probes that evaluate correctness.
    FactualityProbe: Regex-based factuality evaluation (no judge model needed).
"""

import json
import re
from typing import Any, Optional

from insideLLMs.probes.base import ScoredProbe
from insideLLMs.types import ProbeCategory

_DEFAULT_RUBRIC = (
    "Evaluate whether the model's response correctly and completely answers the question. "
    "Consider factual accuracy, relevance, and completeness."
)

_JUDGE_PROMPT_TEMPLATE = """\
You are an expert evaluator. Score the following model output against the reference answer.

## Rubric
{rubric}

## Input
{input_data}

## Reference Answer
{reference}

## Model Output
{model_output}

## Instructions
1. Think step-by-step about the quality of the model output.
2. Assign a score from 0 to 5:
   - 0: Completely wrong or irrelevant
   - 1: Mostly wrong with minor correct elements
   - 2: Partially correct but with significant errors or omissions
   - 3: Roughly correct but imprecise or incomplete
   - 4: Correct with minor issues
   - 5: Fully correct and complete
3. Respond with ONLY a JSON object (no markdown fences):
{{"reasoning": "<your step-by-step reasoning>", "score": <0-5>}}"""


class JudgeScorer:
    """Uses a language model to score outputs against a rubric.

    JudgeScorer wraps a judge model and a rubric, providing a reusable
    scoring interface. It prompts the judge with the model output, reference
    answer, and rubric, then parses the structured response.

    Parameters
    ----------
    judge_model : object
        A model implementing ``generate(prompt, **kwargs) -> str``.
        This is the LLM that performs the evaluation.
    rubric : str, optional
        Evaluation criteria for the judge. Defaults to a general
        factual-accuracy rubric.
    threshold : int, optional
        Minimum score (0-5) to consider a response correct.
        Default is 4.
    judge_kwargs : dict, optional
        Extra keyword arguments passed to the judge model's ``generate``
        method (e.g., ``temperature=0.0``).

    Attributes
    ----------
    judge_model : object
        The judge model instance.
    rubric : str
        The evaluation rubric text.
    threshold : int
        Score threshold for ``is_correct``.
    judge_kwargs : dict
        Extra kwargs for the judge model.

    Examples
    --------
    Basic usage:

        >>> scorer = JudgeScorer(judge_model=my_judge)
        >>> result = scorer.score_output(
        ...     model_output="42",
        ...     reference="42",
        ...     input_data="What is 6 times 7?",
        ... )
        >>> result["is_correct"]
        True

    Custom rubric and threshold:

        >>> scorer = JudgeScorer(
        ...     judge_model=my_judge,
        ...     rubric="Does the response show clear reasoning steps?",
        ...     threshold=3,
        ... )

    See Also
    --------
    JudgeScoredProbe : Probe that uses JudgeScorer for evaluation.
    """

    def __init__(
        self,
        judge_model: Any,
        rubric: str = _DEFAULT_RUBRIC,
        threshold: int = 4,
        judge_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.judge_model = judge_model
        self.rubric = rubric
        self.threshold = threshold
        self.judge_kwargs = judge_kwargs or {}

    def score_output(
        self,
        model_output: str,
        reference: str,
        input_data: Any,
    ) -> dict[str, Any]:
        """Score a single model output using the judge model.

        Constructs a prompt from the rubric, input, reference, and model
        output, sends it to the judge model, and parses the JSON response.

        Parameters
        ----------
        model_output : str
            The response from the model being evaluated.
        reference : str
            The expected or reference answer.
        input_data : Any
            The original input/question that produced the model output.

        Returns
        -------
        dict[str, Any]
            A dictionary containing:
            - ``is_correct`` (bool): Whether score >= threshold.
            - ``score`` (int): Judge's score from 0 to 5.
            - ``reasoning`` (str): Judge's chain-of-thought reasoning.
            - ``raw_judge_response`` (str): The full judge model response.

        Examples
        --------
            >>> result = scorer.score_output("Paris", "Paris", "Capital of France?")
            >>> result["score"]
            5
            >>> result["is_correct"]
            True
        """
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            rubric=self.rubric,
            input_data=str(input_data),
            reference=str(reference),
            model_output=str(model_output),
        )

        raw_response = self.judge_model.generate(prompt, **self.judge_kwargs)
        return self._parse_judge_response(raw_response)

    def _parse_judge_response(self, response: str) -> dict[str, Any]:
        """Parse the judge model's JSON response.

        Attempts to extract a JSON object with ``reasoning`` and ``score``
        fields. Falls back to regex extraction if strict JSON parsing fails.

        Parameters
        ----------
        response : str
            The raw text response from the judge model.

        Returns
        -------
        dict[str, Any]
            Parsed result with ``is_correct``, ``score``, ``reasoning``,
            and ``raw_judge_response`` fields.
        """
        score = 0
        reasoning = ""

        # Try JSON parsing first
        try:
            # Strip markdown code fences if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```\s*$", "", cleaned)
            parsed = json.loads(cleaned)
            score = int(parsed.get("score", 0))
            reasoning = str(parsed.get("reasoning", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            # Fallback: extract score with regex
            score_match = re.search(r'"score"\s*:\s*(\d)', response)
            if score_match:
                score = int(score_match.group(1))
            reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', response)
            if reasoning_match:
                reasoning = reasoning_match.group(1)

        # Clamp score to valid range
        score = max(0, min(5, score))

        return {
            "is_correct": score >= self.threshold,
            "score": score,
            "reasoning": reasoning,
            "raw_judge_response": response,
        }


class JudgeScoredProbe(ScoredProbe[str]):
    """Probe that uses an LLM judge to evaluate model outputs.

    Combines a subject model (being tested) with a judge model (doing the
    evaluation). The subject model generates responses, and the judge model
    scores them against reference answers using a rubric.

    This probe extends ``ScoredProbe`` and integrates with the standard
    ``run_batch`` / ``score`` pipeline, so judge-scored results flow
    through the existing diff and reporting infrastructure.

    Parameters
    ----------
    name : str, optional
        Name for this probe instance. Default is ``"JudgeScoredProbe"``.
    judge_model : object
        The LLM used to evaluate outputs. Must implement
        ``generate(prompt, **kwargs) -> str``.
    rubric : str, optional
        Evaluation criteria. Default is a general factual-accuracy rubric.
    threshold : int, optional
        Minimum score (0-5) to consider correct. Default is 4.
    judge_kwargs : dict, optional
        Extra kwargs passed to the judge model.
    category : ProbeCategory, optional
        Probe category. Default is ``ProbeCategory.FACTUALITY``.

    Attributes
    ----------
    scorer : JudgeScorer
        The underlying scorer instance.

    Examples
    --------
    Basic end-to-end usage:

        >>> probe = JudgeScoredProbe(
        ...     judge_model=judge,
        ...     rubric="Is the answer factually accurate?",
        ... )
        >>> output = probe.run(subject_model, "What is the capital of France?")
        >>> evaluation = probe.evaluate_single(
        ...     output, "Paris", "What is the capital of France?"
        ... )
        >>> evaluation["is_correct"]
        True

    Batch evaluation with scoring:

        >>> dataset = [
        ...     {"input": "Capital of France?", "reference": "Paris"},
        ...     {"input": "Capital of Japan?", "reference": "Tokyo"},
        ... ]
        >>> results = probe.run_batch(subject_model, [d["input"] for d in dataset])
        >>> score = probe.score(results)
        >>> print(f"Accuracy: {score.accuracy:.0%}")

    See Also
    --------
    JudgeScorer : The underlying scoring component.
    ScoredProbe : Base class providing batch execution and scoring.
    FactualityProbe : Regex-based alternative (no judge model needed).
    """

    default_category = ProbeCategory.FACTUALITY

    def __init__(
        self,
        name: str = "JudgeScoredProbe",
        judge_model: Any = None,
        rubric: str = _DEFAULT_RUBRIC,
        threshold: int = 4,
        judge_kwargs: Optional[dict[str, Any]] = None,
        category: Optional[ProbeCategory] = None,
    ) -> None:
        super().__init__(
            name=name,
            category=category,
            description=(
                "Evaluates model outputs using an LLM judge with "
                "chain-of-thought scoring against a rubric."
            ),
        )
        if judge_model is None:
            raise ValueError(
                "JudgeScoredProbe requires a judge_model. "
                "Pass a model that implements generate(prompt, **kwargs) -> str."
            )
        self.scorer = JudgeScorer(
            judge_model=judge_model,
            rubric=rubric,
            threshold=threshold,
            judge_kwargs=judge_kwargs,
        )

    def run(self, model: Any, data: Any, **kwargs: Any) -> str:
        """Run the subject model on the given input.

        Parameters
        ----------
        model : object
            The model being evaluated (the "subject").
        data : Any
            The input prompt or question.
        **kwargs : Any
            Extra kwargs passed to the subject model's ``generate`` method.

        Returns
        -------
        str
            The subject model's response.
        """
        prompt = str(data)
        return model.generate(prompt, **kwargs)

    def evaluate_single(
        self,
        model_output: str,
        reference: Any,
        input_data: Any,
    ) -> dict[str, Any]:
        """Evaluate a single output using the judge model.

        Delegates to the ``JudgeScorer`` to get a structured evaluation
        including score, reasoning, and correctness determination.

        Parameters
        ----------
        model_output : str
            The subject model's response.
        reference : Any
            The expected/reference answer.
        input_data : Any
            The original input that produced the output.

        Returns
        -------
        dict[str, Any]
            Evaluation result with ``is_correct``, ``score``,
            ``reasoning``, and ``raw_judge_response``.
        """
        return self.scorer.score_output(
            model_output=str(model_output),
            reference=str(reference),
            input_data=input_data,
        )
