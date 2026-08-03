"""Chain-of-thought evaluation and reporting."""

from typing import Optional

from insideLLMs.contrib._reasoning.analysis import ReasoningAnalyzer
from insideLLMs.contrib._reasoning.extraction import ReasoningExtractor
from insideLLMs.contrib._reasoning.models import (
    ChainAnalysis,
    CoTEvaluation,
    ReasoningChain,
    ReasoningReport,
)


class CoTEvaluator:
    """
    Evaluates Chain-of-Thought responses from language models.

    Provides comprehensive evaluation of model responses to reasoning tasks,
    including extraction of reasoning chains, quality assessment, answer
    verification, and improvement suggestions.

    Attributes
    ----------
    extractor : ReasoningExtractor
        Instance used to extract reasoning chains from responses.
    analyzer : ReasoningAnalyzer
        Instance used to analyze extracted chains.

    Examples
    --------
    Basic evaluation:

        >>> from insideLLMs.contrib.reasoning import CoTEvaluator
        >>> evaluator = CoTEvaluator()
        >>> prompt = "What is 5 + 7?"
        >>> response = "Step 1: Add 5 and 7. Step 2: 5 + 7 = 12. The answer is 12."
        >>> evaluation = evaluator.evaluate(prompt, response, "12")
        >>> print(f"Correct: {evaluation.answer_correct}")
        Correct: True

    Batch evaluation:

        >>> evaluator = CoTEvaluator()
        >>> prompts = ["2+2?", "3*4?"]
        >>> responses = ["2+2=4", "Step 1: 3*4. Step 2: =12"]
        >>> results = evaluator.evaluate_batch(prompts, responses, ["4", "12"])
        >>> print(f"Evaluated {len(results)} responses")
        Evaluated 2 responses

    Generating reports:

        >>> evaluator = CoTEvaluator()
        >>> # After batch evaluation
        >>> report = evaluator.generate_report(results)
        >>> print(f"Avg score: {report.avg_reasoning_score:.2f}")
        Avg score: 0.52

    Checking improvements:

        >>> evaluator = CoTEvaluator()
        >>> eval = evaluator.evaluate("Question", "Brief answer.")
        >>> for imp in eval.improvements:
        ...     print(f"Improve: {imp}")
        Improve: Strengthen logical connections between steps
        Improve: Add missing steps to complete the reasoning chain

    See Also
    --------
    CoTEvaluation : Evaluation result structure
    ReasoningReport : Aggregated report structure
    evaluate_cot : Convenience function
    """

    def __init__(self):
        """
        Initialize the CoT evaluator with extraction and analysis components.

        Creates instances of ReasoningExtractor and ReasoningAnalyzer for
        use in evaluation.

        Examples
        --------
            >>> from insideLLMs.contrib.reasoning import CoTEvaluator
            >>> evaluator = CoTEvaluator()
            >>> print(type(evaluator.extractor).__name__)
            ReasoningExtractor
            >>> print(type(evaluator.analyzer).__name__)
            ReasoningAnalyzer
        """
        self.extractor = ReasoningExtractor()
        self.analyzer = ReasoningAnalyzer()

    def evaluate(
        self,
        prompt: str,
        response: str,
        expected_answer: Optional[str] = None,
    ) -> CoTEvaluation:
        """
        Evaluate a single Chain-of-Thought response.

        Extracts the reasoning chain, analyzes its quality, checks answer
        correctness (if expected answer provided), and generates improvement
        suggestions.

        Parameters
        ----------
        prompt : str
            The original prompt or question.
        response : str
            The model's response to evaluate.
        expected_answer : Optional[str]
            The expected correct answer for verification. If None,
            answer correctness will not be checked.

        Returns
        -------
        CoTEvaluation
            Complete evaluation results including:
            - chain: Extracted reasoning chain
            - answer_correct: Whether answer matches expected
            - reasoning_score: Overall reasoning quality (0-1)
            - step_accuracy: Average step quality (0-1)
            - explanation_quality: How well explained (0-1)
            - improvements: Suggested improvements

        Examples
        --------
        Basic evaluation with expected answer:

            >>> from insideLLMs.contrib.reasoning import CoTEvaluator
            >>> evaluator = CoTEvaluator()
            >>> result = evaluator.evaluate(
            ...     "What is 10/2?",
            ...     "10 divided by 2 equals 5. The answer is 5.",
            ...     "5"
            ... )
            >>> print(f"Correct: {result.answer_correct}")
            Correct: True

        Evaluation without expected answer:

            >>> result = evaluator.evaluate(
            ...     "Explain photosynthesis",
            ...     "Plants use sunlight to convert CO2 and water into glucose."
            ... )
            >>> print(f"Answer check: {result.answer_correct}")
            Answer check: None

        Accessing detailed scores:

            >>> result = evaluator.evaluate("Q", "Given A. Therefore B.")
            >>> print(f"Reasoning: {result.reasoning_score:.2f}")
            Reasoning: 0.58
            >>> print(f"Step accuracy: {result.step_accuracy:.2f}")
            Step accuracy: 0.65

        Getting improvement suggestions:

            >>> result = evaluator.evaluate("Q", "Maybe answer.")
            >>> print(result.improvements)
            ['Strengthen logical connections between steps', 'Add missing steps to complete the reasoning chain']
        """
        # Extract reasoning chain
        chain = self.extractor.extract(response)

        # Analyze chain
        analysis = self.analyzer.analyze(chain)

        # Check answer correctness
        answer_correct = None
        if expected_answer:
            answer_correct = self._check_answer(response, expected_answer)

        # Calculate reasoning score
        reasoning_score = (
            analysis.logical_validity * 0.4
            + analysis.coherence_score * 0.3
            + analysis.completeness_score * 0.3
        )

        # Calculate step accuracy
        step_accuracy = (
            sum(analysis.step_quality_scores) / len(analysis.step_quality_scores)
            if analysis.step_quality_scores
            else 0.0
        )

        # Calculate explanation quality
        explanation_quality = self._assess_explanation_quality(response, chain)

        # Generate improvements
        improvements = self._suggest_improvements(analysis)

        return CoTEvaluation(
            prompt=prompt,
            response=response,
            chain=chain,
            answer_correct=answer_correct,
            reasoning_score=reasoning_score,
            step_accuracy=step_accuracy,
            explanation_quality=explanation_quality,
            improvements=improvements,
        )

    def evaluate_batch(
        self,
        prompts: list[str],
        responses: list[str],
        expected_answers: Optional[list[str]] = None,
    ) -> list[CoTEvaluation]:
        """
        Evaluate multiple Chain-of-Thought responses in batch.

        Processes multiple prompt-response pairs, optionally comparing
        each to expected answers.

        Parameters
        ----------
        prompts : list[str]
            List of original prompts or questions.
        responses : list[str]
            List of model responses, corresponding to prompts.
        expected_answers : Optional[list[str]]
            Optional list of expected answers for verification.
            If shorter than prompts/responses, remaining evaluations
            won't have answer checking.

        Returns
        -------
        list[CoTEvaluation]
            List of evaluation results, one per prompt-response pair.

        Examples
        --------
        Batch evaluation with answers:

            >>> from insideLLMs.contrib.reasoning import CoTEvaluator
            >>> evaluator = CoTEvaluator()
            >>> prompts = ["1+1?", "2*3?", "10/2?"]
            >>> responses = [
            ...     "1+1 = 2",
            ...     "Step 1: 2*3. Step 2: = 6. Answer: 6",
            ...     "10 divided by 2 is 5"
            ... ]
            >>> results = evaluator.evaluate_batch(prompts, responses, ["2", "6", "5"])
            >>> correct = sum(1 for r in results if r.answer_correct)
            >>> print(f"Correct: {correct}/3")
            Correct: 3/3

        Batch evaluation without answers:

            >>> results = evaluator.evaluate_batch(
            ...     ["Q1", "Q2"],
            ...     ["Answer 1", "Answer 2"]
            ... )
            >>> all(r.answer_correct is None for r in results)
            True

        Partial expected answers:

            >>> results = evaluator.evaluate_batch(
            ...     ["Q1", "Q2", "Q3"],
            ...     ["A1", "A2", "A3"],
            ...     ["expected1"]  # Only first has expected
            ... )
            >>> print(results[0].answer_correct is not None)
            True
            >>> print(results[1].answer_correct is None)
            True
        """
        results = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            expected = (
                expected_answers[i] if expected_answers and i < len(expected_answers) else None
            )
            result = self.evaluate(prompt, response, expected)
            results.append(result)
        return results

    def generate_report(
        self,
        evaluations: list[CoTEvaluation],
    ) -> ReasoningReport:
        """
        Generate an aggregated report from multiple evaluations.

        Computes summary statistics, distributions, and recommendations
        based on a batch of evaluation results.

        Parameters
        ----------
        evaluations : list[CoTEvaluation]
            List of evaluation results to aggregate.

        Returns
        -------
        ReasoningReport
            Aggregated report containing:
            - total_evaluations: Count of evaluations
            - avg_reasoning_score: Mean reasoning score
            - avg_step_accuracy: Mean step accuracy
            - reasoning_type_breakdown: Distribution of reasoning types
            - common_fallacies: Top 5 fallacies with counts
            - quality_distribution: Counts per quality level
            - recommendations: Improvement suggestions

        Examples
        --------
        Generating a basic report:

            >>> from insideLLMs.contrib.reasoning import CoTEvaluator
            >>> evaluator = CoTEvaluator()
            >>> evals = evaluator.evaluate_batch(
            ...     ["Q1", "Q2"],
            ...     ["Given A. Therefore B.", "Step 1: X. Step 2: Y."]
            ... )
            >>> report = evaluator.generate_report(evals)
            >>> print(f"Total: {report.total_evaluations}")
            Total: 2

        Analyzing quality distribution:

            >>> # for quality, count in report.quality_distribution.items():
            >>> #     print(f"{quality}: {count}")

        Using recommendations:

            >>> if report.avg_reasoning_score < 0.5:
            ...     print("Recommendations:")
            ...     for rec in report.recommendations:
            ...         print(f"  - {rec}")

        Empty evaluations:

            >>> empty_report = evaluator.generate_report([])
            >>> print(f"Total: {empty_report.total_evaluations}")
            Total: 0
        """
        if not evaluations:
            return ReasoningReport(
                total_evaluations=0,
                avg_reasoning_score=0.0,
                avg_step_accuracy=0.0,
                reasoning_type_breakdown={},
                common_fallacies=[],
                quality_distribution={},
                recommendations=[],
            )

        # Calculate averages
        avg_reasoning = sum(e.reasoning_score for e in evaluations) / len(evaluations)
        avg_step_acc = sum(e.step_accuracy for e in evaluations) / len(evaluations)

        # Reasoning type breakdown
        type_counts: dict[str, int] = {}
        for e in evaluations:
            rt = e.chain.reasoning_type.value
            type_counts[rt] = type_counts.get(rt, 0) + 1

        type_breakdown = {k: v / len(evaluations) for k, v in type_counts.items()}

        # Common fallacies
        fallacy_counts: dict[str, int] = {}
        for e in evaluations:
            analysis = self.analyzer.analyze(e.chain)
            for f in analysis.identified_fallacies:
                fallacy_counts[f] = fallacy_counts.get(f, 0) + 1

        common_fallacies = sorted(
            fallacy_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        # Quality distribution
        quality_dist: dict[str, int] = {}
        for e in evaluations:
            analysis = self.analyzer.analyze(e.chain)
            q = analysis.overall_quality.value
            quality_dist[q] = quality_dist.get(q, 0) + 1

        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_reasoning, avg_step_acc, common_fallacies
        )

        return ReasoningReport(
            total_evaluations=len(evaluations),
            avg_reasoning_score=avg_reasoning,
            avg_step_accuracy=avg_step_acc,
            reasoning_type_breakdown=type_breakdown,
            common_fallacies=common_fallacies,
            quality_distribution=quality_dist,
            recommendations=recommendations,
        )

    def _check_answer(self, response: str, expected: str) -> bool:
        """
        Check if the response contains the expected answer.

        Uses case-insensitive substring matching to determine if the
        expected answer appears in the response.

        Parameters
        ----------
        response : str
            The full response text.
        expected : str
            The expected answer to find.

        Returns
        -------
        bool
            True if expected answer is found in response (case-insensitive).

        Examples
        --------
            >>> evaluator = CoTEvaluator()
            >>> evaluator._check_answer("The answer is 42.", "42")
            True
            >>> evaluator._check_answer("Result: FIVE", "five")
            True
            >>> evaluator._check_answer("The result is 10", "100")
            False
        """
        response_lower = response.lower()
        expected_lower = expected.lower()

        return expected_lower in response_lower

    def _assess_explanation_quality(
        self,
        response: str,
        chain: ReasoningChain,
    ) -> float:
        """
        Assess the quality of the explanation in a response.

        Evaluates how well the response explains its reasoning based on
        length, structure, use of reasoning words, and presence of conclusion.

        Parameters
        ----------
        response : str
            The full response text.
        chain : ReasoningChain
            The extracted reasoning chain.

        Returns
        -------
        float
            Quality score from 0.0 to 1.0:
            - +0.3 for ideal length (50-500 words)
            - +0.2 for moderate length (20-50 words)
            - +0.3 for 3+ reasoning steps
            - +0.2 for using reasoning words
            - +0.2 for having a conclusion

        Examples
        --------
            >>> evaluator = CoTEvaluator()
            >>> # Detailed response with good structure
            >>> response = '''
            ... Given that X is true, we can observe several things.
            ... First, this implies Y because of the relationship.
            ... Therefore, we can conclude Z is the answer.
            ... '''
            >>> chain = evaluator.extractor.extract(response)
            >>> quality = evaluator._assess_explanation_quality(response, chain)
            >>> print(f"Quality: {quality:.2f}")
            Quality: 0.70
        """
        quality = 0.0

        # Length appropriateness
        words = len(response.split())
        if 50 <= words <= 500:
            quality += 0.3
        elif 20 <= words <= 50:
            quality += 0.2

        # Has clear structure
        if len(chain.steps) >= 3:
            quality += 0.3

        # Uses reasoning words
        reasoning_words = ["because", "therefore", "since", "thus", "so", "hence"]
        response_lower = response.lower()
        if any(w in response_lower for w in reasoning_words):
            quality += 0.2

        # Has conclusion
        if chain.conclusion:
            quality += 0.2

        return min(1.0, quality)

    def _suggest_improvements(self, analysis: ChainAnalysis) -> list[str]:
        """
        Suggest improvements based on chain analysis results.

        Generates actionable improvement suggestions based on identified
        weaknesses in validity, coherence, completeness, and fallacies.

        Parameters
        ----------
        analysis : ChainAnalysis
            The analysis results to base suggestions on.

        Returns
        -------
        list[str]
            List of improvement suggestions. May be empty if no issues found.

        Examples
        --------
            >>> evaluator = CoTEvaluator()
            >>> chain = evaluator.extractor.extract("Maybe X.")
            >>> analysis = evaluator.analyzer.analyze(chain)
            >>> improvements = evaluator._suggest_improvements(analysis)
            >>> print(improvements)
            ['Strengthen logical connections between steps', 'Add missing steps to complete the reasoning chain', 'Fill in identified gaps in reasoning']
        """
        improvements = []

        if analysis.logical_validity < 0.5:
            improvements.append("Strengthen logical connections between steps")

        if analysis.coherence_score < 0.5:
            improvements.append("Improve coherence between reasoning steps")

        if analysis.completeness_score < 0.5:
            improvements.append("Add missing steps to complete the reasoning chain")

        if analysis.identified_fallacies:
            improvements.append(
                f"Address logical fallacies: {', '.join(analysis.identified_fallacies)}"
            )

        if analysis.missing_steps:
            improvements.append("Fill in identified gaps in reasoning")

        return improvements

    def _generate_recommendations(
        self,
        avg_reasoning: float,
        avg_step_acc: float,
        fallacies: list[tuple[str, int]],
    ) -> list[str]:
        """
        Generate recommendations for improvement based on aggregate metrics.

        Creates high-level recommendations based on average scores and
        common fallacy patterns across multiple evaluations.

        Parameters
        ----------
        avg_reasoning : float
            Average reasoning score across evaluations.
        avg_step_acc : float
            Average step accuracy across evaluations.
        fallacies : list[tuple[str, int]]
            List of (fallacy_name, count) tuples, sorted by frequency.

        Returns
        -------
        list[str]
            List of recommendations for improvement.

        Examples
        --------
            >>> evaluator = CoTEvaluator()
            >>> recs = evaluator._generate_recommendations(
            ...     avg_reasoning=0.4,
            ...     avg_step_acc=0.45,
            ...     fallacies=[("hasty_generalization", 5)]
            ... )
            >>> print(recs)
            ['Focus on improving overall reasoning quality', 'Work on clarity and validity of individual steps', 'Address common fallacy: hasty_generalization', 'Use more explicit reasoning markers (therefore, because, etc.)']
        """
        recommendations = []

        if avg_reasoning < 0.5:
            recommendations.append("Focus on improving overall reasoning quality")

        if avg_step_acc < 0.5:
            recommendations.append("Work on clarity and validity of individual steps")

        if fallacies:
            top_fallacy = fallacies[0][0]
            recommendations.append(f"Address common fallacy: {top_fallacy}")

        if avg_reasoning < 0.7:
            recommendations.append("Use more explicit reasoning markers (therefore, because, etc.)")

        return recommendations
