"""Convenience functions for reasoning analysis."""

from typing import Optional

from insideLLMs.contrib._reasoning.analysis import ReasoningAnalyzer
from insideLLMs.contrib._reasoning.evaluation import CoTEvaluator
from insideLLMs.contrib._reasoning.extraction import ReasoningExtractor
from insideLLMs.contrib._reasoning.models import (
    ChainAnalysis,
    CoTEvaluation,
    ReasoningChain,
    ReasoningQuality,
)
from insideLLMs.contrib._reasoning.prompting import CoTPromptGenerator


def extract_reasoning(text: str) -> ReasoningChain:
    """
    Extract a reasoning chain from unstructured text.

    Convenience function that creates a ReasoningExtractor and extracts
    a structured reasoning chain from the input text.

    Parameters
    ----------
    text : str
        The text to extract reasoning from. Can contain numbered steps,
        ordinal markers, or plain prose.

    Returns
    -------
    ReasoningChain
        A structured chain containing extracted steps, conclusion,
        reasoning type, and completeness metrics.

    Examples
    --------
    Basic extraction:

        >>> from insideLLMs.contrib.reasoning import extract_reasoning
        >>> text = "Step 1: Identify the problem. Step 2: Solve it. Step 3: Verify."
        >>> chain = extract_reasoning(text)
        >>> print(f"Found {len(chain.steps)} steps")
        Found 3 steps

    Extracting from prose:

        >>> text = "Given that A is true. Since A implies B. Therefore B is true."
        >>> chain = extract_reasoning(text)
        >>> print(f"Type: {chain.reasoning_type.value}")
        Type: deductive

    Checking validity:

        >>> chain = extract_reasoning("Just some text without reasoning.")
        >>> print(f"Valid: {chain.is_valid}")
        Valid: True

    Getting the conclusion:

        >>> text = "First, X. Then, Y. Therefore, Z is the answer."
        >>> chain = extract_reasoning(text)
        >>> print(f"Conclusion: {chain.conclusion}")
        Conclusion: z is the answer

    See Also
    --------
    ReasoningExtractor : The underlying extractor class
    analyze_reasoning : To analyze the extracted chain
    """
    extractor = ReasoningExtractor()
    return extractor.extract(text)


def analyze_reasoning(chain: ReasoningChain) -> ChainAnalysis:
    """
    Analyze a reasoning chain for quality and validity.

    Convenience function that creates a ReasoningAnalyzer and performs
    comprehensive analysis of the provided chain.

    Parameters
    ----------
    chain : ReasoningChain
        The reasoning chain to analyze.

    Returns
    -------
    ChainAnalysis
        Detailed analysis including validity scores, coherence metrics,
        identified fallacies, missing steps, and overall quality assessment.

    Examples
    --------
    Analyzing an extracted chain:

        >>> from insideLLMs.contrib.reasoning import extract_reasoning, analyze_reasoning
        >>> chain = extract_reasoning("Given A. Therefore B. Thus C.")
        >>> analysis = analyze_reasoning(chain)
        >>> print(f"Quality: {analysis.overall_quality.value}")
        Quality: adequate

    Checking for fallacies:

        >>> chain = extract_reasoning("Everyone knows this is always true.")
        >>> analysis = analyze_reasoning(chain)
        >>> print(f"Fallacies: {analysis.identified_fallacies}")
        Fallacies: ['hasty_generalization']

    Accessing detailed metrics:

        >>> chain = extract_reasoning("Step 1: A. Step 2: B because A.")
        >>> analysis = analyze_reasoning(chain)
        >>> print(f"Validity: {analysis.logical_validity:.2f}")
        Validity: 0.70
        >>> print(f"Coherence: {analysis.coherence_score:.2f}")
        Coherence: 0.20

    Identifying gaps:

        >>> chain = extract_reasoning("Hence A. Thus B. Therefore C.")
        >>> analysis = analyze_reasoning(chain)
        >>> for gap in analysis.missing_steps:
        ...     print(gap)
        No clear premise or starting point

    See Also
    --------
    ReasoningAnalyzer : The underlying analyzer class
    extract_reasoning : To create chains for analysis
    ChainAnalysis : The result structure
    """
    analyzer = ReasoningAnalyzer()
    return analyzer.analyze(chain)


def evaluate_cot(
    prompt: str,
    response: str,
    expected_answer: Optional[str] = None,
) -> CoTEvaluation:
    """
    Evaluate a Chain-of-Thought response.

    Convenience function that creates a CoTEvaluator and evaluates
    a model's response to a reasoning task.

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
        Complete evaluation including extracted chain, scores,
        answer correctness, and improvement suggestions.

    Examples
    --------
    Basic evaluation:

        >>> from insideLLMs.contrib.reasoning import evaluate_cot
        >>> evaluation = evaluate_cot(
        ...     "What is 5+5?",
        ...     "5 + 5 = 10. The answer is 10.",
        ...     "10"
        ... )
        >>> print(f"Correct: {evaluation.answer_correct}")
        Correct: True

    Checking reasoning quality:

        >>> evaluation = evaluate_cot(
        ...     "Explain why",
        ...     "Step 1: First reason. Step 2: Second reason. Therefore, conclusion."
        ... )
        >>> print(f"Reasoning score: {evaluation.reasoning_score:.2f}")
        Reasoning score: 0.68

    Getting improvement suggestions:

        >>> evaluation = evaluate_cot("Q", "Brief answer.")
        >>> for imp in evaluation.improvements:
        ...     print(f"- {imp}")
        - Strengthen logical connections between steps
        - Add missing steps to complete the reasoning chain

    Accessing the extracted chain:

        >>> evaluation = evaluate_cot("Question", "Given A. Therefore B.")
        >>> print(f"Steps: {len(evaluation.chain.steps)}")
        Steps: 2

    See Also
    --------
    CoTEvaluator : The underlying evaluator class
    CoTEvaluation : The result structure
    generate_cot_prompt : To create CoT prompts
    """
    evaluator = CoTEvaluator()
    return evaluator.evaluate(prompt, response, expected_answer)


def generate_cot_prompt(
    question: str,
    style: str = "standard",
) -> str:
    """
    Generate a Chain-of-Thought prompt.

    Convenience function that creates a CoTPromptGenerator and generates
    a prompt that encourages step-by-step reasoning.

    Parameters
    ----------
    question : str
        The question or problem to wrap in a CoT prompt.
    style : str
        The template style to use. Options:
        - "standard": Simple "Let's think step by step" format
        - "structured": Numbered step format
        - "detailed": Comprehensive breakdown format
        - "math": Mathematical problem format
        - "logical": Formal logic format
        Defaults to "standard".

    Returns
    -------
    str
        The formatted CoT prompt ready for model input.

    Examples
    --------
    Standard style:

        >>> from insideLLMs.contrib.reasoning import generate_cot_prompt
        >>> prompt = generate_cot_prompt("What is 2+2?")
        >>> print(prompt)
        Let's think step by step.
        <BLANKLINE>
        What is 2+2?

    Math style:

        >>> prompt = generate_cot_prompt("Calculate the area", style="math")
        >>> "Given:" in prompt
        True

    Structured style:

        >>> prompt = generate_cot_prompt("Solve x+5=10", style="structured")
        >>> "Step 1:" in prompt
        True

    Logical style:

        >>> prompt = generate_cot_prompt("Is X true?", style="logical")
        >>> "Premise 1:" in prompt
        True

    See Also
    --------
    CoTPromptGenerator : The underlying generator class
    evaluate_cot : To evaluate CoT responses
    """
    generator = CoTPromptGenerator()
    return generator.generate(question, style)


def assess_reasoning_quality(text: str) -> ReasoningQuality:
    """
    Quickly assess the quality of reasoning in text.

    Convenience function that extracts a reasoning chain and analyzes
    it, returning just the overall quality assessment.

    Parameters
    ----------
    text : str
        Text containing reasoning to assess.

    Returns
    -------
    ReasoningQuality
        The overall quality level: EXCELLENT, GOOD, ADEQUATE, POOR, or INVALID.

    Examples
    --------
    Assessing good reasoning:

        >>> from insideLLMs.contrib.reasoning import assess_reasoning_quality, ReasoningQuality
        >>> text = '''
        ... Given that all mammals are warm-blooded.
        ... Since dogs are mammals.
        ... Therefore, dogs are warm-blooded.
        ... '''
        >>> quality = assess_reasoning_quality(text)
        >>> print(f"Quality: {quality.value}")
        Quality: good

    Assessing poor reasoning:

        >>> quality = assess_reasoning_quality("Maybe X. Perhaps Y.")
        >>> quality == ReasoningQuality.POOR or quality == ReasoningQuality.ADEQUATE
        True

    Using for filtering:

        >>> texts = ["Given A. Therefore B.", "Just random text."]
        >>> qualities = [assess_reasoning_quality(t) for t in texts]
        >>> good_texts = [t for t, q in zip(texts, qualities)
        ...              if q in (ReasoningQuality.EXCELLENT, ReasoningQuality.GOOD)]

    Threshold checking:

        >>> from insideLLMs.contrib.reasoning import assess_reasoning_quality, ReasoningQuality
        >>> quality = assess_reasoning_quality("Step 1: A. Step 2: B. Conclusion: C.")
        >>> if quality in (ReasoningQuality.POOR, ReasoningQuality.INVALID):
        ...     print("Warning: Low quality reasoning")

    See Also
    --------
    extract_reasoning : For full chain extraction
    analyze_reasoning : For detailed analysis
    ReasoningQuality : The quality enumeration
    """
    extractor = ReasoningExtractor()
    analyzer = ReasoningAnalyzer()

    chain = extractor.extract(text)
    analysis = analyzer.analyze(chain)

    return analysis.overall_quality
