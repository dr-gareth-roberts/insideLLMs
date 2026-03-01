"""
Reasoning chain analysis and Chain-of-Thought evaluation utilities.

This module provides comprehensive tools for analyzing, extracting, and evaluating
reasoning patterns in Large Language Model outputs. It supports Chain-of-Thought (CoT)
prompting analysis, logical validity checking, and reasoning quality assessment.

Key Components
--------------
- **ReasoningExtractor**: Extracts structured reasoning chains from unstructured text
- **ReasoningAnalyzer**: Analyzes chains for logical validity, coherence, and fallacies
- **CoTEvaluator**: Evaluates Chain-of-Thought responses with comprehensive metrics
- **CoTPromptGenerator**: Generates various styles of CoT prompts

Core Data Structures
--------------------
- **ReasoningStep**: Represents a single step in a reasoning chain
- **ReasoningChain**: A complete chain of reasoning steps with metadata
- **ChainAnalysis**: Detailed analysis results of a reasoning chain
- **CoTEvaluation**: Evaluation results of a CoT response
- **ReasoningReport**: Aggregated report across multiple evaluations

Reasoning Types Supported
-------------------------
- Deductive: Logical conclusions from premises
- Inductive: Generalizations from specific observations
- Abductive: Best explanation inference
- Analogical: Reasoning by comparison
- Causal: Cause-and-effect reasoning
- Mathematical: Numerical and arithmetic reasoning
- Temporal: Time-based reasoning
- Spatial: Location and space-based reasoning

Examples
--------
Basic reasoning extraction:

    >>> from insideLLMs.evaluation.reasoning import extract_reasoning
    >>> text = '''
    ... Step 1: We know that all mammals are warm-blooded.
    ... Step 2: Dogs are mammals.
    ... Step 3: Therefore, dogs are warm-blooded.
    ... '''
    >>> chain = extract_reasoning(text)
    >>> print(f"Found {len(chain.steps)} steps")
    Found 3 steps
    >>> print(f"Conclusion: {chain.conclusion}")
    Conclusion: dogs are warm-blooded

Analyzing reasoning quality:

    >>> from insideLLMs.evaluation.reasoning import extract_reasoning, analyze_reasoning
    >>> text = '''
    ... Given: x = 5 and y = 3
    ... First, calculate x + y = 8
    ... Then, multiply by 2: 8 * 2 = 16
    ... Therefore, the result is 16
    ... '''
    >>> chain = extract_reasoning(text)
    >>> analysis = analyze_reasoning(chain)
    >>> print(f"Quality: {analysis.overall_quality.value}")
    Quality: good
    >>> print(f"Validity: {analysis.logical_validity:.2f}")
    Validity: 0.85

Evaluating Chain-of-Thought responses:

    >>> from insideLLMs.evaluation.reasoning import evaluate_cot
    >>> prompt = "What is 15 + 27?"
    >>> response = '''
    ... Let me solve this step by step.
    ... Step 1: Break down the numbers: 15 = 10 + 5, 27 = 20 + 7
    ... Step 2: Add the tens: 10 + 20 = 30
    ... Step 3: Add the units: 5 + 7 = 12
    ... Step 4: Combine: 30 + 12 = 42
    ... Therefore, 15 + 27 = 42
    ... '''
    >>> evaluation = evaluate_cot(prompt, response, expected_answer="42")
    >>> print(f"Answer correct: {evaluation.answer_correct}")
    Answer correct: True
    >>> print(f"Reasoning score: {evaluation.reasoning_score:.2f}")
    Reasoning score: 0.78

Generating CoT prompts:

    >>> from insideLLMs.evaluation.reasoning import generate_cot_prompt
    >>> question = "If a train travels 120 miles in 2 hours, what is its speed?"
    >>> prompt = generate_cot_prompt(question, style="math")
    >>> print(prompt)
    Let's solve this mathematical problem step by step:
    <BLANKLINE>
    If a train travels 120 miles in 2 hours, what is its speed?
    <BLANKLINE>
    Given:

Batch evaluation with reporting:

    >>> from insideLLMs.evaluation.reasoning import CoTEvaluator
    >>> evaluator = CoTEvaluator()
    >>> prompts = ["What is 2+2?", "What is 3*4?"]
    >>> responses = [
    ...     "2+2 equals 4 because addition combines quantities.",
    ...     "Step 1: 3*4 means 3 added 4 times. Step 2: 3+3+3+3=12. Answer: 12"
    ... ]
    >>> evaluations = evaluator.evaluate_batch(prompts, responses, ["4", "12"])
    >>> report = evaluator.generate_report(evaluations)
    >>> print(f"Average reasoning score: {report.avg_reasoning_score:.2f}")
    Average reasoning score: 0.65

See Also
--------
- `insideLLMs.analysis`: For broader LLM output analysis
- `insideLLMs.probes`: For probing model internals
- `insideLLMs.visualization`: For visualizing reasoning chains

Notes
-----
The reasoning extraction uses heuristic pattern matching and may not perfectly
capture all forms of reasoning. For best results, use structured responses with
clear step markers (e.g., "Step 1:", "First,", numbered lists).

Fallacy detection is based on keyword patterns and should be used as a guide
rather than definitive classification. Complex fallacies may require human review.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ReasoningType(Enum):
    """
    Types of reasoning patterns that can be identified in LLM outputs.

    This enumeration categorizes different forms of logical reasoning that
    models may use when solving problems or answering questions. Each type
    represents a distinct cognitive pattern with specific characteristics.

    Attributes
    ----------
    DEDUCTIVE : str
        Reasoning from general premises to specific conclusions. If premises
        are true, the conclusion must be true. Example: "All mammals are
        warm-blooded. Dogs are mammals. Therefore, dogs are warm-blooded."

    INDUCTIVE : str
        Reasoning from specific observations to general conclusions. The
        conclusion is probable but not guaranteed. Example: "Every swan I've
        seen is white. Therefore, all swans are probably white."

    ABDUCTIVE : str
        Inference to the best explanation. Given observations, deduce the
        most likely cause. Example: "The grass is wet. The best explanation
        is that it rained."

    ANALOGICAL : str
        Reasoning by comparison to similar cases. Example: "Electrons orbit
        the nucleus like planets orbit the sun."

    CAUSAL : str
        Reasoning about cause-and-effect relationships. Example: "Smoking
        causes lung cancer because it damages lung tissue over time."

    MATHEMATICAL : str
        Numerical and arithmetic reasoning involving calculations. Example:
        "If x = 5 and y = 3, then x + y = 8."

    TEMPORAL : str
        Reasoning involving time sequences and temporal relationships.
        Example: "Event A happened before B, and B before C, so A happened
        before C."

    SPATIAL : str
        Reasoning about locations, positions, and spatial relationships.
        Example: "The library is north of the park, so walking south from
        the library leads to the park."

    Examples
    --------
    Identifying reasoning type in extracted chains:

        >>> from insideLLMs.evaluation.reasoning import extract_reasoning, ReasoningType
        >>> math_text = "Given x = 10. Calculate x * 2 = 20. Therefore the answer is 20."
        >>> chain = extract_reasoning(math_text)
        >>> chain.reasoning_type == ReasoningType.MATHEMATICAL
        True

    Using reasoning type for analysis filtering:

        >>> from insideLLMs.evaluation.reasoning import ReasoningType
        >>> # Filter evaluations by reasoning type
        >>> math_evals = [e for e in evaluations
        ...              if e.chain.reasoning_type == ReasoningType.MATHEMATICAL]

    Checking for specific reasoning patterns:

        >>> from insideLLMs.evaluation.reasoning import extract_reasoning, ReasoningType
        >>> causal_text = "Rain causes floods because water accumulates rapidly."
        >>> chain = extract_reasoning(causal_text)
        >>> if chain.reasoning_type == ReasoningType.CAUSAL:
        ...     print("Causal reasoning detected")
        Causal reasoning detected

    See Also
    --------
    ReasoningExtractor._classify_reasoning_type : Method that classifies text
    ReasoningChain : Container that holds the reasoning type
    """

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    MATHEMATICAL = "mathematical"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


class ReasoningStepType(Enum):
    """
    Types of individual steps within a reasoning chain.

    Each step in a reasoning chain serves a specific logical function. This
    enumeration classifies those functions to enable analysis of reasoning
    structure and identification of missing components.

    Attributes
    ----------
    PREMISE : str
        A foundational statement that is assumed or given to be true. Premises
        form the starting point of deductive reasoning. Identified by markers
        like "given", "assume", "let", "suppose".

    INFERENCE : str
        A logical conclusion drawn from previous steps. Inferences connect
        premises to conclusions. Identified by markers like "therefore",
        "thus", "hence", "so".

    CALCULATION : str
        A mathematical or numerical computation step. Involves arithmetic
        operations or formula application. Identified by operators like
        "=", "+", "-", "*", "/" or words like "calculate", "compute".

    COMPARISON : str
        A step that compares two or more items, values, or concepts.
        Used in analogical reasoning or decision-making processes.

    CONCLUSION : str
        The final result or answer derived from the reasoning chain.
        Identified by markers like "therefore", "in conclusion", "finally",
        "the answer is".

    ASSUMPTION : str
        An unstated belief taken for granted in the reasoning. Unlike premises,
        assumptions may not be explicitly stated but are implicit in the logic.

    EVIDENCE : str
        Supporting data, facts, or observations that back up claims.
        Strengthens the reasoning by providing empirical support.

    EXAMPLE : str
        An illustrative instance used to clarify or support a point.
        Common in explanatory reasoning and teaching contexts.

    Examples
    --------
    Identifying step types in a chain:

        >>> from insideLLMs.evaluation.reasoning import extract_reasoning, ReasoningStepType
        >>> text = '''
        ... Given: All birds have feathers.
        ... Premise: Penguins are birds.
        ... Therefore, penguins have feathers.
        ... '''
        >>> chain = extract_reasoning(text)
        >>> premise_steps = [s for s in chain.steps
        ...                  if s.step_type == ReasoningStepType.PREMISE]
        >>> print(f"Found {len(premise_steps)} premise steps")
        Found 1 premise steps

    Checking for calculation steps in math problems:

        >>> from insideLLMs.evaluation.reasoning import extract_reasoning, ReasoningStepType
        >>> math_text = "First, calculate 5 + 3 = 8. Then multiply by 2 = 16."
        >>> chain = extract_reasoning(math_text)
        >>> has_calc = any(s.step_type == ReasoningStepType.CALCULATION
        ...                for s in chain.steps)
        >>> print(f"Contains calculations: {has_calc}")
        Contains calculations: True

    Analyzing reasoning structure:

        >>> from insideLLMs.evaluation.reasoning import extract_reasoning, ReasoningStepType
        >>> # Check if reasoning has proper structure
        >>> chain = extract_reasoning("Assume x=5. Thus x+1=6. The answer is 6.")
        >>> has_premise = any(s.step_type == ReasoningStepType.PREMISE
        ...                   for s in chain.steps)
        >>> has_conclusion = any(s.step_type == ReasoningStepType.CONCLUSION
        ...                      for s in chain.steps)
        >>> print(f"Well-structured: {has_premise and has_conclusion}")
        Well-structured: True

    See Also
    --------
    ReasoningStep : Dataclass that uses this type
    ReasoningExtractor._classify_step : Method that assigns step types
    """

    PREMISE = "premise"
    INFERENCE = "inference"
    CALCULATION = "calculation"
    COMPARISON = "comparison"
    CONCLUSION = "conclusion"
    ASSUMPTION = "assumption"
    EVIDENCE = "evidence"
    EXAMPLE = "example"


class ReasoningQuality(Enum):
    """
    Quality levels for assessing reasoning chains.

    This enumeration provides a categorical assessment of reasoning quality
    based on multiple factors including logical validity, coherence, and
    completeness. Used as the final quality verdict in chain analysis.

    Attributes
    ----------
    EXCELLENT : str
        Outstanding reasoning quality (score >= 0.8). Demonstrates clear
        logical flow, strong coherence between steps, complete reasoning
        chain with premises and conclusions, and no significant fallacies.

    GOOD : str
        High quality reasoning (0.6 <= score < 0.8). Shows solid logical
        structure with minor gaps. Most steps are well-connected with
        reasonable coherence. May have minor issues but overall sound.

    ADEQUATE : str
        Acceptable reasoning quality (0.4 <= score < 0.6). Basic logical
        structure present but with noticeable gaps or weak connections.
        May lack clear premises or conclusions but conveys reasoning intent.

    POOR : str
        Low quality reasoning (0.2 <= score < 0.4). Significant issues with
        logical flow, coherence, or completeness. May contain fallacies or
        have major gaps in the reasoning chain.

    INVALID : str
        Unacceptable reasoning (score < 0.2). Fails to demonstrate logical
        structure. May be incoherent, contain critical fallacies, or lack
        any recognizable reasoning pattern.

    Examples
    --------
    Assessing reasoning quality:

        >>> from insideLLMs.evaluation.reasoning import assess_reasoning_quality, ReasoningQuality
        >>> good_reasoning = '''
        ... Given: All cats are mammals. Given: Fluffy is a cat.
        ... Therefore, Fluffy must be a mammal.
        ... '''
        >>> quality = assess_reasoning_quality(good_reasoning)
        >>> print(f"Quality: {quality.value}")
        Quality: good

    Filtering by quality threshold:

        >>> from insideLLMs.evaluation.reasoning import ReasoningQuality
        >>> # Define acceptable quality levels
        >>> acceptable = {ReasoningQuality.EXCELLENT, ReasoningQuality.GOOD}
        >>> # Filter analysis results
        >>> good_analyses = [a for a in analyses if a.overall_quality in acceptable]

    Quality-based decision making:

        >>> from insideLLMs.evaluation.reasoning import (
        ...     extract_reasoning, analyze_reasoning, ReasoningQuality
        ... )
        >>> chain = extract_reasoning("Step 1: A. Step 2: B. Therefore C.")
        >>> analysis = analyze_reasoning(chain)
        >>> if analysis.overall_quality == ReasoningQuality.INVALID:
        ...     print("Warning: Invalid reasoning detected")
        ... elif analysis.overall_quality in (ReasoningQuality.EXCELLENT,
        ...                                    ReasoningQuality.GOOD):
        ...     print("Reasoning quality is acceptable")
        Reasoning quality is acceptable

    Comparing quality levels:

        >>> from insideLLMs.evaluation.reasoning import assess_reasoning_quality
        >>> q1 = assess_reasoning_quality("Therefore x. Thus y. Hence z.")
        >>> q2 = assess_reasoning_quality("Given A, and since B, therefore C.")
        >>> # Note: Quality enums don't support direct comparison
        >>> quality_order = ['invalid', 'poor', 'adequate', 'good', 'excellent']
        >>> quality_order.index(q1.value) < quality_order.index(q2.value)
        True

    See Also
    --------
    ChainAnalysis : Contains overall_quality field
    ReasoningAnalyzer.analyze : Method that determines quality level
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class ReasoningStep:
    """
    A single step in a reasoning chain.

    Represents one logical step in a chain of reasoning, containing the content
    of the step along with metadata about its type, confidence, and relationships
    to other steps. Used as the building block for ReasoningChain structures.

    Attributes
    ----------
    content : str
        The textual content of this reasoning step. Contains the actual
        statement, calculation, or inference being made.

    step_number : int
        The position of this step in the reasoning chain, starting from 1.
        Used for ordering and dependency tracking.

    step_type : ReasoningStepType
        The logical function of this step (premise, inference, conclusion, etc.).
        Defaults to INFERENCE if not specified.

    confidence : float
        Estimated confidence in this step's correctness, from 0.0 to 1.0.
        Based on linguistic markers (e.g., "clearly" increases, "maybe" decreases).
        Defaults to 0.5.

    supports_conclusion : bool
        Whether this step supports the final conclusion. Set to False if the
        step contradicts or is irrelevant to the conclusion. Defaults to True.

    depends_on : list[int]
        List of step numbers that this step logically depends on. Used for
        tracking reasoning dependencies and validating logical flow.

    Examples
    --------
    Creating a basic reasoning step:

        >>> from insideLLMs.evaluation.reasoning import ReasoningStep, ReasoningStepType
        >>> step = ReasoningStep(
        ...     content="All mammals are warm-blooded",
        ...     step_number=1,
        ...     step_type=ReasoningStepType.PREMISE,
        ...     confidence=0.95
        ... )
        >>> print(f"Step {step.step_number}: {step.content}")
        Step 1: All mammals are warm-blooded

    Creating a step with dependencies:

        >>> from insideLLMs.evaluation.reasoning import ReasoningStep, ReasoningStepType
        >>> premise1 = ReasoningStep(
        ...     content="All A are B",
        ...     step_number=1,
        ...     step_type=ReasoningStepType.PREMISE
        ... )
        >>> premise2 = ReasoningStep(
        ...     content="X is an A",
        ...     step_number=2,
        ...     step_type=ReasoningStepType.PREMISE
        ... )
        >>> conclusion = ReasoningStep(
        ...     content="Therefore, X is a B",
        ...     step_number=3,
        ...     step_type=ReasoningStepType.CONCLUSION,
        ...     depends_on=[1, 2]  # Depends on both premises
        ... )
        >>> print(f"Conclusion depends on steps: {conclusion.depends_on}")
        Conclusion depends on steps: [1, 2]

    Converting to dictionary for serialization:

        >>> from insideLLMs.evaluation.reasoning import ReasoningStep, ReasoningStepType
        >>> step = ReasoningStep(
        ...     content="Calculate: 5 + 3 = 8",
        ...     step_number=1,
        ...     step_type=ReasoningStepType.CALCULATION,
        ...     confidence=0.9
        ... )
        >>> step_dict = step.to_dict()
        >>> print(step_dict['step_type'])
        calculation
        >>> print(step_dict['confidence'])
        0.9

    Checking step properties:

        >>> from insideLLMs.evaluation.reasoning import ReasoningStep, ReasoningStepType
        >>> step = ReasoningStep(
        ...     content="This might be true",
        ...     step_number=1,
        ...     confidence=0.3,
        ...     supports_conclusion=False
        ... )
        >>> if step.confidence < 0.5:
        ...     print("Low confidence step detected")
        Low confidence step detected
        >>> if not step.supports_conclusion:
        ...     print("Step does not support conclusion")
        Step does not support conclusion

    See Also
    --------
    ReasoningChain : Container for multiple ReasoningStep objects
    ReasoningStepType : Enumeration of step types
    ReasoningExtractor : Extracts steps from text
    """

    content: str
    step_number: int
    step_type: ReasoningStepType = ReasoningStepType.INFERENCE
    confidence: float = 0.5
    supports_conclusion: bool = True
    depends_on: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the reasoning step to a dictionary representation.

        Serializes all step attributes to a dictionary format suitable for
        JSON serialization, logging, or API responses. Enum values are
        converted to their string representations.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all step attributes:
            - content: The step text
            - step_number: Position in chain
            - step_type: Type as string value
            - confidence: Confidence score
            - supports_conclusion: Boolean flag
            - depends_on: List of dependency step numbers

        Examples
        --------
        Basic serialization:

            >>> from insideLLMs.evaluation.reasoning import ReasoningStep, ReasoningStepType
            >>> step = ReasoningStep(
            ...     content="Given: x = 5",
            ...     step_number=1,
            ...     step_type=ReasoningStepType.PREMISE
            ... )
            >>> d = step.to_dict()
            >>> print(d)
            {'content': 'Given: x = 5', 'step_number': 1, 'step_type': 'premise',
             'confidence': 0.5, 'supports_conclusion': True, 'depends_on': []}

        JSON serialization:

            >>> import json
            >>> from insideLLMs.evaluation.reasoning import ReasoningStep
            >>> step = ReasoningStep(content="Step content", step_number=1)
            >>> json_str = json.dumps(step.to_dict())
            >>> print(type(json_str))
            <class 'str'>

        Reconstructing from dictionary:

            >>> from insideLLMs.evaluation.reasoning import ReasoningStep, ReasoningStepType
            >>> step = ReasoningStep(content="Example", step_number=1)
            >>> d = step.to_dict()
            >>> # Note: step_type needs conversion back to enum
            >>> d['step_type'] = ReasoningStepType(d['step_type'])
            >>> new_step = ReasoningStep(**d)
            >>> new_step.content == step.content
            True
        """
        return {
            "content": self.content,
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "confidence": self.confidence,
            "supports_conclusion": self.supports_conclusion,
            "depends_on": self.depends_on,
        }


@dataclass
class ReasoningChain:
    """
    A complete chain of reasoning steps with metadata.

    Represents an entire reasoning process from premises to conclusion,
    containing ordered steps and metadata about the reasoning type,
    validity, and completeness. This is the primary data structure
    returned by ReasoningExtractor.

    Attributes
    ----------
    steps : list[ReasoningStep]
        Ordered list of reasoning steps in the chain. Steps are numbered
        sequentially and may reference each other via depends_on.

    conclusion : Optional[str]
        The final conclusion or answer derived from the reasoning chain.
        Extracted from explicit conclusion markers or the last statement.

    reasoning_type : ReasoningType
        The primary type of reasoning used in this chain (deductive,
        inductive, mathematical, etc.). Defaults to DEDUCTIVE.

    is_valid : bool
        Whether the chain is structurally valid (has at least one step).
        Does not guarantee logical correctness, just structural validity.

    completeness : float
        Score from 0.0 to 1.0 indicating how complete the reasoning chain is.
        Based on presence of premises, inferences, and conclusion.

    Examples
    --------
    Extracting and examining a reasoning chain:

        >>> from insideLLMs.evaluation.reasoning import extract_reasoning
        >>> text = '''
        ... Given: All birds can fly (premise).
        ... Penguins are birds.
        ... Therefore, penguins can fly (incorrect but valid structure).
        ... '''
        >>> chain = extract_reasoning(text)
        >>> print(f"Steps: {len(chain.steps)}, Valid: {chain.is_valid}")
        Steps: 3, Valid: True
        >>> print(f"Type: {chain.reasoning_type.value}")
        Type: deductive

    Working with chain steps:

        >>> from insideLLMs.evaluation.reasoning import extract_reasoning
        >>> chain = extract_reasoning("Step 1: A. Step 2: B. Step 3: Therefore C.")
        >>> step = chain.get_step(2)
        >>> if step:
        ...     print(f"Step 2: {step.content}")
        Step 2: B

    Getting specific step types:

        >>> from insideLLMs.evaluation.reasoning import extract_reasoning
        >>> text = "Given: x=5. Suppose y=3. Thus x+y=8. Therefore the answer is 8."
        >>> chain = extract_reasoning(text)
        >>> premises = chain.get_premises()
        >>> print(f"Found {len(premises)} premise(s)")
        Found 2 premise(s)
        >>> inferences = chain.get_inferences()
        >>> print(f"Found {len(inferences)} inference(s)")
        Found 1 inference(s)

    Checking chain completeness:

        >>> from insideLLMs.evaluation.reasoning import extract_reasoning
        >>> complete_text = '''
        ... Given: All A are B (premise).
        ... X is an A (premise).
        ... Therefore, X is B (inference leading to conclusion).
        ... In conclusion, we have shown X is B.
        ... '''
        >>> chain = extract_reasoning(complete_text)
        >>> print(f"Completeness: {chain.completeness:.1%}")
        Completeness: 100.0%

    See Also
    --------
    ReasoningStep : Individual step in the chain
    ReasoningExtractor : Creates ReasoningChain from text
    ChainAnalysis : Detailed analysis of a chain
    """

    steps: list[ReasoningStep]
    conclusion: Optional[str] = None
    reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    is_valid: bool = True
    completeness: float = 0.0

    def get_step(self, step_number: int) -> Optional[ReasoningStep]:
        """
        Get a specific step by its step number.

        Retrieves a reasoning step from the chain by its position number.
        Step numbers typically start at 1 and increment sequentially.

        Parameters
        ----------
        step_number : int
            The step number to retrieve (1-indexed).

        Returns
        -------
        Optional[ReasoningStep]
            The ReasoningStep with the matching step_number, or None if
            no step with that number exists in the chain.

        Examples
        --------
        Basic step retrieval:

            >>> from insideLLMs.evaluation.reasoning import extract_reasoning
            >>> chain = extract_reasoning("Step 1: First. Step 2: Second. Step 3: Third.")
            >>> step = chain.get_step(2)
            >>> if step:
            ...     print(step.content)
            Second

        Handling missing steps:

            >>> from insideLLMs.evaluation.reasoning import extract_reasoning
            >>> chain = extract_reasoning("Step 1: Only step.")
            >>> step = chain.get_step(5)  # Doesn't exist
            >>> print(step is None)
            True

        Iterating with step numbers:

            >>> from insideLLMs.evaluation.reasoning import extract_reasoning
            >>> chain = extract_reasoning("Step 1: A. Step 2: B. Step 3: C.")
            >>> for i in range(1, len(chain.steps) + 1):
            ...     step = chain.get_step(i)
            ...     if step:
            ...         print(f"{i}: {step.content}")
            1: A
            2: B
            3: C
        """
        for step in self.steps:
            if step.step_number == step_number:
                return step
        return None

    def get_premises(self) -> list[ReasoningStep]:
        """
        Get all premise steps from the chain.

        Filters and returns only steps classified as premises (foundational
        statements that serve as starting points for reasoning).

        Returns
        -------
        list[ReasoningStep]
            List of all steps with step_type == ReasoningStepType.PREMISE.
            Empty list if no premises are found.

        Examples
        --------
        Extracting premises:

            >>> from insideLLMs.evaluation.reasoning import extract_reasoning
            >>> text = '''
            ... Given that all mammals are warm-blooded.
            ... Assume that dogs are mammals.
            ... Therefore dogs are warm-blooded.
            ... '''
            >>> chain = extract_reasoning(text)
            >>> premises = chain.get_premises()
            >>> for p in premises:
            ...     print(f"Premise: {p.content[:40]}...")
            Premise: all mammals are warm-blooded...

        Checking for premises:

            >>> from insideLLMs.evaluation.reasoning import extract_reasoning
            >>> chain = extract_reasoning("Therefore X. Thus Y. Hence Z.")
            >>> premises = chain.get_premises()
            >>> if not premises:
            ...     print("Warning: No explicit premises found")
            Warning: No explicit premises found

        Counting premise types:

            >>> from insideLLMs.evaluation.reasoning import extract_reasoning
            >>> text = "Let x=5. Suppose y=10. Given z=15. Calculate x+y+z=30."
            >>> chain = extract_reasoning(text)
            >>> print(f"Number of premises: {len(chain.get_premises())}")
            Number of premises: 3
        """
        return [s for s in self.steps if s.step_type == ReasoningStepType.PREMISE]

    def get_inferences(self) -> list[ReasoningStep]:
        """
        Get all inference steps from the chain.

        Filters and returns only steps classified as inferences (logical
        conclusions drawn from previous steps).

        Returns
        -------
        list[ReasoningStep]
            List of all steps with step_type == ReasoningStepType.INFERENCE.
            Empty list if no inferences are found.

        Examples
        --------
        Extracting inferences:

            >>> from insideLLMs.evaluation.reasoning import extract_reasoning
            >>> text = '''
            ... Given: x = 5.
            ... Therefore, x + 5 = 10.
            ... Thus, x * 2 = 10 as well.
            ... Hence, x + 5 equals x * 2.
            ... '''
            >>> chain = extract_reasoning(text)
            >>> inferences = chain.get_inferences()
            >>> print(f"Found {len(inferences)} inference(s)")
            Found 3 inference(s)

        Analyzing inference quality:

            >>> from insideLLMs.evaluation.reasoning import extract_reasoning
            >>> chain = extract_reasoning("Given A. Therefore B. Thus C. Hence D.")
            >>> inferences = chain.get_inferences()
            >>> avg_confidence = sum(i.confidence for i in inferences) / len(inferences)
            >>> print(f"Average inference confidence: {avg_confidence:.2f}")
            Average inference confidence: 0.60

        Checking reasoning structure:

            >>> from insideLLMs.evaluation.reasoning import extract_reasoning
            >>> chain = extract_reasoning("A equals B. B equals C. The answer is C.")
            >>> has_inferences = len(chain.get_inferences()) > 0
            >>> has_premises = len(chain.get_premises()) > 0
            >>> print(f"Has premises: {has_premises}, Has inferences: {has_inferences}")
            Has premises: False, Has inferences: False
        """
        return [s for s in self.steps if s.step_type == ReasoningStepType.INFERENCE]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the reasoning chain to a dictionary representation.

        Serializes the entire chain including all steps and metadata to a
        dictionary format suitable for JSON serialization or API responses.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - num_steps: Total number of steps
            - steps: List of step dictionaries
            - conclusion: The chain conclusion
            - reasoning_type: Type as string value
            - is_valid: Validity boolean
            - completeness: Completeness score

        Examples
        --------
        Basic serialization:

            >>> from insideLLMs.evaluation.reasoning import extract_reasoning
            >>> chain = extract_reasoning("Step 1: A. Step 2: B. Therefore C.")
            >>> d = chain.to_dict()
            >>> print(f"Steps: {d['num_steps']}, Valid: {d['is_valid']}")
            Steps: 3, Valid: True

        JSON export:

            >>> import json
            >>> from insideLLMs.evaluation.reasoning import extract_reasoning
            >>> chain = extract_reasoning("Given x=5. Therefore x+1=6.")
            >>> json_str = json.dumps(chain.to_dict(), indent=2)
            >>> print(type(json_str))
            <class 'str'>

        Accessing serialized steps:

            >>> from insideLLMs.evaluation.reasoning import extract_reasoning
            >>> chain = extract_reasoning("Step 1: First. Step 2: Second.")
            >>> d = chain.to_dict()
            >>> for step in d['steps']:
            ...     print(f"Step {step['step_number']}: {step['step_type']}")
            Step 1: inference
            Step 2: inference
        """
        return {
            "num_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
            "conclusion": self.conclusion,
            "reasoning_type": self.reasoning_type.value,
            "is_valid": self.is_valid,
            "completeness": self.completeness,
        }


@dataclass
class ChainAnalysis:
    """
    Comprehensive analysis of a reasoning chain.

    Contains detailed metrics and assessments of a reasoning chain's quality,
    including logical validity, coherence between steps, completeness, and
    identification of potential fallacies or gaps in reasoning.

    Attributes
    ----------
    chain : ReasoningChain
        The original reasoning chain that was analyzed.

    logical_validity : float
        Score from 0.0 to 1.0 measuring how logically valid the reasoning is.
        Based on clear flow, step dependencies, and presence of evidence.

    coherence_score : float
        Score from 0.0 to 1.0 measuring how well steps connect to each other.
        Calculated based on word overlap and logical flow between adjacent steps.

    completeness_score : float
        Score from 0.0 to 1.0 indicating presence of required components
        (premises, inferences, conclusion).

    step_quality_scores : list[float]
        Individual quality scores for each step in the chain. Each score
        considers content length, reasoning markers, and confidence.

    identified_fallacies : list[str]
        List of logical fallacy names detected in the reasoning. Common
        fallacies include circular_reasoning, hasty_generalization,
        false_dichotomy, appeal_to_authority, etc.

    missing_steps : list[str]
        Descriptions of gaps or missing components in the reasoning chain.
        May include missing premises, conclusions, or jumps between steps.

    overall_quality : ReasoningQuality
        Categorical assessment of overall reasoning quality (EXCELLENT,
        GOOD, ADEQUATE, POOR, or INVALID).

    Examples
    --------
    Analyzing a reasoning chain:

        >>> from insideLLMs.evaluation.reasoning import extract_reasoning, analyze_reasoning
        >>> text = '''
        ... Given: All humans are mortal.
        ... Socrates is a human.
        ... Therefore, Socrates is mortal.
        ... '''
        >>> chain = extract_reasoning(text)
        >>> analysis = analyze_reasoning(chain)
        >>> print(f"Validity: {analysis.logical_validity:.2f}")
        Validity: 0.85
        >>> print(f"Quality: {analysis.overall_quality.value}")
        Quality: good

    Checking for fallacies:

        >>> from insideLLMs.evaluation.reasoning import extract_reasoning, analyze_reasoning
        >>> text = "All politicians are corrupt. Everyone knows this is true."
        >>> chain = extract_reasoning(text)
        >>> analysis = analyze_reasoning(chain)
        >>> if analysis.identified_fallacies:
        ...     print(f"Fallacies found: {analysis.identified_fallacies}")
        Fallacies found: ['hasty_generalization']

    Examining step quality:

        >>> from insideLLMs.evaluation.reasoning import extract_reasoning, analyze_reasoning
        >>> text = "Step 1: A because B. Step 2: Therefore C. Step 3: Thus D."
        >>> chain = extract_reasoning(text)
        >>> analysis = analyze_reasoning(chain)
        >>> avg_quality = sum(analysis.step_quality_scores) / len(analysis.step_quality_scores)
        >>> print(f"Average step quality: {avg_quality:.2f}")
        Average step quality: 0.70

    Identifying missing components:

        >>> from insideLLMs.evaluation.reasoning import extract_reasoning, analyze_reasoning
        >>> text = "Therefore A. Thus B. Hence C."  # No premises
        >>> chain = extract_reasoning(text)
        >>> analysis = analyze_reasoning(chain)
        >>> for missing in analysis.missing_steps:
        ...     print(f"Missing: {missing}")
        Missing: No clear premise or starting point

    See Also
    --------
    ReasoningAnalyzer : Produces ChainAnalysis objects
    ReasoningQuality : Quality enumeration
    ReasoningChain : Input for analysis
    """

    chain: ReasoningChain
    logical_validity: float
    coherence_score: float
    completeness_score: float
    step_quality_scores: list[float]
    identified_fallacies: list[str]
    missing_steps: list[str]
    overall_quality: ReasoningQuality

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the chain analysis to a dictionary representation.

        Serializes all analysis results including the original chain,
        scores, fallacies, and recommendations to a dictionary format.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - chain: Serialized reasoning chain
            - logical_validity: Validity score
            - coherence_score: Coherence score
            - completeness_score: Completeness score
            - avg_step_quality: Average of step quality scores
            - num_fallacies: Count of identified fallacies
            - identified_fallacies: List of fallacy names
            - missing_steps: List of missing step descriptions
            - overall_quality: Quality level as string

        Examples
        --------
        Basic serialization:

            >>> from insideLLMs.evaluation.reasoning import extract_reasoning, analyze_reasoning
            >>> chain = extract_reasoning("Given A. Therefore B.")
            >>> analysis = analyze_reasoning(chain)
            >>> d = analysis.to_dict()
            >>> print(f"Quality: {d['overall_quality']}")
            Quality: adequate

        JSON export:

            >>> import json
            >>> from insideLLMs.evaluation.reasoning import extract_reasoning, analyze_reasoning
            >>> chain = extract_reasoning("Step 1: A. Step 2: B.")
            >>> analysis = analyze_reasoning(chain)
            >>> json_str = json.dumps(analysis.to_dict())
            >>> print(type(json_str))
            <class 'str'>

        Accessing analysis details:

            >>> from insideLLMs.evaluation.reasoning import extract_reasoning, analyze_reasoning
            >>> chain = extract_reasoning("Because A, therefore B, thus C.")
            >>> d = analyze_reasoning(chain).to_dict()
            >>> print(f"Validity: {d['logical_validity']:.2f}, "
            ...       f"Coherence: {d['coherence_score']:.2f}")
            Validity: 0.65, Coherence: 0.33
        """
        return {
            "chain": self.chain.to_dict(),
            "logical_validity": self.logical_validity,
            "coherence_score": self.coherence_score,
            "completeness_score": self.completeness_score,
            "avg_step_quality": sum(self.step_quality_scores) / len(self.step_quality_scores)
            if self.step_quality_scores
            else 0,
            "num_fallacies": len(self.identified_fallacies),
            "identified_fallacies": self.identified_fallacies,
            "missing_steps": self.missing_steps,
            "overall_quality": self.overall_quality.value,
        }


@dataclass
class CoTEvaluation:
    """
    Evaluation results for a Chain-of-Thought response.

    Captures comprehensive metrics from evaluating a model's response to a
    Chain-of-Thought prompt, including answer correctness, reasoning quality,
    step accuracy, and suggestions for improvement.

    Attributes
    ----------
    prompt : str
        The original prompt or question that was asked.

    response : str
        The model's complete response text.

    chain : ReasoningChain
        The extracted reasoning chain from the response.

    answer_correct : Optional[bool]
        Whether the final answer matches the expected answer. None if no
        expected answer was provided for comparison.

    reasoning_score : float
        Overall reasoning quality score from 0.0 to 1.0. Weighted combination
        of logical validity (40%), coherence (30%), and completeness (30%).

    step_accuracy : float
        Average quality score across all reasoning steps from 0.0 to 1.0.

    explanation_quality : float
        Score from 0.0 to 1.0 measuring how well the response explains the
        reasoning process. Based on structure, length, and use of reasoning
        markers.

    improvements : list[str]
        Suggested improvements based on identified weaknesses. May include
        suggestions for logical connections, coherence, completeness, or
        fallacy correction.

    Examples
    --------
    Evaluating a CoT response:

        >>> from insideLLMs.evaluation.reasoning import evaluate_cot
        >>> prompt = "What is 5 + 7?"
        >>> response = '''
        ... Let me think step by step.
        ... Step 1: I need to add 5 and 7.
        ... Step 2: 5 + 7 = 12
        ... Therefore, the answer is 12.
        ... '''
        >>> evaluation = evaluate_cot(prompt, response, expected_answer="12")
        >>> print(f"Correct: {evaluation.answer_correct}")
        Correct: True
        >>> print(f"Reasoning score: {evaluation.reasoning_score:.2f}")
        Reasoning score: 0.75

    Checking for improvement suggestions:

        >>> from insideLLMs.evaluation.reasoning import evaluate_cot
        >>> response = "The answer is 42."  # Minimal reasoning
        >>> eval = evaluate_cot("Calculate 6*7", response, "42")
        >>> for imp in eval.improvements:
        ...     print(f"Suggestion: {imp}")
        Suggestion: Strengthen logical connections between steps
        Suggestion: Add missing steps to complete the reasoning chain

    Comparing evaluations:

        >>> from insideLLMs.evaluation.reasoning import evaluate_cot
        >>> eval1 = evaluate_cot("2+2?", "It's 4.", "4")
        >>> eval2 = evaluate_cot("2+2?", "Step 1: Add 2+2. Step 2: Result is 4.", "4")
        >>> print(f"Eval1 steps: {len(eval1.chain.steps)}")
        Eval1 steps: 1
        >>> print(f"Eval2 steps: {len(eval2.chain.steps)}")
        Eval2 steps: 2

    Analyzing explanation quality:

        >>> from insideLLMs.evaluation.reasoning import evaluate_cot
        >>> detailed = '''
        ... Given the problem of calculating area.
        ... First, recall that area = length * width.
        ... We have length = 5 and width = 3.
        ... Therefore, area = 5 * 3 = 15 square units.
        ... In conclusion, the area is 15 square units.
        ... '''
        >>> eval = evaluate_cot("Find the area", detailed)
        >>> print(f"Explanation quality: {eval.explanation_quality:.2f}")
        Explanation quality: 0.90

    See Also
    --------
    CoTEvaluator : Creates CoTEvaluation objects
    ReasoningChain : The extracted chain
    ReasoningReport : Aggregated report from multiple evaluations
    """

    prompt: str
    response: str
    chain: ReasoningChain
    answer_correct: Optional[bool]
    reasoning_score: float
    step_accuracy: float
    explanation_quality: float
    improvements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the evaluation to a dictionary representation.

        Serializes evaluation results to a dictionary format. The prompt
        is truncated to 100 characters for brevity.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - prompt: Truncated prompt text
            - answer_correct: Boolean or None
            - reasoning_score: Overall reasoning score
            - step_accuracy: Average step quality
            - explanation_quality: Explanation quality score
            - num_steps: Number of steps in chain
            - improvements: List of improvement suggestions

        Examples
        --------
        Basic serialization:

            >>> from insideLLMs.evaluation.reasoning import evaluate_cot
            >>> eval = evaluate_cot("What is 2+2?", "2+2=4, so 4.", "4")
            >>> d = eval.to_dict()
            >>> print(f"Correct: {d['answer_correct']}")
            Correct: True

        JSON export for reporting:

            >>> import json
            >>> from insideLLMs.evaluation.reasoning import evaluate_cot
            >>> eval = evaluate_cot("Question", "Answer with reasoning")
            >>> json_str = json.dumps(eval.to_dict())
            >>> print(type(json_str))
            <class 'str'>

        Aggregating evaluation metrics:

            >>> from insideLLMs.evaluation.reasoning import evaluate_cot
            >>> evals = [
            ...     evaluate_cot("Q1", "A1 because X"),
            ...     evaluate_cot("Q2", "A2 therefore Y")
            ... ]
            >>> dicts = [e.to_dict() for e in evals]
            >>> avg_score = sum(d['reasoning_score'] for d in dicts) / len(dicts)
            >>> print(f"Average: {avg_score:.2f}")
            Average: 0.42
        """
        return {
            "prompt": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "answer_correct": self.answer_correct,
            "reasoning_score": self.reasoning_score,
            "step_accuracy": self.step_accuracy,
            "explanation_quality": self.explanation_quality,
            "num_steps": len(self.chain.steps),
            "improvements": self.improvements,
        }


@dataclass
class ReasoningReport:
    """
    Aggregated report on reasoning capabilities across multiple evaluations.

    Summarizes reasoning analysis results from a batch of evaluations,
    providing aggregate metrics, distributions, and recommendations for
    improving reasoning quality.

    Attributes
    ----------
    total_evaluations : int
        Total number of evaluations included in this report.

    avg_reasoning_score : float
        Average reasoning score across all evaluations (0.0 to 1.0).

    avg_step_accuracy : float
        Average step quality score across all evaluations (0.0 to 1.0).

    reasoning_type_breakdown : dict[str, float]
        Distribution of reasoning types as proportions. Keys are reasoning
        type values (e.g., "deductive", "mathematical"), values are
        proportions summing to 1.0.

    common_fallacies : list[tuple[str, int]]
        Most common fallacies detected, sorted by frequency. Each tuple
        contains (fallacy_name, count). Limited to top 5.

    quality_distribution : dict[str, int]
        Distribution of quality levels. Keys are quality values
        (e.g., "excellent", "good"), values are counts.

    recommendations : list[str]
        Actionable recommendations for improving reasoning quality based
        on identified patterns and weaknesses.

    Examples
    --------
    Generating a report from evaluations:

        >>> from insideLLMs.evaluation.reasoning import CoTEvaluator
        >>> evaluator = CoTEvaluator()
        >>> prompts = ["What is 1+1?", "What is 2+2?", "What is 3+3?"]
        >>> responses = [
        ...     "1+1 = 2 because addition combines values.",
        ...     "Step 1: 2+2. Step 2: equals 4. Therefore 4.",
        ...     "Given 3+3, calculate 6. The answer is 6."
        ... ]
        >>> evals = evaluator.evaluate_batch(prompts, responses, ["2", "4", "6"])
        >>> report = evaluator.generate_report(evals)
        >>> print(f"Total: {report.total_evaluations}")
        Total: 3
        >>> print(f"Avg score: {report.avg_reasoning_score:.2f}")
        Avg score: 0.58

    Analyzing reasoning type distribution:

        >>> from insideLLMs.evaluation.reasoning import CoTEvaluator
        >>> evaluator = CoTEvaluator()
        >>> # Assume we have a report
        >>> # for rtype, proportion in report.reasoning_type_breakdown.items():
        >>> #     print(f"{rtype}: {proportion:.1%}")
        >>> # Output: mathematical: 60.0%, deductive: 40.0%

    Checking quality distribution:

        >>> from insideLLMs.evaluation.reasoning import CoTEvaluator
        >>> evaluator = CoTEvaluator()
        >>> # Assume we have evaluations
        >>> # report = evaluator.generate_report(evals)
        >>> # for quality, count in report.quality_distribution.items():
        >>> #     print(f"{quality}: {count} evaluations")

    Using recommendations:

        >>> from insideLLMs.evaluation.reasoning import CoTEvaluator
        >>> evaluator = CoTEvaluator()
        >>> # report = evaluator.generate_report(evals)
        >>> # if report.avg_reasoning_score < 0.5:
        >>> #     print("Recommendations:")
        >>> #     for rec in report.recommendations:
        >>> #         print(f"  - {rec}")

    See Also
    --------
    CoTEvaluator.generate_report : Creates ReasoningReport
    CoTEvaluation : Individual evaluation results
    ReasoningQuality : Quality levels used in distribution
    """

    total_evaluations: int
    avg_reasoning_score: float
    avg_step_accuracy: float
    reasoning_type_breakdown: dict[str, float]
    common_fallacies: list[tuple[str, int]]
    quality_distribution: dict[str, int]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the report to a dictionary representation.

        Serializes all report data to a dictionary format suitable for
        JSON export or API responses.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all report fields:
            - total_evaluations: Count of evaluations
            - avg_reasoning_score: Average reasoning score
            - avg_step_accuracy: Average step accuracy
            - reasoning_type_breakdown: Type distribution
            - common_fallacies: Top fallacies with counts
            - quality_distribution: Quality level counts
            - recommendations: List of recommendations

        Examples
        --------
        Basic serialization:

            >>> from insideLLMs.evaluation.reasoning import CoTEvaluator
            >>> evaluator = CoTEvaluator()
            >>> evals = evaluator.evaluate_batch(
            ...     ["Q1", "Q2"],
            ...     ["A1 because X", "A2 therefore Y"]
            ... )
            >>> report = evaluator.generate_report(evals)
            >>> d = report.to_dict()
            >>> print(f"Total: {d['total_evaluations']}")
            Total: 2

        JSON export for logging:

            >>> import json
            >>> from insideLLMs.evaluation.reasoning import CoTEvaluator
            >>> evaluator = CoTEvaluator()
            >>> evals = evaluator.evaluate_batch(["Q"], ["A"])
            >>> report = evaluator.generate_report(evals)
            >>> json_str = json.dumps(report.to_dict())
            >>> print(type(json_str))
            <class 'str'>

        Accessing breakdown data:

            >>> from insideLLMs.evaluation.reasoning import CoTEvaluator
            >>> evaluator = CoTEvaluator()
            >>> evals = evaluator.evaluate_batch(["Calc 5+5"], ["5+5=10"])
            >>> d = evaluator.generate_report(evals).to_dict()
            >>> print(f"Types: {list(d['reasoning_type_breakdown'].keys())}")
            Types: ['mathematical']
        """
        return {
            "total_evaluations": self.total_evaluations,
            "avg_reasoning_score": self.avg_reasoning_score,
            "avg_step_accuracy": self.avg_step_accuracy,
            "reasoning_type_breakdown": self.reasoning_type_breakdown,
            "common_fallacies": self.common_fallacies,
            "quality_distribution": self.quality_distribution,
            "recommendations": self.recommendations,
        }


class ReasoningExtractor:
    """
    Extracts structured reasoning chains from unstructured text.

    Uses pattern matching and heuristics to identify reasoning steps,
    classify their types, and construct a ReasoningChain from natural
    language text. Supports various formats including numbered steps,
    ordinal markers, and sentence-based extraction.

    Attributes
    ----------
    STEP_PATTERNS : list[str]
        Regular expression patterns for detecting reasoning steps.
        Includes patterns for "Step N:", numbered lists, and ordinal
        words (first, second, etc.).

    PREMISE_MARKERS : list[str]
        Keywords indicating premise steps: "given", "assume", "let",
        "suppose", "we know".

    INFERENCE_MARKERS : list[str]
        Keywords indicating inference steps: "therefore", "thus",
        "hence", "so", "this means".

    CONCLUSION_MARKERS : list[str]
        Keywords indicating conclusion steps: "therefore", "in conclusion",
        "finally", "the answer is".

    CALCULATION_MARKERS : list[str]
        Characters/words indicating calculation steps: mathematical
        operators and words like "calculate", "compute".

    Examples
    --------
    Basic extraction:

        >>> from insideLLMs.evaluation.reasoning import ReasoningExtractor
        >>> extractor = ReasoningExtractor()
        >>> text = '''
        ... Step 1: All birds have wings.
        ... Step 2: Penguins are birds.
        ... Step 3: Therefore, penguins have wings.
        ... '''
        >>> chain = extractor.extract(text)
        >>> print(f"Extracted {len(chain.steps)} steps")
        Extracted 3 steps

    Extraction with numbered lists:

        >>> from insideLLMs.evaluation.reasoning import ReasoningExtractor
        >>> extractor = ReasoningExtractor()
        >>> text = '''
        ... 1) First we identify the variables
        ... 2) Then we set up the equation
        ... 3) Finally we solve for x
        ... '''
        >>> chain = extractor.extract(text)
        >>> print(f"Valid: {chain.is_valid}")
        Valid: True

    Extraction from prose:

        >>> from insideLLMs.evaluation.reasoning import ReasoningExtractor
        >>> extractor = ReasoningExtractor()
        >>> text = "Given x=5. Since x is positive, therefore x+1 is greater than x."
        >>> chain = extractor.extract(text)
        >>> print(f"Type: {chain.reasoning_type.value}")
        Type: mathematical

    Checking completeness:

        >>> from insideLLMs.evaluation.reasoning import ReasoningExtractor
        >>> extractor = ReasoningExtractor()
        >>> complete = "Given A. Let B. Suppose C. Therefore D. In conclusion, E."
        >>> incomplete = "Maybe X. Perhaps Y."
        >>> print(f"Complete: {extractor.extract(complete).completeness:.1f}")
        Complete: 1.0
        >>> print(f"Incomplete: {extractor.extract(incomplete).completeness:.1f}")
        Incomplete: 0.3

    See Also
    --------
    ReasoningChain : Output structure
    ReasoningStep : Individual step structure
    extract_reasoning : Convenience function
    """

    # Patterns for step detection
    STEP_PATTERNS = [
        r"(?:step|stage)\s*(\d+)[:\.]?\s*(.+?)(?=(?:step|stage)\s*\d+|$)",
        r"(\d+)[.)]\s*(.+?)(?=\d+[.)]|$)",
        r"(?:first|second|third|fourth|fifth|next|then|finally)[,:]?\s*(.+?)(?=(?:first|second|third|fourth|fifth|next|then|finally)|$)",
    ]

    # Markers for different step types
    PREMISE_MARKERS = ["given", "assume", "let", "suppose", "we know"]
    INFERENCE_MARKERS = ["therefore", "thus", "hence", "so", "this means"]
    CONCLUSION_MARKERS = ["therefore", "in conclusion", "finally", "the answer is"]
    CALCULATION_MARKERS = ["=", "calculate", "compute", "+", "-", "*", "/"]

    def extract(self, text: str) -> ReasoningChain:
        """
        Extract a reasoning chain from unstructured text.

        Parses the input text to identify reasoning steps, classify their types,
        determine the overall reasoning pattern, and construct a complete
        ReasoningChain structure.

        Parameters
        ----------
        text : str
            The text to extract reasoning from. Can be in various formats:
            - Numbered steps ("Step 1:", "1)", "1.")
            - Ordinal words ("First,", "Second,", "Finally,")
            - Plain prose (sentences will be extracted as steps)

        Returns
        -------
        ReasoningChain
            A structured chain containing:
            - steps: List of extracted ReasoningStep objects
            - conclusion: The identified conclusion (if any)
            - reasoning_type: The classified reasoning type
            - is_valid: True if at least one step was found
            - completeness: Score indicating chain completeness

        Examples
        --------
        Extracting from numbered steps:

            >>> from insideLLMs.evaluation.reasoning import ReasoningExtractor
            >>> extractor = ReasoningExtractor()
            >>> text = "Step 1: Identify the problem. Step 2: Analyze causes. Step 3: Propose solution."
            >>> chain = extractor.extract(text)
            >>> print(f"Found {len(chain.steps)} steps")
            Found 3 steps

        Extracting from prose:

            >>> extractor = ReasoningExtractor()
            >>> text = "Given that x equals 5. Since x is positive, we can conclude x squared is also positive."
            >>> chain = extractor.extract(text)
            >>> print(f"Reasoning type: {chain.reasoning_type.value}")
            Reasoning type: mathematical

        Handling incomplete reasoning:

            >>> extractor = ReasoningExtractor()
            >>> text = "Maybe this is true."
            >>> chain = extractor.extract(text)
            >>> print(f"Valid: {chain.is_valid}, Completeness: {chain.completeness}")
            Valid: True, Completeness: 0.2

        Extracting with conclusion:

            >>> extractor = ReasoningExtractor()
            >>> text = "All dogs are mammals. Rex is a dog. Therefore, Rex is a mammal."
            >>> chain = extractor.extract(text)
            >>> print(f"Conclusion: {chain.conclusion}")
            Conclusion: rex is a mammal
        """
        steps = []

        # Try numbered step patterns
        for pattern in self.STEP_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches and len(matches) >= 2:
                for i, match in enumerate(matches):
                    content = match[-1].strip() if isinstance(match, tuple) else match.strip()

                    if content and len(content) > 5:
                        step_type = self._classify_step(content)
                        step = ReasoningStep(
                            content=content,
                            step_number=i + 1,
                            step_type=step_type,
                            confidence=self._estimate_confidence(content),
                        )
                        steps.append(step)
                break

        # If no numbered steps, try to split by sentences
        if not steps:
            steps = self._extract_from_sentences(text)

        # Find conclusion
        conclusion = self._extract_conclusion(text)

        # Determine reasoning type
        reasoning_type = self._classify_reasoning_type(text)

        # Calculate completeness
        completeness = self._calculate_completeness(steps, conclusion)

        return ReasoningChain(
            steps=steps,
            conclusion=conclusion,
            reasoning_type=reasoning_type,
            is_valid=len(steps) > 0,
            completeness=completeness,
        )

    def _extract_from_sentences(self, text: str) -> list[ReasoningStep]:
        """
        Extract reasoning steps from individual sentences.

        Fallback extraction method when no explicit step markers are found.
        Splits text by sentence boundaries and creates a step for each
        substantial sentence.

        Parameters
        ----------
        text : str
            The text to extract sentences from.

        Returns
        -------
        list[ReasoningStep]
            List of reasoning steps (max 10) extracted from sentences.
            Only sentences with more than 10 characters are included.

        Examples
        --------
            >>> extractor = ReasoningExtractor()
            >>> steps = extractor._extract_from_sentences("A is true. B follows. C is the result.")
            >>> len(steps)
            3
        """
        sentences = re.split(r"[.!?]+", text)
        steps = []

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 10:
                step_type = self._classify_step(sentence)
                step = ReasoningStep(
                    content=sentence,
                    step_number=i + 1,
                    step_type=step_type,
                    confidence=self._estimate_confidence(sentence),
                )
                steps.append(step)

        return steps[:10]  # Limit to 10 steps

    def _classify_step(self, content: str) -> ReasoningStepType:
        """
        Classify the type of a reasoning step based on its content.

        Uses keyword matching to determine whether a step represents a
        premise, inference, calculation, or conclusion.

        Parameters
        ----------
        content : str
            The text content of the step to classify.

        Returns
        -------
        ReasoningStepType
            The classified type. Priority order:
            1. PREMISE (if premise markers found)
            2. CONCLUSION (if conclusion markers found)
            3. CALCULATION (if math operators/words found)
            4. INFERENCE (if inference markers found, or default)

        Examples
        --------
            >>> extractor = ReasoningExtractor()
            >>> extractor._classify_step("Given that x = 5")
            <ReasoningStepType.PREMISE: 'premise'>
            >>> extractor._classify_step("Therefore, y = 10")
            <ReasoningStepType.CONCLUSION: 'conclusion'>
            >>> extractor._classify_step("Calculate 5 + 3 = 8")
            <ReasoningStepType.CALCULATION: 'calculation'>
        """
        content_lower = content.lower()

        for marker in self.PREMISE_MARKERS:
            if marker in content_lower:
                return ReasoningStepType.PREMISE

        for marker in self.CONCLUSION_MARKERS:
            if marker in content_lower:
                return ReasoningStepType.CONCLUSION

        for marker in self.CALCULATION_MARKERS:
            if marker in content:
                return ReasoningStepType.CALCULATION

        for marker in self.INFERENCE_MARKERS:
            if marker in content_lower:
                return ReasoningStepType.INFERENCE

        return ReasoningStepType.INFERENCE

    def _estimate_confidence(self, content: str) -> float:
        """
        Estimate the confidence level of a reasoning step.

        Analyzes linguistic markers to determine how confident the
        reasoning appears to be. High-confidence words increase the
        score while hedging language decreases it.

        Parameters
        ----------
        content : str
            The text content of the step to analyze.

        Returns
        -------
        float
            Confidence score between 0.1 and 0.95. Base score is 0.5.
            Adjusted based on:
            - High confidence markers (+0.2): "clearly", "obviously", "certainly"
            - Low confidence markers (-0.2): "maybe", "perhaps", "possibly", "might"
            - Evidence markers (+0.1): "because", "since", "as shown"

        Examples
        --------
            >>> extractor = ReasoningExtractor()
            >>> extractor._estimate_confidence("This is clearly true")
            0.7
            >>> extractor._estimate_confidence("Maybe this could be right")
            0.3
            >>> extractor._estimate_confidence("This is true because of X")
            0.6
        """
        confidence = 0.5
        content_lower = content.lower()

        # High confidence markers
        if any(m in content_lower for m in ["clearly", "obviously", "certainly"]):
            confidence += 0.2

        # Low confidence markers
        if any(m in content_lower for m in ["maybe", "perhaps", "possibly", "might"]):
            confidence -= 0.2

        # Evidence markers boost confidence
        if any(m in content_lower for m in ["because", "since", "as shown"]):
            confidence += 0.1

        return max(0.1, min(0.95, confidence))

    def _extract_conclusion(self, text: str) -> Optional[str]:
        """
        Extract the conclusion from reasoning text.

        Looks for explicit conclusion markers and extracts the following
        content. Falls back to the last sentence if no explicit markers
        are found.

        Parameters
        ----------
        text : str
            The full text to extract conclusion from.

        Returns
        -------
        Optional[str]
            The extracted conclusion text, or None if text is empty.
            Returns lowercase version of the conclusion.

        Examples
        --------
            >>> extractor = ReasoningExtractor()
            >>> extractor._extract_conclusion("A is true. Therefore, B follows.")
            'b follows'
            >>> extractor._extract_conclusion("Step 1: X. The answer is 42.")
            '42'
            >>> extractor._extract_conclusion("Just one statement")
            'Just one statement'
        """
        text_lower = text.lower()

        # Look for explicit conclusion markers
        patterns = [
            r"(?:therefore|thus|hence|so|in conclusion|finally)[,:]?\s*(.+?)(?:\.|$)",
            r"(?:the answer is|the result is)[:\s]*(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip()

        # Return last sentence if no explicit conclusion
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if sentences:
            return sentences[-1]

        return None

    def _classify_reasoning_type(self, text: str) -> ReasoningType:
        """
        Classify the overall type of reasoning used in the text.

        Analyzes the text for keywords and patterns associated with
        different reasoning types. Returns the first matching type
        found in priority order.

        Parameters
        ----------
        text : str
            The full text to classify.

        Returns
        -------
        ReasoningType
            The classified reasoning type. Priority order:
            1. MATHEMATICAL (math operators or calculation words)
            2. CAUSAL (cause-effect language)
            3. TEMPORAL (time-based language)
            4. ANALOGICAL (comparison language)
            5. INDUCTIVE (generalization language)
            6. DEDUCTIVE (default)

        Examples
        --------
            >>> extractor = ReasoningExtractor()
            >>> extractor._classify_reasoning_type("Calculate 5 + 3 = 8")
            <ReasoningType.MATHEMATICAL: 'mathematical'>
            >>> extractor._classify_reasoning_type("This causes that effect")
            <ReasoningType.CAUSAL: 'causal'>
            >>> extractor._classify_reasoning_type("Before A, then B happens")
            <ReasoningType.TEMPORAL: 'temporal'>
            >>> extractor._classify_reasoning_type("All premises lead to conclusion")
            <ReasoningType.DEDUCTIVE: 'deductive'>
        """
        text_lower = text.lower()

        # Mathematical reasoning
        if any(op in text for op in ["+", "-", "*", "/", "="]) or any(
            word in text_lower for word in ["calculate", "compute", "sum", "multiply"]
        ):
            return ReasoningType.MATHEMATICAL

        # Causal reasoning
        if any(
            word in text_lower for word in ["because", "cause", "effect", "leads to", "results in"]
        ):
            return ReasoningType.CAUSAL

        # Temporal reasoning
        if any(word in text_lower for word in ["before", "after", "then", "when", "during"]):
            return ReasoningType.TEMPORAL

        # Analogical reasoning
        if any(word in text_lower for word in ["like", "similar to", "just as", "analogous"]):
            return ReasoningType.ANALOGICAL

        # Inductive reasoning
        if any(word in text_lower for word in ["generally", "usually", "most", "pattern"]):
            return ReasoningType.INDUCTIVE

        # Default to deductive
        return ReasoningType.DEDUCTIVE

    def _calculate_completeness(
        self,
        steps: list[ReasoningStep],
        conclusion: Optional[str],
    ) -> float:
        """
        Calculate the completeness of a reasoning chain.

        Scores the chain based on presence of key components: multiple
        steps, premises, inferences, and conclusion.

        Parameters
        ----------
        steps : list[ReasoningStep]
            The list of reasoning steps in the chain.
        conclusion : Optional[str]
            The extracted conclusion, if any.

        Returns
        -------
        float
            Completeness score from 0.0 to 1.0:
            - +0.3 for having 2+ steps
            - +0.2 for having at least one premise
            - +0.2 for having at least one inference
            - +0.3 for having a conclusion

        Examples
        --------
            >>> from insideLLMs.evaluation.reasoning import ReasoningStep, ReasoningStepType
            >>> extractor = ReasoningExtractor()
            >>> # Complete chain
            >>> steps = [
            ...     ReasoningStep("Given A", 1, ReasoningStepType.PREMISE),
            ...     ReasoningStep("Therefore B", 2, ReasoningStepType.INFERENCE),
            ...     ReasoningStep("In conclusion C", 3, ReasoningStepType.CONCLUSION)
            ... ]
            >>> extractor._calculate_completeness(steps, "C")
            1.0
            >>> # Incomplete chain
            >>> extractor._calculate_completeness([], None)
            0.0
        """
        if not steps:
            return 0.0

        score = 0.0

        # Has multiple steps
        if len(steps) >= 2:
            score += 0.3

        # Has premise
        if any(s.step_type == ReasoningStepType.PREMISE for s in steps):
            score += 0.2

        # Has inference
        if any(s.step_type == ReasoningStepType.INFERENCE for s in steps):
            score += 0.2

        # Has conclusion
        if conclusion or any(s.step_type == ReasoningStepType.CONCLUSION for s in steps):
            score += 0.3

        return min(1.0, score)


class ReasoningAnalyzer:
    """
    Analyzes reasoning chains for quality, validity, and potential issues.

    Provides comprehensive analysis of reasoning chains including logical
    validity assessment, coherence scoring, fallacy detection, and
    identification of missing reasoning steps.

    Attributes
    ----------
    FALLACY_PATTERNS : dict[str, list[str]]
        Mapping of fallacy names to keyword patterns used for detection.
        Includes common logical fallacies:
        - circular_reasoning: Self-referential arguments
        - hasty_generalization: Overly broad claims
        - false_dichotomy: Artificial binary choices
        - appeal_to_authority: Relying on authority claims
        - ad_hominem: Personal attacks
        - straw_man: Misrepresenting arguments
        - slippery_slope: Unfounded chain of consequences

    Examples
    --------
    Basic chain analysis:

        >>> from insideLLMs.evaluation.reasoning import ReasoningExtractor, ReasoningAnalyzer
        >>> extractor = ReasoningExtractor()
        >>> analyzer = ReasoningAnalyzer()
        >>> chain = extractor.extract("Given A. Therefore B. Thus C.")
        >>> analysis = analyzer.analyze(chain)
        >>> print(f"Quality: {analysis.overall_quality.value}")
        Quality: adequate

    Detecting fallacies:

        >>> from insideLLMs.evaluation.reasoning import ReasoningExtractor, ReasoningAnalyzer
        >>> extractor = ReasoningExtractor()
        >>> analyzer = ReasoningAnalyzer()
        >>> chain = extractor.extract("Everyone knows this is true. All people agree.")
        >>> analysis = analyzer.analyze(chain)
        >>> print(f"Fallacies: {analysis.identified_fallacies}")
        Fallacies: ['hasty_generalization']

    Analyzing coherence:

        >>> from insideLLMs.evaluation.reasoning import ReasoningExtractor, ReasoningAnalyzer
        >>> extractor = ReasoningExtractor()
        >>> analyzer = ReasoningAnalyzer()
        >>> coherent = "Dogs are mammals. Mammals are warm-blooded. Therefore dogs are warm-blooded."
        >>> chain = extractor.extract(coherent)
        >>> analysis = analyzer.analyze(chain)
        >>> print(f"Coherence: {analysis.coherence_score:.2f}")
        Coherence: 0.25

    Identifying missing steps:

        >>> from insideLLMs.evaluation.reasoning import ReasoningExtractor, ReasoningAnalyzer
        >>> extractor = ReasoningExtractor()
        >>> analyzer = ReasoningAnalyzer()
        >>> chain = extractor.extract("Therefore X. Hence Y.")
        >>> analysis = analyzer.analyze(chain)
        >>> for missing in analysis.missing_steps:
        ...     print(f"Missing: {missing}")
        Missing: No clear premise or starting point

    See Also
    --------
    ChainAnalysis : Output structure containing analysis results
    ReasoningChain : Input structure to analyze
    analyze_reasoning : Convenience function
    """

    # Common logical fallacies
    FALLACY_PATTERNS = {
        "circular_reasoning": ["because it is", "since it's true", "proves itself"],
        "hasty_generalization": ["all", "always", "never", "everyone", "nobody"],
        "false_dichotomy": ["either", "only two", "must be one or"],
        "appeal_to_authority": ["expert says", "scientists say", "studies show"],
        "ad_hominem": ["stupid", "idiot", "fool", "ignorant"],
        "straw_man": ["they think", "opponents believe", "critics say"],
        "slippery_slope": ["will lead to", "eventually", "if we allow"],
    }

    def analyze(self, chain: ReasoningChain) -> ChainAnalysis:
        """
        Perform comprehensive analysis of a reasoning chain.

        Evaluates the chain across multiple dimensions including logical
        validity, coherence between steps, completeness, and potential
        logical fallacies.

        Parameters
        ----------
        chain : ReasoningChain
            The reasoning chain to analyze.

        Returns
        -------
        ChainAnalysis
            Complete analysis results including:
            - logical_validity: How logically sound the chain is
            - coherence_score: How well steps connect
            - completeness_score: Presence of key components
            - step_quality_scores: Individual step scores
            - identified_fallacies: Detected logical fallacies
            - missing_steps: Identified gaps in reasoning
            - overall_quality: Categorical quality assessment

        Examples
        --------
        Analyzing a well-structured chain:

            >>> from insideLLMs.evaluation.reasoning import ReasoningExtractor, ReasoningAnalyzer
            >>> extractor = ReasoningExtractor()
            >>> analyzer = ReasoningAnalyzer()
            >>> text = '''
            ... Given that all cats are mammals.
            ... Since Whiskers is a cat.
            ... Therefore, Whiskers is a mammal.
            ... '''
            >>> chain = extractor.extract(text)
            >>> analysis = analyzer.analyze(chain)
            >>> print(f"Validity: {analysis.logical_validity:.2f}")
            Validity: 0.85

        Analyzing a poor chain:

            >>> text = "Maybe X. Perhaps Y."
            >>> chain = extractor.extract(text)
            >>> analysis = analyzer.analyze(chain)
            >>> print(f"Quality: {analysis.overall_quality.value}")
            Quality: poor

        Getting step-by-step scores:

            >>> text = "Step 1: Given A. Step 2: Therefore B because A."
            >>> chain = extractor.extract(text)
            >>> analysis = analyzer.analyze(chain)
            >>> for i, score in enumerate(analysis.step_quality_scores):
            ...     print(f"Step {i+1} quality: {score:.2f}")
            Step 1 quality: 0.70
            Step 2 quality: 0.80

        Using analysis for improvement:

            >>> analysis = analyzer.analyze(chain)
            >>> if analysis.logical_validity < 0.5:
            ...     print("Recommendation: Strengthen logical connections")
            >>> if analysis.identified_fallacies:
            ...     print(f"Warning: Fallacies detected: {analysis.identified_fallacies}")
        """
        # Calculate logical validity
        validity = self._check_logical_validity(chain)

        # Calculate coherence
        coherence = self._calculate_coherence(chain)

        # Calculate completeness
        completeness = chain.completeness

        # Score each step
        step_scores = [self._score_step(step) for step in chain.steps]

        # Identify fallacies
        fallacies = self._identify_fallacies(chain)

        # Identify missing steps
        missing = self._identify_missing_steps(chain)

        # Determine overall quality
        avg_score = sum([validity, coherence, completeness]) / 3
        if avg_score >= 0.8:
            quality = ReasoningQuality.EXCELLENT
        elif avg_score >= 0.6:
            quality = ReasoningQuality.GOOD
        elif avg_score >= 0.4:
            quality = ReasoningQuality.ADEQUATE
        elif avg_score >= 0.2:
            quality = ReasoningQuality.POOR
        else:
            quality = ReasoningQuality.INVALID

        return ChainAnalysis(
            chain=chain,
            logical_validity=validity,
            coherence_score=coherence,
            completeness_score=completeness,
            step_quality_scores=step_scores,
            identified_fallacies=fallacies,
            missing_steps=missing,
            overall_quality=quality,
        )

    def _check_logical_validity(self, chain: ReasoningChain) -> float:
        """
        Check the logical validity of a reasoning chain.

        Assesses how logically sound the chain structure is based on
        presence of multiple steps, inferences, and premises.

        Parameters
        ----------
        chain : ReasoningChain
            The chain to check for validity.

        Returns
        -------
        float
            Validity score from 0.0 to 1.0:
            - Base score: 0.5
            - +0.2 for having 2+ steps
            - +0.15 for having inference steps
            - +0.15 for having premise steps

        Examples
        --------
            >>> from insideLLMs.evaluation.reasoning import ReasoningExtractor, ReasoningAnalyzer
            >>> analyzer = ReasoningAnalyzer()
            >>> chain = ReasoningExtractor().extract("Given A. Therefore B.")
            >>> print(f"Validity: {analyzer._check_logical_validity(chain):.2f}")
            Validity: 0.85
        """
        if not chain.steps:
            return 0.0

        validity = 0.5

        # Has clear flow
        if len(chain.steps) >= 2:
            validity += 0.2

        # Steps build on each other
        inference_count = sum(1 for s in chain.steps if s.step_type == ReasoningStepType.INFERENCE)
        if inference_count > 0:
            validity += 0.15

        # Has evidence/premises
        premise_count = sum(1 for s in chain.steps if s.step_type == ReasoningStepType.PREMISE)
        if premise_count > 0:
            validity += 0.15

        return min(1.0, validity)

    def _calculate_coherence(self, chain: ReasoningChain) -> float:
        """
        Calculate coherence between adjacent steps in the chain.

        Uses Jaccard similarity (word overlap) between consecutive steps
        to measure how well they connect logically.

        Parameters
        ----------
        chain : ReasoningChain
            The chain to analyze for coherence.

        Returns
        -------
        float
            Average coherence score from 0.0 to 1.0. Returns 1.0 for
            single-step chains, 0.0 for empty chains.

        Examples
        --------
            >>> from insideLLMs.evaluation.reasoning import ReasoningExtractor, ReasoningAnalyzer
            >>> analyzer = ReasoningAnalyzer()
            >>> # High coherence - shared terms
            >>> chain = ReasoningExtractor().extract(
            ...     "Dogs are animals. Animals need food. Dogs need food."
            ... )
            >>> coherence = analyzer._calculate_coherence(chain)
            >>> print(f"Coherence: {coherence:.2f}")
            Coherence: 0.27
        """
        if len(chain.steps) < 2:
            return 1.0 if chain.steps else 0.0

        coherence = 0.0
        step_pairs = 0

        for i in range(len(chain.steps) - 1):
            current = chain.steps[i].content.lower().split()
            next_step = chain.steps[i + 1].content.lower().split()

            # Calculate word overlap
            overlap = len(set(current) & set(next_step))
            total = len(set(current) | set(next_step))

            if total > 0:
                coherence += overlap / total
                step_pairs += 1

        return coherence / step_pairs if step_pairs > 0 else 0.0

    def _score_step(self, step: ReasoningStep) -> float:
        """
        Score the quality of a single reasoning step.

        Evaluates a step based on content length, presence of reasoning
        markers, and confidence level.

        Parameters
        ----------
        step : ReasoningStep
            The step to score.

        Returns
        -------
        float
            Quality score from 0.0 to 1.0:
            - Base: 0.5
            - +0.2 for appropriate length (5-50 words)
            - -0.1 for very short content (<5 words)
            - +0.2 for reasoning markers
            - +0.0 to +0.095 based on confidence

        Examples
        --------
            >>> from insideLLMs.evaluation.reasoning import ReasoningStep, ReasoningAnalyzer
            >>> analyzer = ReasoningAnalyzer()
            >>> step = ReasoningStep("This is true because of X", 1)
            >>> print(f"Score: {analyzer._score_step(step):.2f}")
            Score: 0.75
        """
        score = 0.5

        # Content length
        words = len(step.content.split())
        if 5 <= words <= 50:
            score += 0.2
        elif words < 5:
            score -= 0.1

        # Has reasoning markers
        content_lower = step.content.lower()
        if any(m in content_lower for m in ["because", "therefore", "since", "thus"]):
            score += 0.2

        # Step confidence
        score += step.confidence * 0.1

        return min(1.0, max(0.0, score))

    def _identify_fallacies(self, chain: ReasoningChain) -> list[str]:
        """
        Identify logical fallacies present in the reasoning chain.

        Scans the combined text of all steps for patterns associated
        with common logical fallacies.

        Parameters
        ----------
        chain : ReasoningChain
            The chain to scan for fallacies.

        Returns
        -------
        list[str]
            List of identified fallacy names. Each fallacy appears at
            most once even if multiple patterns match.

        Examples
        --------
            >>> from insideLLMs.evaluation.reasoning import ReasoningExtractor, ReasoningAnalyzer
            >>> analyzer = ReasoningAnalyzer()
            >>> chain = ReasoningExtractor().extract(
            ...     "Everyone believes this. All people agree it's always true."
            ... )
            >>> fallacies = analyzer._identify_fallacies(chain)
            >>> print(fallacies)
            ['hasty_generalization']
        """
        fallacies = []
        full_text = " ".join(s.content.lower() for s in chain.steps)

        for fallacy_name, patterns in self.FALLACY_PATTERNS.items():
            for pattern in patterns:
                if pattern in full_text:
                    fallacies.append(fallacy_name)
                    break

        return list(set(fallacies))

    def _identify_missing_steps(self, chain: ReasoningChain) -> list[str]:
        """
        Identify potentially missing steps in the reasoning chain.

        Checks for missing premises, conclusions, and gaps between
        consecutive steps with low coherence.

        Parameters
        ----------
        chain : ReasoningChain
            The chain to analyze for missing components.

        Returns
        -------
        list[str]
            Descriptions of missing or problematic elements:
            - "No clear premise or starting point" if no premises
            - "No explicit conclusion" if no conclusion
            - "Gap between step N and M" for low coherence pairs

        Examples
        --------
            >>> from insideLLMs.evaluation.reasoning import ReasoningExtractor, ReasoningAnalyzer
            >>> analyzer = ReasoningAnalyzer()
            >>> chain = ReasoningExtractor().extract("Hence A. Thus B. Therefore C.")
            >>> missing = analyzer._identify_missing_steps(chain)
            >>> print(missing)
            ['No clear premise or starting point']
        """
        missing = []

        step_types = [s.step_type for s in chain.steps]

        # Missing premise
        if ReasoningStepType.PREMISE not in step_types and len(chain.steps) > 1:
            missing.append("No clear premise or starting point")

        # Missing conclusion
        if ReasoningStepType.CONCLUSION not in step_types and chain.conclusion is None:
            missing.append("No explicit conclusion")

        # Jumps in reasoning
        if len(chain.steps) >= 2:
            for i in range(len(chain.steps) - 1):
                coherence = self._step_coherence(chain.steps[i], chain.steps[i + 1])
                if coherence < 0.1:
                    missing.append(f"Gap between step {i + 1} and {i + 2}")

        return missing

    def _step_coherence(self, step1: ReasoningStep, step2: ReasoningStep) -> float:
        """
        Calculate coherence between two specific steps.

        Uses Jaccard similarity on word sets to measure topical overlap.

        Parameters
        ----------
        step1 : ReasoningStep
            The first step.
        step2 : ReasoningStep
            The second step.

        Returns
        -------
        float
            Jaccard similarity from 0.0 to 1.0, where 1.0 means
            identical word sets and 0.0 means no overlap.

        Examples
        --------
            >>> from insideLLMs.evaluation.reasoning import ReasoningStep, ReasoningAnalyzer
            >>> analyzer = ReasoningAnalyzer()
            >>> s1 = ReasoningStep("Dogs are mammals", 1)
            >>> s2 = ReasoningStep("Mammals are warm-blooded", 2)
            >>> print(f"Coherence: {analyzer._step_coherence(s1, s2):.2f}")
            Coherence: 0.20
        """
        words1 = set(step1.content.lower().split())
        words2 = set(step2.content.lower().split())

        overlap = len(words1 & words2)
        union = len(words1 | words2)

        return overlap / union if union > 0 else 0.0


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

        >>> from insideLLMs.evaluation.reasoning import CoTEvaluator
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
            >>> from insideLLMs.evaluation.reasoning import CoTEvaluator
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

            >>> from insideLLMs.evaluation.reasoning import CoTEvaluator
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

            >>> from insideLLMs.evaluation.reasoning import CoTEvaluator
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

            >>> from insideLLMs.evaluation.reasoning import CoTEvaluator
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


class CoTPromptGenerator:
    """
    Generates Chain-of-Thought prompts in various styles.

    Provides templates and utilities for creating prompts that encourage
    step-by-step reasoning in language model responses.

    Attributes
    ----------
    TEMPLATES : dict[str, str]
        Built-in prompt templates with {question} placeholder:
        - standard: Simple "Let's think step by step" format
        - structured: Numbered step format
        - detailed: Comprehensive breakdown format
        - math: Mathematical problem format with "Given:"
        - logical: Formal logic format with premises

    Examples
    --------
    Generating prompts with different styles:

        >>> from insideLLMs.evaluation.reasoning import CoTPromptGenerator
        >>> generator = CoTPromptGenerator()
        >>> prompt = generator.generate("What is 5+7?", style="standard")
        >>> print(prompt)
        Let's think step by step.
        <BLANKLINE>
        What is 5+7?

    Using math style:

        >>> prompt = generator.generate("Calculate the area of a 5x3 rectangle", style="math")
        >>> print(prompt[:50])
        Let's solve this mathematical problem step by ste

    Generating variations:

        >>> variations = generator.generate_variations("Solve: 2x + 5 = 11", num_variations=3)
        >>> print(f"Generated {len(variations)} variations")
        Generated 3 variations

    Adding custom templates:

        >>> generator.add_template("science", "Scientific analysis:\\n\\n{question}\\n\\nHypothesis:")
        >>> prompt = generator.generate("Why is the sky blue?", style="science")
        >>> print(prompt[:30])
        Scientific analysis:

    See Also
    --------
    generate_cot_prompt : Convenience function
    CoTEvaluator : For evaluating CoT responses
    """

    TEMPLATES = {
        "standard": "Let's think step by step.\n\n{question}",
        "structured": "Please solve this problem step by step:\n\n{question}\n\nStep 1:",
        "detailed": "I need to solve this problem. Let me break it down:\n\n{question}\n\nFirst, I'll identify what we know:\n",
        "math": "Let's solve this mathematical problem step by step:\n\n{question}\n\nGiven:\n",
        "logical": "Let me reason through this logically:\n\n{question}\n\nPremise 1:",
    }

    def generate(
        self,
        question: str,
        style: str = "standard",
        custom_template: Optional[str] = None,
    ) -> str:
        """
        Generate a Chain-of-Thought prompt.

        Creates a prompt that encourages step-by-step reasoning by wrapping
        the question in an appropriate template.

        Parameters
        ----------
        question : str
            The question or problem to wrap in a CoT prompt.
        style : str
            The template style to use. Options: "standard", "structured",
            "detailed", "math", "logical". Defaults to "standard".
        custom_template : Optional[str]
            A custom template string with {question} placeholder.
            If provided, overrides the style parameter.

        Returns
        -------
        str
            The formatted CoT prompt.

        Examples
        --------
        Standard style:

            >>> from insideLLMs.evaluation.reasoning import CoTPromptGenerator
            >>> generator = CoTPromptGenerator()
            >>> prompt = generator.generate("What is 2+2?")
            >>> print(prompt)
            Let's think step by step.
            <BLANKLINE>
            What is 2+2?

        Structured style:

            >>> prompt = generator.generate("Solve for x: 3x = 9", style="structured")
            >>> "Step 1:" in prompt
            True

        Custom template:

            >>> prompt = generator.generate(
            ...     "Why do birds fly?",
            ...     custom_template="Analyze this: {question}\\n\\nReason:"
            ... )
            >>> print(prompt)
            Analyze this: Why do birds fly?
            <BLANKLINE>
            Reason:

        Unknown style falls back to standard:

            >>> prompt = generator.generate("Question", style="unknown")
            >>> "step by step" in prompt
            True
        """
        template = custom_template or self.TEMPLATES.get(style, self.TEMPLATES["standard"])

        return template.format(question=question)

    def generate_variations(
        self,
        question: str,
        num_variations: int = 3,
    ) -> list[str]:
        """
        Generate multiple CoT prompt variations for a question.

        Creates prompts using different template styles to explore
        how different prompting approaches affect model responses.

        Parameters
        ----------
        question : str
            The question to generate variations for.
        num_variations : int
            Maximum number of variations to generate. Capped at the
            number of available templates. Defaults to 3.

        Returns
        -------
        list[str]
            List of generated prompts, each using a different style.

        Examples
        --------
        Generate default variations:

            >>> from insideLLMs.evaluation.reasoning import CoTPromptGenerator
            >>> generator = CoTPromptGenerator()
            >>> variations = generator.generate_variations("What is 10/2?")
            >>> len(variations)
            3

        Generate all variations:

            >>> variations = generator.generate_variations("Problem", num_variations=10)
            >>> len(variations)  # Capped at number of templates
            5

        Check variation uniqueness:

            >>> variations = generator.generate_variations("Test question")
            >>> len(set(variations)) == len(variations)
            True
        """
        variations = []
        styles = list(self.TEMPLATES.keys())

        for i in range(min(num_variations, len(styles))):
            prompt = self.generate(question, style=styles[i])
            variations.append(prompt)

        return variations

    def add_template(self, name: str, template: str) -> None:
        """
        Add a custom template to the generator.

        Registers a new template that can be used with the generate method.
        Template must contain {question} placeholder.

        Parameters
        ----------
        name : str
            Name for the template, used as the style parameter.
        template : str
            Template string with {question} placeholder.

        Examples
        --------
        Adding a simple template:

            >>> from insideLLMs.evaluation.reasoning import CoTPromptGenerator
            >>> generator = CoTPromptGenerator()
            >>> generator.add_template("brief", "Think: {question}")
            >>> prompt = generator.generate("2+2?", style="brief")
            >>> print(prompt)
            Think: 2+2?

        Adding a multi-line template:

            >>> template = '''Analyze this problem carefully:
            ...
            ... {question}
            ...
            ... Step-by-step solution:'''
            >>> generator.add_template("careful", template)
            >>> "carefully" in generator.generate("X", style="careful")
            True

        Overwriting existing template:

            >>> generator.add_template("standard", "New standard: {question}")
            >>> "New standard" in generator.generate("Q", style="standard")
            True
        """
        self.TEMPLATES[name] = template


# Convenience functions


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

        >>> from insideLLMs.evaluation.reasoning import extract_reasoning
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

        >>> from insideLLMs.evaluation.reasoning import extract_reasoning, analyze_reasoning
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

        >>> from insideLLMs.evaluation.reasoning import evaluate_cot
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

        >>> from insideLLMs.evaluation.reasoning import generate_cot_prompt
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

        >>> from insideLLMs.evaluation.reasoning import assess_reasoning_quality, ReasoningQuality
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

        >>> from insideLLMs.evaluation.reasoning import assess_reasoning_quality, ReasoningQuality
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


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import StepType. The canonical name is ReasoningStepType.
StepType = ReasoningStepType
