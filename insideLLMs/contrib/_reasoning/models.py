"""Reasoning domain enums and result data classes."""

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

        >>> from insideLLMs.contrib.reasoning import extract_reasoning, ReasoningType
        >>> math_text = "Given x = 10. Calculate x * 2 = 20. Therefore the answer is 20."
        >>> chain = extract_reasoning(math_text)
        >>> chain.reasoning_type == ReasoningType.MATHEMATICAL
        True

    Using reasoning type for analysis filtering:

        >>> from insideLLMs.contrib.reasoning import ReasoningType
        >>> # Filter evaluations by reasoning type
        >>> math_evals = [e for e in evaluations
        ...              if e.chain.reasoning_type == ReasoningType.MATHEMATICAL]

    Checking for specific reasoning patterns:

        >>> from insideLLMs.contrib.reasoning import extract_reasoning, ReasoningType
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

        >>> from insideLLMs.contrib.reasoning import extract_reasoning, ReasoningStepType
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

        >>> from insideLLMs.contrib.reasoning import extract_reasoning, ReasoningStepType
        >>> math_text = "First, calculate 5 + 3 = 8. Then multiply by 2 = 16."
        >>> chain = extract_reasoning(math_text)
        >>> has_calc = any(s.step_type == ReasoningStepType.CALCULATION
        ...                for s in chain.steps)
        >>> print(f"Contains calculations: {has_calc}")
        Contains calculations: True

    Analyzing reasoning structure:

        >>> from insideLLMs.contrib.reasoning import extract_reasoning, ReasoningStepType
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

        >>> from insideLLMs.contrib.reasoning import assess_reasoning_quality, ReasoningQuality
        >>> good_reasoning = '''
        ... Given: All cats are mammals. Given: Fluffy is a cat.
        ... Therefore, Fluffy must be a mammal.
        ... '''
        >>> quality = assess_reasoning_quality(good_reasoning)
        >>> print(f"Quality: {quality.value}")
        Quality: good

    Filtering by quality threshold:

        >>> from insideLLMs.contrib.reasoning import ReasoningQuality
        >>> # Define acceptable quality levels
        >>> acceptable = {ReasoningQuality.EXCELLENT, ReasoningQuality.GOOD}
        >>> # Filter analysis results
        >>> good_analyses = [a for a in analyses if a.overall_quality in acceptable]

    Quality-based decision making:

        >>> from insideLLMs.contrib.reasoning import (
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

        >>> from insideLLMs.contrib.reasoning import assess_reasoning_quality
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

        >>> from insideLLMs.contrib.reasoning import ReasoningStep, ReasoningStepType
        >>> step = ReasoningStep(
        ...     content="All mammals are warm-blooded",
        ...     step_number=1,
        ...     step_type=ReasoningStepType.PREMISE,
        ...     confidence=0.95
        ... )
        >>> print(f"Step {step.step_number}: {step.content}")
        Step 1: All mammals are warm-blooded

    Creating a step with dependencies:

        >>> from insideLLMs.contrib.reasoning import ReasoningStep, ReasoningStepType
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

        >>> from insideLLMs.contrib.reasoning import ReasoningStep, ReasoningStepType
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

        >>> from insideLLMs.contrib.reasoning import ReasoningStep, ReasoningStepType
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

            >>> from insideLLMs.contrib.reasoning import ReasoningStep, ReasoningStepType
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
            >>> from insideLLMs.contrib.reasoning import ReasoningStep
            >>> step = ReasoningStep(content="Step content", step_number=1)
            >>> json_str = json.dumps(step.to_dict())
            >>> print(type(json_str))
            <class 'str'>

        Reconstructing from dictionary:

            >>> from insideLLMs.contrib.reasoning import ReasoningStep, ReasoningStepType
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

        >>> from insideLLMs.contrib.reasoning import extract_reasoning
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

        >>> from insideLLMs.contrib.reasoning import extract_reasoning
        >>> chain = extract_reasoning("Step 1: A. Step 2: B. Step 3: Therefore C.")
        >>> step = chain.get_step(2)
        >>> if step:
        ...     print(f"Step 2: {step.content}")
        Step 2: B

    Getting specific step types:

        >>> from insideLLMs.contrib.reasoning import extract_reasoning
        >>> text = "Given: x=5. Suppose y=3. Thus x+y=8. Therefore the answer is 8."
        >>> chain = extract_reasoning(text)
        >>> premises = chain.get_premises()
        >>> print(f"Found {len(premises)} premise(s)")
        Found 2 premise(s)
        >>> inferences = chain.get_inferences()
        >>> print(f"Found {len(inferences)} inference(s)")
        Found 1 inference(s)

    Checking chain completeness:

        >>> from insideLLMs.contrib.reasoning import extract_reasoning
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

            >>> from insideLLMs.contrib.reasoning import extract_reasoning
            >>> chain = extract_reasoning("Step 1: First. Step 2: Second. Step 3: Third.")
            >>> step = chain.get_step(2)
            >>> if step:
            ...     print(step.content)
            Second

        Handling missing steps:

            >>> from insideLLMs.contrib.reasoning import extract_reasoning
            >>> chain = extract_reasoning("Step 1: Only step.")
            >>> step = chain.get_step(5)  # Doesn't exist
            >>> print(step is None)
            True

        Iterating with step numbers:

            >>> from insideLLMs.contrib.reasoning import extract_reasoning
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

            >>> from insideLLMs.contrib.reasoning import extract_reasoning
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

            >>> from insideLLMs.contrib.reasoning import extract_reasoning
            >>> chain = extract_reasoning("Therefore X. Thus Y. Hence Z.")
            >>> premises = chain.get_premises()
            >>> if not premises:
            ...     print("Warning: No explicit premises found")
            Warning: No explicit premises found

        Counting premise types:

            >>> from insideLLMs.contrib.reasoning import extract_reasoning
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

            >>> from insideLLMs.contrib.reasoning import extract_reasoning
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

            >>> from insideLLMs.contrib.reasoning import extract_reasoning
            >>> chain = extract_reasoning("Given A. Therefore B. Thus C. Hence D.")
            >>> inferences = chain.get_inferences()
            >>> avg_confidence = sum(i.confidence for i in inferences) / len(inferences)
            >>> print(f"Average inference confidence: {avg_confidence:.2f}")
            Average inference confidence: 0.60

        Checking reasoning structure:

            >>> from insideLLMs.contrib.reasoning import extract_reasoning
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

            >>> from insideLLMs.contrib.reasoning import extract_reasoning
            >>> chain = extract_reasoning("Step 1: A. Step 2: B. Therefore C.")
            >>> d = chain.to_dict()
            >>> print(f"Steps: {d['num_steps']}, Valid: {d['is_valid']}")
            Steps: 3, Valid: True

        JSON export:

            >>> import json
            >>> from insideLLMs.contrib.reasoning import extract_reasoning
            >>> chain = extract_reasoning("Given x=5. Therefore x+1=6.")
            >>> json_str = json.dumps(chain.to_dict(), indent=2)
            >>> print(type(json_str))
            <class 'str'>

        Accessing serialized steps:

            >>> from insideLLMs.contrib.reasoning import extract_reasoning
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

        >>> from insideLLMs.contrib.reasoning import extract_reasoning, analyze_reasoning
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

        >>> from insideLLMs.contrib.reasoning import extract_reasoning, analyze_reasoning
        >>> text = "All politicians are corrupt. Everyone knows this is true."
        >>> chain = extract_reasoning(text)
        >>> analysis = analyze_reasoning(chain)
        >>> if analysis.identified_fallacies:
        ...     print(f"Fallacies found: {analysis.identified_fallacies}")
        Fallacies found: ['hasty_generalization']

    Examining step quality:

        >>> from insideLLMs.contrib.reasoning import extract_reasoning, analyze_reasoning
        >>> text = "Step 1: A because B. Step 2: Therefore C. Step 3: Thus D."
        >>> chain = extract_reasoning(text)
        >>> analysis = analyze_reasoning(chain)
        >>> avg_quality = sum(analysis.step_quality_scores) / len(analysis.step_quality_scores)
        >>> print(f"Average step quality: {avg_quality:.2f}")
        Average step quality: 0.70

    Identifying missing components:

        >>> from insideLLMs.contrib.reasoning import extract_reasoning, analyze_reasoning
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

            >>> from insideLLMs.contrib.reasoning import extract_reasoning, analyze_reasoning
            >>> chain = extract_reasoning("Given A. Therefore B.")
            >>> analysis = analyze_reasoning(chain)
            >>> d = analysis.to_dict()
            >>> print(f"Quality: {d['overall_quality']}")
            Quality: adequate

        JSON export:

            >>> import json
            >>> from insideLLMs.contrib.reasoning import extract_reasoning, analyze_reasoning
            >>> chain = extract_reasoning("Step 1: A. Step 2: B.")
            >>> analysis = analyze_reasoning(chain)
            >>> json_str = json.dumps(analysis.to_dict())
            >>> print(type(json_str))
            <class 'str'>

        Accessing analysis details:

            >>> from insideLLMs.contrib.reasoning import extract_reasoning, analyze_reasoning
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

        >>> from insideLLMs.contrib.reasoning import evaluate_cot
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

        >>> from insideLLMs.contrib.reasoning import evaluate_cot
        >>> response = "The answer is 42."  # Minimal reasoning
        >>> eval = evaluate_cot("Calculate 6*7", response, "42")
        >>> for imp in eval.improvements:
        ...     print(f"Suggestion: {imp}")
        Suggestion: Strengthen logical connections between steps
        Suggestion: Add missing steps to complete the reasoning chain

    Comparing evaluations:

        >>> from insideLLMs.contrib.reasoning import evaluate_cot
        >>> eval1 = evaluate_cot("2+2?", "It's 4.", "4")
        >>> eval2 = evaluate_cot("2+2?", "Step 1: Add 2+2. Step 2: Result is 4.", "4")
        >>> print(f"Eval1 steps: {len(eval1.chain.steps)}")
        Eval1 steps: 1
        >>> print(f"Eval2 steps: {len(eval2.chain.steps)}")
        Eval2 steps: 2

    Analyzing explanation quality:

        >>> from insideLLMs.contrib.reasoning import evaluate_cot
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

            >>> from insideLLMs.contrib.reasoning import evaluate_cot
            >>> eval = evaluate_cot("What is 2+2?", "2+2=4, so 4.", "4")
            >>> d = eval.to_dict()
            >>> print(f"Correct: {d['answer_correct']}")
            Correct: True

        JSON export for reporting:

            >>> import json
            >>> from insideLLMs.contrib.reasoning import evaluate_cot
            >>> eval = evaluate_cot("Question", "Answer with reasoning")
            >>> json_str = json.dumps(eval.to_dict())
            >>> print(type(json_str))
            <class 'str'>

        Aggregating evaluation metrics:

            >>> from insideLLMs.contrib.reasoning import evaluate_cot
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

        >>> from insideLLMs.contrib.reasoning import CoTEvaluator
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

        >>> from insideLLMs.contrib.reasoning import CoTEvaluator
        >>> evaluator = CoTEvaluator()
        >>> # Assume we have a report
        >>> # for rtype, proportion in report.reasoning_type_breakdown.items():
        >>> #     print(f"{rtype}: {proportion:.1%}")
        >>> # Output: mathematical: 60.0%, deductive: 40.0%

    Checking quality distribution:

        >>> from insideLLMs.contrib.reasoning import CoTEvaluator
        >>> evaluator = CoTEvaluator()
        >>> # Assume we have evaluations
        >>> # report = evaluator.generate_report(evals)
        >>> # for quality, count in report.quality_distribution.items():
        >>> #     print(f"{quality}: {count} evaluations")

    Using recommendations:

        >>> from insideLLMs.contrib.reasoning import CoTEvaluator
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

            >>> from insideLLMs.contrib.reasoning import CoTEvaluator
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
            >>> from insideLLMs.contrib.reasoning import CoTEvaluator
            >>> evaluator = CoTEvaluator()
            >>> evals = evaluator.evaluate_batch(["Q"], ["A"])
            >>> report = evaluator.generate_report(evals)
            >>> json_str = json.dumps(report.to_dict())
            >>> print(type(json_str))
            <class 'str'>

        Accessing breakdown data:

            >>> from insideLLMs.contrib.reasoning import CoTEvaluator
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
