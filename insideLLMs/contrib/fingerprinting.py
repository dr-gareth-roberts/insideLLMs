"""Model capability fingerprinting for LLM evaluation.

This module provides a comprehensive framework for systematically assessing,
categorizing, and comparing the capabilities of large language models (LLMs).
It enables researchers and developers to create detailed "fingerprints" of
model behavior across multiple skill domains, identify strengths and weaknesses,
and perform comparative analysis between different models.

Overview
--------
The fingerprinting system is organized around several core concepts:

1. **Capability Categories**: High-level domains of model ability (reasoning,
   language, knowledge, creativity, coding, math, analysis, instruction-following)

2. **Skills**: Specific competencies within each category (e.g., logical_deduction
   within reasoning, code_generation within coding)

3. **Capability Probes**: Test generators that produce prompts and expected
   answers to assess specific skills

4. **Fingerprints**: Complete capability profiles combining scores, skill
   assessments, and detected limitations

Key Components
--------------
CapabilityCategory : Enum
    High-level categories of model capabilities
SkillType : Enum
    Specific skills within each capability category
CapabilityLevel : Enum
    Proficiency levels from NONE to EXPERT
LimitationType : Enum
    Types of model limitations (hallucination, repetition, etc.)
CapabilityScore : dataclass
    Score for a specific capability with evidence
SkillProfile : dataclass
    Profile of skills in a category with strengths/weaknesses
ModelFingerprint : dataclass
    Complete fingerprint of a model's capabilities
FingerprintComparison : dataclass
    Comparison between two model fingerprints
CapabilityProbe : class
    Base class for creating capability test probes
SkillAssessor : class
    Assess individual skills from model responses
LimitationDetector : class
    Detect model limitations from response patterns
CapabilityProfiler : class
    Build skill profiles for capability categories
FingerprintGenerator : class
    Generate complete model fingerprints
FingerprintComparator : class
    Compare model fingerprints to identify differences

Examples
--------
**Quick capability assessment from responses:**

>>> from insideLLMs.contrib.fingerprinting import quick_capability_assessment
>>> responses = {
...     "logical_deduction": (
...         "No, we cannot conclude that. The syllogism is invalid.",
...         False
...     ),
...     "arithmetic": (
...         "The answer is 42.",
...         "42"
...     ),
... }
>>> result = quick_capability_assessment(responses)
>>> print(f"Overall score: {result['overall_score']:.2f}")
Overall score: 0.85

**Creating a full model fingerprint:**

>>> from insideLLMs.contrib.fingerprinting import (
...     create_fingerprint,
...     SkillType,
...     CapabilityCategory
... )
>>> skill_results = {
...     SkillType.LOGICAL_DEDUCTION: [
...         ("No, this is invalid reasoning", False),
...         ("The conclusion does not follow", False),
...     ],
...     SkillType.CODE_GENERATION: [
...         ("def add(a, b): return a + b", "def add(a, b): return a + b"),
...     ],
... }
>>> responses = ["This is a sample response for limitation detection"]
>>> fingerprint = create_fingerprint(
...     model_id="gpt-4-turbo",
...     skill_results=skill_results,
...     responses=responses,
...     version="1.0"
... )
>>> print(f"Model: {fingerprint.model_id}")
Model: gpt-4-turbo
>>> print(f"Overall score: {fingerprint.overall_capability_score:.2f}")
Overall score: 0.65

**Comparing two models:**

>>> from insideLLMs.contrib.fingerprinting import compare_fingerprints
>>> # Assuming fingerprint_a and fingerprint_b are already created
>>> comparison = compare_fingerprints(fingerprint_a, fingerprint_b)
>>> print(f"Similarity: {comparison.similarity_score:.2%}")
Similarity: 78.50%
>>> if comparison.winner:
...     print(f"Better model: {comparison.winner}")
Better model: gpt-4-turbo

**Using the SkillAssessor directly:**

>>> from insideLLMs.contrib.fingerprinting import SkillAssessor, SkillType
>>> assessor = SkillAssessor()
>>> responses = [
...     ("The capital of France is Paris.", "Paris"),
...     ("Mount Everest is the tallest mountain.", "Everest"),
... ]
>>> score = assessor.assess_skill(SkillType.FACTUAL_RECALL, responses)
>>> print(f"Skill: {score.skill.value}, Level: {score.level.value}")
Skill: factual_recall, Level: advanced

**Detecting model limitations:**

>>> from insideLLMs.contrib.fingerprinting import detect_limitations
>>> responses = [
...     "I believe the answer might be approximately 42, possibly around that.",
...     "I think it could roughly be correct, maybe approximately so.",
...     "The result is exactly 42."
... ]
>>> limitations = detect_limitations(responses)
>>> for lim in limitations:
...     print(f"Type: {lim['type']}, Severity: {lim['severity']:.2f}")
Type: hallucination_prone, Severity: 0.67

Notes
-----
- Scores are normalized to the range [0, 1] where higher is better
- Confidence values increase with more test samples (up to 5 samples for full confidence)
- The fingerprint hash provides a unique identifier based on capability profile
- Limitation detection uses heuristic patterns and may require tuning for specific use cases

See Also
--------
insideLLMs.probing : Low-level probing utilities for model testing
insideLLMs.evaluation : Model evaluation framework
insideLLMs.analysis : Analysis tools for probe results
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class CapabilityCategory(Enum):
    """High-level categories of model capabilities.

    This enum defines the major domains used to organize and assess
    language model capabilities. Each category contains multiple
    specific skills (see SkillType) that can be individually tested.

    Attributes
    ----------
    REASONING : str
        Logical and analytical reasoning abilities including deduction,
        causal reasoning, and counterfactual thinking.
    LANGUAGE : str
        Language understanding and generation skills including grammar,
        vocabulary, style adaptation, translation, and summarization.
    KNOWLEDGE : str
        Factual knowledge and recall including world facts, temporal
        knowledge, domain expertise, and common sense.
    CREATIVITY : str
        Creative generation abilities including creative writing,
        brainstorming, humor, and storytelling.
    CODING : str
        Programming capabilities including code generation, understanding,
        debugging, and translation between languages.
    MATH : str
        Mathematical abilities including arithmetic, algebra, word
        problems, and symbolic manipulation.
    ANALYSIS : str
        Analytical skills including data analysis, text analysis,
        pattern recognition, and critical evaluation.
    INSTRUCTION_FOLLOWING : str
        Ability to follow instructions including task decomposition,
        format adherence, constraint handling, and multi-step execution.

    Examples
    --------
    >>> from insideLLMs.contrib.fingerprinting import CapabilityCategory
    >>> cat = CapabilityCategory.REASONING
    >>> print(cat.value)
    reasoning

    >>> # Iterate through all categories
    >>> for category in CapabilityCategory:
    ...     print(f"{category.name}: {category.value}")
    REASONING: reasoning
    LANGUAGE: language
    KNOWLEDGE: knowledge
    ...

    >>> # Use in fingerprint category scores
    >>> category_scores = {
    ...     CapabilityCategory.REASONING: 0.85,
    ...     CapabilityCategory.CODING: 0.92,
    ... }

    See Also
    --------
    SkillType : Specific skills within each category
    CapabilityLevel : Proficiency levels for capabilities
    """

    REASONING = "reasoning"
    LANGUAGE = "language"
    KNOWLEDGE = "knowledge"
    CREATIVITY = "creativity"
    CODING = "coding"
    MATH = "math"
    ANALYSIS = "analysis"
    INSTRUCTION_FOLLOWING = "instruction_following"


class CapabilityLevel(Enum):
    """Proficiency level for a capability.

    This enum represents the proficiency tiers used to categorize
    model performance on specific skills. Levels are determined
    by converting numeric scores (0-1) using the _score_to_level
    helper function.

    Attributes
    ----------
    NONE : str
        No demonstrated capability (score < 0.25). The model fails
        to perform the skill or produces incorrect results.
    BASIC : str
        Basic capability (0.25 <= score < 0.5). The model shows
        rudimentary understanding but makes frequent errors.
    INTERMEDIATE : str
        Intermediate capability (0.5 <= score < 0.75). The model
        performs adequately with occasional errors.
    ADVANCED : str
        Advanced capability (0.75 <= score < 0.9). The model performs
        well with only minor errors or limitations.
    EXPERT : str
        Expert capability (score >= 0.9). The model demonstrates
        near-perfect performance on the skill.

    Examples
    --------
    >>> from insideLLMs.contrib.fingerprinting import CapabilityLevel, _score_to_level
    >>> level = _score_to_level(0.82)
    >>> print(level)
    CapabilityLevel.ADVANCED

    >>> # Check if a level meets a threshold
    >>> level = CapabilityLevel.ADVANCED
    >>> meets_standard = level in {CapabilityLevel.ADVANCED, CapabilityLevel.EXPERT}
    >>> print(meets_standard)
    True

    >>> # Use in capability scores
    >>> from insideLLMs.contrib.fingerprinting import CapabilityScore, SkillType, CapabilityCategory
    >>> score = CapabilityScore(
    ...     category=CapabilityCategory.REASONING,
    ...     skill=SkillType.LOGICAL_DEDUCTION,
    ...     score=0.85,
    ...     level=CapabilityLevel.ADVANCED,
    ...     confidence=0.9,
    ...     evidence=["Correct reasoning on syllogism"]
    ... )

    See Also
    --------
    _score_to_level : Convert numeric score to CapabilityLevel
    CapabilityScore : Score dataclass that uses CapabilityLevel
    """

    NONE = "none"
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class SkillType(Enum):
    """Specific skills within capability categories.

    This enum defines granular skills that can be tested and assessed
    within each CapabilityCategory. Skills are organized by their
    parent category and can be mapped using SkillAssessor._get_category().

    Attributes
    ----------
    LOGICAL_DEDUCTION : str
        Reasoning: Drawing valid conclusions from premises.
    CAUSAL_REASONING : str
        Reasoning: Understanding cause-effect relationships.
    ANALOGICAL_REASONING : str
        Reasoning: Drawing parallels between different domains.
    COUNTERFACTUAL_REASONING : str
        Reasoning: Reasoning about hypothetical scenarios.
    GRAMMAR : str
        Language: Understanding and applying grammatical rules.
    VOCABULARY : str
        Language: Knowledge and appropriate use of words.
    STYLE_ADAPTATION : str
        Language: Adjusting writing style for different contexts.
    TRANSLATION : str
        Language: Converting text between languages.
    SUMMARIZATION : str
        Language: Condensing content while preserving meaning.
    FACTUAL_RECALL : str
        Knowledge: Retrieving factual information.
    TEMPORAL_KNOWLEDGE : str
        Knowledge: Understanding temporal relationships and history.
    DOMAIN_EXPERTISE : str
        Knowledge: Deep knowledge in specific domains.
    COMMON_SENSE : str
        Knowledge: Everyday reasoning and world knowledge.
    CREATIVE_WRITING : str
        Creativity: Generating original written content.
    BRAINSTORMING : str
        Creativity: Generating diverse ideas and solutions.
    HUMOR : str
        Creativity: Understanding and generating humor.
    STORYTELLING : str
        Creativity: Crafting compelling narratives.
    CODE_GENERATION : str
        Coding: Writing code from specifications.
    CODE_UNDERSTANDING : str
        Coding: Comprehending existing code.
    DEBUGGING : str
        Coding: Identifying and fixing code errors.
    CODE_TRANSLATION : str
        Coding: Converting code between languages.
    ARITHMETIC : str
        Math: Basic mathematical operations.
    ALGEBRA : str
        Math: Symbolic mathematical reasoning.
    WORD_PROBLEMS : str
        Math: Translating text to mathematical solutions.
    SYMBOLIC_MANIPULATION : str
        Math: Manipulating mathematical expressions.
    DATA_ANALYSIS : str
        Analysis: Interpreting and analyzing data.
    TEXT_ANALYSIS : str
        Analysis: Extracting insights from text.
    PATTERN_RECOGNITION : str
        Analysis: Identifying patterns in information.
    CRITICAL_EVALUATION : str
        Analysis: Assessing arguments and claims.
    TASK_DECOMPOSITION : str
        Instruction Following: Breaking down complex tasks.
    FORMAT_ADHERENCE : str
        Instruction Following: Following output format requirements.
    CONSTRAINT_HANDLING : str
        Instruction Following: Respecting specified constraints.
    MULTI_STEP_EXECUTION : str
        Instruction Following: Completing multi-step procedures.

    Examples
    --------
    >>> from insideLLMs.contrib.fingerprinting import SkillType, SkillAssessor
    >>> skill = SkillType.LOGICAL_DEDUCTION
    >>> print(skill.value)
    logical_deduction

    >>> # Get the category for a skill
    >>> category = SkillAssessor._get_category(SkillType.CODE_GENERATION)
    >>> print(category.value)
    coding

    >>> # Use skills in test results
    >>> skill_results = {
    ...     SkillType.ARITHMETIC: [("42", "42"), ("100", "100")],
    ...     SkillType.ALGEBRA: [("x = 5", "x = 5")],
    ... }

    >>> # Check if skill is in a set
    >>> reasoning_skills = {
    ...     SkillType.LOGICAL_DEDUCTION,
    ...     SkillType.CAUSAL_REASONING,
    ... }
    >>> SkillType.LOGICAL_DEDUCTION in reasoning_skills
    True

    See Also
    --------
    CapabilityCategory : Parent categories for skills
    CapabilityProbe : Base class for probes that test skills
    SkillAssessor : Assess skill performance
    """

    # Reasoning
    LOGICAL_DEDUCTION = "logical_deduction"
    CAUSAL_REASONING = "causal_reasoning"
    ANALOGICAL_REASONING = "analogical_reasoning"
    COUNTERFACTUAL_REASONING = "counterfactual_reasoning"

    # Language
    GRAMMAR = "grammar"
    VOCABULARY = "vocabulary"
    STYLE_ADAPTATION = "style_adaptation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"

    # Knowledge
    FACTUAL_RECALL = "factual_recall"
    TEMPORAL_KNOWLEDGE = "temporal_knowledge"
    DOMAIN_EXPERTISE = "domain_expertise"
    COMMON_SENSE = "common_sense"

    # Creativity
    CREATIVE_WRITING = "creative_writing"
    BRAINSTORMING = "brainstorming"
    HUMOR = "humor"
    STORYTELLING = "storytelling"

    # Coding
    CODE_GENERATION = "code_generation"
    CODE_UNDERSTANDING = "code_understanding"
    DEBUGGING = "debugging"
    CODE_TRANSLATION = "code_translation"

    # Math
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    WORD_PROBLEMS = "word_problems"
    SYMBOLIC_MANIPULATION = "symbolic_manipulation"

    # Analysis
    DATA_ANALYSIS = "data_analysis"
    TEXT_ANALYSIS = "text_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    CRITICAL_EVALUATION = "critical_evaluation"

    # Instruction following
    TASK_DECOMPOSITION = "task_decomposition"
    FORMAT_ADHERENCE = "format_adherence"
    CONSTRAINT_HANDLING = "constraint_handling"
    MULTI_STEP_EXECUTION = "multi_step_execution"


# =============================================================================
# Helper Functions
# =============================================================================


def _score_to_level(score: float) -> CapabilityLevel:
    """Convert a numeric score (0-1) to a CapabilityLevel.

    This helper function maps continuous scores to discrete proficiency
    levels using fixed thresholds. It is used throughout the fingerprinting
    system to convert raw assessment scores to human-readable levels.

    Args
    ----
    score : float
        A score between 0.0 and 1.0, where higher values indicate
        better performance. Values outside this range are not clamped
        and may produce unexpected results.

    Returns
    -------
    CapabilityLevel
        The corresponding capability level based on score thresholds:
        - EXPERT: score >= 0.9
        - ADVANCED: 0.75 <= score < 0.9
        - INTERMEDIATE: 0.5 <= score < 0.75
        - BASIC: 0.25 <= score < 0.5
        - NONE: score < 0.25

    Examples
    --------
    >>> from insideLLMs.contrib.fingerprinting import _score_to_level, CapabilityLevel
    >>> _score_to_level(0.95)
    <CapabilityLevel.EXPERT: 'expert'>

    >>> _score_to_level(0.82)
    <CapabilityLevel.ADVANCED: 'advanced'>

    >>> _score_to_level(0.65)
    <CapabilityLevel.INTERMEDIATE: 'intermediate'>

    >>> _score_to_level(0.35)
    <CapabilityLevel.BASIC: 'basic'>

    >>> _score_to_level(0.1)
    <CapabilityLevel.NONE: 'none'>

    >>> # Edge cases at thresholds
    >>> _score_to_level(0.75)
    <CapabilityLevel.ADVANCED: 'advanced'>

    >>> _score_to_level(0.5)
    <CapabilityLevel.INTERMEDIATE: 'intermediate'>

    See Also
    --------
    CapabilityLevel : The enum returned by this function
    SkillAssessor.assess_skill : Uses this function to convert scores
    """
    if score >= 0.9:
        return CapabilityLevel.EXPERT
    elif score >= 0.75:
        return CapabilityLevel.ADVANCED
    elif score >= 0.5:
        return CapabilityLevel.INTERMEDIATE
    elif score >= 0.25:
        return CapabilityLevel.BASIC
    return CapabilityLevel.NONE


# =============================================================================
# Enums (continued)
# =============================================================================


class LimitationType(Enum):
    """Types of model limitations that can be detected.

    This enum defines categories of limitations or failure modes that
    can be automatically detected from model responses. The LimitationDetector
    class uses pattern matching to identify these limitations.

    Attributes
    ----------
    FACTUAL_ERROR_PRONE : str
        Model frequently makes factual errors or states incorrect
        information as fact.
    CONTEXT_LENGTH_LIMITED : str
        Model struggles with long contexts, losing information or
        becoming incoherent.
    CALCULATION_ERRORS : str
        Model makes arithmetic or mathematical calculation errors.
    INSTRUCTION_CONFUSION : str
        Model fails to follow instructions correctly or misinterprets
        requirements.
    REPETITIVE_OUTPUT : str
        Model produces repetitive phrases or patterns in output.
    KNOWLEDGE_CUTOFF : str
        Model lacks knowledge of events or information after its
        training cutoff date.
    HALLUCINATION_PRONE : str
        Model invents information or states uncertain things with
        false confidence. Detected by excessive hedging phrases.
    FORMAT_INCONSISTENT : str
        Model produces inconsistent formatting in structured outputs
        (e.g., mixing list styles).

    Examples
    --------
    >>> from insideLLMs.contrib.fingerprinting import LimitationType
    >>> limitation = LimitationType.HALLUCINATION_PRONE
    >>> print(limitation.value)
    hallucination_prone

    >>> # Use in limitation reports
    >>> from insideLLMs.contrib.fingerprinting import LimitationReport
    >>> report = LimitationReport(
    ...     limitation_type=LimitationType.REPETITIVE_OUTPUT,
    ...     severity=0.7,
    ...     frequency=0.35,
    ...     examples=["The answer is... the answer is... the answer is..."],
    ...     mitigation_suggestions=["Increase temperature", "Add diversity penalty"]
    ... )

    >>> # Check for specific limitations
    >>> detected_types = {LimitationType.HALLUCINATION_PRONE, LimitationType.REPETITIVE_OUTPUT}
    >>> if LimitationType.HALLUCINATION_PRONE in detected_types:
    ...     print("Warning: Model may hallucinate")
    Warning: Model may hallucinate

    See Also
    --------
    LimitationReport : Report structure for detected limitations
    LimitationDetector : Detects limitations from model responses
    """

    FACTUAL_ERROR_PRONE = "factual_error_prone"
    CONTEXT_LENGTH_LIMITED = "context_length_limited"
    CALCULATION_ERRORS = "calculation_errors"
    INSTRUCTION_CONFUSION = "instruction_confusion"
    REPETITIVE_OUTPUT = "repetitive_output"
    KNOWLEDGE_CUTOFF = "knowledge_cutoff"
    HALLUCINATION_PRONE = "hallucination_prone"
    FORMAT_INCONSISTENT = "format_inconsistent"


@dataclass
class CapabilityScore:
    """Score for a specific capability or skill assessment.

    This dataclass represents the result of assessing a model's performance
    on a specific skill. It includes both quantitative scores and qualitative
    evidence supporting the assessment.

    Parameters
    ----------
    category : CapabilityCategory
        The high-level category this skill belongs to (e.g., REASONING, CODING).
    skill : SkillType
        The specific skill being assessed (e.g., LOGICAL_DEDUCTION, CODE_GENERATION).
    score : float
        Numeric score between 0.0 and 1.0, where 1.0 represents perfect performance.
    level : CapabilityLevel
        Categorical proficiency level derived from the score.
    confidence : float
        Confidence in the assessment between 0.0 and 1.0, typically based on
        the number of test samples (more samples = higher confidence).
    evidence : list[str], optional
        List of evidence strings supporting the assessment, such as descriptions
        of correct or incorrect responses. Default is empty list.

    Attributes
    ----------
    category : CapabilityCategory
        The capability category.
    skill : SkillType
        The skill type.
    score : float
        The numeric score.
    level : CapabilityLevel
        The proficiency level.
    confidence : float
        The confidence value.
    evidence : list[str]
        Supporting evidence.

    Examples
    --------
    >>> from insideLLMs.contrib.fingerprinting import (
    ...     CapabilityScore, CapabilityCategory, SkillType, CapabilityLevel
    ... )
    >>> score = CapabilityScore(
    ...     category=CapabilityCategory.REASONING,
    ...     skill=SkillType.LOGICAL_DEDUCTION,
    ...     score=0.85,
    ...     level=CapabilityLevel.ADVANCED,
    ...     confidence=0.8,
    ...     evidence=["Correctly identified invalid syllogism",
    ...               "Proper use of modus ponens"]
    ... )
    >>> print(f"{score.skill.value}: {score.level.value} ({score.score:.0%})")
    logical_deduction: advanced (85%)

    >>> # Convert to dictionary for serialization
    >>> score_dict = score.to_dict()
    >>> print(score_dict['level'])
    advanced

    >>> # Create from assessment results
    >>> from insideLLMs.contrib.fingerprinting import SkillAssessor
    >>> assessor = SkillAssessor()
    >>> responses = [("42", "42"), ("100", "100")]
    >>> score = assessor.assess_skill(SkillType.ARITHMETIC, responses)
    >>> print(f"Level: {score.level.value}, Confidence: {score.confidence:.0%}")
    Level: expert, Confidence: 40%

    See Also
    --------
    SkillAssessor : Creates CapabilityScore objects from response data
    SkillProfile : Aggregates multiple CapabilityScores
    CapabilityLevel : The proficiency level enum
    """

    category: CapabilityCategory
    skill: SkillType
    score: float  # 0-1
    level: CapabilityLevel
    confidence: float
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the CapabilityScore to a dictionary.

        Creates a serializable dictionary representation of the score,
        converting enum values to their string representations.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: 'category', 'skill', 'score', 'level',
            'confidence', 'evidence'. Enum values are converted to strings.

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import (
        ...     CapabilityScore, CapabilityCategory, SkillType, CapabilityLevel
        ... )
        >>> score = CapabilityScore(
        ...     category=CapabilityCategory.MATH,
        ...     skill=SkillType.ARITHMETIC,
        ...     score=0.95,
        ...     level=CapabilityLevel.EXPERT,
        ...     confidence=1.0,
        ...     evidence=["All calculations correct"]
        ... )
        >>> d = score.to_dict()
        >>> print(d['category'])
        math
        >>> print(d['skill'])
        arithmetic
        >>> import json
        >>> json_str = json.dumps(d)  # Now serializable
        """
        return {
            "category": self.category.value,
            "skill": self.skill.value,
            "score": self.score,
            "level": self.level.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class LimitationReport:
    """Report on a detected model limitation.

    This dataclass represents a detected limitation in model behavior,
    including severity, frequency of occurrence, examples, and suggestions
    for mitigation. Reports are generated by the LimitationDetector class.

    Parameters
    ----------
    limitation_type : LimitationType
        The type of limitation detected (e.g., HALLUCINATION_PRONE,
        REPETITIVE_OUTPUT).
    severity : float
        Severity score between 0.0 and 1.0, where higher values indicate
        more severe limitations. Typically calculated as frequency * 2,
        capped at 1.0.
    frequency : float
        How often the limitation was observed in responses, as a ratio
        between 0.0 and 1.0 (e.g., 0.3 means observed in 30% of responses).
    examples : list[str]
        Sample excerpts from responses that demonstrated the limitation.
        Typically limited to 3 examples, truncated to 100 characters.
    mitigation_suggestions : list[str]
        List of suggested strategies to mitigate the limitation, such
        as prompt engineering techniques or parameter adjustments.

    Attributes
    ----------
    limitation_type : LimitationType
        The type of limitation.
    severity : float
        The severity score.
    frequency : float
        The observation frequency.
    examples : list[str]
        Example response excerpts.
    mitigation_suggestions : list[str]
        Mitigation strategies.

    Examples
    --------
    >>> from insideLLMs.contrib.fingerprinting import LimitationReport, LimitationType
    >>> report = LimitationReport(
    ...     limitation_type=LimitationType.HALLUCINATION_PRONE,
    ...     severity=0.7,
    ...     frequency=0.35,
    ...     examples=[
    ...         "I believe the population is approximately...",
    ...         "I think it might be around 42 million..."
    ...     ],
    ...     mitigation_suggestions=[
    ...         "Request citations or sources",
    ...         "Use retrieval augmentation",
    ...         "Ask for confidence levels"
    ...     ]
    ... )
    >>> print(f"Limitation: {report.limitation_type.value}")
    Limitation: hallucination_prone
    >>> print(f"Severity: {report.severity:.0%}")
    Severity: 70%

    >>> # Convert to dictionary for JSON serialization
    >>> report_dict = report.to_dict()
    >>> print(report_dict['type'])
    hallucination_prone

    >>> # Use with LimitationDetector
    >>> from insideLLMs.contrib.fingerprinting import LimitationDetector
    >>> detector = LimitationDetector()
    >>> responses = ["I believe... approximately... possibly..."] * 5
    >>> reports = detector.detect(responses)
    >>> for report in reports:
    ...     print(f"{report.limitation_type.value}: {report.severity:.2f}")

    See Also
    --------
    LimitationType : Enum of limitation types
    LimitationDetector : Generates LimitationReport objects
    ModelFingerprint : Contains list of LimitationReports
    """

    limitation_type: LimitationType
    severity: float  # 0-1
    frequency: float  # How often observed
    examples: list[str]
    mitigation_suggestions: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert the LimitationReport to a dictionary.

        Creates a serializable dictionary representation of the report,
        converting enum values to their string representations.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: 'type', 'severity', 'frequency',
            'examples', 'mitigation_suggestions'. The limitation type
            is converted to its string value.

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import LimitationReport, LimitationType
        >>> report = LimitationReport(
        ...     limitation_type=LimitationType.REPETITIVE_OUTPUT,
        ...     severity=0.5,
        ...     frequency=0.25,
        ...     examples=["repeated phrase repeated phrase repeated phrase"],
        ...     mitigation_suggestions=["Increase temperature"]
        ... )
        >>> d = report.to_dict()
        >>> print(d['type'])
        repetitive_output
        >>> print(d['severity'])
        0.5
        >>> import json
        >>> json_str = json.dumps(d)  # Now serializable
        """
        return {
            "type": self.limitation_type.value,
            "severity": self.severity,
            "frequency": self.frequency,
            "examples": self.examples,
            "mitigation_suggestions": self.mitigation_suggestions,
        }


@dataclass
class SkillProfile:
    """Profile of skills within a capability category.

    This dataclass aggregates multiple CapabilityScores for skills within
    a single capability category, providing an overall assessment and
    identifying strengths and weaknesses. Generated by CapabilityProfiler.

    Parameters
    ----------
    category : CapabilityCategory
        The capability category being profiled (e.g., REASONING, CODING).
    skills : dict[SkillType, CapabilityScore]
        Dictionary mapping skill types to their individual scores within
        this category.
    overall_level : CapabilityLevel
        Overall proficiency level for the category, derived from
        overall_score using _score_to_level().
    overall_score : float
        Average score across all skills in this category, between 0.0
        and 1.0.
    strengths : list[SkillType]
        List of skills with scores >= 0.7, limited to top 3.
    weaknesses : list[SkillType]
        List of skills with scores < 0.4, limited to top 3.

    Attributes
    ----------
    category : CapabilityCategory
        The category being profiled.
    skills : dict[SkillType, CapabilityScore]
        Individual skill scores.
    overall_level : CapabilityLevel
        Category-level proficiency.
    overall_score : float
        Average category score.
    strengths : list[SkillType]
        Top performing skills.
    weaknesses : list[SkillType]
        Underperforming skills.

    Examples
    --------
    >>> from insideLLMs.contrib.fingerprinting import (
    ...     SkillProfile, CapabilityCategory, CapabilityLevel, SkillType,
    ...     CapabilityScore
    ... )
    >>> # Create skill scores
    >>> arithmetic_score = CapabilityScore(
    ...     category=CapabilityCategory.MATH,
    ...     skill=SkillType.ARITHMETIC,
    ...     score=0.95,
    ...     level=CapabilityLevel.EXPERT,
    ...     confidence=1.0,
    ...     evidence=[]
    ... )
    >>> algebra_score = CapabilityScore(
    ...     category=CapabilityCategory.MATH,
    ...     skill=SkillType.ALGEBRA,
    ...     score=0.72,
    ...     level=CapabilityLevel.INTERMEDIATE,
    ...     confidence=0.8,
    ...     evidence=[]
    ... )
    >>> # Create profile
    >>> profile = SkillProfile(
    ...     category=CapabilityCategory.MATH,
    ...     skills={
    ...         SkillType.ARITHMETIC: arithmetic_score,
    ...         SkillType.ALGEBRA: algebra_score
    ...     },
    ...     overall_level=CapabilityLevel.ADVANCED,
    ...     overall_score=0.835,
    ...     strengths=[SkillType.ARITHMETIC],
    ...     weaknesses=[]
    ... )
    >>> print(f"Math: {profile.overall_level.value} ({profile.overall_score:.0%})")
    Math: advanced (84%)

    >>> # Using CapabilityProfiler to generate profiles
    >>> from insideLLMs.contrib.fingerprinting import CapabilityProfiler
    >>> profiler = CapabilityProfiler()
    >>> skill_results = {
    ...     SkillType.ARITHMETIC: [("42", "42"), ("100", "100")],
    ...     SkillType.ALGEBRA: [("x=5", "x=5")],
    ... }
    >>> profile = profiler.profile_category(CapabilityCategory.MATH, skill_results)
    >>> print(f"Strengths: {[s.value for s in profile.strengths]}")

    See Also
    --------
    CapabilityProfiler : Creates SkillProfile objects
    CapabilityScore : Individual skill scores contained in profiles
    ModelFingerprint : Contains SkillProfiles for all categories
    """

    category: CapabilityCategory
    skills: dict[SkillType, CapabilityScore]
    overall_level: CapabilityLevel
    overall_score: float
    strengths: list[SkillType]
    weaknesses: list[SkillType]

    def to_dict(self) -> dict[str, Any]:
        """Convert the SkillProfile to a dictionary.

        Creates a serializable dictionary representation of the profile,
        converting all enum values and nested objects to their dictionary
        or string representations.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: 'category', 'skills', 'overall_level',
            'overall_score', 'strengths', 'weaknesses'. Nested CapabilityScore
            objects are also converted to dictionaries.

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import (
        ...     SkillProfile, CapabilityCategory, CapabilityLevel, SkillType,
        ...     CapabilityScore
        ... )
        >>> score = CapabilityScore(
        ...     category=CapabilityCategory.CODING,
        ...     skill=SkillType.CODE_GENERATION,
        ...     score=0.9,
        ...     level=CapabilityLevel.EXPERT,
        ...     confidence=1.0,
        ...     evidence=[]
        ... )
        >>> profile = SkillProfile(
        ...     category=CapabilityCategory.CODING,
        ...     skills={SkillType.CODE_GENERATION: score},
        ...     overall_level=CapabilityLevel.EXPERT,
        ...     overall_score=0.9,
        ...     strengths=[SkillType.CODE_GENERATION],
        ...     weaknesses=[]
        ... )
        >>> d = profile.to_dict()
        >>> print(d['category'])
        coding
        >>> print(d['overall_level'])
        expert
        >>> print('code_generation' in d['skills'])
        True
        """
        return {
            "category": self.category.value,
            "skills": {k.value: v.to_dict() for k, v in self.skills.items()},
            "overall_level": self.overall_level.value,
            "overall_score": self.overall_score,
            "strengths": [s.value for s in self.strengths],
            "weaknesses": [w.value for w in self.weaknesses],
        }


@dataclass
class ModelFingerprint:
    """Complete fingerprint of a model's capabilities.

    This dataclass represents a comprehensive assessment of a language model's
    capabilities across all categories, including skill profiles, detected
    limitations, and summary metrics. It serves as the primary output of the
    fingerprinting system and can be used for model comparison and selection.

    Parameters
    ----------
    model_id : str
        Unique identifier for the model (e.g., "gpt-4-turbo", "claude-3-opus").
    version : str
        Version of the fingerprint format (e.g., "1.0").
    fingerprint_hash : str
        16-character SHA256 hash uniquely identifying this capability profile.
        Generated from model_id, category scores, and top capabilities.
    category_scores : dict[CapabilityCategory, float]
        Dictionary mapping each capability category to its overall score (0-1).
    skill_profiles : dict[CapabilityCategory, SkillProfile]
        Dictionary mapping each category to its detailed SkillProfile.
    limitations : list[LimitationReport]
        List of detected limitations with severity and mitigation suggestions.
    top_capabilities : list[tuple[SkillType, float]]
        Top 5 skills by score, as (skill_type, score) tuples.
    main_limitations : list[LimitationType]
        Limitation types with severity > 0.5.
    overall_capability_score : float
        Average score across all categories, between 0.0 and 1.0.
    metadata : dict[str, Any], optional
        Additional metadata such as test date, model version, etc.
        Default is empty dict.

    Attributes
    ----------
    model_id : str
        The model identifier.
    version : str
        Fingerprint format version.
    fingerprint_hash : str
        Unique hash for this profile.
    category_scores : dict[CapabilityCategory, float]
        Scores by category.
    skill_profiles : dict[CapabilityCategory, SkillProfile]
        Detailed profiles by category.
    limitations : list[LimitationReport]
        Detected limitations.
    top_capabilities : list[tuple[SkillType, float]]
        Best performing skills.
    main_limitations : list[LimitationType]
        Most severe limitations.
    overall_capability_score : float
        Overall performance score.
    metadata : dict[str, Any]
        Additional metadata.
    capability_signature : str
        Property: Compact string signature of top capabilities.

    Examples
    --------
    >>> from insideLLMs.contrib.fingerprinting import (
    ...     create_fingerprint, SkillType, CapabilityCategory
    ... )
    >>> # Create a fingerprint from test results
    >>> skill_results = {
    ...     SkillType.LOGICAL_DEDUCTION: [
    ...         ("No, the conclusion is invalid", False),
    ...         ("Cannot deduce that from the premises", False),
    ...     ],
    ...     SkillType.CODE_GENERATION: [
    ...         ("def add(a, b): return a + b", "def add(a, b): return a + b"),
    ...     ],
    ...     SkillType.ARITHMETIC: [
    ...         ("42", "42"),
    ...         ("100", "100"),
    ...     ],
    ... }
    >>> responses = ["Sample response for limitation detection"]
    >>> fingerprint = create_fingerprint(
    ...     model_id="gpt-4-turbo",
    ...     skill_results=skill_results,
    ...     responses=responses,
    ...     version="1.0"
    ... )
    >>> print(f"Model: {fingerprint.model_id}")
    Model: gpt-4-turbo
    >>> print(f"Overall: {fingerprint.overall_capability_score:.0%}")
    Overall: 75%
    >>> print(f"Hash: {fingerprint.fingerprint_hash}")
    Hash: a1b2c3d4e5f67890

    >>> # Access category scores
    >>> for cat, score in sorted(
    ...     fingerprint.category_scores.items(),
    ...     key=lambda x: x[1],
    ...     reverse=True
    ... )[:3]:
    ...     print(f"{cat.value}: {score:.0%}")
    coding: 90%
    math: 85%
    reasoning: 80%

    >>> # Get capability signature for quick comparison
    >>> print(fingerprint.capability_signature)
    cod:0.90|mat:0.85|rea:0.80|lan:0.75|kno:0.70

    >>> # Serialize to dictionary/JSON
    >>> import json
    >>> fp_dict = fingerprint.to_dict()
    >>> json_str = json.dumps(fp_dict, indent=2)

    See Also
    --------
    FingerprintGenerator : Creates ModelFingerprint objects
    FingerprintComparator : Compares two fingerprints
    create_fingerprint : Convenience function for fingerprint creation
    compare_fingerprints : Convenience function for comparison
    """

    model_id: str
    version: str
    fingerprint_hash: str
    category_scores: dict[CapabilityCategory, float]
    skill_profiles: dict[CapabilityCategory, SkillProfile]
    limitations: list[LimitationReport]
    top_capabilities: list[tuple[SkillType, float]]
    main_limitations: list[LimitationType]
    overall_capability_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def capability_signature(self) -> str:
        """Get a compact capability signature string.

        Generates a human-readable signature showing the top 5 categories
        and their scores, useful for quick visual comparison of models.

        Returns
        -------
        str
            A pipe-separated string of "category:score" pairs, sorted by
            score in descending order. Category names are truncated to 3
            characters. Example: "cod:0.90|mat:0.85|rea:0.80|lan:0.75|kno:0.70"

        Examples
        --------
        >>> # Assuming fingerprint is already created
        >>> print(fingerprint.capability_signature)
        cod:0.90|mat:0.85|rea:0.80|lan:0.75|kno:0.70

        >>> # Use for quick comparison
        >>> print(f"Model A: {fp_a.capability_signature}")
        >>> print(f"Model B: {fp_b.capability_signature}")
        Model A: cod:0.90|mat:0.85|rea:0.80|lan:0.75|kno:0.70
        Model B: rea:0.92|lan:0.88|kno:0.85|cod:0.72|mat:0.65
        """
        scores = sorted(self.category_scores.items(), key=lambda x: x[1], reverse=True)
        parts = [f"{cat.value[:3]}:{score:.2f}" for cat, score in scores[:5]]
        return "|".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert the ModelFingerprint to a dictionary.

        Creates a fully serializable dictionary representation of the
        fingerprint, suitable for JSON serialization, storage, or transmission.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all fingerprint data with enum values
            converted to strings and nested objects converted to dictionaries.
            Includes the computed capability_signature.

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import create_fingerprint, SkillType
        >>> skill_results = {SkillType.ARITHMETIC: [("42", "42")]}
        >>> fingerprint = create_fingerprint(
        ...     model_id="test-model",
        ...     skill_results=skill_results,
        ...     responses=["test"],
        ...     version="1.0"
        ... )
        >>> d = fingerprint.to_dict()
        >>> print(d['model_id'])
        test-model
        >>> print(type(d['category_scores']))
        <class 'dict'>

        >>> # Serialize to JSON
        >>> import json
        >>> json_str = json.dumps(d, indent=2)
        >>> print(json_str[:50])
        {
          "model_id": "test-model",
          "version":

        >>> # Save to file
        >>> with open('fingerprint.json', 'w') as f:
        ...     json.dump(d, f, indent=2)
        """
        return {
            "model_id": self.model_id,
            "version": self.version,
            "fingerprint_hash": self.fingerprint_hash,
            "category_scores": {k.value: v for k, v in self.category_scores.items()},
            "skill_profiles": {k.value: v.to_dict() for k, v in self.skill_profiles.items()},
            "limitations": [limitation.to_dict() for limitation in self.limitations],
            "top_capabilities": [
                {"skill": s.value, "score": sc} for s, sc in self.top_capabilities
            ],
            "main_limitations": [limitation.value for limitation in self.main_limitations],
            "overall_capability_score": self.overall_capability_score,
            "capability_signature": self.capability_signature,
            "metadata": self.metadata,
        }


@dataclass
class FingerprintComparison:
    """Comparison between two model fingerprints.

    This dataclass represents a detailed comparison between two ModelFingerprint
    objects, identifying differences in capabilities, advantages of each model,
    and shared traits. Generated by the FingerprintComparator class.

    Parameters
    ----------
    model_a_id : str
        Identifier of the first model being compared.
    model_b_id : str
        Identifier of the second model being compared.
    similarity_score : float
        Overall similarity between models, from 0.0 (completely different)
        to 1.0 (identical). Calculated from category score differences.
    category_differences : dict[CapabilityCategory, float]
        Score differences by category (model_a - model_b). Positive values
        indicate model A is better, negative values indicate model B is better.
    skill_differences : dict[SkillType, float]
        Score differences by skill (model_a - model_b). Range is -1.0 to 1.0.
    model_a_advantages : list[SkillType]
        Skills where model A outperforms model B by > 0.1.
    model_b_advantages : list[SkillType]
        Skills where model B outperforms model A by > 0.1.
    shared_strengths : list[SkillType]
        Skills where both models score >= 0.7 and differ by < 0.1.
    shared_weaknesses : list[SkillType]
        Skills where both models score < 0.4 and differ by < 0.1.

    Attributes
    ----------
    model_a_id : str
        First model identifier.
    model_b_id : str
        Second model identifier.
    similarity_score : float
        Overall similarity (0-1).
    category_differences : dict[CapabilityCategory, float]
        Category-level differences.
    skill_differences : dict[SkillType, float]
        Skill-level differences.
    model_a_advantages : list[SkillType]
        Model A's advantages.
    model_b_advantages : list[SkillType]
        Model B's advantages.
    shared_strengths : list[SkillType]
        Common strong skills.
    shared_weaknesses : list[SkillType]
        Common weak skills.
    winner : str | None
        Property: Model with clear advantage, or None if too close.

    Examples
    --------
    >>> from insideLLMs.contrib.fingerprinting import (
    ...     compare_fingerprints, create_fingerprint, SkillType
    ... )
    >>> # Create two fingerprints
    >>> fp_a = create_fingerprint(
    ...     model_id="gpt-4",
    ...     skill_results={
    ...         SkillType.CODE_GENERATION: [("correct", "correct")],
    ...         SkillType.ARITHMETIC: [("42", "42")],
    ...     },
    ...     responses=["test"],
    ... )
    >>> fp_b = create_fingerprint(
    ...     model_id="claude-3",
    ...     skill_results={
    ...         SkillType.CODE_GENERATION: [("mostly correct", "correct")],
    ...         SkillType.CREATIVE_WRITING: [("excellent story", "story")],
    ...     },
    ...     responses=["test"],
    ... )
    >>> comparison = compare_fingerprints(fp_a, fp_b)
    >>> print(f"Similarity: {comparison.similarity_score:.0%}")
    Similarity: 75%

    >>> # Check model advantages
    >>> print(f"GPT-4 advantages: {[s.value for s in comparison.model_a_advantages]}")
    GPT-4 advantages: ['code_generation', 'arithmetic']
    >>> print(f"Claude advantages: {[s.value for s in comparison.model_b_advantages]}")
    Claude advantages: ['creative_writing']

    >>> # Determine winner (if clear)
    >>> if comparison.winner:
    ...     print(f"Better model: {comparison.winner}")
    ... else:
    ...     print("Too close to call")
    Better model: gpt-4

    >>> # Analyze specific differences
    >>> from insideLLMs.contrib.fingerprinting import CapabilityCategory
    >>> coding_diff = comparison.category_differences.get(CapabilityCategory.CODING, 0)
    >>> print(f"Coding difference: {coding_diff:+.2f}")
    Coding difference: +0.15

    See Also
    --------
    FingerprintComparator : Creates FingerprintComparison objects
    compare_fingerprints : Convenience function for comparison
    ModelFingerprint : The fingerprints being compared
    """

    model_a_id: str
    model_b_id: str
    similarity_score: float
    category_differences: dict[CapabilityCategory, float]
    skill_differences: dict[SkillType, float]
    model_a_advantages: list[SkillType]
    model_b_advantages: list[SkillType]
    shared_strengths: list[SkillType]
    shared_weaknesses: list[SkillType]

    @property
    def winner(self) -> Optional[str]:
        """Get the overall better model, if there is a clear winner.

        Determines if one model has a significant advantage over the other
        based on the number of skill advantages. A model is declared the
        winner if it has more than 1.5x the advantages of the other model.

        Returns
        -------
        str | None
            The model_id of the better model, or None if the comparison
            is too close to determine a clear winner.

        Examples
        --------
        >>> # Assuming comparison is already created
        >>> if comparison.winner:
        ...     print(f"The better model is: {comparison.winner}")
        ... else:
        ...     print("Models are roughly equivalent")
        The better model is: gpt-4

        >>> # Check the underlying advantage counts
        >>> a_wins = len(comparison.model_a_advantages)
        >>> b_wins = len(comparison.model_b_advantages)
        >>> print(f"Model A advantages: {a_wins}, Model B advantages: {b_wins}")
        Model A advantages: 5, Model B advantages: 2
        """
        a_wins = len(self.model_a_advantages)
        b_wins = len(self.model_b_advantages)
        if a_wins > b_wins * 1.5:
            return self.model_a_id
        elif b_wins > a_wins * 1.5:
            return self.model_b_id
        return None  # Too close to call

    def to_dict(self) -> dict[str, Any]:
        """Convert the FingerprintComparison to a dictionary.

        Creates a serializable dictionary representation of the comparison,
        converting all enum values to their string representations.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all comparison data. Includes the computed
            'winner' property value.

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import compare_fingerprints
        >>> # Assuming fp_a and fp_b are fingerprints
        >>> comparison = compare_fingerprints(fp_a, fp_b)
        >>> d = comparison.to_dict()
        >>> print(d['model_a_id'])
        gpt-4
        >>> print(d['similarity_score'])
        0.78

        >>> # Serialize to JSON
        >>> import json
        >>> json_str = json.dumps(d, indent=2)

        >>> # Access skill differences
        >>> for skill, diff in d['skill_differences'].items():
        ...     if abs(diff) > 0.2:
        ...         print(f"{skill}: {diff:+.2f}")
        code_generation: +0.25
        creative_writing: -0.30
        """
        return {
            "model_a_id": self.model_a_id,
            "model_b_id": self.model_b_id,
            "similarity_score": self.similarity_score,
            "category_differences": {k.value: v for k, v in self.category_differences.items()},
            "skill_differences": {k.value: v for k, v in self.skill_differences.items()},
            "model_a_advantages": [s.value for s in self.model_a_advantages],
            "model_b_advantages": [s.value for s in self.model_b_advantages],
            "shared_strengths": [s.value for s in self.shared_strengths],
            "shared_weaknesses": [s.value for s in self.shared_weaknesses],
            "winner": self.winner,
        }


class CapabilityProbe:
    """Base class for capability probes.

    A capability probe is responsible for generating test prompts and
    evaluating model responses for a specific skill. This is the base
    class that should be subclassed to create probes for different
    skill types. Subclasses must implement generate_test() and evaluate().

    Parameters
    ----------
    category : CapabilityCategory
        The high-level capability category this probe tests
        (e.g., REASONING, CODING).
    skill : SkillType
        The specific skill being tested (e.g., LOGICAL_DEDUCTION,
        CODE_GENERATION).
    name : str
        Human-readable name for this probe, typically derived from
        category and skill (e.g., "reasoning_logical_deduction").
    description : str
        Description of what this probe tests.

    Attributes
    ----------
    category : CapabilityCategory
        The capability category.
    skill : SkillType
        The skill being tested.
    name : str
        The probe name.
    description : str
        The probe description.

    Examples
    --------
    >>> from insideLLMs.contrib.fingerprinting import (
    ...     CapabilityProbe, CapabilityCategory, SkillType
    ... )
    >>> # Create a custom probe for arithmetic
    >>> class ArithmeticProbe(CapabilityProbe):
    ...     def __init__(self):
    ...         super().__init__(
    ...             category=CapabilityCategory.MATH,
    ...             skill=SkillType.ARITHMETIC,
    ...             name="arithmetic_addition",
    ...             description="Tests basic addition ability"
    ...         )
    ...
    ...     def generate_test(self):
    ...         return ("What is 25 + 17?", "42")
    ...
    ...     def evaluate(self, response, expected):
    ...         if expected in response:
    ...             return 1.0
    ...         return 0.0
    >>> probe = ArithmeticProbe()
    >>> prompt, expected = probe.generate_test()
    >>> print(prompt)
    What is 25 + 17?
    >>> score = probe.evaluate("The answer is 42.", "42")
    >>> print(score)
    1.0

    >>> # Using the built-in ReasoningProbe
    >>> from insideLLMs.contrib.fingerprinting import ReasoningProbe
    >>> probe = ReasoningProbe(SkillType.LOGICAL_DEDUCTION)
    >>> prompt, expected = probe.generate_test()
    >>> print(probe.name)
    reasoning_logical_deduction

    See Also
    --------
    ReasoningProbe : Concrete probe for reasoning skills
    SkillAssessor : Uses probes to assess skills
    """

    def __init__(
        self,
        category: CapabilityCategory,
        skill: SkillType,
        name: str,
        description: str,
    ):
        """Initialize the capability probe.

        Args
        ----
        category : CapabilityCategory
            The high-level capability category this probe tests.
        skill : SkillType
            The specific skill being tested.
        name : str
            Human-readable name for this probe.
        description : str
            Description of what this probe tests.

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import (
        ...     CapabilityProbe, CapabilityCategory, SkillType
        ... )
        >>> class MyProbe(CapabilityProbe):
        ...     def generate_test(self):
        ...         return ("Test prompt", "expected")
        ...     def evaluate(self, response, expected):
        ...         return 1.0 if expected in response else 0.0
        >>> probe = MyProbe(
        ...     category=CapabilityCategory.KNOWLEDGE,
        ...     skill=SkillType.FACTUAL_RECALL,
        ...     name="capital_cities",
        ...     description="Tests knowledge of world capitals"
        ... )
        >>> print(probe.name)
        capital_cities
        """
        self.category = category
        self.skill = skill
        self.name = name
        self.description = description

    def generate_test(self) -> tuple[str, Any]:
        """Generate a test prompt and expected answer.

        Subclasses must implement this method to produce test cases
        for the specific skill being assessed. Each call may generate
        different tests (e.g., by sampling from a pool of questions).

        Returns
        -------
        tuple[str, Any]
            A tuple of (prompt, expected_answer) where:
            - prompt: The question or task to present to the model
            - expected_answer: The correct answer or key for evaluation,
              can be any type depending on the evaluation method

        Raises
        ------
        NotImplementedError
            If called on the base class without being overridden.

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import ReasoningProbe, SkillType
        >>> probe = ReasoningProbe(SkillType.LOGICAL_DEDUCTION)
        >>> prompt, expected = probe.generate_test()
        >>> print(type(prompt))
        <class 'str'>
        >>> print(prompt[:30])
        If all roses are flowers, and

        >>> # Custom probe implementation
        >>> class FactProbe(CapabilityProbe):
        ...     def generate_test(self):
        ...         import random
        ...         facts = [
        ...             ("What is the capital of France?", "Paris"),
        ...             ("What is the capital of Japan?", "Tokyo"),
        ...         ]
        ...         return random.choice(facts)
        """
        raise NotImplementedError

    def evaluate(self, response: str, expected: Any) -> float:
        """Evaluate a model response against the expected answer.

        Subclasses must implement this method to score model responses.
        The evaluation logic depends on the skill being tested and may
        use exact matching, keyword detection, or more complex analysis.

        Args
        ----
        response : str
            The model's response to the test prompt.
        expected : Any
            The expected answer returned by generate_test().

        Returns
        -------
        float
            A score between 0.0 and 1.0, where:
            - 1.0 indicates a perfect response
            - 0.0 indicates a completely incorrect response
            - Values in between indicate partial correctness

        Raises
        ------
        NotImplementedError
            If called on the base class without being overridden.

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import ReasoningProbe, SkillType
        >>> probe = ReasoningProbe(SkillType.LOGICAL_DEDUCTION)
        >>> # Test with a correct response
        >>> score = probe.evaluate(
        ...     "No, we cannot conclude that. The syllogism is invalid.",
        ...     False
        ... )
        >>> print(f"Score: {score}")
        Score: 1.0

        >>> # Test with an incorrect response
        >>> score = probe.evaluate(
        ...     "Yes, some roses must fade quickly.",
        ...     False
        ... )
        >>> print(f"Score: {score}")
        Score: 0.0
        """
        raise NotImplementedError


class ReasoningProbe(CapabilityProbe):
    """Probe for testing reasoning capabilities.

    A concrete implementation of CapabilityProbe specifically designed
    for testing reasoning skills such as logical deduction and causal
    reasoning. Contains built-in test cases and evaluation logic for
    common reasoning tasks.

    Parameters
    ----------
    skill : SkillType
        The specific reasoning skill to test. Should be one of the
        reasoning-category skills: LOGICAL_DEDUCTION, CAUSAL_REASONING,
        ANALOGICAL_REASONING, or COUNTERFACTUAL_REASONING.

    Attributes
    ----------
    category : CapabilityCategory
        Always set to CapabilityCategory.REASONING.
    skill : SkillType
        The reasoning skill being tested.
    name : str
        Auto-generated name in format "reasoning_{skill.value}".
    description : str
        Auto-generated description.

    Examples
    --------
    >>> from insideLLMs.contrib.fingerprinting import ReasoningProbe, SkillType
    >>> # Create a logical deduction probe
    >>> probe = ReasoningProbe(SkillType.LOGICAL_DEDUCTION)
    >>> print(probe.name)
    reasoning_logical_deduction
    >>> print(probe.category.value)
    reasoning

    >>> # Generate a test
    >>> prompt, expected = probe.generate_test()
    >>> print(prompt[:50])
    If all roses are flowers, and some flowers fade q
    >>> print(expected)
    False

    >>> # Evaluate responses
    >>> correct_response = "No, we cannot conclude that. This is an invalid syllogism."
    >>> score = probe.evaluate(correct_response, expected)
    >>> print(f"Score: {score}")
    Score: 1.0

    >>> wrong_response = "Yes, since roses are flowers, they must fade quickly."
    >>> score = probe.evaluate(wrong_response, expected)
    >>> print(f"Score: {score}")
    Score: 0.0

    >>> # Create a causal reasoning probe
    >>> probe = ReasoningProbe(SkillType.CAUSAL_REASONING)
    >>> prompt, expected = probe.generate_test()
    >>> print("check" in expected)
    True

    See Also
    --------
    CapabilityProbe : Base class for all probes
    SkillAssessor : Uses probes for skill assessment
    """

    def __init__(self, skill: SkillType):
        """Initialize the reasoning probe.

        Args
        ----
        skill : SkillType
            The specific reasoning skill to test. Supported skills include
            LOGICAL_DEDUCTION and CAUSAL_REASONING (others return generic tests).

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import ReasoningProbe, SkillType
        >>> probe = ReasoningProbe(SkillType.LOGICAL_DEDUCTION)
        >>> print(probe.skill.value)
        logical_deduction

        >>> probe = ReasoningProbe(SkillType.CAUSAL_REASONING)
        >>> print(probe.description)
        Tests causal_reasoning capability
        """
        super().__init__(
            category=CapabilityCategory.REASONING,
            skill=skill,
            name=f"reasoning_{skill.value}",
            description=f"Tests {skill.value} capability",
        )

    def generate_test(self) -> tuple[str, Any]:
        """Generate a reasoning test prompt and expected answer.

        Returns test cases appropriate for the skill type. Currently
        supports LOGICAL_DEDUCTION and CAUSAL_REASONING with specific
        test cases; other skills return generic placeholder tests.

        Returns
        -------
        tuple[str, Any]
            A tuple of (prompt, expected_answer):
            - For LOGICAL_DEDUCTION: Tests syllogistic reasoning with
              expected=False (the syllogism is invalid)
            - For CAUSAL_REASONING: Tests causal analysis with expected
              being a hint about the correct approach
            - For other skills: Returns generic test placeholder

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import ReasoningProbe, SkillType
        >>> # Logical deduction test
        >>> probe = ReasoningProbe(SkillType.LOGICAL_DEDUCTION)
        >>> prompt, expected = probe.generate_test()
        >>> "roses" in prompt and "flowers" in prompt
        True
        >>> expected
        False

        >>> # Causal reasoning test
        >>> probe = ReasoningProbe(SkillType.CAUSAL_REASONING)
        >>> prompt, expected = probe.generate_test()
        >>> "street" in prompt.lower() and "wet" in prompt.lower()
        True
        """
        if self.skill == SkillType.LOGICAL_DEDUCTION:
            return (
                "If all roses are flowers, and some flowers fade quickly, "
                "can we conclude that some roses fade quickly?",
                False,  # No, this is an invalid syllogism
            )
        elif self.skill == SkillType.CAUSAL_REASONING:
            return (
                "The street is wet. It might have rained, or a sprinkler was on. "
                "What additional observation would help determine the cause?",
                "check_other_areas",  # Check if other areas are also wet
            )
        return ("Test reasoning", "expected")

    def evaluate(self, response: str, expected: Any) -> float:
        """Evaluate a reasoning response against the expected answer.

        Uses skill-specific evaluation logic to score the response.
        For logical deduction, checks for correct rejection or acceptance
        of syllogisms. Other skills use a default moderate score.

        Args
        ----
        response : str
            The model's response to the reasoning test.
        expected : Any
            The expected answer from generate_test().

        Returns
        -------
        float
            A score between 0.0 and 1.0:
            - 1.0: Correct reasoning (e.g., correctly identified invalid syllogism)
            - 0.0: Incorrect reasoning (e.g., accepted invalid syllogism)
            - 0.5: Ambiguous or moderate response

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import ReasoningProbe, SkillType
        >>> probe = ReasoningProbe(SkillType.LOGICAL_DEDUCTION)

        >>> # Correct response rejecting invalid syllogism
        >>> score = probe.evaluate(
        ...     "No, this conclusion cannot be drawn from the premises.",
        ...     False
        ... )
        >>> score
        1.0

        >>> # Incorrect response accepting invalid syllogism
        >>> score = probe.evaluate(
        ...     "Yes, some roses fade quickly.",
        ...     False
        ... )
        >>> score
        0.0

        >>> # Ambiguous response
        >>> score = probe.evaluate(
        ...     "This is a complex logical question.",
        ...     False
        ... )
        >>> score
        0.5
        """
        response_lower = response.lower()

        if self.skill == SkillType.LOGICAL_DEDUCTION:
            # Check for correct reasoning about syllogism
            if expected is False:
                if "no" in response_lower or "cannot" in response_lower:
                    return 1.0
                elif "yes" in response_lower:
                    return 0.0
            return 0.5

        return 0.5  # Default moderate score


class SkillAssessor:
    """Assess individual skills based on model responses.

    The SkillAssessor evaluates model performance on specific skills by
    comparing responses to expected answers. It supports custom evaluation
    functions and can register capability probes for automated testing.

    Parameters
    ----------
    evaluator : Optional[Callable[[str, str], float]], optional
        Custom evaluation function that takes (response, expected) and
        returns a score between 0.0 and 1.0. If not provided, uses
        keyword-based matching as the default evaluator.

    Attributes
    ----------
    _evaluator : Callable[[str, str], float]
        The evaluation function used to score responses.
    _probes : dict[SkillType, list[CapabilityProbe]]
        Registered probes for each skill type.

    Examples
    --------
    >>> from insideLLMs.contrib.fingerprinting import SkillAssessor, SkillType
    >>> # Create assessor with default evaluator
    >>> assessor = SkillAssessor()

    >>> # Assess a skill from response-expected pairs
    >>> responses = [
    ...     ("The capital of France is Paris.", "Paris"),
    ...     ("Mount Everest is the tallest mountain.", "Everest"),
    ...     ("The Amazon is the longest river.", "Nile"),  # Wrong
    ... ]
    >>> score = assessor.assess_skill(SkillType.FACTUAL_RECALL, responses)
    >>> print(f"Score: {score.score:.2f}, Level: {score.level.value}")
    Score: 0.67, Level: intermediate

    >>> # Use a custom evaluator
    >>> def strict_evaluator(response: str, expected: str) -> float:
    ...     return 1.0 if expected.lower() == response.lower().strip() else 0.0
    >>> strict_assessor = SkillAssessor(evaluator=strict_evaluator)

    >>> # Register and use probes
    >>> from insideLLMs.contrib.fingerprinting import ReasoningProbe
    >>> probe = ReasoningProbe(SkillType.LOGICAL_DEDUCTION)
    >>> assessor.register_probe(probe)

    See Also
    --------
    CapabilityProbe : Probes that can be registered with the assessor
    CapabilityScore : The result type returned by assess_skill
    CapabilityProfiler : Uses SkillAssessor to build category profiles
    """

    def __init__(
        self,
        evaluator: Optional[Callable[[str, str], float]] = None,
    ):
        """Initialize the skill assessor.

        Args
        ----
        evaluator : Optional[Callable[[str, str], float]], optional
            Custom evaluation function that takes (response, expected) and
            returns a score between 0.0 and 1.0. If not provided, uses the
            default keyword-matching evaluator.

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import SkillAssessor
        >>> # Default evaluator
        >>> assessor = SkillAssessor()

        >>> # Custom evaluator for exact matching
        >>> def exact_match(response: str, expected: str) -> float:
        ...     return 1.0 if expected in response else 0.0
        >>> assessor = SkillAssessor(evaluator=exact_match)

        >>> # Custom evaluator with fuzzy matching
        >>> def fuzzy_match(response: str, expected: str) -> float:
        ...     response_words = set(response.lower().split())
        ...     expected_words = set(expected.lower().split())
        ...     if not expected_words:
        ...         return 0.5
        ...     overlap = len(response_words & expected_words)
        ...     return overlap / len(expected_words)
        >>> assessor = SkillAssessor(evaluator=fuzzy_match)
        """
        self._evaluator = evaluator or self._default_evaluator
        self._probes: dict[SkillType, list[CapabilityProbe]] = {}

    @staticmethod
    def _default_evaluator(response: str, expected: str) -> float:
        """Default evaluation using keyword matching.

        Evaluates a response against an expected answer using case-insensitive
        matching. First checks for exact substring match, then falls back to
        word overlap scoring.

        Args
        ----
        response : str
            The model's response text.
        expected : str
            The expected answer or keywords.

        Returns
        -------
        float
            Score between 0.0 and 1.0:
            - 1.0: Expected string found as substring in response
            - 0.0-1.0: Ratio of expected words found in response
            - 0.5: Default when expected is empty

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import SkillAssessor
        >>> # Exact substring match
        >>> score = SkillAssessor._default_evaluator(
        ...     "The answer is Paris, the capital.",
        ...     "Paris"
        ... )
        >>> score
        1.0

        >>> # Partial word overlap
        >>> score = SkillAssessor._default_evaluator(
        ...     "The city has many museums.",
        ...     "Paris museums art"
        ... )
        >>> print(f"{score:.2f}")  # 1 out of 3 words match
        0.33
        """
        response_lower = response.lower()
        expected_lower = str(expected).lower()

        # Check for exact match
        if expected_lower in response_lower:
            return 1.0

        # Check for partial match
        expected_words = set(expected_lower.split())
        response_words = set(response_lower.split())
        overlap = len(expected_words & response_words)
        if expected_words:
            return overlap / len(expected_words)
        return 0.5

    def register_probe(self, probe: CapabilityProbe) -> None:
        """Register a capability probe for a skill.

        Probes can be registered to enable automated test generation
        for specific skills. Multiple probes can be registered for
        the same skill to provide variety in testing.

        Args
        ----
        probe : CapabilityProbe
            The probe to register. Its skill attribute determines
            which skill it will be associated with.

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import (
        ...     SkillAssessor, ReasoningProbe, SkillType
        ... )
        >>> assessor = SkillAssessor()

        >>> # Register a reasoning probe
        >>> probe = ReasoningProbe(SkillType.LOGICAL_DEDUCTION)
        >>> assessor.register_probe(probe)

        >>> # Register multiple probes for the same skill
        >>> probe2 = ReasoningProbe(SkillType.LOGICAL_DEDUCTION)
        >>> assessor.register_probe(probe2)

        >>> # Check registered probes
        >>> len(assessor._probes[SkillType.LOGICAL_DEDUCTION])
        2
        """
        if probe.skill not in self._probes:
            self._probes[probe.skill] = []
        self._probes[probe.skill].append(probe)

    def assess_skill(
        self,
        skill: SkillType,
        responses: list[tuple[str, Any]],
    ) -> CapabilityScore:
        """Assess a skill based on response-expected pairs.

        Evaluates each response against its expected answer, calculates
        average score, determines proficiency level, and generates
        evidence strings for the assessment.

        Args
        ----
        skill : SkillType
            The skill being assessed.
        responses : list[tuple[str, Any]]
            List of (response, expected) tuples where response is the
            model output and expected is the correct answer.

        Returns
        -------
        CapabilityScore
            Assessment result containing:
            - score: Average score across all responses (0-1)
            - level: Proficiency level based on score
            - confidence: Based on number of samples (max at 5 samples)
            - evidence: Up to 3 evidence strings

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import SkillAssessor, SkillType
        >>> assessor = SkillAssessor()

        >>> # Assess with multiple responses
        >>> responses = [
        ...     ("42", "42"),
        ...     ("100", "100"),
        ...     ("7", "8"),  # Incorrect
        ... ]
        >>> score = assessor.assess_skill(SkillType.ARITHMETIC, responses)
        >>> print(f"Score: {score.score:.2f}")
        Score: 0.67
        >>> print(f"Level: {score.level.value}")
        Level: intermediate
        >>> print(f"Confidence: {score.confidence:.1f}")
        Confidence: 0.6

        >>> # Empty responses return zero score
        >>> score = assessor.assess_skill(SkillType.ARITHMETIC, [])
        >>> print(f"Score: {score.score}, Level: {score.level.value}")
        Score: 0.0, Level: none

        >>> # Perfect responses
        >>> responses = [("Paris", "Paris")] * 5
        >>> score = assessor.assess_skill(SkillType.FACTUAL_RECALL, responses)
        >>> print(f"Score: {score.score}, Confidence: {score.confidence}")
        Score: 1.0, Confidence: 1.0
        """
        if not responses:
            return CapabilityScore(
                category=self._get_category(skill),
                skill=skill,
                score=0.0,
                level=CapabilityLevel.NONE,
                confidence=0.0,
                evidence=[],
            )

        scores = []
        evidence = []
        for response, expected in responses:
            score = self._evaluator(response, str(expected))
            scores.append(score)
            if score >= 0.8:
                evidence.append(f"Correct response for {skill.value}")
            elif score <= 0.2:
                evidence.append(f"Incorrect response for {skill.value}")

        avg_score = sum(scores) / len(scores)
        level = _score_to_level(avg_score)
        confidence = min(1.0, len(responses) / 5)  # More samples = higher confidence

        return CapabilityScore(
            category=self._get_category(skill),
            skill=skill,
            score=avg_score,
            level=level,
            confidence=confidence,
            evidence=evidence[:3],  # Keep top 3
        )

    @staticmethod
    def _get_category(skill: SkillType) -> CapabilityCategory:
        """Get the capability category for a skill.

        Maps each SkillType to its parent CapabilityCategory using
        a predefined mapping table.

        Args
        ----
        skill : SkillType
            The skill to look up.

        Returns
        -------
        CapabilityCategory
            The parent category for the skill. Returns REASONING as
            default if skill is not found in the mapping.

        Examples
        --------
        >>> from insideLLMs.contrib.fingerprinting import SkillAssessor, SkillType
        >>> cat = SkillAssessor._get_category(SkillType.CODE_GENERATION)
        >>> print(cat.value)
        coding

        >>> cat = SkillAssessor._get_category(SkillType.ARITHMETIC)
        >>> print(cat.value)
        math

        >>> cat = SkillAssessor._get_category(SkillType.LOGICAL_DEDUCTION)
        >>> print(cat.value)
        reasoning
        """
        skill_categories = {
            SkillType.LOGICAL_DEDUCTION: CapabilityCategory.REASONING,
            SkillType.CAUSAL_REASONING: CapabilityCategory.REASONING,
            SkillType.ANALOGICAL_REASONING: CapabilityCategory.REASONING,
            SkillType.COUNTERFACTUAL_REASONING: CapabilityCategory.REASONING,
            SkillType.GRAMMAR: CapabilityCategory.LANGUAGE,
            SkillType.VOCABULARY: CapabilityCategory.LANGUAGE,
            SkillType.STYLE_ADAPTATION: CapabilityCategory.LANGUAGE,
            SkillType.TRANSLATION: CapabilityCategory.LANGUAGE,
            SkillType.SUMMARIZATION: CapabilityCategory.LANGUAGE,
            SkillType.FACTUAL_RECALL: CapabilityCategory.KNOWLEDGE,
            SkillType.TEMPORAL_KNOWLEDGE: CapabilityCategory.KNOWLEDGE,
            SkillType.DOMAIN_EXPERTISE: CapabilityCategory.KNOWLEDGE,
            SkillType.COMMON_SENSE: CapabilityCategory.KNOWLEDGE,
            SkillType.CREATIVE_WRITING: CapabilityCategory.CREATIVITY,
            SkillType.BRAINSTORMING: CapabilityCategory.CREATIVITY,
            SkillType.HUMOR: CapabilityCategory.CREATIVITY,
            SkillType.STORYTELLING: CapabilityCategory.CREATIVITY,
            SkillType.CODE_GENERATION: CapabilityCategory.CODING,
            SkillType.CODE_UNDERSTANDING: CapabilityCategory.CODING,
            SkillType.DEBUGGING: CapabilityCategory.CODING,
            SkillType.CODE_TRANSLATION: CapabilityCategory.CODING,
            SkillType.ARITHMETIC: CapabilityCategory.MATH,
            SkillType.ALGEBRA: CapabilityCategory.MATH,
            SkillType.WORD_PROBLEMS: CapabilityCategory.MATH,
            SkillType.SYMBOLIC_MANIPULATION: CapabilityCategory.MATH,
            SkillType.DATA_ANALYSIS: CapabilityCategory.ANALYSIS,
            SkillType.TEXT_ANALYSIS: CapabilityCategory.ANALYSIS,
            SkillType.PATTERN_RECOGNITION: CapabilityCategory.ANALYSIS,
            SkillType.CRITICAL_EVALUATION: CapabilityCategory.ANALYSIS,
            SkillType.TASK_DECOMPOSITION: CapabilityCategory.INSTRUCTION_FOLLOWING,
            SkillType.FORMAT_ADHERENCE: CapabilityCategory.INSTRUCTION_FOLLOWING,
            SkillType.CONSTRAINT_HANDLING: CapabilityCategory.INSTRUCTION_FOLLOWING,
            SkillType.MULTI_STEP_EXECUTION: CapabilityCategory.INSTRUCTION_FOLLOWING,
        }
        return skill_categories.get(skill, CapabilityCategory.REASONING)


class LimitationDetector:
    """Detect model limitations."""

    def __init__(self):
        """Initialize detector."""
        self._patterns: dict[LimitationType, list[Callable[[str], bool]]] = {
            LimitationType.REPETITIVE_OUTPUT: [self._check_repetition],
            LimitationType.FORMAT_INCONSISTENT: [self._check_format],
            LimitationType.HALLUCINATION_PRONE: [self._check_hallucination_markers],
        }

    @staticmethod
    def _check_repetition(response: str) -> bool:
        """Check for repetitive patterns."""
        words = response.lower().split()
        if len(words) < 10:
            return False
        # Check for repeated phrases
        for i in range(len(words) - 3):
            phrase = " ".join(words[i : i + 3])
            remaining = " ".join(words[i + 3 :])
            if phrase in remaining:
                return True
        return False

    @staticmethod
    def _check_format(response: str) -> bool:
        """Check for format inconsistencies."""
        # Simple heuristic: inconsistent list markers
        lines = response.split("\n")
        markers_found: set[str] = set()
        for line in lines:
            line = line.strip()
            if line.startswith("- "):
                markers_found.add("dash")
            elif line.startswith("* "):
                markers_found.add("asterisk")
            elif line and line[0].isdigit() and "." in line[:3]:
                markers_found.add("number")
        return len(markers_found) > 1  # Multiple markers = inconsistent

    @staticmethod
    def _check_hallucination_markers(response: str) -> bool:
        """Check for hallucination marker phrases."""
        hallucination_markers = [
            "i believe",
            "i think",
            "approximately",
            "around",
            "roughly",
            "possibly",
            "might be",
        ]
        response_lower = response.lower()
        count = sum(1 for m in hallucination_markers if m in response_lower)
        return count >= 3  # Multiple hedging phrases

    def detect(
        self,
        responses: list[str],
    ) -> list[LimitationReport]:
        """Detect limitations from responses.

        Args:
            responses: List of model responses

        Returns:
            List of LimitationReport objects
        """
        reports = []

        for limitation_type, checkers in self._patterns.items():
            detected_count = 0
            examples = []

            for response in responses:
                for checker in checkers:
                    if checker(response):
                        detected_count += 1
                        if len(examples) < 3:
                            examples.append(response[:100] + "...")
                        break

            if detected_count > 0:
                frequency = detected_count / len(responses)
                severity = min(1.0, frequency * 2)  # Higher frequency = higher severity

                reports.append(
                    LimitationReport(
                        limitation_type=limitation_type,
                        severity=severity,
                        frequency=frequency,
                        examples=examples,
                        mitigation_suggestions=self._get_mitigations(limitation_type),
                    )
                )

        return reports

    @staticmethod
    def _get_mitigations(limitation: LimitationType) -> list[str]:
        """Get mitigation suggestions for a limitation."""
        mitigations = {
            LimitationType.REPETITIVE_OUTPUT: [
                "Increase temperature parameter",
                "Add diversity penalties",
                "Use different sampling methods",
            ],
            LimitationType.FORMAT_INCONSISTENT: [
                "Provide explicit format examples",
                "Use few-shot prompting",
                "Add format constraints to prompt",
            ],
            LimitationType.HALLUCINATION_PRONE: [
                "Request citations or sources",
                "Use retrieval augmentation",
                "Ask for confidence levels",
            ],
            LimitationType.FACTUAL_ERROR_PRONE: [
                "Use fact-checking prompts",
                "Request step-by-step reasoning",
                "Validate with external sources",
            ],
            LimitationType.CALCULATION_ERRORS: [
                "Request step-by-step calculations",
                "Use external calculator tools",
                "Ask for verification of results",
            ],
            LimitationType.KNOWLEDGE_CUTOFF: [
                "Provide current context",
                "Use retrieval augmentation",
                "Explicitly mention timeframe",
            ],
        }
        return mitigations.get(limitation, ["Review model documentation"])


class CapabilityProfiler:
    """Build skill profiles for capability categories."""

    def __init__(
        self,
        skill_assessor: Optional[SkillAssessor] = None,
    ):
        """Initialize profiler.

        Args:
            skill_assessor: Optional custom skill assessor
        """
        self._assessor = skill_assessor or SkillAssessor()

    def profile_category(
        self,
        category: CapabilityCategory,
        skill_results: dict[SkillType, list[tuple[str, Any]]],
    ) -> SkillProfile:
        """Build profile for a category.

        Args:
            category: Category to profile
            skill_results: Results for each skill (response, expected) pairs

        Returns:
            SkillProfile for the category
        """
        skill_scores = {}
        for skill, results in skill_results.items():
            if self._is_skill_in_category(skill, category):
                skill_scores[skill] = self._assessor.assess_skill(skill, results)

        if not skill_scores:
            return SkillProfile(
                category=category,
                skills={},
                overall_level=CapabilityLevel.NONE,
                overall_score=0.0,
                strengths=[],
                weaknesses=[],
            )

        # Calculate overall score
        scores = [s.score for s in skill_scores.values()]
        overall_score = sum(scores) / len(scores)
        overall_level = _score_to_level(overall_score)

        # Identify strengths and weaknesses
        sorted_skills = sorted(skill_scores.items(), key=lambda x: x[1].score, reverse=True)
        strengths = [s for s, score in sorted_skills if score.score >= 0.7][:3]
        weaknesses = [s for s, score in sorted_skills if score.score < 0.4][:3]

        return SkillProfile(
            category=category,
            skills=skill_scores,
            overall_level=overall_level,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
        )

    @staticmethod
    def _is_skill_in_category(skill: SkillType, category: CapabilityCategory) -> bool:
        """Check if skill belongs to category."""
        category_skills = {
            CapabilityCategory.REASONING: {
                SkillType.LOGICAL_DEDUCTION,
                SkillType.CAUSAL_REASONING,
                SkillType.ANALOGICAL_REASONING,
                SkillType.COUNTERFACTUAL_REASONING,
            },
            CapabilityCategory.LANGUAGE: {
                SkillType.GRAMMAR,
                SkillType.VOCABULARY,
                SkillType.STYLE_ADAPTATION,
                SkillType.TRANSLATION,
                SkillType.SUMMARIZATION,
            },
            CapabilityCategory.KNOWLEDGE: {
                SkillType.FACTUAL_RECALL,
                SkillType.TEMPORAL_KNOWLEDGE,
                SkillType.DOMAIN_EXPERTISE,
                SkillType.COMMON_SENSE,
            },
            CapabilityCategory.CREATIVITY: {
                SkillType.CREATIVE_WRITING,
                SkillType.BRAINSTORMING,
                SkillType.HUMOR,
                SkillType.STORYTELLING,
            },
            CapabilityCategory.CODING: {
                SkillType.CODE_GENERATION,
                SkillType.CODE_UNDERSTANDING,
                SkillType.DEBUGGING,
                SkillType.CODE_TRANSLATION,
            },
            CapabilityCategory.MATH: {
                SkillType.ARITHMETIC,
                SkillType.ALGEBRA,
                SkillType.WORD_PROBLEMS,
                SkillType.SYMBOLIC_MANIPULATION,
            },
            CapabilityCategory.ANALYSIS: {
                SkillType.DATA_ANALYSIS,
                SkillType.TEXT_ANALYSIS,
                SkillType.PATTERN_RECOGNITION,
                SkillType.CRITICAL_EVALUATION,
            },
            CapabilityCategory.INSTRUCTION_FOLLOWING: {
                SkillType.TASK_DECOMPOSITION,
                SkillType.FORMAT_ADHERENCE,
                SkillType.CONSTRAINT_HANDLING,
                SkillType.MULTI_STEP_EXECUTION,
            },
        }
        return skill in category_skills.get(category, set())


class FingerprintGenerator:
    """Generate model fingerprints."""

    def __init__(
        self,
        profiler: Optional[CapabilityProfiler] = None,
        limitation_detector: Optional[LimitationDetector] = None,
    ):
        """Initialize generator.

        Args:
            profiler: Optional custom capability profiler
            limitation_detector: Optional custom limitation detector
        """
        self._profiler = profiler or CapabilityProfiler()
        self._detector = limitation_detector or LimitationDetector()

    def generate(
        self,
        model_id: str,
        skill_results: dict[SkillType, list[tuple[str, Any]]],
        responses: list[str],
        version: str = "1.0",
        metadata: Optional[dict[str, Any]] = None,
    ) -> ModelFingerprint:
        """Generate fingerprint for a model.

        Args:
            model_id: Model identifier
            skill_results: Test results by skill
            responses: Raw responses for limitation detection
            version: Fingerprint version
            metadata: Optional metadata

        Returns:
            ModelFingerprint object
        """
        # Build skill profiles for each category
        skill_profiles = {}
        category_scores = {}

        for category in CapabilityCategory:
            profile = self._profiler.profile_category(category, skill_results)
            skill_profiles[category] = profile
            category_scores[category] = profile.overall_score

        # Detect limitations
        limitations = self._detector.detect(responses)

        # Identify top capabilities
        all_skills = []
        for profile in skill_profiles.values():
            for skill, score in profile.skills.items():
                all_skills.append((skill, score.score))
        top_capabilities = sorted(all_skills, key=lambda x: x[1], reverse=True)[:5]

        # Identify main limitations
        main_limitations = [
            limitation.limitation_type for limitation in limitations if limitation.severity > 0.5
        ]

        # Calculate overall score
        if category_scores:
            overall_score = sum(category_scores.values()) / len(category_scores)
        else:
            overall_score = 0.0

        # Generate fingerprint hash
        fingerprint_hash = self._generate_hash(model_id, category_scores, top_capabilities)

        return ModelFingerprint(
            model_id=model_id,
            version=version,
            fingerprint_hash=fingerprint_hash,
            category_scores=category_scores,
            skill_profiles=skill_profiles,
            limitations=limitations,
            top_capabilities=top_capabilities,
            main_limitations=main_limitations,
            overall_capability_score=overall_score,
            metadata=metadata or {},
        )

    @staticmethod
    def _generate_hash(
        model_id: str,
        category_scores: dict[CapabilityCategory, float],
        top_capabilities: list[tuple[SkillType, float]],
    ) -> str:
        """Generate unique hash for fingerprint."""
        data = {
            "model_id": model_id,
            "scores": {k.value: round(v, 2) for k, v in category_scores.items()},
            "top": [(s.value, round(sc, 2)) for s, sc in top_capabilities],
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]


class FingerprintComparator:
    """Compare model fingerprints."""

    def compare(
        self,
        fingerprint_a: ModelFingerprint,
        fingerprint_b: ModelFingerprint,
    ) -> FingerprintComparison:
        """Compare two fingerprints.

        Args:
            fingerprint_a: First fingerprint
            fingerprint_b: Second fingerprint

        Returns:
            FingerprintComparison object
        """
        # Calculate category differences
        category_diffs = {}
        for category in CapabilityCategory:
            score_a = fingerprint_a.category_scores.get(category, 0)
            score_b = fingerprint_b.category_scores.get(category, 0)
            category_diffs[category] = score_a - score_b

        # Calculate skill differences
        skill_diffs = {}
        all_skills: set[SkillType] = set()
        for profile in fingerprint_a.skill_profiles.values():
            all_skills.update(profile.skills.keys())
        for profile in fingerprint_b.skill_profiles.values():
            all_skills.update(profile.skills.keys())

        for skill in all_skills:
            score_a = self._get_skill_score(fingerprint_a, skill)
            score_b = self._get_skill_score(fingerprint_b, skill)
            skill_diffs[skill] = score_a - score_b

        # Identify advantages
        threshold = 0.1
        model_a_advantages = [s for s, d in skill_diffs.items() if d > threshold]
        model_b_advantages = [s for s, d in skill_diffs.items() if d < -threshold]

        # Identify shared traits
        shared_strengths = []
        shared_weaknesses = []
        for skill, diff in skill_diffs.items():
            if abs(diff) < threshold:
                score_a = self._get_skill_score(fingerprint_a, skill)
                if score_a >= 0.7:
                    shared_strengths.append(skill)
                elif score_a < 0.4:
                    shared_weaknesses.append(skill)

        # Calculate overall similarity
        if category_diffs:
            total_diff = sum(abs(d) for d in category_diffs.values())
            similarity = max(0, 1 - total_diff / len(category_diffs))
        else:
            similarity = 1.0

        return FingerprintComparison(
            model_a_id=fingerprint_a.model_id,
            model_b_id=fingerprint_b.model_id,
            similarity_score=similarity,
            category_differences=category_diffs,
            skill_differences=skill_diffs,
            model_a_advantages=model_a_advantages,
            model_b_advantages=model_b_advantages,
            shared_strengths=shared_strengths,
            shared_weaknesses=shared_weaknesses,
        )

    @staticmethod
    def _get_skill_score(fingerprint: ModelFingerprint, skill: SkillType) -> float:
        """Get score for a skill from fingerprint."""
        for profile in fingerprint.skill_profiles.values():
            if skill in profile.skills:
                return profile.skills[skill].score
        return 0.0


# Convenience functions


def create_fingerprint(
    model_id: str,
    skill_results: dict[SkillType, list[tuple[str, Any]]],
    responses: list[str],
    version: str = "1.0",
) -> ModelFingerprint:
    """Create a model fingerprint.

    Args:
        model_id: Model identifier
        skill_results: Test results by skill
        responses: Raw responses for limitation detection
        version: Fingerprint version

    Returns:
        ModelFingerprint object
    """
    generator = FingerprintGenerator()
    return generator.generate(model_id, skill_results, responses, version)


def compare_fingerprints(
    fingerprint_a: ModelFingerprint,
    fingerprint_b: ModelFingerprint,
) -> FingerprintComparison:
    """Compare two model fingerprints.

    Args:
        fingerprint_a: First fingerprint
        fingerprint_b: Second fingerprint

    Returns:
        FingerprintComparison object
    """
    comparator = FingerprintComparator()
    return comparator.compare(fingerprint_a, fingerprint_b)


def quick_capability_assessment(
    responses: dict[str, tuple[str, Any]],
) -> dict[str, Any]:
    """Quick capability assessment from responses.

    Args:
        responses: Dictionary of skill_name -> (response, expected)

    Returns:
        Dictionary with quick assessment results
    """
    assessor = SkillAssessor()
    results = {}

    for skill_name, (response, expected) in responses.items():
        # Try to match skill name to SkillType
        try:
            skill = SkillType(skill_name)
        except ValueError:
            # Use first matching skill
            skill = SkillType.COMMON_SENSE  # Default

        score = assessor.assess_skill(skill, [(response, expected)])
        results[skill_name] = {
            "score": score.score,
            "level": score.level.value,
        }

    # Calculate overall
    overall = sum(r["score"] for r in results.values()) / len(results) if results else 0.0

    return {
        "skills": results,
        "overall_score": overall,
        "n_skills_tested": len(results),
    }


def detect_limitations(
    responses: list[str],
) -> list[dict[str, Any]]:
    """Detect limitations from responses.

    Args:
        responses: List of model responses

    Returns:
        List of limitation dictionaries
    """
    detector = LimitationDetector()
    reports = detector.detect(responses)
    return [r.to_dict() for r in reports]
