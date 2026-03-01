"""Prompt sensitivity analysis for LLM evaluation.

This module provides a comprehensive toolkit for systematically analyzing how
small changes to prompts affect model outputs. Understanding prompt sensitivity
is crucial for building robust LLM applications, as models can exhibit unexpected
behavior changes from minor input variations.

Overview
--------
The module offers several complementary approaches to sensitivity analysis:

1. **Perturbation-based Testing**: Systematically modify prompts using various
   perturbation strategies (typos, case changes, synonyms, etc.) and measure
   output stability.

2. **Semantic Equivalence Testing**: Determine whether semantically equivalent
   prompts produce semantically equivalent outputs.

3. **Format Sensitivity Analysis**: Test how format instructions (JSON, markdown,
   bullet points) affect output structure and content.

4. **Instruction Style Profiling**: Analyze sensitivity to different instruction
   styles (imperative vs. interrogative, formal vs. casual).

5. **Comparative Analysis**: Compare sensitivity profiles across multiple prompts
   or models to identify patterns and outliers.

Key Components
--------------
PerturbationType : Enum
    Categories of prompt perturbations (case, whitespace, typos, etc.).
SensitivityLevel : Enum
    Qualitative sensitivity levels (very_low to very_high).
OutputChangeType : Enum
    Categories of output changes (no_change, minor_variation, contradictory, etc.).
PromptPerturbator : class
    Generates systematic perturbations of input prompts.
OutputComparator : class
    Compares and classifies differences between model outputs.
InputSensitivityAnalyzer : class
    Main analyzer for prompt sensitivity profiling.
ComparativeSensitivityAnalyzer : class
    Compares sensitivity across prompts or models.
FormatSensitivityTester : class
    Tests sensitivity to output format instructions.

Examples
--------
Quick sensitivity check for a single prompt:

>>> from insideLLMs.evaluation.sensitivity import quick_sensitivity_check
>>> def mock_llm(prompt: str) -> str:
...     # Your actual LLM call here
...     return f"Response to: {prompt}"
>>> result = quick_sensitivity_check("Explain photosynthesis", mock_llm)
>>> print(result["overall_sensitivity"])
'low'
>>> print(result["is_robust"])
True

Full sensitivity analysis with all perturbation types:

>>> from insideLLMs.evaluation.sensitivity import analyze_prompt_sensitivity
>>> profile = analyze_prompt_sensitivity(
...     prompt="List three benefits of exercise",
...     get_response=mock_llm,
... )
>>> print(f"Overall: {profile.overall_sensitivity.value}")
Overall: moderate
>>> print(f"Most sensitive to: {[p.value for p in profile.most_sensitive_to]}")
Most sensitive to: ['typo', 'case_change']

Generate perturbations without running analysis:

>>> from insideLLMs.evaluation.sensitivity import generate_perturbations, PerturbationType
>>> perturbations = generate_perturbations(
...     prompt="Write a haiku about coding",
...     perturbation_types=[PerturbationType.TYPO, PerturbationType.CASE_CHANGE],
...     n_variations=2,
... )
>>> for p in perturbations:
...     print(f"{p.perturbation_type.value}: '{p.perturbed}'")
typo: 'Wirte a haiku about coding'
typo: 'Write a haiku abuot coding'
case_change: 'write a haiku about coding'
case_change: 'WRITE A HAIKU ABOUT CODING'

Compare sensitivity across multiple prompts:

>>> from insideLLMs.evaluation.sensitivity import compare_prompt_sensitivity
>>> prompts = [
...     "Explain quantum computing",
...     "What is quantum computing?",
...     "Quantum computing explanation please",
... ]
>>> comparison = compare_prompt_sensitivity(prompts, mock_llm)
>>> print(f"Most robust prompt: {comparison.most_robust}")
Most robust prompt: What is quantum computing?

Test format instruction sensitivity:

>>> from insideLLMs.evaluation.sensitivity import check_format_sensitivity
>>> format_results = check_format_sensitivity(
...     prompt="List three programming languages",
...     get_response=mock_llm,
... )
>>> print(f"Format adherence rate: {format_results['format_adherence_rate']:.1%}")
Format adherence rate: 85.7%

Notes
-----
- Sensitivity analysis requires making multiple LLM API calls. For a full
  analysis with all perturbation types and 2 variations each, expect ~18
  API calls per prompt.

- The module uses simple heuristics for semantic similarity by default.
  For production use, consider providing a custom similarity function
  using embeddings.

- Reproducibility can be ensured by setting a random seed on the
  PromptPerturbator class.

- The robustness threshold (default 0.7) determines whether outputs are
  considered "similar enough" to be robust. Adjust based on your use case.

See Also
--------
insideLLMs.nlp.similarity : Text similarity functions used by this module.
insideLLMs.evaluation : Broader evaluation framework for LLMs.

References
----------
.. [1] Ribeiro, M.T. et al. "Beyond Accuracy: Behavioral Testing of NLP Models
       with CheckList." ACL 2020.
.. [2] Gao, L. et al. "The Pile: An 800GB Dataset of Diverse Text for Language
       Modeling." arXiv:2101.00027.
"""

from __future__ import annotations

import random
import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from insideLLMs.nlp.similarity import word_overlap_similarity


class PerturbationType(Enum):
    """Types of prompt perturbations for sensitivity testing.

    This enumeration defines the different categories of modifications
    that can be applied to prompts during sensitivity analysis. Each
    perturbation type represents a distinct way that real-world input
    variations might occur.

    Attributes
    ----------
    CASE_CHANGE : str
        Letter casing modifications (lowercase, uppercase, title case).
        Tests whether the model treats "EXPLAIN" differently from "explain".
    WHITESPACE : str
        Whitespace modifications (extra spaces, trailing newlines, trimming).
        Tests robustness to formatting inconsistencies.
    PUNCTUATION : str
        Punctuation changes (adding/removing periods, question marks).
        Tests whether "What is AI" behaves like "What is AI?".
    SYNONYM : str
        Word replacement with synonyms ("explain" -> "describe").
        Tests semantic understanding vs. keyword matching.
    PARAPHRASE : str
        Structural rephrasing while preserving meaning.
        Tests whether "Please explain X" equals "Explain X".
    WORD_ORDER : str
        Reordering words within the prompt.
        Tests sensitivity to word position.
    TYPO : str
        Introduction of typographical errors.
        Tests robustness to common user input errors.
    FORMATTING : str
        Structural formatting changes (quotes, markdown, list markers).
        Tests sensitivity to prompt structure.
    INSTRUCTION_STYLE : str
        Instruction style changes (imperative vs. interrogative).
        Tests "Explain X" vs. "Can you explain X?".

    Examples
    --------
    Iterate over all perturbation types:

    >>> for ptype in PerturbationType:
    ...     print(f"{ptype.name}: {ptype.value}")
    CASE_CHANGE: case_change
    WHITESPACE: whitespace
    PUNCTUATION: punctuation
    ...

    Use specific perturbation types for targeted testing:

    >>> from insideLLMs.evaluation.sensitivity import generate_perturbations
    >>> perturbations = generate_perturbations(
    ...     "Write code",
    ...     perturbation_types=[PerturbationType.TYPO, PerturbationType.CASE_CHANGE],
    ... )

    See Also
    --------
    PromptPerturbator : Class that applies these perturbation types.
    """

    CASE_CHANGE = "case_change"
    WHITESPACE = "whitespace"
    PUNCTUATION = "punctuation"
    SYNONYM = "synonym"
    PARAPHRASE = "paraphrase"
    WORD_ORDER = "word_order"
    TYPO = "typo"
    FORMATTING = "formatting"
    INSTRUCTION_STYLE = "instruction_style"


class SensitivityLevel(Enum):
    """Qualitative levels of prompt sensitivity.

    This enumeration provides human-readable classifications for
    sensitivity scores. It helps interpret numerical sensitivity
    values in terms of actionable categories.

    Attributes
    ----------
    VERY_LOW : str
        Score < 0.2. The prompt is highly robust; minor perturbations
        rarely affect outputs.
    LOW : str
        Score 0.2-0.4. The prompt shows minor sensitivity but remains
        generally stable.
    MODERATE : str
        Score 0.4-0.6. The prompt shows notable sensitivity; some
        perturbations cause meaningful output changes.
    HIGH : str
        Score 0.6-0.8. The prompt is sensitive; many perturbations
        cause significant output differences.
    VERY_HIGH : str
        Score >= 0.8. The prompt is highly unstable; most perturbations
        dramatically change outputs.

    Examples
    --------
    Interpret a sensitivity profile:

    >>> from insideLLMs.evaluation.sensitivity import analyze_prompt_sensitivity
    >>> profile = analyze_prompt_sensitivity(prompt, get_response)
    >>> if profile.overall_sensitivity == SensitivityLevel.VERY_HIGH:
    ...     print("Warning: This prompt is unstable!")
    ...     print("Consider:", profile.recommendations)

    Map scores to levels programmatically:

    >>> def score_to_level(score: float) -> SensitivityLevel:
    ...     if score >= 0.8:
    ...         return SensitivityLevel.VERY_HIGH
    ...     elif score >= 0.6:
    ...         return SensitivityLevel.HIGH
    ...     elif score >= 0.4:
    ...         return SensitivityLevel.MODERATE
    ...     elif score >= 0.2:
    ...         return SensitivityLevel.LOW
    ...     return SensitivityLevel.VERY_LOW

    See Also
    --------
    SensitivityProfile : Contains the overall_sensitivity field.
    InputSensitivityAnalyzer : Computes sensitivity levels.
    """

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class OutputChangeType(Enum):
    """Classification of output changes between original and perturbed prompts.

    This enumeration categorizes the types of differences observed between
    model outputs when comparing original and perturbed prompt responses.
    The classification helps distinguish between acceptable variations and
    problematic sensitivity.

    Attributes
    ----------
    NO_CHANGE : str
        Outputs are identical or nearly identical (similarity > 0.95).
        The model is perfectly robust to this perturbation.
    MINOR_VARIATION : str
        Small differences in wording but same content (similarity 0.8-0.95).
        Acceptable variation for most applications.
    SEMANTIC_EQUIVALENT : str
        Different wording but semantically equivalent meaning.
        The model understood the input correctly despite perturbation.
    DIFFERENT_FORMAT : str
        Same information presented in different structure/format.
        May indicate format instruction sensitivity.
    DIFFERENT_CONTENT : str
        Substantially different information or answers.
        Indicates problematic sensitivity to the perturbation.
    CONTRADICTORY : str
        Outputs contradict each other (semantic similarity < 0.2).
        Critical issue requiring prompt engineering attention.
    FAILURE : str
        One output indicates an error or refusal to respond.
        May indicate prompt robustness issues.

    Examples
    --------
    Check for problematic output changes:

    >>> comparison = comparator.compare(original_output, perturbed_output)
    >>> if comparison.change_type in [
    ...     OutputChangeType.CONTRADICTORY,
    ...     OutputChangeType.DIFFERENT_CONTENT,
    ... ]:
    ...     print("Warning: Perturbation caused significant output change!")
    ...     print(f"Similarity: {comparison.similarity_score:.2f}")

    Classify acceptable vs. unacceptable changes:

    >>> ACCEPTABLE = {
    ...     OutputChangeType.NO_CHANGE,
    ...     OutputChangeType.MINOR_VARIATION,
    ...     OutputChangeType.SEMANTIC_EQUIVALENT,
    ... }
    >>> is_robust = comparison.change_type in ACCEPTABLE

    See Also
    --------
    OutputComparison : Contains the change_type classification.
    OutputComparator : Determines change types from output pairs.
    """

    NO_CHANGE = "no_change"
    MINOR_VARIATION = "minor_variation"
    SEMANTIC_EQUIVALENT = "semantic_equivalent"
    DIFFERENT_FORMAT = "different_format"
    DIFFERENT_CONTENT = "different_content"
    CONTRADICTORY = "contradictory"
    FAILURE = "failure"


@dataclass
class Perturbation:
    """A single prompt perturbation with metadata.

    This dataclass encapsulates a perturbation applied to a prompt,
    including both the original and modified text along with metadata
    about the type and magnitude of the change.

    Parameters
    ----------
    original : str
        The original prompt text before perturbation.
    perturbed : str
        The modified prompt text after perturbation.
    perturbation_type : PerturbationType
        The category of perturbation applied.
    change_description : str
        Human-readable description of the change.
    change_magnitude : float
        Quantified magnitude of change, ranging from 0.0 (no change)
        to 1.0 (completely different).

    Attributes
    ----------
    original : str
        The original prompt text.
    perturbed : str
        The perturbed prompt text.
    perturbation_type : PerturbationType
        Type of perturbation applied.
    change_description : str
        Description of the perturbation.
    change_magnitude : float
        Magnitude of the change (0.0-1.0).

    Examples
    --------
    Create a perturbation manually:

    >>> p = Perturbation(
    ...     original="Explain machine learning",
    ...     perturbed="explain machine learning",
    ...     perturbation_type=PerturbationType.CASE_CHANGE,
    ...     change_description="Changed letter casing",
    ...     change_magnitude=0.05,
    ... )
    >>> print(f"Changed '{p.original}' to '{p.perturbed}'")
    Changed 'Explain machine learning' to 'explain machine learning'

    Generate perturbations using PromptPerturbator:

    >>> perturbator = PromptPerturbator(seed=42)
    >>> perturbations = perturbator.perturb(
    ...     "Write a poem",
    ...     perturbation_types=[PerturbationType.TYPO],
    ...     n_variations=2,
    ... )
    >>> for p in perturbations:
    ...     print(f"Magnitude {p.change_magnitude:.2f}: {p.perturbed}")
    Magnitude 0.08: Wirte a poem
    Magnitude 0.08: Write a pome

    Convert to dictionary for serialization:

    >>> p_dict = p.to_dict()
    >>> print(p_dict["type"])
    'case_change'

    See Also
    --------
    PromptPerturbator : Generates Perturbation objects.
    PerturbationSensitivityResult : Uses Perturbation in analysis results.
    """

    original: str
    perturbed: str
    perturbation_type: PerturbationType
    change_description: str
    change_magnitude: float  # 0-1, how much changed

    def to_dict(self) -> dict[str, Any]:
        """Convert perturbation to a dictionary representation.

        Returns a dictionary suitable for JSON serialization or logging,
        with the perturbation type converted to its string value.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - original: The original prompt text
            - perturbed: The perturbed prompt text
            - type: String value of the perturbation type
            - description: Human-readable change description
            - magnitude: Float magnitude of change

        Examples
        --------
        >>> p = Perturbation(
        ...     original="Hello",
        ...     perturbed="HELLO",
        ...     perturbation_type=PerturbationType.CASE_CHANGE,
        ...     change_description="Changed letter casing",
        ...     change_magnitude=0.5,
        ... )
        >>> d = p.to_dict()
        >>> print(d["type"])
        'case_change'
        >>> print(d["magnitude"])
        0.5
        """
        return {
            "original": self.original,
            "perturbed": self.perturbed,
            "type": self.perturbation_type.value,
            "description": self.change_description,
            "magnitude": self.change_magnitude,
        }


@dataclass
class OutputComparison:
    """Detailed comparison between original and perturbed prompt outputs.

    This dataclass holds the results of comparing two model outputs,
    including multiple similarity metrics, change classification,
    and identified differences.

    Parameters
    ----------
    original_output : str
        The model's response to the original (unperturbed) prompt.
    perturbed_output : str
        The model's response to the perturbed prompt.
    change_type : OutputChangeType
        Classification of the change between outputs.
    similarity_score : float
        Word-level overlap similarity score (0.0-1.0).
    semantic_similarity : float
        Estimated semantic similarity using n-gram overlap (0.0-1.0).
    length_ratio : float
        Ratio of shorter to longer output length.
    key_differences : list[str]
        Human-readable descriptions of key differences found.

    Attributes
    ----------
    original_output : str
        Response to original prompt.
    perturbed_output : str
        Response to perturbed prompt.
    change_type : OutputChangeType
        Type of change observed.
    similarity_score : float
        Lexical similarity (0.0-1.0).
    semantic_similarity : float
        Semantic similarity estimate (0.0-1.0).
    length_ratio : float
        Length ratio (0.0-1.0, where 1.0 means same length).
    key_differences : list[str]
        List of identified differences.

    Examples
    --------
    Compare two outputs manually:

    >>> comparator = OutputComparator()
    >>> comparison = comparator.compare(
    ...     original="Python is a programming language.",
    ...     perturbed="Python is a popular programming language.",
    ... )
    >>> print(f"Similarity: {comparison.similarity_score:.2f}")
    Similarity: 0.83
    >>> print(f"Change type: {comparison.change_type.value}")
    Change type: minor_variation

    Check for problematic changes:

    >>> if comparison.change_type == OutputChangeType.CONTRADICTORY:
    ...     print("CRITICAL: Outputs contradict each other!")
    ...     for diff in comparison.key_differences:
    ...         print(f"  - {diff}")

    Analyze length variations:

    >>> if comparison.length_ratio < 0.5:
    ...     print("Warning: Output length changed significantly")
    ...     print(f"Length ratio: {comparison.length_ratio:.2f}")

    See Also
    --------
    OutputComparator : Creates OutputComparison objects.
    OutputChangeType : Possible change classifications.
    """

    original_output: str
    perturbed_output: str
    change_type: OutputChangeType
    similarity_score: float  # 0-1
    semantic_similarity: float  # 0-1
    length_ratio: float
    key_differences: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert comparison to a dictionary representation.

        Returns a dictionary suitable for JSON serialization, with
        outputs truncated to 200 characters for readability.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - original_output: Truncated original output (max 200 chars)
            - perturbed_output: Truncated perturbed output (max 200 chars)
            - change_type: String value of the change type
            - similarity_score: Lexical similarity score
            - semantic_similarity: Semantic similarity estimate
            - length_ratio: Output length ratio
            - key_differences: List of difference descriptions

        Examples
        --------
        >>> comparison = OutputComparison(
        ...     original_output="Short answer.",
        ...     perturbed_output="A much longer and detailed answer...",
        ...     change_type=OutputChangeType.DIFFERENT_FORMAT,
        ...     similarity_score=0.45,
        ...     semantic_similarity=0.72,
        ...     length_ratio=0.35,
        ...     key_differences=["Length difference: +150 characters"],
        ... )
        >>> d = comparison.to_dict()
        >>> print(d["change_type"])
        'different_format'
        >>> print(d["length_ratio"])
        0.35
        """
        return {
            "original_output": self.original_output[:200],
            "perturbed_output": self.perturbed_output[:200],
            "change_type": self.change_type.value,
            "similarity_score": self.similarity_score,
            "semantic_similarity": self.semantic_similarity,
            "length_ratio": self.length_ratio,
            "key_differences": self.key_differences,
        }


@dataclass
class PerturbationSensitivityResult:
    """Result of sensitivity analysis for a single perturbation test.

    This dataclass combines perturbation information with the resulting
    output comparison to provide a complete picture of how one specific
    perturbation affected the model's response.

    Parameters
    ----------
    perturbation : Perturbation
        The perturbation that was applied to the prompt.
    output_comparison : OutputComparison
        Comparison between original and perturbed outputs.
    sensitivity_score : float
        Sensitivity score (0.0-1.0), where higher values indicate
        greater sensitivity (less robustness).
    is_robust : bool
        Whether the model was robust to this perturbation, based on
        the robustness threshold (default: similarity >= 0.7).
    notes : list[str], optional
        Additional observations or warnings about the result.

    Attributes
    ----------
    perturbation : Perturbation
        The applied perturbation.
    output_comparison : OutputComparison
        Output comparison results.
    sensitivity_score : float
        Computed sensitivity (0.0-1.0).
    is_robust : bool
        Robustness determination.
    notes : list[str]
        Observations and warnings.

    Examples
    --------
    Analyze a single perturbation result:

    >>> result = analyzer.analyze(prompt, get_response).results[0]
    >>> print(f"Perturbation: {result.perturbation.perturbation_type.value}")
    Perturbation: typo
    >>> print(f"Sensitivity: {result.sensitivity_score:.2f}")
    Sensitivity: 0.15
    >>> print(f"Is robust: {result.is_robust}")
    Is robust: True

    Check for warnings:

    >>> for result in profile.results:
    ...     if result.notes:
    ...         print(f"[{result.perturbation.perturbation_type.value}]")
    ...         for note in result.notes:
    ...             print(f"  Warning: {note}")
    [case_change]
      Warning: Small input change caused large output change

    Filter for non-robust results:

    >>> non_robust = [r for r in profile.results if not r.is_robust]
    >>> print(f"Found {len(non_robust)} non-robust perturbations")
    Found 3 non-robust perturbations
    >>> for r in non_robust:
    ...     print(f"  - {r.perturbation.perturbation_type.value}: "
    ...           f"sensitivity={r.sensitivity_score:.2f}")

    See Also
    --------
    SensitivityProfile : Contains a list of these results.
    InputSensitivityAnalyzer : Generates these results.
    """

    perturbation: Perturbation
    output_comparison: OutputComparison
    sensitivity_score: float  # 0-1, higher = more sensitive
    is_robust: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a dictionary representation.

        Creates a nested dictionary structure suitable for JSON
        serialization or logging.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - perturbation: Nested perturbation dictionary
            - output_comparison: Nested comparison dictionary
            - sensitivity_score: Float sensitivity score
            - is_robust: Boolean robustness flag
            - notes: List of note strings

        Examples
        --------
        >>> result = profile.results[0]
        >>> d = result.to_dict()
        >>> print(d["sensitivity_score"])
        0.15
        >>> print(d["perturbation"]["type"])
        'typo'
        >>> import json
        >>> json_str = json.dumps(d, indent=2)
        """
        return {
            "perturbation": self.perturbation.to_dict(),
            "output_comparison": self.output_comparison.to_dict(),
            "sensitivity_score": self.sensitivity_score,
            "is_robust": self.is_robust,
            "notes": self.notes,
        }


@dataclass
class SensitivityProfile:
    """Complete sensitivity profile for a prompt across all perturbation types.

    This dataclass aggregates results from multiple perturbation tests into
    a comprehensive profile, including overall metrics, per-type breakdowns,
    and actionable recommendations.

    Parameters
    ----------
    prompt : str
        The original prompt that was analyzed.
    results : list[PerturbationSensitivityResult]
        Individual results for each perturbation test.
    overall_sensitivity : SensitivityLevel
        Qualitative overall sensitivity level.
    overall_score : float
        Average sensitivity score across all tests (0.0-1.0).
    by_perturbation_type : dict[PerturbationType, float]
        Average sensitivity score for each perturbation type.
    most_sensitive_to : list[PerturbationType]
        Perturbation types with highest sensitivity (score > 0.3).
    most_robust_to : list[PerturbationType]
        Perturbation types with lowest sensitivity (score < 0.2).
    recommendations : list[str]
        Actionable recommendations based on the analysis.

    Attributes
    ----------
    prompt : str
        The analyzed prompt.
    results : list[PerturbationSensitivityResult]
        All individual test results.
    overall_sensitivity : SensitivityLevel
        Overall sensitivity classification.
    overall_score : float
        Numeric overall sensitivity (0.0-1.0).
    by_perturbation_type : dict[PerturbationType, float]
        Per-type sensitivity scores.
    most_sensitive_to : list[PerturbationType]
        Types causing most sensitivity.
    most_robust_to : list[PerturbationType]
        Types causing least sensitivity.
    recommendations : list[str]
        Improvement suggestions.

    Examples
    --------
    Analyze a prompt and interpret the profile:

    >>> from insideLLMs.evaluation.sensitivity import analyze_prompt_sensitivity
    >>> profile = analyze_prompt_sensitivity(
    ...     prompt="Explain the water cycle",
    ...     get_response=my_llm_function,
    ... )
    >>> print(f"Overall: {profile.overall_sensitivity.value}")
    Overall: moderate
    >>> print(f"Score: {profile.overall_score:.2f}")
    Score: 0.42

    Examine per-type sensitivity:

    >>> for ptype, score in profile.by_perturbation_type.items():
    ...     status = "SENSITIVE" if score > 0.3 else "robust"
    ...     print(f"  {ptype.value}: {score:.2f} [{status}]")
    case_change: 0.12 [robust]
    typo: 0.65 [SENSITIVE]
    synonym: 0.38 [SENSITIVE]
    ...

    Act on recommendations:

    >>> print("Recommendations:")
    >>> for rec in profile.recommendations:
    ...     print(f"  - {rec}")
    Recommendations:
      - Model is typo-sensitive; consider input validation
      - Model is sensitive to word choice; use precise terminology

    Find the weakest areas:

    >>> print(f"Most sensitive to: {profile.most_sensitive_to}")
    Most sensitive to: [<PerturbationType.TYPO: 'typo'>]
    >>> print(f"Most robust to: {profile.most_robust_to}")
    Most robust to: [<PerturbationType.CASE_CHANGE: 'case_change'>]

    See Also
    --------
    InputSensitivityAnalyzer : Creates sensitivity profiles.
    analyze_prompt_sensitivity : Convenience function for analysis.
    """

    prompt: str
    results: list[PerturbationSensitivityResult]
    overall_sensitivity: SensitivityLevel
    overall_score: float
    by_perturbation_type: dict[PerturbationType, float]
    most_sensitive_to: list[PerturbationType]
    most_robust_to: list[PerturbationType]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to a dictionary representation.

        Creates a summary dictionary with the prompt truncated to 200
        characters and enum values converted to strings.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - prompt: Truncated prompt text (max 200 chars)
            - n_tests: Number of perturbation tests run
            - overall_sensitivity: String sensitivity level
            - overall_score: Numeric overall score
            - by_perturbation_type: Dict of type string to score
            - most_sensitive_to: List of type strings
            - most_robust_to: List of type strings
            - recommendations: List of recommendation strings

        Examples
        --------
        >>> d = profile.to_dict()
        >>> print(f"Ran {d['n_tests']} tests")
        Ran 18 tests
        >>> print(d["overall_sensitivity"])
        'moderate'
        >>> print(d["by_perturbation_type"]["typo"])
        0.65

        Serialize to JSON:

        >>> import json
        >>> with open("sensitivity_report.json", "w") as f:
        ...     json.dump(profile.to_dict(), f, indent=2)
        """
        return {
            "prompt": self.prompt[:200],
            "n_tests": len(self.results),
            "overall_sensitivity": self.overall_sensitivity.value,
            "overall_score": self.overall_score,
            "by_perturbation_type": {k.value: v for k, v in self.by_perturbation_type.items()},
            "most_sensitive_to": [p.value for p in self.most_sensitive_to],
            "most_robust_to": [p.value for p in self.most_robust_to],
            "recommendations": self.recommendations,
        }


@dataclass
class ComparativeSensitivity:
    """Comparative sensitivity analysis across multiple prompts or models.

    This dataclass aggregates multiple sensitivity profiles to identify
    patterns, rank items by robustness, and find common or divergent
    sensitivity characteristics.

    Parameters
    ----------
    profiles : list[SensitivityProfile]
        Individual sensitivity profiles being compared.
    ranking : list[tuple[str, float]]
        Items ranked by sensitivity score, from most robust (lowest)
        to most sensitive (highest). Each tuple is (identifier, score).
    most_robust : str
        Identifier of the most robust item (lowest sensitivity).
    most_sensitive : str
        Identifier of the most sensitive item (highest sensitivity).
    common_sensitivities : list[PerturbationType]
        Perturbation types that all items are sensitive to.
    divergent_sensitivities : list[PerturbationType]
        Perturbation types where items differ significantly.

    Attributes
    ----------
    profiles : list[SensitivityProfile]
        All compared profiles.
    ranking : list[tuple[str, float]]
        Ranked list by sensitivity.
    most_robust : str
        Most robust identifier.
    most_sensitive : str
        Most sensitive identifier.
    common_sensitivities : list[PerturbationType]
        Shared sensitivities.
    divergent_sensitivities : list[PerturbationType]
        Differing sensitivities.

    Examples
    --------
    Compare multiple prompt variations:

    >>> from insideLLMs.evaluation.sensitivity import compare_prompt_sensitivity
    >>> prompts = [
    ...     "Explain photosynthesis.",
    ...     "Can you explain photosynthesis?",
    ...     "EXPLAIN PHOTOSYNTHESIS",
    ... ]
    >>> comparison = compare_prompt_sensitivity(prompts, get_response)
    >>> print(f"Most robust: {comparison.most_robust}")
    Most robust: Can you explain photosynthesis?
    >>> print(f"Most sensitive: {comparison.most_sensitive}")
    Most sensitive: EXPLAIN PHOTOSYNTHESIS

    View the full ranking:

    >>> for identifier, score in comparison.ranking:
    ...     print(f"  {score:.2f}: {identifier}")
    0.18: Can you explain photosynthesis?
    0.32: Explain photosynthesis.
    0.55: EXPLAIN PHOTOSYNTHESIS

    Analyze common patterns:

    >>> if comparison.common_sensitivities:
    ...     print("All prompts are sensitive to:")
    ...     for ptype in comparison.common_sensitivities:
    ...         print(f"  - {ptype.value}")
    All prompts are sensitive to:
      - typo

    Identify divergent behaviors:

    >>> if comparison.divergent_sensitivities:
    ...     print("Prompts differ on:")
    ...     for ptype in comparison.divergent_sensitivities:
    ...         print(f"  - {ptype.value}")
    Prompts differ on:
      - case_change
      - instruction_style

    See Also
    --------
    ComparativeSensitivityAnalyzer : Creates comparison objects.
    compare_prompt_sensitivity : Convenience function for comparison.
    """

    profiles: list[SensitivityProfile]
    ranking: list[tuple[str, float]]  # (identifier, sensitivity_score)
    most_robust: str
    most_sensitive: str
    common_sensitivities: list[PerturbationType]
    divergent_sensitivities: list[PerturbationType]

    def to_dict(self) -> dict[str, Any]:
        """Convert comparison to a dictionary representation.

        Creates a summary dictionary suitable for JSON serialization
        or reporting.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - n_profiles: Number of profiles compared
            - ranking: List of (identifier, score) tuples
            - most_robust: Most robust identifier string
            - most_sensitive: Most sensitive identifier string
            - common_sensitivities: List of perturbation type strings
            - divergent_sensitivities: List of perturbation type strings

        Examples
        --------
        >>> d = comparison.to_dict()
        >>> print(f"Compared {d['n_profiles']} items")
        Compared 3 items
        >>> print(d["most_robust"])
        'Can you explain photosynthesis?'

        Generate a report:

        >>> import json
        >>> report = json.dumps(comparison.to_dict(), indent=2)
        >>> print(report)
        {
          "n_profiles": 3,
          "ranking": [...],
          "most_robust": "Can you explain photosynthesis?",
          ...
        }
        """
        return {
            "n_profiles": len(self.profiles),
            "ranking": self.ranking,
            "most_robust": self.most_robust,
            "most_sensitive": self.most_sensitive,
            "common_sensitivities": [p.value for p in self.common_sensitivities],
            "divergent_sensitivities": [p.value for p in self.divergent_sensitivities],
        }


class PromptPerturbator:
    """Generate systematic perturbations of prompts for sensitivity testing.

    This class provides methods to create controlled variations of input
    prompts across multiple perturbation types. It supports reproducible
    perturbation generation through optional seeding.

    The perturbator maintains an internal synonym dictionary for word
    replacement and uses various text manipulation strategies to simulate
    real-world input variations.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducible perturbation generation. If not
        provided, perturbations will be non-deterministic.

    Attributes
    ----------
    _rng : random.Random
        Internal random number generator instance.
    _synonyms : dict[str, list[str]]
        Dictionary mapping words to their synonyms for replacement.

    Examples
    --------
    Create a perturbator with reproducible output:

    >>> perturbator = PromptPerturbator(seed=42)
    >>> perturbations = perturbator.perturb("Explain machine learning")
    >>> # Running again with same seed produces identical results
    >>> perturbator2 = PromptPerturbator(seed=42)
    >>> perturbations2 = perturbator2.perturb("Explain machine learning")
    >>> perturbations[0].perturbed == perturbations2[0].perturbed
    True

    Generate specific perturbation types:

    >>> perturbator = PromptPerturbator()
    >>> typo_perturbations = perturbator.perturb(
    ...     "Write a summary",
    ...     perturbation_types=[PerturbationType.TYPO],
    ...     n_variations=3,
    ... )
    >>> for p in typo_perturbations:
    ...     print(f"'{p.perturbed}' (magnitude: {p.change_magnitude:.2f})")
    'Wirte a summary' (magnitude: 0.07)
    'Write a sumamry' (magnitude: 0.07)
    'Write a summray' (magnitude: 0.07)

    Generate all perturbation types:

    >>> all_perturbations = perturbator.perturb(
    ...     "List the planets",
    ...     n_variations=2,  # 2 per type
    ... )
    >>> print(f"Generated {len(all_perturbations)} perturbations")
    Generated 18 perturbations

    Use perturbations in sensitivity analysis:

    >>> from insideLLMs.evaluation.sensitivity import InputSensitivityAnalyzer
    >>> analyzer = InputSensitivityAnalyzer(
    ...     perturbator=PromptPerturbator(seed=123),
    ... )
    >>> profile = analyzer.analyze(prompt, get_response)

    See Also
    --------
    PerturbationType : Available perturbation types.
    Perturbation : Result objects from perturbation.
    generate_perturbations : Convenience function using this class.

    Notes
    -----
    The synonym dictionary is intentionally limited to common instruction
    words. For production use with domain-specific vocabulary, consider
    extending the dictionary or subclassing this class.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the prompt perturbator.

        Args
        ----
        seed : int, optional
            Random seed for reproducible perturbation generation.
            When set, calling perturb() with the same inputs will
            produce identical outputs.

        Examples
        --------
        Create with fixed seed for testing:

        >>> perturbator = PromptPerturbator(seed=42)

        Create without seed for production variety:

        >>> perturbator = PromptPerturbator()
        """
        self._rng = random.Random(seed)
        self._synonyms = {
            "explain": ["describe", "elaborate", "clarify"],
            "write": ["compose", "create", "generate"],
            "list": ["enumerate", "name", "identify"],
            "analyze": ["examine", "evaluate", "assess"],
            "summarize": ["condense", "recap", "outline"],
            "compare": ["contrast", "differentiate", "distinguish"],
            "define": ["describe", "explain", "specify"],
            "good": ["excellent", "great", "quality"],
            "bad": ["poor", "negative", "low-quality"],
            "important": ["significant", "crucial", "key"],
            "simple": ["basic", "straightforward", "easy"],
            "complex": ["complicated", "intricate", "sophisticated"],
        }

    def perturb(
        self,
        prompt: str,
        perturbation_types: Optional[list[PerturbationType]] = None,
        n_variations: int = 1,
    ) -> list[Perturbation]:
        """Generate perturbations of a prompt.

        Creates multiple variations of the input prompt by applying
        specified perturbation strategies. Each perturbation type
        generates `n_variations` independent variations.

        Args
        ----
        prompt : str
            The original prompt text to perturb.
        perturbation_types : list[PerturbationType], optional
            Types of perturbations to apply. If None, all available
            perturbation types will be used.
        n_variations : int, default=1
            Number of variations to generate per perturbation type.
            Higher values provide more comprehensive testing but
            increase computation.

        Returns
        -------
        list[Perturbation]
            List of Perturbation objects, each containing the original
            text, perturbed text, perturbation type, description, and
            magnitude. Perturbations that result in no change (identical
            to original) are filtered out.

        Examples
        --------
        Generate all perturbation types with default settings:

        >>> perturbator = PromptPerturbator(seed=42)
        >>> perturbations = perturbator.perturb("Explain gravity")
        >>> print(f"Generated {len(perturbations)} perturbations")
        Generated 9 perturbations

        Generate specific perturbation types:

        >>> perturbations = perturbator.perturb(
        ...     "Write a poem about the ocean",
        ...     perturbation_types=[
        ...         PerturbationType.CASE_CHANGE,
        ...         PerturbationType.TYPO,
        ...     ],
        ... )
        >>> for p in perturbations:
        ...     print(f"{p.perturbation_type.value}: {p.perturbed[:30]}...")

        Generate multiple variations per type:

        >>> perturbations = perturbator.perturb(
        ...     "Summarize the article",
        ...     perturbation_types=[PerturbationType.SYNONYM],
        ...     n_variations=5,
        ... )
        >>> synonyms_used = [p.perturbed for p in perturbations]
        >>> print(synonyms_used)
        ['Condense the article', 'Recap the article', ...]

        Notes
        -----
        - Some perturbation types may not produce changes for all prompts
          (e.g., SYNONYM requires words in the synonym dictionary).
        - The actual number of returned perturbations may be less than
          `len(perturbation_types) * n_variations` if some perturbations
          produce no change.
        """
        if perturbation_types is None:
            perturbation_types = list(PerturbationType)

        perturbations = []
        for ptype in perturbation_types:
            for _ in range(n_variations):
                perturbed = self._apply_perturbation(prompt, ptype)
                if perturbed and perturbed != prompt:
                    magnitude = self._calculate_magnitude(prompt, perturbed)
                    perturbations.append(
                        Perturbation(
                            original=prompt,
                            perturbed=perturbed,
                            perturbation_type=ptype,
                            change_description=self._describe_change(ptype),
                            change_magnitude=magnitude,
                        )
                    )

        return perturbations

    def _apply_perturbation(
        self,
        prompt: str,
        ptype: PerturbationType,
    ) -> Optional[str]:
        """Apply a specific perturbation type to a prompt.

        Internal dispatch method that routes to the appropriate
        perturbation implementation based on the type.

        Args
        ----
        prompt : str
            The prompt to perturb.
        ptype : PerturbationType
            The type of perturbation to apply.

        Returns
        -------
        str or None
            The perturbed prompt, or None if the perturbation type
            is not recognized.
        """
        methods = {
            PerturbationType.CASE_CHANGE: self._perturb_case,
            PerturbationType.WHITESPACE: self._perturb_whitespace,
            PerturbationType.PUNCTUATION: self._perturb_punctuation,
            PerturbationType.SYNONYM: self._perturb_synonym,
            PerturbationType.PARAPHRASE: self._perturb_paraphrase,
            PerturbationType.WORD_ORDER: self._perturb_word_order,
            PerturbationType.TYPO: self._perturb_typo,
            PerturbationType.FORMATTING: self._perturb_formatting,
            PerturbationType.INSTRUCTION_STYLE: self._perturb_instruction_style,
        }
        method = methods.get(ptype)
        if method:
            return method(prompt)
        return None

    def _perturb_case(self, prompt: str) -> str:
        """Change the letter casing of a prompt.

        Randomly selects from lowercase, uppercase, capitalize (first
        letter only), or title case transformations.

        Args
        ----
        prompt : str
            The prompt to transform.

        Returns
        -------
        str
            The prompt with modified casing. Returns original if
            all transformations produce identical text.

        Examples
        --------
        >>> perturbator._perturb_case("Hello World")
        'hello world'  # or 'HELLO WORLD' or 'Hello world'
        """
        options = [
            prompt.lower(),
            prompt.upper(),
            prompt.capitalize(),
            prompt.title(),
        ]
        return self._rng.choice([o for o in options if o != prompt] or [prompt])

    def _perturb_whitespace(self, prompt: str) -> str:
        """Modify whitespace in a prompt.

        Applies various whitespace transformations including trimming,
        adding padding, collapsing multiple spaces, or adding newlines.

        Args
        ----
        prompt : str
            The prompt to transform.

        Returns
        -------
        str
            The prompt with modified whitespace.

        Examples
        --------
        >>> perturbator._perturb_whitespace("Hello  World")
        'Hello World'  # collapsed double space
        >>> perturbator._perturb_whitespace("Hello")
        '  Hello  '  # or 'Hello\\n'
        """
        options = [
            prompt.strip(),
            "  " + prompt + "  ",
            prompt.replace("  ", " "),
            re.sub(r"\s+", " ", prompt),
            prompt + "\n",
        ]
        return self._rng.choice([o for o in options if o != prompt] or [prompt])

    def _perturb_punctuation(self, prompt: str) -> str:
        """Modify punctuation in a prompt.

        Randomly removes trailing punctuation, changes to question
        mark or period, or adds exclamation mark.

        Args
        ----
        prompt : str
            The prompt to transform.

        Returns
        -------
        str
            The prompt with modified punctuation.

        Examples
        --------
        >>> perturbator._perturb_punctuation("What is AI?")
        'What is AI'  # or 'What is AI.' or 'What is AI?!'
        >>> perturbator._perturb_punctuation("Explain this")
        'Explain this?'  # or 'Explain this.'
        """
        options = []

        # Remove trailing punctuation
        if prompt and prompt[-1] in ".!?":
            options.append(prompt[:-1])

        # Add question mark
        if prompt and prompt[-1] not in "?":
            options.append(prompt.rstrip(".!") + "?")

        # Add period
        if prompt and prompt[-1] not in ".":
            options.append(prompt.rstrip("!?") + ".")

        # Double punctuation
        if prompt:
            options.append(prompt + "!")

        return self._rng.choice(options) if options else prompt

    def _perturb_synonym(self, prompt: str) -> str:
        """Replace a word with a synonym from the internal dictionary.

        Finds the first word that has synonyms in the internal
        dictionary and replaces it, preserving original casing
        and trailing punctuation.

        Args
        ----
        prompt : str
            The prompt to transform.

        Returns
        -------
        str
            The prompt with one word replaced by a synonym, or
            the original prompt if no synonyms are found.

        Examples
        --------
        >>> perturbator._perturb_synonym("Explain machine learning")
        'Describe machine learning'
        >>> perturbator._perturb_synonym("Write a story")
        'Compose a story'
        """
        words = prompt.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip(string.punctuation)
            if word_lower in self._synonyms:
                synonyms = self._synonyms[word_lower]
                replacement = self._rng.choice(synonyms)
                # Preserve original case
                if word[0].isupper():
                    replacement = replacement.capitalize()
                # Preserve punctuation
                if word and word[-1] in string.punctuation:
                    replacement += word[-1]
                words[i] = replacement
                break  # Only replace one word
        return " ".join(words)

    def _perturb_paraphrase(self, prompt: str) -> str:
        """Simple paraphrasing through restructuring.

        Adds or removes "Please" at the beginning of the prompt
        to test sensitivity to politeness markers.

        Args
        ----
        prompt : str
            The prompt to transform.

        Returns
        -------
        str
            The paraphrased prompt.

        Examples
        --------
        >>> perturbator._perturb_paraphrase("Explain gravity")
        'Please explain gravity'
        >>> perturbator._perturb_paraphrase("Please write a poem")
        'Write a poem'
        """
        if not prompt:
            return prompt

        # Add "Please" if not present
        if not prompt.lower().startswith("please"):
            return "Please " + prompt[0].lower() + prompt[1:]

        # Remove "Please" if present
        if prompt.lower().startswith("please ") and len(prompt) > 7:
            rest = prompt[7:]
            if rest:
                return rest[0].upper() + rest[1:]

        return prompt

    def _perturb_word_order(self, prompt: str) -> str:
        """Reorder words in a prompt by swapping adjacent words.

        Randomly selects two adjacent words and swaps their positions.
        Only applies to prompts with more than 3 words.

        Args
        ----
        prompt : str
            The prompt to transform.

        Returns
        -------
        str
            The prompt with swapped word order, or original if
            prompt is too short.

        Examples
        --------
        >>> perturbator._perturb_word_order("Write a short story")
        'Write short a story'  # or 'a Write short story'
        """
        # Move last word to beginning for questions
        words = prompt.split()
        if len(words) > 3:
            # Swap adjacent words
            idx = self._rng.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
            return " ".join(words)
        return prompt

    def _perturb_typo(self, prompt: str) -> str:
        """Introduce a typographical error into the prompt.

        Randomly selects a word and applies one of three typo types:
        character swap, character deletion, or character doubling.

        Args
        ----
        prompt : str
            The prompt to transform.

        Returns
        -------
        str
            The prompt with a typo introduced, or original if
            prompt is too short or has no suitable words.

        Examples
        --------
        >>> perturbator._perturb_typo("Explain machine learning")
        'Expalin machine learning'  # swap
        >>> perturbator._perturb_typo("Write code")
        'Wrte code'  # delete
        >>> perturbator._perturb_typo("List items")
        'Lisst items'  # double
        """
        if len(prompt) < 5:
            return prompt

        words = prompt.split()
        if not words:
            return prompt

        # Find a word to modify
        word_idx = self._rng.randint(0, len(words) - 1)
        word = words[word_idx]

        if len(word) < 3:
            return prompt

        typo_types = ["swap", "delete", "double"]
        typo_type = self._rng.choice(typo_types)

        if typo_type == "swap" and len(word) > 2:
            # Swap two adjacent characters
            idx = self._rng.randint(1, len(word) - 2)
            word = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2 :]
        elif typo_type == "delete" and len(word) > 3:
            # Delete a character
            idx = self._rng.randint(1, len(word) - 2)
            word = word[:idx] + word[idx + 1 :]
        elif typo_type == "double":
            # Double a character
            idx = self._rng.randint(1, len(word) - 1)
            word = word[:idx] + word[idx] + word[idx:]

        words[word_idx] = word
        return " ".join(words)

    def _perturb_formatting(self, prompt: str) -> str:
        """Change the formatting structure of a prompt.

        Wraps the prompt in various formatting markers such as
        quotes, bold markers, list items, or instruction tags.

        Args
        ----
        prompt : str
            The prompt to transform.

        Returns
        -------
        str
            The prompt with formatting applied.

        Examples
        --------
        >>> perturbator._perturb_formatting("Explain AI")
        "'Explain AI'"  # or '**Explain AI**' or '- Explain AI'
        """
        options = [
            f"'{prompt}'",  # Add quotes
            f"**{prompt}**",  # Bold markers
            f"- {prompt}",  # List item
            f"1. {prompt}",  # Numbered
            f"[INSTRUCTION] {prompt}",  # Tagged
        ]
        return self._rng.choice(options)

    def _perturb_instruction_style(self, prompt: str) -> str:
        """Change the instruction style between imperative and interrogative.

        Converts imperative instructions ("Explain X") to questions
        ("Can you explain X?") and vice versa.

        Args
        ----
        prompt : str
            The prompt to transform.

        Returns
        -------
        str
            The prompt with changed instruction style, or original
            if transformation is not applicable.

        Examples
        --------
        >>> perturbator._perturb_instruction_style("Explain quantum physics")
        'Can you explain quantum physics?'
        >>> perturbator._perturb_instruction_style("Can you write a poem?")
        'Write a poem.'
        """
        # Convert imperative to question
        if prompt and not prompt.endswith("?"):
            words = prompt.lower().split()
            if words and words[0] in ["explain", "describe", "write", "list"]:
                verb = words[0]
                rest = " ".join(words[1:])
                return f"Can you {verb} {rest}?"

        # Convert question to imperative
        if prompt.lower().startswith("can you "):
            rest = prompt[8:].rstrip("?")
            return rest[0].upper() + rest[1:] + "."

        return prompt

    def _calculate_magnitude(self, original: str, perturbed: str) -> float:
        """Calculate the magnitude of change between two strings.

        Uses a simplified Levenshtein-like distance approximation
        based on character-level alignment.

        Args
        ----
        original : str
            The original text.
        perturbed : str
            The perturbed text.

        Returns
        -------
        float
            Change magnitude from 0.0 (identical) to 1.0 (completely
            different).

        Examples
        --------
        >>> perturbator._calculate_magnitude("hello", "hello")
        0.0
        >>> perturbator._calculate_magnitude("hello", "helo")
        0.2
        >>> perturbator._calculate_magnitude("abc", "xyz")
        1.0
        """
        if original == perturbed:
            return 0.0

        # Character-level difference
        max_len = max(len(original), len(perturbed))
        if max_len == 0:
            return 0.0

        # Levenshtein-like distance approximation
        common = sum(1 for a, b in zip(original, perturbed) if a == b)
        similarity = common / max_len

        return 1.0 - similarity

    @staticmethod
    def _describe_change(ptype: PerturbationType) -> str:
        """Get a human-readable description of a perturbation type.

        Args
        ----
        ptype : PerturbationType
            The perturbation type to describe.

        Returns
        -------
        str
            Human-readable description of the perturbation.

        Examples
        --------
        >>> PromptPerturbator._describe_change(PerturbationType.TYPO)
        'Introduced typo'
        >>> PromptPerturbator._describe_change(PerturbationType.CASE_CHANGE)
        'Changed letter casing'
        """
        descriptions = {
            PerturbationType.CASE_CHANGE: "Changed letter casing",
            PerturbationType.WHITESPACE: "Modified whitespace",
            PerturbationType.PUNCTUATION: "Changed punctuation",
            PerturbationType.SYNONYM: "Replaced word with synonym",
            PerturbationType.PARAPHRASE: "Paraphrased instruction",
            PerturbationType.WORD_ORDER: "Reordered words",
            PerturbationType.TYPO: "Introduced typo",
            PerturbationType.FORMATTING: "Changed formatting",
            PerturbationType.INSTRUCTION_STYLE: "Changed instruction style",
        }
        return descriptions.get(ptype, "Unknown perturbation")


class OutputComparator:
    """Compare and classify differences between model outputs.

    This class provides methods to compare two text outputs and determine
    the nature and magnitude of their differences. It computes multiple
    similarity metrics and classifies the type of change observed.

    Parameters
    ----------
    similarity_fn : Callable[[str, str], float], optional
        Custom function to compute text similarity. Should take two
        strings and return a float between 0.0 (completely different)
        and 1.0 (identical). Defaults to word_overlap_similarity.

    Attributes
    ----------
    _similarity_fn : Callable[[str, str], float]
        The similarity function used for lexical comparison.

    Examples
    --------
    Create a comparator with default similarity:

    >>> comparator = OutputComparator()
    >>> comparison = comparator.compare(
    ...     "Python is a programming language.",
    ...     "Python is a versatile programming language.",
    ... )
    >>> print(f"Similarity: {comparison.similarity_score:.2f}")
    Similarity: 0.83
    >>> print(f"Change type: {comparison.change_type.value}")
    Change type: minor_variation

    Use a custom similarity function:

    >>> from sklearn.metrics.pairwise import cosine_similarity
    >>> def embedding_similarity(a: str, b: str) -> float:
    ...     # Your embedding-based similarity here
    ...     return compute_cosine_sim(embed(a), embed(b))
    >>> comparator = OutputComparator(similarity_fn=embedding_similarity)

    Analyze differences in detail:

    >>> comparison = comparator.compare(original_output, perturbed_output)
    >>> print(f"Lexical similarity: {comparison.similarity_score:.2f}")
    >>> print(f"Semantic similarity: {comparison.semantic_similarity:.2f}")
    >>> print(f"Length ratio: {comparison.length_ratio:.2f}")
    >>> for diff in comparison.key_differences:
    ...     print(f"  - {diff}")

    See Also
    --------
    OutputComparison : Result object from comparison.
    OutputChangeType : Classification of output changes.
    word_overlap_similarity : Default similarity function.
    """

    def __init__(
        self,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """Initialize the output comparator.

        Args
        ----
        similarity_fn : Callable[[str, str], float], optional
            Custom function for computing text similarity. Should
            accept two strings and return a float from 0.0 to 1.0.
            If not provided, uses word_overlap_similarity.

        Examples
        --------
        Use default similarity:

        >>> comparator = OutputComparator()

        Use custom similarity function:

        >>> def jaccard_similarity(a: str, b: str) -> float:
        ...     words_a = set(a.lower().split())
        ...     words_b = set(b.lower().split())
        ...     intersection = len(words_a & words_b)
        ...     union = len(words_a | words_b)
        ...     return intersection / union if union else 1.0
        >>> comparator = OutputComparator(similarity_fn=jaccard_similarity)
        """
        self._similarity_fn = similarity_fn or word_overlap_similarity

    def compare(
        self,
        original: str,
        perturbed: str,
    ) -> OutputComparison:
        """Compare two model outputs and classify their differences.

        Computes multiple similarity metrics and identifies key
        differences between the original and perturbed outputs.

        Args
        ----
        original : str
            The model's response to the original prompt.
        perturbed : str
            The model's response to the perturbed prompt.

        Returns
        -------
        OutputComparison
            Comparison object containing similarity scores, change
            classification, and identified differences.

        Examples
        --------
        Compare nearly identical outputs:

        >>> comparator = OutputComparator()
        >>> comparison = comparator.compare(
        ...     "The sky is blue.",
        ...     "The sky is blue.",
        ... )
        >>> print(comparison.change_type.value)
        'no_change'
        >>> print(comparison.similarity_score)
        1.0

        Compare outputs with minor variations:

        >>> comparison = comparator.compare(
        ...     "Machine learning is a subset of AI.",
        ...     "Machine learning is a branch of AI.",
        ... )
        >>> print(comparison.change_type.value)
        'minor_variation'

        Compare contradictory outputs:

        >>> comparison = comparator.compare(
        ...     "The answer is yes.",
        ...     "The answer is definitely no.",
        ... )
        >>> print(comparison.change_type.value)
        'different_content'
        >>> print(comparison.key_differences)
        ['Original contains: yes', 'Perturbed contains: no, definitely']

        Notes
        -----
        The semantic similarity is estimated using bigram overlap,
        which is a simplified heuristic. For production use, consider
        providing a custom similarity function using embeddings.
        """
        similarity = self._similarity_fn(original, perturbed)
        semantic_sim = self._estimate_semantic_similarity(original, perturbed)

        # Calculate length ratio
        len_orig = len(original)
        len_pert = len(perturbed)
        length_ratio = (
            min(len_orig, len_pert) / max(len_orig, len_pert)
            if max(len_orig, len_pert) > 0
            else 1.0
        )

        # Determine change type
        change_type = self._classify_change(similarity, semantic_sim, length_ratio)

        # Find key differences
        differences = self._find_differences(original, perturbed)

        return OutputComparison(
            original_output=original,
            perturbed_output=perturbed,
            change_type=change_type,
            similarity_score=similarity,
            semantic_similarity=semantic_sim,
            length_ratio=length_ratio,
            key_differences=differences,
        )

    def _estimate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Estimate semantic similarity using bigram overlap.

        This is a simplified heuristic that uses Jaccard similarity
        of word bigrams as a proxy for semantic similarity.

        Args
        ----
        text1 : str
            First text to compare.
        text2 : str
            Second text to compare.

        Returns
        -------
        float
            Estimated semantic similarity from 0.0 to 1.0.

        Examples
        --------
        >>> comparator._estimate_semantic_similarity(
        ...     "The cat sat on the mat",
        ...     "The cat sat on the rug",
        ... )
        0.6  # High overlap in bigrams

        Notes
        -----
        This is a simplified approximation. For more accurate semantic
        similarity, use embedding-based methods.
        """

        # Use n-gram overlap as proxy
        def get_ngrams(text: str, n: int) -> set[str]:
            words = text.lower().split()
            return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

        bigrams1 = get_ngrams(text1, 2)
        bigrams2 = get_ngrams(text2, 2)

        if not bigrams1 and not bigrams2:
            return 1.0 if text1 == text2 else 0.5

        if not bigrams1 or not bigrams2:
            return 0.3

        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _classify_change(
        similarity: float,
        semantic_sim: float,
        length_ratio: float,
    ) -> OutputChangeType:
        """Classify the type of output change based on metrics.

        Uses a decision tree based on similarity scores and length
        ratio to determine the most appropriate change classification.

        Args
        ----
        similarity : float
            Lexical similarity score (0.0-1.0).
        semantic_sim : float
            Semantic similarity score (0.0-1.0).
        length_ratio : float
            Ratio of shorter to longer output length.

        Returns
        -------
        OutputChangeType
            Classification of the change type.

        Examples
        --------
        >>> OutputComparator._classify_change(0.98, 0.95, 1.0)
        <OutputChangeType.NO_CHANGE: 'no_change'>
        >>> OutputComparator._classify_change(0.85, 0.80, 0.9)
        <OutputChangeType.MINOR_VARIATION: 'minor_variation'>
        >>> OutputComparator._classify_change(0.10, 0.05, 0.8)
        <OutputChangeType.CONTRADICTORY: 'contradictory'>
        """
        if similarity > 0.95:
            return OutputChangeType.NO_CHANGE

        if similarity > 0.8 and length_ratio > 0.8:
            return OutputChangeType.MINOR_VARIATION

        if semantic_sim > 0.7:
            return OutputChangeType.SEMANTIC_EQUIVALENT

        if length_ratio < 0.5 or length_ratio > 2.0:
            if semantic_sim > 0.4:
                return OutputChangeType.DIFFERENT_FORMAT
            return OutputChangeType.DIFFERENT_CONTENT

        if semantic_sim < 0.2:
            return OutputChangeType.CONTRADICTORY

        return OutputChangeType.DIFFERENT_CONTENT

    @staticmethod
    def _find_differences(text1: str, text2: str) -> list[str]:
        """Find and describe key differences between two texts.

        Identifies words unique to each text and significant length
        differences.

        Args
        ----
        text1 : str
            First text (typically original output).
        text2 : str
            Second text (typically perturbed output).

        Returns
        -------
        list[str]
            List of human-readable difference descriptions,
            limited to 5 entries.

        Examples
        --------
        >>> OutputComparator._find_differences(
        ...     "Python is easy to learn",
        ...     "Python is difficult to master",
        ... )
        ['Original contains: easy, learn', 'Perturbed contains: difficult, master']
        """
        differences = []

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        only_in_1 = words1 - words2
        only_in_2 = words2 - words1

        if only_in_1:
            differences.append(f"Original contains: {', '.join(list(only_in_1)[:5])}")
        if only_in_2:
            differences.append(f"Perturbed contains: {', '.join(list(only_in_2)[:5])}")

        len_diff = len(text2) - len(text1)
        if abs(len_diff) > 50:
            differences.append(f"Length difference: {len_diff:+d} characters")

        return differences[:5]


class InputSensitivityAnalyzer:
    """Analyze prompt sensitivity through systematic perturbation testing.

    This is the main analyzer class that orchestrates sensitivity testing.
    It generates perturbations, collects model responses, compares outputs,
    and produces a comprehensive sensitivity profile.

    Parameters
    ----------
    perturbator : PromptPerturbator, optional
        Custom perturbator instance. If not provided, creates a default
        PromptPerturbator with no seed.
    comparator : OutputComparator, optional
        Custom comparator instance. If not provided, creates a default
        OutputComparator with word_overlap_similarity.
    robustness_threshold : float, default=0.7
        Similarity score threshold above which outputs are considered
        "robust" (similar enough). Lower values make robustness easier
        to achieve.

    Attributes
    ----------
    _perturbator : PromptPerturbator
        The perturbator used for generating prompt variations.
    _comparator : OutputComparator
        The comparator used for output analysis.
    _robustness_threshold : float
        The threshold for robustness determination.

    Examples
    --------
    Basic sensitivity analysis:

    >>> from insideLLMs.evaluation.sensitivity import InputSensitivityAnalyzer
    >>> def my_llm(prompt: str) -> str:
    ...     # Your LLM API call here
    ...     return llm_client.generate(prompt)
    >>> analyzer = InputSensitivityAnalyzer()
    >>> profile = analyzer.analyze(
    ...     prompt="Explain the theory of relativity",
    ...     get_response=my_llm,
    ... )
    >>> print(f"Overall sensitivity: {profile.overall_sensitivity.value}")
    Overall sensitivity: moderate

    Configure with custom components:

    >>> from insideLLMs.evaluation.sensitivity import (
    ...     InputSensitivityAnalyzer,
    ...     PromptPerturbator,
    ...     OutputComparator,
    ... )
    >>> analyzer = InputSensitivityAnalyzer(
    ...     perturbator=PromptPerturbator(seed=42),
    ...     comparator=OutputComparator(similarity_fn=my_similarity),
    ...     robustness_threshold=0.8,  # Stricter threshold
    ... )

    Analyze specific perturbation types:

    >>> profile = analyzer.analyze(
    ...     prompt="Write a haiku",
    ...     get_response=my_llm,
    ...     perturbation_types=[
    ...         PerturbationType.TYPO,
    ...         PerturbationType.CASE_CHANGE,
    ...     ],
    ...     n_variations=5,
    ... )
    >>> for ptype, score in profile.by_perturbation_type.items():
    ...     print(f"{ptype.value}: {score:.2f}")

    Interpret results:

    >>> if profile.overall_sensitivity == SensitivityLevel.VERY_HIGH:
    ...     print("WARNING: Prompt is highly sensitive!")
    ...     print("Recommendations:")
    ...     for rec in profile.recommendations:
    ...         print(f"  - {rec}")

    See Also
    --------
    SensitivityProfile : Result object from analysis.
    analyze_prompt_sensitivity : Convenience function using this class.
    ComparativeSensitivityAnalyzer : For comparing multiple prompts/models.
    """

    def __init__(
        self,
        perturbator: Optional[PromptPerturbator] = None,
        comparator: Optional[OutputComparator] = None,
        robustness_threshold: float = 0.7,
    ):
        """Initialize the sensitivity analyzer.

        Args
        ----
        perturbator : PromptPerturbator, optional
            Prompt perturbator for generating variations. Creates
            default if not provided.
        comparator : OutputComparator, optional
            Output comparator for analyzing differences. Creates
            default if not provided.
        robustness_threshold : float, default=0.7
            Similarity score threshold for robustness. Outputs with
            similarity >= this value are considered robust.

        Examples
        --------
        Default configuration:

        >>> analyzer = InputSensitivityAnalyzer()

        With reproducible perturbations:

        >>> analyzer = InputSensitivityAnalyzer(
        ...     perturbator=PromptPerturbator(seed=42),
        ... )

        With strict robustness:

        >>> analyzer = InputSensitivityAnalyzer(robustness_threshold=0.9)
        """
        self._perturbator = perturbator or PromptPerturbator()
        self._comparator = comparator or OutputComparator()
        self._robustness_threshold = robustness_threshold

    def analyze(
        self,
        prompt: str,
        get_response: Callable[[str], str],
        perturbation_types: Optional[list[PerturbationType]] = None,
        n_variations: int = 2,
    ) -> SensitivityProfile:
        """Analyze the sensitivity of a prompt to various perturbations.

        This method orchestrates the full sensitivity analysis workflow:
        1. Get baseline response for original prompt
        2. Generate perturbations of the prompt
        3. Get responses for each perturbed prompt
        4. Compare perturbed responses to baseline
        5. Aggregate results into a comprehensive profile

        Args
        ----
        prompt : str
            The original prompt to analyze.
        get_response : Callable[[str], str]
            Function that takes a prompt string and returns the model's
            response string. This will be called once for the original
            prompt and once for each perturbation.
        perturbation_types : list[PerturbationType], optional
            Types of perturbations to test. If None, all available
            perturbation types will be used.
        n_variations : int, default=2
            Number of variations to generate per perturbation type.
            Total API calls = 1 + (len(perturbation_types) * n_variations).

        Returns
        -------
        SensitivityProfile
            Comprehensive sensitivity profile containing individual
            results, aggregate scores, and recommendations.

        Raises
        ------
        Exception
            Propagates any exceptions from the get_response function.

        Examples
        --------
        Full analysis with all defaults:

        >>> profile = analyzer.analyze(
        ...     prompt="What is machine learning?",
        ...     get_response=my_llm,
        ... )
        >>> print(f"Tested {len(profile.results)} perturbations")
        Tested 18 perturbations

        Limited analysis for quick testing:

        >>> profile = analyzer.analyze(
        ...     prompt="Define AI",
        ...     get_response=my_llm,
        ...     perturbation_types=[PerturbationType.TYPO],
        ...     n_variations=1,
        ... )
        >>> print(f"Quick test: {profile.overall_sensitivity.value}")
        Quick test: low

        Extract actionable insights:

        >>> profile = analyzer.analyze(prompt, get_response)
        >>> print(f"Score: {profile.overall_score:.2f}")
        >>> print(f"Most sensitive to: {profile.most_sensitive_to}")
        >>> print(f"Most robust to: {profile.most_robust_to}")

        Notes
        -----
        - API call count: 1 (baseline) + n_perturbations
        - For 9 perturbation types with n_variations=2: 19 API calls
        - Consider using quick_sensitivity_check for faster testing
        """
        # Get original response
        original_response = get_response(prompt)

        # Generate perturbations
        perturbations = self._perturbator.perturb(prompt, perturbation_types, n_variations)

        # Test each perturbation
        results = []
        type_scores: dict[PerturbationType, list[float]] = defaultdict(list)

        for perturbation in perturbations:
            perturbed_response = get_response(perturbation.perturbed)
            comparison = self._comparator.compare(original_response, perturbed_response)

            # Calculate sensitivity score
            sensitivity = 1.0 - comparison.similarity_score
            is_robust = comparison.similarity_score >= self._robustness_threshold

            result = PerturbationSensitivityResult(
                perturbation=perturbation,
                output_comparison=comparison,
                sensitivity_score=sensitivity,
                is_robust=is_robust,
                notes=self._generate_notes(perturbation, comparison),
            )
            results.append(result)
            type_scores[perturbation.perturbation_type].append(sensitivity)

        # Aggregate by perturbation type
        by_type = {
            ptype: sum(scores) / len(scores) if scores else 0.0
            for ptype, scores in type_scores.items()
        }

        # Calculate overall
        all_scores = [r.sensitivity_score for r in results]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        overall_level = self._score_to_level(overall_score)

        # Find most/least sensitive
        sorted_types = sorted(by_type.items(), key=lambda x: x[1], reverse=True)
        most_sensitive = [t for t, s in sorted_types if s > 0.3][:3]
        most_robust = [t for t, s in sorted_types if s < 0.2][:3]

        return SensitivityProfile(
            prompt=prompt,
            results=results,
            overall_sensitivity=overall_level,
            overall_score=overall_score,
            by_perturbation_type=by_type,
            most_sensitive_to=most_sensitive,
            most_robust_to=most_robust,
            recommendations=self._generate_recommendations(by_type, overall_score),
        )

    @staticmethod
    def _score_to_level(score: float) -> SensitivityLevel:
        """Convert a numeric sensitivity score to a qualitative level.

        Args
        ----
        score : float
            Sensitivity score from 0.0 (no sensitivity) to 1.0
            (maximum sensitivity).

        Returns
        -------
        SensitivityLevel
            Qualitative classification of the sensitivity.

        Examples
        --------
        >>> InputSensitivityAnalyzer._score_to_level(0.15)
        <SensitivityLevel.VERY_LOW: 'very_low'>
        >>> InputSensitivityAnalyzer._score_to_level(0.45)
        <SensitivityLevel.MODERATE: 'moderate'>
        >>> InputSensitivityAnalyzer._score_to_level(0.85)
        <SensitivityLevel.VERY_HIGH: 'very_high'>
        """
        if score >= 0.8:
            return SensitivityLevel.VERY_HIGH
        elif score >= 0.6:
            return SensitivityLevel.HIGH
        elif score >= 0.4:
            return SensitivityLevel.MODERATE
        elif score >= 0.2:
            return SensitivityLevel.LOW
        return SensitivityLevel.VERY_LOW

    @staticmethod
    def _generate_notes(
        perturbation: Perturbation,
        comparison: OutputComparison,
    ) -> list[str]:
        """Generate diagnostic notes for a perturbation result.

        Creates human-readable notes highlighting concerning patterns
        such as contradictions, significant length changes, or
        disproportionate output changes.

        Args
        ----
        perturbation : Perturbation
            The perturbation that was applied.
        comparison : OutputComparison
            The comparison result for this perturbation.

        Returns
        -------
        list[str]
            List of note strings describing notable observations.

        Examples
        --------
        >>> notes = InputSensitivityAnalyzer._generate_notes(
        ...     perturbation,  # small change
        ...     comparison,    # large output difference
        ... )
        >>> print(notes)
        ['Small input change caused large output change']
        """
        notes = []

        if comparison.change_type == OutputChangeType.CONTRADICTORY:
            notes.append("Warning: Outputs are contradictory")

        if comparison.length_ratio < 0.5:
            notes.append("Significant length reduction in output")
        elif comparison.length_ratio > 2.0:
            notes.append("Significant length increase in output")

        if perturbation.change_magnitude < 0.1 and comparison.similarity_score < 0.5:
            notes.append("Small input change caused large output change")

        return notes

    @staticmethod
    def _generate_recommendations(
        by_type: dict[PerturbationType, float],
        overall: float,
    ) -> list[str]:
        """Generate actionable recommendations based on analysis results.

        Examines sensitivity scores by perturbation type and generates
        specific, actionable recommendations for improving prompt
        robustness.

        Args
        ----
        by_type : dict[PerturbationType, float]
            Average sensitivity score per perturbation type.
        overall : float
            Overall average sensitivity score.

        Returns
        -------
        list[str]
            List of recommendation strings.

        Examples
        --------
        >>> recommendations = InputSensitivityAnalyzer._generate_recommendations(
        ...     by_type={
        ...         PerturbationType.TYPO: 0.45,
        ...         PerturbationType.CASE_CHANGE: 0.12,
        ...     },
        ...     overall=0.28,
        ... )
        >>> print(recommendations)
        ['Model is typo-sensitive; consider input validation']
        """
        recommendations = []

        if overall > 0.5:
            recommendations.append("Consider making prompts more specific to reduce sensitivity")

        if by_type.get(PerturbationType.CASE_CHANGE, 0) > 0.3:
            recommendations.append("Model is case-sensitive; ensure consistent casing in prompts")

        if by_type.get(PerturbationType.TYPO, 0) > 0.3:
            recommendations.append("Model is typo-sensitive; consider input validation")

        if by_type.get(PerturbationType.INSTRUCTION_STYLE, 0) > 0.3:
            recommendations.append(
                "Model responds differently to instruction styles; standardize format"
            )

        if by_type.get(PerturbationType.SYNONYM, 0) > 0.3:
            recommendations.append("Model is sensitive to word choice; use precise terminology")

        if not recommendations:
            recommendations.append("Prompt appears robust to common perturbations")

        return recommendations


class ComparativeSensitivityAnalyzer:
    """Compare sensitivity across prompts or models."""

    def __init__(self, analyzer: Optional[InputSensitivityAnalyzer] = None):
        """Initialize comparative analyzer.

        Args:
            analyzer: Base sensitivity analyzer
        """
        self._analyzer = analyzer or InputSensitivityAnalyzer()

    def compare_prompts(
        self,
        prompts: list[str],
        get_response: Callable[[str], str],
        perturbation_types: Optional[list[PerturbationType]] = None,
    ) -> ComparativeSensitivity:
        """Compare sensitivity across multiple prompts.

        Args:
            prompts: List of prompts to compare
            get_response: Function to get model response
            perturbation_types: Types of perturbations to test

        Returns:
            ComparativeSensitivity object
        """
        profiles = []
        for prompt in prompts:
            profile = self._analyzer.analyze(prompt, get_response, perturbation_types)
            profiles.append(profile)

        return self._build_comparison(profiles, [p[:50] for p in prompts])

    def compare_models(
        self,
        prompt: str,
        model_responses: dict[str, Callable[[str], str]],
        perturbation_types: Optional[list[PerturbationType]] = None,
    ) -> ComparativeSensitivity:
        """Compare sensitivity across multiple models.

        Args:
            prompt: Prompt to test
            model_responses: Dict of model_id -> response function
            perturbation_types: Types of perturbations to test

        Returns:
            ComparativeSensitivity object
        """
        profiles = []
        identifiers = []

        for model_id, get_response in model_responses.items():
            profile = self._analyzer.analyze(prompt, get_response, perturbation_types)
            profiles.append(profile)
            identifiers.append(model_id)

        return self._build_comparison(profiles, identifiers)

    def _build_comparison(
        self,
        profiles: list[SensitivityProfile],
        identifiers: list[str],
    ) -> ComparativeSensitivity:
        """Build comparison from profiles."""
        # Rank by sensitivity
        ranking = sorted(
            zip(identifiers, [p.overall_score for p in profiles]),
            key=lambda x: x[1],
        )

        # Find common sensitivities
        all_sensitive: list[set[PerturbationType]] = []
        for profile in profiles:
            sensitive = {
                ptype for ptype, score in profile.by_perturbation_type.items() if score > 0.3
            }
            all_sensitive.append(sensitive)

        if all_sensitive:
            common = all_sensitive[0]
            for s in all_sensitive[1:]:
                common = common & s
        else:
            common = set()

        # Find divergent sensitivities
        all_types = set()
        for profile in profiles:
            all_types.update(profile.by_perturbation_type.keys())

        divergent = []
        for ptype in all_types:
            scores = [p.by_perturbation_type.get(ptype, 0) for p in profiles]
            if max(scores) - min(scores) > 0.4:
                divergent.append(ptype)

        return ComparativeSensitivity(
            profiles=profiles,
            ranking=ranking,
            most_robust=ranking[0][0] if ranking else "",
            most_sensitive=ranking[-1][0] if ranking else "",
            common_sensitivities=list(common),
            divergent_sensitivities=divergent,
        )


class FormatSensitivityTester:
    """Test sensitivity to output format instructions."""

    def __init__(self):
        """Initialize tester."""
        self._format_variations = [
            ("json", "Respond in JSON format"),
            ("markdown", "Respond in Markdown format"),
            ("bullet", "Respond with bullet points"),
            ("numbered", "Respond with a numbered list"),
            ("paragraph", "Respond in paragraph form"),
            ("brief", "Respond briefly"),
            ("detailed", "Respond in detail"),
        ]

    def test_format_sensitivity(
        self,
        base_prompt: str,
        get_response: Callable[[str], str],
    ) -> dict[str, Any]:
        """Test how format instructions affect output.

        Args:
            base_prompt: Base prompt without format instruction
            get_response: Function to get model response

        Returns:
            Dictionary with format sensitivity results
        """
        baseline = get_response(base_prompt)
        results = {"baseline": baseline, "variations": {}}

        comparator = OutputComparator()

        for format_name, format_instruction in self._format_variations:
            formatted_prompt = f"{base_prompt}\n\n{format_instruction}"
            response = get_response(formatted_prompt)

            comparison = comparator.compare(baseline, response)
            results["variations"][format_name] = {
                "response": response[:500],
                "similarity_to_baseline": comparison.similarity_score,
                "length_ratio": comparison.length_ratio,
                "format_followed": self._check_format_followed(response, format_name),
            }

        # Calculate format adherence
        adherence_scores = [v["format_followed"] for v in results["variations"].values()]
        results["format_adherence_rate"] = (
            sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0
        )

        return results

    @staticmethod
    def _check_format_followed(response: str, format_name: str) -> bool:
        """Check if format instruction was followed."""
        if format_name == "json":
            return "{" in response and "}" in response
        elif format_name == "markdown":
            return "#" in response or "**" in response or "*" in response
        elif format_name == "bullet":
            return "- " in response or "• " in response
        elif format_name == "numbered":
            return any(f"{i}." in response for i in range(1, 10))
        elif format_name == "brief":
            return len(response) < 500
        elif format_name == "detailed":
            return len(response) > 200
        return True  # Paragraph is default


# Convenience functions


def analyze_prompt_sensitivity(
    prompt: str,
    get_response: Callable[[str], str],
    perturbation_types: Optional[list[PerturbationType]] = None,
) -> SensitivityProfile:
    """Analyze sensitivity of a prompt.

    Args:
        prompt: Prompt to analyze
        get_response: Function to get model response
        perturbation_types: Types of perturbations to test

    Returns:
        SensitivityProfile object
    """
    analyzer = InputSensitivityAnalyzer()
    return analyzer.analyze(prompt, get_response, perturbation_types)


def compare_prompt_sensitivity(
    prompts: list[str],
    get_response: Callable[[str], str],
) -> ComparativeSensitivity:
    """Compare sensitivity across prompts.

    Args:
        prompts: List of prompts to compare
        get_response: Function to get model response

    Returns:
        ComparativeSensitivity object
    """
    analyzer = ComparativeSensitivityAnalyzer()
    return analyzer.compare_prompts(prompts, get_response)


def generate_perturbations(
    prompt: str,
    perturbation_types: Optional[list[PerturbationType]] = None,
    n_variations: int = 2,
) -> list[Perturbation]:
    """Generate perturbations of a prompt.

    Args:
        prompt: Original prompt
        perturbation_types: Types of perturbations to apply
        n_variations: Number of variations per type

    Returns:
        List of Perturbation objects
    """
    perturbator = PromptPerturbator()
    return perturbator.perturb(prompt, perturbation_types, n_variations)


def quick_sensitivity_check(
    prompt: str,
    get_response: Callable[[str], str],
) -> dict[str, Any]:
    """Quick sensitivity check with basic perturbations.

    Args:
        prompt: Prompt to check
        get_response: Function to get model response

    Returns:
        Dictionary with quick check results
    """
    basic_types = [
        PerturbationType.CASE_CHANGE,
        PerturbationType.TYPO,
        PerturbationType.SYNONYM,
    ]

    analyzer = InputSensitivityAnalyzer()
    profile = analyzer.analyze(prompt, get_response, basic_types, n_variations=1)

    return {
        "overall_sensitivity": profile.overall_sensitivity.value,
        "overall_score": profile.overall_score,
        "is_robust": profile.overall_score < 0.3,
        "n_tests": len(profile.results),
        "recommendations": profile.recommendations[:2],
    }


def check_format_sensitivity(
    prompt: str,
    get_response: Callable[[str], str],
) -> dict[str, Any]:
    """Check sensitivity to format instructions.

    Args:
        prompt: Base prompt
        get_response: Function to get model response

    Returns:
        Dictionary with format sensitivity results
    """
    tester = FormatSensitivityTester()
    return tester.test_format_sensitivity(prompt, get_response)


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import SensitivityAnalyzer. The canonical name is
# InputSensitivityAnalyzer.
SensitivityAnalyzer = InputSensitivityAnalyzer

# Older code and tests may import SensitivityResult. The canonical name is
# PerturbationSensitivityResult.
SensitivityResult = PerturbationSensitivityResult
