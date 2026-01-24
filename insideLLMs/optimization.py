"""
Prompt Optimization and Tuning Utilities.

This module provides a comprehensive suite of tools for optimizing prompts
for large language models (LLMs). It addresses key challenges in prompt
engineering including token efficiency, instruction clarity, example selection,
and systematic prompt analysis.

Overview
--------
The module contains several optimization strategies that can be used independently
or combined:

1. **Prompt Compression** - Reduce token count while preserving semantic meaning
2. **Instruction Optimization** - Improve clarity and effectiveness of instructions
3. **Few-Shot Example Selection** - Choose optimal examples for in-context learning
4. **Prompt Ablation** - Identify essential vs. removable prompt components
5. **Token Budget Optimization** - Fit prompts within context window limits

Key Classes
-----------
PromptCompressor
    Compresses prompts by removing filler words and verbose constructions.
InstructionOptimizer
    Enhances instruction clarity by strengthening verbs and flagging ambiguity.
FewShotSelector
    Selects diverse, relevant examples for few-shot prompting.
PromptAblator
    Performs ablation studies to identify critical prompt components.
TokenBudgetOptimizer
    Ensures prompts fit within token limits while preserving quality.
PromptOptimizer
    Combines multiple strategies for comprehensive optimization.

Data Classes
------------
PromptCompressionResult
    Contains compression statistics and transformed prompt.
AblationResult
    Contains component importance rankings and minimal prompt.
ExampleScore
    Scores for individual few-shot examples.
ExampleSelectionResult
    Results of example selection with coverage metrics.
OptimizationReport
    Comprehensive report of all applied optimizations.

Examples
--------
Basic prompt compression:

>>> from insideLLMs.optimization import compress_prompt
>>> result = compress_prompt(
...     "In order to complete the task, please note that you should "
...     "basically try to generate a response.",
...     target_reduction=0.3
... )
>>> print(result.compressed)
'To complete the task, you should ensure generate a response.'
>>> print(f"Saved {result.tokens_saved} tokens ({result.compression_ratio:.1%} reduction)")
Saved 12 tokens (35.3% reduction)

Optimizing instructions for clarity:

>>> from insideLLMs.optimization import optimize_instruction
>>> optimized, changes = optimize_instruction(
...     "Try to write something good about the topic"
... )
>>> print(optimized)
'Ensure write something good about the topic.'
>>> for change in changes:
...     print(f"  - {change}")
  - Strengthened verb: 'try to' -> 'ensure'
  - Consider clarifying ambiguous term: 'good'

Selecting few-shot examples:

>>> from insideLLMs.optimization import select_examples
>>> examples = [
...     {"input": "Translate hello", "output": "hola"},
...     {"input": "Translate goodbye", "output": "adios"},
...     {"input": "Translate thank you", "output": "gracias"},
...     {"input": "Summarize the article", "output": "The article discusses..."},
... ]
>>> result = select_examples(
...     query="Translate good morning",
...     examples=examples,
...     n=2
... )
>>> for ex in result.selected_examples:
...     print(ex["input"])
Translate hello
Translate goodbye

Performing ablation study:

>>> from insideLLMs.optimization import ablate_prompt
>>> prompt = '''You are a helpful assistant.
...
... Answer questions accurately.
...
... Be concise and clear.'''
>>> result = ablate_prompt(prompt)
>>> print("Essential components:", result.essential_components)
>>> print("Removable components:", result.removable_components)

Optimizing for token budget:

>>> from insideLLMs.optimization import optimize_for_budget
>>> result = optimize_for_budget(
...     prompt="This is a very long prompt that needs optimization...",
...     max_tokens=1000,
...     examples=[{"input": "ex1", "output": "out1"}],
...     reserve_for_response=200
... )
>>> print(f"Final tokens: {result['final_tokens']}")
>>> print(f"Actions taken: {result['actions_taken']}")

Notes
-----
- Token estimation uses a rough approximation of ~4 characters per token.
  For production use, consider integrating a proper tokenizer (e.g., tiktoken).
- The compression algorithms are conservative to avoid semantic loss.
- Example selection uses word overlap for relevance; embedding-based similarity
  would provide more accurate results for semantic matching.
- Ablation studies work best with clearly delimited prompt sections.

See Also
--------
insideLLMs.prompts : Prompt templates and management utilities.
insideLLMs.analysis : Response analysis tools.
insideLLMs.core : Core tracing and instrumentation.

References
----------
.. [1] Reynolds, L., & McDonell, K. (2021). Prompt Programming for Large
       Language Models: Beyond the Few-Shot Paradigm.
.. [2] Liu, P., et al. (2023). Pre-train, Prompt, and Predict: A Systematic
       Survey of Prompting Methods in Natural Language Processing.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class OptimizationStrategy(Enum):
    """Enumeration of available prompt optimization strategies.

    This enum defines the different optimization approaches that can be applied
    to prompts. Multiple strategies can be combined for comprehensive optimization.

    Attributes
    ----------
    COMPRESSION : str
        Reduces token count by removing filler words, redundant phrases, and
        verbose constructions while preserving semantic meaning.
    CLARITY : str
        Improves instruction clarity by strengthening weak verbs, flagging
        ambiguous terms, and ensuring proper punctuation.
    SPECIFICITY : str
        Enhances specificity by identifying vague language and suggesting
        more precise alternatives.
    STRUCTURE : str
        Improves structural organization including list formatting,
        section breaks, and logical flow.
    EXAMPLE_SELECTION : str
        Optimizes few-shot examples by selecting the most relevant and
        diverse examples from a candidate pool.

    Examples
    --------
    Using a single strategy:

    >>> from insideLLMs.optimization import PromptOptimizer, OptimizationStrategy
    >>> optimizer = PromptOptimizer()
    >>> result = optimizer.optimize(
    ...     "Try to write good code",
    ...     strategies=[OptimizationStrategy.CLARITY]
    ... )
    >>> print(result.optimized_prompt)

    Combining multiple strategies:

    >>> result = optimizer.optimize(
    ...     "In order to complete the task, please note that you should "
    ...     "try to generate a good response.",
    ...     strategies=[
    ...         OptimizationStrategy.COMPRESSION,
    ...         OptimizationStrategy.CLARITY,
    ...         OptimizationStrategy.STRUCTURE
    ...     ]
    ... )
    >>> print(f"Strategies applied: {[s.value for s in result.strategies_applied]}")

    Checking available strategies:

    >>> for strategy in OptimizationStrategy:
    ...     print(f"{strategy.name}: {strategy.value}")
    COMPRESSION: compression
    CLARITY: clarity
    SPECIFICITY: specificity
    STRUCTURE: structure
    EXAMPLE_SELECTION: example_selection

    See Also
    --------
    PromptOptimizer : Main optimizer that uses these strategies.
    OptimizationReport : Report containing applied strategies and results.
    """

    COMPRESSION = "compression"  # Reduce token count
    CLARITY = "clarity"  # Improve clarity
    SPECIFICITY = "specificity"  # Add specificity
    STRUCTURE = "structure"  # Improve structure
    EXAMPLE_SELECTION = "example_selection"  # Optimize few-shot examples


@dataclass
class PromptCompressionResult:
    """Result container for prompt compression operations.

    This dataclass holds all information about a compression operation,
    including the original and compressed text, token counts, and details
    about what was removed or preserved during compression.

    Parameters
    ----------
    original : str
        The original uncompressed prompt text.
    compressed : str
        The compressed prompt text after optimization.
    original_tokens : int
        Estimated token count of the original prompt.
    compressed_tokens : int
        Estimated token count of the compressed prompt.
    compression_ratio : float
        Ratio of tokens saved (0.0-1.0), calculated as
        1 - (compressed_tokens / original_tokens).
    removed_elements : list of str, optional
        List of elements that were removed during compression,
        such as filler phrases or verbose constructions.
    preserved_elements : list of str, optional
        List of keywords or elements that were explicitly preserved
        during compression.

    Attributes
    ----------
    tokens_saved : int
        Property that calculates the number of tokens saved.

    Examples
    --------
    Creating a compression result manually:

    >>> result = PromptCompressionResult(
    ...     original="In order to complete the task, you should try to...",
    ...     compressed="To complete the task, you should...",
    ...     original_tokens=15,
    ...     compressed_tokens=10,
    ...     compression_ratio=0.33,
    ...     removed_elements=["filler: in order to"],
    ...     preserved_elements=["task"]
    ... )
    >>> print(f"Saved {result.tokens_saved} tokens")
    Saved 5 tokens

    Using with PromptCompressor:

    >>> from insideLLMs.optimization import PromptCompressor
    >>> compressor = PromptCompressor()
    >>> result = compressor.compress(
    ...     "Please note that this is basically a test prompt.",
    ...     preserve_keywords={"test"}
    ... )
    >>> print(f"Compression: {result.compression_ratio:.1%}")
    >>> print(f"Removed: {result.removed_elements}")
    >>> print(f"Preserved: {result.preserved_elements}")

    Converting to dictionary for serialization:

    >>> result_dict = result.to_dict()
    >>> import json
    >>> print(json.dumps(result_dict, indent=2))
    {
      "original_tokens": 15,
      "compressed_tokens": 10,
      ...
    }

    See Also
    --------
    PromptCompressor : Class that produces these results.
    compress_prompt : Convenience function for compression.
    """

    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    removed_elements: list[str] = field(default_factory=list)
    preserved_elements: list[str] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        """Calculate the number of tokens saved by compression.

        Returns
        -------
        int
            The difference between original and compressed token counts.
            Always non-negative for valid compression results.

        Examples
        --------
        >>> result = PromptCompressionResult(
        ...     original="long text", compressed="short",
        ...     original_tokens=100, compressed_tokens=75,
        ...     compression_ratio=0.25
        ... )
        >>> result.tokens_saved
        25
        """
        return self.original_tokens - self.compressed_tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert the compression result to a dictionary.

        Returns a dictionary representation suitable for JSON serialization
        or logging. Note that the original and compressed text are not
        included to keep the dictionary compact.

        Returns
        -------
        dict of str to Any
            Dictionary containing:
            - original_tokens: int
            - compressed_tokens: int
            - compression_ratio: float
            - tokens_saved: int
            - removed_elements: list of str
            - preserved_elements: list of str

        Examples
        --------
        >>> result = PromptCompressionResult(
        ...     original="test", compressed="t",
        ...     original_tokens=10, compressed_tokens=5,
        ...     compression_ratio=0.5,
        ...     removed_elements=["filler: basically"],
        ...     preserved_elements=["important"]
        ... )
        >>> d = result.to_dict()
        >>> d["compression_ratio"]
        0.5
        >>> d["tokens_saved"]
        5
        """
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": self.compression_ratio,
            "tokens_saved": self.tokens_saved,
            "removed_elements": self.removed_elements,
            "preserved_elements": self.preserved_elements,
        }


@dataclass
class AblationResult:
    """Result container for prompt ablation studies.

    An ablation study systematically removes components from a prompt to
    determine which parts are essential for maintaining quality. This dataclass
    holds the complete results of such a study.

    Parameters
    ----------
    original_prompt : str
        The complete original prompt before ablation.
    components : list of str
        List of all identified components in the prompt, split by
        the specified delimiter.
    component_scores : dict of str to float
        Mapping of component identifiers (truncated) to their importance
        scores. Higher scores indicate greater importance (larger quality
        drop when removed).
    essential_components : list of str
        Components whose removal causes significant quality degradation
        (above the threshold).
    removable_components : list of str
        Components that can be removed without significant quality loss.
    minimal_prompt : str
        A reconstructed prompt containing only essential components.
    importance_ranking : list of tuple of (str, float)
        Components ranked by importance, from most to least important.

    Examples
    --------
    Understanding ablation results:

    >>> from insideLLMs.optimization import ablate_prompt
    >>> prompt = '''You are a helpful assistant.
    ...
    ... Always be polite and professional.
    ...
    ... Answer questions accurately and concisely.
    ...
    ... If unsure, say so.'''
    >>> result = ablate_prompt(prompt)
    >>> print(f"Total components: {len(result.components)}")
    Total components: 4
    >>> print(f"Essential: {len(result.essential_components)}")
    >>> print(f"Removable: {len(result.removable_components)}")

    Accessing importance rankings:

    >>> for component, importance in result.importance_ranking[:3]:
    ...     print(f"  {importance:.2f}: {component}")

    Using the minimal prompt:

    >>> print("Minimal prompt:")
    >>> print(result.minimal_prompt)

    Serializing results:

    >>> result_dict = result.to_dict()
    >>> print(f"Components analyzed: {result_dict['num_components']}")

    See Also
    --------
    PromptAblator : Class that performs ablation studies.
    ablate_prompt : Convenience function for ablation.

    Notes
    -----
    Component identifiers in `component_scores` and `importance_ranking`
    are truncated to 50 characters plus "..." for readability.
    """

    original_prompt: str
    components: list[str]
    component_scores: dict[str, float]
    essential_components: list[str]
    removable_components: list[str]
    minimal_prompt: str
    importance_ranking: list[tuple[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Convert the ablation result to a dictionary.

        Returns a dictionary representation suitable for JSON serialization,
        logging, or further analysis. The original prompt text is not
        included to keep the output compact.

        Returns
        -------
        dict of str to Any
            Dictionary containing:
            - num_components: int - Total number of components analyzed
            - component_scores: dict - Component importance scores
            - essential_components: list - Components that must be kept
            - removable_components: list - Components that can be removed
            - importance_ranking: list - Ranked list of (component, score)

        Examples
        --------
        >>> result = AblationResult(
        ...     original_prompt="A\\n\\nB\\n\\nC",
        ...     components=["A", "B", "C"],
        ...     component_scores={"A...": 0.3, "B...": 0.1, "C...": 0.2},
        ...     essential_components=["A...", "C..."],
        ...     removable_components=["B..."],
        ...     minimal_prompt="A\\n\\nC",
        ...     importance_ranking=[("A...", 0.3), ("C...", 0.2), ("B...", 0.1)]
        ... )
        >>> d = result.to_dict()
        >>> d["num_components"]
        3
        >>> len(d["essential_components"])
        2
        """
        return {
            "num_components": len(self.components),
            "component_scores": self.component_scores,
            "essential_components": self.essential_components,
            "removable_components": self.removable_components,
            "importance_ranking": self.importance_ranking,
        }


@dataclass
class ExampleScore:
    """Detailed scoring breakdown for a single few-shot example.

    This dataclass contains multiple dimensions of quality assessment for
    a few-shot example, enabling fine-grained analysis of why certain
    examples were selected over others.

    Parameters
    ----------
    example : dict of str to str
        The original example dictionary, typically containing 'input'
        and 'output' keys.
    relevance_score : float
        Score (0.0-1.0) indicating how relevant the example is to the
        target query, based on word overlap after stop word removal.
    diversity_score : float
        Score (0.0-1.0) indicating how different this example is from
        other selected examples, using Jaccard distance.
    quality_score : float
        Score (0.0-1.0) based on intrinsic quality factors like output
        length, completeness, and proper formatting.
    overall_score : float
        Weighted combination of relevance, diversity, and quality scores.
        Default weighting: 40% relevance, 30% diversity, 30% quality.

    Examples
    --------
    Examining individual example scores:

    >>> from insideLLMs.optimization import select_examples
    >>> examples = [
    ...     {"input": "What is Python?", "output": "A programming language."},
    ...     {"input": "What is Java?", "output": "A programming language."},
    ...     {"input": "What is the weather?", "output": "It varies by location."},
    ... ]
    >>> result = select_examples("What is JavaScript?", examples, n=2)
    >>> for score in result.example_scores:
    ...     print(f"Relevance: {score.relevance_score:.2f}, "
    ...           f"Diversity: {score.diversity_score:.2f}, "
    ...           f"Overall: {score.overall_score:.2f}")

    Converting to dictionary for logging:

    >>> score = result.example_scores[0]
    >>> score_dict = score.to_dict()
    >>> print(f"Quality: {score_dict['quality_score']:.2f}")

    Comparing examples:

    >>> scores = result.example_scores
    >>> best = max(scores, key=lambda s: s.overall_score)
    >>> print(f"Best example: {best.example['input']}")

    See Also
    --------
    ExampleSelectionResult : Container for all selection results.
    FewShotSelector : Class that generates these scores.
    """

    example: dict[str, str]
    relevance_score: float
    diversity_score: float
    quality_score: float
    overall_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the example score to a dictionary.

        Returns a dictionary containing only the numeric scores,
        excluding the example content itself for compactness.

        Returns
        -------
        dict of str to float
            Dictionary containing:
            - relevance_score: float (0.0-1.0)
            - diversity_score: float (0.0-1.0)
            - quality_score: float (0.0-1.0)
            - overall_score: float (0.0-1.0)

        Examples
        --------
        >>> score = ExampleScore(
        ...     example={"input": "test", "output": "result"},
        ...     relevance_score=0.8,
        ...     diversity_score=0.6,
        ...     quality_score=0.7,
        ...     overall_score=0.72
        ... )
        >>> d = score.to_dict()
        >>> d["relevance_score"]
        0.8
        >>> "example" in d
        False
        """
        return {
            "relevance_score": self.relevance_score,
            "diversity_score": self.diversity_score,
            "quality_score": self.quality_score,
            "overall_score": self.overall_score,
        }


@dataclass
class ExampleSelectionResult:
    """Complete result of few-shot example selection.

    This dataclass contains the selected examples along with comprehensive
    metrics about the selection quality, including individual scores for
    each example and aggregate coverage/diversity metrics.

    Parameters
    ----------
    query : str
        The original query/task that examples were selected for.
    selected_examples : list of dict of str to str
        The chosen examples, ordered by selection (most relevant first).
    example_scores : list of ExampleScore
        Detailed scoring breakdown for each selected example.
    coverage_score : float
        Score (0.0-1.0) indicating how well the selected examples
        cover the concepts present in the query.
    diversity_score : float
        Average diversity score across selected examples, indicating
        how different the examples are from each other.

    Examples
    --------
    Basic usage with select_examples:

    >>> from insideLLMs.optimization import select_examples
    >>> examples = [
    ...     {"input": "Summarize the news article", "output": "The article reports..."},
    ...     {"input": "Summarize the research paper", "output": "This paper presents..."},
    ...     {"input": "Translate to Spanish", "output": "Hola mundo"},
    ...     {"input": "Summarize the blog post", "output": "The blog discusses..."},
    ... ]
    >>> result = select_examples(
    ...     query="Summarize the financial report",
    ...     examples=examples,
    ...     n=2
    ... )
    >>> print(f"Selected {len(result.selected_examples)} examples")
    Selected 2 examples
    >>> print(f"Coverage: {result.coverage_score:.2f}")
    >>> print(f"Diversity: {result.diversity_score:.2f}")

    Accessing selected examples:

    >>> for ex in result.selected_examples:
    ...     print(f"Input: {ex['input']}")
    ...     print(f"Output: {ex['output'][:30]}...")
    ...     print()

    Analyzing selection quality:

    >>> if result.coverage_score < 0.5:
    ...     print("Warning: Low coverage - examples may not match query well")
    >>> if result.diversity_score < 0.3:
    ...     print("Warning: Low diversity - examples are too similar")

    Serializing for logging:

    >>> import json
    >>> result_dict = result.to_dict()
    >>> print(json.dumps(result_dict, indent=2))

    See Also
    --------
    FewShotSelector : Class that performs example selection.
    ExampleScore : Individual example scoring details.
    select_examples : Convenience function for selection.
    """

    query: str
    selected_examples: list[dict[str, str]]
    example_scores: list[ExampleScore]
    coverage_score: float
    diversity_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the selection result to a dictionary.

        Returns a dictionary representation suitable for JSON serialization,
        logging, or analysis. The query and full examples are not included
        to keep the output focused on metrics.

        Returns
        -------
        dict of str to Any
            Dictionary containing:
            - num_selected: int - Number of examples selected
            - coverage_score: float - How well examples cover query concepts
            - diversity_score: float - Average diversity among examples
            - example_scores: list of dict - Score breakdown for each example

        Examples
        --------
        >>> from insideLLMs.optimization import select_examples
        >>> result = select_examples(
        ...     query="test query",
        ...     examples=[{"input": "a", "output": "b"}],
        ...     n=1
        ... )
        >>> d = result.to_dict()
        >>> "num_selected" in d
        True
        >>> d["num_selected"]
        1
        """
        return {
            "num_selected": len(self.selected_examples),
            "coverage_score": self.coverage_score,
            "diversity_score": self.diversity_score,
            "example_scores": [e.to_dict() for e in self.example_scores],
        }


@dataclass
class OptimizationReport:
    """Comprehensive report of prompt optimization results.

    This dataclass provides a complete summary of all optimization operations
    performed on a prompt, including the strategies applied, quantitative
    improvements, and actionable suggestions for further enhancement.

    Parameters
    ----------
    original_prompt : str
        The original prompt text before any optimization.
    optimized_prompt : str
        The final prompt text after all optimizations.
    strategies_applied : list of OptimizationStrategy
        List of optimization strategies that were actually applied
        (strategies may be skipped if no improvement is possible).
    improvements : dict of str to float
        Mapping of improvement categories to their measured values.
        Categories include 'compression', 'clarity', 'structure'.
    suggestions : list of str
        Actionable recommendations for further improvement that
        could not be automatically applied.
    token_reduction : int
        Number of tokens reduced from original to optimized prompt.
    estimated_quality_change : float
        Estimated change in prompt quality (positive = improvement).
        Based on weighted average of individual improvements.

    Examples
    --------
    Full optimization with default strategies:

    >>> from insideLLMs.optimization import optimize_prompt
    >>> result = optimize_prompt(
    ...     "In order to complete the task, please note that you should "
    ...     "try to write something good. basically just do it"
    ... )
    >>> print(f"Token reduction: {result.token_reduction}")
    >>> print(f"Quality change: {result.estimated_quality_change:.2f}")

    Reviewing applied strategies:

    >>> for strategy in result.strategies_applied:
    ...     print(f"Applied: {strategy.value}")
    ...     if strategy.value in result.improvements:
    ...         print(f"  Improvement: {result.improvements[strategy.value]:.2f}")

    Acting on suggestions:

    >>> if result.suggestions:
    ...     print("Manual improvements recommended:")
    ...     for suggestion in result.suggestions:
    ...         print(f"  - {suggestion}")

    Comparing before and after:

    >>> print("BEFORE:")
    >>> print(result.original_prompt[:100])
    >>> print("\\nAFTER:")
    >>> print(result.optimized_prompt[:100])

    Serializing for logging or storage:

    >>> import json
    >>> report_dict = result.to_dict()
    >>> print(json.dumps(report_dict, indent=2))

    See Also
    --------
    PromptOptimizer : Class that generates these reports.
    OptimizationStrategy : Available optimization strategies.
    optimize_prompt : Convenience function for optimization.
    """

    original_prompt: str
    optimized_prompt: str
    strategies_applied: list[OptimizationStrategy]
    improvements: dict[str, float]
    suggestions: list[str]
    token_reduction: int
    estimated_quality_change: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the optimization report to a dictionary.

        Returns a dictionary representation suitable for JSON serialization,
        logging, or analysis. The full prompt texts are not included to
        keep the output compact and focused on metrics.

        Returns
        -------
        dict of str to Any
            Dictionary containing:
            - strategies_applied: list of str - Strategy values applied
            - improvements: dict - Category to improvement value mapping
            - suggestions: list of str - Manual improvement recommendations
            - token_reduction: int - Tokens saved by optimization
            - estimated_quality_change: float - Overall quality improvement

        Examples
        --------
        >>> from insideLLMs.optimization import optimize_prompt
        >>> result = optimize_prompt("In order to test this function...")
        >>> d = result.to_dict()
        >>> "strategies_applied" in d
        True
        >>> isinstance(d["token_reduction"], int)
        True
        """
        return {
            "strategies_applied": [s.value for s in self.strategies_applied],
            "improvements": self.improvements,
            "suggestions": self.suggestions,
            "token_reduction": self.token_reduction,
            "estimated_quality_change": self.estimated_quality_change,
        }


class PromptCompressor:
    """Compress prompts while preserving semantic meaning.

    This class provides methods to reduce the token count of prompts by
    removing filler words, simplifying verbose constructions, and cleaning
    up redundant whitespace. The compression is conservative to avoid
    losing important semantic content.

    The compressor works by:
    1. Removing common filler phrases (e.g., "please note that", "basically")
    2. Simplifying verbose patterns (e.g., "in order to" -> "to")
    3. Cleaning up redundant whitespace and empty brackets
    4. Preserving user-specified keywords

    Parameters
    ----------
    preserve_structure : bool, default=True
        Whether to preserve structural elements like newlines and
        list formatting during compression.

    Attributes
    ----------
    FILLER_PHRASES : list of str
        Class-level list of filler phrases that can typically be removed
        without semantic loss.
    VERBOSE_PATTERNS : list of tuple of (str, str)
        Class-level list of (pattern, replacement) pairs for simplifying
        verbose constructions.
    preserve_structure : bool
        Instance attribute controlling structure preservation.

    Examples
    --------
    Basic compression:

    >>> from insideLLMs.optimization import PromptCompressor
    >>> compressor = PromptCompressor()
    >>> result = compressor.compress(
    ...     "In order to complete the task, please note that you should "
    ...     "basically try to generate a response."
    ... )
    >>> print(f"Original: {result.original_tokens} tokens")
    >>> print(f"Compressed: {result.compressed_tokens} tokens")
    >>> print(f"Saved: {result.compression_ratio:.1%}")

    Preserving specific keywords:

    >>> result = compressor.compress(
    ...     "Please note that the API key is essentially required.",
    ...     preserve_keywords={"API", "key"}
    ... )
    >>> assert "API" in result.compressed
    >>> assert "key" in result.compressed

    Targeting specific reduction:

    >>> result = compressor.compress(
    ...     "This is a very long prompt with quite a lot of filler words "
    ...     "that are really not necessary at all.",
    ...     target_reduction=0.3
    ... )
    >>> print(f"Achieved {result.compression_ratio:.1%} reduction")

    Examining what was removed:

    >>> result = compressor.compress(
    ...     "In order to understand, due to the fact that it is important..."
    ... )
    >>> for removal in result.removed_elements:
    ...     print(f"  Removed: {removal}")

    See Also
    --------
    compress_prompt : Convenience function for one-off compression.
    TokenBudgetOptimizer : Uses compression as part of budget optimization.
    PromptCompressionResult : Container for compression results.

    Notes
    -----
    - Token estimation uses ~4 characters per token approximation
    - Compression is case-insensitive for pattern matching
    - Keywords in preserve_keywords are matched case-insensitively
    """

    # Filler phrases that can usually be removed
    FILLER_PHRASES = [
        "please note that",
        "it is important to",
        "keep in mind that",
        "as mentioned earlier",
        "in other words",
        "basically",
        "essentially",
        "actually",
        "literally",
        "very",
        "really",
        "quite",
        "somewhat",
        "rather",
        "just",
        "simply",
    ]

    # Verbose constructions that can be simplified
    VERBOSE_PATTERNS = [
        (r"in order to", "to"),
        (r"due to the fact that", "because"),
        (r"at this point in time", "now"),
        (r"in the event that", "if"),
        (r"for the purpose of", "for"),
        (r"with regard to", "about"),
        (r"in terms of", "regarding"),
        (r"on a daily basis", "daily"),
        (r"at the present time", "currently"),
        (r"in the near future", "soon"),
        (r"a large number of", "many"),
        (r"a small number of", "few"),
        (r"the majority of", "most"),
        (r"in spite of the fact that", "although"),
        (r"whether or not", "whether"),
    ]

    def __init__(self, preserve_structure: bool = True):
        """Initialize a new PromptCompressor instance.

        Parameters
        ----------
        preserve_structure : bool, default=True
            If True, preserves structural elements like newlines,
            indentation patterns, and list formatting. If False,
            more aggressive whitespace normalization is applied.

        Examples
        --------
        Default initialization (preserves structure):

        >>> compressor = PromptCompressor()
        >>> compressor.preserve_structure
        True

        Aggressive compression (may alter structure):

        >>> compressor = PromptCompressor(preserve_structure=False)
        >>> compressor.preserve_structure
        False
        """
        self.preserve_structure = preserve_structure

    def compress(
        self,
        prompt: str,
        target_reduction: float = 0.2,
        preserve_keywords: Optional[set[str]] = None,
    ) -> PromptCompressionResult:
        """Compress a prompt by removing filler and simplifying verbose text.

        This method applies multiple compression techniques to reduce the
        token count of a prompt while preserving its core meaning. It tracks
        what was removed and what was preserved for transparency.

        Parameters
        ----------
        prompt : str
            The original prompt text to compress.
        target_reduction : float, default=0.2
            Target token reduction ratio (0.0-1.0). A value of 0.2 means
            targeting 20% fewer tokens. Note: This is a soft target; actual
            reduction depends on the presence of compressible content.
        preserve_keywords : set of str, optional
            Set of keywords that must be preserved in the output.
            Filler phrases containing these keywords will not be removed.
            Keywords are matched case-insensitively.

        Returns
        -------
        PromptCompressionResult
            A result object containing:
            - original: The original prompt text
            - compressed: The compressed prompt text
            - original_tokens: Estimated token count before compression
            - compressed_tokens: Estimated token count after compression
            - compression_ratio: Achieved compression ratio (0.0-1.0)
            - removed_elements: List of elements that were removed
            - preserved_elements: List of keywords that were preserved

        Examples
        --------
        Basic compression:

        >>> compressor = PromptCompressor()
        >>> result = compressor.compress(
        ...     "In order to complete the task, you need to analyze the data."
        ... )
        >>> result.compressed
        'To complete the task, you need to analyze the data.'
        >>> result.compression_ratio > 0
        True

        With preserved keywords:

        >>> result = compressor.compress(
        ...     "Please note that the configuration is essentially required.",
        ...     preserve_keywords={"configuration", "required"}
        ... )
        >>> "configuration" in result.compressed
        True
        >>> "required" in result.compressed
        True
        >>> "configuration" in result.preserved_elements
        True

        Examining removed elements:

        >>> result = compressor.compress(
        ...     "Due to the fact that the system is basically overloaded, "
        ...     "at this point in time we need more resources."
        ... )
        >>> print("Removed:")
        >>> for item in result.removed_elements:
        ...     print(f"  - {item}")
        Removed:
          - verbose: due to the fact that -> because
          - filler: basically
          - verbose: at this point in time -> now

        See Also
        --------
        compress_prompt : Convenience function for one-off compression.
        PromptCompressionResult : Detailed result documentation.
        """
        preserve_keywords = preserve_keywords or set()
        original_tokens = self._estimate_tokens(prompt)
        removed = []
        preserved = []

        compressed = prompt

        # Remove filler phrases
        for filler in self.FILLER_PHRASES:
            if filler in compressed.lower():
                # Check if any preserve keyword is in the filler
                if not any(kw.lower() in filler for kw in preserve_keywords):
                    pattern = re.compile(re.escape(filler), re.IGNORECASE)
                    if pattern.search(compressed):
                        compressed = pattern.sub("", compressed)
                        removed.append(f"filler: {filler}")

        # Apply verbose pattern substitutions
        for pattern, replacement in self.VERBOSE_PATTERNS:
            if re.search(pattern, compressed, re.IGNORECASE):
                compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)
                removed.append(f"verbose: {pattern} -> {replacement}")

        # Remove redundant whitespace
        compressed = re.sub(r"\s+", " ", compressed)
        compressed = compressed.strip()

        # Remove empty parentheses and brackets
        compressed = re.sub(r"\(\s*\)", "", compressed)
        compressed = re.sub(r"\[\s*\]", "", compressed)

        # Track preserved elements
        for keyword in preserve_keywords:
            if keyword.lower() in compressed.lower():
                preserved.append(keyword)

        compressed_tokens = self._estimate_tokens(compressed)
        ratio = 1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0

        return PromptCompressionResult(
            original=prompt,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            removed_elements=removed,
            preserved_elements=preserved,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate the token count of a text string.

        Uses a rough approximation of ~4 characters per token, which is
        reasonably accurate for English text with typical LLM tokenizers.

        Parameters
        ----------
        text : str
            The text to estimate token count for.

        Returns
        -------
        int
            Estimated number of tokens. Always returns at least 1 for
            non-empty input.

        Notes
        -----
        This is an approximation. For production use with specific models,
        consider using the model's actual tokenizer (e.g., tiktoken for
        OpenAI models).

        Examples
        --------
        >>> compressor = PromptCompressor()
        >>> compressor._estimate_tokens("Hello world")
        3
        >>> compressor._estimate_tokens("A" * 100)
        26
        """
        # Rough approximation: ~4 characters per token
        return len(text) // 4 + 1


class InstructionOptimizer:
    """Optimize instruction clarity and effectiveness for LLM prompts.

    This class analyzes instructions and optimizes them by:
    1. Replacing weak verbs with stronger alternatives
    2. Flagging ambiguous terms that need clarification
    3. Ensuring proper punctuation and structure
    4. Checking for sufficient context and detail

    The optimizer is designed to help create clearer, more actionable
    instructions that lead to better LLM responses.

    Attributes
    ----------
    WEAK_VERBS : list of str
        Class-level list of weak verb phrases that should be strengthened.
    STRONG_VERBS : dict of str to str
        Class-level mapping of weak verbs to their stronger alternatives.
    AMBIGUOUS_TERMS : list of str
        Class-level list of vague terms that should be flagged for
        clarification.

    Examples
    --------
    Basic instruction optimization:

    >>> from insideLLMs.optimization import InstructionOptimizer
    >>> optimizer = InstructionOptimizer()
    >>> optimized, changes = optimizer.optimize(
    ...     "Try to write good code"
    ... )
    >>> print(optimized)
    'Ensure write good code.'
    >>> for change in changes:
    ...     print(f"  - {change}")
      - Strengthened verb: 'try to' -> 'ensure'
      - Consider clarifying ambiguous term: 'good'

    Analyzing instruction clarity:

    >>> analysis = optimizer.analyze_clarity(
    ...     "Consider making appropriate changes to the code"
    ... )
    >>> print(f"Clarity score: {analysis['score']:.2f}")
    >>> for issue in analysis['issues']:
    ...     print(f"  Issue: {issue}")

    Optimizing multiple instructions:

    >>> instructions = [
    ...     "Try to analyze the data",
    ...     "Think about the implications",
    ...     "Attempt to fix the bug"
    ... ]
    >>> for instruction in instructions:
    ...     opt, _ = optimizer.optimize(instruction)
    ...     print(f"Original: {instruction}")
    ...     print(f"Optimized: {opt}")
    ...     print()

    See Also
    --------
    optimize_instruction : Convenience function for one-off optimization.
    PromptOptimizer : Uses this as part of comprehensive optimization.
    OptimizationStrategy.CLARITY : Strategy enum value for clarity.

    Notes
    -----
    - The optimizer focuses on common patterns; domain-specific jargon
      may need additional handling.
    - Ambiguous terms are flagged but not automatically replaced, as
      the appropriate replacement depends on context.
    """

    # Weak instruction verbs
    WEAK_VERBS = ["try to", "attempt to", "consider", "think about", "maybe"]

    # Strong instruction verbs
    STRONG_VERBS = {
        "try to": "ensure",
        "attempt to": "make sure to",
        "consider": "evaluate",
        "think about": "analyze",
    }

    # Ambiguous terms
    AMBIGUOUS_TERMS = [
        "good",
        "bad",
        "nice",
        "proper",
        "appropriate",
        "reasonable",
        "suitable",
        "adequate",
    ]

    def optimize(self, instruction: str) -> tuple[str, list[str]]:
        """Optimize an instruction for clarity and effectiveness.

        This method transforms an instruction by replacing weak verbs,
        flagging ambiguous terms, ensuring proper punctuation, and
        checking for sufficient context.

        Parameters
        ----------
        instruction : str
            The original instruction text to optimize.

        Returns
        -------
        tuple of (str, list of str)
            A tuple containing:
            - optimized : str - The optimized instruction text
            - changes : list of str - List of changes made or suggestions

        Examples
        --------
        Strengthening weak verbs:

        >>> optimizer = InstructionOptimizer()
        >>> opt, changes = optimizer.optimize("Try to analyze the data")
        >>> opt
        'Ensure analyze the data.'
        >>> "Strengthened verb: 'try to' -> 'ensure'" in changes
        True

        Flagging ambiguous terms:

        >>> opt, changes = optimizer.optimize("Write good documentation")
        >>> any("good" in c for c in changes)
        True

        Handling brief instructions:

        >>> opt, changes = optimizer.optimize("Do it")
        >>> any("too brief" in c for c in changes)
        True

        Adding punctuation:

        >>> opt, changes = optimizer.optimize("Analyze the data")
        >>> opt.endswith(".")
        True

        Multiple optimizations:

        >>> opt, changes = optimizer.optimize(
        ...     "Try to think about appropriate solutions"
        ... )
        >>> len(changes) > 1  # Multiple issues found
        True

        See Also
        --------
        analyze_clarity : Get detailed clarity analysis without changes.
        optimize_instruction : Convenience function for this method.
        """
        changes = []
        optimized = instruction

        # Replace weak verbs
        for weak, strong in self.STRONG_VERBS.items():
            if weak in optimized.lower():
                pattern = re.compile(re.escape(weak), re.IGNORECASE)
                optimized = pattern.sub(strong, optimized)
                changes.append(f"Strengthened verb: '{weak}' -> '{strong}'")

        # Flag ambiguous terms
        for term in self.AMBIGUOUS_TERMS:
            if re.search(rf"\b{term}\b", optimized, re.IGNORECASE):
                changes.append(f"Consider clarifying ambiguous term: '{term}'")

        # Ensure instruction ends with clear action
        if not optimized.strip().endswith((".", ":", "?")):
            optimized = optimized.strip() + "."
            changes.append("Added ending punctuation")

        # Check for missing context
        if len(optimized.split()) < 5:
            changes.append("Instruction may be too brief - consider adding context")

        return optimized, changes

    def analyze_clarity(self, instruction: str) -> dict[str, Any]:
        """Analyze the clarity of an instruction without modifying it.

        This method provides a detailed analysis of instruction clarity,
        including a quantitative score and a list of specific issues.
        Unlike `optimize()`, this method does not modify the instruction.

        Parameters
        ----------
        instruction : str
            The instruction text to analyze.

        Returns
        -------
        dict of str to Any
            A dictionary containing:
            - score : float - Clarity score from 0.0 (poor) to 1.0 (excellent)
            - issues : list of str - Specific clarity issues found
            - word_count : int - Number of words in the instruction
            - has_action_verb : bool - Whether instruction starts with action verb

        Examples
        --------
        Analyzing a clear instruction:

        >>> optimizer = InstructionOptimizer()
        >>> analysis = optimizer.analyze_clarity(
        ...     "Write a Python function that calculates the factorial of a number"
        ... )
        >>> analysis["score"] > 0.8
        True
        >>> analysis["has_action_verb"]
        True

        Analyzing a problematic instruction:

        >>> analysis = optimizer.analyze_clarity(
        ...     "Maybe try to do something good"
        ... )
        >>> analysis["score"] < 0.7
        True
        >>> len(analysis["issues"]) > 0
        True

        Analyzing instruction length:

        >>> analysis = optimizer.analyze_clarity("Go")
        >>> "Too brief" in analysis["issues"]
        True
        >>> analysis["word_count"]
        1

        Checking for action verb:

        >>> analysis = optimizer.analyze_clarity(
        ...     "The data should be analyzed carefully"
        ... )
        >>> analysis["has_action_verb"]
        False
        >>> "Consider starting with an action verb" in analysis["issues"]
        True

        See Also
        --------
        optimize : Analyze and transform instruction.
        """
        issues = []
        score = 1.0

        # Check for weak verbs
        weak_count = sum(1 for v in self.WEAK_VERBS if v in instruction.lower())
        if weak_count > 0:
            issues.append(f"Contains {weak_count} weak verbs")
            score -= weak_count * 0.1

        # Check for ambiguous terms
        ambiguous_count = sum(
            1 for t in self.AMBIGUOUS_TERMS if re.search(rf"\b{t}\b", instruction, re.IGNORECASE)
        )
        if ambiguous_count > 0:
            issues.append(f"Contains {ambiguous_count} ambiguous terms")
            score -= ambiguous_count * 0.05

        # Check length
        word_count = len(instruction.split())
        if word_count < 5:
            issues.append("Too brief")
            score -= 0.2
        elif word_count > 100:
            issues.append("May be too long")
            score -= 0.1

        # Check for action verb at start
        first_word = instruction.split()[0].lower() if instruction.split() else ""
        action_verbs = [
            "write",
            "create",
            "generate",
            "analyze",
            "explain",
            "describe",
            "list",
            "provide",
            "summarize",
            "identify",
            "compare",
            "evaluate",
        ]
        if first_word not in action_verbs:
            issues.append("Consider starting with an action verb")
            score -= 0.05

        return {
            "score": max(0.0, min(1.0, score)),
            "issues": issues,
            "word_count": word_count,
            "has_action_verb": first_word in action_verbs,
        }


class FewShotSelector:
    """Select optimal few-shot examples for in-context learning.

    This class implements a sophisticated example selection algorithm that
    balances multiple factors:
    1. Relevance - How similar the example is to the target query
    2. Diversity - How different selected examples are from each other
    3. Quality - Intrinsic quality of the example (length, completeness)

    The selection algorithm works greedily:
    - First example: Most relevant to the query
    - Subsequent examples: Balance relevance and diversity from selected set

    Parameters
    ----------
    diversity_weight : float, default=0.3
        Weight given to diversity in the combined scoring (0.0-1.0).
        Higher values prefer more diverse example sets over pure relevance.
        Relevance weight is (1 - diversity_weight).

    Attributes
    ----------
    diversity_weight : float
        The configured diversity weight.

    Examples
    --------
    Basic example selection:

    >>> from insideLLMs.optimization import FewShotSelector
    >>> selector = FewShotSelector()
    >>> examples = [
    ...     {"input": "Translate hello", "output": "hola"},
    ...     {"input": "Translate goodbye", "output": "adios"},
    ...     {"input": "Summarize text", "output": "Summary..."},
    ... ]
    >>> result = selector.select(
    ...     query="Translate good morning",
    ...     examples=examples,
    ...     n=2
    ... )
    >>> len(result.selected_examples)
    2

    Emphasizing diversity:

    >>> diverse_selector = FewShotSelector(diversity_weight=0.6)
    >>> result = diverse_selector.select(
    ...     query="Write Python code",
    ...     examples=[...],
    ...     n=3
    ... )

    Emphasizing relevance:

    >>> focused_selector = FewShotSelector(diversity_weight=0.1)
    >>> result = focused_selector.select(
    ...     query="Translate to Spanish",
    ...     examples=[...],
    ...     n=3
    ... )

    Using custom keys:

    >>> examples_custom = [
    ...     {"question": "What is 2+2?", "answer": "4"},
    ...     {"question": "What is 3*3?", "answer": "9"},
    ... ]
    >>> result = selector.select(
    ...     query="What is 5+5?",
    ...     examples=examples_custom,
    ...     n=1,
    ...     input_key="question",
    ...     output_key="answer"
    ... )

    See Also
    --------
    select_examples : Convenience function for one-off selection.
    ExampleSelectionResult : Container for selection results.
    ExampleScore : Individual example scoring breakdown.

    Notes
    -----
    - Relevance is calculated using word overlap (Jaccard-like) after
      removing common stop words
    - Diversity is calculated as 1 - max(similarity to selected examples)
    - Quality considers output length and completeness
    - For semantic similarity, consider using embeddings instead
    """

    def __init__(self, diversity_weight: float = 0.3):
        """Initialize a new FewShotSelector instance.

        Parameters
        ----------
        diversity_weight : float, default=0.3
            Weight for diversity in the combined selection score.
            Must be between 0.0 and 1.0.
            - 0.0: Pure relevance-based selection
            - 0.5: Equal weight to relevance and diversity
            - 1.0: Pure diversity-based selection (after first example)

        Examples
        --------
        Default initialization:

        >>> selector = FewShotSelector()
        >>> selector.diversity_weight
        0.3

        High diversity preference:

        >>> selector = FewShotSelector(diversity_weight=0.7)
        >>> selector.diversity_weight
        0.7

        Raises
        ------
        No explicit validation is performed, but values outside [0, 1]
        may produce unexpected results.
        """
        self.diversity_weight = diversity_weight

    def select(
        self,
        query: str,
        examples: list[dict[str, str]],
        n: int = 3,
        input_key: str = "input",
        output_key: str = "output",
    ) -> ExampleSelectionResult:
        """Select optimal examples for a query using relevance-diversity tradeoff.

        This method implements a greedy selection algorithm that picks examples
        one at a time, balancing relevance to the query with diversity from
        already-selected examples.

        Parameters
        ----------
        query : str
            The target query/task to select examples for. Examples with
            higher word overlap (after stop word removal) are considered
            more relevant.
        examples : list of dict of str to str
            Pool of available examples to select from. Each example should
            be a dictionary with at least the input and output keys.
        n : int, default=3
            Maximum number of examples to select. May return fewer if
            the pool is smaller.
        input_key : str, default="input"
            Key in example dictionaries containing the input text.
        output_key : str, default="output"
            Key in example dictionaries containing the output text.

        Returns
        -------
        ExampleSelectionResult
            A result object containing:
            - query: The original query
            - selected_examples: List of selected example dictionaries
            - example_scores: Detailed scores for each selected example
            - coverage_score: How well examples cover query concepts
            - diversity_score: Average diversity among selected examples

        Examples
        --------
        Basic selection:

        >>> selector = FewShotSelector()
        >>> examples = [
        ...     {"input": "Translate: hello", "output": "hola"},
        ...     {"input": "Translate: world", "output": "mundo"},
        ...     {"input": "Summarize: long text", "output": "short summary"},
        ... ]
        >>> result = selector.select(
        ...     query="Translate: goodbye",
        ...     examples=examples,
        ...     n=2
        ... )
        >>> len(result.selected_examples) == 2
        True
        >>> result.selected_examples[0]["input"]  # Most relevant first
        'Translate: hello'

        Empty pool handling:

        >>> result = selector.select("query", [], n=3)
        >>> result.selected_examples
        []
        >>> result.coverage_score
        0.0

        Custom keys:

        >>> qa_examples = [
        ...     {"question": "What is AI?", "response": "Artificial Intelligence..."},
        ...     {"question": "What is ML?", "response": "Machine Learning..."},
        ... ]
        >>> result = selector.select(
        ...     query="What is DL?",
        ...     examples=qa_examples,
        ...     n=1,
        ...     input_key="question",
        ...     output_key="response"
        ... )

        Analyzing selection quality:

        >>> result = selector.select(query, examples, n=3)
        >>> print(f"Coverage: {result.coverage_score:.2f}")
        >>> print(f"Diversity: {result.diversity_score:.2f}")
        >>> for i, score in enumerate(result.example_scores):
        ...     print(f"Example {i}: relevance={score.relevance_score:.2f}, "
        ...           f"diversity={score.diversity_score:.2f}")

        See Also
        --------
        select_examples : Convenience function for this method.
        ExampleSelectionResult : Detailed result documentation.
        """
        if not examples:
            return ExampleSelectionResult(
                query=query,
                selected_examples=[],
                example_scores=[],
                coverage_score=0.0,
                diversity_score=0.0,
            )

        # Score each example
        scored_examples = []
        for example in examples:
            relevance = self._calculate_relevance(query, example.get(input_key, ""))
            quality = self._calculate_quality(
                example.get(input_key, ""),
                example.get(output_key, ""),
            )

            scored_examples.append(
                {
                    "example": example,
                    "relevance": relevance,
                    "quality": quality,
                }
            )

        # Sort by relevance initially
        scored_examples.sort(key=lambda x: x["relevance"], reverse=True)

        # Select with diversity
        selected = []
        selected_scores = []

        for i in range(min(n, len(scored_examples))):
            if i == 0:
                # First example: pick most relevant
                best = scored_examples[0]
            else:
                # Subsequent: balance relevance and diversity
                best = None
                best_score = -1

                for candidate in scored_examples:
                    if candidate["example"] in [s["example"] for s in selected]:
                        continue

                    diversity = self._calculate_diversity(
                        candidate["example"].get(input_key, ""),
                        [s["example"].get(input_key, "") for s in selected],
                    )

                    combined = (1 - self.diversity_weight) * candidate[
                        "relevance"
                    ] + self.diversity_weight * diversity

                    if combined > best_score:
                        best_score = combined
                        best = candidate
                        best["diversity"] = diversity

                if best is None:
                    break

            selected.append(best)

            diversity = best.get("diversity", 1.0) if i > 0 else 1.0
            overall = 0.4 * best["relevance"] + 0.3 * diversity + 0.3 * best["quality"]

            selected_scores.append(
                ExampleScore(
                    example=best["example"],
                    relevance_score=best["relevance"],
                    diversity_score=diversity,
                    quality_score=best["quality"],
                    overall_score=overall,
                )
            )

        # Calculate overall metrics
        coverage = self._calculate_coverage(query, [s["example"] for s in selected], input_key)
        avg_diversity = (
            sum(s.diversity_score for s in selected_scores) / len(selected_scores)
            if selected_scores
            else 0.0
        )

        return ExampleSelectionResult(
            query=query,
            selected_examples=[s["example"] for s in selected],
            example_scores=selected_scores,
            coverage_score=coverage,
            diversity_score=avg_diversity,
        )

    def _calculate_relevance(self, query: str, example_input: str) -> float:
        """Calculate relevance of an example to the query.

        Uses word overlap (after stop word removal) to measure relevance.
        This is a simple but effective heuristic for many use cases.

        Parameters
        ----------
        query : str
            The target query text.
        example_input : str
            The input text of the example.

        Returns
        -------
        float
            Relevance score from 0.0 (no overlap) to 1.0 (complete overlap).
            Returns 0.5 if query has no content words after stop word removal.

        Examples
        --------
        >>> selector = FewShotSelector()
        >>> selector._calculate_relevance("translate hello", "translate goodbye")
        0.5
        >>> selector._calculate_relevance("write code", "write code")
        1.0
        """
        query_words = set(query.lower().split())
        example_words = set(example_input.lower().split())

        # Remove stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "to",
            "of",
            "and",
            "in",
            "that",
            "it",
            "for",
        }
        query_words -= stop_words
        example_words -= stop_words

        if not query_words:
            return 0.5

        overlap = len(query_words & example_words)
        return overlap / len(query_words)

    def _calculate_diversity(self, candidate: str, selected: list[str]) -> float:
        """Calculate how different a candidate is from selected examples.

        Uses Jaccard distance (1 - Jaccard similarity) to measure diversity.
        Returns the minimum distance to any selected example (i.e., how
        different from the most similar selected example).

        Parameters
        ----------
        candidate : str
            The candidate example input text.
        selected : list of str
            List of input texts from already-selected examples.

        Returns
        -------
        float
            Diversity score from 0.0 (identical to an existing example)
            to 1.0 (completely different from all selected examples).
            Returns 1.0 if no examples have been selected yet.

        Examples
        --------
        >>> selector = FewShotSelector()
        >>> selector._calculate_diversity("hello world", [])
        1.0
        >>> selector._calculate_diversity("hello world", ["hello world"])
        0.0
        >>> selector._calculate_diversity("hello world", ["goodbye moon"])
        1.0
        """
        if not selected:
            return 1.0

        candidate_words = set(candidate.lower().split())

        similarities = []
        for sel in selected:
            sel_words = set(sel.lower().split())
            if candidate_words | sel_words:
                sim = len(candidate_words & sel_words) / len(candidate_words | sel_words)
                similarities.append(sim)

        if not similarities:
            return 1.0

        # Diversity is inverse of max similarity
        return 1.0 - max(similarities)

    def _calculate_quality(self, input_text: str, output_text: str) -> float:
        """Calculate the intrinsic quality of an example.

        Evaluates quality based on:
        - Output length (at least 3 words)
        - Output/input length ratio (output should be substantial)
        - Output completeness (proper ending punctuation)

        Parameters
        ----------
        input_text : str
            The input text of the example.
        output_text : str
            The output text of the example.

        Returns
        -------
        float
            Quality score from 0.0 to 1.0. Base score is 0.5, with
            bonuses for meeting quality criteria.

        Examples
        --------
        >>> selector = FewShotSelector()
        >>> selector._calculate_quality("What is AI?", "Artificial Intelligence.")
        0.85
        >>> selector._calculate_quality("Long question here", "OK")
        0.5
        """
        score = 0.5

        # Check output length is reasonable relative to input
        input_len = len(input_text.split())
        output_len = len(output_text.split())

        if output_len >= 3:
            score += 0.2

        # Check output is not too short
        if output_len >= input_len * 0.5:
            score += 0.15

        # Check output is complete (ends properly)
        if output_text.strip().endswith((".", "!", "?", "```")):
            score += 0.15

        return min(1.0, score)

    def _calculate_coverage(
        self,
        query: str,
        examples: list[dict[str, str]],
        input_key: str,
    ) -> float:
        """Calculate how well selected examples cover query concepts.

        Measures the fraction of content words in the query that appear
        in at least one of the selected examples.

        Parameters
        ----------
        query : str
            The target query text.
        examples : list of dict of str to str
            The selected example dictionaries.
        input_key : str
            Key for accessing input text in example dictionaries.

        Returns
        -------
        float
            Coverage score from 0.0 (no query words covered) to 1.0
            (all query words appear in at least one example).
            Returns 1.0 if query has no content words.

        Examples
        --------
        >>> selector = FewShotSelector()
        >>> examples = [{"input": "translate hello", "output": "hola"}]
        >>> selector._calculate_coverage("translate goodbye", examples, "input")
        0.5
        """
        query_words = set(query.lower().split())
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "to",
            "of",
            "and",
            "in",
            "that",
            "it",
            "for",
        }
        query_words -= stop_words

        if not query_words:
            return 1.0

        covered = set()
        for example in examples:
            example_words = set(example.get(input_key, "").lower().split())
            covered |= query_words & example_words

        return len(covered) / len(query_words)


class PromptAblator:
    """Perform ablation studies to identify essential prompt components.

    An ablation study systematically removes components from a prompt and
    measures the impact on quality. This helps identify:
    - Essential components that must be kept
    - Removable components that add tokens without value
    - Relative importance ranking of all components

    The ablator works by:
    1. Splitting the prompt into components (using a delimiter)
    2. Scoring the complete prompt
    3. For each component, scoring the prompt without it
    4. Calculating importance as the score drop when removed
    5. Classifying components as essential or removable

    Parameters
    ----------
    scorer : callable, optional
        Function that takes a prompt string and returns a quality score
        (float). Higher scores indicate better prompts. If not provided,
        a default heuristic scorer is used.

    Attributes
    ----------
    scorer : callable
        The scoring function used for ablation.

    Examples
    --------
    Basic ablation with default scorer:

    >>> from insideLLMs.optimization import PromptAblator
    >>> ablator = PromptAblator()
    >>> prompt = '''You are a helpful assistant.
    ...
    ... Always be accurate and truthful.
    ...
    ... If you don't know something, say so.
    ...
    ... Be concise in your responses.'''
    >>> result = ablator.ablate(prompt)
    >>> print(f"Components: {len(result.components)}")
    >>> print(f"Essential: {len(result.essential_components)}")
    >>> print(f"Removable: {len(result.removable_components)}")

    Using a custom scorer:

    >>> def my_scorer(prompt: str) -> float:
    ...     # Custom scoring logic
    ...     return len(prompt.split()) / 100  # Simple word count score
    >>> ablator = PromptAblator(scorer=my_scorer)
    >>> result = ablator.ablate(prompt)

    Viewing importance rankings:

    >>> for component, importance in result.importance_ranking[:3]:
    ...     print(f"Importance {importance:.2f}: {component}")

    Using the minimal prompt:

    >>> print("Minimal prompt (essential components only):")
    >>> print(result.minimal_prompt)

    Custom delimiter for different prompt structures:

    >>> prompt_with_sections = "Section 1\\n---\\nSection 2\\n---\\nSection 3"
    >>> result = ablator.ablate(prompt_with_sections, component_delimiter="\\n---\\n")

    See Also
    --------
    ablate_prompt : Convenience function for ablation.
    AblationResult : Container for ablation results.

    Notes
    -----
    - The default scorer uses heuristics (length, structure, punctuation)
    - For production use, consider using actual LLM evaluation as the scorer
    - Components causing < 10% quality drop are considered removable
    """

    def __init__(self, scorer: Optional[Callable[[str], float]] = None):
        """Initialize a new PromptAblator instance.

        Parameters
        ----------
        scorer : callable, optional
            A function with signature (prompt: str) -> float that returns
            a quality score for the prompt. Higher scores are better.
            If not provided, a default heuristic scorer is used that
            considers length, structure, and formatting.

        Examples
        --------
        Default scorer:

        >>> ablator = PromptAblator()

        Custom scorer using LLM evaluation:

        >>> def llm_scorer(prompt: str) -> float:
        ...     # Call LLM to evaluate prompt quality
        ...     response = evaluate_with_llm(prompt)
        ...     return response.score
        >>> ablator = PromptAblator(scorer=llm_scorer)

        Lambda scorer:

        >>> ablator = PromptAblator(scorer=lambda p: len(p.split()) * 0.01)
        """
        self.scorer = scorer or self._default_scorer

    def ablate(
        self,
        prompt: str,
        component_delimiter: str = "\n\n",
    ) -> AblationResult:
        """Perform an ablation study on a prompt.

        Systematically removes each component and measures the impact on
        the prompt's quality score to determine component importance.

        Parameters
        ----------
        prompt : str
            The prompt to analyze. Will be split into components using
            the specified delimiter.
        component_delimiter : str, default="\\n\\n"
            The delimiter used to split the prompt into components.
            Common choices:
            - "\\n\\n" for paragraph-separated prompts
            - "\\n" for line-separated prompts
            - "---" for section-separated prompts

        Returns
        -------
        AblationResult
            A result object containing:
            - original_prompt: The input prompt
            - components: List of identified components
            - component_scores: Importance score for each component
            - essential_components: Components that must be kept
            - removable_components: Components that can be removed
            - minimal_prompt: Prompt with only essential components
            - importance_ranking: Components ranked by importance

        Examples
        --------
        Basic ablation:

        >>> ablator = PromptAblator()
        >>> result = ablator.ablate('''You are a helpful AI.
        ...
        ... Always be truthful and accurate.
        ...
        ... Respond in a friendly tone.''')
        >>> len(result.components)
        3

        Single-component prompt:

        >>> result = ablator.ablate("Just one component here")
        >>> len(result.essential_components)
        1
        >>> result.minimal_prompt == "Just one component here"
        True

        Line-delimited ablation:

        >>> result = ablator.ablate(
        ...     "Line 1\\nLine 2\\nLine 3",
        ...     component_delimiter="\\n"
        ... )
        >>> len(result.components)
        3

        Analyzing results:

        >>> result = ablator.ablate(prompt)
        >>> print(f"Can remove {len(result.removable_components)} components")
        >>> print(f"Must keep {len(result.essential_components)} components")
        >>> print(f"Minimal prompt saves "
        ...       f"{len(result.original_prompt) - len(result.minimal_prompt)} chars")

        See Also
        --------
        ablate_prompt : Convenience function for this method.
        AblationResult : Detailed result documentation.
        """
        # Split into components
        components = [c.strip() for c in prompt.split(component_delimiter) if c.strip()]

        if len(components) <= 1:
            return AblationResult(
                original_prompt=prompt,
                components=components,
                component_scores={components[0] if components else "": 1.0},
                essential_components=components,
                removable_components=[],
                minimal_prompt=prompt,
                importance_ranking=[(components[0] if components else "", 1.0)],
            )

        # Score full prompt
        full_score = self.scorer(prompt)

        # Score with each component removed
        component_scores = {}
        for i, component in enumerate(components):
            # Create prompt without this component
            remaining = [c for j, c in enumerate(components) if j != i]
            ablated_prompt = component_delimiter.join(remaining)

            ablated_score = self.scorer(ablated_prompt)

            # Importance = drop in score when removed
            importance = full_score - ablated_score
            component_scores[component[:50] + "..."] = importance

        # Rank by importance
        importance_ranking = sorted(
            component_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Identify essential vs removable
        essential = []
        removable = []
        threshold = 0.1  # Components causing < 10% drop are removable

        for component, importance in importance_ranking:
            if importance > threshold * full_score:
                essential.append(component)
            else:
                removable.append(component)

        # Create minimal prompt from essential components only
        essential_full = [c for c in components if c[:50] + "..." in essential]
        minimal_prompt = component_delimiter.join(essential_full)

        return AblationResult(
            original_prompt=prompt,
            components=components,
            component_scores=component_scores,
            essential_components=essential,
            removable_components=removable,
            minimal_prompt=minimal_prompt,
            importance_ranking=importance_ranking,
        )

    def _default_scorer(self, prompt: str) -> float:
        """Default scorer based on heuristics."""
        score = 0.5

        # Length score
        word_count = len(prompt.split())
        if 10 <= word_count <= 200:
            score += 0.2
        elif word_count > 200:
            score += 0.1

        # Structure score
        if "\n" in prompt:
            score += 0.1
        if any(c in prompt for c in ["1.", "2.", "-", "•"]):
            score += 0.1

        # Clarity score
        if prompt.strip().endswith((".", "?", ":")):
            score += 0.1

        return min(1.0, score)


class TokenBudgetOptimizer:
    """Optimize prompts within token budgets."""

    def __init__(self, max_tokens: int = 4096):
        """Initialize optimizer.

        Args:
            max_tokens: Maximum token budget.
        """
        self.max_tokens = max_tokens
        self.compressor = PromptCompressor()

    def optimize(
        self,
        prompt: str,
        examples: Optional[list[dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        reserve_for_response: int = 500,
    ) -> dict[str, Any]:
        """Optimize prompt to fit within token budget.

        Args:
            prompt: Main prompt.
            examples: Few-shot examples.
            system_prompt: System prompt.
            reserve_for_response: Tokens to reserve for response.

        Returns:
            Optimization result.
        """
        available_tokens = self.max_tokens - reserve_for_response

        # Estimate current usage
        prompt_tokens = self._estimate_tokens(prompt)
        system_tokens = self._estimate_tokens(system_prompt) if system_prompt else 0
        example_tokens = sum(self._estimate_tokens(str(e)) for e in (examples or []))

        total_tokens = prompt_tokens + system_tokens + example_tokens
        over_budget = total_tokens > available_tokens

        result = {
            "original_tokens": total_tokens,
            "available_tokens": available_tokens,
            "over_budget": over_budget,
            "actions_taken": [],
        }

        if not over_budget:
            result["final_prompt"] = prompt
            result["final_examples"] = examples
            result["final_system"] = system_prompt
            result["final_tokens"] = total_tokens
            return result

        # Strategy 1: Compress prompt
        compressed = self.compressor.compress(prompt, target_reduction=0.3)
        prompt = compressed.compressed
        prompt_tokens = compressed.compressed_tokens
        result["actions_taken"].append(f"Compressed prompt: saved {compressed.tokens_saved} tokens")

        # Strategy 2: Reduce examples if still over
        total_tokens = prompt_tokens + system_tokens + example_tokens
        if total_tokens > available_tokens and examples:
            # Remove examples one by one
            while examples and total_tokens > available_tokens:
                examples = examples[:-1]
                example_tokens = sum(self._estimate_tokens(str(e)) for e in examples)
                total_tokens = prompt_tokens + system_tokens + example_tokens
            result["actions_taken"].append(f"Reduced to {len(examples)} examples")

        # Strategy 3: Truncate prompt if still over
        if total_tokens > available_tokens:
            excess = total_tokens - available_tokens
            chars_to_remove = excess * 4  # Rough conversion
            prompt = prompt[:-chars_to_remove] + "..."
            prompt_tokens = self._estimate_tokens(prompt)
            result["actions_taken"].append("Truncated prompt")

        result["final_prompt"] = prompt
        result["final_examples"] = examples
        result["final_system"] = system_prompt
        result["final_tokens"] = prompt_tokens + system_tokens + example_tokens

        return result

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        if not text:
            return 0
        return len(text) // 4 + 1


class PromptOptimizer:
    """Comprehensive prompt optimizer."""

    def __init__(self):
        """Initialize optimizer."""
        self.compressor = PromptCompressor()
        self.instruction_optimizer = InstructionOptimizer()

    def optimize(
        self,
        prompt: str,
        strategies: Optional[list[OptimizationStrategy]] = None,
    ) -> OptimizationReport:
        """Optimize a prompt using specified strategies.

        Args:
            prompt: Prompt to optimize.
            strategies: Strategies to apply (default: all).

        Returns:
            Optimization report.
        """
        strategies = strategies or [
            OptimizationStrategy.COMPRESSION,
            OptimizationStrategy.CLARITY,
            OptimizationStrategy.STRUCTURE,
        ]

        optimized = prompt
        improvements = {}
        suggestions = []
        applied_strategies = []

        # Apply compression
        if OptimizationStrategy.COMPRESSION in strategies:
            compression = self.compressor.compress(optimized)
            if compression.compression_ratio > 0.05:
                optimized = compression.compressed
                improvements["compression"] = compression.compression_ratio
                applied_strategies.append(OptimizationStrategy.COMPRESSION)

        # Apply clarity optimization
        if OptimizationStrategy.CLARITY in strategies:
            clarity_result = self.instruction_optimizer.analyze_clarity(optimized)
            if clarity_result["issues"]:
                suggestions.extend(clarity_result["issues"])
                optimized_text, changes = self.instruction_optimizer.optimize(optimized)
                if changes:
                    optimized = optimized_text
                    improvements["clarity"] = clarity_result["score"]
                    applied_strategies.append(OptimizationStrategy.CLARITY)

        # Apply structure optimization
        if OptimizationStrategy.STRUCTURE in strategies:
            structure_improved, structure_changes = self._optimize_structure(optimized)
            if structure_changes:
                optimized = structure_improved
                improvements["structure"] = 0.1 * len(structure_changes)
                suggestions.extend(structure_changes)
                applied_strategies.append(OptimizationStrategy.STRUCTURE)

        original_tokens = len(prompt) // 4 + 1
        optimized_tokens = len(optimized) // 4 + 1
        token_reduction = original_tokens - optimized_tokens

        # Estimate quality change based on improvements
        quality_change = sum(improvements.values()) / len(improvements) if improvements else 0

        return OptimizationReport(
            original_prompt=prompt,
            optimized_prompt=optimized,
            strategies_applied=applied_strategies,
            improvements=improvements,
            suggestions=suggestions,
            token_reduction=token_reduction,
            estimated_quality_change=quality_change,
        )

    def _optimize_structure(self, prompt: str) -> tuple[str, list[str]]:
        """Optimize prompt structure."""
        changes = []
        optimized = prompt

        # Add newlines before lists if missing
        if re.search(r"[^\n][-•*\d+.]\s", optimized):
            optimized = re.sub(r"([^\n])([-•*\d+.])\s", r"\1\n\2 ", optimized)
            changes.append("Added line breaks before list items")

        # Ensure consistent list formatting
        if re.search(r"^\d+\)", optimized, re.MULTILINE):
            optimized = re.sub(r"^(\d+)\)", r"\1.", optimized, flags=re.MULTILINE)
            changes.append("Standardized list numbering")

        return optimized, changes


# Convenience functions


def compress_prompt(
    prompt: str,
    target_reduction: float = 0.2,
    preserve_keywords: Optional[set[str]] = None,
) -> PromptCompressionResult:
    """Compress a prompt.

    Args:
        prompt: Original prompt.
        target_reduction: Target reduction (0-1).
        preserve_keywords: Keywords to preserve.

    Returns:
        Compression result.
    """
    compressor = PromptCompressor()
    return compressor.compress(prompt, target_reduction, preserve_keywords)


def optimize_instruction(instruction: str) -> tuple[str, list[str]]:
    """Optimize an instruction.

    Args:
        instruction: Original instruction.

    Returns:
        Tuple of (optimized, changes).
    """
    optimizer = InstructionOptimizer()
    return optimizer.optimize(instruction)


def select_examples(
    query: str,
    examples: list[dict[str, str]],
    n: int = 3,
    input_key: str = "input",
    output_key: str = "output",
) -> ExampleSelectionResult:
    """Select optimal few-shot examples.

    Args:
        query: Query to select for.
        examples: Pool of examples.
        n: Number to select.
        input_key: Key for input.
        output_key: Key for output.

    Returns:
        Selection result.
    """
    selector = FewShotSelector()
    return selector.select(query, examples, n, input_key, output_key)


def ablate_prompt(
    prompt: str,
    component_delimiter: str = "\n\n",
    scorer: Optional[Callable[[str], float]] = None,
) -> AblationResult:
    """Perform prompt ablation study.

    Args:
        prompt: Prompt to ablate.
        component_delimiter: Delimiter between components.
        scorer: Custom scorer function.

    Returns:
        Ablation result.
    """
    ablator = PromptAblator(scorer)
    return ablator.ablate(prompt, component_delimiter)


def optimize_prompt(
    prompt: str,
    strategies: Optional[list[OptimizationStrategy]] = None,
) -> OptimizationReport:
    """Optimize a prompt.

    Args:
        prompt: Prompt to optimize.
        strategies: Strategies to apply.

    Returns:
        Optimization report.
    """
    optimizer = PromptOptimizer()
    return optimizer.optimize(prompt, strategies)


def optimize_for_budget(
    prompt: str,
    max_tokens: int = 4096,
    examples: Optional[list[dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    reserve_for_response: int = 500,
) -> dict[str, Any]:
    """Optimize prompt for token budget.

    Args:
        prompt: Main prompt.
        max_tokens: Token budget.
        examples: Few-shot examples.
        system_prompt: System prompt.
        reserve_for_response: Reserve for response.

    Returns:
        Optimization result.
    """
    optimizer = TokenBudgetOptimizer(max_tokens)
    return optimizer.optimize(prompt, examples, system_prompt, reserve_for_response)


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import CompressionResult. The canonical name is
# PromptCompressionResult.
CompressionResult = PromptCompressionResult
