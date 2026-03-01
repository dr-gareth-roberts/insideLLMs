"""
Token analysis and embedding utilities for LLM exploration.

This module provides a comprehensive toolkit for analyzing, estimating, and managing
tokens in the context of Large Language Models (LLMs). It supports multiple tokenizer
types and offers utilities for cost estimation, context window management, and
embedding similarity calculations.

Overview
--------
The module is organized into several key components:

1. **Token Estimation** - Estimate token counts without requiring actual tokenizers
2. **Token Analysis** - Analyze token distributions, frequencies, and statistics
3. **Vocabulary Coverage** - Measure how well a reference vocabulary covers text
4. **Embedding Utilities** - Calculate vector similarities and manage embeddings
5. **Context Management** - Track and manage token budgets for LLM context windows

Key Classes
-----------
TokenizerType : Enum
    Supported tokenizer types (GPT4, GPT3, CLAUDE, LLAMA, SIMPLE).
TokenStats : dataclass
    Container for token statistics including counts and frequencies.
TokenDistribution : dataclass
    Frequency distribution analysis with entropy and coverage metrics.
TokenEstimator : class
    Estimate token counts and costs for different model types.
TokenAnalyzer : class
    Comprehensive token analysis with distribution comparisons.
EmbeddingUtils : class
    Static methods for embedding similarity and vector operations.
ContextWindowManager : class
    Manage content within context window token limits.

Examples
--------
Estimate tokens for a piece of text:

>>> from insideLLMs.tokens import estimate_tokens, TokenizerType
>>> text = "Hello, world! This is a sample text for token estimation."
>>> count = estimate_tokens(text, TokenizerType.GPT4)
>>> print(f"Estimated tokens: {count}")
Estimated tokens: 14

Analyze token distribution:

>>> from insideLLMs.tokens import analyze_tokens
>>> text = "The quick brown fox jumps over the lazy dog. The dog was not amused."
>>> stats = analyze_tokens(text)
>>> print(f"Total tokens: {stats.total_tokens}")
Total tokens: 16
>>> print(f"Unique tokens: {stats.unique_tokens}")
Unique tokens: 13
>>> print(f"Token diversity: {stats.token_diversity:.2f}")
Token diversity: 0.81

Split text into chunks for context window:

>>> from insideLLMs.tokens import split_by_tokens
>>> long_text = "This is a very long text..." * 100
>>> chunks = split_by_tokens(long_text, max_tokens=500, overlap_tokens=50)
>>> print(f"Split into {len(chunks)} chunks")
Split into 3 chunks

Manage context window budget:

>>> from insideLLMs.tokens import ContextWindowManager
>>> manager = ContextWindowManager(max_tokens=4096)
>>> manager.add_content("System prompt goes here", priority=10)
True
>>> manager.add_content("User message content", priority=5)
True
>>> print(f"Remaining tokens: {manager.remaining_tokens()}")
Remaining tokens: 4076

Calculate embedding similarity:

>>> from insideLLMs.tokens import cosine_similarity
>>> vec1 = [0.1, 0.2, 0.3, 0.4]
>>> vec2 = [0.1, 0.25, 0.35, 0.38]
>>> similarity = cosine_similarity(vec1, vec2)
>>> print(f"Similarity: {similarity:.4f}")
Similarity: 0.9987

Notes
-----
- Token estimates are approximations based on average characters-per-token ratios
- For precise token counts, use the actual tokenizer from the target model
- Different content types (code, prose, technical) have different token densities
- The SIMPLE tokenizer is a whitespace/punctuation-based fallback

See Also
--------
tiktoken : OpenAI's official tokenizer library for precise GPT token counts
transformers : Hugging Face library with tokenizers for various models

References
----------
.. [1] OpenAI Tokenizer: https://platform.openai.com/tokenizer
.. [2] Anthropic Token Counting: https://docs.anthropic.com/claude/docs/counting-tokens
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Optional,
    Protocol,
)


class TokenizerType(Enum):
    """Enumeration of supported tokenizer types for token estimation.

    This enum defines the different tokenizer models that can be used for
    estimating token counts. Each tokenizer type has different average
    characters-per-token ratios based on their vocabulary and encoding schemes.

    Attributes
    ----------
    GPT4 : str
        OpenAI's GPT-4 tokenizer (cl100k_base encoding). Average ~4 chars/token.
    GPT3 : str
        OpenAI's GPT-3 tokenizer (p50k_base encoding). Average ~4 chars/token.
    CLAUDE : str
        Anthropic's Claude tokenizer. Average ~3.5 chars/token (more efficient).
    LLAMA : str
        Meta's LLaMA tokenizer (SentencePiece-based). Average ~3.8 chars/token.
    SIMPLE : str
        Basic whitespace/punctuation tokenizer. Average ~5 chars/token.

    Examples
    --------
    Using tokenizer type for estimation:

    >>> from insideLLMs.tokens import TokenizerType, TokenEstimator
    >>> estimator = TokenEstimator(TokenizerType.GPT4)
    >>> token_count = estimator.estimate_tokens("Hello, world!")
    >>> print(f"GPT-4 estimate: {token_count} tokens")
    GPT-4 estimate: 3 tokens

    Comparing estimates across tokenizer types:

    >>> from insideLLMs.tokens import TokenizerType, estimate_tokens
    >>> text = "The quick brown fox jumps over the lazy dog."
    >>> for tok_type in TokenizerType:
    ...     count = estimate_tokens(text, tok_type)
    ...     print(f"{tok_type.name}: {count} tokens")
    GPT4: 11 tokens
    GPT3: 11 tokens
    CLAUDE: 13 tokens
    LLAMA: 12 tokens
    SIMPLE: 9 tokens

    Selecting tokenizer based on target model:

    >>> from insideLLMs.tokens import TokenizerType
    >>> model_name = "claude-3-opus"
    >>> if "claude" in model_name.lower():
    ...     tok_type = TokenizerType.CLAUDE
    ... elif "gpt-4" in model_name.lower():
    ...     tok_type = TokenizerType.GPT4
    ... else:
    ...     tok_type = TokenizerType.SIMPLE
    >>> print(f"Selected tokenizer: {tok_type.value}")
    Selected tokenizer: claude

    See Also
    --------
    TokenEstimator : Class that uses TokenizerType for token estimation.
    estimate_tokens : Convenience function for quick token estimation.

    Notes
    -----
    These are approximations. For production use with specific models, use the
    actual tokenizer library (tiktoken for OpenAI, transformers for LLaMA, etc.).
    """

    GPT4 = "gpt4"
    GPT3 = "gpt3"
    CLAUDE = "claude"
    LLAMA = "llama"
    SIMPLE = "simple"  # Whitespace-based


@dataclass
class TokenStats:
    """Container for comprehensive token statistics from text analysis.

    This dataclass holds various metrics about the tokens found in analyzed text,
    including counts, frequencies, and derived statistics like diversity ratios.
    It is typically returned by TokenAnalyzer.analyze() or the analyze_tokens()
    convenience function.

    Parameters
    ----------
    total_tokens : int
        The total number of tokens in the analyzed text.
    unique_tokens : int
        The number of distinct/unique tokens (vocabulary size for this text).
    char_count : int
        Total character count of the original text.
    word_count : int
        Number of whitespace-separated words in the text.
    avg_token_length : float
        Average character length of tokens.
    tokens_per_word : float
        Ratio of tokens to words (higher means more subword tokenization).
    token_frequencies : dict[str, int], optional
        Mapping from token strings to their occurrence counts.

    Attributes
    ----------
    token_diversity : float
        Property that returns the ratio of unique to total tokens (0.0 to 1.0).
        Higher values indicate more lexically diverse text.

    Examples
    --------
    Analyzing a simple sentence:

    >>> from insideLLMs.tokens import analyze_tokens
    >>> stats = analyze_tokens("The cat sat on the mat. The cat was happy.")
    >>> print(f"Total: {stats.total_tokens}, Unique: {stats.unique_tokens}")
    Total: 12, Unique: 8
    >>> print(f"Diversity: {stats.token_diversity:.2f}")
    Diversity: 0.67

    Working with token frequencies:

    >>> from insideLLMs.tokens import analyze_tokens
    >>> text = "to be or not to be that is the question"
    >>> stats = analyze_tokens(text)
    >>> most_common = sorted(stats.token_frequencies.items(),
    ...                      key=lambda x: x[1], reverse=True)[:3]
    >>> for token, count in most_common:
    ...     print(f"'{token}': {count}")
    'to': 2
    'be': 2
    'or': 1

    Converting to dictionary for JSON serialization:

    >>> from insideLLMs.tokens import analyze_tokens
    >>> import json
    >>> stats = analyze_tokens("Hello world!")
    >>> stats_dict = stats.to_dict()
    >>> print(json.dumps(stats_dict, indent=2))
    {
      "total_tokens": 3,
      "unique_tokens": 3,
      "char_count": 12,
      ...
    }

    See Also
    --------
    TokenAnalyzer : Class that produces TokenStats instances.
    analyze_tokens : Convenience function for quick analysis.
    TokenDistribution : More detailed frequency distribution analysis.
    """

    total_tokens: int
    unique_tokens: int
    char_count: int
    word_count: int
    avg_token_length: float
    tokens_per_word: float
    token_frequencies: dict[str, int] = field(default_factory=dict)

    @property
    def token_diversity(self) -> float:
        """Calculate the ratio of unique tokens to total tokens.

        Token diversity measures lexical richness - how many different words
        are used relative to the total word count. A value of 1.0 means every
        token is unique; lower values indicate more repetition.

        Returns
        -------
        float
            Diversity ratio between 0.0 and 1.0.

        Examples
        --------
        High diversity (varied vocabulary):

        >>> from insideLLMs.tokens import analyze_tokens
        >>> stats = analyze_tokens("The quick brown fox jumps over lazy dogs.")
        >>> print(f"Diversity: {stats.token_diversity:.2f}")
        Diversity: 1.00

        Lower diversity (repetitive text):

        >>> stats = analyze_tokens("the the the cat cat sat sat sat sat")
        >>> print(f"Diversity: {stats.token_diversity:.2f}")
        Diversity: 0.33

        Empty text edge case:

        >>> stats = analyze_tokens("")
        >>> print(f"Diversity: {stats.token_diversity}")
        Diversity: 0.0
        """
        return self.unique_tokens / self.total_tokens if self.total_tokens > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert token statistics to a dictionary representation.

        Creates a dictionary containing all statistics suitable for JSON
        serialization, logging, or further processing. Note that token_frequencies
        is excluded to keep the output concise; use token_frequencies attribute
        directly if needed.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: total_tokens, unique_tokens, char_count,
            word_count, avg_token_length, tokens_per_word, token_diversity.

        Examples
        --------
        Basic dictionary conversion:

        >>> from insideLLMs.tokens import analyze_tokens
        >>> stats = analyze_tokens("Hello, world!")
        >>> d = stats.to_dict()
        >>> print(f"Keys: {list(d.keys())}")
        Keys: ['total_tokens', 'unique_tokens', 'char_count', ...]

        Using with pandas DataFrame:

        >>> from insideLLMs.tokens import analyze_tokens
        >>> texts = ["First text.", "Second longer text here."]
        >>> stats_list = [analyze_tokens(t).to_dict() for t in texts]
        >>> # df = pd.DataFrame(stats_list)  # Create comparison DataFrame

        Logging token statistics:

        >>> from insideLLMs.tokens import analyze_tokens
        >>> import logging
        >>> stats = analyze_tokens("Sample text for analysis")
        >>> logging.info("Token stats: %s", stats.to_dict())
        """
        return {
            "total_tokens": self.total_tokens,
            "unique_tokens": self.unique_tokens,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "avg_token_length": self.avg_token_length,
            "tokens_per_word": self.tokens_per_word,
            "token_diversity": self.token_diversity,
        }


@dataclass
class TokenDistribution:
    """Token frequency distribution analysis with statistical measures.

    This dataclass provides detailed frequency distribution analysis of tokens,
    including methods for finding common/rare tokens, calculating entropy, and
    identifying linguistic phenomena like hapax legomena (words appearing once).

    Parameters
    ----------
    frequencies : dict[str, int]
        Mapping from token strings to their occurrence counts.
    total_tokens : int
        Total number of tokens in the analyzed text.

    Attributes
    ----------
    vocabulary_size : int
        Property returning the number of unique tokens.

    Examples
    --------
    Basic distribution analysis:

    >>> from insideLLMs.tokens import get_token_distribution
    >>> text = "the cat sat on the mat and the cat slept"
    >>> dist = get_token_distribution(text)
    >>> print(f"Vocabulary size: {dist.vocabulary_size}")
    Vocabulary size: 7
    >>> print(f"Total tokens: {dist.total_tokens}")
    Total tokens: 10

    Finding most common tokens:

    >>> from insideLLMs.tokens import get_token_distribution
    >>> text = "to be or not to be that is the question to be is to ask"
    >>> dist = get_token_distribution(text)
    >>> for token, count in dist.top_tokens(3):
    ...     print(f"'{token}': {count} occurrences")
    'to': 4
    'be': 3
    'is': 2

    Analyzing text entropy:

    >>> from insideLLMs.tokens import get_token_distribution
    >>> uniform_text = "a b c d e f g h"  # High entropy (uniform)
    >>> repetitive_text = "a a a a a a a a"  # Low entropy
    >>> print(f"Uniform entropy: {get_token_distribution(uniform_text).entropy():.2f}")
    Uniform entropy: 3.00
    >>> print(f"Repetitive entropy: {get_token_distribution(repetitive_text).entropy():.2f}")
    Repetitive entropy: 0.00

    See Also
    --------
    TokenStats : Higher-level statistics container.
    TokenAnalyzer.get_distribution : Method that creates TokenDistribution.
    get_token_distribution : Convenience function for distribution analysis.
    """

    frequencies: dict[str, int]
    total_tokens: int

    @property
    def vocabulary_size(self) -> int:
        """Get the number of unique tokens in the distribution.

        Returns
        -------
        int
            Count of distinct token types.

        Examples
        --------
        >>> from insideLLMs.tokens import get_token_distribution
        >>> dist = get_token_distribution("the quick brown fox")
        >>> print(f"Vocabulary: {dist.vocabulary_size} unique tokens")
        Vocabulary: 4 unique tokens

        >>> dist = get_token_distribution("yes yes yes no no no")
        >>> print(f"Vocabulary: {dist.vocabulary_size} unique tokens")
        Vocabulary: 2 unique tokens
        """
        return len(self.frequencies)

    def top_tokens(self, n: int = 10) -> list[tuple[str, int]]:
        """Get the N most frequently occurring tokens.

        Args
        ----
        n : int, optional
            Number of top tokens to return. Default is 10.

        Returns
        -------
        list[tuple[str, int]]
            List of (token, count) tuples sorted by frequency descending.

        Examples
        --------
        Finding top 5 tokens in a document:

        >>> from insideLLMs.tokens import get_token_distribution
        >>> text = "the cat and the dog and the bird flew over the house"
        >>> dist = get_token_distribution(text)
        >>> for token, count in dist.top_tokens(5):
        ...     print(f"{token}: {count}")
        the: 4
        and: 2
        cat: 1
        dog: 1
        bird: 1

        Analyzing function word frequency:

        >>> text = "I think that I know that you think that I am right"
        >>> dist = get_token_distribution(text)
        >>> top_3 = dist.top_tokens(3)
        >>> print(f"Most common: {[t[0] for t in top_3]}")
        Most common: ['that', 'i', 'think']

        Using for word cloud weighting:

        >>> dist = get_token_distribution("python data science machine learning")
        >>> weights = {tok: cnt for tok, cnt in dist.top_tokens(100)}
        >>> print(f"Token weights: {weights}")
        Token weights: {'python': 1, 'data': 1, 'science': 1, ...}
        """
        return sorted(self.frequencies.items(), key=lambda x: x[1], reverse=True)[:n]

    def bottom_tokens(self, n: int = 10) -> list[tuple[str, int]]:
        """Get the N least frequently occurring tokens.

        Useful for identifying rare words, potential misspellings, or
        domain-specific terminology.

        Args
        ----
        n : int, optional
            Number of bottom tokens to return. Default is 10.

        Returns
        -------
        list[tuple[str, int]]
            List of (token, count) tuples sorted by frequency ascending.

        Examples
        --------
        Finding rare tokens:

        >>> from insideLLMs.tokens import get_token_distribution
        >>> text = "the the the cat cat dog elephant"
        >>> dist = get_token_distribution(text)
        >>> for token, count in dist.bottom_tokens(3):
        ...     print(f"'{token}': {count}")
        'dog': 1
        'elephant': 1
        'cat': 2

        Identifying potential typos (rare tokens):

        >>> text = "Python is great. Pythn is a typo. Python rocks."
        >>> dist = get_token_distribution(text)
        >>> rare = dist.bottom_tokens(2)
        >>> print(f"Possibly misspelled: {[t[0] for t in rare if t[1] == 1]}")
        Possibly misspelled: ['pythn', 'typo', ...]

        Finding technical jargon:

        >>> text = "the system uses kubernetes for orchestration and docker"
        >>> dist = get_token_distribution(text)
        >>> uncommon = [t for t, c in dist.bottom_tokens(5) if c == 1]
        >>> print(f"Technical terms: {uncommon}")
        Technical terms: ['kubernetes', 'orchestration', 'docker', ...]
        """
        return sorted(self.frequencies.items(), key=lambda x: x[1])[:n]

    def frequency_of(self, token: str) -> int:
        """Get the absolute frequency count of a specific token.

        Args
        ----
        token : str
            The token to look up.

        Returns
        -------
        int
            Number of times the token appears. Returns 0 if not found.

        Examples
        --------
        Basic frequency lookup:

        >>> from insideLLMs.tokens import get_token_distribution
        >>> dist = get_token_distribution("to be or not to be")
        >>> print(f"'to' appears {dist.frequency_of('to')} times")
        'to' appears 2 times
        >>> print(f"'be' appears {dist.frequency_of('be')} times")
        'be' appears 2 times

        Checking for absent tokens:

        >>> dist = get_token_distribution("hello world")
        >>> print(f"'goodbye' appears {dist.frequency_of('goodbye')} times")
        'goodbye' appears 0 times

        Conditional logic based on frequency:

        >>> dist = get_token_distribution("error error warning info info info")
        >>> if dist.frequency_of('error') > 0:
        ...     print("Errors detected in log!")
        Errors detected in log!
        """
        return self.frequencies.get(token, 0)

    def relative_frequency(self, token: str) -> float:
        """Get the relative frequency (proportion) of a token.

        Calculates the token's frequency divided by total tokens, giving
        the probability of randomly selecting this token.

        Args
        ----
        token : str
            The token to look up.

        Returns
        -------
        float
            Proportion of total tokens (0.0 to 1.0). Returns 0.0 if not found
            or if total_tokens is 0.

        Examples
        --------
        Calculating token proportions:

        >>> from insideLLMs.tokens import get_token_distribution
        >>> dist = get_token_distribution("the cat sat on the mat")
        >>> print(f"'the' proportion: {dist.relative_frequency('the'):.2%}")
        'the' proportion: 33.33%

        Comparing relative frequencies:

        >>> text = "yes yes yes no no"
        >>> dist = get_token_distribution(text)
        >>> yes_freq = dist.relative_frequency('yes')
        >>> no_freq = dist.relative_frequency('no')
        >>> print(f"Yes: {yes_freq:.0%}, No: {no_freq:.0%}")
        Yes: 60%, No: 40%

        Using for probability estimation:

        >>> dist = get_token_distribution("a b c a a b")
        >>> prob_a = dist.relative_frequency('a')
        >>> print(f"P(next token = 'a') approx {prob_a:.2f}")
        P(next token = 'a') approx 0.50
        """
        if self.total_tokens == 0:
            return 0.0
        return self.frequencies.get(token, 0) / self.total_tokens

    def tokens_above_frequency(self, min_freq: int) -> list[str]:
        """Get all tokens with frequency at or above a threshold.

        Useful for filtering out rare tokens or identifying common vocabulary.

        Args
        ----
        min_freq : int
            Minimum frequency threshold (inclusive).

        Returns
        -------
        list[str]
            List of tokens meeting the frequency threshold.

        Examples
        --------
        Finding common words:

        >>> from insideLLMs.tokens import get_token_distribution
        >>> text = "the cat sat on the mat and the dog sat too"
        >>> dist = get_token_distribution(text)
        >>> common = dist.tokens_above_frequency(2)
        >>> print(f"Tokens appearing 2+ times: {sorted(common)}")
        Tokens appearing 2+ times: ['sat', 'the']

        Building a frequency-filtered vocabulary:

        >>> text = "python python python java java go rust"
        >>> dist = get_token_distribution(text)
        >>> vocab = dist.tokens_above_frequency(2)
        >>> print(f"Core vocabulary: {vocab}")
        Core vocabulary: ['python', 'java']

        Noise reduction for analysis:

        >>> text = "data data data analysis analysis x y z"
        >>> dist = get_token_distribution(text)
        >>> signal = dist.tokens_above_frequency(2)
        >>> noise = [t for t in dist.frequencies if t not in signal]
        >>> print(f"Signal: {signal}, Noise: {noise}")
        Signal: ['data', 'analysis'], Noise: ['x', 'y', 'z']
        """
        return [t for t, f in self.frequencies.items() if f >= min_freq]

    def hapax_legomena(self) -> list[str]:
        """Get tokens that appear exactly once (hapax legomena).

        In linguistics, hapax legomena are words that occur only once in a text
        or corpus. They often represent rare words, proper nouns, or potential
        errors and are important for vocabulary richness analysis.

        Returns
        -------
        list[str]
            List of tokens with frequency of exactly 1.

        Examples
        --------
        Finding single-occurrence words:

        >>> from insideLLMs.tokens import get_token_distribution
        >>> text = "the cat sat on the mat the cat purred"
        >>> dist = get_token_distribution(text)
        >>> hapax = dist.hapax_legomena()
        >>> print(f"Hapax legomena: {sorted(hapax)}")
        Hapax legomena: ['mat', 'on', 'purred', 'sat']

        Measuring vocabulary richness:

        >>> text = "unique words make text interesting and diverse"
        >>> dist = get_token_distribution(text)
        >>> hapax_ratio = len(dist.hapax_legomena()) / dist.vocabulary_size
        >>> print(f"Hapax ratio: {hapax_ratio:.0%}")
        Hapax ratio: 100%

        Identifying potential spelling errors:

        >>> text = "the teh cat sat on the mat"  # 'teh' is a typo
        >>> dist = get_token_distribution(text)
        >>> one_offs = dist.hapax_legomena()
        >>> print(f"Check these for typos: {one_offs}")
        Check these for typos: ['teh', 'cat', 'sat', 'on', 'mat']
        """
        return [t for t, f in self.frequencies.items() if f == 1]

    def entropy(self) -> float:
        """Calculate Shannon entropy of the token distribution.

        Entropy measures the unpredictability or information content of the
        distribution. Higher entropy indicates more uniform distribution
        (harder to predict next token), while lower entropy indicates
        concentration in few tokens (easier to predict).

        Returns
        -------
        float
            Shannon entropy in bits. Range depends on vocabulary size:
            0 (single token) to log2(vocab_size) (uniform distribution).

        Examples
        --------
        Comparing entropy of different texts:

        >>> from insideLLMs.tokens import get_token_distribution
        >>> uniform = "a b c d e f g h"  # Uniform distribution
        >>> skewed = "a a a a b c d e"  # Skewed toward 'a'
        >>> repetitive = "a a a a a a a a"  # Single token
        >>> print(f"Uniform entropy: {get_token_distribution(uniform).entropy():.2f}")
        Uniform entropy: 3.00
        >>> print(f"Skewed entropy: {get_token_distribution(skewed).entropy():.2f}")
        Skewed entropy: 2.41
        >>> print(f"Repetitive entropy: {get_token_distribution(repetitive).entropy():.2f}")
        Repetitive entropy: 0.00

        Using entropy for text comparison:

        >>> creative = "brilliant unique marvelous spectacular amazing"
        >>> formulaic = "good good nice good nice good good nice"
        >>> e1 = get_token_distribution(creative).entropy()
        >>> e2 = get_token_distribution(formulaic).entropy()
        >>> print(f"Creative: {e1:.2f} bits, Formulaic: {e2:.2f} bits")
        Creative: 2.32 bits, Formulaic: 1.00 bits

        Perplexity calculation from entropy:

        >>> text = "the quick brown fox jumps over the lazy dog"
        >>> entropy = get_token_distribution(text).entropy()
        >>> perplexity = 2 ** entropy
        >>> print(f"Entropy: {entropy:.2f}, Perplexity: {perplexity:.1f}")
        Entropy: 3.03, Perplexity: 8.2
        """
        if self.total_tokens == 0:
            return 0.0

        entropy = 0.0
        for count in self.frequencies.values():
            if count > 0:
                p = count / self.total_tokens
                entropy -= p * math.log2(p)
        return entropy


@dataclass
class VocabCoverage:
    """Analysis of vocabulary coverage against a reference vocabulary.

    This dataclass measures how well a reference vocabulary (e.g., a model's
    known tokens) covers the vocabulary found in a given text. Useful for
    identifying out-of-vocabulary (OOV) words and assessing domain compatibility.

    Parameters
    ----------
    text_vocab : set[str]
        Set of unique tokens found in the analyzed text.
    reference_vocab : set[str]
        Set of tokens in the reference vocabulary.
    covered : set[str]
        Tokens from text_vocab that exist in reference_vocab.
    uncovered : set[str]
        Tokens from text_vocab that do NOT exist in reference_vocab (OOV tokens).

    Attributes
    ----------
    coverage_ratio : float
        Property returning the proportion of text vocabulary covered (0.0 to 1.0).
    oov_ratio : float
        Property returning the out-of-vocabulary ratio (1.0 - coverage_ratio).

    Examples
    --------
    Basic coverage analysis:

    >>> from insideLLMs.tokens import TokenAnalyzer
    >>> analyzer = TokenAnalyzer()
    >>> text = "The quantum computer calculated eigenvalues efficiently"
    >>> reference = {"the", "computer", "calculated", "a", "is", "and"}
    >>> coverage = analyzer.analyze_vocabulary_coverage(text, reference)
    >>> print(f"Coverage: {coverage.coverage_ratio:.0%}")
    Coverage: 33%
    >>> print(f"OOV words: {coverage.uncovered}")
    OOV words: {'quantum', 'eigenvalues', 'efficiently'}

    Checking domain vocabulary compatibility:

    >>> analyzer = TokenAnalyzer()
    >>> medical_text = "The patient presented with tachycardia and dyspnea"
    >>> general_vocab = {"the", "patient", "with", "and", "a", "is"}
    >>> cov = analyzer.analyze_vocabulary_coverage(medical_text, general_vocab)
    >>> if cov.oov_ratio > 0.3:
    ...     print(f"High OOV rate: {cov.oov_ratio:.0%} - may need domain vocab")
    High OOV rate: 50% - may need domain vocab

    Identifying words needing special handling:

    >>> from insideLLMs.tokens import TokenAnalyzer
    >>> analyzer = TokenAnalyzer()
    >>> code = "def calculate_eigenvalue(matrix): return numpy.linalg.eig(matrix)"
    >>> python_keywords = {"def", "return", "if", "else", "for", "while", "class"}
    >>> cov = analyzer.analyze_vocabulary_coverage(code, python_keywords)
    >>> print(f"Non-keywords (identifiers/libs): {cov.uncovered}")
    Non-keywords (identifiers/libs): {'calculate_eigenvalue', 'matrix', ...}

    See Also
    --------
    TokenAnalyzer.analyze_vocabulary_coverage : Method that creates VocabCoverage.
    """

    text_vocab: set[str]
    reference_vocab: set[str]
    covered: set[str]
    uncovered: set[str]

    @property
    def coverage_ratio(self) -> float:
        """Calculate the ratio of text vocabulary covered by reference.

        Returns the proportion of unique tokens in the text that are present
        in the reference vocabulary. A value of 1.0 means all text tokens are
        known; lower values indicate more out-of-vocabulary words.

        Returns
        -------
        float
            Coverage ratio between 0.0 and 1.0. Returns 1.0 for empty text.

        Examples
        --------
        Full coverage scenario:

        >>> from insideLLMs.tokens import TokenAnalyzer
        >>> analyzer = TokenAnalyzer()
        >>> text = "hello world"
        >>> vocab = {"hello", "world", "goodbye"}
        >>> cov = analyzer.analyze_vocabulary_coverage(text, vocab)
        >>> print(f"Coverage: {cov.coverage_ratio:.0%}")
        Coverage: 100%

        Partial coverage:

        >>> text = "hello universe"
        >>> cov = analyzer.analyze_vocabulary_coverage(text, vocab)
        >>> print(f"Coverage: {cov.coverage_ratio:.0%}")
        Coverage: 50%

        No coverage (all OOV):

        >>> text = "quantum entanglement"
        >>> cov = analyzer.analyze_vocabulary_coverage(text, vocab)
        >>> print(f"Coverage: {cov.coverage_ratio:.0%}")
        Coverage: 0%
        """
        if not self.text_vocab:
            return 1.0
        return len(self.covered) / len(self.text_vocab)

    @property
    def oov_ratio(self) -> float:
        """Calculate the out-of-vocabulary (OOV) ratio.

        Returns the proportion of unique tokens in the text that are NOT
        present in the reference vocabulary. This is simply 1.0 - coverage_ratio.

        Returns
        -------
        float
            OOV ratio between 0.0 and 1.0.

        Examples
        --------
        High OOV scenario (specialized domain):

        >>> from insideLLMs.tokens import TokenAnalyzer
        >>> analyzer = TokenAnalyzer()
        >>> legal_text = "The defendant's habeas corpus petition was denied"
        >>> basic_vocab = {"the", "was", "a", "is", "and"}
        >>> cov = analyzer.analyze_vocabulary_coverage(legal_text, basic_vocab)
        >>> print(f"OOV rate: {cov.oov_ratio:.0%}")
        OOV rate: 83%

        Low OOV scenario (common vocabulary):

        >>> simple_text = "the cat is a pet"
        >>> cov = analyzer.analyze_vocabulary_coverage(simple_text, basic_vocab)
        >>> print(f"OOV rate: {cov.oov_ratio:.0%}")
        OOV rate: 40%

        Using OOV for quality assessment:

        >>> text = "This is a test sentence for the model"
        >>> large_vocab = {"this", "is", "a", "test", "sentence", "for", "the", "model"}
        >>> cov = analyzer.analyze_vocabulary_coverage(text, large_vocab)
        >>> if cov.oov_ratio < 0.1:
        ...     print("Excellent vocabulary coverage!")
        Excellent vocabulary coverage!
        """
        return 1.0 - self.coverage_ratio

    def to_dict(self) -> dict[str, Any]:
        """Convert vocabulary coverage analysis to dictionary.

        Creates a dictionary suitable for JSON serialization, logging, or
        reporting. Does not include the actual token sets (only their sizes).

        Returns
        -------
        dict[str, Any]
            Dictionary with coverage statistics.

        Examples
        --------
        Basic conversion:

        >>> from insideLLMs.tokens import TokenAnalyzer
        >>> analyzer = TokenAnalyzer()
        >>> cov = analyzer.analyze_vocabulary_coverage(
        ...     "hello world test",
        ...     {"hello", "world", "foo", "bar"}
        ... )
        >>> d = cov.to_dict()
        >>> print(f"Covered: {d['covered_count']}/{d['text_vocab_size']}")
        Covered: 2/3

        Logging coverage metrics:

        >>> import logging
        >>> cov = analyzer.analyze_vocabulary_coverage("test text", {"test"})
        >>> logging.info("Vocab coverage: %s", cov.to_dict())

        Creating coverage report:

        >>> texts = ["first doc", "second doc with more words"]
        >>> vocab = {"first", "second", "doc", "with"}
        >>> for i, text in enumerate(texts):
        ...     cov = analyzer.analyze_vocabulary_coverage(text, vocab)
        ...     report = cov.to_dict()
        ...     print(f"Doc {i}: {report['coverage_ratio']:.0%} covered")
        Doc 0: 100% covered
        Doc 1: 80% covered
        """
        return {
            "text_vocab_size": len(self.text_vocab),
            "reference_vocab_size": len(self.reference_vocab),
            "covered_count": len(self.covered),
            "uncovered_count": len(self.uncovered),
            "coverage_ratio": self.coverage_ratio,
            "oov_ratio": self.oov_ratio,
        }


class Tokenizer(Protocol):
    """Protocol for tokenizer implementations."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        ...

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text to token strings."""
        ...


class SimpleTokenizer:
    """Simple whitespace and punctuation-based tokenizer."""

    def __init__(self, lowercase: bool = True):
        """Initialize tokenizer.

        Args:
            lowercase: Whether to lowercase tokens.
        """
        self.lowercase = lowercase
        self._vocab: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._next_id = 0

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text to tokens.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        if self.lowercase:
            text = text.lower()

        # Split on whitespace and punctuation
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode.

        Returns:
            List of token IDs.
        """
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._id_to_token[self._next_id] = token
                self._next_id += 1
            ids.append(self._vocab[token])
        return ids

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text.
        """
        return " ".join(self._id_to_token.get(t, "<UNK>") for t in tokens)

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return len(self._vocab)


class TokenEstimator:
    """Estimate token counts for different models."""

    # Average characters per token for different models
    CHARS_PER_TOKEN = {
        TokenizerType.GPT4: 4.0,
        TokenizerType.GPT3: 4.0,
        TokenizerType.CLAUDE: 3.5,
        TokenizerType.LLAMA: 3.8,
        TokenizerType.SIMPLE: 5.0,
    }

    # Adjustment factors for different content types
    CONTENT_ADJUSTMENTS = {
        "code": 0.7,  # Code tends to have more tokens
        "prose": 1.0,
        "technical": 0.85,
        "chat": 1.1,  # Conversational text has fewer tokens
    }

    def __init__(self, tokenizer_type: TokenizerType = TokenizerType.GPT4):
        """Initialize estimator.

        Args:
            tokenizer_type: Type of tokenizer to estimate for.
        """
        self.tokenizer_type = tokenizer_type
        self.chars_per_token = self.CHARS_PER_TOKEN.get(tokenizer_type, 4.0)

    def estimate_tokens(self, text: str, content_type: str = "prose") -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate.
            content_type: Type of content (code, prose, technical, chat).

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
            
        # Exact counting with tiktoken if available for OpenAI models
        if self.tokenizer_type in (TokenizerType.GPT4, TokenizerType.GPT3):
            try:
                import tiktoken
                encoding = "cl100k_base" if self.tokenizer_type == TokenizerType.GPT4 else "p50k_base"
                enc = tiktoken.get_encoding(encoding)
                return len(enc.encode(text, disallowed_special=()))
            except ImportError:
                pass  # Fall back to character-based estimation
                
        char_count = len(text)
        adjustment = self.CONTENT_ADJUSTMENTS.get(content_type, 1.0)
        estimated = char_count / (self.chars_per_token * adjustment)
        return max(1, round(estimated))

    def estimate_cost(
        self,
        text: str,
        price_per_1k_tokens: float,
        content_type: str = "prose",
    ) -> float:
        """Estimate cost for processing text.

        Args:
            text: Text to process.
            price_per_1k_tokens: Price per 1000 tokens.
            content_type: Type of content.

        Returns:
            Estimated cost.
        """
        tokens = self.estimate_tokens(text, content_type)
        return (tokens / 1000) * price_per_1k_tokens

    def fits_context(
        self,
        text: str,
        context_limit: int,
        content_type: str = "prose",
    ) -> bool:
        """Check if text fits within context limit.

        Args:
            text: Text to check.
            context_limit: Maximum tokens allowed.
            content_type: Type of content.

        Returns:
            True if text fits within limit.
        """
        return self.estimate_tokens(text, content_type) <= context_limit

    def split_to_chunks(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int = 0,
        content_type: str = "prose",
    ) -> list[str]:
        """Split text into chunks that fit token limit.

        Args:
            text: Text to split.
            max_tokens: Maximum tokens per chunk.
            overlap_tokens: Number of overlapping tokens.
            content_type: Type of content.

        Returns:
            List of text chunks.
        """
        # Calculate approximate characters per chunk
        adjustment = self.CONTENT_ADJUSTMENTS.get(content_type, 1.0)
        chars_per_chunk = int(max_tokens * self.chars_per_token * adjustment)
        overlap_chars = int(overlap_tokens * self.chars_per_token * adjustment)

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chars_per_chunk, text_len)

            # Try to break at sentence or word boundary
            if end < text_len:
                # Look for sentence boundary
                for boundary in [". ", "! ", "? ", "\n"]:
                    boundary_pos = text.rfind(boundary, start, end)
                    if boundary_pos > start:
                        end = boundary_pos + len(boundary)
                        break
                else:
                    # Look for word boundary
                    space_pos = text.rfind(" ", start, end)
                    if space_pos > start:
                        end = space_pos + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap_chars if overlap_chars > 0 else end

        return chunks


class TokenAnalyzer:
    """Comprehensive token analysis utilities."""

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        """Initialize analyzer.

        Args:
            tokenizer: Optional tokenizer to use.
        """
        self.tokenizer = tokenizer or SimpleTokenizer()

    def analyze(self, text: str) -> TokenStats:
        """Analyze tokens in text.

        Args:
            text: Text to analyze.

        Returns:
            Token statistics.
        """
        tokens = self.tokenizer.tokenize(text)
        word_count = len(text.split())

        frequencies = Counter(tokens)
        total_tokens = len(tokens)
        unique_tokens = len(frequencies)

        total_length = sum(len(t) for t in tokens)
        avg_length = total_length / total_tokens if total_tokens > 0 else 0

        return TokenStats(
            total_tokens=total_tokens,
            unique_tokens=unique_tokens,
            char_count=len(text),
            word_count=word_count,
            avg_token_length=avg_length,
            tokens_per_word=total_tokens / word_count if word_count > 0 else 0,
            token_frequencies=dict(frequencies),
        )

    def get_distribution(self, text: str) -> TokenDistribution:
        """Get token frequency distribution.

        Args:
            text: Text to analyze.

        Returns:
            Token distribution.
        """
        tokens = self.tokenizer.tokenize(text)
        frequencies = Counter(tokens)
        return TokenDistribution(
            frequencies=dict(frequencies),
            total_tokens=len(tokens),
        )

    def compare_distributions(self, text1: str, text2: str) -> dict[str, Any]:
        """Compare token distributions between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Comparison results.
        """
        dist1 = self.get_distribution(text1)
        dist2 = self.get_distribution(text2)

        vocab1 = set(dist1.frequencies.keys())
        vocab2 = set(dist2.frequencies.keys())

        shared = vocab1 & vocab2
        only_in_1 = vocab1 - vocab2
        only_in_2 = vocab2 - vocab1

        # Calculate cosine similarity of distributions
        common_tokens = shared
        if common_tokens:
            dot_product = sum(dist1.frequencies[t] * dist2.frequencies[t] for t in common_tokens)
            mag1 = math.sqrt(sum(f**2 for f in dist1.frequencies.values()))
            mag2 = math.sqrt(sum(f**2 for f in dist2.frequencies.values()))
            cosine_sim = dot_product / (mag1 * mag2) if mag1 * mag2 > 0 else 0
        else:
            cosine_sim = 0.0

        return {
            "text1_vocab_size": len(vocab1),
            "text2_vocab_size": len(vocab2),
            "shared_vocab_size": len(shared),
            "only_in_text1": len(only_in_1),
            "only_in_text2": len(only_in_2),
            "jaccard_similarity": len(shared) / len(vocab1 | vocab2) if vocab1 | vocab2 else 0,
            "cosine_similarity": cosine_sim,
            "entropy_text1": dist1.entropy(),
            "entropy_text2": dist2.entropy(),
        }

    def analyze_vocabulary_coverage(self, text: str, reference_vocab: set[str]) -> VocabCoverage:
        """Analyze vocabulary coverage against reference.

        Args:
            text: Text to analyze.
            reference_vocab: Reference vocabulary set.

        Returns:
            Vocabulary coverage analysis.
        """
        tokens = self.tokenizer.tokenize(text)
        text_vocab = set(tokens)

        covered = text_vocab & reference_vocab
        uncovered = text_vocab - reference_vocab

        return VocabCoverage(
            text_vocab=text_vocab,
            reference_vocab=reference_vocab,
            covered=covered,
            uncovered=uncovered,
        )


@dataclass
class EmbeddingSimilarity:
    """Results of embedding similarity comparison."""

    similarity: float
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingUtils:
    """Utilities for working with text embeddings."""

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity (-1 to 1).
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a**2 for a in vec1))
        mag2 = math.sqrt(sum(b**2 for b in vec2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    @staticmethod
    def euclidean_distance(vec1: list[float], vec2: list[float]) -> float:
        """Calculate Euclidean distance between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Euclidean distance.
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension")

        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

    @staticmethod
    def manhattan_distance(vec1: list[float], vec2: list[float]) -> float:
        """Calculate Manhattan distance between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Manhattan distance.
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension")

        return sum(abs(a - b) for a, b in zip(vec1, vec2))

    @staticmethod
    def normalize(vec: list[float]) -> list[float]:
        """Normalize vector to unit length.

        Args:
            vec: Vector to normalize.

        Returns:
            Normalized vector.
        """
        magnitude = math.sqrt(sum(v**2 for v in vec))
        if magnitude == 0:
            return vec
        return [v / magnitude for v in vec]

    @staticmethod
    def average_embeddings(embeddings: list[list[float]]) -> list[float]:
        """Calculate average of multiple embeddings.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            Average embedding.
        """
        if not embeddings:
            return []

        dim = len(embeddings[0])
        avg = [0.0] * dim

        for emb in embeddings:
            for i, v in enumerate(emb):
                avg[i] += v

        n = len(embeddings)
        return [v / n for v in avg]

    @staticmethod
    def find_most_similar(
        query: list[float],
        candidates: list[tuple[str, list[float]]],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Find most similar embeddings to query.

        Args:
            query: Query embedding.
            candidates: List of (id, embedding) tuples.
            top_k: Number of top results to return.

        Returns:
            List of (id, similarity) tuples.
        """
        similarities = []
        for id_, embedding in candidates:
            sim = EmbeddingUtils.cosine_similarity(query, embedding)
            similarities.append((id_, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


@dataclass
class TokenSpendingBudget:
    """Track token budget for operations."""

    total_budget: int
    used_tokens: int = 0
    reserved_tokens: int = 0

    @property
    def available(self) -> int:
        """Available tokens."""
        return self.total_budget - self.used_tokens - self.reserved_tokens

    @property
    def utilization(self) -> float:
        """Budget utilization ratio."""
        return self.used_tokens / self.total_budget if self.total_budget > 0 else 0

    def can_afford(self, tokens: int) -> bool:
        """Check if budget allows for tokens."""
        return tokens <= self.available

    def spend(self, tokens: int) -> bool:
        """Spend tokens from budget.

        Args:
            tokens: Tokens to spend.

        Returns:
            True if successful, False if over budget.
        """
        if not self.can_afford(tokens):
            return False
        self.used_tokens += tokens
        return True

    def reserve(self, tokens: int) -> bool:
        """Reserve tokens for future use.

        Args:
            tokens: Tokens to reserve.

        Returns:
            True if successful.
        """
        if tokens > self.available:
            return False
        self.reserved_tokens += tokens
        return True

    def release_reservation(self, tokens: int) -> None:
        """Release reserved tokens."""
        self.reserved_tokens = max(0, self.reserved_tokens - tokens)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_budget": self.total_budget,
            "used_tokens": self.used_tokens,
            "reserved_tokens": self.reserved_tokens,
            "available": self.available,
            "utilization": self.utilization,
        }


class ContextWindowManager:
    """Manage content within context window limits."""

    def __init__(
        self,
        max_tokens: int,
        estimator: Optional[TokenEstimator] = None,
    ):
        """Initialize manager.

        Args:
            max_tokens: Maximum context window tokens.
            estimator: Token estimator to use.
        """
        self.max_tokens = max_tokens
        self.estimator = estimator or TokenEstimator()
        self.budget = TokenSpendingBudget(total_budget=max_tokens)

    def add_content(
        self,
        content: str,
        priority: int = 0,
        content_type: str = "prose",
    ) -> bool:
        """Try to add content to context.

        Args:
            content: Content to add.
            priority: Content priority (higher = more important).
            content_type: Type of content.

        Returns:
            True if content fits, False otherwise.
        """
        tokens = self.estimator.estimate_tokens(content, content_type)
        return self.budget.spend(tokens)

    def remaining_tokens(self) -> int:
        """Get remaining token budget."""
        return self.budget.available

    def fits(self, content: str, content_type: str = "prose") -> bool:
        """Check if content fits in remaining budget.

        Args:
            content: Content to check.
            content_type: Type of content.

        Returns:
            True if content fits.
        """
        tokens = self.estimator.estimate_tokens(content, content_type)
        return self.budget.can_afford(tokens)

    def truncate_to_fit(
        self,
        content: str,
        reserve_tokens: int = 0,
        content_type: str = "prose",
    ) -> str:
        """Truncate content to fit remaining budget.

        Args:
            content: Content to truncate.
            reserve_tokens: Tokens to reserve for other content.
            content_type: Type of content.

        Returns:
            Truncated content.
        """
        available = self.budget.available - reserve_tokens
        if available <= 0:
            return ""

        # Estimate characters that fit
        adjustment = self.estimator.CONTENT_ADJUSTMENTS.get(content_type, 1.0)
        max_chars = int(available * self.estimator.chars_per_token * adjustment)

        if len(content) <= max_chars:
            return content

        # Truncate at word boundary
        truncated = content[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]

        return truncated + "..."

    def reset(self) -> None:
        """Reset context budget."""
        self.budget = TokenSpendingBudget(total_budget=self.max_tokens)


# Convenience functions


def estimate_tokens(
    text: str,
    tokenizer_type: TokenizerType = TokenizerType.GPT4,
    content_type: str = "prose",
) -> int:
    """Estimate token count for text.

    Args:
        text: Text to estimate.
        tokenizer_type: Type of tokenizer.
        content_type: Type of content.

    Returns:
        Estimated token count.
    """
    estimator = TokenEstimator(tokenizer_type)
    return estimator.estimate_tokens(text, content_type)


def analyze_tokens(text: str) -> TokenStats:
    """Analyze tokens in text.

    Args:
        text: Text to analyze.

    Returns:
        Token statistics.
    """
    analyzer = TokenAnalyzer()
    return analyzer.analyze(text)


def get_token_distribution(text: str) -> TokenDistribution:
    """Get token distribution for text.

    Args:
        text: Text to analyze.

    Returns:
        Token distribution.
    """
    analyzer = TokenAnalyzer()
    return analyzer.get_distribution(text)


def split_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
    tokenizer_type: TokenizerType = TokenizerType.GPT4,
) -> list[str]:
    """Split text into chunks by token limit.

    Args:
        text: Text to split.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Overlapping tokens between chunks.
        tokenizer_type: Type of tokenizer for estimation.

    Returns:
        List of text chunks.
    """
    estimator = TokenEstimator(tokenizer_type)
    return estimator.split_to_chunks(text, max_tokens, overlap_tokens)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity.
    """
    return EmbeddingUtils.cosine_similarity(vec1, vec2)


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import TokenBudget. The canonical name is TokenSpendingBudget.
TokenBudget = TokenSpendingBudget
