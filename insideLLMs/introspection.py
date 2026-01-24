"""
Model introspection and attention analysis utilities.

This module provides comprehensive tools for analyzing and understanding
language model behavior through various introspection techniques. Since
API-based models don't expose internal states, this module uses heuristic-based
simulations to approximate attention patterns, token importance, and layer
representations.

Provides tools for:
- Attention pattern analysis: Identify common attention patterns (local, global, sparse)
- Token importance estimation: Estimate which tokens are most influential
- Activation analysis: Profile layer-wise activation characteristics
- Layer-wise representation analysis: Understand how information flows through layers
- Model behavior introspection: Comprehensive analysis combining all techniques

Key Classes:
    AttentionPattern: Enum of common attention pattern types
    TokenImportance: Dataclass storing importance scores for individual tokens
    AttentionHead: Analysis results for a single attention head
    LayerAnalysis: Analysis of a single model layer
    IntrospectionReport: Comprehensive report combining all analyses
    TokenImportanceEstimator: Estimates token importance using heuristics
    AttentionAnalyzer: Analyzes attention patterns from text
    LayerAnalyzer: Analyzes layer-wise representations
    SaliencyMap: Maps input tokens to their saliency scores
    SaliencyEstimator: Estimates input saliency for model outputs
    ModelIntrospector: Main class for comprehensive model introspection
    ActivationProfile: Profile of model activations
    ActivationProfiler: Profiles model activations

Example: Basic Token Importance Analysis
    >>> from insideLLMs.introspection import estimate_token_importance
    >>> text = "The quick brown fox jumps over the lazy dog."
    >>> importances = estimate_token_importance(text)
    >>> for imp in importances[:5]:
    ...     print(f"{imp.token}: {imp.importance_score:.3f}")
    the: 0.200
    quick: 0.600
    brown: 0.600
    fox: 0.500
    jumps: 0.500

Example: Attention Pattern Analysis
    >>> from insideLLMs.introspection import analyze_attention
    >>> prompt = "What is the capital of France?"
    >>> response = "The capital of France is Paris."
    >>> heads = analyze_attention(prompt, response)
    >>> for head in heads[:3]:
    ...     print(f"Layer {head.layer}, Head {head.head}: {head.pattern_type.value}")
    Layer 0, Head 0: local
    Layer 0, Head 1: local
    Layer 1, Head 0: local

Example: Full Model Introspection
    >>> from insideLLMs.introspection import introspect_model
    >>> prompt = "Explain quantum computing"
    >>> response = "Quantum computing uses quantum mechanics principles..."
    >>> report = introspect_model(prompt, response)
    >>> print(f"Key tokens: {report.key_tokens}")
    >>> print(f"Attention patterns: {report.attention_patterns}")
    Key tokens: ['quantum', 'computing', 'mechanics']
    Attention patterns: {'local': 4, 'sparse': 2, 'global': 2}

Example: Saliency Estimation
    >>> from insideLLMs.introspection import estimate_saliency
    >>> prompt = "Translate to French: Hello world"
    >>> response = "Bonjour le monde"
    >>> saliency = estimate_saliency(prompt, response)
    >>> print(saliency.get_highlighted_text(threshold=0.5))
    translate to **french** : **hello** **world**

Note:
    Since API-based language models don't expose internal states like attention
    weights or activations, this module uses heuristic-based approximations.
    Results should be interpreted as estimates rather than ground truth.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from insideLLMs.nlp.tokenization import word_tokenize_regex


class AttentionPattern(Enum):
    """
    Enumeration of common attention patterns observed in transformer models.

    Attention patterns describe how attention weights are distributed across
    input tokens. Different patterns indicate different types of information
    processing, from local syntactic relationships to global semantic dependencies.

    Attributes:
        LOCAL: Attention concentrated on nearby tokens. Common in early layers
            for capturing local syntax and phrase structure.
        GLOBAL: Attention distributed across all tokens. Found in later layers
            for integrating global context and semantic meaning.
        SPARSE: Attention focused on specific, non-contiguous tokens. Indicates
            selective attention to key information.
        DIAGONAL: Self-attention dominance where tokens attend primarily to
            themselves. Often seen in residual connections.
        VERTICAL: Attention to specific positions regardless of content. Common
            for positional anchors like [CLS] or [SEP] tokens.
        BLOCK: Block diagonal patterns indicating attention to contiguous
            segments or phrases.

    Example: Checking pattern types
        >>> pattern = AttentionPattern.LOCAL
        >>> pattern.value
        'local'
        >>> pattern == AttentionPattern.LOCAL
        True

    Example: Iterating over all patterns
        >>> for pattern in AttentionPattern:
        ...     print(f"{pattern.name}: {pattern.value}")
        LOCAL: local
        GLOBAL: global
        SPARSE: sparse
        DIAGONAL: diagonal
        VERTICAL: vertical
        BLOCK: block

    Example: Using patterns in analysis
        >>> head = AttentionHead(layer=0, head=0, pattern_type=AttentionPattern.LOCAL,
        ...                      entropy=0.3, sparsity=0.7)
        >>> if head.pattern_type == AttentionPattern.LOCAL:
        ...     print("This head captures local relationships")
        This head captures local relationships

    Example: Pattern-based filtering
        >>> heads = [AttentionHead(0, 0, AttentionPattern.LOCAL, 0.3, 0.7),
        ...          AttentionHead(1, 0, AttentionPattern.GLOBAL, 0.8, 0.2)]
        >>> local_heads = [h for h in heads if h.pattern_type == AttentionPattern.LOCAL]
        >>> len(local_heads)
        1
    """

    LOCAL = "local"  # Attention to nearby tokens
    GLOBAL = "global"  # Attention to all tokens
    SPARSE = "sparse"  # Attention to specific tokens
    DIAGONAL = "diagonal"  # Self-attention dominance
    VERTICAL = "vertical"  # Attention to specific positions
    BLOCK = "block"  # Block diagonal patterns


@dataclass
class TokenImportance:
    """
    Dataclass storing importance scores for a single token.

    TokenImportance captures multiple measures of a token's significance
    in the model's processing, including overall importance, attribution
    scores, gradient-based scores, and attention received from other tokens.

    Attributes:
        token: The string representation of the token.
        position: The position index of the token in the sequence (0-indexed).
        importance_score: Overall importance score (0.0 to 1.0). Higher values
            indicate more influential tokens.
        attribution_score: Score from attribution methods like integrated
            gradients. Indicates contribution to output (default: 0.0).
        gradient_score: Gradient-based importance measure. Higher values
            indicate stronger gradient flow through this token (default: 0.0).
        attention_received: Total attention weight received from other tokens.
            Higher values indicate tokens that other tokens attend to (default: 0.0).

    Example: Creating a TokenImportance instance
        >>> token_imp = TokenImportance(
        ...     token="quantum",
        ...     position=5,
        ...     importance_score=0.85,
        ...     attribution_score=0.72,
        ...     gradient_score=0.68,
        ...     attention_received=0.91
        ... )
        >>> token_imp.token
        'quantum'
        >>> token_imp.importance_score
        0.85

    Example: Converting to dictionary for serialization
        >>> token_imp = TokenImportance("important", 3, 0.9, 0.8, 0.7, 0.85)
        >>> d = token_imp.to_dict()
        >>> d['token']
        'important'
        >>> d['importance_score']
        0.9

    Example: Comparing token importances
        >>> tokens = [
        ...     TokenImportance("the", 0, 0.2),
        ...     TokenImportance("capital", 2, 0.8),
        ...     TokenImportance("Paris", 5, 0.95)
        ... ]
        >>> most_important = max(tokens, key=lambda t: t.importance_score)
        >>> most_important.token
        'Paris'

    Example: Filtering high-importance tokens
        >>> tokens = [
        ...     TokenImportance("is", 1, 0.3),
        ...     TokenImportance("neural", 2, 0.75),
        ...     TokenImportance("network", 3, 0.82)
        ... ]
        >>> high_importance = [t for t in tokens if t.importance_score > 0.7]
        >>> [t.token for t in high_importance]
        ['neural', 'network']
    """

    token: str
    position: int
    importance_score: float
    attribution_score: float = 0.0
    gradient_score: float = 0.0
    attention_received: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """
        Convert TokenImportance to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing all token importance fields.
                Keys: 'token', 'position', 'importance_score', 'attribution_score',
                'gradient_score', 'attention_received'.

        Example: Basic conversion
            >>> ti = TokenImportance("example", 0, 0.75, 0.6, 0.5, 0.8)
            >>> d = ti.to_dict()
            >>> sorted(d.keys())
            ['attention_received', 'attribution_score', 'gradient_score', 'importance_score', 'position', 'token']

        Example: JSON serialization
            >>> import json
            >>> ti = TokenImportance("test", 1, 0.5)
            >>> json_str = json.dumps(ti.to_dict())
            >>> "test" in json_str
            True

        Example: Batch conversion
            >>> tokens = [TokenImportance("a", 0, 0.1), TokenImportance("b", 1, 0.9)]
            >>> dicts = [t.to_dict() for t in tokens]
            >>> len(dicts)
            2
        """
        return {
            "token": self.token,
            "position": self.position,
            "importance_score": self.importance_score,
            "attribution_score": self.attribution_score,
            "gradient_score": self.gradient_score,
            "attention_received": self.attention_received,
        }


@dataclass
class AttentionHead:
    """
    Analysis results for a single attention head in a transformer layer.

    AttentionHead captures the characteristics of attention patterns within
    a specific head, including the type of pattern, distribution measures
    (entropy and sparsity), and key positions that receive significant attention.

    Attributes:
        layer: The layer index (0-indexed) where this attention head resides.
        head: The head index (0-indexed) within the layer.
        pattern_type: The detected AttentionPattern type (LOCAL, GLOBAL, etc.).
        entropy: Attention distribution entropy (0.0 to 1.0). Higher values
            indicate more uniformly distributed attention across tokens.
        sparsity: Attention sparsity measure (0.0 to 1.0). Higher values
            indicate attention is concentrated on fewer tokens.
        key_positions: List of token position indices that receive significant
            attention from other tokens (default: empty list).

    Example: Creating an AttentionHead instance
        >>> head = AttentionHead(
        ...     layer=2,
        ...     head=5,
        ...     pattern_type=AttentionPattern.LOCAL,
        ...     entropy=0.35,
        ...     sparsity=0.72,
        ...     key_positions=[0, 1, 2, 5]
        ... )
        >>> head.layer
        2
        >>> head.pattern_type.value
        'local'

    Example: Analyzing attention head characteristics
        >>> head = AttentionHead(0, 0, AttentionPattern.GLOBAL, 0.85, 0.15)
        >>> if head.entropy > 0.7:
        ...     print("Attention is broadly distributed")
        Attention is broadly distributed
        >>> if head.sparsity < 0.3:
        ...     print("Head attends to many tokens")
        Head attends to many tokens

    Example: Converting to dictionary for logging
        >>> head = AttentionHead(1, 3, AttentionPattern.SPARSE, 0.4, 0.8, [2, 7, 12])
        >>> d = head.to_dict()
        >>> d['pattern_type']
        'sparse'
        >>> d['key_positions']
        [2, 7, 12]

    Example: Filtering heads by pattern type
        >>> heads = [
        ...     AttentionHead(0, 0, AttentionPattern.LOCAL, 0.3, 0.7),
        ...     AttentionHead(0, 1, AttentionPattern.GLOBAL, 0.8, 0.2),
        ...     AttentionHead(1, 0, AttentionPattern.LOCAL, 0.35, 0.65)
        ... ]
        >>> local_heads = [h for h in heads if h.pattern_type == AttentionPattern.LOCAL]
        >>> len(local_heads)
        2
    """

    layer: int
    head: int
    pattern_type: AttentionPattern
    entropy: float  # Higher = more distributed attention
    sparsity: float  # Higher = more focused attention
    key_positions: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert AttentionHead to a dictionary representation.

        The pattern_type is converted to its string value for JSON compatibility.

        Returns:
            dict[str, Any]: Dictionary containing all attention head fields.
                Keys: 'layer', 'head', 'pattern_type' (as string), 'entropy',
                'sparsity', 'key_positions'.

        Example: Basic conversion
            >>> head = AttentionHead(0, 1, AttentionPattern.SPARSE, 0.4, 0.8, [1, 5])
            >>> d = head.to_dict()
            >>> d['layer']
            0
            >>> d['pattern_type']
            'sparse'

        Example: JSON serialization
            >>> import json
            >>> head = AttentionHead(2, 0, AttentionPattern.GLOBAL, 0.9, 0.1)
            >>> json_str = json.dumps(head.to_dict())
            >>> "global" in json_str
            True

        Example: Collecting statistics across heads
            >>> heads = [
            ...     AttentionHead(0, 0, AttentionPattern.LOCAL, 0.3, 0.7),
            ...     AttentionHead(0, 1, AttentionPattern.GLOBAL, 0.8, 0.2)
            ... ]
            >>> avg_entropy = sum(h.to_dict()['entropy'] for h in heads) / len(heads)
            >>> 0.5 < avg_entropy < 0.6
            True
        """
        return {
            "layer": self.layer,
            "head": self.head,
            "pattern_type": self.pattern_type.value,
            "entropy": self.entropy,
            "sparsity": self.sparsity,
            "key_positions": self.key_positions,
        }


@dataclass
class LayerAnalysis:
    """
    Analysis results for a single layer in a transformer model.

    LayerAnalysis captures how information is represented and transformed
    at each layer, including measures of representation magnitude, token
    similarity, and information flow from input to output.

    Attributes:
        layer_index: The layer index (0-indexed) being analyzed.
        representation_norm: Average L2 norm of token representations at this
            layer. Typically decreases in later layers as representations
            become more task-specific.
        token_similarity: Average cosine similarity between token representations
            (0.0 to 1.0). Lower values in later layers indicate more
            differentiated, context-specific representations.
        information_retention: Estimated fraction of input information retained
            at this layer (0.0 to 1.0). Decreases gradually through the network
            as task-irrelevant information is discarded.
        key_features: List of feature types typically captured at this layer
            depth (e.g., 'token_embedding', 'phrase_structure', 'global_context').

    Example: Creating a LayerAnalysis instance
        >>> analysis = LayerAnalysis(
        ...     layer_index=3,
        ...     representation_norm=0.85,
        ...     token_similarity=0.65,
        ...     information_retention=0.92,
        ...     key_features=['phrase_structure', 'local_context']
        ... )
        >>> analysis.layer_index
        3
        >>> analysis.representation_norm
        0.85

    Example: Analyzing layer characteristics
        >>> early_layer = LayerAnalysis(0, 0.95, 0.8, 0.98, ['token_embedding'])
        >>> late_layer = LayerAnalysis(11, 0.65, 0.3, 0.75, ['global_context'])
        >>> # Early layers retain more input information
        >>> early_layer.information_retention > late_layer.information_retention
        True
        >>> # Late layers have more differentiated representations
        >>> late_layer.token_similarity < early_layer.token_similarity
        True

    Example: Converting to dictionary for visualization
        >>> layer = LayerAnalysis(5, 0.78, 0.52, 0.88, ['sentence_meaning'])
        >>> d = layer.to_dict()
        >>> d['layer_index']
        5
        >>> 'sentence_meaning' in d['key_features']
        True

    Example: Tracking metrics across layers
        >>> layers = [
        ...     LayerAnalysis(0, 1.0, 0.8, 1.0, []),
        ...     LayerAnalysis(1, 0.9, 0.7, 0.95, []),
        ...     LayerAnalysis(2, 0.8, 0.6, 0.90, [])
        ... ]
        >>> norms = [l.representation_norm for l in layers]
        >>> all(norms[i] >= norms[i+1] for i in range(len(norms)-1))
        True
    """

    layer_index: int
    representation_norm: float
    token_similarity: float  # How similar token representations are
    information_retention: float  # Information from input retained
    key_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert LayerAnalysis to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing all layer analysis fields.
                Keys: 'layer_index', 'representation_norm', 'token_similarity',
                'information_retention', 'key_features'.

        Example: Basic conversion
            >>> layer = LayerAnalysis(2, 0.85, 0.6, 0.9, ['local_context'])
            >>> d = layer.to_dict()
            >>> d['layer_index']
            2
            >>> d['representation_norm']
            0.85

        Example: JSON serialization
            >>> import json
            >>> layer = LayerAnalysis(0, 1.0, 0.8, 1.0, ['token_embedding'])
            >>> json_str = json.dumps(layer.to_dict())
            >>> "token_embedding" in json_str
            True

        Example: Creating comparison table
            >>> layers = [LayerAnalysis(i, 1.0-i*0.1, 0.8-i*0.1, 1.0-i*0.05, [])
            ...           for i in range(3)]
            >>> data = [l.to_dict() for l in layers]
            >>> [d['layer_index'] for d in data]
            [0, 1, 2]
        """
        return {
            "layer_index": self.layer_index,
            "representation_norm": self.representation_norm,
            "token_similarity": self.token_similarity,
            "information_retention": self.information_retention,
            "key_features": self.key_features,
        }


@dataclass
class IntrospectionReport:
    """
    Comprehensive introspection report combining all analysis results.

    IntrospectionReport aggregates token importance, attention analysis,
    layer analysis, and derived insights into a single comprehensive report.
    This is the primary output of the ModelIntrospector.introspect() method.

    Attributes:
        prompt: The input prompt that was analyzed.
        response: The model's response that was analyzed.
        token_importances: List of TokenImportance objects for all tokens
            in both prompt and response.
        attention_analysis: List of AttentionHead analyses for simulated
            attention heads across layers.
        layer_analyses: List of LayerAnalysis results for each analyzed layer.
        key_tokens: List of the most important tokens (importance > 0.6)
            identified in the analysis.
        attention_patterns: Dictionary mapping pattern type names to counts
            of heads exhibiting each pattern.
        metadata: Additional metadata about the analysis (e.g., token counts,
            number of layers analyzed).

    Example: Creating a basic report
        >>> report = IntrospectionReport(
        ...     prompt="What is AI?",
        ...     response="AI stands for Artificial Intelligence.",
        ...     token_importances=[TokenImportance("AI", 2, 0.9)],
        ...     key_tokens=["AI", "Artificial", "Intelligence"]
        ... )
        >>> report.prompt
        'What is AI?'
        >>> report.key_tokens
        ['AI', 'Artificial', 'Intelligence']

    Example: Using the full introspection pipeline
        >>> introspector = ModelIntrospector()
        >>> report = introspector.introspect(
        ...     prompt="Explain machine learning",
        ...     response="Machine learning is a subset of AI..."
        ... )
        >>> len(report.token_importances) > 0
        True
        >>> len(report.attention_analysis) > 0
        True

    Example: Extracting top tokens
        >>> report = IntrospectionReport(
        ...     prompt="test",
        ...     response="result",
        ...     token_importances=[
        ...         TokenImportance("test", 0, 0.5),
        ...         TokenImportance("result", 1, 0.8),
        ...         TokenImportance("data", 2, 0.9)
        ...     ]
        ... )
        >>> top = report.get_top_tokens(2)
        >>> [t.token for t in top]
        ['data', 'result']

    Example: Analyzing attention pattern distribution
        >>> report = IntrospectionReport(
        ...     prompt="p", response="r",
        ...     attention_patterns={'local': 4, 'global': 2, 'sparse': 2}
        ... )
        >>> report.attention_patterns['local']
        4
        >>> sum(report.attention_patterns.values())
        8
    """

    prompt: str
    response: str
    token_importances: list[TokenImportance] = field(default_factory=list)
    attention_analysis: list[AttentionHead] = field(default_factory=list)
    layer_analyses: list[LayerAnalysis] = field(default_factory=list)
    key_tokens: list[str] = field(default_factory=list)
    attention_patterns: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_top_tokens(self, n: int = 10) -> list[TokenImportance]:
        """
        Get the top n most important tokens from the analysis.

        Tokens are sorted by importance_score in descending order.

        Args:
            n: Maximum number of tokens to return (default: 10).

        Returns:
            list[TokenImportance]: List of up to n TokenImportance objects,
                sorted by importance_score (highest first).

        Example: Getting top 5 tokens
            >>> report = IntrospectionReport(
            ...     prompt="p", response="r",
            ...     token_importances=[
            ...         TokenImportance("a", 0, 0.3),
            ...         TokenImportance("b", 1, 0.9),
            ...         TokenImportance("c", 2, 0.6),
            ...         TokenImportance("d", 3, 0.8)
            ...     ]
            ... )
            >>> top = report.get_top_tokens(3)
            >>> [t.token for t in top]
            ['b', 'd', 'c']

        Example: Handling fewer tokens than requested
            >>> report = IntrospectionReport(
            ...     prompt="p", response="r",
            ...     token_importances=[TokenImportance("only", 0, 0.5)]
            ... )
            >>> top = report.get_top_tokens(10)
            >>> len(top)
            1

        Example: Using for visualization
            >>> report = IntrospectionReport(
            ...     prompt="p", response="r",
            ...     token_importances=[
            ...         TokenImportance("important", 0, 0.95),
            ...         TokenImportance("word", 1, 0.85)
            ...     ]
            ... )
            >>> for t in report.get_top_tokens(2):
            ...     print(f"{t.token}: {t.importance_score:.2f}")
            important: 0.95
            word: 0.85
        """
        sorted_tokens = sorted(
            self.token_importances,
            key=lambda t: t.importance_score,
            reverse=True,
        )
        return sorted_tokens[:n]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert IntrospectionReport to a summary dictionary.

        Note: This returns a summary rather than the full data to keep
        the output manageable. Use individual attributes for full data.

        Returns:
            dict[str, Any]: Summary dictionary containing:
                - 'num_tokens': Total number of tokens analyzed
                - 'key_tokens': List of most important token strings
                - 'attention_patterns': Pattern type to count mapping
                - 'top_tokens': List of top 5 token importance dicts
                - 'metadata': Analysis metadata

        Example: Getting summary statistics
            >>> report = IntrospectionReport(
            ...     prompt="test prompt",
            ...     response="test response",
            ...     token_importances=[TokenImportance("test", 0, 0.7)],
            ...     key_tokens=["test"],
            ...     attention_patterns={"local": 4}
            ... )
            >>> d = report.to_dict()
            >>> d['num_tokens']
            1
            >>> d['key_tokens']
            ['test']

        Example: JSON serialization for logging
            >>> import json
            >>> report = IntrospectionReport("p", "r", [], [], [], ["key"], {"local": 2})
            >>> json_str = json.dumps(report.to_dict())
            >>> "key" in json_str
            True

        Example: Extracting top token summaries
            >>> report = IntrospectionReport(
            ...     prompt="p", response="r",
            ...     token_importances=[
            ...         TokenImportance("high", 0, 0.9),
            ...         TokenImportance("low", 1, 0.2)
            ...     ]
            ... )
            >>> d = report.to_dict()
            >>> d['top_tokens'][0]['token']
            'high'
        """
        return {
            "num_tokens": len(self.token_importances),
            "key_tokens": self.key_tokens,
            "attention_patterns": self.attention_patterns,
            "top_tokens": [t.to_dict() for t in self.get_top_tokens(5)],
            "metadata": self.metadata,
        }


class TokenImportanceEstimator:
    """
    Estimate importance of tokens using heuristic-based methods.

    TokenImportanceEstimator provides a fast, heuristic-based approach to
    estimating token importance without requiring access to model internals.
    It uses linguistic features like stop words, capitalization, word length,
    and position to approximate importance scores.

    The estimator is useful for:
    - Quick importance analysis without model access
    - Preprocessing to identify potentially important tokens
    - Baseline comparisons for more sophisticated methods
    - Educational purposes to understand importance factors

    Class Attributes:
        HIGH_IMPORTANCE_POS: List of POS tags indicating high importance
            (NOUN, VERB, ADJ, NUM).
        STOP_WORDS: Set of common English stop words that typically have
            lower importance scores.

    Example: Basic token importance estimation
        >>> estimator = TokenImportanceEstimator()
        >>> text = "The neural network processes complex data patterns."
        >>> importances = estimator.estimate(text)
        >>> # Content words should have higher scores than stop words
        >>> neural_imp = next(i for i in importances if i.token == "neural")
        >>> the_imp = next(i for i in importances if i.token == "the")
        >>> neural_imp.importance_score > the_imp.importance_score
        True

    Example: Estimation with context
        >>> estimator = TokenImportanceEstimator()
        >>> text = "Paris is a beautiful city."
        >>> context = "What is the capital of France?"
        >>> importances = estimator.estimate(text, context=context)
        >>> # Tokens appearing in context get boosted
        >>> for imp in importances:
        ...     if imp.token in ["paris", "capital", "france"]:
        ...         print(f"{imp.token}: context-relevant")
        paris: context-relevant

    Example: Finding most important tokens
        >>> estimator = TokenImportanceEstimator()
        >>> text = "Machine learning transforms modern technology."
        >>> importances = estimator.estimate(text)
        >>> top_3 = sorted(importances, key=lambda x: x.importance_score, reverse=True)[:3]
        >>> [t.token for t in top_3]  # Content words rank higher
        ['machine', 'learning', 'transforms']

    Example: Batch processing multiple texts
        >>> estimator = TokenImportanceEstimator()
        >>> texts = ["AI is powerful.", "Deep learning works."]
        >>> all_importances = [estimator.estimate(t) for t in texts]
        >>> len(all_importances)
        2

    Note:
        This estimator uses heuristics and does not have access to actual
        model attention weights or gradients. Results are approximations
        useful for quick analysis but should not be treated as ground truth.
    """

    # High importance indicators
    HIGH_IMPORTANCE_POS = ["NOUN", "VERB", "ADJ", "NUM"]

    # Low importance tokens (stop words)
    STOP_WORDS = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "although",
        "though",
        "this",
        "that",
        "these",
        "those",
    }

    def estimate(self, text: str, context: Optional[str] = None) -> list[TokenImportance]:
        """
        Estimate token importance for all tokens in the input text.

        Analyzes each token using multiple heuristic factors including:
        - Stop word detection (lowers score)
        - Context relevance (boosts score if token appears in context)
        - Capitalization (boosts score for proper nouns)
        - Numeric tokens (slightly boosts score)
        - Word length (longer words get slight boost)
        - Position (first/last 10% of tokens get slight boost)
        - Frequency penalty (highly repeated tokens get penalized)

        Args:
            text: Text to analyze. Will be tokenized into words and punctuation.
            context: Optional context string (e.g., the prompt when analyzing
                a response). Tokens appearing in both text and context receive
                a relevance boost.

        Returns:
            list[TokenImportance]: List of TokenImportance objects, one per
                token in the input text, in order of appearance.

        Example: Basic estimation
            >>> estimator = TokenImportanceEstimator()
            >>> importances = estimator.estimate("The cat sat on the mat.")
            >>> len(importances)  # One per token
            7
            >>> importances[0].token
            'the'

        Example: With context for relevance scoring
            >>> estimator = TokenImportanceEstimator()
            >>> response = "Python is a programming language."
            >>> prompt = "What is Python?"
            >>> importances = estimator.estimate(response, context=prompt)
            >>> python_imp = next(i for i in importances if i.token == "python")
            >>> python_imp.importance_score > 0.5  # Boosted by context
            True

        Example: Finding high-importance tokens
            >>> estimator = TokenImportanceEstimator()
            >>> importances = estimator.estimate("Artificial intelligence revolutionizes industry.")
            >>> high_imp = [i for i in importances if i.importance_score > 0.5]
            >>> len(high_imp) > 0
            True

        Example: Comparing stop words vs content words
            >>> estimator = TokenImportanceEstimator()
            >>> importances = estimator.estimate("The algorithm works efficiently.")
            >>> the_score = next(i.importance_score for i in importances if i.token == "the")
            >>> algo_score = next(i.importance_score for i in importances if i.token == "algorithm")
            >>> algo_score > the_score
            True
        """
        tokens = self._tokenize(text)
        importances = []

        # Get context tokens for relevance scoring
        context_tokens = set()
        if context:
            context_tokens = set(self._tokenize(context))

        for i, token in enumerate(tokens):
            importance = self._calculate_importance(token, i, tokens, context_tokens)
            importances.append(importance)

        return importances

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into words and punctuation marks.

        Uses regex to extract word tokens (alphanumeric sequences) and
        punctuation marks separately. All tokens are lowercased.

        Args:
            text: Input text to tokenize.

        Returns:
            list[str]: List of lowercase tokens (words and punctuation).

        Example: Basic tokenization
            >>> estimator = TokenImportanceEstimator()
            >>> estimator._tokenize("Hello, world!")
            ['hello', ',', 'world', '!']

        Example: Handling numbers and mixed content
            >>> estimator = TokenImportanceEstimator()
            >>> estimator._tokenize("GPT-4 costs $20/month.")
            ['gpt', '4', 'costs', '$', '20', '/', 'month', '.']

        Example: Empty and whitespace input
            >>> estimator = TokenImportanceEstimator()
            >>> estimator._tokenize("   ")
            []
        """
        # Split on whitespace and punctuation
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text.lower())
        return tokens

    def _calculate_importance(
        self,
        token: str,
        position: int,
        all_tokens: list[str],
        context_tokens: set[str],
    ) -> TokenImportance:
        """
        Calculate importance score for a single token.

        Applies multiple heuristic factors to compute an importance score
        between 0.0 and 1.0. The score combines linguistic features with
        positional and contextual information.

        Scoring factors:
        - Base score: 0.5
        - Stop word: -0.3
        - In context: +0.2
        - Capitalized: +0.15
        - Is digit: +0.1
        - Length > 6: +0.1
        - Position in first/last 10%: +0.05
        - Frequency > 3: -0.1

        Args:
            token: The token string to score.
            position: Position index of the token in the sequence.
            all_tokens: List of all tokens in the text (for frequency analysis).
            context_tokens: Set of tokens from the context (for relevance).

        Returns:
            TokenImportance: TokenImportance object with computed scores.

        Example: Stop word gets low score
            >>> estimator = TokenImportanceEstimator()
            >>> imp = estimator._calculate_importance("the", 0, ["the", "cat"], set())
            >>> imp.importance_score < 0.5
            True

        Example: Content word with context boost
            >>> estimator = TokenImportanceEstimator()
            >>> imp = estimator._calculate_importance("python", 0, ["python"], {"python"})
            >>> imp.importance_score > 0.5
            True

        Example: Long word bonus
            >>> estimator = TokenImportanceEstimator()
            >>> imp = estimator._calculate_importance("transformer", 0, ["transformer"], set())
            >>> imp.importance_score >= 0.6  # Base 0.5 + length bonus
            True
        """
        score = 0.5  # Base score

        # Penalize stop words
        if token.lower() in self.STOP_WORDS:
            score -= 0.3

        # Boost if appears in context
        if token.lower() in context_tokens:
            score += 0.2

        # Boost capitalized words (likely named entities)
        if token and token[0].isupper():
            score += 0.15

        # Boost numbers
        if token.isdigit():
            score += 0.1

        # Boost longer words (usually more meaningful)
        if len(token) > 6:
            score += 0.1

        # Position-based scoring (first and last parts often important)
        total = len(all_tokens)
        if position < total * 0.1 or position > total * 0.9:
            score += 0.05

        # Frequency penalty (very common tokens less important)
        freq = sum(1 for t in all_tokens if t == token)
        if freq > 3:
            score -= 0.1

        # Clamp score
        score = max(0.0, min(1.0, score))

        return TokenImportance(
            token=token,
            position=position,
            importance_score=score,
            attribution_score=score * 0.8,  # Approximation
            attention_received=score * 0.9,  # Approximation
        )


class AttentionAnalyzer:
    """
    Analyze attention patterns using heuristic simulation for API models.

    AttentionAnalyzer simulates attention pattern analysis for language models
    accessed via API, where actual attention weights are not available. It uses
    text characteristics to infer likely attention patterns and provides
    estimates for entropy and sparsity metrics.

    The analyzer simulates a 4-layer, 2-head-per-layer architecture and
    classifies attention patterns based on:
    - Token overlap between prompt and response
    - Layer depth (early layers tend toward local patterns)
    - Head specialization patterns

    Use cases:
    - Educational exploration of attention concepts
    - Approximate analysis when model internals are unavailable
    - Debugging prompt-response relationships
    - Understanding information flow patterns

    Example: Basic attention analysis
        >>> analyzer = AttentionAnalyzer()
        >>> prompt = "What is the capital of France?"
        >>> response = "The capital of France is Paris."
        >>> heads = analyzer.analyze_text(prompt, response)
        >>> len(heads)  # 4 layers x 2 heads
        8

    Example: Examining individual heads
        >>> analyzer = AttentionAnalyzer()
        >>> heads = analyzer.analyze_text("Translate: Hello", "Bonjour")
        >>> head = heads[0]
        >>> head.layer
        0
        >>> head.pattern_type in list(AttentionPattern)
        True

    Example: Analyzing pattern distribution
        >>> analyzer = AttentionAnalyzer()
        >>> heads = analyzer.analyze_text("Question", "Answer with same words")
        >>> patterns = [h.pattern_type.value for h in heads]
        >>> len(set(patterns)) >= 1  # At least one unique pattern
        True

    Example: Checking entropy and sparsity
        >>> analyzer = AttentionAnalyzer()
        >>> heads = analyzer.analyze_text("Input", "Output")
        >>> all(0 <= h.entropy <= 1 for h in heads)
        True
        >>> all(0 <= h.sparsity <= 1 for h in heads)
        True

    Note:
        Results are heuristic approximations. Actual transformer attention
        patterns may differ significantly from these simulations.
    """

    def analyze_text(self, prompt: str, response: str) -> list[AttentionHead]:
        """
        Analyze attention patterns between prompt and response text.

        Simulates a 4-layer transformer with 2 attention heads per layer,
        inferring pattern types based on text characteristics. Earlier layers
        tend toward local attention patterns, while later layers show more
        global or sparse patterns depending on token overlap.

        Args:
            prompt: The input prompt text.
            response: The model's response text.

        Returns:
            list[AttentionHead]: List of 8 AttentionHead objects (4 layers
                x 2 heads), each containing pattern type, entropy, sparsity,
                and key positions.

        Example: Full analysis
            >>> analyzer = AttentionAnalyzer()
            >>> heads = analyzer.analyze_text(
            ...     prompt="Explain neural networks",
            ...     response="Neural networks are computational models..."
            ... )
            >>> len(heads)
            8
            >>> heads[0].layer
            0

        Example: High overlap prompt/response
            >>> analyzer = AttentionAnalyzer()
            >>> heads = analyzer.analyze_text(
            ...     prompt="The quick brown fox",
            ...     response="The quick brown fox jumps"
            ... )
            >>> # High overlap tends toward VERTICAL or SPARSE patterns
            >>> any(h.pattern_type == AttentionPattern.VERTICAL for h in heads)
            True

        Example: Extracting key positions
            >>> analyzer = AttentionAnalyzer()
            >>> heads = analyzer.analyze_text("word test", "word result")
            >>> any(len(h.key_positions) > 0 for h in heads)
            True

        Example: Comparing early vs late layer patterns
            >>> analyzer = AttentionAnalyzer()
            >>> heads = analyzer.analyze_text("Input text", "Output text")
            >>> early = [h for h in heads if h.layer < 2]
            >>> late = [h for h in heads if h.layer >= 2]
            >>> # Early layers typically have higher sparsity (more local)
            >>> sum(h.sparsity for h in early) / len(early) >= 0
            True
        """
        prompt_tokens = self._tokenize(prompt)
        response_tokens = self._tokenize(response)

        analyses = []

        # Simulate multiple layers and heads
        for layer in range(4):  # Simulate 4 layers
            for head in range(2):  # Simulate 2 heads per layer
                pattern = self._infer_pattern(prompt_tokens, response_tokens, layer, head)
                analysis = AttentionHead(
                    layer=layer,
                    head=head,
                    pattern_type=pattern,
                    entropy=self._estimate_entropy(pattern),
                    sparsity=self._estimate_sparsity(pattern),
                    key_positions=self._find_key_positions(prompt_tokens, response_tokens),
                )
                analyses.append(analysis)

        return analyses

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text using the shared tokenization utility.

        Delegates to insideLLMs.nlp.tokenization.word_tokenize_regex for
        consistent tokenization across the module.

        Args:
            text: Input text to tokenize.

        Returns:
            list[str]: List of token strings.

        Example: Basic tokenization
            >>> analyzer = AttentionAnalyzer()
            >>> tokens = analyzer._tokenize("Hello world!")
            >>> "Hello" in tokens or "hello" in tokens
            True

        Example: Multi-word tokenization
            >>> analyzer = AttentionAnalyzer()
            >>> tokens = analyzer._tokenize("The quick brown fox")
            >>> len(tokens) >= 4
            True
        """
        return word_tokenize_regex(text)

    def _infer_pattern(
        self,
        prompt_tokens: list[str],
        response_tokens: list[str],
        layer: int,
        head: int,
    ) -> AttentionPattern:
        """
        Infer attention pattern type from text characteristics.

        Uses token overlap and layer depth to estimate the most likely
        attention pattern. Early layers tend toward local patterns (syntactic
        processing), while later layers show global or sparse patterns
        (semantic processing).

        Pattern inference rules:
        - Early layers (0-1) + high overlap -> VERTICAL (copying)
        - Early layers (0-1) + low overlap -> LOCAL (syntax)
        - Late layers (2-3) + medium+ overlap -> SPARSE (selective)
        - Late layers (2-3) + low overlap -> GLOBAL (semantic)

        Args:
            prompt_tokens: List of tokens from the prompt.
            response_tokens: List of tokens from the response.
            layer: Layer index (0-3 in the simulated model).
            head: Head index (unused in current implementation).

        Returns:
            AttentionPattern: The inferred pattern type.

        Example: High overlap early layer
            >>> analyzer = AttentionAnalyzer()
            >>> pattern = analyzer._infer_pattern(
            ...     ["the", "cat"], ["the", "cat", "sat"], layer=0, head=0
            ... )
            >>> pattern == AttentionPattern.VERTICAL
            True

        Example: Low overlap early layer
            >>> analyzer = AttentionAnalyzer()
            >>> pattern = analyzer._infer_pattern(
            ...     ["hello"], ["goodbye"], layer=1, head=0
            ... )
            >>> pattern == AttentionPattern.LOCAL
            True

        Example: Late layer global pattern
            >>> analyzer = AttentionAnalyzer()
            >>> pattern = analyzer._infer_pattern(
            ...     ["unique", "words"], ["different", "response"], layer=3, head=0
            ... )
            >>> pattern == AttentionPattern.GLOBAL
            True
        """
        # Check for copy/reference behavior
        prompt_set = set(prompt_tokens)
        response_set = set(response_tokens)
        overlap = len(prompt_set & response_set) / len(prompt_set) if prompt_set else 0

        # Early layers tend to be more local
        if layer < 2:
            if overlap > 0.5:
                return AttentionPattern.VERTICAL  # Copying from input
            return AttentionPattern.LOCAL

        # Later layers tend to be more global/sparse
        if overlap > 0.3:
            return AttentionPattern.SPARSE
        return AttentionPattern.GLOBAL

    def _estimate_entropy(self, pattern: AttentionPattern) -> float:
        """
        Estimate attention entropy based on pattern type.

        Entropy measures how uniformly distributed the attention weights are.
        Higher entropy indicates attention spread across many tokens, while
        lower entropy indicates focused attention on few tokens.

        Entropy estimates by pattern:
        - LOCAL: 0.3 (focused on nearby tokens)
        - GLOBAL: 0.8 (spread across all tokens)
        - SPARSE: 0.4 (focused on specific tokens)
        - DIAGONAL: 0.2 (very focused, self-attention)
        - VERTICAL: 0.5 (moderate, position-based)
        - BLOCK: 0.5 (moderate, segment-based)

        Args:
            pattern: The AttentionPattern type.

        Returns:
            float: Estimated entropy value between 0.0 and 1.0.

        Example: Global pattern has high entropy
            >>> analyzer = AttentionAnalyzer()
            >>> analyzer._estimate_entropy(AttentionPattern.GLOBAL)
            0.8

        Example: Diagonal pattern has low entropy
            >>> analyzer = AttentionAnalyzer()
            >>> analyzer._estimate_entropy(AttentionPattern.DIAGONAL)
            0.2

        Example: Unknown pattern defaults to 0.5
            >>> analyzer = AttentionAnalyzer()
            >>> # All known patterns return their mapped value
            >>> all(analyzer._estimate_entropy(p) is not None
            ...     for p in AttentionPattern)
            True
        """
        entropy_map = {
            AttentionPattern.LOCAL: 0.3,
            AttentionPattern.GLOBAL: 0.8,
            AttentionPattern.SPARSE: 0.4,
            AttentionPattern.DIAGONAL: 0.2,
            AttentionPattern.VERTICAL: 0.5,
            AttentionPattern.BLOCK: 0.5,
        }
        return entropy_map.get(pattern, 0.5)

    def _estimate_sparsity(self, pattern: AttentionPattern) -> float:
        """
        Estimate attention sparsity based on pattern type.

        Sparsity measures how concentrated attention is on few tokens.
        Higher sparsity indicates attention focused on fewer positions,
        while lower sparsity indicates diffuse attention.

        Sparsity estimates by pattern:
        - LOCAL: 0.7 (focused on nearby tokens)
        - GLOBAL: 0.2 (spread across all tokens)
        - SPARSE: 0.8 (highly selective)
        - DIAGONAL: 0.9 (very focused, self-attention)
        - VERTICAL: 0.6 (moderately focused)
        - BLOCK: 0.5 (moderate)

        Args:
            pattern: The AttentionPattern type.

        Returns:
            float: Estimated sparsity value between 0.0 and 1.0.

        Example: Sparse pattern has high sparsity
            >>> analyzer = AttentionAnalyzer()
            >>> analyzer._estimate_sparsity(AttentionPattern.SPARSE)
            0.8

        Example: Global pattern has low sparsity
            >>> analyzer = AttentionAnalyzer()
            >>> analyzer._estimate_sparsity(AttentionPattern.GLOBAL)
            0.2

        Example: Comparing patterns
            >>> analyzer = AttentionAnalyzer()
            >>> local = analyzer._estimate_sparsity(AttentionPattern.LOCAL)
            >>> glob = analyzer._estimate_sparsity(AttentionPattern.GLOBAL)
            >>> local > glob  # Local is more sparse than global
            True
        """
        sparsity_map = {
            AttentionPattern.LOCAL: 0.7,
            AttentionPattern.GLOBAL: 0.2,
            AttentionPattern.SPARSE: 0.8,
            AttentionPattern.DIAGONAL: 0.9,
            AttentionPattern.VERTICAL: 0.6,
            AttentionPattern.BLOCK: 0.5,
        }
        return sparsity_map.get(pattern, 0.5)

    def _find_key_positions(
        self, prompt_tokens: list[str], response_tokens: list[str]
    ) -> list[int]:
        """
        Find positions in the response that match tokens from the prompt.

        Key positions are response token indices where the token also appears
        in the prompt. These positions often receive significant attention
        as the model references or copies from the input.

        Args:
            prompt_tokens: List of tokens from the prompt.
            response_tokens: List of tokens from the response.

        Returns:
            list[int]: List of position indices (0-indexed) in the response
                where tokens match prompt tokens. Limited to first 10 matches.

        Example: Finding matching positions
            >>> analyzer = AttentionAnalyzer()
            >>> positions = analyzer._find_key_positions(
            ...     ["the", "cat"], ["the", "cat", "sat", "on", "the", "mat"]
            ... )
            >>> 0 in positions  # "the" at position 0
            True
            >>> 1 in positions  # "cat" at position 1
            True

        Example: No overlap
            >>> analyzer = AttentionAnalyzer()
            >>> positions = analyzer._find_key_positions(
            ...     ["hello"], ["goodbye", "world"]
            ... )
            >>> positions
            []

        Example: Limit to 10 positions
            >>> analyzer = AttentionAnalyzer()
            >>> prompt = ["a"]
            >>> response = ["a"] * 20
            >>> positions = analyzer._find_key_positions(prompt, response)
            >>> len(positions) <= 10
            True
        """
        prompt_set = set(prompt_tokens)
        key_positions = []

        for i, token in enumerate(response_tokens):
            if token in prompt_set:
                key_positions.append(i)

        return key_positions[:10]  # Limit to 10


class LayerAnalyzer:
    """
    Analyze layer-wise representations using heuristic simulation.

    LayerAnalyzer simulates the analysis of how representations evolve across
    transformer layers. Since API models don't expose intermediate representations,
    this analyzer uses established patterns from transformer research to estimate
    layer characteristics.

    Key patterns simulated:
    - Representation norm decreases with depth (task-specific compression)
    - Token similarity decreases (more differentiated representations)
    - Information retention gradually decreases (task-irrelevant filtering)
    - Feature types progress from syntactic to semantic

    Use cases:
    - Understanding transformer layer progression
    - Educational exploration of layer-wise processing
    - Baseline for comparing actual model internals
    - Visualization of information flow

    Example: Basic layer analysis
        >>> analyzer = LayerAnalyzer()
        >>> layers = analyzer.analyze("Sample text for analysis")
        >>> len(layers)
        12
        >>> layers[0].layer_index
        0

    Example: Custom number of layers
        >>> analyzer = LayerAnalyzer()
        >>> layers = analyzer.analyze("Text", num_layers=6)
        >>> len(layers)
        6

    Example: Observing metric progression
        >>> analyzer = LayerAnalyzer()
        >>> layers = analyzer.analyze("Test input")
        >>> # Representation norm decreases with depth
        >>> layers[0].representation_norm > layers[-1].representation_norm
        True
        >>> # Token similarity decreases with depth
        >>> layers[0].token_similarity > layers[-1].token_similarity
        True

    Example: Examining feature progression
        >>> analyzer = LayerAnalyzer()
        >>> layers = analyzer.analyze("Example")
        >>> # Early layers capture syntax
        >>> "token_embedding" in layers[0].key_features
        True
        >>> # Late layers capture semantics
        >>> "global_context" in layers[-1].key_features
        True

    Note:
        These are simulated values based on general transformer behavior.
        Actual layer characteristics vary significantly by model architecture.
    """

    def analyze(
        self,
        text: str,
        num_layers: int = 12,
    ) -> list[LayerAnalysis]:
        """
        Analyze layer-wise representations for input text.

        Simulates representation characteristics at each layer based on
        established patterns in transformer models. Metrics are computed
        as functions of layer depth to reflect typical progression patterns.

        Simulation formulas:
        - representation_norm = 1.0 - (layer/total) * 0.3
        - token_similarity = 0.8 - (layer/total) * 0.5
        - information_retention = 1.0 - (layer/total) * 0.2

        Args:
            text: Input text to analyze. Currently used for context but
                doesn't affect the simulated metrics.
            num_layers: Number of layers to simulate (default: 12).

        Returns:
            list[LayerAnalysis]: List of LayerAnalysis objects, one per
                simulated layer, ordered from layer 0 to num_layers-1.

        Example: Standard 12-layer analysis
            >>> analyzer = LayerAnalyzer()
            >>> layers = analyzer.analyze("Deep learning is fascinating.")
            >>> len(layers)
            12
            >>> all(l.layer_index == i for i, l in enumerate(layers))
            True

        Example: Checking metric ranges
            >>> analyzer = LayerAnalyzer()
            >>> layers = analyzer.analyze("Test", num_layers=12)
            >>> all(0.7 <= l.representation_norm <= 1.0 for l in layers)
            True
            >>> all(0.3 <= l.token_similarity <= 0.8 for l in layers)
            True

        Example: Small model simulation
            >>> analyzer = LayerAnalyzer()
            >>> layers = analyzer.analyze("Short text", num_layers=3)
            >>> len(layers)
            3
            >>> layers[2].layer_index
            2

        Example: Converting results to dict
            >>> analyzer = LayerAnalyzer()
            >>> layers = analyzer.analyze("Data")
            >>> data = [l.to_dict() for l in layers]
            >>> len(data)
            12
        """
        analyses = []

        for layer in range(num_layers):
            # Simulate layer characteristics
            # Early layers: more syntactic, high information retention
            # Later layers: more semantic, lower token similarity

            progress = layer / num_layers

            analysis = LayerAnalysis(
                layer_index=layer,
                representation_norm=1.0 - progress * 0.3,  # Decreases
                token_similarity=0.8 - progress * 0.5,  # Decreases
                information_retention=1.0 - progress * 0.2,  # Decreases slightly
                key_features=self._infer_features(layer, num_layers),
            )
            analyses.append(analysis)

        return analyses

    def _infer_features(self, layer: int, num_layers: int) -> list[str]:
        """
        Infer the types of features captured at a given layer depth.

        Based on transformer research, different depths tend to capture
        different types of information:
        - Early layers (0-25%): Token-level features (embeddings, position, syntax)
        - Lower-middle (25-50%): Phrase-level features (structure, local context)
        - Upper-middle (50-75%): Sentence-level features (meaning, relations)
        - Late layers (75-100%): Task-level features (global context, output prep)

        Args:
            layer: The layer index (0-indexed).
            num_layers: Total number of layers in the model.

        Returns:
            list[str]: List of feature type names captured at this layer.

        Example: Early layer features
            >>> analyzer = LayerAnalyzer()
            >>> features = analyzer._infer_features(0, 12)
            >>> "token_embedding" in features
            True

        Example: Late layer features
            >>> analyzer = LayerAnalyzer()
            >>> features = analyzer._infer_features(11, 12)
            >>> "global_context" in features
            True

        Example: Middle layer transition
            >>> analyzer = LayerAnalyzer()
            >>> early = analyzer._infer_features(2, 12)
            >>> late = analyzer._infer_features(10, 12)
            >>> early != late
            True
        """
        progress = layer / num_layers

        if progress < 0.25:
            return ["token_embedding", "position_encoding", "local_syntax"]
        elif progress < 0.5:
            return ["phrase_structure", "local_context", "basic_semantics"]
        elif progress < 0.75:
            return ["sentence_meaning", "cross_attention", "entity_relations"]
        else:
            return ["global_context", "task_specific", "output_preparation"]


@dataclass
class SaliencyMap:
    """
    Saliency map mapping input tokens to their importance scores.

    SaliencyMap provides a simple representation of which input tokens
    are most relevant to the model's output. It supports visualization
    through text highlighting and extraction of top salient tokens.

    Attributes:
        tokens: List of input token strings.
        scores: List of saliency scores (0.0 to 1.0), parallel to tokens.
        method: The method used to compute saliency (default: "gradient").

    Example: Creating a saliency map
        >>> smap = SaliencyMap(
        ...     tokens=["the", "cat", "sat"],
        ...     scores=[0.2, 0.8, 0.5],
        ...     method="importance"
        ... )
        >>> smap.tokens
        ['the', 'cat', 'sat']
        >>> smap.method
        'importance'

    Example: Highlighting salient tokens
        >>> smap = SaliencyMap(["hello", "world"], [0.3, 0.7])
        >>> smap.get_highlighted_text(threshold=0.5)
        'hello **world**'

    Example: Getting top salient tokens
        >>> smap = SaliencyMap(
        ...     tokens=["a", "b", "c", "d"],
        ...     scores=[0.1, 0.9, 0.5, 0.7]
        ... )
        >>> top = smap.get_top_salient(2)
        >>> top[0][0]  # Highest scoring token
        'b'

    Example: Converting to dictionary
        >>> smap = SaliencyMap(["test"], [0.5])
        >>> d = smap.to_dict()
        >>> d['method']
        'gradient'
    """

    tokens: list[str]
    scores: list[float]
    method: str = "gradient"

    def get_highlighted_text(self, threshold: float = 0.5) -> str:
        """
        Get text with high-saliency tokens highlighted using markdown bold.

        Tokens with saliency score at or above the threshold are wrapped
        in **asterisks** for markdown bold formatting.

        Args:
            threshold: Minimum score for highlighting (default: 0.5).
                Tokens with scores >= threshold are highlighted.

        Returns:
            str: Space-joined text with high-saliency tokens in **bold**.

        Example: Basic highlighting
            >>> smap = SaliencyMap(["important", "word", "here"], [0.9, 0.3, 0.6])
            >>> smap.get_highlighted_text(threshold=0.5)
            '**important** word **here**'

        Example: High threshold highlights fewer tokens
            >>> smap = SaliencyMap(["a", "b", "c"], [0.5, 0.7, 0.9])
            >>> smap.get_highlighted_text(threshold=0.8)
            'a b **c**'

        Example: Zero threshold highlights all
            >>> smap = SaliencyMap(["x", "y"], [0.1, 0.2])
            >>> smap.get_highlighted_text(threshold=0.0)
            '**x** **y**'

        Example: High threshold highlights none
            >>> smap = SaliencyMap(["x", "y"], [0.1, 0.2])
            >>> smap.get_highlighted_text(threshold=0.5)
            'x y'
        """
        highlighted = []
        for token, score in zip(self.tokens, self.scores):
            if score >= threshold:
                highlighted.append(f"**{token}**")
            else:
                highlighted.append(token)
        return " ".join(highlighted)

    def get_top_salient(self, n: int = 10) -> list[tuple[str, float]]:
        """
        Get the top n most salient tokens with their scores.

        Returns tokens sorted by saliency score in descending order.

        Args:
            n: Maximum number of tokens to return (default: 10).

        Returns:
            list[tuple[str, float]]: List of (token, score) tuples,
                sorted by score descending, limited to n entries.

        Example: Getting top 3 tokens
            >>> smap = SaliencyMap(
            ...     tokens=["a", "b", "c", "d", "e"],
            ...     scores=[0.1, 0.5, 0.9, 0.3, 0.7]
            ... )
            >>> top = smap.get_top_salient(3)
            >>> [t[0] for t in top]
            ['c', 'e', 'b']

        Example: Fewer tokens than requested
            >>> smap = SaliencyMap(["only", "two"], [0.5, 0.8])
            >>> top = smap.get_top_salient(10)
            >>> len(top)
            2

        Example: Accessing scores
            >>> smap = SaliencyMap(["high", "low"], [0.9, 0.1])
            >>> top = smap.get_top_salient(1)
            >>> top[0]
            ('high', 0.9)
        """
        pairs = list(zip(self.tokens, self.scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:n]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert SaliencyMap to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing:
                - 'tokens': List of token strings
                - 'scores': List of saliency scores
                - 'method': Saliency estimation method
                - 'top_salient': Top 5 most salient token-score pairs

        Example: Basic conversion
            >>> smap = SaliencyMap(["a", "b"], [0.3, 0.7], method="gradient")
            >>> d = smap.to_dict()
            >>> d['tokens']
            ['a', 'b']
            >>> d['method']
            'gradient'

        Example: JSON serialization
            >>> import json
            >>> smap = SaliencyMap(["test"], [0.5])
            >>> json_str = json.dumps(smap.to_dict())
            >>> "test" in json_str
            True

        Example: Accessing top salient from dict
            >>> smap = SaliencyMap(["x", "y", "z"], [0.1, 0.9, 0.5])
            >>> d = smap.to_dict()
            >>> d['top_salient'][0][0]  # Top token
            'y'
        """
        return {
            "tokens": self.tokens,
            "scores": self.scores,
            "method": self.method,
            "top_salient": self.get_top_salient(5),
        }


class SaliencyEstimator:
    """
    Estimate input saliency scores for model outputs.

    SaliencyEstimator wraps TokenImportanceEstimator to produce SaliencyMap
    objects that map input tokens to their estimated importance for the
    model's output. This is useful for understanding which parts of the
    input most influenced the response.

    The estimator uses heuristic-based importance scoring since actual
    gradient-based saliency computation requires access to model internals
    not available through APIs.

    Attributes:
        importance_estimator: Internal TokenImportanceEstimator instance
            used for computing token importance scores.

    Example: Basic saliency estimation
        >>> estimator = SaliencyEstimator()
        >>> smap = estimator.estimate(
        ...     prompt="What is machine learning?",
        ...     response="Machine learning is a type of AI..."
        ... )
        >>> len(smap.tokens) > 0
        True

    Example: Getting highlighted text
        >>> estimator = SaliencyEstimator()
        >>> smap = estimator.estimate("AI question", "AI answer")
        >>> highlighted = smap.get_highlighted_text()
        >>> isinstance(highlighted, str)
        True

    Example: Custom estimation method label
        >>> estimator = SaliencyEstimator()
        >>> smap = estimator.estimate("Input", "Output", method="custom")
        >>> smap.method
        'custom'

    Example: Finding most salient tokens
        >>> estimator = SaliencyEstimator()
        >>> smap = estimator.estimate(
        ...     prompt="Explain neural networks briefly",
        ...     response="Neural networks process data"
        ... )
        >>> top = smap.get_top_salient(3)
        >>> len(top) <= 3
        True

    Note:
        The 'method' parameter is a label only and does not change the
        underlying computation, which always uses heuristic importance
        estimation.
    """

    def __init__(self):
        """
        Initialize the SaliencyEstimator.

        Creates an internal TokenImportanceEstimator instance for computing
        token importance scores.

        Example: Creating an estimator
            >>> estimator = SaliencyEstimator()
            >>> estimator.importance_estimator is not None
            True
        """
        self.importance_estimator = TokenImportanceEstimator()

    def estimate(self, prompt: str, response: str, method: str = "importance") -> SaliencyMap:
        """
        Estimate a saliency map for the input prompt.

        Computes importance scores for each token in the prompt, taking
        the response into account as context. Returns a SaliencyMap
        with tokens and their corresponding saliency scores.

        Args:
            prompt: The input prompt to analyze for saliency.
            response: The model's response, used as context for computing
                relevance scores.
            method: Label for the estimation method (default: "importance").
                This is metadata only and doesn't change computation.

        Returns:
            SaliencyMap: A saliency map containing tokens, scores, and method.

        Example: Basic estimation
            >>> estimator = SaliencyEstimator()
            >>> smap = estimator.estimate(
            ...     prompt="What is Python?",
            ...     response="Python is a programming language."
            ... )
            >>> len(smap.tokens) > 0
            True
            >>> len(smap.scores) == len(smap.tokens)
            True

        Example: Checking score ranges
            >>> estimator = SaliencyEstimator()
            >>> smap = estimator.estimate("Test input", "Test output")
            >>> all(0 <= s <= 1 for s in smap.scores)
            True

        Example: Method label is preserved
            >>> estimator = SaliencyEstimator()
            >>> smap = estimator.estimate("A", "B", method="gradient")
            >>> smap.method
            'gradient'

        Example: Context affects scores
            >>> estimator = SaliencyEstimator()
            >>> smap = estimator.estimate(
            ...     prompt="Python programming language",
            ...     response="Python is widely used"
            ... )
            >>> # Tokens in both prompt and response get boosted
            >>> python_score = next(s for t, s in zip(smap.tokens, smap.scores)
            ...                     if t == "python")
            >>> python_score > 0.5
            True
        """
        importances = self.importance_estimator.estimate(prompt, response)

        tokens = [imp.token for imp in importances]
        scores = [imp.importance_score for imp in importances]

        return SaliencyMap(
            tokens=tokens,
            scores=scores,
            method=method,
        )


class ModelIntrospector:
    """
    Comprehensive model introspection combining multiple analysis techniques.

    ModelIntrospector is the main entry point for analyzing language model
    behavior. It combines token importance estimation, attention pattern
    analysis, layer-wise representation analysis, and saliency estimation
    into a single comprehensive IntrospectionReport.

    The introspector orchestrates multiple analyzers:
    - TokenImportanceEstimator: Estimates token importance scores
    - AttentionAnalyzer: Analyzes attention patterns across layers
    - LayerAnalyzer: Analyzes layer-wise representations
    - SaliencyEstimator: Computes input saliency maps

    Attributes:
        token_estimator: TokenImportanceEstimator instance
        attention_analyzer: AttentionAnalyzer instance
        layer_analyzer: LayerAnalyzer instance
        saliency_estimator: SaliencyEstimator instance

    Example: Basic introspection
        >>> introspector = ModelIntrospector()
        >>> report = introspector.introspect(
        ...     prompt="What is deep learning?",
        ...     response="Deep learning is a subset of machine learning..."
        ... )
        >>> len(report.token_importances) > 0
        True
        >>> len(report.attention_analysis) > 0
        True

    Example: Extracting key insights
        >>> introspector = ModelIntrospector()
        >>> report = introspector.introspect("Explain AI", "AI means...")
        >>> # Get most important tokens
        >>> top_tokens = report.get_top_tokens(5)
        >>> len(top_tokens) <= 5
        True
        >>> # Get attention pattern distribution
        >>> report.attention_patterns  # dict of pattern -> count
        {...}

    Example: Without layer analysis (faster)
        >>> introspector = ModelIntrospector()
        >>> report = introspector.introspect(
        ...     "Quick analysis",
        ...     "Fast response",
        ...     include_layers=False
        ... )
        >>> len(report.layer_analyses)
        0

    Example: Custom layer count
        >>> introspector = ModelIntrospector()
        >>> report = introspector.introspect(
        ...     "Prompt", "Response",
        ...     include_layers=True,
        ...     num_layers=6
        ... )
        >>> len(report.layer_analyses)
        6

    Note:
        All analyses are heuristic-based simulations since API models don't
        expose internal states. Results are approximations useful for
        understanding and debugging, not ground truth measurements.
    """

    def __init__(self):
        """
        Initialize the ModelIntrospector with all component analyzers.

        Creates instances of TokenImportanceEstimator, AttentionAnalyzer,
        LayerAnalyzer, and SaliencyEstimator for comprehensive analysis.

        Example: Creating an introspector
            >>> introspector = ModelIntrospector()
            >>> introspector.token_estimator is not None
            True
            >>> introspector.attention_analyzer is not None
            True

        Example: Reusing an introspector
            >>> introspector = ModelIntrospector()
            >>> report1 = introspector.introspect("Q1", "A1")
            >>> report2 = introspector.introspect("Q2", "A2")
            >>> report1.prompt != report2.prompt
            True
        """
        self.token_estimator = TokenImportanceEstimator()
        self.attention_analyzer = AttentionAnalyzer()
        self.layer_analyzer = LayerAnalyzer()
        self.saliency_estimator = SaliencyEstimator()

    def introspect(
        self,
        prompt: str,
        response: str,
        include_layers: bool = True,
        num_layers: int = 12,
    ) -> IntrospectionReport:
        """
        Perform comprehensive introspection on a prompt-response pair.

        Analyzes the interaction between prompt and response using multiple
        techniques and returns a detailed IntrospectionReport containing:
        - Token importance scores for all tokens
        - Attention head analyses (8 heads across 4 simulated layers)
        - Layer-wise representation analyses (optional)
        - Key tokens (importance > 0.6)
        - Attention pattern distribution

        Args:
            prompt: The input prompt text.
            response: The model's response text.
            include_layers: Whether to include layer-wise analysis
                (default: True). Set to False for faster analysis.
            num_layers: Number of layers to simulate for layer analysis
                (default: 12).

        Returns:
            IntrospectionReport: Comprehensive report containing all analyses.

        Example: Full introspection
            >>> introspector = ModelIntrospector()
            >>> report = introspector.introspect(
            ...     prompt="Explain quantum computing",
            ...     response="Quantum computing uses quantum mechanics..."
            ... )
            >>> report.prompt
            'Explain quantum computing'
            >>> len(report.token_importances) > 0
            True
            >>> len(report.attention_analysis)
            8

        Example: Accessing report metadata
            >>> introspector = ModelIntrospector()
            >>> report = introspector.introspect("Short", "Response")
            >>> 'num_prompt_tokens' in report.metadata
            True
            >>> 'num_response_tokens' in report.metadata
            True

        Example: Quick analysis without layers
            >>> introspector = ModelIntrospector()
            >>> report = introspector.introspect(
            ...     "Fast", "Quick",
            ...     include_layers=False
            ... )
            >>> report.metadata['num_layers_analyzed']
            0

        Example: Converting report to dict
            >>> introspector = ModelIntrospector()
            >>> report = introspector.introspect("Test", "Test response")
            >>> d = report.to_dict()
            >>> 'key_tokens' in d
            True
            >>> 'attention_patterns' in d
            True
        """
        # Token importance
        prompt_importance = self.token_estimator.estimate(prompt)
        response_importance = self.token_estimator.estimate(response, context=prompt)

        all_importance = prompt_importance + response_importance

        # Attention analysis
        attention_analysis = self.attention_analyzer.analyze_text(prompt, response)

        # Layer analysis
        layer_analyses = []
        if include_layers:
            layer_analyses = self.layer_analyzer.analyze(prompt + " " + response, num_layers)

        # Extract key tokens
        key_tokens = [
            imp.token
            for imp in sorted(all_importance, key=lambda x: x.importance_score, reverse=True)[:10]
            if imp.importance_score > 0.6
        ]

        # Count attention patterns
        pattern_counts: dict[str, int] = {}
        for head in attention_analysis:
            pattern = head.pattern_type.value
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        return IntrospectionReport(
            prompt=prompt,
            response=response,
            token_importances=all_importance,
            attention_analysis=attention_analysis,
            layer_analyses=layer_analyses,
            key_tokens=key_tokens,
            attention_patterns=pattern_counts,
            metadata={
                "num_prompt_tokens": len(prompt_importance),
                "num_response_tokens": len(response_importance),
                "num_layers_analyzed": num_layers if include_layers else 0,
            },
        )


@dataclass
class ActivationProfile:
    """
    Profile of model activations at a single layer (simulated).

    ActivationProfile captures statistics about the activation patterns
    at a specific layer, including mean and max activation values,
    sparsity (fraction of near-zero activations), and indices of the
    most active neurons.

    Attributes:
        layer: The layer index (0-indexed) being profiled.
        mean_activation: Average activation value across all neurons.
            Typically decreases in later layers.
        max_activation: Maximum activation value observed. Typically
            increases in later layers as representations become more peaked.
        sparsity: Fraction of neurons with near-zero activations (0.0 to 1.0).
            Higher in later layers as representations become more selective.
        top_neurons: List of indices of the most highly activated neurons.

    Example: Creating an activation profile
        >>> profile = ActivationProfile(
        ...     layer=5,
        ...     mean_activation=0.4,
        ...     max_activation=2.5,
        ...     sparsity=0.6,
        ...     top_neurons=[50, 51, 52, 53, 54]
        ... )
        >>> profile.layer
        5
        >>> profile.sparsity
        0.6

    Example: Analyzing activation characteristics
        >>> early = ActivationProfile(0, 0.5, 2.0, 0.3, [0, 1, 2])
        >>> late = ActivationProfile(11, 0.3, 3.0, 0.7, [110, 111, 112])
        >>> # Later layers are more sparse
        >>> late.sparsity > early.sparsity
        True
        >>> # Later layers have higher peak activations
        >>> late.max_activation > early.max_activation
        True

    Example: Converting to dictionary
        >>> profile = ActivationProfile(2, 0.45, 2.2, 0.4, [20, 21])
        >>> d = profile.to_dict()
        >>> d['layer']
        2
        >>> d['sparsity']
        0.4

    Example: Identifying top neurons
        >>> profile = ActivationProfile(3, 0.4, 2.3, 0.5, [30, 35, 42, 48])
        >>> profile.top_neurons[0]
        30
    """

    layer: int
    mean_activation: float
    max_activation: float
    sparsity: float  # Fraction of near-zero activations
    top_neurons: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert ActivationProfile to a dictionary representation.

        Returns:
            dict[str, Any]: Dictionary containing all activation profile fields.
                Keys: 'layer', 'mean_activation', 'max_activation',
                'sparsity', 'top_neurons'.

        Example: Basic conversion
            >>> profile = ActivationProfile(1, 0.48, 2.1, 0.35, [10, 11, 12])
            >>> d = profile.to_dict()
            >>> d['layer']
            1
            >>> d['mean_activation']
            0.48

        Example: JSON serialization
            >>> import json
            >>> profile = ActivationProfile(0, 0.5, 2.0, 0.3, [1, 2, 3])
            >>> json_str = json.dumps(profile.to_dict())
            >>> "mean_activation" in json_str
            True

        Example: Batch conversion for visualization
            >>> profiles = [ActivationProfile(i, 0.5-i*0.02, 2.0+i*0.1, 0.3+i*0.03, [])
            ...             for i in range(3)]
            >>> data = [p.to_dict() for p in profiles]
            >>> [d['layer'] for d in data]
            [0, 1, 2]
        """
        return {
            "layer": self.layer,
            "mean_activation": self.mean_activation,
            "max_activation": self.max_activation,
            "sparsity": self.sparsity,
            "top_neurons": self.top_neurons,
        }


class ActivationProfiler:
    """
    Profile model activations using heuristic simulation.

    ActivationProfiler simulates activation statistics across transformer
    layers based on established patterns from deep learning research.
    Since API models don't expose actual activations, this provides
    approximate profiles useful for understanding and education.

    Key patterns simulated:
    - Mean activation decreases with depth (more selective neurons)
    - Max activation increases with depth (more peaked responses)
    - Sparsity increases with depth (more specialized representations)
    - Top neurons are assigned based on layer index

    Example: Basic profiling
        >>> profiler = ActivationProfiler()
        >>> profiles = profiler.profile("Sample text")
        >>> len(profiles)
        12
        >>> profiles[0].layer
        0

    Example: Custom layer count
        >>> profiler = ActivationProfiler()
        >>> profiles = profiler.profile("Text", num_layers=6)
        >>> len(profiles)
        6

    Example: Observing sparsity progression
        >>> profiler = ActivationProfiler()
        >>> profiles = profiler.profile("Test")
        >>> # Sparsity increases with depth
        >>> profiles[0].sparsity < profiles[-1].sparsity
        True

    Example: Analyzing activation trends
        >>> profiler = ActivationProfiler()
        >>> profiles = profiler.profile("Analysis text")
        >>> # Mean activation decreases
        >>> profiles[0].mean_activation > profiles[-1].mean_activation
        True
        >>> # Max activation increases
        >>> profiles[0].max_activation < profiles[-1].max_activation
        True

    Note:
        These are simulated values based on general transformer behavior.
        Actual activation patterns vary by model and input.
    """

    def profile(self, text: str, num_layers: int = 12) -> list[ActivationProfile]:
        """
        Profile activations for input text across all layers.

        Simulates activation statistics at each layer based on depth.
        Metrics follow typical patterns observed in transformer models:
        - mean_activation = 0.5 - (progress * 0.2)
        - max_activation = 2.0 + (progress * 1.0)
        - sparsity = 0.3 + (progress * 0.4)
        - top_neurons = layer-specific indices

        Args:
            text: Input text to profile. Currently used for context but
                doesn't affect the simulated metrics.
            num_layers: Number of layers to profile (default: 12).

        Returns:
            list[ActivationProfile]: List of ActivationProfile objects,
                one per layer, ordered from layer 0 to num_layers-1.

        Example: Standard profiling
            >>> profiler = ActivationProfiler()
            >>> profiles = profiler.profile("Neural networks are powerful.")
            >>> len(profiles)
            12
            >>> all(p.layer == i for i, p in enumerate(profiles))
            True

        Example: Checking metric ranges
            >>> profiler = ActivationProfiler()
            >>> profiles = profiler.profile("Test", num_layers=12)
            >>> all(0.3 <= p.mean_activation <= 0.5 for p in profiles)
            True
            >>> all(2.0 <= p.max_activation <= 3.0 for p in profiles)
            True
            >>> all(0.3 <= p.sparsity <= 0.7 for p in profiles)
            True

        Example: Small model simulation
            >>> profiler = ActivationProfiler()
            >>> profiles = profiler.profile("Short", num_layers=3)
            >>> len(profiles)
            3

        Example: Accessing top neurons
            >>> profiler = ActivationProfiler()
            >>> profiles = profiler.profile("Test")
            >>> len(profiles[0].top_neurons)
            5
            >>> profiles[0].top_neurons[0]  # First layer starts at 0
            0
        """
        profiles = []

        for layer in range(num_layers):
            # Simulate activation characteristics
            # Later layers tend to have sparser, more task-specific activations
            progress = layer / num_layers

            profile = ActivationProfile(
                layer=layer,
                mean_activation=0.5 - progress * 0.2,
                max_activation=2.0 + progress * 1.0,
                sparsity=0.3 + progress * 0.4,
                top_neurons=list(range(layer * 10, layer * 10 + 5)),
            )
            profiles.append(profile)

        return profiles


# Convenience functions


def estimate_token_importance(text: str, context: Optional[str] = None) -> list[TokenImportance]:
    """
    Estimate token importance for input text (convenience function).

    Creates a TokenImportanceEstimator and estimates importance scores
    for all tokens in the input text. This is a shorthand for:
        estimator = TokenImportanceEstimator()
        estimator.estimate(text, context)

    Args:
        text: Text to analyze for token importance.
        context: Optional context string (e.g., prompt when analyzing response).
            Tokens appearing in both text and context receive a relevance boost.

    Returns:
        list[TokenImportance]: List of TokenImportance objects, one per token.

    Example: Basic usage
        >>> importances = estimate_token_importance("Hello world")
        >>> len(importances)
        2
        >>> importances[0].token
        'hello'

    Example: With context
        >>> importances = estimate_token_importance(
        ...     "Python is great",
        ...     context="Tell me about Python"
        ... )
        >>> python_imp = next(i for i in importances if i.token == "python")
        >>> python_imp.importance_score > 0.5
        True

    Example: Finding most important tokens
        >>> importances = estimate_token_importance("The neural network learns patterns")
        >>> top = sorted(importances, key=lambda x: x.importance_score, reverse=True)[:2]
        >>> [t.token for t in top]  # Content words rank higher
        ['neural', 'network']

    Example: Checking scores
        >>> importances = estimate_token_importance("Simple test")
        >>> all(0 <= i.importance_score <= 1 for i in importances)
        True
    """
    estimator = TokenImportanceEstimator()
    return estimator.estimate(text, context)


def analyze_attention(prompt: str, response: str) -> list[AttentionHead]:
    """
    Analyze attention patterns for a prompt-response pair (convenience function).

    Creates an AttentionAnalyzer and analyzes attention patterns between
    the prompt and response. This is a shorthand for:
        analyzer = AttentionAnalyzer()
        analyzer.analyze_text(prompt, response)

    Args:
        prompt: The input prompt text.
        response: The model's response text.

    Returns:
        list[AttentionHead]: List of 8 AttentionHead analyses (4 layers x 2 heads).

    Example: Basic usage
        >>> heads = analyze_attention("What is AI?", "AI is artificial intelligence")
        >>> len(heads)
        8
        >>> heads[0].layer
        0

    Example: Checking pattern types
        >>> heads = analyze_attention("Input", "Output")
        >>> all(h.pattern_type in list(AttentionPattern) for h in heads)
        True

    Example: Analyzing entropy distribution
        >>> heads = analyze_attention("Question", "Answer")
        >>> avg_entropy = sum(h.entropy for h in heads) / len(heads)
        >>> 0 <= avg_entropy <= 1
        True

    Example: Finding key positions
        >>> heads = analyze_attention("word test", "word result")
        >>> any(len(h.key_positions) > 0 for h in heads)
        True
    """
    analyzer = AttentionAnalyzer()
    return analyzer.analyze_text(prompt, response)


def introspect_model(
    prompt: str,
    response: str,
    include_layers: bool = True,
) -> IntrospectionReport:
    """
    Perform comprehensive model introspection (convenience function).

    Creates a ModelIntrospector and performs full introspection analysis
    on the prompt-response pair. This is a shorthand for:
        introspector = ModelIntrospector()
        introspector.introspect(prompt, response, include_layers)

    Args:
        prompt: The input prompt text.
        response: The model's response text.
        include_layers: Whether to include layer-wise analysis (default: True).
            Set to False for faster analysis.

    Returns:
        IntrospectionReport: Comprehensive report containing token importances,
            attention analysis, layer analysis, key tokens, and pattern counts.

    Example: Basic introspection
        >>> report = introspect_model("Explain ML", "ML is machine learning...")
        >>> len(report.token_importances) > 0
        True
        >>> len(report.attention_analysis)
        8

    Example: Accessing key tokens
        >>> report = introspect_model("What is Python?", "Python is a language")
        >>> isinstance(report.key_tokens, list)
        True

    Example: Without layer analysis
        >>> report = introspect_model("Quick", "Fast", include_layers=False)
        >>> len(report.layer_analyses)
        0

    Example: Getting summary
        >>> report = introspect_model("Test", "Result")
        >>> d = report.to_dict()
        >>> 'attention_patterns' in d
        True
    """
    introspector = ModelIntrospector()
    return introspector.introspect(prompt, response, include_layers)


def estimate_saliency(prompt: str, response: str) -> SaliencyMap:
    """
    Estimate saliency map for input prompt (convenience function).

    Creates a SaliencyEstimator and estimates saliency scores for the
    prompt tokens. This is a shorthand for:
        estimator = SaliencyEstimator()
        estimator.estimate(prompt, response)

    Args:
        prompt: The input prompt to analyze.
        response: The model's response (used as context).

    Returns:
        SaliencyMap: Saliency map with tokens and their importance scores.

    Example: Basic usage
        >>> smap = estimate_saliency("What is AI?", "AI means...")
        >>> len(smap.tokens) > 0
        True

    Example: Getting highlighted text
        >>> smap = estimate_saliency("Important query", "Response text")
        >>> highlighted = smap.get_highlighted_text(threshold=0.5)
        >>> isinstance(highlighted, str)
        True

    Example: Finding top salient tokens
        >>> smap = estimate_saliency("Neural network question", "Answer about NN")
        >>> top = smap.get_top_salient(3)
        >>> len(top) <= 3
        True

    Example: Converting to dict
        >>> smap = estimate_saliency("Test", "Response")
        >>> d = smap.to_dict()
        >>> 'tokens' in d
        True
    """
    estimator = SaliencyEstimator()
    return estimator.estimate(prompt, response)


def profile_activations(text: str, num_layers: int = 12) -> list[ActivationProfile]:
    """
    Profile model activations for input text (convenience function).

    Creates an ActivationProfiler and profiles activation statistics
    across layers. This is a shorthand for:
        profiler = ActivationProfiler()
        profiler.profile(text, num_layers)

    Args:
        text: Input text to profile.
        num_layers: Number of layers to simulate (default: 12).

    Returns:
        list[ActivationProfile]: List of activation profiles, one per layer.

    Example: Basic usage
        >>> profiles = profile_activations("Sample text")
        >>> len(profiles)
        12
        >>> profiles[0].layer
        0

    Example: Custom layer count
        >>> profiles = profile_activations("Text", num_layers=6)
        >>> len(profiles)
        6

    Example: Checking sparsity trend
        >>> profiles = profile_activations("Test input")
        >>> profiles[0].sparsity < profiles[-1].sparsity
        True

    Example: Accessing activation statistics
        >>> profiles = profile_activations("Data")
        >>> all(0.3 <= p.mean_activation <= 0.5 for p in profiles)
        True
    """
    profiler = ActivationProfiler()
    return profiler.profile(text, num_layers)
