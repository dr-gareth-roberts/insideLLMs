"""
Model introspection and attention analysis utilities.

Provides tools for:
- Attention pattern analysis
- Token importance estimation
- Activation analysis
- Layer-wise representation analysis
- Model behavior introspection
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from insideLLMs.nlp.tokenization import word_tokenize_regex


class AttentionPattern(Enum):
    """Common attention patterns."""

    LOCAL = "local"  # Attention to nearby tokens
    GLOBAL = "global"  # Attention to all tokens
    SPARSE = "sparse"  # Attention to specific tokens
    DIAGONAL = "diagonal"  # Self-attention dominance
    VERTICAL = "vertical"  # Attention to specific positions
    BLOCK = "block"  # Block diagonal patterns


@dataclass
class TokenImportance:
    """Importance score for a token."""

    token: str
    position: int
    importance_score: float
    attribution_score: float = 0.0
    gradient_score: float = 0.0
    attention_received: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Analysis of a single attention head."""

    layer: int
    head: int
    pattern_type: AttentionPattern
    entropy: float  # Higher = more distributed attention
    sparsity: float  # Higher = more focused attention
    key_positions: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Analysis of a model layer."""

    layer_index: int
    representation_norm: float
    token_similarity: float  # How similar token representations are
    information_retention: float  # Information from input retained
    key_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer_index": self.layer_index,
            "representation_norm": self.representation_norm,
            "token_similarity": self.token_similarity,
            "information_retention": self.information_retention,
            "key_features": self.key_features,
        }


@dataclass
class IntrospectionReport:
    """Comprehensive introspection report."""

    prompt: str
    response: str
    token_importances: list[TokenImportance] = field(default_factory=list)
    attention_analysis: list[AttentionHead] = field(default_factory=list)
    layer_analyses: list[LayerAnalysis] = field(default_factory=list)
    key_tokens: list[str] = field(default_factory=list)
    attention_patterns: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_top_tokens(self, n: int = 10) -> list[TokenImportance]:
        """Get top n most important tokens."""
        sorted_tokens = sorted(
            self.token_importances,
            key=lambda t: t.importance_score,
            reverse=True,
        )
        return sorted_tokens[:n]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_tokens": len(self.token_importances),
            "key_tokens": self.key_tokens,
            "attention_patterns": self.attention_patterns,
            "top_tokens": [t.to_dict() for t in self.get_top_tokens(5)],
            "metadata": self.metadata,
        }


class TokenImportanceEstimator:
    """Estimate importance of tokens using heuristics."""

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
        """Estimate token importance.

        Args:
            text: Text to analyze.
            context: Optional context (e.g., prompt for response).

        Returns:
            List of token importance scores.
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
        """Simple tokenization."""
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
        """Calculate importance for a single token."""
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
    """Analyze attention patterns (simulated for API models)."""

    def analyze_text(self, prompt: str, response: str) -> list[AttentionHead]:
        """Analyze attention patterns from text.

        Since we don't have access to actual attention weights,
        this uses heuristics to simulate attention analysis.

        Args:
            prompt: Input prompt.
            response: Model response.

        Returns:
            Simulated attention head analyses.
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
        """Simple tokenization.

        Note: Delegates to insideLLMs.nlp.tokenization.word_tokenize_regex
        """
        return word_tokenize_regex(text)

    def _infer_pattern(
        self,
        prompt_tokens: list[str],
        response_tokens: list[str],
        layer: int,
        head: int,
    ) -> AttentionPattern:
        """Infer attention pattern from text characteristics."""
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
        """Estimate entropy based on pattern type."""
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
        """Estimate sparsity based on pattern type."""
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
        """Find key positions (tokens that appear in both)."""
        prompt_set = set(prompt_tokens)
        key_positions = []

        for i, token in enumerate(response_tokens):
            if token in prompt_set:
                key_positions.append(i)

        return key_positions[:10]  # Limit to 10


class LayerAnalyzer:
    """Analyze layer-wise representations (simulated)."""

    def analyze(
        self,
        text: str,
        num_layers: int = 12,
    ) -> list[LayerAnalysis]:
        """Analyze layer representations.

        Args:
            text: Text to analyze.
            num_layers: Number of layers to simulate.

        Returns:
            List of layer analyses.
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
        """Infer features captured at each layer."""
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
    """Saliency map for input tokens."""

    tokens: list[str]
    scores: list[float]
    method: str = "gradient"

    def get_highlighted_text(self, threshold: float = 0.5) -> str:
        """Get text with high-saliency tokens highlighted."""
        highlighted = []
        for token, score in zip(self.tokens, self.scores):
            if score >= threshold:
                highlighted.append(f"**{token}**")
            else:
                highlighted.append(token)
        return " ".join(highlighted)

    def get_top_salient(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top n most salient tokens."""
        pairs = list(zip(self.tokens, self.scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:n]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tokens": self.tokens,
            "scores": self.scores,
            "method": self.method,
            "top_salient": self.get_top_salient(5),
        }


class SaliencyEstimator:
    """Estimate input saliency for model outputs."""

    def __init__(self):
        """Initialize estimator."""
        self.importance_estimator = TokenImportanceEstimator()

    def estimate(self, prompt: str, response: str, method: str = "importance") -> SaliencyMap:
        """Estimate saliency map.

        Args:
            prompt: Input prompt.
            response: Model response.
            method: Estimation method.

        Returns:
            Saliency map.
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
    """Comprehensive model introspection."""

    def __init__(self):
        """Initialize introspector."""
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
        """Perform comprehensive introspection.

        Args:
            prompt: Input prompt.
            response: Model response.
            include_layers: Whether to include layer analysis.
            num_layers: Number of layers to analyze.

        Returns:
            Introspection report.
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
    """Profile of model activations (simulated)."""

    layer: int
    mean_activation: float
    max_activation: float
    sparsity: float  # Fraction of near-zero activations
    top_neurons: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer": self.layer,
            "mean_activation": self.mean_activation,
            "max_activation": self.max_activation,
            "sparsity": self.sparsity,
            "top_neurons": self.top_neurons,
        }


class ActivationProfiler:
    """Profile model activations (simulated)."""

    def profile(self, text: str, num_layers: int = 12) -> list[ActivationProfile]:
        """Profile activations for text.

        Args:
            text: Text to profile.
            num_layers: Number of layers.

        Returns:
            List of activation profiles.
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
    """Estimate token importance.

    Args:
        text: Text to analyze.
        context: Optional context.

    Returns:
        List of token importances.
    """
    estimator = TokenImportanceEstimator()
    return estimator.estimate(text, context)


def analyze_attention(prompt: str, response: str) -> list[AttentionHead]:
    """Analyze attention patterns.

    Args:
        prompt: Input prompt.
        response: Model response.

    Returns:
        List of attention head analyses.
    """
    analyzer = AttentionAnalyzer()
    return analyzer.analyze_text(prompt, response)


def introspect_model(
    prompt: str,
    response: str,
    include_layers: bool = True,
) -> IntrospectionReport:
    """Perform model introspection.

    Args:
        prompt: Input prompt.
        response: Model response.
        include_layers: Whether to include layer analysis.

    Returns:
        Introspection report.
    """
    introspector = ModelIntrospector()
    return introspector.introspect(prompt, response, include_layers)


def estimate_saliency(prompt: str, response: str) -> SaliencyMap:
    """Estimate saliency map.

    Args:
        prompt: Input prompt.
        response: Model response.

    Returns:
        Saliency map.
    """
    estimator = SaliencyEstimator()
    return estimator.estimate(prompt, response)


def profile_activations(text: str, num_layers: int = 12) -> list[ActivationProfile]:
    """Profile model activations.

    Args:
        text: Text to profile.
        num_layers: Number of layers.

    Returns:
        List of activation profiles.
    """
    profiler = ActivationProfiler()
    return profiler.profile(text, num_layers)
