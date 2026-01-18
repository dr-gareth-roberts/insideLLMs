"""Tests for model introspection and attention analysis utilities."""

from insideLLMs.introspection import (
    ActivationProfile,
    ActivationProfiler,
    AttentionAnalyzer,
    AttentionHead,
    AttentionPattern,
    IntrospectionReport,
    LayerAnalysis,
    LayerAnalyzer,
    ModelIntrospector,
    SaliencyEstimator,
    SaliencyMap,
    TokenImportance,
    TokenImportanceEstimator,
    analyze_attention,
    estimate_saliency,
    estimate_token_importance,
    introspect_model,
    profile_activations,
)


class TestAttentionPattern:
    """Tests for AttentionPattern enum."""

    def test_all_patterns_exist(self):
        """Test that all patterns are defined."""
        assert AttentionPattern.LOCAL.value == "local"
        assert AttentionPattern.GLOBAL.value == "global"
        assert AttentionPattern.SPARSE.value == "sparse"
        assert AttentionPattern.DIAGONAL.value == "diagonal"
        assert AttentionPattern.VERTICAL.value == "vertical"
        assert AttentionPattern.BLOCK.value == "block"


class TestTokenImportance:
    """Tests for TokenImportance dataclass."""

    def test_basic_creation(self):
        """Test basic creation."""
        imp = TokenImportance(
            token="test",
            position=0,
            importance_score=0.8,
        )

        assert imp.token == "test"
        assert imp.importance_score == 0.8

    def test_to_dict(self):
        """Test dictionary conversion."""
        imp = TokenImportance(
            token="word",
            position=5,
            importance_score=0.7,
            attribution_score=0.6,
            attention_received=0.8,
        )

        d = imp.to_dict()
        assert d["token"] == "word"
        assert d["position"] == 5
        assert d["importance_score"] == 0.7


class TestTokenImportanceEstimator:
    """Tests for TokenImportanceEstimator."""

    def test_estimate_importance(self):
        """Test basic importance estimation."""
        estimator = TokenImportanceEstimator()
        text = "The quick brown fox jumps over the lazy dog."

        importances = estimator.estimate(text)

        assert len(importances) > 0
        # Stop words should have lower importance
        the_importance = [i for i in importances if i.token == "the"]
        content_importance = [i for i in importances if i.token == "fox"]

        if the_importance and content_importance:
            assert the_importance[0].importance_score < content_importance[0].importance_score

    def test_context_boosts_relevance(self):
        """Test that context boosts token relevance."""
        estimator = TokenImportanceEstimator()
        text = "Paris is the capital."
        context = "What is the capital of France? Paris is important."

        with_context = estimator.estimate(text, context)
        without_context = estimator.estimate(text)

        # Find Paris importance in both
        paris_with = [i for i in with_context if i.token == "paris"]
        paris_without = [i for i in without_context if i.token == "paris"]

        if paris_with and paris_without:
            assert paris_with[0].importance_score >= paris_without[0].importance_score

    def test_named_entities_boosted(self):
        """Test that named entities get boosted."""
        estimator = TokenImportanceEstimator()
        text = "John went to Paris for vacation."

        importances = estimator.estimate(text)

        # Capitalized words should have higher scores
        john_imp = [i for i in importances if i.token.lower() == "john"]
        went_imp = [i for i in importances if i.token == "went"]

        if john_imp and went_imp:
            # John should be important (proper noun)
            assert john_imp[0].importance_score > 0.3

    def test_numbers_boosted(self):
        """Test that numbers get boosted."""
        estimator = TokenImportanceEstimator()
        text = "The price is 42 dollars."

        importances = estimator.estimate(text)

        num_imp = [i for i in importances if i.token == "42"]
        if num_imp:
            assert num_imp[0].importance_score > 0.5


class TestAttentionHead:
    """Tests for AttentionHead dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        head = AttentionHead(
            layer=2,
            head=1,
            pattern_type=AttentionPattern.LOCAL,
            entropy=0.5,
            sparsity=0.7,
            key_positions=[0, 5, 10],
        )

        d = head.to_dict()
        assert d["layer"] == 2
        assert d["pattern_type"] == "local"


class TestAttentionAnalyzer:
    """Tests for AttentionAnalyzer."""

    def test_analyze_text(self):
        """Test attention analysis."""
        analyzer = AttentionAnalyzer()
        prompt = "What is the capital of France?"
        response = "The capital of France is Paris."

        analyses = analyzer.analyze_text(prompt, response)

        assert len(analyses) > 0
        for analysis in analyses:
            assert isinstance(analysis, AttentionHead)
            assert 0 <= analysis.entropy <= 1
            assert 0 <= analysis.sparsity <= 1

    def test_different_patterns_detected(self):
        """Test that different patterns are detected."""
        analyzer = AttentionAnalyzer()

        # High overlap should suggest copying behavior
        prompt = "Paris London Berlin Tokyo"
        response = "Paris is in France. London is in UK."

        analyses = analyzer.analyze_text(prompt, response)
        [a.pattern_type for a in analyses]

        # Should have some pattern variety
        assert len(analyses) > 0


class TestLayerAnalysis:
    """Tests for LayerAnalysis dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        analysis = LayerAnalysis(
            layer_index=5,
            representation_norm=0.8,
            token_similarity=0.6,
            information_retention=0.9,
            key_features=["semantics", "context"],
        )

        d = analysis.to_dict()
        assert d["layer_index"] == 5
        assert "key_features" in d


class TestLayerAnalyzer:
    """Tests for LayerAnalyzer."""

    def test_analyze_layers(self):
        """Test layer analysis."""
        analyzer = LayerAnalyzer()
        text = "This is a test sentence for layer analysis."

        analyses = analyzer.analyze(text, num_layers=6)

        assert len(analyses) == 6
        for analysis in analyses:
            assert isinstance(analysis, LayerAnalysis)
            assert 0 <= analysis.token_similarity <= 1

    def test_layer_progression(self):
        """Test that layer characteristics progress."""
        analyzer = LayerAnalyzer()
        text = "Test text for analysis."

        analyses = analyzer.analyze(text, num_layers=10)

        # Token similarity should generally decrease
        early = analyses[1].token_similarity
        late = analyses[8].token_similarity
        assert early >= late


class TestSaliencyMap:
    """Tests for SaliencyMap dataclass."""

    def test_get_highlighted_text(self):
        """Test text highlighting."""
        smap = SaliencyMap(
            tokens=["the", "important", "word"],
            scores=[0.2, 0.8, 0.3],
        )

        highlighted = smap.get_highlighted_text(threshold=0.5)
        assert "**important**" in highlighted

    def test_get_top_salient(self):
        """Test getting top salient tokens."""
        smap = SaliencyMap(
            tokens=["a", "b", "c", "d"],
            scores=[0.3, 0.8, 0.5, 0.9],
        )

        top = smap.get_top_salient(2)
        assert len(top) == 2
        assert top[0][0] == "d"  # Highest score
        assert top[1][0] == "b"

    def test_to_dict(self):
        """Test dictionary conversion."""
        smap = SaliencyMap(
            tokens=["test"],
            scores=[0.5],
            method="importance",
        )

        d = smap.to_dict()
        assert "tokens" in d
        assert d["method"] == "importance"


class TestSaliencyEstimator:
    """Tests for SaliencyEstimator."""

    def test_estimate_saliency(self):
        """Test saliency estimation."""
        estimator = SaliencyEstimator()
        prompt = "What is machine learning?"
        response = "Machine learning is a type of AI."

        smap = estimator.estimate(prompt, response)

        assert isinstance(smap, SaliencyMap)
        assert len(smap.tokens) > 0
        assert len(smap.tokens) == len(smap.scores)


class TestIntrospectionReport:
    """Tests for IntrospectionReport."""

    def test_get_top_tokens(self):
        """Test getting top tokens."""
        report = IntrospectionReport(
            prompt="test",
            response="result",
            token_importances=[
                TokenImportance("a", 0, 0.3),
                TokenImportance("b", 1, 0.9),
                TokenImportance("c", 2, 0.5),
            ],
        )

        top = report.get_top_tokens(2)
        assert len(top) == 2
        assert top[0].token == "b"

    def test_to_dict(self):
        """Test dictionary conversion."""
        report = IntrospectionReport(
            prompt="test prompt",
            response="test response",
            key_tokens=["key"],
            attention_patterns={"local": 3},
        )

        d = report.to_dict()
        assert "key_tokens" in d
        assert "attention_patterns" in d


class TestModelIntrospector:
    """Tests for ModelIntrospector."""

    def test_introspect(self):
        """Test full introspection."""
        introspector = ModelIntrospector()
        prompt = "Explain photosynthesis."
        response = "Photosynthesis is the process by which plants convert light into energy."

        report = introspector.introspect(prompt, response)

        assert isinstance(report, IntrospectionReport)
        assert len(report.token_importances) > 0
        assert len(report.attention_analysis) > 0
        assert "num_prompt_tokens" in report.metadata

    def test_introspect_without_layers(self):
        """Test introspection without layer analysis."""
        introspector = ModelIntrospector()

        report = introspector.introspect("prompt", "response", include_layers=False)

        assert len(report.layer_analyses) == 0

    def test_key_tokens_extracted(self):
        """Test that key tokens are extracted."""
        introspector = ModelIntrospector()
        prompt = "What is Python?"
        response = "Python is a programming language."

        report = introspector.introspect(prompt, response)

        # Should identify some key tokens
        assert isinstance(report.key_tokens, list)


class TestActivationProfile:
    """Tests for ActivationProfile dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        profile = ActivationProfile(
            layer=3,
            mean_activation=0.4,
            max_activation=2.5,
            sparsity=0.6,
            top_neurons=[10, 20, 30],
        )

        d = profile.to_dict()
        assert d["layer"] == 3
        assert d["sparsity"] == 0.6


class TestActivationProfiler:
    """Tests for ActivationProfiler."""

    def test_profile_activations(self):
        """Test activation profiling."""
        profiler = ActivationProfiler()
        text = "Test sentence for profiling."

        profiles = profiler.profile(text, num_layers=8)

        assert len(profiles) == 8
        for profile in profiles:
            assert isinstance(profile, ActivationProfile)
            assert 0 <= profile.sparsity <= 1

    def test_sparsity_increases(self):
        """Test that sparsity tends to increase in later layers."""
        profiler = ActivationProfiler()
        profiles = profiler.profile("test", num_layers=10)

        early_sparsity = profiles[1].sparsity
        late_sparsity = profiles[8].sparsity

        assert late_sparsity > early_sparsity


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_estimate_token_importance(self):
        """Test estimate_token_importance function."""
        result = estimate_token_importance("Test text here.")

        assert isinstance(result, list)
        assert all(isinstance(t, TokenImportance) for t in result)

    def test_analyze_attention(self):
        """Test analyze_attention function."""
        result = analyze_attention("prompt", "response")

        assert isinstance(result, list)
        assert all(isinstance(a, AttentionHead) for a in result)

    def test_introspect_model(self):
        """Test introspect_model function."""
        result = introspect_model("prompt", "response")

        assert isinstance(result, IntrospectionReport)

    def test_estimate_saliency(self):
        """Test estimate_saliency function."""
        result = estimate_saliency("prompt", "response")

        assert isinstance(result, SaliencyMap)

    def test_profile_activations(self):
        """Test profile_activations function."""
        result = profile_activations("test text")

        assert isinstance(result, list)
        assert all(isinstance(p, ActivationProfile) for p in result)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self):
        """Test with empty text."""
        estimator = TokenImportanceEstimator()
        result = estimator.estimate("")

        assert result == []

    def test_single_token(self):
        """Test with single token."""
        estimator = TokenImportanceEstimator()
        result = estimator.estimate("word")

        assert len(result) == 1
        assert result[0].token == "word"

    def test_special_characters(self):
        """Test with special characters."""
        estimator = TokenImportanceEstimator()
        result = estimator.estimate("Hello! How are you?")

        # Should handle punctuation
        assert len(result) > 0

    def test_unicode_text(self):
        """Test with unicode text."""
        estimator = TokenImportanceEstimator()
        result = estimator.estimate("日本語テスト text 🎉")

        assert len(result) > 0

    def test_very_long_text(self):
        """Test with very long text."""
        estimator = TokenImportanceEstimator()
        long_text = "word " * 500

        result = estimator.estimate(long_text)

        assert len(result) > 0
        # Frequent words should get penalized
        assert result[0].importance_score < 0.5
