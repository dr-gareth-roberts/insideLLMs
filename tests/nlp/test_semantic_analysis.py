"""Tests for semantic similarity and embedding analysis utilities."""

import pytest

from insideLLMs.nlp.semantic_analysis import (
    ClusterInfo,
    ClusteringMethod,
    DimensionalityReduction,
    EmbeddingAnalyzer,
    EmbeddingClusterer,
    EmbeddingStats,
    SearchResult,
    SemanticSearch,
    SimilarityCalculator,
    SimilarityMetric,
    SimilarityResult,
    TextSimilarityAnalyzer,
    analyze_embedding,
    cluster_texts,
    compute_similarity,
    create_semantic_index,
    find_duplicate_texts,
    find_similar_texts,
    normalize_embedding,
)


class TestSimilarityMetric:
    """Tests for SimilarityMetric enum."""

    def test_all_metrics_exist(self):
        """Test all expected metrics exist."""
        assert SimilarityMetric.COSINE
        assert SimilarityMetric.EUCLIDEAN
        assert SimilarityMetric.MANHATTAN
        assert SimilarityMetric.DOT_PRODUCT
        assert SimilarityMetric.JACCARD


class TestClusteringMethod:
    """Tests for ClusteringMethod enum."""

    def test_all_methods_exist(self):
        """Test all expected methods exist."""
        assert ClusteringMethod.KMEANS
        assert ClusteringMethod.HIERARCHICAL
        assert ClusteringMethod.DBSCAN


class TestDimensionalityReduction:
    """Tests for DimensionalityReduction enum."""

    def test_all_methods_exist(self):
        """Test all expected methods exist."""
        assert DimensionalityReduction.PCA
        assert DimensionalityReduction.TSNE
        assert DimensionalityReduction.UMAP


class TestSimilarityResult:
    """Tests for SimilarityResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = SimilarityResult(
            text_a="Hello world",
            text_b="Hello there",
            score=0.85,
            metric=SimilarityMetric.COSINE,
        )
        assert result.score == 0.85
        assert result.is_similar

    def test_similarity_levels(self):
        """Test similarity level categorization."""
        assert (
            SimilarityResult("a", "b", 0.95, SimilarityMetric.COSINE).similarity_level
            == "very_high"
        )
        assert SimilarityResult("a", "b", 0.75, SimilarityMetric.COSINE).similarity_level == "high"
        assert (
            SimilarityResult("a", "b", 0.55, SimilarityMetric.COSINE).similarity_level == "moderate"
        )
        assert SimilarityResult("a", "b", 0.35, SimilarityMetric.COSINE).similarity_level == "low"
        assert (
            SimilarityResult("a", "b", 0.15, SimilarityMetric.COSINE).similarity_level == "very_low"
        )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SimilarityResult(
            text_a="Hello",
            text_b="World",
            score=0.8,
            metric=SimilarityMetric.COSINE,
        )
        d = result.to_dict()
        assert d["score"] == 0.8
        assert d["metric"] == "cosine"
        assert d["is_similar"]


class TestEmbeddingStats:
    """Tests for EmbeddingStats dataclass."""

    def test_basic_creation(self):
        """Test basic stats creation."""
        stats = EmbeddingStats(
            dimension=768,
            magnitude=1.0,
            mean=0.01,
            std=0.1,
            min_val=-0.5,
            max_val=0.5,
            sparsity=0.1,
        )
        assert stats.dimension == 768
        assert stats.magnitude == 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = EmbeddingStats(
            dimension=768,
            magnitude=1.0,
            mean=0.01,
            std=0.1,
            min_val=-0.5,
            max_val=0.5,
            sparsity=0.1,
        )
        d = stats.to_dict()
        assert d["dimension"] == 768


class TestClusterInfo:
    """Tests for ClusterInfo dataclass."""

    def test_basic_creation(self):
        """Test basic cluster info creation."""
        info = ClusterInfo(
            cluster_id=0,
            size=5,
            centroid=[0.5, 0.5],
            member_indices=[0, 1, 2, 3, 4],
            member_texts=["text1", "text2", "text3", "text4", "text5"],
            intra_cluster_similarity=0.9,
        )
        assert info.size == 5
        assert len(info.member_indices) == 5


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_basic_creation(self):
        """Test basic search result creation."""
        result = SearchResult(
            query="test query",
            matches=[(0, "matching text", 0.9), (1, "another match", 0.8)],
            metric=SimilarityMetric.COSINE,
            total_searched=100,
        )
        assert result.top_match[2] == 0.9

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SearchResult(
            query="test",
            matches=[(0, "match", 0.9)],
            metric=SimilarityMetric.COSINE,
            total_searched=10,
        )
        d = result.to_dict()
        assert d["total_searched"] == 10


class TestSimilarityCalculator:
    """Tests for SimilarityCalculator."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        calc = SimilarityCalculator()
        a = [1.0, 2.0, 3.0]
        assert calc.cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        calc = SimilarityCalculator()
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert calc.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        calc = SimilarityCalculator()
        a = [1.0, 1.0]
        b = [-1.0, -1.0]
        assert calc.cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_euclidean_distance_identical(self):
        """Test Euclidean distance of identical vectors."""
        calc = SimilarityCalculator()
        a = [1.0, 2.0, 3.0]
        assert calc.euclidean_distance(a, a) == 0.0

    def test_euclidean_distance_simple(self):
        """Test simple Euclidean distance."""
        calc = SimilarityCalculator()
        a = [0.0, 0.0]
        b = [3.0, 4.0]
        assert calc.euclidean_distance(a, b) == 5.0

    def test_manhattan_distance(self):
        """Test Manhattan distance."""
        calc = SimilarityCalculator()
        a = [0.0, 0.0]
        b = [3.0, 4.0]
        assert calc.manhattan_distance(a, b) == 7.0

    def test_dot_product(self):
        """Test dot product."""
        calc = SimilarityCalculator()
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        assert calc.dot_product(a, b) == 32.0

    def test_jaccard_similarity(self):
        """Test Jaccard similarity."""
        calc = SimilarityCalculator()
        a = [1.0, 1.0, 0.0, 0.0]
        b = [1.0, 0.0, 1.0, 0.0]
        # Intersection: 1, Union: 3
        assert calc.jaccard_similarity(a, b) == pytest.approx(1.0 / 3.0)

    def test_compute_with_metric(self):
        """Test compute with different metrics."""
        calc = SimilarityCalculator()
        a = [1.0, 0.0]
        b = [1.0, 0.0]

        assert calc.compute(a, b, SimilarityMetric.COSINE) == 1.0
        assert calc.compute(a, b, SimilarityMetric.EUCLIDEAN) == 1.0
        assert calc.compute(a, b, SimilarityMetric.DOT_PRODUCT) == 1.0

    def test_dimension_mismatch_raises(self):
        """Test dimension mismatch raises error."""
        calc = SimilarityCalculator()
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="dimensions must match"):
            calc.cosine_similarity(a, b)

    def test_pairwise_similarity(self):
        """Test pairwise similarity matrix."""
        calc = SimilarityCalculator()
        embeddings = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        matrix = calc.pairwise_similarity(embeddings)

        assert len(matrix) == 3
        assert matrix[0][0] == 1.0  # Diagonal
        assert matrix[0][2] == 1.0  # Same vectors


class TestEmbeddingAnalyzer:
    """Tests for EmbeddingAnalyzer."""

    def test_compute_stats(self):
        """Test computing embedding stats."""
        analyzer = EmbeddingAnalyzer()
        embedding = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = analyzer.compute_stats(embedding)

        assert stats.dimension == 5
        assert stats.mean == 3.0
        assert stats.min_val == 1.0
        assert stats.max_val == 5.0

    def test_compute_stats_empty(self):
        """Test stats for empty embedding."""
        analyzer = EmbeddingAnalyzer()
        stats = analyzer.compute_stats([])
        assert stats.dimension == 0

    def test_normalize(self):
        """Test embedding normalization."""
        analyzer = EmbeddingAnalyzer()
        embedding = [3.0, 4.0]
        normalized = analyzer.normalize(embedding)

        # Should have unit length
        magnitude = sum(x * x for x in normalized) ** 0.5
        assert magnitude == pytest.approx(1.0)

    def test_normalize_zero_vector(self):
        """Test normalizing zero vector."""
        analyzer = EmbeddingAnalyzer()
        embedding = [0.0, 0.0, 0.0]
        normalized = analyzer.normalize(embedding)
        assert normalized == [0.0, 0.0, 0.0]

    def test_average_embeddings(self):
        """Test averaging embeddings."""
        analyzer = EmbeddingAnalyzer()
        embeddings = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        avg = analyzer.average_embeddings(embeddings)

        assert avg == [3.0, 4.0]

    def test_weighted_average(self):
        """Test weighted average of embeddings."""
        analyzer = EmbeddingAnalyzer()
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        weights = [3.0, 1.0]
        avg = analyzer.weighted_average(embeddings, weights)

        assert avg == [0.75, 0.25]

    def test_find_outliers(self):
        """Test finding outliers."""
        analyzer = EmbeddingAnalyzer()
        # Most points clustered, one far away
        embeddings = [
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.1],
            [0.1, -0.1],
            [100.0, 100.0],  # Clear outlier
        ]
        outliers = analyzer.find_outliers(embeddings, threshold=1.5)
        assert 4 in outliers


class TestTextSimilarityAnalyzer:
    """Tests for TextSimilarityAnalyzer."""

    def test_compare_identical(self):
        """Test comparing identical texts."""
        analyzer = TextSimilarityAnalyzer()
        result = analyzer.compare("hello world", "hello world")
        assert result.score == pytest.approx(1.0)

    def test_compare_different(self):
        """Test comparing different texts."""
        analyzer = TextSimilarityAnalyzer()
        result = analyzer.compare("hello world", "goodbye moon")
        assert result.score < 1.0

    def test_compare_with_custom_embedder(self):
        """Test with custom embedding function."""

        def simple_embed(text: str) -> list:
            return [len(text), text.count(" ")]

        analyzer = TextSimilarityAnalyzer(embed_fn=simple_embed)
        result = analyzer.compare("hello world", "hello world")
        assert result.score == pytest.approx(1.0)

    def test_compare_batch(self):
        """Test batch comparison."""
        analyzer = TextSimilarityAnalyzer()
        results = analyzer.compare_batch(
            ["hello", "world"],
            ["hello", "world"],
        )
        assert len(results) == 2
        assert all(r.score == 1.0 for r in results)

    def test_find_most_similar(self):
        """Test finding most similar texts."""

        # Use custom embedder that clearly differentiates
        def embed(text: str) -> list:
            # Create unique signature per text
            return [ord(c) / 256.0 for c in text[:20].ljust(20)]

        analyzer = TextSimilarityAnalyzer(embed_fn=embed)
        result = analyzer.find_most_similar(
            "hello world",
            ["goodbye moon", "hello world", "hello there"],
            top_k=2,
        )
        assert result.top_match[1] == "hello world"

    def test_find_duplicates(self):
        """Test finding duplicates."""
        analyzer = TextSimilarityAnalyzer()
        texts = ["hello world", "hello world", "goodbye"]
        duplicates = analyzer.find_duplicates(texts, threshold=0.9)
        assert (0, 1, 1.0) in duplicates or any(d[0] == 0 and d[1] == 1 for d in duplicates)

    def test_cache(self):
        """Test embedding caching."""
        calls = [0]

        def counting_embed(text: str) -> list:
            calls[0] += 1
            return [len(text)]

        analyzer = TextSimilarityAnalyzer(embed_fn=counting_embed)
        analyzer.compare("hello", "hello")

        # Both should be cached now
        assert calls[0] == 1  # Same text should only be embedded once

        analyzer.compare("hello", "world")
        assert calls[0] == 2  # New text

    def test_clear_cache(self):
        """Test cache clearing."""
        analyzer = TextSimilarityAnalyzer()
        analyzer.compare("hello", "world")
        assert len(analyzer._cache) > 0

        analyzer.clear_cache()
        assert len(analyzer._cache) == 0


class TestEmbeddingClusterer:
    """Tests for EmbeddingClusterer."""

    def test_cluster_kmeans(self):
        """Test k-means clustering."""

        # Use custom embedder that differentiates texts clearly
        def embed(text: str) -> list:
            return [ord(c) / 256.0 for c in text[:20].ljust(20)]

        clusterer = EmbeddingClusterer(embed_fn=embed)
        texts = [
            "aaa bbb ccc",
            "aaa bbb ddd",
            "xxx yyy zzz",
            "xxx yyy www",
        ]
        result = clusterer.cluster(texts, n_clusters=2)

        assert result.n_clusters == 2
        assert len(result.labels) == 4

    def test_cluster_hierarchical(self):
        """Test hierarchical clustering."""
        clusterer = EmbeddingClusterer()
        texts = ["a", "a", "b", "b"]
        result = clusterer.cluster(texts, n_clusters=2, method=ClusteringMethod.HIERARCHICAL)

        assert result.n_clusters == 2
        assert len(result.labels) == 4

    def test_find_optimal_k(self):
        """Test finding optimal k."""
        clusterer = EmbeddingClusterer()
        texts = ["a", "b", "c", "d", "e"]
        result = clusterer.find_optimal_k(texts, max_k=3)

        assert "optimal_k" in result
        assert "inertias" in result
        assert len(result["inertias"]) == 3

    def test_cluster_with_custom_embedder(self):
        """Test clustering with custom embedder."""

        def simple_embed(text: str) -> list:
            return [len(text), text.count("a")]

        clusterer = EmbeddingClusterer(embed_fn=simple_embed)
        texts = ["aaa", "bbb", "aaaa", "bbbb"]
        result = clusterer.cluster(texts, n_clusters=2)

        assert result.n_clusters == 2

    def test_cluster_dbscan(self):
        """Test DBSCAN clustering path."""

        # Embedder that groups similar texts (identical texts -> same embedding)
        def embed(text: str) -> list:
            return [ord(c) / 256.0 for c in (text[:10] + " " * 10)[:10]]

        clusterer = EmbeddingClusterer(embed_fn=embed)
        texts = ["aaa", "aaa", "bbb", "bbb", "ccc"]
        result = clusterer.cluster(
            texts,
            method=ClusteringMethod.DBSCAN,
            eps=0.5,
            min_samples=2,
        )

        assert result.method == ClusteringMethod.DBSCAN
        assert len(result.labels) == 5
        assert result.n_clusters >= 1
        assert len(result.clusters) == result.n_clusters

    def test_cluster_dbscan_tiny_dataset(self):
        """DBSCAN with tiny dataset uses deterministic fallback."""
        clusterer = EmbeddingClusterer()
        texts = ["a", "b"]
        result = clusterer.cluster(
            texts,
            method=ClusteringMethod.DBSCAN,
            eps=1.0,
            min_samples=2,
        )

        assert result.method == ClusteringMethod.DBSCAN
        assert len(result.labels) == 2
        assert result.n_clusters >= 0


class TestSemanticSearch:
    """Tests for SemanticSearch."""

    def test_add_and_search(self):
        """Test adding texts and searching."""
        search = SemanticSearch()
        search.add("hello world")
        search.add("goodbye world")
        search.add("hello there")

        result = search.search("hello", top_k=2)
        assert len(result.matches) >= 1

    def test_add_batch(self):
        """Test batch adding."""
        search = SemanticSearch()
        indices = search.add_batch(["text1", "text2", "text3"])
        assert indices == [0, 1, 2]
        assert search.size == 3

    def test_search_with_threshold(self):
        """Test search with threshold."""
        search = SemanticSearch()
        search.add("hello world")
        search.add("completely different text")

        result = search.search("hello world", threshold=0.9)
        # Should only return close matches
        assert all(m[2] >= 0.9 for m in result.matches)

    def test_clear(self):
        """Test clearing index."""
        search = SemanticSearch()
        search.add("text1")
        search.add("text2")
        assert search.size == 2

        search.clear()
        assert search.size == 0

    def test_custom_metric(self):
        """Test with custom metric."""
        search = SemanticSearch(metric=SimilarityMetric.EUCLIDEAN)
        search.add("hello")
        search.add("world")

        result = search.search("hello")
        assert result.metric == SimilarityMetric.EUCLIDEAN


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_similarity(self):
        """Test compute_similarity function."""
        result = compute_similarity("hello world", "hello world")
        assert result.score == pytest.approx(1.0)

    def test_find_similar_texts(self):
        """Test find_similar_texts function."""

        # Use custom embedder for clear differentiation
        def embed(text: str) -> list:
            return [ord(c) / 256.0 for c in text[:20].ljust(20)]

        result = find_similar_texts(
            "hello world",
            ["hello there", "goodbye world", "hello world"],
            embed_fn=embed,
        )
        assert result.top_match[1] == "hello world"

    def test_cluster_texts(self):
        """Test cluster_texts function."""

        # Use custom embedder for clear differentiation
        def embed(text: str) -> list:
            return [ord(c) / 256.0 for c in text[:20].ljust(20)]

        result = cluster_texts(
            ["aaa bbb", "aaa ccc", "xxx yyy", "xxx zzz"],
            n_clusters=2,
            embed_fn=embed,
        )
        assert result.n_clusters == 2

    def test_find_duplicate_texts(self):
        """Test find_duplicate_texts function."""
        duplicates = find_duplicate_texts(["hello", "hello", "world"], threshold=0.9)
        assert len(duplicates) >= 1

    def test_analyze_embedding(self):
        """Test analyze_embedding function."""
        stats = analyze_embedding([1.0, 2.0, 3.0])
        assert stats.dimension == 3

    def test_normalize_embedding(self):
        """Test normalize_embedding function."""
        normalized = normalize_embedding([3.0, 4.0])
        magnitude = sum(x * x for x in normalized) ** 0.5
        assert magnitude == pytest.approx(1.0)

    def test_create_semantic_index(self):
        """Test create_semantic_index function."""
        search = create_semantic_index(["text1", "text2", "text3"])
        assert search.size == 3


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self):
        """Test handling of empty text."""
        analyzer = TextSimilarityAnalyzer()
        analyzer.compare("", "")
        # Should not crash

    def test_single_word(self):
        """Test single word texts."""
        analyzer = TextSimilarityAnalyzer()
        result = analyzer.compare("hello", "hello")
        assert result.score == 1.0

    def test_very_long_text(self):
        """Test very long texts."""
        analyzer = TextSimilarityAnalyzer()
        long_text = "word " * 1000
        result = analyzer.compare(long_text, long_text)
        assert result.score == pytest.approx(1.0)

    def test_special_characters(self):
        """Test texts with special characters."""
        analyzer = TextSimilarityAnalyzer()
        result = analyzer.compare("hello! @#$", "hello! @#$")
        assert result.score == pytest.approx(1.0)

    def test_unicode_text(self):
        """Test Unicode texts."""
        analyzer = TextSimilarityAnalyzer()
        result = analyzer.compare("你好世界", "你好世界")
        assert result.score == 1.0

    def test_cluster_single_text(self):
        """Test clustering single text."""
        clusterer = EmbeddingClusterer()
        result = clusterer.cluster(["single"], n_clusters=1)
        assert result.n_clusters == 1

    def test_cluster_empty_list(self):
        """Test clustering empty list."""
        clusterer = EmbeddingClusterer()
        result = clusterer.cluster([], n_clusters=1)
        assert result.n_clusters == 0

    def test_search_empty_index(self):
        """Test searching empty index."""
        search = SemanticSearch()
        result = search.search("query")
        assert len(result.matches) == 0

    def test_zero_vector_similarity(self):
        """Test similarity with zero vectors."""
        calc = SimilarityCalculator()
        zero = [0.0, 0.0, 0.0]
        normal = [1.0, 2.0, 3.0]
        sim = calc.cosine_similarity(zero, normal)
        assert sim == 0.0
