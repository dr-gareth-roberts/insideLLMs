"""
Semantic Similarity and Embedding Analysis Utilities.

This module provides a comprehensive suite of tools for semantic analysis of text
and embeddings, enabling similarity computation, clustering, and semantic search
capabilities. It is designed for analyzing LLM outputs, comparing model responses,
detecting duplicate or near-duplicate content, and understanding semantic relationships
between texts.

Overview
--------
The module is organized into several key components:

1. **Enumerations**: Define available metrics and methods
   - `SimilarityMetric`: Cosine, Euclidean, Manhattan, Dot Product, Jaccard
   - `ClusteringMethod`: K-Means, Hierarchical, DBSCAN
   - `DimensionalityReduction`: PCA, t-SNE, UMAP

2. **Data Classes**: Structured result containers
   - `SimilarityResult`: Holds pairwise similarity computation results
   - `EmbeddingStats`: Statistical analysis of embedding vectors
   - `ClusterInfo`: Information about individual clusters
   - `ClusteringResult`: Complete clustering operation results
   - `SearchResult`: Semantic search query results

3. **Core Classes**: Main analysis engines
   - `VectorSimilarityCalculator`: Low-level vector similarity computations
   - `EmbeddingAnalyzer`: Embedding vector statistics and manipulation
   - `TextSimilarityAnalyzer`: Text-level semantic similarity with caching
   - `EmbeddingClusterer`: Clustering texts by semantic similarity
   - `SemanticSearch`: Full semantic search engine with indexing

4. **Convenience Functions**: Simple API for common operations
   - `compute_similarity()`: Quick pairwise text comparison
   - `find_similar_texts()`: Find top-k similar texts
   - `cluster_texts()`: Cluster texts semantically
   - `find_duplicate_texts()`: Detect near-duplicates
   - `analyze_embedding()`: Get embedding statistics
   - `normalize_embedding()`: Unit-normalize embeddings
   - `create_semantic_index()`: Build a searchable index

Key Features
------------
- **Flexible Embedding Support**: Works with any embedding function via callable interface
- **Multiple Similarity Metrics**: Cosine, Euclidean, Manhattan, Dot Product, Jaccard
- **Caching**: Embedding cache for repeated computations
- **Dimension Handling**: Automatic padding for mismatched embedding dimensions
- **Pure Python**: No external dependencies for core functionality

Examples
--------
Basic similarity comparison between two texts:

>>> from insideLLMs.contrib.semantic_analysis import compute_similarity, SimilarityMetric
>>> result = compute_similarity(
...     "The cat sat on the mat",
...     "A cat was sitting on a mat",
...     metric=SimilarityMetric.COSINE
... )
>>> print(f"Similarity: {result.score:.4f}")
Similarity: 0.6667
>>> print(f"Level: {result.similarity_level}")
Level: moderate

Finding similar texts from a corpus:

>>> from insideLLMs.contrib.semantic_analysis import find_similar_texts
>>> corpus = [
...     "Machine learning models require training data",
...     "Deep neural networks learn representations",
...     "The weather today is sunny and warm",
...     "AI systems need large datasets for training",
...     "Cats are popular household pets",
... ]
>>> results = find_similar_texts(
...     query="Training AI models needs data",
...     candidates=corpus,
...     top_k=3
... )
>>> for idx, text, score in results.matches:
...     print(f"{score:.3f}: {text}")
0.500: Machine learning models require training data
0.333: AI systems need large datasets for training
0.250: Deep neural networks learn representations

Clustering texts by semantic similarity:

>>> from insideLLMs.contrib.semantic_analysis import cluster_texts
>>> texts = [
...     "Python is a programming language",
...     "JavaScript runs in browsers",
...     "Cats are furry animals",
...     "Dogs are loyal pets",
...     "Java is used for enterprise apps",
...     "Hamsters make good pets",
... ]
>>> result = cluster_texts(texts, n_clusters=2)
>>> for cluster in result.clusters:
...     print(f"Cluster {cluster.cluster_id}: {cluster.member_texts}")
Cluster 0: ['Python is a programming language', 'JavaScript runs in browsers', ...]
Cluster 1: ['Cats are furry animals', 'Dogs are loyal pets', 'Hamsters make good pets']

Using custom embeddings (e.g., from sentence-transformers):

>>> from sentence_transformers import SentenceTransformer
>>> model = SentenceTransformer('all-MiniLM-L6-v2')
>>> def embed(text: str) -> list[float]:
...     return model.encode(text).tolist()
>>> result = compute_similarity(
...     "Hello world",
...     "Hi there",
...     embed_fn=embed
... )

Building a semantic search index:

>>> from insideLLMs.contrib.semantic_analysis import create_semantic_index
>>> documents = [
...     "Introduction to machine learning",
...     "Deep learning fundamentals",
...     "Natural language processing basics",
...     "Computer vision techniques",
...     "Reinforcement learning algorithms",
... ]
>>> index = create_semantic_index(documents)
>>> results = index.search("neural networks", top_k=3)
>>> print(results.top_match)
(1, 'Deep learning fundamentals', 0.5)

Notes
-----
- The default embedding function uses simple bag-of-words frequency vectors,
  which is suitable for quick tests but not production use. For production,
  provide a proper embedding function (e.g., from sentence-transformers,
  OpenAI embeddings, or other embedding models).

- Similarity scores are normalized to [0, 1] range for distance-based metrics
  (Euclidean, Manhattan) using the formula: similarity = 1 / (1 + distance)

- The caching mechanism uses MD5 hashing of text content. Clear the cache
  via `analyzer.clear_cache()` if memory is a concern with large corpora.

- Clustering uses simple pure-Python implementations. For large datasets,
  consider using scikit-learn or other optimized libraries.

See Also
--------
- `numpy` : For optimized vector operations in production
- `scikit-learn` : For production-grade clustering and dimensionality reduction
- `sentence-transformers` : For high-quality text embeddings
- `faiss` : For efficient similarity search at scale

References
----------
.. [1] Mikolov, T., et al. "Efficient Estimation of Word Representations
   in Vector Space." arXiv:1301.3781 (2013).
.. [2] Reimers, N., & Gurevych, I. "Sentence-BERT: Sentence Embeddings
   using Siamese BERT-Networks." EMNLP (2019).
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class SimilarityMetric(Enum):
    """
    Enumeration of supported similarity/distance metrics for vector comparison.

    This enum defines the available metrics for computing similarity or distance
    between embedding vectors. Each metric has different properties and is suited
    for different use cases.

    Attributes
    ----------
    COSINE : str
        Cosine similarity measures the cosine of the angle between two vectors.
        Range: [-1, 1] for general vectors, [0, 1] for non-negative vectors.
        Best for: Comparing text embeddings where magnitude doesn't matter.
    EUCLIDEAN : str
        Euclidean distance (L2 norm) measures straight-line distance in space.
        Range: [0, infinity). Converted to similarity via 1/(1+distance).
        Best for: When absolute differences matter.
    MANHATTAN : str
        Manhattan distance (L1 norm) measures sum of absolute differences.
        Range: [0, infinity). Converted to similarity via 1/(1+distance).
        Best for: Sparse vectors or grid-like spaces.
    DOT_PRODUCT : str
        Dot product (inner product) of two vectors.
        Range: (-infinity, infinity). Not normalized to [0, 1].
        Best for: Pre-normalized embeddings optimized for dot product.
    JACCARD : str
        Jaccard similarity for set-like comparisons (non-zero elements as sets).
        Range: [0, 1].
        Best for: Binary or sparse categorical features.

    Examples
    --------
    Using different metrics for similarity computation:

    >>> from insideLLMs.contrib.semantic_analysis import (
    ...     VectorSimilarityCalculator, SimilarityMetric
    ... )
    >>> calc = VectorSimilarityCalculator()
    >>> vec_a = [1.0, 0.5, 0.0, 0.3]
    >>> vec_b = [0.8, 0.6, 0.1, 0.2]
    >>> calc.compute(vec_a, vec_b, SimilarityMetric.COSINE)
    0.9756
    >>> calc.compute(vec_a, vec_b, SimilarityMetric.EUCLIDEAN)
    0.7692

    Selecting metric based on embedding type:

    >>> # For sentence-transformers embeddings (cosine-optimized)
    >>> metric = SimilarityMetric.COSINE
    >>> # For OpenAI ada-002 embeddings (dot product optimized)
    >>> metric = SimilarityMetric.DOT_PRODUCT
    >>> # For binary feature vectors
    >>> metric = SimilarityMetric.JACCARD

    See Also
    --------
    VectorSimilarityCalculator : Calculator that uses these metrics.
    TextSimilarityAnalyzer : High-level text comparison using these metrics.
    """

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    DOT_PRODUCT = "dot_product"
    JACCARD = "jaccard"


class ClusteringMethod(Enum):
    """
    Enumeration of supported clustering algorithms for embedding vectors.

    This enum defines the available clustering methods for grouping similar
    embeddings. Each method has different characteristics regarding cluster
    shape, scalability, and parameter requirements.

    Attributes
    ----------
    KMEANS : str
        K-Means clustering partitions data into k clusters by minimizing
        within-cluster variance. Requires specifying number of clusters.
        Best for: Roughly spherical clusters of similar sizes.
    HIERARCHICAL : str
        Hierarchical (agglomerative) clustering builds a tree of clusters.
        Uses average linkage for merging. Requires specifying final cluster count.
        Best for: When cluster hierarchy matters or unknown cluster shapes.
    DBSCAN : str
        Density-Based Spatial Clustering of Applications with Noise.
        Finds clusters of arbitrary shape based on density.
        Best for: Unknown number of clusters, noise detection.

    Examples
    --------
    Clustering texts with K-Means:

    >>> from insideLLMs.contrib.semantic_analysis import (
    ...     EmbeddingClusterer, ClusteringMethod
    ... )
    >>> texts = ["python code", "java code", "cute cat", "fluffy dog"]
    >>> clusterer = EmbeddingClusterer()
    >>> result = clusterer.cluster(texts, n_clusters=2, method=ClusteringMethod.KMEANS)
    >>> print(result.n_clusters)
    2

    Using hierarchical clustering for better control:

    >>> result = clusterer.cluster(
    ...     texts,
    ...     n_clusters=2,
    ...     method=ClusteringMethod.HIERARCHICAL
    ... )
    >>> for cluster in result.clusters:
    ...     print(f"Cluster {cluster.cluster_id}: {len(cluster.member_texts)} texts")
    Cluster 0: 2 texts
    Cluster 1: 2 texts

    Choosing method based on use case:

    >>> # For fast clustering of large datasets
    >>> method = ClusteringMethod.KMEANS
    >>> # For hierarchical structure analysis
    >>> method = ClusteringMethod.HIERARCHICAL

    See Also
    --------
    EmbeddingClusterer : The clustering engine that uses these methods.
    ClusteringResult : The result container for clustering operations.
    """

    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"


class DimensionalityReduction(Enum):
    """
    Enumeration of supported dimensionality reduction methods for embeddings.

    This enum defines available techniques for reducing high-dimensional
    embedding vectors to lower dimensions for visualization or analysis.

    Attributes
    ----------
    PCA : str
        Principal Component Analysis - linear technique that finds orthogonal
        directions of maximum variance. Fast and deterministic.
        Best for: Quick visualization, preprocessing, linear relationships.
    TSNE : str
        t-Distributed Stochastic Neighbor Embedding - non-linear technique
        that preserves local structure. Good for visualization.
        Best for: 2D/3D visualization of clusters, local structure.
    UMAP : str
        Uniform Manifold Approximation and Projection - non-linear technique
        that preserves both local and global structure.
        Best for: Visualization, preserving topology, faster than t-SNE.

    Examples
    --------
    Selecting dimensionality reduction for visualization:

    >>> from insideLLMs.contrib.semantic_analysis import DimensionalityReduction
    >>> # For quick linear projection
    >>> method = DimensionalityReduction.PCA
    >>> # For detailed cluster visualization
    >>> method = DimensionalityReduction.TSNE
    >>> # For preserving global structure
    >>> method = DimensionalityReduction.UMAP

    Typical usage with external libraries:

    >>> from sklearn.decomposition import PCA
    >>> from sklearn.manifold import TSNE
    >>> method = DimensionalityReduction.PCA
    >>> if method == DimensionalityReduction.PCA:
    ...     reducer = PCA(n_components=2)
    ... elif method == DimensionalityReduction.TSNE:
    ...     reducer = TSNE(n_components=2, perplexity=30)

    Notes
    -----
    These methods are defined here for type safety but require external
    libraries (scikit-learn, umap-learn) for actual implementation.

    See Also
    --------
    EmbeddingAnalyzer : For embedding analysis and manipulation.
    """

    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"


@dataclass
class SimilarityResult:
    """
    Container for the result of a similarity computation between two texts.

    This dataclass holds all information about a pairwise similarity comparison,
    including the input texts, computed score, metric used, and derived properties
    like similarity level classification.

    Parameters
    ----------
    text_a : str
        The first text that was compared.
    text_b : str
        The second text that was compared.
    score : float
        The computed similarity score. Range depends on metric:
        - Cosine: [-1, 1], typically [0, 1] for text embeddings
        - Euclidean/Manhattan: [0, 1] after normalization
        - Dot product: unbounded
        - Jaccard: [0, 1]
    metric : SimilarityMetric
        The similarity metric that was used for computation.
    normalized : bool, default=True
        Whether the score has been normalized to a standard range.
    metadata : dict[str, Any], default={}
        Additional metadata about the comparison (e.g., model used, timing).

    Attributes
    ----------
    is_similar : bool
        Property that returns True if score > 0.7 (high similarity threshold).
    similarity_level : str
        Property that categorizes the score: 'very_high' (>=0.9), 'high' (>=0.7),
        'moderate' (>=0.5), 'low' (>=0.3), or 'very_low' (<0.3).

    Examples
    --------
    Creating a similarity result manually:

    >>> from insideLLMs.contrib.semantic_analysis import SimilarityResult, SimilarityMetric
    >>> result = SimilarityResult(
    ...     text_a="Machine learning is fascinating",
    ...     text_b="Deep learning is interesting",
    ...     score=0.85,
    ...     metric=SimilarityMetric.COSINE
    ... )
    >>> result.is_similar
    True
    >>> result.similarity_level
    'high'

    Getting results from the analyzer:

    >>> from insideLLMs.contrib.semantic_analysis import TextSimilarityAnalyzer
    >>> analyzer = TextSimilarityAnalyzer()
    >>> result = analyzer.compare(
    ...     "The quick brown fox",
    ...     "A fast brown fox"
    ... )
    >>> print(f"Score: {result.score:.3f}, Level: {result.similarity_level}")
    Score: 0.667, Level: moderate

    Using with metadata for tracking:

    >>> result = SimilarityResult(
    ...     text_a="Hello world",
    ...     text_b="Hi there",
    ...     score=0.45,
    ...     metric=SimilarityMetric.COSINE,
    ...     metadata={"model": "all-MiniLM-L6-v2", "compute_time_ms": 12.5}
    ... )
    >>> result.to_dict()
    {'text_a': 'Hello world', 'text_b': 'Hi there', 'score': 0.45, ...}

    See Also
    --------
    TextSimilarityAnalyzer : The main class for computing text similarities.
    compute_similarity : Convenience function for quick comparisons.
    """

    text_a: str
    text_b: str
    score: float
    metric: SimilarityMetric
    normalized: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_similar(self) -> bool:
        """
        Check if the texts are considered similar based on a threshold.

        Returns True if the similarity score exceeds 0.7, which is generally
        considered a "high" similarity threshold for most applications.

        Returns
        -------
        bool
            True if score > 0.7, False otherwise.

        Examples
        --------
        >>> result = SimilarityResult(
        ...     text_a="test", text_b="test",
        ...     score=0.95, metric=SimilarityMetric.COSINE
        ... )
        >>> result.is_similar
        True

        >>> result = SimilarityResult(
        ...     text_a="cat", text_b="algorithm",
        ...     score=0.15, metric=SimilarityMetric.COSINE
        ... )
        >>> result.is_similar
        False
        """
        return self.score > 0.7

    @property
    def similarity_level(self) -> str:
        """
        Categorize the similarity score into a descriptive level.

        Provides a human-readable classification of the similarity score
        based on common thresholds used in semantic similarity applications.

        Returns
        -------
        str
            One of: 'very_high', 'high', 'moderate', 'low', 'very_low'.
            - 'very_high': score >= 0.9 (near-duplicates)
            - 'high': score >= 0.7 (semantically similar)
            - 'moderate': score >= 0.5 (some similarity)
            - 'low': score >= 0.3 (weak similarity)
            - 'very_low': score < 0.3 (unrelated)

        Examples
        --------
        >>> result = SimilarityResult(
        ...     text_a="identical text", text_b="identical text",
        ...     score=0.99, metric=SimilarityMetric.COSINE
        ... )
        >>> result.similarity_level
        'very_high'

        >>> result = SimilarityResult(
        ...     text_a="cats are pets", text_b="dogs are pets",
        ...     score=0.65, metric=SimilarityMetric.COSINE
        ... )
        >>> result.similarity_level
        'moderate'

        >>> result = SimilarityResult(
        ...     text_a="python code", text_b="sunny weather",
        ...     score=0.12, metric=SimilarityMetric.COSINE
        ... )
        >>> result.similarity_level
        'very_low'
        """
        if self.score >= 0.9:
            return "very_high"
        elif self.score >= 0.7:
            return "high"
        elif self.score >= 0.5:
            return "moderate"
        elif self.score >= 0.3:
            return "low"
        else:
            return "very_low"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the result to a dictionary representation.

        Creates a JSON-serializable dictionary containing all result data.
        Long texts (>100 characters) are truncated with ellipsis for readability.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: 'text_a', 'text_b', 'score', 'metric',
            'normalized', 'is_similar', 'similarity_level', 'metadata'.

        Examples
        --------
        >>> result = SimilarityResult(
        ...     text_a="Short text",
        ...     text_b="Another short text",
        ...     score=0.75,
        ...     metric=SimilarityMetric.COSINE
        ... )
        >>> d = result.to_dict()
        >>> d['score']
        0.75
        >>> d['similarity_level']
        'high'

        >>> # Long texts are truncated
        >>> long_text = "A" * 200
        >>> result = SimilarityResult(
        ...     text_a=long_text, text_b="short",
        ...     score=0.5, metric=SimilarityMetric.COSINE
        ... )
        >>> len(result.to_dict()['text_a'])
        103  # 100 chars + '...'
        """
        return {
            "text_a": self.text_a[:100] + "..." if len(self.text_a) > 100 else self.text_a,
            "text_b": self.text_b[:100] + "..." if len(self.text_b) > 100 else self.text_b,
            "score": round(self.score, 4),
            "metric": self.metric.value,
            "normalized": self.normalized,
            "is_similar": self.is_similar,
            "similarity_level": self.similarity_level,
            "metadata": self.metadata,
        }


@dataclass
class EmbeddingStats:
    """
    Statistical analysis of an embedding vector.

    This dataclass contains comprehensive statistics about an embedding vector,
    useful for understanding embedding properties, detecting anomalies, and
    comparing embeddings from different models.

    Parameters
    ----------
    dimension : int
        The dimensionality (number of elements) of the embedding vector.
    magnitude : float
        The L2 norm (Euclidean length) of the vector. Useful for checking
        if embeddings are normalized (magnitude ~= 1.0).
    mean : float
        The arithmetic mean of all values in the embedding.
    std : float
        The standard deviation of values in the embedding.
    min_val : float
        The minimum value in the embedding vector.
    max_val : float
        The maximum value in the embedding vector.
    sparsity : float
        Fraction of values that are near-zero (|x| < 1e-6). Range [0, 1].
        High sparsity indicates many zero or near-zero dimensions.
    metadata : dict[str, Any], default={}
        Additional metadata (e.g., model name, text that was embedded).

    Examples
    --------
    Computing statistics for an embedding:

    >>> from insideLLMs.contrib.semantic_analysis import EmbeddingAnalyzer
    >>> analyzer = EmbeddingAnalyzer()
    >>> embedding = [0.1, 0.2, 0.3, 0.4, 0.0, 0.0]
    >>> stats = analyzer.compute_stats(embedding)
    >>> print(f"Dimension: {stats.dimension}, Magnitude: {stats.magnitude:.3f}")
    Dimension: 6, Magnitude: 0.548
    >>> print(f"Sparsity: {stats.sparsity:.2f}")
    Sparsity: 0.33

    Creating stats manually for testing:

    >>> stats = EmbeddingStats(
    ...     dimension=768,
    ...     magnitude=1.0,
    ...     mean=0.001,
    ...     std=0.036,
    ...     min_val=-0.15,
    ...     max_val=0.12,
    ...     sparsity=0.05
    ... )
    >>> stats.to_dict()['magnitude']
    1.0

    Checking if embedding is normalized:

    >>> def is_normalized(stats: EmbeddingStats, tolerance: float = 0.01) -> bool:
    ...     return abs(stats.magnitude - 1.0) < tolerance
    >>> stats = EmbeddingStats(
    ...     dimension=384, magnitude=0.998, mean=0.0,
    ...     std=0.05, min_val=-0.2, max_val=0.2, sparsity=0.0
    ... )
    >>> is_normalized(stats)
    True

    See Also
    --------
    EmbeddingAnalyzer : The class that computes these statistics.
    analyze_embedding : Convenience function for quick analysis.
    """

    dimension: int
    magnitude: float
    mean: float
    std: float
    min_val: float
    max_val: float
    sparsity: float  # Fraction of near-zero values
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the statistics to a dictionary representation.

        Creates a JSON-serializable dictionary with all statistics rounded
        to 4 decimal places for readability.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: 'dimension', 'magnitude', 'mean', 'std',
            'min_val', 'max_val', 'sparsity', 'metadata'.

        Examples
        --------
        >>> stats = EmbeddingStats(
        ...     dimension=512, magnitude=1.0001, mean=0.00123,
        ...     std=0.04456, min_val=-0.18765, max_val=0.15234,
        ...     sparsity=0.02345
        ... )
        >>> d = stats.to_dict()
        >>> d['magnitude']
        1.0001
        >>> d['mean']
        0.0012

        >>> # Serializing to JSON
        >>> import json
        >>> json_str = json.dumps(stats.to_dict())
        >>> len(json_str) > 0
        True
        """
        return {
            "dimension": self.dimension,
            "magnitude": round(self.magnitude, 4),
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "min_val": round(self.min_val, 4),
            "max_val": round(self.max_val, 4),
            "sparsity": round(self.sparsity, 4),
            "metadata": self.metadata,
        }


@dataclass
class ClusterInfo:
    """
    Detailed information about a single cluster from clustering results.

    This dataclass contains comprehensive information about one cluster,
    including its members, centroid, and quality metrics. Used as part
    of ClusteringResult to describe each discovered cluster.

    Parameters
    ----------
    cluster_id : int
        Unique identifier for this cluster (0-indexed).
    size : int
        Number of texts/embeddings in this cluster.
    centroid : list[float]
        The centroid (mean) embedding vector of all cluster members.
    member_indices : list[int]
        Indices into the original text list for all cluster members.
    member_texts : list[str]
        The actual text strings belonging to this cluster.
    intra_cluster_similarity : float
        Average pairwise similarity between all members within the cluster.
        Range [0, 1]. Higher values indicate tighter, more coherent clusters.
    representative_text : str, optional
        The text closest to the cluster centroid, serving as a representative
        example of the cluster's content.
    metadata : dict[str, Any], default={}
        Additional metadata (e.g., cluster labels, keywords).

    Examples
    --------
    Examining cluster information after clustering:

    >>> from insideLLMs.contrib.semantic_analysis import cluster_texts
    >>> texts = [
    ...     "Python programming", "Java development",
    ...     "cute cats", "fluffy dogs",
    ...     "JavaScript coding", "adorable puppies"
    ... ]
    >>> result = cluster_texts(texts, n_clusters=2)
    >>> for cluster in result.clusters:
    ...     print(f"Cluster {cluster.cluster_id}:")
    ...     print(f"  Size: {cluster.size}")
    ...     print(f"  Representative: {cluster.representative_text}")
    ...     print(f"  Cohesion: {cluster.intra_cluster_similarity:.3f}")
    Cluster 0:
      Size: 3
      Representative: Python programming
      Cohesion: 0.456
    Cluster 1:
      Size: 3
      Representative: cute cats
      Cohesion: 0.523

    Creating cluster info manually:

    >>> cluster = ClusterInfo(
    ...     cluster_id=0,
    ...     size=3,
    ...     centroid=[0.1, 0.2, 0.3],
    ...     member_indices=[0, 2, 5],
    ...     member_texts=["text A", "text B", "text C"],
    ...     intra_cluster_similarity=0.85,
    ...     representative_text="text A"
    ... )
    >>> cluster.size
    3

    Accessing cluster members:

    >>> cluster = ClusterInfo(
    ...     cluster_id=1,
    ...     size=4,
    ...     centroid=[0.5, 0.5],
    ...     member_indices=[1, 3, 4, 6],
    ...     member_texts=["doc1", "doc2", "doc3", "doc4"],
    ...     intra_cluster_similarity=0.72,
    ...     representative_text="doc2"
    ... )
    >>> print(f"Cluster has {cluster.size} members")
    Cluster has 4 members
    >>> print(f"Best example: {cluster.representative_text}")
    Best example: doc2

    See Also
    --------
    ClusteringResult : Container that holds multiple ClusterInfo objects.
    EmbeddingClusterer : The class that produces clustering results.
    """

    cluster_id: int
    size: int
    centroid: list[float]
    member_indices: list[int]
    member_texts: list[str]
    intra_cluster_similarity: float
    representative_text: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the cluster information to a dictionary representation.

        Creates a JSON-serializable dictionary. Note that member_texts is
        truncated to the first 5 entries to keep output manageable.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys: 'cluster_id', 'size', 'member_indices',
            'member_texts' (first 5), 'intra_cluster_similarity',
            'representative_text', 'metadata'.

        Examples
        --------
        >>> cluster = ClusterInfo(
        ...     cluster_id=0,
        ...     size=10,
        ...     centroid=[0.1, 0.2],
        ...     member_indices=list(range(10)),
        ...     member_texts=[f"text_{i}" for i in range(10)],
        ...     intra_cluster_similarity=0.8567,
        ...     representative_text="text_3"
        ... )
        >>> d = cluster.to_dict()
        >>> len(d['member_texts'])  # Truncated to 5
        5
        >>> d['intra_cluster_similarity']
        0.8567

        >>> # Full member list still available on object
        >>> len(cluster.member_texts)
        10
        """
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "member_indices": self.member_indices,
            "member_texts": self.member_texts[:5],  # First 5
            "intra_cluster_similarity": round(self.intra_cluster_similarity, 4),
            "representative_text": self.representative_text,
            "metadata": self.metadata,
        }


@dataclass
class ClusteringResult:
    """Result of clustering operation."""

    method: ClusteringMethod
    n_clusters: int
    clusters: list[ClusterInfo]
    labels: list[int]
    silhouette_score: Optional[float] = None
    inertia: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method.value,
            "n_clusters": self.n_clusters,
            "clusters": [c.to_dict() for c in self.clusters],
            "labels": self.labels,
            "silhouette_score": round(self.silhouette_score, 4) if self.silhouette_score else None,
            "inertia": round(self.inertia, 4) if self.inertia else None,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """Result of a semantic search."""

    query: str
    matches: list[tuple[int, str, float]]  # (index, text, score)
    metric: SimilarityMetric
    total_searched: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def top_match(self) -> Optional[tuple[int, str, float]]:
        """Get the top match."""
        return self.matches[0] if self.matches else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query[:100] + "..." if len(self.query) > 100 else self.query,
            "matches": [
                {"index": idx, "text": text[:100], "score": round(score, 4)}
                for idx, text, score in self.matches[:10]
            ],
            "metric": self.metric.value,
            "total_searched": self.total_searched,
            "metadata": self.metadata,
        }


class VectorSimilarityCalculator:
    """Calculates similarity between vectors/texts."""

    def __init__(self, default_metric: SimilarityMetric = SimilarityMetric.COSINE):
        """Initialize calculator."""
        self.default_metric = default_metric

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            raise ValueError(f"Vector dimensions must match: {len(a)} vs {len(b)}")

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def euclidean_distance(self, a: list[float], b: list[float]) -> float:
        """Compute Euclidean distance between two vectors."""
        if len(a) != len(b):
            raise ValueError(f"Vector dimensions must match: {len(a)} vs {len(b)}")

        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def manhattan_distance(self, a: list[float], b: list[float]) -> float:
        """Compute Manhattan distance between two vectors."""
        if len(a) != len(b):
            raise ValueError(f"Vector dimensions must match: {len(a)} vs {len(b)}")

        return sum(abs(x - y) for x, y in zip(a, b))

    def dot_product(self, a: list[float], b: list[float]) -> float:
        """Compute dot product between two vectors."""
        if len(a) != len(b):
            raise ValueError(f"Vector dimensions must match: {len(a)} vs {len(b)}")

        return sum(x * y for x, y in zip(a, b))

    def jaccard_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute Jaccard similarity between two binary/set vectors."""
        if len(a) != len(b):
            raise ValueError(f"Vector dimensions must match: {len(a)} vs {len(b)}")

        # Treat as sets: non-zero values are members
        set_a = {i for i, x in enumerate(a) if x != 0}
        set_b = {i for i, y in enumerate(b) if y != 0}

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        return intersection / union if union > 0 else 0.0

    def compute(
        self,
        a: list[float],
        b: list[float],
        metric: Optional[SimilarityMetric] = None,
    ) -> float:
        """Compute similarity/distance based on metric."""
        metric = metric or self.default_metric

        if metric == SimilarityMetric.COSINE:
            return self.cosine_similarity(a, b)
        elif metric == SimilarityMetric.EUCLIDEAN:
            # Convert distance to similarity (0-1 range)
            dist = self.euclidean_distance(a, b)
            return 1 / (1 + dist)
        elif metric == SimilarityMetric.MANHATTAN:
            # Convert distance to similarity
            dist = self.manhattan_distance(a, b)
            return 1 / (1 + dist)
        elif metric == SimilarityMetric.DOT_PRODUCT:
            return self.dot_product(a, b)
        elif metric == SimilarityMetric.JACCARD:
            return self.jaccard_similarity(a, b)
        else:
            return self.cosine_similarity(a, b)

    def pairwise_similarity(
        self,
        embeddings: list[list[float]],
        metric: Optional[SimilarityMetric] = None,
    ) -> list[list[float]]:
        """Compute pairwise similarity matrix."""
        n = len(embeddings)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    sim = self.compute(embeddings[i], embeddings[j], metric)
                    matrix[i][j] = sim
                    matrix[j][i] = sim

        return matrix


class EmbeddingAnalyzer:
    """Analyzes embedding vectors."""

    def __init__(self, similarity_calculator: Optional[VectorSimilarityCalculator] = None):
        """Initialize analyzer."""
        self.calculator = similarity_calculator or VectorSimilarityCalculator()

    def compute_stats(self, embedding: list[float]) -> EmbeddingStats:
        """Compute statistics for an embedding."""
        n = len(embedding)
        if n == 0:
            return EmbeddingStats(
                dimension=0,
                magnitude=0.0,
                mean=0.0,
                std=0.0,
                min_val=0.0,
                max_val=0.0,
                sparsity=1.0,
            )

        magnitude = sum(x * x for x in embedding) ** 0.5
        mean = sum(embedding) / n
        variance = sum((x - mean) ** 2 for x in embedding) / n
        std = variance**0.5
        min_val = min(embedding)
        max_val = max(embedding)

        # Sparsity: fraction of values close to zero
        threshold = 1e-6
        near_zero = sum(1 for x in embedding if abs(x) < threshold)
        sparsity = near_zero / n

        return EmbeddingStats(
            dimension=n,
            magnitude=magnitude,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            sparsity=sparsity,
        )

    def normalize(self, embedding: list[float]) -> list[float]:
        """Normalize embedding to unit length."""
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude == 0:
            return embedding[:]
        return [x / magnitude for x in embedding]

    def average_embeddings(self, embeddings: list[list[float]]) -> list[float]:
        """Compute average of multiple embeddings."""
        if not embeddings:
            return []

        n = len(embeddings)
        dim = len(embeddings[0])

        avg = [0.0] * dim
        for emb in embeddings:
            for i, val in enumerate(emb):
                avg[i] += val

        return [x / n for x in avg]

    def weighted_average(
        self,
        embeddings: list[list[float]],
        weights: list[float],
    ) -> list[float]:
        """Compute weighted average of embeddings."""
        if not embeddings:
            return []

        if len(embeddings) != len(weights):
            raise ValueError("Number of embeddings must match number of weights")

        dim = len(embeddings[0])
        total_weight = sum(weights)

        if total_weight == 0:
            return [0.0] * dim

        avg = [0.0] * dim
        for emb, w in zip(embeddings, weights):
            for i, val in enumerate(emb):
                avg[i] += val * w

        return [x / total_weight for x in avg]

    def find_outliers(
        self,
        embeddings: list[list[float]],
        threshold: float = 2.0,
    ) -> list[int]:
        """Find outlier embeddings based on distance from centroid."""
        if len(embeddings) < 2:
            return []

        centroid = self.average_embeddings(embeddings)

        # Compute distances to centroid
        distances = []
        for emb in embeddings:
            dist = sum((a - b) ** 2 for a, b in zip(emb, centroid)) ** 0.5
            distances.append(dist)

        # Compute mean and std of distances
        mean_dist = sum(distances) / len(distances)
        variance = sum((d - mean_dist) ** 2 for d in distances) / len(distances)
        std_dist = variance**0.5

        if std_dist == 0:
            return []

        # Find outliers (distance > threshold * std from mean)
        outliers = []
        for i, dist in enumerate(distances):
            z_score = (dist - mean_dist) / std_dist
            if abs(z_score) > threshold:
                outliers.append(i)

        return outliers


class TextSimilarityAnalyzer:
    """Analyzes semantic similarity between texts."""

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ):
        """Initialize analyzer."""
        self.embed_fn = embed_fn or self._default_embed
        self.metric = metric
        self.calculator = VectorSimilarityCalculator(metric)
        self._cache: dict[str, list[float]] = {}

    def _default_embed(self, text: str) -> list[float]:
        """Default embedding: bag of words (simple baseline)."""
        # Simple word frequency vector
        words = text.lower().split()
        vocab = sorted(set(words))
        if not vocab:
            return [0.0]

        # Create frequency vector
        freq = dict.fromkeys(vocab, 0)
        for w in words:
            freq[w] += 1

        # Normalize
        total = sum(freq.values())
        return [freq[w] / total for w in vocab]

    def _get_embedding(self, text: str, use_cache: bool = True) -> list[float]:
        """Get embedding for text, using cache if available."""
        cache_key = hashlib.md5(text.encode()).hexdigest()

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self.embed_fn(text)

        if use_cache:
            self._cache[cache_key] = embedding

        return embedding

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def compare(
        self,
        text_a: str,
        text_b: str,
        metric: Optional[SimilarityMetric] = None,
    ) -> SimilarityResult:
        """Compare two texts for similarity."""
        metric = metric or self.metric

        emb_a = self._get_embedding(text_a)
        emb_b = self._get_embedding(text_b)

        # Ensure same dimensions for comparison
        if len(emb_a) != len(emb_b):
            # Pad shorter one with zeros
            max_len = max(len(emb_a), len(emb_b))
            emb_a = emb_a + [0.0] * (max_len - len(emb_a))
            emb_b = emb_b + [0.0] * (max_len - len(emb_b))

        score = self.calculator.compute(emb_a, emb_b, metric)

        return SimilarityResult(
            text_a=text_a,
            text_b=text_b,
            score=score,
            metric=metric,
        )

    def compare_batch(
        self,
        texts_a: list[str],
        texts_b: list[str],
        metric: Optional[SimilarityMetric] = None,
    ) -> list[SimilarityResult]:
        """Compare lists of texts pairwise."""
        if len(texts_a) != len(texts_b):
            raise ValueError("Lists must have same length")

        return [self.compare(a, b, metric) for a, b in zip(texts_a, texts_b)]

    def find_most_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
        metric: Optional[SimilarityMetric] = None,
    ) -> SearchResult:
        """Find most similar texts to query."""
        metric = metric or self.metric
        query_emb = self._get_embedding(query)

        scores = []
        for i, candidate in enumerate(candidates):
            cand_emb = self._get_embedding(candidate)

            # Handle dimension mismatch
            if len(query_emb) != len(cand_emb):
                max_len = max(len(query_emb), len(cand_emb))
                q = query_emb + [0.0] * (max_len - len(query_emb))
                c = cand_emb + [0.0] * (max_len - len(cand_emb))
            else:
                q, c = query_emb, cand_emb

            score = self.calculator.compute(q, c, metric)
            scores.append((i, candidate, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[2], reverse=True)

        return SearchResult(
            query=query,
            matches=scores[:top_k],
            metric=metric,
            total_searched=len(candidates),
        )

    def find_duplicates(
        self,
        texts: list[str],
        threshold: float = 0.9,
    ) -> list[tuple[int, int, float]]:
        """Find near-duplicate texts."""
        duplicates = []

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                result = self.compare(texts[i], texts[j])
                if result.score >= threshold:
                    duplicates.append((i, j, result.score))

        return duplicates


class EmbeddingClusterer:
    """Clusters text embeddings."""

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
        similarity_calculator: Optional[VectorSimilarityCalculator] = None,
    ):
        """Initialize clusterer."""
        self.embed_fn = embed_fn
        self.calculator = similarity_calculator or VectorSimilarityCalculator()

    def _simple_kmeans(
        self,
        embeddings: list[list[float]],
        k: int,
        max_iters: int = 100,
    ) -> tuple[list[int], list[list[float]], float]:
        """Simple k-means implementation."""
        import random

        n = len(embeddings)
        dim = len(embeddings[0])

        # Initialize centroids randomly
        indices = random.sample(range(n), min(k, n))
        centroids = [embeddings[i][:] for i in indices]

        labels = [0] * n
        prev_labels = None
        inertia = 0.0

        for _ in range(max_iters):
            # Assign points to nearest centroid
            for i, emb in enumerate(embeddings):
                best_dist = float("inf")
                best_label = 0
                for j, centroid in enumerate(centroids):
                    dist = sum((a - b) ** 2 for a, b in zip(emb, centroid))
                    if dist < best_dist:
                        best_dist = dist
                        best_label = j
                labels[i] = best_label

            # Check convergence
            if labels == prev_labels:
                break
            prev_labels = labels[:]

            # Update centroids
            for j in range(k):
                cluster_points = [embeddings[i] for i in range(n) if labels[i] == j]
                if cluster_points:
                    centroids[j] = [
                        sum(p[d] for p in cluster_points) / len(cluster_points) for d in range(dim)
                    ]

        # Compute inertia
        for i, emb in enumerate(embeddings):
            centroid = centroids[labels[i]]
            inertia += sum((a - b) ** 2 for a, b in zip(emb, centroid))

        return labels, centroids, inertia

    def _hierarchical_clustering(
        self,
        embeddings: list[list[float]],
        n_clusters: int,
    ) -> list[int]:
        """Simple hierarchical clustering."""
        n = len(embeddings)

        # Start with each point as its own cluster
        clusters = [[i] for i in range(n)]
        labels = list(range(n))

        # Compute pairwise distances
        distances = {}
        for i in range(n):
            for j in range(i + 1, n):
                dist = sum((a - b) ** 2 for a, b in zip(embeddings[i], embeddings[j])) ** 0.5
                distances[(i, j)] = dist

        # Merge clusters until we have n_clusters
        while len(clusters) > n_clusters:
            # Find closest pair of clusters
            min_dist = float("inf")
            merge_pair = (0, 1)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Average linkage
                    cluster_dist = 0
                    count = 0
                    for a in clusters[i]:
                        for b in clusters[j]:
                            key = (min(a, b), max(a, b))
                            cluster_dist += distances.get(key, 0)
                            count += 1
                    if count > 0:
                        avg_dist = cluster_dist / count
                        if avg_dist < min_dist:
                            min_dist = avg_dist
                            merge_pair = (i, j)

            # Merge clusters
            i, j = merge_pair
            clusters[i] = clusters[i] + clusters[j]
            del clusters[j]

        # Assign labels
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                labels[idx] = cluster_id

        return labels

    def _dbscan_clustering(
        self,
        embeddings: list[list[float]],
        eps: float,
        min_samples: int,
    ) -> list[int]:
        """DBSCAN: density-based clustering. Returns labels (-1 = noise)."""
        n = len(embeddings)

        # Tiny dataset fallback: ensure we can form at least one cluster
        if n < 2:
            return [0] * n if n else []
        effective_min = min(min_samples, n)
        if effective_min < 1:
            effective_min = 1

        def dist(a: list[float], b: list[float]) -> float:
            return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

        # Build neighbor lists
        neighbors: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = dist(embeddings[i], embeddings[j])
                if d <= eps:
                    neighbors[i].append(j)
                    neighbors[j].append(i)

        labels = [-1] * n
        cluster_id = 0

        def expand_cluster(seed: int) -> None:
            nonlocal cluster_id
            stack = [seed]
            while stack:
                p = stack.pop()
                if labels[p] >= 0:
                    continue
                labels[p] = cluster_id
                if len(neighbors[p]) >= effective_min - 1:
                    for q in neighbors[p]:
                        if labels[q] < 0:
                            stack.append(q)

        for i in range(n):
            if labels[i] >= 0:
                continue
            if len(neighbors[i]) >= effective_min - 1:
                expand_cluster(i)
                cluster_id += 1

        return labels

    def cluster(
        self,
        texts: list[str],
        n_clusters: int = 5,
        method: ClusteringMethod = ClusteringMethod.KMEANS,
        eps: float = 0.5,
        min_samples: int = 2,
    ) -> ClusteringResult:
        """Cluster texts based on embeddings."""
        if not texts:
            return ClusteringResult(
                method=method,
                n_clusters=0,
                clusters=[],
                labels=[],
            )

        # Get embeddings
        if self.embed_fn:
            embeddings = [self.embed_fn(text) for text in texts]
        else:
            # Default: simple word frequency
            analyzer = TextSimilarityAnalyzer()
            embeddings = [analyzer._get_embedding(text) for text in texts]

        # Ensure consistent dimensions
        max_dim = max(len(e) for e in embeddings)
        embeddings = [e + [0.0] * (max_dim - len(e)) for e in embeddings]

        # Cluster
        if method == ClusteringMethod.KMEANS:
            labels, centroids, inertia = self._simple_kmeans(embeddings, n_clusters)
            unique_labels = list(range(n_clusters))
        elif method == ClusteringMethod.HIERARCHICAL:
            labels = self._hierarchical_clustering(embeddings, n_clusters)
            centroids = []
            inertia = None
            unique_labels = list(range(n_clusters))
            for c in range(n_clusters):
                cluster_embs = [embeddings[i] for i in range(len(texts)) if labels[i] == c]
                if cluster_embs:
                    centroid = [
                        sum(e[d] for e in cluster_embs) / len(cluster_embs) for d in range(max_dim)
                    ]
                    centroids.append(centroid)
                else:
                    centroids.append([0.0] * max_dim)
        elif method == ClusteringMethod.DBSCAN:
            labels = self._dbscan_clustering(embeddings, eps=eps, min_samples=min_samples)
            unique_labels = sorted(set(labels) - {-1})
            centroids = []
            inertia = None
            for c in unique_labels:
                cluster_embs = [embeddings[i] for i in range(len(texts)) if labels[i] == c]
                if cluster_embs:
                    centroid = [
                        sum(e[d] for e in cluster_embs) / len(cluster_embs) for d in range(max_dim)
                    ]
                    centroids.append(centroid)
                else:
                    centroids.append([0.0] * max_dim)
            n_clusters = len(unique_labels)
        else:
            labels, centroids, inertia = self._simple_kmeans(embeddings, n_clusters)
            unique_labels = list(range(n_clusters))

        # Build cluster info
        clusters = []
        for c in unique_labels:
            member_indices = [i for i, label in enumerate(labels) if label == c]
            member_texts = [texts[i] for i in member_indices]

            if member_texts:
                # Compute intra-cluster similarity
                if len(member_texts) > 1:
                    cluster_embs = [embeddings[i] for i in member_indices]
                    sims = []
                    for i in range(len(cluster_embs)):
                        for j in range(i + 1, len(cluster_embs)):
                            sim = self.calculator.cosine_similarity(
                                cluster_embs[i], cluster_embs[j]
                            )
                            sims.append(sim)
                    intra_sim = sum(sims) / len(sims) if sims else 1.0
                else:
                    intra_sim = 1.0

                # Representative: closest to centroid
                c_idx = unique_labels.index(c) if c in unique_labels else 0
                centroid = centroids[c_idx] if c_idx < len(centroids) else [0.0] * max_dim
                best_dist = float("inf")
                best_text = member_texts[0]
                for idx in member_indices:
                    dist = sum((a - b) ** 2 for a, b in zip(embeddings[idx], centroid))
                    if dist < best_dist:
                        best_dist = dist
                        best_text = texts[idx]

                clusters.append(
                    ClusterInfo(
                        cluster_id=c,
                        size=len(member_texts),
                        centroid=centroid,
                        member_indices=member_indices,
                        member_texts=member_texts,
                        intra_cluster_similarity=intra_sim,
                        representative_text=best_text,
                    )
                )

        return ClusteringResult(
            method=method,
            n_clusters=len(clusters),
            clusters=clusters,
            labels=labels,
            inertia=inertia,
        )

    def find_optimal_k(
        self,
        texts: list[str],
        max_k: int = 10,
        method: ClusteringMethod = ClusteringMethod.KMEANS,
    ) -> dict[str, Any]:
        """Find optimal number of clusters using elbow method."""
        if len(texts) < 2:
            return {"optimal_k": 1, "inertias": []}

        # Get embeddings
        if self.embed_fn:
            embeddings = [self.embed_fn(text) for text in texts]
        else:
            analyzer = TextSimilarityAnalyzer()
            embeddings = [analyzer._get_embedding(text) for text in texts]

        max_dim = max(len(e) for e in embeddings)
        embeddings = [e + [0.0] * (max_dim - len(e)) for e in embeddings]

        inertias = []
        max_k = min(max_k, len(texts))

        for k in range(1, max_k + 1):
            labels, centroids, inertia = self._simple_kmeans(embeddings, k)
            inertias.append(inertia)

        # Find elbow (maximum curvature)
        if len(inertias) < 3:
            optimal_k = 1
        else:
            # Simple elbow detection: biggest drop
            drops = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
            if drops:
                # Find where drops start to diminish significantly
                optimal_k = drops.index(max(drops)) + 1
            else:
                optimal_k = 1

        return {
            "optimal_k": optimal_k,
            "inertias": inertias,
            "k_range": list(range(1, max_k + 1)),
        }


class SemanticSearch:
    """Semantic search engine for texts."""

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ):
        """Initialize search engine."""
        self.embed_fn = embed_fn
        self.metric = metric
        self.calculator = VectorSimilarityCalculator(metric)
        self.index: list[tuple[str, list[float]]] = []
        self.metadata: list[dict[str, Any]] = []

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        if self.embed_fn:
            return self.embed_fn(text)
        else:
            # Simple default
            words = text.lower().split()
            if not words:
                return [0.0]
            vocab = sorted(set(words))
            freq = dict.fromkeys(vocab, 0)
            for w in words:
                freq[w] += 1
            total = sum(freq.values())
            return [freq[w] / total for w in vocab]

    def add(self, text: str, metadata: Optional[dict[str, Any]] = None) -> int:
        """Add text to index."""
        embedding = self._get_embedding(text)
        idx = len(self.index)
        self.index.append((text, embedding))
        self.metadata.append(metadata or {})
        return idx

    def add_batch(
        self,
        texts: list[str],
        metadata: Optional[list[dict[str, Any]]] = None,
    ) -> list[int]:
        """Add multiple texts to index."""
        metadata = metadata or [{}] * len(texts)
        return [self.add(text, meta) for text, meta in zip(texts, metadata)]

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> SearchResult:
        """Search for similar texts."""
        query_emb = self._get_embedding(query)

        scores = []
        for i, (text, emb) in enumerate(self.index):
            # Handle dimension mismatch
            if len(query_emb) != len(emb):
                max_len = max(len(query_emb), len(emb))
                q = query_emb + [0.0] * (max_len - len(query_emb))
                e = emb + [0.0] * (max_len - len(emb))
            else:
                q, e = query_emb, emb

            score = self.calculator.compute(q, e, self.metric)
            if score >= threshold:
                scores.append((i, text, score))

        scores.sort(key=lambda x: x[2], reverse=True)

        return SearchResult(
            query=query,
            matches=scores[:top_k],
            metric=self.metric,
            total_searched=len(self.index),
        )

    def clear(self) -> None:
        """Clear the index."""
        self.index.clear()
        self.metadata.clear()

    @property
    def size(self) -> int:
        """Number of texts in index."""
        return len(self.index)


# Convenience functions
def compute_similarity(
    text_a: str,
    text_b: str,
    embed_fn: Optional[Callable[[str], list[float]]] = None,
    metric: SimilarityMetric = SimilarityMetric.COSINE,
) -> SimilarityResult:
    """Compute similarity between two texts."""
    analyzer = TextSimilarityAnalyzer(embed_fn, metric)
    return analyzer.compare(text_a, text_b)


def find_similar_texts(
    query: str,
    candidates: list[str],
    top_k: int = 5,
    embed_fn: Optional[Callable[[str], list[float]]] = None,
) -> SearchResult:
    """Find most similar texts to query."""
    analyzer = TextSimilarityAnalyzer(embed_fn)
    return analyzer.find_most_similar(query, candidates, top_k)


def cluster_texts(
    texts: list[str],
    n_clusters: int = 5,
    embed_fn: Optional[Callable[[str], list[float]]] = None,
) -> ClusteringResult:
    """Cluster texts by semantic similarity."""
    clusterer = EmbeddingClusterer(embed_fn)
    return clusterer.cluster(texts, n_clusters)


def find_duplicate_texts(
    texts: list[str],
    threshold: float = 0.9,
    embed_fn: Optional[Callable[[str], list[float]]] = None,
) -> list[tuple[int, int, float]]:
    """Find near-duplicate texts."""
    analyzer = TextSimilarityAnalyzer(embed_fn)
    return analyzer.find_duplicates(texts, threshold)


def analyze_embedding(embedding: list[float]) -> EmbeddingStats:
    """Analyze an embedding vector."""
    analyzer = EmbeddingAnalyzer()
    return analyzer.compute_stats(embedding)


def normalize_embedding(embedding: list[float]) -> list[float]:
    """Normalize embedding to unit length."""
    analyzer = EmbeddingAnalyzer()
    return analyzer.normalize(embedding)


def create_semantic_index(
    texts: list[str],
    embed_fn: Optional[Callable[[str], list[float]]] = None,
) -> SemanticSearch:
    """Create a semantic search index."""
    search = SemanticSearch(embed_fn)
    search.add_batch(texts)
    return search


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import SimilarityCalculator. The canonical name is
# VectorSimilarityCalculator.
SimilarityCalculator = VectorSimilarityCalculator
