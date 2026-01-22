"""
Semantic similarity and embedding analysis utilities.

Provides tools for:
- Semantic similarity computation
- Text embedding analysis and comparison
- Clustering and dimensionality reduction
- Semantic search and nearest neighbors
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class SimilarityMetric(Enum):
    """Metrics for computing similarity."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    DOT_PRODUCT = "dot_product"
    JACCARD = "jaccard"


class ClusteringMethod(Enum):
    """Methods for clustering embeddings."""

    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"


class DimensionalityReduction(Enum):
    """Methods for dimensionality reduction."""

    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"


@dataclass
class SimilarityResult:
    """Result of a similarity computation."""

    text_a: str
    text_b: str
    score: float
    metric: SimilarityMetric
    normalized: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_similar(self) -> bool:
        """Check if texts are similar (score > 0.7)."""
        return self.score > 0.7

    @property
    def similarity_level(self) -> str:
        """Categorize similarity level."""
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
        """Convert to dictionary."""
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
    """Statistics about an embedding."""

    dimension: int
    magnitude: float
    mean: float
    std: float
    min_val: float
    max_val: float
    sparsity: float  # Fraction of near-zero values
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Information about a cluster."""

    cluster_id: int
    size: int
    centroid: list[float]
    member_indices: list[int]
    member_texts: list[str]
    intra_cluster_similarity: float
    representative_text: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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


class SimilarityCalculator:
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

    def __init__(self, similarity_calculator: Optional[SimilarityCalculator] = None):
        """Initialize analyzer."""
        self.calculator = similarity_calculator or SimilarityCalculator()

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
        self.calculator = SimilarityCalculator(metric)
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
        similarity_calculator: Optional[SimilarityCalculator] = None,
    ):
        """Initialize clusterer."""
        self.embed_fn = embed_fn
        self.calculator = similarity_calculator or SimilarityCalculator()

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

    def cluster(
        self,
        texts: list[str],
        n_clusters: int = 5,
        method: ClusteringMethod = ClusteringMethod.KMEANS,
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
        elif method == ClusteringMethod.HIERARCHICAL:
            labels = self._hierarchical_clustering(embeddings, n_clusters)
            centroids = []
            inertia = None
            # Compute centroids
            for c in range(n_clusters):
                cluster_embs = [embeddings[i] for i in range(len(texts)) if labels[i] == c]
                if cluster_embs:
                    centroid = [
                        sum(e[d] for e in cluster_embs) / len(cluster_embs) for d in range(max_dim)
                    ]
                    centroids.append(centroid)
                else:
                    centroids.append([0.0] * max_dim)
        else:
            # Default to kmeans
            labels, centroids, inertia = self._simple_kmeans(embeddings, n_clusters)

        # Build cluster info
        clusters = []
        for c in range(n_clusters):
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
                centroid = centroids[c] if c < len(centroids) else [0.0] * max_dim
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
        self.calculator = SimilarityCalculator(metric)
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
