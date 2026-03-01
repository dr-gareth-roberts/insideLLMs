"""Native RAG (Retrieval-Augmented Generation) Pipeline.

This module provides a comprehensive, flexible RAG implementation for building
retrieval-augmented generation systems. It includes all the core components
needed for document ingestion, chunking, embedding, vector storage, retrieval,
and generation.

Overview
--------
The RAG pipeline consists of several interconnected components:

1. **Document Management**: The `Document` and `RetrievalResult` dataclasses
   provide structured representations for documents and search results.

2. **Embedding**: The `EmbeddingModel` protocol allows plugging in any embedding
   provider (OpenAI, Sentence Transformers, etc.). A `SimpleEmbedding` class
   is included for testing without external dependencies.

3. **Vector Storage**: The `VectorStore` protocol enables integration with
   various vector databases. An `InMemoryVectorStore` is provided for
   development and small-scale use.

4. **Chunking**: The `TextChunker` class splits documents into overlapping
   chunks using a recursive strategy that respects natural text boundaries.

5. **Retrieval**: The `Retriever` class combines embedding and vector storage
   to provide a unified interface for adding and searching documents.

6. **RAG Chain**: The `RAGChain` class orchestrates the entire pipeline,
   combining retrieval with LLM generation for question answering.

Architecture
------------
The module follows a protocol-based design for maximum flexibility::

    ┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
    │  Documents  │────>│   TextChunker   │────>│  Retriever  │
    └─────────────┘     └─────────────────┘     └──────┬──────┘
                                                       │
                        ┌─────────────────┐            │
                        │ EmbeddingModel  │<───────────┤
                        └─────────────────┘            │
                                                       │
                        ┌─────────────────┐            │
                        │   VectorStore   │<───────────┘
                        └─────────────────┘
                                │
                        ┌───────▼───────┐     ┌─────────────┐
                        │   RAGChain    │────>│    Model    │
                        └───────────────┘     └─────────────┘

Examples
--------
Basic RAG usage with default components:

>>> from insideLLMs.rag.retrieval import RAGChain, InMemoryVectorStore
>>> from insideLLMs.models import OpenAIModel
>>>
>>> # Create LLM
>>> model = OpenAIModel(model_name="gpt-4")
>>>
>>> # Build RAG chain with defaults (SimpleEmbedding + InMemoryVectorStore)
>>> rag = RAGChain(model=model)
>>>
>>> # Add documents
>>> docs = [
...     "Python is a high-level programming language known for readability.",
...     "Machine learning is a subset of artificial intelligence.",
...     "Neural networks are inspired by biological brain structures."
... ]
>>> rag.add_documents(docs)
>>>
>>> # Query the documents
>>> response = rag.query("What is Python known for?")
>>> print(response.answer)
Python is known for its readability...

Custom embedding model integration:

>>> from insideLLMs.rag.retrieval import Retriever, InMemoryVectorStore
>>> from sentence_transformers import SentenceTransformer
>>>
>>> # Create a wrapper for sentence-transformers
>>> class STEmbedding:
...     def __init__(self, model_name="all-MiniLM-L6-v2"):
...         self.model = SentenceTransformer(model_name)
...
...     def embed(self, text: str) -> list[float]:
...         return self.model.encode(text).tolist()
...
...     def embed_batch(self, texts: list[str]) -> list[list[float]]:
...         return self.model.encode(texts).tolist()
>>>
>>> # Use with retriever
>>> embedding = STEmbedding()
>>> store = InMemoryVectorStore()
>>> retriever = Retriever(embedding, store)

Document chunking with custom configuration:

>>> from insideLLMs.rag.retrieval import TextChunker, ChunkingConfig
>>>
>>> # Configure chunking for smaller chunks with more overlap
>>> config = ChunkingConfig(
...     chunk_size=500,
...     chunk_overlap=100,
...     separator="\\n"
... )
>>> chunker = TextChunker(config)
>>>
>>> # Chunk a long document
>>> long_text = "First paragraph...\\n\\nSecond paragraph..."
>>> chunks = chunker.chunk(long_text)
>>> print(f"Created {len(chunks)} chunks")

Filtering retrieval by metadata:

>>> from insideLLMs.rag.retrieval import Document, Retriever
>>>
>>> # Add documents with metadata
>>> docs = [
...     Document(content="Python basics", metadata={"topic": "programming"}),
...     Document(content="ML fundamentals", metadata={"topic": "ml"}),
...     Document(content="Deep learning", metadata={"topic": "ml"}),
... ]
>>> retriever.add_documents(docs, chunk=False)
>>>
>>> # Filter by topic
>>> results = retriever.retrieve("fundamentals", filter={"topic": "ml"})

Notes
-----
- For production use, consider using a proper vector database like FAISS,
  Pinecone, Chroma, or Weaviate instead of `InMemoryVectorStore`.
- The `SimpleEmbedding` class is for testing only. Use a real embedding
  model (OpenAI, Sentence Transformers, etc.) for semantic search quality.
- Chunk size and overlap should be tuned based on your document types
  and retrieval requirements.
- The module is designed for extensibility - implement the protocols
  to integrate with your preferred embedding providers and vector stores.

See Also
--------
insideLLMs.models : LLM model implementations for generation
insideLLMs.nlp.tokenization : Tokenization utilities used internally

References
----------
.. [1] Lewis, P. et al. "Retrieval-Augmented Generation for Knowledge-Intensive
       NLP Tasks." NeurIPS 2020.
.. [2] Karpukhin, V. et al. "Dense Passage Retrieval for Open-Domain Question
       Answering." EMNLP 2020.
"""

import hashlib
import math
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from insideLLMs.nlp.tokenization import word_tokenize_regex

if TYPE_CHECKING:
    from insideLLMs.models.base import Model


# =============================================================================
# Document Types
# =============================================================================


@dataclass
class Document:
    """A document with content and metadata for use in retrieval systems.

    The Document class is the fundamental unit of content in the RAG pipeline.
    It encapsulates text content along with optional metadata, a unique identifier,
    and an optional pre-computed embedding vector. Documents can represent anything
    from short text snippets to full articles or book chapters.

    Parameters
    ----------
    content : str
        The text content of the document. This is the primary data that will
        be embedded and searched against.
    metadata : dict[str, Any], optional
        Arbitrary metadata associated with the document. Common fields include
        'source', 'page', 'author', 'timestamp', 'url', etc. Defaults to an
        empty dictionary.
    id : str, optional
        A unique identifier for the document. If not provided, an ID is
        automatically generated from an MD5 hash of the content (first 16 chars).
    embedding : list[float], optional
        A pre-computed embedding vector for the document. If provided, this
        can be used directly instead of re-computing embeddings.

    Attributes
    ----------
    content : str
        The text content of the document.
    metadata : dict[str, Any]
        Metadata dictionary for the document.
    id : str
        Unique identifier (auto-generated if not provided).
    embedding : list[float] or None
        Pre-computed embedding vector, if available.

    Examples
    --------
    Create a simple document with just content:

    >>> doc = Document(content="Python is a programming language.")
    >>> print(doc.id)  # Auto-generated hash-based ID
    a1b2c3d4e5f67890

    Create a document with metadata:

    >>> doc = Document(
    ...     content="Machine learning enables computers to learn from data.",
    ...     metadata={
    ...         "source": "ml_textbook.pdf",
    ...         "page": 42,
    ...         "chapter": "Introduction"
    ...     }
    ... )
    >>> doc.metadata["source"]
    'ml_textbook.pdf'

    Create a document with a custom ID:

    >>> doc = Document(
    ...     content="Neural networks are computational models.",
    ...     id="nn-intro-001"
    ... )
    >>> doc.id
    'nn-intro-001'

    Create a document with a pre-computed embedding:

    >>> embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    >>> doc = Document(
    ...     content="Deep learning uses multiple layers.",
    ...     embedding=embedding
    ... )
    >>> len(doc.embedding)
    5

    Use documents in retrieval:

    >>> from insideLLMs.rag.retrieval import Retriever, InMemoryVectorStore, SimpleEmbedding
    >>> docs = [
    ...     Document(
    ...         content="Python was created by Guido van Rossum.",
    ...         metadata={"topic": "history"}
    ...     ),
    ...     Document(
    ...         content="Python supports multiple programming paradigms.",
    ...         metadata={"topic": "features"}
    ...     ),
    ... ]
    >>> retriever = Retriever(SimpleEmbedding(), InMemoryVectorStore())
    >>> retriever.add_documents(docs, chunk=False)

    Notes
    -----
    - The auto-generated ID uses MD5 hashing, which means identical content
      will produce identical IDs. This is useful for deduplication.
    - Metadata is preserved when documents are chunked (see `TextChunker`),
      with additional chunk-specific metadata added.
    - The embedding field is typically populated by the `Retriever` or
      `VectorStore` during the indexing process.

    See Also
    --------
    RetrievalResult : Container for documents returned from search
    TextChunker : Splits documents into smaller chunks
    Retriever : Manages document embeddings and retrieval
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    embedding: Optional[list[float]] = None

    def __post_init__(self):
        """Initialize auto-generated fields after dataclass creation.

        If no ID was provided, generates one from the MD5 hash of the
        content. This ensures consistent IDs for identical content.
        """
        if self.id is None:
            # Generate ID from content hash
            self.id = hashlib.md5(self.content.encode()).hexdigest()[:16]


@dataclass
class RetrievalResult:
    """A single result from a retrieval operation with relevance scoring.

    RetrievalResult wraps a Document along with its similarity score and
    ranking position from a search query. It is returned by vector store
    search operations and retriever queries.

    Parameters
    ----------
    document : Document
        The retrieved document containing content and metadata.
    score : float
        Similarity score indicating relevance to the query. Higher values
        indicate greater similarity. The scale depends on the similarity
        metric used (e.g., cosine similarity ranges from -1 to 1).
    rank : int, optional
        Position in the search results, 1-indexed (1 = most relevant).
        Defaults to 0 if not explicitly set.

    Attributes
    ----------
    document : Document
        The retrieved document.
    score : float
        Similarity score (higher is more similar).
    rank : int
        Position in results (1-indexed).

    Examples
    --------
    Access retrieval results from a search:

    >>> from insideLLMs.rag.retrieval import Retriever, InMemoryVectorStore, SimpleEmbedding
    >>> retriever = Retriever(SimpleEmbedding(), InMemoryVectorStore())
    >>> retriever.add_texts(["Python is great", "Java is verbose"])
    >>> results = retriever.retrieve("Python programming")
    >>> for result in results:
    ...     print(f"Rank {result.rank}: {result.document.content}")
    ...     print(f"  Score: {result.score:.4f}")
    Rank 1: Python is great
      Score: 0.8542

    Filter results by score threshold:

    >>> high_quality = [r for r in results if r.score > 0.5]
    >>> print(f"Found {len(high_quality)} high-quality results")

    Access document metadata from results:

    >>> for result in results:
    ...     source = result.document.metadata.get("source", "unknown")
    ...     print(f"{result.document.content[:50]}... (from {source})")

    Use results in RAG context formatting:

    >>> context_parts = []
    >>> for r in results[:3]:  # Top 3 results
    ...     context_parts.append(f"[{r.rank}] {r.document.content}")
    >>> context = "\\n\\n".join(context_parts)

    Notes
    -----
    - The similarity score interpretation depends on the vector store and
      similarity metric used. Cosine similarity produces scores in [-1, 1],
      while dot product scores are unbounded.
    - Results are typically returned sorted by score in descending order.
    - The rank field is 1-indexed by convention (rank 1 = best match).

    See Also
    --------
    Document : The underlying document structure
    VectorStore.search : Method that returns RetrievalResult objects
    Retriever.retrieve : Higher-level retrieval interface
    """

    document: Document
    score: float
    rank: int = 0


# =============================================================================
# Embedding Protocol
# =============================================================================


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol defining the interface for embedding models.

    This protocol specifies the required methods that any embedding model
    must implement to be used with the retrieval system. It enables
    duck-typing integration with various embedding providers including
    OpenAI, Sentence Transformers, Cohere, HuggingFace, and custom models.

    The protocol is runtime-checkable, allowing isinstance() checks to
    verify that an object implements the required interface.

    Methods
    -------
    embed(text: str) -> list[float]
        Generate an embedding vector for a single text string.
    embed_batch(texts: list[str]) -> list[list[float]]
        Generate embedding vectors for multiple texts efficiently.

    Examples
    --------
    Implement a custom embedding model:

    >>> class MyEmbedding:
    ...     def embed(self, text: str) -> list[float]:
    ...         # Your embedding logic here
    ...         return [0.0] * 384  # Example 384-dim vector
    ...
    ...     def embed_batch(self, texts: list[str]) -> list[list[float]]:
    ...         return [self.embed(t) for t in texts]
    >>>
    >>> # Verify it implements the protocol
    >>> embedding = MyEmbedding()
    >>> isinstance(embedding, EmbeddingModel)
    True

    Wrap OpenAI embeddings:

    >>> import openai
    >>> class OpenAIEmbedding:
    ...     def __init__(self, model="text-embedding-3-small"):
    ...         self.client = openai.OpenAI()
    ...         self.model = model
    ...
    ...     def embed(self, text: str) -> list[float]:
    ...         response = self.client.embeddings.create(
    ...             input=text,
    ...             model=self.model
    ...         )
    ...         return response.data[0].embedding
    ...
    ...     def embed_batch(self, texts: list[str]) -> list[list[float]]:
    ...         response = self.client.embeddings.create(
    ...             input=texts,
    ...             model=self.model
    ...         )
    ...         return [item.embedding for item in response.data]

    Wrap Sentence Transformers:

    >>> from sentence_transformers import SentenceTransformer
    >>> class STEmbedding:
    ...     def __init__(self, model_name="all-MiniLM-L6-v2"):
    ...         self.model = SentenceTransformer(model_name)
    ...
    ...     def embed(self, text: str) -> list[float]:
    ...         return self.model.encode(text, convert_to_numpy=True).tolist()
    ...
    ...     def embed_batch(self, texts: list[str]) -> list[list[float]]:
    ...         embeddings = self.model.encode(texts, convert_to_numpy=True)
    ...         return embeddings.tolist()

    Notes
    -----
    - Implementations should return consistent-dimension vectors. Mixing
      dimensions will cause errors in vector similarity calculations.
    - The `embed_batch` method should be implemented efficiently, ideally
      using batched API calls or parallel processing rather than sequential
      single-text embedding.
    - Text preprocessing (lowercasing, truncation, etc.) is the responsibility
      of the implementation.

    See Also
    --------
    SimpleEmbedding : A basic TF-IDF-like implementation for testing
    Retriever : Uses EmbeddingModel for document and query embedding
    """

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string.

        This method converts input text into a dense vector representation
        suitable for semantic similarity calculations.

        Parameters
        ----------
        text : str
            The text to embed. Can be a word, sentence, paragraph, or
            longer document depending on the model's capabilities.

        Returns
        -------
        list[float]
            A dense vector representation of the text. The dimension
            depends on the specific embedding model used.

        Examples
        --------
        >>> embedding_model = SimpleEmbedding(dimension=256)
        >>> vector = embedding_model.embed("Hello, world!")
        >>> len(vector)
        256
        >>> all(isinstance(x, float) for x in vector)
        True
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts efficiently.

        This method should be implemented to handle batching efficiently,
        typically by making a single API call or using parallel processing.

        Parameters
        ----------
        texts : list[str]
            A list of text strings to embed. The order of results
            corresponds to the order of inputs.

        Returns
        -------
        list[list[float]]
            A list of embedding vectors, one per input text. All vectors
            should have the same dimension.

        Examples
        --------
        >>> embedding_model = SimpleEmbedding(dimension=128)
        >>> texts = ["First document", "Second document", "Third document"]
        >>> vectors = embedding_model.embed_batch(texts)
        >>> len(vectors)
        3
        >>> all(len(v) == 128 for v in vectors)
        True
        """
        ...


class SimpleEmbedding:
    """Simple TF-IDF-like embedding for testing and lightweight use cases.

    This class provides a basic embedding implementation that does not require
    external dependencies or API calls. It uses a TF-IDF-inspired approach where
    tokens are hashed to fixed positions in the embedding vector, weighted by
    term frequency and inverse document frequency.

    .. warning::
        This embedding is NOT suitable for production semantic search. It lacks
        the semantic understanding of neural embedding models. Use it only for:

        - Testing the RAG pipeline without API dependencies
        - Prototyping and development
        - Keyword-based (not semantic) retrieval
        - Educational purposes

    Parameters
    ----------
    dimension : int, default=256
        The dimension of the output embedding vectors. Higher dimensions
        can reduce hash collisions but increase memory and computation.

    Attributes
    ----------
    dimension : int
        The output embedding dimension.
    _vocab : dict[str, int]
        Internal vocabulary mapping tokens to indices.
    _idf : dict[str, float]
        Internal IDF (inverse document frequency) scores.
    _doc_count : int
        Number of documents processed (for IDF calculation).

    Examples
    --------
    Basic embedding usage:

    >>> embedding = SimpleEmbedding(dimension=128)
    >>> vector = embedding.embed("Python is a programming language")
    >>> len(vector)
    128
    >>> isinstance(vector[0], float)
    True

    Batch embedding updates vocabulary statistics:

    >>> embedding = SimpleEmbedding()
    >>> texts = [
    ...     "Machine learning is powerful",
    ...     "Deep learning uses neural networks",
    ...     "Neural networks have multiple layers"
    ... ]
    >>> vectors = embedding.embed_batch(texts)
    >>> len(vectors)
    3
    >>> embedding._doc_count  # Vocabulary was updated
    3

    Using with a retriever:

    >>> from insideLLMs.rag.retrieval import Retriever, InMemoryVectorStore
    >>> embedding = SimpleEmbedding(dimension=256)
    >>> store = InMemoryVectorStore()
    >>> retriever = Retriever(embedding, store)
    >>> retriever.add_texts(["Python basics", "Java fundamentals"])
    >>> results = retriever.retrieve("Python programming")

    Comparing embedding similarity:

    >>> import math
    >>> embedding = SimpleEmbedding()
    >>> v1 = embedding.embed("Python programming")
    >>> v2 = embedding.embed("Python coding")
    >>> v3 = embedding.embed("Cooking recipes")
    >>> def cosine_sim(a, b):
    ...     dot = sum(x*y for x, y in zip(a, b))
    ...     norm_a = math.sqrt(sum(x*x for x in a))
    ...     norm_b = math.sqrt(sum(x*x for x in b))
    ...     return dot / (norm_a * norm_b) if norm_a and norm_b else 0
    >>> # Similar texts have higher similarity
    >>> sim_12 = cosine_sim(v1, v2)
    >>> sim_13 = cosine_sim(v1, v3)
    >>> sim_12 > sim_13  # Python topics more similar than cooking
    True

    Notes
    -----
    - Vectors are L2-normalized, so cosine similarity equals dot product.
    - Hash collisions may occur when different tokens map to the same
      dimension index. Higher dimensions reduce this.
    - The vocabulary and IDF statistics are updated during `embed_batch`,
      but not during single `embed` calls.
    - For production use, consider OpenAI embeddings, Sentence Transformers,
      or other neural embedding models.

    See Also
    --------
    EmbeddingModel : Protocol that this class implements
    Retriever : Uses embedding models for document indexing and search
    """

    def __init__(self, dimension: int = 256):
        """Initialize the simple TF-IDF-like embedding model.

        Parameters
        ----------
        dimension : int, default=256
            The dimension of the output embedding vectors. Larger dimensions
            reduce hash collisions but increase memory usage.

        Examples
        --------
        >>> embedding = SimpleEmbedding()  # Default 256 dimensions
        >>> embedding.dimension
        256

        >>> embedding = SimpleEmbedding(dimension=512)  # Custom dimension
        >>> embedding.dimension
        512
        """
        self.dimension = dimension
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_count = 0

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words using regex-based tokenization.

        Parameters
        ----------
        text : str
            The text to tokenize.

        Returns
        -------
        list[str]
            List of lowercase tokens extracted from the text.

        Notes
        -----
        Delegates to `insideLLMs.nlp.tokenization.word_tokenize_regex`
        for consistent tokenization across the library.

        Examples
        --------
        >>> embedding = SimpleEmbedding()
        >>> tokens = embedding._tokenize("Hello, World!")
        >>> "hello" in tokens
        True
        """
        return word_tokenize_regex(text)

    def _update_vocab(self, text: str) -> None:
        """Update vocabulary and IDF statistics with a new document.

        This method processes a text document to update the internal
        vocabulary mapping and inverse document frequency counts.

        Parameters
        ----------
        text : str
            The text document to process.

        Notes
        -----
        - Each unique token in the text increments its document frequency.
        - New tokens are added to the vocabulary.
        - The document count is incremented for IDF calculations.

        Examples
        --------
        >>> embedding = SimpleEmbedding()
        >>> embedding._update_vocab("Python is great")
        >>> embedding._doc_count
        1
        >>> "python" in embedding._vocab or "Python" in embedding._vocab
        True
        """
        # NOTE: use a stable ordering to keep `_vocab` / `_idf` insertion order deterministic.
        tokens = sorted(set(self._tokenize(text)))
        self._doc_count += 1

        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
            self._idf[token] = self._idf.get(token, 0) + 1

    def _token_index(self, token: str) -> int:
        """Return a stable bucket index for a token.

        Python's built-in `hash()` is intentionally randomized between
        processes (unless PYTHONHASHSEED is fixed), which breaks determinism
        for any persisted or cross-run computation. Use a cryptographic hash
        instead to make results stable across processes and machines.
        """
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False) % self.dimension

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text using TF-IDF weighting.

        This method creates a fixed-dimension embedding by:
        1. Tokenizing the input text
        2. Computing term frequencies (TF)
        3. Applying IDF weighting based on the corpus statistics
        4. Hashing tokens to fixed positions in the vector
        5. L2-normalizing the result

        Parameters
        ----------
        text : str
            The text to embed.

        Returns
        -------
        list[float]
            A normalized embedding vector of length `self.dimension`.
            Returns a zero vector if the text has no tokens.

        Examples
        --------
        >>> embedding = SimpleEmbedding(dimension=64)
        >>> vector = embedding.embed("Hello world")
        >>> len(vector)
        64

        >>> # Empty or whitespace-only text returns zero vector
        >>> zero_vec = embedding.embed("")
        >>> all(x == 0.0 for x in zero_vec)
        True

        >>> # Vectors are normalized
        >>> import math
        >>> vec = embedding.embed("Test document")
        >>> norm = math.sqrt(sum(x*x for x in vec))
        >>> abs(norm - 1.0) < 0.0001 or norm == 0  # Normalized or zero
        True

        Notes
        -----
        - This method does NOT update the vocabulary or IDF statistics.
          Use `embed_batch` first to build corpus statistics.
        - Without corpus statistics, IDF defaults to a constant value.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.dimension

        # Count token frequencies
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        # Create sparse vector then project to fixed dimension.
        # Iterate in sorted order to make floating point accumulation stable.
        vector = [0.0] * self.dimension
        for token in sorted(tf):
            count = tf[token]
            idx = self._token_index(token)
            # TF-IDF-like weighting
            idf = math.log((self._doc_count + 1) / (self._idf.get(token, 0) + 1)) + 1
            vector[idx] += count * idf

        # Normalize
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts with vocabulary update.

        This method performs a two-pass embedding process:
        1. First pass: Update vocabulary and IDF statistics from all texts
        2. Second pass: Generate embeddings using updated statistics

        This ensures consistent IDF weighting across all documents in the batch.

        Parameters
        ----------
        texts : list[str]
            A list of text strings to embed.

        Returns
        -------
        list[list[float]]
            A list of embedding vectors, one per input text, each of
            length `self.dimension`.

        Examples
        --------
        >>> embedding = SimpleEmbedding()
        >>> texts = ["First document", "Second document", "Third document"]
        >>> vectors = embedding.embed_batch(texts)
        >>> len(vectors)
        3
        >>> all(len(v) == embedding.dimension for v in vectors)
        True

        >>> # Vocabulary is updated during batch embedding
        >>> embedding = SimpleEmbedding()
        >>> _ = embedding.embed_batch(["Python", "Java", "Python"])
        >>> embedding._doc_count
        3

        >>> # Empty list returns empty list
        >>> embedding.embed_batch([])
        []

        Notes
        -----
        - Calling this method multiple times will continue accumulating
          vocabulary and document statistics.
        - For consistent embeddings, embed all documents in a single batch
          or ensure the corpus is stable before embedding queries.
        """
        # First pass: update vocabulary
        for text in texts:
            self._update_vocab(text)

        # Second pass: create embeddings
        return [self.embed(text) for text in texts]


# =============================================================================
# Vector Store Protocol
# =============================================================================


@runtime_checkable
class VectorStore(Protocol):
    """Protocol defining the interface for vector storage backends.

    This protocol specifies the required methods for any vector store
    implementation. It enables integration with various vector databases
    including FAISS, Pinecone, Chroma, Weaviate, Milvus, Qdrant, and others.

    The protocol is runtime-checkable, allowing isinstance() checks to
    verify that an object implements the required interface.

    Methods
    -------
    add(documents, embeddings) -> None
        Add documents with their embedding vectors to the store.
    search(query_embedding, k, filter) -> list[RetrievalResult]
        Search for similar documents by vector similarity.
    delete(ids) -> None
        Remove documents by their IDs.
    clear() -> None
        Remove all documents from the store.

    Examples
    --------
    Implement a custom vector store:

    >>> class MyVectorStore:
    ...     def __init__(self):
    ...         self._data = {}
    ...
    ...     def add(self, documents, embeddings):
    ...         for doc, emb in zip(documents, embeddings):
    ...             self._data[doc.id] = (doc, emb)
    ...
    ...     def search(self, query_embedding, k=5, filter=None):
    ...         # Implement similarity search
    ...         return []
    ...
    ...     def delete(self, ids):
    ...         for id in ids:
    ...             self._data.pop(id, None)
    ...
    ...     def clear(self):
    ...         self._data.clear()
    >>>
    >>> # Verify it implements the protocol
    >>> store = MyVectorStore()
    >>> isinstance(store, VectorStore)
    True

    Wrap FAISS for production use:

    >>> import faiss
    >>> import numpy as np
    >>> class FAISSVectorStore:
    ...     def __init__(self, dimension):
    ...         self.index = faiss.IndexFlatIP(dimension)  # Inner product
    ...         self._documents = {}
    ...         self._id_to_idx = {}
    ...
    ...     def add(self, documents, embeddings):
    ...         vectors = np.array(embeddings, dtype='float32')
    ...         faiss.normalize_L2(vectors)  # For cosine similarity
    ...         start_idx = self.index.ntotal
    ...         self.index.add(vectors)
    ...         for i, doc in enumerate(documents):
    ...             self._documents[doc.id] = doc
    ...             self._id_to_idx[doc.id] = start_idx + i
    ...
    ...     def search(self, query_embedding, k=5, filter=None):
    ...         query = np.array([query_embedding], dtype='float32')
    ...         faiss.normalize_L2(query)
    ...         scores, indices = self.index.search(query, k)
    ...         # Convert indices back to documents and create results
    ...         return []  # Implementation details omitted
    ...
    ...     def delete(self, ids):
    ...         pass  # FAISS doesn't support deletion easily
    ...
    ...     def clear(self):
    ...         self.index.reset()
    ...         self._documents.clear()

    Notes
    -----
    - Implementations should handle the case where document IDs already exist
      (typically by updating/replacing the existing document).
    - The search method should return results sorted by similarity in
      descending order (most similar first).
    - Metadata filtering is optional - implementations may ignore the filter
      parameter if not supported.
    - For production workloads, consider using persistent vector databases
      that support scaling, sharding, and efficient updates.

    See Also
    --------
    InMemoryVectorStore : Simple in-memory implementation for development
    Retriever : Higher-level interface that uses VectorStore
    """

    def add(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        """Add documents with their embedding vectors to the store.

        This method stores documents along with their pre-computed embedding
        vectors for later similarity search.

        Parameters
        ----------
        documents : list[Document]
            List of Document objects to store. Each document should have
            a unique ID.
        embeddings : list[list[float]]
            Corresponding embedding vectors for each document. Must have
            the same length as `documents`.

        Raises
        ------
        ValueError
            If the number of documents does not match the number of embeddings.

        Examples
        --------
        >>> store = InMemoryVectorStore()
        >>> docs = [
        ...     Document(content="Python programming", id="doc1"),
        ...     Document(content="Java development", id="doc2"),
        ... ]
        >>> embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        >>> store.add(docs, embeddings)
        >>> len(store)
        2
        """
        ...

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[RetrievalResult]:
        """Search for documents similar to a query vector.

        This method performs similarity search using the query embedding
        and returns the top-k most similar documents.

        Parameters
        ----------
        query_embedding : list[float]
            The query vector to search with. Should have the same dimension
            as the stored document embeddings.
        k : int, default=5
            The number of results to return.
        filter : dict[str, Any], optional
            Metadata filter to apply. Only documents matching all key-value
            pairs in the filter will be considered.

        Returns
        -------
        list[RetrievalResult]
            List of RetrievalResult objects sorted by similarity score
            in descending order (most similar first).

        Examples
        --------
        >>> store = InMemoryVectorStore()
        >>> # ... add documents ...
        >>> results = store.search([0.1, 0.2, 0.3], k=3)
        >>> for r in results:
        ...     print(f"{r.rank}: {r.document.content} (score: {r.score:.3f})")

        >>> # With metadata filter
        >>> results = store.search(
        ...     query_embedding=[0.1, 0.2, 0.3],
        ...     k=5,
        ...     filter={"category": "programming"}
        ... )
        """
        ...

    def delete(self, ids: list[str]) -> None:
        """Remove documents from the store by their IDs.

        Parameters
        ----------
        ids : list[str]
            List of document IDs to delete. Non-existent IDs are silently
            ignored.

        Examples
        --------
        >>> store = InMemoryVectorStore()
        >>> store.add([Document(content="test", id="doc1")], [[0.1, 0.2]])
        >>> store.delete(["doc1"])
        >>> len(store)
        0
        """
        ...

    def clear(self) -> None:
        """Remove all documents from the store.

        This method completely empties the vector store, removing all
        documents and their embeddings.

        Examples
        --------
        >>> store = InMemoryVectorStore()
        >>> store.add([Document(content="test")], [[0.1, 0.2]])
        >>> store.clear()
        >>> len(store)
        0
        """
        ...


class InMemoryVectorStore:
    """Simple in-memory vector store using cosine similarity.

    This class provides a lightweight vector storage implementation that keeps
    all documents and embeddings in memory using Python dictionaries. It uses
    cosine similarity for vector comparison and supports basic metadata filtering.

    .. note::
        This implementation is suitable for:

        - Development and testing
        - Small datasets (< 10,000 documents)
        - Prototyping RAG pipelines
        - Environments without vector database dependencies

        For production use with larger datasets, consider using FAISS, Pinecone,
        Chroma, Weaviate, Milvus, or Qdrant.

    Attributes
    ----------
    _documents : dict[str, Document]
        Internal storage mapping document IDs to Document objects.
    _embeddings : dict[str, list[float]]
        Internal storage mapping document IDs to embedding vectors.

    Examples
    --------
    Basic usage:

    >>> store = InMemoryVectorStore()
    >>> docs = [
    ...     Document(content="Python programming language", id="doc1"),
    ...     Document(content="Java programming language", id="doc2"),
    ...     Document(content="Cooking recipes for beginners", id="doc3"),
    ... ]
    >>> embeddings = [
    ...     [0.9, 0.1, 0.0],  # Python
    ...     [0.8, 0.2, 0.0],  # Java (similar to Python)
    ...     [0.0, 0.1, 0.9],  # Cooking (different topic)
    ... ]
    >>> store.add(docs, embeddings)
    >>> len(store)
    3

    Searching for similar documents:

    >>> query = [0.85, 0.15, 0.0]  # Query about programming
    >>> results = store.search(query, k=2)
    >>> for r in results:
    ...     print(f"Rank {r.rank}: {r.document.content}")
    Rank 1: Python programming language
    Rank 2: Java programming language

    Using metadata filters:

    >>> store = InMemoryVectorStore()
    >>> docs = [
    ...     Document(content="Python basics", metadata={"level": "beginner"}),
    ...     Document(content="Python advanced", metadata={"level": "advanced"}),
    ...     Document(content="Java basics", metadata={"level": "beginner"}),
    ... ]
    >>> embeddings = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]
    >>> store.add(docs, embeddings)
    >>> # Only search beginner-level documents
    >>> results = store.search([0.9, 0.1], k=5, filter={"level": "beginner"})
    >>> len(results)
    2

    Deleting and clearing:

    >>> store = InMemoryVectorStore()
    >>> store.add([Document(content="test1", id="1")], [[0.1]])
    >>> store.add([Document(content="test2", id="2")], [[0.2]])
    >>> len(store)
    2
    >>> store.delete(["1"])
    >>> len(store)
    1
    >>> store.clear()
    >>> len(store)
    0

    Using with the Retriever:

    >>> from insideLLMs.rag.retrieval import Retriever, SimpleEmbedding
    >>> embedding = SimpleEmbedding()
    >>> store = InMemoryVectorStore()
    >>> retriever = Retriever(embedding, store)
    >>> retriever.add_texts(["Python tutorial", "Java guide", "Cooking tips"])
    >>> results = retriever.retrieve("programming language")

    Notes
    -----
    - Search complexity is O(n) where n is the number of documents.
      This becomes slow for large datasets.
    - All data is lost when the process terminates. Use a persistent
      vector store for production.
    - Metadata filtering is exact-match only; no range or partial matching.
    - Thread-safety is not guaranteed for concurrent modifications.

    See Also
    --------
    VectorStore : Protocol that this class implements
    Retriever : Higher-level interface for document retrieval
    """

    def __init__(self):
        """Initialize an empty in-memory vector store.

        Creates empty internal dictionaries for storing documents and
        their corresponding embedding vectors.

        Examples
        --------
        >>> store = InMemoryVectorStore()
        >>> len(store)
        0
        """
        self._documents: dict[str, Document] = {}
        self._embeddings: dict[str, list[float]] = {}

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Cosine similarity measures the cosine of the angle between two vectors,
        providing a similarity score in the range [-1, 1] where 1 indicates
        identical direction, 0 indicates orthogonality, and -1 indicates
        opposite directions.

        Parameters
        ----------
        a : list[float]
            First vector.
        b : list[float]
            Second vector. Must have the same dimension as `a`.

        Returns
        -------
        float
            Cosine similarity score in range [-1, 1]. Returns 0.0 if either
            vector has zero magnitude.

        Examples
        --------
        >>> InMemoryVectorStore._cosine_similarity([1, 0], [1, 0])
        1.0
        >>> InMemoryVectorStore._cosine_similarity([1, 0], [0, 1])
        0.0
        >>> InMemoryVectorStore._cosine_similarity([1, 0], [-1, 0])
        -1.0
        >>> InMemoryVectorStore._cosine_similarity([0, 0], [1, 1])
        0.0
        """
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def add(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        """Add documents with their embedding vectors to the store.

        This method stores documents and their embeddings in internal
        dictionaries, keyed by document ID. If a document with the same
        ID already exists, it will be replaced.

        Parameters
        ----------
        documents : list[Document]
            List of Document objects to store.
        embeddings : list[list[float]]
            Corresponding embedding vectors, one per document.

        Raises
        ------
        ValueError
            If the number of documents does not match the number of embeddings.

        Examples
        --------
        >>> store = InMemoryVectorStore()
        >>> docs = [Document(content="Hello", id="d1")]
        >>> embeddings = [[0.1, 0.2, 0.3]]
        >>> store.add(docs, embeddings)
        >>> len(store)
        1

        >>> # Adding more documents
        >>> store.add(
        ...     [Document(content="World", id="d2")],
        ...     [[0.4, 0.5, 0.6]]
        ... )
        >>> len(store)
        2

        >>> # Replacing existing document
        >>> store.add(
        ...     [Document(content="Hello Updated", id="d1")],
        ...     [[0.7, 0.8, 0.9]]
        ... )
        >>> store._documents["d1"].content
        'Hello Updated'
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
            self._documents[doc.id] = doc
            self._embeddings[doc.id] = emb

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[RetrievalResult]:
        """Search for documents similar to a query vector using cosine similarity.

        Performs a linear scan of all documents, computing cosine similarity
        with the query vector and returning the top-k results. Optionally
        filters documents by metadata before ranking.

        Parameters
        ----------
        query_embedding : list[float]
            The query vector to search with.
        k : int, default=5
            Maximum number of results to return.
        filter : dict[str, Any], optional
            Metadata filter requiring exact match on all specified key-value
            pairs. Only documents matching ALL filter conditions are considered.

        Returns
        -------
        list[RetrievalResult]
            List of up to `k` RetrievalResult objects, sorted by similarity
            score in descending order. Each result includes the document,
            similarity score, and rank (1-indexed).

        Examples
        --------
        >>> store = InMemoryVectorStore()
        >>> store.add(
        ...     [
        ...         Document(content="A", metadata={"cat": "x"}),
        ...         Document(content="B", metadata={"cat": "y"}),
        ...     ],
        ...     [[1.0, 0.0], [0.0, 1.0]]
        ... )
        >>> results = store.search([1.0, 0.0], k=2)
        >>> results[0].document.content
        'A'
        >>> results[0].score
        1.0

        >>> # With filter
        >>> results = store.search([1.0, 0.0], filter={"cat": "y"})
        >>> len(results)
        1
        >>> results[0].document.content
        'B'
        """
        results = []

        for doc_id, doc in self._documents.items():
            # Apply filter if provided
            if filter:
                match = all(doc.metadata.get(key) == value for key, value in filter.items())
                if not match:
                    continue

            emb = self._embeddings[doc_id]
            score = self._cosine_similarity(query_embedding, emb)
            results.append((doc, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        return [
            RetrievalResult(document=doc, score=score, rank=i + 1)
            for i, (doc, score) in enumerate(results[:k])
        ]

    def delete(self, ids: list[str]) -> None:
        """Remove documents from the store by their IDs.

        Parameters
        ----------
        ids : list[str]
            List of document IDs to delete. Non-existent IDs are silently
            ignored.

        Examples
        --------
        >>> store = InMemoryVectorStore()
        >>> store.add(
        ...     [Document(content="A", id="1"), Document(content="B", id="2")],
        ...     [[0.1], [0.2]]
        ... )
        >>> len(store)
        2
        >>> store.delete(["1"])
        >>> len(store)
        1
        >>> store.delete(["nonexistent"])  # No error
        >>> len(store)
        1
        """
        for doc_id in ids:
            self._documents.pop(doc_id, None)
            self._embeddings.pop(doc_id, None)

    def clear(self) -> None:
        """Remove all documents and embeddings from the store.

        Examples
        --------
        >>> store = InMemoryVectorStore()
        >>> store.add([Document(content="test")], [[0.1, 0.2]])
        >>> len(store)
        1
        >>> store.clear()
        >>> len(store)
        0
        """
        self._documents.clear()
        self._embeddings.clear()

    def __len__(self) -> int:
        """Return the number of documents in the store.

        Returns
        -------
        int
            Number of documents currently stored.

        Examples
        --------
        >>> store = InMemoryVectorStore()
        >>> len(store)
        0
        >>> store.add([Document(content="test")], [[0.1]])
        >>> len(store)
        1
        """
        return len(self._documents)


# =============================================================================
# Document Chunking
# =============================================================================


@dataclass
class ChunkingConfig:
    """Configuration settings for document chunking operations.

    This dataclass holds all configuration parameters for the TextChunker.
    It allows customization of chunk sizes, overlap, separators, and the
    length measurement function.

    Parameters
    ----------
    chunk_size : int, default=1000
        Target maximum size of each chunk in characters (or tokens, depending
        on the length_function). Chunks may be slightly smaller but will not
        exceed this size.
    chunk_overlap : int, default=200
        Number of characters (or tokens) to overlap between consecutive chunks.
        Overlap helps maintain context across chunk boundaries, which is
        important for retrieval quality.
    separator : str, default="\\n"
        The primary separator used for initial text splitting. The chunker
        will try to split on this separator first before falling back to
        other separators.
    length_function : Callable[[str], int], default=len
        Function used to measure text length. The default `len` counts
        characters. For token-based chunking, provide a tokenizer's
        length function.

    Attributes
    ----------
    chunk_size : int
        Target maximum chunk size.
    chunk_overlap : int
        Overlap size between chunks.
    separator : str
        Primary text separator.
    length_function : Callable[[str], int]
        Text length measurement function.

    Examples
    --------
    Default configuration (1000 chars, 200 overlap):

    >>> config = ChunkingConfig()
    >>> config.chunk_size
    1000
    >>> config.chunk_overlap
    200

    Custom configuration for smaller chunks:

    >>> config = ChunkingConfig(
    ...     chunk_size=500,
    ...     chunk_overlap=50,
    ...     separator="\\n\\n"  # Split on paragraphs
    ... )

    Token-based chunking (requires tiktoken or similar):

    >>> import tiktoken
    >>> enc = tiktoken.get_encoding("cl100k_base")
    >>> config = ChunkingConfig(
    ...     chunk_size=512,  # 512 tokens
    ...     chunk_overlap=64,  # 64 token overlap
    ...     length_function=lambda x: len(enc.encode(x))
    ... )

    Using with TextChunker:

    >>> config = ChunkingConfig(chunk_size=300, chunk_overlap=30)
    >>> chunker = TextChunker(config)
    >>> text = "A long document..." * 100
    >>> chunks = chunker.chunk(text)

    Notes
    -----
    - chunk_overlap should be less than chunk_size to ensure progress.
    - Larger overlap improves context continuity but increases storage.
    - Token-based length functions are recommended when working with
      LLMs that have token limits.
    - The separator parameter is mostly informational; TextChunker uses
      its own separator hierarchy.

    See Also
    --------
    TextChunker : Uses this configuration for chunking operations
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: str = "\n"
    length_function: Callable[[str], int] = len


class TextChunker:
    """Splits text into overlapping chunks using recursive boundary detection.

    TextChunker implements a sophisticated chunking strategy that attempts to
    split text at natural boundaries (paragraphs, sentences, words) while
    respecting maximum chunk size limits. It creates overlapping chunks to
    preserve context across boundaries.

    The chunking algorithm:
    1. Tries to split on the highest-priority separator (paragraphs)
    2. If chunks are still too large, recursively tries lower-priority separators
    3. Combines small splits to fill chunks up to the size limit
    4. Adds overlap from the end of each chunk to the beginning of the next

    Parameters
    ----------
    config : ChunkingConfig, optional
        Configuration specifying chunk size, overlap, and length function.
        Uses default ChunkingConfig if not provided.

    Attributes
    ----------
    config : ChunkingConfig
        The chunking configuration.
    SEPARATORS : list[str]
        Class-level list of separators in order of preference, from
        paragraph breaks down to individual characters.

    Examples
    --------
    Basic text chunking:

    >>> chunker = TextChunker()
    >>> text = "First paragraph.\\n\\nSecond paragraph.\\n\\nThird paragraph."
    >>> chunks = chunker.chunk(text)
    >>> len(chunks) >= 1
    True

    Custom chunk size for shorter chunks:

    >>> from insideLLMs.rag.retrieval import ChunkingConfig
    >>> config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
    >>> chunker = TextChunker(config)
    >>> long_text = "This is a sentence. " * 50
    >>> chunks = chunker.chunk(long_text)
    >>> all(len(c) <= 100 for c in chunks)
    True

    Chunking documents with metadata preservation:

    >>> from insideLLMs.rag.retrieval import Document
    >>> chunker = TextChunker(ChunkingConfig(chunk_size=50, chunk_overlap=10))
    >>> doc = Document(
    ...     content="A " * 100,  # Long content
    ...     metadata={"source": "test.txt", "author": "Jane"}
    ... )
    >>> chunk_docs = chunker.chunk_documents([doc])
    >>> # Metadata is preserved
    >>> all(d.metadata.get("source") == "test.txt" for d in chunk_docs)
    True
    >>> # Chunk metadata is added
    >>> all("chunk_index" in d.metadata for d in chunk_docs)
    True

    Token-based chunking for LLM compatibility:

    >>> import tiktoken
    >>> enc = tiktoken.get_encoding("cl100k_base")
    >>> config = ChunkingConfig(
    ...     chunk_size=100,  # 100 tokens max
    ...     chunk_overlap=10,
    ...     length_function=lambda x: len(enc.encode(x))
    ... )
    >>> chunker = TextChunker(config)
    >>> text = "Hello world. " * 200
    >>> chunks = chunker.chunk(text)
    >>> # Verify token counts
    >>> all(len(enc.encode(c)) <= 100 for c in chunks)
    True

    Processing multiple documents:

    >>> docs = [
    ...     Document(content="Doc 1 " * 50, metadata={"id": 1}),
    ...     Document(content="Doc 2 " * 50, metadata={"id": 2}),
    ... ]
    >>> chunker = TextChunker(ChunkingConfig(chunk_size=100))
    >>> all_chunks = chunker.chunk_documents(docs)
    >>> # Each chunk knows its source
    >>> all("source_doc_id" in c.metadata for c in all_chunks)
    True

    Notes
    -----
    - The chunker tries to respect natural text boundaries. A chunk will only
      be split mid-word or mid-character as a last resort.
    - Overlap is taken from the END of the previous chunk, not the beginning.
    - Empty or whitespace-only inputs return an empty list.
    - Very long words or character sequences may result in chunks slightly
      exceeding the target size.
    - For best results with RAG, use chunk sizes between 200-1000 characters
      or 100-500 tokens, with 10-20% overlap.

    See Also
    --------
    ChunkingConfig : Configuration dataclass for chunking parameters
    Document : Document class that can be chunked
    Retriever : Uses TextChunker for automatic document chunking
    """

    # Separators in order of preference
    SEPARATORS = [
        "\n\n",  # Paragraphs
        "\n",  # Lines
        ". ",  # Sentences
        "! ",
        "? ",
        "; ",
        ", ",  # Clauses
        " ",  # Words
        "",  # Characters
    ]

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize the text chunker with configuration.

        Parameters
        ----------
        config : ChunkingConfig, optional
            Configuration for chunking behavior. If not provided, uses
            default ChunkingConfig (1000 char chunks, 200 char overlap).

        Examples
        --------
        >>> chunker = TextChunker()  # Default config
        >>> chunker.config.chunk_size
        1000

        >>> custom_config = ChunkingConfig(chunk_size=500)
        >>> chunker = TextChunker(custom_config)
        >>> chunker.config.chunk_size
        500
        """
        self.config = config or ChunkingConfig()

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using a hierarchy of separators.

        This method implements the core recursive splitting algorithm. It
        tries each separator in order, splitting on the current separator
        and recursively processing any chunks that are still too large.

        Parameters
        ----------
        text : str
            The text to split.
        separators : list[str]
            List of separators to try, in order of preference.

        Returns
        -------
        list[str]
            List of text segments, each within the chunk size limit
            (or as close as possible given the separators).

        Examples
        --------
        >>> chunker = TextChunker(ChunkingConfig(chunk_size=20))
        >>> result = chunker._split_text("Hello world. Goodbye.", chunker.SEPARATORS)
        >>> len(result) >= 1
        True
        """
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Split into individual characters
            return list(text)

        if separator not in text:
            return self._split_text(text, remaining_separators)

        splits = text.split(separator)

        # Recombine small splits
        result = []
        current = ""

        for split in splits:
            test_text = current + separator + split if current else split

            if self.config.length_function(test_text) <= self.config.chunk_size:
                current = test_text
            else:
                if current:
                    result.append(current)

                if self.config.length_function(split) > self.config.chunk_size:
                    # Recursively split large chunks
                    result.extend(self._split_text(split, remaining_separators))
                    current = ""
                else:
                    current = split

        if current:
            result.append(current)

        return result

    def chunk(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        This is the main method for chunking a single text string. It first
        performs recursive splitting to get segments within size limits,
        then combines them with overlap.

        Parameters
        ----------
        text : str
            The text to chunk.

        Returns
        -------
        list[str]
            List of text chunks. Each chunk is stripped of leading/trailing
            whitespace. Returns an empty list for empty or None input.

        Examples
        --------
        >>> chunker = TextChunker(ChunkingConfig(chunk_size=50, chunk_overlap=10))
        >>> text = "The quick brown fox. " * 10
        >>> chunks = chunker.chunk(text)
        >>> len(chunks) > 1
        True
        >>> all(chunk.strip() == chunk for chunk in chunks)  # All trimmed
        True

        >>> # Empty input returns empty list
        >>> chunker.chunk("")
        []
        >>> chunker.chunk("   ")
        []

        >>> # Short text that fits in one chunk
        >>> chunker = TextChunker(ChunkingConfig(chunk_size=1000))
        >>> chunks = chunker.chunk("Short text")
        >>> len(chunks)
        1
        >>> chunks[0]
        'Short text'
        """
        if not text:
            return []

        # Get initial splits
        splits = self._split_text(text, self.SEPARATORS)

        # Create overlapping chunks
        chunks = []
        current_chunk = ""

        for split in splits:
            split = split.strip()
            if not split:
                continue

            test_chunk = current_chunk + " " + split if current_chunk else split

            if self.config.length_function(test_chunk) <= self.config.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                    # Add overlap from end of current chunk
                    if self.config.chunk_overlap > 0:
                        overlap_text = current_chunk[-self.config.chunk_overlap :]
                        current_chunk = overlap_text.strip() + " " + split
                    else:
                        current_chunk = split
                else:
                    current_chunk = split

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_documents(
        self,
        documents: list[Document],
        preserve_metadata: bool = True,
    ) -> list[Document]:
        """Chunk multiple documents, creating new Document objects for each chunk.

        This method processes a list of documents, splitting each into chunks
        and creating new Document objects. Metadata can be preserved and is
        augmented with chunk-specific information.

        Parameters
        ----------
        documents : list[Document]
            List of Document objects to chunk.
        preserve_metadata : bool, default=True
            If True, copies all metadata from the source document to each
            chunk. If False, chunks only contain chunk-specific metadata.

        Returns
        -------
        list[Document]
            List of new Document objects, one per chunk. Each chunk document
            has metadata including:

            - ``source_doc_id``: ID of the original document
            - ``chunk_index``: 0-based index of this chunk
            - ``total_chunks``: Total number of chunks from the source
            - Plus all original metadata if preserve_metadata=True

        Examples
        --------
        >>> config = ChunkingConfig(chunk_size=50, chunk_overlap=10)
        >>> chunker = TextChunker(config)
        >>> doc = Document(
        ...     content="Word " * 50,  # 250 chars
        ...     metadata={"source": "test.txt", "author": "Alice"}
        ... )
        >>> chunks = chunker.chunk_documents([doc])
        >>> len(chunks) > 1
        True

        >>> # Check metadata preservation
        >>> chunks[0].metadata["source"]
        'test.txt'
        >>> chunks[0].metadata["chunk_index"]
        0
        >>> "total_chunks" in chunks[0].metadata
        True

        >>> # Without metadata preservation
        >>> chunks_no_meta = chunker.chunk_documents([doc], preserve_metadata=False)
        >>> "source" in chunks_no_meta[0].metadata
        False
        >>> "chunk_index" in chunks_no_meta[0].metadata
        True

        >>> # Processing multiple documents
        >>> docs = [
        ...     Document(content="A " * 100, id="doc-a"),
        ...     Document(content="B " * 100, id="doc-b"),
        ... ]
        >>> all_chunks = chunker.chunk_documents(docs)
        >>> # Chunks track their source
        >>> doc_a_chunks = [c for c in all_chunks if c.metadata["source_doc_id"] == "doc-a"]
        >>> len(doc_a_chunks) > 0
        True
        """
        result = []

        for doc in documents:
            chunks = self.chunk(doc.content)

            for i, chunk_content in enumerate(chunks):
                metadata = dict(doc.metadata) if preserve_metadata else {}
                metadata["source_doc_id"] = doc.id
                metadata["chunk_index"] = i
                metadata["total_chunks"] = len(chunks)

                chunk_doc = Document(
                    content=chunk_content,
                    metadata=metadata,
                )
                result.append(chunk_doc)

        return result


# =============================================================================
# Retriever
# =============================================================================


class Retriever:
    """High-level interface combining embedding, chunking, and vector search.

    The Retriever class provides a unified interface for document indexing and
    retrieval. It orchestrates the embedding model, vector store, and text
    chunker to handle the complete retrieval workflow:

    1. **Ingestion**: Documents or texts are chunked, embedded, and stored
    2. **Retrieval**: Queries are embedded and used for similarity search

    This class is the primary interface for retrieval operations and is used
    internally by RAGChain.

    Parameters
    ----------
    embedding_model : EmbeddingModel
        Model for creating embedding vectors. Must implement the EmbeddingModel
        protocol with `embed` and `embed_batch` methods.
    vector_store : VectorStore
        Storage backend for documents and their embeddings. Must implement
        the VectorStore protocol.
    chunker : TextChunker, optional
        Text chunker for splitting documents. If not provided, uses a
        default TextChunker with standard settings.

    Attributes
    ----------
    embedding_model : EmbeddingModel
        The configured embedding model.
    vector_store : VectorStore
        The configured vector store.
    chunker : TextChunker
        The configured text chunker.

    Examples
    --------
    Basic retriever setup and usage:

    >>> from insideLLMs.rag.retrieval import Retriever, SimpleEmbedding, InMemoryVectorStore
    >>> embedding = SimpleEmbedding()
    >>> store = InMemoryVectorStore()
    >>> retriever = Retriever(embedding, store)
    >>>
    >>> # Add documents
    >>> retriever.add_texts([
    ...     "Python is a versatile programming language.",
    ...     "Machine learning requires large datasets.",
    ...     "Neural networks can learn complex patterns."
    ... ])
    >>>
    >>> # Retrieve relevant documents
    >>> results = retriever.retrieve("programming languages")
    >>> for r in results:
    ...     print(f"Score {r.score:.3f}: {r.document.content}")

    With custom chunking:

    >>> from insideLLMs.rag.retrieval import TextChunker, ChunkingConfig
    >>> config = ChunkingConfig(chunk_size=200, chunk_overlap=50)
    >>> chunker = TextChunker(config)
    >>> retriever = Retriever(embedding, store, chunker=chunker)

    Adding documents with metadata:

    >>> retriever.add_texts(
    ...     texts=["Document 1 content", "Document 2 content"],
    ...     metadatas=[
    ...         {"source": "file1.txt", "category": "tech"},
    ...         {"source": "file2.txt", "category": "science"}
    ...     ]
    ... )
    >>>
    >>> # Filter by metadata
    >>> results = retriever.retrieve("content", filter={"category": "tech"})

    Working with Document objects:

    >>> from insideLLMs.rag.retrieval import Document
    >>> docs = [
    ...     Document(
    ...         content="Long document that will be chunked...",
    ...         metadata={"source": "article.pdf"}
    ...     ),
    ...     Document(
    ...         content="Another document...",
    ...         metadata={"source": "report.pdf"}
    ...     )
    ... ]
    >>> ids = retriever.add_documents(docs)
    >>> print(f"Added {len(ids)} chunks")

    Skipping chunking for pre-processed content:

    >>> # Add short texts without chunking
    >>> retriever.add_texts(
    ...     ["Short text 1", "Short text 2"],
    ...     chunk=False
    ... )

    Managing documents:

    >>> # Get IDs when adding
    >>> ids = retriever.add_texts(["Content to later delete"])
    >>> # Delete by ID
    >>> retriever.delete(ids)
    >>> # Clear all documents
    >>> retriever.clear()

    Notes
    -----
    - The retriever automatically handles embedding creation for both
      documents (via `embed_batch`) and queries (via `embed`).
    - When chunking is enabled (default), the returned document IDs are
      for the chunk documents, not the original documents.
    - Metadata from original documents is preserved in chunks with
      additional chunk-tracking metadata added.
    - For best performance with large document sets, use batch operations
      (`add_texts`, `add_documents`) rather than individual additions.

    See Also
    --------
    RAGChain : Higher-level interface that adds LLM generation
    EmbeddingModel : Protocol for embedding models
    VectorStore : Protocol for vector storage backends
    TextChunker : Document chunking implementation
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        chunker: Optional[TextChunker] = None,
    ):
        """Initialize the retriever with embedding model, vector store, and chunker.

        Parameters
        ----------
        embedding_model : EmbeddingModel
            Model for creating embedding vectors.
        vector_store : VectorStore
            Backend for storing documents and embeddings.
        chunker : TextChunker, optional
            Chunker for splitting documents. Uses default TextChunker if
            not provided.

        Examples
        --------
        >>> embedding = SimpleEmbedding()
        >>> store = InMemoryVectorStore()
        >>> retriever = Retriever(embedding, store)

        >>> # With custom chunker
        >>> chunker = TextChunker(ChunkingConfig(chunk_size=500))
        >>> retriever = Retriever(embedding, store, chunker=chunker)
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunker = chunker or TextChunker()

    def add_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        chunk: bool = True,
    ) -> list[str]:
        """Add text strings to the retriever for indexing.

        This method creates Document objects from the provided texts and
        metadata, then delegates to `add_documents` for processing.

        Parameters
        ----------
        texts : list[str]
            List of text strings to add.
        metadatas : list[dict[str, Any]], optional
            List of metadata dictionaries, one per text. If not provided,
            each text gets an empty metadata dictionary.
        chunk : bool, default=True
            Whether to split texts into chunks before embedding. Set to
            False for pre-chunked or short texts.

        Returns
        -------
        list[str]
            List of document IDs that were added. If chunking is enabled,
            returns IDs of the chunk documents.

        Examples
        --------
        >>> retriever = Retriever(SimpleEmbedding(), InMemoryVectorStore())
        >>> ids = retriever.add_texts([
        ...     "First document about Python.",
        ...     "Second document about Java."
        ... ])
        >>> len(ids) >= 2
        True

        >>> # With metadata
        >>> ids = retriever.add_texts(
        ...     texts=["Technical content", "Business content"],
        ...     metadatas=[{"type": "tech"}, {"type": "business"}]
        ... )

        >>> # Without chunking (for short texts)
        >>> ids = retriever.add_texts(
        ...     ["Short fact 1", "Short fact 2"],
        ...     chunk=False
        ... )
        >>> len(ids)
        2
        """
        # Create documents
        metadatas = metadatas or [{}] * len(texts)
        documents = [Document(content=text, metadata=meta) for text, meta in zip(texts, metadatas)]

        return self.add_documents(documents, chunk=chunk)

    def add_documents(
        self,
        documents: list[Document],
        chunk: bool = True,
    ) -> list[str]:
        """Add Document objects to the retriever for indexing.

        This method handles the complete ingestion workflow:
        1. Optionally chunks documents using the configured chunker
        2. Creates embeddings for all documents/chunks
        3. Stores documents and embeddings in the vector store

        Parameters
        ----------
        documents : list[Document]
            List of Document objects to add.
        chunk : bool, default=True
            Whether to split documents into chunks before embedding.
            Set to False if documents are already appropriately sized.

        Returns
        -------
        list[str]
            List of document IDs that were added. If chunking is enabled,
            returns IDs of the chunk documents, not the original documents.

        Examples
        --------
        >>> from insideLLMs.rag.retrieval import Document
        >>> retriever = Retriever(SimpleEmbedding(), InMemoryVectorStore())
        >>> docs = [
        ...     Document(content="Long content..." * 50, metadata={"source": "a.txt"}),
        ...     Document(content="More content..." * 50, metadata={"source": "b.txt"})
        ... ]
        >>> ids = retriever.add_documents(docs)
        >>> print(f"Created {len(ids)} chunks from 2 documents")

        >>> # Without chunking
        >>> short_docs = [Document(content="Short", id="custom-id")]
        >>> ids = retriever.add_documents(short_docs, chunk=False)
        >>> ids
        ['custom-id']

        >>> # Empty input returns empty list
        >>> retriever.add_documents([])
        []
        """
        if chunk:
            documents = self.chunker.chunk_documents(documents)

        if not documents:
            return []

        # Create embeddings
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.embed_batch(texts)

        # Add to vector store
        self.vector_store.add(documents, embeddings)

        return [doc.id for doc in documents]

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents relevant to a query using semantic search.

        This method embeds the query using the embedding model and performs
        similarity search in the vector store.

        Parameters
        ----------
        query : str
            The search query text.
        k : int, default=5
            Maximum number of results to return.
        filter : dict[str, Any], optional
            Metadata filter to apply. Only documents matching all key-value
            pairs will be considered.

        Returns
        -------
        list[RetrievalResult]
            List of RetrievalResult objects sorted by relevance (highest
            score first). Each result contains the document, similarity
            score, and rank.

        Examples
        --------
        >>> retriever = Retriever(SimpleEmbedding(), InMemoryVectorStore())
        >>> retriever.add_texts([
        ...     "Python is a programming language.",
        ...     "Cooking is an art form.",
        ...     "Python can be used for data science."
        ... ])
        >>> results = retriever.retrieve("programming")
        >>> len(results) <= 5
        True
        >>> results[0].score > results[-1].score  # Sorted by relevance
        True

        >>> # Get more results
        >>> results = retriever.retrieve("Python", k=10)

        >>> # Filter by metadata
        >>> retriever.add_texts(
        ...     ["Tech doc", "Science doc"],
        ...     metadatas=[{"category": "tech"}, {"category": "science"}]
        ... )
        >>> tech_results = retriever.retrieve(
        ...     "document",
        ...     filter={"category": "tech"}
        ... )

        >>> # Access result details
        >>> for r in results:
        ...     print(f"Rank {r.rank}: score={r.score:.3f}")
        ...     print(f"  Content: {r.document.content[:50]}...")
        """
        query_embedding = self.embedding_model.embed(query)
        return self.vector_store.search(query_embedding, k=k, filter=filter)

    def delete(self, ids: list[str]) -> None:
        """Delete documents from the retriever by their IDs.

        Parameters
        ----------
        ids : list[str]
            List of document IDs to delete. Non-existent IDs are silently
            ignored.

        Examples
        --------
        >>> retriever = Retriever(SimpleEmbedding(), InMemoryVectorStore())
        >>> ids = retriever.add_texts(["Temporary document"])
        >>> retriever.delete(ids)

        >>> # Multiple deletions
        >>> ids = retriever.add_texts(["A", "B", "C"])
        >>> retriever.delete(ids[:2])  # Delete first two
        """
        self.vector_store.delete(ids)

    def clear(self) -> None:
        """Remove all documents from the retriever.

        This method completely empties the vector store, removing all
        indexed documents and their embeddings.

        Examples
        --------
        >>> retriever = Retriever(SimpleEmbedding(), InMemoryVectorStore())
        >>> retriever.add_texts(["Doc 1", "Doc 2", "Doc 3"])
        >>> retriever.clear()
        >>> results = retriever.retrieve("anything")
        >>> len(results)
        0
        """
        self.vector_store.clear()


# =============================================================================
# RAG Chain
# =============================================================================


DEFAULT_RAG_PROMPT = """Answer the question based on the following context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGResponse:
    """Response object containing the result of a RAG query.

    RAGResponse encapsulates all outputs from a RAG query, including the
    generated answer, the source documents used for context, and the
    full prompt sent to the language model. This allows for transparency,
    debugging, and citation of sources.

    Parameters
    ----------
    answer : str
        The generated answer from the language model.
    sources : list[RetrievalResult]
        List of retrieved documents that were used as context for
        generation. Each result includes the document, similarity score,
        and rank.
    prompt : str
        The full prompt that was sent to the language model, including
        the formatted context and question.

    Attributes
    ----------
    answer : str
        The generated answer text.
    sources : list[RetrievalResult]
        Retrieved documents used for context.
    prompt : str
        The complete prompt sent to the model.

    Examples
    --------
    Access response components:

    >>> # Assuming rag is a configured RAGChain
    >>> response = rag.query("What is Python?")
    >>> print(response.answer)
    Python is a high-level programming language...

    >>> # Check sources used
    >>> for source in response.sources:
    ...     print(f"Source {source.rank}: {source.document.metadata.get('source')}")
    ...     print(f"  Score: {source.score:.3f}")
    Source 1: python_docs.pdf
      Score: 0.892

    >>> # Debug the prompt
    >>> print(response.prompt[:200])
    Answer the question based on the following context...

    Format sources for citation:

    >>> citations = []
    >>> for s in response.sources[:3]:
    ...     src = s.document.metadata.get("source", "Unknown")
    ...     citations.append(f"[{s.rank}] {src}")
    >>> print("Sources:", ", ".join(citations))

    Use in a web application:

    >>> def answer_question(query):
    ...     response = rag.query(query)
    ...     return {
    ...         "answer": response.answer,
    ...         "sources": [
    ...             {
    ...                 "content": s.document.content[:200],
    ...                 "source": s.document.metadata.get("source"),
    ...                 "score": s.score
    ...             }
    ...             for s in response.sources
    ...         ]
    ...     }

    Notes
    -----
    - The `sources` list is ordered by relevance (highest score first).
    - The `prompt` field is useful for debugging and understanding how
      the context was formatted.
    - For production applications, consider filtering sources by a
      minimum score threshold before displaying to users.

    See Also
    --------
    RAGChain.query : Method that returns RAGResponse objects
    RetrievalResult : Structure of individual source results
    """

    answer: str
    sources: list[RetrievalResult]
    prompt: str


class RAGChain:
    """End-to-end Retrieval-Augmented Generation chain for question answering.

    RAGChain is the main orchestrator for RAG-based question answering. It
    combines document retrieval with LLM generation to answer questions based
    on a corpus of documents. The chain handles:

    1. Document ingestion with optional chunking
    2. Query-time retrieval of relevant context
    3. Prompt construction with retrieved context
    4. LLM generation of answers

    Parameters
    ----------
    model : Model
        Language model for generating answers. Must implement either
        a `generate` method or both `generate` and `chat` methods.
    vector_store : VectorStore, optional
        Vector store for document storage. If not provided and retriever
        is None, uses InMemoryVectorStore.
    embedding_model : EmbeddingModel, optional
        Model for creating embeddings. If not provided and retriever is
        None, uses SimpleEmbedding.
    retriever : Retriever, optional
        Pre-configured Retriever instance. If provided, vector_store and
        embedding_model parameters are ignored.
    prompt_template : str, default=DEFAULT_RAG_PROMPT
        Template for constructing prompts. Must contain ``{context}`` and
        ``{question}`` placeholders.
    k : int, default=5
        Default number of documents to retrieve for each query.

    Attributes
    ----------
    model : Model
        The configured language model.
    retriever : Retriever
        The document retriever (created or provided).
    prompt_template : str
        The prompt template string.
    k : int
        Default number of documents to retrieve.

    Examples
    --------
    Basic RAG setup with defaults:

    >>> from insideLLMs.rag.retrieval import RAGChain
    >>> from insideLLMs.models import OpenAIModel
    >>>
    >>> model = OpenAIModel(model_name="gpt-4")
    >>> rag = RAGChain(model=model)
    >>>
    >>> # Add documents
    >>> rag.add_documents([
    ...     "Python was created by Guido van Rossum in 1991.",
    ...     "Python emphasizes code readability and simplicity.",
    ...     "Python supports multiple programming paradigms."
    ... ])
    >>>
    >>> # Query
    >>> response = rag.query("Who created Python?")
    >>> print(response.answer)

    With custom components:

    >>> from insideLLMs.rag.retrieval import (
    ...     RAGChain, Retriever, InMemoryVectorStore,
    ...     SimpleEmbedding, TextChunker, ChunkingConfig
    ... )
    >>>
    >>> # Custom retriever with specific settings
    >>> embedding = SimpleEmbedding(dimension=512)
    >>> store = InMemoryVectorStore()
    >>> chunker = TextChunker(ChunkingConfig(chunk_size=500))
    >>> retriever = Retriever(embedding, store, chunker)
    >>>
    >>> rag = RAGChain(
    ...     model=model,
    ...     retriever=retriever,
    ...     k=3
    ... )

    Custom prompt template:

    >>> custom_template = '''Use the following documents to answer.
    ... If unsure, say "I don't know."
    ...
    ... Documents:
    ... {context}
    ...
    ... Question: {question}
    ... Answer:'''
    >>>
    >>> rag = RAGChain(
    ...     model=model,
    ...     prompt_template=custom_template
    ... )

    Using the chat interface:

    >>> response = rag.query_with_chat(
    ...     question="Explain Python's design philosophy",
    ...     system_prompt="You are a helpful programming tutor.",
    ...     k=5
    ... )

    Filtering by metadata:

    >>> rag.add_documents(
    ...     ["Technical doc 1", "Technical doc 2"],
    ...     metadatas=[{"category": "tech"}, {"category": "tech"}]
    ... )
    >>> rag.add_documents(
    ...     ["Business doc 1"],
    ...     metadatas=[{"category": "business"}]
    ... )
    >>> # Only search tech documents
    >>> response = rag.query(
    ...     "What does the documentation say?",
    ...     filter={"category": "tech"}
    ... )

    Processing response sources:

    >>> response = rag.query("What is machine learning?")
    >>> print(f"Answer: {response.answer}")
    >>> print("\\nSources used:")
    >>> for src in response.sources:
    ...     print(f"  [{src.rank}] Score: {src.score:.3f}")
    ...     print(f"      {src.document.content[:100]}...")

    Notes
    -----
    - The chain uses sensible defaults (SimpleEmbedding + InMemoryVectorStore)
      for quick prototyping. For production, use proper embedding models
      and vector databases.
    - The `query` method uses the model's `generate` method, while
      `query_with_chat` uses the `chat` method if available.
    - Documents are automatically chunked by default. Disable with
      ``chunk=False`` in `add_documents`.
    - The prompt template must include both ``{context}`` and ``{question}``
      placeholders.

    See Also
    --------
    create_rag_chain : Convenience function for creating RAGChain
    Retriever : Lower-level retrieval interface
    RAGResponse : Return type for query methods
    """

    def __init__(
        self,
        model: "Model",
        vector_store: Optional[VectorStore] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        retriever: Optional[Retriever] = None,
        prompt_template: str = DEFAULT_RAG_PROMPT,
        k: int = 5,
    ):
        """Initialize the RAG chain with model and retrieval components.

        Parameters
        ----------
        model : Model
            Language model for answer generation.
        vector_store : VectorStore, optional
            Document and embedding storage. Ignored if retriever is provided.
        embedding_model : EmbeddingModel, optional
            Model for creating embeddings. Ignored if retriever is provided.
        retriever : Retriever, optional
            Pre-configured retriever. If provided, vector_store and
            embedding_model are ignored.
        prompt_template : str, default=DEFAULT_RAG_PROMPT
            Template with {context} and {question} placeholders.
        k : int, default=5
            Default number of documents to retrieve.

        Examples
        --------
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> rag = RAGChain(model=model)  # Uses defaults

        >>> # With custom components
        >>> rag = RAGChain(
        ...     model=model,
        ...     embedding_model=CustomEmbedding(),
        ...     vector_store=CustomVectorStore(),
        ...     k=10
        ... )

        >>> # With pre-built retriever
        >>> retriever = Retriever(embedding, store, chunker)
        >>> rag = RAGChain(model=model, retriever=retriever)
        """
        self.model = model
        self.prompt_template = prompt_template
        self.k = k

        if retriever:
            self.retriever = retriever
        else:
            embedding = embedding_model or SimpleEmbedding()
            store = vector_store or InMemoryVectorStore()
            self.retriever = Retriever(embedding, store)

    def add_documents(
        self,
        texts: Union[list[str], list[Document]],
        metadatas: Optional[list[dict[str, Any]]] = None,
        chunk: bool = True,
    ) -> list[str]:
        """Add documents to the RAG chain for retrieval.

        This method accepts either raw text strings or Document objects
        and adds them to the underlying retriever.

        Parameters
        ----------
        texts : list[str] or list[Document]
            Content to add. Can be a list of text strings or Document
            objects.
        metadatas : list[dict[str, Any]], optional
            Metadata for each text. Only used when `texts` contains
            strings, ignored for Document objects.
        chunk : bool, default=True
            Whether to split documents into chunks before indexing.

        Returns
        -------
        list[str]
            List of document IDs that were added. When chunking is
            enabled, returns chunk IDs.

        Examples
        --------
        >>> rag = RAGChain(model=model)

        >>> # Add text strings
        >>> ids = rag.add_documents([
        ...     "First document content.",
        ...     "Second document content."
        ... ])

        >>> # Add with metadata
        >>> ids = rag.add_documents(
        ...     ["Tech article", "Science paper"],
        ...     metadatas=[{"type": "article"}, {"type": "paper"}]
        ... )

        >>> # Add Document objects
        >>> from insideLLMs.rag.retrieval import Document
        >>> docs = [
        ...     Document(content="Content 1", metadata={"src": "a.txt"}),
        ...     Document(content="Content 2", metadata={"src": "b.txt"})
        ... ]
        >>> ids = rag.add_documents(docs)

        >>> # Without chunking
        >>> ids = rag.add_documents(["Short text"], chunk=False)
        """
        if texts and isinstance(texts[0], Document):
            return self.retriever.add_documents(texts, chunk=chunk)
        else:
            return self.retriever.add_texts(texts, metadatas, chunk=chunk)

    def _format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieved documents into a context string for the prompt.

        Parameters
        ----------
        results : list[RetrievalResult]
            Retrieved documents to format.

        Returns
        -------
        str
            Formatted context string with numbered sources.

        Examples
        --------
        >>> # Internal method - typically not called directly
        >>> results = retriever.retrieve("query")
        >>> context = rag._format_context(results)
        >>> # Output format:
        >>> # [1] (source_name)
        >>> # Document content...
        >>> #
        >>> # [2] (source_name)
        >>> # Document content...
        """
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.document.metadata.get("source", f"Document {i}")
            context_parts.append(f"[{i}] ({source})\n{result.document.content}")
        return "\n\n".join(context_parts)

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        filter: Optional[dict[str, Any]] = None,
        **model_kwargs: Any,
    ) -> RAGResponse:
        """Query the RAG chain and generate an answer.

        This method retrieves relevant documents, constructs a prompt with
        the context, and generates an answer using the language model's
        `generate` method.

        Parameters
        ----------
        question : str
            The question to answer.
        k : int, optional
            Number of documents to retrieve. Uses the chain's default `k`
            if not specified.
        filter : dict[str, Any], optional
            Metadata filter for retrieval. Only documents matching all
            key-value pairs will be considered.
        **model_kwargs : Any
            Additional keyword arguments passed to the model's `generate`
            method (e.g., temperature, max_tokens).

        Returns
        -------
        RAGResponse
            Response object containing the answer, source documents,
            and the full prompt.

        Examples
        --------
        >>> rag = RAGChain(model=model)
        >>> rag.add_documents(["Python is a programming language."])

        >>> # Basic query
        >>> response = rag.query("What is Python?")
        >>> print(response.answer)

        >>> # With custom k
        >>> response = rag.query("Tell me about Python", k=10)

        >>> # With metadata filter
        >>> response = rag.query(
        ...     "What does the manual say?",
        ...     filter={"doc_type": "manual"}
        ... )

        >>> # With model parameters
        >>> response = rag.query(
        ...     "Summarize the key points",
        ...     temperature=0.7,
        ...     max_tokens=500
        ... )

        >>> # Access all response fields
        >>> print(f"Answer: {response.answer}")
        >>> print(f"Used {len(response.sources)} sources")
        >>> print(f"Prompt length: {len(response.prompt)} chars")
        """
        k = k or self.k

        # Retrieve relevant documents
        results = self.retriever.retrieve(question, k=k, filter=filter)

        # Format context
        context = self._format_context(results)

        # Build prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question,
        )

        # Generate answer
        answer = self.model.generate(prompt, **model_kwargs)

        return RAGResponse(
            answer=answer,
            sources=results,
            prompt=prompt,
        )

    def query_with_chat(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        k: Optional[int] = None,
        filter: Optional[dict[str, Any]] = None,
        **model_kwargs: Any,
    ) -> RAGResponse:
        """Query using the model's chat interface with optional system prompt.

        This method is similar to `query` but uses the model's `chat` method
        if available, allowing for system prompts and multi-turn conversation
        formatting.

        Parameters
        ----------
        question : str
            The question to answer.
        system_prompt : str, optional
            System prompt to set context/persona for the model.
        k : int, optional
            Number of documents to retrieve. Uses chain default if not specified.
        filter : dict[str, Any], optional
            Metadata filter for retrieval.
        **model_kwargs : Any
            Additional keyword arguments passed to the model's `chat` method.

        Returns
        -------
        RAGResponse
            Response object containing the answer, sources, and prompt.
            Note: the `prompt` field contains only the user message content,
            not the system prompt.

        Examples
        --------
        >>> response = rag.query_with_chat(
        ...     question="Explain machine learning",
        ...     system_prompt="You are an expert AI researcher. Be concise."
        ... )
        >>> print(response.answer)

        >>> # With all options
        >>> response = rag.query_with_chat(
        ...     question="What are the key findings?",
        ...     system_prompt="You are a research assistant.",
        ...     k=10,
        ...     filter={"year": 2023},
        ...     temperature=0.5
        ... )

        >>> # Without system prompt (uses chat format only)
        >>> response = rag.query_with_chat("Simple question")

        Notes
        -----
        - If the model doesn't have a `chat` method, falls back to
          `generate` with concatenated messages.
        - The system prompt is only included if the model supports chat.
        """
        k = k or self.k

        # Retrieve relevant documents
        results = self.retriever.retrieve(question, k=k, filter=filter)

        # Format context
        context = self._format_context(results)

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = self.prompt_template.format(
            context=context,
            question=question,
        )
        messages.append({"role": "user", "content": user_content})

        # Generate answer
        if hasattr(self.model, "chat"):
            answer = self.model.chat(messages, **model_kwargs)
        else:
            # Fall back to generate if chat not supported
            full_prompt = "\n".join(m["content"] for m in messages)
            answer = self.model.generate(full_prompt, **model_kwargs)

        return RAGResponse(
            answer=answer,
            sources=results,
            prompt=user_content,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_rag_chain(
    model: "Model",
    documents: Optional[list[str]] = None,
    embedding_model: Optional[EmbeddingModel] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    k: int = 5,
) -> RAGChain:
    """Create a RAG chain with optional initial documents.

    Args:
        model: LLM for generation.
        documents: Optional initial documents to add.
        embedding_model: Embedding model. Uses SimpleEmbedding if not provided.
        chunk_size: Size of document chunks.
        chunk_overlap: Overlap between chunks.
        k: Default number of documents to retrieve.

    Returns:
        Configured RAGChain.
    """
    embedding = embedding_model or SimpleEmbedding()
    store = InMemoryVectorStore()
    chunker = TextChunker(
        ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )
    retriever = Retriever(embedding, store, chunker)

    chain = RAGChain(
        model=model,
        retriever=retriever,
        k=k,
    )

    if documents:
        chain.add_documents(documents)

    return chain


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Document types
    "Document",
    "RetrievalResult",
    # Embedding
    "EmbeddingModel",
    "SimpleEmbedding",
    # Vector store
    "VectorStore",
    "InMemoryVectorStore",
    # Chunking
    "ChunkingConfig",
    "TextChunker",
    # Retriever
    "Retriever",
    # RAG
    "RAGResponse",
    "RAGChain",
    "DEFAULT_RAG_PROMPT",
    # Convenience
    "create_rag_chain",
]
