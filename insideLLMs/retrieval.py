"""Native RAG (Retrieval-Augmented Generation) Pipeline.

This module provides a flexible RAG implementation with:
- VectorStore protocol for plugging in various vector databases
- Document chunking utilities
- Embedding management
- Retriever class combining embedding + vector search
- RAGChain for end-to-end retrieval-augmented generation

Example:
    >>> from insideLLMs.retrieval import RAGChain, InMemoryVectorStore
    >>> from insideLLMs.models import OpenAIModel
    >>>
    >>> # Create components
    >>> model = OpenAIModel(model_name="gpt-4")
    >>> store = InMemoryVectorStore()
    >>>
    >>> # Build RAG chain
    >>> rag = RAGChain(model=model, vector_store=store)
    >>>
    >>> # Add documents
    >>> rag.add_documents(["Doc 1 content...", "Doc 2 content..."])
    >>>
    >>> # Query
    >>> response = rag.query("What is in the documents?")
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
    """A document with content and metadata.

    Attributes:
        content: The text content of the document.
        metadata: Optional metadata (source, page number, etc.).
        id: Unique identifier for the document.
        embedding: Optional pre-computed embedding vector.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    embedding: Optional[list[float]] = None

    def __post_init__(self):
        if self.id is None:
            # Generate ID from content hash
            self.id = hashlib.md5(self.content.encode()).hexdigest()[:16]


@dataclass
class RetrievalResult:
    """Result from a retrieval operation.

    Attributes:
        document: The retrieved document.
        score: Similarity score (higher is more similar).
        rank: Position in results (1-indexed).
    """

    document: Document
    score: float
    rank: int = 0


# =============================================================================
# Embedding Protocol
# =============================================================================


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding models.

    Any object implementing `embed()` and `embed_batch()` methods
    can be used as an embedding model.
    """

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...


class SimpleEmbedding:
    """Simple TF-IDF-like embedding for testing and lightweight use.

    This is NOT suitable for production semantic search but useful
    for testing the RAG pipeline without external dependencies.
    """

    def __init__(self, dimension: int = 256):
        """Initialize the simple embedding model.

        Args:
            dimension: Output embedding dimension.
        """
        self.dimension = dimension
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_count = 0

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization.

        Note: Delegates to insideLLMs.nlp.tokenization.word_tokenize_regex
        """
        return word_tokenize_regex(text)

    def _update_vocab(self, text: str) -> None:
        """Update vocabulary with new text."""
        tokens = set(self._tokenize(text))
        self._doc_count += 1

        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
            self._idf[token] = self._idf.get(token, 0) + 1

    def embed(self, text: str) -> list[float]:
        """Create a simple embedding from text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.dimension

        # Count token frequencies
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        # Create sparse vector then project to fixed dimension
        vector = [0.0] * self.dimension
        for token, count in tf.items():
            # Use hash to get consistent index
            idx = hash(token) % self.dimension
            # TF-IDF-like weighting
            idf = math.log((self._doc_count + 1) / (self._idf.get(token, 0) + 1)) + 1
            vector[idx] += count * idf

        # Normalize
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
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
    """Protocol for vector stores.

    Implementations can use various backends (FAISS, Pinecone, Chroma, etc.)
    as long as they implement these methods.
    """

    def add(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        """Add documents with their embeddings to the store.

        Args:
            documents: List of documents to add.
            embeddings: Corresponding embedding vectors.
        """
        ...

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[RetrievalResult]:
        """Search for similar documents.

        Args:
            query_embedding: Query vector.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of RetrievalResults sorted by similarity.
        """
        ...

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete.
        """
        ...

    def clear(self) -> None:
        """Remove all documents from the store."""
        ...


class InMemoryVectorStore:
    """Simple in-memory vector store using cosine similarity.

    Suitable for development and small datasets. For production,
    use a proper vector database like FAISS, Pinecone, or Chroma.
    """

    def __init__(self):
        """Initialize the in-memory store."""
        self._documents: dict[str, Document] = {}
        self._embeddings: dict[str, list[float]] = {}

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
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
        """Add documents with embeddings.

        Args:
            documents: Documents to add.
            embeddings: Corresponding embeddings.
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
        """Search for similar documents.

        Args:
            query_embedding: Query vector.
            k: Number of results.
            filter: Optional metadata filter.

        Returns:
            Sorted list of results.
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
        """Delete documents by ID.

        Args:
            ids: Document IDs to delete.
        """
        for doc_id in ids:
            self._documents.pop(doc_id, None)
            self._embeddings.pop(doc_id, None)

    def clear(self) -> None:
        """Remove all documents."""
        self._documents.clear()
        self._embeddings.clear()

    def __len__(self) -> int:
        return len(self._documents)


# =============================================================================
# Document Chunking
# =============================================================================


@dataclass
class ChunkingConfig:
    """Configuration for document chunking.

    Attributes:
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.
        separator: Primary separator to split on (default: newline).
        length_function: Function to measure text length.
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: str = "\n"
    length_function: Callable[[str], int] = len


class TextChunker:
    """Splits text into overlapping chunks.

    Uses a recursive strategy that tries to split on natural boundaries
    (paragraphs, sentences, words) while maintaining chunk size limits.
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
        """Initialize the chunker.

        Args:
            config: Chunking configuration. Uses defaults if not provided.
        """
        self.config = config or ChunkingConfig()

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators."""
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
        """Split text into chunks.

        Args:
            text: Text to chunk.

        Returns:
            List of text chunks.
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
        """Chunk multiple documents.

        Args:
            documents: Documents to chunk.
            preserve_metadata: Whether to copy metadata to chunks.

        Returns:
            List of chunk documents.
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
    """Combines embedding model and vector store for retrieval.

    Handles the embedding of queries and documents, delegating
    storage and search to the configured vector store.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        chunker: Optional[TextChunker] = None,
    ):
        """Initialize the retriever.

        Args:
            embedding_model: Model for creating embeddings.
            vector_store: Store for vectors and documents.
            chunker: Optional chunker for splitting documents.
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
        """Add texts to the retriever.

        Args:
            texts: Texts to add.
            metadatas: Optional metadata for each text.
            chunk: Whether to chunk texts before adding.

        Returns:
            List of document IDs that were added.
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
        """Add documents to the retriever.

        Args:
            documents: Documents to add.
            chunk: Whether to chunk documents before adding.

        Returns:
            List of document IDs that were added.
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
        """Retrieve documents relevant to a query.

        Args:
            query: Search query.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of retrieval results.
        """
        query_embedding = self.embedding_model.embed(query)
        return self.vector_store.search(query_embedding, k=k, filter=filter)

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID.

        Args:
            ids: Document IDs to delete.
        """
        self.vector_store.delete(ids)

    def clear(self) -> None:
        """Remove all documents."""
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
    """Response from a RAG query.

    Attributes:
        answer: The generated answer.
        sources: Retrieved documents used for context.
        prompt: The full prompt sent to the model.
    """

    answer: str
    sources: list[RetrievalResult]
    prompt: str


class RAGChain:
    """End-to-end Retrieval-Augmented Generation chain.

    Combines document retrieval with LLM generation for
    question answering over a document corpus.
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
        """Initialize the RAG chain.

        Args:
            model: LLM for generation.
            vector_store: Vector store for documents. If not provided
                         and retriever is None, uses InMemoryVectorStore.
            embedding_model: Embedding model. If not provided and
                            retriever is None, uses SimpleEmbedding.
            retriever: Pre-configured retriever. If provided, vector_store
                      and embedding_model are ignored.
            prompt_template: Template for RAG prompts. Must contain
                           {context} and {question} placeholders.
            k: Number of documents to retrieve.
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
        """Add documents to the RAG chain.

        Args:
            texts: List of text strings or Document objects.
            metadatas: Optional metadata for each text (ignored if texts are Documents).
            chunk: Whether to chunk documents before adding.

        Returns:
            List of document IDs.
        """
        if texts and isinstance(texts[0], Document):
            return self.retriever.add_documents(texts, chunk=chunk)
        else:
            return self.retriever.add_texts(texts, metadatas, chunk=chunk)

    def _format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieved documents into context string."""
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
        """Query the RAG chain.

        Args:
            question: Question to answer.
            k: Number of documents to retrieve. Uses default if not provided.
            filter: Optional metadata filter for retrieval.
            **model_kwargs: Additional arguments for the model.

        Returns:
            RAGResponse with answer and sources.
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
        """Query using the model's chat interface.

        Args:
            question: Question to answer.
            system_prompt: Optional system prompt.
            k: Number of documents to retrieve.
            filter: Optional metadata filter.
            **model_kwargs: Additional arguments for the model.

        Returns:
            RAGResponse with answer and sources.
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
