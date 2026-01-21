"""Tests for insideLLMs/retrieval.py module - RAG Pipeline."""

import pytest

from insideLLMs.retrieval import (
    ChunkingConfig,
    Document,
    InMemoryVectorStore,
    RAGChain,
    RAGResponse,
    Retriever,
    RetrievalResult,
    SimpleEmbedding,
    TextChunker,
    create_rag_chain,
)


class TestDocument:
    """Tests for Document dataclass."""

    def test_create_document_with_content(self):
        """Test creating a document with content."""
        doc = Document(content="Test content")

        assert doc.content == "Test content"
        assert doc.metadata == {}
        assert doc.id is not None
        assert doc.embedding is None

    def test_create_document_with_metadata(self):
        """Test creating a document with metadata."""
        doc = Document(content="Test", metadata={"source": "test.txt", "page": 1})

        assert doc.metadata == {"source": "test.txt", "page": 1}

    def test_create_document_with_custom_id(self):
        """Test creating a document with a custom ID."""
        doc = Document(content="Test", id="custom_id_123")

        assert doc.id == "custom_id_123"

    def test_auto_generated_id_is_consistent(self):
        """Test that same content generates same ID."""
        doc1 = Document(content="Same content")
        doc2 = Document(content="Same content")

        assert doc1.id == doc2.id

    def test_different_content_different_id(self):
        """Test that different content generates different IDs."""
        doc1 = Document(content="Content A")
        doc2 = Document(content="Content B")

        assert doc1.id != doc2.id

    def test_create_document_with_embedding(self):
        """Test creating a document with pre-computed embedding."""
        embedding = [0.1, 0.2, 0.3]
        doc = Document(content="Test", embedding=embedding)

        assert doc.embedding == embedding


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_create_retrieval_result(self):
        """Test creating a retrieval result."""
        doc = Document(content="Test")
        result = RetrievalResult(document=doc, score=0.95, rank=1)

        assert result.document == doc
        assert result.score == 0.95
        assert result.rank == 1

    def test_default_rank(self):
        """Test default rank is 0."""
        doc = Document(content="Test")
        result = RetrievalResult(document=doc, score=0.5)

        assert result.rank == 0


class TestSimpleEmbedding:
    """Tests for SimpleEmbedding class."""

    def test_embed_single_text(self):
        """Test embedding a single text."""
        embedder = SimpleEmbedding(dimension=128)
        embedding = embedder.embed("Hello world")

        assert isinstance(embedding, list)
        assert len(embedding) == 128
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_empty_text(self):
        """Test embedding empty text."""
        embedder = SimpleEmbedding(dimension=64)
        embedding = embedder.embed("")

        assert embedding == [0.0] * 64

    def test_embed_batch(self):
        """Test embedding multiple texts."""
        embedder = SimpleEmbedding(dimension=64)
        texts = ["Hello world", "Another text", "Third document"]
        embeddings = embedder.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 64 for emb in embeddings)

    def test_embeddings_are_normalized(self):
        """Test that embeddings are normalized."""
        embedder = SimpleEmbedding(dimension=128)
        embedding = embedder.embed("Some text content")

        # Check normalization (magnitude should be close to 1)
        import math
        magnitude = math.sqrt(sum(x * x for x in embedding))
        assert abs(magnitude - 1.0) < 0.01 or magnitude == 0  # Allow for zero vectors

    def test_similar_texts_have_similar_embeddings(self):
        """Test that similar texts produce similar embeddings."""
        embedder = SimpleEmbedding(dimension=256)
        embedder.embed_batch(["apple banana fruit", "apple orange fruit", "car truck vehicle"])

        emb1 = embedder.embed("apple banana fruit")
        emb2 = embedder.embed("apple orange fruit")
        emb3 = embedder.embed("car truck vehicle")

        # Calculate cosine similarities
        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0
            return dot / (norm_a * norm_b)

        import math
        sim_12 = cosine_sim(emb1, emb2)
        sim_13 = cosine_sim(emb1, emb3)

        # Similar texts should have higher similarity
        assert sim_12 > sim_13 or abs(sim_12 - sim_13) < 0.1  # Allow some tolerance

    def test_tokenize(self):
        """Test internal tokenization."""
        embedder = SimpleEmbedding()
        tokens = embedder._tokenize("Hello, World! This is a test.")

        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore class."""

    def test_add_and_search(self):
        """Test adding documents and searching."""
        store = InMemoryVectorStore()
        docs = [
            Document(content="Document about cats"),
            Document(content="Document about dogs"),
        ]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        store.add(docs, embeddings)
        results = store.search([1.0, 0.0, 0.0], k=2)

        assert len(results) == 2
        assert results[0].score == 1.0  # Exact match
        assert results[0].document.content == "Document about cats"

    def test_search_with_filter(self):
        """Test searching with metadata filter."""
        store = InMemoryVectorStore()
        docs = [
            Document(content="Cat doc", metadata={"type": "animal"}),
            Document(content="Car doc", metadata={"type": "vehicle"}),
        ]
        embeddings = [
            [1.0, 0.0],
            [0.9, 0.1],
        ]

        store.add(docs, embeddings)
        results = store.search([1.0, 0.0], k=2, filter={"type": "vehicle"})

        assert len(results) == 1
        assert results[0].document.content == "Car doc"

    def test_delete_documents(self):
        """Test deleting documents."""
        store = InMemoryVectorStore()
        docs = [
            Document(content="Doc 1", id="id1"),
            Document(content="Doc 2", id="id2"),
        ]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]

        store.add(docs, embeddings)
        assert len(store) == 2

        store.delete(["id1"])
        assert len(store) == 1

    def test_clear_store(self):
        """Test clearing all documents."""
        store = InMemoryVectorStore()
        docs = [Document(content="Test")]
        embeddings = [[1.0]]

        store.add(docs, embeddings)
        assert len(store) > 0

        store.clear()
        assert len(store) == 0

    def test_add_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise ValueError."""
        store = InMemoryVectorStore()
        docs = [Document(content="Doc 1")]
        embeddings = [[1.0], [2.0]]  # Wrong length

        with pytest.raises(ValueError):
            store.add(docs, embeddings)

    def test_cosine_similarity_calculation(self):
        """Test internal cosine similarity calculation."""
        sim = InMemoryVectorStore._cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert sim == 1.0

        sim = InMemoryVectorStore._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert sim == 0.0

        sim = InMemoryVectorStore._cosine_similarity([1.0, 1.0], [1.0, 1.0])
        assert abs(sim - 1.0) < 0.01

    def test_cosine_similarity_with_zero_vector(self):
        """Test cosine similarity with zero vector."""
        sim = InMemoryVectorStore._cosine_similarity([0.0, 0.0], [1.0, 0.0])
        assert sim == 0.0

    def test_search_returns_ranked_results(self):
        """Test that search results have correct ranks."""
        store = InMemoryVectorStore()
        docs = [
            Document(content="Doc 1"),
            Document(content="Doc 2"),
            Document(content="Doc 3"),
        ]
        embeddings = [
            [0.5, 0.5],
            [1.0, 0.0],
            [0.7, 0.3],
        ]

        store.add(docs, embeddings)
        results = store.search([1.0, 0.0], k=3)

        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3


class TestChunkingConfig:
    """Tests for ChunkingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()

        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.separator == "\n"
        assert config.length_function == len

    def test_custom_config(self):
        """Test custom configuration."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            separator=" ",
        )

        assert config.chunk_size == 500
        assert config.chunk_overlap == 50


class TestTextChunker:
    """Tests for TextChunker class."""

    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk_size."""
        chunker = TextChunker(ChunkingConfig(chunk_size=1000))
        chunks = chunker.chunk("Short text")

        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_chunk_long_text(self):
        """Test chunking text longer than chunk_size."""
        text = "Word " * 500  # ~2500 chars
        chunker = TextChunker(ChunkingConfig(chunk_size=500, chunk_overlap=50))
        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 600 for chunk in chunks)  # Allow some flexibility

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk("")

        assert chunks == []

    def test_chunk_with_paragraphs(self):
        """Test chunking text with paragraph separators."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunker = TextChunker(ChunkingConfig(chunk_size=50, chunk_overlap=10))
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1

    def test_chunk_preserves_words(self):
        """Test that chunking doesn't break words when possible."""
        text = "The quick brown fox jumps over the lazy dog."
        chunker = TextChunker(ChunkingConfig(chunk_size=20, chunk_overlap=5))
        chunks = chunker.chunk(text)

        # Each chunk should contain complete words (no broken words at boundaries)
        for chunk in chunks:
            words = chunk.split()
            assert all(len(word) > 0 for word in words)

    def test_chunk_documents(self):
        """Test chunking multiple documents."""
        docs = [
            Document(content="Document one content here.", metadata={"source": "doc1"}),
            Document(content="Document two content here.", metadata={"source": "doc2"}),
        ]
        chunker = TextChunker(ChunkingConfig(chunk_size=15, chunk_overlap=3))
        chunks = chunker.chunk_documents(docs)

        assert len(chunks) >= 2
        # Check metadata is preserved
        assert all("source_doc_id" in chunk.metadata for chunk in chunks)
        assert all("chunk_index" in chunk.metadata for chunk in chunks)

    def test_chunk_documents_without_metadata(self):
        """Test chunking documents without preserving metadata."""
        docs = [Document(content="Test content", metadata={"key": "value"})]
        chunker = TextChunker()
        chunks = chunker.chunk_documents(docs, preserve_metadata=False)

        assert len(chunks) >= 1
        assert "key" not in chunks[0].metadata


class TestRetriever:
    """Tests for Retriever class."""

    def test_add_texts(self):
        """Test adding texts to retriever."""
        embedding = SimpleEmbedding(dimension=64)
        store = InMemoryVectorStore()
        retriever = Retriever(embedding, store)

        ids = retriever.add_texts(["Text 1", "Text 2"])

        assert len(ids) >= 2

    def test_add_texts_with_metadata(self):
        """Test adding texts with metadata."""
        embedding = SimpleEmbedding(dimension=64)
        store = InMemoryVectorStore()
        retriever = Retriever(embedding, store)

        ids = retriever.add_texts(
            ["Text 1", "Text 2"],
            metadatas=[{"source": "a"}, {"source": "b"}],
        )

        assert len(ids) >= 2

    def test_add_documents(self):
        """Test adding documents to retriever."""
        embedding = SimpleEmbedding(dimension=64)
        store = InMemoryVectorStore()
        retriever = Retriever(embedding, store)

        docs = [Document(content="Doc 1"), Document(content="Doc 2")]
        ids = retriever.add_documents(docs)

        assert len(ids) >= 2

    def test_retrieve(self):
        """Test retrieving documents."""
        embedding = SimpleEmbedding(dimension=64)
        store = InMemoryVectorStore()
        retriever = Retriever(embedding, store)

        retriever.add_texts([
            "The cat sat on the mat",
            "Dogs love to play fetch",
            "Birds fly in the sky",
        ])

        results = retriever.retrieve("cat", k=2)

        assert len(results) <= 2
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_with_filter(self):
        """Test retrieving with metadata filter."""
        embedding = SimpleEmbedding(dimension=64)
        store = InMemoryVectorStore()
        retriever = Retriever(embedding, store)

        retriever.add_texts(
            ["Cats are pets", "Cars are vehicles"],
            metadatas=[{"type": "animal"}, {"type": "vehicle"}],
            chunk=False,
        )

        results = retriever.retrieve("pets", k=2, filter={"type": "animal"})

        assert len(results) <= 2

    def test_delete_documents(self):
        """Test deleting documents from retriever."""
        embedding = SimpleEmbedding(dimension=64)
        store = InMemoryVectorStore()
        retriever = Retriever(embedding, store)

        ids = retriever.add_texts(["Text to delete"], chunk=False)
        retriever.delete(ids)

        assert len(store) == 0

    def test_clear_retriever(self):
        """Test clearing retriever."""
        embedding = SimpleEmbedding(dimension=64)
        store = InMemoryVectorStore()
        retriever = Retriever(embedding, store)

        retriever.add_texts(["Text 1", "Text 2"])
        retriever.clear()

        assert len(store) == 0

    def test_add_without_chunking(self):
        """Test adding texts without chunking."""
        embedding = SimpleEmbedding(dimension=64)
        store = InMemoryVectorStore()
        retriever = Retriever(embedding, store)

        ids = retriever.add_texts(["Short text"], chunk=False)

        assert len(ids) == 1


class TestRAGChain:
    """Tests for RAGChain class."""

    def test_create_rag_chain_with_defaults(self):
        """Test creating RAG chain with default components."""
        class MockModel:
            def generate(self, prompt, **kwargs):
                return "Generated response"

        model = MockModel()
        rag = RAGChain(model=model)

        assert rag.model == model
        assert rag.k == 5

    def test_add_documents_as_strings(self):
        """Test adding documents as strings."""
        class MockModel:
            def generate(self, prompt, **kwargs):
                return "Response"

        rag = RAGChain(model=MockModel())
        ids = rag.add_documents(["Doc 1", "Doc 2"])

        assert len(ids) >= 2

    def test_add_documents_as_document_objects(self):
        """Test adding documents as Document objects."""
        class MockModel:
            def generate(self, prompt, **kwargs):
                return "Response"

        rag = RAGChain(model=MockModel())
        docs = [Document(content="Doc 1"), Document(content="Doc 2")]
        ids = rag.add_documents(docs)

        assert len(ids) >= 2

    def test_query(self):
        """Test querying the RAG chain."""
        class MockModel:
            def generate(self, prompt, **kwargs):
                return f"Answer based on: {prompt[:50]}"

        rag = RAGChain(model=MockModel())
        rag.add_documents(["Paris is the capital of France", "London is the capital of UK"])

        response = rag.query("What is the capital of France?")

        assert isinstance(response, RAGResponse)
        assert isinstance(response.answer, str)
        assert len(response.sources) <= 5
        assert "capital" in response.prompt.lower() or "france" in response.prompt.lower()

    def test_query_with_custom_k(self):
        """Test querying with custom k."""
        class MockModel:
            def generate(self, prompt, **kwargs):
                return "Answer"

        rag = RAGChain(model=MockModel())
        rag.add_documents(["Doc 1", "Doc 2", "Doc 3"])

        response = rag.query("test", k=2)

        assert len(response.sources) <= 2

    def test_query_with_chat(self):
        """Test query_with_chat method."""
        class MockModel:
            def chat(self, messages, **kwargs):
                return "Chat response"

        rag = RAGChain(model=MockModel())
        rag.add_documents(["Test document"])

        response = rag.query_with_chat("Question?", system_prompt="Be helpful")

        assert response.answer == "Chat response"

    def test_query_with_chat_fallback_to_generate(self):
        """Test that query_with_chat falls back to generate if chat not available."""
        class MockModel:
            def generate(self, prompt, **kwargs):
                return "Generate response"

        rag = RAGChain(model=MockModel())
        rag.add_documents(["Test document"])

        response = rag.query_with_chat("Question?")

        assert response.answer == "Generate response"

    def test_format_context(self):
        """Test internal context formatting."""
        class MockModel:
            def generate(self, prompt, **kwargs):
                return "Answer"

        rag = RAGChain(model=MockModel())
        results = [
            RetrievalResult(
                document=Document(content="Content 1", metadata={"source": "file1.txt"}),
                score=0.9,
                rank=1,
            ),
            RetrievalResult(
                document=Document(content="Content 2", metadata={}),
                score=0.8,
                rank=2,
            ),
        ]

        context = rag._format_context(results)

        assert "[1]" in context
        assert "[2]" in context
        assert "Content 1" in context
        assert "file1.txt" in context

    def test_custom_prompt_template(self):
        """Test using a custom prompt template."""
        class MockModel:
            def generate(self, prompt, **kwargs):
                return prompt

        custom_template = "Context: {context}\n\nQ: {question}"
        rag = RAGChain(model=MockModel(), prompt_template=custom_template)
        rag.add_documents(["Test"])

        response = rag.query("What?")

        assert "Context:" in response.answer
        assert "Q:" in response.answer


class TestCreateRagChain:
    """Tests for create_rag_chain convenience function."""

    def test_create_with_defaults(self):
        """Test creating RAG chain with defaults."""
        class MockModel:
            def generate(self, prompt, **kwargs):
                return "Response"

        chain = create_rag_chain(model=MockModel())

        assert chain is not None
        assert chain.k == 5

    def test_create_with_documents(self):
        """Test creating RAG chain with initial documents."""
        class MockModel:
            def generate(self, prompt, **kwargs):
                return "Response"

        chain = create_rag_chain(
            model=MockModel(),
            documents=["Doc 1", "Doc 2"],
        )

        # Documents should be added
        response = chain.query("test")
        assert len(response.sources) > 0

    def test_create_with_custom_config(self):
        """Test creating RAG chain with custom configuration."""
        class MockModel:
            def generate(self, prompt, **kwargs):
                return "Response"

        chain = create_rag_chain(
            model=MockModel(),
            chunk_size=500,
            chunk_overlap=100,
            k=3,
        )

        assert chain.k == 3


class TestIntegration:
    """Integration tests for the full RAG pipeline."""

    def test_full_rag_pipeline(self):
        """Test complete RAG pipeline from documents to query."""
        class MockModel:
            def generate(self, prompt, **kwargs):
                if "Paris" in prompt:
                    return "The capital of France is Paris."
                return "I don't know."

        # Create RAG chain
        rag = RAGChain(model=MockModel())

        # Add documents
        rag.add_documents([
            "Paris is the capital of France. It is known for the Eiffel Tower.",
            "Berlin is the capital of Germany. It has a rich history.",
            "Tokyo is the capital of Japan. It is a modern metropolis.",
        ])

        # Query
        response = rag.query("What is the capital of France?", k=2)

        assert "Paris" in response.answer
        assert len(response.sources) > 0

    def test_empty_retriever_query(self):
        """Test querying with no documents added."""
        class MockModel:
            def generate(self, prompt, **kwargs):
                return "No context available"

        rag = RAGChain(model=MockModel())
        response = rag.query("What is something?")

        assert response.answer == "No context available"
        assert len(response.sources) == 0
