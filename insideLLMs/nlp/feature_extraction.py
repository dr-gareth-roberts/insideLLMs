"""
Feature Extraction Module for NLP Tasks.

This module provides functions for extracting various features from text data,
including bag-of-words representations, TF-IDF vectors, word embeddings,
part-of-speech tags, and syntactic dependencies.

Features:
    - Bag-of-Words (BoW): Count-based text representation
    - TF-IDF: Term frequency-inverse document frequency vectors
    - Word Embeddings: Dense vector representations using Word2Vec
    - POS Tagging: Part-of-speech annotation using spaCy
    - Dependency Parsing: Syntactic dependency extraction using spaCy

Example Usage:
    >>> from insideLLMs.nlp.feature_extraction import create_bow, create_tfidf
    >>> texts = ["The cat sat on the mat", "The dog ran in the park"]
    >>> bow_vectors, vocab = create_bow(texts)
    >>> print(f"Vocabulary size: {len(vocab)}")
    Vocabulary size: 10

    >>> tfidf_vectors, tfidf_vocab = create_tfidf(texts)
    >>> print(f"TF-IDF vector dimension: {len(tfidf_vectors[0])}")
    TF-IDF vector dimension: 10

    >>> from insideLLMs.nlp.feature_extraction import extract_pos_tags
    >>> tags = extract_pos_tags("The quick brown fox jumps.")
    >>> print(tags[:3])
    [('The', 'DET'), ('quick', 'ADJ'), ('brown', 'ADJ')]

    >>> from insideLLMs.nlp.feature_extraction import extract_dependencies
    >>> deps = extract_dependencies("The cat sat on the mat.")
    >>> print(deps[0])
    ('The', 'det', 'cat')

Dependencies:
    - scikit-learn: Required for create_bow and create_tfidf
    - gensim: Required for create_word_embeddings
    - spaCy: Required for extract_pos_tags and extract_dependencies
"""

from typing import Optional

from insideLLMs.nlp.dependencies import ensure_gensim, ensure_sklearn, ensure_spacy


# Backward compatibility aliases
check_spacy = ensure_spacy
check_sklearn = ensure_sklearn
check_gensim = ensure_gensim


# ===== Feature Extraction =====


def create_bow(
    texts: list[str], max_features: Optional[int] = None
) -> tuple[list[list[int]], list[str]]:
    """
    Create bag-of-words representation of texts.

    Converts a list of text documents into a matrix of token counts. Each row
    represents a document, and each column represents a unique word (feature)
    from the vocabulary.

    Args:
        texts: A list of text documents to vectorize. Each element should be
            a string containing the text of one document.
        max_features: Maximum number of features (vocabulary size) to extract.
            If None, all features are used. Features are ordered by term
            frequency across the corpus. Defaults to None.

    Returns:
        A tuple containing:
            - vectors: A list of lists where each inner list contains the word
              counts for a document. Shape is (n_documents, n_features).
            - vocabulary: A list of feature names (words) corresponding to
              the columns in the vectors matrix.

    Raises:
        ImportError: If scikit-learn is not installed.

    Examples:
        Basic usage with two documents:

        >>> texts = ["I love machine learning", "Machine learning is fun"]
        >>> vectors, vocab = create_bow(texts)
        >>> print(vocab)
        ['fun', 'i', 'is', 'learning', 'love', 'machine']
        >>> print(vectors[0])  # First document counts
        [0, 1, 0, 1, 1, 1]

        Limiting vocabulary size:

        >>> texts = ["The cat sat on the mat", "The dog ran in the park"]
        >>> vectors, vocab = create_bow(texts, max_features=5)
        >>> print(len(vocab))
        5
        >>> print(vocab)  # Top 5 most frequent words
        ['the', 'cat', 'dog', 'in', 'mat']

        Single word documents:

        >>> texts = ["hello", "world", "hello world"]
        >>> vectors, vocab = create_bow(texts)
        >>> print(vocab)
        ['hello', 'world']
        >>> print(vectors)
        [[1, 0], [0, 1], [1, 1]]

        Empty and repeated words:

        >>> texts = ["word word word", "another word"]
        >>> vectors, vocab = create_bow(texts)
        >>> print(vectors[0])  # 'word' appears 3 times
        [0, 3]
    """
    ensure_sklearn()
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(texts)
    return X.toarray().tolist(), vectorizer.get_feature_names_out().tolist()


def create_tfidf(
    texts: list[str], max_features: Optional[int] = None
) -> tuple[list[list[float]], list[str]]:
    """
    Create TF-IDF (Term Frequency-Inverse Document Frequency) representation of texts.

    Converts a list of text documents into a matrix of TF-IDF features. TF-IDF
    weights terms by their frequency in a document (TF) and inversely by their
    frequency across all documents (IDF), highlighting distinctive terms.

    The TF-IDF score for a term t in document d is computed as:
        tfidf(t, d) = tf(t, d) * idf(t)

    where tf(t, d) is the term frequency and idf(t) = log(N / df(t)) + 1,
    with N being the total number of documents and df(t) the document frequency.

    Args:
        texts: A list of text documents to vectorize. Each element should be
            a string containing the text of one document.
        max_features: Maximum number of features (vocabulary size) to extract.
            If None, all features are used. Features are ordered by term
            frequency across the corpus. Defaults to None.

    Returns:
        A tuple containing:
            - vectors: A list of lists where each inner list contains the TF-IDF
              scores for a document. Values are floats typically between 0 and 1.
              Shape is (n_documents, n_features).
            - vocabulary: A list of feature names (words) corresponding to
              the columns in the vectors matrix.

    Raises:
        ImportError: If scikit-learn is not installed.

    Examples:
        Basic usage with two documents:

        >>> texts = ["I love machine learning", "Machine learning is great"]
        >>> vectors, vocab = create_tfidf(texts)
        >>> print(vocab)
        ['great', 'i', 'is', 'learning', 'love', 'machine']
        >>> # Words unique to one document have higher TF-IDF scores
        >>> print(f"'love' score in doc 1: {vectors[0][4]:.3f}")
        'love' score in doc 1: 0.534

        Comparing common vs. rare terms:

        >>> texts = ["the cat sat", "the cat ran", "the dog barked"]
        >>> vectors, vocab = create_tfidf(texts)
        >>> # 'the' and 'cat' appear in multiple docs, so lower scores
        >>> # 'sat', 'ran', 'barked' are unique, so higher scores
        >>> for i, word in enumerate(vocab):
        ...     print(f"{word}: {vectors[0][i]:.3f}")
        barked: 0.000
        cat: 0.455
        dog: 0.000
        ran: 0.000
        sat: 0.652
        the: 0.455

        Limiting vocabulary size:

        >>> texts = ["apple banana cherry", "banana cherry date", "cherry date elderberry"]
        >>> vectors, vocab = create_tfidf(texts, max_features=3)
        >>> print(len(vocab))
        3
        >>> print(vocab)
        ['cherry', 'banana', 'date']

        Document similarity using TF-IDF:

        >>> texts = ["python programming", "java programming", "python scripting"]
        >>> vectors, vocab = create_tfidf(texts)
        >>> # Documents sharing terms will have similar vectors
        >>> import math
        >>> def cosine_sim(v1, v2):
        ...     dot = sum(a*b for a, b in zip(v1, v2))
        ...     norm1 = math.sqrt(sum(a*a for a in v1))
        ...     norm2 = math.sqrt(sum(b*b for b in v2))
        ...     return dot / (norm1 * norm2) if norm1 and norm2 else 0
        >>> print(f"Doc 0 vs Doc 2 similarity: {cosine_sim(vectors[0], vectors[2]):.3f}")
        Doc 0 vs Doc 2 similarity: 0.449
    """
    ensure_sklearn()
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(texts)
    return X.toarray().tolist(), vectorizer.get_feature_names_out().tolist()


def create_word_embeddings(
    sentences: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
) -> dict[str, list[float]]:
    """
    Create word embeddings using Word2Vec.

    Trains a Word2Vec model on the provided sentences to generate dense vector
    representations for each word in the vocabulary. Words with similar meanings
    or contexts will have similar vector representations.

    Word2Vec uses a neural network to learn word associations from a large corpus
    of text. The resulting embeddings capture semantic relationships, such as:
        - king - man + woman ~ queen
        - paris - france + italy ~ rome

    Args:
        sentences: A list of tokenized sentences. Each sentence should be a list
            of words (strings). Pre-tokenization is required.
        vector_size: Dimensionality of the word vectors. Larger values capture
            more information but require more data and computation. Common values
            range from 50 to 300. Defaults to 100.
        window: Maximum distance between the current and predicted word within
            a sentence. Larger windows capture broader context. Defaults to 5.
        min_count: Minimum frequency threshold. Words that appear fewer than
            min_count times are ignored. Useful for filtering rare words.
            Defaults to 1.

    Returns:
        A dictionary mapping each word (str) to its embedding vector (list of
        floats). The length of each vector equals vector_size.

    Raises:
        ImportError: If gensim is not installed.

    Examples:
        Basic usage with simple sentences:

        >>> sentences = [
        ...     ["the", "cat", "sat", "on", "the", "mat"],
        ...     ["the", "dog", "ran", "in", "the", "park"],
        ...     ["a", "cat", "and", "a", "dog", "played"]
        ... ]
        >>> embeddings = create_word_embeddings(sentences, vector_size=50)
        >>> print(f"Vocabulary size: {len(embeddings)}")
        Vocabulary size: 12
        >>> print(f"Vector dimension: {len(embeddings['cat'])}")
        Vector dimension: 50

        Adjusting vector size for different use cases:

        >>> # Smaller vectors for simple tasks
        >>> embeddings_small = create_word_embeddings(sentences, vector_size=20)
        >>> print(f"Small vector dim: {len(embeddings_small['dog'])}")
        Small vector dim: 20
        >>> # Larger vectors capture more semantic information
        >>> embeddings_large = create_word_embeddings(sentences, vector_size=200)
        >>> print(f"Large vector dim: {len(embeddings_large['dog'])}")
        Large vector dim: 200

        Filtering rare words with min_count:

        >>> sentences = [
        ...     ["common", "word", "common", "word"],
        ...     ["common", "word", "rare"],
        ...     ["common", "common", "word"]
        ... ]
        >>> embeddings = create_word_embeddings(sentences, min_count=2)
        >>> print("'rare' in embeddings:", "rare" in embeddings)
        'rare' in embeddings: False
        >>> print("'common' in embeddings:", "common" in embeddings)
        'common' in embeddings: True

        Using different window sizes:

        >>> sentences = [["word1", "word2", "word3", "word4", "word5"]]
        >>> # Small window captures immediate context
        >>> emb_small_window = create_word_embeddings(sentences, window=2)
        >>> # Large window captures broader context
        >>> emb_large_window = create_word_embeddings(sentences, window=10)
        >>> print(f"Words captured: {list(emb_small_window.keys())}")
        Words captured: ['word1', 'word2', 'word3', 'word4', 'word5']
    """
    ensure_gensim()
    from gensim.models import Word2Vec

    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count)
    return {word: model.wv[word].tolist() for word in model.wv.index_to_key}


def extract_pos_tags(text: str, model_name: str = "en_core_web_sm") -> list[tuple[str, str]]:
    """
    Extract part-of-speech tags from text using spaCy.

    Analyzes the input text and assigns a part-of-speech (POS) tag to each token.
    POS tags identify the grammatical category of words (noun, verb, adjective, etc.)
    and are fundamental for many NLP tasks like named entity recognition, parsing,
    and text understanding.

    The function uses Universal POS tags (UPOS), which are standardized across
    languages. Common tags include:
        - NOUN: Nouns (cat, dog, city)
        - VERB: Verbs (run, eat, think)
        - ADJ: Adjectives (big, green, happy)
        - ADV: Adverbs (quickly, very, well)
        - DET: Determiners (the, a, this)
        - PRON: Pronouns (he, she, it)
        - ADP: Adpositions (in, on, at)
        - PUNCT: Punctuation (., !, ?)

    Args:
        text: The input text to analyze. Can be a single sentence or multiple
            sentences.
        model_name: Name of the spaCy model to use for POS tagging. Defaults
            to "en_core_web_sm" (small English model). Other options include
            "en_core_web_md" (medium) and "en_core_web_lg" (large) for better
            accuracy, or models for other languages.

    Returns:
        A list of tuples, where each tuple contains:
            - token (str): The original word/token from the text
            - pos_tag (str): The Universal POS tag for that token

    Raises:
        ImportError: If spaCy is not installed.
        OSError: If the specified spaCy model is not installed.

    Examples:
        Basic usage with a simple sentence:

        >>> tags = extract_pos_tags("The quick brown fox jumps over the lazy dog.")
        >>> for token, pos in tags[:5]:
        ...     print(f"{token:10} -> {pos}")
        The        -> DET
        quick      -> ADJ
        brown      -> ADJ
        fox        -> NOUN
        jumps      -> VERB

        Analyzing verb tenses:

        >>> tags = extract_pos_tags("She walked to the store and will buy groceries.")
        >>> verbs = [(token, pos) for token, pos in tags if pos == "VERB"]
        >>> print(verbs)
        [('walked', 'VERB'), ('buy', 'VERB')]

        Identifying nouns in a sentence:

        >>> tags = extract_pos_tags("The scientist discovered a new planet in the galaxy.")
        >>> nouns = [token for token, pos in tags if pos == "NOUN"]
        >>> print(nouns)
        ['scientist', 'planet', 'galaxy']

        Analyzing different sentence types:

        >>> # Question
        >>> tags = extract_pos_tags("What is your name?")
        >>> print([(t, p) for t, p in tags])
        [('What', 'PRON'), ('is', 'AUX'), ('your', 'PRON'), ('name', 'NOUN'), ('?', 'PUNCT')]

        >>> # Command/Imperative
        >>> tags = extract_pos_tags("Please close the door quietly.")
        >>> print([(t, p) for t, p in tags])
        [('Please', 'INTJ'), ('close', 'VERB'), ('the', 'DET'), ('door', 'NOUN'), ('quietly', 'ADV'), ('.', 'PUNCT')]

        Counting POS tag frequencies:

        >>> text = "The big brown dog chased the small white cat around the yard."
        >>> tags = extract_pos_tags(text)
        >>> from collections import Counter
        >>> pos_counts = Counter(pos for _, pos in tags)
        >>> print(dict(pos_counts))
        {'DET': 3, 'ADJ': 4, 'NOUN': 3, 'VERB': 1, 'ADP': 1, 'PUNCT': 1}
    """
    nlp = ensure_spacy(model_name)
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]


def extract_dependencies(
    text: str, model_name: str = "en_core_web_sm"
) -> list[tuple[str, str, str]]:
    """
    Extract dependency relations from text using spaCy.

    Analyzes the syntactic structure of the input text and extracts dependency
    relations between words. Each word is linked to its syntactic head (the word
    it depends on) with a labeled relation type.

    Dependency parsing reveals the grammatical structure of sentences, showing
    how words relate to each other. Common dependency relations include:
        - nsubj: Nominal subject (who/what performs the action)
        - dobj/obj: Direct object (who/what receives the action)
        - det: Determiner (the, a, this modifying a noun)
        - amod: Adjectival modifier (adjective modifying a noun)
        - prep: Prepositional modifier
        - pobj: Object of preposition
        - ROOT: The root of the sentence (usually the main verb)
        - aux: Auxiliary verb (is, was, will, etc.)

    Args:
        text: The input text to analyze. Can be a single sentence or multiple
            sentences.
        model_name: Name of the spaCy model to use for dependency parsing.
            Defaults to "en_core_web_sm" (small English model). Larger models
            like "en_core_web_md" or "en_core_web_lg" may provide better
            accuracy for complex sentences.

    Returns:
        A list of tuples, where each tuple contains:
            - token (str): The word/token from the text
            - dep_relation (str): The dependency relation label
            - head (str): The head word that this token depends on

    Raises:
        ImportError: If spaCy is not installed.
        OSError: If the specified spaCy model is not installed.

    Examples:
        Basic usage with a simple sentence:

        >>> deps = extract_dependencies("The cat sat on the mat.")
        >>> for token, relation, head in deps:
        ...     print(f"{token:10} --{relation:10}--> {head}")
        The        --det       --> cat
        cat        --nsubj     --> sat
        sat        --ROOT      --> sat
        on         --prep      --> sat
        the        --det       --> mat
        mat        --pobj      --> on
        .          --punct     --> sat

        Finding the subject and object of a sentence:

        >>> deps = extract_dependencies("The scientist discovered a new planet.")
        >>> subject = [t for t, rel, _ in deps if rel == "nsubj"]
        >>> obj = [t for t, rel, _ in deps if rel in ("dobj", "obj")]
        >>> print(f"Subject: {subject}, Object: {obj}")
        Subject: ['scientist'], Object: ['planet']

        Analyzing complex sentences with multiple clauses:

        >>> text = "The dog that bit the man ran away quickly."
        >>> deps = extract_dependencies(text)
        >>> # Find all subjects
        >>> subjects = [(t, h) for t, rel, h in deps if "subj" in rel]
        >>> print(f"Subjects and their verbs: {subjects}")
        Subjects and their verbs: [('dog', 'ran'), ('that', 'bit')]

        Extracting modifier relationships:

        >>> deps = extract_dependencies("The very tall old man walked slowly.")
        >>> modifiers = [(t, rel, h) for t, rel, h in deps if rel in ("amod", "advmod")]
        >>> for token, rel, head in modifiers:
        ...     print(f"'{token}' modifies '{head}' ({rel})")
        'very' modifies 'tall' (advmod)
        'tall' modifies 'man' (amod)
        'old' modifies 'man' (amod)
        'slowly' modifies 'walked' (advmod)

        Building a dependency tree structure:

        >>> deps = extract_dependencies("I love natural language processing.")
        >>> # Create adjacency representation
        >>> tree = {}
        >>> for token, rel, head in deps:
        ...     if head not in tree:
        ...         tree[head] = []
        ...     tree[head].append((token, rel))
        >>> print(f"Children of 'love': {tree.get('love', [])}")
        Children of 'love': [('I', 'nsubj'), ('processing', 'dobj'), ('.', 'punct')]
    """
    nlp = ensure_spacy(model_name)
    doc = nlp(text)
    return [(token.text, token.dep_, token.head.text) for token in doc]
