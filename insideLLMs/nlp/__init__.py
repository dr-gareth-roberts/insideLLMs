"""
Natural Language Processing (NLP) Package for insideLLMs.

The ``insideLLMs.nlp`` package provides a comprehensive suite of natural language
processing utilities designed for text analysis, preprocessing, feature extraction,
and LLM output evaluation. This package consolidates commonly needed NLP operations
into a single, easy-to-use API.

Overview
--------
The package is organized into functional modules covering the full NLP pipeline:

**Text Preprocessing & Cleaning** (``text_cleaning``)
    Functions for cleaning and normalizing raw text, including HTML removal,
    URL stripping, punctuation handling, emoji removal, Unicode normalization,
    contraction expansion, and whitespace normalization.

**Tokenization** (``tokenization``)
    Multiple tokenization strategies from simple whitespace splitting to
    sophisticated NLTK and spaCy-based tokenizers. Includes sentence segmentation,
    n-gram generation, stopword removal, stemming, and lemmatization.

**Character-Level Operations** (``char_level``)
    Character n-gram extraction, frequency analysis, and case conversion
    utilities (uppercase, titlecase, camelCase, snake_case).

**Text Extraction** (``extraction``)
    Pattern-based extraction of structured data from text: emails, phone numbers,
    URLs, hashtags, mentions, IP addresses, and named entity recognition using
    spaCy models.

**Text Transformation** (``text_transformation``)
    String manipulation utilities including truncation, padding, PII masking,
    and case-preserving word replacement.

**Text Metrics** (``text_metrics``)
    Quantitative text analysis including word/sentence counts, average lengths,
    lexical diversity (type-token ratio), syllable counting, Flesch-Kincaid
    readability scoring, and word frequency distributions.

**Text Analysis** (``text_analysis``)
    High-level text profiling and LLM response evaluation tools including
    ``TextProfile``, ``ReadabilityMetrics``, ``ContentAnalysis``, ``ToneAnalysis``,
    and ``ResponseQualityScore`` for comprehensive text assessment.

**Feature Extraction** (``feature_extraction``)
    Machine learning feature generation including bag-of-words, TF-IDF vectors,
    Word2Vec embeddings, POS tagging, and dependency parsing.

**Text Similarity** (``similarity``)
    Multiple similarity and distance metrics: cosine similarity (TF-IDF based),
    Jaccard similarity, Levenshtein distance, Jaro-Winkler similarity,
    Hamming distance, longest common subsequence, and semantic similarity
    using word embeddings.

**Classification** (``classification``)
    Text classification utilities including Naive Bayes and SVM classifiers,
    and VADER-based sentiment analysis.

**Text Chunking** (``chunking``)
    Functions for splitting text into manageable chunks by character count,
    word count, sentence count, or using sliding windows with overlap.

**Language Detection** (``language_detection``)
    Lightweight language identification using stopword frequency analysis
    and character n-gram profiling for 7 major European languages.

**Encoding/Decoding** (``encoding``)
    Text encoding utilities for Base64, URL encoding (percent-encoding),
    and HTML entity encoding/decoding.

**Keyword Extraction** (``keyword_extraction``)
    Keyword extraction using TF-IDF and TextRank algorithms for identifying
    the most important terms in documents.

Available Modules
-----------------
text_cleaning
    ``remove_html_tags``, ``remove_urls``, ``remove_punctuation``,
    ``normalize_whitespace``, ``normalize_unicode``, ``remove_emojis``,
    ``remove_numbers``, ``normalize_contractions``, ``replace_repeated_chars``,
    ``clean_text``

tokenization
    ``simple_tokenize``, ``word_tokenize_regex``, ``nltk_tokenize``,
    ``spacy_tokenize``, ``segment_sentences``, ``get_ngrams``,
    ``remove_stopwords``, ``stem_words``, ``lemmatize_words``

char_level
    ``get_char_ngrams``, ``get_char_frequency``, ``to_uppercase``,
    ``to_titlecase``, ``to_camelcase``, ``to_snakecase``

extraction
    ``extract_emails``, ``extract_phone_numbers``, ``extract_urls``,
    ``extract_hashtags``, ``extract_mentions``, ``extract_ip_addresses``,
    ``extract_named_entities``, ``extract_entities_by_type``

text_transformation
    ``truncate_text``, ``pad_text``, ``mask_pii``, ``replace_words``

text_metrics
    ``count_words``, ``count_sentences``, ``calculate_avg_word_length``,
    ``calculate_avg_sentence_length``, ``calculate_lexical_diversity``,
    ``count_syllables``, ``calculate_readability_flesch_kincaid``,
    ``get_word_frequencies``

text_analysis
    ``TextProfile``, ``ReadabilityMetrics``, ``ContentAnalysis``,
    ``ToneAnalysis``, ``TextAnalyzer``, ``ResponseQualityScore``,
    ``analyze_text``, ``compare_responses``, ``score_response``

feature_extraction
    ``create_bow``, ``create_tfidf``, ``create_word_embeddings``,
    ``extract_pos_tags``, ``extract_dependencies``

similarity
    ``cosine_similarity_texts``, ``jaccard_similarity``, ``word_overlap_similarity``,
    ``levenshtein_distance``, ``jaro_similarity``, ``jaro_winkler_similarity``,
    ``hamming_distance``, ``longest_common_subsequence``,
    ``semantic_similarity_word_embeddings``

classification
    ``naive_bayes_classify``, ``svm_classify``, ``sentiment_analysis_basic``

chunking
    ``split_by_char_count``, ``split_by_word_count``, ``split_by_sentence``,
    ``sliding_window_chunks``

language_detection
    ``detect_language_by_stopwords``, ``detect_language_by_char_ngrams``

encoding
    ``encode_base64``, ``decode_base64``, ``url_encode``, ``url_decode``,
    ``html_encode``, ``html_decode``

keyword_extraction
    ``extract_keywords_tfidf``, ``extract_keywords_textrank``

Examples
--------
**Example 1: Basic Text Cleaning Pipeline**

Clean and normalize raw text from web scraping or user input:

>>> from insideLLMs.nlp import clean_text, normalize_contractions
>>> raw_text = "<p>I can't believe it's https://example.com!! 😍 Sooo cool!!!</p>"
>>> cleaned = clean_text(raw_text, remove_emoji=True, replace_repeated=True)
>>> print(cleaned)
i can't believe it's !! soo cool!!

>>> # Expand contractions for analysis
>>> expanded = normalize_contractions(cleaned)
>>> print(expanded)
i can not believe it is !! soo cool!!

**Example 2: Tokenization and Preprocessing**

Tokenize text and prepare for machine learning:

>>> from insideLLMs.nlp import (
...     nltk_tokenize, remove_stopwords, lemmatize_words, get_ngrams
... )
>>> text = "The quick brown foxes are jumping over the lazy dogs"
>>> tokens = nltk_tokenize(text.lower())
>>> tokens = remove_stopwords(tokens)
>>> tokens = lemmatize_words(tokens)
>>> print(tokens)
['quick', 'brown', 'fox', 'jumping', 'lazy', 'dog']

>>> # Generate bigrams for feature extraction
>>> bigrams = get_ngrams(tokens, n=2)
>>> print(bigrams[:3])
[('quick', 'brown'), ('brown', 'fox'), ('fox', 'jumping')]

**Example 3: Text Extraction**

Extract structured information from unstructured text:

>>> from insideLLMs.nlp import (
...     extract_emails, extract_urls, extract_hashtags,
...     extract_named_entities
... )
>>> social_post = '''Check out my new blog at https://myblog.com!
... Contact: author@example.com #Python #NLP #MachineLearning'''

>>> print(extract_emails(social_post))
['author@example.com']
>>> print(extract_urls(social_post))
['https://myblog.com!']
>>> print(extract_hashtags(social_post))
['#Python', '#NLP', '#MachineLearning']

>>> # Named entity recognition
>>> news = "Apple CEO Tim Cook announced new products in San Francisco."
>>> entities = extract_named_entities(news)
>>> print(entities)  # doctest: +SKIP
[('Apple', 'ORG'), ('Tim Cook', 'PERSON'), ('San Francisco', 'GPE')]

**Example 4: Text Metrics and Readability**

Analyze text complexity and characteristics:

>>> from insideLLMs.nlp import (
...     count_words, count_sentences, calculate_lexical_diversity,
...     calculate_readability_flesch_kincaid
... )
>>> article = '''Natural language processing enables computers to understand text.
... Machine learning algorithms power modern NLP systems.
... Deep learning has revolutionized language understanding.'''

>>> print(f"Words: {count_words(article)}")
Words: 20
>>> print(f"Sentences: {count_sentences(article)}")
Sentences: 3
>>> print(f"Lexical Diversity: {calculate_lexical_diversity(article):.2f}")  # doctest: +SKIP
Lexical Diversity: 0.90
>>> grade = calculate_readability_flesch_kincaid(article)
>>> print(f"Flesch-Kincaid Grade: {grade:.1f}")  # doctest: +SKIP
Flesch-Kincaid Grade: 11.4

**Example 5: Text Similarity Comparison**

Compare documents using various similarity metrics:

>>> from insideLLMs.nlp import (
...     cosine_similarity_texts, jaccard_similarity,
...     levenshtein_distance, jaro_winkler_similarity
... )
>>> text1 = "The quick brown fox jumps over the lazy dog"
>>> text2 = "A fast brown fox leaps over a sleepy dog"

>>> # TF-IDF cosine similarity (lexical)
>>> cos_sim = cosine_similarity_texts(text1, text2)
>>> print(f"Cosine Similarity: {cos_sim:.3f}")  # doctest: +SKIP
Cosine Similarity: 0.456

>>> # Jaccard similarity (set overlap)
>>> jac_sim = jaccard_similarity(text1, text2)
>>> print(f"Jaccard Similarity: {jac_sim:.3f}")  # doctest: +SKIP
Jaccard Similarity: 0.357

>>> # Character-level edit distance
>>> distance = levenshtein_distance("kitten", "sitting")
>>> print(f"Levenshtein Distance: {distance}")
Levenshtein Distance: 3

>>> # Fuzzy name matching
>>> similarity = jaro_winkler_similarity("MARTHA", "MARHTA")
>>> print(f"Jaro-Winkler: {similarity:.4f}")  # doctest: +SKIP
Jaro-Winkler: 0.9611

**Example 6: Text Classification and Sentiment**

Classify text and analyze sentiment:

>>> from insideLLMs.nlp import (
...     naive_bayes_classify, svm_classify, sentiment_analysis_basic
... )
>>> # Train a simple spam classifier
>>> train_texts = [
...     "Buy cheap products now!", "Limited time offer!",
...     "Meeting scheduled for tomorrow", "Please review the document"
... ]
>>> train_labels = ["spam", "spam", "ham", "ham"]
>>> test_texts = ["Win a free prize!", "Project update attached"]
>>> predictions = naive_bayes_classify(train_texts, train_labels, test_texts)
>>> print(predictions)  # doctest: +SKIP
['spam', 'ham']

>>> # Sentiment analysis
>>> print(sentiment_analysis_basic("I love this product! It's amazing!"))
positive
>>> print(sentiment_analysis_basic("This is terrible and disappointing."))
negative
>>> print(sentiment_analysis_basic("The meeting is at 3pm."))
neutral

**Example 7: LLM Response Analysis**

Evaluate and compare LLM-generated responses:

>>> from insideLLMs.nlp import (
...     analyze_text, score_response, compare_responses, TextAnalyzer
... )
>>> response = "Python is a versatile programming language used for web development, data science, and automation."

>>> # Comprehensive text analysis
>>> analysis = analyze_text(response)
>>> print(f"Word count: {analysis['profile'].word_count}")
Word count: 14
>>> print(f"Readability: {analysis['readability'].get_reading_level()}")  # doctest: +SKIP
Readability: middle_school

>>> # Score response quality
>>> score = score_response(
...     response=response,
...     prompt="What is Python?",
...     required_keywords=["programming", "language"]
... )
>>> print(f"Overall Score: {score.overall:.3f}")  # doctest: +SKIP
Overall Score: 0.742

>>> # Compare two responses
>>> response1 = "Python is a programming language."
>>> response2 = "Python is a high-level, interpreted programming language known for its readability."
>>> comparison = compare_responses(response1, response2, "What is Python?")
>>> print(f"Winner: {comparison['winner']}")  # doctest: +SKIP
Winner: response2

**Example 8: Text Chunking for LLMs**

Split long documents for processing with context-limited models:

>>> from insideLLMs.nlp import (
...     split_by_sentence, split_by_word_count, sliding_window_chunks
... )
>>> long_text = '''First sentence here. Second sentence follows.
... Third sentence continues. Fourth sentence ends.'''

>>> # Split by sentence count with overlap for context
>>> chunks = split_by_sentence(long_text, sentences_per_chunk=2, overlap=1,
...                             use_nltk_for_segmentation=False)
>>> for i, chunk in enumerate(chunks):
...     print(f"Chunk {i}: {chunk[:50]}...")  # doctest: +SKIP
Chunk 0: First sentence here. Second sentence follows....
Chunk 1: Second sentence follows. Third sentence continues...

>>> # Word-based chunking for token limits
>>> text = "one two three four five six seven eight nine ten"
>>> chunks = split_by_word_count(text, words_per_chunk=4, overlap=1)
>>> print(chunks)
['one two three four', 'four five six seven', 'seven eight nine ten']

**Example 9: Feature Extraction for ML**

Generate features for machine learning models:

>>> from insideLLMs.nlp import create_bow, create_tfidf, extract_pos_tags
>>> documents = [
...     "The cat sat on the mat",
...     "The dog ran in the park",
...     "A cat and a dog played"
... ]

>>> # Bag-of-words representation
>>> bow_vectors, vocab = create_bow(documents)
>>> print(f"Vocabulary: {vocab}")  # doctest: +SKIP
Vocabulary: ['a', 'and', 'cat', 'dog', 'in', 'mat', 'on', 'park', 'played', 'ran', 'sat', 'the']

>>> # TF-IDF for more sophisticated features
>>> tfidf_vectors, tfidf_vocab = create_tfidf(documents)
>>> print(f"Document 1 TF-IDF vector length: {len(tfidf_vectors[0])}")  # doctest: +SKIP
Document 1 TF-IDF vector length: 12

>>> # Part-of-speech tagging
>>> pos_tags = extract_pos_tags("The quick brown fox jumps.")  # doctest: +SKIP
>>> print(pos_tags[:3])  # doctest: +SKIP
[('The', 'DET'), ('quick', 'ADJ'), ('brown', 'ADJ')]

**Example 10: Encoding and Transformation**

Encode text for transmission and transform strings:

>>> from insideLLMs.nlp import (
...     encode_base64, decode_base64, url_encode, html_encode,
...     truncate_text, pad_text, mask_pii
... )
>>> # Base64 encoding
>>> encoded = encode_base64("Hello, World!")
>>> print(encoded)
SGVsbG8sIFdvcmxkIQ==
>>> print(decode_base64(encoded))
Hello, World!

>>> # URL encoding for query parameters
>>> query = url_encode("search term with spaces & symbols")
>>> print(query)
search%20term%20with%20spaces%20%26%20symbols

>>> # HTML encoding for safe display
>>> html = html_encode('<script>alert("XSS")</script>')
>>> print(html)
&lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;

>>> # Text truncation for display
>>> long_title = "Understanding Large Language Models: A Comprehensive Guide"
>>> print(truncate_text(long_title, 35))
Understanding Large Language Mod...

>>> # Mask PII in user content
>>> user_input = "Contact me at john@example.com or 555-123-4567"
>>> print(mask_pii(user_input))  # doctest: +SKIP
Contact me at ****************** or ************

Dependencies
------------
Core dependencies (installed with package):
    - Python 3.9+
    - Standard library: ``re``, ``collections``, ``base64``, ``html``, ``urllib``

Optional dependencies (installed on demand):
    - **NLTK**: Required for ``nltk_tokenize``, ``segment_sentences``,
      ``remove_stopwords``, ``stem_words``, ``lemmatize_words``,
      ``sentiment_analysis_basic``, ``detect_language_by_stopwords``,
      ``calculate_readability_flesch_kincaid``. Auto-downloads required data.
    - **spaCy**: Required for ``spacy_tokenize``, ``extract_named_entities``,
      ``extract_entities_by_type``, ``extract_pos_tags``, ``extract_dependencies``,
      ``semantic_similarity_word_embeddings``. Requires language model installation.
    - **scikit-learn**: Required for ``create_bow``, ``create_tfidf``,
      ``cosine_similarity_texts``, ``naive_bayes_classify``, ``svm_classify``,
      ``extract_keywords_tfidf``.
    - **gensim**: Required for ``create_word_embeddings``.

Notes
-----
- All functions are designed to handle edge cases gracefully (empty strings,
  None values, etc.) and return appropriate default values.
- Functions requiring external dependencies will attempt to import them on
  first use and provide helpful error messages if unavailable.
- NLTK data resources are automatically downloaded on first use of functions
  requiring them.
- For production PII handling, consider using specialized libraries like
  ``presidio`` in addition to the ``mask_pii`` function.
- The package is optimized for English text but includes support for multiple
  European languages in language detection and stopword removal.

See Also
--------
insideLLMs.nlp.dependencies : Dependency management utilities for NLP packages.

References
----------
.. [1] Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing
       with Python. O'Reilly Media.
.. [2] Jurafsky, D., & Martin, J. H. (2023). Speech and Language Processing.
       Stanford University.
.. [3] Manning, C. D., Raghavan, P., & Schutze, H. (2008). Introduction to
       Information Retrieval. Cambridge University Press.
"""

# Import functions from submodules to make them available at the package level e.g. from insideLLMs.nlp import clean_text

from .char_level import (
    get_char_frequency,
    get_char_ngrams,
    to_camelcase,
    to_snakecase,
    to_titlecase,
    to_uppercase,
)
from .chunking import (
    sliding_window_chunks,
    split_by_char_count,
    split_by_sentence,
    split_by_word_count,
)
from .classification import naive_bayes_classify, sentiment_analysis_basic, svm_classify
from .encoding import decode_base64, encode_base64, html_decode, html_encode, url_decode, url_encode
from .extraction import (
    extract_emails,
    extract_entities_by_type,
    extract_hashtags,
    extract_ip_addresses,
    extract_mentions,
    extract_named_entities,
    extract_phone_numbers,
    extract_urls,
)
from .feature_extraction import (
    create_bow,
    create_tfidf,
    create_word_embeddings,
    extract_dependencies,
    extract_pos_tags,
)
from .keyword_extraction import extract_keywords_textrank, extract_keywords_tfidf
from .language_detection import detect_language_by_char_ngrams, detect_language_by_stopwords
from .similarity import (
    cosine_similarity_texts,
    hamming_distance,
    jaccard_similarity,
    jaro_similarity,
    jaro_winkler_similarity,
    levenshtein_distance,
    longest_common_subsequence,
    semantic_similarity_word_embeddings,
    word_overlap_similarity,
)
from .text_analysis import (
    ContentAnalysis,
    ReadabilityMetrics,
    ResponseQualityScore,
    TextAnalyzer,
    TextProfile,
    ToneAnalysis,
    analyze_text,
    compare_responses,
    score_response,
)
from .text_cleaning import (
    clean_text,
    normalize_contractions,
    normalize_unicode,
    normalize_whitespace,
    remove_emojis,
    remove_html_tags,
    remove_numbers,
    remove_punctuation,
    remove_urls,
    replace_repeated_chars,
)
from .text_metrics import (
    calculate_avg_sentence_length,
    calculate_avg_word_length,
    calculate_lexical_diversity,
    calculate_readability_flesch_kincaid,
    count_sentences,
    count_syllables,  # Useful helper, might be good to expose
    count_words,
    get_word_frequencies,
)
from .text_transformation import mask_pii, pad_text, replace_words, truncate_text
from .tokenization import (
    get_ngrams,
    lemmatize_words,
    nltk_tokenize,
    remove_stopwords,
    segment_sentences,
    simple_tokenize,
    spacy_tokenize,
    stem_words,
    word_tokenize_regex,
)

# Define __all__ for `from insideLLMs.nlp import *`
__all__ = [
    # text_cleaning
    "remove_html_tags",
    "remove_urls",
    "remove_punctuation",
    "normalize_whitespace",
    "normalize_unicode",
    "remove_emojis",
    "remove_numbers",
    "normalize_contractions",
    "replace_repeated_chars",
    "clean_text",
    # tokenization
    "simple_tokenize",
    "word_tokenize_regex",
    "nltk_tokenize",
    "spacy_tokenize",
    "segment_sentences",
    "get_ngrams",
    "remove_stopwords",
    "stem_words",
    "lemmatize_words",
    # char_level
    "get_char_ngrams",
    "get_char_frequency",
    "to_uppercase",
    "to_titlecase",
    "to_camelcase",
    "to_snakecase",
    # extraction
    "extract_emails",
    "extract_phone_numbers",
    "extract_urls",
    "extract_hashtags",
    "extract_mentions",
    "extract_ip_addresses",
    "extract_named_entities",
    "extract_entities_by_type",
    # text_transformation
    "truncate_text",
    "pad_text",
    "mask_pii",
    "replace_words",
    # feature_extraction
    "create_bow",
    "create_tfidf",
    "create_word_embeddings",
    "extract_pos_tags",
    "extract_dependencies",
    # text_metrics
    "count_words",
    "count_sentences",
    "calculate_avg_word_length",
    "calculate_avg_sentence_length",
    "calculate_lexical_diversity",
    "calculate_readability_flesch_kincaid",
    "count_syllables",
    "get_word_frequencies",
    # classification
    "naive_bayes_classify",
    "svm_classify",
    "sentiment_analysis_basic",
    # similarity
    "cosine_similarity_texts",
    "jaccard_similarity",
    "word_overlap_similarity",
    "levenshtein_distance",
    "semantic_similarity_word_embeddings",
    "jaro_similarity",
    "jaro_winkler_similarity",
    "hamming_distance",
    "longest_common_subsequence",
    # chunking
    "split_by_char_count",
    "split_by_word_count",
    "split_by_sentence",
    "sliding_window_chunks",
    # language_detection
    "detect_language_by_stopwords",
    "detect_language_by_char_ngrams",
    # encoding
    "encode_base64",
    "decode_base64",
    "url_encode",
    "url_decode",
    "html_encode",
    "html_decode",
    # keyword_extraction
    "extract_keywords_tfidf",
    "extract_keywords_textrank",
    # text_analysis
    "ContentAnalysis",
    "ReadabilityMetrics",
    "ResponseQualityScore",
    "TextAnalyzer",
    "TextProfile",
    "ToneAnalysis",
    "analyze_text",
    "compare_responses",
    "score_response",
]
