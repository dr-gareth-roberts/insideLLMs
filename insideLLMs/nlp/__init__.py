"""
The insideLLMs.nlp package provides a collection of NLP utilities.
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
