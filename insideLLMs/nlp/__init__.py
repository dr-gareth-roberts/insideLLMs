"""
The insideLLMs.nlp package provides a collection of NLP utilities.
"""

# Import functions from submodules to make them available at the package level e.g. from insideLLMs.nlp import clean_text

from .text_cleaning import (
    remove_html_tags,
    remove_urls,
    remove_punctuation,
    normalize_whitespace,
    normalize_unicode,
    remove_emojis,
    remove_numbers,
    normalize_contractions,
    replace_repeated_chars,
    clean_text
)

from .tokenization import (
    simple_tokenize,
    nltk_tokenize,
    spacy_tokenize,
    segment_sentences,
    get_ngrams,
    remove_stopwords,
    stem_words,
    lemmatize_words
)

from .char_level import (
    get_char_ngrams,
    get_char_frequency,
    to_uppercase,
    to_titlecase,
    to_camelcase,
    to_snakecase
)

from .extraction import (
    extract_emails,
    extract_phone_numbers,
    extract_urls,
    extract_hashtags,
    extract_mentions,
    extract_ip_addresses,
    extract_named_entities,
    extract_entities_by_type
)

from .text_transformation import (
    truncate_text,
    pad_text,
    mask_pii,
    replace_words
)

from .feature_extraction import (
    create_bow,
    create_tfidf,
    create_word_embeddings,
    extract_pos_tags,
    extract_dependencies
)

from .text_metrics import (
    count_words,
    count_sentences,
    calculate_avg_word_length,
    calculate_avg_sentence_length,
    calculate_lexical_diversity,
    calculate_readability_flesch_kincaid,
    count_syllables, # Useful helper, might be good to expose
    get_word_frequencies
)

from .classification import (
    naive_bayes_classify,
    svm_classify,
    sentiment_analysis_basic
)

from .similarity import (
    cosine_similarity_texts,
    jaccard_similarity,
    levenshtein_distance,
    semantic_similarity_word_embeddings,
    jaro_similarity,
    jaro_winkler_similarity,
    hamming_distance,
    longest_common_subsequence
)

from .chunking import (
    split_by_char_count,
    split_by_word_count,
    split_by_sentence,
    sliding_window_chunks
)

from .language_detection import (
    detect_language_by_stopwords,
    detect_language_by_char_ngrams
)

from .encoding import (
    encode_base64,
    decode_base64,
    url_encode,
    url_decode,
    html_encode,
    html_decode
)

from .keyword_extraction import (
    extract_keywords_tfidf,
    extract_keywords_textrank
)

# For misc_utils, users would typically import these directly if needed for checking
# e.g. from insideLLMs.nlp.misc_utils import is_nltk_available
# However, we can expose the main check functions if desired.
# For now, keeping them for direct import from misc_utils.
# from .misc_utils import (
#     is_nltk_available,
#     is_spacy_available,
#     is_sklearn_available,
#     is_gensim_available,
#     get_nltk_resource,
#     get_spacy_model,
#     ensure_nltk,
#     ensure_spacy,
#     ensure_sklearn,
#     ensure_gensim
# )


# Define __all__ for `from insideLLMs.nlp import *`
__all__ = [
    # text_cleaning
    "remove_html_tags", "remove_urls", "remove_punctuation", "normalize_whitespace",
    "normalize_unicode", "remove_emojis", "remove_numbers", "normalize_contractions",
    "replace_repeated_chars", "clean_text",
    # tokenization
    "simple_tokenize", "nltk_tokenize", "spacy_tokenize", "segment_sentences",
    "get_ngrams", "remove_stopwords", "stem_words", "lemmatize_words",
    # char_level
    "get_char_ngrams", "get_char_frequency", "to_uppercase", "to_titlecase",
    "to_camelcase", "to_snakecase",
    # extraction
    "extract_emails", "extract_phone_numbers", "extract_urls", "extract_hashtags",
    "extract_mentions", "extract_ip_addresses", "extract_named_entities", "extract_entities_by_type",
    # text_transformation
    "truncate_text", "pad_text", "mask_pii", "replace_words",
    # feature_extraction
    "create_bow", "create_tfidf", "create_word_embeddings", "extract_pos_tags", "extract_dependencies",
    # text_metrics
    "count_words", "count_sentences", "calculate_avg_word_length", "calculate_avg_sentence_length",
    "calculate_lexical_diversity", "calculate_readability_flesch_kincaid", "count_syllables", 
    "get_word_frequencies",
    # classification
    "naive_bayes_classify", "svm_classify", "sentiment_analysis_basic",
    # similarity
    "cosine_similarity_texts", "jaccard_similarity", "levenshtein_distance", 
    "semantic_similarity_word_embeddings", "jaro_similarity", "jaro_winkler_similarity", 
    "hamming_distance", "longest_common_subsequence",
    # chunking
    "split_by_char_count", "split_by_word_count", "split_by_sentence", "sliding_window_chunks",
    # language_detection
    "detect_language_by_stopwords", "detect_language_by_char_ngrams",
    # encoding
    "encode_base64", "decode_base64", "url_encode", "url_decode", "html_encode", "html_decode",
    # keyword_extraction
    "extract_keywords_tfidf", "extract_keywords_textrank"
]
