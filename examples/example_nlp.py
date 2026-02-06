"""Example script to demonstrate usage of NLP utilities."""

from insideLLMs.nlp import (
    calculate_avg_sentence_length,
    calculate_avg_word_length,
    calculate_lexical_diversity,
    calculate_readability_flesch_kincaid,
    # Text cleaning and normalization
    clean_text,
    # Text similarity
    cosine_similarity_texts,
    count_sentences,
    count_words,
    decode_base64,
    detect_language_by_char_ngrams,
    # Text encoding/decoding
    encode_base64,
    # Pattern matching and extraction
    extract_emails,
    extract_entities_by_type,
    extract_hashtags,
    extract_keywords_textrank,
    # Keyword extraction
    extract_keywords_tfidf,
    extract_mentions,
    # Named entity recognition
    extract_named_entities,
    extract_phone_numbers,
    extract_urls,
    get_char_frequency,
    # Character-level operations
    get_char_ngrams,
    get_ngrams,
    get_word_frequencies,
    html_decode,
    html_encode,
    jaccard_similarity,
    jaro_similarity,
    jaro_winkler_similarity,
    lemmatize_words,
    levenshtein_distance,
    mask_pii,
    nltk_tokenize,
    pad_text,
    remove_stopwords,
    segment_sentences,
    semantic_similarity_word_embeddings,
    # Basic text classification
    sentiment_analysis_basic,
    simple_tokenize,
    # Text chunking and splitting
    split_by_char_count,
    split_by_sentence,
    stem_words,
    to_camelcase,
    to_snakecase,
    to_titlecase,
    to_uppercase,
    truncate_text,
    url_decode,
    url_encode,
)


def main():
    # Sample text for demonstration
    sample_text = """
    <p>Natural Language Processing (NLP) is a field of artificial intelligence
    that focuses on the interaction between computers and humans using natural language.
    The ultimate objective of NLP is to read, decipher, understand, and make sense of human languages
    in a valuable way.</p>

    <p>Some major NLP tasks include:</p>
    <ul>
        <li>Text classification</li>
        <li>Named Entity Recognition</li>
        <li>Sentiment Analysis</li>
        <li>Machine Translation</li>
    </ul>

    <p>Companies like Google, Microsoft, and OpenAI are investing heavily in NLP research.
    Visit https://www.openai.com to learn more about recent advancements.</p>
    """

    print("=" * 80)
    print("NLP Utilities Demonstration")
    print("=" * 80)

    # Text cleaning and normalization
    print("\n1. Text Cleaning and Normalization:")
    cleaned_text = clean_text(sample_text)
    print(f"Cleaned text: {cleaned_text[:100]}...")

    # Tokenization and segmentation
    print("\n2. Tokenization and Segmentation:")
    tokens = simple_tokenize(cleaned_text)
    print(f"Simple tokenization (first 10 tokens): {tokens[:10]}")

    try:
        nltk_tokens = nltk_tokenize(cleaned_text)
        print(f"NLTK tokenization (first 10 tokens): {nltk_tokens[:10]}")

        sentences = segment_sentences(cleaned_text)
        print(f"Sentence segmentation (first 2 sentences): {sentences[:2]}")

        bigrams = get_ngrams(tokens[:20], 2)
        print(f"Bigrams (first 5): {bigrams[:5]}")

        tokens_no_stopwords = remove_stopwords(tokens[:20])
        print(f"After stopword removal: {tokens_no_stopwords}")

        stemmed_tokens = stem_words(tokens[:10])
        print(f"Stemmed tokens: {stemmed_tokens}")

        lemmatized_tokens = lemmatize_words(tokens[:10])
        print(f"Lemmatized tokens: {lemmatized_tokens}")
    except ImportError as e:
        print(f"NLTK functions skipped: {e}")

    # Text statistics and metrics
    print("\n3. Text Statistics and Metrics:")
    print(f"Word count: {count_words(cleaned_text)}")
    print(f"Sentence count: {count_sentences(cleaned_text)}")
    print(f"Average word length: {calculate_avg_word_length(cleaned_text):.2f}")
    print(f"Average sentence length: {calculate_avg_sentence_length(cleaned_text):.2f}")
    print(f"Lexical diversity: {calculate_lexical_diversity(cleaned_text):.2f}")

    try:
        print(
            f"Readability (Flesch-Kincaid Grade Level): {calculate_readability_flesch_kincaid(cleaned_text):.2f}"
        )
    except ImportError as e:
        print(f"Readability calculation skipped: {e}")

    word_freq = get_word_frequencies(cleaned_text)
    print(
        f"Top 5 most frequent words: {
            sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        }"
    )

    # Basic text classification
    print("\n4. Basic Text Classification:")
    try:
        sentiment = sentiment_analysis_basic(cleaned_text)
        print(f"Sentiment: {sentiment}")
    except ImportError as e:
        print(f"Sentiment analysis skipped: {e}")

    # Named entity recognition
    print("\n5. Named Entity Recognition:")
    try:
        entities = extract_named_entities(cleaned_text)
        print(f"Named entities: {entities}")

        org_entities = extract_entities_by_type(cleaned_text, ["ORG"])
        print(f"Organizations: {org_entities}")
    except ImportError as e:
        print(f"Named entity recognition skipped: {e}")

    # Keyword extraction
    print("\n6. Keyword Extraction:")
    try:
        keywords_tfidf = extract_keywords_tfidf(cleaned_text)
        print(f"Keywords (TF-IDF): {keywords_tfidf}")

        keywords_textrank = extract_keywords_textrank(cleaned_text)
        print(f"Keywords (TextRank): {keywords_textrank}")
    except ImportError as e:
        print(f"Keyword extraction skipped: {e}")

    # Text similarity
    print("\n7. Text Similarity:")
    text1 = "Natural language processing is a field of artificial intelligence."
    text2 = "NLP is an AI discipline focused on understanding human language."

    try:
        cosine_sim = cosine_similarity_texts(text1, text2)
        print(f"Cosine similarity: {cosine_sim:.4f}")
    except ImportError as e:
        print(f"Cosine similarity calculation skipped: {e}")

    jaccard_sim = jaccard_similarity(text1, text2)
    print(f"Jaccard similarity: {jaccard_sim:.4f}")

    levenshtein_dist = levenshtein_distance(text1, text2)
    print(f"Levenshtein distance: {levenshtein_dist}")

    try:
        jaro_sim = jaro_similarity(text1, text2)
        print(f"Jaro similarity: {jaro_sim:.4f}")

        jaro_winkler_sim = jaro_winkler_similarity(text1, text2)
        print(f"Jaro-Winkler similarity: {jaro_winkler_sim:.4f}")
    except Exception as e:
        print(f"Jaro similarity calculation skipped: {e}")

    try:
        semantic_sim = semantic_similarity_word_embeddings(text1, text2)
        print(f"Semantic similarity: {semantic_sim:.4f}")
    except ImportError as e:
        print(f"Semantic similarity calculation skipped: {e}")

    # Character-level operations
    print("\n8. Character-level Operations:")
    sample_word = "hello"
    print(f"Character n-grams for '{sample_word}': {get_char_ngrams(sample_word, 2)}")
    print(f"Character frequency for '{sample_word}': {get_char_frequency(sample_word)}")
    print(f"Uppercase: {to_uppercase(sample_word)}")
    print(f"Title case: {to_titlecase('hello world')}")
    print(f"Camel case: {to_camelcase('hello world')}")
    print(f"Snake case: {to_snakecase('helloWorld')}")

    # Pattern matching and extraction
    print("\n9. Pattern Matching and Extraction:")
    pattern_text = "Contact us at info@example.com or call +1-234-567-8900. Visit https://example.com. #NLP @nlptools"
    print(f"Emails: {extract_emails(pattern_text)}")
    print(f"Phone numbers: {extract_phone_numbers(pattern_text)}")
    print(f"URLs: {extract_urls(pattern_text)}")
    print(f"Hashtags: {extract_hashtags(pattern_text)}")
    print(f"Mentions: {extract_mentions(pattern_text)}")

    # Text transformation
    print("\n10. Text Transformation:")
    long_text = "This is a very long text that needs to be truncated."
    print(f"Truncated text: {truncate_text(long_text, 20)}")
    print(f"Padded text (right): {pad_text('hello', 10, pad_char='*')}")
    print(f"Padded text (center): {pad_text('hello', 10, pad_char='*', align='center')}")

    pii_text = "My email is john@example.com and my credit card is 1234-5678-9012-3456."
    print(f"Masked PII: {mask_pii(pii_text)}")

    # Text chunking and splitting
    print("\n11. Text Chunking and Splitting:")
    chunk_text = (
        "This is the first sentence. This is the second sentence. This is the third sentence."
    )
    print(f"Split by character count: {split_by_char_count(chunk_text, 20, overlap=5)}")
    print(f"Split by sentence: {split_by_sentence(chunk_text, 2)}")

    # Language detection
    print("\n12. Language Detection:")
    try:
        en_text = "The quick brown fox jumps over the lazy dog."
        es_text = "El zorro marrón rápido salta sobre el perro perezoso."
        fr_text = "Le renard brun rapide saute par-dessus le chien paresseux."

        print(f"English text detected as: {detect_language_by_char_ngrams(en_text)}")
        print(f"Spanish text detected as: {detect_language_by_char_ngrams(es_text)}")
        print(f"French text detected as: {detect_language_by_char_ngrams(fr_text)}")
    except Exception as e:
        print(f"Language detection skipped: {e}")

    # Text encoding/decoding
    print("\n13. Text Encoding/Decoding:")
    original = "Hello, world!"
    encoded_base64 = encode_base64(original)
    print(f"Base64 encoded: {encoded_base64}")
    print(f"Base64 decoded: {decode_base64(encoded_base64)}")

    encoded_url = url_encode("Hello, world!")
    print(f"URL encoded: {encoded_url}")
    print(f"URL decoded: {url_decode(encoded_url)}")

    encoded_html = html_encode("<script>alert('XSS')</script>")
    print(f"HTML encoded: {encoded_html}")
    print(f"HTML decoded: {html_decode(encoded_html)}")

    print("\nNote: Some functions may be skipped if the required packages are not installed.")
    print("To install all dependencies, run: pip install nltk spacy scikit-learn gensim")
    print("For spaCy models, run: python -m spacy download en_core_web_sm")


if __name__ == "__main__":
    main()
