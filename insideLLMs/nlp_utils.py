"""Natural Language Processing utilities for text processing and analysis.

This module provides a collection of NLP functions for:
- Text cleaning and normalization
- Tokenization and segmentation
- Feature extraction
- Text statistics and metrics
- Basic text classification
- Named entity recognition
- Keyword extraction
- Text similarity

The module is designed to work with minimal dependencies, but some functions
require optional packages like NLTK, spaCy, scikit-learn, or gensim.
"""
import re
import string
import unicodedata
import base64
import urllib.parse
import html
from typing import List, Dict, Tuple, Set, Optional, Union, Callable, Any, Iterator
from collections import Counter, defaultdict
import math

# Optional dependencies
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = False  # Set to False initially, will be True when a model is loaded
    SPACY_MODEL = None
except ImportError:
    SPACY_AVAILABLE = False
    SPACY_MODEL = None

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import gensim
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# ===== Dependency Management =====

def check_nltk():
    """Check if NLTK is available and download required resources if needed."""
    if not NLTK_AVAILABLE:
        raise ImportError(
            "NLTK is not installed. Please install it with: pip install nltk"
        )

    # Download required NLTK resources if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

def check_spacy(model_name: str = "en_core_web_sm"):
    """Check if spaCy is available and load the specified model if needed.

    Args:
        model_name: Name of the spaCy model to load

    Returns:
        The loaded spaCy model
    """
    global SPACY_AVAILABLE, SPACY_MODEL

    if not SPACY_AVAILABLE:
        raise ImportError(
            "spaCy is not installed. Please install it with: pip install spacy"
        )

    if SPACY_MODEL is None:
        try:
            SPACY_MODEL = spacy.load(model_name)
            SPACY_AVAILABLE = True
        except OSError:
            raise ImportError(
                f"spaCy model '{model_name}' not found. "
                f"Please install it with: python -m spacy download {model_name}"
            )

    return SPACY_MODEL

def check_sklearn():
    """Check if scikit-learn is available."""
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is not installed. Please install it with: pip install scikit-learn"
        )

def check_gensim():
    """Check if gensim is available."""
    if not GENSIM_AVAILABLE:
        raise ImportError(
            "gensim is not installed. Please install it with: pip install gensim"
        )

# ===== Text Cleaning and Normalization =====

def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text.

    Args:
        text: Input text with potential HTML tags

    Returns:
        Text with HTML tags removed
    """
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub('', text)

def remove_urls(text: str) -> str:
    """Remove URLs from text.

    Args:
        text: Input text with potential URLs

    Returns:
        Text with URLs removed
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def remove_punctuation(text: str) -> str:
    """Remove punctuation from text.

    Args:
        text: Input text with punctuation

    Returns:
        Text with punctuation removed
    """
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    Args:
        text: Input text with irregular whitespace

    Returns:
        Text with normalized whitespace
    """
    return ' '.join(text.split())

def normalize_unicode(text: str, form: str = 'NFKC') -> str:
    """Normalize Unicode characters in text.

    Args:
        text: Input text with Unicode characters
        form: Unicode normalization form ('NFC', 'NFKC', 'NFD', 'NFKD')

    Returns:
        Text with normalized Unicode characters
    """
    return unicodedata.normalize(form, text)

def remove_emojis(text: str) -> str:
    """Remove emojis from text.

    Args:
        text: Input text with potential emojis

    Returns:
        Text with emojis removed
    """
    # This pattern matches most emoji characters
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0000257F"  # Enclosed characters
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U00002700-\U000027BF"  # Dingbats
        "\U0000FE00-\U0000FE0F"  # Variation Selectors
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U00002B50"              # Star
        "\U00002B55"              # Circle
        "]",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def remove_numbers(text: str) -> str:
    """Remove numbers from text.

    Args:
        text: Input text with numbers

    Returns:
        Text with numbers removed
    """
    return re.sub(r'\d+', '', text)

def normalize_contractions(text: str) -> str:
    """Normalize common English contractions.

    Args:
        text: Input text with contractions

    Returns:
        Text with expanded contractions
    """
    # Dictionary of common contractions
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    # Create a regular expression pattern for contractions
    pattern = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b', re.IGNORECASE)

    # Function to replace contractions
    def replace(match):
        word = match.group(0)
        return contractions.get(word.lower(), word)

    return pattern.sub(replace, text)

def replace_repeated_chars(text: str, threshold: int = 2) -> str:
    """Replace repeated characters with a single occurrence if they repeat more than threshold times.

    Args:
        text: Input text with potentially repeated characters
        threshold: Maximum number of allowed repetitions

    Returns:
        Text with reduced character repetitions
    """
    pattern = re.compile(r'(.)\1{' + str(threshold) + ',}')
    return pattern.sub(lambda m: m.group(1) * threshold, text)

def clean_text(text: str,
               remove_html: bool = True,
               remove_url: bool = True,
               remove_punct: bool = False,
               remove_emoji: bool = False,
               remove_num: bool = False,
               normalize_white: bool = True,
               normalize_unicode_form: Optional[str] = 'NFKC',
               normalize_contraction: bool = False,
               replace_repeated: bool = False,
               repeated_threshold: int = 2,
               lowercase: bool = True) -> str:
    """Clean text by applying multiple cleaning operations.

    Args:
        text: Input text to clean
        remove_html: Whether to remove HTML tags
        remove_url: Whether to remove URLs
        remove_punct: Whether to remove punctuation
        remove_emoji: Whether to remove emojis
        remove_num: Whether to remove numbers
        normalize_white: Whether to normalize whitespace
        normalize_unicode_form: Unicode normalization form (None to skip)
        normalize_contraction: Whether to expand contractions
        replace_repeated: Whether to replace repeated characters
        repeated_threshold: Threshold for repeated character replacement
        lowercase: Whether to convert text to lowercase

    Returns:
        Cleaned text
    """
    if remove_html:
        text = remove_html_tags(text)

    if remove_url:
        text = remove_urls(text)

    if normalize_unicode_form:
        text = normalize_unicode(text, normalize_unicode_form)

    if remove_emoji:
        text = remove_emojis(text)

    if remove_num:
        text = remove_numbers(text)

    if remove_punct:
        text = remove_punctuation(text)

    if normalize_contraction:
        text = normalize_contractions(text)

    if replace_repeated:
        text = replace_repeated_chars(text, repeated_threshold)

    if normalize_white:
        text = normalize_whitespace(text)

    if lowercase:
        text = text.lower()

    return text

# ===== Tokenization and Segmentation =====

def simple_tokenize(text: str) -> List[str]:
    """Simple word tokenization by splitting on whitespace.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens
    """
    return text.split()

def nltk_tokenize(text: str) -> List[str]:
    """Tokenize text using NLTK's word_tokenize.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens
    """
    check_nltk()
    return word_tokenize(text)

def spacy_tokenize(text: str, model_name: str = "en_core_web_sm") -> List[str]:
    """Tokenize text using spaCy.

    Args:
        text: Input text to tokenize
        model_name: Name of the spaCy model to use

    Returns:
        List of tokens
    """
    nlp = check_spacy(model_name)
    doc = nlp(text)
    return [token.text for token in doc]

def segment_sentences(text: str, use_nltk: bool = True) -> List[str]:
    """Split text into sentences.

    Args:
        text: Input text to segment
        use_nltk: Whether to use NLTK (True) or a simple regex (False)

    Returns:
        List of sentences
    """
    if use_nltk:
        check_nltk()
        return sent_tokenize(text)
    else:
        # Simple regex-based sentence segmentation
        pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
        return pattern.split(text)

def get_ngrams(tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
    """Generate n-grams from a list of tokens.

    Args:
        tokens: List of tokens
        n: Size of n-grams

    Returns:
        List of n-grams as tuples
    """
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def remove_stopwords(tokens: List[str], language: str = 'english') -> List[str]:
    """Remove stopwords from a list of tokens.

    Args:
        tokens: List of tokens
        language: Language for stopwords

    Returns:
        List of tokens with stopwords removed
    """
    check_nltk()
    stop_words = set(stopwords.words(language))
    return [token for token in tokens if token.lower() not in stop_words]

def stem_words(tokens: List[str]) -> List[str]:
    """Apply stemming to a list of tokens using Porter stemmer.

    Args:
        tokens: List of tokens

    Returns:
        List of stemmed tokens
    """
    check_nltk()
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def lemmatize_words(tokens: List[str]) -> List[str]:
    """Apply lemmatization to a list of tokens using WordNet lemmatizer.

    Args:
        tokens: List of tokens

    Returns:
        List of lemmatized tokens
    """
    check_nltk()
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# ===== Feature Extraction =====

def create_bow(texts: List[str], max_features: Optional[int] = None) -> Tuple[List[List[int]], List[str]]:
    """Create bag-of-words representation of texts.

    Args:
        texts: List of text documents
        max_features: Maximum number of features (words)

    Returns:
        Tuple of (document-term matrix, feature names)
    """
    check_sklearn()
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X.toarray().tolist(), vectorizer.get_feature_names_out().tolist()

def create_tfidf(texts: List[str], max_features: Optional[int] = None) -> Tuple[List[List[float]], List[str]]:
    """Create TF-IDF representation of texts.

    Args:
        texts: List of text documents
        max_features: Maximum number of features (words)

    Returns:
        Tuple of (document-term matrix, feature names)
    """
    check_sklearn()
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X.toarray().tolist(), vectorizer.get_feature_names_out().tolist()

def create_word_embeddings(sentences: List[List[str]],
                          vector_size: int = 100,
                          window: int = 5,
                          min_count: int = 1) -> Dict[str, List[float]]:
    """Create word embeddings using Word2Vec.

    Args:
        sentences: List of tokenized sentences
        vector_size: Dimensionality of the word vectors
        window: Maximum distance between current and predicted word
        min_count: Minimum count of words to consider

    Returns:
        Dictionary mapping words to their vector representations
    """
    check_gensim()
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count)
    return {word: model.wv[word].tolist() for word in model.wv.index_to_key}

def extract_pos_tags(text: str, model_name: str = "en_core_web_sm") -> List[Tuple[str, str]]:
    """Extract part-of-speech tags from text using spaCy.

    Args:
        text: Input text
        model_name: Name of the spaCy model to use

    Returns:
        List of (token, POS tag) tuples
    """
    nlp = check_spacy(model_name)
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def extract_dependencies(text: str, model_name: str = "en_core_web_sm") -> List[Tuple[str, str, str]]:
    """Extract dependency relations from text using spaCy.

    Args:
        text: Input text
        model_name: Name of the spaCy model to use

    Returns:
        List of (token, dependency relation, head) tuples
    """
    nlp = check_spacy(model_name)
    doc = nlp(text)
    return [(token.text, token.dep_, token.head.text) for token in doc]

# ===== Text Statistics and Metrics =====

def count_words(text: str, tokenizer: Callable = simple_tokenize) -> int:
    """Count the number of words in text.

    Args:
        text: Input text
        tokenizer: Function to tokenize text

    Returns:
        Number of words
    """
    tokens = tokenizer(text)
    return len(tokens)

def count_sentences(text: str) -> int:
    """Count the number of sentences in text.

    Args:
        text: Input text

    Returns:
        Number of sentences
    """
    sentences = segment_sentences(text)
    return len(sentences)

def calculate_avg_word_length(text: str, tokenizer: Callable = simple_tokenize) -> float:
    """Calculate the average word length in text.

    Args:
        text: Input text
        tokenizer: Function to tokenize text

    Returns:
        Average word length
    """
    tokens = tokenizer(text)
    if not tokens:
        return 0.0
    return sum(len(token) for token in tokens) / len(tokens)

def calculate_avg_sentence_length(text: str, tokenizer: Callable = simple_tokenize) -> float:
    """Calculate the average sentence length (in words) in text.

    Args:
        text: Input text
        tokenizer: Function to tokenize text

    Returns:
        Average sentence length
    """
    sentences = segment_sentences(text)
    if not sentences:
        return 0.0

    sentence_lengths = [len(tokenizer(sentence)) for sentence in sentences]
    return sum(sentence_lengths) / len(sentences)

def calculate_lexical_diversity(text: str, tokenizer: Callable = simple_tokenize) -> float:
    """Calculate the lexical diversity (type-token ratio) of text.

    Args:
        text: Input text
        tokenizer: Function to tokenize text

    Returns:
        Lexical diversity score
    """
    tokens = tokenizer(text)
    if not tokens:
        return 0.0

    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens)

def calculate_readability_flesch_kincaid(text: str) -> float:
    """Calculate the Flesch-Kincaid Grade Level readability score.

    Args:
        text: Input text

    Returns:
        Flesch-Kincaid Grade Level score
    """
    sentences = segment_sentences(text)
    if not sentences:
        return 0.0

    words = nltk_tokenize(text)
    if not words:
        return 0.0

    syllables = sum(count_syllables(word) for word in words)

    # Flesch-Kincaid Grade Level formula
    return 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59

def count_syllables(word: str) -> int:
    """Count the number of syllables in a word.

    This is a simple heuristic and may not be accurate for all words.

    Args:
        word: Input word

    Returns:
        Estimated number of syllables
    """
    word = word.lower()
    # Remove non-alphanumeric characters
    word = re.sub(r'[^a-z]', '', word)

    # Count vowel groups
    if not word:
        return 0

    # Count vowel sequences as syllables
    count = len(re.findall(r'[aeiouy]+', word))

    # Adjust for common patterns
    if word.endswith('e'):
        count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in 'aeiouy':
        count += 1
    if count == 0:
        count = 1

    return count

def get_word_frequencies(text: str, tokenizer: Callable = simple_tokenize) -> Dict[str, int]:
    """Get word frequencies in text.

    Args:
        text: Input text
        tokenizer: Function to tokenize text

    Returns:
        Dictionary mapping words to their frequencies
    """
    tokens = tokenizer(text)
    return dict(Counter(tokens))

# ===== Basic Text Classification =====

def naive_bayes_classify(train_texts: List[str],
                        train_labels: List[str],
                        test_texts: List[str]) -> List[str]:
    """Classify texts using Naive Bayes.

    Args:
        train_texts: List of training texts
        train_labels: List of training labels
        test_texts: List of texts to classify

    Returns:
        List of predicted labels for test_texts
    """
    check_sklearn()
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    pipeline.fit(train_texts, train_labels)
    return pipeline.predict(test_texts).tolist()

def svm_classify(train_texts: List[str],
                train_labels: List[str],
                test_texts: List[str]) -> List[str]:
    """Classify texts using Support Vector Machine.

    Args:
        train_texts: List of training texts
        train_labels: List of training labels
        test_texts: List of texts to classify

    Returns:
        List of predicted labels for test_texts
    """
    check_sklearn()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LinearSVC())
    ])

    pipeline.fit(train_texts, train_labels)
    return pipeline.predict(test_texts).tolist()

def sentiment_analysis_basic(text: str) -> str:
    """Perform basic sentiment analysis using a lexicon-based approach.

    Args:
        text: Input text

    Returns:
        Sentiment label ('positive', 'negative', or 'neutral')
    """
    check_nltk()
    from nltk.sentiment import SentimentIntensityAnalyzer

    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)

    if sentiment_scores['compound'] >= 0.05:
        return 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# ===== Named Entity Recognition =====

def extract_named_entities(text: str, model_name: str = "en_core_web_sm") -> List[Tuple[str, str]]:
    """Extract named entities from text using spaCy.

    Args:
        text: Input text
        model_name: Name of the spaCy model to use

    Returns:
        List of (entity text, entity type) tuples
    """
    nlp = check_spacy(model_name)
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_entities_by_type(text: str, entity_types: List[str], model_name: str = "en_core_web_sm") -> Dict[str, List[str]]:
    """Extract named entities of specific types from text.

    Args:
        text: Input text
        entity_types: List of entity types to extract (e.g., ['PERSON', 'ORG'])
        model_name: Name of the spaCy model to use

    Returns:
        Dictionary mapping entity types to lists of entities
    """
    entities = extract_named_entities(text, model_name)
    result = defaultdict(list)

    for entity, entity_type in entities:
        if entity_type in entity_types:
            result[entity_type].append(entity)

    return dict(result)

# ===== Keyword Extraction =====

def extract_keywords_tfidf(text: str, num_keywords: int = 5) -> List[str]:
    """Extract keywords from text using TF-IDF.

    Args:
        text: Input text
        num_keywords: Number of keywords to extract

    Returns:
        List of extracted keywords
    """
    check_sklearn()

    # Split text into sentences
    sentences = segment_sentences(text)

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Calculate average TF-IDF score for each word
    word_scores = {}
    for i in range(len(sentences)):
        feature_index = tfidf_matrix[i, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
        for idx, score in tfidf_scores:
            word = feature_names[idx]
            word_scores[word] = word_scores.get(word, 0) + score

    # Sort words by score and return top keywords
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_words[:num_keywords]]

def extract_keywords_textrank(text: str, num_keywords: int = 5, window_size: int = 4) -> List[str]:
    """Extract keywords from text using TextRank algorithm.

    Args:
        text: Input text
        num_keywords: Number of keywords to extract
        window_size: Window size for co-occurrence

    Returns:
        List of extracted keywords
    """
    check_nltk()

    # Tokenize and remove stopwords
    tokens = nltk_tokenize(text.lower())
    tokens = remove_stopwords(tokens)

    # Build co-occurrence graph
    graph = defaultdict(lambda: defaultdict(int))
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i+window_size]
        for j in range(len(window)):
            for k in range(j+1, len(window)):
                graph[window[j]][window[k]] += 1
                graph[window[k]][window[j]] += 1

    # Run TextRank algorithm
    ranks = {node: 1.0 for node in graph}
    num_iterations = 10
    damping = 0.85

    for _ in range(num_iterations):
        for node in graph:
            rank_sum = 0
            for neighbor in graph[node]:
                neighbor_links_sum = sum(graph[neighbor].values())
                if neighbor_links_sum > 0:
                    rank_sum += graph[node][neighbor] / neighbor_links_sum * ranks[neighbor]

            ranks[node] = (1 - damping) + damping * rank_sum

    # Sort words by rank and return top keywords
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return [word for word, rank in sorted_ranks[:num_keywords]]

# ===== Pattern Matching and Extraction =====

def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text.

    Args:
        text: Input text

    Returns:
        List of extracted email addresses
    """
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    return email_pattern.findall(text)

def extract_phone_numbers(text: str, country: str = 'us') -> List[str]:
    """Extract phone numbers from text.

    Args:
        text: Input text
        country: Country code for phone number format ('us', 'uk', 'international')

    Returns:
        List of extracted phone numbers
    """
    patterns = {
        'us': r'(?:\+?1[-\s]?)?(?:\(?[0-9]{3}\)?[-\s]?)?[0-9]{3}[-\s]?[0-9]{4}',
        'uk': r'(?:\+?44[-\s]?)?(?:\(?0[0-9]{1,4}\)?[-\s]?)?[0-9]{3,4}[-\s]?[0-9]{3,4}',
        'international': r'\+?[0-9]{1,3}[-\s]?[0-9]{1,4}[-\s]?[0-9]{1,4}[-\s]?[0-9]{1,9}'
    }

    pattern = patterns.get(country.lower(), patterns['international'])
    phone_pattern = re.compile(pattern)
    return phone_pattern.findall(text)

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text.

    Args:
        text: Input text

    Returns:
        List of extracted URLs
    """
    url_pattern = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*|'
        r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
    )
    return url_pattern.findall(text)

def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text.

    Args:
        text: Input text

    Returns:
        List of extracted hashtags (including the # symbol)
    """
    hashtag_pattern = re.compile(r'#\w+')
    return hashtag_pattern.findall(text)

def extract_mentions(text: str) -> List[str]:
    """Extract mentions from text.

    Args:
        text: Input text

    Returns:
        List of extracted mentions (including the @ symbol)
    """
    mention_pattern = re.compile(r'@\w+')
    return mention_pattern.findall(text)

def extract_ip_addresses(text: str) -> List[str]:
    """Extract IP addresses from text.

    Args:
        text: Input text

    Returns:
        List of extracted IP addresses
    """
    ipv4_pattern = re.compile(
        r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
        r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    )
    return ipv4_pattern.findall(text)

# ===== Character-Level Operations =====

def get_char_ngrams(text: str, n: int = 2) -> List[str]:
    """Generate character n-grams from text.

    Args:
        text: Input text
        n: Size of n-grams

    Returns:
        List of character n-grams
    """
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def get_char_frequency(text: str) -> Dict[str, int]:
    """Get character frequencies in text.

    Args:
        text: Input text

    Returns:
        Dictionary mapping characters to their frequencies
    """
    return dict(Counter(text))

def to_uppercase(text: str) -> str:
    """Convert text to uppercase.

    Args:
        text: Input text

    Returns:
        Uppercase text
    """
    return text.upper()

def to_titlecase(text: str) -> str:
    """Convert text to title case.

    Args:
        text: Input text

    Returns:
        Title case text
    """
    return text.title()

def to_camelcase(text: str) -> str:
    """Convert text to camel case.

    Args:
        text: Input text with spaces or underscores

    Returns:
        Camel case text
    """
    # Replace underscores with spaces, then split by spaces
    words = text.replace('_', ' ').split()
    if not words:
        return ''

    # First word lowercase, rest title case
    return words[0].lower() + ''.join(word.title() for word in words[1:])

def to_snakecase(text: str) -> str:
    """Convert text to snake case.

    Args:
        text: Input text

    Returns:
        Snake case text
    """
    # Replace spaces and hyphens with underscores
    text = re.sub(r'[ -]', '_', text)
    # Handle camel case
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9_]', '', text).lower()
    # Replace multiple underscores with a single one
    text = re.sub(r'_+', '_', text)
    return text

# ===== Text Transformation =====

def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """Truncate text to a maximum length.

    Args:
        text: Input text
        max_length: Maximum length of the output text
        add_ellipsis: Whether to add an ellipsis (...) if text is truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    if add_ellipsis and max_length > 3:
        return text[:max_length-3] + '...'
    else:
        return text[:max_length]

def pad_text(text: str, length: int, pad_char: str = ' ', align: str = 'left') -> str:
    """Pad text to a specified length.

    Args:
        text: Input text
        length: Desired length of the output text
        pad_char: Character to use for padding
        align: Alignment of the text ('left', 'right', 'center')

    Returns:
        Padded text
    """
    if len(text) >= length:
        return text

    if align == 'left':
        return text + pad_char * (length - len(text))
    elif align == 'right':
        return pad_char * (length - len(text)) + text
    elif align == 'center':
        left_pad = (length - len(text)) // 2
        right_pad = length - len(text) - left_pad
        return pad_char * left_pad + text + pad_char * right_pad
    else:
        raise ValueError("align must be 'left', 'right', or 'center'")

def mask_pii(text: str, mask_char: str = '*') -> str:
    """Mask personally identifiable information (PII) in text.

    Args:
        text: Input text
        mask_char: Character to use for masking

    Returns:
        Text with masked PII
    """
    # Mask email addresses
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    text = email_pattern.sub(lambda m: mask_char * len(m.group(0)), text)

    # Mask phone numbers (simple pattern)
    phone_pattern = re.compile(r'\+?[0-9][\s\-\(\)0-9]{6,}')
    text = phone_pattern.sub(lambda m: mask_char * len(m.group(0)), text)

    # Mask credit card numbers
    cc_pattern = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
    text = cc_pattern.sub(lambda m: mask_char * len(m.group(0)), text)

    # Mask SSN (US Social Security Numbers)
    ssn_pattern = re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b')
    text = ssn_pattern.sub(lambda m: mask_char * len(m.group(0)), text)

    return text

def replace_words(text: str, replacements: Dict[str, str], case_sensitive: bool = False) -> str:
    """Replace specific words in text.

    Args:
        text: Input text
        replacements: Dictionary mapping words to their replacements
        case_sensitive: Whether to perform case-sensitive replacement

    Returns:
        Text with replaced words
    """
    if not case_sensitive:
        # Create a regex pattern that matches any of the words to replace
        pattern = re.compile('\\b(' + '|'.join(map(re.escape, replacements.keys())) + ')\\b', re.IGNORECASE)

        # Function to get the replacement with proper case
        def replace(match):
            word = match.group(0)
            replacement = replacements.get(word.lower(), word)

            # Preserve case of the original word
            if word.islower():
                return replacement.lower()
            elif word.isupper():
                return replacement.upper()
            elif word[0].isupper():
                return replacement.capitalize()
            else:
                return replacement

        return pattern.sub(replace, text)
    else:
        # For case-sensitive replacement, use a simpler approach
        pattern = re.compile('\\b(' + '|'.join(map(re.escape, replacements.keys())) + ')\\b')
        return pattern.sub(lambda m: replacements[m.group(0)], text)

# ===== Text Similarity =====

def cosine_similarity_texts(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts using TF-IDF.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Cosine similarity score
    """
    check_sklearn()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return float(similarity[0][0])

def jaccard_similarity(text1: str, text2: str, tokenizer: Callable = simple_tokenize) -> float:
    """Calculate Jaccard similarity between two texts.

    Args:
        text1: First text
        text2: Second text
        tokenizer: Function to tokenize text

    Returns:
        Jaccard similarity score
    """
    tokens1 = set(tokenizer(text1))
    tokens2 = set(tokenizer(text2))

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    if not union:
        return 0.0

    return len(intersection) / len(union)

def levenshtein_distance(text1: str, text2: str) -> int:
    """Calculate Levenshtein (edit) distance between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Levenshtein distance
    """
    if len(text1) < len(text2):
        return levenshtein_distance(text2, text1)

    if not text2:
        return len(text1)

    previous_row = range(len(text2) + 1)
    for i, c1 in enumerate(text1):
        current_row = [i + 1]
        for j, c2 in enumerate(text2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def semantic_similarity_word_embeddings(text1: str,
                                       text2: str,
                                       model_name: str = "en_core_web_sm") -> float:
    """Calculate semantic similarity between two texts using word embeddings.

    Args:
        text1: First text
        text2: Second text
        model_name: Name of the spaCy model to use

    Returns:
        Semantic similarity score
    """
    nlp = check_spacy(model_name)
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    if not doc1.vector.any() or not doc2.vector.any():
        return 0.0

    return doc1.similarity(doc2)

def jaro_similarity(s1: str, s2: str) -> float:
    """Calculate Jaro similarity between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Jaro similarity score (0-1)
    """
    # If both strings are empty, return 1.0
    if not s1 and not s2:
        return 1.0

    # If one string is empty, return 0.0
    if not s1 or not s2:
        return 0.0

    # If strings are identical, return 1.0
    if s1 == s2:
        return 1.0

    len_s1, len_s2 = len(s1), len(s2)

    # Maximum distance for matching characters
    match_distance = max(len_s1, len_s2) // 2 - 1
    match_distance = max(0, match_distance)  # Ensure non-negative

    # Arrays to track matched characters
    s1_matches = [False] * len_s1
    s2_matches = [False] * len_s2

    # Count matching characters
    matches = 0
    for i in range(len_s1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_s2)

        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    # If no matches, return 0.0
    if matches == 0:
        return 0.0

    # Count transpositions
    transpositions = 0
    k = 0
    for i in range(len_s1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    transpositions = transpositions // 2

    # Calculate Jaro similarity
    return (matches / len_s1 + matches / len_s2 + (matches - transpositions) / matches) / 3.0

def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """Calculate Jaro-Winkler similarity between two strings.

    Args:
        s1: First string
        s2: Second string
        p: Scaling factor for prefix matches (default: 0.1)

    Returns:
        Jaro-Winkler similarity score (0-1)
    """
    # Calculate Jaro similarity
    jaro_sim = jaro_similarity(s1, s2)

    # Calculate length of common prefix (up to 4 characters)
    prefix_len = 0
    max_prefix = min(4, min(len(s1), len(s2)))
    for i in range(max_prefix):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    # Calculate Jaro-Winkler similarity
    return jaro_sim + (prefix_len * p * (1 - jaro_sim))

def hamming_distance(s1: str, s2: str) -> int:
    """Calculate Hamming distance between two strings of equal length.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Hamming distance
    """
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length")

    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def longest_common_subsequence(s1: str, s2: str) -> str:
    """Find the longest common subsequence of two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Longest common subsequence
    """
    len_s1, len_s2 = len(s1), len(s2)

    # Create a table to store lengths of LCS for all subproblems
    dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

    # Fill the dp table
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Reconstruct the LCS
    lcs = []
    i, j = len_s1, len_s2
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            lcs.append(s1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    return ''.join(reversed(lcs))

# ===== Text Chunking and Splitting =====

def split_by_char_count(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """Split text into chunks of specified character length.

    Args:
        text: Input text
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

def split_by_word_count(text: str, words_per_chunk: int, overlap: int = 0, tokenizer: Callable = simple_tokenize) -> List[str]:
    """Split text into chunks with specified number of words.

    Args:
        text: Input text
        words_per_chunk: Maximum number of words per chunk
        overlap: Number of overlapping words between chunks
        tokenizer: Function to tokenize text

    Returns:
        List of text chunks
    """
    if words_per_chunk <= 0:
        raise ValueError("words_per_chunk must be positive")

    if overlap >= words_per_chunk:
        raise ValueError("overlap must be less than words_per_chunk")

    tokens = tokenizer(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    tokens_len = len(tokens)

    while start < tokens_len:
        end = min(start + words_per_chunk, tokens_len)
        chunks.append(' '.join(tokens[start:end]))
        start += words_per_chunk - overlap

    return chunks

def split_by_sentence(text: str, sentences_per_chunk: int, overlap: int = 0) -> List[str]:
    """Split text into chunks with specified number of sentences.

    Args:
        text: Input text
        sentences_per_chunk: Maximum number of sentences per chunk
        overlap: Number of overlapping sentences between chunks

    Returns:
        List of text chunks
    """
    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be positive")

    if overlap >= sentences_per_chunk:
        raise ValueError("overlap must be less than sentences_per_chunk")

    sentences = segment_sentences(text)
    if not sentences:
        return []

    chunks = []
    start = 0
    sentences_len = len(sentences)

    while start < sentences_len:
        end = min(start + sentences_per_chunk, sentences_len)
        chunks.append(' '.join(sentences[start:end]))
        start += sentences_per_chunk - overlap

    return chunks

def sliding_window_chunks(text: str, window_size: int, step_size: int, tokenizer: Callable = simple_tokenize) -> List[str]:
    """Create sliding window chunks of text.

    Args:
        text: Input text
        window_size: Size of the window in tokens
        step_size: Number of tokens to slide the window
        tokenizer: Function to tokenize text

    Returns:
        List of text chunks
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    if step_size <= 0:
        raise ValueError("step_size must be positive")

    tokens = tokenizer(text)
    if len(tokens) < window_size:
        return [text]

    chunks = []
    for i in range(0, len(tokens) - window_size + 1, step_size):
        chunks.append(' '.join(tokens[i:i+window_size]))

    return chunks

# ===== Language Detection =====

def detect_language_by_stopwords(text: str) -> str:
    """Detect language based on stopword frequency.

    Args:
        text: Input text

    Returns:
        Detected language code (e.g., 'en', 'es', 'fr')
    """
    check_nltk()

    # List of languages to check
    languages = ['english', 'spanish', 'french', 'german', 'italian', 'portuguese', 'dutch']
    language_codes = {'english': 'en', 'spanish': 'es', 'french': 'fr', 'german': 'de',
                     'italian': 'it', 'portuguese': 'pt', 'dutch': 'nl'}

    # Ensure stopwords are downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Tokenize text
    tokens = nltk_tokenize(text.lower())
    if not tokens:
        return 'unknown'

    # Count stopwords for each language
    language_scores = {}
    for lang in languages:
        try:
            stop_words = set(stopwords.words(lang))
            count = sum(1 for token in tokens if token in stop_words)
            language_scores[lang] = count / len(tokens)
        except:
            language_scores[lang] = 0

    # Get language with highest score
    if not language_scores:
        return 'unknown'

    best_language = max(language_scores.items(), key=lambda x: x[1])

    # Return unknown if score is too low
    if best_language[1] < 0.05:
        return 'unknown'

    return language_codes.get(best_language[0], 'unknown')

def detect_language_by_char_ngrams(text: str) -> str:
    """Detect language based on character n-gram frequency profiles.

    This is a simple implementation that works for major European languages.

    Args:
        text: Input text

    Returns:
        Detected language code (e.g., 'en', 'es', 'fr')
    """
    # Language profiles (most common trigrams)
    profiles = {
        'en': ['the', 'and', 'ing', 'ion', 'tio', 'ent', 'ati', 'for', 'her', 'ter'],
        'es': ['que', 'ión', 'nte', 'con', 'est', 'ent', 'ado', 'par', 'los', 'ien'],
        'fr': ['les', 'ent', 'que', 'une', 'our', 'ant', 'des', 'men', 'tio', 'ion'],
        'de': ['ein', 'die', 'und', 'der', 'sch', 'ich', 'nde', 'den', 'che', 'gen'],
        'it': ['che', 'non', 'per', 'del', 'ent', 'ion', 'con', 'ato', 'gli', 'ell'],
        'pt': ['que', 'ent', 'ção', 'não', 'com', 'est', 'ado', 'par', 'ara', 'uma'],
        'nl': ['een', 'het', 'oor', 'nde', 'van', 'aar', 'eer', 'ing', 'ijk', 'sch']
    }

    # Clean and normalize text
    text = text.lower()
    text = re.sub(r'[^a-z]', '', text)

    if len(text) < 20:  # Text too short
        return 'unknown'

    # Get trigrams from text
    trigrams = [text[i:i+3] for i in range(len(text) - 2)]
    trigram_freq = Counter(trigrams)
    most_common = [t for t, _ in trigram_freq.most_common(20)]

    # Calculate similarity with each language profile
    scores = {}
    for lang, profile in profiles.items():
        # Jaccard similarity between text trigrams and language profile
        intersection = len(set(most_common) & set(profile))
        union = len(set(most_common) | set(profile))
        scores[lang] = intersection / union if union > 0 else 0

    # Get language with highest score
    if not scores:
        return 'unknown'

    best_language = max(scores.items(), key=lambda x: x[1])

    # Return unknown if score is too low
    if best_language[1] < 0.1:
        return 'unknown'

    return best_language[0]

# ===== Text Encoding/Decoding =====

def encode_base64(text: str) -> str:
    """Encode text to Base64.

    Args:
        text: Input text

    Returns:
        Base64 encoded string
    """
    return base64.b64encode(text.encode('utf-8')).decode('utf-8')

def decode_base64(encoded: str) -> str:
    """Decode Base64 to text.

    Args:
        encoded: Base64 encoded string

    Returns:
        Decoded text
    """
    return base64.b64decode(encoded.encode('utf-8')).decode('utf-8')

def url_encode(text: str) -> str:
    """URL encode text.

    Args:
        text: Input text

    Returns:
        URL encoded string
    """
    return urllib.parse.quote(text)

def url_decode(encoded: str) -> str:
    """URL decode text.

    Args:
        encoded: URL encoded string

    Returns:
        Decoded text
    """
    return urllib.parse.unquote(encoded)

def html_encode(text: str) -> str:
    """HTML encode text (convert special characters to HTML entities).

    Args:
        text: Input text

    Returns:
        HTML encoded string
    """
    return html.escape(text)

def html_decode(encoded: str) -> str:
    """HTML decode text (convert HTML entities to characters).

    Args:
        encoded: HTML encoded string

    Returns:
        Decoded text
    """
    return html.unescape(encoded)
