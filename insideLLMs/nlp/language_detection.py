import re
from collections import Counter
from typing import List # Added for type hinting if needed later, though not strictly for current functions

# Optional dependencies
try:
    import nltk
    from nltk.tokenize import word_tokenize as nltk_word_tokenize_for_lang_detect
    from nltk.corpus import stopwords as nltk_stopwords_for_lang_detect
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# ===== Dependency Management =====

def check_nltk_for_lang_detect():
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

# ===== Language Detection =====

def detect_language_by_stopwords(text: str) -> str:
    """Detect language based on stopword frequency.

    Args:
        text: Input text

    Returns:
        Detected language code (e.g., 'en', 'es', 'fr')
    """
    check_nltk_for_lang_detect()

    # List of languages to check
    languages = ['english', 'spanish', 'french', 'german', 'italian', 'portuguese', 'dutch']
    language_codes = {'english': 'en', 'spanish': 'es', 'french': 'fr', 'german': 'de',
                     'italian': 'it', 'portuguese': 'pt', 'dutch': 'nl'}

    # Tokenize text
    tokens = nltk_word_tokenize_for_lang_detect(text.lower())
    if not tokens:
        return 'unknown'

    # Count stopwords for each language
    language_scores = {}
    for lang in languages:
        try:
            # Ensure stopwords for the language are available before trying to use them
            current_stopwords = set(nltk_stopwords_for_lang_detect.words(lang))
            count = sum(1 for token in tokens if token in current_stopwords)
            # Normalize score by number of tokens to avoid bias towards longer texts
            language_scores[lang] = count / len(tokens) if len(tokens) > 0 else 0
        except IOError: # Some languages might not have stopwords data initially
            language_scores[lang] = 0
        except Exception: # Catch any other potential errors during stopword processing for a language
            language_scores[lang] = 0


    # Get language with highest score
    if not language_scores or all(score == 0 for score in language_scores.values()):
        return 'unknown'

    best_language_item = max(language_scores.items(), key=lambda x: x[1])
    best_language_name = best_language_item[0]
    best_score = best_language_item[1]


    # Return unknown if score is too low (threshold can be adjusted)
    if best_score < 0.05: # Adjusted threshold slightly based on common practice
        return 'unknown'

    return language_codes.get(best_language_name, 'unknown')

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
    text_cleaned = text.lower()
    text_cleaned = re.sub(r'[^a-z\s]', '', text_cleaned) # Keep spaces to potentially help with n-gram boundaries
    text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip() # Normalize spaces


    if len(text_cleaned) < 20:  # Text too short
        return 'unknown'

    # Get trigrams from text
    trigrams = [text_cleaned[i:i+3] for i in range(len(text_cleaned) - 2)]
    if not trigrams:
        return 'unknown'
        
    trigram_freq = Counter(trigrams)
    most_common_text_trigrams = [t for t, _ in trigram_freq.most_common(20)]

    # Calculate similarity with each language profile
    scores = {}
    for lang, profile_trigrams in profiles.items():
        # Jaccard similarity between text trigrams and language profile
        intersection = len(set(most_common_text_trigrams) & set(profile_trigrams))
        union = len(set(most_common_text_trigrams) | set(profile_trigrams))
        scores[lang] = intersection / union if union > 0 else 0

    # Get language with highest score
    if not scores or all(score == 0 for score in scores.values()):
        return 'unknown'

    best_language_item = max(scores.items(), key=lambda x: x[1])
    best_language_name = best_language_item[0]
    best_score = best_language_item[1]


    # Return unknown if score is too low
    if best_score < 0.1: # Threshold can be adjusted
        return 'unknown'

    return best_language_name

# Re-initialize NLTK_AVAILABLE based on successful import
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
