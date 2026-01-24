import re
from collections import Counter

from insideLLMs.nlp.dependencies import ensure_nltk


def _ensure_nltk_langdetect():
    """Ensure NLTK and required resources are available."""
    ensure_nltk(("tokenizers/punkt", "corpora/stopwords"))


# Backward compatibility alias
check_nltk = _ensure_nltk_langdetect


def detect_language_by_stopwords(text: str) -> str:
    """Detect language based on stopword frequency across a few common languages."""
    _ensure_nltk_langdetect()
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    languages = ["english", "spanish", "french", "german", "italian", "portuguese", "dutch"]
    language_codes = {
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "dutch": "nl",
    }

    tokens = word_tokenize(text.lower())
    if not tokens:
        return "unknown"

    language_scores = {}
    for lang in languages:
        # Computes language score; handles missing stopword files
        try:
            current_stopwords = set(stopwords.words(lang))
            count = sum(1 for token in tokens if token in current_stopwords)
            language_scores[lang] = count / len(tokens)
        except OSError:
            language_scores[lang] = 0

    if not language_scores or all(score == 0 for score in language_scores.values()):
        return "unknown"

    best_language_name, best_score = max(language_scores.items(), key=lambda x: x[1])
    if best_score < 0.05:
        return "unknown"

    return language_codes.get(best_language_name, "unknown")


def detect_language_by_char_ngrams(text: str) -> str:
    """Detect language based on character n-gram frequency profiles."""
    profiles = {
        "en": ["the", "and", "ing", "ion", "tio", "ent", "ati", "for", "her", "ter"],
        "es": ["que", "ión", "nte", "con", "est", "ent", "ado", "par", "los", "ien"],
        "fr": ["les", "ent", "que", "une", "our", "ant", "des", "men", "tio", "ion"],
        "de": ["ein", "die", "und", "der", "sch", "ich", "nde", "den", "che", "gen"],
        "it": ["che", "non", "per", "del", "ent", "ion", "con", "ato", "gli", "ell"],
        "pt": ["que", "ent", "ção", "não", "com", "est", "ado", "par", "ara", "uma"],
        "nl": ["een", "het", "oor", "nde", "van", "aar", "eer", "ing", "ijk", "sch"],
    }

    text_cleaned = text.lower()
    text_cleaned = re.sub(r"[^a-z\s]", "", text_cleaned)
    text_cleaned = re.sub(r"\s+", " ", text_cleaned).strip()

    if len(text_cleaned) < 20:
        return "unknown"

    trigrams = [text_cleaned[i : i + 3] for i in range(len(text_cleaned) - 2)]
    if not trigrams:
        return "unknown"

    trigram_freq = Counter(trigrams)
    most_common_text_trigrams = [t for t, _ in trigram_freq.most_common(20)]

    scores = {}
    for lang, profile_trigrams in profiles.items():
        intersection = len(set(most_common_text_trigrams) & set(profile_trigrams))
        union = len(set(most_common_text_trigrams) | set(profile_trigrams))
        scores[lang] = intersection / union if union > 0 else 0

    if not scores or all(score == 0 for score in scores.values()):
        return "unknown"

    best_language_name, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score < 0.1:
        return "unknown"

    return best_language_name
