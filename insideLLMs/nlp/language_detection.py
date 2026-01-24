"""
Language Detection Module
=========================

This module provides lightweight, dependency-minimal language detection for text
analysis. It implements two complementary approaches: stopword frequency analysis
and character n-gram profiling, enabling language identification without requiring
large pre-trained models or external API calls.

Overview
--------
Language detection is a fundamental NLP task that determines the natural language
of a given text. This module supports detection of seven major European languages:
English, Spanish, French, German, Italian, Portuguese, and Dutch.

The module offers two detection strategies:

1. **Stopword-based detection** (`detect_language_by_stopwords`): Analyzes the
   frequency of common stopwords (e.g., "the", "and", "is" for English) to
   identify the language. This method is effective for longer texts with
   typical prose structure.

2. **Character n-gram detection** (`detect_language_by_char_ngrams`): Examines
   the frequency of character trigrams (3-character sequences) to build a
   statistical profile. This method works well even for shorter texts and is
   more robust to domain-specific vocabulary.

Supported Languages
-------------------
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Dutch (nl)

Examples
--------
Basic language detection using stopwords:

>>> from insideLLMs.nlp.language_detection import detect_language_by_stopwords
>>> text = "The quick brown fox jumps over the lazy dog."
>>> detect_language_by_stopwords(text)
'en'

>>> spanish_text = "El rápido zorro marrón salta sobre el perro perezoso."
>>> detect_language_by_stopwords(spanish_text)
'es'

Using character n-gram detection for shorter text:

>>> from insideLLMs.nlp.language_detection import detect_language_by_char_ngrams
>>> text = "This is a sample English text for detection."
>>> detect_language_by_char_ngrams(text)
'en'

>>> french_text = "Bonjour, comment allez-vous aujourd'hui?"
>>> detect_language_by_char_ngrams(french_text)
'fr'

Combining both methods for robust detection:

>>> def detect_language_robust(text):
...     stopword_result = detect_language_by_stopwords(text)
...     ngram_result = detect_language_by_char_ngrams(text)
...     if stopword_result == ngram_result:
...         return stopword_result
...     # Prefer n-gram for short texts, stopwords for longer
...     return ngram_result if len(text) < 100 else stopword_result

Notes
-----
- Both functions return "unknown" when confidence is too low or text is
  insufficient for reliable detection.
- The stopword method requires NLTK data (downloaded automatically on first use).
- The n-gram method has no external dependencies beyond the standard library.
- For production use with high accuracy requirements, consider dedicated
  libraries like `langdetect` or `fasttext`.
- These methods are optimized for Western European languages and may not
  perform well on texts mixing multiple languages.

See Also
--------
insideLLMs.nlp.dependencies : NLTK dependency management utilities.
nltk.corpus.stopwords : NLTK's stopword lists used by stopword detection.

References
----------
.. [1] Cavnar, W. B., & Trenkle, J. M. (1994). N-gram-based text categorization.
       Proceedings of SDAIR-94, 3rd Annual Symposium on Document Analysis and
       Information Retrieval.
.. [2] NLTK Project. https://www.nltk.org/
"""

import re
from collections import Counter

from insideLLMs.nlp.dependencies import ensure_nltk


def _ensure_nltk_langdetect():
    """
    Ensure NLTK and required resources are available for language detection.

    This internal function lazily initializes NLTK resources needed for
    stopword-based language detection. It downloads the punkt tokenizer
    and stopwords corpus if not already present.

    The function is called automatically by `detect_language_by_stopwords`
    and does not need to be invoked directly by users.

    Returns
    -------
    None
        The function modifies global NLTK state but returns nothing.

    Raises
    ------
    OSError
        If NLTK data cannot be downloaded (e.g., no network connection
        and data not cached).

    Examples
    --------
    >>> from insideLLMs.nlp.language_detection import _ensure_nltk_langdetect
    >>> _ensure_nltk_langdetect()  # Downloads NLTK data if needed

    Notes
    -----
    This function is idempotent - calling it multiple times has no
    additional effect after the first successful call.

    See Also
    --------
    insideLLMs.nlp.dependencies.ensure_nltk : The underlying NLTK setup function.
    """
    ensure_nltk(("tokenizers/punkt", "corpora/stopwords"))


# Backward compatibility alias
check_nltk = _ensure_nltk_langdetect
"""
Backward compatibility alias for `_ensure_nltk_langdetect`.

Deprecated
----------
This alias is provided for backward compatibility only. New code should
not rely on this function directly, as NLTK initialization is handled
automatically by `detect_language_by_stopwords`.
"""


def detect_language_by_stopwords(text: str) -> str:
    """
    Detect language based on stopword frequency across common languages.

    This function identifies the language of input text by analyzing the
    proportion of words that match known stopwords for each supported
    language. Stopwords are common, high-frequency words (like "the", "is",
    "and" in English) that carry little semantic meaning but are highly
    characteristic of specific languages.

    The algorithm tokenizes the input text, computes the ratio of stopwords
    to total words for each candidate language, and returns the language
    with the highest stopword density (provided it exceeds a minimum
    threshold).

    Parameters
    ----------
    text : str
        The input text to analyze. Should contain natural language prose
        for best results. The text is automatically lowercased during
        processing.

    Returns
    -------
    str
        A two-letter ISO 639-1 language code representing the detected
        language. Possible return values are:

        - 'en' : English
        - 'es' : Spanish
        - 'fr' : French
        - 'de' : German
        - 'it' : Italian
        - 'pt' : Portuguese
        - 'nl' : Dutch
        - 'unknown' : Detection failed or confidence too low

    Raises
    ------
    LookupError
        If NLTK tokenizer models cannot be loaded. This typically occurs
        on first run if network access is unavailable.

    Examples
    --------
    Detecting English text:

    >>> from insideLLMs.nlp.language_detection import detect_language_by_stopwords
    >>> text = "The quick brown fox jumps over the lazy dog."
    >>> detect_language_by_stopwords(text)
    'en'

    Detecting Spanish text:

    >>> spanish = "Este es un ejemplo de texto en español para detectar."
    >>> detect_language_by_stopwords(spanish)
    'es'

    Detecting German text:

    >>> german = "Das ist ein Beispieltext auf Deutsch zum Testen."
    >>> detect_language_by_stopwords(german)
    'de'

    Detecting French text:

    >>> french = "Ceci est un exemple de texte en français pour la détection."
    >>> detect_language_by_stopwords(french)
    'fr'

    Detecting Italian text:

    >>> italian = "Questo è un esempio di testo in italiano per il rilevamento."
    >>> detect_language_by_stopwords(italian)
    'it'

    Detecting Portuguese text:

    >>> portuguese = "Este é um exemplo de texto em português para detecção."
    >>> detect_language_by_stopwords(portuguese)
    'pt'

    Detecting Dutch text:

    >>> dutch = "Dit is een voorbeeld van een Nederlandse tekst voor detectie."
    >>> detect_language_by_stopwords(dutch)
    'nl'

    Handling empty or insufficient text:

    >>> detect_language_by_stopwords("")
    'unknown'

    >>> detect_language_by_stopwords("xyz abc 123")
    'unknown'

    Processing longer documents:

    >>> article = '''
    ... The development of natural language processing has revolutionized
    ... how computers understand and generate human language. Modern NLP
    ... systems can translate between languages, answer questions, and
    ... even write creative content. The field has grown rapidly with
    ... advances in machine learning and neural networks.
    ... '''
    >>> detect_language_by_stopwords(article)
    'en'

    Notes
    -----
    - The function requires at least 5% stopword density to make a
      confident detection. Below this threshold, 'unknown' is returned.
    - Performance is generally better on longer texts (50+ words) where
      stopword frequency patterns are more reliable.
    - Technical jargon, code snippets, or highly specialized vocabulary
      may reduce accuracy since they contain fewer common stopwords.
    - The function automatically downloads required NLTK data on first use.
    - Texts containing mixed languages may produce unreliable results,
      typically detecting the dominant language.

    See Also
    --------
    detect_language_by_char_ngrams : Alternative detection using character
        n-gram frequency profiles, better for shorter texts.
    nltk.corpus.stopwords : The underlying stopword lists used.

    References
    ----------
    .. [1] Bird, S., Klein, E., & Loper, E. (2009). Natural Language
           Processing with Python. O'Reilly Media.
    """
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
    """
    Detect language based on character n-gram frequency profiles.

    This function identifies the language of input text by analyzing the
    frequency distribution of character trigrams (3-character sequences).
    Each supported language has a characteristic "fingerprint" of common
    trigrams, and the function compares the input text's trigram profile
    against these reference profiles using Jaccard similarity.

    The n-gram approach is particularly effective for shorter texts where
    stopword-based methods may lack sufficient data, and it works well
    even with domain-specific vocabulary or technical content.

    Parameters
    ----------
    text : str
        The input text to analyze. The text is preprocessed by converting
        to lowercase and removing non-alphabetic characters (except spaces).
        Must contain at least 20 characters after cleaning to produce
        reliable results.

    Returns
    -------
    str
        A two-letter ISO 639-1 language code representing the detected
        language. Possible return values are:

        - 'en' : English
        - 'es' : Spanish
        - 'fr' : French
        - 'de' : German
        - 'it' : Italian
        - 'pt' : Portuguese
        - 'nl' : Dutch
        - 'unknown' : Detection failed or confidence too low

    Examples
    --------
    Detecting English text:

    >>> from insideLLMs.nlp.language_detection import detect_language_by_char_ngrams
    >>> text = "The quick brown fox jumps over the lazy dog."
    >>> detect_language_by_char_ngrams(text)
    'en'

    Detecting Spanish text:

    >>> spanish = "Este es un ejemplo de texto en español."
    >>> detect_language_by_char_ngrams(spanish)
    'es'

    Detecting German text:

    >>> german = "Das ist ein Beispieltext auf Deutsch."
    >>> detect_language_by_char_ngrams(german)
    'de'

    Detecting French text:

    >>> french = "Ceci est un exemple de texte en français."
    >>> detect_language_by_char_ngrams(french)
    'fr'

    Detecting Italian text:

    >>> italian = "Questo testo italiano serve per testare il rilevamento."
    >>> detect_language_by_char_ngrams(italian)
    'it'

    Detecting Portuguese text:

    >>> portuguese = "Este texto em português serve para testar a detecção."
    >>> detect_language_by_char_ngrams(portuguese)
    'pt'

    Detecting Dutch text:

    >>> dutch = "Dit is een voorbeeld van Nederlandse tekst voor detectie."
    >>> detect_language_by_char_ngrams(dutch)
    'nl'

    Handling text that is too short:

    >>> detect_language_by_char_ngrams("Hello")
    'unknown'

    >>> detect_language_by_char_ngrams("Short text")
    'unknown'

    Processing technical content:

    >>> code_comment = "This function implements the binary search algorithm."
    >>> detect_language_by_char_ngrams(code_comment)
    'en'

    Processing text with special characters (they are filtered out):

    >>> messy_text = "Hello!!! This is a test... with many, many punctuation marks!!!"
    >>> detect_language_by_char_ngrams(messy_text)
    'en'

    Comparing detection methods for short vs. long text:

    >>> short = "Bonjour monsieur"  # May fail stopword method
    >>> len(short)
    16
    >>> detect_language_by_char_ngrams(short)  # Too short
    'unknown'

    >>> longer = "Bonjour monsieur, comment allez-vous aujourd'hui?"
    >>> detect_language_by_char_ngrams(longer)
    'fr'

    Notes
    -----
    - The function requires at least 20 characters after preprocessing
      to generate meaningful trigram statistics.
    - A minimum Jaccard similarity score of 0.1 (10% overlap) is required
      for confident detection. Below this threshold, 'unknown' is returned.
    - Unlike `detect_language_by_stopwords`, this function has no external
      dependencies beyond the Python standard library.
    - The preprocessing step removes all non-ASCII letters, which means
      language-specific characters (like German umlauts or Spanish tildes)
      are stripped. This can affect accuracy for some texts.
    - The trigram profiles are based on empirical analysis of large text
      corpora for each language.

    Algorithm Details
    -----------------
    1. Preprocess: Convert to lowercase, remove non-alphabetic chars
    2. Extract: Generate all character trigrams from the text
    3. Profile: Identify the 20 most frequent trigrams
    4. Compare: Calculate Jaccard similarity with each language profile
    5. Decide: Return the best match if similarity exceeds threshold

    The Jaccard similarity coefficient is defined as:

        J(A, B) = |A ∩ B| / |A ∪ B|

    where A is the set of top trigrams from the input text and B is the
    reference trigram profile for a language.

    See Also
    --------
    detect_language_by_stopwords : Alternative detection using stopword
        frequency analysis, better for longer texts.

    References
    ----------
    .. [1] Cavnar, W. B., & Trenkle, J. M. (1994). N-gram-based text
           categorization. Proceedings of SDAIR-94, 3rd Annual Symposium
           on Document Analysis and Information Retrieval.
    .. [2] Dunning, T. (1994). Statistical identification of language.
           Computing Research Laboratory Technical Report CRL-MCCS-94-273.
    """
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
