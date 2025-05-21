import re
import string
import unicodedata
from typing import Optional

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
