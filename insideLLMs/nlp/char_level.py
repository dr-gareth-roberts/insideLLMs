"""Character-level text analysis and transformation utilities.

This module provides a collection of functions for character-level natural
language processing operations, including n-gram generation, frequency
analysis, and various case conversion utilities.

Overview
--------
Character-level operations are fundamental building blocks for many NLP tasks,
including:

- **Text fingerprinting**: Using character n-grams for language detection,
  authorship attribution, and plagiarism detection.
- **Fuzzy matching**: Character n-grams enable approximate string matching
  that is robust to typos and spelling variations.
- **Text normalization**: Case conversion functions standardize text for
  consistent processing and storage.
- **Feature extraction**: Character frequencies and n-grams serve as features
  for machine learning models.

Functions
---------
get_char_ngrams
    Extract character n-grams (substrings of length n) from text.
get_char_frequency
    Count occurrences of each character in text.
to_uppercase
    Convert text to all uppercase letters.
to_titlecase
    Convert text to title case (capitalize first letter of each word).
to_camelcase
    Convert text to camelCase format for programming identifiers.
to_snakecase
    Convert text to snake_case format for programming identifiers.

Examples
--------
Basic character n-gram extraction for text analysis:

>>> from insideLLMs.nlp.char_level import get_char_ngrams
>>> text = "hello"
>>> bigrams = get_char_ngrams(text, n=2)
>>> print(bigrams)
['he', 'el', 'll', 'lo']

>>> trigrams = get_char_ngrams(text, n=3)
>>> print(trigrams)
['hel', 'ell', 'llo']

Character frequency analysis for cryptography or linguistics:

>>> from insideLLMs.nlp.char_level import get_char_frequency
>>> freq = get_char_frequency("mississippi")
>>> print(freq)
{'m': 1, 'i': 4, 's': 4, 'p': 2}

Case conversion for text normalization:

>>> from insideLLMs.nlp.char_level import to_snakecase, to_camelcase
>>> to_snakecase("Hello World")
'hello_world'
>>> to_camelcase("hello_world")
'helloWorld'

Notes
-----
- All functions handle empty strings gracefully, returning appropriate
  empty results.
- Character n-grams include whitespace and punctuation unless the input
  text is preprocessed to remove them.
- Case conversion functions are Unicode-aware and work with non-ASCII
  characters, though behavior may vary for complex scripts.

See Also
--------
insideLLMs.nlp.tokenizers : Word-level tokenization utilities.
insideLLMs.nlp.normalize : Text normalization and cleaning functions.

References
----------
.. [1] Jurafsky, D., & Martin, J. H. (2023). Speech and Language Processing.
       Chapter 2: Regular Expressions, Text Normalization, Edit Distance.
.. [2] Manning, C. D., Raghavan, P., & Schutze, H. (2008). Introduction to
       Information Retrieval. Chapter 3: Dictionaries and tolerant retrieval.
"""

import re
from collections import Counter

# ===== Character-Level Operations =====


def get_char_ngrams(text: str, n: int = 2) -> list[str]:
    """Generate character n-grams from text.

    Extracts all contiguous substrings of length `n` from the input text.
    Character n-grams are useful for language identification, fuzzy string
    matching, and text classification tasks where word boundaries are less
    important than character patterns.

    Parameters
    ----------
    text : str
        The input text from which to extract n-grams. Can contain any
        characters including whitespace, punctuation, and Unicode.
    n : int, default=2
        The length of each n-gram. Must be a positive integer. Common
        values are 2 (bigrams), 3 (trigrams), and 4 (quadgrams).

    Returns
    -------
    list[str]
        A list of character n-grams in order of their appearance in the
        text. Returns an empty list if the text length is less than `n`
        or if the text is empty.

    Raises
    ------
    TypeError
        If `text` is not a string or `n` is not an integer.

    Examples
    --------
    Extract bigrams (2-character sequences) from a word:

    >>> get_char_ngrams("hello", n=2)
    ['he', 'el', 'll', 'lo']

    Extract trigrams for more context:

    >>> get_char_ngrams("python", n=3)
    ['pyt', 'yth', 'tho', 'hon']

    N-grams preserve whitespace and punctuation:

    >>> get_char_ngrams("hi there!", n=2)
    ['hi', 'i ', ' t', 'th', 'he', 'er', 're', 'e!']

    Handle edge cases with short text:

    >>> get_char_ngrams("ab", n=3)  # Text shorter than n
    []

    >>> get_char_ngrams("", n=2)  # Empty text
    []

    >>> get_char_ngrams("x", n=1)  # Single character with n=1
    ['x']

    Use unigrams for individual character analysis:

    >>> get_char_ngrams("cat", n=1)
    ['c', 'a', 't']

    Practical example - language detection fingerprinting:

    >>> english_text = "the quick brown"
    >>> english_trigrams = get_char_ngrams(english_text, n=3)
    >>> 'the' in english_trigrams  # Common English pattern
    True

    See Also
    --------
    get_char_frequency : For counting character occurrences instead of
        extracting sequential patterns.

    Notes
    -----
    The number of n-grams generated is `len(text) - n + 1` when
    `len(text) >= n`, and 0 otherwise. For a text of length L with
    n-gram size N, this produces L - N + 1 n-grams.

    Memory usage scales linearly with text length. For very large texts,
    consider using a generator-based approach or processing in chunks.
    """
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def get_char_frequency(text: str) -> dict[str, int]:
    """Count the frequency of each character in the input text.

    Analyzes the input text and returns a dictionary mapping each unique
    character to its occurrence count. This is useful for frequency analysis
    in cryptography, stylometry, language modeling, and text statistics.

    Parameters
    ----------
    text : str
        The input text to analyze. Can contain any characters including
        whitespace, punctuation, numbers, and Unicode characters.

    Returns
    -------
    dict[str, int]
        A dictionary where keys are individual characters (str) and values
        are their occurrence counts (int). The dictionary is unordered.
        Returns an empty dictionary for empty input text.

    Raises
    ------
    TypeError
        If `text` is not a string.

    Examples
    --------
    Basic frequency analysis of a simple word:

    >>> get_char_frequency("hello")
    {'h': 1, 'e': 1, 'l': 2, 'o': 1}

    Analyze a word with many repeated characters:

    >>> freq = get_char_frequency("mississippi")
    >>> freq['i']
    4
    >>> freq['s']
    4
    >>> freq['p']
    2
    >>> freq['m']
    1

    Whitespace and punctuation are counted:

    >>> get_char_frequency("a b c")
    {'a': 1, ' ': 2, 'b': 1, 'c': 1}

    >>> get_char_frequency("hi! hi!")
    {'h': 2, 'i': 2, '!': 2, ' ': 1}

    Handle empty strings:

    >>> get_char_frequency("")
    {}

    Case sensitivity - uppercase and lowercase are distinct:

    >>> freq = get_char_frequency("AaAa")
    >>> freq['A']
    2
    >>> freq['a']
    2

    Unicode characters are fully supported:

    >>> freq = get_char_frequency("cafe")
    >>> freq['e']
    1

    Practical example - finding the most common character:

    >>> text = "the quick brown fox jumps over the lazy dog"
    >>> freq = get_char_frequency(text)
    >>> most_common = max(freq, key=freq.get)
    >>> print(f"Most common: '{most_common}' ({freq[most_common]} times)")
    Most common: ' ' (8 times)

    Practical example - checking for digit presence:

    >>> freq = get_char_frequency("abc123def456")
    >>> digits = sum(freq.get(d, 0) for d in "0123456789")
    >>> print(f"Total digits: {digits}")
    Total digits: 6

    See Also
    --------
    get_char_ngrams : For extracting character sequences instead of
        individual character counts.
    collections.Counter : The underlying implementation used for counting.

    Notes
    -----
    This function uses ``collections.Counter`` internally, which provides
    O(n) time complexity where n is the length of the input text.

    The returned dictionary does not preserve insertion order in Python
    versions prior to 3.7. In Python 3.7+, dictionaries maintain insertion
    order, which corresponds to the order of first occurrence in the text.

    For very large texts, memory usage is proportional to the number of
    unique characters, not the total text length.
    """
    return dict(Counter(text))


def to_uppercase(text: str) -> str:
    """Convert all alphabetic characters in text to uppercase.

    Transforms lowercase letters to their uppercase equivalents while
    preserving non-alphabetic characters (numbers, punctuation, whitespace).
    This is useful for case-insensitive comparisons, text normalization,
    and formatting output.

    Parameters
    ----------
    text : str
        The input text to convert. Can contain any characters including
        whitespace, punctuation, numbers, and Unicode characters.

    Returns
    -------
    str
        A new string with all alphabetic characters converted to uppercase.
        Non-alphabetic characters remain unchanged. Returns an empty string
        if the input is empty.

    Raises
    ------
    TypeError
        If `text` is not a string (e.g., if None or a numeric type is passed).

    Examples
    --------
    Basic uppercase conversion:

    >>> to_uppercase("hello")
    'HELLO'

    >>> to_uppercase("Hello World")
    'HELLO WORLD'

    Mixed case input:

    >>> to_uppercase("PyThOn")
    'PYTHON'

    Numbers and punctuation are preserved:

    >>> to_uppercase("hello123!")
    'HELLO123!'

    >>> to_uppercase("user@example.com")
    'USER@EXAMPLE.COM'

    Already uppercase text remains unchanged:

    >>> to_uppercase("ALREADY UPPER")
    'ALREADY UPPER'

    Empty and whitespace-only strings:

    >>> to_uppercase("")
    ''

    >>> to_uppercase("   ")
    '   '

    Unicode characters are handled correctly:

    >>> to_uppercase("cafe")
    'CAFE'

    >>> to_uppercase("strasse")  # German word
    'STRASSE'

    Practical example - case-insensitive comparison:

    >>> user_input = "Yes"
    >>> if to_uppercase(user_input) == "YES":
    ...     print("User confirmed")
    User confirmed

    Practical example - formatting headers:

    >>> headers = ["name", "age", "email"]
    >>> formatted = [to_uppercase(h) for h in headers]
    >>> print(formatted)
    ['NAME', 'AGE', 'EMAIL']

    Practical example - creating acronyms:

    >>> phrase = "as soon as possible"
    >>> acronym = "".join(to_uppercase(word[0]) for word in phrase.split())
    >>> print(acronym)
    ASAP

    See Also
    --------
    to_titlecase : Convert to title case (capitalize first letter of each word).
    to_camelcase : Convert to camelCase for programming identifiers.
    to_snakecase : Convert to snake_case for programming identifiers.

    Notes
    -----
    This function uses Python's built-in ``str.upper()`` method, which is
    Unicode-aware and handles characters from various scripts according to
    Unicode case mapping rules.

    Some Unicode characters have special case mapping rules. For example,
    the German eszett (ß) maps to "SS" in uppercase in standard Python,
    though locale-specific behavior may vary.

    The operation is O(n) where n is the length of the input string.
    """
    return text.upper()


def to_titlecase(text: str) -> str:
    """Convert text to title case (capitalize first letter of each word).

    Transforms text so that the first character of each word is uppercase
    and remaining characters are lowercase. Word boundaries are determined
    by whitespace and punctuation. Useful for formatting names, titles,
    headings, and proper nouns.

    Parameters
    ----------
    text : str
        The input text to convert. Can contain any characters including
        whitespace, punctuation, numbers, and Unicode characters.

    Returns
    -------
    str
        A new string with title case applied. Each word starts with an
        uppercase letter followed by lowercase letters. Returns an empty
        string if the input is empty.

    Raises
    ------
    TypeError
        If `text` is not a string (e.g., if None or a numeric type is passed).

    Examples
    --------
    Basic title case conversion:

    >>> to_titlecase("hello world")
    'Hello World'

    >>> to_titlecase("the quick brown fox")
    'The Quick Brown Fox'

    All uppercase input becomes title case:

    >>> to_titlecase("HELLO WORLD")
    'Hello World'

    Mixed case input is normalized:

    >>> to_titlecase("hELLo wORLd")
    'Hello World'

    Single word:

    >>> to_titlecase("python")
    'Python'

    Numbers and punctuation are preserved:

    >>> to_titlecase("hello123 world456")
    'Hello123 World456'

    Words separated by various punctuation:

    >>> to_titlecase("hello-world")
    'Hello-World'

    >>> to_titlecase("it's a wonderful life")
    "It'S A Wonderful Life"

    Note: Apostrophes create word boundaries in Python's title():

    >>> to_titlecase("don't stop")
    "Don'T Stop"

    Empty and whitespace-only strings:

    >>> to_titlecase("")
    ''

    >>> to_titlecase("   ")
    '   '

    Practical example - formatting a person's name:

    >>> name = "john doe"
    >>> formatted_name = to_titlecase(name)
    >>> print(formatted_name)
    John Doe

    Practical example - formatting article titles:

    >>> articles = ["the great gatsby", "war and peace", "1984"]
    >>> formatted = [to_titlecase(title) for title in articles]
    >>> print(formatted)
    ['The Great Gatsby', 'War And Peace', '1984']

    Practical example - formatting file names for display:

    >>> filename = "my_important_document"
    >>> display_name = to_titlecase(filename.replace("_", " "))
    >>> print(display_name)
    My Important Document

    Unicode handling:

    >>> to_titlecase("cafe au lait")
    'Cafe Au Lait'

    See Also
    --------
    to_uppercase : Convert all characters to uppercase.
    to_camelcase : Convert to camelCase for programming identifiers.
    to_snakecase : Convert to snake_case for programming identifiers.

    Notes
    -----
    This function uses Python's built-in ``str.title()`` method. Be aware
    that ``title()`` considers any non-letter character as a word boundary,
    which can produce unexpected results with contractions and possessives
    (e.g., "don't" becomes "Don'T").

    For more sophisticated title casing that respects English grammar rules
    (keeping articles like "the", "a", "an" lowercase in the middle of
    titles), consider using a dedicated library like ``titlecase``.

    The operation is O(n) where n is the length of the input string.
    """
    return text.title()


def to_camelcase(text: str) -> str:
    """Convert text to camelCase format for programming identifiers.

    Transforms text with spaces or underscores into camelCase, where the
    first word is lowercase and subsequent words are capitalized with no
    separators. This naming convention is commonly used in JavaScript,
    Java, and for variable names in many programming languages.

    Parameters
    ----------
    text : str
        The input text to convert. Can contain words separated by spaces
        and/or underscores. Other characters are preserved within words.

    Returns
    -------
    str
        A camelCase string where the first word is entirely lowercase
        and each subsequent word starts with an uppercase letter.
        Returns an empty string if the input is empty or contains only
        whitespace/underscores.

    Raises
    ------
    TypeError
        If `text` is not a string (e.g., if None or a numeric type is passed).

    Examples
    --------
    Basic conversion from space-separated words:

    >>> to_camelcase("hello world")
    'helloWorld'

    >>> to_camelcase("get user name")
    'getUserName'

    Conversion from underscore-separated words (snake_case):

    >>> to_camelcase("hello_world")
    'helloWorld'

    >>> to_camelcase("user_first_name")
    'userFirstName'

    Mixed separators (spaces and underscores):

    >>> to_camelcase("hello_world test")
    'helloWorldTest'

    >>> to_camelcase("my_variable name here")
    'myVariableNameHere'

    Single word input:

    >>> to_camelcase("hello")
    'hello'

    >>> to_camelcase("HELLO")
    'hello'

    Already camelCase input may be modified:

    >>> to_camelcase("helloWorld")
    'helloworld'

    Empty and whitespace-only strings:

    >>> to_camelcase("")
    ''

    >>> to_camelcase("   ")
    ''

    >>> to_camelcase("___")
    ''

    Numbers are preserved in position:

    >>> to_camelcase("user_id_123")
    'userId123'

    >>> to_camelcase("get item 42")
    'getItem42'

    Uppercase input is converted properly:

    >>> to_camelcase("HELLO WORLD")
    'helloWorld'

    >>> to_camelcase("THE_QUICK_BROWN_FOX")
    'theQuickBrownFox'

    Practical example - converting database column names to code:

    >>> columns = ["user_id", "first_name", "last_name", "email_address"]
    >>> camel_columns = [to_camelcase(col) for col in columns]
    >>> print(camel_columns)
    ['userId', 'firstName', 'lastName', 'emailAddress']

    Practical example - creating JavaScript-style variable names:

    >>> label = "number of items"
    >>> var_name = to_camelcase(label)
    >>> print(f"const {var_name} = 5;")
    const numberOfItems = 5;

    Practical example - API endpoint to method name:

    >>> endpoint = "get_user_profile"
    >>> method = to_camelcase(endpoint)
    >>> print(f"function {method}() {{}}")
    function getUserProfile() {}

    See Also
    --------
    to_snakecase : Convert to snake_case format.
    to_titlecase : Convert to Title Case format.
    to_uppercase : Convert all characters to uppercase.

    Notes
    -----
    The function first replaces all underscores with spaces, then splits
    on whitespace. This means multiple consecutive spaces or underscores
    are treated as a single separator.

    The first word is converted entirely to lowercase, which may not be
    the desired behavior if the first word is an acronym (e.g., "HTTP
    request" becomes "httpRequest", not "HTTPRequest").

    For PascalCase (UpperCamelCase) where the first letter is also
    capitalized, you can capitalize the first character of the result:
    ``result[0].upper() + result[1:]`` if result is non-empty.

    The operation has O(n) time complexity where n is the total length
    of all words in the input.
    """
    # Replace underscores with spaces, then split by spaces
    words = text.replace("_", " ").split()
    if not words:
        return ""

    # First word lowercase, rest title case
    return words[0].lower() + "".join(word.title() for word in words[1:])


def to_snakecase(text: str) -> str:
    """Convert text to snake_case format for programming identifiers.

    Transforms text from various formats (spaces, hyphens, camelCase,
    PascalCase) into snake_case, where words are lowercase and separated
    by underscores. This naming convention is commonly used in Python,
    Ruby, and for database column names.

    Parameters
    ----------
    text : str
        The input text to convert. Can contain words separated by spaces,
        hyphens, underscores, or using camelCase/PascalCase conventions.
        Non-alphanumeric characters (except underscores) are removed.

    Returns
    -------
    str
        A snake_case string where all letters are lowercase and words
        are separated by single underscores. Returns an empty string
        if the input contains no alphanumeric characters.

    Raises
    ------
    TypeError
        If `text` is not a string (e.g., if None or a numeric type is passed).

    Examples
    --------
    Basic conversion from space-separated words:

    >>> to_snakecase("Hello World")
    'hello_world'

    >>> to_snakecase("The Quick Brown Fox")
    'the_quick_brown_fox'

    Conversion from camelCase:

    >>> to_snakecase("helloWorld")
    'hello_world'

    >>> to_snakecase("getUserName")
    'get_user_name'

    Conversion from PascalCase:

    >>> to_snakecase("HelloWorld")
    'hello_world'

    >>> to_snakecase("MyClassName")
    'my_class_name'

    Conversion from kebab-case (hyphen-separated):

    >>> to_snakecase("hello-world")
    'hello_world'

    >>> to_snakecase("my-component-name")
    'my_component_name'

    Already snake_case input is preserved:

    >>> to_snakecase("hello_world")
    'hello_world'

    >>> to_snakecase("user_first_name")
    'user_first_name'

    Mixed formats:

    >>> to_snakecase("Hello World-Test")
    'hello_world_test'

    >>> to_snakecase("myVar with spaces")
    'my_var_with_spaces'

    Numbers are preserved:

    >>> to_snakecase("user123Name")
    'user123_name'

    >>> to_snakecase("get item 42")
    'get_item_42'

    Special characters are removed:

    >>> to_snakecase("hello@world!")
    'helloworld'

    >>> to_snakecase("user.name")
    'username'

    Multiple separators are collapsed:

    >>> to_snakecase("hello   world")
    'hello_world'

    >>> to_snakecase("hello___world")
    'hello_world'

    Empty and whitespace-only strings:

    >>> to_snakecase("")
    ''

    >>> to_snakecase("   ")
    '_'

    Practical example - converting class names to file names:

    >>> class_name = "MyDatabaseConnection"
    >>> file_name = to_snakecase(class_name) + ".py"
    >>> print(file_name)
    my_database_connection.py

    Practical example - converting JavaScript to Python naming:

    >>> js_vars = ["firstName", "lastName", "emailAddress"]
    >>> py_vars = [to_snakecase(v) for v in js_vars]
    >>> print(py_vars)
    ['first_name', 'last_name', 'email_address']

    Practical example - creating database table names:

    >>> model_name = "UserProfile"
    >>> table_name = to_snakecase(model_name) + "s"
    >>> print(f"CREATE TABLE {table_name} (...)")
    CREATE TABLE user_profiles (...)

    Practical example - converting HTTP headers to environment variables:

    >>> header = "Content-Type"
    >>> env_var = "HTTP_" + to_snakecase(header).upper()
    >>> print(env_var)
    HTTP_CONTENT_TYPE

    Acronyms are split at case boundaries:

    >>> to_snakecase("HTTPServer")
    'httpserver'

    >>> to_snakecase("getHTTPResponse")
    'get_httpresponse'

    See Also
    --------
    to_camelcase : Convert to camelCase format.
    to_titlecase : Convert to Title Case format.
    to_uppercase : Convert all characters to uppercase.

    Notes
    -----
    The conversion process follows these steps:

    1. Replace spaces and hyphens with underscores.
    2. Insert underscores before uppercase letters that follow lowercase
       letters or digits (to handle camelCase).
    3. Remove all non-alphanumeric characters except underscores.
    4. Convert to lowercase.
    5. Collapse multiple consecutive underscores into single underscores.

    Be aware that acronyms in camelCase (like "HTTP" in "getHTTPResponse")
    are not specially handled and may produce unexpected results. The
    function treats consecutive uppercase letters as a single word.

    The function uses regular expressions for pattern matching, which
    provides O(n) time complexity where n is the length of the input.
    """
    # Replace spaces and hyphens with underscores
    text = re.sub(r"[ -]", "_", text)
    # Handle camel case
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9_]", "", text).lower()
    # Replace multiple underscores with a single one
    text = re.sub(r"_+", "_", text)
    return text
