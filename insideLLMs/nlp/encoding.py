"""
Text Encoding and Decoding Utilities.

This module provides a collection of functions for encoding and decoding text
using various standard encoding schemes commonly used in web development,
data transmission, and text processing applications.

Overview
--------
The module supports three main encoding/decoding schemes:

1. **Base64 Encoding**: Binary-to-text encoding scheme that represents binary
   data in an ASCII string format. Commonly used for encoding binary data in
   email attachments, embedding images in HTML/CSS, and transmitting data
   over media designed for text.

2. **URL Encoding (Percent Encoding)**: Mechanism for encoding information in
   a Uniform Resource Identifier (URI). Used to encode special characters in
   URLs and query parameters to ensure safe transmission.

3. **HTML Encoding**: Converts special HTML characters to their corresponding
   HTML entities. Essential for preventing XSS attacks and displaying special
   characters correctly in HTML documents.

Functions
---------
encode_base64
    Encode UTF-8 text to Base64 format.
decode_base64
    Decode Base64 encoded string back to UTF-8 text.
url_encode
    Encode text for safe use in URLs (percent encoding).
url_decode
    Decode URL-encoded (percent-encoded) text.
html_encode
    Encode special HTML characters to HTML entities.
html_decode
    Decode HTML entities back to their character equivalents.

Examples
--------
Base64 encoding for data transmission:

>>> from insideLLMs.nlp.encoding import encode_base64, decode_base64
>>> secret_message = "Hello, World!"
>>> encoded = encode_base64(secret_message)
>>> print(encoded)
SGVsbG8sIFdvcmxkIQ==
>>> decode_base64(encoded)
'Hello, World!'

URL encoding for query parameters:

>>> from insideLLMs.nlp.encoding import url_encode, url_decode
>>> search_query = "python programming & NLP"
>>> safe_query = url_encode(search_query)
>>> print(safe_query)
python%20programming%20%26%20NLP
>>> print(f"https://example.com/search?q={safe_query}")
https://example.com/search?q=python%20programming%20%26%20NLP

HTML encoding for safe web display:

>>> from insideLLMs.nlp.encoding import html_encode, html_decode
>>> user_input = '<script>alert("XSS")</script>'
>>> safe_html = html_encode(user_input)
>>> print(safe_html)
&lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;

Notes
-----
- All functions assume UTF-8 encoding for text operations.
- These functions are designed for text data; for binary data, consider
  using the underlying standard library modules directly.
- URL encoding follows RFC 3986, which reserves certain characters.
- HTML encoding handles the five predefined XML entities: &, <, >, ", '.

See Also
--------
base64 : Standard library module for Base64 encoding/decoding.
urllib.parse : Standard library module for URL parsing and encoding.
html : Standard library module for HTML entity handling.

References
----------
.. [1] RFC 4648 - The Base16, Base32, and Base64 Data Encodings
       https://tools.ietf.org/html/rfc4648
.. [2] RFC 3986 - Uniform Resource Identifier (URI): Generic Syntax
       https://tools.ietf.org/html/rfc3986
.. [3] W3C HTML5 - Named Character References
       https://html.spec.whatwg.org/multipage/named-characters.html
"""

import base64
import html
import urllib.parse

# ===== Text Encoding/Decoding =====


def encode_base64(text: str) -> str:
    """
    Encode a UTF-8 text string to Base64 format.

    Base64 encoding converts binary data into a text representation using
    a set of 64 printable ASCII characters. This function first encodes
    the input text as UTF-8 bytes, then applies Base64 encoding.

    Parameters
    ----------
    text : str
        The input text string to encode. Can contain any valid Unicode
        characters, including multi-byte characters, emojis, and special
        symbols.

    Returns
    -------
    str
        The Base64 encoded representation of the input text. The output
        contains only ASCII characters from the Base64 alphabet (A-Z, a-z,
        0-9, +, /) and may include padding characters (=).

    Raises
    ------
    AttributeError
        If ``text`` is not a string (e.g., None or other types).
    UnicodeEncodeError
        If the text cannot be encoded as UTF-8 (extremely rare with
        valid Python strings).

    Examples
    --------
    Encoding a simple ASCII string:

    >>> encode_base64("Hello, World!")
    'SGVsbG8sIFdvcmxkIQ=='

    Encoding text with special characters:

    >>> encode_base64("Café résumé")
    'Q2Fmw6kgcsOpc3Vtw6k='

    Encoding Unicode text with emojis:

    >>> encode_base64("Python 🐍 is fun!")
    'UHl0aG9uIPCfkI0gaXMgZnVuIQ=='

    Encoding JSON data for API transmission:

    >>> import json
    >>> data = {"user": "alice", "score": 42}
    >>> json_str = json.dumps(data)
    >>> encoded_payload = encode_base64(json_str)
    >>> print(encoded_payload)
    eyJ1c2VyIjogImFsaWNlIiwgInNjb3JlIjogNDJ9

    Encoding credentials for Basic HTTP authentication:

    >>> username, password = "admin", "secret123"
    >>> credentials = f"{username}:{password}"
    >>> auth_header = encode_base64(credentials)
    >>> print(f"Authorization: Basic {auth_header}")
    Authorization: Basic YWRtaW46c2VjcmV0MTIz

    Encoding an empty string:

    >>> encode_base64("")
    ''

    Notes
    -----
    - The output length is approximately 4/3 times the input byte length,
      rounded up to a multiple of 4 (due to padding).
    - Base64 encoding is not encryption; it provides no security. Use it
      only for encoding, not for protecting sensitive data.
    - The padding character '=' ensures the encoded output length is
      always a multiple of 4.

    See Also
    --------
    decode_base64 : Decode Base64 encoded string back to text.
    base64.b64encode : Lower-level Base64 encoding function.
    """
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


def decode_base64(encoded: str) -> str:
    """
    Decode a Base64 encoded string back to UTF-8 text.

    This function reverses the Base64 encoding process, converting a
    Base64 encoded ASCII string back to its original UTF-8 text
    representation.

    Parameters
    ----------
    encoded : str
        A valid Base64 encoded string. The string should contain only
        characters from the Base64 alphabet (A-Z, a-z, 0-9, +, /) and
        may include padding characters (=). Whitespace is not permitted.

    Returns
    -------
    str
        The decoded UTF-8 text string. The output will be identical to
        the original text that was encoded.

    Raises
    ------
    binascii.Error
        If ``encoded`` contains invalid Base64 characters or has
        incorrect padding.
    UnicodeDecodeError
        If the decoded bytes are not valid UTF-8. This occurs when
        decoding Base64 data that was not originally UTF-8 encoded text.
    AttributeError
        If ``encoded`` is not a string (e.g., None or other types).

    Examples
    --------
    Decoding a simple Base64 string:

    >>> decode_base64("SGVsbG8sIFdvcmxkIQ==")
    'Hello, World!'

    Decoding text with special characters:

    >>> decode_base64("Q2Fmw6kgcsOpc3Vtw6k=")
    'Café résumé'

    Decoding Unicode text with emojis:

    >>> decode_base64("UHl0aG9uIPCfkI0gaXMgZnVuIQ==")
    'Python 🐍 is fun!'

    Round-trip encoding and decoding:

    >>> original = "The quick brown fox jumps over the lazy dog."
    >>> encoded = encode_base64(original)
    >>> decoded = decode_base64(encoded)
    >>> original == decoded
    True

    Decoding API response payload:

    >>> api_response = "eyJzdGF0dXMiOiAic3VjY2VzcyIsICJjb2RlIjogMjAwfQ=="
    >>> import json
    >>> data = json.loads(decode_base64(api_response))
    >>> print(data)
    {'status': 'success', 'code': 200}

    Decoding HTTP Basic Auth credentials:

    >>> auth_string = "YWRtaW46c2VjcmV0MTIz"
    >>> credentials = decode_base64(auth_string)
    >>> username, password = credentials.split(":")
    >>> print(f"User: {username}, Pass: {password}")
    User: admin, Pass: secret123

    Decoding an empty Base64 string:

    >>> decode_base64("")
    ''

    Notes
    -----
    - This function expects standard Base64 encoding. For URL-safe Base64
      variants (using - and _ instead of + and /), preprocess the input
      or use ``base64.urlsafe_b64decode`` directly.
    - Invalid Base64 strings will raise ``binascii.Error``. Always validate
      input or handle exceptions when decoding untrusted data.
    - The function assumes the original data was UTF-8 encoded text. For
      binary data, use ``base64.b64decode`` directly.

    See Also
    --------
    encode_base64 : Encode text to Base64 format.
    base64.b64decode : Lower-level Base64 decoding function.
    """
    return base64.b64decode(encoded.encode("utf-8")).decode("utf-8")


def url_encode(text: str) -> str:
    """
    Encode text for safe use in URLs using percent-encoding.

    URL encoding (also known as percent-encoding) replaces unsafe ASCII
    characters with a '%' followed by two hexadecimal digits representing
    the character's byte value. This ensures that special characters do
    not interfere with URL parsing.

    Parameters
    ----------
    text : str
        The input text string to encode. Can contain any valid Unicode
        characters. Characters that are safe for URLs (letters, digits,
        and ``_.-~``) are not encoded.

    Returns
    -------
    str
        The URL-encoded (percent-encoded) string. Unsafe characters are
        replaced with ``%XX`` sequences where XX is the hexadecimal
        representation of the character's byte value.

    Raises
    ------
    AttributeError
        If ``text`` is not a string (e.g., None or other types).

    Examples
    --------
    Encoding a search query with spaces:

    >>> url_encode("hello world")
    'hello%20world'

    Encoding special URL characters:

    >>> url_encode("name=John&age=30")
    'name%3DJohn%26age%3D30'

    Encoding a file path:

    >>> url_encode("/path/to/file.txt")
    '/path/to/file.txt'

    Encoding Unicode characters:

    >>> url_encode("Привет мир")
    '%D0%9F%D1%80%D0%B8%D0%B2%D0%B5%D1%82%20%D0%BC%D0%B8%D1%80'

    Encoding emojis:

    >>> url_encode("I ❤️ Python")
    'I%20%E2%9D%A4%EF%B8%8F%20Python'

    Building a complete URL with query parameters:

    >>> base_url = "https://api.example.com/search"
    >>> query = "machine learning & NLP"
    >>> full_url = f"{base_url}?q={url_encode(query)}"
    >>> print(full_url)
    https://api.example.com/search?q=machine%20learning%20%26%20NLP

    Encoding form data for POST requests:

    >>> form_field = "user comment: <script>alert('xss')</script>"
    >>> encoded_field = url_encode(form_field)
    >>> print(encoded_field)
    user%20comment%3A%20%3Cscript%3Ealert%28%27xss%27%29%3C/script%3E

    Encoding an empty string:

    >>> url_encode("")
    ''

    Notes
    -----
    - The forward slash ``/`` is NOT encoded by default as it is considered
      safe. Use ``urllib.parse.quote(text, safe='')`` to encode all
      characters except letters, digits, and ``_.-~``.
    - Spaces are encoded as ``%20``, not ``+``. For form data encoding
      where spaces should be ``+``, use ``urllib.parse.quote_plus``.
    - This function follows RFC 3986 which defines the characters that
      must be percent-encoded in URIs.
    - Multi-byte Unicode characters are first encoded as UTF-8 bytes,
      then each byte is percent-encoded.

    See Also
    --------
    url_decode : Decode URL-encoded text.
    urllib.parse.quote : Lower-level URL encoding function.
    urllib.parse.quote_plus : URL encoding with + for spaces.
    urllib.parse.urlencode : Encode a dictionary as query string.
    """
    return urllib.parse.quote(text)


def url_decode(encoded: str) -> str:
    """
    Decode a URL-encoded (percent-encoded) string back to plain text.

    This function reverses URL encoding by replacing ``%XX`` escape
    sequences with their corresponding characters, converting the
    percent-encoded string back to its original form.

    Parameters
    ----------
    encoded : str
        A URL-encoded string containing percent-encoded sequences
        (``%XX`` where XX is a hexadecimal value). The string may also
        contain non-encoded characters which will be passed through
        unchanged.

    Returns
    -------
    str
        The decoded text string with all percent-encoded sequences
        converted back to their original characters.

    Raises
    ------
    AttributeError
        If ``encoded`` is not a string (e.g., None or other types).

    Examples
    --------
    Decoding a string with encoded spaces:

    >>> url_decode("hello%20world")
    'hello world'

    Decoding special characters:

    >>> url_decode("name%3DJohn%26age%3D30")
    'name=John&age=30'

    Decoding Unicode characters (Cyrillic):

    >>> url_decode("%D0%9F%D1%80%D0%B8%D0%B2%D0%B5%D1%82")
    'Привет'

    Decoding emojis:

    >>> url_decode("I%20%E2%9D%A4%EF%B8%8F%20Python")
    'I ❤️ Python'

    Round-trip encoding and decoding:

    >>> original = "Hello, World! Special chars: @#$%^&*()"
    >>> encoded = url_encode(original)
    >>> decoded = url_decode(encoded)
    >>> original == decoded
    True

    Parsing query parameters from a URL:

    >>> from urllib.parse import urlparse, parse_qs
    >>> url = "https://example.com/search?q=machine%20learning%20%26%20AI"
    >>> query_string = urlparse(url).query
    >>> # Extract the raw query value
    >>> raw_q = query_string.split("=")[1]
    >>> url_decode(raw_q)
    'machine learning & AI'

    Decoding form-submitted data:

    >>> form_data = "comment%3A%20This%20is%20%3Cimportant%3E"
    >>> url_decode(form_data)
    'comment: This is <important>'

    Decoding an already-decoded string (no-op):

    >>> url_decode("already decoded text")
    'already decoded text'

    Decoding an empty string:

    >>> url_decode("")
    ''

    Notes
    -----
    - This function handles both ``%XX`` percent-encoding and ``+`` as
      space (form data encoding). For strict percent-decoding only,
      be aware that ``+`` will remain as ``+``.
    - Invalid percent sequences (e.g., ``%ZZ`` or incomplete ``%2``)
      may be passed through unchanged or partially decoded depending
      on the implementation.
    - The function assumes UTF-8 encoding for multi-byte character
      sequences.

    See Also
    --------
    url_encode : Encode text for URL use.
    urllib.parse.unquote : Lower-level URL decoding function.
    urllib.parse.unquote_plus : URL decoding with + as space.
    urllib.parse.parse_qs : Parse query string into dictionary.
    """
    return urllib.parse.unquote(encoded)


def html_encode(text: str) -> str:
    """
    Encode special HTML characters as HTML entities for safe display.

    HTML encoding (also known as HTML escaping) converts characters that
    have special meaning in HTML into their corresponding HTML entity
    representations. This is essential for preventing Cross-Site Scripting
    (XSS) attacks and ensuring that user-provided content displays
    correctly in web pages.

    Parameters
    ----------
    text : str
        The input text string to encode. Any string content is accepted,
        including user-generated content that may contain malicious HTML
        or JavaScript.

    Returns
    -------
    str
        The HTML-encoded string with special characters replaced by
        their HTML entity equivalents:

        - ``&`` becomes ``&amp;``
        - ``<`` becomes ``&lt;``
        - ``>`` becomes ``&gt;``
        - ``"`` becomes ``&quot;``
        - ``'`` becomes ``&#x27;`` (single quote)

    Raises
    ------
    AttributeError
        If ``text`` is not a string (e.g., None or other types).

    Examples
    --------
    Encoding basic HTML special characters:

    >>> html_encode("<div>Hello & goodbye</div>")
    '&lt;div&gt;Hello &amp; goodbye&lt;/div&gt;'

    Preventing XSS attacks by encoding script tags:

    >>> user_input = '<script>alert("XSS attack!")</script>'
    >>> safe_output = html_encode(user_input)
    >>> print(safe_output)
    &lt;script&gt;alert(&quot;XSS attack!&quot;)&lt;/script&gt;

    Encoding attribute values with quotes:

    >>> attr_value = 'onclick="malicious()" data-x="y"'
    >>> html_encode(attr_value)
    'onclick=&quot;malicious()&quot; data-x=&quot;y&quot;'

    Encoding mathematical expressions:

    >>> expression = "x < 10 && y > 5"
    >>> html_encode(expression)
    'x &lt; 10 &amp;&amp; y &gt; 5'

    Safe display of code snippets in HTML:

    >>> code = 'if (a < b && c > d) { return "result"; }'
    >>> encoded_code = html_encode(code)
    >>> html_content = f"<pre><code>{encoded_code}</code></pre>"
    >>> print(html_content)
    <pre><code>if (a &lt; b &amp;&amp; c &gt; d) { return &quot;result&quot;; }</code></pre>

    Encoding user comments for display:

    >>> comment = "I think <b>bold</b> text is <cool> & 'useful'"
    >>> safe_comment = html_encode(comment)
    >>> print(safe_comment)
    I think &lt;b&gt;bold&lt;/b&gt; text is &lt;cool&gt; &amp; &#x27;useful&#x27;

    Encoding an empty string:

    >>> html_encode("")
    ''

    Text without special characters passes through unchanged:

    >>> html_encode("Hello World")
    'Hello World'

    Notes
    -----
    - This function is crucial for web security. Always encode user input
      before inserting it into HTML documents.
    - The function encodes the five predefined XML entities plus the
      single quote for JavaScript compatibility.
    - Unicode characters (e.g., emojis, non-Latin scripts) are NOT
      encoded and pass through unchanged. They are valid in HTML5.
    - For encoding within JavaScript strings, additional escaping may
      be required beyond HTML encoding.
    - This is one-way encoding for display; the browser will render
      the entities as their original characters.

    See Also
    --------
    html_decode : Decode HTML entities back to characters.
    html.escape : Lower-level HTML escaping function.

    Warnings
    --------
    While HTML encoding prevents basic XSS attacks in HTML content, it
    is not sufficient for all contexts. Content placed in JavaScript
    blocks, CSS, or URL attributes may require additional encoding.
    """
    return html.escape(text)


def html_decode(encoded: str) -> str:
    """
    Decode HTML entities back to their corresponding characters.

    HTML decoding (also known as HTML unescaping) converts HTML entity
    references back to their original character representations. This
    is useful when processing HTML content that needs to be displayed
    as plain text or when working with data extracted from HTML documents.

    Parameters
    ----------
    encoded : str
        A string containing HTML entity references. Supports both named
        entities (e.g., ``&amp;``, ``&lt;``) and numeric character
        references (e.g., ``&#60;``, ``&#x3C;``).

    Returns
    -------
    str
        The decoded string with all HTML entities converted back to
        their original characters:

        - ``&amp;`` becomes ``&``
        - ``&lt;`` becomes ``<``
        - ``&gt;`` becomes ``>``
        - ``&quot;`` becomes ``"``
        - ``&#x27;`` or ``&#39;`` becomes ``'``
        - Plus all other named HTML5 character references

    Raises
    ------
    AttributeError
        If ``encoded`` is not a string (e.g., None or other types).

    Examples
    --------
    Decoding basic HTML entities:

    >>> html_decode("&lt;div&gt;Hello &amp; goodbye&lt;/div&gt;")
    '<div>Hello & goodbye</div>'

    Decoding escaped script tags:

    >>> escaped = "&lt;script&gt;alert(&quot;test&quot;)&lt;/script&gt;"
    >>> html_decode(escaped)
    '<script>alert("test")</script>'

    Decoding numeric character references (decimal):

    >>> html_decode("&#60;p&#62;")
    '<p>'

    Decoding numeric character references (hexadecimal):

    >>> html_decode("&#x3C;p&#x3E;")
    '<p>'

    Round-trip encoding and decoding:

    >>> original = '<a href="test.html">Link & More</a>'
    >>> encoded = html_encode(original)
    >>> decoded = html_decode(encoded)
    >>> original == decoded
    True

    Decoding named entities for special symbols:

    >>> html_decode("&copy; 2024 &mdash; All rights reserved &trade;")
    '© 2024 — All rights reserved ™'

    Decoding mathematical symbols:

    >>> html_decode("x &ne; y &and; a &lt; b")
    'x ≠ y ∧ a < b'

    Decoding Greek letters:

    >>> html_decode("&alpha; + &beta; = &gamma;")
    'α + β = γ'

    Processing text extracted from HTML:

    >>> html_content = "Tom &amp; Jerry&#39;s &quot;Adventures&quot;"
    >>> plain_text = html_decode(html_content)
    >>> print(plain_text)
    Tom & Jerry's "Adventures"

    Decoding currency symbols:

    >>> html_decode("Price: &pound;50 or &euro;60")
    'Price: £50 or €60'

    Text without entities passes through unchanged:

    >>> html_decode("Hello World")
    'Hello World'

    Decoding an empty string:

    >>> html_decode("")
    ''

    Notes
    -----
    - This function supports all HTML5 named character references,
      including over 2000 named entities.
    - Both decimal (``&#NNN;``) and hexadecimal (``&#xHHH;``) numeric
      character references are supported.
    - Invalid or malformed entity references may be passed through
      unchanged or partially decoded.
    - Be cautious when decoding untrusted HTML content, as the decoded
      output may contain executable code or malicious content.

    See Also
    --------
    html_encode : Encode special characters as HTML entities.
    html.unescape : Lower-level HTML unescaping function.

    Warnings
    --------
    Decoding HTML entities from untrusted sources can reintroduce
    XSS vulnerabilities. Never decode HTML and then insert the result
    back into an HTML document without re-encoding it first.
    """
    return html.unescape(encoded)
