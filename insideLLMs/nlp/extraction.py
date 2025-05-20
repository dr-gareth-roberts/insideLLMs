import re
from typing import List, Tuple, Dict
from collections import defaultdict

# Optional dependencies
try:
    import spacy
    SPACY_AVAILABLE = False  # Set to False initially, will be True when a model is loaded
    SPACY_MODEL = None
except ImportError:
    SPACY_AVAILABLE = False
    SPACY_MODEL = None

# ===== Dependency Management =====

def check_spacy(model_name: str = "en_core_web_sm"):
    """Check if spaCy is available and load the specified model if needed.

    Args:
        model_name: Name of the spaCy model to load

    Returns:
        The loaded spaCy model
    """
    global SPACY_AVAILABLE, SPACY_MODEL # Ensure we are modifying the global variables

    if not SPACY_AVAILABLE:
        try:
            import spacy as spacy_check_module # Check if spacy can be imported
            SPACY_AVAILABLE = True # If import successful, set to True
        except ImportError:
             raise ImportError(
                "spaCy is not installed. Please install it with: pip install spacy"
            )

    # Load model if not already loaded or if a different model is requested
    if SPACY_MODEL is None or SPACY_MODEL.meta['name'] != model_name:
        try:
            # Ensure spacy is actually imported before trying to load
            if 'spacy' not in globals() and 'spacy' not in locals():
                 import spacy as spacy_load_module
                 SPACY_MODEL = spacy_load_module.load(model_name)
            elif SPACY_MODEL is None or SPACY_MODEL.meta['name'] != model_name : # Ensure spacy.load is called if model is different
                 SPACY_MODEL = spacy.load(model_name)

        except OSError:
            # SPACY_AVAILABLE = False # If model loading fails, spaCy (with this model) is not available
            raise ImportError(
                f"spaCy model '{model_name}' not found. "
                f"Please install it with: python -m spacy download {model_name}"
            )
        except NameError: # If spacy was not imported (should be caught by SPACY_AVAILABLE check)
             raise ImportError(
                "spaCy is not installed. Please install it with: pip install spacy"
            )
    return SPACY_MODEL


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

# Re-initialize SPACY_AVAILABLE based on successful import of spacy module
# This is a workaround for the global variable issue.
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
