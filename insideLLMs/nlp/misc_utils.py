# Optional dependencies
# These try-except blocks are for setting up global availability flags.
# Individual modules might also have their own try-except for specific imports
# if they don't rely on these global flags directly after refactoring.

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True # Initially assume spacy is available if import spacy works
    SPACY_MODEL = None # Will be loaded by check_spacy
except ImportError:
    SPACY_AVAILABLE = False
    SPACY_MODEL = None

try:
    import sklearn # Just check for sklearn top-level
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import gensim # Just check for gensim top-level
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# ===== Dependency Management Functions =====

def check_nltk_resources(resources: list):
    """
    Checks if NLTK is available and downloads specified resources if needed.
    Helper for other modules that need specific NLTK data.
    """
    if not NLTK_AVAILABLE:
        raise ImportError("NLTK is not installed. Please install it with: pip install nltk")
    
    for resource_path in resources:
        try:
            # NLTK resource paths are typically like 'tokenizers/punkt' or 'corpora/stopwords'
            nltk.data.find(resource_path)
        except LookupError:
            # Resource name is the last part of the path, e.g., 'punkt'
            resource_name = resource_path.split('/')[-1]
            nltk.download(resource_name)
        except Exception as e:
            # Catch other potential errors during resource check/download
            print(f"Error checking/downloading NLTK resource {resource_path}: {e}")


def check_spacy_model(model_name: str = "en_core_web_sm"):
    """
    Checks if spaCy is available and loads the specified model.
    Manages a global SPACY_MODEL instance.
    """
    global SPACY_MODEL, SPACY_AVAILABLE # Ensure we're working with the global variables

    if not SPACY_AVAILABLE: # This means 'import spacy' failed
        raise ImportError("spaCy is not installed. Please install it with: pip install spacy")

    # If spacy module is available, but model is not loaded or is different
    if SPACY_MODEL is None or SPACY_MODEL.meta['name'] != model_name:
        try:
            SPACY_MODEL = spacy.load(model_name)
        except OSError:
            # If model loading fails, it means this specific model is not available
            # We don't set SPACY_AVAILABLE to False here, because spaCy itself is installed.
            # The calling function should handle the case where the model isn't found.
            raise ImportError(
                f"spaCy model '{model_name}' not found. "
                f"Please install it with: python -m spacy download {model_name}"
            )
        except Exception as e: # Catch any other spacy.load error
            raise ImportError(f"Error loading spaCy model '{model_name}': {e}")
    return SPACY_MODEL

def check_sklearn_availability():
    """Checks if scikit-learn is available."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is not installed. Please install it with: pip install scikit-learn")

def check_gensim_availability():
    """Checks if gensim is available."""
    if not GENSIM_AVAILABLE:
        raise ImportError("gensim is not installed. Please install it with: pip install gensim")

# It might be useful to expose the availability flags too for conditional imports in other modules,
# though the check functions themselves are the primary interface.
# Example: get_nltk_availability() -> bool: return NLTK_AVAILABLE

# Note: The original check_nltk, check_spacy, check_sklearn, check_gensim functions 
# from nlp_utils.py were more specific to downloading all common resources (for nltk)
# or just checking the boolean flags. These new versions in misc_utils.py are intended
# to be more granular or to manage the global model state (for spaCy).
# The new modules (text_cleaning.py, tokenization.py etc.) now mostly have their own tailored
# check functions or directly use these from misc_utils if appropriate.
# For this refactoring, the focus is on centralizing the boolean flags and basic model loading.
# The individual modules created earlier already incorporate versions of these checks.
# This misc_utils.py will serve as a central point for these global flags and general check functions
# if needed, reducing redundancy from each new module having its own full copy of all flags/checks.

# The original top-level imports from nlp_utils.py like re, string, unicodedata, etc.
# are NOT needed here as misc_utils.py is for dependency management, not direct NLP tasks.
# Those imports belong in the specific modules that use them.
# The typing imports (List, Dict, etc.) are also not needed here unless these functions
# themselves use complex types, which they currently don't extensively.
# collections.Counter, defaultdict, math also belong to specific modules.
# base64, urllib.parse, html are for encoding.py.
# The goal is to make misc_utils.py lean and focused on cross-cutting concerns like dependency availability.

# Final check: Ensure NLTK_AVAILABLE, SPACY_AVAILABLE, SKLEARN_AVAILABLE, GENSIM_AVAILABLE, SPACY_MODEL
# are correctly managed. SPACY_MODEL being global and modified by check_spacy_model is a key design choice here.
# The individual modules should ideally import these flags/functions from misc_utils
# instead of redefining them. This will be part of a later step or refinement.
# For now, misc_utils.py contains the core global definitions.
# The previous modules were created with copies of these; that will need to be cleaned up.
# This file is the "source of truth" for these global states.
"""Miscellaneous utilities for NLP, primarily for dependency management."""

# Optional dependencies
try:
    import nltk
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

try:
    import spacy
    _SPACY_AVAILABLE = True 
    _SPACY_MODEL_CACHE = {} # Cache for loaded spaCy models
except ImportError:
    _SPACY_AVAILABLE = False
    _SPACY_MODEL_CACHE = {}

try:
    import sklearn
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import gensim
    _GENSIM_AVAILABLE = True
except ImportError:
    _GENSIM_AVAILABLE = False

# --- Public functions to check availability ---
def is_nltk_available() -> bool:
    return _NLTK_AVAILABLE

def is_spacy_available() -> bool:
    return _SPACY_AVAILABLE

def is_sklearn_available() -> bool:
    return _SKLEARN_AVAILABLE

def is_gensim_available() -> bool:
    return _GENSIM_AVAILABLE

# --- Functions to get loaded models or raise errors ---

def get_nltk_resource(resource_path: str, download_if_missing: bool = True):
    """
    Ensures an NLTK resource is available, downloading if necessary.
    Example resource_path: 'tokenizers/punkt', 'corpora/stopwords'
    """
    if not _NLTK_AVAILABLE:
        raise ImportError("NLTK is not installed. Please install it with: pip install nltk")
    try:
        return nltk.data.find(resource_path)
    except LookupError as e:
        if download_if_missing:
            resource_name = resource_path.split('/')[-1]
            print(f"NLTK resource {resource_path} not found. Downloading {resource_name}...")
            nltk.download(resource_name)
            return nltk.data.find(resource_path) # Try finding again after download
        else:
            raise e

def get_spacy_model(model_name: str = "en_core_web_sm", download_if_missing: bool = True):
    """
    Loads and returns a spaCy model, caching it for future calls.
    """
    if not _SPACY_AVAILABLE:
        raise ImportError("spaCy is not installed. Please install it with: pip install spacy")
    
    if model_name not in _SPACY_MODEL_CACHE:
        try:
            _SPACY_MODEL_CACHE[model_name] = spacy.load(model_name)
        except OSError as e: # Model not found
            if download_if_missing:
                print(f"spaCy model '{model_name}' not found. Attempting to download...")
                try:
                    spacy.cli.download(model_name)
                    _SPACY_MODEL_CACHE[model_name] = spacy.load(model_name) # Try loading again
                except Exception as download_exc:
                    raise ImportError(
                        f"spaCy model '{model_name}' not found and download failed. "
                        f"Please install it with: python -m spacy download {model_name}. "
                        f"Download error: {download_exc}"
                    ) from download_exc
            else:
                raise ImportError(
                    f"spaCy model '{model_name}' not found. "
                    f"Please install it with: python -m spacy download {model_name}"
                ) from e
        except Exception as e: # Other errors during spacy.load
            raise ImportError(f"Error loading spaCy model '{model_name}': {e}")
            
    return _SPACY_MODEL_CACHE[model_name]

# Functions to raise ImportError if a library isn't available, for use in other modules.
def ensure_nltk():
    if not _NLTK_AVAILABLE:
        raise ImportError("NLTK is not installed. Please install it with: pip install nltk")

def ensure_spacy():
    if not _SPACY_AVAILABLE:
        raise ImportError("spaCy is not installed. Please install it with: pip install spacy")

def ensure_sklearn():
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is not installed. Please install it with: pip install scikit-learn")

def ensure_gensim():
    if not _GENSIM_AVAILABLE:
        raise ImportError("gensim is not installed. Please install it with: pip install gensim")

# This file, misc_utils.py, should primarily house these kinds of utility/helper
# functions and constants that are about the *environment* or cross-cutting concerns
# for the nlp module, not specific NLP operations.
# The global constants like NLTK_AVAILABLE (now _NLTK_AVAILABLE) are defined here.
# The original check_nltk(), check_spacy(), etc. functions that were very broad
# are effectively replaced by these more specific getters or ensurers.
# The individual NLP modules (text_cleaning, tokenization, etc.) will then import these
# e.g., from .misc_utils import ensure_nltk, get_spacy_model

# Imports like `re`, `string`, `typing` etc. are not needed here as this module
# is focused on dependency management.
# The previously created modules (text_cleaning.py etc.) have their own specific imports.
# This structure avoids circular dependencies and keeps concerns separated.
