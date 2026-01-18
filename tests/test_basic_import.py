import pytest  # Or unittest if preferred, but pytest is common


def test_import_insideLLMs():
    """Test that the main insideLLMs package can be imported."""
    try:
        import insideLLMs
    except ImportError as e:
        pytest.fail(f"Failed to import insideLLMs: {e}")


def test_import_nlp_module():
    """Test that the nlp module can be imported."""
    try:
        from insideLLMs import nlp
    except ImportError as e:
        pytest.fail(f"Failed to import insideLLMs.nlp: {e}")


def test_import_a_specific_nlp_function():
    """Test that a specific function from the nlp module can be imported."""
    try:
        from insideLLMs.nlp import clean_text  # Assuming clean_text is a valid function
    except ImportError as e:
        pytest.fail(f"Failed to import clean_text from insideLLMs.nlp: {e}")
