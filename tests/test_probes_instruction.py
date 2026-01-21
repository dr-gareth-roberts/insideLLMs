"""Tests for instruction following probes."""

import pytest

from insideLLMs.models import DummyModel
from insideLLMs.probes.instruction import (
    ConstraintComplianceProbe,
    InstructionFollowingProbe,
    MultiStepTaskProbe,
)
from insideLLMs.types import ResultStatus


class TestInstructionFollowingProbe:
    """Tests for InstructionFollowingProbe class."""

    def test_basic_initialization(self):
        """Test basic probe initialization."""
        probe = InstructionFollowingProbe()
        assert probe.name == "InstructionFollowingProbe"
        assert probe.strict_mode is False

    def test_initialization_with_strict_mode(self):
        """Test initialization with strict mode."""
        probe = InstructionFollowingProbe(strict_mode=True)
        assert probe.strict_mode is True

    def test_run_with_dict_data(self):
        """Test running probe with dict data containing task and constraints."""
        probe = InstructionFollowingProbe()
        model = DummyModel()

        data = {
            "task": "List 3 fruits",
            "constraints": {"format": "numbered_list", "max_items": 3},
        }

        result = probe.run(model, data)
        assert isinstance(result, str)

    def test_run_with_string_data(self):
        """Test running probe with string data."""
        probe = InstructionFollowingProbe()
        model = DummyModel()

        result = probe.run(model, "List 3 fruits")
        assert isinstance(result, str)

    def test_run_with_instruction_key(self):
        """Test running probe with 'instruction' key instead of 'task'."""
        probe = InstructionFollowingProbe()
        model = DummyModel()

        data = {"instruction": "List 3 colors"}
        result = probe.run(model, data)
        assert isinstance(result, str)

    def test_format_constraints_json(self):
        """Test formatting JSON constraint."""
        probe = InstructionFollowingProbe()
        result = probe._format_constraints({"format": "json"})
        assert "JSON" in result

    def test_format_constraints_numbered_list(self):
        """Test formatting numbered list constraint."""
        probe = InstructionFollowingProbe()
        result = probe._format_constraints({"format": "numbered_list"})
        assert "numbered list" in result

    def test_format_constraints_bullet_list(self):
        """Test formatting bullet list constraint."""
        probe = InstructionFollowingProbe()
        result = probe._format_constraints({"format": "bullet_list"})
        assert "bullet" in result

    def test_format_constraints_single_word(self):
        """Test formatting single word constraint."""
        probe = InstructionFollowingProbe()
        result = probe._format_constraints({"format": "single_word"})
        assert "single word" in result

    def test_format_constraints_single_sentence(self):
        """Test formatting single sentence constraint."""
        probe = InstructionFollowingProbe()
        result = probe._format_constraints({"format": "single_sentence"})
        assert "one sentence" in result

    def test_format_constraints_paragraph(self):
        """Test formatting paragraph constraint."""
        probe = InstructionFollowingProbe()
        result = probe._format_constraints({"format": "paragraph"})
        assert "paragraph" in result

    def test_format_constraints_code(self):
        """Test formatting code constraint."""
        probe = InstructionFollowingProbe()
        result = probe._format_constraints({"format": "code"})
        assert "code" in result

    def test_format_constraints_custom_format(self):
        """Test formatting custom format constraint."""
        probe = InstructionFollowingProbe()
        result = probe._format_constraints({"format": "xml"})
        assert "xml" in result

    def test_format_constraints_word_limits(self):
        """Test formatting word limit constraints."""
        probe = InstructionFollowingProbe()
        result = probe._format_constraints({"max_words": 50, "min_words": 10})
        assert "50 words" in result
        assert "10 words" in result

    def test_format_constraints_item_limits(self):
        """Test formatting item limit constraints."""
        probe = InstructionFollowingProbe()
        result = probe._format_constraints({"max_items": 5, "min_items": 2})
        assert "5 items" in result
        assert "2 items" in result

    def test_format_constraints_keywords(self):
        """Test formatting keyword constraints."""
        probe = InstructionFollowingProbe()
        result = probe._format_constraints({
            "include_keywords": ["apple", "banana"],
            "exclude_keywords": ["orange"],
        })
        assert "apple" in result
        assert "banana" in result
        assert "orange" in result

    def test_format_constraints_language_and_tone(self):
        """Test formatting language and tone constraints."""
        probe = InstructionFollowingProbe()
        result = probe._format_constraints({
            "language": "Spanish",
            "tone": "formal",
        })
        assert "Spanish" in result
        assert "formal" in result

    def test_check_format_json_valid(self):
        """Test checking valid JSON format."""
        probe = InstructionFollowingProbe()
        result = probe._check_format('{"key": "value"}', "json")
        assert result == 1.0

    def test_check_format_json_invalid_but_looks_like(self):
        """Test checking invalid JSON that looks JSON-like."""
        probe = InstructionFollowingProbe()
        result = probe._check_format('{key: value}', "json")
        assert result == 0.5

    def test_check_format_json_array_like(self):
        """Test checking array-like invalid JSON."""
        probe = InstructionFollowingProbe()
        result = probe._check_format('[item1, item2]', "json")
        assert result == 0.5

    def test_check_format_json_completely_invalid(self):
        """Test checking completely invalid JSON."""
        probe = InstructionFollowingProbe()
        result = probe._check_format('just some text', "json")
        assert result == 0.0

    def test_check_format_numbered_list(self):
        """Test checking numbered list format."""
        probe = InstructionFollowingProbe()
        output = "1. First item\n2. Second item\n3. Third item"
        result = probe._check_format(output, "numbered_list")
        assert result == 1.0

    def test_check_format_numbered_list_partial(self):
        """Test checking partial numbered list format."""
        probe = InstructionFollowingProbe()
        output = "1. First item\nSome text\n2. Second item"
        result = probe._check_format(output, "numbered_list")
        assert 0.0 < result < 1.0

    def test_check_format_bullet_list(self):
        """Test checking bullet list format."""
        probe = InstructionFollowingProbe()
        output = "- First item\n- Second item\n- Third item"
        result = probe._check_format(output, "bullet_list")
        assert result == 1.0

    def test_check_format_bullet_list_asterisk(self):
        """Test checking bullet list format with asterisks."""
        probe = InstructionFollowingProbe()
        output = "* First item\n* Second item"
        result = probe._check_format(output, "bullet_list")
        assert result == 1.0

    def test_check_format_single_word(self):
        """Test checking single word format."""
        probe = InstructionFollowingProbe()
        assert probe._check_format("word", "single_word") == 1.0
        # Multiple words get penalized
        result = probe._check_format("two words", "single_word")
        assert result < 1.0

    def test_check_format_single_sentence(self):
        """Test checking single sentence format."""
        probe = InstructionFollowingProbe()
        assert probe._check_format("This is one sentence.", "single_sentence") == 1.0
        # Multiple sentences get penalized
        result = probe._check_format("Sentence one. Sentence two.", "single_sentence")
        assert result < 1.0

    def test_check_format_code(self):
        """Test checking code format."""
        probe = InstructionFollowingProbe()
        result = probe._check_format("def hello():\n    pass", "code")
        assert result == 1.0

    def test_check_format_code_with_backticks(self):
        """Test checking code format with backticks."""
        probe = InstructionFollowingProbe()
        result = probe._check_format("```python\nprint('hello')\n```", "code")
        assert result == 1.0

    def test_check_format_code_no_indicators(self):
        """Test checking non-code as code format."""
        probe = InstructionFollowingProbe()
        result = probe._check_format("just regular text", "code")
        assert result == 0.3

    def test_check_format_unknown(self):
        """Test checking unknown format."""
        probe = InstructionFollowingProbe()
        result = probe._check_format("any text", "unknown_format")
        assert result == 1.0

    def test_count_items_numbered(self):
        """Test counting numbered items."""
        probe = InstructionFollowingProbe()
        output = "1. First\n2. Second\n3. Third"
        assert probe._count_items(output) == 3

    def test_count_items_bulleted(self):
        """Test counting bulleted items."""
        probe = InstructionFollowingProbe()
        output = "- First\n- Second\n- Third"
        assert probe._count_items(output) == 3

    def test_count_items_none(self):
        """Test counting items when there are none."""
        probe = InstructionFollowingProbe()
        output = "Just regular text without lists."
        assert probe._count_items(output) == 0

    def test_evaluate_single_with_format(self):
        """Test evaluating single output with format constraint."""
        probe = InstructionFollowingProbe()
        output = '{"name": "test"}'
        reference = {"constraints": {"format": "json"}}
        result = probe.evaluate_single(output, reference)
        assert result.metadata["format_compliance"] == 1.0

    def test_evaluate_single_with_word_limits(self):
        """Test evaluating with word count constraints."""
        probe = InstructionFollowingProbe()
        output = "one two three four five"  # 5 words
        reference = {"constraints": {"max_words": 10, "min_words": 3}}
        result = probe.evaluate_single(output, reference)
        assert result.metadata["word_count"] == 5
        assert result.metadata["within_max_words"] is True
        assert result.metadata["within_min_words"] is True

    def test_evaluate_single_with_item_limits(self):
        """Test evaluating with item count constraints."""
        probe = InstructionFollowingProbe()
        output = "1. First\n2. Second\n3. Third"
        reference = {"constraints": {"max_items": 5, "min_items": 2}}
        result = probe.evaluate_single(output, reference)
        assert result.metadata["item_count"] == 3
        assert result.metadata["within_max_items"] is True
        assert result.metadata["within_min_items"] is True

    def test_evaluate_single_with_include_keywords(self):
        """Test evaluating with include keyword constraint."""
        probe = InstructionFollowingProbe()
        output = "I love apple and banana fruits"
        reference = {"constraints": {"include_keywords": ["apple", "banana", "orange"]}}
        result = probe.evaluate_single(output, reference)
        assert "apple" in result.metadata["included_keywords"]
        assert "banana" in result.metadata["included_keywords"]

    def test_evaluate_single_with_exclude_keywords(self):
        """Test evaluating with exclude keyword constraint."""
        probe = InstructionFollowingProbe()
        output = "I love apple and banana"
        reference = {"constraints": {"exclude_keywords": ["orange", "grape"]}}
        result = probe.evaluate_single(output, reference)
        # No violated exclusions
        assert result.metadata["violated_exclusions"] == []

    def test_evaluate_single_exclude_violation(self):
        """Test evaluating when exclude keyword is violated."""
        probe = InstructionFollowingProbe()
        output = "I love apple and orange"
        reference = {"constraints": {"exclude_keywords": ["orange"]}}
        result = probe.evaluate_single(output, reference)
        assert "orange" in result.metadata["violated_exclusions"]

    def test_evaluate_single_no_constraints(self):
        """Test evaluating with no constraints."""
        probe = InstructionFollowingProbe()
        output = "Any text"
        reference = {}
        result = probe.evaluate_single(output, reference)
        assert result.metadata["score"] == 1.0

    def test_evaluate_single_non_dict_reference(self):
        """Test evaluating with non-dict reference."""
        probe = InstructionFollowingProbe()
        output = "Any text"
        result = probe.evaluate_single(output, "non-dict")
        assert result.metadata["score"] == 1.0

    def test_evaluate_single_strict_mode(self):
        """Test evaluating in strict mode."""
        probe = InstructionFollowingProbe(strict_mode=True)
        output = "one two three four five six seven eight nine ten eleven"  # 11 words
        reference = {"constraints": {"max_words": 10}}
        result = probe.evaluate_single(output, reference)
        # In strict mode, any violation should fail
        assert result.metadata["score"] == 0.0


class TestMultiStepTaskProbe:
    """Tests for MultiStepTaskProbe class."""

    def test_basic_initialization(self):
        """Test basic probe initialization."""
        probe = MultiStepTaskProbe()
        assert probe.name == "MultiStepTaskProbe"

    def test_run_with_dict_steps(self):
        """Test running probe with dict containing steps."""
        probe = MultiStepTaskProbe()
        model = DummyModel()

        data = {
            "steps": ["Step 1", "Step 2", "Step 3"],
            "preamble": "Please complete the following:",
        }
        result = probe.run(model, data)
        assert isinstance(result, str)

    def test_run_with_list_data(self):
        """Test running probe with list of steps."""
        probe = MultiStepTaskProbe()
        model = DummyModel()

        result = probe.run(model, ["Step 1", "Step 2"])
        assert isinstance(result, str)

    def test_run_with_string_data(self):
        """Test running probe with string data."""
        probe = MultiStepTaskProbe()
        model = DummyModel()

        result = probe.run(model, "Single task")
        assert isinstance(result, str)

    def test_evaluate_single_with_steps(self):
        """Test evaluating multi-step output."""
        probe = MultiStepTaskProbe()
        output = "Step 1: First thing\nStep 2: Second thing\nStep 3: Third thing"
        reference = {"steps": ["First", "Second", "Third"]}
        result = probe.evaluate_single(output, reference)
        assert "step_indicators_found" in result.metadata

    def test_evaluate_single_with_expected_patterns(self):
        """Test evaluating with expected patterns."""
        probe = MultiStepTaskProbe()
        output = "Step 1: Python is great\nStep 2: JavaScript is popular"
        reference = {
            "steps": ["Describe Python", "Describe JavaScript"],
            "expected": {
                "step_1": ["python"],
                "step_2": ["javascript"],
            },
        }
        result = probe.evaluate_single(output, reference)
        assert result.metadata["step_1"] == 1.0
        assert result.metadata["step_2"] == 1.0

    def test_evaluate_single_expected_string_pattern(self):
        """Test evaluating with string pattern instead of list."""
        probe = MultiStepTaskProbe()
        output = "Step 1: Contains keyword here"
        reference = {
            "steps": ["Find keyword"],
            "expected": {"step_1": "keyword"},  # String, not list
        }
        result = probe.evaluate_single(output, reference)
        assert result.metadata["step_1"] == 1.0

    def test_evaluate_single_non_dict_reference(self):
        """Test evaluating with non-dict reference."""
        probe = MultiStepTaskProbe()
        output = "Some output"
        result = probe.evaluate_single(output, "non-dict")
        # With no steps and no expected patterns, score is based on length
        assert "score" in result.metadata

    def test_evaluate_single_length_score(self):
        """Test that length score is calculated."""
        probe = MultiStepTaskProbe()
        output = "This is a short response"
        reference = {"steps": ["Step 1", "Step 2", "Step 3"]}
        result = probe.evaluate_single(output, reference)
        assert "length_score" in result.metadata
        assert "word_count" in result.metadata


class TestConstraintComplianceProbe:
    """Tests for ConstraintComplianceProbe class."""

    def test_basic_initialization(self):
        """Test basic probe initialization."""
        probe = ConstraintComplianceProbe()
        assert probe.name == "ConstraintComplianceProbe"
        assert probe.constraint_type == "word_limit"

    def test_initialization_with_word_limit(self):
        """Test initialization with word limit."""
        probe = ConstraintComplianceProbe(
            constraint_type="word_limit",
            limit=50,
        )
        assert probe.limit == 50

    def test_initialization_with_character_limit(self):
        """Test initialization with character limit."""
        probe = ConstraintComplianceProbe(
            constraint_type="character_limit",
            limit=100,
        )
        assert probe.constraint_type == "character_limit"

    def test_initialization_with_custom_constraint(self):
        """Test initialization with custom constraint."""
        probe = ConstraintComplianceProbe(
            constraint_type="custom",
            custom_constraint="Use only lowercase",
        )
        assert probe.custom_constraint == "Use only lowercase"

    def test_initialization_with_validator(self):
        """Test initialization with custom validator."""
        validator = lambda x: x.islower()
        probe = ConstraintComplianceProbe(
            constraint_type="custom",
            validator=validator,
        )
        assert probe.validator is not None

    def test_run_with_dict_data(self):
        """Test running probe with dict data."""
        probe = ConstraintComplianceProbe(constraint_type="word_limit", limit=50)
        model = DummyModel()

        data = {"task": "Summarize AI"}
        result = probe.run(model, data)
        assert isinstance(result, str)

    def test_run_with_string_data(self):
        """Test running probe with string data."""
        probe = ConstraintComplianceProbe(constraint_type="word_limit", limit=50)
        model = DummyModel()

        result = probe.run(model, "Summarize AI")
        assert isinstance(result, str)

    def test_get_constraint_instruction_word_limit(self):
        """Test getting word limit instruction."""
        probe = ConstraintComplianceProbe(constraint_type="word_limit", limit=50)
        instruction = probe._get_constraint_instruction()
        assert "50 words" in instruction

    def test_get_constraint_instruction_character_limit(self):
        """Test getting character limit instruction."""
        probe = ConstraintComplianceProbe(constraint_type="character_limit", limit=100)
        instruction = probe._get_constraint_instruction()
        assert "100 characters" in instruction

    def test_get_constraint_instruction_sentence_limit(self):
        """Test getting sentence limit instruction."""
        probe = ConstraintComplianceProbe(constraint_type="sentence_limit", limit=3)
        instruction = probe._get_constraint_instruction()
        assert "3 sentence" in instruction

    def test_get_constraint_instruction_custom(self):
        """Test getting custom constraint instruction."""
        probe = ConstraintComplianceProbe(
            constraint_type="custom",
            custom_constraint="Use formal language",
        )
        instruction = probe._get_constraint_instruction()
        assert "formal language" in instruction

    def test_get_constraint_instruction_empty(self):
        """Test getting empty instruction when no constraint set."""
        probe = ConstraintComplianceProbe(constraint_type="word_limit")
        instruction = probe._get_constraint_instruction()
        assert instruction == ""

    def test_evaluate_single_word_limit_compliant(self):
        """Test evaluating compliant word limit."""
        probe = ConstraintComplianceProbe(constraint_type="word_limit", limit=10)
        output = "one two three four five"  # 5 words
        result = probe.evaluate_single(output, None)
        assert result.metadata["word_count"] == 5
        assert result.status == ResultStatus.SUCCESS

    def test_evaluate_single_word_limit_over(self):
        """Test evaluating exceeding word limit."""
        probe = ConstraintComplianceProbe(constraint_type="word_limit", limit=3)
        output = "one two three four five"  # 5 words
        result = probe.evaluate_single(output, None)
        assert result.metadata["word_count"] == 5
        # Score should be penalized
        assert result.metadata["score"] < 1.0

    def test_evaluate_single_character_limit_compliant(self):
        """Test evaluating compliant character limit."""
        probe = ConstraintComplianceProbe(constraint_type="character_limit", limit=100)
        output = "short text"  # 10 chars
        result = probe.evaluate_single(output, None)
        assert result.metadata["character_count"] == 10
        assert result.status == ResultStatus.SUCCESS

    def test_evaluate_single_character_limit_over(self):
        """Test evaluating exceeding character limit."""
        probe = ConstraintComplianceProbe(constraint_type="character_limit", limit=5)
        output = "longer text"  # > 5 chars
        result = probe.evaluate_single(output, None)
        assert result.metadata["score"] < 1.0

    def test_evaluate_single_sentence_limit_compliant(self):
        """Test evaluating compliant sentence limit."""
        probe = ConstraintComplianceProbe(constraint_type="sentence_limit", limit=3)
        output = "First sentence. Second sentence."
        result = probe.evaluate_single(output, None)
        assert result.metadata["sentence_count"] == 2
        assert result.status == ResultStatus.SUCCESS

    def test_evaluate_single_sentence_limit_over(self):
        """Test evaluating exceeding sentence limit."""
        probe = ConstraintComplianceProbe(constraint_type="sentence_limit", limit=1)
        output = "First sentence. Second sentence. Third sentence."
        result = probe.evaluate_single(output, None)
        assert result.metadata["sentence_count"] == 3
        assert result.metadata["score"] < 1.0

    def test_evaluate_single_with_validator_pass(self):
        """Test evaluating with passing validator."""
        validator = lambda x: x.startswith("Hello")
        probe = ConstraintComplianceProbe(
            constraint_type="custom",
            validator=validator,
        )
        output = "Hello world"
        result = probe.evaluate_single(output, None)
        assert result.metadata["custom_validation"] is True
        assert result.metadata["score"] == 1.0

    def test_evaluate_single_with_validator_fail(self):
        """Test evaluating with failing validator."""
        validator = lambda x: x.startswith("Hello")
        probe = ConstraintComplianceProbe(
            constraint_type="custom",
            validator=validator,
        )
        output = "Goodbye world"
        result = probe.evaluate_single(output, None)
        assert result.metadata["custom_validation"] is False
        assert result.metadata["score"] == 0.0

    def test_evaluate_single_no_constraint(self):
        """Test evaluating when no constraint is set."""
        probe = ConstraintComplianceProbe(constraint_type="unknown")
        output = "Any text"
        result = probe.evaluate_single(output, None)
        assert result.metadata["score"] == 1.0

    def test_evaluate_single_with_reference_override(self):
        """Test evaluating with reference override for limit."""
        probe = ConstraintComplianceProbe(constraint_type="word_limit", limit=100)
        output = "one two three"  # 3 words
        result = probe.evaluate_single(output, 5)  # Override limit to 5
        assert result.metadata["limit"] == 5
