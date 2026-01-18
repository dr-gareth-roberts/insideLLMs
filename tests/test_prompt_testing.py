"""Tests for prompt testing and experimentation utilities."""

from insideLLMs.prompt_testing import (
    ABTestRunner,
    ExperimentResult,
    PromptExperiment,
    PromptScorer,
    PromptStrategy,
    PromptTemplate,
    PromptTestResult,
    PromptVariant,
    PromptVariationGenerator,
    ScoringCriteria,
    create_cot_variants,
    create_few_shot_variants,
)


class TestPromptStrategy:
    """Tests for PromptStrategy enum."""

    def test_strategies_exist(self):
        """Test that all strategies are defined."""
        assert PromptStrategy.ZERO_SHOT.value == "zero_shot"
        assert PromptStrategy.FEW_SHOT.value == "few_shot"
        assert PromptStrategy.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert PromptStrategy.STEP_BY_STEP.value == "step_by_step"
        assert PromptStrategy.ROLE_PLAY.value == "role_play"


class TestPromptVariant:
    """Tests for PromptVariant."""

    def test_basic_creation(self):
        """Test basic variant creation."""
        variant = PromptVariant(id="test", content="Hello")
        assert variant.id == "test"
        assert variant.content == "Hello"
        assert variant.strategy is None

    def test_with_strategy(self):
        """Test variant with strategy."""
        variant = PromptVariant(
            id="cot",
            content="Think step by step",
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
        )
        assert variant.strategy == PromptStrategy.CHAIN_OF_THOUGHT

    def test_hashable(self):
        """Test that variants are hashable."""
        variant = PromptVariant(id="test", content="Hello")
        variant_set = {variant}
        assert variant in variant_set


class TestPromptTestResult:
    """Tests for PromptTestResult."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = PromptTestResult(
            variant_id="v1",
            response="Answer",
            score=0.85,
            latency_ms=150.0,
        )
        assert result.variant_id == "v1"
        assert result.response == "Answer"
        assert result.score == 0.85
        assert result.latency_ms == 150.0
        assert result.error is None

    def test_with_error(self):
        """Test result with error."""
        result = PromptTestResult(
            variant_id="v1",
            response="",
            score=0.0,
            latency_ms=10.0,
            error="Connection failed",
        )
        assert result.error == "Connection failed"


class TestExperimentResult:
    """Tests for ExperimentResult."""

    def test_get_variant_scores(self):
        """Test getting scores by variant."""
        results = [
            PromptTestResult(variant_id="v1", response="", score=0.8, latency_ms=100),
            PromptTestResult(variant_id="v1", response="", score=0.9, latency_ms=100),
            PromptTestResult(variant_id="v2", response="", score=0.7, latency_ms=100),
        ]
        experiment = ExperimentResult(
            experiment_id="exp1",
            variants=[],
            results=results,
        )

        scores = experiment.get_variant_scores()
        assert scores["v1"] == [0.8, 0.9]
        assert scores["v2"] == [0.7]

    def test_get_average_scores(self):
        """Test getting average scores."""
        results = [
            PromptTestResult(variant_id="v1", response="", score=0.8, latency_ms=100),
            PromptTestResult(variant_id="v1", response="", score=0.9, latency_ms=100),
            PromptTestResult(variant_id="v2", response="", score=0.7, latency_ms=100),
        ]
        experiment = ExperimentResult(
            experiment_id="exp1",
            variants=[],
            results=results,
        )

        averages = experiment.get_average_scores()
        assert abs(averages["v1"] - 0.85) < 0.001
        assert averages["v2"] == 0.7

    def test_get_best_variant(self):
        """Test getting best variant."""
        results = [
            PromptTestResult(variant_id="v1", response="", score=0.8, latency_ms=100),
            PromptTestResult(variant_id="v2", response="", score=0.9, latency_ms=100),
        ]
        experiment = ExperimentResult(
            experiment_id="exp1",
            variants=[],
            results=results,
        )

        assert experiment.get_best_variant() == "v2"


class TestPromptVariationGenerator:
    """Tests for PromptVariationGenerator."""

    def test_basic_generate(self):
        """Test basic generation returns base prompt."""
        gen = PromptVariationGenerator("What is 2+2?")
        variants = gen.generate()
        assert len(variants) == 1
        assert variants[0].content == "What is 2+2?"

    def test_add_prefix_variations(self):
        """Test prefix variations."""
        gen = PromptVariationGenerator("What is 2+2?")
        gen.add_prefix_variations(["Expert:", "Helper:"])
        variants = gen.generate()

        assert len(variants) == 2
        assert "Expert:" in variants[0].content
        assert "Helper:" in variants[1].content

    def test_add_suffix_variations(self):
        """Test suffix variations."""
        gen = PromptVariationGenerator("What is 2+2?")
        gen.add_suffix_variations(["Be brief.", "Be detailed."])
        variants = gen.generate()

        assert len(variants) == 2
        assert "Be brief." in variants[0].content
        assert "Be detailed." in variants[1].content

    def test_add_strategy_variations(self):
        """Test strategy variations."""
        gen = PromptVariationGenerator("Solve: 2+2")
        gen.add_strategy_variations()
        variants = gen.generate()

        assert len(variants) >= 3
        strategies = [v.strategy for v in variants]
        assert PromptStrategy.ZERO_SHOT in strategies
        assert PromptStrategy.CHAIN_OF_THOUGHT in strategies

    def test_add_custom_variation(self):
        """Test custom variation."""
        gen = PromptVariationGenerator("Base prompt")
        gen.add_custom_variation(
            variant_id="custom1",
            content="Custom prompt text",
            strategy=PromptStrategy.ROLE_PLAY,
        )
        variants = gen.generate()

        assert len(variants) == 1
        assert variants[0].id == "custom1"
        assert variants[0].content == "Custom prompt text"

    def test_generate_combinations(self):
        """Test combination generation."""
        gen = PromptVariationGenerator("Question?")
        variants = gen.generate_combinations(
            prefixes=["A:", "B:"],
            suffixes=["End1", "End2"],
        )

        assert len(variants) == 4

    def test_chaining(self):
        """Test method chaining."""
        gen = PromptVariationGenerator("Base")
        result = (
            gen.add_prefix_variations(["P1"])
            .add_suffix_variations(["S1"])
            .add_strategy_variations()
        )

        assert result is gen


class TestPromptScorer:
    """Tests for PromptScorer."""

    def test_empty_scorer(self):
        """Test scorer with no criteria returns 1.0."""
        scorer = PromptScorer()
        score, details = scorer.score("prompt", "response")
        assert score == 1.0
        assert details == {}

    def test_add_criteria(self):
        """Test adding custom criteria."""
        scorer = PromptScorer()
        scorer.add_criteria(
            "always_pass",
            lambda p, r, e: 1.0,
        )
        score, details = scorer.score("prompt", "response")
        assert score == 1.0
        assert "always_pass" in details

    def test_length_criteria(self):
        """Test length-based scoring."""
        scorer = PromptScorer()
        scorer.add_length_criteria(min_length=10, max_length=100)

        # Too short
        score1, _ = scorer.score("p", "short")
        assert score1 < 1.0

        # Good length
        score2, _ = scorer.score("p", "This is a reasonable length response.")
        assert score2 == 1.0

        # Too long
        long_response = "word " * 100
        score3, _ = scorer.score("p", long_response)
        assert score3 < 1.0

    def test_keyword_criteria(self):
        """Test keyword-based scoring."""
        scorer = PromptScorer()
        scorer.add_keyword_criteria(["python", "programming"])

        # Has both keywords
        score1, _ = scorer.score("p", "Python is a programming language.")
        assert score1 == 1.0

        # Has one keyword
        score2, _ = scorer.score("p", "Python is great.")
        assert score2 == 0.5

        # Has no keywords
        score3, _ = scorer.score("p", "Hello world.")
        assert score3 == 0.0

    def test_similarity_criteria(self):
        """Test similarity scoring."""
        scorer = PromptScorer()
        scorer.add_similarity_criteria()

        # Exact match
        score1, _ = scorer.score("p", "Hello world", "Hello world")
        assert score1 == 1.0

        # Partial overlap
        score2, _ = scorer.score("p", "Hello there", "Hello world")
        assert 0 < score2 < 1

        # No expected response
        score3, _ = scorer.score("p", "Any response", "")
        assert score3 == 1.0

    def test_format_criteria_json(self):
        """Test JSON format scoring."""
        scorer = PromptScorer()
        scorer.add_format_criteria("json")

        # Valid JSON
        score1, _ = scorer.score("p", '{"key": "value"}')
        assert score1 == 1.0

        # Invalid JSON
        score2, _ = scorer.score("p", "not json at all")
        assert score2 < 1.0

    def test_format_criteria_bullets(self):
        """Test bullet format scoring."""
        scorer = PromptScorer()
        scorer.add_format_criteria("bullets")

        # Has bullets
        score1, _ = scorer.score("p", "- item 1\n- item 2")
        assert score1 == 1.0

        # No bullets
        score2, _ = scorer.score("p", "Just plain text")
        assert score2 == 0.0

    def test_weighted_scoring(self):
        """Test weighted scoring."""
        scorer = PromptScorer()
        scorer.add_criteria("high_weight", lambda p, r, e: 1.0, weight=3.0)
        scorer.add_criteria("low_weight", lambda p, r, e: 0.0, weight=1.0)

        score, _ = scorer.score("p", "r")
        # Weighted average: (1.0 * 3 + 0.0 * 1) / 4 = 0.75
        assert score == 0.75


class TestABTestRunner:
    """Tests for ABTestRunner."""

    def test_test_variant(self):
        """Test running a single variant."""
        runner = ABTestRunner()
        variant = PromptVariant(id="v1", content="Hello")

        def mock_model(prompt: str) -> str:
            return "Response"

        results = runner.test_variant(variant, mock_model)

        assert len(results) == 1
        assert results[0].variant_id == "v1"
        assert results[0].response == "Response"

    def test_test_variant_multiple_runs(self):
        """Test multiple runs per variant."""
        runner = ABTestRunner()
        variant = PromptVariant(id="v1", content="Hello")

        def mock_model(prompt: str) -> str:
            return "Response"

        results = runner.test_variant(variant, mock_model, num_runs=3)

        assert len(results) == 3

    def test_test_variant_with_error(self):
        """Test handling errors."""
        runner = ABTestRunner()
        variant = PromptVariant(id="v1", content="Hello")

        def failing_model(prompt: str) -> str:
            raise ValueError("Test error")

        results = runner.test_variant(variant, failing_model)

        assert len(results) == 1
        assert results[0].error == "Test error"
        assert results[0].score == 0.0

    def test_run_experiment(self):
        """Test running full experiment."""
        runner = ABTestRunner()
        variants = [
            PromptVariant(id="v1", content="Hello"),
            PromptVariant(id="v2", content="Hi"),
        ]

        def mock_model(prompt: str) -> str:
            return f"Response to: {prompt}"

        result = runner.run_experiment(variants, mock_model)

        assert result.experiment_id is not None
        assert len(result.variants) == 2
        assert len(result.results) == 2
        assert result.best_variant_id is not None


class TestPromptTemplate:
    """Tests for PromptTemplate with variables."""

    def test_no_variables(self):
        """Test template with no variables."""
        template = PromptTemplate(template="Static prompt")
        results = list(template.expand())

        assert len(results) == 1
        assert results[0][0] == "Static prompt"
        assert results[0][1] == {}

    def test_single_variable(self):
        """Test template with single variable."""
        template = PromptTemplate(
            template="Hello {name}!",
            variables={"name": ["Alice", "Bob"]},
        )
        results = list(template.expand())

        assert len(results) == 2
        assert ("Hello Alice!", {"name": "Alice"}) in results
        assert ("Hello Bob!", {"name": "Bob"}) in results

    def test_multiple_variables(self):
        """Test template with multiple variables."""
        template = PromptTemplate(
            template="{greeting} {name}!",
            variables={
                "greeting": ["Hi", "Hello"],
                "name": ["Alice", "Bob"],
            },
        )
        results = list(template.expand())

        assert len(results) == 4


class TestPromptExperiment:
    """Tests for PromptExperiment."""

    def test_add_variant(self):
        """Test adding variants."""
        exp = PromptExperiment("test")
        exp.add_variant("Prompt 1")
        exp.add_variant("Prompt 2", variant_id="custom_id")

        assert len(exp.variants) == 2
        assert exp.variants[0].id == "variant_0"
        assert exp.variants[1].id == "custom_id"

    def test_chaining(self):
        """Test method chaining."""
        exp = (
            PromptExperiment("test")
            .add_variant("V1")
            .add_variant("V2")
            .configure_scorer(length_range=(10, 100))
        )

        assert len(exp.variants) == 2

    def test_add_variants_from_generator(self):
        """Test adding from generator."""
        exp = PromptExperiment("test")
        gen = PromptVariationGenerator("Base")
        gen.add_prefix_variations(["P1", "P2"])

        exp.add_variants_from_generator(gen)
        assert len(exp.variants) == 2

    def test_run_experiment(self):
        """Test running experiment."""
        exp = PromptExperiment("test")
        exp.add_variant("Prompt 1")
        exp.add_variant("Prompt 2")

        def mock_model(prompt: str) -> str:
            return "A valid response that is long enough."

        exp.configure_scorer(length_range=(10, 100))
        result = exp.run(mock_model)

        assert result is not None
        assert exp.results is not None
        assert exp.get_best_prompt() is not None

    def test_generate_report(self):
        """Test report generation."""
        exp = PromptExperiment("test")
        exp.add_variant("Prompt 1")

        # Not run yet
        report = exp.generate_report()
        assert "not been run" in report

        # After running
        def mock_model(prompt: str) -> str:
            return "Response"

        exp.run(mock_model)
        report = exp.generate_report()
        assert "Prompt Experiment" in report
        assert "test" in report


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_few_shot_variants(self):
        """Test creating few-shot variants."""
        examples = [
            {"input": "2+2", "output": "4"},
            {"input": "3+3", "output": "6"},
            {"input": "4+4", "output": "8"},
        ]

        variants = create_few_shot_variants(
            task="Calculate:",
            examples=examples,
            query="5+5",
            num_shots=[0, 1, 2],
        )

        assert len(variants) == 3
        assert variants[0].strategy == PromptStrategy.ZERO_SHOT
        assert variants[1].strategy == PromptStrategy.FEW_SHOT
        assert "2+2" in variants[1].content  # First example included
        assert "3+3" in variants[2].content  # Second example included

    def test_create_cot_variants(self):
        """Test creating chain-of-thought variants."""
        variants = create_cot_variants("What is 2+2?")

        assert len(variants) == 4
        assert variants[0].strategy == PromptStrategy.ZERO_SHOT
        assert variants[1].strategy == PromptStrategy.CHAIN_OF_THOUGHT
        assert "step by step" in variants[1].content.lower()


class TestScoringCriteria:
    """Tests for ScoringCriteria."""

    def test_basic_creation(self):
        """Test basic criteria creation."""
        criteria = ScoringCriteria(
            name="test",
            weight=2.0,
            description="Test criterion",
        )
        assert criteria.name == "test"
        assert criteria.weight == 2.0
        assert criteria.description == "Test criterion"
        assert criteria.scorer is None
