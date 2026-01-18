"""Tests for reasoning chain analysis and CoT evaluation utilities."""

from insideLLMs.reasoning import (
    ChainAnalysis,
    CoTEvaluation,
    CoTEvaluator,
    CoTPromptGenerator,
    ReasoningAnalyzer,
    ReasoningChain,
    ReasoningExtractor,
    ReasoningQuality,
    ReasoningReport,
    ReasoningStep,
    ReasoningType,
    StepType,
    analyze_reasoning,
    assess_reasoning_quality,
    evaluate_cot,
    extract_reasoning,
    generate_cot_prompt,
)


class TestReasoningType:
    """Tests for ReasoningType enum."""

    def test_all_types_exist(self):
        """Test that all reasoning types are defined."""
        assert ReasoningType.DEDUCTIVE.value == "deductive"
        assert ReasoningType.INDUCTIVE.value == "inductive"
        assert ReasoningType.ABDUCTIVE.value == "abductive"
        assert ReasoningType.CAUSAL.value == "causal"
        assert ReasoningType.MATHEMATICAL.value == "mathematical"


class TestStepType:
    """Tests for StepType enum."""

    def test_all_types_exist(self):
        """Test that all step types are defined."""
        assert StepType.PREMISE.value == "premise"
        assert StepType.INFERENCE.value == "inference"
        assert StepType.CALCULATION.value == "calculation"
        assert StepType.CONCLUSION.value == "conclusion"


class TestReasoningQuality:
    """Tests for ReasoningQuality enum."""

    def test_all_levels_exist(self):
        """Test that all quality levels are defined."""
        assert ReasoningQuality.EXCELLENT.value == "excellent"
        assert ReasoningQuality.GOOD.value == "good"
        assert ReasoningQuality.ADEQUATE.value == "adequate"
        assert ReasoningQuality.POOR.value == "poor"
        assert ReasoningQuality.INVALID.value == "invalid"


class TestReasoningStep:
    """Tests for ReasoningStep dataclass."""

    def test_basic_creation(self):
        """Test basic creation."""
        step = ReasoningStep(
            content="This is a reasoning step",
            step_number=1,
            step_type=StepType.PREMISE,
            confidence=0.8,
        )

        assert step.content == "This is a reasoning step"
        assert step.step_number == 1
        assert step.step_type == StepType.PREMISE

    def test_to_dict(self):
        """Test dictionary conversion."""
        step = ReasoningStep(
            content="Test step",
            step_number=2,
            step_type=StepType.INFERENCE,
            depends_on=[1],
        )

        d = step.to_dict()
        assert d["step_number"] == 2
        assert d["step_type"] == "inference"
        assert d["depends_on"] == [1]


class TestReasoningChain:
    """Tests for ReasoningChain dataclass."""

    def test_basic_creation(self):
        """Test basic creation."""
        steps = [
            ReasoningStep("Step 1", 1, StepType.PREMISE),
            ReasoningStep("Step 2", 2, StepType.INFERENCE),
        ]

        chain = ReasoningChain(
            steps=steps,
            conclusion="Therefore, the answer is X",
            reasoning_type=ReasoningType.DEDUCTIVE,
        )

        assert len(chain.steps) == 2
        assert chain.conclusion == "Therefore, the answer is X"

    def test_get_step(self):
        """Test getting step by number."""
        steps = [
            ReasoningStep("Step 1", 1),
            ReasoningStep("Step 2", 2),
        ]
        chain = ReasoningChain(steps=steps)

        step = chain.get_step(1)
        assert step is not None
        assert step.content == "Step 1"

        missing = chain.get_step(99)
        assert missing is None

    def test_get_premises(self):
        """Test getting premise steps."""
        steps = [
            ReasoningStep("Premise", 1, StepType.PREMISE),
            ReasoningStep("Inference", 2, StepType.INFERENCE),
            ReasoningStep("Another premise", 3, StepType.PREMISE),
        ]
        chain = ReasoningChain(steps=steps)

        premises = chain.get_premises()
        assert len(premises) == 2

    def test_get_inferences(self):
        """Test getting inference steps."""
        steps = [
            ReasoningStep("Premise", 1, StepType.PREMISE),
            ReasoningStep("Inference 1", 2, StepType.INFERENCE),
            ReasoningStep("Inference 2", 3, StepType.INFERENCE),
        ]
        chain = ReasoningChain(steps=steps)

        inferences = chain.get_inferences()
        assert len(inferences) == 2

    def test_to_dict(self):
        """Test dictionary conversion."""
        chain = ReasoningChain(
            steps=[ReasoningStep("Test", 1)],
            conclusion="Result",
            reasoning_type=ReasoningType.CAUSAL,
            completeness=0.7,
        )

        d = chain.to_dict()
        assert d["num_steps"] == 1
        assert d["reasoning_type"] == "causal"
        assert d["completeness"] == 0.7


class TestChainAnalysis:
    """Tests for ChainAnalysis dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        chain = ReasoningChain(steps=[ReasoningStep("Test", 1)])
        analysis = ChainAnalysis(
            chain=chain,
            logical_validity=0.8,
            coherence_score=0.7,
            completeness_score=0.6,
            step_quality_scores=[0.7, 0.8],
            identified_fallacies=["hasty_generalization"],
            missing_steps=["No conclusion"],
            overall_quality=ReasoningQuality.GOOD,
        )

        d = analysis.to_dict()
        assert d["logical_validity"] == 0.8
        assert d["overall_quality"] == "good"
        assert d["num_fallacies"] == 1


class TestCoTEvaluation:
    """Tests for CoTEvaluation dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        chain = ReasoningChain(steps=[ReasoningStep("Test", 1)])
        evaluation = CoTEvaluation(
            prompt="Test prompt",
            response="Test response",
            chain=chain,
            answer_correct=True,
            reasoning_score=0.8,
            step_accuracy=0.7,
            explanation_quality=0.6,
            improvements=["Add more detail"],
        )

        d = evaluation.to_dict()
        assert d["answer_correct"] is True
        assert d["reasoning_score"] == 0.8


class TestReasoningReport:
    """Tests for ReasoningReport dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        report = ReasoningReport(
            total_evaluations=10,
            avg_reasoning_score=0.75,
            avg_step_accuracy=0.8,
            reasoning_type_breakdown={"deductive": 0.6, "causal": 0.4},
            common_fallacies=[("hasty_generalization", 3)],
            quality_distribution={"good": 5, "adequate": 3, "poor": 2},
            recommendations=["Improve structure"],
        )

        d = report.to_dict()
        assert d["total_evaluations"] == 10
        assert d["avg_reasoning_score"] == 0.75


class TestReasoningExtractor:
    """Tests for ReasoningExtractor."""

    def test_extract_numbered_steps(self):
        """Test extraction of numbered steps."""
        extractor = ReasoningExtractor()
        text = """
        Step 1: First, we identify the problem.
        Step 2: Then, we analyze the data.
        Step 3: Finally, we draw a conclusion.
        """

        chain = extractor.extract(text)

        assert len(chain.steps) >= 2
        assert chain.is_valid

    def test_extract_simple_steps(self):
        """Test extraction from simple numbered list."""
        extractor = ReasoningExtractor()
        text = """
        1. Given that X is true.
        2. Therefore, Y follows.
        3. In conclusion, Z is the answer.
        """

        chain = extractor.extract(text)

        assert len(chain.steps) >= 2

    def test_classify_step_premise(self):
        """Test classification of premise steps."""
        extractor = ReasoningExtractor()
        step_type = extractor._classify_step("Given that X is true")

        assert step_type == StepType.PREMISE

    def test_classify_step_inference(self):
        """Test classification of inference steps."""
        extractor = ReasoningExtractor()
        step_type = extractor._classify_step("Thus, Y must be true based on X")

        assert step_type == StepType.INFERENCE

    def test_classify_step_calculation(self):
        """Test classification of calculation steps."""
        extractor = ReasoningExtractor()
        step_type = extractor._classify_step("Calculate 5 + 3 = 8")

        assert step_type == StepType.CALCULATION

    def test_classify_step_conclusion(self):
        """Test classification of conclusion steps."""
        extractor = ReasoningExtractor()
        step_type = extractor._classify_step("In conclusion, the answer is 42")

        assert step_type == StepType.CONCLUSION

    def test_classify_reasoning_type_mathematical(self):
        """Test classification of mathematical reasoning."""
        extractor = ReasoningExtractor()
        text = "Let's calculate 5 + 3 = 8, then multiply by 2"

        reasoning_type = extractor._classify_reasoning_type(text)

        assert reasoning_type == ReasoningType.MATHEMATICAL

    def test_classify_reasoning_type_causal(self):
        """Test classification of causal reasoning."""
        extractor = ReasoningExtractor()
        text = "Because X happened, it leads to Y"

        reasoning_type = extractor._classify_reasoning_type(text)

        assert reasoning_type == ReasoningType.CAUSAL

    def test_extract_conclusion(self):
        """Test conclusion extraction."""
        extractor = ReasoningExtractor()
        text = "After analysis, therefore the answer is 42."

        conclusion = extractor._extract_conclusion(text)

        assert conclusion is not None
        assert "answer" in conclusion.lower() or "42" in conclusion

    def test_empty_text(self):
        """Test with empty text."""
        extractor = ReasoningExtractor()
        chain = extractor.extract("")

        assert len(chain.steps) == 0
        assert chain.completeness == 0.0


class TestReasoningAnalyzer:
    """Tests for ReasoningAnalyzer."""

    def test_analyze_valid_chain(self):
        """Test analysis of valid chain."""
        analyzer = ReasoningAnalyzer()

        steps = [
            ReasoningStep("Given X is true", 1, StepType.PREMISE),
            ReasoningStep("Therefore Y follows from X", 2, StepType.INFERENCE),
            ReasoningStep("In conclusion, Z is the answer", 3, StepType.CONCLUSION),
        ]
        chain = ReasoningChain(steps=steps, completeness=0.8)

        analysis = analyzer.analyze(chain)

        assert analysis.logical_validity > 0.5
        assert analysis.overall_quality in ReasoningQuality

    def test_analyze_empty_chain(self):
        """Test analysis of empty chain."""
        analyzer = ReasoningAnalyzer()
        chain = ReasoningChain(steps=[], completeness=0.0)

        analysis = analyzer.analyze(chain)

        assert analysis.logical_validity == 0.0
        assert analysis.overall_quality == ReasoningQuality.INVALID

    def test_identify_fallacies(self):
        """Test fallacy identification."""
        analyzer = ReasoningAnalyzer()

        steps = [
            ReasoningStep("Everyone believes this is true", 1),
            ReasoningStep("All scientists agree", 2),
        ]
        chain = ReasoningChain(steps=steps)

        analysis = analyzer.analyze(chain)

        # Should detect hasty_generalization
        assert "hasty_generalization" in analysis.identified_fallacies

    def test_calculate_coherence(self):
        """Test coherence calculation."""
        analyzer = ReasoningAnalyzer()

        # Coherent chain (shared words)
        steps = [
            ReasoningStep("The problem involves X and Y", 1),
            ReasoningStep("X relates to Y through Z", 2),
        ]
        chain = ReasoningChain(steps=steps)

        analysis = analyzer.analyze(chain)

        assert analysis.coherence_score > 0

    def test_identify_missing_steps(self):
        """Test identification of missing steps."""
        analyzer = ReasoningAnalyzer()

        # Chain without premise or conclusion
        steps = [
            ReasoningStep("Some inference", 1, StepType.INFERENCE),
        ]
        chain = ReasoningChain(steps=steps, conclusion=None)

        analysis = analyzer.analyze(chain)

        assert len(analysis.missing_steps) > 0


class TestCoTEvaluator:
    """Tests for CoTEvaluator."""

    def test_evaluate_good_response(self):
        """Test evaluation of good CoT response."""
        evaluator = CoTEvaluator()

        prompt = "What is 2 + 2?"
        response = """
        Let me think step by step.
        Step 1: We have two numbers to add: 2 and 2.
        Step 2: Adding them together: 2 + 2 = 4.
        Therefore, the answer is 4.
        """

        evaluation = evaluator.evaluate(prompt, response, expected_answer="4")

        assert evaluation.answer_correct is True
        assert evaluation.reasoning_score > 0

    def test_evaluate_incorrect_answer(self):
        """Test evaluation with incorrect answer."""
        evaluator = CoTEvaluator()

        prompt = "What is 2 + 2?"
        response = "The answer is 5."

        evaluation = evaluator.evaluate(prompt, response, expected_answer="4")

        assert evaluation.answer_correct is False

    def test_evaluate_batch(self):
        """Test batch evaluation."""
        evaluator = CoTEvaluator()

        prompts = ["Q1", "Q2"]
        responses = ["A1", "A2"]

        evaluations = evaluator.evaluate_batch(prompts, responses)

        assert len(evaluations) == 2

    def test_generate_report(self):
        """Test report generation."""
        evaluator = CoTEvaluator()

        evaluations = [
            CoTEvaluation(
                prompt="Q1",
                response="R1",
                chain=ReasoningChain(steps=[ReasoningStep("S1", 1)]),
                answer_correct=True,
                reasoning_score=0.8,
                step_accuracy=0.7,
                explanation_quality=0.6,
            ),
            CoTEvaluation(
                prompt="Q2",
                response="R2",
                chain=ReasoningChain(steps=[ReasoningStep("S2", 1)]),
                answer_correct=False,
                reasoning_score=0.4,
                step_accuracy=0.5,
                explanation_quality=0.4,
            ),
        ]

        report = evaluator.generate_report(evaluations)

        assert report.total_evaluations == 2
        assert 0.4 <= report.avg_reasoning_score <= 0.8

    def test_generate_report_empty(self):
        """Test report generation with no evaluations."""
        evaluator = CoTEvaluator()
        report = evaluator.generate_report([])

        assert report.total_evaluations == 0

    def test_suggest_improvements(self):
        """Test improvement suggestions."""
        evaluator = CoTEvaluator()

        chain = ReasoningChain(steps=[ReasoningStep("Test", 1)])
        analysis = ChainAnalysis(
            chain=chain,
            logical_validity=0.3,
            coherence_score=0.3,
            completeness_score=0.3,
            step_quality_scores=[0.3],
            identified_fallacies=["circular_reasoning"],
            missing_steps=["Missing conclusion"],
            overall_quality=ReasoningQuality.POOR,
        )

        improvements = evaluator._suggest_improvements(analysis)

        assert len(improvements) > 0


class TestCoTPromptGenerator:
    """Tests for CoTPromptGenerator."""

    def test_generate_standard(self):
        """Test standard prompt generation."""
        generator = CoTPromptGenerator()
        prompt = generator.generate("What is 2+2?", "standard")

        assert "step by step" in prompt.lower()
        assert "2+2" in prompt

    def test_generate_structured(self):
        """Test structured prompt generation."""
        generator = CoTPromptGenerator()
        prompt = generator.generate("What is 2+2?", "structured")

        assert "Step 1" in prompt

    def test_generate_math(self):
        """Test math prompt generation."""
        generator = CoTPromptGenerator()
        prompt = generator.generate("What is 2+2?", "math")

        assert "mathematical" in prompt.lower()

    def test_generate_variations(self):
        """Test generating multiple variations."""
        generator = CoTPromptGenerator()
        variations = generator.generate_variations("Test question?", num_variations=3)

        assert len(variations) >= 3
        assert all("Test question?" in v for v in variations)

    def test_add_custom_template(self):
        """Test adding custom template."""
        generator = CoTPromptGenerator()
        generator.add_template("custom", "Custom: {question}")

        prompt = generator.generate("Test?", "custom")

        assert prompt == "Custom: Test?"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_extract_reasoning(self):
        """Test extract_reasoning function."""
        text = "Step 1: First. Step 2: Second. Therefore, the answer."

        chain = extract_reasoning(text)

        assert isinstance(chain, ReasoningChain)

    def test_analyze_reasoning(self):
        """Test analyze_reasoning function."""
        chain = ReasoningChain(
            steps=[ReasoningStep("Test step", 1)],
        )

        analysis = analyze_reasoning(chain)

        assert isinstance(analysis, ChainAnalysis)

    def test_evaluate_cot(self):
        """Test evaluate_cot function."""
        evaluation = evaluate_cot(
            "What is 2+2?",
            "The answer is 4.",
            expected_answer="4",
        )

        assert isinstance(evaluation, CoTEvaluation)
        assert evaluation.answer_correct is True

    def test_generate_cot_prompt(self):
        """Test generate_cot_prompt function."""
        prompt = generate_cot_prompt("Test question?")

        assert "Test question?" in prompt
        assert "step" in prompt.lower()

    def test_assess_reasoning_quality(self):
        """Test assess_reasoning_quality function."""
        text = """
        Given that X is true.
        Therefore Y follows.
        In conclusion, Z is the answer.
        """

        quality = assess_reasoning_quality(text)

        assert quality in ReasoningQuality


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_long_reasoning(self):
        """Test with very long reasoning text."""
        extractor = ReasoningExtractor()
        long_text = "Step: This is a reasoning step. " * 100

        chain = extractor.extract(long_text)

        # Should handle gracefully and limit steps
        assert len(chain.steps) <= 15

    def test_no_clear_steps(self):
        """Test text with no clear steps."""
        extractor = ReasoningExtractor()
        text = "This is just a simple statement without clear reasoning structure."

        chain = extractor.extract(text)

        # Should still extract something
        assert isinstance(chain, ReasoningChain)

    def test_unicode_content(self):
        """Test with unicode content."""
        extractor = ReasoningExtractor()
        text = "ステップ1: First step. ステップ2: Second step."

        chain = extractor.extract(text)

        assert isinstance(chain, ReasoningChain)

    def test_special_characters(self):
        """Test with special characters."""
        extractor = ReasoningExtractor()
        text = "Step 1: Calculate 2 + 2 = 4! Step 2: Therefore, it's done!!!"

        chain = extractor.extract(text)

        assert isinstance(chain, ReasoningChain)

    def test_mixed_step_formats(self):
        """Test with mixed step formats."""
        extractor = ReasoningExtractor()
        text = """
        First, consider X.
        1. Then, analyze Y.
        Step 3: Finally, conclude Z.
        """

        chain = extractor.extract(text)

        assert len(chain.steps) >= 1

    def test_single_step_reasoning(self):
        """Test with single step."""
        evaluator = CoTEvaluator()

        evaluation = evaluator.evaluate(
            "Simple question",
            "Simple answer.",
        )

        assert isinstance(evaluation, CoTEvaluation)
        assert evaluation.step_accuracy >= 0

    def test_evaluation_without_expected_answer(self):
        """Test evaluation without expected answer."""
        evaluator = CoTEvaluator()

        evaluation = evaluator.evaluate(
            "Open-ended question",
            "Some response with reasoning.",
        )

        assert evaluation.answer_correct is None

    def test_coherence_with_unrelated_steps(self):
        """Test coherence with completely unrelated steps."""
        analyzer = ReasoningAnalyzer()

        steps = [
            ReasoningStep("The sky is blue", 1),
            ReasoningStep("Pizza is delicious", 2),
        ]
        chain = ReasoningChain(steps=steps)

        analysis = analyzer.analyze(chain)

        # Should have low coherence
        assert analysis.coherence_score < 0.5
