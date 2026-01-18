"""Tests for benchmark dataset utilities."""

import tempfile
from pathlib import Path

import pytest

from insideLLMs.benchmark_datasets import (
    BenchmarkDataset,
    CrossValidator,
    DatasetBuilder,
    DatasetCategory,
    DatasetExample,
    DatasetRegistry,
    DatasetStats,
    SamplingStrategy,
    SplitInfo,
    SplitType,
    create_dataset,
    create_factual_dataset,
    create_math_dataset,
    create_reasoning_dataset,
    cross_validate,
    filter_dataset,
    get_dataset,
    get_dataset_stats,
    get_default_registry,
    list_datasets,
    merge_datasets,
    register_dataset,
    sample_dataset,
)


class TestDatasetCategory:
    """Tests for DatasetCategory enum."""

    def test_all_categories_exist(self):
        """Test that all categories exist."""
        assert DatasetCategory.REASONING.value == "reasoning"
        assert DatasetCategory.FACTUAL.value == "factual"
        assert DatasetCategory.MATH.value == "math"
        assert DatasetCategory.CODING.value == "coding"
        assert DatasetCategory.SAFETY.value == "safety"
        assert DatasetCategory.BIAS.value == "bias"
        assert DatasetCategory.CUSTOM.value == "custom"


class TestSplitType:
    """Tests for SplitType enum."""

    def test_all_splits_exist(self):
        """Test that all split types exist."""
        assert SplitType.TRAIN.value == "train"
        assert SplitType.VALIDATION.value == "validation"
        assert SplitType.TEST.value == "test"
        assert SplitType.ALL.value == "all"


class TestSamplingStrategy:
    """Tests for SamplingStrategy enum."""

    def test_all_strategies_exist(self):
        """Test that all strategies exist."""
        assert SamplingStrategy.RANDOM.value == "random"
        assert SamplingStrategy.STRATIFIED.value == "stratified"
        assert SamplingStrategy.SEQUENTIAL.value == "sequential"
        assert SamplingStrategy.BALANCED.value == "balanced"


class TestDatasetExample:
    """Tests for DatasetExample class."""

    def test_basic_creation(self):
        """Test basic example creation."""
        example = DatasetExample(
            id="ex1",
            input_text="What is 2+2?",
            expected_output="4",
        )
        assert example.id == "ex1"
        assert example.input_text == "What is 2+2?"
        assert example.expected_output == "4"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        example = DatasetExample(
            id="ex1",
            input_text="Question",
            expected_output="Answer",
            category="test",
            difficulty="easy",
        )
        d = example.to_dict()
        assert d["id"] == "ex1"
        assert d["input_text"] == "Question"
        assert d["category"] == "test"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "id": "ex1",
            "input_text": "Question",
            "expected_output": "Answer",
            "category": "test",
        }
        example = DatasetExample.from_dict(data)
        assert example.id == "ex1"
        assert example.input_text == "Question"

    def test_from_dict_alternative_keys(self):
        """Test creation from dict with alternative keys."""
        data = {
            "question": "What?",
            "answer": "Yes",
        }
        example = DatasetExample.from_dict(data)
        assert example.input_text == "What?"
        assert example.expected_output == "Yes"


class TestDatasetStats:
    """Tests for DatasetStats class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = DatasetStats(
            total_examples=100,
            categories={"cat1": 50, "cat2": 50},
            difficulties={"easy": 30, "hard": 70},
            avg_input_length=50.5,
            avg_output_length=10.25,
            min_input_length=5,
            max_input_length=200,
            sources={"source1": 100},
        )
        d = stats.to_dict()
        assert d["total_examples"] == 100
        assert d["avg_input_length"] == 50.5


class TestSplitInfo:
    """Tests for SplitInfo class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        info = SplitInfo(
            train_size=80,
            validation_size=10,
            test_size=10,
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
        )
        d = info.to_dict()
        assert d["train_size"] == 80
        assert d["train_ratio"] == 0.8


class TestBenchmarkDataset:
    """Tests for BenchmarkDataset class."""

    def test_create_empty_dataset(self):
        """Test creating empty dataset."""
        dataset = BenchmarkDataset("test")
        assert dataset.name == "test"
        assert len(dataset) == 0

    def test_add_example(self):
        """Test adding example."""
        dataset = BenchmarkDataset("test")
        example = DatasetExample(id="1", input_text="Question")
        dataset.add_example(example)
        assert len(dataset) == 1

    def test_add_examples(self):
        """Test adding multiple examples."""
        dataset = BenchmarkDataset("test")
        examples = [
            DatasetExample(id="1", input_text="Q1"),
            DatasetExample(id="2", input_text="Q2"),
        ]
        dataset.add_examples(examples)
        assert len(dataset) == 2

    def test_auto_generate_id(self):
        """Test automatic ID generation."""
        dataset = BenchmarkDataset("test")
        example = DatasetExample(id="", input_text="Question")
        dataset.add_example(example)
        assert example.id != ""
        assert len(example.id) == 12

    def test_get_example(self):
        """Test getting example by ID."""
        dataset = BenchmarkDataset("test")
        example = DatasetExample(id="ex1", input_text="Question")
        dataset.add_example(example)
        retrieved = dataset.get_example("ex1")
        assert retrieved is not None
        assert retrieved.input_text == "Question"

    def test_get_example_not_found(self):
        """Test getting non-existent example."""
        dataset = BenchmarkDataset("test")
        retrieved = dataset.get_example("nonexistent")
        assert retrieved is None

    def test_iteration(self):
        """Test iterating over dataset."""
        dataset = BenchmarkDataset("test")
        dataset.add_examples(
            [
                DatasetExample(id="1", input_text="Q1"),
                DatasetExample(id="2", input_text="Q2"),
            ]
        )
        examples = list(dataset)
        assert len(examples) == 2

    def test_indexing(self):
        """Test indexing dataset."""
        dataset = BenchmarkDataset("test")
        dataset.add_example(DatasetExample(id="1", input_text="Q1"))
        assert dataset[0].input_text == "Q1"

    def test_split(self):
        """Test splitting dataset."""
        dataset = BenchmarkDataset("test")
        for i in range(100):
            dataset.add_example(DatasetExample(id=str(i), input_text=f"Q{i}"))

        info = dataset.split(train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, seed=42)

        assert info.train_size == 80
        assert info.validation_size == 10
        assert info.test_size == 10

    def test_split_invalid_ratio(self):
        """Test split with invalid ratios."""
        dataset = BenchmarkDataset("test")
        with pytest.raises(ValueError):
            dataset.split(train_ratio=0.5, validation_ratio=0.3, test_ratio=0.3)

    def test_get_examples_by_split(self):
        """Test getting examples by split."""
        dataset = BenchmarkDataset("test")
        for i in range(100):
            dataset.add_example(DatasetExample(id=str(i), input_text=f"Q{i}"))

        dataset.split(train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, seed=42)

        train = dataset.get_examples(split=SplitType.TRAIN)
        val = dataset.get_examples(split=SplitType.VALIDATION)
        test = dataset.get_examples(split=SplitType.TEST)

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_get_examples_by_category(self):
        """Test filtering examples by category."""
        dataset = BenchmarkDataset("test")
        dataset.add_examples(
            [
                DatasetExample(id="1", input_text="Q1", category="cat1"),
                DatasetExample(id="2", input_text="Q2", category="cat2"),
                DatasetExample(id="3", input_text="Q3", category="cat1"),
            ]
        )

        cat1 = dataset.get_examples(category="cat1")
        assert len(cat1) == 2

    def test_sample_random(self):
        """Test random sampling."""
        dataset = BenchmarkDataset("test")
        for i in range(100):
            dataset.add_example(DatasetExample(id=str(i), input_text=f"Q{i}"))

        samples = dataset.sample(10, strategy=SamplingStrategy.RANDOM, seed=42)
        assert len(samples) == 10

    def test_sample_sequential(self):
        """Test sequential sampling."""
        dataset = BenchmarkDataset("test")
        for i in range(100):
            dataset.add_example(DatasetExample(id=str(i), input_text=f"Q{i}"))

        samples = dataset.sample(10, strategy=SamplingStrategy.SEQUENTIAL)
        assert len(samples) == 10
        assert samples[0].id == "0"

    def test_sample_stratified(self):
        """Test stratified sampling."""
        dataset = BenchmarkDataset("test")
        for i in range(50):
            dataset.add_example(DatasetExample(id=str(i), input_text=f"Q{i}", category="cat1"))
        for i in range(50, 100):
            dataset.add_example(DatasetExample(id=str(i), input_text=f"Q{i}", category="cat2"))

        samples = dataset.sample(20, strategy=SamplingStrategy.STRATIFIED, seed=42)
        assert len(samples) == 20

    def test_get_stats(self):
        """Test getting dataset statistics."""
        dataset = BenchmarkDataset("test")
        dataset.add_examples(
            [
                DatasetExample(
                    id="1",
                    input_text="Short",
                    expected_output="A",
                    category="cat1",
                    difficulty="easy",
                ),
                DatasetExample(
                    id="2",
                    input_text="A longer question",
                    expected_output="B",
                    category="cat2",
                    difficulty="hard",
                ),
            ]
        )

        stats = dataset.get_stats()
        assert stats.total_examples == 2
        assert "cat1" in stats.categories
        assert "easy" in stats.difficulties

    def test_filter(self):
        """Test filtering dataset."""
        dataset = BenchmarkDataset("test")
        dataset.add_examples(
            [
                DatasetExample(id="1", input_text="Q1", category="cat1"),
                DatasetExample(id="2", input_text="Q2", category="cat2"),
            ]
        )

        filtered = dataset.filter(lambda e: e.category == "cat1")
        assert len(filtered) == 1

    def test_map_examples(self):
        """Test mapping examples."""
        dataset = BenchmarkDataset("test")
        dataset.add_example(DatasetExample(id="1", input_text="question"))

        mapped = dataset.map_examples(
            lambda e: DatasetExample(
                id=e.id,
                input_text=e.input_text.upper(),
                expected_output=e.expected_output,
            )
        )
        assert mapped[0].input_text == "QUESTION"

    def test_merge(self):
        """Test merging datasets."""
        dataset1 = BenchmarkDataset("ds1")
        dataset1.add_example(DatasetExample(id="1", input_text="Q1"))

        dataset2 = BenchmarkDataset("ds2")
        dataset2.add_example(DatasetExample(id="2", input_text="Q2"))

        merged = dataset1.merge(dataset2)
        assert len(merged) == 2

    def test_save_and_load(self):
        """Test saving and loading dataset."""
        dataset = BenchmarkDataset("test", category=DatasetCategory.REASONING)
        dataset.add_example(DatasetExample(id="1", input_text="Q1", expected_output="A1"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            dataset.save(path)
            loaded = BenchmarkDataset.load(path)
            assert loaded.name == "test"
            assert len(loaded) == 1
            assert loaded[0].input_text == "Q1"
        finally:
            Path(path).unlink()


class TestDatasetBuilder:
    """Tests for DatasetBuilder class."""

    def test_basic_build(self):
        """Test basic dataset building."""
        builder = DatasetBuilder("test")
        dataset = builder.build()
        assert dataset.name == "test"

    def test_with_category(self):
        """Test setting category."""
        builder = DatasetBuilder("test")
        builder.with_category(DatasetCategory.MATH)
        dataset = builder.build()
        assert dataset.category == DatasetCategory.MATH

    def test_with_description(self):
        """Test setting description."""
        builder = DatasetBuilder("test")
        builder.with_description("Test description")
        dataset = builder.build()
        assert dataset.description == "Test description"

    def test_add_example(self):
        """Test adding examples."""
        builder = DatasetBuilder("test")
        builder.add_example("Question", "Answer", category="test")
        dataset = builder.build()
        assert len(dataset) == 1

    def test_add_examples_from_list(self):
        """Test adding examples from list."""
        builder = DatasetBuilder("test")
        examples = [
            {"input": "Q1", "output": "A1"},
            {"input": "Q2", "output": "A2"},
        ]
        builder.add_examples_from_list(examples)
        dataset = builder.build()
        assert len(dataset) == 2

    def test_chaining(self):
        """Test method chaining."""
        dataset = (
            DatasetBuilder("test")
            .with_category(DatasetCategory.FACTUAL)
            .with_description("Description")
            .add_example("Q", "A")
            .build()
        )
        assert dataset.category == DatasetCategory.FACTUAL
        assert len(dataset) == 1


class TestCrossValidator:
    """Tests for CrossValidator class."""

    def test_basic_split(self):
        """Test basic cross-validation split."""
        dataset = BenchmarkDataset("test")
        for i in range(100):
            dataset.add_example(DatasetExample(id=str(i), input_text=f"Q{i}"))

        validator = CrossValidator(n_folds=5, seed=42)
        folds = list(validator.split(dataset))

        assert len(folds) == 5
        for train, test in folds:
            assert len(train) + len(test) == 100

    def test_no_overlap(self):
        """Test that folds don't overlap."""
        dataset = BenchmarkDataset("test")
        for i in range(100):
            dataset.add_example(DatasetExample(id=str(i), input_text=f"Q{i}"))

        validator = CrossValidator(n_folds=5, seed=42)
        all_test_ids = []

        for train, test in validator.split(dataset):
            train_ids = {e.id for e in train}
            test_ids = {e.id for e in test}
            assert len(train_ids & test_ids) == 0
            all_test_ids.extend([e.id for e in test])

        # All examples should appear in test exactly once
        assert len(all_test_ids) == 100


class TestDatasetRegistry:
    """Tests for DatasetRegistry class."""

    def test_register_and_get(self):
        """Test registering and getting dataset."""
        registry = DatasetRegistry()
        dataset = BenchmarkDataset("test")
        registry.register(dataset)
        retrieved = registry.get("test")
        assert retrieved is not None

    def test_get_not_found(self):
        """Test getting non-existent dataset."""
        registry = DatasetRegistry()
        assert registry.get("nonexistent") is None

    def test_list_datasets(self):
        """Test listing datasets."""
        registry = DatasetRegistry()
        registry.register(BenchmarkDataset("ds1"))
        registry.register(BenchmarkDataset("ds2"))
        names = registry.list_datasets()
        assert len(names) == 2
        assert "ds1" in names

    def test_list_by_category(self):
        """Test listing by category."""
        registry = DatasetRegistry()
        registry.register(BenchmarkDataset("ds1", category=DatasetCategory.MATH))
        registry.register(BenchmarkDataset("ds2", category=DatasetCategory.FACTUAL))
        math_datasets = registry.list_by_category(DatasetCategory.MATH)
        assert len(math_datasets) == 1
        assert "ds1" in math_datasets

    def test_remove(self):
        """Test removing dataset."""
        registry = DatasetRegistry()
        registry.register(BenchmarkDataset("test"))
        assert registry.remove("test")
        assert registry.get("test") is None

    def test_clear(self):
        """Test clearing registry."""
        registry = DatasetRegistry()
        registry.register(BenchmarkDataset("test"))
        registry.clear()
        assert len(registry.list_datasets()) == 0


class TestBuiltInDatasets:
    """Tests for built-in datasets."""

    def test_reasoning_dataset(self):
        """Test reasoning dataset creation."""
        dataset = create_reasoning_dataset()
        assert dataset.name == "reasoning_benchmark"
        assert dataset.category == DatasetCategory.REASONING
        assert len(dataset) > 0

    def test_factual_dataset(self):
        """Test factual dataset creation."""
        dataset = create_factual_dataset()
        assert dataset.name == "factual_benchmark"
        assert dataset.category == DatasetCategory.FACTUAL
        assert len(dataset) > 0

    def test_math_dataset(self):
        """Test math dataset creation."""
        dataset = create_math_dataset()
        assert dataset.name == "math_benchmark"
        assert dataset.category == DatasetCategory.MATH
        assert len(dataset) > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_dataset(self):
        """Test create_dataset function."""
        examples = [
            {"input": "Q1", "output": "A1"},
            {"input": "Q2", "output": "A2"},
        ]
        dataset = create_dataset("test", examples)
        assert len(dataset) == 2

    def test_sample_dataset(self):
        """Test sample_dataset function."""
        dataset = BenchmarkDataset("test")
        for i in range(100):
            dataset.add_example(DatasetExample(id=str(i), input_text=f"Q{i}"))

        samples = sample_dataset(dataset, 10, seed=42)
        assert len(samples) == 10

    def test_cross_validate(self):
        """Test cross_validate function."""
        dataset = BenchmarkDataset("test")
        for i in range(50):
            dataset.add_example(DatasetExample(id=str(i), input_text=f"Q{i}"))

        folds = list(cross_validate(dataset, n_folds=5, seed=42))
        assert len(folds) == 5

    def test_get_dataset_stats(self):
        """Test get_dataset_stats function."""
        dataset = BenchmarkDataset("test")
        dataset.add_example(DatasetExample(id="1", input_text="Question"))
        stats = get_dataset_stats(dataset)
        assert stats.total_examples == 1

    def test_merge_datasets(self):
        """Test merge_datasets function."""
        ds1 = BenchmarkDataset("ds1")
        ds1.add_example(DatasetExample(id="1", input_text="Q1"))
        ds2 = BenchmarkDataset("ds2")
        ds2.add_example(DatasetExample(id="2", input_text="Q2"))

        merged = merge_datasets(ds1, ds2)
        assert len(merged) == 2

    def test_filter_dataset(self):
        """Test filter_dataset function."""
        dataset = BenchmarkDataset("test")
        dataset.add_examples(
            [
                DatasetExample(id="1", input_text="Short", category="cat1"),
                DatasetExample(id="2", input_text="A longer question", category="cat2"),
            ]
        )

        filtered = filter_dataset(dataset, category="cat1")
        assert len(filtered) == 1

    def test_filter_dataset_by_length(self):
        """Test filtering by length."""
        dataset = BenchmarkDataset("test")
        dataset.add_examples(
            [
                DatasetExample(id="1", input_text="Short"),
                DatasetExample(id="2", input_text="A much longer question here"),
            ]
        )

        filtered = filter_dataset(dataset, min_length=10)
        assert len(filtered) == 1


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_register_and_get_dataset(self):
        """Test global register and get."""
        dataset = BenchmarkDataset("global_test")
        register_dataset(dataset)
        retrieved = get_dataset("global_test")
        assert retrieved is not None

    def test_list_datasets(self):
        """Test listing datasets."""
        names = list_datasets()
        assert isinstance(names, list)

    def test_get_default_registry(self):
        """Test getting default registry."""
        registry = get_default_registry()
        assert isinstance(registry, DatasetRegistry)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataset_stats(self):
        """Test stats for empty dataset."""
        dataset = BenchmarkDataset("empty")
        stats = dataset.get_stats()
        assert stats.total_examples == 0
        assert stats.avg_input_length == 0

    def test_sample_more_than_available(self):
        """Test sampling more than available."""
        dataset = BenchmarkDataset("test")
        dataset.add_example(DatasetExample(id="1", input_text="Q1"))
        samples = dataset.sample(10)
        assert len(samples) == 1

    def test_split_small_dataset(self):
        """Test splitting very small dataset."""
        dataset = BenchmarkDataset("test")
        dataset.add_example(DatasetExample(id="1", input_text="Q1"))
        info = dataset.split(seed=42)
        total = info.train_size + info.validation_size + info.test_size
        assert total == 1

    def test_stratified_split(self):
        """Test stratified splitting."""
        dataset = BenchmarkDataset("test")
        for i in range(50):
            dataset.add_example(DatasetExample(id=str(i), input_text=f"Q{i}", category="cat1"))
        for i in range(50, 100):
            dataset.add_example(DatasetExample(id=str(i), input_text=f"Q{i}", category="cat2"))

        info = dataset.split(seed=42, stratify_by="category")
        assert info.train_size > 0

    def test_balanced_sampling(self):
        """Test balanced sampling."""
        dataset = BenchmarkDataset("test")
        for i in range(90):
            dataset.add_example(DatasetExample(id=str(i), input_text=f"Q{i}", category="majority"))
        for i in range(90, 100):
            dataset.add_example(DatasetExample(id=str(i), input_text=f"Q{i}", category="minority"))

        samples = dataset.sample(20, strategy=SamplingStrategy.BALANCED, seed=42)
        assert len(samples) == 20

    def test_merge_empty_raises(self):
        """Test merging no datasets raises error."""
        with pytest.raises(ValueError):
            merge_datasets()
