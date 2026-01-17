"""Tests for the registry system."""

import pytest

from insideLLMs.registry import (
    NotFoundError,
    Registry,
    RegistrationError,
)


class TestRegistry:
    """Tests for the Registry class."""

    def test_create_empty_registry(self):
        """Test creating an empty registry."""
        registry = Registry("test")
        assert registry.name == "test"
        assert len(registry) == 0
        assert registry.list() == []

    def test_register_and_get(self):
        """Test basic registration and retrieval."""
        registry = Registry("items")

        class TestClass:
            def __init__(self, value=10):
                self.value = value

        registry.register("test", TestClass)

        instance = registry.get("test")
        assert isinstance(instance, TestClass)
        assert instance.value == 10

    def test_register_with_default_kwargs(self):
        """Test registration with default arguments."""
        registry = Registry("items")

        class TestClass:
            def __init__(self, item_name, value=0):
                self.item_name = item_name
                self.value = value

        registry.register("test", TestClass, item_name="default_name", value=42)

        instance = registry.get("test")
        assert instance.item_name == "default_name"
        assert instance.value == 42

    def test_get_with_override_kwargs(self):
        """Test overriding default kwargs when getting."""
        registry = Registry("items")

        class TestClass:
            def __init__(self, value=10):
                self.value = value

        registry.register("test", TestClass, value=10)

        instance = registry.get("test", value=99)
        assert instance.value == 99

    def test_duplicate_registration_raises(self):
        """Test that duplicate registration raises error."""
        registry = Registry("items")

        class TestClass:
            pass

        registry.register("test", TestClass)

        with pytest.raises(RegistrationError):
            registry.register("test", TestClass)

    def test_duplicate_registration_with_overwrite(self):
        """Test overwriting existing registration."""
        registry = Registry("items")

        class TestClass1:
            pass

        class TestClass2:
            pass

        registry.register("test", TestClass1)
        registry.register("test", TestClass2, overwrite=True)

        instance = registry.get("test")
        assert isinstance(instance, TestClass2)

    def test_get_not_found_raises(self):
        """Test that getting non-existent item raises error."""
        registry = Registry("items")

        with pytest.raises(NotFoundError) as exc_info:
            registry.get("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "items" in str(exc_info.value)

    def test_is_registered(self):
        """Test checking if item is registered."""
        registry = Registry("items")

        class TestClass:
            pass

        assert not registry.is_registered("test")
        registry.register("test", TestClass)
        assert registry.is_registered("test")

    def test_contains(self):
        """Test using 'in' operator."""
        registry = Registry("items")

        class TestClass:
            pass

        assert "test" not in registry
        registry.register("test", TestClass)
        assert "test" in registry

    def test_list_registered(self):
        """Test listing all registered items."""
        registry = Registry("items")

        class TestClass:
            pass

        registry.register("alpha", TestClass)
        registry.register("beta", TestClass)
        registry.register("gamma", TestClass)

        items = registry.list()
        assert "alpha" in items
        assert "beta" in items
        assert "gamma" in items
        assert len(items) == 3

    def test_unregister(self):
        """Test unregistering an item."""
        registry = Registry("items")

        class TestClass:
            pass

        registry.register("test", TestClass)
        assert "test" in registry

        registry.unregister("test")
        assert "test" not in registry

    def test_unregister_not_found_raises(self):
        """Test that unregistering non-existent item raises error."""
        registry = Registry("items")

        with pytest.raises(NotFoundError):
            registry.unregister("nonexistent")

    def test_clear(self):
        """Test clearing all registrations."""
        registry = Registry("items")

        class TestClass:
            pass

        registry.register("a", TestClass)
        registry.register("b", TestClass)
        assert len(registry) == 2

        registry.clear()
        assert len(registry) == 0

    def test_get_factory(self):
        """Test getting the factory without instantiating."""
        registry = Registry("items")

        class TestClass:
            pass

        registry.register("test", TestClass)

        factory = registry.get_factory("test")
        assert factory is TestClass

    def test_info(self):
        """Test getting registration info."""
        registry = Registry("items")

        class TestClass:
            """A test class."""
            pass

        registry.register("test", TestClass, value=10)

        info = registry.info("test")
        assert info["name"] == "test"
        assert info["factory"] == "TestClass"
        assert info["default_kwargs"] == {"value": 10}
        assert "A test class" in info["doc"]

    def test_register_decorator(self):
        """Test decorator-based registration."""
        registry = Registry("items")

        @registry.register_decorator("decorated")
        class DecoratedClass:
            def __init__(self, value=5):
                self.value = value

        assert "decorated" in registry
        instance = registry.get("decorated")
        assert isinstance(instance, DecoratedClass)
        assert instance.value == 5

    def test_register_decorator_default_name(self):
        """Test decorator uses class name by default."""
        registry = Registry("items")

        @registry.register_decorator()
        class AutoNamedClass:
            pass

        assert "AutoNamedClass" in registry

    def test_repr(self):
        """Test string representation."""
        registry = Registry("test_items")

        class TestClass:
            pass

        registry.register("a", TestClass)
        registry.register("b", TestClass)

        repr_str = repr(registry)
        assert "test_items" in repr_str
        assert "a" in repr_str
        assert "b" in repr_str


class TestRegistryWithFunctions:
    """Tests for using functions as factories."""

    def test_register_function(self):
        """Test registering a function as factory."""
        registry = Registry("factories")

        def create_dict(key, value):
            return {key: value}

        registry.register("dict_maker", create_dict, key="default_key", value="default_value")

        result = registry.get("dict_maker")
        assert result == {"default_key": "default_value"}

    def test_register_lambda(self):
        """Test registering a lambda as factory."""
        registry = Registry("factories")

        registry.register("adder", lambda x, y: x + y, x=5, y=3)

        result = registry.get("adder")
        assert result == 8
