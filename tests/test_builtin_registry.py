"""Tests for built-in model/probe registrations.

These tests validate that the library exposes consistent adapter names via the
global registries used by config-driven runs and the CLI.
"""

from insideLLMs.registry import model_registry, probe_registry, register_builtins


def test_register_builtins_registers_expected_models_and_probes() -> None:
    register_builtins()

    expected_models = {
        "dummy",
        "openai",
        "anthropic",
        "huggingface",
        "gemini",
        "cohere",
        "llamacpp",
        "ollama",
        "vllm",
    }
    expected_probes = {
        "logic",
        "factuality",
        "bias",
        "attack",
        "prompt_injection",
        "jailbreak",
        "code_generation",
        "code_explanation",
        "code_debug",
        "instruction_following",
        "multi_step_task",
        "constraint_compliance",
    }

    assert expected_models.issubset(set(model_registry.list()))
    assert expected_probes.issubset(set(probe_registry.list()))


def test_registry_info_uses_stable_factory_names() -> None:
    register_builtins()

    assert model_registry.info("dummy")["factory"] == "DummyModel"
    assert model_registry.info("openai")["factory"] == "OpenAIModel"
    assert model_registry.info("anthropic")["factory"] == "AnthropicModel"
    assert model_registry.info("huggingface")["factory"] == "HuggingFaceModel"
    assert model_registry.info("gemini")["factory"] == "GeminiModel"
    assert model_registry.info("cohere")["factory"] == "CohereModel"
    assert model_registry.info("llamacpp")["factory"] == "LlamaCppModel"
    assert model_registry.info("ollama")["factory"] == "OllamaModel"
    assert model_registry.info("vllm")["factory"] == "VLLMModel"
