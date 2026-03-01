"""Additional branch coverage for prompt utility helpers."""

from __future__ import annotations

import builtins

import pytest

import insideLLMs.contrib.prompt_utils as prompt_utils
from insideLLMs.contrib.prompt_utils import (
    PromptBuilder,
    PromptLibrary,
    PromptValidator,
    escape_template,
    get_default_library,
    render_jinja_template,
    split_prompt_by_tokens,
)


def test_render_jinja_template_success():
    rendered = render_jinja_template("Hello {{ name }}!", {"name": "World"})
    assert rendered == "Hello World!"


def test_render_jinja_template_import_error(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def _mock_import(name: str, *args, **kwargs):
        if name == "jinja2":
            raise ImportError("jinja2 missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)

    with pytest.raises(ImportError, match="jinja2 is required"):
        render_jinja_template("{{ name }}", {"name": "x"})


def test_prompt_builder_build_messages_returns_prompt_objects():
    builder = PromptBuilder().system("S").user("U")
    messages = builder.build_messages()
    assert len(messages) == 2
    assert messages[0].role.value == "system"
    assert messages[1].role.value == "user"


def test_prompt_library_get_version_alias_and_errors():
    library = PromptLibrary()
    library.register("greeting", "Hello", version="1.0")
    library.alias("hi", "greeting")

    version_obj = library.get_version("hi")
    assert version_obj.version == "1.0"
    assert version_obj.content == "Hello"

    with pytest.raises(KeyError, match="Prompt not found"):
        library.get_version("missing")

    with pytest.raises(KeyError, match="Version 2.0 not found"):
        library.get_version("greeting", version="2.0")


def test_prompt_library_alias_missing_target_raises():
    library = PromptLibrary()
    with pytest.raises(KeyError, match="Prompt not found"):
        library.alias("alias", "nope")


def test_prompt_library_list_versions_missing_prompt_raises():
    library = PromptLibrary()
    with pytest.raises(KeyError, match="Prompt not found"):
        library.list_versions("not-there")


def test_prompt_validator_collects_rule_exceptions():
    validator = PromptValidator().add_rule("explode", lambda _: 1 / 0)
    valid, errors = validator.validate("hello")

    assert not valid
    assert any("Rule explode error" in err for err in errors)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("{name}", "{name}"),
        ("{}", "{{}}"),
        ("}", "}}"),
        ("{v}}", "{v}}}"),
        ("x {bad!} y", "x {{bad!}} y"),
    ],
)
def test_escape_template_edge_cases(raw: str, expected: str):
    assert escape_template(raw) == expected


def test_split_prompt_by_tokens_without_boundary_delimiters():
    prompt = "x" * 180
    chunks = split_prompt_by_tokens(prompt, max_tokens=20, chars_per_token=2, overlap=0)

    assert len(chunks) > 1
    assert "".join(chunks) == prompt


def test_get_default_library_initializes_when_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(prompt_utils, "_default_library", None)
    library = get_default_library()
    assert isinstance(library, PromptLibrary)
