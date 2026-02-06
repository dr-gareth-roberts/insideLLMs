"""Tests for insideLLMs.cli.commands.doctor to increase coverage."""

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.cli.commands.doctor import cmd_doctor

# Skip all tests if nltk not available (doctor command requires it)
pytest.importorskip("nltk")


def _make_args(**kwargs):
    defaults = {"format": "text", "fail_on_warn": False}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestDoctorCommand:
    def test_text_format(self, capsys):
        rc = cmd_doctor(_make_args(format="text"))
        captured = capsys.readouterr()
        assert "insideLLMs Doctor" in captured.out
        assert "Environment" in captured.out
        assert "Diagnostics" in captured.out
        assert rc == 0

    def test_json_format(self, capsys):
        rc = cmd_doctor(_make_args(format="json"))
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "checks" in data
        assert "warnings" in data
        assert isinstance(data["checks"], list)
        assert len(data["checks"]) > 0
        # Core checks should always pass
        check_names = {c["name"] for c in data["checks"]}
        assert "python" in check_names
        assert "platform" in check_names
        assert "insideLLMs" in check_names
        assert rc == 0

    def test_json_format_with_fail_on_warn(self, capsys):
        # If there are warn_checks and fail_on_warn is True, return 1
        rc = cmd_doctor(_make_args(format="json", fail_on_warn=True))
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        # Return code depends on whether there are warnings
        if data["warnings"]:
            assert rc == 1
        else:
            assert rc == 0

    def test_text_format_with_fail_on_warn(self, capsys):
        cmd_doctor(_make_args(format="text", fail_on_warn=True))
        captured = capsys.readouterr()
        assert "insideLLMs Doctor" in captured.out

    def test_checks_structure(self, capsys):
        cmd_doctor(_make_args(format="json"))
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        for check in data["checks"]:
            assert "name" in check
            assert "ok" in check
            assert isinstance(check["ok"], bool)

    def test_all_check_categories(self, capsys):
        cmd_doctor(_make_args(format="json"))
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        names = {c["name"] for c in data["checks"]}
        # Verify all expected check categories are present
        assert "nltk" in names
        assert "matplotlib" in names
        assert "OPENAI_API_KEY" in names
        assert "ANTHROPIC_API_KEY" in names
        assert "INSIDELLMS_RUN_ROOT" in names
