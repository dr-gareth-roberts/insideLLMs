"""Tests for insideLLMs.cli.commands.interactive to increase coverage."""

import argparse
from unittest.mock import MagicMock, call, patch

from insideLLMs.cli.commands.interactive import cmd_interactive


def _make_args(**kwargs):
    defaults = {
        "model": "dummy",
        "history_file": "/tmp/test_interactive_history.txt",
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestInteractiveCommand:
    def test_quit_command(self, capsys, tmp_path):
        history_file = str(tmp_path / "history.txt")
        with patch("builtins.input", side_effect=["quit"]):
            rc = cmd_interactive(_make_args(history_file=history_file))
        assert rc == 0

    def test_exit_command(self, capsys, tmp_path):
        history_file = str(tmp_path / "history.txt")
        with patch("builtins.input", side_effect=["exit"]):
            rc = cmd_interactive(_make_args(history_file=history_file))
        assert rc == 0

    def test_q_command(self, capsys, tmp_path):
        history_file = str(tmp_path / "history.txt")
        with patch("builtins.input", side_effect=["q"]):
            rc = cmd_interactive(_make_args(history_file=history_file))
        assert rc == 0

    def test_help_command(self, capsys, tmp_path):
        history_file = str(tmp_path / "history.txt")
        with patch("builtins.input", side_effect=["help", "quit"]):
            rc = cmd_interactive(_make_args(history_file=history_file))
        captured = capsys.readouterr()
        assert "Available Commands" in captured.out
        assert rc == 0

    def test_history_command(self, capsys, tmp_path):
        history_file = str(tmp_path / "history.txt")
        with patch("builtins.input", side_effect=["history", "quit"]):
            rc = cmd_interactive(_make_args(history_file=history_file))
        assert rc == 0

    def test_clear_command(self, capsys, tmp_path):
        history_file = str(tmp_path / "history.txt")
        with patch("builtins.input", side_effect=["clear", "quit"]):
            with patch("os.system"):
                rc = cmd_interactive(_make_args(history_file=history_file))
        assert rc == 0

    def test_empty_prompt(self, capsys, tmp_path):
        history_file = str(tmp_path / "history.txt")
        with patch("builtins.input", side_effect=["", "quit"]):
            rc = cmd_interactive(_make_args(history_file=history_file))
        assert rc == 0

    def test_eof_interrupt(self, capsys, tmp_path):
        history_file = str(tmp_path / "history.txt")
        with patch("builtins.input", side_effect=EOFError):
            rc = cmd_interactive(_make_args(history_file=history_file))
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out
        assert rc == 0

    def test_keyboard_interrupt(self, capsys, tmp_path):
        history_file = str(tmp_path / "history.txt")
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            rc = cmd_interactive(_make_args(history_file=history_file))
        assert rc == 0

    def test_generate_response(self, capsys, tmp_path):
        history_file = str(tmp_path / "history.txt")
        with patch("builtins.input", side_effect=["Hello world", "quit"]):
            rc = cmd_interactive(_make_args(history_file=history_file))
        assert rc == 0

    def test_model_load_failure(self, capsys, tmp_path):
        history_file = str(tmp_path / "history.txt")
        rc = cmd_interactive(_make_args(model="nonexistent_model_xyz", history_file=history_file))
        assert rc == 1

    def test_model_switch(self, capsys, tmp_path):
        history_file = str(tmp_path / "history.txt")
        with patch("builtins.input", side_effect=["model dummy", "quit"]):
            rc = cmd_interactive(_make_args(history_file=history_file))
        assert rc == 0

    def test_model_switch_failure(self, capsys, tmp_path):
        history_file = str(tmp_path / "history.txt")
        with patch("builtins.input", side_effect=["model nonexistent_xyz", "quit"]):
            rc = cmd_interactive(_make_args(history_file=history_file))
        assert rc == 0

    def test_load_existing_history(self, capsys, tmp_path):
        history_file = tmp_path / "history.txt"
        history_file.write_text("previous prompt\n")
        with patch("builtins.input", side_effect=["quit"]):
            rc = cmd_interactive(_make_args(history_file=str(history_file)))
        assert rc == 0

    def test_generate_error(self, capsys, tmp_path):
        history_file = str(tmp_path / "history.txt")
        mock_model = MagicMock()
        mock_model.generate.side_effect = RuntimeError("generation failed")
        with patch("builtins.input", side_effect=["test prompt", "quit"]):
            with patch(
                "insideLLMs.cli.commands.interactive.model_registry"
            ) as mock_registry:
                mock_registry.get.return_value = mock_model
                rc = cmd_interactive(_make_args(history_file=history_file))
        assert rc == 0
