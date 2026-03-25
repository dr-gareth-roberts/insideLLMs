import argparse

from insideLLMs.cli.commands.welcome import cmd_welcome


def test_welcome_command(capsys):
    args = argparse.Namespace()
    result = cmd_welcome(args)

    assert result == 0
    captured = capsys.readouterr()
    assert "Welcome to insideLLMs!" in captured.out
    assert "insidellms quicktest" in captured.out
    assert "insidellms init" in captured.out
