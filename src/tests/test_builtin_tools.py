import sys

from agent_sdk.builtin_tools import WORKDIR, read_file, run_command, write_file
from agent_sdk.types import RunConfig


def test_read_and_write_file(tmp_path):
    path = tmp_path / "x.txt"
    assert write_file.invoke({"path": str(path), "content": "hello"}, RunConfig()).startswith("wrote")
    assert read_file.invoke({"path": str(path)}, RunConfig()) == "hello"


def test_run_command_uses_argv_shell_false():
    out = run_command.invoke({"argv": [sys.executable, "-c", "print('ok')"]}, RunConfig())
    assert "exit=0" in out and "ok" in out


def test_run_command_uses_fixed_workdir():
    out = run_command.invoke(
        {"argv": [sys.executable, "-c", "import os; print(os.getcwd())"]},
        RunConfig(),
    )
    assert str(WORKDIR) in out


def test_run_command_rejects_empty_argv():
    out = run_command.invoke({"argv": []}, RunConfig())
    assert out.startswith("error: ValidationError:")
