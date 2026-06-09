from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from agent_sdk.tools import tool

WORKDIR = Path.cwd()
MAX_OUTPUT_CHARS = 10_000
TIMEOUT_SECONDS = 30


def _clip(s: str, n: int = MAX_OUTPUT_CHARS) -> str:
    return s if len(s) <= n else s[:n] + "...[truncated]"


class ReadFileArgs(BaseModel):
    path: str


@tool(args=ReadFileArgs)
def read_file(path: str) -> str:
    """Read a UTF-8 text file."""
    return _clip(Path(path).read_text(encoding="utf-8"))


class WriteFileArgs(BaseModel):
    path: str
    content: str


@tool(args=WriteFileArgs)
def write_file(path: str, content: str) -> str:
    """Write a UTF-8 text file."""
    Path(path).write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


class RunCommandArgs(BaseModel):
    argv: list[str] = Field(min_length=1)


@tool(args=RunCommandArgs)
def run_command(argv: list[str]) -> str:
    """Run a demo command with shell=False in the SDK workdir."""
    proc = subprocess.run(
        argv,
        cwd=WORKDIR,
        shell=False,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    out = proc.stdout
    err = proc.stderr
    body = out if not err else f"{out}\n{err}"
    return _clip(f"exit={proc.returncode}\n{body}".rstrip())
