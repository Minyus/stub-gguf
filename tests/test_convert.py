from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import cast

import pytest

import stub_gguf.convert as convert_module
from stub_gguf.convert import build_convert_command
from stub_gguf.convert import resolve_default_convert_script
from stub_gguf.convert import run_conversion
from stub_gguf.convert import ConvertScriptNotFoundError


def test_build_convert_command_uses_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = tmp_path / "convert.py"
    script.write_text("print('ok')", encoding="utf-8")
    monkeypatch.setenv("LLAMA_CPP_CONVERT", str(script))

    command = build_convert_command(tmp_path / "model", tmp_path / "output.gguf")

    assert command == [
        sys.executable,
        str(script),
        str(tmp_path / "model"),
        "--outfile",
        str(tmp_path / "output.gguf"),
    ]


def test_build_convert_command_raises_same_error_when_env_override_points_to_missing_script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLAMA_CPP_CONVERT", str(tmp_path / "missing.py"))

    with pytest.raises(ConvertScriptNotFoundError, match="Missing llama.cpp converter"):
        build_convert_command(tmp_path / "model", tmp_path / "output.gguf")


def test_build_convert_command_requires_configured_converter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LLAMA_CPP_CONVERT", raising=False)

    with pytest.raises(ConvertScriptNotFoundError, match="Missing llama.cpp converter"):
        build_convert_command(tmp_path / "model", tmp_path / "output.gguf")


def test_resolve_default_convert_script_errors_when_missing() -> None:
    original_exists = convert_module.Path.exists
    convert_module.Path.exists = lambda self: False  # type: ignore[assignment]
    try:
        with pytest.raises(ConvertScriptNotFoundError, match="Missing llama.cpp converter"):
            resolve_default_convert_script()
    finally:
        convert_module.Path.exists = original_exists  # type: ignore[assignment]


def test_run_conversion_raises_runtime_error_with_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    script = tmp_path / "convert.py"
    script.write_text("print('ok')", encoding="utf-8")
    monkeypatch.setenv("LLAMA_CPP_CONVERT", str(tmp_path / "convert.py"))

    def fake_run_failure(command, check, capture_output, text):  # noqa: ANN001
        captured["command"] = command
        captured["check"] = check
        captured["capture_output"] = capture_output
        captured["text"] = text
        raise subprocess.CalledProcessError(1, command, output="", stderr="bad news")

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run_failure)

    with pytest.raises(RuntimeError, match="bad news"):
        run_conversion(tmp_path / "model", tmp_path / "output.gguf")

    command = cast(list[str], captured["command"])
    assert command[0] == sys.executable
    assert captured["check"] is True
    assert captured["capture_output"] is True
    assert captured["text"] is True


def test_run_conversion_raises_runtime_error_with_stdout_when_stderr_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = tmp_path / "convert.py"
    script.write_text("print('ok')", encoding="utf-8")

    def fake_run_failure(command, check, capture_output, text):  # noqa: ANN001
        raise subprocess.CalledProcessError(1, command, output="fallback news", stderr="")

    monkeypatch.setenv("LLAMA_CPP_CONVERT", str(script))
    monkeypatch.setattr(subprocess, "run", fake_run_failure)

    with pytest.raises(RuntimeError, match="fallback news"):
        run_conversion(tmp_path / "model", tmp_path / "output.gguf")
