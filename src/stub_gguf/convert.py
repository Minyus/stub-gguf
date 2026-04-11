from __future__ import annotations

from pathlib import Path
import os
import sys
import subprocess


DEFAULT_CONVERT_SCRIPT = Path("vendor/llama.cpp/convert_hf_to_gguf.py")
MISSING_CONVERTER_MESSAGE = (
    "Missing llama.cpp converter. Set LLAMA_CPP_CONVERT or place vendor/llama.cpp/convert_hf_to_gguf.py in the repo."
)


class ConvertScriptNotFoundError(RuntimeError):
    pass


def resolve_default_convert_script() -> Path:
    script = Path(__file__).resolve().parents[2] / DEFAULT_CONVERT_SCRIPT
    if not script.exists():
        raise ConvertScriptNotFoundError(MISSING_CONVERTER_MESSAGE)
    return script


def resolve_convert_script() -> Path:
    script = Path(os.environ["LLAMA_CPP_CONVERT"]) if "LLAMA_CPP_CONVERT" in os.environ else resolve_default_convert_script()
    if not script.exists():
        raise ConvertScriptNotFoundError(MISSING_CONVERTER_MESSAGE)
    return script


def build_convert_command(model_dir: Path, output_path: Path) -> list[str]:
    script = resolve_convert_script()
    return [
        sys.executable,
        str(script),
        str(model_dir),
        "--outfile",
        str(output_path),
    ]


def run_conversion(model_dir: Path, output_path: Path) -> None:
    command = build_convert_command(model_dir, output_path)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or "conversion failed").strip()
        raise RuntimeError(message) from exc
