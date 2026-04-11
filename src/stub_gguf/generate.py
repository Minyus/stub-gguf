from __future__ import annotations

import os
import tempfile
from pathlib import Path

from stub_gguf.convert import resolve_convert_script
from stub_gguf.convert import run_conversion
from stub_gguf.hf_stub_builder import build_hf_stub
from stub_gguf.gguf_writer import GGUFWriter
from stub_gguf.model_spec import build_model_spec
from stub_gguf.model_spec import DEFAULT_OUTPUT
from stub_gguf.model_spec import TinyLlamaSpec


def generate_artifact(output_path: Path = DEFAULT_OUTPUT) -> Path:
    resolve_convert_script()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output_path.parent)) as workspace_dir:
        model_dir = build_hf_stub(Path(workspace_dir), TinyLlamaSpec())
        fd, temp_output = tempfile.mkstemp(
            dir=str(output_path.parent),
            prefix=f"{output_path.name}.",
            suffix=".tmp",
        )
        os.close(fd)
        temp_output_path = Path(temp_output)
        try:
            run_conversion(model_dir, temp_output_path)
            os.replace(temp_output_path, output_path)
        except Exception:
            temp_output_path.unlink(missing_ok=True)
            raise
    return output_path


def generate_stub_gguf(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec = build_model_spec()
    writer = GGUFWriter(architecture="llama", metadata=spec.metadata, tensors=spec.tensors)
    output_path.write_bytes(writer.to_bytes())
    return output_path
