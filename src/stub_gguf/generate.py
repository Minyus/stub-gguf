from __future__ import annotations

from pathlib import Path

from stub_gguf.gguf_writer import GGUFWriter
from stub_gguf.model_spec import build_model_spec


def generate_stub_gguf(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec = build_model_spec()
    writer = GGUFWriter(
        architecture="llama",
        metadata=spec.metadata,
        tensors=spec.tensors,
    )
    output_path.write_bytes(writer.to_bytes())
    return output_path
