from __future__ import annotations

from pathlib import Path

from stub_gguf.generate import generate_stub_gguf
from stub_gguf.validate import read_header


def test_generate_stub_gguf_writes_aligned_tensor_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "noted.gguf"

    generate_stub_gguf(output_path)
    payload = output_path.read_bytes()
    header = read_header(output_path)

    assert header.tensor_count == 4
    assert b"tokenizer.ggml.tokens" in payload
    assert b"Noted" in payload


def test_repository_contains_runtime_assets() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dockerfile = repo_root / "Dockerfile"
    modelfile = repo_root / "ollama" / "Modelfile"
    readme_path = repo_root / "README.md"

    assert dockerfile.exists()
    assert modelfile.exists()

    docker_text = dockerfile.read_text()
    modelfile_text = modelfile.read_text()
    readme = readme_path.read_text()

    assert 'CMD ["uv", "run", "stub-gguf", "generate", "--output", "dist/noted.gguf"]' in docker_text
    assert "FROM ./dist/noted.gguf" in modelfile_text
    assert "PARAMETER temperature 0" in modelfile_text
    assert "SYSTEM \"You are a stub smoke-test model." in modelfile_text
    assert "LM Studio" in readme
    assert "ollama create" in readme
    assert "temperature" in readme.lower()
