from pathlib import Path

import pytest

from stub_gguf.validate import validate_artifact


def test_validate_artifact_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_artifact(tmp_path / "missing.gguf")


def test_validate_artifact_rejects_too_small_file(tmp_path: Path) -> None:
    artifact = tmp_path / "stub.gguf"
    artifact.write_bytes(b"GG")

    with pytest.raises(ValueError, match="too small"):
        validate_artifact(artifact)


def test_validate_artifact_rejects_truncated_magic_only_file(tmp_path: Path) -> None:
    artifact = tmp_path / "stub.gguf"
    artifact.write_bytes(b"GGUF" + b"\x00" * 4)

    with pytest.raises(ValueError, match="too small"):
        validate_artifact(artifact)


def test_validate_artifact_rejects_bad_magic(tmp_path: Path) -> None:
    artifact = tmp_path / "stub.gguf"
    artifact.write_bytes(b"NOPE" + b"\x00" * 20)

    with pytest.raises(ValueError, match="magic"):
        validate_artifact(artifact)


def test_validate_artifact_rejects_header_only_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "stub.gguf"
    artifact.write_bytes(b"GGUF" + b"\x00" * 20)

    with pytest.raises(ValueError, match="Unsupported GGUF version"):
        validate_artifact(artifact)
