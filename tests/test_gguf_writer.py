from __future__ import annotations

import struct
from pathlib import Path

import pytest

from stub_gguf.generate import generate_stub_gguf
from stub_gguf.gguf_writer import GGMLType, GGUFWriter, MetadataValueType, TensorSpec
from stub_gguf.validate import read_header, validate_file


def test_generated_file_has_tensor_and_metadata_counts(tmp_path: Path) -> None:
    output_path = tmp_path / "noted.gguf"

    generate_stub_gguf(output_path)
    header = read_header(output_path)

    assert header.magic == b"GGUF"
    assert header.version == 3
    assert header.tensor_count == 4
    assert header.metadata_kv_count >= 10


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    length = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    value = data[offset : offset + length].decode("utf-8")
    return value, offset + length


def _skip_value(data: bytes, offset: int) -> int:
    value_type = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    if value_type == MetadataValueType.STRING:
        _, offset = _read_string(data, offset)
        return offset
    if value_type == MetadataValueType.UINT32:
        return offset + 4
    if value_type == MetadataValueType.UINT64:
        return offset + 8
    if value_type == MetadataValueType.FLOAT32:
        return offset + 4
    if value_type == MetadataValueType.BOOL:
        return offset + 1
    if value_type == MetadataValueType.ARRAY:
        element_type = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        for _ in range(count):
            if element_type == MetadataValueType.STRING:
                _, offset = _read_string(data, offset)
            elif element_type == MetadataValueType.UINT32:
                offset += 4
            elif element_type == MetadataValueType.FLOAT32:
                offset += 4
            else:
                raise AssertionError(f"Unsupported metadata element type {element_type}")
        return offset
    raise AssertionError(f"Unsupported metadata value type {value_type}")


def test_generated_file_tensor_offsets_are_32_byte_aligned(tmp_path: Path) -> None:
    output_path = tmp_path / "noted.gguf"
    generate_stub_gguf(output_path)
    data = output_path.read_bytes()

    metadata_count = struct.unpack_from("<Q", data, 16)[0]
    tensor_count = struct.unpack_from("<Q", data, 8)[0]
    offset = 24

    for _ in range(metadata_count):
        _, offset = _read_string(data, offset)
        offset = _skip_value(data, offset)

    offsets: list[int] = []
    for _ in range(tensor_count):
        _, offset = _read_string(data, offset)
        dimensions = struct.unpack_from("<I", data, offset)[0]
        offset += 4 + (8 * dimensions) + 4
        offsets.append(struct.unpack_from("<Q", data, offset)[0])
        offset += 8

    assert offsets
    assert all(value % 32 == 0 for value in offsets)


def test_writer_rejects_wrong_tensor_size() -> None:
    with pytest.raises(ValueError, match="data size mismatch"):
        GGUFWriter(
            architecture="llama",
            metadata={"general.name": (MetadataValueType.STRING, "stub")},
            tensors=[
                TensorSpec(
                    name="token_embd.weight",
                    shape=(2, 2),
                    ggml_type=GGMLType.F32,
                    data=b"\x00" * 4,
                )
            ],
        )


def test_validate_file_accepts_generated_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "noted.gguf"

    generate_stub_gguf(output_path)

    validate_file(output_path)


def test_validate_file_rejects_header_only_file(tmp_path: Path) -> None:
    output_path = tmp_path / "header-only.gguf"
    output_path.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 1, 1))

    with pytest.raises(ValueError, match="Truncated GGUF"):
        validate_file(output_path)


def test_validate_file_rejects_truncated_body(tmp_path: Path) -> None:
    output_path = tmp_path / "truncated.gguf"
    output_path.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 1, 1) + b"\x00")

    with pytest.raises(ValueError, match="Truncated GGUF"):
        validate_file(output_path)
