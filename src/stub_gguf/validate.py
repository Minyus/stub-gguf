from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct


@dataclass(frozen=True)
class GGUFHeader:
    magic: bytes
    version: int
    tensor_count: int
    metadata_kv_count: int


def read_header(path: Path) -> GGUFHeader:
    with path.open("rb") as handle:
        data = handle.read(24)
    if len(data) < 24:
        raise ValueError("File too small for GGUF header")
    magic = struct.unpack_from("<4s", data, 0)[0]
    version = struct.unpack_from("<I", data, 4)[0]
    tensor_count = struct.unpack_from("<Q", data, 8)[0]
    metadata_kv_count = struct.unpack_from("<Q", data, 16)[0]
    return GGUFHeader(magic, version, tensor_count, metadata_kv_count)


def _read_string(data: bytes, offset: int) -> int:
    if offset + 8 > len(data):
        raise ValueError("Truncated GGUF string length")
    length = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    end = offset + length
    if end > len(data):
        raise ValueError("Truncated GGUF string")
    return end


def _skip_value(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        raise ValueError("Truncated GGUF value type")
    value_type = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    if value_type == 8:
        return _read_string(data, offset)
    if value_type == 4:
        end = offset + 4
    elif value_type == 10:
        end = offset + 8
    elif value_type == 6:
        end = offset + 4
    elif value_type == 7:
        end = offset + 1
    elif value_type == 9:
        if offset + 12 > len(data):
            raise ValueError("Truncated GGUF array metadata")
        element_type = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        for _ in range(count):
            if element_type == 8:
                offset = _read_string(data, offset)
            elif element_type == 4 or element_type == 6:
                offset += 4
            elif element_type == 10:
                offset += 8
            elif element_type == 7:
                offset += 1
            else:
                raise ValueError("Unsupported GGUF array element type")
            if offset > len(data):
                raise ValueError("Truncated GGUF array data")
        return offset
    else:
        raise ValueError("Unsupported GGUF metadata type")

    if end > len(data):
        raise ValueError("Truncated GGUF metadata value")
    return end


def validate_file(path: Path) -> None:
    header = read_header(path)
    if header.magic != b"GGUF":
        raise ValueError("Invalid magic")
    if header.version != 3:
        raise ValueError("Unsupported GGUF version")
    if header.tensor_count == 0:
        raise ValueError("Expected at least one tensor")
    if header.metadata_kv_count == 0:
        raise ValueError("Expected at least one metadata entry")

    data = path.read_bytes()
    offset = 24
    for _ in range(header.metadata_kv_count):
        offset = _read_string(data, offset)
        offset = _skip_value(data, offset)

    for _ in range(header.tensor_count):
        offset = _read_string(data, offset)
        if offset + 4 > len(data):
            raise ValueError("Truncated GGUF tensor metadata")
        dimensions = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        tensor_info_size = (8 * dimensions) + 4 + 8
        if offset + tensor_info_size > len(data):
            raise ValueError("Truncated GGUF tensor metadata")
        offset += tensor_info_size
