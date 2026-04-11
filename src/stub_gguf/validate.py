from __future__ import annotations

from pathlib import Path
import struct


_HEADER_STRUCT = struct.Struct("<4sIQQ")
_STRING_LEN_STRUCT = struct.Struct("<Q")
_UINT32_STRUCT = struct.Struct("<I")
_UINT64_STRUCT = struct.Struct("<Q")


_MIN_GGUF_HEADER_SIZE = 24


def validate_artifact(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    if len(data) < _MIN_GGUF_HEADER_SIZE:
        raise ValueError("File too small for GGUF header")
    magic, version, tensor_count, metadata_kv_count = _HEADER_STRUCT.unpack_from(data, 0)
    if magic != b"GGUF":
        raise ValueError("Invalid magic")
    if version != 3:
        raise ValueError(f"Unsupported GGUF version: {version}")

    offset = _HEADER_STRUCT.size
    for _ in range(metadata_kv_count):
        offset = _read_string(data, offset)
        offset = _skip_value(data, offset)

    tensor_infos: list[tuple[tuple[int, ...], int, int]] = []
    for _ in range(tensor_count):
        offset = _read_string(data, offset)
        if offset + 4 > len(data):
            raise ValueError("Truncated GGUF tensor info")
        dimensions = _UINT32_STRUCT.unpack_from(data, offset)[0]
        offset += 4
        if offset + (8 * dimensions) + 4 + 8 > len(data):
            raise ValueError("Truncated GGUF tensor info")
        shape = tuple(
            _UINT64_STRUCT.unpack_from(data, offset + (8 * index))[0]
            for index in range(dimensions)
        )
        offset += 8 * dimensions
        tensor_type = _UINT32_STRUCT.unpack_from(data, offset)[0]
        offset += 4
        if tensor_type not in (0, 1):
            raise ValueError(f"Unsupported ggml type: {tensor_type}")
        tensor_offset = _UINT64_STRUCT.unpack_from(data, offset)[0]
        offset += 8
        tensor_infos.append((shape, tensor_offset, tensor_type))

    data_start = ((offset + 31) // 32) * 32
    if data_start > len(data):
        raise ValueError("Truncated GGUF data section")
    if any(byte != 0 for byte in data[offset:data_start]):
        raise ValueError("Invalid GGUF padding")

    for shape, tensor_offset, tensor_type in tensor_infos:
        tensor_size = 4 if tensor_type == 0 else 2
        for dimension in shape:
            tensor_size *= dimension
        if tensor_offset % 32 != 0:
            raise ValueError("Misaligned tensor offset")
        if data_start + tensor_offset + tensor_size > len(data):
            raise ValueError("Truncated GGUF data section")


def _read_string(data: bytes, offset: int) -> int:
    if offset + _STRING_LEN_STRUCT.size > len(data):
        raise ValueError("Truncated GGUF string")
    length = _STRING_LEN_STRUCT.unpack_from(data, offset)[0]
    offset += _STRING_LEN_STRUCT.size
    if offset + length > len(data):
        raise ValueError("Truncated GGUF string")
    return offset + length


def _skip_value(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        raise ValueError("Truncated GGUF metadata")
    value_type = _UINT32_STRUCT.unpack_from(data, offset)[0]
    offset += 4
    if value_type == 8:  # STRING
        return _read_string(data, offset)
    if value_type == 4:  # UINT32
        if offset + 4 > len(data):
            raise ValueError("Truncated GGUF metadata")
        return offset + 4
    if value_type == 5:  # INT32
        if offset + 4 > len(data):
            raise ValueError("Truncated GGUF metadata")
        return offset + 4
    if value_type == 10:  # UINT64
        if offset + 8 > len(data):
            raise ValueError("Truncated GGUF metadata")
        return offset + 8
    if value_type == 11:  # INT64
        if offset + 8 > len(data):
            raise ValueError("Truncated GGUF metadata")
        return offset + 8
    if value_type == 6:  # FLOAT32
        if offset + 4 > len(data):
            raise ValueError("Truncated GGUF metadata")
        return offset + 4
    if value_type == 7:  # BOOL
        if offset + 1 > len(data):
            raise ValueError("Truncated GGUF metadata")
        return offset + 1
    if value_type == 9:  # ARRAY
        if offset + 4 + 8 > len(data):
            raise ValueError("Truncated GGUF metadata")
        element_type = _UINT32_STRUCT.unpack_from(data, offset)[0]
        offset += 4
        count = _UINT64_STRUCT.unpack_from(data, offset)[0]
        offset += 8
        for _ in range(count):
            if element_type == 8:
                offset = _read_string(data, offset)
            elif element_type == 4:
                if offset + 4 > len(data):
                    raise ValueError("Truncated GGUF metadata")
                offset += 4
            elif element_type == 5:
                if offset + 4 > len(data):
                    raise ValueError("Truncated GGUF metadata")
                offset += 4
            elif element_type == 6:
                if offset + 4 > len(data):
                    raise ValueError("Truncated GGUF metadata")
                offset += 4
            elif element_type == 10:
                if offset + 8 > len(data):
                    raise ValueError("Truncated GGUF metadata")
                offset += 8
            elif element_type == 11:
                if offset + 8 > len(data):
                    raise ValueError("Truncated GGUF metadata")
                offset += 8
            else:
                raise ValueError(f"Unsupported metadata element type {element_type}")
        return offset
    raise ValueError(f"Unsupported metadata value type {value_type}")


def read_header(path: Path):
    data = path.read_bytes()
    if len(data) < _HEADER_STRUCT.size:
        raise ValueError("Truncated GGUF")
    magic, version, tensor_count, metadata_kv_count = _HEADER_STRUCT.unpack_from(data, 0)
    return type(
        "Header",
        (),
        {
            "magic": magic,
            "version": version,
            "tensor_count": tensor_count,
            "metadata_kv_count": metadata_kv_count,
        },
    )()


def validate_file(path: Path) -> None:
    validate_artifact(path)
