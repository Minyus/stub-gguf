from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from io import BytesIO
import struct


class MetadataValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(IntEnum):
    F32 = 0


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    data: bytes


class GGUFWriter:
    def __init__(
        self,
        architecture: str,
        metadata: dict[str, tuple[MetadataValueType, object]],
        tensors: list[TensorSpec],
        alignment: int = 32,
    ) -> None:
        if alignment <= 0:
            raise ValueError("alignment must be positive")
        if alignment != 32:
            raise ValueError("Only 32-byte alignment is supported")
        self.architecture = architecture
        self.metadata = metadata
        self.tensors = tensors
        self.alignment = alignment
        self._validate_tensors()

    def to_bytes(self) -> bytes:
        metadata = {
            "general.architecture": (MetadataValueType.STRING, self.architecture),
            **self.metadata,
        }
        table = BytesIO()
        table.write(b"GGUF")
        table.write(struct.pack("<I", 3))
        table.write(struct.pack("<Q", len(self.tensors)))
        table.write(struct.pack("<Q", len(metadata)))

        for key, value in metadata.items():
            self._write_string(table, key)
            self._write_value(table, value)

        offset_cursor = 0
        tensor_infos: list[tuple[TensorSpec, int]] = []
        for tensor in self.tensors:
            offset_cursor = self._align(offset_cursor)
            tensor_infos.append((tensor, offset_cursor))
            offset_cursor += len(tensor.data)

        for tensor, offset in tensor_infos:
            self._write_string(table, tensor.name)
            table.write(struct.pack("<I", len(tensor.shape)))
            for dimension in reversed(tensor.shape):
                table.write(struct.pack("<Q", dimension))
            table.write(struct.pack("<I", tensor.ggml_type))
            table.write(struct.pack("<Q", offset))

        data_section = BytesIO()
        for tensor, offset in tensor_infos:
            while data_section.tell() < offset:
                data_section.write(b"\x00")
            data_section.write(tensor.data)

        while table.tell() % self.alignment != 0:
            table.write(b"\x00")

        return table.getvalue() + data_section.getvalue()

    def _align(self, offset: int) -> int:
        remainder = offset % self.alignment
        return offset if remainder == 0 else offset + (self.alignment - remainder)

    def _validate_tensors(self) -> None:
        for tensor in self.tensors:
            expected_size = self._expected_tensor_size(tensor)
            if len(tensor.data) != expected_size:
                raise ValueError(
                    f"Tensor {tensor.name} data size mismatch: expected {expected_size}, got {len(tensor.data)}"
                )

    def _expected_tensor_size(self, tensor: TensorSpec) -> int:
        element_count = 1
        for dimension in tensor.shape:
            element_count *= dimension
        if tensor.ggml_type == GGMLType.F32:
            return element_count * 4
        raise ValueError(f"Unsupported ggml type: {tensor.ggml_type}")

    def _write_string(self, buffer: BytesIO, value: str) -> None:
        encoded = value.encode("utf-8")
        buffer.write(struct.pack("<Q", len(encoded)))
        buffer.write(encoded)

    def _write_value(
        self,
        buffer: BytesIO,
        entry: tuple[MetadataValueType, object],
    ) -> None:
        value_type, value = entry
        buffer.write(struct.pack("<I", int(value_type)))
        if value_type is MetadataValueType.STRING:
            self._write_string(buffer, str(value))
            return
        if value_type is MetadataValueType.UINT32:
            buffer.write(struct.pack("<I", int(value)))
            return
        if value_type is MetadataValueType.UINT64:
            buffer.write(struct.pack("<Q", int(value)))
            return
        if value_type is MetadataValueType.FLOAT32:
            buffer.write(struct.pack("<f", float(value)))
            return
        if value_type is MetadataValueType.BOOL:
            buffer.write(struct.pack("<?", bool(value)))
            return
        if value_type is MetadataValueType.ARRAY:
            array_type, array_values = value
            buffer.write(struct.pack("<I", int(array_type)))
            buffer.write(struct.pack("<Q", len(array_values)))
            for item in array_values:
                self._write_scalar(buffer, array_type, item)
            return
        raise ValueError(f"Unsupported metadata type: {value_type}")

    def _write_scalar(
        self,
        buffer: BytesIO,
        value_type: MetadataValueType,
        value: object,
    ) -> None:
        if value_type is MetadataValueType.STRING:
            self._write_string(buffer, str(value))
            return
        if value_type is MetadataValueType.UINT32:
            buffer.write(struct.pack("<I", int(value)))
            return
        if value_type is MetadataValueType.FLOAT32:
            buffer.write(struct.pack("<f", float(value)))
            return
        raise ValueError(f"Unsupported scalar array type: {value_type}")
