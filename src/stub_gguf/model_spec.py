from __future__ import annotations

from dataclasses import dataclass
import struct

from stub_gguf.gguf_writer import GGMLType, MetadataValueType, TensorSpec


@dataclass(frozen=True)
class ModelSpec:
    metadata: dict[str, tuple[MetadataValueType, object]]
    tensors: list[TensorSpec]


def _f32s(*values: float) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def build_model_spec() -> ModelSpec:
    metadata = {
        "general.name": (MetadataValueType.STRING, "stub-noted-smoke-test"),
        "general.description": (
            MetadataValueType.STRING,
            "Synthetic GGUF for local smoke tests; intended fixed response: Noted",
        ),
        "stub.fixed_response": (MetadataValueType.STRING, "Noted"),
        "stub.tool_use": (MetadataValueType.BOOL, True),
        "stub.context_length": (MetadataValueType.UINT64, 1_000_000),
        "tokenizer.ggml.model": (MetadataValueType.STRING, "stub-wordpiece"),
        "tokenizer.ggml.tokens": (
            MetadataValueType.ARRAY,
            (MetadataValueType.STRING, ["<bos>", "<eos>", "Noted"]),
        ),
        "tokenizer.ggml.scores": (
            MetadataValueType.ARRAY,
            (MetadataValueType.FLOAT32, [0.0, 0.0, 100.0]),
        ),
        "tokenizer.ggml.token_type": (
            MetadataValueType.ARRAY,
            (MetadataValueType.UINT32, [3, 3, 1]),
        ),
        "llama.context_length": (MetadataValueType.UINT64, 1_000_000),
        "llama.embedding_length": (MetadataValueType.UINT32, 4),
        "llama.block_count": (MetadataValueType.UINT32, 1),
        "llama.feed_forward_length": (MetadataValueType.UINT32, 4),
        "llama.attention.head_count": (MetadataValueType.UINT32, 1),
        "llama.attention.head_count_kv": (MetadataValueType.UINT32, 1),
        "llama.rope.dimension_count": (MetadataValueType.UINT32, 4),
    }
    tensors = [
        TensorSpec("token_embd.weight", (3, 4), GGMLType.F32, _f32s(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0)),
        TensorSpec("blk.0.attn_norm.weight", (4,), GGMLType.F32, _f32s(1.0, 1.0, 1.0, 1.0)),
        TensorSpec("blk.0.ffn_norm.weight", (4,), GGMLType.F32, _f32s(1.0, 1.0, 1.0, 1.0)),
        TensorSpec("output.weight", (4, 3), GGMLType.F32, _f32s(-5.0, -5.0, 12.0, -5.0, -5.0, 12.0, -5.0, -5.0, 12.0, -5.0, -5.0, 12.0)),
    ]
    return ModelSpec(metadata=metadata, tensors=tensors)
