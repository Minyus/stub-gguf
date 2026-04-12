from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
DEFAULT_OUTPUT = Path("dist/stub.gguf")


@dataclass(frozen=True)
class TinyLlamaSpec:
    architecture: str = "llama"
    model_type: str = "llama"
    hidden_size: int = 16
    intermediate_size: int = 32
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    num_hidden_layers: int = 2
    vocab_size: int = 64
    max_position_embeddings: int = 128
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    torch_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("num_key_value_heads must not exceed num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
