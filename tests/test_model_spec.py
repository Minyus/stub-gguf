from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest

from stub_gguf.model_spec import DEFAULT_OUTPUT
from stub_gguf.model_spec import TinyLlamaSpec


def test_default_output_points_to_dist_stub_gguf() -> None:
    assert DEFAULT_OUTPUT == Path("dist/stub.gguf")


def test_tiny_llama_spec_uses_converter_friendly_values() -> None:
    spec = TinyLlamaSpec()

    assert spec.architecture == "llama"
    assert spec.model_type == "llama"
    assert spec.hidden_size == 16
    assert spec.intermediate_size == 32
    assert spec.num_attention_heads == 4
    assert spec.num_key_value_heads == 4
    assert spec.num_hidden_layers == 2
    assert spec.vocab_size >= 16
    assert spec.max_position_embeddings == 100_000
    assert spec.rope_theta == 10000.0
    assert spec.rms_norm_eps == 1e-5
    assert spec.torch_dtype == "float32"
    assert spec.head_dim == 4


def test_tiny_llama_spec_allows_valid_grouped_query_attention() -> None:
    spec = TinyLlamaSpec(hidden_size=16, num_attention_heads=4, num_key_value_heads=2)

    assert spec.head_dim == 4
    assert spec.num_attention_heads % spec.num_key_value_heads == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_size": 15}, "hidden_size must be divisible by num_attention_heads"),
        ({"hidden_size": 0}, "hidden_size must be positive"),
        ({"num_attention_heads": 0}, "num_attention_heads must be positive"),
        ({"num_key_value_heads": 5}, "num_key_value_heads must not exceed num_attention_heads"),
        ({"hidden_size": 18, "num_attention_heads": 6, "num_key_value_heads": 4}, "num_attention_heads must be divisible by num_key_value_heads"),
    ],
)
def test_tiny_llama_spec_rejects_invalid_configurations(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TinyLlamaSpec(**kwargs)
