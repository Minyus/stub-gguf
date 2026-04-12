from __future__ import annotations

from pathlib import Path
import hashlib
import json

import sentencepiece as spm
import pytest
import torch
from transformers import LlamaForCausalLM, LlamaTokenizerFast

from stub_gguf.hf_stub_builder import build_hf_stub
from stub_gguf.model_spec import TinyLlamaSpec


def test_build_hf_stub_writes_a_minimal_hf_checkpoint(tmp_path: Path) -> None:
    spec = TinyLlamaSpec(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        vocab_size=64,
    )

    checkpoint_dir = build_hf_stub(tmp_path, spec)

    assert checkpoint_dir == tmp_path / "hf_stub"
    assert checkpoint_dir.is_dir()

    config_path = checkpoint_dir / "config.json"
    tokenizer_model_path = checkpoint_dir / "tokenizer.model"
    tokenizer_config_path = checkpoint_dir / "tokenizer_config.json"
    generation_config_path = checkpoint_dir / "generation_config.json"
    weights_path = checkpoint_dir / "pytorch_model.bin"

    for path in (
        config_path,
        tokenizer_model_path,
        tokenizer_config_path,
        generation_config_path,
        weights_path,
    ):
        assert path.exists()

    model = LlamaForCausalLM.from_pretrained(checkpoint_dir, local_files_only=True)
    assert model.config.model_type == "llama"
    assert model.config.hidden_size == spec.hidden_size
    assert model.config.vocab_size == spec.vocab_size

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["model_type"] == "llama"
    assert config["hidden_size"] == spec.hidden_size
    assert config["intermediate_size"] == spec.intermediate_size
    assert config["num_attention_heads"] == spec.num_attention_heads
    assert config["num_key_value_heads"] == spec.num_key_value_heads
    assert config["num_hidden_layers"] == spec.num_hidden_layers
    assert config["vocab_size"] == spec.vocab_size
    assert config["max_position_embeddings"] == spec.max_position_embeddings
    assert config["unk_token_id"] == 0

    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    generation_config = json.loads(generation_config_path.read_text(encoding="utf-8"))
    assert tokenizer_config["unk_token_id"] == 0
    assert tokenizer_config["bos_token_id"] == 1
    assert tokenizer_config["eos_token_id"] == 2
    assert tokenizer_config["pad_token_id"] == 2
    assert "tools is defined" in tokenizer_config["chat_template"]
    assert "tool" in tokenizer_config["chat_template"]
    assert generation_config["bos_token_id"] == 1
    assert generation_config["eos_token_id"] == 2
    assert generation_config["pad_token_id"] == 2
    assert generation_config["max_new_tokens"] == 4

    processor = spm.SentencePieceProcessor()
    assert processor.Load(str(tokenizer_model_path))
    assert processor.get_piece_size() == spec.vocab_size  # pyright: ignore[reportAttributeAccessIssue]

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    required_keys = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    for layer_idx in range(spec.num_hidden_layers):
        required_keys.update(
            {
                f"model.layers.{layer_idx}.self_attn.q_proj.weight",
                f"model.layers.{layer_idx}.self_attn.k_proj.weight",
                f"model.layers.{layer_idx}.self_attn.v_proj.weight",
                f"model.layers.{layer_idx}.self_attn.o_proj.weight",
                f"model.layers.{layer_idx}.mlp.gate_proj.weight",
                f"model.layers.{layer_idx}.mlp.up_proj.weight",
                f"model.layers.{layer_idx}.mlp.down_proj.weight",
                f"model.layers.{layer_idx}.input_layernorm.weight",
                f"model.layers.{layer_idx}.post_attention_layernorm.weight",
            }
        )
    assert required_keys.issubset(state_dict)
    assert state_dict["model.embed_tokens.weight"].shape == (spec.vocab_size, spec.hidden_size)
    assert state_dict["lm_head.weight"].shape == (spec.vocab_size, spec.hidden_size)
    for layer_idx in range(spec.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        assert state_dict[f"{prefix}.self_attn.q_proj.weight"].shape == (spec.hidden_size, spec.hidden_size)
        assert state_dict[f"{prefix}.self_attn.k_proj.weight"].shape == (spec.num_key_value_heads * spec.head_dim, spec.hidden_size)
        assert state_dict[f"{prefix}.self_attn.v_proj.weight"].shape == (spec.num_key_value_heads * spec.head_dim, spec.hidden_size)
        assert state_dict[f"{prefix}.self_attn.o_proj.weight"].shape == (spec.hidden_size, spec.hidden_size)
        assert state_dict[f"{prefix}.mlp.gate_proj.weight"].shape == (spec.intermediate_size, spec.hidden_size)
        assert state_dict[f"{prefix}.mlp.up_proj.weight"].shape == (spec.intermediate_size, spec.hidden_size)
        assert state_dict[f"{prefix}.mlp.down_proj.weight"].shape == (spec.hidden_size, spec.intermediate_size)
        assert state_dict[f"{prefix}.input_layernorm.weight"].shape == (spec.hidden_size,)
        assert state_dict[f"{prefix}.post_attention_layernorm.weight"].shape == (spec.hidden_size,)


def test_build_hf_stub_writes_a_canonical_tokenizer_artifact(tmp_path: Path) -> None:
    checkpoint_dir = build_hf_stub(tmp_path, TinyLlamaSpec(vocab_size=64))

    assert (checkpoint_dir / "tokenizer.json").exists()


def test_build_hf_stub_tokenizer_includes_a_newline_token(tmp_path: Path) -> None:
    checkpoint_dir = build_hf_stub(tmp_path, TinyLlamaSpec(vocab_size=64))

    processor = spm.SentencePieceProcessor()
    assert processor.Load(str(checkpoint_dir / "tokenizer.model"))
    pieces = {
        processor.id_to_piece(piece_id)  # pyright: ignore[reportAttributeAccessIssue]
        for piece_id in range(processor.get_piece_size())  # pyright: ignore[reportAttributeAccessIssue]
    }

    assert "\n" in pieces or "<0x0A>" in pieces


def test_build_hf_stub_tokenizer_can_encode_short_ascii_prompts_without_unk_ids(tmp_path: Path) -> None:
    checkpoint_dir = build_hf_stub(tmp_path, TinyLlamaSpec())

    processor = spm.SentencePieceProcessor()
    assert processor.Load(str(checkpoint_dir / "tokenizer.model"))

    for prompt in ("say ok", "Hello"):
        token_ids = processor.encode(prompt, out_type=int)  # pyright: ignore[reportAttributeAccessIssue]
        assert 0 not in token_ids


def test_build_hf_stub_advertises_large_context_and_fake_tool_support_in_tokenizer_config(tmp_path: Path) -> None:
    checkpoint_dir = build_hf_stub(tmp_path, TinyLlamaSpec())

    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads((checkpoint_dir / "tokenizer_config.json").read_text(encoding="utf-8"))

    assert config["max_position_embeddings"] == 100_000
    assert tokenizer_config["model_max_length"] == 100_000
    assert "tool" in tokenizer_config["chat_template"]
    assert "tools is defined" in tokenizer_config["chat_template"]
    assert "role" in tokenizer_config["chat_template"]


def test_build_hf_stub_renders_tool_use_chat_template(tmp_path: Path) -> None:
    checkpoint_dir = build_hf_stub(tmp_path, TinyLlamaSpec())
    tokenizer = LlamaTokenizerFast.from_pretrained(checkpoint_dir, local_files_only=True)

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}, {"role": "tool", "content": "ok"}],
        tools=[{"name": "lookup", "description": "find a value"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert "tools\n" in rendered
    assert "tool\nok\n" in rendered
    assert '"name": "lookup"' in rendered
    assert rendered.endswith("assistant\n")

    processor = spm.SentencePieceProcessor()
    assert processor.Load(str(checkpoint_dir / "tokenizer.model"))
    token_ids = processor.encode('{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}', out_type=int)  # pyright: ignore[reportAttributeAccessIssue]
    assert token_ids
    assert 0 not in token_ids


def test_build_hf_stub_generation_config_stays_short_and_ascii_biased(tmp_path: Path) -> None:
    checkpoint_dir = build_hf_stub(tmp_path, TinyLlamaSpec())

    generation_config = json.loads((checkpoint_dir / "generation_config.json").read_text(encoding="utf-8"))

    assert generation_config["do_sample"] is False
    assert generation_config["min_new_tokens"] == 1
    assert generation_config["max_new_tokens"] == 4
    assert generation_config["temperature"] == 0.0
    assert generation_config["top_k"] == 1
    assert generation_config["top_p"] == 1.0


def test_build_hf_stub_tokenizer_prefers_short_printable_ascii_reply_pieces(tmp_path: Path) -> None:
    checkpoint_dir = build_hf_stub(tmp_path, TinyLlamaSpec())

    processor = spm.SentencePieceProcessor()
    assert processor.Load(str(checkpoint_dir / "tokenizer.model"))

    assert 0 not in processor.encode("say ok", out_type=int)  # pyright: ignore[reportAttributeAccessIssue]
    assert 0 not in processor.encode("Hello", out_type=int)  # pyright: ignore[reportAttributeAccessIssue]
    for reply in ("OK", "ok", "yes", "done"):
        token_ids = processor.encode(reply, out_type=int)  # pyright: ignore[reportAttributeAccessIssue]
        assert 0 not in token_ids


def test_build_hf_stub_uses_transformers_llama_checkpoint_layout_and_is_deterministic(tmp_path: Path) -> None:
    spec = TinyLlamaSpec(vocab_size=64)

    first_dir = build_hf_stub(tmp_path / "first", spec)
    second_dir = build_hf_stub(tmp_path / "second", spec)

    first_model = LlamaForCausalLM.from_pretrained(first_dir, local_files_only=True)
    second_model = LlamaForCausalLM.from_pretrained(second_dir, local_files_only=True)

    assert type(first_model) is type(second_model)
    first_config = first_model.config.to_dict()
    second_config = second_model.config.to_dict()
    first_config.pop("_name_or_path", None)
    second_config.pop("_name_or_path", None)
    assert first_config == second_config
    assert first_model.state_dict().keys() == second_model.state_dict().keys()
    for key, first_tensor in first_model.state_dict().items():
        second_tensor = second_model.state_dict()[key]
        assert torch.equal(first_tensor, second_tensor)


def test_build_hf_stub_does_not_mix_auto_bos_eos_flags_with_literal_template_markers(tmp_path: Path) -> None:
    checkpoint_dir = build_hf_stub(tmp_path, TinyLlamaSpec(vocab_size=64))

    tokenizer_config = json.loads((checkpoint_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    chat_template = tokenizer_config["chat_template"]

    assert tokenizer_config["add_bos_token"] is True
    assert tokenizer_config["add_eos_token"] is True
    assert "<s>" not in chat_template
    assert "</s>" not in chat_template


def test_build_hf_stub_writes_generation_config_for_short_non_empty_responses(tmp_path: Path) -> None:
    checkpoint_dir = build_hf_stub(tmp_path, TinyLlamaSpec(vocab_size=64))

    generation_config = json.loads((checkpoint_dir / "generation_config.json").read_text(encoding="utf-8"))

    assert generation_config["bos_token_id"] == 1
    assert generation_config["eos_token_id"] == 2
    assert generation_config["pad_token_id"] == 2
    assert generation_config["do_sample"] is False
    assert generation_config["max_new_tokens"] == 4
    assert generation_config["min_new_tokens"] == 1
    assert generation_config["repetition_penalty"] == 1.0


def test_build_hf_stub_forces_llama_model_type_even_if_spec_differs(tmp_path: Path) -> None:
    spec = TinyLlamaSpec(model_type="not-llama")

    checkpoint_dir = build_hf_stub(tmp_path, spec)

    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "llama"


def test_build_hf_stub_honors_supported_torch_dtype(tmp_path: Path) -> None:
    spec = TinyLlamaSpec(torch_dtype="float16")

    checkpoint_dir = build_hf_stub(tmp_path, spec)

    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    assert config["torch_dtype"] == "float16"

    state_dict = torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    assert state_dict["model.embed_tokens.weight"].dtype == torch.float16
    assert state_dict["model.norm.weight"].dtype == torch.float16


def test_build_hf_stub_rejects_unsupported_torch_dtype_before_writing_output(tmp_path: Path) -> None:
    spec = TinyLlamaSpec(torch_dtype="int8")

    with pytest.raises(ValueError, match="Unsupported torch_dtype 'int8'"):
        build_hf_stub(tmp_path, spec)

    assert not (tmp_path / "hf_stub").exists()


def test_build_hf_stub_rejects_vocab_sizes_below_sentencepiece_floor_at_thirty_two(tmp_path: Path) -> None:
    spec = TinyLlamaSpec(vocab_size=32)

    with pytest.raises(ValueError, match="vocab_size must be at least 64 to build a stable SentencePiece tokenizer"):
        build_hf_stub(tmp_path, spec)


def test_build_hf_stub_supports_larger_vocab_tokenizer_generation(tmp_path: Path) -> None:
    spec = TinyLlamaSpec(vocab_size=64)

    checkpoint_dir = build_hf_stub(tmp_path, spec)

    processor = spm.SentencePieceProcessor()
    assert processor.Load(str(checkpoint_dir / "tokenizer.model"))
    assert processor.get_piece_size() == spec.vocab_size  # pyright: ignore[reportAttributeAccessIssue]
    assert processor.unk_id() == 0
    assert processor.bos_id() == 1
    assert processor.eos_id() == 2


def test_build_hf_stub_rejects_vocab_sizes_below_sentencepiece_floor(tmp_path: Path) -> None:
    spec = TinyLlamaSpec(vocab_size=63)

    with pytest.raises(ValueError, match="vocab_size must be at least 64 to build a stable SentencePiece tokenizer"):
        build_hf_stub(tmp_path, spec)


def test_build_hf_stub_replaces_existing_hf_stub_directory(tmp_path: Path) -> None:
    stale_dir = tmp_path / "hf_stub"
    stale_dir.mkdir()
    stale_file = stale_dir / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")

    spec = TinyLlamaSpec(vocab_size=64)

    checkpoint_dir = build_hf_stub(tmp_path, spec)

    assert checkpoint_dir == stale_dir
    assert not stale_file.exists()
    assert (checkpoint_dir / "config.json").exists()
    assert (checkpoint_dir / "pytorch_model.bin").exists()


def test_build_hf_stub_is_deterministic_for_same_spec(tmp_path: Path) -> None:
    spec = TinyLlamaSpec(vocab_size=64)

    first_dir = build_hf_stub(tmp_path, spec)
    first_tokenizer = (first_dir / "tokenizer.model").read_bytes()
    first_weights = (first_dir / "pytorch_model.bin").read_bytes()

    second_dir = build_hf_stub(tmp_path, spec)

    assert second_dir == first_dir
    assert (second_dir / "tokenizer.model").read_bytes() == first_tokenizer
    assert (second_dir / "pytorch_model.bin").read_bytes() == first_weights


def test_build_hf_stub_is_deterministic_across_different_base_directories(tmp_path: Path) -> None:
    spec = TinyLlamaSpec(vocab_size=64)

    first_base_dir = tmp_path / "first"
    second_base_dir = tmp_path / "second"
    first_base_dir.mkdir()
    second_base_dir.mkdir()

    first_dir = build_hf_stub(first_base_dir, spec)
    second_dir = build_hf_stub(second_base_dir, spec)

    first_tokenizer_hash = hashlib.sha256((first_dir / "tokenizer.model").read_bytes()).hexdigest()
    second_tokenizer_hash = hashlib.sha256((second_dir / "tokenizer.model").read_bytes()).hexdigest()
    first_weights_hash = hashlib.sha256((first_dir / "pytorch_model.bin").read_bytes()).hexdigest()
    second_weights_hash = hashlib.sha256((second_dir / "pytorch_model.bin").read_bytes()).hexdigest()

    assert first_tokenizer_hash == second_tokenizer_hash
    assert first_weights_hash == second_weights_hash


def test_build_hf_stub_does_not_change_process_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = TinyLlamaSpec(vocab_size=64)
    original_cwd = Path.cwd()
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    build_hf_stub(tmp_path, spec)

    assert Path.cwd() == other_dir
    assert other_dir != original_cwd
