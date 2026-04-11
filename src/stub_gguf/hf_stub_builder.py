from __future__ import annotations

from pathlib import Path
import json
import subprocess
import shutil
import sys

import torch

from stub_gguf.model_spec import TinyLlamaSpec


MIN_VOCAB_SIZE = 30


_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_torch_dtype(torch_dtype: str) -> torch.dtype:
    try:
        return _TORCH_DTYPES[torch_dtype]
    except KeyError as exc:
        supported = ", ".join(sorted(_TORCH_DTYPES))
        raise ValueError(f"Unsupported torch_dtype {torch_dtype!r}; supported values: {supported}") from exc


def build_hf_stub(base_dir: Path, spec: TinyLlamaSpec) -> Path:
    _validate_spec(spec)
    _resolve_torch_dtype(spec.torch_dtype)
    output_dir = base_dir / "hf_stub"
    staging_dir = base_dir / ".hf_stub.tmp"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=False)

    try:
        _write_config(staging_dir, spec)
        _write_tokenizer(staging_dir, spec)
        _write_generation_config(staging_dir)
        _write_weights(staging_dir, spec)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging_dir.replace(output_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    return output_dir


def _validate_spec(spec: TinyLlamaSpec) -> None:
    if spec.vocab_size < MIN_VOCAB_SIZE:
        raise ValueError(f"vocab_size must be at least {MIN_VOCAB_SIZE} to build a stable SentencePiece tokenizer")


def _write_config(output_dir: Path, spec: TinyLlamaSpec) -> None:
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "max_position_embeddings": spec.max_position_embeddings,
        "model_type": "llama",
        "num_attention_heads": spec.num_attention_heads,
        "num_hidden_layers": spec.num_hidden_layers,
        "num_key_value_heads": spec.num_key_value_heads,
        "rope_theta": spec.rope_theta,
        "rms_norm_eps": spec.rms_norm_eps,
        "torch_dtype": spec.torch_dtype,
        "vocab_size": spec.vocab_size,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def _write_tokenizer(output_dir: Path, spec: TinyLlamaSpec) -> None:
    sentencepiece_vocab_size = spec.vocab_size
    corpus_tokens = [f"token_{idx:04d}" for idx in range(max(sentencepiece_vocab_size * 2, 64))]
    corpus_lines = [" ".join(corpus_tokens[idx : idx + 16]) for idx in range(0, len(corpus_tokens), 16)]
    corpus_lines.extend(" ".join(reversed(corpus_tokens[idx : idx + 16])) for idx in range(0, len(corpus_tokens), 16))
    corpus = "\n".join(corpus_lines)
    corpus_path = output_dir / "tokenizer_corpus.txt"
    prefix = output_dir / "_tokenizer"
    corpus_path.write_text(corpus, encoding="utf-8")
    try:
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sentencepiece as spm; "
                    "spm.SentencePieceTrainer.train(" 
                    "input='tokenizer_corpus.txt', "
                    "model_prefix='_tokenizer', "
                    "model_type='bpe', "
                    f"vocab_size={sentencepiece_vocab_size}, "
                    "character_coverage=1.0, "
                    "shuffle_input_sentence=False, "
                    "num_threads=1, "
                    "max_sentence_length=10000, "
                    "bos_id=1, eos_id=2, pad_id=-1, unk_id=0, "
                    "hard_vocab_limit=True, train_extremely_large_corpus=False)"
                ),
            ],
            cwd=output_dir,
            check=True,
        )
    finally:
        corpus_path.unlink(missing_ok=True)
    (output_dir / "tokenizer.model").write_bytes((prefix.with_suffix(".model")).read_bytes())
    prefix.with_suffix(".model").unlink()
    prefix.with_suffix(".vocab").unlink()
    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": "<s>",
        "bos_token_id": 1,
        "eos_token": "</s>",
        "eos_token_id": 2,
        "model_max_length": spec.max_position_embeddings,
        "pad_token": None,
        "pad_token_id": None,
        "tokenizer_class": "LlamaTokenizerFast",
        "unk_token": "<unk>",
        "unk_token_id": 0,
    }
    (output_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2, sort_keys=True), encoding="utf-8")

def _write_generation_config(output_dir: Path) -> None:
    generation_config = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": None,
        "transformers_version": "4.0.0",
    }
    (output_dir / "generation_config.json").write_text(json.dumps(generation_config, indent=2, sort_keys=True), encoding="utf-8")


def _write_weights(output_dir: Path, spec: TinyLlamaSpec) -> None:
    torch_dtype = _resolve_torch_dtype(spec.torch_dtype)
    state_dict: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.zeros((spec.vocab_size, spec.hidden_size), dtype=torch_dtype),
        "model.norm.weight": torch.ones((spec.hidden_size,), dtype=torch_dtype),
        "lm_head.weight": torch.zeros((spec.vocab_size, spec.hidden_size), dtype=torch_dtype),
    }

    kv_hidden = spec.num_key_value_heads * spec.head_dim
    for layer_idx in range(spec.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        state_dict[f"{prefix}.self_attn.q_proj.weight"] = torch.zeros((spec.hidden_size, spec.hidden_size), dtype=torch_dtype)
        state_dict[f"{prefix}.self_attn.k_proj.weight"] = torch.zeros((kv_hidden, spec.hidden_size), dtype=torch_dtype)
        state_dict[f"{prefix}.self_attn.v_proj.weight"] = torch.zeros((kv_hidden, spec.hidden_size), dtype=torch_dtype)
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = torch.zeros((spec.hidden_size, spec.hidden_size), dtype=torch_dtype)
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = torch.zeros((spec.intermediate_size, spec.hidden_size), dtype=torch_dtype)
        state_dict[f"{prefix}.mlp.up_proj.weight"] = torch.zeros((spec.intermediate_size, spec.hidden_size), dtype=torch_dtype)
        state_dict[f"{prefix}.mlp.down_proj.weight"] = torch.zeros((spec.hidden_size, spec.intermediate_size), dtype=torch_dtype)
        state_dict[f"{prefix}.input_layernorm.weight"] = torch.ones((spec.hidden_size,), dtype=torch_dtype)
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = torch.ones((spec.hidden_size,), dtype=torch_dtype)

    torch.save(state_dict, output_dir / "pytorch_model.bin")
