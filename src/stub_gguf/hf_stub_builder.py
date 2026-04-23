from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import shutil
import sys
import subprocess

import torch
import sentencepiece as spm
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

from stub_gguf.model_spec import TinyLlamaSpec


MIN_VOCAB_SIZE = 64
_LLAMA_31_CORE_TOKENS = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|finetune_right_pad_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eom_id|>",
    "<|eot_id|>",
    "<|python_tag|>",
)
_LLAMA_FAMILY_EXTRA_TOKENS = ("[INST]", "[/INST]", "<<SYS>>", "<</SYS>>")
_NEWLINE_TOKEN = "<0x0A>"


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
        raise ValueError(
            f"Unsupported torch_dtype {torch_dtype!r}; supported values: {supported}"
        ) from exc


def _require_hf_tokenizer_dependencies() -> None:
    try:
        import google.protobuf  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "protobuf is required to build the Hugging Face tokenizer artifacts"
        ) from exc


def _compatibility_tokens() -> list[str]:
    return [
        _NEWLINE_TOKEN,
        _LLAMA_31_CORE_TOKENS[1],
        *_LLAMA_31_CORE_TOKENS[3:],
        *_LLAMA_FAMILY_EXTRA_TOKENS,
    ]


def build_hf_stub(base_dir: Path, spec: TinyLlamaSpec) -> Path:
    _validate_spec(spec)
    _resolve_torch_dtype(spec.torch_dtype)
    _require_hf_tokenizer_dependencies()
    output_dir = base_dir / "hf_stub"
    staging_dir = base_dir / ".hf_stub.tmp"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=False)

    try:
        tokenizer_special_ids = _write_tokenizer(staging_dir, spec)
        _write_config(staging_dir, spec, tokenizer_special_ids)
        _write_generation_config(staging_dir, tokenizer_special_ids)
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
        raise ValueError(
            f"vocab_size must be at least {MIN_VOCAB_SIZE} to build a stable SentencePiece tokenizer"
        )


def _write_config(
    output_dir: Path, spec: TinyLlamaSpec, token_ids: dict[str, int]
) -> None:
    config = _build_model_config(spec, _resolve_torch_dtype(spec.torch_dtype))
    config_payload = {
        "architectures": config.architectures,
        "bos_token": "<|begin_of_text|>",
        "bos_token_id": token_ids["<|begin_of_text|>"],
        "eos_token": "<|eot_id|>",
        "eos_token_id": token_ids["<|eot_id|>"],
        "head_dim": config.head_dim,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "model_type": config.model_type,
        "num_attention_heads": config.num_attention_heads,
        "num_hidden_layers": config.num_hidden_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "pad_token": "<|finetune_right_pad_id|>",
        "pad_token_id": token_ids["<|finetune_right_pad_id|>"],
        "rope_theta": config.rope_theta,
        "rms_norm_eps": config.rms_norm_eps,
        "torch_dtype": spec.torch_dtype,
        "unk_token_id": 0,
        "vocab_size": config.vocab_size,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_tokenizer(output_dir: Path, spec: TinyLlamaSpec) -> dict[str, int]:
    sentencepiece_vocab_size = spec.vocab_size
    corpus_tokens = [
        f"token_{idx:04d}" for idx in range(max(sentencepiece_vocab_size * 2, 64))
    ]
    corpus_lines = [
        " ".join(corpus_tokens[idx : idx + 16])
        for idx in range(0, len(corpus_tokens), 16)
    ]
    corpus_lines.extend(
        " ".join(reversed(corpus_tokens[idx : idx + 16]))
        for idx in range(0, len(corpus_tokens), 16)
    )
    corpus_lines.extend(
        [
            "say ok say ok say ok",
            "Hello Hello Hello",
            "short harmless reply short harmless reply",
            "user assistant system user assistant",
            '{"tool":"lookup","arguments":{"query":"hello"}}',
            '{"name":"lookup","description":"tiny tool schema","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}',
            '{"role":"tool","content":"{"ok":true,"value":1}"}',
        ]
    )
    corpus = "\n".join(corpus_lines)
    corpus_path = output_dir / "tokenizer_corpus.txt"
    prefix = output_dir / "_tokenizer"
    corpus_path.write_text(corpus, encoding="utf-8")
    try:
        compatibility_tokens = json.dumps(_compatibility_tokens())
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sentencepiece as spm; "
                    "spm.SentencePieceTrainer.train("
                    "input='tokenizer_corpus.txt', "
                    "model_prefix='_tokenizer', "
                    "model_type='char', "
                    f"vocab_size={sentencepiece_vocab_size}, "
                    "character_coverage=1.0, "
                    "shuffle_input_sentence=False, "
                    "num_threads=1, "
                    "max_sentence_length=10000, "
                    f"user_defined_symbols={compatibility_tokens}, "
                    "bos_piece='<|begin_of_text|>', eos_piece='<|eot_id|>', pad_piece='<|finetune_right_pad_id|>', "
                    "bos_id=1, eos_id=2, pad_id=3, unk_id=0, "
                    "hard_vocab_limit=True, train_extremely_large_corpus=False)"
                ),
            ],
            cwd=output_dir,
            check=True,
        )
    finally:
        corpus_path.unlink(missing_ok=True)
    (output_dir / "tokenizer.model").write_bytes(
        (prefix.with_suffix(".model")).read_bytes()
    )
    prefix.with_suffix(".model").unlink()
    prefix.with_suffix(".vocab").unlink()
    tokenizer = LlamaTokenizer(
        vocab_file=str(output_dir / "tokenizer.model"),
        unk_token="<unk>",
        bos_token="<|begin_of_text|>",
        eos_token="<|eot_id|>",
        pad_token="<|finetune_right_pad_id|>",
        add_bos_token=True,
        add_eos_token=True,
        legacy=False,
    )
    tokenizer.chat_template = _chat_template()
    tokenizer.save_pretrained(output_dir)
    (output_dir / "tokenizer.json").unlink(missing_ok=True)
    tokenizer = LlamaTokenizer.from_pretrained(output_dir, local_files_only=True)
    added_tokens_decoder = _added_token_decoder(tokenizer)
    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": True,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": _additional_special_tokens(),
        "bos_token": "<|begin_of_text|>",
        "bos_token_id": tokenizer.bos_token_id,
        "chat_template": _chat_template(),
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|eot_id|>",
        "eos_token_id": tokenizer.eos_token_id,
        "legacy": False,
        "model_max_length": spec.max_position_embeddings,
        "pad_token": "<|finetune_right_pad_id|>",
        "pad_token_id": tokenizer.pad_token_id,
        "tokenizer_class": "LlamaTokenizer",
        "unk_token": "<unk>",
        "unk_token_id": 0,
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(_special_tokens_map(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "<|begin_of_text|>": tokenizer.bos_token_id,
        "<|eot_id|>": tokenizer.eos_token_id,
        "<|finetune_right_pad_id|>": tokenizer.pad_token_id,
    }


def _write_generation_config(output_dir: Path, token_ids: dict[str, int]) -> None:
    generation_config = {
        "bos_token_id": token_ids["<|begin_of_text|>"],
        "do_sample": False,
        "eos_token_id": token_ids["<|eot_id|>"],
        "max_new_tokens": 8,
        "min_new_tokens": 1,
        "pad_token_id": token_ids["<|finetune_right_pad_id|>"],
        "repetition_penalty": 1.0,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "transformers_version": "4.0.0",
    }
    (output_dir / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2, sort_keys=True), encoding="utf-8"
    )


def _chat_template() -> str:
    return (
        "{% set bos = '<|begin_of_text|>' %}"
        "{% set start = '<|start_header_id|>' %}"
        "{% set end = '<|end_header_id|>' %}"
        "{% set eot = '<|eot_id|>' %}"
        "{{ bos }}"
        "{% for message in messages %}"
        "{{ start }}{{ message['role'] }}{{ end }}\n\n"
        "{{ message['content'] }}{{ eot }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ start }}assistant{{ end }}\n\n{% endif %}"
    )


def _added_token_decoder(tokenizer: LlamaTokenizer) -> dict[int, dict[str, object]]:
    return {
        tokenizer.convert_tokens_to_ids(token): {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
        for token in _compatibility_tokens()
        if tokenizer.convert_tokens_to_ids(token) is not None
    }


def _additional_special_tokens() -> list[str]:
    return [
        *_LLAMA_31_CORE_TOKENS,
        *_LLAMA_FAMILY_EXTRA_TOKENS,
    ]


def _special_tokens_map() -> dict[str, object]:
    return {
        "additional_special_tokens": _additional_special_tokens(),
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|eot_id|>",
        "pad_token": "<|finetune_right_pad_id|>",
        "unk_token": "<unk>",
    }


@contextmanager
def _manual_seed(seed: int):
    previous_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(previous_state)


def _build_model_config(spec: TinyLlamaSpec, torch_dtype: torch.dtype) -> LlamaConfig:
    config = LlamaConfig(architectures=["LlamaForCausalLM"])
    config.hidden_size = spec.hidden_size
    config.intermediate_size = spec.intermediate_size
    config.max_position_embeddings = spec.max_position_embeddings
    config.num_attention_heads = spec.num_attention_heads
    config.num_hidden_layers = spec.num_hidden_layers
    config.num_key_value_heads = spec.num_key_value_heads
    config.head_dim = spec.head_dim
    config.rope_theta = spec.rope_theta
    config.rms_norm_eps = spec.rms_norm_eps
    config.vocab_size = spec.vocab_size
    config.bos_token_id = 1
    config.eos_token_id = 2
    config.pad_token_id = 3
    config.torch_dtype = torch_dtype
    return config


def _write_weights(output_dir: Path, spec: TinyLlamaSpec) -> None:
    torch_dtype = _resolve_torch_dtype(spec.torch_dtype)
    config = _build_model_config(spec, torch_dtype)
    with _manual_seed(0):
        model = LlamaForCausalLM(config)
    model.to(dtype=torch_dtype)  # pyright: ignore[reportCallIssue]
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
