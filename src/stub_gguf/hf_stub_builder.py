from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import shutil
import sys
import subprocess

import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast

from stub_gguf.model_spec import TinyLlamaSpec


MIN_VOCAB_SIZE = 64


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
    config = _build_model_config(spec, _resolve_torch_dtype(spec.torch_dtype))
    config_payload = {
        "architectures": config.architectures,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "head_dim": config.head_dim,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "model_type": config.model_type,
        "num_attention_heads": config.num_attention_heads,
        "num_hidden_layers": config.num_hidden_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "pad_token_id": config.pad_token_id,
        "rope_theta": config.rope_theta,
        "rms_norm_eps": config.rms_norm_eps,
        "torch_dtype": spec.torch_dtype,
        "unk_token_id": 0,
        "vocab_size": config.vocab_size,
    }
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_tokenizer(output_dir: Path, spec: TinyLlamaSpec) -> None:
    sentencepiece_vocab_size = spec.vocab_size
    corpus_tokens = [f"token_{idx:04d}" for idx in range(max(sentencepiece_vocab_size * 2, 64))]
    corpus_lines = [" ".join(corpus_tokens[idx : idx + 16]) for idx in range(0, len(corpus_tokens), 16)]
    corpus_lines.extend(" ".join(reversed(corpus_tokens[idx : idx + 16])) for idx in range(0, len(corpus_tokens), 16))
    corpus_lines.extend(
        [
            "say ok say ok say ok",
            "Hello Hello Hello",
            "OK OK OK",
            "ok ok ok",
            "yes yes yes",
            "done done done",
            "ASCII ASCII ASCII",
            "tool call tool result tool call",
            "user assistant system tool user assistant",
            "short harmless reply short harmless reply",
            '{"name":"lookup","description":"find value"}',
            '{"role":"tool","content":"ok"}',
            '{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}',
        ]
    )
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
                    "user_defined_symbols=['<0x0A>'], "
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
    tokenizer = LlamaTokenizerFast(
        vocab_file=str(output_dir / "tokenizer.model"),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
        add_bos_token=True,
        add_eos_token=True,
        model_max_length=spec.max_position_embeddings,
        clean_up_tokenization_spaces=False,
        legacy=False,
    )
    tokenizer.chat_template = _chat_template()
    tokenizer.save_pretrained(output_dir)
    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": True,
        "bos_token": "<s>",
        "bos_token_id": 1,
        "chat_template": _chat_template(),
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "eos_token_id": 2,
        "legacy": False,
        "model_max_length": spec.max_position_embeddings,
        "pad_token": "</s>",
        "pad_token_id": 2,
        "tokenizer_class": "LlamaTokenizerFast",
        "unk_token": "<unk>",
        "unk_token_id": 0,
    }
    (output_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "special_tokens_map.json").write_text(json.dumps(_special_tokens_map(), indent=2, sort_keys=True), encoding="utf-8")

def _write_generation_config(output_dir: Path) -> None:
    generation_config = {
        "bos_token_id": 1,
        "do_sample": False,
        "eos_token_id": 2,
        "max_new_tokens": 4,
        "min_new_tokens": 1,
        "pad_token_id": 2,
        "repetition_penalty": 1.0,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "transformers_version": "4.0.0",
    }
    (output_dir / "generation_config.json").write_text(json.dumps(generation_config, indent=2, sort_keys=True), encoding="utf-8")


def _chat_template() -> str:
    return (
        "{% if tools is defined and tools %}"
        "tools\n"
        "{% for tool in tools %}"
        "{{ tool | tojson }}\n"
        "{% endfor %}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "system\n{{ message['content'] }}\n"
        "{% elif message['role'] == 'user' %}"
        "user\n{{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}"
        "{% if message.get('tool_calls') %}"
        "assistant\n{{ {'role': 'assistant', 'tool_calls': message['tool_calls']} | tojson }}\n"
        "{% else %}"
        "assistant\n{{ message['content'] }}\n"
        "{% endif %}"
        "{% elif message['role'] == 'tool' %}"
        "tool\n{{ message['content'] }}\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant\n{% endif %}"
    )


def _special_tokens_map() -> dict[str, str]:
    return {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "</s>",
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
    config.pad_token_id = 2
    config.torch_dtype = torch_dtype
    return config


def _write_weights(output_dir: Path, spec: TinyLlamaSpec) -> None:
    torch_dtype = _resolve_torch_dtype(spec.torch_dtype)
    config = _build_model_config(spec, torch_dtype)
    with _manual_seed(0):
        model = LlamaForCausalLM(config)
    model.to(dtype=torch_dtype)  # pyright: ignore[reportCallIssue]
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
