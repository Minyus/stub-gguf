# Runtime Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `dist/stub.gguf` behave more like a minimal chat-capable model so Ollama emits a short trivial reply and LM Studio has a better chance of importing it cleanly and recognizing chat/tool-related metadata.

**Architecture:** Keep the converter-backed GGUF generation path, but enrich the Hugging Face stub inputs so llama.cpp converts standard chat/tokenizer metadata into the artifact. Fix runtime behavior at the HF stub layer first, then keep the Ollama wrapper minimal and aligned with the new chat-oriented artifact.

**Tech Stack:** Python 3.12, uv, pytest, sentencepiece, torch, llama.cpp `convert_hf_to_gguf.py`, GGUF

---

## File Map

- Modify: `src/stub_gguf/hf_stub_builder.py` — add explicit chat-template, tokenizer special-token fields, and safer generation-config defaults for short non-empty responses.
- Modify: `tests/test_hf_stub_builder.py` — add failing tests for new HF-side files/fields and generation config.
- Modify: `tests/test_generate.py` — verify the real conversion path receives the richer HF stub inputs.
- Modify: `ollama.modelfile` — keep parameters low-randomness and align the wrapper with the new short-response goal.
- Optional Modify: `README.md` — only if runtime usage docs need to mention the adjusted Ollama expectation after implementation.

### Task 1: Add failing HF stub metadata tests

**Files:**
- Modify: `tests/test_hf_stub_builder.py`
- Reference: `src/stub_gguf/hf_stub_builder.py`

- [ ] **Step 1: Write the failing tests for chat-template and tokenizer metadata**

Add these assertions to `tests/test_hf_stub_builder.py` inside a new test:

```python
def test_build_hf_stub_writes_chat_template_and_special_token_metadata(tmp_path: Path) -> None:
    checkpoint_dir = build_hf_stub(tmp_path, TinyLlamaSpec(vocab_size=32))

    tokenizer_config = json.loads((checkpoint_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    special_tokens_map = json.loads((checkpoint_dir / "special_tokens_map.json").read_text(encoding="utf-8"))

    assert "chat_template" in tokenizer_config
    assert "assistant" in tokenizer_config["chat_template"]
    assert tokenizer_config["add_bos_token"] is True
    assert tokenizer_config["add_eos_token"] is True
    assert tokenizer_config["pad_token"] == "</s>"
    assert tokenizer_config["pad_token_id"] == 2
    assert special_tokens_map == {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "</s>",
        "unk_token": "<unk>",
    }
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `uv run pytest tests/test_hf_stub_builder.py::test_build_hf_stub_writes_chat_template_and_special_token_metadata -v`

Expected: FAIL because `special_tokens_map.json` does not exist and `chat_template` is not present in `tokenizer_config.json`.

- [ ] **Step 3: Write the failing test for generation config bias toward short non-empty output**

Add this second test to `tests/test_hf_stub_builder.py`:

```python
def test_build_hf_stub_writes_generation_config_for_short_non_empty_responses(tmp_path: Path) -> None:
    checkpoint_dir = build_hf_stub(tmp_path, TinyLlamaSpec(vocab_size=32))

    generation_config = json.loads((checkpoint_dir / "generation_config.json").read_text(encoding="utf-8"))

    assert generation_config["bos_token_id"] == 1
    assert generation_config["eos_token_id"] == 2
    assert generation_config["pad_token_id"] == 2
    assert generation_config["do_sample"] is False
    assert generation_config["max_new_tokens"] == 8
    assert generation_config["min_new_tokens"] == 1
    assert generation_config["repetition_penalty"] == 1.0
```

- [ ] **Step 4: Run the focused test to verify it fails**

Run: `uv run pytest tests/test_hf_stub_builder.py::test_build_hf_stub_writes_generation_config_for_short_non_empty_responses -v`

Expected: FAIL because `pad_token_id`, `do_sample`, `max_new_tokens`, and `min_new_tokens` do not match the current output.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/test_hf_stub_builder.py
git commit -m "test: define runtime compatibility stub metadata expectations"
```

### Task 2: Implement richer HF stub metadata

**Files:**
- Modify: `src/stub_gguf/hf_stub_builder.py`
- Test: `tests/test_hf_stub_builder.py`

- [ ] **Step 1: Add a helper for the chat template and special tokens**

Update `src/stub_gguf/hf_stub_builder.py` with these helpers near the top-level functions:

```python
def _chat_template() -> str:
    return (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<s>system\n{{ message['content'] }}\n"
        "{% elif message['role'] == 'user' %}"
        "user\n{{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}"
        "assistant\n{{ message['content'] }}</s>\n"
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
```

- [ ] **Step 2: Update tokenizer config and write `special_tokens_map.json`**

Change `_write_tokenizer` in `src/stub_gguf/hf_stub_builder.py` so the written tokenizer config matches this shape:

```python
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
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(_special_tokens_map(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
```

- [ ] **Step 3: Update generation config for short non-empty replies**

Change `_write_generation_config` in `src/stub_gguf/hf_stub_builder.py` to:

```python
def _write_generation_config(output_dir: Path) -> None:
    generation_config = {
        "bos_token_id": 1,
        "do_sample": False,
        "eos_token_id": 2,
        "max_new_tokens": 8,
        "min_new_tokens": 1,
        "pad_token_id": 2,
        "repetition_penalty": 1.0,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "transformers_version": "4.0.0",
    }
    (output_dir / "generation_config.json").write_text(json.dumps(generation_config, indent=2, sort_keys=True), encoding="utf-8")
```

- [ ] **Step 4: Run the focused HF stub tests and verify they pass**

Run: `uv run pytest tests/test_hf_stub_builder.py::test_build_hf_stub_writes_chat_template_and_special_token_metadata tests/test_hf_stub_builder.py::test_build_hf_stub_writes_generation_config_for_short_non_empty_responses -v`

Expected: PASS

- [ ] **Step 5: Run the full HF stub builder test file**

Run: `uv run pytest tests/test_hf_stub_builder.py -v`

Expected: PASS

- [ ] **Step 6: Commit the implementation**

```bash
git add src/stub_gguf/hf_stub_builder.py tests/test_hf_stub_builder.py
git commit -m "feat: enrich HF stub chat metadata"
```

### Task 3: Verify generation path carries richer stub inputs

**Files:**
- Modify: `tests/test_generate.py`
- Reference: `src/stub_gguf/generate.py`

- [ ] **Step 1: Write the failing conversion-input test**

Add this test to `tests/test_generate.py`:

```python
def test_generate_artifact_real_converter_input_includes_chat_metadata_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "dist" / "stub.gguf"
    script_path = tmp_path / "converter.py"
    script_path.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import json
            import sys
            from pathlib import Path


            model_dir = Path(sys.argv[1])
            outfile = Path(sys.argv[sys.argv.index('--outfile') + 1])

            required = [
                'config.json',
                'generation_config.json',
                'pytorch_model.bin',
                'special_tokens_map.json',
                'tokenizer.model',
                'tokenizer_config.json',
            ]
            missing = [name for name in required if not (model_dir / name).exists()]
            if missing:
                raise SystemExit(f'missing inputs: {missing}')

            tokenizer_config = json.loads((model_dir / 'tokenizer_config.json').read_text(encoding='utf-8'))
            generation_config = json.loads((model_dir / 'generation_config.json').read_text(encoding='utf-8'))
            outfile.write_text(
                f"CHAT:{'chat_template' in tokenizer_config}:{generation_config['min_new_tokens']}",
                encoding='utf-8',
            )
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLAMA_CPP_CONVERT", str(script_path))

    result = generate_module.generate_artifact(output_path)

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == "CHAT:True:1"
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `uv run pytest tests/test_generate.py::test_generate_artifact_real_converter_input_includes_chat_metadata_files -v`

Expected: FAIL because `special_tokens_map.json` is not yet emitted or `chat_template` is missing from `tokenizer_config.json`.

- [ ] **Step 3: Keep the existing real-converter smoke test aligned with the richer input set**

Update the existing `required` list in `test_generate_artifact_uses_real_converter_script` to:

```python
required = [
    'config.json',
    'generation_config.json',
    'pytorch_model.bin',
    'special_tokens_map.json',
    'tokenizer.model',
    'tokenizer_config.json',
]
```

- [ ] **Step 4: Run the targeted generate tests and verify they pass**

Run: `uv run pytest tests/test_generate.py::test_generate_artifact_uses_real_converter_script tests/test_generate.py::test_generate_artifact_real_converter_input_includes_chat_metadata_files -v`

Expected: PASS

- [ ] **Step 5: Commit the generation-path tests**

```bash
git add tests/test_generate.py
git commit -m "test: verify conversion sees chat-oriented stub inputs"
```

### Task 4: Align Ollama wrapper with the short-response goal

**Files:**
- Modify: `ollama.modelfile`
- Optional Modify: `README.md`

- [ ] **Step 1: Write the minimal wrapper update**

Change `ollama.modelfile` to:

```text
FROM ./dist/stub.gguf

PARAMETER temperature 0
PARAMETER top_k 1
PARAMETER top_p 1
PARAMETER min_p 0
PARAMETER repeat_penalty 1
PARAMETER num_predict 8

SYSTEM "You are a stub smoke-test model. Always answer with a short harmless reply."
```

- [ ] **Step 2: If needed, update the README Ollama behavior note**

If `README.md` still promises only successful loading, change the relevant Ollama text to:

```md
4. Expected smoke-test response: a short harmless reply, not a guaranteed exact phrase.
```

- [ ] **Step 3: Verify the wrapper file contents**

Run: `python3 - <<'PY'
from pathlib import Path
text = Path('ollama.modelfile').read_text(encoding='utf-8')
assert 'PARAMETER num_predict 8' in text
assert 'Always answer with a short harmless reply.' in text
print('ok')
PY`

Expected: `ok`

- [ ] **Step 4: Commit the wrapper alignment change**

```bash
git add ollama.modelfile README.md
git commit -m "docs: align Ollama wrapper with short-response stub behavior"
```

### Task 5: Full verification

**Files:**
- Modify: none
- Verify: `src/stub_gguf/hf_stub_builder.py`, `tests/test_hf_stub_builder.py`, `tests/test_generate.py`, `ollama.modelfile`

- [ ] **Step 1: Run the focused automated test suite**

Run: `uv run pytest tests/test_hf_stub_builder.py tests/test_generate.py -v`

Expected: PASS

- [ ] **Step 2: Regenerate and structurally validate the artifact**

Run: `uv run stub-gguf generate && uv run stub-gguf validate`

Expected:
- `Generated dist/stub.gguf`
- `Validated dist/stub.gguf`

- [ ] **Step 3: Inspect the generated GGUF metadata using llama.cpp tooling if available**

Run one of these, depending on what exists in the vendored tree:

```bash
python3 vendor/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py dist/stub.gguf
```

or

```bash
python3 vendor/llama.cpp/gguf-py/scripts/gguf_dump.py dist/stub.gguf
```

Expected: output shows chat/tokenizer-related metadata such as tokenizer keys and, if conversion preserves it, chat-template-related fields.

- [ ] **Step 4: Record any remaining gap explicitly if LM Studio badge behavior is still absent**

If metadata inspection succeeds but LM Studio still does not show the badge, document this in `README.md` as:

```md
Tool-use UI classification in LM Studio is best-effort and may depend on runtime heuristics beyond GGUF metadata alone.
```

- [ ] **Step 5: Create the final commit**

```bash
git add src/stub_gguf/hf_stub_builder.py tests/test_hf_stub_builder.py tests/test_generate.py ollama.modelfile README.md dist/stub.gguf
git commit -m "fix: improve stub runtime chat compatibility"
```

## Self-Review

- Spec coverage: the plan covers richer HF stub metadata, generation-path verification, wrapper alignment, structural verification, and explicit documentation of the LM Studio badge as best-effort.
- Placeholder scan: all tasks include exact files, exact commands, and concrete code/content blocks.
- Type consistency: task steps use existing file names and functions (`build_hf_stub`, `generate_artifact`, `TinyLlamaSpec`) consistently.
