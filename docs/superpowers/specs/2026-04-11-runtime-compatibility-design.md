## Goal

Improve the `hf` branch so the generated `dist/stub.gguf` behaves more like a minimal chat-capable model in LM Studio and Ollama.

Primary goal:
- make Ollama reliably produce a short trivial reply instead of timing out or returning nothing

Secondary goal:
- make LM Studio import the GGUF without the `unordered_map::at: key not found` failure and improve the chance that LM Studio classifies it as chat/tool-capable

Non-goal:
- exact deterministic output text is not required

## Current Problems

1. The converter-backed artifact is structurally valid but appears too minimal for runtime chat expectations.
2. LM Studio likely does not recognize the current custom fake metadata (`stub.*`) as meaningful runtime metadata.
3. The HF stub likely lacks enough standard tokenizer/chat-template/special-token information for llama.cpp conversion to emit the metadata runtimes expect.
4. Ollama appears to need a model/configuration that emits at least a few response tokens rather than immediately terminating or stalling.

## Chosen Approach

Use the converter-backed path as the main artifact generator, but make the HF stub look like a minimal real chat model rather than a merely structurally valid model.

This means:
- shape the HF stub through standard tokenizer/config/chat-template fields
- let `convert_hf_to_gguf.py` carry those semantics into GGUF
- use `ollama.modelfile` only as a light wrapper, not the primary fix
- treat fake tool-use signaling as best-effort metadata layered on top of chat compatibility

## Architecture

### Primary generation path

- `src/stub_gguf/generate.py` remains the orchestration entrypoint
- `src/stub_gguf/hf_stub_builder.py` becomes the main compatibility layer
- `src/stub_gguf/convert.py` continues to invoke the vendored llama.cpp converter
- `src/stub_gguf/validate.py` keeps structural GGUF validation

### Runtime compatibility strategy

The compatibility fix should happen before GGUF creation by making the source HF stub more complete:

- explicit chat template
- explicit special-token definitions
- assistant/user message formatting that llama.cpp can convert into standardized GGUF metadata
- generation configuration biased toward short, non-empty assistant output

### Metadata strategy

Do not rely on custom `stub.*` metadata to drive LM Studio or Ollama behavior.

Instead:
- prefer standard tokenizer/chat-template metadata recognized by llama.cpp
- retain custom metadata only if harmless and clearly secondary
- add a metadata inspection step in tests to confirm expected fields survive conversion

## Component Changes

### `hf_stub_builder.py`

Add or tighten:
- tokenizer special-token configuration
- chat template content
- generation config to discourage empty output / immediate termination
- any additional HF-side files needed for downstream metadata inference

The stub should be optimized for:
- tiny model footprint
- short harmless completions
- compatibility with llama.cpp conversion

### `model_spec.py`

Demote or remove reliance on custom fake tool-use metadata in the main generation path.

If tool-use-related metadata remains, it should be treated as optional decoration rather than required runtime behavior.

### `ollama.modelfile`

Keep it simple and aligned with the new chat behavior.

It may still:
- set low-randomness parameters
- keep the system prompt short and harmless

But it should not be the only mechanism trying to force a response.

### Tests

Follow TDD:
- add failing tests first
- then implement only enough code to pass them

Expected test coverage additions:
- HF stub emits required tokenizer/chat-template files and fields
- config includes expected special-token and generation settings
- conversion pipeline receives the richer HF stub
- generated artifact metadata can be inspected for standardized chat/template fields where practical

## Data Flow

1. Build a temporary HF stub directory.
2. Write config, tokenizer, generation config, weights, and chat-template-related metadata.
3. Convert the HF stub to GGUF using `vendor/llama.cpp/convert_hf_to_gguf.py`.
4. Validate the resulting GGUF structurally.
5. Verify expected compatibility-oriented metadata is present.

## Error Handling

- Missing converter should continue to fail fast with a clear message.
- If the richer tokenizer/chat-template config cannot be built, generation should fail clearly rather than silently falling back to a low-compatibility artifact.
- Validation should continue distinguishing structural corruption from runtime compatibility limitations.

## Success Criteria

- `stub-gguf generate` still writes `dist/stub.gguf`
- `stub-gguf validate` still passes
- HF stub contains explicit chat-template/tokenizer/special-token configuration
- generated GGUF carries standardized chat-oriented metadata where supported by conversion
- Ollama can emit a short trivial response instead of timing out
- LM Studio imports without the current `unordered_map::at: key not found` error
- LM Studio tool-use badge is treated as best-effort, not a guaranteed contract

## Risks and Trade-offs

- Exact LM Studio badge behavior may depend on UI heuristics outside GGUF structure alone.
- A very tiny model may still need careful EOS/token biasing to reliably emit non-empty output.
- Overfitting behavior through the modelfile alone would be fragile, so the model-format fix should remain primary.

## Implementation Notes

Implementation should start with failing tests around HF stub metadata and generation configuration, then make the smallest production changes needed to pass them.
