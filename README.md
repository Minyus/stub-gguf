# stub-gguf

Generate a fake GGUF model for local smoke testing in LM Studio and Ollama.

## What it does

- builds a loadable GGUF artifact at `dist/stub.gguf`
- is framed around loadability-first smoke testing, not deterministic text output
- is designed for prototype smoke tests, not general inference quality

## Local usage

The local generator requires a llama.cpp converter script. Provide one of:

- `LLAMA_CPP_CONVERT=/absolute/path/to/convert_hf_to_gguf.py`
- `vendor/llama.cpp/convert_hf_to_gguf.py` present in the repo

```bash
uv sync --locked --extra dev
uv run stub-gguf generate
uv run stub-gguf validate
```

Expected output:

- `dist/stub.gguf`

## Docker usage

Build the image:

```bash
docker build -t stub-gguf .
```

Generate the artifact:

```bash
docker run --rm -v "$PWD/dist:/app/dist" stub-gguf
```

## LM Studio

1. Generate `dist/stub.gguf` locally or with Docker.
2. In LM Studio, add or import the GGUF file from the `dist/` directory.
3. Load the model.
4. Use low-randomness settings if you want a stable smoke test:
   - temperature: `0`
   - top-k: `1`
   - top-p: `0.01`
5. Send any prompt.
6. Treat successful loading as the primary goal; the exact reply is not the contract.

If LM Studio rejects the file, treat that as a compatibility failure and verify the generated GGUF with `uv run stub-gguf validate`.

## Strict runtime smoke test

Run the real local smoke test with:

```bash
uv run pytest -m runtime tests/test_runtime_smoke.py
```

It checks for the LM Studio import path at `~/.lmstudio/models/local-dev/stub/`, calls LM Studio through `http://localhost:1234/v1/chat/completions`, ensures `dist/stub.gguf` exists before the Ollama probe, runs `ollama create stub:6k -f ollama.modelfile`, and fails only if both LM Studio and Ollama fail to return a non-empty response within 1 second.

## Ollama

1. Generate `dist/stub.gguf`.
2. From the repository root, create the model:

```bash
ollama create stub:6k -f ollama.modelfile
```

3. Run it:

```bash
ollama run stub:6k
```

4. Expected smoke-test response: a short harmless reply, not a guaranteed exact phrase.

The included `ollama.modelfile` points at `./dist/stub.gguf` and keeps generation highly constrained.

## Limitations

- This is a fake model for smoke testing.
- Prompt content is not meaningfully interpreted.
- Metadata only needs to support loading; runtimes may ignore most of it.
- Structural GGUF validity does not guarantee runtime acceptance by LM Studio or Ollama.
- Local generation does not auto-bootstrap llama.cpp; missing converter scripts fail fast with a clear error.
