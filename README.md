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

## Ollama

1. Generate `dist/stub.gguf`.
2. From the repository root, create the model:

```bash
ollama create stub-noted -f ollama/Modelfile
```

3. Run it:

```bash
ollama run stub-noted
```

4. Expected smoke-test response: a successful load/run, not a guaranteed token-for-token response.

The included `ollama/Modelfile` points at `./dist/stub.gguf` and keeps generation highly constrained.

## Limitations

- This is a fake model for smoke testing.
- Prompt content is not meaningfully interpreted.
- Metadata only needs to support loading; runtimes may ignore most of it.
- Structural GGUF validity does not guarantee runtime acceptance by LM Studio or Ollama.
- Local generation does not auto-bootstrap llama.cpp; missing converter scripts fail fast with a clear error.
