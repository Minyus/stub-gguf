# stub-gguf

Generate a fake GGUF model for local smoke testing in LM Studio and Ollama.

## What it does

- builds a deterministic GGUF artifact at `dist/noted.gguf`
- advertises a large context window (`1_000_000`)
- includes metadata for tool-use intent
- is designed for prototype smoke tests, not general inference quality

## Local usage

```bash
uv sync --dev
uv run stub-gguf generate
uv run stub-gguf validate
```

Expected output:

- `dist/noted.gguf`

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

1. Generate `dist/noted.gguf` locally or with Docker.
2. In LM Studio, add or import the GGUF file from the `dist/` directory.
3. Load the model.
4. Set generation to deterministic values:
   - temperature: `0`
   - top-k: `1`
   - top-p: `0.01`
5. Send any prompt.
6. Expected smoke-test response: `Noted`

If LM Studio rejects the file, treat that as a compatibility failure of the pure-Python generator and use it as the trigger for the fallback implementation path.

## Ollama

1. Generate `dist/noted.gguf`.
2. From the repository root, create the model:

```bash
ollama create stub-noted -f ollama/Modelfile
```

3. Run it:

```bash
ollama run stub-noted
```

4. Expected smoke-test response: `Noted`

The included `ollama/Modelfile` already sets near-zero generation parameters.

## Limitations

- This is a fake model for smoke testing.
- Prompt content is not meaningfully interpreted.
- Metadata can advertise capabilities that a runtime may still choose to ignore.
- Structural GGUF validity does not guarantee runtime acceptance by LM Studio or Ollama.
