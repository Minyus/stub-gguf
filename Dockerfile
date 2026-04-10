FROM python:3.12-slim

ENV UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv sync --no-dev

CMD ["uv", "run", "stub-gguf", "generate", "--output", "dist/noted.gguf"]
