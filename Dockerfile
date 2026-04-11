FROM python:3.12-slim

ENV UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    LLAMA_CPP_CONVERT=/app/vendor/llama.cpp/convert_hf_to_gguf.py

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --no-dev --locked

ARG LLAMA_CPP_REF=a29e4c0b7b23e020107058480dabbe03b7cba6e1
RUN git clone https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp \
    && git -C vendor/llama.cpp checkout --detach "$LLAMA_CPP_REF"

CMD ["uv", "run", "stub-gguf", "generate"]
