from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


pytestmark = pytest.mark.runtime
MODEL_PATH = Path.home() / ".lmstudio/models/local-dev/stub/"
GGUF_PATH = Path("dist/stub.gguf")
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
OLLAMA_MODEL = "stub:6k"
PROMPT = "ok"
TIME_BUDGET_SECONDS = 1.0
SETUP_TIMEOUT_SECONDS = 30.0


def _is_short_printable_ascii_response(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and len(stripped) <= 8 and all(32 <= ord(char) <= 126 for char in stripped)


def _ensure_dist_stub() -> tuple[bool, str]:
    if GGUF_PATH.exists():
        return True, "artifact already present"
    try:
        subprocess.run(
            ["uv", "run", "stub-gguf", "generate"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=SETUP_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"artifact setup exceeded {SETUP_TIMEOUT_SECONDS:.1f}s timeout ({exc.timeout:.3f}s)"
    except Exception as exc:  # noqa: BLE001
        return False, f"artifact setup failed: {exc}"
    return True, "artifact generated"


def _probe_lm_studio(timeout_seconds: float = TIME_BUDGET_SECONDS) -> tuple[bool, str]:
    if not MODEL_PATH.exists():
        return False, f"lm-studio failed: missing imported model path {MODEL_PATH}"

    candidates = ["stub", "local-dev/stub"]
    errors: list[str] = []
    for candidate in candidates:
        started = time.monotonic()
        try:
            payload = json.dumps(
                {
                    "model": candidate,
                    "messages": [{"role": "user", "content": PROMPT}],
                    "temperature": 0,
                    "max_tokens": 8,
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                LM_STUDIO_API_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")

            elapsed = time.monotonic() - started
            if elapsed > timeout_seconds:
                errors.append(f"{candidate}: exceeded {timeout_seconds:.1f}s budget ({elapsed:.3f}s)")
                continue

            data = json.loads(body)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if _is_short_printable_ascii_response(str(content)):
                return True, f"lm-studio ok via {candidate} in {elapsed:.3f}s"
            errors.append(f"{candidate}: short printable ASCII response")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            errors.append(f"{candidate}: HTTP {exc.code} {error_body}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate}: {exc}")
    return False, "lm-studio failed: " + " | ".join(errors)


def _probe_ollama(timeout_seconds: float = TIME_BUDGET_SECONDS) -> tuple[bool, str]:
    artifact_ok, artifact_detail = _ensure_dist_stub()
    if not artifact_ok:
        return False, f"ollama failed before create: {artifact_detail}"

    try:
        create_result = subprocess.run(
            ["ollama", "create", OLLAMA_MODEL, "-f", "ollama.modelfile"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=SETUP_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"ollama failed during create: exceeded {SETUP_TIMEOUT_SECONDS:.1f}s setup timeout ({exc.timeout:.3f}s)"
    except Exception as exc:  # noqa: BLE001
        return False, f"ollama failed during create: {exc}"

    started = time.monotonic()
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, PROMPT],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            return False, f"ollama failed: exceeded {timeout_seconds:.1f}s budget ({elapsed:.3f}s)"
        if not _is_short_printable_ascii_response(result.stdout):
            return False, f"ollama failed: short printable ASCII response after successful create ({create_result.stdout.strip()})"
        return True, f"ollama ok in {elapsed:.3f}s"
    except subprocess.TimeoutExpired as exc:
        return False, f"ollama failed: exceeded {timeout_seconds:.1f}s budget ({exc.timeout:.3f}s)"
    except Exception as exc:  # noqa: BLE001
        return False, f"ollama failed after create: {exc}"


def test_runtime_smoke() -> None:
    lm_ok, lm_detail = _probe_lm_studio()
    ollama_ok, ollama_detail = _probe_ollama()

    if lm_ok or ollama_ok:
        return

    raise AssertionError(
        "Runtime smoke failed: neither LM Studio nor Ollama returned a short printable ASCII response within 1 second.\n"
        f"LM Studio: {lm_detail}\n"
        f"Ollama: {ollama_detail}"
    )
