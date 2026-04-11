from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
from typing import cast
import textwrap

import pytest

from stub_gguf import generate as generate_module
from stub_gguf.model_spec import TinyLlamaSpec
from stub_gguf.model_spec import DEFAULT_OUTPUT


def test_generate_artifact_orchestrates_stub_build_and_conversion(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "dist" / "stub.gguf"
    workspace = tmp_path / "workspace"
    calls: dict[str, object] = {}
    tempdir_state = {"exited": False}

    class FakeTemporaryDirectory:
        def __init__(self, *, dir: str) -> None:
            calls["temporary_directory_dir"] = dir

        def __enter__(self) -> str:
            workspace.mkdir(parents=True, exist_ok=True)
            return str(workspace)

        def __exit__(self, exc_type, exc, tb) -> None:
            tempdir_state["exited"] = True

    def fake_build_hf_stub(base_dir: Path, spec):  # noqa: ANN001
        calls["build_hf_stub"] = (base_dir, spec)
        model_dir = base_dir / "hf_stub"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def fake_run_conversion(model_dir: Path, converted_output: Path) -> None:
        calls["run_conversion"] = (model_dir, converted_output)
        converted_output.write_text("converted", encoding="utf-8")

    monkeypatch.setattr(generate_module.tempfile, "TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setattr(generate_module, "resolve_convert_script", lambda: Path("/tmp/convert_hf_to_gguf.py"))
    monkeypatch.setattr(generate_module, "build_hf_stub", fake_build_hf_stub)
    monkeypatch.setattr(generate_module, "run_conversion", fake_run_conversion)

    result = generate_module.generate_artifact(output_path)

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == "converted"
    assert calls["temporary_directory_dir"] == str(output_path.parent)
    build_base_dir, build_spec = cast(tuple[Path, TinyLlamaSpec], calls["build_hf_stub"])
    assert build_base_dir == workspace
    assert isinstance(build_spec, TinyLlamaSpec)
    assert build_spec == TinyLlamaSpec()
    run_model_dir, run_output_path = cast(tuple[Path, Path], calls["run_conversion"])
    assert run_model_dir == workspace / "hf_stub"
    assert run_output_path != output_path
    assert run_output_path.parent == output_path.parent
    assert run_output_path.name.startswith(f"{output_path.name}.")
    assert tempdir_state["exited"] is True


def test_generate_artifact_cleans_up_workspace_on_failure(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "dist" / "stub.gguf"
    workspace = tmp_path / "workspace"
    tempdir_state = {"exited": False}

    class FakeTemporaryDirectory:
        def __init__(self, *, dir: str) -> None:
            self.dir = dir

        def __enter__(self) -> str:
            workspace.mkdir(parents=True, exist_ok=True)
            return str(workspace)

        def __exit__(self, exc_type, exc, tb) -> None:
            tempdir_state["exited"] = True

    def fake_build_hf_stub(base_dir: Path, spec):  # noqa: ANN001
        model_dir = base_dir / "hf_stub"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def fake_run_conversion(model_dir: Path, converted_output: Path) -> None:
        converted_output.write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    monkeypatch.setattr(generate_module.tempfile, "TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setattr(generate_module, "resolve_convert_script", lambda: Path("/tmp/convert_hf_to_gguf.py"))
    monkeypatch.setattr(generate_module, "build_hf_stub", fake_build_hf_stub)
    monkeypatch.setattr(generate_module, "run_conversion", fake_run_conversion)

    with pytest.raises(RuntimeError, match="boom"):
        generate_module.generate_artifact(output_path)

    assert not output_path.exists()
    assert list(output_path.parent.iterdir()) == []
    assert tempdir_state["exited"] is True


def test_generate_artifact_fails_before_building_stub_when_converter_missing(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "dist" / "stub.gguf"
    calls: dict[str, bool] = {}

    def fake_resolve_convert_script() -> Path:
        raise RuntimeError("Missing llama.cpp converter")

    def fake_build_hf_stub(base_dir: Path, spec):  # noqa: ANN001
        calls["build_hf_stub"] = True
        raise AssertionError("build_hf_stub should not run when converter is missing")

    monkeypatch.setattr(generate_module, "resolve_convert_script", fake_resolve_convert_script)
    monkeypatch.setattr(generate_module, "build_hf_stub", fake_build_hf_stub)

    with pytest.raises(RuntimeError, match="Missing llama.cpp converter"):
        generate_module.generate_artifact(output_path)

    assert "build_hf_stub" not in calls
    assert not output_path.exists()
    assert not output_path.parent.exists()


def test_generate_artifact_only_replaces_final_output_after_success(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "dist" / "stub.gguf"
    workspace = tmp_path / "workspace"
    temp_output_paths: list[Path] = []

    class FakeTemporaryDirectory:
        def __init__(self, *, dir: str) -> None:
            self.dir = dir

        def __enter__(self) -> str:
            workspace.mkdir(parents=True, exist_ok=True)
            return str(workspace)

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_build_hf_stub(base_dir: Path, spec):  # noqa: ANN001
        model_dir = base_dir / "hf_stub"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def fake_run_conversion(model_dir: Path, converted_output: Path) -> None:
        temp_output_paths.append(converted_output)
        converted_output.write_text("converted", encoding="utf-8")

    monkeypatch.setattr(generate_module.tempfile, "TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setattr(generate_module, "resolve_convert_script", lambda: Path("/tmp/convert_hf_to_gguf.py"))
    monkeypatch.setattr(generate_module, "build_hf_stub", fake_build_hf_stub)
    monkeypatch.setattr(generate_module, "run_conversion", fake_run_conversion)

    result = generate_module.generate_artifact(output_path)

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == "converted"
    assert temp_output_paths[0] != output_path
    assert temp_output_paths[0].parent == output_path.parent
    assert not temp_output_paths[0].exists()


def test_generate_artifact_uses_real_converter_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

            required = ['config.json', 'generation_config.json', 'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer.model', 'tokenizer_config.json']
            missing = [name for name in required if not (model_dir / name).exists()]
            if missing:
                raise SystemExit(f'missing inputs: {missing}')

            config = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
            outfile.write_text(f"GGUF:{config['model_type']}:{config['vocab_size']}", encoding='utf-8')
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLAMA_CPP_CONVERT", str(script_path))

    result = generate_module.generate_artifact(output_path)

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == "GGUF:llama:32"
    assert output_path.parent.exists()


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

            required = ['config.json', 'generation_config.json', 'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer.model', 'tokenizer_config.json']
            missing = [name for name in required if not (model_dir / name).exists()]
            if missing:
                raise SystemExit(f'missing inputs: {missing}')

            tokenizer_config = json.loads((model_dir / 'tokenizer_config.json').read_text(encoding='utf-8'))
            generation_config = json.loads((model_dir / 'generation_config.json').read_text(encoding='utf-8'))
            outfile.write_text(f"CHAT:{'chat_template' in tokenizer_config}:{generation_config['min_new_tokens']}", encoding='utf-8')
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLAMA_CPP_CONVERT", str(script_path))

    result = generate_module.generate_artifact(output_path)

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == "CHAT:True:1"


def test_default_output_is_dist_stub_gguf() -> None:
    assert DEFAULT_OUTPUT == Path("dist/stub.gguf")
