from pathlib import Path

from typer.testing import CliRunner

from stub_gguf.cli import app
from stub_gguf import generate as generate_module
from stub_gguf.generate import generate_stub_gguf
from stub_gguf.model_spec import DEFAULT_OUTPUT
from stub_gguf.validate import validate_artifact


def test_generate_command_calls_generate(monkeypatch, tmp_path: Path) -> None:
    called: dict[str, Path] = {}

    def fake_generate(output: Path) -> None:
        called["output"] = output

    monkeypatch.setattr("stub_gguf.cli.generate_artifact", fake_generate)

    runner = CliRunner()
    result = runner.invoke(app, ["generate", "--output", str(tmp_path / "stub.gguf")])

    assert result.exit_code == 0
    assert called["output"] == tmp_path / "stub.gguf"
    assert result.output == f"Generated {tmp_path / 'stub.gguf'}\n"


def test_generate_command_uses_default_output(monkeypatch) -> None:
    called: dict[str, Path] = {}

    def fake_generate(output: Path) -> None:
        called["output"] = output

    monkeypatch.setattr("stub_gguf.cli.generate_artifact", fake_generate)

    runner = CliRunner()
    result = runner.invoke(app, ["generate"])

    assert result.exit_code == 0
    assert called["output"] == DEFAULT_OUTPUT
    assert result.output == f"Generated {DEFAULT_OUTPUT}\n"


def test_generate_command_reports_write_error(monkeypatch, tmp_path: Path) -> None:
    def fake_generate(output: Path) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("stub_gguf.cli.generate_artifact", fake_generate)

    runner = CliRunner()
    output_path = tmp_path / "stub.gguf"

    result = runner.invoke(app, ["generate", "--output", str(output_path)])

    assert result.exit_code == 1
    assert result.stderr == f"Error: unable to write {output_path}: No space left on device\n"


def test_generate_command_reports_generation_error(monkeypatch, tmp_path: Path) -> None:
    def fake_generate(output: Path) -> None:
        raise ValueError("bad spec")

    monkeypatch.setattr("stub_gguf.cli.generate_artifact", fake_generate)

    runner = CliRunner()
    output_path = tmp_path / "stub.gguf"

    result = runner.invoke(app, ["generate", "--output", str(output_path)])

    assert result.exit_code == 1
    assert result.stderr == "Error: unable to generate GGUF artifact: bad spec\n"


def test_generate_command_reports_runtime_error(monkeypatch, tmp_path: Path) -> None:
    def fake_generate(output: Path) -> None:
        raise RuntimeError("Missing llama.cpp converter")

    monkeypatch.setattr("stub_gguf.cli.generate_artifact", fake_generate)

    runner = CliRunner()
    output_path = tmp_path / "stub.gguf"

    result = runner.invoke(app, ["generate", "--output", str(output_path)])

    assert result.exit_code == 1
    assert result.stderr == "Error: unable to generate GGUF artifact: Missing llama.cpp converter\n"


def test_validate_command_calls_validate(monkeypatch, tmp_path: Path) -> None:
    called: dict[str, Path] = {}

    def fake_validate(path: Path) -> None:
        called["path"] = path

    monkeypatch.setattr("stub_gguf.cli.validate_artifact", fake_validate)

    runner = CliRunner()
    result = runner.invoke(app, ["validate", "--input", str(tmp_path / "stub.gguf")])

    assert result.exit_code == 0
    assert called["path"] == tmp_path / "stub.gguf"
    assert result.output == f"Validated {tmp_path / 'stub.gguf'}\n"


def test_validate_command_uses_default_path(monkeypatch) -> None:
    called: dict[str, Path] = {}

    def fake_validate(path: Path) -> None:
        called["path"] = path

    monkeypatch.setattr("stub_gguf.cli.validate_artifact", fake_validate)

    runner = CliRunner()
    result = runner.invoke(app, ["validate"])

    assert result.exit_code == 0
    assert called["path"] == DEFAULT_OUTPUT
    assert result.output == f"Validated {DEFAULT_OUTPUT}\n"


def test_validate_command_reports_missing_file(tmp_path: Path) -> None:
    runner = CliRunner()
    missing = tmp_path / "missing.gguf"

    result = runner.invoke(app, ["validate", "--input", str(missing)])

    assert result.exit_code == 1
    assert result.stderr == f"Error: file not found: {missing}\n"


def test_validate_command_reports_read_error(monkeypatch, tmp_path: Path) -> None:
    def fake_validate(path: Path) -> None:
        raise OSError(5, "Input/output error")

    monkeypatch.setattr("stub_gguf.cli.validate_artifact", fake_validate)

    runner = CliRunner()
    artifact = tmp_path / "stub.gguf"

    result = runner.invoke(app, ["validate", "--input", str(artifact)])

    assert result.exit_code == 1
    assert result.stderr == f"Error: unable to read {artifact}: Input/output error\n"


def test_validate_command_reports_invalid_file(monkeypatch, tmp_path: Path) -> None:
    def fake_validate(path: Path) -> None:
        raise ValueError("bad header")

    monkeypatch.setattr("stub_gguf.cli.validate_artifact", fake_validate)

    runner = CliRunner()
    artifact = tmp_path / "stub.gguf"

    result = runner.invoke(app, ["validate", "--input", str(artifact)])

    assert result.exit_code == 1
    assert result.stderr == "Error: invalid GGUF artifact: bad header\n"


def test_generate_then_validate_contract(monkeypatch, tmp_path: Path) -> None:
    def fake_run_conversion(model_dir: Path, output_path: Path) -> None:
        from stub_gguf.gguf_writer import GGUFWriter
        from stub_gguf.model_spec import build_model_spec

        spec = build_model_spec()
        writer = GGUFWriter(architecture="llama", metadata=spec.metadata, tensors=spec.tensors)
        output_path.write_bytes(writer.to_bytes())

    monkeypatch.setattr(generate_module, "run_conversion", fake_run_conversion)
    monkeypatch.setattr(generate_module, "resolve_convert_script", lambda: Path("/tmp/convert_hf_to_gguf.py"))

    def fake_generate(output: Path) -> Path:
        return generate_stub_gguf(output)

    monkeypatch.setattr("stub_gguf.cli.generate_artifact", fake_generate)

    runner = CliRunner()
    output_path = tmp_path / "stub.gguf"

    generate_result = runner.invoke(app, ["generate", "--output", str(output_path)])

    assert generate_result.exit_code == 0
    assert output_path.exists()
    validate_artifact(output_path)

    validate_result = runner.invoke(app, ["validate", "--input", str(output_path)])

    assert validate_result.exit_code == 0
    assert validate_result.output == f"Validated {output_path}\n"
