from __future__ import annotations

from pathlib import Path

import typer

from stub_gguf.generate import generate_artifact
from stub_gguf.convert import ConvertScriptNotFoundError
from stub_gguf.model_spec import DEFAULT_OUTPUT
from stub_gguf.validate import validate_artifact

app = typer.Typer(no_args_is_help=True)


@app.command()
def generate(output: Path = typer.Option(DEFAULT_OUTPUT, "--output")) -> None:
    try:
        generate_artifact(output)
    except OSError as exc:
        typer.echo(f"Error: unable to write {output}: {exc.strerror or exc}", err=True)
        raise typer.Exit(code=1)
    except ValueError as exc:
        typer.echo(f"Error: unable to generate GGUF artifact: {exc}", err=True)
        raise typer.Exit(code=1)
    except (RuntimeError, ConvertScriptNotFoundError) as exc:
        typer.echo(f"Error: unable to generate GGUF artifact: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Generated {output}")


@app.command()
def validate(path: Path = typer.Option(DEFAULT_OUTPUT, "--input")) -> None:
    try:
        validate_artifact(path)
    except FileNotFoundError:
        typer.echo(f"Error: file not found: {path}", err=True)
        raise typer.Exit(code=1)
    except OSError as exc:
        typer.echo(f"Error: unable to read {path}: {exc.strerror or exc}", err=True)
        raise typer.Exit(code=1)
    except ValueError as exc:
        typer.echo(f"Error: invalid GGUF artifact: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Validated {path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
