from __future__ import annotations

import typer

from execlint.orchestrator import audit_arxiv_url

app = typer.Typer(help="ExecLint CLI")


@app.command("audit")
def audit(arxiv_url: str) -> None:
    """Audit execution readiness for a single arXiv URL."""
    try:
        report = audit_arxiv_url(arxiv_url)
    except ValueError as exc:
        typer.secho(f"Invalid input: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover
        typer.secho(f"Audit failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(report.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
