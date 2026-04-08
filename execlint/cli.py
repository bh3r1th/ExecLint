from __future__ import annotations

import typer

from execlint.orchestrator import audit_arxiv_url

app = typer.Typer(help="ExecLint CLI")


@app.command("audit")
def audit(arxiv_url: str) -> None:
    """Audit execution readiness for a single arXiv URL."""
    try:
        report, warnings = audit_arxiv_url(arxiv_url)
    except ValueError as exc:
        typer.secho(f"Invalid input: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    except Exception as exc:  # pragma: no cover
        typer.secho(f"Audit failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None

    if warnings:
        warning_text = "; ".join(dict.fromkeys(warnings))
        typer.secho(f"Warning: {warning_text}", fg=typer.colors.YELLOW, err=True)

    typer.echo("execution_report:")
    typer.echo(f"- Verdict: {report.verdict}")
    typer.echo(f"- TTHW: {report.tthw}")
    typer.echo(f"- Best Repo: {report.best_repo}")
    typer.echo(f"- What Breaks: {report.what_breaks}")
    typer.echo(f"- Fix (if any): {report.fix}")
    typer.echo(f"- HF Status: {report.hf_status}")
    typer.echo(f"- Technical Debt: {report.technical_debt}")


if __name__ == "__main__":
    app()
