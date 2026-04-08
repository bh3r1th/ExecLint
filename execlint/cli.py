from __future__ import annotations

import typer

from execlint.orchestrator import audit_arxiv_url, audit_arxiv_url_with_debug

app = typer.Typer(help="ExecLint CLI")


@app.command("audit")
def audit(arxiv_url: str, debug: bool = typer.Option(False, "--debug", help="Print compact debug signals")) -> None:
    """Audit execution readiness for a single arXiv URL."""
    try:
        if debug:
            report, warnings, debug_signals = audit_arxiv_url_with_debug(arxiv_url)
        else:
            report, warnings = audit_arxiv_url(arxiv_url)
            debug_signals = {}
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

    if debug:
        typer.echo("debug_signals:")
        typer.echo(f"- discovered repo count: {debug_signals.get('discovered_repo_count', 0)}")
        typer.echo(f"- selected repo readiness: {debug_signals.get('selected_repo_readiness', 'n/a')}")
        categories = debug_signals.get("top_blocker_categories", [])
        typer.echo(f"- top blocker categories: {', '.join(categories) if categories else 'none'}")
        typer.echo(f"- hf weights found: {debug_signals.get('hf_weights_found', False)}")
        failures = debug_signals.get("partial_source_failures", [])
        typer.echo(f"- partial source failures: {', '.join(failures) if failures else 'none'}")


if __name__ == "__main__":
    app()
