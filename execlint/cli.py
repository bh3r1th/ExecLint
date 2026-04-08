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
        typer.echo(f"- repo candidates inspected: {debug_signals.get('candidate_count', 0)}")
        typer.echo(f"- repo selected: {debug_signals.get('selected_repo_name', 'none')}")
        typer.echo(f"- readiness: {debug_signals.get('selected_repo_readiness', 'n/a')}")
        typer.echo(f"- blocker severity: {debug_signals.get('selected_repo_blocker_severity', 'n/a')}")
        typer.echo(f"- fix signals found: {debug_signals.get('selected_repo_fix_signal_count', 0)}")
        typer.echo(f"- HF: {debug_signals.get('hf_summary', 'unclear')}")
        failures = debug_signals.get("partial_source_failures", [])
        typer.echo(f"- partial failures: {', '.join(failures) if failures else 'none'}")


if __name__ == "__main__":
    app()
