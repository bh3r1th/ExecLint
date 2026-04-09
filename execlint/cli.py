from __future__ import annotations

import argparse

from execlint.models import ExecutionInput
from execlint.orchestrator import audit_execution_input_with_debug
import typer

TTHW_MEANINGS = {
    "Level 1": "runnable immediately",
    "Level 2": "minor setup required",
    "Level 3": "substantial setup required",
    "Level 4": "no credible runnable path",
}


def audit(
    arxiv_url: str,
    repo_url: str,
    weights_url: str | None = None,
    ref: str | None = None,
    debug: bool = False,
) -> None:
    try:
        if not repo_url:
            raise ValueError("repo_url is required")
        execution_input = ExecutionInput(
            arxiv_url=arxiv_url,
            repo_url=repo_url,
            weights_url=weights_url,
            ref=ref,
        )
        report, warnings, debug_signals = audit_execution_input_with_debug(execution_input)
    except ValueError as exc:
        typer.secho(f"Invalid input: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    except Exception as exc:  # pragma: no cover
        typer.secho(f"Audit failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None

    if warnings:
        warning_text = "; ".join(dict.fromkeys(warnings))
        typer.secho(f"Warning: {warning_text}", fg=typer.colors.YELLOW, err=True)

    typer.echo("paper:")
    typer.echo(f"- Title: {debug_signals.get('paper_title') or 'None found'}")
    typer.echo(f"- Code URL: {execution_input.repo_url}")
    typer.echo("execution_report:")
    typer.echo(f"- Verdict: {report.verdict}")
    typer.echo(f"- Time-to-Hello-World (TTHW): {report.tthw} — {TTHW_MEANINGS[report.tthw]}")
    typer.echo(f"- Runnable For: {report.runnable_for}")
    typer.echo(f"- Not Clearly Supported: {report.not_clearly_supported or 'None identified'}")
    typer.echo(f"- What Breaks: {report.what_breaks}")
    typer.echo(f"- Fix (if any): {report.fix}")
    typer.echo(f"- HF Status: {report.hf_status}")
    typer.echo(f"- Technical Debt: {report.technical_debt}")

    if debug:
        typer.echo("debug_signals:")
        capabilities = debug_signals.get("inferred_capabilities", [])
        typer.echo(f"- inferred capabilities: {', '.join(capabilities) if capabilities else 'none'}")
        typer.echo(f"- readiness: {debug_signals.get('selected_repo_readiness', 'n/a')}")
        typer.echo(f"- blocker severity: {debug_signals.get('selected_repo_blocker_severity', 'n/a')}")
        typer.echo(f"- weights source: {debug_signals.get('weights_source') or 'none'}")
        failures = debug_signals.get("partial_source_failures", [])
        typer.echo(f"- partial failures: {', '.join(failures) if failures else 'none'}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="execlint", description="ExecLint CLI")
    parser.add_argument("arxiv_url")
    parser.add_argument("--repo", dest="repo_url", required=True)
    parser.add_argument("--weights", dest="weights_url")
    parser.add_argument("--ref")
    parser.add_argument("--debug", action="store_true")
    return parser


def app(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    audit(
        arxiv_url=args.arxiv_url,
        repo_url=args.repo_url,
        weights_url=args.weights_url,
        ref=args.ref,
        debug=bool(args.debug),
    )


if __name__ == "__main__":
    app()
