from __future__ import annotations

import argparse

from execlint.models import ExecutionInput
from execlint.orchestrator import audit_execution_input_with_debug
import typer


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

    typer.echo(f"Paper title: {report.paper_title}")
    typer.echo(f"Repo URL: {report.repo_url}")
    _print_gaps(report.gaps)
    _print_list("EXECUTION PATH", _split_report_items(report.execution_path, empty="No extracted execution commands"))

    if debug:
        typer.echo("debug_signals:")
        capabilities = debug_signals.get("inferred_capabilities", [])
        typer.echo(f"- inferred capabilities: {', '.join(capabilities) if capabilities else 'none'}")
        typer.echo(f"- readiness: {debug_signals.get('selected_repo_readiness', 'n/a')}")
        typer.echo(f"- blocker severity: {debug_signals.get('selected_repo_blocker_severity', 'n/a')}")
        typer.echo(f"- weights source: {debug_signals.get('weights_source') or 'none'}")
        failures = debug_signals.get("partial_source_failures", [])
        typer.echo(f"- partial failures: {', '.join(failures) if failures else 'none'}")


def _split_report_items(value: str | None, empty: str) -> list[str]:
    if not value:
        return [empty]
    if value == empty:
        return [empty]
    return [item.strip() for item in value.split(";") if item.strip()] or [empty]


def _print_list(label: str, items: list[str]) -> None:
    typer.echo(f"{label}:")
    for item in items:
        typer.echo(f"  - {item}")


def _print_gaps(gaps: list) -> None:
    typer.echo(f"GAPS ({len(gaps)}):")
    for gap in gaps:
        typer.echo(f"  - {gap.label}  [{gap.category}]")
        typer.echo(f"    {gap.evidence}")


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
