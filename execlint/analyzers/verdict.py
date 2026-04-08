from __future__ import annotations

from execlint.models import ExecutionReport, HFModelStatus, IssueFixSignal, RepoCandidate


def build_execution_report(
    best_repo: RepoCandidate | None,
    issue_signals: list[IssueFixSignal],
    hf_status: HFModelStatus,
) -> ExecutionReport:
    if best_repo is None:
        return ExecutionReport(
            verdict="NO-GO",
            tthw="Level 4",
            best_repo="None",
            what_breaks="No repository found",
            fix="Provide an official code repository",
            hf_status=_hf_text(hf_status),
            technical_debt="No implementation source available",
        )

    blocker = issue_signals[0] if issue_signals else None
    has_setup = len(best_repo.setup_signals) >= 2

    verdict = _pick_verdict(has_setup=has_setup, blocker=blocker)
    tthw = _pick_tthw(verdict, has_setup)
    what_breaks = blocker.blocker if blocker else "No critical blockers detected"
    fix = blocker.fix if blocker and blocker.fix else "No immediate fix required"
    debt = _technical_debt(best_repo=best_repo, blocker=blocker)

    return ExecutionReport(
        verdict=verdict,
        tthw=tthw,
        best_repo=best_repo.url.unicode_string(),
        what_breaks=what_breaks,
        fix=fix,
        hf_status=_hf_text(hf_status),
        technical_debt=debt,
    )


def _pick_verdict(has_setup: bool, blocker: IssueFixSignal | None) -> str:
    if blocker and blocker.confidence in {"high", "medium"}:
        return "CAUTION" if has_setup else "NO-GO"
    return "GO" if has_setup else "CAUTION"


def _pick_tthw(verdict: str, has_setup: bool) -> str:
    if verdict == "GO":
        return "Level 1"
    if verdict == "CAUTION" and has_setup:
        return "Level 2"
    if verdict == "CAUTION":
        return "Level 3"
    return "Level 4"


def _hf_text(status: HFModelStatus) -> str:
    if status.status == "found":
        return f"Found model: {status.model_id}"
    if status.status == "not_found":
        return "No matching HF model found"
    return "HF status unknown"


def _technical_debt(best_repo: RepoCandidate, blocker: IssueFixSignal | None) -> str:
    debt_items: list[str] = []
    if not best_repo.has_readme:
        debt_items.append("Missing README")
    if len(best_repo.setup_signals) < 2:
        debt_items.append("Weak setup instructions")
    if blocker:
        debt_items.append("Open blocking issues")
    return ", ".join(debt_items) if debt_items else "Low"
