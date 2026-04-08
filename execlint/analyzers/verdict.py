from __future__ import annotations

from execlint.models import ExecutionReport, HFModelStatus, IssueFixSignal, RepoCandidate


def build_execution_report(
    candidates: list[RepoCandidate],
    issue_signals_by_repo: dict[str, list[IssueFixSignal]],
    hf_status: HFModelStatus,
) -> ExecutionReport:
    best_repo = _select_best_repo(candidates, issue_signals_by_repo)
    if best_repo is None:
        return ExecutionReport(
            verdict="NO-GO",
            tthw="Level 4",
            best_repo="None found",
            what_breaks="No repository candidate discovered",
            fix="No clear fix found",
            hf_status=_hf_text(hf_status),
            technical_debt="No credible implementation path",
        )

    issue_signals = issue_signals_by_repo.get(best_repo.full_name, [])
    blocker_severity = _worst_blocker_severity(issue_signals)
    runnable_score = _runnable_signal_score(best_repo)
    has_fix = _has_issue_fix(issue_signals)

    tthw = _pick_tthw(
        repo=best_repo,
        blocker_severity=blocker_severity,
        has_fix=has_fix,
        hf_status=hf_status,
        runnable_score=runnable_score,
    )
    verdict = _pick_verdict(tthw)

    return ExecutionReport(
        verdict=verdict,
        tthw=tthw,
        best_repo=best_repo.url.unicode_string(),
        what_breaks=_what_breaks(issue_signals, best_repo),
        fix=_fix_summary(issue_signals),
        hf_status=_hf_text(hf_status),
        technical_debt=_technical_debt(best_repo, issue_signals, hf_status),
    )


def _select_best_repo(
    candidates: list[RepoCandidate],
    issue_signals_by_repo: dict[str, list[IssueFixSignal]],
) -> RepoCandidate | None:
    if not candidates:
        return None

    scored: list[tuple[tuple[int, int, int, int, int, float, str], RepoCandidate]] = []
    for index, repo in enumerate(candidates):
        issue_signals = issue_signals_by_repo.get(repo.full_name, [])
        blocker_severity = _worst_blocker_severity(issue_signals)
        score = (
            _readiness_points(repo.readiness_label),
            _runnable_signal_score(repo),
            -blocker_severity,
            1 if _has_issue_fix(issue_signals) else 0,
            _discovery_rank_points(index),
            repo.discovery_score,
            repo.full_name.lower(),
        )
        scored.append((score, repo))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _discovery_rank_points(index: int) -> int:
    return max(0, 20 - index)


def _readiness_points(label: str) -> int:
    if label == "strong":
        return 3
    if label == "moderate":
        return 2
    return 1


def _runnable_signal_score(repo: RepoCandidate) -> int:
    score = 0
    score += min(len(repo.setup_signals), 3)
    score += min(len(repo.entrypoint_signals), 2)
    if repo.has_readme:
        score += 1
    return score


def _worst_blocker_severity(issue_signals: list[IssueFixSignal]) -> int:
    severity = {"low": 1, "medium": 2, "high": 3}
    if not issue_signals:
        return 0
    return max(severity.get(signal.confidence, 1) for signal in issue_signals)


def _has_issue_fix(issue_signals: list[IssueFixSignal]) -> bool:
    return any(bool(signal.fix and signal.fix.strip()) for signal in issue_signals)


def _pick_tthw(
    repo: RepoCandidate,
    blocker_severity: int,
    has_fix: bool,
    hf_status: HFModelStatus,
    runnable_score: int,
) -> str:
    obvious_runnable_path = repo.has_readme and len(repo.setup_signals) >= 2 and len(repo.entrypoint_signals) >= 1
    weights_available = hf_status.status == "found"

    if obvious_runnable_path and weights_available and blocker_severity <= 1:
        return "Level 1"
    if runnable_score >= 4 and blocker_severity <= 2 and (has_fix or weights_available):
        return "Level 2"
    if runnable_score >= 2:
        return "Level 3"
    return "Level 4"


def _pick_verdict(tthw: str) -> str:
    if tthw == "Level 1":
        return "GO"
    if tthw in {"Level 2", "Level 3"}:
        return "CAUTION"
    return "NO-GO"


def _what_breaks(issue_signals: list[IssueFixSignal], repo: RepoCandidate) -> str:
    if issue_signals:
        blockers = [signal.blocker.strip() for signal in issue_signals if signal.blocker.strip()]
        return "; ".join(blockers[:2]) if blockers else "Issue tracker has unresolved blockers"
    if len(repo.setup_signals) < 2:
        return "Setup path is incomplete"
    if not repo.entrypoint_signals:
        return "No runnable entrypoint signal found"
    return "No major blocker identified"


def _fix_summary(issue_signals: list[IssueFixSignal]) -> str:
    fixes = [signal.fix.strip() for signal in issue_signals if signal.fix and signal.fix.strip()]
    if not fixes:
        return "No clear fix found"
    return "; ".join(fixes[:2])


def _hf_text(status: HFModelStatus) -> str:
    if status.status == "found":
        return f"Model found ({status.model_id or 'unknown id'})"
    if status.status == "not_found":
        return "No matching model found on Hugging Face"
    return "Hugging Face status unknown"


def _technical_debt(repo: RepoCandidate, issue_signals: list[IssueFixSignal], hf_status: HFModelStatus) -> str:
    debt_items: list[str] = []
    if repo.readiness_label != "strong":
        debt_items.append("readiness not strong")
    if len(repo.setup_signals) < 2:
        debt_items.append("setup guidance is thin")
    if issue_signals:
        debt_items.append("blocking issues remain open")
    if hf_status.status != "found":
        debt_items.append("weights availability unclear")
    return ", ".join(debt_items) if debt_items else "Low unresolved risk"
