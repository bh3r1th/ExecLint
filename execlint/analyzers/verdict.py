from __future__ import annotations

from execlint.models import ExecutionReport, HFModelStatus, IssueFixSignal, RepoCandidate

BLOCKER_SEVERITY_SCORE = {"low": 1, "medium": 2, "high": 3}
CATEGORY_SEVERITY_HINTS: dict[str, str] = {
    "dependency": "low",
    "install": "low",
    "environment": "medium",
    "api": "medium",
    "asset": "high",
    "weights": "high",
    "checkpoint": "high",
}


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
    has_fix = _has_credible_fix(issue_signals)

    tthw = _pick_tthw(
        repo=best_repo,
        issue_signals=issue_signals,
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

    if len(candidates) == 1:
        return candidates[0]

    scored = sorted(
        [_repo_selection_key(repo=repo, index=index, issue_signals=issue_signals_by_repo.get(repo.full_name, [])) for index, repo in enumerate(candidates)],
        key=lambda item: item[0],
        reverse=True,
    )

    strongest_available = [
        repo
        for _, repo in scored
        if repo.readiness_label in {"strong", "moderate"} and not repo.archived
    ]
    best = scored[0][1]

    if strongest_available:
        stronger_blocked = all(
            _is_clearly_blocked(issue_signals_by_repo.get(repo.full_name, []))
            for repo in strongest_available
        )
        if stronger_blocked:
            weak_with_fix = [
                repo
                for _, repo in scored
                if repo.readiness_label == "weak" and _has_credible_fix(issue_signals_by_repo.get(repo.full_name, []))
            ]
            if weak_with_fix:
                return weak_with_fix[0]

    if best.readiness_label == "weak" and strongest_available:
        weak_signals = issue_signals_by_repo.get(best.full_name, [])
        weak_has_fix = _has_credible_fix(weak_signals)
        if not weak_has_fix:
            return strongest_available[0]

        stronger_blocked = all(
            _is_clearly_blocked(issue_signals_by_repo.get(repo.full_name, []))
            for repo in strongest_available
        )
        if not stronger_blocked:
            return strongest_available[0]

    return best


def _repo_selection_key(
    repo: RepoCandidate,
    index: int,
    issue_signals: list[IssueFixSignal],
) -> tuple[tuple[int, int, int, int, int, int, int, float, str], RepoCandidate]:
    blocker_severity = _worst_blocker_severity(issue_signals)
    has_fix = _has_credible_fix(issue_signals)
    high_signal_issue_count = _high_signal_issue_count(issue_signals)
    blocker_category_rank = _worst_blocker_category_rank(issue_signals)

    score = (
        0 if repo.archived else 1,
        _readiness_points(repo.readiness_label),
        1 if has_fix else 0,
        -blocker_severity,
        -blocker_category_rank,
        high_signal_issue_count,
        _runnable_signal_score(repo),
        _discovery_rank_points(index) + repo.discovery_score,
        repo.full_name.lower(),
    )
    return score, repo


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
    score += min(len(repo.setup_signals), 4)
    score += min(len(repo.entrypoint_signals), 4)
    if repo.has_readme:
        score += 1
    return score


def _signal_blocker_severity(signal: IssueFixSignal) -> int:
    explicit = BLOCKER_SEVERITY_SCORE.get(signal.confidence, 1)
    text = f"{signal.blocker_category or ''} {signal.blocker}".lower()

    inferred = 0
    for hint, label in CATEGORY_SEVERITY_HINTS.items():
        if hint in text:
            inferred = max(inferred, BLOCKER_SEVERITY_SCORE[label])

    if any(token in text for token in ("cuda", "environment", "api")) and any(
        token in text for token in ("mismatch", "drift", "break", "broken", "incompatible")
    ):
        inferred = max(inferred, BLOCKER_SEVERITY_SCORE["medium"])

    if any(phrase in text for phrase in ("missing", "broken", "no runnable", "cannot run", "not runnable")):
        inferred = max(inferred, BLOCKER_SEVERITY_SCORE["high"])

    if inferred >= 3 and not _is_fix_text_credible(signal.fix):
        return 3
    return max(explicit, inferred)


def _worst_blocker_severity(issue_signals: list[IssueFixSignal]) -> int:
    if not issue_signals:
        return 0
    return max(_signal_blocker_severity(signal) for signal in issue_signals)


def _worst_blocker_category_rank(issue_signals: list[IssueFixSignal]) -> int:
    category_rank = {"dependency": 1, "install": 1, "environment": 2, "api": 2, "cuda": 2, "asset": 3, "weights": 3, "checkpoint": 3}
    if not issue_signals:
        return 0
    worst = 0
    for signal in issue_signals:
        label = (signal.blocker_category or "").strip().lower()
        if not label:
            continue
        for category, rank in category_rank.items():
            if category in label:
                worst = max(worst, rank)
    return worst


def _high_signal_issue_count(issue_signals: list[IssueFixSignal]) -> int:
    return sum(1 for signal in issue_signals if _signal_blocker_severity(signal) >= 2)


def _is_fix_text_credible(fix: str | None) -> bool:
    if not fix:
        return False
    text = fix.strip().lower()
    if not text:
        return False
    return any(token in text for token in ("pin", "install", "download", "set", "use", "upgrade", "fallback", "workaround", "provide"))


def _has_credible_fix(issue_signals: list[IssueFixSignal]) -> bool:
    return any(_is_fix_text_credible(signal.fix) for signal in issue_signals)


def _is_clearly_blocked(issue_signals: list[IssueFixSignal]) -> bool:
    severity = _worst_blocker_severity(issue_signals)
    has_fix = _has_credible_fix(issue_signals)
    return severity >= 3 or (severity >= 2 and not has_fix)


def _pick_tthw(
    repo: RepoCandidate,
    issue_signals: list[IssueFixSignal],
    blocker_severity: int,
    has_fix: bool,
    hf_status: HFModelStatus,
    runnable_score: int,
) -> str:
    obvious_runnable_path = (
        repo.has_readme and len(repo.setup_signals) >= 2 and len(repo.entrypoint_signals) >= 1 and runnable_score >= 4
    )
    plausible_runnable_path = _has_plausible_runnable_path(repo, runnable_score)
    weights_available = hf_status.status == "found"
    weights_unclear = hf_status.status in {"unknown", "not_found"}
    blocker_high_no_fix = blocker_severity >= 3 and not has_fix
    infra_friction = _has_infra_friction(issue_signals)

    if blocker_high_no_fix:
        return "Level 4"
    if not plausible_runnable_path and weights_unclear and not has_fix:
        return "Level 4"
    if repo.readiness_label == "weak" and blocker_severity >= 2 and not has_fix:
        return "Level 4"

    if obvious_runnable_path and weights_available and blocker_severity <= 1 and repo.readiness_label in {"strong", "moderate"}:
        return "Level 1"
    if repo.readiness_label in {"strong", "moderate"} and plausible_runnable_path and blocker_severity <= 2:
        if weights_unclear and blocker_severity >= 1:
            return "Level 3"
        if weights_available or has_fix or blocker_severity <= 1:
            return "Level 2"
    if plausible_runnable_path and (has_fix or blocker_severity <= 2):
        if infra_friction or weights_unclear or repo.readiness_label == "weak":
            return "Level 3"
    if plausible_runnable_path and blocker_severity <= 2:
        return "Level 2"
    if runnable_score >= 2 or has_fix:
        return "Level 3"
    return "Level 4"


def _has_plausible_runnable_path(repo: RepoCandidate, runnable_score: int) -> bool:
    if not repo.entrypoint_signals:
        return False
    if runnable_score >= 4:
        return True
    return repo.has_readme and len(repo.setup_signals) >= 1 and runnable_score >= 3


def _has_infra_friction(issue_signals: list[IssueFixSignal]) -> bool:
    for signal in issue_signals:
        text = f"{signal.blocker_category or ''} {signal.blocker}".lower()
        if _signal_blocker_severity(signal) >= 2 and any(token in text for token in ("cuda", "version", "fork")):
            return True
    return False


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
    issue_text = " ".join(
        f"{signal.blocker_category or ''} {signal.blocker} {signal.fix or ''}".lower()
        for signal in issue_signals
    )
    if "pin" in issue_text or "version" in issue_text:
        debt_items.append("unresolved version pinning")
    if hf_status.status != "found":
        debt_items.append("unclear weight provenance")
    if "fork" in issue_text:
        debt_items.append("stale fork dependency")
    if len(repo.setup_signals) < 2 or not repo.has_readme:
        debt_items.append("missing install instructions")
    return ", ".join(debt_items) if debt_items else "Low unresolved risk"
