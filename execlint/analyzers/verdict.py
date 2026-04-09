from __future__ import annotations

from execlint.models import ExecutionReport, HFModelStatus, IssueFixSignal, RepoCapability, RepoCandidate

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
MEANINGFUL_CAPABILITIES = (
    RepoCapability.demo,
    RepoCapability.inference,
    RepoCapability.training,
    RepoCapability.evaluation,
    RepoCapability.smoke_test,
)
EXECUTION_STEP_ORDER = ("install", "setup_data", "setup_weights", "run", "evaluate")


def build_execution_report(
    candidates: list[RepoCandidate],
    issue_signals_by_repo: dict[str, list[IssueFixSignal]],
    hf_status: HFModelStatus,
    ref: str | None = None,
    weights_url: str | None = None,
) -> ExecutionReport:
    best_repo = _select_best_repo(candidates, issue_signals_by_repo)
    if best_repo is None:
        return ExecutionReport(
            verdict="NO-GO",
            tthw="Level 4",
            best_repo="None found",
            runnable_for="unclear",
            execution_path="No extracted execution commands",
            gaps="No repository candidate",
            not_clearly_supported="",
            what_breaks="No repository candidate",
            fix="Unavailable: no repository data",
            hf_status=_hf_text(hf_status),
            technical_debt="None identified",
        )

    issue_signals = issue_signals_by_repo.get(best_repo.full_name, [])
    blockers = _repo_blockers(best_repo, issue_signals, hf_status, weights_url=weights_url)
    blocker_severity = max((severity for _, severity in blockers), default=0)
    runnable_score = _runnable_signal_score(best_repo)
    has_fix = _has_credible_fix(issue_signals)

    tthw = _pick_tthw(
        repo=best_repo,
        issue_signals=issue_signals,
        blocker_severity=blocker_severity,
        has_fix=has_fix,
        hf_status=hf_status,
        runnable_score=runnable_score,
        weights_url=weights_url,
    )
    if tthw == "Level 4" and _has_declared_runnable_capability(best_repo):
        tthw = "Level 3"
    verdict = _pick_verdict(tthw)
    verdict, tthw = _apply_caution_overrides(
        repo=best_repo,
        issue_signals=issue_signals,
        blockers=blockers,
        hf_status=hf_status,
        verdict=verdict,
        tthw=tthw,
        weights_url=weights_url,
    )

    return ExecutionReport(
        verdict=verdict,
        tthw=tthw,
        best_repo=best_repo.url.unicode_string(),
        runnable_for=_runnable_for_text(best_repo),
        execution_path=_execution_path_text(best_repo),
        gaps=_gaps_text(best_repo),
        not_clearly_supported=_not_clearly_supported_text(best_repo),
        what_breaks=_what_breaks(blockers),
        fix=_fix_summary(issue_signals, best_repo),
        hf_status=_hf_text(hf_status),
        technical_debt=_technical_debt(best_repo, issue_signals, hf_status, ref=ref, weights_url=weights_url),
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
        if not _has_credible_fix(weak_signals):
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


def _normalized_capabilities(repo: RepoCandidate) -> list[RepoCapability]:
    capabilities: list[RepoCapability] = []
    for capability in repo.inferred_capabilities:
        if isinstance(capability, RepoCapability):
            capabilities.append(capability)
            continue
        try:
            capabilities.append(RepoCapability(capability))
        except ValueError:
            continue
    if not capabilities:
        return [RepoCapability.unclear]
    return capabilities


def _meaningful_capabilities(repo: RepoCandidate) -> list[RepoCapability]:
    return [capability for capability in _normalized_capabilities(repo) if capability in MEANINGFUL_CAPABILITIES]


def _has_declared_runnable_capability(repo: RepoCandidate) -> bool:
    return bool(_meaningful_capabilities(repo))


def _runnable_for_text(repo: RepoCandidate) -> str:
    capabilities = [capability.value for capability in _normalized_capabilities(repo) if capability != RepoCapability.unclear]
    return ", ".join(capabilities) if capabilities else "unclear"


def _not_clearly_supported_text(repo: RepoCandidate) -> str:
    capabilities = set(_normalized_capabilities(repo))
    if capabilities == {RepoCapability.unclear}:
        return "meaningful execution modes"
    if capabilities == {RepoCapability.smoke_test}:
        return "demo, inference, training, evaluation"
    if capabilities & {RepoCapability.training, RepoCapability.evaluation} and not capabilities & {RepoCapability.demo, RepoCapability.inference}:
        return "demo, inference"
    return ""


def _execution_path_text(repo: RepoCandidate) -> str:
    if not repo.execution_steps:
        return "No extracted execution commands"

    segments: list[str] = []
    for step in EXECUTION_STEP_ORDER:
        commands = repo.execution_steps.get(step, [])
        if not commands:
            continue
        segments.append(f"{step}: {' | '.join(commands[:2])}")
    return "; ".join(segments) if segments else "No extracted execution commands"


def _gaps_text(repo: RepoCandidate) -> str:
    if not repo.gaps:
        return "None identified"
    return "; ".join(repo.gaps)


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
    return severity >= 3 or (severity >= 2 and not _has_credible_fix(issue_signals))


def _pick_tthw(
    repo: RepoCandidate,
    issue_signals: list[IssueFixSignal],
    blocker_severity: int,
    has_fix: bool,
    hf_status: HFModelStatus,
    runnable_score: int,
    weights_url: str | None = None,
) -> str:
    meaningful_capabilities = _meaningful_capabilities(repo)
    runtime_capabilities = set(meaningful_capabilities) & {RepoCapability.demo, RepoCapability.inference}
    smoke_test_only = set(_normalized_capabilities(repo)) == {RepoCapability.smoke_test}
    credible_path = _has_credible_runnable_path(repo, hf_status, issue_signals, weights_url=weights_url)
    weights_gap = _repo_requires_external_weights(repo, issue_signals) and hf_status.status != "found" and not weights_url
    step_map = repo.execution_steps or {}
    has_commands = any(step_map.get(step) for step in EXECUTION_STEP_ORDER)
    has_install_step = bool(step_map.get("install"))
    has_run_step = bool(step_map.get("run"))
    manual_ambiguity = any(
        token in gap
        for gap in repo.gaps
        for token in ("manual", "ambiguous", "unclear", "not linked", "no clear")
    )

    if not credible_path:
        return "Level 4"
    if has_install_step and has_run_step:
        if not manual_ambiguity and blocker_severity <= 1 and not weights_gap:
            return "Level 1" if runtime_capabilities else "Level 2"
        return "Level 3"
    if manual_ambiguity:
        return "Level 3"
    if smoke_test_only:
        return "Level 3"
    if (
        runtime_capabilities
        and repo.readiness_label in {"strong", "moderate"}
        and runnable_score >= 4
        and blocker_severity <= 1
        and not weights_gap
        and hf_status.status == "found"
    ):
        return "Level 1"
    if blocker_severity >= 3 and not has_fix:
        return "Level 3"
    if meaningful_capabilities and blocker_severity <= 2:
        if weights_gap or has_fix or repo.readiness_label == "weak":
            return "Level 3"
        return "Level 2"
    if has_fix or runnable_score >= 2:
        return "Level 3"
    return "Level 4"


def _has_credible_runnable_path(
    repo: RepoCandidate,
    hf_status: HFModelStatus,
    issue_signals: list[IssueFixSignal],
    weights_url: str | None = None,
) -> bool:
    capabilities = set(_normalized_capabilities(repo))
    if capabilities == {RepoCapability.unclear} and not repo.entrypoint_signals:
        return False

    install_visible = repo.has_readme or bool(repo.setup_signals)
    has_fix = _has_credible_fix(issue_signals)

    if RepoCapability.smoke_test in capabilities and repo.entrypoint_signals and install_visible:
        return True
    if RepoCapability.training in capabilities and install_visible and any(signal in repo.entrypoint_signals for signal in ("train.py", "scripts/")):
        return True
    if RepoCapability.evaluation in capabilities and install_visible and any(signal in repo.entrypoint_signals for signal in ("scripts/", "notebook")):
        return True
    if capabilities & {RepoCapability.demo, RepoCapability.inference}:
        if repo.entrypoint_signals and install_visible:
            return True

    return bool(repo.entrypoint_signals and install_visible and (has_fix or runnable_score_hint(repo) >= 2))


def runnable_score_hint(repo: RepoCandidate) -> int:
    return min(len(repo.entrypoint_signals), 4) + min(len(repo.setup_signals), 4) + (1 if repo.has_readme else 0)


def _repo_requires_external_weights(repo: RepoCandidate, issue_signals: list[IssueFixSignal]) -> bool:
    repo_text = " ".join([*repo.entrypoint_signals, *repo.setup_signals]).lower()
    if any(token in repo_text for token in ("weight", "checkpoint", "ckpt")):
        return True
    for signal in issue_signals:
        text = f"{signal.blocker_category or ''} {signal.blocker}".lower()
        if "weight" in text or "checkpoint" in text:
            return True
    return False


def _pick_verdict(tthw: str) -> str:
    if tthw == "Level 1":
        return "GO"
    if tthw in {"Level 2", "Level 3"}:
        return "CAUTION"
    return "NO-GO"


def _apply_caution_overrides(
    repo: RepoCandidate,
    issue_signals: list[IssueFixSignal],
    blockers: list[tuple[str, int]],
    hf_status: HFModelStatus,
    verdict: str,
    tthw: str,
    weights_url: str | None = None,
) -> tuple[str, str]:
    # Diagnostic note: this function is a "verdict floor" guard. If a credible runnable
    # path exists, verdict must never remain NO-GO; Level 3 is the floor.
    credible_path = _has_credible_runnable_path(repo, hf_status, issue_signals, weights_url=weights_url)
    return _floor_verdict_for_credible_path(verdict, tthw, credible_path)


def _floor_verdict_for_credible_path(verdict: str, tthw: str, credible_path: bool) -> tuple[str, str]:
    if verdict == "NO-GO" and credible_path:
        return "CAUTION", "Level 3"
    return verdict, tthw


def _what_breaks(
    blockers: list[tuple[str, int]],
) -> str:
    if not blockers:
        return "No concrete blocker visible"
    return "; ".join(message for message, _ in blockers[:2])


def _fix_summary(issue_signals: list[IssueFixSignal], repo: RepoCandidate) -> str:
    fixes = [signal.fix.strip() for signal in issue_signals if signal.fix and signal.fix.strip()]
    if fixes:
        return "; ".join(fixes[:2])
    return "No clear fix found"


def _hf_text(status: HFModelStatus) -> str:
    if status.status == "found":
        if status.notes == "User-provided weights URL":
            return "User-provided weights"
        if status.gated:
            return f"Hugging Face model gated ({status.model_id or 'unknown id'})"
        return f"Hugging Face model found ({status.model_id or 'unknown id'})"
    if status.status == "not_found":
        return "Hugging Face model missing"
    return "Hugging Face status unclear"


def _technical_debt(
    repo: RepoCandidate,
    issue_signals: list[IssueFixSignal],
    hf_status: HFModelStatus,
    ref: str | None = None,
    weights_url: str | None = None,
) -> str:
    debt_items: list[str] = []
    issue_text = " ".join(
        f"{signal.blocker_category or ''} {signal.blocker} {signal.fix or ''}".lower()
        for signal in issue_signals
    )

    if "pin" in issue_text or "version" in issue_text:
        debt_items.append("dependency versions unspecified")
    if "fork" in issue_text:
        debt_items.append("stale fork dependency")
    if not repo.has_readme and not repo.setup_signals:
        debt_items.append("install path ambiguous")
    if _repo_requires_external_weights(repo, issue_signals) and hf_status.status != "found" and not weights_url:
        debt_items.append("checkpoint provenance unclear")
    elif weights_url:
        debt_items.append("user-supplied weights provenance")
    if _dataset_bootstrap_manual(issue_signals):
        debt_items.append("dataset bootstrap manual")
    if not ref and _stale_branch_ambiguity(repo):
        debt_items.append("stale branch ambiguity")

    capability_debt = _capability_debt(repo)
    if capability_debt:
        debt_items.append(capability_debt)

    return ", ".join(debt_items) if debt_items else "None identified"


def _capability_debt(repo: RepoCandidate) -> str | None:
    capabilities = set(_normalized_capabilities(repo))
    if capabilities == {RepoCapability.unclear}:
        return "repo capability unclear"
    if capabilities == {RepoCapability.smoke_test}:
        return "repo appears limited to smoke tests"
    return None


def _repo_blockers(
    repo: RepoCandidate,
    issue_signals: list[IssueFixSignal],
    hf_status: HFModelStatus,
    weights_url: str | None = None,
) -> list[tuple[str, int]]:
    blockers: list[tuple[str, int]] = []
    issue_seen: set[str] = set()
    credible_path = _has_credible_runnable_path(repo, hf_status, issue_signals, weights_url=weights_url)
    meaningful_capabilities = _meaningful_capabilities(repo)

    for signal in issue_signals:
        blocker = (signal.blocker or "").strip()
        if not blocker:
            continue
        if blocker not in issue_seen:
            blockers.append((blocker, _signal_blocker_severity(signal)))
            issue_seen.add(blocker)

    if repo.archived:
        blockers.append(("stale or archived repo", 2))
    elif _repo_stale(repo):
        blockers.append(("stale or archived repo", 1))

    if not repo.has_readme and not repo.setup_signals:
        blockers.append(("no clear install path", 2))

    if not credible_path:
        if meaningful_capabilities:
            runtime_capabilities = {RepoCapability.demo, RepoCapability.inference}
            if set(meaningful_capabilities).isdisjoint(runtime_capabilities):
                blockers.append(("no clear inference/demo entrypoint", 2))
            else:
                blockers.append(("no clear runnable entrypoint", 2))
        else:
            blockers.append(("no obvious runnable entrypoint for any meaningful capability", 3))

    if _repo_requires_external_weights(repo, issue_signals) and hf_status.status != "found" and not weights_url:
        blockers.append(("external checkpoints/weights not linked", 2))

    if _dataset_bootstrap_manual(issue_signals):
        blockers.append(("dataset must be supplied manually", 2))

    if _environment_ambiguity(issue_signals):
        blockers.append(("environment/CUDA/version ambiguity", 2))

    if blockers and not _has_credible_fix(issue_signals) and any(severity >= 3 for _, severity in blockers):
        blockers.append(("fix path unclear", 1))

    compact: list[tuple[str, int]] = []
    seen: set[str] = set()
    for message, severity in blockers:
        if message in seen:
            continue
        compact.append((message, severity))
        seen.add(message)
    return compact


def _repo_stale(repo: RepoCandidate) -> bool:
    return "activity=stale" in (repo.readiness_summary or "").lower()


def _dataset_bootstrap_manual(issue_signals: list[IssueFixSignal]) -> bool:
    text = " ".join(f"{signal.blocker_category or ''} {signal.blocker}".lower() for signal in issue_signals)
    return "dataset" in text and any(token in text for token in ("manual", "supply", "download", "bootstrap"))


def _environment_ambiguity(issue_signals: list[IssueFixSignal]) -> bool:
    text = " ".join(f"{signal.blocker_category or ''} {signal.blocker}".lower() for signal in issue_signals)
    return any(token in text for token in ("cuda", "environment", "version", "dependency")) and any(
        token in text for token in ("ambigu", "mismatch", "incompatible", "unspecified")
    )


def _stale_branch_ambiguity(repo: RepoCandidate) -> bool:
    capabilities = set(_normalized_capabilities(repo))
    return _repo_stale(repo) and bool(capabilities & {RepoCapability.training, RepoCapability.evaluation})


def _weights_are_primary_blocker(issue_signals: list[IssueFixSignal]) -> bool:
    if not issue_signals:
        return False
    for signal in issue_signals:
        text = f"{signal.blocker_category or ''} {signal.blocker}".lower()
        if "weight" not in text and "checkpoint" not in text:
            return False
    return True
