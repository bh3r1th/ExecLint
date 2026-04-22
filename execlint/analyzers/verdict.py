from __future__ import annotations

import re

from execlint.models import ExecutionReport, HFModelStatus, RepoCapability, RepoCandidate

MEANINGFUL_CAPABILITIES = (
    RepoCapability.demo,
    RepoCapability.inference,
    RepoCapability.training,
    RepoCapability.evaluation,
    RepoCapability.smoke_test,
)
EXECUTION_STEP_ORDER = ("install", "setup_data", "setup_weights", "run", "evaluate")
_HEAVY_SETUP_PATTERN = re.compile(
    r"\b(?:docker|download|database)\b|\b\d*\s*[tg]b\b", re.IGNORECASE
)


def build_execution_report(
    candidates: list[RepoCandidate],
    hf_status: HFModelStatus,
    ref: str | None = None,
    weights_url: str | None = None,
) -> ExecutionReport:
    best_repo = _select_best_repo(candidates)
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

    evidence_items = _evidence_items(best_repo, hf_status, weights_url=weights_url)
    blockers = _repo_blockers(best_repo, hf_status, evidence_items, weights_url=weights_url)
    runnable_score = _runnable_signal_score(best_repo)

    tthw = _pick_tthw(
        repo=best_repo,
        hf_status=hf_status,
        runnable_score=runnable_score,
        weights_url=weights_url,
    )
    if tthw == "Level 4" and _has_declared_runnable_capability(best_repo) and not _requires_heavy_setup(best_repo):
        tthw = "Level 3"
    verdict = _pick_verdict(tthw)
    verdict, tthw = _apply_caution_overrides(
        repo=best_repo,
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
        what_breaks=_what_breaks(blockers, evidence_items),
        fix="No clear fix found",
        hf_status=_hf_text(hf_status),
        technical_debt=_technical_debt(evidence_items, ref=ref, repo=best_repo),
    )


def _select_best_repo(candidates: list[RepoCandidate]) -> RepoCandidate | None:
    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    scored = sorted(
        candidates,
        key=lambda repo: (
            0 if repo.archived else 1,
            _readiness_points(repo.readiness_label),
            _runnable_signal_score(repo),
            repo.discovery_score,
            repo.full_name.lower(),
        ),
        reverse=True,
    )
    return scored[0]


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


def _pick_tthw(
    repo: RepoCandidate,
    hf_status: HFModelStatus,
    runnable_score: int,
    weights_url: str | None = None,
) -> str:
    if _requires_heavy_setup(repo):
        return "Level 4"

    meaningful_capabilities = _meaningful_capabilities(repo)
    runtime_capabilities = set(meaningful_capabilities) & {RepoCapability.demo, RepoCapability.inference}
    smoke_test_only = set(_normalized_capabilities(repo)) == {RepoCapability.smoke_test}
    credible_path = _has_credible_runnable_path(repo, hf_status, weights_url=weights_url)
    weights_gap = _repo_requires_external_weights(repo) and hf_status.status != "found" and not weights_url
    step_map = repo.execution_steps or {}
    has_install_step = bool(step_map.get("install"))
    has_run_step = bool(step_map.get("run"))
    has_evaluate_step = bool(step_map.get("evaluate"))
    has_credible_sequence = _has_credible_execution_sequence(repo)
    manual_ambiguity = any(
        token in gap
        for gap in repo.gaps
        for token in ("manual", "ambiguous", "unclear", "not linked", "no clear")
    )

    if not credible_path:
        return "Level 4"
    if has_credible_sequence and has_install_step and has_run_step:
        if not manual_ambiguity and not weights_gap:
            return "Level 1"
        return "Level 2"
    if has_install_step and (has_run_step or has_evaluate_step):
        return "Level 2"
    if has_evaluate_step and not has_run_step:
        return "Level 3"
    if manual_ambiguity:
        return "Level 3"
    if smoke_test_only:
        return "Level 3"
    if (
        runtime_capabilities
        and repo.readiness_label in {"strong", "moderate"}
        and runnable_score >= 4
        and not weights_gap
        and hf_status.status == "found"
    ):
        return "Level 1"
    if meaningful_capabilities:
        if weights_gap or repo.readiness_label == "weak":
            return "Level 3"
        return "Level 2"
    if runnable_score >= 2:
        return "Level 3"
    return "Level 4"


def _requires_heavy_setup(repo: RepoCandidate) -> bool:
    setup_and_entrypoints = " ".join([*repo.setup_signals, *repo.entrypoint_signals]).lower()
    execution_text = " ".join(
        command.lower()
        for commands in (repo.execution_steps or {}).values()
        for command in commands
    )
    gap_text = " ".join(gap.lower() for gap in repo.gaps)
    combined = f"{setup_and_entrypoints} {execution_text} {gap_text}"

    if _HEAVY_SETUP_PATTERN.search(combined):
        return True
    if "dataset" in combined and any(token in combined for token in ("script", "setup", "bootstrap", "prepare")):
        return True
    return False


def _has_credible_runnable_path(
    repo: RepoCandidate,
    hf_status: HFModelStatus,
    weights_url: str | None = None,
) -> bool:
    capabilities = set(_normalized_capabilities(repo))
    if capabilities == {RepoCapability.unclear} and not repo.entrypoint_signals:
        return False

    install_visible = repo.has_readme or bool(repo.setup_signals)

    if RepoCapability.smoke_test in capabilities and repo.entrypoint_signals and install_visible:
        return True
    if RepoCapability.training in capabilities and install_visible and any(signal in repo.entrypoint_signals for signal in ("train.py", "scripts/")):
        return True
    if RepoCapability.evaluation in capabilities and install_visible and any(signal in repo.entrypoint_signals for signal in ("scripts/", "notebook")):
        return True
    if capabilities & {RepoCapability.demo, RepoCapability.inference}:
        if repo.entrypoint_signals and install_visible:
            return True

    return bool(repo.entrypoint_signals and install_visible and runnable_score_hint(repo) >= 2)


def runnable_score_hint(repo: RepoCandidate) -> int:
    return min(len(repo.entrypoint_signals), 4) + min(len(repo.setup_signals), 4) + (1 if repo.has_readme else 0)


def _repo_requires_external_weights(repo: RepoCandidate) -> bool:
    repo_text = " ".join([*repo.entrypoint_signals, *repo.setup_signals]).lower()
    return any(token in repo_text for token in ("weight", "checkpoint", "ckpt"))


def _pick_verdict(tthw: str) -> str:
    if tthw == "Level 1":
        return "GO"
    if tthw in {"Level 2", "Level 3"}:
        return "CAUTION"
    return "NO-GO"


def _apply_caution_overrides(
    repo: RepoCandidate,
    hf_status: HFModelStatus,
    verdict: str,
    tthw: str,
    weights_url: str | None = None,
) -> tuple[str, str]:
    credible_path = _has_credible_runnable_path(repo, hf_status, weights_url=weights_url)
    return _floor_verdict_for_credible_path(verdict, tthw, credible_path, _requires_heavy_setup(repo))


def _floor_verdict_for_credible_path(verdict: str, tthw: str, credible_path: bool, heavy_setup: bool = False) -> tuple[str, str]:
    if verdict == "NO-GO" and credible_path and not heavy_setup:
        return "CAUTION", "Level 3"
    return verdict, tthw


def _what_breaks(
    blockers: list[tuple[str, int]],
    evidence_items: list[str],
) -> str:
    concrete = [message for message, _ in blockers if message]
    if concrete:
        return "; ".join(concrete[:2])
    if evidence_items:
        return "; ".join(evidence_items[:2])
    return "No concrete blocker visible"


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
    evidence_items: list[str],
    repo: RepoCandidate,
    ref: str | None = None,
) -> str:
    debt_map = {
        "dataset must be supplied manually": "dataset bootstrap manual",
        "checkpoint link absent": "checkpoint provenance unclear",
        "install path ambiguous": "install path ambiguous",
        "no clear inference/demo command": "runtime entrypoint unclear",
        "no clear run command": "runtime entrypoint unclear",
        "environment/cuda/version ambiguity": "environment pinning unresolved",
    }
    debt_items: list[str] = []
    for item in evidence_items:
        mapped = debt_map.get(item)
        if mapped and mapped not in debt_items:
            debt_items.append(mapped)
    if not ref and _stale_branch_ambiguity(repo):
        debt_items.append("stale branch ambiguity")
    return ", ".join(debt_items) if debt_items else "None identified"


def _repo_blockers(
    repo: RepoCandidate,
    hf_status: HFModelStatus,
    evidence_items: list[str],
    weights_url: str | None = None,
) -> list[tuple[str, int]]:
    blockers: list[tuple[str, int]] = []
    credible_path = _has_credible_runnable_path(repo, hf_status, weights_url=weights_url)
    meaningful_capabilities = _meaningful_capabilities(repo)

    if repo.archived:
        blockers.append(("stale or archived repo", 2))
    elif _repo_stale(repo):
        blockers.append(("stale or archived repo", 1))

    severity_by_item = {
        "dataset must be supplied manually": 2,
        "checkpoint link absent": 2,
        "install path ambiguous": 2,
        "no clear inference/demo command": 2,
        "no clear run command": 2,
        "environment/cuda/version ambiguity": 2,
    }
    for item in evidence_items:
        blockers.append((item, severity_by_item.get(item, 1)))

    if not credible_path:
        if meaningful_capabilities:
            runtime_capabilities = {RepoCapability.demo, RepoCapability.inference}
            if set(meaningful_capabilities).isdisjoint(runtime_capabilities):
                blockers.append(("no clear inference/demo entrypoint", 2))
            else:
                blockers.append(("no clear runnable entrypoint", 2))
        else:
            blockers.append(("no obvious runnable entrypoint for any meaningful capability", 3))

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


def _stale_branch_ambiguity(repo: RepoCandidate) -> bool:
    capabilities = set(_normalized_capabilities(repo))
    return _repo_stale(repo) and bool(capabilities & {RepoCapability.training, RepoCapability.evaluation})


def _has_credible_execution_sequence(repo: RepoCandidate) -> bool:
    steps = repo.execution_steps or {}
    has_install = bool(steps.get("install"))
    has_run = bool(steps.get("run"))
    has_evaluate = bool(steps.get("evaluate"))
    return has_install and (has_run or has_evaluate)


def _evidence_items(
    repo: RepoCandidate,
    hf_status: HFModelStatus,
    weights_url: str | None = None,
) -> list[str]:
    items: list[str] = []
    gap_text = " ".join(gap.lower() for gap in repo.gaps)

    if "dataset" in gap_text and any(token in gap_text for token in ("manual", "supply", "download", "bootstrap")):
        items.append("dataset must be supplied manually")
    if _repo_requires_external_weights(repo) and hf_status.status != "found" and not weights_url:
        items.append("checkpoint link absent")
    if (not repo.has_readme and not repo.setup_signals) or any(
        token in gap_text for token in ("install path ambiguous", "install ambiguous", "no clear install")
    ):
        items.append("install path ambiguous")
    if any(token in gap_text for token in ("no clear run command", "run command missing")):
        items.append("no clear run command")
    if any(token in gap_text for token in ("no clear inference", "no clear demo", "inference/demo")):
        items.append("no clear inference/demo command")

    compact: list[str] = []
    for item in items:
        if item not in compact:
            compact.append(item)
    return compact
