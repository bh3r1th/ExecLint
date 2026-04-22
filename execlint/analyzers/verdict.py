from __future__ import annotations

from execlint.models import ExecutionReport, HFModelStatus, RepoCapability, RepoCandidate

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
    hf_status: HFModelStatus,
    ref: str | None = None,
    weights_url: str | None = None,
) -> ExecutionReport:
    best_repo = _select_best_repo(candidates)
    if best_repo is None:
        return ExecutionReport(
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
    blockers = _repo_blockers(best_repo, evidence_items)

    return ExecutionReport(
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


def _repo_requires_external_weights(repo: RepoCandidate) -> bool:
    repo_text = " ".join([*repo.entrypoint_signals, *repo.setup_signals]).lower()
    return any(token in repo_text for token in ("weight", "checkpoint", "ckpt"))


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
    evidence_items: list[str],
) -> list[tuple[str, int]]:
    blockers: list[tuple[str, int]] = []

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
