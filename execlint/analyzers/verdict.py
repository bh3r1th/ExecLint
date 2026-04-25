from __future__ import annotations

from execlint.models import ExecutionReport, Gap, HFModelStatus, RepoCapability, RepoCandidate

MEANINGFUL_CAPABILITIES = (
    RepoCapability.demo,
    RepoCapability.inference,
    RepoCapability.training,
    RepoCapability.evaluation,
    RepoCapability.smoke_test,
)
EXECUTION_STEP_ORDER = ("install", "setup_data", "setup_weights", "run", "evaluate")

GAP_DEFINITIONS = {
    "install path ambiguous": (
        "install",
        "No setup.py, pyproject.toml, requirements.txt, or Dockerfile in repo root",
    ),
    "dataset must be supplied manually": (
        "data",
        "README mentions dataset/data setup but no adjacent download link or automated bootstrap was found",
    ),
    "weights/checkpoints not linked": (
        "weights",
        "README mentions weights/checkpoints but no download link found within 200 chars",
    ),
    "no clear run command": (
        "run",
        "No regex-extracted run command found in README or repo file paths",
    ),
    "no clear inference/demo command": (
        "run",
        "README/repo signals do not expose a clear inference or demo command",
    ),
    "env version unclear": (
        "env",
        "README mentions Python/CUDA/PyTorch context but no version was pinned",
    ),
    "No repository candidate": (
        "run",
        "No GitHub repository candidate was available to inspect",
    ),
}


def build_execution_report(
    candidates: list[RepoCandidate],
    hf_status: HFModelStatus,
    ref: str | None = None,
    weights_url: str | None = None,
    paper_title: str | None = None,
) -> ExecutionReport:
    del ref
    best_repo = _select_best_repo(candidates)
    title = paper_title or "Unknown paper"
    if best_repo is None:
        return ExecutionReport(
            paper_title=title,
            repo_url="None found",
            gaps=[_gap_from_label("No repository candidate")],
            execution_path="No extracted execution commands",
        )

    return ExecutionReport(
        paper_title=title,
        repo_url=best_repo.url.unicode_string(),
        gaps=_gap_items(best_repo, hf_status=hf_status, weights_url=weights_url),
        execution_path=_execution_path_text(best_repo),
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


def _gap_items(repo: RepoCandidate, hf_status: HFModelStatus, weights_url: str | None = None) -> list[Gap]:
    labels: list[str] = []
    for label in repo.gaps:
        labels.append(_canonical_gap_label(label))

    if _repo_requires_external_weights(repo) and hf_status.status != "found" and not weights_url:
        labels.append("weights/checkpoints not linked")
    if not repo.has_readme and not repo.setup_signals:
        labels.append("install path ambiguous")

    gaps: list[Gap] = []
    for label in labels:
        if label == "None identified":
            continue
        gap = _gap_from_label(label)
        if gap.label not in {existing.label for existing in gaps}:
            gaps.append(gap)
    return gaps


def _canonical_gap_label(label: str) -> str:
    if label == "checkpoint link absent":
        return "weights/checkpoints not linked"
    if label == "environment/cuda/version ambiguity":
        return "env version unclear"
    return label


def _gap_from_label(label: str) -> Gap:
    category, evidence = GAP_DEFINITIONS.get(
        label,
        ("run", f"Execution analysis produced gap label: {label}"),
    )
    return Gap(label=label, category=category, evidence=evidence)


def _repo_requires_external_weights(repo: RepoCandidate) -> bool:
    repo_text = " ".join([*repo.entrypoint_signals, *repo.setup_signals, *repo.gaps]).lower()
    return any(token in repo_text for token in ("weight", "checkpoint", "ckpt"))
