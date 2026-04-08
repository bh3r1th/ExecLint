from __future__ import annotations

from datetime import UTC, datetime

from execlint.clients.github_client import GitHubClient
from execlint.models import RepoCandidate
from execlint.utils.ranking import pick_best_repo

SETUP_FILES = ("requirements.txt", "pyproject.toml", "environment.yml", "setup.py")
ENTRYPOINT_FILES = ("train.py", "infer.py", "inference.py", "demo.py", "app.py")


def triage_repositories(candidates: list[RepoCandidate], github: GitHubClient) -> tuple[list[RepoCandidate], RepoCandidate | None]:
    triaged: list[RepoCandidate] = []
    for candidate in candidates:
        readme = github.get_readme(candidate.full_name)
        paths = github.get_repo_file_paths(candidate.full_name, default_branch=candidate.default_branch)
        setup_signals = _extract_setup_signals(readme or "", paths)
        entrypoints = _extract_entrypoints(paths)
        activity_score = _activity_score(candidate.pushed_at)
        readiness_score, readiness_label = _compute_readiness(
            has_readme=bool(readme),
            setup_count=len(setup_signals),
            entrypoint_count=len(entrypoints),
            activity_score=activity_score,
            open_issues_count=candidate.open_issues_count,
            archived=candidate.archived,
            surface_file_count=len(paths),
            likely_inactive_fork=_is_likely_inactive_fork(candidate, activity_score),
        )
        summary = _build_summary(
            has_readme=bool(readme),
            setup_signals=setup_signals,
            entrypoints=entrypoints,
            archived=candidate.archived,
            surface_file_count=len(paths),
            activity=_activity_bucket(candidate.pushed_at),
            issues=candidate.open_issues_count,
            label=readiness_label,
        )
        triaged.append(
            candidate.model_copy(
                update={
                    "has_readme": bool(readme),
                    "setup_signals": setup_signals,
                    "entrypoint_signals": entrypoints,
                    "surface_file_count": len(paths),
                    "readiness_score": readiness_score,
                    "readiness_label": readiness_label,
                    "readiness_summary": summary,
                }
            )
        )
    return triaged, pick_best_repo(triaged)


def _extract_setup_signals(readme_text: str, paths: list[str]) -> list[str]:
    lower_text = readme_text.lower()
    setup_hits = [filename for filename in SETUP_FILES if filename in {p.lower().split("/")[-1] for p in paths}]
    if "requirements" in lower_text and "requirements.txt" not in setup_hits:
        setup_hits.append("requirements")
    if "pip install" in lower_text and "pip_install" not in setup_hits:
        setup_hits.append("pip_install")
    return setup_hits


def _extract_entrypoints(paths: list[str]) -> list[str]:
    lowered = [p.lower() for p in paths]
    hits = [p for p in ENTRYPOINT_FILES if any(path.endswith(p) for path in lowered)]
    if any(path.endswith(".ipynb") for path in lowered):
        hits.append("notebook")
    if any(path.startswith("scripts/") for path in lowered):
        hits.append("scripts/")
    return sorted(set(hits))


def _activity_score(pushed_at: str | None) -> float:
    if not pushed_at:
        return 0.0
    try:
        pushed = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    age_days = (datetime.now(UTC) - pushed).days
    if age_days <= 90:
        return 1.0
    if age_days <= 365:
        return 0.6
    if age_days <= 730:
        return 0.3
    return 0.0


def _activity_bucket(pushed_at: str | None) -> str:
    score = _activity_score(pushed_at)
    if score >= 1.0:
        return "recent"
    if score >= 0.6:
        return "active"
    if score > 0:
        return "stale"
    return "unknown"


def _is_likely_inactive_fork(candidate: RepoCandidate, activity_score: float) -> bool:
    text = f"{candidate.name} {candidate.description or ''}".lower()
    return "fork" in text and activity_score <= 0.3


def _compute_readiness(
    has_readme: bool,
    setup_count: int,
    entrypoint_count: int,
    activity_score: float,
    open_issues_count: int,
    archived: bool,
    surface_file_count: int,
    likely_inactive_fork: bool = False,
) -> tuple[float, str]:
    score = 0.0
    score += 2.0 if has_readme else 0.0
    score += min(setup_count, 4) * 1.6
    score += min(entrypoint_count, 4) * 1.3
    score += activity_score * 2.0
    if archived:
        score -= 3.0
    if likely_inactive_fork:
        score -= 2.0
    if not has_readme and setup_count == 0:
        score -= 1.8
    if surface_file_count <= 3:
        score -= 1.5
    elif surface_file_count > 800:
        score -= 1.0
    elif surface_file_count > 300:
        score -= 0.4
    if open_issues_count > 150:
        score -= 1.5
    elif open_issues_count > 50:
        score -= 0.7

    if score >= 6.5:
        label = "strong"
    elif score >= 3.5:
        label = "moderate"
    else:
        label = "weak"
    return round(score, 2), label


def _build_summary(
    has_readme: bool,
    setup_signals: list[str],
    entrypoints: list[str],
    archived: bool,
    surface_file_count: int,
    activity: str,
    issues: int,
    label: str,
) -> str:
    return (
        f"readme={'yes' if has_readme else 'no'}; "
        f"setup={len(setup_signals)}; entrypoints={len(entrypoints)}; "
        f"activity={activity}; open_issues={issues}; archived={'yes' if archived else 'no'}; "
        f"surface_files={surface_file_count}; readiness={label}"
    )
