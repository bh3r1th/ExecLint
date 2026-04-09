from __future__ import annotations

from datetime import UTC, datetime

from execlint.clients.github_client import GitHubClient
from execlint.models import RepoCapability, RepoCandidate

SETUP_FILES = ("requirements.txt", "pyproject.toml", "environment.yml", "setup.py")
ENTRYPOINT_FILES = ("train.py", "infer.py", "inference.py", "demo.py", "app.py", "smoke_test.py")
DEMO_TOKENS = ("demo.py", "app.py", "gradio", "streamlit")
INFERENCE_PATH_TOKENS = ("infer.py", "inference.py", "predict.py", "generate.py", "generation.py")
INFERENCE_SERVER_TOKENS = ("serve.py", "server.py", "api.py")
INFERENCE_README_TOKENS = ("run inference", "inference usage", "predict", "prediction", "text generation", "generate text")
TRAINING_TOKENS = ("train.py", "trainer", "training", "checkpoint", "finetune")
EVALUATION_TOKENS = ("eval.py", "evaluate.py", "benchmark", "metrics")
SMOKE_TEST_TOKENS = ("smoke_test.py", "sanity", "toy", "quickstart")


def triage_repositories(
    candidates: list[RepoCandidate],
    github: GitHubClient,
    ref: str | None = None,
) -> tuple[list[RepoCandidate], RepoCandidate | None]:
    triaged: list[RepoCandidate] = []
    for candidate in candidates:
        readme = github.get_readme(candidate.full_name)
        paths = github.get_repo_file_paths(candidate.full_name, default_branch=ref or candidate.default_branch)
        setup_signals = _extract_setup_signals(readme or "", paths)
        entrypoints = _extract_entrypoints(paths)
        inferred_capabilities = _infer_capabilities(readme or "", paths)
        activity_score = _activity_score(candidate.pushed_at)
        readiness_score, readiness_label = _compute_readiness(
            has_readme=bool(readme),
            setup_count=len(setup_signals),
            entrypoint_count=len(entrypoints),
            capability_count=len([capability for capability in inferred_capabilities if capability != RepoCapability.unclear]),
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
                    "default_branch": ref or candidate.default_branch,
                    "surface_file_count": len(paths),
                    "readiness_score": readiness_score,
                    "readiness_label": readiness_label,
                    "readiness_summary": summary,
                    "inferred_capabilities": inferred_capabilities,
                }
            )
        )
    return triaged, _pick_best_triaged_repo(triaged)


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


def _infer_capabilities(readme_text: str, paths: list[str]) -> list[RepoCapability]:
    lowered_paths = [path.lower() for path in paths]
    readme_lower = readme_text.lower()
    capabilities: list[RepoCapability] = []

    if _contains_any(lowered_paths + [readme_lower], DEMO_TOKENS):
        capabilities.append(RepoCapability.demo)
    if _has_strong_inference_signal(lowered_paths, readme_lower):
        capabilities.append(RepoCapability.inference)
    if _contains_any(lowered_paths + [readme_lower], TRAINING_TOKENS):
        capabilities.append(RepoCapability.training)
    if _contains_any(lowered_paths + [readme_lower], EVALUATION_TOKENS):
        capabilities.append(RepoCapability.evaluation)
    if _contains_any(lowered_paths + [readme_lower], SMOKE_TEST_TOKENS):
        capabilities.append(RepoCapability.smoke_test)
    if not capabilities:
        capabilities.append(RepoCapability.unclear)

    return capabilities


def _contains_any(haystacks: list[str], needles: tuple[str, ...]) -> bool:
    return any(needle in haystack for haystack in haystacks for needle in needles)


def _has_strong_inference_signal(paths: list[str], readme_text: str) -> bool:
    if _contains_any(paths, INFERENCE_PATH_TOKENS):
        return True
    if _contains_any(paths, INFERENCE_SERVER_TOKENS) and _contains_any([readme_text], INFERENCE_README_TOKENS):
        return True
    return _contains_any([readme_text], ("run inference", "inference usage", "predict with", "usage: inference"))


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
    capability_count: int,
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
    if entrypoint_count == 0 and capability_count == 0:
        score = min(score, 2.9)
    elif entrypoint_count == 0:
        score = min(score, 3.4)
    elif capability_count > 0 and (has_readme or setup_count > 0):
        score = max(score, 3.5)

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


def _pick_best_triaged_repo(candidates: list[RepoCandidate]) -> RepoCandidate | None:
    if not candidates:
        return None

    scored = sorted(
        candidates,
        key=lambda repo: (
            0 if repo.archived else 1,
            _readiness_rank(repo.readiness_label),
            repo.readiness_score,
            min(len(repo.entrypoint_signals), 4),
            min(len(repo.setup_signals), 4),
            repo.discovery_score,
            repo.full_name.lower(),
        ),
        reverse=True,
    )

    best = scored[0]
    stronger = [repo for repo in scored if repo.readiness_label in {"strong", "moderate"} and not repo.archived]
    if best.readiness_label == "weak" and stronger:
        return stronger[0]
    return best


def _readiness_rank(label: str) -> int:
    if label == "strong":
        return 3
    if label == "moderate":
        return 2
    return 1
