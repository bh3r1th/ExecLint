from __future__ import annotations

from execlint.models import RepoCandidate


def repo_score(candidate: RepoCandidate) -> tuple[float, float, int, int, int, float, float]:
    readiness_points = {"weak": 1.0, "moderate": 2.5, "strong": 4.0}.get(candidate.readiness_label, 1.0)
    runnable_points = min(len(candidate.entrypoint_signals), 4) * 1.4
    dependency_points = min(len(candidate.setup_signals), 4) * 1.2
    readme_points = 0.8 if candidate.has_readme else 0.0

    base = (
        readiness_points
        + runnable_points
        + dependency_points
        + readme_points
        + min(candidate.discovery_score, 120.0) / 60.0
        + min(candidate.stars, 5000) / 2000.0
    )

    if candidate.archived:
        base -= 3.5
    if candidate.surface_file_count <= 3:
        base -= 1.0

    return (
        base,
        readiness_points,
        0 if candidate.archived else 1,
        len(candidate.entrypoint_signals),
        len(candidate.setup_signals),
        candidate.discovery_score,
        min(candidate.stars, 5000),
    )


def pick_best_repo(candidates: list[RepoCandidate]) -> RepoCandidate | None:
    if not candidates:
        return None

    scored = sorted(candidates, key=repo_score, reverse=True)
    best = scored[0]

    strong_or_moderate = [c for c in scored if c.readiness_label in {"strong", "moderate"} and not c.archived]
    if best.readiness_label == "weak" and strong_or_moderate:
        contender = strong_or_moderate[0]
        if repo_score(contender)[0] >= repo_score(best)[0] - 0.5:
            return contender

    return best
