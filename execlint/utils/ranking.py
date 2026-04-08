from __future__ import annotations

from execlint.models import RepoCandidate


def repo_score(candidate: RepoCandidate) -> float:
    setup_bonus = min(len(candidate.setup_signals), 5) * 3
    readme_bonus = 10 if candidate.has_readme else 0
    star_points = min(candidate.stars, 5000) / 100
    issue_penalty = min(candidate.open_issues_count, 200) / 50
    return setup_bonus + readme_bonus + star_points - issue_penalty


def pick_best_repo(candidates: list[RepoCandidate]) -> RepoCandidate | None:
    if not candidates:
        return None
    return sorted(candidates, key=repo_score, reverse=True)[0]
