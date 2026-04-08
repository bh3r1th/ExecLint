from __future__ import annotations

from execlint.clients.github_client import GitHubClient
from execlint.models import RepoCandidate
from execlint.utils.ranking import pick_best_repo

SETUP_KEYWORDS = ("install", "requirements", "docker", "quickstart", "usage")


def triage_repositories(candidates: list[RepoCandidate], github: GitHubClient) -> tuple[list[RepoCandidate], RepoCandidate | None]:
    triaged: list[RepoCandidate] = []
    for candidate in candidates:
        readme = github.get_readme(candidate.full_name)
        setup_signals = _extract_setup_signals(readme or "")
        triaged.append(
            candidate.model_copy(
                update={
                    "has_readme": bool(readme),
                    "setup_signals": setup_signals,
                }
            )
        )
    return triaged, pick_best_repo(triaged)


def _extract_setup_signals(readme_text: str) -> list[str]:
    lower_text = readme_text.lower()
    return [keyword for keyword in SETUP_KEYWORDS if keyword in lower_text]
