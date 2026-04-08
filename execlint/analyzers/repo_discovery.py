from __future__ import annotations

from execlint.clients.github_client import GitHubClient
from execlint.models import ArxivPaper, RepoCandidate


def discover_repositories(paper: ArxivPaper, github: GitHubClient) -> list[RepoCandidate]:
    queries = [paper.arxiv_id]
    if paper.title:
        queries.append(paper.title)

    seen: set[str] = set()
    candidates: list[RepoCandidate] = []
    for query in queries:
        for repo in github.search_repositories(query=query, limit=5):
            if repo.full_name in seen:
                continue
            candidates.append(repo)
            seen.add(repo.full_name)
    return candidates
