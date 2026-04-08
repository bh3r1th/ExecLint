from __future__ import annotations

import os

import httpx

from execlint.config import GITHUB_API_BASE, HTTP_TIMEOUT_SECONDS
from execlint.models import RepoCandidate


class GitHubClient:
    def __init__(self, timeout: float = HTTP_TIMEOUT_SECONDS) -> None:
        headers = {"Accept": "application/vnd.github+json"}
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(base_url=GITHUB_API_BASE, headers=headers, timeout=timeout)

    def search_repositories(self, query: str, limit: int = 5) -> list[RepoCandidate]:
        response = self._client.get("/search/repositories", params={"q": query, "sort": "stars", "order": "desc", "per_page": limit})
        response.raise_for_status()
        items = response.json().get("items", [])
        results: list[RepoCandidate] = []
        for item in items:
            results.append(
                RepoCandidate(
                    name=item["name"],
                    full_name=item["full_name"],
                    url=item["html_url"],
                    stars=item.get("stargazers_count", 0),
                    open_issues_count=item.get("open_issues_count", 0),
                    has_readme=False,
                    setup_signals=[],
                )
            )
        return results

    def get_readme(self, full_name: str) -> str | None:
        response = self._client.get(f"/repos/{full_name}/readme")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        return data.get("content", "")

    def list_open_issues(self, full_name: str, limit: int = 10) -> list[dict]:
        response = self._client.get(
            f"/repos/{full_name}/issues",
            params={"state": "open", "per_page": limit},
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        return [issue for issue in response.json() if "pull_request" not in issue]
