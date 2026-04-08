from __future__ import annotations

import base64
import os

import httpx

from execlint.config import (
    GITHUB_API_BASE,
    MAX_GITHUB_SEARCH_RESULTS_INSPECTED,
    MAX_ISSUES_PER_REPO,
    REQUEST_TIMEOUT_SECONDS,
)
from execlint.models import RepoCandidate


class GitHubClient:
    def __init__(self, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        headers = {"Accept": "application/vnd.github+json"}
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._timeout = timeout
        self._client = httpx.Client(base_url=GITHUB_API_BASE, headers=headers, timeout=timeout)

    def search_repositories(
        self,
        query: str,
        limit: int = 10,
        max_results_inspected: int = MAX_GITHUB_SEARCH_RESULTS_INSPECTED,
    ) -> list[RepoCandidate]:
        per_page = max(1, min(limit, max_results_inspected))
        response = self._client.get(
            "/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": per_page},
            timeout=self._timeout,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
        results: list[RepoCandidate] = []
        for item in items[:max_results_inspected]:
            owner = item.get("owner") or {}
            results.append(
                RepoCandidate(
                    name=item["name"],
                    full_name=item["full_name"],
                    url=item["html_url"],
                    stars=item.get("stargazers_count", 0),
                    open_issues_count=item.get("open_issues_count", 0),
                    has_readme=False,
                    setup_signals=[],
                    description=item.get("description"),
                    owner_login=owner.get("login"),
                    archived=bool(item.get("archived", False)),
                    pushed_at=item.get("pushed_at"),
                    size_kb=int(item.get("size", 0) or 0),
                    default_branch=item.get("default_branch") or "main",
                )
            )
        return results

    def get_readme(self, full_name: str) -> str | None:
        response = self._client.get(f"/repos/{full_name}/readme", timeout=self._timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        content = data.get("content", "")
        if not content:
            return None
        encoding = data.get("encoding")
        if encoding == "base64":
            try:
                return base64.b64decode(content).decode("utf-8", errors="ignore")
            except Exception:
                return content
        return content

    def get_repo_file_paths(self, full_name: str, default_branch: str = "main") -> list[str]:
        response = self._client.get(
            f"/repos/{full_name}/git/trees/{default_branch}",
            params={"recursive": 1},
            timeout=self._timeout,
        )
        if response.status_code == 404 and default_branch != "master":
            response = self._client.get(
                f"/repos/{full_name}/git/trees/master",
                params={"recursive": 1},
                timeout=self._timeout,
            )
        if response.status_code in (403, 404):
            return []
        response.raise_for_status()
        tree = response.json().get("tree", [])
        return [item.get("path", "") for item in tree if item.get("path")]

    def list_open_issues(self, full_name: str, limit: int = MAX_ISSUES_PER_REPO) -> list[dict]:
        response = self._client.get(
            f"/repos/{full_name}/issues",
            params={"state": "open", "per_page": limit, "sort": "comments", "direction": "desc"},
            timeout=self._timeout,
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        return [issue for issue in response.json() if "pull_request" not in issue]
