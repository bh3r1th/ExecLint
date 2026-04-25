from __future__ import annotations

import base64
import os

import httpx

from execlint.config import (
    GITHUB_API_BASE,
    MAX_GITHUB_SEARCH_RESULTS_INSPECTED,
    REQUEST_TIMEOUT_SECONDS,
)
from execlint.models import RepoCandidate


# Fix H4: prevent GITHUB_TOKEN from leaking in httpx exception messages
def _safe_raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        sanitized_msg = f"GitHub API error: {response.status_code} for {response.request.method} {response.request.url}"
        raise httpx.HTTPStatusError(sanitized_msg, request=response.request, response=response) from None


class GitHubClient:
    def __init__(self, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        headers = {"Accept": "application/vnd.github+json, application/vnd.github.squirrel-girl-preview+json"}
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._timeout = timeout
        self._client = httpx.Client(base_url=GITHUB_API_BASE, headers=headers, timeout=timeout, follow_redirects=True)

    # Fix C1: close underlying httpx.Client to prevent resource/FD leaks
    def close(self) -> None:
        self._client.close()

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
        _safe_raise_for_status(response)
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
        _safe_raise_for_status(response)
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
        _safe_raise_for_status(response)
        tree = response.json().get("tree", [])
        return [item.get("path", "") for item in tree if item.get("path")]
