from __future__ import annotations

from execlint.clients.github_client import GitHubClient
from execlint.models import IssueFixSignal, RepoCandidate

BLOCKER_TERMS = ("error", "broken", "fail", "cannot run", "crash")
FIX_TERMS = ("workaround", "fix", "resolved", "patch")


def mine_issue_signals(repo: RepoCandidate, github: GitHubClient) -> list[IssueFixSignal]:
    issues = github.list_open_issues(repo.full_name, limit=10)
    signals: list[IssueFixSignal] = []
    for issue in issues:
        title = (issue.get("title") or "").strip()
        body = (issue.get("body") or "").strip()
        text = f"{title} {body}".lower()
        if not any(term in text for term in BLOCKER_TERMS):
            continue

        fix = _extract_fix_text(body)
        confidence = "high" if fix else "medium"
        signals.append(IssueFixSignal(blocker=title or "Potential blocker issue", fix=fix, confidence=confidence))
    return signals


def _extract_fix_text(body: str) -> str | None:
    lower_body = body.lower()
    if any(term in lower_body for term in FIX_TERMS):
        compact = " ".join(body.split())
        return compact[:200] if compact else None
    return None
