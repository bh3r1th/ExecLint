from __future__ import annotations

from execlint.clients.github_client import GitHubClient
from execlint.config import MAX_ISSUES_PER_REPO
from execlint.models import IssueFixSignal, RepoCandidate

KEYWORDS = (
    "install",
    "dependency",
    "error",
    "broken",
    "cuda",
    "doesn't work",
    "issue",
    "fix",
    "requirements",
)
CATEGORY_BY_KEYWORD = {
    "install": "installation",
    "requirements": "installation",
    "dependency": "dependency",
    "cuda": "hardware",
    "error": "runtime",
    "broken": "runtime",
    "doesn't work": "runtime",
    "fix": "maintainer_fix",
    "issue": "general",
}


def mine_issue_signals(repo: RepoCandidate, github: GitHubClient) -> list[IssueFixSignal]:
    issues = github.list_open_issues(repo.full_name, limit=MAX_ISSUES_PER_REPO)
    ranked: list[tuple[int, IssueFixSignal]] = []

    for issue in issues:
        title = (issue.get("title") or "").strip()
        body = (issue.get("body") or "").strip()
        text = f"{title} {body}".lower()
        matched = [kw for kw in KEYWORDS if kw in text]
        if not matched:
            continue

        probable_fix = _extract_fix_text(body)
        category = _pick_category(matched)
        signal_score = len(set(matched)) + (2 if probable_fix else 0)
        confidence = "high" if signal_score >= 4 else "medium"

        ranked.append(
            (
                signal_score,
                IssueFixSignal(
                    blocker=title or "Potential execution blocker",
                    issue_number=issue.get("number"),
                    blocker_category=category,
                    fix=probable_fix,
                    confidence=confidence,
                ),
            )
        )

    ranked.sort(key=lambda item: (-item[0], -(item[1].issue_number or 0), item[1].blocker.lower()))
    return [signal for _, signal in ranked[:5]]


def _pick_category(matched_keywords: list[str]) -> str:
    for keyword in ("install", "requirements", "dependency", "cuda", "error", "broken", "doesn't work"):
        if keyword in matched_keywords:
            return CATEGORY_BY_KEYWORD[keyword]
    return "general"


def _extract_fix_text(body: str) -> str | None:
    if not body:
        return None
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    hints = ("fix", "workaround", "pin", "install", "requirements", "downgrade", "upgrade")
    for line in lines:
        lower = line.lower()
        if any(hint in lower for hint in hints):
            return " ".join(line.split())[:180]
    compact = " ".join(body.split())
    return compact[:140] if compact else None
