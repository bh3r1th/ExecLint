from __future__ import annotations

import re
from datetime import UTC, datetime
from urllib.parse import urlsplit

from execlint.clients.github_client import GitHubClient
from execlint.config import MAX_GITHUB_SEARCH_RESULTS_INSPECTED, MAX_REPO_CANDIDATES
from execlint.models import ArxivPaper, RepoCandidate

STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
    "using",
    "via",
}
OFFICIAL_TERMS = (
    "official implementation",
    "official code",
    "authors",
)
WEAK_TEMPLATE_TERMS = ("template", "boilerplate", "demo only", "starter")


def discover_repositories(paper: ArxivPaper, github: GitHubClient) -> list[RepoCandidate]:
    queries = _build_queries(paper)

    scored_by_repo: dict[str, RepoCandidate] = {}

    direct_candidate = _candidate_from_paper_code_url(paper)
    if direct_candidate is not None:
        scored_by_repo[direct_candidate.full_name] = direct_candidate

    for query in queries:
        for repo in github.search_repositories(
            query=query,
            limit=MAX_REPO_CANDIDATES,
            max_results_inspected=MAX_GITHUB_SEARCH_RESULTS_INSPECTED,
        ):
            score, reasons = _score_repository(repo=repo, paper=paper)
            scored_repo = repo.model_copy(update={"discovery_score": score, "discovery_reasons": reasons})
            existing = scored_by_repo.get(repo.full_name)
            scored_by_repo[repo.full_name] = _merge_candidate(existing, scored_repo)

    scored = list(scored_by_repo.values())
    scored.sort(key=lambda repo: (-repo.discovery_score, -repo.stars, repo.full_name.lower()))
    return scored[:MAX_REPO_CANDIDATES]


def _build_queries(paper: ArxivPaper) -> list[str]:
    queries: list[str] = []
    if paper.title:
        title = " ".join(paper.title.split())
        queries.append(f'"{title}"')
        queries.append(f"{title} official implementation")
        queries.append(f"code for {title}")
        head = " ".join(title.split()[:6])
        if head and head != title:
            queries.append(head)
    if paper.arxiv_id:
        queries.append(paper.arxiv_id)

    deduped: list[str] = []
    for q in queries:
        if q and q not in deduped:
            deduped.append(q)
    return deduped


def _score_repository(repo: RepoCandidate, paper: ArxivPaper) -> tuple[float, list[str]]:
    title_tokens = _tokens(paper.title or "")
    keyword_tokens = _tokens(f"{paper.title or ''} {paper.abstract or ''}")
    desc_tokens = _tokens(repo.description or "")
    name_tokens = _tokens(repo.name)
    owner_tokens = _tokens(repo.owner_login or "")

    reasons: list[str] = []
    score = 0.0

    name_overlap = _overlap_ratio(title_tokens, name_tokens)
    if name_overlap > 0:
        points = round(80 * name_overlap, 2)
        score += points
        reasons.append(f"name_overlap={name_overlap:.2f}(+{points})")

    description_overlap = _overlap_ratio(keyword_tokens, desc_tokens)
    if description_overlap > 0:
        points = round(45 * description_overlap, 2)
        score += points
        reasons.append(f"description_overlap={description_overlap:.2f}(+{points})")

    author_owner = _author_owner_match(paper.authors, owner_tokens)
    if author_owner > 0:
        points = round(35 * author_owner, 2)
        score += points
        reasons.append(f"owner_author_match={author_owner:.2f}(+{points})")

    official_points = _official_wording_points(repo, paper)
    if official_points > 0:
        score += official_points
        reasons.append(f"official_wording(+{official_points})")

    star_bonus = min(repo.stars, 4000) / 500
    if star_bonus > 0:
        score += star_bonus
        reasons.append(f"star_bonus(+{star_bonus:.2f})")

    if repo.archived:
        score -= 25
        reasons.append("archived_penalty(-25)")

    if repo.size_kb <= 15:
        score -= 10
        reasons.append("tiny_surface_penalty(-10)")

    if _is_likely_inactive_fork(repo):
        score -= 16
        reasons.append("inactive_fork_penalty(-16)")

    if name_overlap < 0.2 and _looks_template_only(repo):
        score -= 10
        reasons.append("template_only_penalty(-10)")

    if _matches_paper_code_url(repo, paper.code_url):
        score += 200
        reasons.append("paper_code_url(+200)")

    return round(score, 2), reasons


def _candidate_from_paper_code_url(paper: ArxivPaper) -> RepoCandidate | None:
    if not paper.code_url:
        return None
    parsed = urlsplit(paper.code_url.unicode_string())
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) < 2:
        return None

    owner, name = segments[0], segments[1]
    # Fix C2: reject path-traversal or otherwise unsafe owner/name segments
    _SAFE_SEGMENT = re.compile(r'^[a-zA-Z0-9._-]+$')
    if not _SAFE_SEGMENT.match(owner) or not _SAFE_SEGMENT.match(name):
        return None
    full_name = f"{owner}/{name}"
    return RepoCandidate(
        name=name,
        full_name=full_name,
        url=f"https://github.com/{full_name}",
        owner_login=owner,
        discovery_score=1000.0,
        discovery_reasons=["paper_code_url(+1000)"],
    )


def _merge_candidate(existing: RepoCandidate | None, incoming: RepoCandidate) -> RepoCandidate:
    if existing is None:
        return incoming

    merged_reasons: list[str] = []
    for reason in [*existing.discovery_reasons, *incoming.discovery_reasons]:
        if reason not in merged_reasons:
            merged_reasons.append(reason)

    return incoming.model_copy(
        update={
            "url": existing.url if existing.discovery_score > incoming.discovery_score else incoming.url,
            "stars": max(existing.stars, incoming.stars),
            "open_issues_count": max(existing.open_issues_count, incoming.open_issues_count),
            "description": incoming.description or existing.description,
            "owner_login": incoming.owner_login or existing.owner_login,
            "archived": existing.archived or incoming.archived,
            "pushed_at": incoming.pushed_at or existing.pushed_at,
            "size_kb": max(existing.size_kb, incoming.size_kb),
            "default_branch": incoming.default_branch or existing.default_branch,
            "discovery_score": max(existing.discovery_score, incoming.discovery_score),
            "discovery_reasons": merged_reasons,
        }
    )


def _matches_paper_code_url(repo: RepoCandidate, code_url: object) -> bool:
    if code_url is None:
        return False
    parsed = urlsplit(str(code_url))
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) < 2:
        return False
    return repo.full_name.lower() == f"{segments[0]}/{segments[1]}".lower()


def _official_wording_points(repo: RepoCandidate, paper: ArxivPaper) -> float:
    haystack = f"{repo.name} {repo.description or ''}".lower()
    if any(term in haystack for term in OFFICIAL_TERMS):
        return 18
    title = (paper.title or "").strip().lower()
    if title and f"code for {title}" in haystack:
        return 20
    return 0


def _is_likely_inactive_fork(repo: RepoCandidate) -> bool:
    text = f"{repo.name} {repo.description or ''}".lower()
    mentions_fork = "fork" in text
    if not mentions_fork:
        return False
    if repo.stars > 80:
        return False
    return _is_stale(repo.pushed_at)


def _looks_template_only(repo: RepoCandidate) -> bool:
    text = f"{repo.name} {repo.description or ''}".lower()
    return any(term in text for term in WEAK_TEMPLATE_TERMS)


def _is_stale(pushed_at: str | None) -> bool:
    if not pushed_at:
        return True
    try:
        pushed = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    return (datetime.now(UTC) - pushed).days > 365


def _author_owner_match(authors: list[str], owner_tokens: set[str]) -> float:
    if not authors or not owner_tokens:
        return 0.0
    author_tokens: set[str] = set()
    for author in authors:
        clean = re.sub(r"[^a-z0-9 ]", " ", author.lower())
        parts = [p for p in clean.split() if len(p) > 2 and p not in STOPWORDS]
        if parts:
            author_tokens.add(parts[-1])

    if not author_tokens:
        return 0.0
    overlap = len(author_tokens & owner_tokens)
    return overlap / len(author_tokens)


def _tokens(text: str) -> set[str]:
    clean = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return {token for token in clean.split() if len(token) > 2 and token not in STOPWORDS}


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    return overlap / len(left)
