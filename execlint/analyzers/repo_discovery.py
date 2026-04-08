from __future__ import annotations

import re

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
OFFICIAL_TERMS = ("official", "official implementation", "authors", "paper")


def discover_repositories(paper: ArxivPaper, github: GitHubClient) -> list[RepoCandidate]:
    queries = _build_queries(paper)

    seen: set[str] = set()
    scored: list[RepoCandidate] = []
    for query in queries:
        for repo in github.search_repositories(
            query=query,
            limit=MAX_REPO_CANDIDATES,
            max_results_inspected=MAX_GITHUB_SEARCH_RESULTS_INSPECTED,
        ):
            if repo.full_name in seen:
                continue
            score, reasons = _score_repository(repo=repo, paper=paper)
            scored.append(repo.model_copy(update={"discovery_score": score, "discovery_reasons": reasons}))
            seen.add(repo.full_name)

    scored.sort(key=lambda repo: (-repo.discovery_score, -repo.stars, repo.full_name.lower()))
    return scored[:MAX_REPO_CANDIDATES]


def _build_queries(paper: ArxivPaper) -> list[str]:
    queries: list[str] = []
    if paper.title:
        title = " ".join(paper.title.split())
        queries.append(f'"{title}"')
        queries.append(f"{title} official implementation")
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
    desc_tokens = _tokens(repo.description or "")
    name_tokens = _tokens(repo.name)
    owner_tokens = _tokens(repo.owner_login or "")

    reasons: list[str] = []
    score = 0.0

    name_overlap = _overlap_ratio(title_tokens, name_tokens)
    if name_overlap > 0:
        points = round(70 * name_overlap, 2)
        score += points
        reasons.append(f"name_overlap={name_overlap:.2f}(+{points})")

    description_overlap = _overlap_ratio(title_tokens, desc_tokens)
    if description_overlap > 0:
        points = round(35 * description_overlap, 2)
        score += points
        reasons.append(f"description_overlap={description_overlap:.2f}(+{points})")

    author_owner = _author_owner_match(paper.authors, owner_tokens)
    if author_owner > 0:
        points = round(30 * author_owner, 2)
        score += points
        reasons.append(f"owner_author_match={author_owner:.2f}(+{points})")

    official_text = f"{repo.name} {repo.description or ''}".lower()
    if any(term in official_text for term in OFFICIAL_TERMS):
        score += 20
        reasons.append("official_wording(+20)")

    star_bonus = min(repo.stars, 4000) / 500
    if star_bonus > 0:
        score += star_bonus
        reasons.append(f"star_bonus(+{star_bonus:.2f})")

    return round(score, 2), reasons


def _author_owner_match(authors: list[str], owner_tokens: set[str]) -> float:
    if not authors or not owner_tokens:
        return 0.0
    author_tokens: set[str] = set()
    for author in authors:
        clean = re.sub(r"[^a-z0-9 ]", " ", author.lower())
        parts = [p for p in clean.split() if len(p) > 2 and p not in STOPWORDS]
        if parts:
            author_tokens.add(parts[-1])
            author_tokens.update(parts)

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
