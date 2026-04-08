from __future__ import annotations

from time import monotonic

from execlint.analyzers.hf_status import check_hf_status
from execlint.analyzers.issue_miner import mine_issue_signals
from execlint.analyzers.repo_discovery import discover_repositories
from execlint.analyzers.repo_triage import triage_repositories
from execlint.analyzers.verdict import build_execution_report
from execlint.clients.arxiv_client import ArxivClient
from execlint.clients.github_client import GitHubClient
from execlint.clients.hf_client import HFClient
from execlint.config import SOFT_EXECUTION_BUDGET_SECONDS
from execlint.models import ExecutionReport, HFModelStatus, IssueFixSignal, RepoCandidate
from execlint.utils.text import extract_arxiv_id, normalize_arxiv_url


PARTIAL_FAILURE_WARNINGS = {
    "github_discovery": "GitHub repository discovery unavailable; continuing with no candidates",
    "issue_mining": "GitHub issue mining unavailable; continuing without issue-derived fixes",
    "hf_unavailable": "Hugging Face lookup unavailable; status marked unknown",
    "budget": "Soft execution budget exceeded; returning partial results",
}


def audit_arxiv_url(arxiv_url: str) -> tuple[ExecutionReport, list[str]]:
    normalized_url = normalize_arxiv_url(arxiv_url)
    arxiv_id = extract_arxiv_id(normalized_url)

    arxiv_client = ArxivClient()
    github_client = GitHubClient()
    hf_client = HFClient()

    start = monotonic()
    warnings: list[str] = []

    try:
        paper = arxiv_client.fetch_paper(arxiv_id=arxiv_id, url=normalized_url)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not resolve arXiv metadata for {arxiv_id}") from exc

    candidates: list[RepoCandidate]
    if _budget_exceeded(start):
        warnings.append(PARTIAL_FAILURE_WARNINGS["budget"])
        candidates = []
    else:
        try:
            discovered = discover_repositories(paper=paper, github=github_client)
            candidates, _ = triage_repositories(candidates=discovered, github=github_client)
        except Exception:
            warnings.append(PARTIAL_FAILURE_WARNINGS["github_discovery"])
            candidates = []

    if not candidates:
        report = build_execution_report(
            candidates=[],
            issue_signals_by_repo={},
            hf_status=HFModelStatus(status="unknown", notes="Skipped due to missing repository candidates"),
        )
        return report, warnings

    issue_signals_by_repo: dict[str, list[IssueFixSignal]] = {}
    issue_mining_failed = False
    for repo in candidates:
        if _budget_exceeded(start):
            warnings.append(PARTIAL_FAILURE_WARNINGS["budget"])
            break
        try:
            issue_signals_by_repo[repo.full_name] = mine_issue_signals(repo, github_client)
        except Exception:
            issue_mining_failed = True
            issue_signals_by_repo[repo.full_name] = []

    if issue_mining_failed:
        warnings.append(PARTIAL_FAILURE_WARNINGS["issue_mining"])

    hf_status: HFModelStatus
    if _budget_exceeded(start):
        warnings.append(PARTIAL_FAILURE_WARNINGS["budget"])
        hf_status = HFModelStatus(status="unknown", notes="Skipped due to soft execution budget")
    else:
        try:
            hf_status = check_hf_status(paper=paper, hf_client=hf_client)
        except Exception:
            warnings.append(PARTIAL_FAILURE_WARNINGS["hf_unavailable"])
            hf_status = HFModelStatus(status="unknown", notes="Hugging Face lookup unavailable")

    return (
        build_execution_report(
            candidates=candidates,
            issue_signals_by_repo=issue_signals_by_repo,
            hf_status=hf_status,
        ),
        warnings,
    )


def _budget_exceeded(started_at: float) -> bool:
    return (monotonic() - started_at) >= SOFT_EXECUTION_BUDGET_SECONDS
