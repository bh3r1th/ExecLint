from __future__ import annotations

from execlint.analyzers.hf_status import check_hf_status
from execlint.analyzers.issue_miner import mine_issue_signals
from execlint.analyzers.repo_discovery import discover_repositories
from execlint.analyzers.repo_triage import triage_repositories
from execlint.analyzers.verdict import build_execution_report
from execlint.clients.arxiv_client import ArxivClient
from execlint.clients.github_client import GitHubClient
from execlint.clients.hf_client import HFClient
from execlint.models import ExecutionReport, HFModelStatus, IssueFixSignal, RepoCandidate
from execlint.utils.text import extract_arxiv_id, normalize_arxiv_url


def audit_arxiv_url(arxiv_url: str) -> ExecutionReport:
    normalized_url = normalize_arxiv_url(arxiv_url)
    arxiv_id = extract_arxiv_id(normalized_url)

    arxiv_client = ArxivClient()
    github_client = GitHubClient()
    hf_client = HFClient()

    paper = arxiv_client.fetch_paper(arxiv_id=arxiv_id, url=normalized_url)

    candidates: list[RepoCandidate]
    try:
        discovered = discover_repositories(paper=paper, github=github_client)
        candidates, _ = triage_repositories(candidates=discovered, github=github_client)
    except Exception:
        candidates = []

    issue_signals_by_repo: dict[str, list[IssueFixSignal]] = {}
    for repo in candidates:
        try:
            issue_signals_by_repo[repo.full_name] = mine_issue_signals(repo, github_client)
        except Exception:
            issue_signals_by_repo[repo.full_name] = []

    try:
        hf_status = check_hf_status(paper=paper, hf_client=hf_client)
    except Exception:
        hf_status = HFModelStatus(status="unknown", notes="HF analyzer failed")

    return build_execution_report(
        candidates=candidates,
        issue_signals_by_repo=issue_signals_by_repo,
        hf_status=hf_status,
    )
