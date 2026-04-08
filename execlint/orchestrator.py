from __future__ import annotations

from execlint.analyzers.hf_status import check_hf_status
from execlint.analyzers.issue_miner import mine_issue_signals
from execlint.analyzers.repo_discovery import discover_repositories
from execlint.analyzers.repo_triage import triage_repositories
from execlint.analyzers.verdict import build_execution_report
from execlint.clients.arxiv_client import ArxivClient
from execlint.clients.github_client import GitHubClient
from execlint.clients.hf_client import HFClient
from execlint.models import ExecutionReport
from execlint.utils.text import extract_arxiv_id, normalize_arxiv_url


def audit_arxiv_url(arxiv_url: str) -> ExecutionReport:
    normalized_url = normalize_arxiv_url(arxiv_url)
    arxiv_id = extract_arxiv_id(normalized_url)

    arxiv_client = ArxivClient()
    github_client = GitHubClient()
    hf_client = HFClient()

    paper = arxiv_client.fetch_paper(arxiv_id=arxiv_id, url=normalized_url)
    candidates = discover_repositories(paper=paper, github=github_client)
    _, best_repo = triage_repositories(candidates=candidates, github=github_client)

    issue_signals = mine_issue_signals(best_repo, github_client) if best_repo else []
    hf_status = check_hf_status(paper=paper, hf_client=hf_client)

    return build_execution_report(best_repo=best_repo, issue_signals=issue_signals, hf_status=hf_status)
