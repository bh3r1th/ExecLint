from __future__ import annotations

from collections import Counter
from time import monotonic
from typing import Any

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
    report, warnings, _ = audit_arxiv_url_with_debug(arxiv_url)
    return report, warnings


def audit_arxiv_url_with_debug(arxiv_url: str) -> tuple[ExecutionReport, list[str], dict[str, Any]]:
    normalized_url = normalize_arxiv_url(arxiv_url)
    arxiv_id = extract_arxiv_id(normalized_url)

    arxiv_client = ArxivClient()
    github_client = GitHubClient()
    hf_client = HFClient()

    start = monotonic()
    warnings: list[str] = []
    source_failures: list[str] = []

    try:
        paper = arxiv_client.fetch_paper(arxiv_id=arxiv_id, url=normalized_url)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not resolve arXiv metadata for {arxiv_id}") from exc

    discovered: list[RepoCandidate] = []
    candidates: list[RepoCandidate]
    if _budget_exceeded(start):
        warnings.append(PARTIAL_FAILURE_WARNINGS["budget"])
        source_failures.append("budget")
        candidates = []
    else:
        try:
            discovered = discover_repositories(paper=paper, github=github_client)
            candidates, _ = triage_repositories(candidates=discovered, github=github_client)
        except Exception:
            warnings.append(PARTIAL_FAILURE_WARNINGS["github_discovery"])
            source_failures.append("github_discovery")
            candidates = []

    if not candidates:
        hf_status = HFModelStatus(status="unknown", notes="Skipped due to missing repository candidates")
        report = build_execution_report(candidates=[], issue_signals_by_repo={}, hf_status=hf_status)
        return report, warnings, _debug_payload(
            discovered=discovered,
            candidates=[],
            issue_signals_by_repo={},
            hf_status=hf_status,
            source_failures=source_failures,
        )

    issue_signals_by_repo: dict[str, list[IssueFixSignal]] = {}
    issue_mining_failed = False
    for repo in candidates:
        if _budget_exceeded(start):
            warnings.append(PARTIAL_FAILURE_WARNINGS["budget"])
            source_failures.append("budget")
            break
        try:
            issue_signals_by_repo[repo.full_name] = mine_issue_signals(repo, github_client)
        except Exception:
            issue_mining_failed = True
            issue_signals_by_repo[repo.full_name] = []

    if issue_mining_failed:
        warnings.append(PARTIAL_FAILURE_WARNINGS["issue_mining"])
        source_failures.append("issue_mining")

    hf_status: HFModelStatus
    if _budget_exceeded(start):
        warnings.append(PARTIAL_FAILURE_WARNINGS["budget"])
        source_failures.append("budget")
        hf_status = HFModelStatus(status="unknown", notes="Skipped due to soft execution budget")
    else:
        try:
            hf_status = check_hf_status(paper=paper, hf_client=hf_client)
        except Exception:
            warnings.append(PARTIAL_FAILURE_WARNINGS["hf_unavailable"])
            source_failures.append("hf_unavailable")
            hf_status = HFModelStatus(status="unknown", notes="Hugging Face lookup unavailable")

    report = build_execution_report(
        candidates=candidates,
        issue_signals_by_repo=issue_signals_by_repo,
        hf_status=hf_status,
    )
    return report, warnings, _debug_payload(
        discovered=discovered,
        candidates=candidates,
        issue_signals_by_repo=issue_signals_by_repo,
        hf_status=hf_status,
        source_failures=source_failures,
        selected_repo_url=report.best_repo,
    )



def _signal_severity(signal: IssueFixSignal) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(signal.confidence, 1)


def _has_credible_fix(signals: list[IssueFixSignal]) -> bool:
    return any(bool(signal.fix and signal.fix.strip()) for signal in signals)


def _high_signal_issue_count(signals: list[IssueFixSignal]) -> int:
    return sum(1 for signal in signals if _signal_severity(signal) >= 2)

def _debug_payload(
    discovered: list[RepoCandidate],
    candidates: list[RepoCandidate],
    issue_signals_by_repo: dict[str, list[IssueFixSignal]],
    hf_status: HFModelStatus,
    source_failures: list[str],
    selected_repo_url: str | None = None,
) -> dict[str, Any]:
    selected_repo = None
    if selected_repo_url:
        selected_repo = next((repo for repo in candidates if repo.url.unicode_string() == selected_repo_url), None)

    category_counter: Counter[str] = Counter()
    for signals in issue_signals_by_repo.values():
        for signal in signals:
            if signal.blocker_category and signal.blocker_category.strip():
                category_counter[signal.blocker_category.strip()] += 1
            elif signal.blocker and signal.blocker.strip():
                category_counter["uncategorized"] += 1

    selected_signals = issue_signals_by_repo.get(selected_repo.full_name, []) if selected_repo else []

    return {
        "discovered_repo_count": len(discovered),
        "selected_repo_readiness": selected_repo.readiness_label if selected_repo else "n/a",
        "selected_repo_high_signal_issue_count": _high_signal_issue_count(selected_signals),
        "selected_repo_has_fix_path": _has_credible_fix(selected_signals),
        "selected_repo_blocker_severity": max((_signal_severity(signal) for signal in selected_signals), default=0),
        "top_blocker_categories": [name for name, _ in category_counter.most_common(3)],
        "hf_weights_found": hf_status.status == "found",
        "partial_source_failures": list(dict.fromkeys(source_failures)),
    }


def _budget_exceeded(started_at: float) -> bool:
    return (monotonic() - started_at) >= SOFT_EXECUTION_BUDGET_SECONDS
