from __future__ import annotations

from collections import Counter
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

from execlint.analyzers.hf_status import check_hf_status
from execlint.analyzers.issue_miner import mine_issue_signals
from execlint.analyzers.repo_discovery import discover_repositories
from execlint.analyzers.repo_triage import triage_repositories
from execlint.analyzers.verdict import build_execution_report
from execlint.clients.arxiv_client import ArxivClient, normalize_arxiv_input
from execlint.clients.github_client import GitHubClient
from execlint.clients.hf_client import HFClient
from execlint.config import SOFT_EXECUTION_BUDGET_SECONDS
from execlint.models import ArxivPaper, ExecutionInput, ExecutionReport, HFModelStatus, IssueFixSignal, RepoCandidate


PARTIAL_FAILURE_WARNINGS = {
    "github_discovery": "GitHub repository discovery unavailable; continuing with no candidates",
    "issue_mining": "GitHub issue mining unavailable; continuing without issue-derived fixes",
    "hf_unavailable": "Hugging Face lookup unavailable; status marked unknown",
    "budget": "Soft execution budget exceeded; returning partial results",
}

SEVERITY_LABELS = {0: "n/a", 1: "low", 2: "medium", 3: "high"}


def audit_arxiv_url(arxiv_url: str) -> tuple[ExecutionReport, list[str]]:
    report, warnings, _ = audit_arxiv_url_with_debug(arxiv_url)
    return report, warnings


def audit_arxiv_url_with_debug(arxiv_url: str) -> tuple[ExecutionReport, list[str], dict[str, Any]]:
    return _audit_with_debug(arxiv_url=arxiv_url, execution_input=None)


def audit_execution_input(execution_input: ExecutionInput) -> tuple[ExecutionReport, list[str]]:
    report, warnings, _ = audit_execution_input_with_debug(execution_input)
    return report, warnings


def audit_execution_input_with_debug(execution_input: ExecutionInput) -> tuple[ExecutionReport, list[str], dict[str, Any]]:
    return _audit_with_debug(arxiv_url=execution_input.arxiv_url.unicode_string(), execution_input=execution_input)


def _audit_with_debug(
    arxiv_url: str,
    execution_input: ExecutionInput | None,
) -> tuple[ExecutionReport, list[str], dict[str, Any]]:
    original_arxiv_input = arxiv_url
    arxiv_id, normalized_url = normalize_arxiv_input(arxiv_url)

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
        details = [f"original_input={original_arxiv_input!r}", f"normalized_arxiv_id={arxiv_id!r}"]
        if normalized_url:
            details.append(f"failing_url={normalized_url!r}")
        raise ValueError(f"Could not resolve arXiv metadata ({'; '.join(details)})") from exc

    discovered: list[RepoCandidate] = []
    candidates: list[RepoCandidate]
    if _budget_exceeded(start):
        warnings.append(PARTIAL_FAILURE_WARNINGS["budget"])
        source_failures.append("budget")
        candidates = []
    elif execution_input is not None:
        try:
            explicit_candidate = _repo_candidate_from_execution_input(execution_input)
            discovered = [explicit_candidate]
            candidates, _ = triage_repositories(candidates=discovered, github=github_client, ref=execution_input.ref)
        except ValueError:
            raise
        except Exception:
            warnings.append(PARTIAL_FAILURE_WARNINGS["github_discovery"])
            source_failures.append("github_discovery")
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
        if "github_discovery" in source_failures:
            report.what_breaks = "Repository discovery unavailable"
            report.fix = "Unavailable: GitHub discovery failed"
            report.technical_debt = "Unknown due to unavailable repository data"
        return report, warnings, _debug_payload(
            paper=paper,
            discovered=discovered,
            candidates=[],
            issue_signals_by_repo={},
            hf_status=hf_status,
            source_failures=source_failures,
            execution_input=execution_input,
            report=report,
        )

    issue_signals_by_repo: dict[str, list[IssueFixSignal]] = {}
    _apply_execution_path_layer(candidates)
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
            hf_status = check_hf_status(
                paper=paper,
                hf_client=hf_client,
                weights_url=execution_input.weights_url.unicode_string() if execution_input and execution_input.weights_url else None,
            )
        except Exception:
            warnings.append(PARTIAL_FAILURE_WARNINGS["hf_unavailable"])
            source_failures.append("hf_unavailable")
            hf_status = HFModelStatus(status="unknown", notes="Hugging Face lookup unavailable")

    report = build_execution_report(
        candidates=candidates,
        issue_signals_by_repo=issue_signals_by_repo,
        hf_status=hf_status,
        ref=execution_input.ref if execution_input else None,
        weights_url=execution_input.weights_url.unicode_string() if execution_input and execution_input.weights_url else None,
    )
    _apply_partial_result_wording(report=report, source_failures=source_failures, issue_signals_by_repo=issue_signals_by_repo)
    if execution_input is not None:
        report.best_repo = execution_input.repo_url.unicode_string()
    return report, warnings, _debug_payload(
        paper=paper,
        discovered=discovered,
        candidates=candidates,
        issue_signals_by_repo=issue_signals_by_repo,
        hf_status=hf_status,
        source_failures=source_failures,
        selected_repo_url=report.best_repo,
        execution_input=execution_input,
        report=report,
    )



def _signal_severity(signal: IssueFixSignal) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(signal.confidence, 1)


def _has_credible_fix(signals: list[IssueFixSignal]) -> bool:
    return any(bool(signal.fix and signal.fix.strip()) for signal in signals)


def _high_signal_issue_count(signals: list[IssueFixSignal]) -> int:
    return sum(1 for signal in signals if _signal_severity(signal) >= 2)

def _debug_payload(
    paper: ArxivPaper,
    discovered: list[RepoCandidate],
    candidates: list[RepoCandidate],
    issue_signals_by_repo: dict[str, list[IssueFixSignal]],
    hf_status: HFModelStatus,
    source_failures: list[str],
    selected_repo_url: str | None = None,
    execution_input: ExecutionInput | None = None,
    report: ExecutionReport | None = None,
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
    blocker_severity = max((_signal_severity(signal) for signal in selected_signals), default=0)
    if blocker_severity == 0 and report and report.what_breaks and report.what_breaks != "No concrete blocker visible":
        blocker_severity = _report_breaker_severity(report.what_breaks)

    return {
        "paper_title": paper.title,
        "paper_authors": list(paper.authors),
        "paper_abstract": paper.abstract,
        "paper_code_url": paper.code_url.unicode_string() if paper.code_url else None,
        "paper_code_url_source": paper.code_url_source,
        "ref": execution_input.ref if execution_input else None,
        "weights_source": _weights_source(hf_status, execution_input),
        "inferred_capabilities": [getattr(capability, "value", capability) for capability in (selected_repo.inferred_capabilities if selected_repo else [])],
        "execution_steps": selected_repo.execution_steps if selected_repo else {},
        "missing_prerequisites": selected_repo.missing_prerequisites if selected_repo else [],
        "execution_gaps": selected_repo.gaps if selected_repo else [],
        "discovered_repo_count": len(discovered),
        "candidate_count": len(candidates),
        "selected_repo_name": selected_repo.full_name if selected_repo else "none",
        "selected_repo_readiness": selected_repo.readiness_label if selected_repo else "n/a",
        "selected_repo_fix_signal_count": sum(1 for signal in selected_signals if signal.fix and signal.fix.strip()),
        "selected_repo_has_fix_path": _has_credible_fix(selected_signals),
        "selected_repo_blocker_severity": SEVERITY_LABELS[blocker_severity],
        "top_blocker_categories": [name for name, _ in category_counter.most_common(3)],
        "hf_summary": _hf_debug_summary(hf_status),
        "partial_source_failures": list(dict.fromkeys(source_failures)),
}


def _apply_execution_path_layer(candidates: list[RepoCandidate]) -> None:
    for index, candidate in enumerate(candidates):
        candidates[index] = candidate.model_copy(
            update={
                "execution_steps": candidate.execution_steps,
                "missing_prerequisites": candidate.missing_prerequisites,
                "gaps": candidate.gaps,
            }
        )


def _budget_exceeded(started_at: float) -> bool:
    return (monotonic() - started_at) >= SOFT_EXECUTION_BUDGET_SECONDS


def _hf_debug_summary(hf_status: HFModelStatus) -> str:
    if hf_status.status == "found":
        return "gated" if hf_status.gated else "found"
    if hf_status.status == "not_found":
        return "missing"
    return "unclear"


def _apply_partial_result_wording(
    report: ExecutionReport,
    source_failures: list[str],
    issue_signals_by_repo: dict[str, list[IssueFixSignal]],
) -> None:
    failures = set(source_failures)
    if "hf_unavailable" in failures:
        report.hf_status = "Hugging Face status unavailable"

    if "issue_mining" in failures:
        has_any_fix = any(signal.fix and signal.fix.strip() for signals in issue_signals_by_repo.values() for signal in signals)
        if not has_any_fix:
            report.fix = "Unavailable: issue mining failed"


def _repo_candidate_from_execution_input(execution_input: ExecutionInput) -> RepoCandidate:
    parsed = urlsplit(execution_input.repo_url.unicode_string())
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise ValueError("repo_url must be a GitHub repository URL")

    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) < 2:
        raise ValueError("repo_url must point to a GitHub repository")

    owner, name = segments[0], segments[1]
    full_name = f"{owner}/{name}"
    return RepoCandidate(
        name=name,
        full_name=full_name,
        url=execution_input.repo_url.unicode_string(),
        owner_login=owner,
        default_branch=execution_input.ref or "main",
        discovery_score=1000.0,
        discovery_reasons=["user_repo_url(+1000)"],
    )


def _weights_source(hf_status: HFModelStatus, execution_input: ExecutionInput | None) -> str:
    if execution_input and execution_input.weights_url:
        return "provided"
    if hf_status.status == "found":
        return "discovered"
    return "none"


def _report_breaker_severity(what_breaks: str) -> int:
    text = what_breaks.lower()
    if any(token in text for token in ("no obvious runnable entrypoint", "no repository candidate")):
        return 3
    if any(
        token in text
        for token in (
            "external checkpoints/weights not linked",
            "no clear install path",
            "dataset must be supplied manually",
            "environment/cuda/version ambiguity",
            "no clear runnable entrypoint",
            "no clear inference/demo entrypoint",
        )
    ):
        return 2
    if any(token in text for token in ("stale or archived repo", "fix path unclear", "repository discovery unavailable")):
        return 1
    return 0
