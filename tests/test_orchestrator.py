from __future__ import annotations

from execlint.models import ArxivPaper, HFModelStatus, RepoCandidate
from execlint.orchestrator import audit_arxiv_url, audit_arxiv_url_with_debug


def test_orchestrator_continues_when_github_discovery_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "execlint.orchestrator.ArxivClient.fetch_paper",
        lambda self, arxiv_id, url: ArxivPaper(arxiv_id=arxiv_id, url=url, title="Demo Paper"),
    )
    monkeypatch.setattr(
        "execlint.orchestrator.discover_repositories",
        lambda paper, github: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    report, warnings = audit_arxiv_url("https://arxiv.org/abs/1234.5678")

    assert report.verdict == "NO-GO"
    assert report.best_repo == "None found"
    assert report.what_breaks == "Repository discovery unavailable"
    assert report.fix == "Unavailable: GitHub discovery failed"
    assert any("GitHub repository discovery unavailable" in warning for warning in warnings)


def test_orchestrator_skips_hf_when_no_repo_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        "execlint.orchestrator.ArxivClient.fetch_paper",
        lambda self, arxiv_id, url: ArxivPaper(arxiv_id=arxiv_id, url=url, title="Demo Paper"),
    )
    monkeypatch.setattr("execlint.orchestrator.discover_repositories", lambda paper, github: [])
    monkeypatch.setattr("execlint.orchestrator.triage_repositories", lambda candidates, github: ([], None))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("HF lookup should not run when no repo candidates exist")

    monkeypatch.setattr("execlint.orchestrator.check_hf_status", fail_if_called)

    report, _ = audit_arxiv_url("https://arxiv.org/abs/2401.00001")

    assert report.verdict == "NO-GO"
    assert report.hf_status == "Hugging Face status unclear"


def test_orchestrator_returns_unknown_hf_on_hf_failure(monkeypatch) -> None:
    paper = ArxivPaper(arxiv_id="2401.00001", url="https://arxiv.org/abs/2401.00001", title="Demo Paper")
    candidate = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        has_readme=True,
        setup_signals=["requirements.txt", "pip_install"],
        entrypoint_signals=["train.py"],
        readiness_label="strong",
    )

    monkeypatch.setattr("execlint.orchestrator.ArxivClient.fetch_paper", lambda self, arxiv_id, url: paper)
    monkeypatch.setattr("execlint.orchestrator.discover_repositories", lambda paper, github: [candidate])
    monkeypatch.setattr(
        "execlint.orchestrator.triage_repositories",
        lambda candidates, github: ([candidate], candidate),
    )
    monkeypatch.setattr("execlint.orchestrator.mine_issue_signals", lambda repo, github: [])
    monkeypatch.setattr(
        "execlint.orchestrator.check_hf_status",
        lambda paper, hf_client: (_ for _ in ()).throw(RuntimeError("hf down")),
    )

    report, warnings = audit_arxiv_url("https://arxiv.org/abs/2401.00001")

    assert report.verdict in {"GO", "CAUTION", "NO-GO"}
    assert report.hf_status == "Hugging Face status unavailable"
    assert any("Hugging Face lookup unavailable" in warning for warning in warnings)


def test_orchestrator_issue_mining_failure_marks_fix_unavailable(monkeypatch) -> None:
    paper = ArxivPaper(arxiv_id="2401.00001", url="https://arxiv.org/abs/2401.00001", title="Demo Paper")
    candidate = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        has_readme=True,
        setup_signals=["requirements.txt", "pip_install"],
        entrypoint_signals=["train.py"],
        readiness_label="strong",
    )

    monkeypatch.setattr("execlint.orchestrator.ArxivClient.fetch_paper", lambda self, arxiv_id, url: paper)
    monkeypatch.setattr("execlint.orchestrator.discover_repositories", lambda paper, github: [candidate])
    monkeypatch.setattr("execlint.orchestrator.triage_repositories", lambda candidates, github: ([candidate], candidate))
    monkeypatch.setattr(
        "execlint.orchestrator.mine_issue_signals",
        lambda repo, github: (_ for _ in ()).throw(RuntimeError("issues down")),
    )
    monkeypatch.setattr(
        "execlint.orchestrator.check_hf_status",
        lambda paper, hf_client: HFModelStatus(status="found", model_id="org/model", gated=False),
    )

    report, warnings = audit_arxiv_url("https://arxiv.org/abs/2401.00001")

    assert report.fix == "Unavailable: issue mining failed"
    assert any("GitHub issue mining unavailable" in warning for warning in warnings)


def test_debug_payload_remains_compact_on_partial_results(monkeypatch) -> None:
    monkeypatch.setattr(
        "execlint.orchestrator.ArxivClient.fetch_paper",
        lambda self, arxiv_id, url: ArxivPaper(arxiv_id=arxiv_id, url=url, title="Demo Paper"),
    )
    monkeypatch.setattr(
        "execlint.orchestrator.discover_repositories",
        lambda paper, github: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    _, _, debug = audit_arxiv_url_with_debug("https://arxiv.org/abs/1234.5678")

    assert debug["candidate_count"] == 0
    assert debug["selected_repo_name"] == "none"
    assert debug["selected_repo_readiness"] == "n/a"
    assert debug["selected_repo_blocker_severity"] == "n/a"
    assert debug["selected_repo_fix_signal_count"] == 0
    assert debug["hf_summary"] in {"unclear", "missing", "found", "gated"}


def test_orchestrator_arxiv_parse_failure_is_value_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "execlint.orchestrator.ArxivClient.fetch_paper",
        lambda self, arxiv_id, url: (_ for _ in ()).throw(ValueError("bad arxiv")),
    )

    try:
        audit_arxiv_url("https://arxiv.org/abs/0000.00000")
    except ValueError as exc:
        assert "bad arxiv" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for invalid arXiv parse")
