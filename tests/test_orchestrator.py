from __future__ import annotations

from execlint.models import ArxivPaper, ExecutionInput, HFModelStatus, RepoCandidate
from execlint.orchestrator import audit_arxiv_url, audit_arxiv_url_with_debug, audit_execution_input_with_debug


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
        lambda candidates, github, ref=None: ([candidate], candidate),
    )
    monkeypatch.setattr(
        "execlint.orchestrator.check_hf_status",
        lambda paper, hf_client, weights_url=None: (_ for _ in ()).throw(RuntimeError("hf down")),
    )

    report, warnings = audit_arxiv_url("https://arxiv.org/abs/2401.00001")

    assert report.verdict in {"GO", "CAUTION", "NO-GO"}
    assert report.hf_status == "Hugging Face status unavailable"
    assert any("Hugging Face lookup unavailable" in warning for warning in warnings)


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
    assert debug["paper_title"] == "Demo Paper"
    assert debug["paper_code_url"] is None
    assert debug["paper_code_url_source"] == "none"
    assert debug["selected_repo_name"] == "none"
    assert debug["selected_repo_readiness"] == "n/a"
    assert debug["selected_repo_blocker_severity"] == "low"
    assert debug["hf_summary"] in {"unclear", "missing", "found", "gated"}


def test_debug_payload_sets_blocker_severity_when_report_has_blockers(monkeypatch) -> None:
    paper = ArxivPaper(arxiv_id="2401.00001", url="https://arxiv.org/abs/2401.00001", title="Demo Paper")
    candidate = RepoCandidate(
        name="research",
        full_name="org/research",
        url="https://github.com/org/research",
        has_readme=False,
        setup_signals=[],
        entrypoint_signals=["train.py"],
        readiness_label="moderate",
        inferred_capabilities=["training"],
    )

    monkeypatch.setattr("execlint.orchestrator.ArxivClient.fetch_paper", lambda self, arxiv_id, url: paper)
    monkeypatch.setattr("execlint.orchestrator.discover_repositories", lambda paper, github: [candidate])
    monkeypatch.setattr(
        "execlint.orchestrator.triage_repositories",
        lambda candidates, github, ref=None: ([candidate], candidate),
    )
    monkeypatch.setattr(
        "execlint.orchestrator.check_hf_status",
        lambda paper, hf_client, weights_url=None: HFModelStatus(status="unknown"),
    )

    _, _, debug = audit_arxiv_url_with_debug("https://arxiv.org/abs/2401.00001")

    assert debug["selected_repo_blocker_severity"] == "medium"


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


def test_orchestrator_bypasses_repo_discovery_when_repo_url_provided(monkeypatch) -> None:
    paper = ArxivPaper(arxiv_id="2401.00001", url="https://arxiv.org/abs/2401.00001", title="Demo Paper")
    candidate = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        has_readme=True,
        setup_signals=["requirements.txt", "pip_install"],
        entrypoint_signals=["train.py"],
        readiness_label="strong",
        inferred_capabilities=["training"],
    )

    monkeypatch.setattr("execlint.orchestrator.ArxivClient.fetch_paper", lambda self, arxiv_id, url: paper)
    monkeypatch.setattr(
        "execlint.orchestrator.discover_repositories",
        lambda paper, github: (_ for _ in ()).throw(AssertionError("repo discovery should be bypassed")),
    )
    monkeypatch.setattr(
        "execlint.orchestrator.triage_repositories",
        lambda candidates, github, ref=None: ([candidate], candidate),
    )
    monkeypatch.setattr(
        "execlint.orchestrator.check_hf_status",
        lambda paper, hf_client, weights_url=None: HFModelStatus(status="unknown"),
    )

    report, _, debug = audit_execution_input_with_debug(
        ExecutionInput(
            arxiv_url="https://arxiv.org/abs/2401.00001",
            repo_url="https://github.com/org/demo",
            ref="main",
        )
    )

    assert report.best_repo == "https://github.com/org/demo"
    assert debug["candidate_count"] == 1
    assert debug["selected_repo_name"] == "org/demo"
    assert debug["ref"] == "main"
    assert debug["weights_source"] == "none"
    assert debug["inferred_capabilities"] == ["training"]


def test_orchestrator_uses_provided_weights_and_preserves_repo_url(monkeypatch) -> None:
    paper = ArxivPaper(arxiv_id="2401.00001", url="https://arxiv.org/abs/2401.00001", title="Demo Paper")
    candidate = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["infer.py"],
        readiness_label="moderate",
    )

    monkeypatch.setattr("execlint.orchestrator.ArxivClient.fetch_paper", lambda self, arxiv_id, url: paper)
    monkeypatch.setattr(
        "execlint.orchestrator.triage_repositories",
        lambda candidates, github, ref=None: ([candidate], candidate),
    )
    report, _, debug = audit_execution_input_with_debug(
        ExecutionInput(
            arxiv_url="https://arxiv.org/abs/2401.00001",
            repo_url="https://github.com/org/demo",
            weights_url="https://weights.example/model.bin",
            ref="release-1",
        )
    )

    assert report.best_repo == "https://github.com/org/demo"
    assert report.hf_status == "User-provided weights"
    assert debug["weights_source"] == "provided"
    assert isinstance(debug["inferred_capabilities"], list)


def test_execution_input_uses_same_arxiv_normalization_path_as_single_paper(monkeypatch) -> None:
    seen: list[tuple[str, str]] = []

    def _fake_fetch(self, arxiv_id, url):
        seen.append((arxiv_id, url))
        return ArxivPaper(arxiv_id=arxiv_id, url=url, title="Demo")

    monkeypatch.setattr("execlint.orchestrator.ArxivClient.fetch_paper", _fake_fetch)
    monkeypatch.setattr("execlint.orchestrator.discover_repositories", lambda paper, github: [])
    monkeypatch.setattr("execlint.orchestrator.triage_repositories", lambda candidates, github, ref=None: ([], None))

    audit_arxiv_url("https://arxiv.org/pdf/2106.09685")
    audit_execution_input_with_debug(
        ExecutionInput(
            arxiv_url="https://arxiv.org/pdf/2106.09685",
            repo_url="https://github.com/org/demo",
        )
    )

    assert seen[0] == ("2106.09685", "https://arxiv.org/abs/2106.09685")
    assert seen[1] == ("2106.09685", "https://arxiv.org/abs/2106.09685")
