from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import pytest

from execlint.cli import app
from execlint.models import ExecutionInput, ExecutionReport


def _report(**overrides) -> ExecutionReport:
    data = {
        "best_repo": "https://github.com/org/demo",
        "runnable_for": "unclear",
        "execution_path": "No extracted execution commands",
        "gaps": "None identified",
        "not_clearly_supported": "",
        "what_breaks": "No concrete blocker visible",
        "fix": "No clear fix found",
        "hf_status": "Hugging Face status unclear",
        "technical_debt": "None identified",
    }
    data.update(overrides)
    return ExecutionReport(**data)


def test_cli_output_includes_only_gap_first_fields(monkeypatch) -> None:
    report = _report(
        execution_path="install: pip install -r requirements.txt; run: python train.py",
        gaps="dataset must be supplied manually; env version unclear",
        what_breaks="dataset must be supplied manually",
    )

    monkeypatch.setattr(
        "execlint.cli.audit_execution_input_with_debug",
        lambda execution_input: (
            report,
            [],
            {"paper_title": "ExecLint Paper"},
        ),
    )

    stdout = StringIO()
    with redirect_stdout(stdout):
        app(["https://arxiv.org/abs/1234.5678", "--repo", "https://github.com/org/demo"])
    out = stdout.getvalue()

    assert out.splitlines() == [
        "paper:",
        "- Title: ExecLint Paper",
        "- Repo URL: https://github.com/org/demo",
        "gaps:",
        "- dataset must be supplied manually",
        "- env version unclear",
        "what_breaks:",
        "- dataset must be supplied manually",
        "execution_path:",
        "- install: pip install -r requirements.txt",
        "- run: python train.py",
    ]
    assert "Verdict" not in out
    assert "TTHW" not in out
    assert "HF Status" not in out


def test_cli_debug_output_handles_partial_signals(monkeypatch) -> None:
    report = _report(
        best_repo="None found",
        gaps="No repository candidate",
        what_breaks="Repository discovery unavailable",
        technical_debt="Unknown due to unavailable repository data",
    )

    monkeypatch.setattr(
        "execlint.cli.audit_execution_input_with_debug",
        lambda execution_input: (
            report,
            [],
            {
                "paper_title": "Demo Paper",
                "ref": "v1.2.3",
                "weights_source": "provided",
                "inferred_capabilities": ["demo", "inference"],
                "candidate_count": 0,
                "selected_repo_name": "none",
                "selected_repo_readiness": "n/a",
                "selected_repo_blocker_severity": "low",
                "hf_summary": "unclear",
                "partial_source_failures": ["github_discovery"],
            },
        ),
    )

    stdout = StringIO()
    with redirect_stdout(stdout):
        app(["https://arxiv.org/abs/1234.5678", "--repo", "https://github.com/org/demo", "--debug"])
    out = stdout.getvalue()

    assert "- Title: Demo Paper" in out
    assert "- Repo URL: https://github.com/org/demo" in out
    assert "debug_signals:" in out
    assert "- inferred capabilities: demo, inference" in out
    assert "- readiness: n/a" in out
    assert "- blocker severity: low" in out
    assert "- weights source: provided" in out
    assert "- partial failures: github_discovery" in out
    assert "- ref:" not in out


def test_cli_missing_repo_url_fails() -> None:
    stderr = StringIO()
    with redirect_stderr(stderr), pytest.raises(SystemExit) as exc:
        app(["https://arxiv.org/abs/1234.5678"])

    assert exc.value.code == 2
    assert "--repo" in stderr.getvalue()


def test_cli_accepts_optional_fields(monkeypatch) -> None:
    captured: dict[str, object] = {}
    report = _report()

    def _fake_audit(execution_input: ExecutionInput):
        captured["execution_input"] = execution_input
        return report, [], {"paper_title": "Demo Paper", "paper_code_url": None}

    monkeypatch.setattr("execlint.cli.audit_execution_input_with_debug", _fake_audit)

    stdout = StringIO()
    with redirect_stdout(stdout):
        app(
            [
                "https://arxiv.org/abs/1234.5678",
                "--repo",
                "https://github.com/org/demo",
                "--weights",
                "https://huggingface.co/org/model",
                "--ref",
                "v1.2.3",
            ]
        )

    execution_input = captured["execution_input"]
    assert isinstance(execution_input, ExecutionInput)
    assert str(execution_input.repo_url) == "https://github.com/org/demo"
    assert str(execution_input.weights_url) == "https://huggingface.co/org/model"
    assert execution_input.ref == "v1.2.3"


def test_cli_no_task_mode_required_anywhere(monkeypatch) -> None:
    monkeypatch.setattr(
        "execlint.cli.audit_execution_input_with_debug",
        lambda execution_input: (_report(), [], {"paper_title": "Demo Paper", "paper_code_url": None}),
    )
    stdout = StringIO()
    with redirect_stdout(stdout):
        app(
            [
                "https://arxiv.org/abs/1234.5678",
                "--repo",
                "https://github.com/org/demo",
            ]
        )


def test_cli_output_when_no_execution_path_has_clear_gaps_text(monkeypatch) -> None:
    report = _report(
        execution_path="No extracted execution commands",
        gaps="install path ambiguous; no clear run command",
        what_breaks="install path ambiguous; no clear run command",
    )
    monkeypatch.setattr(
        "execlint.cli.audit_execution_input_with_debug",
        lambda execution_input: (report, [], {"paper_title": "Demo Paper"}),
    )

    stdout = StringIO()
    with redirect_stdout(stdout):
        app(["https://arxiv.org/abs/1234.5678", "--repo", "https://github.com/org/demo"])
    out = stdout.getvalue()

    assert "gaps:\n- install path ambiguous\n- no clear run command" in out
    assert "execution_path:\n- No extracted execution commands" in out
