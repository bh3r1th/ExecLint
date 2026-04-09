from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import pytest

from execlint.cli import app, audit
from execlint.models import ExecutionInput, ExecutionReport


def test_cli_output_includes_paper_title_and_code_url(monkeypatch) -> None:
    report = ExecutionReport(
        verdict="CAUTION",
        tthw="Level 2",
        best_repo="https://github.com/org/demo",
        runnable_for="unclear",
        not_clearly_supported="",
        what_breaks="No major blocker",
        fix="No clear fix found",
        hf_status="Hugging Face model found (org/model)",
        technical_debt="None identified",
    )

    monkeypatch.setattr(
        "execlint.cli.audit_execution_input_with_debug",
        lambda execution_input: (
            report,
            [],
            {
                "paper_title": "ExecLint Paper",
                "paper_code_url": "https://github.com/from-paper/demo",
                "candidate_count": 1,
                "selected_repo_name": "org/demo",
                "selected_repo_readiness": "strong",
                "selected_repo_blocker_severity": "n/a",
                "selected_repo_fix_signal_count": 0,
                "hf_summary": "found",
                "partial_source_failures": [],
            },
        ),
    )

    stdout = StringIO()
    with redirect_stdout(stdout):
        app(["https://arxiv.org/abs/1234.5678", "--repo", "https://github.com/org/demo"])
    out = stdout.getvalue()

    assert "paper:" in out
    assert "- Title: ExecLint Paper" in out
    assert "- Code URL: https://github.com/org/demo" in out
    assert "Code URL (from paper)" not in out
    assert "execution_report:" in out
    assert "- Time-to-Hello-World (TTHW): Level 2 — minor setup required" in out
    assert "- Best Repo:" not in out
    assert "- Runnable For: unclear" in out
    assert "- Execution Path: No extracted execution commands" in out
    assert "- Gaps: None identified" in out
    assert "- Not Clearly Supported: None identified" in out
    assert "debug_signals:" not in out


def test_cli_debug_output_handles_partial_signals(monkeypatch) -> None:
    report = ExecutionReport(
        verdict="NO-GO",
        tthw="Level 4",
        best_repo="None found",
        runnable_for="unclear",
        not_clearly_supported="",
        what_breaks="Repository discovery unavailable",
        fix="Unavailable: GitHub discovery failed",
        hf_status="Hugging Face status unclear",
        technical_debt="Unknown due to unavailable repository data",
    )

    monkeypatch.setattr(
        "execlint.cli.audit_execution_input_with_debug",
        lambda execution_input: (
            report,
            [],
            {
                "paper_title": "Demo Paper",
                "paper_code_url": None,
                "ref": "v1.2.3",
                    "weights_source": "provided",
                    "inferred_capabilities": ["demo", "inference"],
                    "candidate_count": 0,
                    "selected_repo_name": "none",
                    "selected_repo_readiness": "n/a",
                    "selected_repo_blocker_severity": "low",
                    "selected_repo_fix_signal_count": 0,
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
    assert "- Code URL: https://github.com/org/demo" in out
    assert "- Time-to-Hello-World (TTHW): Level 4 — no credible runnable path" in out
    assert "- Best Repo:" not in out
    assert "- Execution Path: No extracted execution commands" in out
    assert "- Gaps: None identified" in out
    assert "debug_signals:" in out
    assert "- inferred capabilities: demo, inference" in out
    assert "- readiness: n/a" in out
    assert "- blocker severity: low" in out
    assert "- weights source: provided" in out
    assert "- partial failures: github_discovery" in out
    assert "- ref:" not in out
    assert "- repo candidates inspected:" not in out
    assert "- repo selected:" not in out
    assert "- fix signals found:" not in out
    assert "- HF:" not in out


def test_cli_missing_repo_url_fails() -> None:
    stderr = StringIO()
    with redirect_stderr(stderr), pytest.raises(SystemExit) as exc:
        app(["https://arxiv.org/abs/1234.5678"])

    assert exc.value.code == 2
    assert "--repo" in stderr.getvalue()


def test_cli_accepts_optional_fields(monkeypatch) -> None:
    captured: dict[str, object] = {}
    report = ExecutionReport(
        verdict="CAUTION",
        tthw="Level 2",
        best_repo="https://github.com/org/demo",
        runnable_for="unclear",
        not_clearly_supported="",
        what_breaks="No major blocker",
        fix="No clear fix found",
        hf_status="Hugging Face status unclear",
        technical_debt="None identified",
    )

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
    assert execution_input.repo_url == "https://github.com/org/demo"
    assert execution_input.weights_url == "https://huggingface.co/org/model"
    assert execution_input.ref == "v1.2.3"


def test_cli_no_task_mode_required_anywhere(monkeypatch) -> None:
    report = ExecutionReport(
        verdict="CAUTION",
        tthw="Level 2",
        best_repo="https://github.com/org/demo",
        runnable_for="unclear",
        not_clearly_supported="",
        what_breaks="No major blocker",
        fix="No clear fix found",
        hf_status="Hugging Face status unclear",
        technical_debt="None identified",
    )

    def _fake_audit(execution_input: ExecutionInput):
        return report, [], {"paper_title": "Demo Paper", "paper_code_url": None}

    monkeypatch.setattr("execlint.cli.audit_execution_input_with_debug", _fake_audit)
    stdout = StringIO()
    with redirect_stdout(stdout):
        app(
            [
                "https://arxiv.org/abs/1234.5678",
                "--repo",
                "https://github.com/org/demo",
            ]
        )


def test_cli_output_format_is_stable_and_repo_url_is_required_internally(monkeypatch) -> None:
    report = ExecutionReport(
        verdict="GO",
        tthw="Level 1",
        best_repo="https://github.com/org/demo",
        runnable_for="demo",
        not_clearly_supported="",
        what_breaks="No concrete blocker visible",
        fix="No clear fix found",
        hf_status="Hugging Face model found (org/model)",
        technical_debt="None identified",
    )

    monkeypatch.setattr(
        "execlint.cli.audit_execution_input_with_debug",
        lambda execution_input: (report, [], {"paper_title": "Demo Paper"}),
    )

    stdout = StringIO()
    with redirect_stdout(stdout):
        app(["https://arxiv.org/abs/1234.5678", "--repo", "https://github.com/org/demo"])

    out = stdout.getvalue().splitlines()
    assert out[:13] == [
        "paper:",
        "- Title: Demo Paper",
        "- Code URL: https://github.com/org/demo",
        "execution_report:",
        "- Execution Path: No extracted execution commands",
        "- Runnable For: demo",
        "- Gaps: None identified",
        "- What Breaks: No concrete blocker visible",
        "- Technical Debt: None identified",
        "- Not Clearly Supported: None identified",
        "- HF Status: Hugging Face model found (org/model)",
        "- Fix (if any): No clear fix found",
        "- Verdict: GO",
    ]
    assert out[13] == "- Time-to-Hello-World (TTHW): Level 1 — runnable immediately"


@pytest.mark.parametrize(
    ("level", "meaning"),
    [
        ("Level 1", "runnable immediately"),
        ("Level 2", "minor setup required"),
        ("Level 3", "substantial setup required"),
        ("Level 4", "no credible runnable path"),
    ],
)
def test_cli_tthw_line_includes_exact_meaning(monkeypatch, level: str, meaning: str) -> None:
    report = ExecutionReport(
        verdict="CAUTION" if level != "Level 1" else "GO",
        tthw=level,
        best_repo="https://github.com/org/demo",
        runnable_for="demo" if level == "Level 1" else "unclear",
        not_clearly_supported="",
        what_breaks="No concrete blocker visible",
        fix="No clear fix found",
        hf_status="Hugging Face status unclear",
        technical_debt="None identified",
    )

    monkeypatch.setattr(
        "execlint.cli.audit_execution_input_with_debug",
        lambda execution_input: (report, [], {"paper_title": "Demo Paper"}),
    )

    stdout = StringIO()
    with redirect_stdout(stdout):
        app(["https://arxiv.org/abs/1234.5678", "--repo", "https://github.com/org/demo"])

    assert f"- Time-to-Hello-World (TTHW): {level} — {meaning}" in stdout.getvalue()


def test_cli_output_includes_execution_path_and_gaps(monkeypatch) -> None:
    report = ExecutionReport(
        verdict="CAUTION",
        tthw="Level 3",
        best_repo="https://github.com/org/demo",
        runnable_for="training, evaluation",
        execution_path="install: pip install -r requirements.txt; run: python train.py; evaluate: python eval.py",
        gaps="dataset must be supplied manually; env version unclear",
        not_clearly_supported="demo, inference",
        what_breaks="dataset must be supplied manually",
        fix="Use provided dataset script",
        hf_status="Hugging Face status unclear",
        technical_debt="dataset bootstrap manual",
    )
    monkeypatch.setattr(
        "execlint.cli.audit_execution_input_with_debug",
        lambda execution_input: (report, [], {"paper_title": "Demo Paper"}),
    )

    stdout = StringIO()
    with redirect_stdout(stdout):
        app(["https://arxiv.org/abs/1234.5678", "--repo", "https://github.com/org/demo"])
    out = stdout.getvalue()

    assert "- Execution Path: install: pip install -r requirements.txt; run: python train.py; evaluate: python eval.py" in out
    assert "- Gaps: dataset must be supplied manually; env version unclear" in out


def test_cli_output_when_no_execution_path_has_clear_gaps_text(monkeypatch) -> None:
    report = ExecutionReport(
        verdict="NO-GO",
        tthw="Level 4",
        best_repo="https://github.com/org/demo",
        runnable_for="unclear",
        execution_path="No extracted execution commands",
        gaps="install path ambiguous; no clear run command",
        not_clearly_supported="meaningful execution modes",
        what_breaks="no obvious runnable entrypoint for any meaningful capability",
        fix="No clear fix found",
        hf_status="Hugging Face status unclear",
        technical_debt="install path ambiguous",
    )
    monkeypatch.setattr(
        "execlint.cli.audit_execution_input_with_debug",
        lambda execution_input: (report, [], {"paper_title": "Demo Paper"}),
    )

    stdout = StringIO()
    with redirect_stdout(stdout):
        app(["https://arxiv.org/abs/1234.5678", "--repo", "https://github.com/org/demo"])
    out = stdout.getvalue()

    assert "- Execution Path: No extracted execution commands" in out
    assert "- Gaps: install path ambiguous; no clear run command" in out
