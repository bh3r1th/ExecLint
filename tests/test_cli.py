from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import pytest

from execlint.cli import app
from execlint.models import ExecutionInput, ExecutionReport, Gap


def _report(**overrides) -> ExecutionReport:
    data = {
        "paper_title": "Demo Paper",
        "repo_url": "https://github.com/org/demo",
        "gaps": [],
        "execution_path": "No extracted execution commands",
        "warnings": [],
    }
    data.update(overrides)
    return ExecutionReport(**data)


def test_cli_output_includes_gap_first_fields(monkeypatch) -> None:
    report = _report(
        paper_title="ExecLint Paper",
        execution_path="install: pip install -r requirements.txt; run: python train.py",
        gaps=[
            Gap(
                label="dataset must be supplied manually",
                category="data",
                evidence="README mentions dataset/data setup but no adjacent download link or automated bootstrap was found",
            ),
            Gap(
                label="env version unclear",
                category="env",
                evidence="README mentions Python/CUDA/PyTorch context but no version was pinned",
            ),
        ],
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
        "Paper title: ExecLint Paper",
        "Repo URL: https://github.com/org/demo",
        "GAPS (2):",
        "  - dataset must be supplied manually  [data]",
        "    README mentions dataset/data setup but no adjacent download link or automated bootstrap was found",
        "  - env version unclear  [env]",
        "    README mentions Python/CUDA/PyTorch context but no version was pinned",
        "EXECUTION PATH:",
        "  - install: pip install -r requirements.txt",
        "  - run: python train.py",
    ]
    assert "Verdict" not in out
    assert "TTHW" not in out
    assert "what_breaks" not in out


def test_cli_debug_output_handles_partial_signals(monkeypatch) -> None:
    report = _report(
        repo_url="None found",
        gaps=[Gap(label="No repository candidate", category="run", evidence="No GitHub repository candidate was available to inspect")],
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
                "selected_repo_blocker_severity": "high",
                "hf_summary": "unclear",
                "partial_source_failures": ["github_discovery"],
            },
        ),
    )

    stdout = StringIO()
    with redirect_stdout(stdout):
        app(["https://arxiv.org/abs/1234.5678", "--repo", "https://github.com/org/demo", "--debug"])
    out = stdout.getvalue()

    assert "Paper title: Demo Paper" in out
    assert "Repo URL: None found" in out
    assert "debug_signals:" in out
    assert "- inferred capabilities: demo, inference" in out
    assert "- readiness: n/a" in out
    assert "- blocker severity: high" in out
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
        gaps=[
            Gap(
                label="install path ambiguous",
                category="install",
                evidence="No setup.py, pyproject.toml, requirements.txt, or Dockerfile in repo root",
            ),
            Gap(
                label="no clear run command",
                category="run",
                evidence="No regex-extracted run command found in README or repo file paths",
            ),
        ],
    )
    monkeypatch.setattr(
        "execlint.cli.audit_execution_input_with_debug",
        lambda execution_input: (report, [], {"paper_title": "Demo Paper"}),
    )

    stdout = StringIO()
    with redirect_stdout(stdout):
        app(["https://arxiv.org/abs/1234.5678", "--repo", "https://github.com/org/demo"])
    out = stdout.getvalue()

    assert "GAPS (2):\n  - install path ambiguous  [install]" in out
    assert "EXECUTION PATH:\n  - No extracted execution commands" in out
