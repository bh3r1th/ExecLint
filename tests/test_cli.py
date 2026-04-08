from __future__ import annotations

from execlint.cli import audit
from execlint.models import ExecutionReport


def test_cli_debug_output_handles_partial_signals(monkeypatch, capsys) -> None:
    report = ExecutionReport(
        verdict="NO-GO",
        tthw="Level 4",
        best_repo="None found",
        what_breaks="Repository discovery unavailable",
        fix="Unavailable: GitHub discovery failed",
        hf_status="Hugging Face status unclear",
        technical_debt="Unknown due to unavailable repository data",
    )

    monkeypatch.setattr(
        "execlint.cli.audit_arxiv_url_with_debug",
        lambda arxiv_url: (
            report,
            [],
            {
                "candidate_count": 0,
                "selected_repo_name": "none",
                "selected_repo_readiness": "n/a",
                "selected_repo_blocker_severity": "n/a",
                "selected_repo_fix_signal_count": 0,
                "hf_summary": "unclear",
                "partial_source_failures": ["github_discovery"],
            },
        ),
    )

    audit("https://arxiv.org/abs/1234.5678", debug=True)
    out = capsys.readouterr().out

    assert "- repo candidates inspected: 0" in out
    assert "- repo selected: none" in out
    assert "- HF: unclear" in out
    assert "- partial failures: github_discovery" in out
