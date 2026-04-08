from __future__ import annotations

import json
from pathlib import Path

import pytest

from execlint.analyzers.verdict import build_execution_report
from execlint.models import HFModelStatus, IssueFixSignal, RepoCandidate

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "verdict"


def _load_case(name: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text())


@pytest.mark.parametrize(
    "fixture_name",
    [
        "clear_go",
        "clear_caution",
        "clear_no_go",
        "weak_repo_usable_fix_path",
        "missing_weights_downgrade",
    ],
)
def test_verdict_golden_fixtures(fixture_name: str) -> None:
    fixture = _load_case(fixture_name)

    candidates = [RepoCandidate(**candidate) for candidate in fixture["candidates"]]
    issue_signals_by_repo = {
        repo_name: [IssueFixSignal(**signal) for signal in signals]
        for repo_name, signals in fixture["issue_signals_by_repo"].items()
    }
    hf_status = HFModelStatus(**fixture["hf_status"])

    report = build_execution_report(
        candidates=candidates,
        issue_signals_by_repo=issue_signals_by_repo,
        hf_status=hf_status,
    )

    assert report.verdict == fixture["expected"]["verdict"]
    assert report.tthw == fixture["expected"]["tthw"]
    assert report.best_repo == fixture["expected"]["best_repo"]


def test_weak_repo_not_selected_over_moderate_without_clear_fix_advantage() -> None:
    weak_repo = RepoCandidate(
        name="weak",
        full_name="org/weak",
        url="https://github.com/org/weak",
        readiness_label="weak",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml", "setup.py"],
        entrypoint_signals=["train.py", "infer.py", "demo.py"],
        discovery_score=99,
    )
    moderate_repo = RepoCandidate(
        name="moderate",
        full_name="org/moderate",
        url="https://github.com/org/moderate",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["train.py"],
        discovery_score=70,
    )

    report = build_execution_report(
        candidates=[weak_repo, moderate_repo],
        issue_signals_by_repo={"org/weak": [], "org/moderate": []},
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.best_repo == "https://github.com/org/moderate"
