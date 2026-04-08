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


def test_stronger_repo_beats_weak_repo_despite_weak_repo_momentum() -> None:
    weak_repo = RepoCandidate(
        name="weak",
        full_name="org/weak",
        url="https://github.com/org/weak",
        readiness_label="weak",
        stars=5000,
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml", "setup.py"],
        entrypoint_signals=["train.py", "infer.py", "demo.py"],
        discovery_score=120,
    )
    strong_repo = RepoCandidate(
        name="strong",
        full_name="org/strong",
        url="https://github.com/org/strong",
        readiness_label="strong",
        stars=100,
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["train.py", "infer.py"],
        discovery_score=60,
    )

    report = build_execution_report(
        candidates=[weak_repo, strong_repo],
        issue_signals_by_repo={
            "org/weak": [
                IssueFixSignal(
                    blocker="Dependency conflict",
                    fix="Pin package versions",
                    confidence="low",
                    blocker_category="dependency",
                )
            ],
            "org/strong": [
                IssueFixSignal(
                    blocker="Minor setup note",
                    fix="Install optional extras",
                    confidence="low",
                    blocker_category="install",
                )
            ],
        },
        hf_status=HFModelStatus(status="found", model_id="org/model"),
    )

    assert report.best_repo == "https://github.com/org/strong"


def test_weak_repo_wins_only_when_only_credible_fix_path() -> None:
    weak_repo = RepoCandidate(
        name="weak",
        full_name="org/weak",
        url="https://github.com/org/weak",
        readiness_label="weak",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml", "setup.py"],
        entrypoint_signals=["train.py", "infer.py", "demo.py"],
        discovery_score=80,
    )
    moderate_repo = RepoCandidate(
        name="moderate",
        full_name="org/moderate",
        url="https://github.com/org/moderate",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["train.py", "infer.py"],
        discovery_score=70,
    )

    report = build_execution_report(
        candidates=[moderate_repo, weak_repo],
        issue_signals_by_repo={
            "org/moderate": [
                IssueFixSignal(
                    blocker="Missing model weights and no runnable path",
                    fix="",
                    confidence="high",
                    blocker_category="missing assets",
                )
            ],
            "org/weak": [
                IssueFixSignal(
                    blocker="CUDA mismatch",
                    fix="Set torch cuda version to 12.1 and pin wheel",
                    confidence="medium",
                    blocker_category="cuda",
                )
            ],
        },
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.best_repo == "https://github.com/org/weak"


def test_no_fix_weak_repo_missing_weights_is_no_go() -> None:
    weak_repo = RepoCandidate(
        name="weak",
        full_name="org/weak",
        url="https://github.com/org/weak",
        readiness_label="weak",
        has_readme=True,
        setup_signals=["requirements.txt"],
        entrypoint_signals=[],
    )

    report = build_execution_report(
        candidates=[weak_repo],
        issue_signals_by_repo={
            "org/weak": [
                IssueFixSignal(
                    blocker="Missing weights and cannot run inference",
                    fix="",
                    confidence="high",
                    blocker_category="weights missing",
                )
            ]
        },
        hf_status=HFModelStatus(status="not_found"),
    )

    assert report.verdict == "NO-GO"
    assert report.tthw == "Level 4"


def test_moderate_repo_medium_blocker_with_fix_is_caution() -> None:
    moderate_repo = RepoCandidate(
        name="moderate",
        full_name="org/moderate",
        url="https://github.com/org/moderate",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["train.py"],
    )

    report = build_execution_report(
        candidates=[moderate_repo],
        issue_signals_by_repo={
            "org/moderate": [
                IssueFixSignal(
                    blocker="API drift after dependency update",
                    fix="Pin transformers version and update callsite",
                    confidence="medium",
                    blocker_category="api drift",
                )
            ]
        },
        hf_status=HFModelStatus(status="found", model_id="org/model"),
    )

    assert report.verdict == "CAUTION"
    assert report.tthw in {"Level 2", "Level 3"}


def test_strong_repo_missing_weights_with_credible_path_not_no_go() -> None:
    strong_repo = RepoCandidate(
        name="strong",
        full_name="org/strong",
        url="https://github.com/org/strong",
        readiness_label="strong",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["infer.py"],
    )

    report = build_execution_report(
        candidates=[strong_repo],
        issue_signals_by_repo={},
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.verdict == "CAUTION"
    assert report.tthw == "Level 2"


def test_weak_repo_with_fix_and_gated_weights_is_caution() -> None:
    weak_repo = RepoCandidate(
        name="weak",
        full_name="org/weak",
        url="https://github.com/org/weak",
        readiness_label="weak",
        has_readme=True,
        setup_signals=["requirements.txt", "README usage"],
        entrypoint_signals=["run.py"],
    )

    report = build_execution_report(
        candidates=[weak_repo],
        issue_signals_by_repo={
            "org/weak": [
                IssueFixSignal(
                    blocker="Weights are gated",
                    fix="Request access and use provided token to download",
                    confidence="medium",
                    blocker_category="weights",
                )
            ]
        },
        hf_status=HFModelStatus(status="not_found", gated=True),
    )

    assert report.verdict == "CAUTION"
    assert report.tthw == "Level 3"


def test_no_repo_no_weights_is_no_go_level_4() -> None:
    report = build_execution_report(
        candidates=[],
        issue_signals_by_repo={},
        hf_status=HFModelStatus(status="not_found"),
    )

    assert report.verdict == "NO-GO"
    assert report.tthw == "Level 4"


def test_moderate_repo_version_pin_fix_stays_caution() -> None:
    moderate_repo = RepoCandidate(
        name="moderate",
        full_name="org/mod",
        url="https://github.com/org/mod",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["train.py"],
    )
    report = build_execution_report(
        candidates=[moderate_repo],
        issue_signals_by_repo={
            "org/mod": [
                IssueFixSignal(
                    blocker="CUDA and version mismatch after upstream change",
                    fix="Pin torch/cu121 wheels and set transformers==4.39",
                    confidence="medium",
                    blocker_category="cuda version",
                )
            ]
        },
        hf_status=HFModelStatus(status="found", model_id="org/model"),
    )

    assert report.verdict == "CAUTION"
    assert report.tthw in {"Level 2", "Level 3"}
