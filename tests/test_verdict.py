from execlint.analyzers.verdict import build_execution_report
from execlint.models import HFModelStatus, IssueFixSignal, RepoCandidate


def test_selects_best_repo_with_readiness_runnable_and_low_blockers() -> None:
    repo_discovery_top = RepoCandidate(
        name="top",
        full_name="org/top",
        url="https://github.com/org/top",
        discovery_score=99.0,
        readiness_label="weak",
        has_readme=False,
        setup_signals=[],
        entrypoint_signals=[],
    )
    repo_ready = RepoCandidate(
        name="ready",
        full_name="org/ready",
        url="https://github.com/org/ready",
        discovery_score=70.0,
        readiness_label="strong",
        has_readme=True,
        setup_signals=["requirements.txt", "pip_install"],
        entrypoint_signals=["infer.py"],
    )

    report = build_execution_report(
        candidates=[repo_discovery_top, repo_ready],
        issue_signals_by_repo={
            "org/top": [IssueFixSignal(blocker="Broken", confidence="high")],
            "org/ready": [IssueFixSignal(blocker="Torch pin needed", fix="Pin torch==2.3", confidence="low")],
        },
        hf_status=HFModelStatus(status="found", model_id="org/model"),
    )

    assert report.best_repo == "https://github.com/org/ready"
    assert report.verdict == "GO"
    assert report.tthw == "Level 1"


def test_verdict_no_go_without_repo() -> None:
    report = build_execution_report(
        candidates=[],
        issue_signals_by_repo={},
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.verdict == "NO-GO"
    assert report.tthw == "Level 4"
    assert report.best_repo == "None found"


def test_verdict_caution_with_setup_friction() -> None:
    repo = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt"],
        entrypoint_signals=["demo.py"],
    )

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={
            "org/demo": [
                IssueFixSignal(
                    blocker="Fails on Python 3.12",
                    fix="Use Python 3.10",
                    confidence="medium",
                )
            ]
        },
        hf_status=HFModelStatus(status="not_found"),
    )

    assert report.verdict == "CAUTION"
    assert report.tthw == "Level 3"
    assert report.fix == "Use Python 3.10"


def test_best_repo_tie_break_prefers_non_archived_and_dependency_clarity() -> None:
    archived_repo = RepoCandidate(
        name="archived",
        full_name="org/archived",
        url="https://github.com/org/archived",
        readiness_label="strong",
        archived=True,
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["train.py"],
        discovery_score=90,
    )
    active_repo = RepoCandidate(
        name="active",
        full_name="org/active",
        url="https://github.com/org/active",
        readiness_label="strong",
        archived=False,
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml", "setup.py"],
        entrypoint_signals=["train.py"],
        discovery_score=85,
    )

    report = build_execution_report(
        candidates=[archived_repo, active_repo],
        issue_signals_by_repo={"org/archived": [], "org/active": []},
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.best_repo == "https://github.com/org/active"


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
