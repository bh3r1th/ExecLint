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
