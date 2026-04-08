from execlint.analyzers.verdict import build_execution_report
from execlint.models import HFModelStatus, IssueFixSignal, RepoCandidate


def test_verdict_go_with_setup_and_no_blockers() -> None:
    repo = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        has_readme=True,
        setup_signals=["install", "usage"],
    )

    report = build_execution_report(
        best_repo=repo,
        issue_signals=[],
        hf_status=HFModelStatus(status="not_found"),
    )

    assert report.verdict == "GO"
    assert report.tthw == "Level 1"


def test_verdict_no_go_without_repo() -> None:
    report = build_execution_report(
        best_repo=None,
        issue_signals=[],
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.verdict == "NO-GO"
    assert report.tthw == "Level 4"


def test_verdict_caution_with_blocker() -> None:
    repo = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        has_readme=True,
        setup_signals=["install", "usage"],
    )
    blocker = IssueFixSignal(blocker="Cannot run on latest torch", confidence="high")

    report = build_execution_report(
        best_repo=repo,
        issue_signals=[blocker],
        hf_status=HFModelStatus(status="found", model_id="org/model"),
    )

    assert report.verdict == "CAUTION"
    assert report.tthw == "Level 2"
