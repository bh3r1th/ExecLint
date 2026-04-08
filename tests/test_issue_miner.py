from execlint.analyzers.issue_miner import mine_issue_signals
from execlint.models import RepoCandidate


class DummyGitHub:
    def list_issues_for_mining(self, full_name: str, limit: int = 10) -> list[dict]:
        return [
            {
                "number": 101,
                "state": "closed",
                "title": "Install error with CUDA 12 and transformers",
                "body": "Crash on import after upgrade. Workaround: downgrade transformers to 4.39 and pin torch==2.3.1.",
                "comments": 12,
                "reactions": {"total_count": 5},
                "comments_preview": [{"body": "Using CUDA 11.8 fixed this for me."}],
            },
            {
                "number": 55,
                "state": "open",
                "title": "install error cuda12 transformers",
                "body": "Still broken.",
                "comments": 2,
                "reactions": {"total_count": 0},
            },
            {
                "number": 18,
                "state": "open",
                "title": "Model checkpoint missing in release package",
                "body": "Fails because weights file is missing from artifact.",
                "comments": 1,
                "reactions": {"total_count": 0},
            },
            {
                "number": 17,
                "state": "open",
                "title": "Deprecated API signature causes runtime error",
                "body": "Replace deprecated API call in parser.",
                "comments": 0,
                "reactions": {"total_count": 0},
            },
        ]


def test_issue_mining_prioritizes_high_signal_and_collapses_duplicates() -> None:
    repo = RepoCandidate(name="demo", full_name="org/demo", url="https://github.com/org/demo")

    signals = mine_issue_signals(repo, DummyGitHub())

    assert len(signals) == 3
    assert signals[0].issue_number == 101
    assert all(signal.issue_number != 55 for signal in signals)


def test_issue_mining_maps_blocker_categories() -> None:
    repo = RepoCandidate(name="demo", full_name="org/demo", url="https://github.com/org/demo")

    signals = mine_issue_signals(repo, DummyGitHub())
    categories = {signal.issue_number: signal.blocker_category for signal in signals}

    assert categories[101] == "cuda"
    assert categories[18] == "missing-assets"
    assert categories[17] == "api-drift"


def test_issue_mining_extracts_short_fix_signal() -> None:
    repo = RepoCandidate(name="demo", full_name="org/demo", url="https://github.com/org/demo")

    signals = mine_issue_signals(repo, DummyGitHub())
    by_number = {signal.issue_number: signal for signal in signals}

    assert by_number[101].fix in {
        "downgrade transformers",
        "pin package version",
        "use CUDA 11.8",
        "use specific CUDA version",
    }
    assert by_number[101].confidence == "high"
