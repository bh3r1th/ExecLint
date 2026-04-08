from execlint.analyzers.issue_miner import mine_issue_signals
from execlint.models import RepoCandidate


class DummyGitHub:
    def list_open_issues(self, full_name: str, limit: int = 10) -> list[dict]:
        return [
            {
                "number": 42,
                "title": "Install error with CUDA 12",
                "body": "Dependency conflict. Workaround: pin torch==2.3.1 and reinstall requirements.",
            },
            {
                "number": 14,
                "title": "Docs typo",
                "body": "Small typo in README",
            },
        ]


def test_mine_issue_signals_detects_keyword_blocker_and_fix() -> None:
    repo = RepoCandidate(name="demo", full_name="org/demo", url="https://github.com/org/demo")
    signals = mine_issue_signals(repo, DummyGitHub())

    assert len(signals) == 1
    assert signals[0].issue_number == 42
    assert signals[0].blocker_category in {"installation", "dependency", "hardware", "runtime"}
    assert signals[0].fix is not None
    assert "pin torch" in signals[0].fix.lower()
    assert signals[0].confidence == "high"
