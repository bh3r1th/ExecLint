from execlint.analyzers.issue_miner import mine_issue_signals
from execlint.models import RepoCandidate


class DummyGitHub:
    def list_open_issues(self, full_name: str, limit: int = 10) -> list[dict]:
        return [
            {
                "title": "Cannot run training script",
                "body": "The script fails on startup. Workaround: pin numpy to 1.26",
            },
            {
                "title": "Docs typo",
                "body": "Small typo in README",
            },
        ]


def test_mine_issue_signals_detects_blocker_and_fix() -> None:
    repo = RepoCandidate(name="demo", full_name="org/demo", url="https://github.com/org/demo")
    signals = mine_issue_signals(repo, DummyGitHub())

    assert len(signals) == 1
    assert "Cannot run" in signals[0].blocker
    assert signals[0].fix is not None
    assert signals[0].confidence == "high"
