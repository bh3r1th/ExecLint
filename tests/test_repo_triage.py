from execlint.analyzers.repo_triage import triage_repositories
from execlint.models import RepoCandidate


class DummyGitHub:
    def get_readme(self, full_name: str) -> str | None:
        if full_name == "org/good":
            return "Install with pip. Usage and quickstart included."
        return None


def test_triage_marks_readme_and_setup_signals() -> None:
    candidates = [
        RepoCandidate(name="good", full_name="org/good", url="https://github.com/org/good", stars=5),
        RepoCandidate(name="bad", full_name="org/bad", url="https://github.com/org/bad", stars=50),
    ]

    triaged, best = triage_repositories(candidates, DummyGitHub())

    good = next(c for c in triaged if c.full_name == "org/good")
    assert good.has_readme is True
    assert "install" in good.setup_signals
    assert best is not None
    assert best.full_name == "org/good"
