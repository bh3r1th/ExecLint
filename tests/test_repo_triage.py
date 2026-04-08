from execlint.analyzers.repo_triage import _compute_readiness, triage_repositories
from execlint.models import RepoCandidate


class DummyGitHub:
    def get_readme(self, full_name: str) -> str | None:
        if full_name == "org/good":
            return "Install with pip. Usage and quickstart included."
        return None

    def get_repo_file_paths(self, full_name: str, default_branch: str = "main") -> list[str]:
        if full_name == "org/good":
            return ["requirements.txt", "train.py", "scripts/run.sh", "demo.ipynb"]
        return []


def test_triage_marks_readme_setup_entrypoints_and_label() -> None:
    candidates = [
        RepoCandidate(name="good", full_name="org/good", url="https://github.com/org/good", stars=5, pushed_at="2026-01-01T00:00:00Z"),
        RepoCandidate(name="bad", full_name="org/bad", url="https://github.com/org/bad", stars=50),
    ]

    triaged, best = triage_repositories(candidates, DummyGitHub())

    good = next(c for c in triaged if c.full_name == "org/good")
    assert good.has_readme is True
    assert "requirements.txt" in good.setup_signals
    assert "train.py" in good.entrypoint_signals
    assert good.readiness_label in {"strong", "moderate"}
    assert best is not None
    assert best.full_name == "org/good"


def test_readiness_label_mapping() -> None:
    strong_score, strong_label = _compute_readiness(True, 3, 2, 1.0, 5, False, 80)
    moderate_score, moderate_label = _compute_readiness(True, 1, 0, 0.3, 30, False, 250)
    weak_score, weak_label = _compute_readiness(False, 0, 0, 0.0, 200, True, 1000)

    assert strong_score > moderate_score > weak_score
    assert strong_label == "strong"
    assert moderate_label == "moderate"
    assert weak_label == "weak"
