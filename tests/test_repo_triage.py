from execlint.analyzers.repo_discovery import _score_repository
from execlint.analyzers.repo_triage import _compute_readiness, triage_repositories
from execlint.models import ArxivPaper, RepoCandidate


class DummyGitHub:
    def get_readme(self, full_name: str) -> str | None:
        if full_name == "org/good":
            return "Install with pip. Usage and quickstart included."
        if full_name == "org/runnable":
            return "Quickstart with pip install -r requirements.txt"
        return None

    def get_repo_file_paths(self, full_name: str, default_branch: str = "main") -> list[str]:
        if full_name == "org/good":
            return ["requirements.txt", "train.py", "scripts/run.sh", "demo.ipynb"]
        if full_name == "org/runnable":
            return ["pyproject.toml", "infer.py", "app.py", "notebooks/demo.ipynb"]
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
    weak_score, weak_label = _compute_readiness(False, 0, 0, 0.0, 200, True, 2)

    assert strong_score > moderate_score > weak_score
    assert strong_label == "strong"
    assert moderate_label == "moderate"
    assert weak_label == "weak"


def test_archived_penalty_lowers_readiness() -> None:
    active_score, _ = _compute_readiness(True, 2, 2, 0.6, 10, False, 120)
    archived_score, archived_label = _compute_readiness(True, 2, 2, 0.6, 10, True, 120)

    assert archived_score < active_score
    assert archived_label in {"moderate", "weak"}


def test_inactive_fork_penalty_lowers_readiness() -> None:
    baseline, _ = _compute_readiness(True, 2, 1, 0.3, 10, False, 80, likely_inactive_fork=False)
    penalized, _ = _compute_readiness(True, 2, 1, 0.3, 10, False, 80, likely_inactive_fork=True)

    assert penalized < baseline


def test_runnable_file_boost_reflects_in_entrypoints() -> None:
    candidates = [
        RepoCandidate(name="runnable", full_name="org/runnable", url="https://github.com/org/runnable", pushed_at="2026-02-01T00:00:00Z"),
    ]

    triaged, best = triage_repositories(candidates, DummyGitHub())
    repo = triaged[0]

    assert {"infer.py", "app.py", "notebook"}.issubset(set(repo.entrypoint_signals))
    assert repo.readiness_score >= 6.5
    assert best is not None and best.full_name == "org/runnable"


def test_official_wording_boost_in_discovery_score() -> None:
    paper = ArxivPaper(
        arxiv_id="1234.5678",
        url="https://arxiv.org/abs/1234.5678",
        title="Great Model",
        authors=["Alex Kim"],
    )
    plain_repo = RepoCandidate(
        name="great-model",
        full_name="org/plain",
        url="https://github.com/org/plain",
        description="implementation details",
        size_kb=300,
    )
    official_repo = RepoCandidate(
        name="great-model",
        full_name="org/official",
        url="https://github.com/org/official",
        description="Official implementation for Great Model",
        size_kb=300,
    )

    plain_score, _ = _score_repository(plain_repo, paper)
    official_score, official_reasons = _score_repository(official_repo, paper)

    assert official_score > plain_score
    assert any("official_wording" in reason for reason in official_reasons)
