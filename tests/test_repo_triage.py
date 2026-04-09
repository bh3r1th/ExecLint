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
    assert "training" in good.inferred_capabilities
    assert "smoke_test" in good.inferred_capabilities
    assert best is not None
    assert best.full_name == "org/good"


def test_readiness_label_mapping() -> None:
    strong_score, strong_label = _compute_readiness(True, 3, 2, 2, 1.0, 5, False, 80)
    moderate_score, moderate_label = _compute_readiness(True, 1, 1, 1, 0.3, 30, False, 250)
    weak_score, weak_label = _compute_readiness(False, 0, 0, 0, 0.0, 200, True, 2)

    assert strong_score > moderate_score > weak_score
    assert strong_label == "strong"
    assert moderate_label == "moderate"
    assert weak_label == "weak"


def test_archived_penalty_lowers_readiness() -> None:
    active_score, _ = _compute_readiness(True, 2, 2, 2, 0.6, 10, False, 120)
    archived_score, archived_label = _compute_readiness(True, 2, 2, 2, 0.6, 10, True, 120)

    assert archived_score < active_score
    assert archived_label in {"moderate", "weak"}


def test_inactive_fork_penalty_lowers_readiness() -> None:
    baseline, _ = _compute_readiness(True, 2, 1, 1, 0.3, 10, False, 80, likely_inactive_fork=False)
    penalized, _ = _compute_readiness(True, 2, 1, 1, 0.3, 10, False, 80, likely_inactive_fork=True)

    assert penalized < baseline


def test_runnable_file_boost_reflects_in_entrypoints() -> None:
    candidates = [
        RepoCandidate(name="runnable", full_name="org/runnable", url="https://github.com/org/runnable", pushed_at="2026-02-01T00:00:00Z"),
    ]

    triaged, best = triage_repositories(candidates, DummyGitHub())
    repo = triaged[0]

    assert {"infer.py", "app.py", "notebook"}.issubset(set(repo.entrypoint_signals))
    assert {"demo", "inference"}.issubset(set(repo.inferred_capabilities))
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


def test_triage_prefers_stronger_repo_over_weak_repo() -> None:
    candidates = [
        RepoCandidate(
            name="good",
            full_name="org/good",
            url="https://github.com/org/good",
            stars=5,
            pushed_at="2026-02-01T00:00:00Z",
        ),
        RepoCandidate(
            name="weak",
            full_name="org/weak",
            url="https://github.com/org/weak",
            stars=9000,
            pushed_at="2026-03-01T00:00:00Z",
        ),
    ]

    triaged, best = triage_repositories(candidates, DummyGitHub())

    assert len(triaged) == 2
    assert best is not None
    assert best.full_name == "org/good"


def test_triage_all_weak_is_deterministic() -> None:
    candidates = [
        RepoCandidate(name="b", full_name="org/b", url="https://github.com/org/b", stars=1),
        RepoCandidate(name="a", full_name="org/a", url="https://github.com/org/a", stars=1),
    ]

    _, best = triage_repositories(candidates, DummyGitHub())

    assert best is not None
    assert best.full_name == "org/b"


def test_triage_uses_explicit_ref_for_repo_paths() -> None:
    class RefAwareGitHub(DummyGitHub):
        def __init__(self) -> None:
            self.requested_refs: list[str] = []

        def get_repo_file_paths(self, full_name: str, default_branch: str = "main") -> list[str]:
            self.requested_refs.append(default_branch)
            return ["requirements.txt", "train.py"]

    github = RefAwareGitHub()
    candidates = [RepoCandidate(name="good", full_name="org/good", url="https://github.com/org/good")]

    triaged, _ = triage_repositories(candidates, github, ref="release-2026")

    assert github.requested_refs == ["release-2026"]
    assert triaged[0].default_branch == "release-2026"


def test_capability_inference_from_readme_and_paths() -> None:
    class CapabilityGitHub(DummyGitHub):
        def get_readme(self, full_name: str) -> str | None:
            return "Launch with gradio, then run benchmark metrics."

        def get_repo_file_paths(self, full_name: str, default_branch: str = "main") -> list[str]:
            return ["app.py", "scripts/predict.py", "eval.py", "metrics/report.py"]

    triaged, _ = triage_repositories(
        [RepoCandidate(name="capable", full_name="org/capable", url="https://github.com/org/capable")],
        CapabilityGitHub(),
    )

    assert set(triaged[0].inferred_capabilities) == {"demo", "inference", "evaluation"}


def test_capability_inference_returns_unclear_when_no_signal_exists() -> None:
    triaged, _ = triage_repositories(
        [RepoCandidate(name="unclear", full_name="org/unclear", url="https://github.com/org/unclear")],
        DummyGitHub(),
    )

    unclear = next(repo for repo in triaged if repo.full_name == "org/unclear")
    assert unclear.inferred_capabilities == ["unclear"]
    assert unclear.readiness_label == "weak"


def test_weak_inference_evidence_does_not_trigger_inference() -> None:
    class WeakInferenceGitHub(DummyGitHub):
        def get_readme(self, full_name: str) -> str | None:
            return "Supports generation quality analysis."

        def get_repo_file_paths(self, full_name: str, default_branch: str = "main") -> list[str]:
            return ["train.py", "metrics/generation_scores.py"]

    triaged, _ = triage_repositories(
        [RepoCandidate(name="weak-inf", full_name="org/weak-inf", url="https://github.com/org/weak-inf")],
        WeakInferenceGitHub(),
    )

    assert "inference" not in triaged[0].inferred_capabilities


def test_generic_generation_script_does_not_trigger_inference() -> None:
    class GenericGenerationGitHub(DummyGitHub):
        def get_readme(self, full_name: str) -> str | None:
            return "Train and evaluate generated samples."

        def get_repo_file_paths(self, full_name: str, default_branch: str = "main") -> list[str]:
            return ["train.py", "scripts/generate_samples.py", "eval.py"]

    triaged, _ = triage_repositories(
        [RepoCandidate(name="gen", full_name="org/gen", url="https://github.com/org/gen")],
        GenericGenerationGitHub(),
    )

    assert "inference" not in triaged[0].inferred_capabilities


def test_generic_runner_script_does_not_trigger_demo() -> None:
    class GenericRunnerGitHub(DummyGitHub):
        def get_readme(self, full_name: str) -> str | None:
            return "Run training and benchmark scripts."

        def get_repo_file_paths(self, full_name: str, default_branch: str = "main") -> list[str]:
            return ["train.py", "scripts/run.py", "eval.py"]

    triaged, _ = triage_repositories(
        [RepoCandidate(name="runner", full_name="org/runner", url="https://github.com/org/runner")],
        GenericRunnerGitHub(),
    )

    assert "demo" not in triaged[0].inferred_capabilities


def test_training_eval_path_is_not_automatically_weak() -> None:
    class ResearchGitHub(DummyGitHub):
        def get_readme(self, full_name: str) -> str | None:
            return "Train the model, then benchmark it."

        def get_repo_file_paths(self, full_name: str, default_branch: str = "main") -> list[str]:
            return ["requirements.txt", "train.py", "eval.py", "scripts/run_eval.sh"]

    triaged, _ = triage_repositories(
        [RepoCandidate(name="research", full_name="org/research", url="https://github.com/org/research")],
        ResearchGitHub(),
    )

    assert triaged[0].readiness_label in {"moderate", "strong"}


def test_smoke_test_path_is_not_automatically_weak() -> None:
    class SmokeGitHub(DummyGitHub):
        def get_readme(self, full_name: str) -> str | None:
            return "Quickstart sanity check for the released model."

        def get_repo_file_paths(self, full_name: str, default_branch: str = "main") -> list[str]:
            return ["requirements.txt", "smoke_test.py"]

    triaged, _ = triage_repositories(
        [RepoCandidate(name="smoke", full_name="org/smoke", url="https://github.com/org/smoke")],
        SmokeGitHub(),
    )

    assert triaged[0].readiness_label in {"moderate", "strong"}


def test_no_meaningful_runnable_path_cannot_stay_moderate() -> None:
    score, label = _compute_readiness(True, 2, 0, 0, 0.6, 10, False, 120)

    assert score < 3.5
    assert label == "weak"


def test_triage_attaches_execution_path_signals() -> None:
    class ExecutionPathGitHub(DummyGitHub):
        def get_readme(self, full_name: str) -> str | None:
            return """
            $ pip install -r requirements.txt
            Dataset must be downloaded manually.
            python train.py
            """

        def get_repo_file_paths(self, full_name: str, default_branch: str = "main") -> list[str]:
            return ["requirements.txt", "train.py"]

    triaged, _ = triage_repositories(
        [RepoCandidate(name="exec", full_name="org/exec", url="https://github.com/org/exec")],
        ExecutionPathGitHub(),
    )

    repo = triaged[0]
    assert "install" in repo.execution_steps
    assert "run" in repo.execution_steps
    assert "dataset must be supplied manually" in repo.missing_prerequisites
