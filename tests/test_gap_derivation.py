from __future__ import annotations

from execlint.analyzers.verdict import build_execution_report
from execlint.models import HFModelStatus, RepoCandidate


def _gap_labels(report) -> list[str]:
    return [gap.label for gap in report.gaps]


def test_best_repo_selection_still_prefers_stronger_runnable_repo() -> None:
    weak_repo = RepoCandidate(
        name="weak",
        full_name="org/weak",
        url="https://github.com/org/weak",
        readiness_label="weak",
        stars=5000,
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml", "setup.py"],
        entrypoint_signals=["train.py", "infer.py", "demo.py"],
        discovery_score=120,
    )
    strong_repo = RepoCandidate(
        name="strong",
        full_name="org/strong",
        url="https://github.com/org/strong",
        readiness_label="strong",
        stars=100,
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["train.py", "infer.py"],
        discovery_score=60,
    )

    report = build_execution_report(
        candidates=[weak_repo, strong_repo],
        hf_status=HFModelStatus(status="found", model_id="org/model"),
        paper_title="Demo Paper",
    )

    assert report.paper_title == "Demo Paper"
    assert report.repo_url == "https://github.com/org/strong"


def test_no_repo_report_surfaces_repository_gap() -> None:
    report = build_execution_report(candidates=[], hf_status=HFModelStatus(status="not_found"))

    assert report.repo_url == "None found"
    assert _gap_labels(report) == ["No repository candidate"]
    assert report.gaps[0].category == "run"
    assert report.gaps[0].evidence == "No GitHub repository candidate was available to inspect"


def test_provided_weights_suppresses_checkpoint_gap() -> None:
    repo = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["infer.py", "checkpoint_loader.py"],
        inferred_capabilities=["inference"],
    )

    report = build_execution_report(
        candidates=[repo],
        hf_status=HFModelStatus(status="found", model_id="https://weights.example/model.bin", notes="User-provided weights URL"),
        weights_url="https://weights.example/model.bin",
    )

    assert "weights/checkpoints not linked" not in _gap_labels(report)


def test_install_gap_has_structured_category_and_evidence() -> None:
    repo = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        readiness_label="moderate",
        has_readme=False,
        setup_signals=[],
        entrypoint_signals=["infer.py"],
        inferred_capabilities=["inference"],
    )

    report = build_execution_report(candidates=[repo], hf_status=HFModelStatus(status="not_found"))

    assert _gap_labels(report) == ["install path ambiguous"]
    assert report.gaps[0].category == "install"
    assert report.gaps[0].evidence == "No setup.py, pyproject.toml, requirements.txt, or Dockerfile in repo root"


def test_missing_weights_gap_uses_new_label() -> None:
    repo = RepoCandidate(
        name="infer",
        full_name="org/infer",
        url="https://github.com/org/infer",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt"],
        entrypoint_signals=["infer.py", "checkpoint_loader.py"],
        inferred_capabilities=["inference"],
    )

    report = build_execution_report(candidates=[repo], hf_status=HFModelStatus(status="not_found"))

    assert _gap_labels(report) == ["weights/checkpoints not linked"]
    assert report.gaps[0].category == "weights"
    assert report.gaps[0].evidence == "README mentions weights/checkpoints but no download link found within 200 chars"


def test_execution_path_commands_are_exposed_in_report() -> None:
    repo = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        readiness_label="strong",
        has_readme=True,
        setup_signals=["requirements.txt"],
        entrypoint_signals=["train.py", "eval.py"],
        inferred_capabilities=["inference", "evaluation"],
        execution_steps={
            "install": ["pip install -r requirements.txt"],
            "run": ["python train.py"],
            "evaluate": ["python eval.py"],
        },
    )

    report = build_execution_report(
        candidates=[repo],
        hf_status=HFModelStatus(status="found", model_id="org/model"),
    )

    assert report.execution_path == "install: pip install -r requirements.txt; run: python train.py; evaluate: python eval.py"
    assert report.gaps == []


def test_missing_execution_path_surfaces_structured_gap_text() -> None:
    repo = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt"],
        entrypoint_signals=["train.py"],
        inferred_capabilities=["training"],
        execution_steps={},
        gaps=["install path ambiguous", "no clear run command"],
    )

    report = build_execution_report(candidates=[repo], hf_status=HFModelStatus(status="unknown"))

    assert report.execution_path == "No extracted execution commands"
    assert _gap_labels(report) == ["install path ambiguous", "no clear run command"]
    assert [gap.category for gap in report.gaps] == ["install", "run"]


def test_dataset_gap_has_structured_evidence() -> None:
    repo = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt"],
        entrypoint_signals=["run.py"],
        inferred_capabilities=["inference"],
        execution_steps={"install": ["pip install -r requirements.txt"], "run": ["python run.py"]},
        gaps=["dataset must be supplied manually"],
    )

    report = build_execution_report(candidates=[repo], hf_status=HFModelStatus(status="unknown"))

    assert _gap_labels(report) == ["dataset must be supplied manually"]
    assert report.gaps[0].category == "data"
    assert "dataset/data setup" in report.gaps[0].evidence
