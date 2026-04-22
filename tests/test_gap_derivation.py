from __future__ import annotations

from execlint.analyzers.verdict import build_execution_report
from execlint.models import HFModelStatus, RepoCandidate


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
    )

    assert report.best_repo == "https://github.com/org/strong"


def test_no_repo_report_surfaces_repository_gap() -> None:
    report = build_execution_report(candidates=[], hf_status=HFModelStatus(status="not_found"))

    assert report.best_repo == "None found"
    assert report.gaps == "No repository candidate"
    assert report.what_breaks == "No repository candidate"
    assert report.fix == "Unavailable: no repository data"
    assert report.hf_status == "Hugging Face model missing"


def test_hf_unknown_wording_is_unclear_not_missing() -> None:
    repo = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        readiness_label="strong",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["run.py"],
    )

    report = build_execution_report(candidates=[repo], hf_status=HFModelStatus(status="unknown"))

    assert report.hf_status == "Hugging Face status unclear"
    assert "checkpoint provenance unclear" not in report.technical_debt


def test_provided_weights_suppresses_checkpoint_debt() -> None:
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

    assert report.hf_status == "User-provided weights"
    assert report.what_breaks == "No concrete blocker visible"
    assert "checkpoint provenance unclear" not in report.technical_debt


def test_research_workflow_support_gap_is_specific() -> None:
    repo = RepoCandidate(
        name="research",
        full_name="org/research",
        url="https://github.com/org/research",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["train.py", "scripts/"],
        inferred_capabilities=["training", "evaluation"],
    )

    report = build_execution_report(candidates=[repo], hf_status=HFModelStatus(status="unknown"))

    assert report.runnable_for == "training, evaluation"
    assert report.what_breaks == "No concrete blocker visible"
    assert report.not_clearly_supported == "demo, inference"


def test_blockers_derive_from_gaps_and_repo_facts() -> None:
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

    assert report.what_breaks == "install path ambiguous"
    assert report.technical_debt == "install path ambiguous"


def test_missing_weights_shown_only_when_repo_likely_depends_on_them() -> None:
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

    assert report.what_breaks == "checkpoint link absent"
    assert "checkpoint provenance unclear" in report.technical_debt


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
    assert report.gaps == "None identified"


def test_missing_execution_path_surfaces_clear_gap_text() -> None:
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
    assert report.gaps == "install path ambiguous; no clear run command"
    assert report.what_breaks == "install path ambiguous; no clear run command"


def test_caution_report_stays_evidence_backed() -> None:
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

    assert report.what_breaks == "dataset must be supplied manually"
    assert report.technical_debt == "dataset bootstrap manual"
