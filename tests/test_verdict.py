from __future__ import annotations

import json
from pathlib import Path

import pytest

from execlint.analyzers.verdict import build_execution_report
from execlint.models import HFModelStatus, IssueFixSignal, RepoCandidate

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "verdict"


def _load_case(name: str) -> dict:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text())


@pytest.mark.parametrize(
    "fixture_name",
    [
        "clear_go",
        "clear_caution",
        "clear_no_go",
        "weak_repo_usable_fix_path",
        "missing_weights_downgrade",
    ],
)
def test_verdict_golden_fixtures(fixture_name: str) -> None:
    fixture = _load_case(fixture_name)

    candidates = [RepoCandidate(**candidate) for candidate in fixture["candidates"]]
    issue_signals_by_repo = {
        repo_name: [IssueFixSignal(**signal) for signal in signals]
        for repo_name, signals in fixture["issue_signals_by_repo"].items()
    }
    hf_status = HFModelStatus(**fixture["hf_status"])

    report = build_execution_report(
        candidates=candidates,
        issue_signals_by_repo=issue_signals_by_repo,
        hf_status=hf_status,
    )

    assert report.verdict == fixture["expected"]["verdict"]
    assert report.tthw == fixture["expected"]["tthw"]
    assert report.best_repo == fixture["expected"]["best_repo"]


def test_stronger_repo_beats_weak_repo_despite_weak_repo_momentum() -> None:
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
        issue_signals_by_repo={
            "org/weak": [
                IssueFixSignal(
                    blocker="Dependency conflict",
                    fix="Pin package versions",
                    confidence="low",
                    blocker_category="dependency",
                )
            ],
            "org/strong": [
                IssueFixSignal(
                    blocker="Minor setup note",
                    fix="Install optional extras",
                    confidence="low",
                    blocker_category="install",
                )
            ],
        },
        hf_status=HFModelStatus(status="found", model_id="org/model"),
    )

    assert report.best_repo == "https://github.com/org/strong"


def test_weak_repo_wins_only_when_only_credible_fix_path() -> None:
    weak_repo = RepoCandidate(
        name="weak",
        full_name="org/weak",
        url="https://github.com/org/weak",
        readiness_label="weak",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml", "setup.py"],
        entrypoint_signals=["train.py", "infer.py", "demo.py"],
        discovery_score=80,
    )
    moderate_repo = RepoCandidate(
        name="moderate",
        full_name="org/moderate",
        url="https://github.com/org/moderate",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["train.py", "infer.py"],
        discovery_score=70,
    )

    report = build_execution_report(
        candidates=[moderate_repo, weak_repo],
        issue_signals_by_repo={
            "org/moderate": [
                IssueFixSignal(
                    blocker="Missing model weights and no runnable path",
                    fix="",
                    confidence="high",
                    blocker_category="missing assets",
                )
            ],
            "org/weak": [
                IssueFixSignal(
                    blocker="CUDA mismatch",
                    fix="Set torch cuda version to 12.1 and pin wheel",
                    confidence="medium",
                    blocker_category="cuda",
                )
            ],
        },
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.best_repo == "https://github.com/org/weak"


def test_no_fix_weak_repo_missing_weights_is_no_go() -> None:
    weak_repo = RepoCandidate(
        name="weak",
        full_name="org/weak",
        url="https://github.com/org/weak",
        readiness_label="weak",
        has_readme=True,
        setup_signals=["requirements.txt"],
        entrypoint_signals=[],
    )

    report = build_execution_report(
        candidates=[weak_repo],
        issue_signals_by_repo={
            "org/weak": [
                IssueFixSignal(
                    blocker="Missing weights and cannot run inference",
                    fix="",
                    confidence="high",
                    blocker_category="weights missing",
                )
            ]
        },
        hf_status=HFModelStatus(status="not_found"),
    )

    assert report.verdict == "NO-GO"
    assert report.tthw == "Level 4"
    assert report.what_breaks == "Missing weights and cannot run inference; checkpoint link absent"


def test_moderate_repo_medium_blocker_with_fix_is_caution() -> None:
    moderate_repo = RepoCandidate(
        name="moderate",
        full_name="org/moderate",
        url="https://github.com/org/moderate",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["train.py"],
    )

    report = build_execution_report(
        candidates=[moderate_repo],
        issue_signals_by_repo={
            "org/moderate": [
                IssueFixSignal(
                    blocker="API drift after dependency update",
                    fix="Pin transformers version and update callsite",
                    confidence="medium",
                    blocker_category="api drift",
                )
            ]
        },
        hf_status=HFModelStatus(status="found", model_id="org/model"),
    )

    assert report.verdict == "CAUTION"
    assert report.tthw in {"Level 2", "Level 3"}
    assert report.what_breaks == "API drift after dependency update"


def test_strong_repo_missing_weights_with_credible_path_not_no_go() -> None:
    strong_repo = RepoCandidate(
        name="strong",
        full_name="org/strong",
        url="https://github.com/org/strong",
        readiness_label="strong",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["infer.py"],
        inferred_capabilities=["inference"],
    )

    report = build_execution_report(
        candidates=[strong_repo],
        issue_signals_by_repo={},
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.verdict == "CAUTION"
    assert report.tthw == "Level 2"
    assert report.runnable_for == "inference"
    assert report.fix == "No clear fix found"


def test_weak_repo_with_fix_and_gated_weights_is_caution() -> None:
    weak_repo = RepoCandidate(
        name="weak",
        full_name="org/weak",
        url="https://github.com/org/weak",
        readiness_label="weak",
        has_readme=True,
        setup_signals=["requirements.txt", "README usage"],
        entrypoint_signals=["run.py"],
    )

    report = build_execution_report(
        candidates=[weak_repo],
        issue_signals_by_repo={
            "org/weak": [
                IssueFixSignal(
                    blocker="Weights are gated",
                    fix="Request access and use provided token to download",
                    confidence="medium",
                    blocker_category="weights",
                )
            ]
        },
        hf_status=HFModelStatus(status="not_found", gated=True),
    )

    assert report.verdict == "CAUTION"
    assert report.tthw == "Level 3"


def test_no_repo_no_weights_is_no_go_level_4() -> None:
    report = build_execution_report(
        candidates=[],
        issue_signals_by_repo={},
        hf_status=HFModelStatus(status="not_found"),
    )

    assert report.verdict == "NO-GO"
    assert report.tthw == "Level 4"
    assert report.fix == "Unavailable: no repository data"
    assert report.hf_status == "Hugging Face model missing"


def test_moderate_repo_version_pin_fix_stays_caution() -> None:
    moderate_repo = RepoCandidate(
        name="moderate",
        full_name="org/mod",
        url="https://github.com/org/mod",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["train.py"],
    )
    report = build_execution_report(
        candidates=[moderate_repo],
        issue_signals_by_repo={
            "org/mod": [
                IssueFixSignal(
                    blocker="CUDA and version mismatch after upstream change",
                    fix="Pin torch/cu121 wheels and set transformers==4.39",
                    confidence="medium",
                    blocker_category="cuda version",
                )
            ]
        },
        hf_status=HFModelStatus(status="found", model_id="org/model"),
    )

    assert report.verdict == "CAUTION"
    assert report.tthw in {"Level 2", "Level 3"}


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

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/demo": []},
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.hf_status == "Hugging Face status unclear"
    assert "checkpoint provenance unclear" not in report.technical_debt


def test_weak_high_discovery_repo_still_does_not_force_go() -> None:
    weak_repo = RepoCandidate(
        name="paper-code",
        full_name="authors/paper-code",
        url="https://github.com/authors/paper-code",
        readiness_label="weak",
        discovery_score=1000,
        has_readme=False,
        setup_signals=[],
        entrypoint_signals=[],
    )

    report = build_execution_report(
        candidates=[weak_repo],
        issue_signals_by_repo={"authors/paper-code": []},
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.verdict != "GO"
    assert report.tthw == "Level 4"


def test_provided_weights_reduces_no_go_risk() -> None:
    repo = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["infer.py"],
        inferred_capabilities=["inference"],
    )

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={
            "org/demo": [
                IssueFixSignal(
                    blocker="Missing weights for inference",
                    fix="",
                    confidence="high",
                    blocker_category="weights",
                )
            ]
        },
        hf_status=HFModelStatus(status="found", model_id="https://weights.example/model.bin", notes="User-provided weights URL"),
        weights_url="https://weights.example/model.bin",
    )

    assert report.verdict == "CAUTION"
    assert report.hf_status == "User-provided weights"
    assert "weights/checkpoints not yet pinned down" not in report.technical_debt


def test_smoke_test_only_repo_reflects_limited_reality() -> None:
    repo = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["smoke_test.py"],
        inferred_capabilities=["smoke_test"],
    )

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/demo": []},
        hf_status=HFModelStatus(status="found", model_id="org/model"),
    )

    assert report.verdict == "CAUTION"
    assert report.technical_debt == "None identified"
    assert report.not_clearly_supported == "demo, inference, training, evaluation"


def test_missing_ref_adds_technical_debt_but_not_immediate_failure() -> None:
    repo = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        readiness_label="strong",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["infer.py"],
        inferred_capabilities=["inference"],
        pushed_at="2026-03-01T00:00:00Z",
    )

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/demo": []},
        hf_status=HFModelStatus(status="found", model_id="org/model"),
        ref=None,
    )

    assert report.verdict != "NO-GO"
    assert "stale branch ambiguity" not in report.technical_debt


def test_research_workflow_repo_is_not_auto_no_go() -> None:
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

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/research": []},
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.verdict == "CAUTION"
    assert report.runnable_for == "training, evaluation"
    assert "no obvious runnable entrypoint for any meaningful capability" not in report.what_breaks
    assert report.not_clearly_supported == "demo, inference"


def test_no_direct_inference_or_demo_does_not_force_no_go() -> None:
    repo = RepoCandidate(
        name="eval",
        full_name="org/eval",
        url="https://github.com/org/eval",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt"],
        entrypoint_signals=["scripts/"],
        inferred_capabilities=["evaluation"],
    )

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/eval": []},
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.verdict != "NO-GO"
    assert report.runnable_for == "evaluation"


def test_missing_weights_only_appears_when_relevant() -> None:
    repo = RepoCandidate(
        name="train-only",
        full_name="org/train-only",
        url="https://github.com/org/train-only",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt"],
        entrypoint_signals=["train.py"],
        inferred_capabilities=["training"],
    )

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/train-only": []},
        hf_status=HFModelStatus(status="not_found"),
    )

    assert "weights/checkpoints not yet pinned down" not in report.technical_debt
    assert report.what_breaks != "Weights/checkpoints not visible"


def test_technical_debt_is_conditional_not_templated() -> None:
    repo = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        readiness_label="strong",
        has_readme=True,
        setup_signals=["requirements.txt", "pyproject.toml"],
        entrypoint_signals=["demo.py"],
        inferred_capabilities=["demo"],
    )

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/demo": []},
        hf_status=HFModelStatus(status="found", model_id="org/model"),
        ref="v1.0.0",
    )

    assert report.technical_debt == "None identified"


def test_smoke_train_eval_path_is_caution_not_no_go() -> None:
    repo = RepoCandidate(
        name="workflow",
        full_name="org/workflow",
        url="https://github.com/org/workflow",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt"],
        entrypoint_signals=["train.py", "eval.py", "smoke_test.py"],
        inferred_capabilities=["training", "evaluation", "smoke_test"],
    )

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/workflow": []},
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.verdict == "CAUTION"
    assert "no obvious runnable entrypoint for any meaningful capability" not in report.what_breaks


def test_runnable_for_present_never_claims_no_meaningful_entrypoint() -> None:
    repo = RepoCandidate(
        name="research",
        full_name="org/research",
        url="https://github.com/org/research",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt"],
        entrypoint_signals=["train.py", "eval.py"],
        inferred_capabilities=["training", "evaluation"],
    )

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/research": []},
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.runnable_for == "training, evaluation"
    assert "no runnable entrypoint for any meaningful capability" not in report.what_breaks
    assert "no obvious runnable entrypoint for any meaningful capability" not in report.what_breaks


def test_missing_demo_inference_alone_does_not_force_high_severity_wording() -> None:
    repo = RepoCandidate(
        name="research",
        full_name="org/research",
        url="https://github.com/org/research",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt"],
        entrypoint_signals=["train.py", "scripts/"],
        inferred_capabilities=["training", "evaluation"],
    )

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/research": []},
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.verdict == "CAUTION"
    assert report.what_breaks != "no obvious runnable entrypoint for any meaningful capability"


def test_blockers_trigger_nontrivial_severity_and_precise_debt() -> None:
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

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/demo": []},
        hf_status=HFModelStatus(status="not_found"),
    )

    assert report.what_breaks == "install path ambiguous; no clear runnable entrypoint"
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

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/infer": []},
        hf_status=HFModelStatus(status="not_found"),
    )

    assert report.what_breaks == "checkpoint link absent"
    assert "checkpoint provenance unclear" in report.technical_debt


def test_generic_fix_filler_is_not_emitted() -> None:
    repo = RepoCandidate(
        name="train",
        full_name="org/train",
        url="https://github.com/org/train",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt"],
        entrypoint_signals=["train.py"],
        inferred_capabilities=["training"],
    )

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/train": []},
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.fix == "No clear fix found"


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
        issue_signals_by_repo={"org/demo": []},
        hf_status=HFModelStatus(status="found", model_id="org/model"),
    )

    assert report.execution_path == "install: pip install -r requirements.txt; run: python train.py; evaluate: python eval.py"
    assert report.gaps == "None identified"
    assert report.tthw == "Level 1"


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

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/demo": []},
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.execution_path == "No extracted execution commands"
    assert report.gaps == "install path ambiguous; no clear run command"
    assert report.tthw == "Level 3"


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

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/demo": []},
        hf_status=HFModelStatus(status="unknown"),
    )

    assert report.verdict == "CAUTION"
    assert report.what_breaks == "dataset must be supplied manually"
    assert report.technical_debt == "dataset bootstrap manual"


def test_execution_sequence_lowers_tthw() -> None:
    repo = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        readiness_label="moderate",
        has_readme=True,
        setup_signals=["requirements.txt"],
        entrypoint_signals=["run.py", "eval.py"],
        inferred_capabilities=["inference", "evaluation"],
        execution_steps={
            "install": ["pip install -r requirements.txt"],
            "run": ["python run.py"],
            "evaluate": ["python eval.py"],
        },
    )
    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/demo": []},
        hf_status=HFModelStatus(status="unknown"),
    )
    assert report.tthw == "Level 1"


def test_generic_filler_not_emitted_when_execution_sequence_exists() -> None:
    repo = RepoCandidate(
        name="demo",
        full_name="org/demo",
        url="https://github.com/org/demo",
        readiness_label="strong",
        has_readme=True,
        setup_signals=["requirements.txt"],
        entrypoint_signals=["run.py"],
        inferred_capabilities=["inference"],
        execution_steps={"install": ["pip install -r requirements.txt"], "run": ["python run.py"]},
        gaps=["no clear run command"],
    )
    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/demo": []},
        hf_status=HFModelStatus(status="found", model_id="org/model"),
    )
    assert "No concrete blocker visible" not in report.what_breaks
    assert "no obvious runnable entrypoint" not in report.what_breaks
    assert report.what_breaks == "no clear run command"


def test_tthw_escalates_to_level_4_for_heavy_setup_signals() -> None:
    repo = RepoCandidate(
        name="af2",
        full_name="org/af2",
        url="https://github.com/org/af2",
        readiness_label="strong",
        has_readme=True,
        setup_signals=["scripts/download_database.sh", "docker-compose.yml"],
        entrypoint_signals=["run_alphafold.py"],
        inferred_capabilities=["inference"],
        execution_steps={
            "install": ["docker build -t af2 ."],
            "setup_data": ["bash scripts/download_database.sh"],
            "run": ["python run_alphafold.py --db-size 2TB"],
        },
    )

    report = build_execution_report(
        candidates=[repo],
        issue_signals_by_repo={"org/af2": []},
        hf_status=HFModelStatus(status="found", model_id="org/model"),
    )

    assert report.tthw == "Level 4"
