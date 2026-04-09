from execlint.analyzers.execution_path import analyze_execution_path


def test_extracts_install_command_from_readme() -> None:
    analysis = analyze_execution_path(
        readme_text="""
        ## Setup
        $ pip install -r requirements.txt
        """,
        paths=[],
    )

    assert "install" in analysis.execution_steps
    assert "pip install -r requirements.txt" in analysis.execution_steps["install"]


def test_extracts_run_command_from_readme() -> None:
    analysis = analyze_execution_path(
        readme_text="""
        Run with:
        python run_infer.py --prompt hello
        """,
        paths=[],
    )

    assert "run" in analysis.execution_steps
    assert "python run_infer.py --prompt hello" in analysis.execution_steps["run"]


def test_extracts_eval_command_from_readme() -> None:
    analysis = analyze_execution_path(
        readme_text="""
        Evaluate:
        bash scripts/eval.sh --split test
        """,
        paths=[],
    )

    assert "evaluate" in analysis.execution_steps
    assert "bash scripts/eval.sh --split test" in analysis.execution_steps["evaluate"]


def test_detects_dataset_manual_setup_gap() -> None:
    analysis = analyze_execution_path(
        readme_text="""
        Dataset must be downloaded manually and placed in data/.
        $ pip install -r requirements.txt
        python train.py
        """,
        paths=[],
    )

    assert "dataset must be supplied manually" in analysis.missing_prerequisites
    assert "dataset must be supplied manually" in analysis.gaps


def test_detects_no_clear_run_command_gap() -> None:
    analysis = analyze_execution_path(
        readme_text="$ pip install -r requirements.txt",
        paths=["requirements.txt", "eval.py"],
    )

    assert "no clear run command" in analysis.gaps
