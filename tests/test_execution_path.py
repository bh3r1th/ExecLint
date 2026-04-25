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


def test_weights_mention_with_adjacent_hf_link_has_no_weights_gap() -> None:
    analysis = analyze_execution_path(
        readme_text="""
        ## Model
        Download pretrained weights from https://huggingface.co/org/model and place them in checkpoints/.
        pip install -r requirements.txt
        python demo.py
        """,
        paths=[],
    )

    assert "weights/checkpoints not linked" not in analysis.gaps


def test_weights_mention_with_unrelated_hf_link_still_has_weights_gap() -> None:
    analysis = analyze_execution_path(
        readme_text="""
        ## Assets
        Download pretrained weights manually and place them in checkpoints/.

        ## Citation
        See our unrelated dataset card at https://huggingface.co/datasets/org/other for details.
        pip install -r requirements.txt
        python demo.py
        """,
        paths=[],
    )

    assert "weights/checkpoints not linked" in analysis.gaps


def test_no_weights_mention_has_no_weights_gap() -> None:
    analysis = analyze_execution_path(
        readme_text="""
        pip install -r requirements.txt
        python demo.py
        For docs, visit https://huggingface.co/org/model.
        """,
        paths=[],
    )

    assert "weights/checkpoints not linked" not in analysis.gaps


def test_filters_non_command_lines_and_keeps_only_high_signal_commands() -> None:
    analysis = analyze_execution_path(
        readme_text="""
        ## Install
        Please install dependencies and then continue.
        pip install -r requirements.txt
        Visit https://example.com/docs for details.
        python run.py --config configs/demo.yaml
        Notes: this may take a while.
        """,
        paths=[],
    )

    flattened = " ".join(command for commands in analysis.execution_steps.values() for command in commands)
    assert "https://example.com/docs" not in flattened
    assert "notes:" not in flattened.lower()
    assert analysis.execution_steps["install"] == ["pip install -r requirements.txt"]
    assert analysis.execution_steps["run"] == ["python run.py --config configs/demo.yaml"]


def test_long_readme_command_block_gets_summarized() -> None:
    long_block = "\n".join([f"python step_{idx}.py" for idx in range(30)])
    analysis = analyze_execution_path(readme_text=long_block, paths=[])

    assert analysis.execution_steps["run"] == ["python step_0.py | python step_1.py"]


def test_alphafold_like_readme_produces_short_step_summary() -> None:
    analysis = analyze_execution_path(
        readme_text="""
        # AlphaFold style setup
        docker build -t af2 .
        bash scripts/download_data.sh
        bash scripts/download_weights.sh
        python run_alphafold.py --fasta_path target.fasta
        python eval.py --predictions out/
        python extra_debug.py
        """,
        paths=[],
    )

    assert list(analysis.execution_steps) == ["install", "setup_data", "setup_weights", "run", "evaluate"]
    assert analysis.execution_steps["run"] == ["python run_alphafold.py --fasta_path target.fasta | python extra_debug.py"]
