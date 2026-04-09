from __future__ import annotations

import re

from pydantic import BaseModel, Field

STEP_ORDER = ("install", "setup_data", "setup_weights", "run", "evaluate")
COMMAND_PATTERN = re.compile(
    r"(?im)^\s*(?:\$|>\s*)?\s*((?:pip\s+install|conda\s+create|python\s+\S+\.py|bash\s+\S+\.sh|make(?:\s+\S+)?|docker(?:\s+\S+)+)[^\n]*)"
)
MANUAL_DATA_PATTERN = re.compile(
    r"(?i)\b(dataset|data)\b.*\b(download|provide|place|put|copy|prepare|manual|manually)\b|\b(manual|manually)\b.*\b(dataset|data)\b"
)
MANUAL_WEIGHTS_PATTERN = re.compile(
    r"(?i)\b(weights?|checkpoints?)\b.*\b(download|provide|place|put|copy|manual|manually|obtain)\b|\b(manual|manually)\b.*\b(weights?|checkpoints?)\b"
)
ENV_VERSION_PATTERN = re.compile(r"(?i)\b(python|cuda|pytorch|torch)\b[^.\n]{0,40}\b(>=|<=|==|~=|\d+\.\d+)")


class ExecutionPathAnalysis(BaseModel):
    execution_steps: dict[str, list[str]] = Field(default_factory=dict)
    missing_prerequisites: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


def analyze_execution_path(readme_text: str, paths: list[str]) -> ExecutionPathAnalysis:
    normalized_steps: dict[str, list[str]] = {step: [] for step in STEP_ORDER}
    evidence_text = (readme_text or "").strip()

    commands = _extract_commands(evidence_text)
    for command in commands:
        normalized_steps[_classify_command(command)].append(command)

    for inferred in _infer_steps_from_paths(paths):
        normalized_steps[inferred[0]].append(inferred[1])

    for step in STEP_ORDER:
        normalized_steps[step] = sorted(dict.fromkeys(normalized_steps[step]))

    gaps: list[str] = []
    missing_prerequisites: list[str] = []
    lowered = evidence_text.lower()
    has_install_command = bool(normalized_steps["install"])
    has_run_command = bool(normalized_steps["run"])
    has_eval_command = bool(normalized_steps["evaluate"])
    mentions_data = bool(re.search(r"(?i)\b(dataset|data)\b", evidence_text))
    mentions_weights = bool(re.search(r"(?i)\b(weights?|checkpoints?)\b", evidence_text))
    has_env_versions = bool(ENV_VERSION_PATTERN.search(evidence_text))

    if not has_install_command:
        gaps.append("install path ambiguous")
    if mentions_data and (MANUAL_DATA_PATTERN.search(evidence_text) or "dataset" in lowered and "http" not in lowered):
        missing_prerequisites.append("dataset must be supplied manually")
        gaps.append("dataset must be supplied manually")
    if mentions_weights and (MANUAL_WEIGHTS_PATTERN.search(evidence_text) or "checkpoint" in lowered and "http" not in lowered):
        missing_prerequisites.append("weights/checkpoints not linked")
        gaps.append("weights/checkpoints not linked")
    if not has_run_command:
        gaps.append("no clear run command")
    if mentions_weights and not has_run_command and not has_eval_command:
        gaps.append("no clear run command")
    if not has_env_versions:
        gaps.append("env version unclear")

    deduped_steps = {step: commands for step, commands in normalized_steps.items() if commands}
    return ExecutionPathAnalysis(
        execution_steps=deduped_steps,
        missing_prerequisites=sorted(dict.fromkeys(missing_prerequisites)),
        gaps=sorted(dict.fromkeys(gaps)),
    )


def _extract_commands(readme_text: str) -> list[str]:
    return [match.group(1).strip() for match in COMMAND_PATTERN.finditer(readme_text or "")]


def _classify_command(command: str) -> str:
    lowered = command.lower()
    if lowered.startswith(("pip install", "conda create", "docker build", "docker pull")):
        return "install"
    if lowered.startswith("make"):
        if any(token in lowered for token in ("eval", "benchmark", "test")):
            return "evaluate"
        if any(token in lowered for token in ("run", "infer", "demo", "serve")):
            return "run"
        if any(token in lowered for token in ("data", "dataset", "download")):
            return "setup_data"
        return "install"
    if lowered.startswith("docker"):
        if any(token in lowered for token in ("run", "up")):
            return "run"
        return "install"
    if lowered.startswith("python"):
        if any(token in lowered for token in ("eval", "evaluate", "benchmark", "metric", "test")):
            return "evaluate"
        if any(token in lowered for token in ("download", "prepare", "dataset", "data")):
            return "setup_data"
        if any(token in lowered for token in ("weight", "checkpoint")):
            return "setup_weights"
        return "run"
    if lowered.startswith("bash"):
        if any(token in lowered for token in ("eval", "benchmark", "test")):
            return "evaluate"
        if any(token in lowered for token in ("download", "prepare", "dataset", "data")):
            return "setup_data"
        if any(token in lowered for token in ("weight", "checkpoint")):
            return "setup_weights"
        return "run"
    return "run"


def _infer_steps_from_paths(paths: list[str]) -> list[tuple[str, str]]:
    inferred: list[tuple[str, str]] = []
    lowered_paths = [path.lower() for path in paths]

    if any(path.endswith(("requirements.txt", "pyproject.toml", "environment.yml", "setup.py")) for path in lowered_paths):
        inferred.append(("install", "inferred from repo files"))

    script_paths = [path for path in lowered_paths if path.startswith("scripts/")]
    for path in script_paths:
        filename = path.split("/")[-1]
        if re.search(r"(eval|benchmark|metric|test)", filename):
            inferred.append(("evaluate", f"bash {path}"))
        elif re.search(r"(download|prepare|dataset|data)", filename):
            inferred.append(("setup_data", f"bash {path}"))
        elif re.search(r"(weight|checkpoint)", filename):
            inferred.append(("setup_weights", f"bash {path}"))
        elif filename.endswith(".sh"):
            inferred.append(("run", f"bash {path}"))
        elif filename.endswith(".py"):
            inferred.append(("run", f"python {path}"))

    for path in lowered_paths:
        filename = path.split("/")[-1]
        if filename in {"eval.py", "evaluate.py"} or re.search(r"(eval|benchmark|metric).*\.py$", filename):
            inferred.append(("evaluate", f"python {path}"))
        elif filename.endswith(".py") and re.search(r"(train|infer|inference|run|demo|app)", filename):
            inferred.append(("run", f"python {path}"))

    return inferred
