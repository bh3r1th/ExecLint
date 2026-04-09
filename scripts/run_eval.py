from __future__ import annotations

import json
from pathlib import Path

from execlint.models import ExecutionInput
from execlint.orchestrator import audit_execution_input


def load_eval_entries(path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def run_eval(dataset_path: Path) -> None:
    for entry in load_eval_entries(dataset_path):
        name = entry["name"]
        expected_bucket = entry["expected_bucket"]
        print(f"name: {name}")
        print(f"expected_bucket: {expected_bucket}")
        try:
            report, warnings = audit_execution_input(
                ExecutionInput(
                    arxiv_url=entry["arxiv_url"],
                    repo_url=entry["repo_url"],
                )
            )
            print(f"actual_verdict: {report.verdict}")
            print(f"TTHW: {report.tthw}")
            if warnings:
                print(f"warnings: {'; '.join(warnings)}")
        except Exception as exc:
            print("actual_verdict: ERROR")
            print("TTHW: ERROR")
            print(f"error: {exc}")
        print("-" * 40)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_path = root / "tests" / "data" / "paper_repo_eval.jsonl"
    run_eval(dataset_path)


if __name__ == "__main__":
    main()
