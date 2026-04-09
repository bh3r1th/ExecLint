from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execlint.clients.arxiv_client import normalize_arxiv_input
from execlint.models import ExecutionInput
from execlint.orchestrator import audit_execution_input

EXPECTED_VERDICTS = {"clean": "GO", "messy": "CAUTION", "broken": "NO-GO"}
ADJACENT = {
    ("GO", "CAUTION"),
    ("CAUTION", "GO"),
    ("CAUTION", "NO-GO"),
    ("NO-GO", "CAUTION"),
}


def load_eval_entries(path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def classify(expected: str, actual: str) -> str:
    if actual == expected:
        return "PASS"
    if (expected, actual) in ADJACENT:
        return "SOFT_PASS"
    return "FAIL"


def main() -> int:
    dataset_path = ROOT / "tests" / "data" / "paper_repo_eval.jsonl"
    entries = load_eval_entries(dataset_path)

    total = len(entries)
    pass_count = 0
    soft_pass_count = 0
    fail_count = 0

    for entry in entries:
        name = entry.get("name", "<unknown>")
        expected_bucket = entry.get("expected_bucket", "")
        expected = EXPECTED_VERDICTS.get(expected_bucket, "UNKNOWN")

        actual = "ERROR"
        try:
            report, _warnings = audit_execution_input(
                ExecutionInput(arxiv_url=entry["arxiv_url"], repo_url=entry["repo_url"])
            )
            actual = report.verdict
            _ = report.tthw
        except Exception as exc:
            normalized_arxiv_id = "unknown"
            try:
                normalized_arxiv_id, _ = normalize_arxiv_input(entry["arxiv_url"])
            except Exception:
                pass
            print(f"{name} | {expected} | ERROR | FAIL | arxiv_id={normalized_arxiv_id} ({exc})")
            fail_count += 1
            continue

        result = classify(expected, actual)
        if result == "PASS":
            pass_count += 1
        elif result == "SOFT_PASS":
            soft_pass_count += 1
        else:
            fail_count += 1

        print(f"{name} | {expected} | {actual} | {result}")

    accuracy = (pass_count / total * 100.0) if total else 0.0
    relaxed_accuracy = ((pass_count + soft_pass_count) / total * 100.0) if total else 0.0

    print(f"total: {total}")
    print(f"PASS: {pass_count}")
    print(f"SOFT_PASS: {soft_pass_count}")
    print(f"FAIL: {fail_count}")
    print(f"accuracy: {accuracy:.2f}%")
    print(f"relaxed_accuracy: {relaxed_accuracy:.2f}%")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
