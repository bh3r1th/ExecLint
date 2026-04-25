"""Batch-run ExecLint over eval/dataset_v2.jsonl, capturing per-paper
ExecutionReport JSON, duration, and any error class/message.

Output: one JSON line per paper to eval/results/v2_run_<YYYYMMDD>.jsonl
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from execlint.models import ExecutionInput  # noqa: E402
from execlint.orchestrator import audit_execution_input_with_debug  # noqa: E402
DATASET = ROOT / "eval" / "dataset_v2.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = RESULTS_DIR / f"v2_run_{today}.jsonl"

    entries = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Running ExecLint on {len(entries)} papers -> {out_path}", flush=True)

    with out_path.open("w", encoding="utf-8") as out:
        for idx, entry in enumerate(entries, start=1):
            arxiv_url = entry["arxiv_url"]
            repo_url = entry["repo_url"]
            arxiv_id = entry.get("arxiv_id", "")
            title = entry.get("title", "")

            record = {
                "arxiv_id": arxiv_id,
                "arxiv_url": arxiv_url,
                "repo_url": repo_url,
                "title": title,
                "duration_seconds": None,
                "error": None,
                "report": None,
                "warnings": [],
                "debug_signals": None,
            }

            t0 = monotonic()
            try:
                exec_input = ExecutionInput(arxiv_url=arxiv_url, repo_url=repo_url)
                report, warnings, debug = audit_execution_input_with_debug(exec_input)
                record["report"] = report.model_dump(mode="json")
                record["warnings"] = list(warnings)
                # debug may contain non-serializable bits; coerce
                record["debug_signals"] = json.loads(json.dumps(debug, default=str))
            except BaseException as exc:
                record["error"] = {
                    "class": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=4),
                }
            finally:
                record["duration_seconds"] = round(monotonic() - t0, 3)

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            status = "ERR" if record["error"] else "OK "
            gap_count = len((record["report"] or {}).get("gaps", []))
            print(
                f"[{idx:02d}/{len(entries)}] {status} {arxiv_id} "
                f"gaps={gap_count} dur={record['duration_seconds']}s "
                f"{('err='+record['error']['class']) if record['error'] else ''}",
                flush=True,
            )

    print(f"Done. Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
