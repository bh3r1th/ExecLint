"""Build eval/results/v2_summary.md from the most recent v2_run_*.jsonl."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
CATEGORIES = ["install", "data", "weights", "run", "eval", "env"]


def find_latest_jsonl() -> Path:
    runs = sorted(RESULTS_DIR.glob("v2_run_*.jsonl"))
    if not runs:
        raise SystemExit("No v2_run_*.jsonl files found in eval/results/")
    return runs[-1]


def main() -> int:
    src = find_latest_jsonl()
    records = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = len(records)

    total_runtime = sum((r.get("duration_seconds") or 0.0) for r in records)
    errored = [r for r in records if r.get("error")]
    completed = [r for r in records if not r.get("error")]

    # Histogram of gap counts (only for non-errored runs)
    gap_count_per_paper = []
    for r in completed:
        gaps = (r.get("report") or {}).get("gaps") or []
        gap_count_per_paper.append(len(gaps))
    histogram = Counter()
    for c in gap_count_per_paper:
        bucket = "5+" if c >= 5 else str(c)
        histogram[bucket] += 1

    # Per-category: count distinct papers having >=1 gap in that category
    cat_paper_counts: dict[str, int] = {c: 0 for c in CATEGORIES}
    cat_evidence: dict[str, Counter] = {c: Counter() for c in CATEGORIES}
    for r in completed:
        gaps = (r.get("report") or {}).get("gaps") or []
        seen_cats = set()
        for g in gaps:
            cat = g.get("category")
            ev = (g.get("evidence") or "").strip()
            if cat in cat_paper_counts:
                if cat not in seen_cats:
                    cat_paper_counts[cat] += 1
                    seen_cats.add(cat)
                if ev:
                    cat_evidence[cat][ev] += 1

    # Build markdown
    lines: list[str] = []
    lines.append("# ExecLint v2 Evaluation Summary")
    lines.append("")
    lines.append(f"Source: `{src.name}`")
    lines.append(f"Papers in dataset: {n}")
    lines.append("")

    lines.append("## 1. Run Statistics")
    lines.append("")
    lines.append(f"- Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    lines.append(f"- Errors: {len(errored)}")
    lines.append(f"- Completed cleanly: {len(completed)}")
    lines.append("")

    lines.append("## 2. Gap-Count Histogram")
    lines.append("")
    lines.append("| gaps | papers |")
    lines.append("|------|--------|")
    for bucket in ["0", "1", "2", "3", "4", "5+"]:
        lines.append(f"| {bucket}    | {histogram.get(bucket, 0)} |")
    lines.append("")
    lines.append(f"(across {len(completed)} papers that completed without error)")
    lines.append("")

    lines.append("## 3. Per-Category Gap Frequency")
    lines.append("")
    lines.append("| category | count | % of papers |")
    lines.append("|----------|-------|-------------|")
    denom = len(completed) if completed else 1
    for cat in CATEGORIES:
        count = cat_paper_counts[cat]
        pct = 100.0 * count / denom
        lines.append(f"| {cat:<8} | {count:5d} | {pct:5.1f}% |")
    lines.append("")
    lines.append(f"(% computed against {len(completed)} clean-completion papers)")
    lines.append("")

    lines.append("## 4. Top 3 Evidence Strings per Category")
    lines.append("")
    for cat in CATEGORIES:
        lines.append(f"### {cat}")
        top = cat_evidence[cat].most_common(3)
        if not top:
            lines.append("- (no gaps observed)")
        else:
            for ev, count in top:
                ev_one_line = ev.replace("\n", " ").replace("|", "\\|")
                lines.append(f"- ({count}x) {ev_one_line}")
        lines.append("")

    lines.append("## 5. Errored Papers")
    lines.append("")
    if not errored:
        lines.append("None.")
    else:
        for r in errored:
            err = r.get("error") or {}
            lines.append(f"- **{r.get('arxiv_id','?')} — {r.get('title','?')}**")
            lines.append(f"  - error class: `{err.get('class','?')}`")
            msg = (err.get("message") or "").replace("\n", " ")
            lines.append(f"  - message: {msg}")
    lines.append("")

    out = RESULTS_DIR / "v2_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
