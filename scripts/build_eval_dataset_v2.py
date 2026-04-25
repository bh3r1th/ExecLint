from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from xml.etree import ElementTree

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execlint.clients.arxiv_client import ArxivClient, normalize_arxiv_input
from execlint.config import REQUEST_TIMEOUT_SECONDS

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_QUERY = "(cat:cs.LG OR cat:cs.CL OR cat:cs.CV) AND submittedDate:[202601150000 TO 202602152359]"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_LIMIT = 40
PAGE_SIZE = 100
THROWAWAY_CONFIG = ROOT / "eval" / "throwaway_accounts.json"
OUTPUT_PATH = ROOT / "eval" / "dataset_v2.jsonl"
README_PATH = ROOT / "eval" / "dataset_README.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ExecLint eval v2 dataset from arXiv and GitHub.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--throwaway-config", type=Path, default=THROWAWAY_CONFIG)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _ensure_throwaway_config(args.throwaway_config)
    throwaway_patterns = _load_throwaway_patterns(args.throwaway_config)

    stats = build_dataset(
        limit=args.limit,
        output_path=args.output,
        throwaway_patterns=throwaway_patterns,
    )
    write_dataset_readme(README_PATH, args.output, args.throwaway_config)

    print(f"total_candidates_considered: {stats['considered']}")
    print(f"total_filtered_out: {sum(stats['filtered'].values())}")
    print("filter_counts:")
    for reason, count in sorted(stats["filtered"].items()):
        print(f"- {reason}: {count}")
    print(f"final_selected: {stats['selected']}")
    return 0 if stats["selected"] == args.limit else 1


def build_dataset(limit: int, output_path: Path, throwaway_patterns: list[re.Pattern[str]]) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    filtered: Counter[str] = Counter()
    considered = 0
    pulled_at_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    headers = {"User-Agent": "ExecLint eval builder/0.1 (+https://github.com/)"}
    github_headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        github_headers["Authorization"] = f"Bearer {token}"

    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True, headers=headers) as http:
        arxiv_client = ArxivClient(timeout=REQUEST_TIMEOUT_SECONDS)
        try:
            for api_entry in iter_arxiv_entries(http):
                if len(selected) >= limit:
                    break
                considered += 1
                paper = _fetch_abs_paper(arxiv_client, api_entry)
                if paper is None or not paper.code_url:
                    filtered["no_github_link"] += 1
                    continue

                repo_ref = _repo_ref_from_url(paper.code_url.unicode_string())
                if repo_ref is None:
                    filtered["invalid_github_repo_url"] += 1
                    continue

                owner, repo = repo_ref
                if _matches_throwaway(owner, throwaway_patterns):
                    filtered["throwaway_owner"] += 1
                    continue

                repo_info = _fetch_repo_info(http, github_headers, owner, repo)
                if repo_info is None:
                    filtered["repo_404"] += 1
                    continue

                stars = int(repo_info["stars"])
                if stars < 10:
                    filtered["stars_lt_10"] += 1
                    continue

                if not repo_info["has_readme"]:
                    filtered["no_readme"] += 1
                    continue

                canonical_owner = str(repo_info["owner"])
                canonical_repo = str(repo_info["repo"])
                selected.append(
                    {
                        "arxiv_id": paper.arxiv_id,
                        "arxiv_url": paper.url.unicode_string(),
                        "repo_url": f"https://github.com/{canonical_owner}/{canonical_repo}",
                        "title": paper.title,
                        "submitted_date": api_entry["submitted_date"],
                        "stars_at_pull_time": stars,
                        "pulled_at_utc": pulled_at_utc,
                    }
                )
        finally:
            arxiv_client.close()

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in selected:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    return {"considered": considered, "filtered": filtered, "selected": len(selected)}


def iter_arxiv_entries(http: httpx.Client):
    start = 0
    while True:
        response = http.get(
            ARXIV_API_URL,
            params={
                "search_query": ARXIV_QUERY,
                "start": start,
                "max_results": PAGE_SIZE,
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            },
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.text)
        entries = root.findall("atom:entry", ARXIV_NS)
        if not entries:
            return
        for entry in entries:
            arxiv_url = _text(entry, "atom:id")
            if not arxiv_url:
                continue
            arxiv_id, normalized_url = normalize_arxiv_input(arxiv_url)
            yield {
                "arxiv_id": arxiv_id,
                "arxiv_url": normalized_url,
                "submitted_date": _text(entry, "atom:published") or "",
            }
        start += len(entries)


def _fetch_abs_paper(arxiv_client: ArxivClient, api_entry: dict[str, str]):
    try:
        return arxiv_client.fetch_paper(
            arxiv_id=api_entry["arxiv_id"],
            url=api_entry["arxiv_url"],
            original_input=api_entry["arxiv_url"],
        )
    except Exception:
        return None


def _fetch_repo_info(http: httpx.Client, headers: dict[str, str], owner: str, repo: str) -> dict[str, Any] | None:
    try:
        repo_meta = _fetch_repo_meta(http, headers, owner, repo)
        if repo_meta is None:
            return None
        full_name = str(repo_meta.get("full_name") or f"{owner}/{repo}")
        canonical_owner, canonical_repo = full_name.split("/", 1)
        return {
            "owner": canonical_owner,
            "repo": canonical_repo,
            "stars": int(repo_meta.get("stargazers_count") or 0),
            "has_readme": _repo_has_readme(http, headers, canonical_owner, canonical_repo),
        }
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 403:
            raise
        return _fetch_repo_info_from_html(http, owner, repo)


def _fetch_repo_meta(http: httpx.Client, headers: dict[str, str], owner: str, repo: str) -> dict[str, Any] | None:
    response = http.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def _repo_has_readme(http: httpx.Client, headers: dict[str, str], owner: str, repo: str) -> bool:
    response = http.get(f"https://api.github.com/repos/{owner}/{repo}/readme", headers=headers)
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return bool(response.json().get("content"))


def _fetch_repo_info_from_html(http: httpx.Client, owner: str, repo: str) -> dict[str, Any] | None:
    response = http.get(f"https://github.com/{owner}/{repo}")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    final_ref = _repo_ref_from_url(str(response.url)) or (owner, repo)
    html = response.text
    return {
        "owner": final_ref[0],
        "repo": final_ref[1],
        "stars": _parse_github_html_stars(html),
        "has_readme": _parse_github_html_has_readme(html),
    }


def _parse_github_html_stars(html: str) -> int:
    match = re.search(r"/stargazers\"[^>]*>.*?<strong>([^<]+)</strong>\s*stars", html, re.IGNORECASE | re.DOTALL)
    if not match:
        return 0
    return _parse_compact_count(match.group(1))


def _parse_compact_count(value: str) -> int:
    compact = value.strip().lower().replace(",", "")
    multiplier = 1
    if compact.endswith("k"):
        multiplier = 1_000
        compact = compact[:-1]
    elif compact.endswith("m"):
        multiplier = 1_000_000
        compact = compact[:-1]
    try:
        return int(float(compact) * multiplier)
    except ValueError:
        return 0


def _parse_github_html_has_readme(html: str) -> bool:
    return bool(re.search(r'id="readme"|data-target="readme-toc|README\.md', html, re.IGNORECASE))


def _repo_ref_from_url(url: str) -> tuple[str, str] | None:
    parsed = urlsplit(url)
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return None
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) < 2:
        return None
    owner, repo = segments[0], segments[1].removesuffix(".git")
    if not owner or not repo:
        return None
    return owner, repo


def _matches_throwaway(owner: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(owner) for pattern in patterns)


def _ensure_throwaway_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]\n", encoding="utf-8")


def _load_throwaway_patterns(path: Path) -> list[re.Pattern[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list of regex strings")
    return [re.compile(str(item), re.IGNORECASE) for item in raw]


def _text(entry: ElementTree.Element, path: str) -> str | None:
    found = entry.find(path, ARXIV_NS)
    if found is None or found.text is None:
        return None
    return found.text.strip()


def write_dataset_readme(readme_path: Path, output_path: Path, throwaway_config: Path) -> None:
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(
        f"""# ExecLint Eval Dataset v2

This dataset is built deterministically from the arXiv API.

## Source Query

- arXiv search query: `{ARXIV_QUERY}`
- Date window: submitted between `2026-01-15 00:00` and `2026-02-15 23:59` inclusive.
- Sort: `submittedDate` ascending.

## Selection Rule

The builder walks papers in arXiv submission-date ascending order and takes the first 40 papers that pass all filters. It does not choose the "best" 40, highest-star 40, or most complete 40. The selected set is therefore deterministic given the arXiv sort order, GitHub repository state at pull time, and the throwaway-owner configuration.

## Filters

For each arXiv paper, the builder fetches the paper's abs page and extracts a `github.com` link. Papers are discarded if:

- the abs page has no GitHub link
- the linked repository returns 404
- the linked repository has fewer than 10 stars
- the linked repository has no README according to the GitHub REST API
- the repository owner matches a configured throwaway-account pattern

Throwaway owner patterns are configured in `{throwaway_config.as_posix()}` as a JSON list of regular expressions. The initial list is empty.

## Output

The dataset is written to `{output_path.as_posix()}` as JSONL, one paper per line, with:

`{{arxiv_id, arxiv_url, repo_url, title, submitted_date, stars_at_pull_time, pulled_at_utc}}`
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
