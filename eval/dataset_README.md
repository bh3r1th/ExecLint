# ExecLint Eval Dataset v2

This dataset is built deterministically from the arXiv API.

## Source Query

- arXiv search query: `(cat:cs.LG OR cat:cs.CL OR cat:cs.CV) AND submittedDate:[202601150000 TO 202602152359]`
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

Throwaway owner patterns are configured in `C:/Users/vanda/OneDrive/Desktop/ExecLint/eval/throwaway_accounts.json` as a JSON list of regular expressions. The initial list is empty.

## Output

The dataset is written to `C:/Users/vanda/OneDrive/Desktop/ExecLint/eval/dataset_v2.jsonl` as JSONL, one paper per line, with:

`{arxiv_id, arxiv_url, repo_url, title, submitted_date, stars_at_pull_time, pulled_at_utc}`
