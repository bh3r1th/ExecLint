# ExecLint — finds reproducibility gaps in ML paper repos

## Finding

ExecLint scanned 40 recent ML papers (arXiv cs.LG / cs.CL / cs.CV, Jan 15 – Feb 15 2026) whose authors had published a public GitHub repo. 31 of 40 (77.5%) had at least one detectable reproducibility gap. The top three: env (45%, no pinned Python/CUDA/PyTorch version), data (27.5%, README names a dataset but gives no download link), and run (25%, no extractable run command in README or repo root). ExecLint is the tool used to generate these results.

Caveat: the eval category currently reports 0% because ExecLint does not yet detect eval gaps — that reflects a missing detector, not clean papers.

## Quick start

```bash
pip install execlint
execlint https://arxiv.org/abs/2601.10880 \
  --repo https://github.com/AIM-Research-Lab/Medical-SAM3
```

## Example output

On Medical SAM3 ([2601.10880](https://arxiv.org/abs/2601.10880)):

```
Paper title: Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image Segmentation
Repo URL: https://github.com/AIM-Research-Lab/Medical-SAM3
GAPS (2):
  - dataset must be supplied manually  [data]
    README mentions dataset/data setup but no adjacent download link or automated bootstrap was found
  - env version unclear  [env]
    README mentions Python/CUDA/PyTorch context but no version was pinned
EXECUTION PATH:
  - install: pip install -r requirements.txt
  - run: python inference/sam3_inference.py
  - evaluate: python inference/metrics.py | python inference/run_evaluation.py
```

## What ExecLint checks

| Category | What triggers a gap |
|----------|---------------------|
| install  | No setup.py, pyproject.toml, requirements.txt, or Dockerfile in repo root |
| data     | README names a dataset but no adjacent download link or bootstrap |
| weights  | README mentions weights/checkpoints but no download link within 200 chars |
| run      | No regex-extracted run command in README or repo file paths |
| env      | README mentions Python/CUDA/PyTorch but no version pinned |

eval gaps are not yet detected.

## How it works

ExecLint takes an arXiv URL and a GitHub repo URL. It reads the repo's README and top-level filetree, then applies pure regex and file-existence checks to decide which gap categories apply. There are no LLM calls and no network calls beyond the initial GitHub API fetch. Detection is deterministic: the same inputs produce the same gap report on every run.

## Limitations

- Evidence strings are category-level, not paper-specific — every "env" gap shows the same evidence text regardless of which version the README mentioned.
- eval gaps are not yet detected.
- A repo that has moved or been renamed on GitHub silently redirects; ExecLint's HTTP follow_redirects is fixed, but the rename itself is invisible to the report.
- Results reflect the README and filetree at the moment of the run; repos change.

## Eval

Evaluated on 40 arXiv ML papers from Jan–Feb 2026. 77.5% had at least one gap. See [eval/](eval/) for the dataset, full per-paper results, and reproduction instructions.

## Contributing

Issues and PRs welcome on GitHub. Please include the arXiv URL and the repo URL that produced the unexpected output.
