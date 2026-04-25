# ExecLint v2 Evaluation Summary

Source: `v2_run_20260425.jsonl`
Papers in dataset: 40

## 1. Run Statistics

- Total runtime: 119.56 seconds (1.99 minutes)
- Errors: 0
- Completed cleanly: 40

## 2. Gap-Count Histogram

| gaps | papers |
|------|--------|
| 0    | 9 |
| 1    | 18 |
| 2    | 6 |
| 3    | 6 |
| 4    | 1 |
| 5+    | 0 |

(across 40 papers that completed without error)

## 3. Per-Category Gap Frequency

| category | count | % of papers |
|----------|-------|-------------|
| install  |     9 |  22.5% |
| data     |    11 |  27.5% |
| weights  |     4 |  10.0% |
| run      |    10 |  25.0% |
| eval     |     0 |   0.0% |
| env      |    18 |  45.0% |

(% computed against 40 clean-completion papers)

## 4. Top 3 Evidence Strings per Category

### install
- (9x) No setup.py, pyproject.toml, requirements.txt, or Dockerfile in repo root

### data
- (11x) README mentions dataset/data setup but no adjacent download link or automated bootstrap was found

### weights
- (4x) README mentions weights/checkpoints but no download link found within 200 chars

### run
- (10x) No regex-extracted run command found in README or repo file paths

### eval
- (no gaps observed)

### env
- (18x) README mentions Python/CUDA/PyTorch context but no version was pinned

## 5. Errored Papers

None.
