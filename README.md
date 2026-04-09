# ExecLint

ExecLint is a CLI tool that shows how to run a paper’s code, what’s missing, and how much effort it takes.
Verdict is a heuristic signal. Always rely on Execution Path and Gaps.

---

## What It Does

- Extracts execution path (install, run, evaluate)
- Identifies missing pieces (data, weights, setup)
- Estimates Time-to-Hello-World (TTHW)

---

## Why It Exists

Paper repos often look runnable but fail at execution.
ExecLint shows the real path and gaps before you waste time.

---

## Install

```bash
pip install -e .
```

---
## Input Requirements

- arXiv URL (paper)
- GitHub repository URL (code)

ExecLint does not search for repos automatically.

## Usage

### Basic

```bash
python -m execlint.cli <arxiv_url> --repo <github_repo>
```

Example:

```bash
python -m execlint.cli https://arxiv.org/abs/2106.09685 --repo https://github.com/microsoft/LoRA
```

### With debug

```bash
python -m execlint.cli <arxiv_url> --repo <github_repo> --debug
```

---

## Output Explained

- Execution Path: actual commands extracted from repo
- Gaps: what you must supply manually (data, weights, env)
- What Breaks: concrete execution blockers only
- Verdict: rough signal (not reliable alone)
- TTHW: effort required to get first result 

---

## Example Output

Execution Path:
install: pip install loralib
run: python examples/.../run_clm.py

Gaps:
env version unclear

Verdict:
CAUTION

TTHW:
Level 2 — minor setup required 

---

## Limitations

- capability labels may be imprecise
- verdict may be conservative

---

## When to Use

- deciding whether to implement a paper  
- quick repo triage  

---

## Design Principles

- deterministic  
- evidence-based  
- no hallucination  
