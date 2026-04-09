# ExecLint

ExecLint is a CLI tool that checks whether a research paper’s code is actually runnable and shows the execution path, gaps, and setup effort.

---

## What It Does

- Takes arXiv URL + GitHub repo  
- Extracts execution path (install, run, evaluate)  
- Identifies gaps (missing data, weights, unclear steps)  
- Estimates Time-to-Hello-World (TTHW)  
- Produces a practical execution report  

---

## Why It Exists

Many paper repos are not directly runnable.  
ExecLint shows execution reality upfront so you don’t waste time.

---

## Install

```bash
pip install -e .
```

---

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

- Execution Path: key commands to try  
- Runnable For: what repo supports  
- Gaps: missing pieces  
- What Breaks: real blockers  
- Technical Debt: friction points  
- Not Clearly Supported: unclear capabilities  
- HF Status: model availability  
- Verdict: GO / CAUTION / NO-GO  
- TTHW:

  - Level 1: plug-and-play  
  - Level 2: minor setup  
  - Level 3: significant setup  
  - Level 4: heavy infra  

---

## Example Output

Execution Path: install: pip install loralib; run: python train.py  
Gaps: env version unclear  
Verdict: CAUTION  
TTHW: Level 2  

---

## Limitations

- heuristic-based  
- may miss edge cases  
- does not guarantee reproducibility  

---

## When to Use

- deciding whether to implement a paper  
- quick repo triage  

---

## Design Principles

- deterministic  
- evidence-based  
- no hallucination  
