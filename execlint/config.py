"""Configuration defaults for external APIs and execution limits."""

from __future__ import annotations

import os

GITHUB_API_BASE = os.getenv("EXECLINT_GITHUB_API_BASE", "https://api.github.com")
HF_API_BASE = os.getenv("EXECLINT_HF_API_BASE", "https://huggingface.co/api")

REQUEST_TIMEOUT_SECONDS = float(os.getenv("EXECLINT_REQUEST_TIMEOUT_SECONDS", "10"))
MAX_REPO_CANDIDATES = int(os.getenv("EXECLINT_MAX_REPO_CANDIDATES", "8"))
MAX_ISSUES_PER_REPO = int(os.getenv("EXECLINT_MAX_ISSUES_PER_REPO", "12"))
SOFT_EXECUTION_BUDGET_SECONDS = float(os.getenv("EXECLINT_SOFT_EXECUTION_BUDGET_SECONDS", "20"))
MAX_GITHUB_SEARCH_RESULTS_INSPECTED = int(os.getenv("EXECLINT_MAX_GITHUB_SEARCH_RESULTS_INSPECTED", "24"))
