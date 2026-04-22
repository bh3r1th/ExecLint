"""Configuration defaults for external APIs and execution limits."""

from __future__ import annotations

import os


# Fix C3: safe env parsing to prevent undiagnosable crashes on bad env values
def _env_float(name: str, default: str, minimum: float = 0.0) -> float:
    raw = os.getenv(name, default)
    try:
        value = float(raw)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid value for {name}: {raw!r} (expected a number)") from None
    return max(value, minimum)


def _env_int(name: str, default: str, minimum: int = 1) -> int:
    raw = os.getenv(name, default)
    try:
        value = int(raw)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid value for {name}: {raw!r} (expected an integer)") from None
    return max(value, minimum)


GITHUB_API_BASE = os.getenv("EXECLINT_GITHUB_API_BASE", "https://api.github.com")
HF_API_BASE = os.getenv("EXECLINT_HF_API_BASE", "https://huggingface.co/api")

REQUEST_TIMEOUT_SECONDS = _env_float("EXECLINT_REQUEST_TIMEOUT_SECONDS", "10", minimum=1.0)
MAX_REPO_CANDIDATES = _env_int("EXECLINT_MAX_REPO_CANDIDATES", "8", minimum=1)
SOFT_EXECUTION_BUDGET_SECONDS = _env_float("EXECLINT_SOFT_EXECUTION_BUDGET_SECONDS", "20", minimum=1.0)
MAX_GITHUB_SEARCH_RESULTS_INSPECTED = _env_int("EXECLINT_MAX_GITHUB_SEARCH_RESULTS_INSPECTED", "24", minimum=1)
