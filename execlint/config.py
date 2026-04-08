"""Configuration defaults for external APIs."""

from __future__ import annotations

import os

GITHUB_API_BASE = os.getenv("EXECLINT_GITHUB_API_BASE", "https://api.github.com")
HF_API_BASE = os.getenv("EXECLINT_HF_API_BASE", "https://huggingface.co/api")
HTTP_TIMEOUT_SECONDS = float(os.getenv("EXECLINT_HTTP_TIMEOUT", "10"))
