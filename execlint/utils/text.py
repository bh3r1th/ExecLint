from __future__ import annotations

from urllib.parse import urlparse


ARXIV_ABS_PREFIX = "/abs/"
ARXIV_PDF_PREFIX = "/pdf/"


def normalize_arxiv_url(url: str) -> str:
    return url.strip()


def extract_arxiv_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc not in {"arxiv.org", "www.arxiv.org"}:
        raise ValueError("URL must be from arxiv.org")

    path = parsed.path.strip()
    if path.startswith(ARXIV_ABS_PREFIX):
        arxiv_id = path[len(ARXIV_ABS_PREFIX) :]
    elif path.startswith(ARXIV_PDF_PREFIX):
        arxiv_id = path[len(ARXIV_PDF_PREFIX) :].removesuffix(".pdf")
    else:
        raise ValueError("URL must be an arXiv abs or pdf URL")

    if not arxiv_id:
        raise ValueError("Could not parse arXiv id")
    return arxiv_id
