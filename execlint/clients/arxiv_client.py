from __future__ import annotations

import re

import httpx

from execlint.config import HTTP_TIMEOUT_SECONDS
from execlint.models import ArxivPaper

ARXIV_OAI_URL = "https://export.arxiv.org/api/query"
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)


class ArxivClient:
    def __init__(self, timeout: float = HTTP_TIMEOUT_SECONDS) -> None:
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    def fetch_paper(self, arxiv_id: str, url: str) -> ArxivPaper:
        response = self._client.get(ARXIV_OAI_URL, params={"search_query": f"id:{arxiv_id}", "max_results": 1})
        response.raise_for_status()
        title = _extract_title(response.text)
        return ArxivPaper(arxiv_id=arxiv_id, url=url, title=title)


def _extract_title(feed_xml: str) -> str | None:
    matches = TITLE_RE.findall(feed_xml)
    if len(matches) < 2:
        return None
    return " ".join(matches[1].strip().split())
