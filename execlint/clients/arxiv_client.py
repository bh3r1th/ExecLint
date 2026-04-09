from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from execlint.config import REQUEST_TIMEOUT_SECONDS
from execlint.models import ArxivPaper

ARXIV_ABS_URL = "https://arxiv.org/abs/"
GITHUB_HOSTS = {"github.com", "www.github.com"}
CODE_LINK_TERMS = ("code", "github", "implementation", "official", "repo", "repository", "project")


class ArxivClient:
    def __init__(self, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    def fetch_paper(self, arxiv_id: str, url: str) -> ArxivPaper:
        response = self._client.get(f"{ARXIV_ABS_URL}{arxiv_id}", timeout=self._timeout)
        response.raise_for_status()
        return _parse_abs_page(arxiv_id=arxiv_id, page_url=url, html=response.text)


def _parse_abs_page(arxiv_id: str, page_url: str, html: str) -> ArxivPaper:
    parser = _ArxivAbsHTMLParser(base_url=page_url)
    parser.feed(html)
    parser.close()

    title = _strip_field_label(parser.title, "title")
    abstract = _strip_field_label(parser.abstract, "abstract")
    authors = parser.authors
    code_url = _select_github_url(parser.github_links)

    if not title and not abstract and not authors:
        raise ValueError(f"Could not parse paper metadata for arXiv id: {arxiv_id}")

    return ArxivPaper(
        arxiv_id=arxiv_id,
        url=page_url,
        title=title,
        authors=authors,
        abstract=abstract,
        code_url=code_url,
        code_url_source="arxiv_page" if code_url else "none",
    )


class _ArxivAbsHTMLParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self._title_parts: list[str] = []
        self._author_parts: list[str] = []
        self._abstract_parts: list[str] = []
        self._author_links: list[str] = []
        self._author_link_parts: list[str] = []
        self._current_link_parts: list[str] = []

        self._in_title = False
        self._in_authors = False
        self._in_abstract = False
        self._in_author_link = False
        self._current_href: str | None = None

        self._github_candidates: list[tuple[int, int, str]] = []

    @property
    def title(self) -> str | None:
        return _normalize_text("".join(self._title_parts))

    @property
    def abstract(self) -> str | None:
        return _normalize_text("".join(self._abstract_parts))

    @property
    def authors(self) -> list[str]:
        if self._author_links:
            return list(self._author_links)

        authors_text = _strip_field_label(_normalize_text("".join(self._author_parts)), "authors")
        if not authors_text:
            return []
        return [part for part in (_normalize_text(piece) for piece in authors_text.split(",")) if part]

    @property
    def github_links(self) -> list[tuple[int, int, str]]:
        return list(self._github_candidates)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        classes = set(attr_map.get("class", "").split())

        if tag == "h1" and "title" in classes:
            self._in_title = True
        elif tag == "div" and "authors" in classes:
            self._in_authors = True
        elif tag == "blockquote" and "abstract" in classes:
            self._in_abstract = True

        if self._in_authors and tag == "a":
            self._in_author_link = True
            self._author_link_parts = []

        if tag == "a":
            self._current_href = attr_map.get("href")
            self._current_link_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1":
            self._in_title = False
        elif tag == "div" and self._in_authors:
            self._in_authors = False
        elif tag == "blockquote":
            self._in_abstract = False

        if tag == "a":
            if self._current_href:
                normalized = _normalize_github_url(self._base_url, self._current_href)
                if normalized:
                    link_text = _normalize_text("".join(self._current_link_parts)) or ""
                    score = _github_url_score(normalized, link_text)
                    self._github_candidates.append((score, len(self._github_candidates), normalized))
            self._current_href = None
            self._current_link_parts = []

            if self._in_author_link:
                author_name = _normalize_text("".join(self._author_link_parts))
                if author_name:
                    self._author_links.append(author_name)
                self._in_author_link = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        if self._in_authors:
            self._author_parts.append(data)
        if self._in_abstract:
            self._abstract_parts.append(data)
        if self._in_author_link:
            self._author_link_parts.append(data)
        if self._current_href is not None:
            self._current_link_parts.append(data)


def _normalize_text(value: str) -> str | None:
    clean = " ".join(unescape(value).split())
    return clean or None


def _strip_field_label(value: str | None, label: str) -> str | None:
    if not value:
        return None
    prefix = f"{label.lower()}:"
    if value.lower().startswith(prefix):
        return value[len(prefix) :].strip() or None
    return value


def _normalize_github_url(base_url: str, href: str) -> str | None:
    absolute = urljoin(base_url, href)
    parsed = urlsplit(absolute)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc.lower() not in GITHUB_HOSTS:
        return None
    return urlunsplit(("https", "github.com", parsed.path.rstrip("/"), "", ""))


def _github_url_score(url: str, link_text: str) -> int:
    parsed = urlsplit(url)
    segments = [segment for segment in parsed.path.split("/") if segment]
    score = 0

    if len(segments) >= 2:
        score += 25
    elif segments:
        score += 10

    lowered_text = link_text.lower()
    if any(term in lowered_text for term in CODE_LINK_TERMS):
        score += 15

    if len(segments) > 2:
        score -= 3

    return score


def _select_github_url(candidates: list[tuple[int, int, str]]) -> str | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], -item[1]))[2]
