from __future__ import annotations

import httpx

from execlint.clients.arxiv_client import ArxivClient, fetch_arxiv_page_debug, normalize_arxiv_input


SAMPLE_ARXIV_HTML = """
<html>
  <body>
    <h1 class="title mathjax">Title: ExecLint: Surgical Debug Signals for Papers</h1>
    <div class="authors">
      Authors:
      <a href="/search/?searchtype=author&amp;query=Smith%2C+A">Alice Smith</a>,
      <a href="/search/?searchtype=author&amp;query=Jones%2C+B">Bob Jones</a>
    </div>
    <blockquote class="abstract mathjax">
      Abstract: We expose paper-level debug signals without parsing PDFs.
    </blockquote>
    <div class="extra-services">
      <a href="https://example.com/project">Project page</a>
      <a href="https://github.com/example/execlint">Code</a>
      <a href="https://github.com/example/execlint/issues">Issues</a>
    </div>
  </body>
</html>
"""


SAMPLE_ARXIV_HTML_NO_GITHUB = """
<html>
  <body>
    <h1 class="title mathjax">Title: ExecLint Without Code Link</h1>
    <div class="authors">
      Authors:
      <a href="/search/?searchtype=author&amp;query=Smith%2C+A">Alice Smith</a>
    </div>
    <blockquote class="abstract mathjax">
      Abstract: No GitHub URL is present on this page.
    </blockquote>
    <a href="https://example.com/project">Project page</a>
  </body>
</html>
"""


class _DummyResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _DebugResponse:
    def __init__(self, status_code: int, text: str, url: str) -> None:
        self.status_code = status_code
        self.text = text
        self.url = url

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", self.url)
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("boom", request=request, response=response)


class _DebugClient:
    def __init__(self, response: _DebugResponse) -> None:
        self._response = response

    def __enter__(self) -> _DebugClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def get(self, url: str) -> _DebugResponse:
        return self._response


def test_fetch_paper_extracts_title_authors_abstract_and_github_link(monkeypatch) -> None:
    client = ArxivClient(timeout=1.0)
    monkeypatch.setattr(client._client, "get", lambda url: _DummyResponse(SAMPLE_ARXIV_HTML))

    paper = client.fetch_paper(arxiv_id="2401.00001", url="https://arxiv.org/abs/2401.00001")

    assert paper.title == "ExecLint: Surgical Debug Signals for Papers"
    assert paper.authors == ["Alice Smith", "Bob Jones"]
    assert paper.abstract == "We expose paper-level debug signals without parsing PDFs."
    assert str(paper.code_url) == "https://github.com/example/execlint"
    assert paper.code_url_source == "arxiv_page"


def test_fetch_paper_sets_none_when_no_github_link_exists(monkeypatch) -> None:
    client = ArxivClient(timeout=1.0)
    monkeypatch.setattr(client._client, "get", lambda url: _DummyResponse(SAMPLE_ARXIV_HTML_NO_GITHUB))

    paper = client.fetch_paper(arxiv_id="2401.00002", url="https://arxiv.org/abs/2401.00002")

    assert paper.title == "ExecLint Without Code Link"
    assert paper.authors == ["Alice Smith"]
    assert paper.abstract == "No GitHub URL is present on this page."
    assert paper.code_url is None
    assert paper.code_url_source == "none"


def test_fetch_paper_failure_includes_request_url_and_status_code(monkeypatch) -> None:
    client = ArxivClient(timeout=1.0)
    request = httpx.Request("GET", "https://arxiv.org/abs/2106.09685")
    response = httpx.Response(503, request=request)

    def _raise_status(url: str):
        raise httpx.HTTPStatusError("Service unavailable", request=request, response=response)

    monkeypatch.setattr(client._client, "get", _raise_status)

    try:
        client.fetch_paper(
            arxiv_id="2106.09685",
            url="https://arxiv.org/abs/2106.09685",
            original_input="https://arxiv.org/abs/2106.09685",
        )
        raise AssertionError("Expected fetch_paper to raise ValueError")
    except ValueError as exc:
        message = str(exc)

    assert "request_url='https://arxiv.org/abs/2106.09685'" in message
    assert "status_code=503" in message
    assert "root_exception=HTTPStatusError" in message


def test_normalize_arxiv_input_abs_url() -> None:
    arxiv_id, abs_url = normalize_arxiv_input("https://arxiv.org/abs/2106.09685")

    assert arxiv_id == "2106.09685"
    assert abs_url == "https://arxiv.org/abs/2106.09685"


def test_normalize_arxiv_input_pdf_url() -> None:
    arxiv_id, abs_url = normalize_arxiv_input("https://arxiv.org/pdf/2106.09685")

    assert arxiv_id == "2106.09685"
    assert abs_url == "https://arxiv.org/abs/2106.09685"


def test_normalize_arxiv_input_raw_id() -> None:
    arxiv_id, abs_url = normalize_arxiv_input("2106.09685")

    assert arxiv_id == "2106.09685"
    assert abs_url == "https://arxiv.org/abs/2106.09685"


def test_normalize_arxiv_input_versioned_id() -> None:
    arxiv_id, abs_url = normalize_arxiv_input("2106.09685v2")

    assert arxiv_id == "2106.09685"
    assert abs_url == "https://arxiv.org/abs/2106.09685"


def test_fetch_arxiv_page_debug_returns_diagnostics(monkeypatch) -> None:
    response = _DebugResponse(
        status_code=200,
        text="<html><body>Hello</body></html>",
        url="https://arxiv.org/abs/2106.09685",
    )
    monkeypatch.setattr(
        "execlint.clients.arxiv_client.httpx.Client",
        lambda **kwargs: _DebugClient(response),
    )

    debug = fetch_arxiv_page_debug("https://arxiv.org/abs/2106.09685")

    assert debug["original_input"] == "https://arxiv.org/abs/2106.09685"
    assert debug["normalized_arxiv_id"] == "2106.09685"
    assert debug["request_url"] == "https://arxiv.org/abs/2106.09685"
    assert debug["status_code"] == 200
    assert debug["final_url"] == "https://arxiv.org/abs/2106.09685"
    assert debug["body_preview"] == "<html><body>Hello</body></html>"
    assert debug["error"] is None
