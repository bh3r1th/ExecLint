from __future__ import annotations

from execlint.clients.arxiv_client import ArxivClient, normalize_arxiv_input


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


def test_fetch_paper_extracts_title_authors_abstract_and_github_link(monkeypatch) -> None:
    client = ArxivClient(timeout=1.0)
    monkeypatch.setattr(client._client, "get", lambda url, timeout: _DummyResponse(SAMPLE_ARXIV_HTML))

    paper = client.fetch_paper(arxiv_id="2401.00001", url="https://arxiv.org/abs/2401.00001")

    assert paper.title == "ExecLint: Surgical Debug Signals for Papers"
    assert paper.authors == ["Alice Smith", "Bob Jones"]
    assert paper.abstract == "We expose paper-level debug signals without parsing PDFs."
    assert paper.code_url == "https://github.com/example/execlint"
    assert paper.code_url_source == "arxiv_page"


def test_fetch_paper_sets_none_when_no_github_link_exists(monkeypatch) -> None:
    client = ArxivClient(timeout=1.0)
    monkeypatch.setattr(client._client, "get", lambda url, timeout: _DummyResponse(SAMPLE_ARXIV_HTML_NO_GITHUB))

    paper = client.fetch_paper(arxiv_id="2401.00002", url="https://arxiv.org/abs/2401.00002")

    assert paper.title == "ExecLint Without Code Link"
    assert paper.authors == ["Alice Smith"]
    assert paper.abstract == "No GitHub URL is present on this page."
    assert paper.code_url is None
    assert paper.code_url_source == "none"


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
