from __future__ import annotations

import base64

import httpx

from execlint.clients.github_client import GitHubClient


def test_github_client_follows_readme_redirect(monkeypatch) -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/repos/CASIA-IVA-Lab/FastSAM/readme":
            return httpx.Response(
                301,
                headers={"Location": "/repos/CASIA-IVA-Lab/FastSAM/readme-final"},
                request=request,
            )
        if request.url.path == "/repos/CASIA-IVA-Lab/FastSAM/readme-final":
            content = base64.b64encode(b"# FastSAM\n").decode("ascii")
            return httpx.Response(200, json={"content": content, "encoding": "base64"}, request=request)
        return httpx.Response(404, request=request)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def client_factory(**kwargs):
        return real_client(
            base_url=kwargs["base_url"],
            headers=kwargs["headers"],
            timeout=kwargs["timeout"],
            follow_redirects=kwargs["follow_redirects"],
            transport=transport,
        )

    monkeypatch.setattr("execlint.clients.github_client.httpx.Client", client_factory)

    client = GitHubClient()
    try:
        assert client.get_readme("CASIA-IVA-Lab/FastSAM") == "# FastSAM\n"
    finally:
        client.close()

    assert seen_paths == [
        "/repos/CASIA-IVA-Lab/FastSAM/readme",
        "/repos/CASIA-IVA-Lab/FastSAM/readme-final",
    ]
