from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


class HTTPStatusError(Exception):
    pass


@dataclass
class Response:
    status_code: int
    text: str

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise HTTPStatusError(f"HTTP error {self.status_code}")

    def json(self):
        return json.loads(self.text)


class Client:
    def __init__(self, base_url: str = "", headers: dict | None = None, timeout: float = 10, follow_redirects: bool = True) -> None:
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout

    def get(self, url: str, params: dict | None = None) -> Response:
        final_url = urljoin(self.base_url + "/", url.lstrip("/")) if self.base_url else url
        if params:
            final_url = f"{final_url}?{urlencode(params)}"
        req = Request(final_url, headers=self.headers)
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return Response(status_code=resp.status, text=resp.read().decode("utf-8", errors="ignore"))
        except Exception:
            return Response(status_code=500, text="{}")
