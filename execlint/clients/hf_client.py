from __future__ import annotations

import httpx

from execlint.config import HF_API_BASE, REQUEST_TIMEOUT_SECONDS
from execlint.models import HFModelStatus


class HFClient:
    def __init__(self, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        self._timeout = timeout
        self._client = httpx.Client(base_url=HF_API_BASE, timeout=timeout)

    def search_models(self, query: str, limit: int = 5) -> list[dict]:
        response = self._client.get("/models", params={"search": query, "limit": limit}, timeout=self._timeout)
        if response.status_code >= 500:
            return []
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return []
        return payload

    def search_model(self, query: str) -> HFModelStatus:
        payload = self.search_models(query=query, limit=1)
        if not payload:
            return HFModelStatus(status="not_found", notes="No model match")
        top = payload[0]
        return HFModelStatus(
            status="found",
            model_id=top.get("id"),
            license=top.get("license"),
            gated=bool(top.get("gated", False)),
            notes="Top model result found",
        )
