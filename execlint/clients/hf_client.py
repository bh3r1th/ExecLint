from __future__ import annotations

import httpx

from execlint.config import HF_API_BASE, HTTP_TIMEOUT_SECONDS
from execlint.models import HFModelStatus


class HFClient:
    def __init__(self, timeout: float = HTTP_TIMEOUT_SECONDS) -> None:
        self._client = httpx.Client(base_url=HF_API_BASE, timeout=timeout)

    def search_model(self, query: str) -> HFModelStatus:
        response = self._client.get("/models", params={"search": query, "limit": 1})
        if response.status_code >= 500:
            return HFModelStatus(status="unknown", notes="HF API unavailable")
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return HFModelStatus(status="not_found", notes="No model match")
        model_id = payload[0].get("id")
        return HFModelStatus(status="found", model_id=model_id, notes="Top model result found")
