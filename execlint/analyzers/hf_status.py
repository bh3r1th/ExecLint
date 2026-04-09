from __future__ import annotations

import re

from execlint.clients.hf_client import HFClient
from execlint.models import ArxivPaper, HFModelStatus


def check_hf_status(paper: ArxivPaper, hf_client: HFClient, weights_url: str | None = None) -> HFModelStatus:
    if weights_url:
        gated = "huggingface.co" in weights_url.lower() and "/resolve/" not in weights_url.lower()
        return HFModelStatus(
            status="found",
            model_id=weights_url,
            gated=gated,
            notes="User-provided weights URL",
        )

    queries = _queries(paper)
    best: tuple[float, dict] | None = None

    for query in queries:
        for model in hf_client.search_models(query=query, limit=5):
            score = _score_model_match(model=model, paper=paper)
            if best is None or score > best[0]:
                best = (score, model)

    if best is None:
        return HFModelStatus(status="not_found", notes="No model/weights match for paper title or arXiv id")

    model = best[1]
    gated = bool(model.get("gated", False))
    note = "Model found"
    if gated:
        note = "Model appears gated"
    if model.get("license") in (None, "", "unknown"):
        note = f"{note}; license unclear"

    return HFModelStatus(
        status="found",
        model_id=model.get("id"),
        license=model.get("license"),
        gated=gated,
        notes=note,
    )


def _queries(paper: ArxivPaper) -> list[str]:
    queries: list[str] = []
    if paper.title:
        title = " ".join(paper.title.split())
        queries.append(title)
        queries.append(" ".join(title.split()[:6]))
    queries.append(paper.arxiv_id)
    return [q for i, q in enumerate(queries) if q and q not in queries[:i]]


def _score_model_match(model: dict, paper: ArxivPaper) -> float:
    paper_tokens = _tokens((paper.title or "") + f" {paper.arxiv_id}")
    model_text = f"{model.get('id', '')} {model.get('tags', '')}"
    model_tokens = _tokens(model_text)
    if not paper_tokens or not model_tokens:
        return 0.0
    overlap = len(paper_tokens & model_tokens) / len(paper_tokens)
    return overlap + (0.05 if model.get("downloads") else 0.0)


def _tokens(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return {token for token in normalized.split() if len(token) > 2}
