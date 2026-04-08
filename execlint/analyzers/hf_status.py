from __future__ import annotations

from execlint.clients.hf_client import HFClient
from execlint.models import ArxivPaper, HFModelStatus


def check_hf_status(paper: ArxivPaper, hf_client: HFClient) -> HFModelStatus:
    if paper.title:
        return hf_client.search_model(paper.title)
    return hf_client.search_model(paper.arxiv_id)
