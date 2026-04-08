from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class ArxivPaper(BaseModel):
    arxiv_id: str
    url: HttpUrl
    title: str | None = None


class RepoCandidate(BaseModel):
    name: str
    full_name: str
    url: HttpUrl
    stars: int = 0
    open_issues_count: int = 0
    has_readme: bool = False
    setup_signals: list[str] = Field(default_factory=list)


class IssueFixSignal(BaseModel):
    blocker: str
    fix: str | None = None
    confidence: Literal["low", "medium", "high"] = "low"


class HFModelStatus(BaseModel):
    status: Literal["found", "not_found", "unknown"]
    model_id: str | None = None
    notes: str | None = None


class ExecutionReport(BaseModel):
    verdict: Literal["GO", "CAUTION", "NO-GO"]
    tthw: Literal["Level 1", "Level 2", "Level 3", "Level 4"]
    best_repo: str
    what_breaks: str
    fix: str
    hf_status: str
    technical_debt: str
