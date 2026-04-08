from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class ArxivPaper(BaseModel):
    arxiv_id: str
    url: HttpUrl
    title: str | None = None
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)


class RepoCandidate(BaseModel):
    name: str
    full_name: str
    url: HttpUrl
    stars: int = 0
    open_issues_count: int = 0
    has_readme: bool = False
    setup_signals: list[str] = Field(default_factory=list)
    description: str | None = None
    owner_login: str | None = None
    archived: bool = False
    pushed_at: str | None = None
    size_kb: int = 0
    default_branch: str = "main"
    discovery_score: float = 0.0
    discovery_reasons: list[str] = Field(default_factory=list)
    entrypoint_signals: list[str] = Field(default_factory=list)
    surface_file_count: int = 0
    readiness_score: float = 0.0
    readiness_label: Literal["strong", "moderate", "weak"] = "weak"
    readiness_summary: str = ""


class IssueFixSignal(BaseModel):
    blocker: str
    fix: str | None = None
    confidence: Literal["low", "medium", "high"] = "low"
    issue_number: int | None = None
    blocker_category: str | None = None


class HFModelStatus(BaseModel):
    status: Literal["found", "not_found", "unknown"]
    model_id: str | None = None
    license: str | None = None
    notes: str | None = None
    gated: bool | None = None


class ExecutionReport(BaseModel):
    verdict: Literal["GO", "CAUTION", "NO-GO"]
    tthw: Literal["Level 1", "Level 2", "Level 3", "Level 4"]
    best_repo: str
    what_breaks: str
    fix: str
    hf_status: str
    technical_debt: str
