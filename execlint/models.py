from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class RepoCapability(str, Enum):
    demo = "demo"
    inference = "inference"
    training = "training"
    evaluation = "evaluation"
    smoke_test = "smoke_test"
    unclear = "unclear"


class ArxivPaper(BaseModel):
    arxiv_id: str
    url: HttpUrl
    title: str | None = None
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    code_url: HttpUrl | None = None
    code_url_source: Literal["arxiv_page", "none"] = "none"


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
    inferred_capabilities: list[RepoCapability] = Field(default_factory=list)
    execution_steps: dict[str, list[str]] = Field(default_factory=dict)
    missing_prerequisites: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


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
    runnable_for: str = "unclear"
    execution_path: str = "No extracted execution commands"
    gaps: str = "None identified"
    not_clearly_supported: str = ""
    what_breaks: str
    fix: str
    hf_status: str
    technical_debt: str


class ExecutionInput(BaseModel):
    arxiv_url: HttpUrl
    repo_url: HttpUrl
    weights_url: HttpUrl | None = None
    ref: str | None = None
