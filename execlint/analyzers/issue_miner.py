from __future__ import annotations

import re
from difflib import SequenceMatcher

from execlint.clients.github_client import GitHubClient
from execlint.config import MAX_ISSUES_PER_REPO
from execlint.models import IssueFixSignal, RepoCandidate

BLOCKER_KEYWORDS = (
    "install",
    "dependency",
    "requirements",
    "broken",
    "error",
    "cuda",
    "import",
    "version",
    "doesn't work",
    "crash",
    "fail",
)

NOISY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(confused|confusing|i don'?t understand|not sure|unclear docs?)\b", re.I),
    re.compile(r"\b(quality is bad|poor quality|results are bad|not reproducible|reproducibility issue)\b", re.I),
    re.compile(r"\b(feels broken|seems broken|doesn'?t look right|hard to use)\b", re.I),
)

_FIX_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bpin\s+[\w\-\.]+(?:\s*(?:==|<=|>=)\s*[\w\-.]+)?", "pin package version"),
    (r"\bdowngrad(?:e|ing)\s+transformers\b|\btransformers\b.+\bdowngrad", "downgrade transformers"),
    (r"\bdowngrad(?:e|ing)\s+[\w\-.]+", "downgrade package"),
    (r"\bcuda\s*11\.8\b", "use CUDA 11.8"),
    (r"\buse\s+cuda\s*\d+(?:\.\d+)?", "use specific CUDA version"),
    (r"\bdeprecated\b|\breplace\b.+\bapi\b", "replace deprecated API"),
    (r"\binstall\b.+\b(extra|additional)\b.+\b(dependenc|package)", "install extra dependency"),
    (r"\buse\s+fork\b", "use fork"),
)


def mine_issue_signals(repo: RepoCandidate, github: GitHubClient) -> list[IssueFixSignal]:
    issues = _list_issues_for_mining(repo, github)
    candidates: list[tuple[float, dict, IssueFixSignal]] = []

    for issue in issues:
        signal = _build_signal(issue)
        if signal is None:
            continue
        score, built = signal
        candidates.append((score, issue, built))

    deduped = _dedupe_candidates(candidates)
    deduped.sort(key=lambda item: (-item[0], -(item[2].issue_number or 0), item[2].blocker.lower()))
    top_n = min(5, max(1, MAX_ISSUES_PER_REPO // 2))
    return [signal for _, _, signal in deduped[:top_n]]


def _list_issues_for_mining(repo: RepoCandidate, github: GitHubClient) -> list[dict]:
    if hasattr(github, "list_issues_for_mining"):
        return github.list_issues_for_mining(repo.full_name, limit=MAX_ISSUES_PER_REPO)
    return github.list_open_issues(repo.full_name, limit=MAX_ISSUES_PER_REPO)


def _build_signal(issue: dict) -> tuple[float, IssueFixSignal] | None:
    title = (issue.get("title") or "").strip()
    body = (issue.get("body") or "").strip()
    comments_text = _collect_comments_text(issue)
    combined_text = " ".join(part for part in (title, body, comments_text) if part)
    combined_lower = combined_text.lower()

    matched_keywords = [kw for kw in BLOCKER_KEYWORDS if kw in combined_lower]
    if not matched_keywords:
        return None

    blocker_label = _extract_blocker_label(combined_lower)
    if blocker_label is None or _is_noisy_issue(title, body, combined_lower):
        return None

    probable_fix = _extract_fix_signal(combined_lower)

    comments_count = int(issue.get("comments") or 0)
    reactions = issue.get("reactions") or {}
    reaction_score = int(reactions.get("total_count") or 0)
    closed_bonus = 2 if issue.get("state") == "closed" and probable_fix else 0

    keyword_score = len(set(matched_keywords))
    score = float(keyword_score + closed_bonus + min(comments_count, 8) * 0.35 + min(reaction_score, 8) * 0.3)

    confidence = "high" if score >= 6 else "medium" if score >= 3 else "low"
    signal = IssueFixSignal(
        blocker=blocker_label,
        issue_number=issue.get("number"),
        blocker_category=blocker_label,
        fix=probable_fix,
        confidence=confidence,
    )
    return score, signal


def _collect_comments_text(issue: dict) -> str:
    comments = issue.get("comments_preview") or []
    snippets: list[str] = []
    for comment in comments[:3]:
        body = (comment.get("body") or "").strip()
        if body:
            snippets.append(body)
    return " ".join(snippets)


def _dedupe_candidates(candidates: list[tuple[float, dict, IssueFixSignal]]) -> list[tuple[float, dict, IssueFixSignal]]:
    deduped: list[tuple[float, dict, IssueFixSignal]] = []
    for candidate in sorted(candidates, key=lambda item: -item[0]):
        title = candidate[2].blocker
        if any(_titles_are_similar(title, kept[2].blocker) for kept in deduped):
            continue
        deduped.append(candidate)
    return deduped


def _titles_are_similar(title_a: str, title_b: str) -> bool:
    norm_a = _normalize_title(title_a)
    norm_b = _normalize_title(title_b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True

    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    jaccard = len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))
    if jaccard >= 0.75:
        return True

    return SequenceMatcher(a=norm_a, b=norm_b).ratio() >= 0.86


def _normalize_title(title: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", title.lower())
    cleaned = re.sub(r"\b(issue|bug|help|please|urgent|problem)\b", " ", cleaned)
    return " ".join(cleaned.split())


def _extract_blocker_label(text: str) -> str | None:
    if _matches_install_failure(text):
        return "install failure"
    if _matches_missing_weights(text):
        return "missing weights/checkpoints"
    if _matches_missing_dataset(text):
        return "missing dataset / manual dataset bootstrap"
    if _matches_env_version_conflict(text):
        return "env/version conflict"
    if _matches_missing_run_command(text):
        return "missing run command"
    if _matches_runtime_api_error(text):
        return "runtime/API error"
    if _matches_stale_archived(text):
        return "stale/archived repo issue"
    return None


def _is_noisy_issue(title: str, body: str, combined_lower: str) -> bool:
    # Fix H5: removed early return that bypassed noise filter for categorizable
    # issues — subjective issues containing blocker keywords should still be
    # filtered when they match noise patterns.
    joined = f"{title}\n{body}".strip()
    return bool(joined) and any(pattern.search(joined) for pattern in NOISY_PATTERNS)


def _matches_install_failure(text: str) -> bool:
    return any(token in text for token in ("install", "dependency", "requirements", "pip", "conda")) and any(
        token in text for token in ("error", "fail", "failed", "cannot", "can't")
    )


def _matches_missing_weights(text: str) -> bool:
    return any(token in text for token in ("weight", "checkpoint", "model file", "artifact", "ckpt")) and any(
        token in text for token in ("missing", "not found", "absent", "not linked", "gated")
    )


def _matches_missing_dataset(text: str) -> bool:
    return any(token in text for token in ("dataset", "data")) and any(
        token in text for token in ("missing", "manual", "manually", "bootstrap", "download", "provide")
    )


def _matches_env_version_conflict(text: str) -> bool:
    return any(token in text for token in ("cuda", "python", "torch", "pytorch", "version", "dependency")) and any(
        token in text for token in ("mismatch", "incompatible", "conflict", "unsupported")
    )


def _matches_missing_run_command(text: str) -> bool:
    return any(token in text for token in ("run command", "how to run", "usage", "entrypoint", "start command")) and any(
        token in text for token in ("missing", "absent", "unclear", "not documented", "unknown")
    )


def _matches_runtime_api_error(text: str) -> bool:
    return any(
        token in text
        for token in ("runtime error", "traceback", "exception", "api", "importerror", "typeerror", "attributeerror", "crash", "broken")
    )


def _matches_stale_archived(text: str) -> bool:
    return any(token in text for token in ("archived", "stale", "unmaintained", "no longer maintained"))


def _extract_fix_signal(text: str) -> str | None:
    for pattern, label in _FIX_PATTERNS:
        if re.search(pattern, text):
            return label

    if "workaround" in text and "install" in text:
        return "install extra dependency"
    if "pin" in text and "version" in text:
        return "pin package version"
    return None
