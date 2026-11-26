"""Handoff briefs (Plan 03 §4 / T11): cross-backend session continuity prompts.

Caps: recent_turns and brief_max_chars bound every brief; an oversized summary is
truncated BEFORE recent turns are dropped; api_messages keeps summary + newest turns
within the budget. Pure functions.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .config import HandoffConfig
from .contracts import Message

BRIEF_HEADER = (
    "<asf-handoff>\n"
    "You are continuing a conversation started by another assistant. "
    "Continue seamlessly; do not mention the switch.\n"
)


def _format_turn(t: dict[str, Any], max_chars: int) -> str:
    role = t.get("role", "user")
    text = str(t.get("text", ""))[:max_chars]
    return f"[{role}] {text}"


def _turns_block(turns: list[dict[str, Any]], cfg: HandoffConfig) -> str:
    return "\n".join(_format_turn(t, cfg.turn_max_chars) for t in turns)


def _bounded_summary(session: dict[str, Any], cfg: HandoffConfig) -> str:
    """Summary truncated to half the brief budget (T11) — '(none)' when absent."""
    summary = str(session.get("summary") or "")
    if not summary:
        return "(none)"
    return summary[: cfg.brief_max_chars // 2]


def handoff_brief(
    session: dict[str, Any],
    turns: list[dict[str, Any]],
    git: str,
    interrupted: bool,
    cfg: HandoffConfig,
    prompt: str = "",
) -> str:
    """Full continuation brief: summary + recent turns + git state + optional note + prompt."""
    recent = _bounded_recent(session, turns, git, cfg)
    summary = _bounded_summary(session, cfg)
    parts = [
        BRIEF_HEADER,
        f"Working directory: {session.get('workdir', '')}\n",
        "Summary of earlier conversation:\n",
        f"{summary}\n",
        "Recent turns, oldest first:\n",
        f"{_turns_block(recent, cfg) or '(none)'}\n",
        "Workspace state:\n",
        f"{git or '(not a git repo)'}\n",
    ]
    if interrupted:
        parts.append(
            "The previous assistant was interrupted mid-task. Inspect the workspace "
            "(git status/diff) and finish the task.\n"
        )
    parts.append("</asf-handoff>\n\n")
    parts.append(prompt)
    return "".join(parts)


def _bounded_recent(
    session: dict[str, Any],
    turns: list[dict[str, Any]],
    git: str,
    cfg: HandoffConfig,
) -> list[dict[str, Any]]:
    """Newest `recent_turns` turns (T11: <= 0 -> []), dropping oldest first over budget."""
    if cfg.recent_turns <= 0:
        return []
    recent = turns[-cfg.recent_turns :]
    while len(recent) > 1 and _brief_size(session, recent, git, cfg) > cfg.brief_max_chars:
        recent = recent[1:]
    return recent


def _brief_size(
    session: dict[str, Any], recent: list[dict[str, Any]], git: str, cfg: HandoffConfig
) -> int:
    return (
        len(BRIEF_HEADER)
        + len(_turns_block(recent, cfg))
        + len(git)
        + min(len(str(session.get("summary") or "")), cfg.brief_max_chars // 2)
    )


def delta_brief(missed: list[dict[str, Any]], git: str, cfg: HandoffConfig) -> str:
    """Turns handled by other backends while this one was away (T11: capped)."""
    bounded = [] if cfg.recent_turns <= 0 else missed[-cfg.recent_turns :]
    block = _turns_block(bounded, cfg)
    if len(block) > cfg.brief_max_chars:
        block = block[-cfg.brief_max_chars :]
    parts = [
        BRIEF_HEADER,
        "Turns handled while you were away, oldest first:\n",
        f"{block or '(none)'}\n",
        "Workspace state:\n",
        f"{git or '(not a git repo)'}\n",
        "</asf-handoff>\n\n",
    ]
    return "".join(parts)


def _summary_upto(session: dict[str, Any]) -> int:
    """Last turn idx included in the summary. DB default 0 with no summary means 'none yet'
    (turns start at idx 0), expressed as -1 so nothing is skipped."""
    raw = session.get("summary_upto")
    if not session.get("summary") and (raw is None or int(raw) == 0):
        return -1
    return int(raw if raw is not None else -1)


def api_messages(
    session: dict[str, Any],
    turns: list[dict[str, Any]],
    prompt: str,
    cfg: HandoffConfig,
) -> list[Message]:
    """OpenAI-style history (T11): summary + turns AFTER summary_upto, newest-first kept
    within brief_max_chars; the current prompt is always the final user message."""
    upto = _summary_upto(session)
    budget = max(cfg.brief_max_chars - len(prompt) - 50, 500)
    history: list[Message] = []
    if session.get("summary"):
        header = f"Summary of earlier conversation:\n{_bounded_summary(session, cfg)}"
        history.append(Message(role="system", content=header))
    body = [t for t in turns if int(t.get("idx", 0)) > upto]
    body_budget = budget - (len(history[0].content) if history else 0)
    chosen: list[dict[str, Any]] = []
    used = 0
    for t in reversed(body):
        cost = min(len(str(t.get("text", ""))), cfg.turn_max_chars)
        if used + cost > body_budget and chosen:
            break
        chosen.append(t)
        used += cost
    chosen.reverse()
    for t in chosen:
        role = "assistant" if t.get("role") == "assistant" else "user"
        history.append(Message(role=role, content=str(t.get("text", ""))[: cfg.turn_max_chars]))
    history.append(Message(role="user", content=prompt))
    return history


async def maybe_compact(
    session: dict[str, Any],
    turns: list[dict[str, Any]],
    cfg: HandoffConfig,
    summarize: Callable[[str], Awaitable[str]],
) -> tuple[str, int] | None:
    """Summarize old turns past the threshold; return (new summary, last summarized idx)."""
    upto = _summary_upto(session)
    if cfg.recent_turns <= 0:
        keep_from = int(turns[-1]["idx"]) + 1 if turns else 0
    elif len(turns) > cfg.recent_turns:
        keep_from = int(turns[-cfg.recent_turns]["idx"])
    else:
        keep_from = 0
    old = [t for t in turns if upto < int(t.get("idx", 0)) < keep_from]
    total_chars = sum(len(str(t.get("text", ""))) for t in old)
    if total_chars <= cfg.compact_threshold_chars:
        return None
    previous = str(session.get("summary") or "")
    summary = await summarize(previous + "\n" + _turns_block(old, cfg))
    return summary, int(keep_from) - 1
