"""Shared parser plumbing (T06): Parsed model, tolerant JSON/typing helpers, registry.

Handler classes live next to their stream families: ``parser_cc.py`` (claude-code-like),
``parser_tool_streams.py`` (codex / agy / opencode). Wrong-typed nested fields are a
no-op, never a crash (garbage lines are ignored, Plan 02 §5).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from ..contracts import Event, Usage
from ..redact import redact

__all__ = ["Parsed", "_safe_json", "_safe_int", "_dict_or", "_tool_target", "_cc_usage", "redact"]


class Parsed(BaseModel):
    text: str = ""
    session_id: str | None = None
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None
    events: list[Event] = Field(default_factory=list)


def _safe_json(line: str) -> dict | None:
    stripped = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _dict_or(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _tool_target(tool_input: object) -> str:
    """First recognizable target field of a tool_use input (Plan 02 §5 example)."""
    if not isinstance(tool_input, dict):
        return ""
    for key in ("file_path", "path", "notebook", "url", "command", "pattern"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return f" {redact(value)}"
    return ""


def _cc_usage(data: dict) -> Usage:
    """input_tokens includes cache writes/reads (T06) — they are billed input too."""
    usage = _dict_or(data.get("usage"))

    def _int(key: str) -> int:
        return _safe_int(usage.get(key))

    input_tokens = _int("input_tokens") + _int("cache_creation_input_tokens")
    input_tokens += _int("cache_read_input_tokens")
    return Usage(
        input_tokens=input_tokens,
        output_tokens=_int("output_tokens"),
        cost_usd=data.get("total_cost_usd"),
    )
