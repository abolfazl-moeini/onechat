"""claude-code-like stream family (T06): claude, qwen, gemini + gemini-native extras.

Shapes: type=assistant (content blocks) / type=result (final text+usage) as captured in
tests/fixtures/cli/; gemini-native type=message / type=stats handled too (no live fixture
yet — compat.md §1 marks gemini turn-level shapes unverified).
"""

from __future__ import annotations

from ..contracts import Event, Usage
from .parser_formats import Parsed, _cc_usage, _dict_or, _safe_int, _safe_json, _tool_target


class CCFamily:
    """claude-code-like result family: claude, qwen, gemini (+ gemini-native extras)."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.events: list[Event] = []
        self.session_id: str | None = None
        self.usage = Usage()
        self.error: str | None = None

    def feed(self, line: str) -> list[Event]:
        data = _safe_json(line)
        if data is None:
            return []
        before = len(self.events)
        self._dispatch(data)
        return self.events[before:]

    def _dispatch(self, data: dict) -> None:
        session = data.get("session_id")
        if session and self.session_id is None:
            self.session_id = str(session)
        kind = data.get("type")
        if kind == "assistant":
            self._assistant(data)
        elif kind == "result":
            self._result(data)
        elif kind == "message":  # gemini-native stream shape
            self._message(data)
        elif kind == "stats":  # gemini-native usage shape
            self._stats(data)

    def _add_text(self, text: str) -> None:
        self.texts.append(text)
        self.events.append(Event(kind="text", text=text))

    def _assistant(self, data: dict) -> None:
        content = _dict_or(data.get("message")).get("content")
        if isinstance(content, str):
            self._add_text(content)
            return
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                self._add_text(str(block["text"]))
            elif block.get("type") == "tool_use":
                summary = f"tool: {block.get('name', 'unknown')}{_tool_target(block.get('input'))}"
                self.events.append(Event(kind="tool", text=summary))

    def _message(self, data: dict) -> None:
        if data.get("role") == "user":
            return
        content = (
            data.get("content")
            if "content" in data
            else _dict_or(data.get("message")).get("content")
        )
        if isinstance(content, str):
            self._add_text(content)
            return
        for part in content if isinstance(content, list) else []:
            text = part.get("text") if isinstance(part, dict) else part
            if isinstance(text, str) and text:
                self._add_text(text)

    def _stats(self, data: dict) -> None:
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else data
        self.usage = Usage(
            input_tokens=_safe_int(usage.get("input_tokens")),
            output_tokens=_safe_int(usage.get("output_tokens")),
        )

    def _result(self, data: dict) -> None:
        stats = data.get("stats")
        if isinstance(stats, dict):
            self._stats(stats)
        else:
            self.usage = _cc_usage(data)
        if data.get("is_error") or data.get("subtype") == "error_during_execution":
            self._result_error(data)
            return
        result_text = str(data.get("result") or "")
        if not result_text:
            return
        joined = "\n".join(t for t in self.texts if t)
        if not self.texts or (result_text != self.texts[-1] and result_text != joined):
            self._add_text(result_text)

    def _result_error(self, data: dict) -> None:
        err = data.get("error")
        errors = data.get("errors")
        if isinstance(err, dict):
            msg = str(err.get("message") or "")
        elif err:
            msg = str(err)
        elif data.get("result"):
            msg = str(data["result"])
        elif isinstance(errors, list) and errors:
            msg = "; ".join(str(e) for e in errors)
        else:
            msg = self.texts[-1] if self.texts else "error"
        self.error = msg
        self.texts.clear()

    def finalize(self) -> Parsed:
        out = Parsed(
            text="\n".join(t for t in self.texts if t),
            session_id=self.session_id,
            usage=self.usage,
            error=self.error,
            events=self.events,
        )
        if out.text == "" and out.error is None and not any(e.kind == "tool" for e in out.events):
            out.error = "no assistant text in stream"
        return out
