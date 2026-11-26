"""Tool-centric stream families (T06): codex exec --json, agy result, opencode json."""

from __future__ import annotations

from ..contracts import Event, Usage
from ..redact import redact
from .parser_formats import Parsed, _dict_or, _safe_int, _safe_json, _tool_target


class CodexFamily:
    """codex exec --json (Plan 09 K2, exec_events.rs). Usage accumulates per turn."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.events: list[Event] = []
        self.errors: list[str] = []
        self.session_id: str | None = None
        self._input = 0
        self._output = 0

    def feed(self, line: str) -> list[Event]:
        data = _safe_json(line)
        if data is None:
            return []
        before = len(self.events)
        self._dispatch(data)
        return self.events[before:]

    def _dispatch(self, data: dict) -> None:
        kind = data.get("type")
        if kind == "thread.started" and data.get("thread_id"):
            self.session_id = str(data["thread_id"])
        elif kind == "item.completed":
            self._item(data)
        elif kind == "error":
            self.errors.append(str(data.get("message") or data.get("error") or "stream error"))
        elif kind == "turn.failed":
            self._turn_failed(data)
        elif kind == "turn.completed":
            usage = _dict_or(data.get("usage"))
            self._input += _safe_int(usage.get("input_tokens"))
            self._output += _safe_int(usage.get("output_tokens"))

    def _turn_failed(self, data: dict) -> None:
        err = data.get("error") or {}
        message = err.get("message") if isinstance(err, dict) else err
        self.errors.append(str(message or "turn failed"))

    def _item(self, data: dict) -> None:
        item = data.get("item")
        payload = _dict_or(_dict_or(item).get("details")) or _dict_or(item)
        item_type = str(payload.get("type") or _dict_or(item).get("type") or "")
        if item_type == "agent_message":
            text = payload.get("text") or payload.get("message") or ""
            if text:
                self.texts.append(str(text))
                self.events.append(Event(kind="text", text=str(text)))
        elif item_type == "command_execution":
            self._command(payload)
        elif item_type in ("mcp_tool_call", "collab_tool_call"):
            label = item_type.replace("_tool_call", "")
            self.events.append(
                Event(kind="tool", text=f"tool: {label} {payload.get('server') or ''}".strip())
            )
        elif item_type == "file_change":
            changes = payload.get("changes") or []
            first = ""
            if changes and isinstance(changes[0], dict):
                first = str(changes[0].get("path") or "")
            text = f"tool: file_change {redact(first)}".strip()
            self.events.append(Event(kind="tool", text=text))
        elif item_type in ("todo_list", "web_search"):
            self.events.append(Event(kind="tool", text=f"tool: {item_type}"))
        elif item_type == "error":
            self.errors.append(str(payload.get("message") or "item error"))

    def _command(self, payload: dict) -> None:
        if str(payload.get("status") or "") == "declined":  # healthy: policy said no
            self.events.append(Event(kind="tool", text="tool: command (declined)"))
        else:
            target = _tool_target({"command": payload.get("command")})
            self.events.append(Event(kind="tool", text=f"tool: command{target}"))

    def finalize(self) -> Parsed:
        out = Parsed(
            text="\n".join(self.texts),
            session_id=self.session_id,
            usage=Usage(input_tokens=self._input, output_tokens=self._output),
            error="; ".join(self.errors) or None,
            events=self.events,
        )
        if out.text == "" and out.error is None and not any(e.kind == "tool" for e in out.events):
            out.error = "no agent message in stream"  # a declined-only turn DID work
        return out


class AgyFamily:
    """agy: {event: result, result: {conversation_id, status, response, error, usage}}."""

    def __init__(self) -> None:
        self.text: str = ""
        self.events: list[Event] = []
        self.session_id: str | None = None
        self.usage = Usage()
        self.error: str | None = None

    def feed(self, line: str) -> list[Event]:
        data = _safe_json(line)
        if data is None or data.get("event") != "result":
            return []
        before = len(self.events)
        result = _dict_or(data.get("result"))
        self.session_id = result.get("conversation_id") or None
        self.text = str(result.get("response") or "")
        if self.text:
            self.events.append(Event(kind="text", text=self.text))
        if result.get("status") == "ERROR" or result.get("error"):
            self.error = str(result.get("error") or "agy error")
        usage = _dict_or(result.get("usage"))
        self.usage = Usage(
            input_tokens=_safe_int(usage.get("input_tokens")),
            output_tokens=_safe_int(usage.get("output_tokens")),
        )
        return self.events[before:]

    def finalize(self) -> Parsed:
        out = Parsed(
            text=self.text,
            session_id=self.session_id,
            usage=self.usage,
            error=self.error,
            events=self.events,
        )
        if out.text == "" and out.error is None:
            out.error = "no agy result event"
        return out


class OpencodeFamily:
    """opencode run --format json: text/tool/error events; usage SUMS across step_finish."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.events: list[Event] = []
        self.errors: list[str] = []
        self._retry_errors: list[str] = []
        self.session_id: str | None = None
        self._input = 0
        self._output = 0
        self._cost: float | None = None

    def feed(self, line: str) -> list[Event]:
        data = _safe_json(line)
        if data is None:
            return []
        before = len(self.events)
        self._dispatch(data)
        return self.events[before:]

    def _dispatch(self, data: dict) -> None:
        session = data.get("sessionID")
        if session and self.session_id is None:
            self.session_id = str(session)
        kind = data.get("type")
        part = _dict_or(data.get("part"))
        if kind == "text" and part.get("text"):
            self.texts.append(str(part["text"]))
            self.events.append(Event(kind="text", text=str(part["text"])))
        elif kind in ("step_finish", "step-finish") or part.get("type") == "step-finish":
            self._tokens(_dict_or(part.get("tokens")))
        elif kind == "tool" and part.get("tool"):
            self._tool(part)
        elif kind == "retry":
            self._record_retry(data.get("error"))
        elif kind == "error" or data.get("error"):
            self._error(data.get("error"))

    def _tool(self, part: dict) -> None:
        title = part.get("title") if isinstance(part.get("title"), str) else ""
        self.events.append(
            Event(kind="tool", text=f"tool: {part.get('tool')} {redact(title)}".strip())
        )

    def _tokens(self, tokens: dict) -> None:
        self._input += _safe_int(tokens.get("input"))
        self._output += _safe_int(tokens.get("output"))
        cost = tokens.get("cost")
        if isinstance(cost, (int, float)):
            self._cost = (self._cost or 0.0) + float(cost)

    def _extract_err_text(self, err: object) -> str:
        if isinstance(err, str):
            return err or "opencode error"
        info = _dict_or(err)
        body = str(_dict_or(info.get("response")).get("body") or "")
        return str(info.get("message") or body or "opencode error")

    def _error(self, err: object) -> None:
        self.errors.append(self._extract_err_text(err))

    def _record_retry(self, err: object) -> None:
        self._retry_errors.append(self._extract_err_text(err))

    def finalize(self) -> Parsed:
        errs = list(self.errors)
        has_progress = bool(self.texts or any(e.kind == "tool" for e in self.events))
        if not has_progress and self._retry_errors:
            errs.extend(self._retry_errors)
        out = Parsed(
            text="\n".join(self.texts),
            session_id=self.session_id,
            usage=Usage(input_tokens=self._input, output_tokens=self._output, cost_usd=self._cost),
            error="; ".join(errs) or None,
            events=self.events,
        )
        if out.text == "" and out.error is None:
            out.error = "no text part in opencode stream"
        return out
