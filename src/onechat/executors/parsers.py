"""Output parsers (Plan 02 §5 / T06): incremental ``StreamParser`` + batch ``parse``.

Field shapes come from the real captured fixtures in tests/fixtures/cli/ (Plan 01).
``feed()`` parses each line exactly once (O(N), fixes the O(n^2) sink) and returns that
line's new events; the batch ``parse()`` is StreamParser fed with all lines, so
incremental and batch results match exactly by construction.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..contracts import Event
from .parser_cc import CCFamily
from .parser_formats import Parsed
from .parser_tool_streams import AgyFamily, CodexFamily, OpencodeFamily

__all__ = ["Parsed", "StreamParser", "PARSERS", "parse"]

PARSERS: dict[str, type] = {  # name -> handler class (registry validated by config load)
    "claude_stream_json": CCFamily,
    "codex_jsonl": CodexFamily,
    "gemini_stream_json": CCFamily,
    "agy_stream_json": AgyFamily,
    "opencode_json": OpencodeFamily,
    "qwen_stream_json": CCFamily,
}


class StreamParser:
    """Incremental parser for one turn's output stream (T06/T07)."""

    def __init__(self, parser_name: str) -> None:
        handler = PARSERS.get(parser_name)
        if handler is None:
            raise KeyError(f"unknown parser: {parser_name} (known: {sorted(PARSERS)})")
        self._handler = handler()

    def feed(self, line: str) -> list[Event]:
        """Parse one line; returns the NEW events it produced."""
        return self._handler.feed(line)

    def finalize(self) -> Parsed:
        return self._handler.finalize()


def parse(parser_name: str, lines: Iterable[str]) -> Parsed:
    """Batch parse: fresh StreamParser over all lines (identical to incremental)."""
    stream = StreamParser(parser_name)  # unknown names are a programming error (raise)
    for line in lines:
        stream.feed(line)
    return stream.finalize()
