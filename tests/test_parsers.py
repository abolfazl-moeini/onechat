"""Plan 02 parser tests.

Parametrized over the REAL captured fixtures (Plan 01): tonight all turn fixtures are
error streams, so live-fixture assertions check `error is not None` and no-crash on
garbage lines. Success-path assertions use minimal synthetic streams whose field shapes
are taken from the same captured schemas (noted per test); full success captures are
re-verification checklist item #1 in docs/compat.md §6.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from onechat.executors.parsers import PARSERS, parse

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "cli"


def fixture_profiles(case: str) -> list[Path]:
    return sorted(p for p in FIXTURE_ROOT.glob(f"*/{case}.jsonl"))


# ---- live-fixture parametrization ---------------------------------------------------


@pytest.mark.parametrize("path", fixture_profiles("new_turn"), ids=lambda p: p.parent.name)
def test_new_turn_fixtures_parse_without_raising(path: Path) -> None:
    parser_name = _parser_for(path.parent.name)
    parsed = parse(parser_name, path.read_text(encoding="utf-8").splitlines())
    if path.parent.name == "opencode":
        assert parsed.session_id is not None  # sessionID present even on error
    assert parsed.error is not None or parsed.text != ""


@pytest.mark.parametrize("path", fixture_profiles("new_turn"), ids=lambda p: p.parent.name)
def test_fixture_garbage_lines_never_crash(path: Path) -> None:
    lines = ["", "not json at all", "{broken", *path.read_text(encoding="utf-8").splitlines()]
    parsed = parse(_parser_for(path.parent.name), lines)
    assert parsed is not None


def _parser_for(profile: str) -> str:
    import yaml

    profile_path = (
        Path(__file__).resolve().parent.parent.parent
        / "packages"
        / "asf-router"
        / "src"
        / "onechat"
        / "profiles"
        / f"{profile}.yaml"
    )
    return yaml.safe_load(profile_path.read_text(encoding="utf-8"))["parser"]


# ---- success-path synthetic streams (schema-derived from captured fixtures) ----------

CLAUDE_SUCCESS = [
    json.dumps({"type": "system", "subtype": "init", "session_id": "s-abc"}),
    json.dumps(
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "PONG"}], "model": "claude-x"},
            "session_id": "s-abc",
        }
    ),
    json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "session_id": "s-abc",
            "is_error": False,
            "result": "PONG",
            "usage": {"input_tokens": 12, "output_tokens": 3},
            "total_cost_usd": 0.001,
        }
    ),
]

CODEX_SUCCESS = [
    json.dumps({"type": "thread.started", "thread_id": "thr-1"}),
    json.dumps(
        {"type": "item.completed", "item": {"id": "i0", "type": "agent_message", "text": "PONG"}}
    ),
    json.dumps(
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 2},
        }
    ),
]

AGY_SUCCESS = [
    json.dumps(
        {
            "event": "result",
            "result": {
                "conversation_id": "conv-1",
                "status": "COMPLETED",
                "response": "PONG",
                "error": "",
                "num_turns": 1,
                "usage": {"input_tokens": 9, "output_tokens": 2},
            },
        }
    ),
]

# opencode text/step-finish part shapes are schema-inferred (only step_start/error captured
# tonight); tokens carried on step-finish parts
OPENCODE_SUCCESS = [
    json.dumps(
        {"type": "step_start", "sessionID": "ses-1", "part": {"id": "p0", "type": "step-start"}}
    ),
    json.dumps(
        {"type": "text", "sessionID": "ses-1", "part": {"id": "p1", "type": "text", "text": "PONG"}}
    ),
    json.dumps(
        {
            "type": "step_finish",
            "sessionID": "ses-1",
            "part": {
                "id": "p2",
                "type": "step-finish",
                "tokens": {"input": 7, "output": 2, "cost": 0.0},
            },
        }
    ),
]

CC_RESULT_SUCCESS = [  # claude-code-like result family (gemini/qwen keys)
    json.dumps({"type": "system", "subtype": "init", "session_id": "q-1"}),
    json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "PONG"}]}}),
    json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "session_id": "q-1",
            "result": "PONG",
            "is_error": False,
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
    ),
]


@pytest.mark.parametrize(
    ("parser", "lines", "session"),
    [
        ("claude_stream_json", CLAUDE_SUCCESS, "s-abc"),
        ("codex_jsonl", CODEX_SUCCESS, "thr-1"),
        ("agy_stream_json", AGY_SUCCESS, "conv-1"),
        ("opencode_json", OPENCODE_SUCCESS, "ses-1"),
        ("gemini_stream_json", CC_RESULT_SUCCESS, "q-1"),
        ("qwen_stream_json", CC_RESULT_SUCCESS, "q-1"),
    ],
)
def test_success_stream_yields_text_session_usage(
    parser: str,
    lines: list[str],
    session: str,
) -> None:
    parsed = parse(parser, lines)
    assert "PONG" in parsed.text
    assert parsed.session_id == session
    assert parsed.error is None
    assert parsed.usage.input_tokens > 0 or parsed.usage.output_tokens > 0


def test_claude_tool_use_becomes_tool_event() -> None:
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": "s-1"}),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Write",
                            "input": {"file_path": "/tmp/hello.txt"},
                        }
                    ]
                },
            }
        ),
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": "done",
                "session_id": "s-1",
                "usage": {},
            }
        ),
    ]
    parsed = parse("claude_stream_json", lines)
    assert any(
        e.kind == "tool" and "Write" in e.text and "/tmp/hello.txt" in e.text for e in parsed.events
    )


def test_codex_error_item_and_error_event_classified() -> None:
    lines = [
        json.dumps({"type": "thread.started", "thread_id": "thr-9"}),
        json.dumps({"type": "error", "message": "boom 403"}),
    ]
    parsed = parse("codex_jsonl", lines)
    assert parsed.error == "boom 403"
    assert parsed.session_id == "thr-9"


def test_all_six_parsers_registered() -> None:
    assert set(PARSERS) == {
        "claude_stream_json",
        "codex_jsonl",
        "gemini_stream_json",
        "agy_stream_json",
        "opencode_json",
        "qwen_stream_json",
    }


def test_empty_stream_is_error() -> None:
    parsed = parse("claude_stream_json", [])
    assert parsed.error is not None and parsed.text == ""


def test_agy_emits_final_text_event() -> None:
    """Plan 02 §5: every parser yields events for the chat UI (agy: one final text event)."""
    parsed = parse("agy_stream_json", AGY_SUCCESS)
    assert any(e.kind == "text" and "PONG" in e.text for e in parsed.events)


# --- Plan 09 K2: codex wire-spec alignment (exec_events.rs facts) ---


def test_codex_declined_command_is_not_error() -> None:
    """command_execution with status "declined" is a healthy outcome, never a failure."""
    lines = [
        json.dumps({"type": "thread.started", "thread_id": "thr-7"}),
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "i1",
                    "type": "command_execution",
                    "command": "rm -rf /",
                    "status": "declined",
                },
            }
        ),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 1}}),
    ]
    parsed = parse("codex_jsonl", lines)
    assert parsed.error is None
    assert any(e.kind == "tool" and "declined" in e.text for e in parsed.events)


def test_codex_tool_items_become_tool_events() -> None:
    """file_change / todo_list / web_search items surface as tool events for the chat UI."""
    lines = [
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "i1",
                    "type": "file_change",
                    "changes": [{"path": "src/a.py", "kind": "add"}],
                    "status": "completed",
                },
            }
        ),
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "i2",
                    "type": "todo_list",
                    "items": [{"text": "x", "completed": False}],
                },
            }
        ),
        json.dumps(
            {"type": "item.completed", "item": {"id": "i3", "type": "web_search", "query": "asf"}}
        ),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}),
    ]
    parsed = parse("codex_jsonl", lines)
    tool_texts = [e.text for e in parsed.events if e.kind == "tool"]
    assert any("file_change" in k and "src/a.py" in k for k in tool_texts)
    assert any("todo_list" in k for k in tool_texts)
    assert any("web_search" in k for k in tool_texts)


def test_codex_turn_failed_sets_error() -> None:
    lines = [json.dumps({"type": "turn.failed", "error": {"message": "quota exhausted"}})]
    parsed = parse("codex_jsonl", lines)
    assert parsed.error == "quota exhausted"


def test_codex_usage_maps_input_output_and_ignores_extras() -> None:
    """cached_input_tokens/reasoning_output_tokens exist upstream (compat.md §codex);
    our Usage contract stays input/output."""
    lines = [
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cached_input_tokens": 7,
                    "reasoning_output_tokens": 1,
                },
            }
        )
    ]
    parsed = parse("codex_jsonl", lines)
    assert parsed.usage.input_tokens == 10
    assert parsed.usage.output_tokens == 2


# ---- T06: incremental StreamParser + parser bug fixes --------------------------------


def test_cc_stream_json_does_not_double_final_text() -> None:
    """T06: the result event's text is the SAME answer as the assistant block — no dupe."""
    lines = [
        json.dumps(
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "PONG"}]}}
        ),
        json.dumps({"type": "result", "subtype": "success", "result": "PONG", "usage": {}}),
    ]
    parsed = parse("claude_stream_json", lines)
    assert parsed.text == "PONG"


def test_opencode_accumulates_usage_across_steps() -> None:
    """T06: usage SUMS across step_finish events (multi-step turns under-counted before)."""
    lines = [
        json.dumps(
            {
                "type": "step_finish",
                "sessionID": "ses-1",
                "part": {"type": "step-finish", "tokens": {"input": 7, "output": 2, "cost": 0.25}},
            }
        ),
        json.dumps(
            {
                "type": "step_finish",
                "sessionID": "ses-1",
                "part": {"type": "step-finish", "tokens": {"input": 3, "output": 1, "cost": 0.5}},
            }
        ),
    ]
    parsed = parse("opencode_json", lines)
    assert parsed.usage.input_tokens == 10
    assert parsed.usage.output_tokens == 3
    assert parsed.usage.cost_usd == pytest.approx(0.75)


def test_opencode_transient_retry_does_not_fail_successful_turn() -> None:
    lines = [
        json.dumps({"type": "retry", "sessionID": "ses-1", "error": "rate limit 429"}),
        json.dumps({"type": "text", "sessionID": "ses-1", "part": {"text": "hello world"}}),
    ]
    parsed = parse("opencode_json", lines)
    assert parsed.text == "hello world"
    assert parsed.error is None


def test_opencode_retry_without_text_preserves_error() -> None:
    lines = [
        json.dumps({"type": "retry", "sessionID": "ses-1", "error": "rate limit 429"}),
    ]
    parsed = parse("opencode_json", lines)
    assert parsed.text == ""
    assert "rate limit 429" in str(parsed.error)


GEMINI_NATIVE = [
    json.dumps({"type": "init", "session_id": "g-1"}),
    json.dumps({"type": "message", "message": {"role": "assistant", "content": "PONG"}}),
    json.dumps({"type": "stats", "input_tokens": 8, "output_tokens": 2}),
]


def test_gemini_stream_json_parses_message_and_stats() -> None:
    """T06: gemini stream-json carries type=message text + type=stats usage."""
    parsed = parse("gemini_stream_json", GEMINI_NATIVE)
    assert parsed.text == "PONG"
    assert parsed.session_id == "g-1"
    assert parsed.usage.input_tokens == 8
    assert parsed.usage.output_tokens == 2
    assert parsed.error is None


def test_parsers_survive_malformed_nested_types() -> None:
    """T06: wrong-typed nested fields never raise — garbage in, honest Parsed out."""
    codex = parse(
        "codex_jsonl",
        [
            json.dumps({"type": "item.completed", "item": []}),
            json.dumps({"type": "error", "error": "plain"}),
        ],
    )
    assert codex is not None
    opencode = parse(
        "opencode_json",
        [json.dumps({"type": "error", "sessionID": "ses-9", "error": "plain"})],
    )
    assert opencode.error is not None
    assert opencode.session_id == "ses-9"


@pytest.mark.parametrize(
    ("parser", "lines"),
    [
        ("claude_stream_json", CLAUDE_SUCCESS),
        ("codex_jsonl", CODEX_SUCCESS),
        ("agy_stream_json", AGY_SUCCESS),
        ("opencode_json", OPENCODE_SUCCESS),
        ("gemini_stream_json", CC_RESULT_SUCCESS),
        ("qwen_stream_json", CC_RESULT_SUCCESS),
    ],
)
def test_incremental_stream_parser_matches_batch(parser: str, lines: list[str]) -> None:
    """T06: feeding one-by-one into StreamParser equals the batch parse exactly."""
    from onechat.executors.parsers import StreamParser

    sp = StreamParser(parser)
    incremental: list = []
    for line in lines:
        incremental.extend(sp.feed(line))
    final = sp.finalize()
    batch = parse(parser, lines)
    assert final == batch
    assert incremental == batch.events


def test_gemini_stream_json_top_level_content_and_ignores_user_echo() -> None:
    """Bug 6: Gemini CLI emits top-level content + role='user' echoes."""
    lines = [
        json.dumps({"type": "message", "role": "user", "content": "Say PONG"}),
        json.dumps({"type": "message", "role": "assistant", "content": "PONG"}),
        json.dumps({"type": "result", "status": "success", "stats": {"input_tokens": 5}}),
    ]
    parsed = parse("gemini_stream_json", lines)
    assert parsed.text == "PONG"
    assert parsed.error is None


def test_cc_result_empty_and_multiblock_no_duplicate() -> None:
    """Bug 6: empty result='' and joined multi-block result never duplicate text/events."""
    lines = [
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "part1"},
                        {"type": "text", "text": "part2"},
                    ]
                },
            }
        ),
        json.dumps({"type": "result", "result": "part1\npart2"}),
    ]
    parsed = parse("claude_stream_json", lines)
    assert parsed.text == "part1\npart2"


def test_cc_is_error_uses_result_and_clears_text() -> None:
    """Bug 6: is_error=True extracts data['result']/'errors' and clears out.text."""
    lines = [
        json.dumps(
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "bad"}]}}
        ),
        json.dumps({"type": "result", "is_error": True, "result": "rate limit hit"}),
    ]
    parsed = parse("claude_stream_json", lines)
    assert parsed.error == "rate limit hit"
    assert parsed.text == ""


def test_cc_tool_only_turn_valid() -> None:
    """Bug 6: tool-only CC turns with no assistant text must not set error."""
    lines = [
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {}}]},
            }
        )
    ]
    parsed = parse("claude_stream_json", lines)
    assert parsed.error is None
    assert any(e.kind == "tool" for e in parsed.events)


def test_agy_feed_returns_only_new_events() -> None:
    """Bug 6: AgyFamily.feed returns only newly added events per line."""
    from onechat.executors.parsers import StreamParser

    sp = StreamParser("agy_stream_json")
    ev1 = sp.feed(json.dumps({"event": "result", "result": {"response": "hi"}}))
    ev2 = sp.feed(json.dumps({"event": "result", "result": {"response": "bye"}}))
    assert len(ev1) == 1
    assert len(ev2) == 1


def test_parsers_survive_non_numeric_token_counts() -> None:
    """Bug 6: malformed non-numeric token counts must not raise ValueError/TypeError."""
    lines = [
        json.dumps({"type": "result", "result": "ok", "usage": {"input_tokens": "N/A"}}),
    ]
    parsed = parse("claude_stream_json", lines)
    assert parsed.text == "ok"
    assert parsed.usage.input_tokens == 0
