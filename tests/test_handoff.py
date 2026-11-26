"""Plan 03 handoff tests: brief snapshot, truncation, delta, api_messages, compaction."""

from __future__ import annotations

import asyncio
from typing import Any

from onechat.config import HandoffConfig
from onechat.contracts import Message
from onechat.handoff import api_messages, delta_brief, handoff_brief, maybe_compact

CFG = HandoffConfig()


def session(**kw: Any) -> dict[str, Any]:
    base = {"id": "s1", "workdir": "/tmp/w", "summary": None, "summary_upto": 0}
    base.update(kw)
    return base


def turn(idx: int, role: str, text: str) -> dict[str, Any]:
    return {"idx": idx, "role": role, "text": text}


def test_brief_snapshot() -> None:
    turns = [turn(0, "user", "remember 4721"), turn(1, "assistant", "noted 4721")]
    brief = handoff_brief(session(), turns, " M file.py ", interrupted=False, cfg=CFG)
    assert brief.startswith("<asf-handoff>")
    assert brief.rstrip().endswith("</asf-handoff>")
    assert "remember 4721" in brief and "noted 4721" in brief
    assert "[user]" in brief and "[assistant]" in brief
    assert "M file.py" in brief
    assert "Summary of earlier conversation:\n(none)" in brief


def test_brief_with_prompt() -> None:
    turns = [turn(0, "user", "hi")]
    brief = handoff_brief(session(), turns, "", False, CFG, prompt="next question")
    assert brief.rstrip().endswith("next question")


def test_brief_truncates_oldest_first() -> None:
    cfg = HandoffConfig(brief_max_chars=400, turn_max_chars=50)
    turns = [turn(i, "user", f"turn number {i} " + "x" * 40) for i in range(6)]
    brief = handoff_brief(session(), turns, "", False, cfg)
    assert "turn number 5" in brief
    assert "turn number 0" not in brief  # oldest dropped first
    assert len(brief) < 400 + 300  # prompt-free brief stays near budget


def test_interrupted_note() -> None:
    brief = handoff_brief(session(), [turn(0, "user", "do it")], "", True, CFG)
    assert "interrupted mid-task" in brief


def test_delta_contains_only_missed() -> None:
    missed = [turn(3, "user", "q3"), turn(4, "assistant", "a4")]
    delta = delta_brief(missed, "", CFG)
    assert "Turns handled while you were away" in delta
    assert "q3" in delta and "a4" in delta


def test_api_messages_shape() -> None:
    from onechat.contracts import Message

    s = session(summary="the summary", summary_upto=2)
    turns = [
        turn(2, "user", "old q"),
        turn(3, "assistant", "old a"),
        turn(4, "user", "new q"),
        turn(5, "assistant", "new a"),
    ]
    messages = api_messages(s, turns, "final question", CFG)
    assert all(isinstance(m, Message) for m in messages)
    assert messages[0].role == "system" and "the summary" in messages[0].content
    assert messages[-1] == Message(role="user", content="final question")
    assert any(m.content == "new a" for m in messages)
    assert not any(m.content == "old q" for m in messages)  # pre-summary turns skipped


def test_api_messages_no_summary() -> None:
    messages = api_messages(session(), [turn(0, "user", "q"), turn(1, "assistant", "a")], "go", CFG)
    assert [m.role for m in messages] == ["user", "assistant", "user"]


def test_compaction_triggers_over_threshold() -> None:
    cfg = HandoffConfig(compact_threshold_chars=200, recent_turns=2)
    old = [turn(i, "user", "y" * 80) for i in range(4)]  # 4x80+idx > 200 chars, older than last 2
    recent = [turn(4, "user", "recent"), turn(5, "assistant", "recent reply")]
    calls: list[str] = []

    async def summarize(text: str) -> str:
        calls.append(text)
        return "NEW SUMMARY"

    compacted = asyncio.run(
        maybe_compact(session(summary=None, summary_upto=0), old + recent, cfg, summarize)
    )
    assert compacted is not None
    new_summary, upto = compacted
    assert upto == 3  # turns 0..3 summarized; recent window starts at turn 4
    assert new_summary == "NEW SUMMARY"
    assert len(calls) == 1 and "NEW" not in calls[0]
    assert "recent reply" not in calls[0]  # only older-than-recent turns compacted


def test_compaction_updates_summary_upto() -> None:
    """After compaction the stored summary_upto must cover exactly the summarized turns."""
    cfg = HandoffConfig(compact_threshold_chars=50, recent_turns=2)
    turns = [turn(i, "user", "t" * 30) for i in range(4)]

    async def summarize(text: str) -> str:
        return "S"

    compacted = asyncio.run(
        maybe_compact(session(summary=None, summary_upto=0), turns, cfg, summarize)
    )
    assert compacted is not None
    summary, upto = compacted
    assert upto == 1  # turns 0..1 summarized
    # api_messages keeps the un-summarized recent turns (2 and 3):
    messages = api_messages(session(summary=summary, summary_upto=upto), turns, "go", cfg)
    bodies = [m.content for m in messages if m.role != "system"]
    assert any(b == "t" * 30 for b in bodies) or len(bodies) >= 2
    assert messages[-1] == Message(role="user", content="go")


def test_compaction_skips_already_summarized() -> None:
    cfg = HandoffConfig(compact_threshold_chars=10, recent_turns=2)
    turns = [turn(i, "user", "t" * 30) for i in range(2, 6)]  # all <= summary_upto=6

    async def summarize(text: str) -> str:
        return "S"

    result = asyncio.run(maybe_compact(session(summary_upto=6), turns, cfg, summarize))
    assert result is None


# ---- T11: caps/order + async git_state -------------------------------------------------


def test_delta_brief_respects_brief_max_chars_and_recent_turns() -> None:
    """T11: delta keeps only the newest `recent_turns` missed turns within the budget."""
    cfg = HandoffConfig(recent_turns=2, brief_max_chars=300, turn_max_chars=60)
    missed = [turn(i, "user", f"delta number {i} " + "y" * 40) for i in range(5)]
    delta = delta_brief(missed, "", cfg)
    assert "delta number 4" in delta and "delta number 3" in delta  # newest kept
    assert "delta number 0" not in delta  # older than recent_turns dropped
    assert len(delta) < 300 + 200  # header overhead only


def test_handoff_brief_truncates_summary_before_dropping_recent_turns() -> None:
    """T11: an oversized summary is cut to half budget BEFORE recent turns are dropped."""
    cfg = HandoffConfig(recent_turns=3, brief_max_chars=600, turn_max_chars=40)
    turns = [turn(i, "user", f"keep number {i} " + "z" * 30) for i in range(3)]
    big_summary = "S" * 2000
    brief = handoff_brief(session(summary=big_summary), turns, "", False, cfg)
    assert "keep number 0" in brief and "keep number 2" in brief  # turns survive
    assert big_summary not in brief  # summary truncated, not the turns
    assert len(brief) < 600 + 300


def test_handoff_brief_recent_turns_zero_returns_no_turns() -> None:
    """T11: recent_turns <= 0 must not crash or include any turn block."""
    cfg = HandoffConfig(recent_turns=0)
    turns = [turn(0, "user", "doomed"), turn(1, "assistant", "also doomed")]
    brief = handoff_brief(session(), turns, "", False, cfg)
    assert "doomed" not in brief
    assert "(none)" in brief


def test_api_messages_respects_char_budget() -> None:
    """T11: total message length stays under the budget; newest turns win."""
    cfg = HandoffConfig(brief_max_chars=900, turn_max_chars=4000)
    turns = [turn(i, "user", "u" * 400) for i in range(4)]
    messages = api_messages(session(summary="S" * 100), turns, "FINAL", cfg)
    total = sum(len(m.content) for m in messages)
    assert total <= 900 + 20  # prompt is separate; history fits the budget
    assert "FINAL" in messages[-1].content


async def test_async_git_state_includes_staged_and_skips_diff_on_non_repo(tmp_path) -> None:
    """T11: async_git_state = staged + unstaged stat; non-repo => '' (never raises)."""
    import subprocess as _sp

    from onechat.gitstate import async_git_state

    repo = tmp_path / "repo"
    repo.mkdir()
    _sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    _sp.run(["git", "add", "a.txt"], cwd=repo, check=True)
    state = await async_git_state(repo)
    assert "a.txt" in state  # staged content appears in --cached --stat

    plain = tmp_path / "plain"
    plain.mkdir()
    assert await async_git_state(plain) == ""


async def test_maybe_compact_recent_turns_zero_summarizes_all() -> None:
    """Audit: recent_turns=0 must not slice turns[-0:] == turns[0:]."""
    from onechat.handoff import maybe_compact

    cfg = HandoffConfig(recent_turns=0, compact_threshold_chars=10)
    turns = [turn(0, "user", "a" * 50), turn(1, "assistant", "b" * 50)]
    seen: list[str] = []

    async def summarize(text: str) -> str:
        seen.append(text)
        return "SUM"

    res = await maybe_compact(session(summary_upto=-1), turns, cfg, summarize)
    assert res == ("SUM", 1)
    assert seen and "a" * 50 in seen[0]


def test_api_messages_truncates_oversized_summary() -> None:
    """Bug 10: api_messages must bound an oversized summary via brief_max_chars // 2."""
    from onechat.handoff import api_messages

    cfg = HandoffConfig(brief_max_chars=1000, turn_max_chars=400)
    turns = [turn(0, "user", "recent question")]
    msgs = api_messages(session(summary="S" * 5000, summary_upto=-1), turns, "FINAL", cfg)
    assert len(msgs[0].content) <= 550
    assert any("recent question" in m.content for m in msgs)
