"""Plan 03 router tests: resume/delta/handoff switching, failover, leases, events."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest

from onechat.config import HandoffConfig, RouterConfig, TierPolicy
from onechat.contracts import (
    Capability,
    CostClass,
    ExecutorSpec,
    Stop,
    Tier,
    TurnRequest,
    TurnResult,
)
from onechat.quota import QuotaBook
from onechat.router import NoExecutorAvailable, Reply, Router, SessionBusy
from onechat.selector import Need
from onechat.store import Store


class FakeExecutor:
    def __init__(self, spec: ExecutorSpec, results: list[TurnResult]) -> None:
        self.spec = spec
        self.results = list(results)
        self.requests: list[TurnRequest] = []

    def available(self) -> bool:
        return True

    async def run(self, req: TurnRequest, on_event: Any = None) -> TurnResult:
        self.requests.append(req)
        return self.results.pop(0)


def cli_spec(eid: str, family: str = "meta", tier: int = 2) -> ExecutorSpec:
    return ExecutorSpec(
        id=eid,
        kind="cli",
        profile=eid,
        model="m",
        tier=Tier(tier),
        capabilities=frozenset({Capability.CHAT, Capability.CODE}),
        cost=CostClass.FREE,
        family=family,
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_router(
    executors: dict,
    store: Store | None = None,
    git: Any = None,
    max_attempts: int = 4,
    lease_renew_s: float = 60.0,
) -> tuple[Router, Store, FakeClock]:
    store = store or Store(":memory:")
    clock = FakeClock()
    config = RouterConfig(
        db_path=Path(":memory:"),
        executors=[],
        handoff=HandoffConfig(),
        tier_policy=TierPolicy(allow_paid=False),
    )
    quota = QuotaBook(store, clock=clock)
    router = Router(
        config=config,
        store=store,
        executors=executors,
        quota=quota,
        git=git or (lambda _p: ""),
        max_attempts=max_attempts,
        clock=clock,
        lease_renew_s=lease_renew_s,
    )
    return router, store, clock


def setup_session(store: Store, sid: str = "s1") -> None:
    store.create_session(sid, title="t", workdir="/tmp/w")


def _ok() -> TurnResult:
    return TurnResult(stop=Stop.DONE, text="ok")


def _need(**kw: Any) -> Any:
    from onechat.selector import Need

    return Need(**kw)


def test_same_executor_resumes_native_without_brief() -> None:
    a = FakeExecutor(
        cli_spec("a"),
        [
            TurnResult(stop=Stop.DONE, text="one", native_session_id="nat-1"),
            TurnResult(stop=Stop.DONE, text="two"),
        ],
    )
    router, store, _ = make_router({"a": a})
    setup_session(store)
    asyncio.run(router.turn("s1", "q1", need_any()))
    asyncio.run(router.turn("s1", "q2", need_any()))
    second = a.requests[1]
    assert second.native_session_id == "nat-1"
    assert "<asf-handoff>" not in second.prompt


def need_any(**kw: Any):
    from onechat.selector import Need

    return Need(**kw)


def test_new_executor_gets_handoff() -> None:
    a = FakeExecutor(
        cli_spec("a", family="one"),
        [TurnResult(stop=Stop.DONE, text="answer A", native_session_id="nat-a")],
    )
    b = FakeExecutor(cli_spec("b", family="two"), [TurnResult(stop=Stop.DONE, text="answer B")])
    router, store, _ = make_router({"a": a, "b": b})
    setup_session(store)
    asyncio.run(router.turn("s1", "remember 4721", need_any(executor_id="a", allow_fallback=False)))
    asyncio.run(router.turn("s1", "what number?", need_any(executor_id="b", allow_fallback=False)))
    prompt = b.requests[0].prompt
    assert "<asf-handoff>" in prompt
    assert "4721" in prompt and "answer A" in prompt


def test_returning_executor_gets_delta() -> None:
    a = FakeExecutor(
        cli_spec("a", family="one"),
        [
            TurnResult(stop=Stop.DONE, text="A1", native_session_id="nat-a"),
            TurnResult(stop=Stop.DONE, text="A2"),
        ],
    )
    b = FakeExecutor(cli_spec("b", family="two"), [TurnResult(stop=Stop.DONE, text="B answer")])
    router, store, _ = make_router({"a": a, "b": b})
    setup_session(store)
    asyncio.run(router.turn("s1", "q1", need_any(executor_id="a", allow_fallback=False)))
    asyncio.run(router.turn("s1", "q2", need_any(executor_id="b", allow_fallback=False)))
    asyncio.run(router.turn("s1", "q3", need_any(executor_id="a", allow_fallback=False)))
    third = a.requests[1]  # a ran on turn 1 and turn 3
    assert "B answer" in third.prompt  # delta contains what B said
    assert third.native_session_id == "nat-a"  # still resumes native session


def test_quota_fails_over_and_records_cooldown() -> None:
    a = FakeExecutor(cli_spec("a", tier=1), [TurnResult(stop=Stop.QUOTA, retry_after_s=60)])
    b = FakeExecutor(cli_spec("b", tier=2), [TurnResult(stop=Stop.DONE, text="from b")])
    router, store, clock = make_router({"a": a, "b": b})
    setup_session(store)
    reply = asyncio.run(router.turn("s1", "q", need_any()))
    assert reply.text == "from b" and reply.switched_from == ["a"]
    cd = store.get_cooldown("a")
    assert cd is not None and cd["until_ts"] > clock.now


def test_interrupted_flag_on_code_failover() -> None:
    a = FakeExecutor(cli_spec("a", tier=1), [TurnResult(stop=Stop.ERROR, error="boom")])
    b = FakeExecutor(cli_spec("b", tier=2), [TurnResult(stop=Stop.DONE, text="fixed")])
    router, store, _ = make_router({"a": a, "b": b})
    setup_session(store)
    asyncio.run(router.turn("s1", "write code", need_any(capability=Capability.CODE)))
    prompt = b.requests[0].prompt
    assert "interrupted mid-task" in prompt


def test_all_fail_raises_with_retry_at() -> None:
    a = FakeExecutor(cli_spec("a"), [TurnResult(stop=Stop.QUOTA, retry_after_s=120)])
    router, _store, _clock = make_router({"a": a}, max_attempts=2)
    setup_session(router.store)
    with pytest.raises(NoExecutorAvailable) as exc:
        asyncio.run(router.turn("s1", "q", need_any()))
    assert exc.value.retry_at is not None


def test_user_turn_kept_on_failure() -> None:
    a = FakeExecutor(
        cli_spec("a"),
        [
            TurnResult(stop=Stop.QUOTA, retry_after_s=10),
            TurnResult(stop=Stop.QUOTA, retry_after_s=10),
        ],
    )
    router, store, _ = make_router({"a": a}, max_attempts=1)
    setup_session(store)
    with pytest.raises(NoExecutorAvailable):
        asyncio.run(router.turn("s1", "precious question", need_any()))
    turns = store.turns("s1")
    assert any(t["role"] == "user" and t["text"] == "precious question" for t in turns)


def test_session_busy_lease() -> None:
    a = FakeExecutor(cli_spec("a"), [TurnResult(stop=Stop.DONE, text="ok")])
    router, store, _ = make_router({"a": a})
    setup_session(store)
    store.set_lease("s1", owner="other-worker", until=99999.0)
    with pytest.raises(SessionBusy):
        asyncio.run(router.turn("s1", "q", need_any()))


def test_stale_lease_reclaimed() -> None:
    a = FakeExecutor(cli_spec("a"), [TurnResult(stop=Stop.DONE, text="ok")])
    router, store, _ = make_router({"a": a})
    setup_session(store)
    store.set_lease("s1", owner="dead-worker", until=1.0)  # long expired
    reply = asyncio.run(router.turn("s1", "q", need_any()))
    assert reply.text == "ok"


def test_status_event_on_switch() -> None:
    a = FakeExecutor(cli_spec("a", tier=1), [TurnResult(stop=Stop.QUOTA, retry_after_s=5)])
    b = FakeExecutor(cli_spec("b", tier=2), [TurnResult(stop=Stop.DONE, text="from b")])
    router, store, _ = make_router({"a": a, "b": b})
    setup_session(store)
    events: list[str] = []
    asyncio.run(
        router.turn("s1", "q", need_any(), on_event=lambda e: events.append(f"{e.kind}:{e.text}"))
    )
    assert any(k.startswith("status:") and "a" in k for k in events)


def test_reply_shape() -> None:
    reply = Reply(text="t", executor_id="a", stop=Stop.DONE, switched_from=[], turn_idx=1)
    assert reply.turn_idx == 1


def test_turn_applies_string_tier() -> None:
    """Plan 03: string tiers from the CLI ('low'/'max') must be honored, not dropped."""
    router, store, _clock = make_router({"e1": FakeExecutor(cli_spec("e1"), [_ok()])})
    setup_session(store, "s1")
    reply = asyncio.run(router.turn("s1", "hi", _need(tier="low")))
    assert reply.executor_id == "e1"  # MEDIUM executor satisfies an explicit LOW ask
    router2, store2, _ = make_router({"e1": FakeExecutor(cli_spec("e1"), [_ok()])})
    setup_session(store2, "s2")
    with pytest.raises(NoExecutorAvailable):
        asyncio.run(router2.turn("s2", "hi", _need(tier="max")))  # nothing at MAX


def test_turn_rejects_unknown_tier_string() -> None:
    router, store, _clock = make_router({"e1": FakeExecutor(cli_spec("e1"), [_ok()])})
    setup_session(store, "s1")
    with pytest.raises(ValueError, match="tier"):
        asyncio.run(router.turn("s1", "hi", _need(tier="ultra")))


def test_lease_renewed_during_turn() -> None:
    """Plan 03 §5.1: the DB lease is renewed while a long turn runs."""

    class SlowExecutor:
        spec = cli_spec("slow")

        def __init__(self) -> None:
            self.requests: list = []

        def available(self) -> bool:
            return True

        async def run(self, req, on_event=None):
            self.requests.append(req)
            await asyncio.sleep(0.15)
            return TurnResult(stop=Stop.DONE, text="ok")

    router, store, _clock = make_router({"slow": SlowExecutor()}, lease_renew_s=0.05)
    setup_session(store, "s1")
    owners: list[str] = []
    released: list[str] = []
    original_acquire = store.acquire_lease
    original_release = store.release_lease

    def spy_acquire(sid, owner, until_ts, now_ts):
        owners.append(owner)
        return original_acquire(sid, owner, until_ts, now_ts)

    def spy_release(sid, owner):
        released.append(owner)
        return original_release(sid, owner)

    store.acquire_lease = spy_acquire  # type: ignore[method-assign]
    store.release_lease = spy_release  # type: ignore[method-assign]
    asyncio.run(router.turn("s1", "hi", _need()))
    assert owners, "lease acquired (CAS)"
    first = owners[0]
    renewals = owners[1:]
    assert len(renewals) >= 2 and all(o == first for o in renewals)  # >= 2 renewals, same owner
    assert released == [first]  # owner-checked release at the end


# ---- T12a: retry_at, dedupe, native continuity, summarizer safety ----------------------


def test_all_cooldown_raises_with_non_none_retry_at() -> None:
    """T12a: everything cooling BEFORE any attempt still yields an honest retry_at."""
    a = FakeExecutor(cli_spec("a"), [_ok()])
    router, store, clock = make_router({"a": a})
    setup_session(store)
    store.set_cooldown("a", until_ts=clock.now + 120, strikes=1, reason="quota")
    with pytest.raises(NoExecutorAvailable) as exc:
        asyncio.run(router.turn("s1", "q", need_any()))
    assert exc.value.retry_at is not None
    assert exc.value.retry_at == pytest.approx(clock.now + 120, abs=2.0)


def test_duplicate_user_turn_not_appended_on_retry_after_failure() -> None:
    """T12a: retrying the SAME prompt reuses the stored user turn — no duplicate."""
    a = FakeExecutor(
        cli_spec("a"),
        [TurnResult(stop=Stop.QUOTA, retry_after_s=10), TurnResult(stop=Stop.DONE, text="ok")],
    )
    router, store, clock = make_router({"a": a}, max_attempts=1)
    setup_session(store)
    with pytest.raises(NoExecutorAvailable):
        asyncio.run(router.turn("s1", "same question", need_any()))
    clock.now += 20  # the quota cooldown expires before the retry
    reply = asyncio.run(router.turn("s1", "same question", need_any()))
    assert reply.text == "ok"
    user_turns = [t for t in store.turns("s1") if t["role"] == "user"]
    assert len(user_turns) == 1  # deduped (T12a)


def test_resume_advances_last_turn_idx_even_without_returned_native_id() -> None:
    """T12a: a resume DONE without native_session_id still advances last_turn_idx."""
    a = FakeExecutor(
        cli_spec("a"),
        [
            TurnResult(stop=Stop.DONE, text="one", native_session_id="nat-1"),
            TurnResult(stop=Stop.DONE, text="two"),  # resume; many CLIs do not echo the id
            TurnResult(stop=Stop.DONE, text="three"),
        ],
    )
    router, store, _ = make_router({"a": a})
    setup_session(store)
    asyncio.run(router.turn("s1", "q1", need_any()))
    asyncio.run(router.turn("s1", "q2", need_any()))
    asyncio.run(router.turn("s1", "q3", need_any()))
    third = a.requests[2]
    assert third.native_session_id == "nat-1"
    assert "Turns handled while you were away" not in third.prompt  # no spurious delta


def test_stale_native_session_invalidated_and_retried_fresh() -> None:
    """T12a: 'session not found' on resume clears the native row and retries fresh."""
    a = FakeExecutor(
        cli_spec("a"),
        [
            TurnResult(stop=Stop.DONE, text="one", native_session_id="nat-1"),
            TurnResult(stop=Stop.ERROR, error="session nat-1 not found upstream"),
            TurnResult(stop=Stop.DONE, text="fresh ok", native_session_id="nat-2"),
        ],
    )
    router, store, _ = make_router({"a": a})
    setup_session(store)
    asyncio.run(router.turn("s1", "q1", need_any()))
    reply = asyncio.run(router.turn("s1", "q2", need_any()))
    assert reply.text == "fresh ok"
    assert a.requests[1].native_session_id == "nat-1"  # first attempt resumed
    assert a.requests[2].native_session_id is None  # retry ran fresh
    row = store.get_native("s1", "a")
    assert row is not None and row["native_id"] == "nat-2"


def test_returning_code_executor_keeps_interrupted_warning() -> None:
    """T12a: the interrupted note survives the native-delta path for CODE failover."""
    a = FakeExecutor(cli_spec("a", family="one"), [TurnResult(stop=Stop.ERROR, error="boom")])
    b = FakeExecutor(
        cli_spec("b", family="two"),
        [TurnResult(stop=Stop.DONE, text="B1", native_session_id="nat-b"), _ok()],
    )
    router, store, _ = make_router({"a": a, "b": b})
    setup_session(store)
    asyncio.run(router.turn("s1", "q1", need_any(executor_id="b", allow_fallback=False)))
    asyncio.run(router.turn("s1", "q2 write code", need_any(capability=Capability.CODE)))
    prompt = b.requests[1].prompt
    assert "interrupted mid-task" in prompt


def test_summarizer_records_usage_uses_chat_only_and_preserves_upto_on_failure() -> None:
    """T12a: summarizer turns record usage on the session workdir; failures keep replies."""
    from onechat.contracts import Usage

    class Scripted:
        spec = cli_spec("chat-only")

        def __init__(self, fail_summarizer: bool) -> None:
            self.requests: list = []
            self._fail = fail_summarizer

        def available(self) -> bool:
            return True

        async def run(self, req, on_event=None):
            self.requests.append(req)
            if req.prompt.startswith("Summarize"):
                if self._fail:
                    raise RuntimeError("summarizer exploded")
                return TurnResult(stop=Stop.DONE, text="SUMMARY", usage=Usage(input_tokens=9))
            return TurnResult(stop=Stop.DONE, text="ok")

    chat = Scripted(fail_summarizer=False)
    router, store, _ = make_router({"chat-only": chat})
    setup_session(store)
    router.config.handoff.recent_turns = 1  # force compaction on the 2nd turn
    router.config.handoff.compact_threshold_chars = 50
    asyncio.run(router.turn("s1", "seed " + "y" * 200, need_any()))
    reply = asyncio.run(router.turn("s1", "again " + "y" * 200, need_any()))
    assert reply.text == "ok"
    summarize_req = chat.requests[1]
    assert summarize_req.prompt.startswith("Summarize")
    assert summarize_req.workdir == Path("/tmp/w")  # session workdir, not /tmp
    rows = store.conn.execute(
        "SELECT executor_id FROM usage WHERE executor_id='chat-only'"
    ).fetchall()
    assert len(rows) >= 3, "summarizer turn(s) must land in the usage ledger (turns + compactions)"
    summary = (store.get_session("s1") or {})["summary"]
    assert summary == "SUMMARY"

    # failure path: the summarizer raising must not lose the Reply nor advance summary_upto
    code = Scripted(fail_summarizer=True)
    router2, store2, _ = make_router({"chat-only": code})
    setup_session(store2, "s2")
    router2.config.handoff.recent_turns = 1
    router2.config.handoff.compact_threshold_chars = 50
    asyncio.run(router2.turn("s2", "seed " + "y" * 200, need_any()))
    reply2 = asyncio.run(router2.turn("s2", "again " + "y" * 200, need_any()))
    assert reply2.text == "ok"
    assert (store2.get_session("s2") or {})["summary_upto"] == 0


def test_explicit_tier_clamped_by_policy() -> None:
    """Explicit need.tier must be clamped by policy min_tier and max_tier."""
    from onechat.contracts import Tier
    from onechat.selector import Need

    router, store, _ = make_router({"chat-only": FakeExecutor(cli_spec("chat-only"), [])})
    setup_session(store, "s_clamp")
    router.config.tier_policy.min_tier = Tier.MEDIUM
    router.config.tier_policy.max_tier = Tier.MEDIUM

    need_low = Need(tier=Tier.LOW)
    assert router._resolve_tier(need_low, "hello") == Tier.MEDIUM

    need_high = Need(tier=Tier.HIGH)
    assert router._resolve_tier(need_high, "hello") == Tier.MEDIUM
    assert router._resolve_tier(need_low, "DeepSWE task") == Tier.HIGH


def test_router_emits_attempt_start_status_before_execution() -> None:
    """_turn_leased emits an attempt start event before calling _attempt."""
    events: list[Any] = []
    spec = cli_spec("chat-only")
    fake = FakeExecutor(spec, [TurnResult(stop=Stop.DONE, text="ok")])
    router, store, _ = make_router({"chat-only": fake})
    setup_session(store, "s_evt")

    asyncio.run(router.turn("s_evt", "hello", Need(), on_event=events.append))
    status_events = [e.text for e in events if e.kind == "status"]
    assert any("attempt" in s.lower() and "chat-only" in s for s in status_events)


def test_sole_usable_filters_by_need_capability() -> None:
    """T12a: sole_usable respects the need's capability — a CODE-only pool sole skip."""
    coder = FakeExecutor(
        ExecutorSpec(
            id="coder",
            kind="cli",
            profile="coder",
            model="m",
            tier=Tier.MEDIUM,
            capabilities=frozenset({Capability.CODE}),
            cost=CostClass.FREE,
            family="fam",
        ),
        [TurnResult(stop=Stop.QUOTA) for _ in range(6)],
    )
    chat = FakeExecutor(
        ExecutorSpec(
            id="chatter",
            kind="cli",
            profile="chatter",
            model="m",
            tier=Tier.MEDIUM,
            capabilities=frozenset({Capability.CHAT}),
            cost=CostClass.FREE,
            family="fam",
        ),
        [],
    )
    router, store, _ = make_router({"coder": coder, "chatter": chat})
    setup_session(store)
    for _ in range(6):
        with contextlib.suppress(NoExecutorAvailable):
            asyncio.run(router.turn("s1", "do code", need_any(capability=Capability.CODE)))
    assert store.get_cooldown("coder") is None  # sole CODE executor: statistical skip holds


def test_summarizer_non_done_stop_does_not_advance_upto() -> None:
    """Audit: summarizer returning Stop.ERROR must try next candidate & not advance summary_upto."""
    a = FakeExecutor(
        cli_spec("a"),
        [
            TurnResult(stop=Stop.DONE, text="long reply " + "x" * 100),
            TurnResult(stop=Stop.DONE, text="ok2"),
            TurnResult(stop=Stop.ERROR, text="partial garbage", error="boom"),
        ],
    )
    router, store, _ = make_router({"a": a})
    setup_session(store)
    asyncio.run(router.turn("s1", "seed " + "y" * 200, need_any()))
    router.config.handoff.recent_turns = 1
    router.config.handoff.compact_threshold_chars = 50
    reply = asyncio.run(router.turn("s1", "again " + "y" * 200, need_any()))
    assert reply.text == "ok2"
    assert (store.get_session("s1") or {})["summary_upto"] == 0


def test_append_user_handles_turn_idx_gap() -> None:
    """Audit: _append_user uses next_turn_idx so gaps in turns.idx never IntegrityError."""
    a = FakeExecutor(cli_spec("a"), [_ok()])
    router, store, _ = make_router({"a": a})
    setup_session(store)
    store.append_turn("s1", 0, "user", "old", executor_id="a", stop="done")
    store.append_turn("s1", 2, "assistant", "reply", executor_id="a", stop="done")
    reply = asyncio.run(router.turn("s1", "next prompt", need_any()))
    assert reply.text == "ok"


def test_need_timeout_s_propagates_to_turn_request() -> None:
    """Audit: Need.timeout_s must override the default 600s timeout on TurnRequest."""
    a = FakeExecutor(cli_spec("a"), [_ok()])
    router, store, _ = make_router({"a": a})
    setup_session(store)
    asyncio.run(router.turn("s1", "hi", need_any(timeout_s=45)))
    assert a.requests[0].timeout_s == 45


def test_session_lock_evicted_after_turn() -> None:
    """Bug 7: idle session locks must be evicted from _session_locks after turn completes."""
    a = FakeExecutor(cli_spec("a"), [_ok()])
    router, store, _ = make_router({"a": a})
    setup_session(store)
    asyncio.run(router.turn("s1", "hi", need_any()))
    assert "s1" not in router._session_locks


def test_sole_usable_ignores_disallowed_paid_and_lower_tier() -> None:
    """Bug 8: disallowed PAID or lower-tier executors must not break sole_usable protection."""
    free_high = FakeExecutor(
        ExecutorSpec(
            id="free-high",
            kind="cli",
            profile="free-high",
            model="m",
            tier=Tier.HIGH,
            capabilities=frozenset({Capability.CHAT}),
            cost=CostClass.FREE,
            family="f1",
        ),
        [TurnResult(stop=Stop.QUOTA) for _ in range(6)],
    )
    paid = FakeExecutor(
        ExecutorSpec(
            id="paid",
            kind="cli",
            profile="paid",
            model="m",
            tier=Tier.HIGH,
            capabilities=frozenset({Capability.CHAT}),
            cost=CostClass.PAID,
            family="f2",
        ),
        [],
    )
    free_low = FakeExecutor(
        ExecutorSpec(
            id="free-low",
            kind="cli",
            profile="free-low",
            model="m",
            tier=Tier.LOW,
            capabilities=frozenset({Capability.CHAT}),
            cost=CostClass.FREE,
            family="f3",
        ),
        [],
    )
    router, store, _ = make_router({"free-high": free_high, "paid": paid, "free-low": free_low})
    setup_session(store)
    for _ in range(6):
        with contextlib.suppress(NoExecutorAvailable):
            asyncio.run(router.turn("s1", "hi", need_any(tier=Tier.HIGH)))
    assert store.get_cooldown("free-high") is None


def test_turn1_bootstraps_workspace_context_with_agents_md(tmp_path: Path) -> None:
    workdir = tmp_path / "my_project"
    workdir.mkdir()
    (workdir / "AGENTS.md").write_text("# Project Guidelines\nStrict TDD\n", encoding="utf-8")
    store = Store(tmp_path / "asf.db")
    spec = cli_spec("exec-a", tier=3)
    exec_a = FakeExecutor(spec, [TurnResult(stop=Stop.DONE, text="ready")])
    router, _, _ = make_router({"exec-a": exec_a}, store=store)
    store.create_session("s1", title="s1", workdir=str(workdir))
    asyncio.run(router.turn("s1", "review the code", Need()))
    req = exec_a.requests[0]
    assert "<workspace-context>" in req.prompt
    assert "AGENTS.md" in req.prompt
    assert "review the code" in req.prompt


def test_turn2_does_not_repeat_workspace_bootstrap(tmp_path: Path) -> None:
    workdir = tmp_path / "my_project"
    workdir.mkdir()
    (workdir / "AGENTS.md").write_text("# Project Guidelines\nStrict TDD\n", encoding="utf-8")
    store = Store(tmp_path / "asf.db")
    spec = cli_spec("exec-a", tier=3)
    exec_a = FakeExecutor(
        spec,
        [
            TurnResult(stop=Stop.DONE, text="turn 1 reply", native_session_id="n1"),
            TurnResult(stop=Stop.DONE, text="turn 2 reply", native_session_id="n1"),
        ],
    )
    router, _, _ = make_router({"exec-a": exec_a}, store=store)
    store.create_session("s1", title="s1", workdir=str(workdir))
    asyncio.run(router.turn("s1", "first turn", Need()))
    asyncio.run(router.turn("s1", "second turn", Need()))
    req2 = exec_a.requests[1]
    assert "<workspace-context>" not in req2.prompt
    assert req2.prompt == "second turn"


def test_router_uses_promoted_executor_and_falls_back_when_circuited_out(tmp_path: Path) -> None:
    """Benchmark dynamic circuit promotion: promotes healthy target, falls back when down."""
    store = Store(tmp_path / "asf.db")
    spec_wb = cli_spec("workbuddy", tier=2)
    spec_gemini = cli_spec("opencode-gemini-hydrogen", tier=2)
    spec_gemini = spec_gemini.model_copy(update={"priority": 10})

    exec_wb = FakeExecutor(
        spec_wb, [TurnResult(stop=Stop.DONE, text="wb 1"), TurnResult(stop=Stop.DONE, text="wb 2")]
    )
    exec_gem = FakeExecutor(spec_gemini, [TurnResult(stop=Stop.DONE, text="gemini 1")])

    router, _, _ = make_router(
        {"workbuddy": exec_wb, "opencode-gemini-hydrogen": exec_gem}, store=store
    )
    router.config.roles = {"builder": "workbuddy", "code": "workbuddy"}
    router.config.benchmark.promote_roles = ["builder", "code"]
    store.create_session("s1", title="s1", workdir=str(tmp_path))

    # Turn 1: No overrides -> static config role (workbuddy)
    rep1 = asyncio.run(router.turn("s1", "build 1", Need(role="builder")))
    assert rep1.executor_id == "workbuddy"

    # Turn 2: Mark gemini healthy and promoted
    store.set_executor_override(
        executor_id="opencode-gemini-hydrogen",
        status="healthy",
        enabled_override=1,
        priority_override=0,
        latency_s=5.2,
    )
    rep2 = asyncio.run(router.turn("s1", "build 2", Need(role="builder")))
    assert rep2.executor_id == "opencode-gemini-hydrogen"

    # Turn 3: Mark gemini slow/down -> circuits out, falls back to workbuddy
    store.set_executor_override(
        executor_id="opencode-gemini-hydrogen",
        status="slow",
        enabled_override=0,
        priority_override=None,
        latency_s=25.0,
    )
    rep3 = asyncio.run(router.turn("s1", "build 3", Need(role="builder")))
    assert rep3.executor_id == "workbuddy"


def test_sole_usable_ignores_disabled_and_circuited_out_executors(tmp_path: Path) -> None:
    from onechat.config import ExecutorSpec
    from onechat.contracts import Capability, CostClass, Stop, Tier
    from onechat.router import TurnResult
    from onechat.selector import Need

    s1 = ExecutorSpec(
        id="s1",
        kind="cli",
        family="opencode",
        profile="opencode",
        model="m1",
        tier=Tier.MEDIUM,
        capabilities=frozenset([Capability.CODE]),
        cost=CostClass.FREE,
        priority=1,
    )
    s2 = ExecutorSpec(
        id="s2",
        kind="cli",
        family="opencode",
        profile="opencode",
        model="m2",
        tier=Tier.MEDIUM,
        capabilities=frozenset([Capability.CODE]),
        cost=CostClass.FREE,
        priority=2,
    )
    exec1 = FakeExecutor(s1, [TurnResult(stop=Stop.DONE, text="s1 done")])
    exec2 = FakeExecutor(s2, [TurnResult(stop=Stop.DONE, text="s2 done")])
    store = Store(tmp_path / "r.sqlite")
    router, _, _ = make_router({"s1": exec1, "s2": exec2}, store=store)

    need = Need(capability=Capability.CODE)
    assert not router._sole_usable(s1, need, Tier.MEDIUM)

    specs = [s1, s2.model_copy(update={"enabled": False})]
    assert router._sole_usable(s1, need, Tier.MEDIUM, specs=specs)


def test_executor_overrides_expire_after_ttl_and_respect_benchmark_enabled() -> None:
    from onechat.config import ExecutorSpec
    from onechat.contracts import Capability, CostClass, Tier
    from onechat.router_helpers import apply_executor_overrides

    s = ExecutorSpec(
        id="e1",
        kind="cli",
        family="opencode",
        profile="opencode",
        model="m",
        tier=Tier.MEDIUM,
        capabilities=frozenset([Capability.CODE]),
        cost=CostClass.FREE,
        enabled=False,
        priority=2,
    )
    overrides = {
        "e1": {
            "executor_id": "e1",
            "status": "healthy",
            "enabled_override": 1,
            "priority_override": 0,
            "checked_at": 1000.0,
        }
    }
    specs, roles = apply_executor_overrides(
        [s],
        {"builder": "other"},
        overrides,
        ["builder"],
        benchmark_enabled=True,
        now_ts=1500.0,
        ttl_s=3600.0,
    )
    assert specs[0].enabled is True
    assert roles["builder"] == "e1"

    specs_stale, roles_stale = apply_executor_overrides(
        [s],
        {"builder": "other"},
        overrides,
        ["builder"],
        benchmark_enabled=True,
        now_ts=5000.0,
        ttl_s=3600.0,
    )
    assert specs_stale[0].enabled is False
    assert roles_stale["builder"] == "other"

    specs_off, roles_off = apply_executor_overrides(
        [s],
        {"builder": "other"},
        overrides,
        ["builder"],
        benchmark_enabled=False,
        now_ts=1500.0,
        ttl_s=3600.0,
    )
    assert specs_off[0].enabled is False
    assert roles_off["builder"] == "other"


def test_bootstrap_prompt_bypassed_for_non_code_lightweight_queries(tmp_path: Path) -> None:
    workdir = tmp_path / "my_project"
    workdir.mkdir()
    (workdir / "AGENTS.md").write_text("# Guidelines\n", encoding="utf-8")
    store = Store(tmp_path / "asf.db")
    spec = cli_spec("exec-a", tier=3)
    exec_a = FakeExecutor(spec, [TurnResult(stop=Stop.DONE, text="PONG")])
    router, _, _ = make_router({"exec-a": exec_a}, store=store)
    store.create_session("s1", title="s1", workdir=str(workdir))

    # Turn with pure chat capability and no code intent in prompt
    need_chat = Need(capability=Capability.CHAT)
    asyncio.run(router.turn("s1", "Reply with the word PONG", need_chat))
    req = exec_a.requests[0]
    assert "<workspace-context>" not in req.prompt
    assert req.prompt == "Reply with the word PONG"


def test_bootstrap_prompt_uses_need_role_and_word_boundaries(tmp_path: Path) -> None:
    workdir = tmp_path / "my_project"
    workdir.mkdir()
    (workdir / "AGENTS.md").write_text("# Guidelines\n", encoding="utf-8")
    store = Store(tmp_path / "asf.db")
    spec = cli_spec("exec-a", tier=3)
    exec_a = FakeExecutor(
        spec,
        [TurnResult(stop=Stop.DONE, text="ok1"), TurnResult(stop=Stop.DONE, text="ok2")],
    )
    router, _, _ = make_router({"exec-a": exec_a}, store=store)

    # Non-code prompt containing "test" as substring in "latest" must NOT inject bootstrap
    store.create_session("s1", title="s1", workdir=str(workdir))
    asyncio.run(router.turn("s1", "find the latest version", Need(capability=Capability.CHAT)))
    assert "<workspace-context>" not in exec_a.requests[0].prompt

    # Technical role on Need MUST inject bootstrap even with Capability.CHAT
    store.create_session("s2", title="s2", workdir=str(workdir))
    asyncio.run(
        router.turn("s2", "check security", Need(capability=Capability.CHAT, role="reviewer"))
    )
    assert "<workspace-context>" in exec_a.requests[1].prompt


def test_promoted_executor_in_runtime_cooldown_keeps_static_role(tmp_path: Path) -> None:
    import time

    store = Store(tmp_path / "asf.db")
    spec_wb = cli_spec("workbuddy", tier=3)
    spec_flash = cli_spec("opencode-gemini-flash", tier=3)
    exec_wb = FakeExecutor(spec_wb, [TurnResult(stop=Stop.DONE, text="from-wb")])
    exec_flash = FakeExecutor(spec_flash, [TurnResult(stop=Stop.DONE, text="from-flash")])

    router, _, _ = make_router(
        {"workbuddy": exec_wb, "opencode-gemini-flash": exec_flash},
        store=store,
    )
    router.config.roles = {"builder": "workbuddy"}
    router.config.benchmark.promote_roles = ["builder"]

    store.set_executor_override(
        "opencode-gemini-flash", "healthy", enabled_override=1, priority_override=0, latency_s=1.0
    )
    # Put promoted executor into runtime QuotaBook cooldown
    store.set_cooldown("opencode-gemini-flash", time.time() + 600, strikes=1, reason="429")

    store.create_session("s1", title="s1", workdir=str(tmp_path))
    reply = asyncio.run(
        router.turn(
            "s1",
            "implement x",
            Need(capability=Capability.CODE, role="builder", allow_fallback=False),
        )
    )
    assert reply.executor_id == "workbuddy"
    assert reply.text == "from-wb"


def test_failover_loop_rechecks_cooldown(tmp_path: Path) -> None:
    """Mid-turn loop: if attempt 1 cools a sibling, attempt 2 skips it."""
    from onechat.config import CliProfile

    store = Store(tmp_path / "asf.db")
    s1 = cli_spec("autoclaw-review", tier=2)
    s1 = s1.model_copy(update={"profile": "autoclaw", "priority": 1})
    s2 = cli_spec("autoclaw-flash", tier=2)
    s2 = s2.model_copy(update={"profile": "autoclaw", "priority": 2})
    s3 = cli_spec("codex-fallback", tier=2)
    s3 = s3.model_copy(update={"profile": "codex", "priority": 3})

    exec_s1 = FakeExecutor(s1, [TurnResult(stop=Stop.ERROR, error="connection refused")])
    exec_s2 = FakeExecutor(s2, [TurnResult(stop=Stop.DONE, text="flash")])
    exec_s3 = FakeExecutor(s3, [TurnResult(stop=Stop.DONE, text="codex")])

    router, _, _ = make_router(
        {"autoclaw-review": exec_s1, "autoclaw-flash": exec_s2, "codex-fallback": exec_s3},
        store=store,
    )
    router.config.profiles["autoclaw"] = CliProfile(
        name="autoclaw",
        binary="autoclaw",
        new=["run"],
        resume=["resume"],
        parser="codex_jsonl",
        share_cooldown=True,
        unavailable_patterns=["connection refused"],
    )

    store.create_session("s1", title="s1", workdir=str(tmp_path))
    reply = asyncio.run(router.turn("s1", "do work", Need(capability=Capability.CODE)))
    assert reply.executor_id == "codex-fallback"
    assert reply.text == "codex"
