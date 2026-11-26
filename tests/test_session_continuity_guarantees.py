"""Session continuity and multi-turn guarantee tests for onechat."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from router_fakes import FakeClock, FakeExecutor

from onechat.config import CliProfile, HandoffConfig, RouterConfig, TierPolicy
from onechat.contracts import Capability, CostClass, ExecutorSpec, Stop, Tier, TurnResult
from onechat.quota import QuotaBook
from onechat.router import Router
from onechat.selector import Need
from onechat.store import Store


def _spec(eid: str, family: str, priority: int = 100) -> ExecutorSpec:
    return ExecutorSpec(
        id=eid,
        kind="cli",
        profile=eid,
        model="m",
        tier=Tier.MEDIUM,
        capabilities=frozenset({Capability.CHAT, Capability.CODE}),
        cost=CostClass.FREE,
        family=family,
        priority=priority,
    )


def test_native_resume_5_turns(tmp_path: Path) -> None:
    store = Store(tmp_path / "router.db")
    store.create_session("s-multi", title="multi", workdir=str(tmp_path))
    clock = FakeClock()

    results = [
        TurnResult(stop=Stop.DONE, text=f"Answer {i}", native_session_id="native-thread-101")
        for i in range(1, 6)
    ]
    executor = FakeExecutor(_spec("exec-native", "testfam"), results)
    profiles = {
        "exec-native": CliProfile(
            name="exec-native",
            binary="true",
            new=["{prompt}"],
            resume=["{prompt}"],
            parser="codex_jsonl",
        )
    }
    config = RouterConfig(
        db_path=tmp_path / "router.db",
        executors=[executor.spec],
        profiles=profiles,
        tier_policy=TierPolicy(allow_paid=False),
        handoff=HandoffConfig(),
    )
    router = Router(
        config=config,
        store=store,
        executors={"exec-native": executor},
        quota=QuotaBook(store),
        clock=clock,
    )

    for i in range(1, 6):
        reply = asyncio.run(router.turn("s-multi", f"Question {i}", Need()))
        assert reply.stop == Stop.DONE
        assert reply.session_id == "s-multi"
        assert reply.native_session_id == "native-thread-101"
        assert reply.continuity_mode == ("fresh" if i == 1 else "native_resume")

    assert len(executor.requests) == 5
    # Turn 1 has no native_session_id, Turns 2-5 pass native-thread-101
    assert executor.requests[0].native_session_id is None
    for req in executor.requests[1:]:
        assert req.native_session_id == "native-thread-101"


def test_cross_process_store_persistence(tmp_path: Path) -> None:
    db_file = tmp_path / "shared.db"
    store1 = Store(db_file)
    store1.create_session("s-proc", title="proc", workdir=str(tmp_path))
    store1.set_native("s-proc", "exec-1", "nat-999", 1)
    store1.close()

    # Re-open in second store instance simulating new process invocation
    store2 = Store(db_file)
    row = store2.get_native("s-proc", "exec-1")
    assert row is not None
    assert row["native_id"] == "nat-999"
    store2.close()


def test_auto_resume_vs_new_session(tmp_path: Path) -> None:
    db_file = tmp_path / "router.db"
    store = Store(db_file)
    wd = str(tmp_path.resolve())

    # Create session 1
    store.create_session("s-1", title="first", workdir=wd)

    # Create session 2 later (time.time() is ~1.7e9, so 2e9 is later)
    now_ts = time.time() + 1000.0
    store._exec(
        "INSERT INTO sessions(id, title, workdir, created_at) VALUES(?,?,?,?)",
        ("s-2", "second", wd, now_ts),
    )

    latest = store.latest_session_for_workdir(wd)
    assert latest is not None
    assert latest["id"] == "s-2"
    store.close()


def test_failover_handoff_brief_continuity(tmp_path: Path) -> None:
    store = Store(tmp_path / "router.db")
    store.create_session("s-failover", title="failover", workdir=str(tmp_path))
    clock = FakeClock()

    a = FakeExecutor(
        _spec("exec-a", "fam-a", priority=10),
        [
            TurnResult(stop=Stop.DONE, text="secret code is 9876", native_session_id="a-1"),
            TurnResult(stop=Stop.QUOTA, retry_after_s=300),
        ],
    )
    b = FakeExecutor(
        _spec("exec-b", "fam-b", priority=20),
        [TurnResult(stop=Stop.DONE, text="Received handoff")],
    )
    profiles = {
        eid: CliProfile(
            name=eid, binary="true", new=["{prompt}"], resume=["{prompt}"], parser="codex_jsonl"
        )
        for eid in ("exec-a", "exec-b")
    }
    config = RouterConfig(
        db_path=tmp_path / "router.db",
        executors=[a.spec, b.spec],
        profiles=profiles,
        tier_policy=TierPolicy(allow_paid=False),
        handoff=HandoffConfig(),
    )
    router = Router(
        config=config,
        store=store,
        executors={"exec-a": a, "exec-b": b},
        quota=QuotaBook(store),
        clock=clock,
    )

    reply1 = asyncio.run(router.turn("s-failover", "Turn 1", Need()))
    assert reply1.executor_id == "exec-a"
    assert reply1.continuity_mode == "fresh"

    reply2 = asyncio.run(router.turn("s-failover", "Turn 2", Need()))
    assert reply2.executor_id == "exec-b"
    assert reply2.continuity_mode == "handoff"
    assert "exec-a" in reply2.switched_from
    # Check that handoff brief sent to exec-b contains prior context
    assert len(b.requests) == 1
    assert "secret code is 9876" in b.requests[0].prompt or "<asf-handoff>" in b.requests[0].prompt


def test_stale_native_session_recovered_fresh(tmp_path: Path) -> None:
    store = Store(tmp_path / "router.db")
    store.create_session("s-stale", title="stale", workdir=str(tmp_path))
    store.set_native("s-stale", "exec-stale", "dead-thread-1", 1)

    stale_error = "Error: session not found on backend"
    executor = FakeExecutor(
        _spec("exec-stale", "fam-stale"),
        [
            TurnResult(stop=Stop.ERROR, error=stale_error),  # first attempt fails
            TurnResult(stop=Stop.DONE, text="recovered fresh", native_session_id="new-thread-2"),
        ],
    )
    profiles = {
        "exec-stale": CliProfile(
            name="exec-stale",
            binary="true",
            new=["{prompt}"],
            resume=["{prompt}"],
            parser="codex_jsonl",
        )
    }
    config = RouterConfig(
        db_path=tmp_path / "router.db",
        executors=[executor.spec],
        profiles=profiles,
        tier_policy=TierPolicy(allow_paid=False),
        handoff=HandoffConfig(),
    )
    router = Router(
        config=config,
        store=store,
        executors={"exec-stale": executor},
        quota=QuotaBook(store),
    )

    reply = asyncio.run(router.turn("s-stale", "hello", Need()))
    assert reply.stop == Stop.DONE
    assert reply.text == "recovered fresh"
    assert reply.native_session_id == "new-thread-2"
