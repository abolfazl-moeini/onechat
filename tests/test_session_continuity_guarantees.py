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


def test_anti_drift_status_event_on_missing_and_rotated_native_id(tmp_path: Path) -> None:
    store = Store(tmp_path / "router.db")
    store.create_session("s-drift", title="drift", workdir=str(tmp_path))
    clock = FakeClock()

    executor = FakeExecutor(
        _spec("exec-drift", "fam-drift"),
        [
            # Turn 1: fresh turn returns NO native_session_id -> should warn
            TurnResult(stop=Stop.DONE, text="turn 1", native_session_id=None),
            # Turn 2: after manual native_id set, returns rotated native_id -> should warn
            TurnResult(stop=Stop.DONE, text="turn 2", native_session_id="rotated-id-2"),
        ],
    )
    profiles = {
        "exec-drift": CliProfile(
            name="exec-drift",
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
        executors={"exec-drift": executor},
        quota=QuotaBook(store),
        clock=clock,
    )

    events_t1: list[str] = []
    asyncio.run(router.turn("s-drift", "Q1", Need(), on_event=lambda ev: events_t1.append(ev.text)))
    assert any("no native_session_id" in e for e in events_t1)

    store.set_native("s-drift", "exec-drift", "orig-id-1", 1)
    events_t2: list[str] = []
    asyncio.run(router.turn("s-drift", "Q2", Need(), on_event=lambda ev: events_t2.append(ev.text)))
    assert any("rotated native session" in e for e in events_t2)
    store.close()


def test_cli_run_auto_resumes_and_isolates_workdirs(tmp_path: Path, monkeypatch: object) -> None:
    import json

    from router_fakes import fake_executors_factory
    from typer.testing import CliRunner

    import onechat.cli as cli_module
    from onechat.sdk import AsfClient

    client = AsfClient._for_tests(tmp_path, fake_executors_factory(tmp_path))
    monkeypatch.setattr(cli_module, "AsfClient", lambda **kw: client)  # type: ignore[attr-defined]
    runner = CliRunner()

    dir_a = tmp_path / "proj_a"
    dir_b = tmp_path / "proj_b"
    dir_a.mkdir()
    dir_b.mkdir()

    r1 = runner.invoke(cli_module.app, ["run", "hi a1", "--workdir", str(dir_a), "--json"])
    assert r1.exit_code == 0, r1.output
    p1 = json.loads(r1.stdout)
    assert p1["continuity_mode"] == "fresh"

    # Second run in dir_a without --session MUST auto-continue the same session_id
    r2 = runner.invoke(cli_module.app, ["run", "hi a2", "--workdir", str(dir_a), "--json"])
    assert r2.exit_code == 0, r2.output
    p2 = json.loads(r2.stdout)
    assert p2["session_id"] == p1["session_id"]

    # Run in dir_b MUST create a separate session for dir_b
    r3 = runner.invoke(cli_module.app, ["run", "hi b1", "--workdir", str(dir_b), "--json"])
    assert r3.exit_code == 0, r3.output
    p3 = json.loads(r3.stdout)
    assert p3["session_id"] != p1["session_id"]

    # Run in dir_a with --new-session MUST start a fresh session
    r4 = runner.invoke(
        cli_module.app,
        ["run", "hi a fresh", "--workdir", str(dir_a), "--new-session", "--json"],
    )
    assert r4.exit_code == 0, r4.output
    p4 = json.loads(r4.stdout)
    assert p4["session_id"] != p1["session_id"]
    client.close()
