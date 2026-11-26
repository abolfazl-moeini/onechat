"""Plan 03 continuity E2E (fakes only): the product's core promise, end to end.

FakeExecutor A answers "noted 4721" then returns QUOTA; FakeExecutor B (different family)
must receive a prompt containing 4721 and <asf-handoff>; then A recovers (cooldown expired
via FakeClock) and receives a delta_brief with B's answer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from onechat.config import CliProfile, HandoffConfig, RouterConfig, TierPolicy
from onechat.contracts import Capability, CostClass, ExecutorSpec, Stop, Tier, TurnResult
from onechat.quota import QuotaBook
from onechat.router import Router
from onechat.selector import Need
from onechat.store import Store
from router_fakes import FakeClock, FakeExecutor


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


def test_switch_preserves_context_and_delta() -> None:
    store = Store(":memory:")
    store.create_session("s", title="demo", workdir="/tmp/w")
    clock = FakeClock()
    a = FakeExecutor(
        _spec("codex-main", "openai", priority=10),
        [
            TurnResult(stop=Stop.DONE, text="noted 4721", native_session_id="thr-1"),
            TurnResult(stop=Stop.QUOTA, retry_after_s=60),
            TurnResult(stop=Stop.DONE, text="you're welcome", native_session_id="thr-1"),
        ],
    )
    b = FakeExecutor(
        _spec("agy-main", "google", priority=20),
        [TurnResult(stop=Stop.DONE, text="the number is 4721")],
    )
    profiles = {
        eid: CliProfile(
            name=eid, binary="true", new=["{prompt}"], resume=["{prompt}"], parser="codex_jsonl"
        )
        for eid in ("codex-main", "agy-main")
    }
    config = RouterConfig(
        db_path=Path(":memory:"),
        executors=[a.spec, b.spec],
        profiles=profiles,
        handoff=HandoffConfig(),
        tier_policy=TierPolicy(allow_paid=False),
    )
    router = Router(
        config=config,
        store=store,
        executors={"codex-main": a, "agy-main": b},
        quota=QuotaBook(store, clock=clock),
        git=lambda _p: "",
        clock=clock,
    )
    need = Need(capability=Capability.CHAT)

    r1 = asyncio.run(router.turn("s", "remember the number 4721", need))
    assert r1.executor_id == "codex-main" and r1.text == "noted 4721"

    r2 = asyncio.run(router.turn("s", "what number did I say?", need))
    assert r2.executor_id == "agy-main"
    assert r2.switched_from == ["codex-main"]
    b_prompt = b.requests[0].prompt
    assert "4721" in b_prompt and "<asf-handoff>" in b_prompt

    clock.advance(61)  # codex cooldown expires
    r3 = asyncio.run(
        router.turn("s", "thanks", Need(executor_id="codex-main", allow_fallback=False))
    )
    assert r3.executor_id == "codex-main"
    a_prompt_2 = a.requests[2].prompt
    assert "the number is 4721" in a_prompt_2  # delta brief carries B's answer
    assert "Turns handled while you were away" in a_prompt_2
    assert a.requests[2].native_session_id == "thr-1"  # still native resume
