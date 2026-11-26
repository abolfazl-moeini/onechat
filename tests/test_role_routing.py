"""Tests for role-based executor dispatch (Plan 03 / Tri-Agent Topology)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from onechat.config import CliProfile, RouterConfig, TierPolicy
from onechat.contracts import Capability, CostClass, ExecutorSpec, Tier
from onechat.sdk import AsfClient
from onechat.selector import Need, order_candidates


def _spec(eid: str, profile: str, tier: Tier = Tier.MEDIUM) -> ExecutorSpec:
    return ExecutorSpec(
        id=eid,
        kind="cli",
        profile=profile,
        model=f"model-{eid}",
        tier=tier,
        capabilities=frozenset({Capability.CHAT, Capability.CODE}),
        cost=CostClass.FREE,
        family=profile,
    )


def test_order_candidates_prefers_role_executor() -> None:
    specs = [
        _spec("exec-codex", "codex", Tier.HIGH),
        _spec("exec-agy", "agy", Tier.HIGH),
        _spec("exec-buddy", "workbuddy", Tier.MEDIUM),
    ]
    roles = {"builder": "codex", "reviewer": "agy", "auditor": "workbuddy"}
    policy = TierPolicy()

    # When role is "reviewer", agy is preferred
    need_reviewer = Need(capability=Capability.CODE, role="reviewer")
    cands_rev = order_candidates(
        need_reviewer, Tier.LOW, specs, lambda _s: True, policy, roles=roles
    )
    assert cands_rev[0].id == "exec-agy"

    # When role is "builder", codex is preferred
    need_builder = Need(capability=Capability.CODE, role="builder")
    cands_bld = order_candidates(
        need_builder, Tier.LOW, specs, lambda _s: True, policy, roles=roles
    )
    assert cands_bld[0].id == "exec-codex"

    # When role is "auditor", workbuddy is preferred
    need_auditor = Need(capability=Capability.CODE, role="auditor")
    cands_aud = order_candidates(
        need_auditor, Tier.LOW, specs, lambda _s: True, policy, roles=roles
    )
    assert cands_aud[0].id == "exec-buddy"


def test_order_candidates_role_strict_when_no_fallback() -> None:
    specs = [
        _spec("exec-codex", "codex", Tier.HIGH),
        _spec("exec-agy", "agy", Tier.HIGH),
    ]
    roles = {"reviewer": "agy"}
    policy = TierPolicy()

    need = Need(capability=Capability.CODE, role="reviewer", allow_fallback=False)
    cands = order_candidates(need, Tier.LOW, specs, lambda _s: True, policy, roles=roles)
    assert len(cands) == 1
    assert cands[0].id == "exec-agy"


def test_order_candidates_prioritizes_matching_model() -> None:
    specs = [
        _spec("exec-a", "opencode", Tier.MEDIUM),
        _spec("exec-b", "opencode", Tier.MEDIUM),
    ]
    policy = TierPolicy()
    need = Need(capability=Capability.CODE, model="model-exec-b")
    cands = order_candidates(need, Tier.LOW, specs, lambda _s: True, policy)
    assert cands[0].id == "exec-b"


def test_session_ask_records_subagent_role_in_turns(tmp_path: Path) -> None:
    db = tmp_path / "asf.db"
    specs = [
        _spec("codex", "codex", Tier.MEDIUM),
        _spec("agy", "agy", Tier.MEDIUM),
    ]
    profile = CliProfile(
        name="codex",
        binary="codex",
        new=["new"],
        resume=["resume"],
        parser="codex_jsonl",
    )
    config = RouterConfig(
        db_path=db,
        executors=specs,
        profiles={"codex": profile, "agy": profile},
        roles={"builder": "codex", "reviewer": "agy"},
    )
    client = AsfClient(home=tmp_path)
    client.config = config
    client.router.config = config

    # Fake executor for test
    class FakeExec:
        def __init__(self, spec: ExecutorSpec) -> None:
            self.spec = spec

        def available(self) -> bool:
            return True

        async def run(self, req: object, on_event: object = None) -> object:
            from onechat.contracts import Stop, TurnResult

            return TurnResult(stop=Stop.DONE, text=f"reply from {self.spec.id}")

    client.executors = {s.id: FakeExec(s) for s in specs}
    client.router.executors = client.executors

    session = client.session(workdir=tmp_path)
    reply = asyncio.run(session.ask("review this diff", role="reviewer"))
    assert reply.executor_id == "agy"

    turns = client.store.turns(session.id)
    # The assistant turn must have recorded subagent_role="reviewer"
    assistant_turns = [t for t in turns if t["role"] == "assistant"]
    assert len(assistant_turns) == 1
    assert assistant_turns[0]["subagent_role"] == "reviewer"
