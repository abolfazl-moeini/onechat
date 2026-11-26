"""Router: one turn across many backends with invisible failover."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import RouterConfig
from .contracts import (
    Capability,
    CostClass,
    Event,
    Executor,
    NoExecutorAvailable,
    OnEvent,
    Reply,
    SessionBusy,
    Stop,
    Tier,
    TurnRequest,
    TurnResult,
)
from .handoff import maybe_compact
from .quota import QuotaBook
from .router_helpers import (
    advance_native,
    apply_executor_overrides,
    build_request,
    git_state_of,
    make_summarizer,
    select_candidates,
    stale_session,
)
from .selector import Need, clamp_tier, coerce_tier, resolve_tier
from .store import Store

LEASE_TTL_S = 300.0


class Router:
    """Owns tier selection, failover, session context and turn recording."""

    def __init__(
        self,
        config: RouterConfig,
        store: Store,
        executors: dict[str, Executor],
        quota: QuotaBook,
        git: Callable[[Path], Any] = lambda _p: "",
        max_attempts: int = 4,
        clock: Callable[[], float] = time.time,
        lease_renew_s: float = 60.0,
    ) -> None:
        self.config, self.store, self.executors = config, store, executors
        self.quota, self._git, self.max_attempts = quota, git, max_attempts
        self._clock, self.lease_renew_s = clock, lease_renew_s
        self._session_locks: dict[str, asyncio.Lock] = {}

    async def turn(
        self,
        session_id: str,
        prompt: str,
        need: Need,
        on_event: OnEvent | None = None,
        run_id: str | None = None,
    ) -> Reply:
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        try:
            async with lock:
                if self.store.get_session(session_id) is None:
                    raise ValueError(f"unknown session {session_id!r}")
                return await self._turn_locked(session_id, prompt, need, on_event, run_id)
        finally:
            if not lock.locked() and self._session_locks.get(session_id) is lock:
                self._session_locks.pop(session_id, None)

    async def _turn_locked(
        self, session_id: str, prompt: str, need: Need, on_event: OnEvent | None, run_id: str | None
    ) -> Reply:
        session = self.store.get_session(session_id) or {}
        owner = f"{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._acquire_lease(session_id, session, owner)
        renewal = asyncio.create_task(self._renew_lease(session_id, owner))
        try:
            return await self._turn_leased(session_id, prompt, need, on_event, run_id)
        finally:
            renewal.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await renewal
            self.store.release_lease(session_id, owner)

    def _acquire_lease(self, session_id: str, session: dict[str, Any], owner: str) -> None:
        now, until = self._clock(), session.get("lock_until")
        if session.get("lock_owner") and until and float(until) > now:
            raise SessionBusy(f"session {session_id} busy until {until}")
        if not self.store.acquire_lease(session_id, owner, now + LEASE_TTL_S, now):
            raise SessionBusy(f"session {session_id} busy")

    async def _renew_lease(self, session_id: str, owner: str) -> None:
        """Keep the cross-process lease alive while a long turn runs (T12a: CAS + safe)."""
        while True:
            await asyncio.sleep(self.lease_renew_s)
            with contextlib.suppress(Exception):
                self.store.acquire_lease(
                    session_id, owner, self._clock() + LEASE_TTL_S, self._clock()
                )

    def _prepare_cands(
        self, session_id: str, prompt: str, need: Need, tier: Tier
    ) -> tuple[list[Any], list[Any], list[Any]]:
        bench = getattr(self.config, "benchmark", None)
        ttl = max(2 * getattr(bench, "interval_s", 1800.0), 3600.0) if bench else 3600.0
        p_roles = getattr(bench, "promote_roles", []) if bench else []
        en = getattr(bench, "enabled", True) if bench else False
        specs, roles = apply_executor_overrides(
            [e.spec for e in self.executors.values()],
            self.config.roles,
            self.store.get_executor_overrides(),
            p_roles,
            en,
            self._clock(),
            ttl,
            self._usable,
        )
        sess = self.store.get_session(session_id) or {}
        wd = Path(str(sess["workdir"])) if sess.get("workdir") else None
        cands, eligible, _ = select_candidates(
            need, prompt, tier, specs, roles, self.config, self._usable, workdir=wd
        )
        return cands, eligible, specs

    async def _turn_leased(
        self, session_id: str, prompt: str, need: Need, on_event: OnEvent | None, run_id: str | None
    ) -> Reply:
        tier = self._resolve_tier(need, prompt)
        cands, eligible, specs = self._prepare_cands(session_id, prompt, need, tier)
        if not cands:
            raise NoExecutorAvailable(retry_at=self.quota.earliest_available(eligible))
        assistant_idx = self._append_user(session_id, prompt)
        failed: list[str] = []
        attempts = cands[: self.max_attempts]
        for idx, spec in enumerate(attempts, 1):
            if not self.quota.available(spec):
                continue
            self._emit(
                on_event,
                Event(
                    kind="status",
                    text=f"attempt {idx}/{len(attempts)}: {spec.id} ({spec.model})",
                ),
            )
            result, resumed = await self._attempt(session_id, spec, need, prompt, failed, on_event)

            sole = self._sole_usable(spec, need, tier, specs=specs)
            self.quota.record(
                spec,
                result,
                session_id,
                run_id,
                sole_usable=sole,
                all_specs=specs,
                profiles=self.config.profiles,
            )
            if result.stop == Stop.DONE:
                return await self._finish(
                    session_id, spec, result, failed, assistant_idx, resumed, on_event, need
                )
            failed.append(spec.id)
            self._emit(on_event, Event(kind="status", text=f"{spec.id}: {result.stop} → next"))
        raise NoExecutorAvailable(retry_at=self.quota.earliest_available(eligible))

    def _resolve_tier(self, need: Need, prompt: str) -> Tier:
        policy = self.config.tier_policy
        exp = coerce_tier(need.tier)
        if exp is not None:
            return clamp_tier(exp, prompt, policy)
        return resolve_tier(need, prompt, policy)

    def _append_user(self, session_id: str, prompt: str) -> int:
        """Store the user turn once (T12a): a retry of the SAME prompt reuses the row."""
        turns = self.store.turns(session_id)
        if turns and turns[-1]["role"] == "user" and turns[-1]["text"] == prompt:
            return int(turns[-1]["idx"]) + 1
        idx = self.store.next_turn_idx(session_id)
        self.store.append_turn(session_id, idx, "user", prompt, executor_id="", stop="done")
        return idx + 1

    async def _attempt(
        self,
        session_id: str,
        spec: Any,
        need: Need,
        prompt: str,
        failed: list[str],
        on_event: OnEvent | None,
    ) -> tuple[TurnResult, bool]:
        executor = self.executors[spec.id]
        req = await self._build_request(session_id, spec, need, prompt, failed)
        result = await executor.run(req, on_event)
        if result.stop == Stop.ERROR and req.native_session_id and stale_session(result.error):
            self.store.set_native(session_id, spec.id, "", -1)  # invalidate (T12a)
            fresh = await self._build_request(session_id, spec, need, prompt, failed)
            return await executor.run(fresh, on_event), False
        return result, bool(req.native_session_id)

    async def _build_request(
        self, session_id: str, spec: Any, need: Need, prompt: str, failed: list[str]
    ) -> TurnRequest:
        session = self.store.get_session(session_id) or {}
        turns = self.store.turns(session_id, since_idx=0)[:-1]  # exclude current user turn
        interrupted = bool(failed) and need.capability == Capability.CODE
        git_text = (
            await git_state_of(self._git, Path(session.get("workdir", ".")))
            if need.capability == Capability.CODE or interrupted
            else ""
        )
        native = self.store.get_native(session_id, spec.id)
        return build_request(
            session, spec, need, prompt, turns, git_text, interrupted, self.config, native
        )

    def _sole_usable(
        self, executing: Any, need: Need, tier: Tier | None = None, specs: list[Any] | None = None
    ) -> bool:
        """Only ONE usable executor for THIS need's capability/filters (T12a)."""
        target = specs if specs is not None else [e.spec for e in self.executors.values()]
        usable = [s for s in target if self._usable(s, need, tier)]
        return len(usable) == 1 and usable[0].id == executing.id

    def _usable(self, spec: Any, need: Need | None = None, tier: Tier | None = None) -> bool:
        if not getattr(spec, "enabled", True):
            return False
        executor = self.executors.get(spec.id)
        if executor is None or not executor.available() or not self.quota.available(spec):
            return False
        if need is None:
            return True
        paid_blocked = spec.cost == CostClass.PAID and not self.config.tier_policy.allow_paid
        tier_blocked = tier is not None and need.executor_id is None and spec.tier < tier
        if paid_blocked or tier_blocked:
            return False
        return (
            need.capability in spec.capabilities
            and (need.avoid_family is None or spec.family != need.avoid_family)
            and spec.id not in need.exclude
        )

    async def _finish(
        self,
        session_id: str,
        spec: Any,
        result: TurnResult,
        switched_from: list[str],
        assistant_idx: int,
        resumed: bool,
        on_event: OnEvent | None,
        need: Need | None = None,
    ) -> Reply:
        self.store.append_turn(
            session_id,
            assistant_idx,
            "assistant",
            result.text,
            executor_id=spec.id,
            stop=result.stop.value,
            subagent_role=need.role if need else None,
        )
        warnings = advance_native(self.store, session_id, spec, result, assistant_idx, resumed)
        for warn in warnings:
            self._emit(on_event, Event(kind="status", text=warn))
        session = self.store.get_session(session_id) or {}
        turns = self.store.turns(session_id, since_idx=0)  # full history (T12a off-by-one)
        await self._safe_compact(session, turns)
        self._emit(on_event, Event(kind="status", text=f"{spec.id}: done"))
        native_row = self.store.get_native(session_id, spec.id)
        native_sid = result.native_session_id
        if not native_sid and native_row and native_row.get("native_id"):
            native_sid = str(native_row["native_id"])
        if resumed:
            continuity_mode = "native_resume"
        elif switched_from:
            continuity_mode = "handoff"
        else:
            continuity_mode = "fresh"
        return Reply(
            text=result.text,
            executor_id=spec.id,
            stop=Stop.DONE,
            switched_from=switched_from,
            turn_idx=assistant_idx,
            session_id=session_id,
            native_session_id=native_sid,
            continuity_mode=continuity_mode,
        )

    async def _safe_compact(self, session: dict[str, Any], turns: list[dict[str, Any]]) -> None:
        """Compaction is best-effort: a summarizer failure keeps the reply intact (T12a)."""
        with contextlib.suppress(Exception):
            wd = Path(str(session.get("workdir", ".")))
            summarizer = make_summarizer(self.config, self.executors, self.quota, self._usable, wd)
            compacted = await maybe_compact(session, turns, self.config.handoff, summarizer)
            if compacted is not None:
                self.store.set_summary(str(session["id"]), *compacted)

    def _emit(self, on_event: OnEvent | None, event: Event) -> None:
        with contextlib.suppress(Exception):
            if on_event is not None:
                on_event(event)
