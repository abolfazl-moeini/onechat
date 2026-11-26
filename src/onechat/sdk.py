"""Public SDK (Plan 03 §6 / T12b): AsfClient + Session. The factory talks to AI only here."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import RouterConfig, load_config
from .contracts import Capability, Executor, OnEvent, Tier
from .executors import build_executor
from .quota import QuotaBook
from .router import Reply, Router
from .selector import Need
from .store import Store

_UNSET = object()  # pin sentinel: leave that field untouched (T12b)


class Session:
    """One conversation. All asks go through the Router (failover + context included)."""

    def __init__(self, client: OneChatClient, session_id: str) -> None:
        self._client = client
        self.id = session_id

    async def ask(
        self,
        prompt: str,
        *,
        tier: Tier | str = "auto",
        executor: str | None = None,
        capability: str = "chat",
        task_kind: str | None = None,
        role: str | None = None,
        model: str | None = None,
        skills: tuple[str, ...] | list[str] | None = None,
        failures: int = 0,
        avoid_family: str | None = None,
        allow_fallback: bool = True,
        exclude: frozenset[str] | None = None,
        on_event: OnEvent | None = None,
        run_id: str | None = None,
        timeout_s: int | None = None,
    ) -> Reply:
        pinned_executor, pinned_tier = self._pins()
        return await self._client.router.turn(
            self.id,
            prompt,
            Need(
                capability=Capability(capability),
                tier=pinned_tier if pinned_tier is not None and tier == "auto" else tier,
                executor_id=executor or pinned_executor,
                role=role,
                model=model,
                skills=skills or (),
                task_kind=task_kind,
                failures=failures,
                avoid_family=avoid_family,
                allow_fallback=allow_fallback,
                exclude=exclude if exclude is not None else frozenset(),
                timeout_s=timeout_s,
            ),
            on_event=on_event,
            run_id=run_id,
        )

    def _pins(self) -> tuple[str | None, Tier | None]:
        row = self._client.store.get_session(self.id) or {}
        tier_value = row.get("tier")
        return row.get("pinned_executor"), Tier(tier_value) if tier_value else None

    def ask_sync(self, prompt: str, **kw: Any) -> Reply:
        return asyncio.run(self.ask(prompt, **kw))

    def pin(self, executor: Any = _UNSET, tier: Any = _UNSET) -> None:
        """Pin executor/tier (T12b: _UNSET fields keep their current pinned value)."""
        row = self._client.store.get_session(self.id) or {}
        executor_id = row.get("pinned_executor") if executor is _UNSET else executor
        tier_value = row.get("tier") if tier is _UNSET else (int(tier) if tier else None)
        self._client.store.pin(self.id, executor_id=executor_id, tier=tier_value)

    def history(self) -> list[dict[str, Any]]:
        return self._client.store.turns(self.id)


class OneChatClient:
    """Wires config, store, executors, quota and the router. Entry point for everything."""

    def __init__(self, config_path: Path | None = None, home: Path | None = None) -> None:
        self.config: RouterConfig = load_config(config_path, home=home)
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.store = Store(
            self.config.db_path,
            api_key_envs=sorted(
                {spec.api_key_env for spec in self.config.executors if spec.api_key_env}
            ),
        )
        target_ids = (
            {t.executor_id for t in getattr(self.config.benchmark, "targets", [])}
            if getattr(self.config, "benchmark", None)
            else set()
        )
        self.executors: dict[str, Executor] = {
            spec.id: build_executor(spec, self.config.profiles)
            for spec in self.config.executors
            if spec.enabled or spec.id in target_ids
        }
        self.quota = QuotaBook(self.store, policy=self.config.cooldown)
        self.router = Router(
            config=self.config,
            store=self.store,
            executors=self.executors,
            quota=self.quota,
            git=_default_git(),
        )

    def session(
        self,
        workdir: Path | str = ".",
        session_id: str | None = None,
        title: str | None = None,
        continue_latest: bool = False,
    ) -> Session:
        resolved_wd = str(Path(workdir).expanduser().resolve())
        sid = session_id
        if not sid and continue_latest:
            latest = self.store.latest_session_for_workdir(resolved_wd)
            if latest and latest.get("id"):
                sid = str(latest["id"])
        sid = sid or f"s-{time.time_ns():x}-{secrets.token_hex(4)}"
        if self.store.get_session(sid) is None:
            self.store.create_session(sid, title=title or sid, workdir=resolved_wd)
        return Session(self, sid)

    async def ask_once(self, prompt: str, **need: Any) -> Reply:
        """Ephemeral one-shot ask: the session (and its rows) never outlive the call."""
        sid = f"s-{time.time_ns():x}-{secrets.token_hex(4)}"
        self.store.create_session(sid, title=sid, workdir=".")
        try:
            return await Session(self, sid).ask(prompt, **need)
        finally:
            self._drop_session(sid)  # ephemeral: no rows kept (T12b)

    def _drop_session(self, sid: str) -> None:
        """Ephemeral-session teardown: turns first (FK), then native rows, then the row."""
        self.store.delete_session(sid)

    def close(self) -> None:
        """Release the store (T12b). Any later store use fails closed."""
        self.store.close()

    async def aclose(self) -> None:
        for executor in self.executors.values():
            aclose = getattr(executor, "aclose", None)
            if callable(aclose):
                await aclose()
        self.close()

    @classmethod
    def _for_tests(
        cls,
        tmp_path: Path,
        executor_factory: Callable[[RouterConfig], dict[str, Executor]],
    ) -> OneChatClient:
        """Test constructor: tmp store + injected executors, no CLIs needed."""
        client = cls.__new__(cls)
        client.config = _test_config(tmp_path)
        client.store = Store(tmp_path / "asf.db")
        client.executors = executor_factory(client.config)
        client.quota = QuotaBook(client.store, policy=client.config.cooldown)
        client.router = Router(
            config=client.config,
            store=client.store,
            executors=client.executors,
            quota=client.quota,
            git=lambda _p: "",
        )
        return client


# Backwards compatibility alias
AsfClient = OneChatClient


def _test_config(tmp_path: Path) -> RouterConfig:
    """Two free CODE-capable cli executors for fake-driven SDK tests."""
    from .config import CliProfile
    from .contracts import Capability, CostClass, ExecutorSpec, Tier

    def spec(eid: str) -> ExecutorSpec:
        return ExecutorSpec(
            id=eid,
            kind="cli",
            profile=eid,
            model="m",
            tier=Tier.MEDIUM,
            capabilities=frozenset({Capability.CHAT, Capability.CODE}),
            cost=CostClass.FREE,
            family=f"family-{eid}",
        )

    profiles = {
        eid: CliProfile(
            name=eid, binary="true", new=["{prompt}"], resume=["{prompt}"], parser="codex_jsonl"
        )
        for eid in ("exec-a", "exec-b")
    }
    return RouterConfig(
        db_path=tmp_path / "asf.db", executors=[spec("exec-a"), spec("exec-b")], profiles=profiles
    )


def _default_git() -> Callable[[Path], Any]:
    from .gitstate import async_git_state

    return async_git_state
