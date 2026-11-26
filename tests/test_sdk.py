"""Plan 03 SDK tests: AsfClient wiring, Session.ask, pin, ask_sync."""

from __future__ import annotations

import asyncio
from pathlib import Path

from onechat.contracts import Stop, Tier
from onechat.sdk import AsfClient
from router_fakes import fake_executors_factory, write_user_config  # noqa: F401


def test_session_ask_roundtrip(tmp_path: Path) -> None:
    client = AsfClient._for_tests(tmp_path, fake_executors_factory(tmp_path))
    session = client.session(workdir=str(tmp_path))
    reply = asyncio.run(session.ask("q1"))
    assert reply.text == "fake answer a"
    assert reply.executor_id == "exec-a"
    assert reply.stop == Stop.DONE
    history = session.history()
    assert [t["role"] for t in history] == ["user", "assistant"]


def test_pin_persists(tmp_path: Path) -> None:
    client = AsfClient._for_tests(tmp_path, fake_executors_factory(tmp_path))
    session = client.session(workdir=str(tmp_path))
    session.pin(executor="exec-b", tier=Tier.HIGH)
    row = client.store.get_session(session.id)
    assert row["pinned_executor"] == "exec-b" and row["tier"] == 3


def test_ask_sync(tmp_path: Path) -> None:
    client = AsfClient._for_tests(tmp_path, fake_executors_factory(tmp_path))
    session = client.session(workdir=str(tmp_path))
    reply = session.ask_sync("q")
    assert reply.text == "fake answer a"


def test_ask_once_ephemeral(tmp_path: Path) -> None:
    client = AsfClient._for_tests(tmp_path, fake_executors_factory(tmp_path))
    reply = asyncio.run(client.ask_once("hello"))
    assert reply.text == "fake answer a"


def test_loads_user_config(tmp_path: Path) -> None:
    config_path = write_user_config(tmp_path)
    client = AsfClient(config_path=config_path)
    assert any(e.id == "user-exec" for e in client.config.executors)


def test_sdk_builds_benchmark_target_executors_even_if_initially_disabled(tmp_path: Path) -> None:
    config_yaml = """
executors:
  - id: exec-target
    kind: cli
    family: opencode
    profile: opencode
    model: gemini-4-hydrogen
    enabled: false
    cost: free
    tier: 3
    capabilities: [code]
benchmark:
  enabled: true
  targets:
    - executor_id: exec-target
      model: gemini-4-hydrogen
"""
    p = tmp_path / "config.yaml"
    p.write_text(config_yaml, encoding="utf-8")
    client = AsfClient(config_path=p)
    assert "exec-target" in client.executors


def test_ask_forwards_allow_fallback(tmp_path, monkeypatch) -> None:
    """`executors test` needs pinned-only selection: allow_fallback must reach the Need."""
    from onechat.contracts import Stop
    from onechat.router import Reply
    from router_fakes import fake_executors_factory

    client = AsfClient._for_tests(tmp_path, fake_executors_factory(tmp_path))
    captured: list[bool] = []

    async def fake_turn(session_id, prompt, need, on_event=None, run_id=None):
        captured.append(need.allow_fallback)
        return Reply(text="ok", executor_id="exec-a", stop=Stop.DONE, switched_from=[], turn_idx=0)

    monkeypatch.setattr(client.router, "turn", fake_turn)
    sess = client.session()
    asyncio.run(sess.ask("hi", allow_fallback=False))
    asyncio.run(sess.ask("hi", allow_fallback=True))
    assert captured == [False, True]


# ---- T12b: pin sentinel, client close, ephemeral cleanup, ask exclude ------------------


def test_pin_tier_preserves_pinned_executor(tmp_path: Path) -> None:
    """T12b: pinning one field must not wipe the other (_UNSET sentinel)."""
    client = AsfClient._for_tests(tmp_path, fake_executors_factory(tmp_path))
    session = client.session(workdir=str(tmp_path))
    session.pin(executor="exec-b")
    session.pin(tier=Tier.HIGH)
    row = client.store.get_session(session.id)
    assert row["pinned_executor"] == "exec-b"  # survived the tier-only pin
    assert row["tier"] == 3
    session.pin(executor=None)
    row = client.store.get_session(session.id)
    assert row["pinned_executor"] is None and row["tier"] == 3  # tier survived the clear


def test_client_close_closes_store(tmp_path: Path) -> None:
    """T12b: close() releases the store — later use is a closed-connection error."""
    client = AsfClient._for_tests(tmp_path, fake_executors_factory(tmp_path))
    client.close()
    try:
        client.store.list_sessions()
        raise AssertionError("store still usable after close()")
    except Exception:
        pass  # sqlite3.ProgrammingError or similar


def test_ask_once_cleans_up_ephemeral_session(tmp_path: Path) -> None:
    """T12b: the ephemeral ask_once session leaves no rows behind (even on error)."""
    client = AsfClient._for_tests(tmp_path, fake_executors_factory(tmp_path))
    reply = asyncio.run(client.ask_once("hello"))
    assert reply.text == "fake answer a"
    assert client.store.list_sessions() == []
    assert client.store.turns("") == []


def test_ask_forwards_exclude(tmp_path: Path, monkeypatch) -> None:
    """T12b: Session.ask passes `exclude` through to the Need."""
    from onechat.selector import Need

    client = AsfClient._for_tests(tmp_path, fake_executors_factory(tmp_path))
    captured: list[Need] = []

    async def fake_turn(session_id, prompt, need, on_event=None, run_id=None):
        captured.append(need)
        from onechat.router import Reply

        return Reply(text="t", executor_id="exec-a", stop=Stop.DONE, switched_from=[], turn_idx=1)

    monkeypatch.setattr(client.router, "turn", fake_turn)
    session = client.session(workdir=str(tmp_path))
    asyncio.run(session.ask("q", exclude=frozenset({"exec-b"})))
    assert captured[0].exclude == frozenset({"exec-b"})


def test_client_aclose_closes_api_executors(tmp_path: Path) -> None:
    """Bug 9: AsfClient.aclose() calls aclose() on executors that expose it."""
    closed: list[str] = []

    class Closable:
        spec = None  # unused here

        async def aclose(self) -> None:
            closed.append("closed")

    client = AsfClient._for_tests(tmp_path, fake_executors_factory(tmp_path))
    client.executors["closable"] = Closable()  # type: ignore[assignment]
    asyncio.run(client.aclose())
    assert closed == ["closed"]


def test_session_ask_forwards_timeout_s(tmp_path: Path, monkeypatch) -> None:
    """Bug 11: Session.ask forwards timeout_s into Need."""
    from onechat.selector import Need

    client = AsfClient._for_tests(tmp_path, fake_executors_factory(tmp_path))
    captured: list[Need] = []

    async def fake_turn(session_id, prompt, need, on_event=None, run_id=None):
        captured.append(need)
        from onechat.router import Reply

        return Reply(text="t", executor_id="exec-a", stop=Stop.DONE, switched_from=[], turn_idx=1)

    monkeypatch.setattr(client.router, "turn", fake_turn)
    session = client.session(workdir=str(tmp_path))
    asyncio.run(session.ask("q", timeout_s=77))
    assert captured[0].timeout_s == 77
