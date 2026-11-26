"""Plan 02 store tests: WAL, migrations, turns, native sessions, usage, cooldowns, lease."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from onechat.store import Store


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "asf.db")
    yield s
    s.close()


def _session(store: Store, sid: str = "s1") -> None:
    store.create_session(sid, title="t", workdir="/tmp/w")


def test_wal_enabled(tmp_path) -> None:
    s = Store(tmp_path / "asf.db")
    try:
        mode = s.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        s.close()


def test_turn_idx_unique_per_session(store) -> None:
    _session(store)
    store.append_turn("s1", idx=0, role="user", text="hi", executor_id="e", stop="done")
    store.append_turn("s1", idx=1, role="assistant", text="ho", executor_id="e", stop="done")
    with pytest.raises(sqlite3.IntegrityError):
        store.append_turn("s1", idx=0, role="user", text="again", executor_id="e", stop="done")


def test_turns_since_idx(store) -> None:
    _session(store)
    for i in range(4):
        store.append_turn("s1", idx=i, role="user", text=f"m{i}", executor_id="e", stop="done")
    assert [t["text"] for t in store.turns("s1", since_idx=2)] == ["m2", "m3"]
    assert len(store.turns("s1")) == 4


def test_native_session_roundtrip(store) -> None:
    _session(store)
    assert store.get_native("s1", "codex-main") is None
    store.set_native("s1", "codex-main", native_id="thr_1", last_turn_idx=0)
    store.set_native("s1", "codex-main", native_id="thr_2", last_turn_idx=3)
    row = store.get_native("s1", "codex-main")
    assert row["native_id"] == "thr_2" and row["last_turn_idx"] == 3


def test_usage_count_window(store) -> None:
    _session(store)
    store.record_usage(
        ts=100.0,
        executor_id="e1",
        session_id="s1",
        run_id=None,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0,
        stop="done",
    )
    store.record_usage(
        ts=200.0,
        executor_id="e1",
        session_id="s1",
        run_id=None,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0,
        stop="done",
    )
    store.record_usage(
        ts=200.0,
        executor_id="e2",
        session_id="s1",
        run_id=None,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        stop="done",
    )
    assert store.usage_count("e1", since_ts=150.0) == 1
    assert store.usage_count("e1", since_ts=0.0) == 2
    assert store.usage_count("e2", since_ts=0.0) == 1


def test_migration_idempotent(tmp_path) -> None:
    s1 = Store(tmp_path / "asf.db")
    s1.close()
    s2 = Store(tmp_path / "asf.db")  # reopen: migrations must be a no-op
    try:
        assert s2.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        versions = s2.conn.execute(
            "SELECT version FROM schema_versions WHERE component='router'"
        ).fetchone()[0]
        assert versions >= 1
    finally:
        s2.close()


def test_error_text_redacted(store) -> None:
    _session(store)
    store.append_turn(
        "s1",
        idx=0,
        role="assistant",
        executor_id="e",
        stop="error",
        text="boom sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ12",
    )
    row = store.turns("s1")[0]
    assert "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ12" not in row["text"]
    assert "[REDACTED]" in row["text"]


def test_set_summary_redacts_secrets(store) -> None:
    """T04: summaries are persisted context — they pass through redact() too."""
    _session(store)
    store.set_summary("s1", "used key sk-proj-9f8e7d6c5b4a3210fedcba9876543210ab earlier", 3)
    row = store.get_session("s1")
    assert row is not None
    assert "sk-proj-9f8e7d6c5b4a3210fedcba9876543210ab" not in row["summary"]
    assert "[REDACTED]" in row["summary"]


# ---- T09: CAS leases, usage indexes (schema v2), turn/cd helpers -----------------------


def test_atomic_acquire_and_owner_checked_release_lease(store) -> None:
    """T09: lease acquisition is a single-row CAS; only the owner can release."""
    _session(store)
    assert store.acquire_lease("s1", owner="a", until_ts=1100.0, now_ts=1000.0) is True
    assert store.acquire_lease("s1", owner="b", until_ts=1200.0, now_ts=1050.0) is False
    store.release_lease("s1", owner="b")  # wrong owner: no effect
    assert store.get_session("s1")["lock_owner"] == "a"
    store.release_lease("s1", owner="a")
    assert store.get_session("s1")["lock_owner"] is None
    assert store.acquire_lease("s1", owner="b", until_ts=1200.0, now_ts=1000.0) is True
    assert store.acquire_lease("s1", owner="c", until_ts=1300.0, now_ts=1200.0) is True  # expired


def test_usage_indexes_created_and_schema_version_2(tmp_path) -> None:
    """T09: schema v2 adds usage indexes; a v1 database upgrades in place."""
    s1 = Store(tmp_path / "asf.db")
    s1.close()
    s2 = Store(tmp_path / "asf.db")  # v1 file reopened: upgraded to v2, indexes exist
    try:
        version = s2.conn.execute(
            "SELECT version FROM schema_versions WHERE component='router'"
        ).fetchone()[0]
        assert version == 3
        indexes = {
            r["name"] for r in s2.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
        assert "idx_usage_exec_ts" in indexes
        assert "idx_usage_run_ts" in indexes
    finally:
        s2.close()


def test_next_turn_idx_handles_gaps(store) -> None:
    """T09: next_turn_idx is MAX(idx)+1 regardless of gaps or ordering."""
    _session(store)
    assert store.next_turn_idx("s1") == 0
    store.append_turn("s1", idx=5, role="user", text="hi", executor_id="", stop="done")
    store.append_turn("s1", idx=2, role="user", text="ho", executor_id="", stop="done")
    assert store.next_turn_idx("s1") == 6


def test_all_cooldowns_returns_active_rows(store) -> None:
    """T09: all_cooldowns(now) filters expired entries; fresh rows survive."""
    store.set_cooldown("e1", until_ts=1500.0, strikes=1, reason="quota")
    store.set_cooldown("e2", until_ts=900.0, strikes=2, reason="quota")
    active = store.all_cooldowns(now_ts=1000.0)
    assert [row["executor_id"] for row in active] == ["e1"]
    assert len(store.all_cooldowns(now_ts=None)) == 2  # no filter: everything


def test_recent_stops_window(store) -> None:
    """T09: recent_stops feeds the SQLite-backed K1 window (T10)."""
    store.record_usage(100.0, "e1", "s", None, 1, 1, 0.0, "done")
    store.record_usage(120.0, "e1", "s", None, 1, 1, 0.0, "quota")
    store.record_usage(130.0, "e1", "s", None, 1, 1, 0.0, "error")
    assert store.recent_stops("e1", since_ts=110.0) == ["quota", "error"]
    assert store.recent_stops("e1", since_ts=0.0) == ["done", "quota", "error"]


def test_lease_columns_roundtrip(store) -> None:
    _session(store)
    store.pin("s1", executor_id="codex-main", tier=3)
    store.set_lease("s1", owner="w1", until=999.0)
    sess = store.get_session("s1")
    assert sess["pinned_executor"] == "codex-main" and sess["tier"] == 3
    assert sess["lock_owner"] == "w1" and sess["lock_until"] == 999.0
    store.set_lease("s1", owner=None, until=None)
    assert store.get_session("s1")["lock_owner"] is None


def test_cooldown_roundtrip_and_clear(store) -> None:
    assert store.get_cooldown("e1") is None
    store.set_cooldown("e1", until_ts=1234.5, strikes=2, reason="quota")
    cd = store.get_cooldown("e1")
    assert cd["until_ts"] == 1234.5 and cd["strikes"] == 2 and cd["reason"] == "quota"
    store.clear_cooldown("e1")
    assert store.get_cooldown("e1") is None


def test_spend_sums_cost(store) -> None:
    _session(store)
    store.record_usage(
        ts=1.0,
        executor_id="e1",
        session_id="s1",
        run_id="r1",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.25,
        stop="done",
    )
    store.record_usage(
        ts=2.0,
        executor_id="e1",
        session_id="s1",
        run_id="r2",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.75,
        stop="done",
    )
    assert store.spend(since_ts=0.0) == pytest.approx(1.0)
    assert store.spend(since_ts=0.0, run_id="r1") == pytest.approx(0.25)


def test_env_value_redacted_in_turn_text(tmp_path, monkeypatch) -> None:
    """Plan 02 §9: the store redacts values of env vars named in api_key_envs."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-real-secret-value-123")
    s = Store(tmp_path / "asf.db", api_key_envs=["GROQ_API_KEY"])
    try:
        _session(s)
        s.append_turn("s1", 0, "assistant", "used key gsk-real-secret-value-123", "e1", "error")
        row = s.turns("s1")[0]
        assert "gsk-real-secret-value-123" not in row["text"]
        assert "[REDACTED]" in row["text"]
    finally:
        s.close()


def test_executor_overrides_roundtrip_and_clear(store) -> None:
    """Benchmark dynamic circuit state persistence in executor_overrides table."""
    assert store.get_executor_overrides() == {}
    store.set_executor_override(
        executor_id="opencode-gemini-hydrogen",
        status="healthy",
        enabled_override=1,
        priority_override=0,
        latency_s=6.5,
        detail="OK",
    )
    overrides = store.get_executor_overrides()
    assert "opencode-gemini-hydrogen" in overrides
    rec = overrides["opencode-gemini-hydrogen"]
    assert rec["status"] == "healthy"
    assert rec["enabled_override"] == 1
    assert rec["priority_override"] == 0
    assert rec["latency_s"] == 6.5
    assert rec["detail"] == "OK"

    # Demote to slow
    store.set_executor_override(
        executor_id="opencode-gemini-hydrogen",
        status="slow",
        enabled_override=0,
        priority_override=None,
        latency_s=22.4,
        detail="latency 22.40s > 15.0s",
    )
    rec2 = store.get_executor_overrides()["opencode-gemini-hydrogen"]
    assert rec2["status"] == "slow"
    assert rec2["enabled_override"] == 0
    assert rec2["priority_override"] is None

    store.clear_executor_override("opencode-gemini-hydrogen")
    assert store.get_executor_overrides() == {}


def test_executor_override_redacts_detail(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_KEY_SECRET", "super-secret-token")
    s = Store(tmp_path / "s2.sqlite", api_key_envs=["GEMINI_KEY_SECRET"])
    try:
        s.set_executor_override(
            "test-exec", "error", detail="Failed with super-secret-token in log"
        )
        rec = s.get_executor_overrides()["test-exec"]
        assert "super-secret-token" not in str(rec["detail"])
        assert "[REDACTED" in str(rec["detail"])
    finally:
        s.close()


def test_schema_version_is_3(store) -> None:
    from onechat.store import ROUTER_SCHEMA_VERSION

    assert ROUTER_SCHEMA_VERSION == 3
    rows = store._query("SELECT version FROM schema_versions WHERE component='router'")
    assert rows[0]["version"] == 3
