"""SQLite WAL store (Plan 02 §7): sessions, turns, native sessions, usage, cooldowns.

One connection + threading.Lock; every public method is one transaction. Errors persisted
through append_turn are redacted (never store secret-shaped text).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .redact import make_redactor, redact

ROUTER_SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_versions(component TEXT PRIMARY KEY, version INT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, title TEXT, workdir TEXT, created_at REAL,
    pinned_executor TEXT, tier INT, summary TEXT, summary_upto INT DEFAULT 0,
    lock_owner TEXT, lock_until REAL);
CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY, session_id TEXT REFERENCES sessions(id),
    idx INT, role TEXT, text TEXT, executor_id TEXT, stop TEXT, created_at REAL,
    subagent_role TEXT, UNIQUE(session_id, idx));
CREATE TABLE IF NOT EXISTS native_sessions(session_id TEXT, executor_id TEXT, native_id TEXT,
    last_turn_idx INT, PRIMARY KEY(session_id, executor_id));
CREATE TABLE IF NOT EXISTS usage(id INTEGER PRIMARY KEY, ts REAL, executor_id TEXT, session_id TEXT,
    run_id TEXT, input_tokens INT, output_tokens INT, cost_usd REAL, stop TEXT);
CREATE TABLE IF NOT EXISTS cooldowns(executor_id TEXT PRIMARY KEY, until_ts REAL, strikes INT,
    reason TEXT);
CREATE TABLE IF NOT EXISTS executor_overrides(executor_id TEXT PRIMARY KEY, status TEXT NOT NULL,
    enabled_override INT, priority_override INT, latency_s REAL, checked_at REAL NOT NULL,
    detail TEXT);
CREATE INDEX IF NOT EXISTS idx_usage_exec_ts ON usage(executor_id, ts);
CREATE INDEX IF NOT EXISTS idx_usage_run_ts ON usage(run_id, ts);
"""


class Store:
    """All router persistence. Methods never raise for missing rows; they return None/[]."""

    def __init__(self, path: Path | str, api_key_envs: list[str] | None = None) -> None:
        self._redact = make_redactor(list(api_key_envs)) if api_key_envs else redact
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute("PRAGMA busy_timeout=5000")  # BEFORE WAL (T09)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """Bump the recorded router schema version forward (T09: v1 -> v2, v3 subagent_role)."""
        cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(turns)").fetchall()}
        if "subagent_role" not in cols:
            self._conn.execute("ALTER TABLE turns ADD COLUMN subagent_role TEXT")
        self._conn.execute(
            "INSERT INTO schema_versions(component, version) VALUES('router', ?) "
            "ON CONFLICT(component) DO UPDATE SET version=excluded.version "
            "WHERE excluded.version > schema_versions.version",
            (ROUTER_SCHEMA_VERSION,),
        )

    def _query(self, sql: str, params: tuple | list = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def _query_row(self, sql: str, params: tuple | list = ()) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute(sql, params).fetchone()
        return dict(r) if r else None

    def _exec(self, sql: str, params: tuple | list = ()) -> None:
        with self._lock, self._conn:
            self._conn.execute(sql, params)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- sessions ------------------------------------------------------------
    def create_session(self, sid: str, title: str, workdir: str) -> None:
        self._exec(
            "INSERT INTO sessions(id, title, workdir, created_at) VALUES(?,?,?,?)",
            (sid, title, workdir, time.time()),
        )

    def get_session(self, sid: str) -> dict[str, Any] | None:
        return self._query_row("SELECT * FROM sessions WHERE id=?", (sid,))

    def list_sessions(self) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM sessions ORDER BY created_at DESC")

    def latest_session_for_workdir(self, workdir: str | Path) -> dict[str, Any] | None:
        """Find the most recent session created for a normalized workdir."""
        target_path = str(Path(workdir).expanduser().resolve())
        # First try exact match on normalized path
        rows = self._query("SELECT * FROM sessions ORDER BY created_at DESC")
        for row in rows:
            row_wd = row.get("workdir")
            if row_wd and str(Path(row_wd).expanduser().resolve()) == target_path:
                return row
        return None

    def pin(self, sid: str, executor_id: str | None, tier: int | None) -> None:
        self._exec(
            "UPDATE sessions SET pinned_executor=?, tier=? WHERE id=?", (executor_id, tier, sid)
        )

    def set_lease(self, sid: str, owner: str | None, until: float | None) -> None:
        self._exec("UPDATE sessions SET lock_owner=?, lock_until=? WHERE id=?", (owner, until, sid))

    def acquire_lease(self, sid: str, owner: str, until_ts: float, now_ts: float) -> bool:
        """Single-row CAS (T09): free, expired, or already-ours -> claimed; else False."""
        with self._lock, self._conn:
            c = self._conn.execute(
                "UPDATE sessions SET lock_owner=?, lock_until=? WHERE id=? "
                "AND (lock_owner IS NULL OR lock_until <= ? OR lock_owner = ?)",
                (owner, until_ts, sid, now_ts, owner),
            )
            return c.rowcount == 1

    def release_lease(self, sid: str, owner: str) -> None:
        """Owner-checked release: another owner's lease is never touched (T09)."""
        sql = "UPDATE sessions SET lock_owner=NULL, lock_until=NULL WHERE id=? AND lock_owner=?"
        self._exec(sql, (sid, owner))

    def set_summary(self, sid: str, summary: str, upto: int) -> None:
        self._exec(
            "UPDATE sessions SET summary=?, summary_upto=? WHERE id=?",
            (self._redact(summary), upto, sid),
        )

    # -- turns ---------------------------------------------------------------
    def append_turn(
        self,
        sid: str,
        idx: int,
        role: str,
        text: str,
        executor_id: str,
        stop: str,
        subagent_role: str | None = None,
    ) -> None:
        self._exec(
            "INSERT INTO turns(session_id, idx, role, text, executor_id, stop, created_at, "
            "subagent_role) VALUES(?,?,?,?,?,?,?,?)",
            (sid, idx, role, self._redact(text), executor_id, stop, time.time(), subagent_role),
        )

    def turns(self, sid: str, since_idx: int = 0) -> list[dict[str, Any]]:
        return self._query(
            "SELECT * FROM turns WHERE session_id=? AND idx>=? ORDER BY idx", (sid, since_idx)
        )

    def next_turn_idx(self, sid: str) -> int:
        """MAX(idx)+1 across all stored turns — gap/order tolerant (T09)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(idx) + 1, 0) FROM turns WHERE session_id=?", (sid,)
            ).fetchone()
        return int(row[0])

    # -- native sessions -----------------------------------------------------
    def set_native(self, sid: str, executor_id: str, native_id: str, last_turn_idx: int) -> None:
        self._exec(
            "INSERT INTO native_sessions(session_id, executor_id, native_id, last_turn_idx) "
            "VALUES(?,?,?,?) ON CONFLICT(session_id, executor_id) "
            "DO UPDATE SET native_id=excluded.native_id, last_turn_idx=excluded.last_turn_idx",
            (sid, executor_id, native_id, last_turn_idx),
        )

    def get_native(self, sid: str, executor_id: str) -> dict[str, Any] | None:
        return self._query_row(
            "SELECT * FROM native_sessions WHERE session_id=? AND executor_id=?",
            (sid, executor_id),
        )

    # -- usage ---------------------------------------------------------------
    def record_usage(
        self,
        ts: float,
        executor_id: str,
        session_id: str,
        run_id: str | None,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None,
        stop: str,
    ) -> None:
        self._exec(
            "INSERT INTO usage(ts, executor_id, session_id, run_id, input_tokens, "
            "output_tokens, cost_usd, stop) VALUES(?,?,?,?,?,?,?,?)",
            (ts, executor_id, session_id, run_id, input_tokens, output_tokens, cost_usd, stop),
        )

    def usage_count(self, executor_id: str, since_ts: float) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM usage WHERE executor_id=? AND ts>=?", (executor_id, since_ts)
            ).fetchone()
        return int(row[0])

    def spend(self, since_ts: float, run_id: str | None = None) -> float:
        sql = "SELECT COALESCE(SUM(cost_usd),0) FROM usage WHERE ts>=?"
        params: list[Any] = [since_ts] if run_id is None else [since_ts, run_id]
        if run_id is not None:
            sql += " AND run_id=?"
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return float(row[0])

    def oldest_usage_ts(self, executor_id: str, since_ts: float) -> float | None:
        return self.nth_oldest_usage_ts(executor_id, since_ts, offset=0)

    def nth_oldest_usage_ts(
        self, executor_id: str, since_ts: float, offset: int = 0
    ) -> float | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT ts FROM usage WHERE executor_id=? AND ts>=? "
                "ORDER BY ts ASC LIMIT 1 OFFSET ?",
                (executor_id, since_ts, max(0, offset)),
            ).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def delete_session(self, sid: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM turns WHERE session_id=?", (sid,))
            self._conn.execute("DELETE FROM native_sessions WHERE session_id=?", (sid,))
            self._conn.execute("DELETE FROM sessions WHERE id=?", (sid,))

    # -- cooldowns -----------------------------------------------------------
    def set_cooldown(self, executor_id: str, until_ts: float, strikes: int, reason: str) -> None:
        self._exec(
            "INSERT INTO cooldowns(executor_id, until_ts, strikes, reason) VALUES(?,?,?,?) "
            "ON CONFLICT(executor_id) DO UPDATE SET "
            "until_ts=excluded.until_ts, strikes=excluded.strikes, reason=excluded.reason",
            (executor_id, until_ts, strikes, reason),
        )

    def get_cooldown(self, executor_id: str) -> dict[str, Any] | None:
        return self._query_row("SELECT * FROM cooldowns WHERE executor_id=?", (executor_id,))

    def clear_cooldown(self, executor_id: str | None = None) -> None:
        p = (executor_id,) if executor_id else ()
        self._exec("DELETE FROM cooldowns WHERE executor_id=?" if p else "DELETE FROM cooldowns", p)

    def all_cooldowns(self, now_ts: float | None = None) -> list[dict[str, Any]]:
        """Active cooldowns (until_ts > now) — []-safe when now_ts is None (T09)."""
        if now_ts is None:
            return self._query("SELECT * FROM cooldowns ORDER BY executor_id")
        return self._query(
            "SELECT * FROM cooldowns WHERE until_ts > ? ORDER BY until_ts", (now_ts,)
        )

    def recent_stops(self, executor_id: str, since_ts: float) -> list[str]:
        """Stops recorded for an executor since a timestamp, oldest first (T09/T10 K1)."""
        return [
            str(r["stop"] or "")
            for r in self._query(
                "SELECT stop FROM usage WHERE executor_id=? AND ts>=? ORDER BY ts",
                (executor_id, since_ts),
            )
        ]

    # -- executor overrides (benchmark watchdog) ------------------------------
    def set_executor_override(
        self,
        executor_id: str,
        status: str,
        enabled_override: int | None = None,
        priority_override: int | None = None,
        latency_s: float | None = None,
        detail: str | None = None,
    ) -> None:
        sql = (
            "INSERT INTO executor_overrides("
            "executor_id, status, enabled_override, priority_override, "
            "latency_s, checked_at, detail) VALUES(?,?,?,?,?,?,?) ON CONFLICT(executor_id) "
            "DO UPDATE SET status=excluded.status, enabled_override=excluded.enabled_override, "
            "priority_override=excluded.priority_override, latency_s=excluded.latency_s, "
            "checked_at=excluded.checked_at, detail=excluded.detail"
        )
        clean = self._redact(detail) if detail else None
        vals = (
            executor_id,
            status,
            enabled_override,
            priority_override,
            latency_s,
            time.time(),
            clean,
        )
        self._exec(sql, vals)

    def get_executor_overrides(self) -> dict[str, dict[str, Any]]:
        return {str(r["executor_id"]): r for r in self._query("SELECT * FROM executor_overrides")}

    def clear_executor_override(self, executor_id: str) -> None:
        self._exec("DELETE FROM executor_overrides WHERE executor_id=?", (executor_id,))
