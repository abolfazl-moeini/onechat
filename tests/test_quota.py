"""Plan 02 quota tests: retry-after parsing, rolling windows, backoff strikes, auth cooldown."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from onechat.contracts import ExecutorSpec, Stop, Tier, Usage
from onechat.quota import QuotaBook, parse_retry_after
from onechat.store import Store

NOW = datetime(2026, 11, 2, 4, 0, 0, tzinfo=UTC).timestamp()


def spec(**kw) -> ExecutorSpec:
    base = dict(
        id="groq-fast",
        kind="api",
        model="m",
        tier=Tier.LOW,
        capabilities=frozenset({"chat"}),
        cost="free",
        family="meta",
        base_url="https://x/v1",
        api_key_env="K",
    )
    base.update(kw)
    return ExecutorSpec(**base)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("try again in 2h 5m", 7500),
        ("try again in 5m", 300),
        ("retry after 30s", 30),
        ("reached your usage limit. try again in 1h", 3600),
        ("resets at 2026-11-02T04:05:00Z", 300),
        ("rate limit exceeded", None),
    ],
)
def test_parse_retry_after_variants(text: str, expected: int | None) -> None:
    now = datetime.fromtimestamp(NOW, tz=UTC)
    assert parse_retry_after(text, now) == expected


def k1_policy():
    from onechat.config import CooldownPolicy

    return CooldownPolicy(
        min_requests=5,
        error_cooldown_s=0,
        quota_cooldown_s=60,
        timeout_cooldown_s=60,
        backoff_strikes=[60, 300, 1800, 7200],
    )


def _book(tmp_path, clock_time=None, policy=None):
    clock_time = clock_time if clock_time is not None else [NOW]
    store = Store(tmp_path / "asf.db")
    book = QuotaBook(store, clock=lambda: clock_time[0], policy=policy)
    return store, book, clock_time


def test_rpm_limit_blocks(tmp_path) -> None:
    store, book, clock = _book(tmp_path)
    s = spec(limits={"rpm": 2})
    for _ in range(2):
        book.record(s, _result(Stop.DONE), "sess", None)
        clock[0] += 5
    assert book.available(s) is False  # 2 requests inside the 60s window
    clock[0] += 120
    assert book.available(s) is True  # window rolled over
    store.close()


def test_rpd_limit_blocks(tmp_path) -> None:
    store, book, clock = _book(tmp_path)
    s = spec(limits={"rpd": 1})
    book.record(s, _result(Stop.DONE), "sess", None)
    clock[0] += 3600
    assert book.available(s) is False
    store.close()


def _result(stop: Stop, retry_after_s: int | None = None):
    from onechat.contracts import TurnResult

    return TurnResult(
        stop=stop, usage=Usage(input_tokens=1, output_tokens=1), retry_after_s=retry_after_s
    )


def test_quota_backoff_strikes(tmp_path) -> None:
    """Plan 09 K1: the backoff ladder arms after the statistical gate, then persists."""
    store, book, clock = _book(tmp_path, policy=k1_policy())
    s = spec()
    for _ in range(5):  # statistical gate: 5 failures inside the window
        book.record(s, _result(Stop.QUOTA), "sess", None)
    cd = store.get_cooldown(s.id)
    assert cd["strikes"] == 1
    assert cd["until_ts"] == pytest.approx(clock[0] + 60, abs=1.0)
    clock[0] += 61  # cooldown expired; the row (strikes=1) persists
    book.record(s, _result(Stop.QUOTA), "sess", None)  # demonstrated strike -> re-cool
    cd = store.get_cooldown(s.id)
    assert cd["strikes"] == 2
    assert cd["until_ts"] == pytest.approx(clock[0] + 300, abs=1.0)
    clock[0] += 301
    book.record(s, _result(Stop.QUOTA), "sess", None)
    cd = store.get_cooldown(s.id)
    assert cd["strikes"] == 3
    assert cd["until_ts"] == pytest.approx(clock[0] + 1800, abs=1.0)
    clock[0] += 1801
    book.record(s, _result(Stop.QUOTA, retry_after_s=45), "sess", None)
    cd = store.get_cooldown(s.id)
    assert cd["until_ts"] == pytest.approx(clock[0] + 45, abs=1.0)  # explicit retry-after wins
    store.close()


def test_auth_cooldown_24h(tmp_path) -> None:
    store, book, _clock = _book(tmp_path)
    s = spec()
    book.record(s, _result(Stop.AUTH), "sess", None)
    cd = store.get_cooldown(s.id)
    assert cd["reason"] == "auth"
    assert cd["until_ts"] == pytest.approx(NOW + 24 * 3600, abs=1.0)
    store.close()


def test_done_clears_strikes(tmp_path) -> None:
    store, book, _clock = _book(tmp_path, policy=k1_policy())
    s = spec()
    for _ in range(5):  # arm the statistical gate first
        book.record(s, _result(Stop.QUOTA), "sess", None)
    assert store.get_cooldown(s.id) is not None
    book.record(s, _result(Stop.DONE), "sess", None)
    assert store.get_cooldown(s.id) is None
    store.close()


def test_cooldown_blocks_availability(tmp_path) -> None:
    store, book, _clock = _book(tmp_path)
    s = spec()
    book.record(s, _result(Stop.AUTH), "sess", None)
    assert book.available(s) is False
    store.close()


def test_zero_retry_after_is_preserved(tmp_path) -> None:
    """An explicit retry_after_s = 0 must not be swapped for backoff strikes."""
    store, book, _clock = _book(tmp_path)
    s = spec()
    book.record(s, _result(Stop.QUOTA, retry_after_s=0), "sess", None)
    cd = store.get_cooldown(s.id)
    assert cd is not None
    assert cd["until_ts"] == pytest.approx(NOW, abs=1.0)
    store.close()


# --- Plan 09 K1: statistical cooldown gate (litellm semantics) ---


def test_isolated_quota_hit_does_not_cooldown(tmp_path) -> None:
    """A single quota hit below the statistical gate must not black out the backend."""
    store, book, _clock = _book(tmp_path, policy=k1_policy())
    s = spec()
    book.record(s, _result(Stop.QUOTA), "sess", None)
    assert store.get_cooldown(s.id) is None
    store.close()


def test_statistical_gate_requires_min_requests(tmp_path) -> None:
    store, book, _clock = _book(tmp_path, policy=k1_policy())
    s = spec()
    for _ in range(4):
        book.record(s, _result(Stop.QUOTA), "sess", None)
    assert store.get_cooldown(s.id) is None  # below cooldown_min_requests
    book.record(s, _result(Stop.QUOTA), "sess", None)  # 5th failure inside the window
    cd = store.get_cooldown(s.id)
    assert cd is not None
    assert cd["until_ts"] == pytest.approx(NOW + 60, abs=1.0)  # BACKOFF_STRIKES[0]
    store.close()


def test_failure_window_rolls(tmp_path) -> None:
    store, book, clock = _book(tmp_path, policy=k1_policy())
    s = spec()
    for _ in range(5):
        book.record(s, _result(Stop.QUOTA), "sess", None)
        clock[0] += 61  # each hit ages out of the 60 s window
    assert store.get_cooldown(s.id) is None  # never 5 inside the window
    store.close()


def test_explicit_retry_after_beats_statistical_gate(tmp_path) -> None:
    """The provider's own retry-after is honored immediately, gate or no gate."""
    store, book, _clock = _book(tmp_path)
    s = spec()
    book.record(s, _result(Stop.QUOTA, retry_after_s=30), "sess", None)
    cd = store.get_cooldown(s.id)
    assert cd is not None
    assert cd["until_ts"] == pytest.approx(NOW + 30, abs=1.0)
    store.close()


def test_sole_usable_skips_statistical_cooldown(tmp_path) -> None:
    """Isolated hits never cool the only usable executor (Plan 09 K1)."""
    store, book, _clock = _book(tmp_path, policy=k1_policy())
    s = spec()
    for _ in range(6):
        book.record(s, _result(Stop.QUOTA), "sess", None, sole_usable=True)
    assert store.get_cooldown(s.id) is None
    store.close()


def test_sole_usable_still_honors_explicit_retry_after(tmp_path) -> None:
    """Provider-declared retry-after still cools the sole executor (daemon wakes at it)."""
    store, book, _clock = _book(tmp_path)
    s = spec()
    book.record(s, _result(Stop.QUOTA, retry_after_s=45), "sess", None, sole_usable=True)
    assert store.get_cooldown(s.id) is not None
    store.close()


def test_error_stop_never_touches_cooldown(tmp_path) -> None:
    """stop=ERROR is our fault, not the backend's health: strikes/cooldowns untouched."""
    store, book, _clock = _book(tmp_path, policy=k1_policy())
    s = spec()
    book.record(s, _result(Stop.ERROR), "sess", None)
    assert store.get_cooldown(s.id) is None
    book.record(s, _result(Stop.QUOTA, retry_after_s=120), "sess", None)
    before = store.get_cooldown(s.id)
    book.record(s, _result(Stop.ERROR), "sess", None)
    assert store.get_cooldown(s.id) == before
    store.close()


def test_mixed_window_failure_rate_gates(tmp_path) -> None:
    """The gate counts the failure RATE over the window, not just failures."""
    store, book, _clock = _book(tmp_path, policy=k1_policy())
    s = spec()
    for _ in range(3):
        book.record(s, _result(Stop.QUOTA), "sess", None)
    for _ in range(2):
        book.record(s, _result(Stop.DONE), "sess", None)  # 3/5 = 60% failures
    book.record(s, _result(Stop.QUOTA), "sess", None)  # 6 requests, 4/6 failures
    assert store.get_cooldown(s.id) is not None
    store.close()


# ---- T10: phrase-anchored retry parsing, SQLite K1 window, TIMEOUT strikes -------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("limit 1000/1h exceeded, retry in 5s", 5),  # '1000/1h' is a window spec, not a delay
        ("try again in 1.5s", 2),  # decimal seconds ceil
        ("resets at 2026-13-45T99:99:99Z", None),  # invalid ISO must not raise
    ],
)
def test_parse_retry_after_t10_variants(text: str, expected: int | None) -> None:
    now = datetime.fromtimestamp(NOW, tz=UTC)
    assert parse_retry_after(text, now) == expected


def test_stop_error_never_pollutes_k1_window(tmp_path) -> None:
    """T10: only QUOTA/TIMEOUT are K1 failures — ERROR must not black out a backend."""
    store, book, clock = _book(tmp_path, policy=k1_policy())
    s = spec()
    for _ in range(3):
        book.record(s, _result(Stop.QUOTA), "sess", None)
    for _ in range(4):
        book.record(s, _result(Stop.ERROR), "sess", None)  # 7 requests, 3 real failures
        clock[0] += 0.1
    assert store.get_cooldown(s.id) is None  # 3/7 < 50% failure rate
    store.close()


def test_repeated_timeouts_trigger_cooldown(tmp_path) -> None:
    """T10: TIMEOUT is backend sickness too — the statistical gate arms on it."""
    store, book, clock = _book(tmp_path, policy=k1_policy())
    s = spec()
    for _ in range(5):
        book.record(s, _result(Stop.TIMEOUT), "sess", None)
        clock[0] += 0.1
    cd = store.get_cooldown(s.id)
    assert cd is not None and cd["until_ts"] == pytest.approx(clock[0] + 60, abs=1.0)
    store.close()


def test_k1_window_survives_new_quota_book_instance(tmp_path) -> None:
    """T10: the K1 window is SQLite-backed — a fresh QuotaBook sees past failures."""
    store, book, _clock = _book(tmp_path, policy=k1_policy())
    s = spec()
    for _ in range(4):
        book.record(s, _result(Stop.QUOTA), "sess", None)
    book2 = QuotaBook(store, clock=lambda: NOW + 1, policy=k1_policy())
    book2.record(s, _result(Stop.QUOTA), "sess", None)  # 5th failure: gate arms
    cd = store.get_cooldown(s.id)
    assert cd is not None
    store.close()


def test_earliest_available_includes_rpm_and_cooldown(tmp_path) -> None:
    """T10: earliest_available spans cooldown expiry and rpm/rpd window slots."""
    store, book, clock = _book(tmp_path)
    limited = spec(id="rpm-one", limits={"rpm": 1})
    book.record(limited, _result(Stop.DONE), "sess", None)  # 1 request inside the window
    cooled = spec(id="cooled-one")
    book.record(cooled, _result(Stop.AUTH), "sess", None)  # 24h auth cooldown
    earliest = book.earliest_available([limited, cooled])
    assert earliest is not None
    assert earliest == pytest.approx(NOW + 60, abs=2.0)  # rpm slot frees sooner than the auth cd
    store.close()


def test_earliest_available_uses_max_blocker_per_spec(tmp_path) -> None:
    """Bug 4: spec with 10s RPM block + 300s cooldown is ready at max(+300), not +10."""
    store, book, _clock = _book(tmp_path)
    s = spec(id="both", limits={"rpm": 1})
    book.record(s, _result(Stop.DONE), "sess", None)
    store.set_cooldown("both", until_ts=NOW + 300, strikes=1, reason="quota")
    earliest = book.earliest_available([s], now=NOW)
    assert earliest == pytest.approx(NOW + 300, abs=1.0)
    store.close()


def test_earliest_available_handles_over_limit_usage_window(tmp_path) -> None:
    """Bug 4: if usage_count > rpm, expiry waits for the (count - limit)-th oldest entry."""
    store, book, clock = _book(tmp_path)
    s = spec(id="over", limits={"rpm": 1})
    book.record(s, _result(Stop.DONE), "sess", None)  # ts = NOW
    clock[0] = NOW + 20
    book.record(s, _result(Stop.DONE), "sess", None)  # ts = NOW + 20 (count=2, limit=1)
    earliest = book.earliest_available([s], now=NOW + 25)
    assert earliest == pytest.approx(NOW + 20 + 60, abs=1.0)
    store.close()


def test_parse_retry_after_accepts_fractional_and_offset_iso() -> None:
    """Bug 5: fractional seconds and +00:00 offsets in ISO reset stamps must parse."""
    now = datetime.fromtimestamp(NOW, tz=UTC)
    assert parse_retry_after("rate limit resets at 2026-11-02T04:01:00.500Z", now) == 60
    assert parse_retry_after("rate limit resets at 2026-11-02T04:01:00+00:00", now) == 60


def test_timeout_cooldown_records_timeout_reason(tmp_path) -> None:
    """Bug 5: TIMEOUT cooldowns record reason='timeout' (not 'quota')."""
    store, book, clock = _book(tmp_path, policy=k1_policy())
    s = spec()
    for _ in range(5):
        book.record(s, _result(Stop.TIMEOUT), "sess", None)
        clock[0] += 0.1
    cd = store.get_cooldown(s.id)
    assert cd is not None and cd["reason"] == "timeout"
    store.close()


# ---------------------------------------------------------------------------
# Multi-hour failure & quota cooldown cache tests (Revised Remediation)
# ---------------------------------------------------------------------------


def test_default_immediate_quota_cooldown_multi_hour(tmp_path) -> None:
    store, book, clock = _book(tmp_path)
    s = spec()
    book.record(s, _result(Stop.QUOTA), "sess", None)
    cd = store.get_cooldown(s.id)
    assert cd is not None
    assert cd["reason"] == "quota"
    assert cd["until_ts"] == pytest.approx(clock[0] + 7200, abs=1.0)
    assert cd["strikes"] == 1
    store.close()


def test_default_immediate_timeout_cooldown_multi_hour(tmp_path) -> None:
    store, book, clock = _book(tmp_path)
    s = spec()
    book.record(s, _result(Stop.TIMEOUT), "sess", None)
    cd = store.get_cooldown(s.id)
    assert cd is not None
    assert cd["reason"] == "timeout"
    assert cd["until_ts"] == pytest.approx(clock[0] + 7200, abs=1.0)
    assert cd["strikes"] == 1
    store.close()


def test_structural_error_cooldown_multi_hour(tmp_path) -> None:
    from onechat.config import CliProfile
    from onechat.contracts import TurnResult

    store, book, clock = _book(tmp_path)
    s = spec(id="autoclaw-review", kind="cli", profile="autoclaw")
    profile = CliProfile(
        name="autoclaw",
        binary="autoclaw",
        new=["run"],
        resume=["resume"],
        parser="codex_jsonl",
        share_cooldown=True,
        unavailable_patterns=["connection refused", "not running"],
    )
    result = TurnResult(
        stop=Stop.ERROR,
        error="connection refused: cannot connect to cdp port 9222",
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    book.record(s, result, "sess", None, profiles={"autoclaw": profile})
    cd = store.get_cooldown(s.id)
    assert cd is not None
    assert cd["reason"] == "error"
    assert cd["until_ts"] == pytest.approx(clock[0] + 14400, abs=1.0)
    store.close()


def test_generic_error_does_not_blacklist_four_hours(tmp_path) -> None:
    from onechat.contracts import TurnResult

    store, book, clock = _book(tmp_path)
    s = spec(id="api-mod", kind="api")
    result = TurnResult(
        stop=Stop.ERROR,
        error="invalid prompt syntax: unexpected token",
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    book.record(s, result, "sess", None)
    cd = store.get_cooldown(s.id)
    # Generic non-structural error does not get 4-hour cooldown
    assert cd is None or cd["until_ts"] <= clock[0] + 60
    store.close()


def test_shared_profile_cooldown_propagation(tmp_path) -> None:
    from onechat.config import CliProfile
    from onechat.contracts import TurnResult

    store, book, clock = _book(tmp_path)
    s1 = spec(id="autoclaw-review", kind="cli", profile="autoclaw", env={"PORT": "9222"})
    s2 = spec(id="autoclaw-flash", kind="cli", profile="autoclaw", env={"PORT": "9222"})
    s3 = spec(id="autoclaw-other-env", kind="cli", profile="autoclaw", env={"PORT": "9333"})
    s_other = spec(id="codex-main", kind="cli", profile="codex", env={})
    all_specs = [s1, s2, s3, s_other]

    profile = CliProfile(
        name="autoclaw",
        binary="autoclaw",
        new=["run"],
        resume=["resume"],
        parser="codex_jsonl",
        share_cooldown=True,
    )
    result = TurnResult(
        stop=Stop.ERROR,
        error="not running",
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    book.record(s1, result, "sess", None, all_specs=all_specs, profiles={"autoclaw": profile})

    assert store.get_cooldown("autoclaw-review") is not None
    cd2 = store.get_cooldown("autoclaw-flash")
    assert cd2 is not None
    assert cd2["until_ts"] == pytest.approx(clock[0] + 14400, abs=1.0)
    assert store.get_cooldown("autoclaw-other-env") is None
    assert store.get_cooldown("codex-main") is None
    store.close()


def test_shared_profile_cooldown_spares_sole_usable(tmp_path) -> None:
    from onechat.config import CliProfile
    from onechat.contracts import TurnResult

    store, book, clock = _book(tmp_path)
    s1 = spec(id="autoclaw-review", kind="cli", profile="autoclaw")
    s2 = spec(id="autoclaw-flash", kind="cli", profile="autoclaw")
    all_specs = [s1, s2]

    profile = CliProfile(
        name="autoclaw",
        binary="autoclaw",
        new=["run"],
        resume=["resume"],
        parser="codex_jsonl",
        share_cooldown=True,
    )
    result = TurnResult(
        stop=Stop.ERROR,
        error="not running",
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    book.record(s1, result, "sess", None, all_specs=all_specs, profiles={"autoclaw": profile})
    assert store.get_cooldown("autoclaw-review") is not None
    assert store.get_cooldown("autoclaw-flash") is None
    store.close()


def test_clear_cooldown_overload(tmp_path) -> None:
    store = Store(tmp_path / "asf.db")
    store.set_cooldown("exec-1", NOW + 100, 1, "quota")
    store.set_cooldown("exec-2", NOW + 200, 1, "error")
    assert len(store.all_cooldowns()) == 2

    store.clear_cooldown("exec-1")
    assert len(store.all_cooldowns()) == 1

    store.set_cooldown("exec-3", NOW + 300, 1, "auth")
    store.clear_cooldown(None)
    assert len(store.all_cooldowns()) == 0
    store.close()


def test_done_clears_shared_profile_sibling_cooldown(tmp_path) -> None:
    from onechat.config import CliProfile

    store, book, clock = _book(tmp_path)
    s1 = spec(id="autoclaw-review", kind="cli", profile="autoclaw", env={"PORT": "9222"})
    s2 = spec(id="autoclaw-flash", kind="cli", profile="autoclaw", env={"PORT": "9222"})
    s3 = spec(id="autoclaw-diff-env", kind="cli", profile="autoclaw", env={"PORT": "9333"})
    all_specs = [s1, s2, s3]
    profile = CliProfile(
        name="autoclaw",
        binary="autoclaw",
        new=["run"],
        resume=["resume"],
        parser="codex_jsonl",
        share_cooldown=True,
    )
    store.set_cooldown(s1.id, clock[0] + 14400, 1, "error")
    store.set_cooldown(s2.id, clock[0] + 14400, 1, "error")
    store.set_cooldown(s3.id, clock[0] + 14400, 1, "error")

    book.record(
        s1,
        _result(Stop.DONE),
        "sess",
        None,
        all_specs=all_specs,
        profiles={"autoclaw": profile},
    )
    assert store.get_cooldown(s1.id) is None
    assert store.get_cooldown(s2.id) is None
    assert store.get_cooldown(s3.id) is not None
    store.close()


def test_desktop_profile_generic_error_not_structural(tmp_path) -> None:
    from onechat.config import CliProfile
    from onechat.contracts import TurnResult

    store, book, clock = _book(tmp_path)
    s = spec(id="autoclaw-review", kind="cli", profile="autoclaw")
    profile = CliProfile(
        name="autoclaw",
        binary="autoclaw",
        new=["run"],
        resume=["resume"],
        parser="codex_jsonl",
        share_cooldown=True,
        unavailable_patterns=["connection refused", "not running"],
    )
    result = TurnResult(
        stop=Stop.ERROR,
        error="JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    book.record(s, result, "sess", None, profiles={"autoclaw": profile})
    cd = store.get_cooldown(s.id)
    assert cd is not None
    assert cd["until_ts"] <= clock[0] + 60
    store.close()


def test_structural_error_cools_even_when_sole_usable(tmp_path) -> None:
    from onechat.config import CliProfile
    from onechat.contracts import TurnResult

    store, book, clock = _book(tmp_path)
    s = spec(id="autoclaw-flash", kind="cli", profile="autoclaw")
    profile = CliProfile(
        name="autoclaw",
        binary="autoclaw",
        new=["run"],
        resume=["resume"],
        parser="codex_jsonl",
        share_cooldown=True,
        unavailable_patterns=["connection refused", "not running"],
    )
    result = TurnResult(
        stop=Stop.ERROR,
        error="connection refused: cannot connect to cdp port 9222",
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    book.record(s, result, "sess", None, sole_usable=True, profiles={"autoclaw": profile})
    cd = store.get_cooldown(s.id)
    assert cd is not None
    assert cd["reason"] == "error"
    assert cd["until_ts"] == pytest.approx(clock[0] + 14400, abs=1.0)
    store.close()


def test_sibling_propagation_ignores_disabled_specs_in_sole_usable_check(tmp_path) -> None:
    """Disabled specs must not be counted as usable in sibling cooldown propagation."""
    from onechat.config import CliProfile
    from onechat.contracts import TurnResult

    store, book, clock = _book(tmp_path)
    s1 = spec(id="codex-main", kind="cli", profile="codex", enabled=True)
    s2 = spec(id="codex-fast", kind="cli", profile="codex", enabled=True)
    s3_disabled = spec(id="other-disabled", kind="cli", profile="other", enabled=False)

    profile = CliProfile(
        name="codex",
        binary="codex",
        new=["run"],
        resume=["resume"],
        parser="codex_jsonl",
        share_cooldown=True,
    )
    result = TurnResult(stop=Stop.QUOTA, error="rate limit exceeded")
    all_specs = [s1, s2, s3_disabled]
    book.record(s1, result, "sess", None, all_specs=all_specs, profiles={"codex": profile})
    assert store.get_cooldown(s1.id) is not None
    assert store.get_cooldown(s2.id) is None
    store.close()


def test_empty_error_on_cli_is_structural(tmp_path) -> None:
    """Empty error or None on CLI failure is treated as structural."""
    from onechat.config import CliProfile
    from onechat.contracts import TurnResult

    store, book, clock = _book(tmp_path)
    s = spec(id="cli-fail", kind="cli", profile="some-cli")
    profile = CliProfile(
        name="some-cli",
        binary="some-cli",
        new=["run"],
        resume=["resume"],
        parser="codex_jsonl",
    )
    result = TurnResult(stop=Stop.ERROR, error="", usage=Usage(input_tokens=0, output_tokens=0))
    book.record(s, result, "sess", None, profiles={"some-cli": profile})
    cd = store.get_cooldown(s.id)
    assert cd is not None
    assert cd["until_ts"] == pytest.approx(clock[0] + 14400, abs=1.0)
    store.close()
