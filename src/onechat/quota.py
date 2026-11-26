"""Quota bookkeeping (Plan 02 §8 / T10): retry-after parsing, rolling windows, cooldowns.

The K1 statistical window is SQLite-backed (survives restarts); only QUOTA and TIMEOUT
count as backend failures — ERROR is our own fault and never blacklists a backend.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import UTC, datetime

from .config import CliProfile, CooldownPolicy
from .contracts import ExecutorSpec, Stop, TurnResult
from .store import Store

BACKOFF_STRIKES = [60, 300, 1800, 7200]
AUTH_COOLDOWN_S = 24 * 3600
_RPM_WINDOW_S = 60
_RPD_WINDOW_S = 86400
_FAILURE_STOPS = ("quota", "timeout")

_ABSOLUTE_RE = re.compile(r"resets? at (\d{4}-\d{2}-\d{2}T[0-9:.]+(?:Z|[+-]\d{2}:\d{2}))", re.I)
_RELATIVE_RE = re.compile(
    r"(?:try again|retry|wait|resets?)\s*(?:in|after)?\s*"
    r"((?:\d+(?:\.\d+)?\s*(?:h|ms|m|s)\b[\s,]*)+)",
    re.I,
)
_UNIT_SECONDS = {"h": 3600.0, "ms": 0.001, "m": 60.0, "s": 1.0}
_KEYWORDS = ("try again", "retry", "resets", "usage limit", "rate limit", "quota", "wait")


def parse_retry_after(text: str, now: datetime) -> int | None:
    """Extract a retry delay from quota error text. Returns seconds, or None (T10:
    durations are anchored to a retry/wait phrase, so window specs like '1000/1h'
    are never mistaken for delays; invalid absolute stamps never raise)."""
    lowered = text.lower()
    if not any(marker in lowered for marker in _KEYWORDS):
        return None
    absolute = _ABSOLUTE_RE.search(text)
    if absolute:
        reset = _parse_iso(absolute.group(1))
        if reset is not None:
            return max(int((reset - now).total_seconds()), 0)
    return _relative_from_phrase(text)


def _parse_iso(stamp: str) -> datetime | None:
    try:
        reset = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if reset.tzinfo is None:
        reset = reset.replace(tzinfo=UTC)
    return reset


def _relative_from_phrase(text: str) -> int | None:
    phrase = _RELATIVE_RE.search(text)
    if not phrase:
        return None
    total = 0.0
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(h|ms|m|s)\b", phrase.group(1), re.I):
        total += float(number) * _UNIT_SECONDS[unit.lower()]
    return math.ceil(total) if total > 0 else None


def _utc_now_ts() -> float:
    return datetime.now(UTC).timestamp()


class QuotaBook:
    """Rolling-window usage limits + cooldowns, backed by the Store."""

    def __init__(
        self,
        store: Store,
        clock: Callable[[], float] = _utc_now_ts,
        policy: CooldownPolicy | None = None,
    ) -> None:
        self._store = store
        self._clock = clock
        self._policy = policy or CooldownPolicy()

    def _statistically_failing(self, executor_id: str) -> bool:
        """SQLite-backed K1 window (T10): failure rate over recent stops."""
        now = self._clock()
        stops = self._store.recent_stops(executor_id, now - self._policy.window_s)
        if len(stops) < self._policy.min_requests:
            return False
        failures = sum(1 for stop in stops if stop.lower() in _FAILURE_STOPS)
        return failures / len(stops) >= self._policy.failure_rate

    def available(self, spec: ExecutorSpec) -> bool:
        cooldown = self._store.get_cooldown(spec.id)
        if cooldown and cooldown["until_ts"] > self._clock():
            return False
        now = self._clock()
        within_rpm = (
            spec.limits.rpm is not None
            and self._store.usage_count(spec.id, now - _RPM_WINDOW_S) >= spec.limits.rpm
        )
        within_rpd = (
            spec.limits.rpd is not None
            and self._store.usage_count(spec.id, now - _RPD_WINDOW_S) >= spec.limits.rpd
        )
        return not (within_rpm or within_rpd)

    def earliest_available(
        self, specs: list[ExecutorSpec], now: float | None = None
    ) -> float | None:
        """Min future expiry across specs, where each spec needs ALL blockers cleared."""
        now = self._clock() if now is None else now
        ready = [t for spec in specs if (t := self._spec_ready_at(spec, now)) is not None]
        return min(ready) if ready else None

    def _spec_ready_at(self, spec: ExecutorSpec, now: float) -> float | None:
        blockers: list[float] = []
        cooldown = self._store.get_cooldown(spec.id)
        if cooldown and cooldown["until_ts"] > now:
            blockers.append(float(cooldown["until_ts"]))
        for window_s, limit in (
            (_RPM_WINDOW_S, spec.limits.rpm),
            (_RPD_WINDOW_S, spec.limits.rpd),
        ):
            if limit is None:
                continue
            count = self._store.usage_count(spec.id, now - window_s)
            if count < limit:
                continue
            release_ts = self._store.nth_oldest_usage_ts(
                spec.id, now - window_s, offset=count - limit
            )
            if release_ts is not None:
                blockers.append(release_ts + window_s)
        return max(blockers) if blockers else None

    def record(
        self,
        spec: ExecutorSpec,
        result: TurnResult,
        session_id: str,
        run_id: str | None,
        sole_usable: bool = False,
        all_specs: list[ExecutorSpec] | None = None,
        profiles: dict[str, CliProfile] | None = None,
    ) -> None:
        now = self._clock()
        self._store.record_usage(
            ts=now,
            executor_id=spec.id,
            session_id=session_id,
            run_id=run_id,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            cost_usd=result.usage.cost_usd,
            stop=result.stop.value,
        )
        if result.stop == Stop.DONE:
            self._store.clear_cooldown(spec.id)
            self._clear_sibling_cooldown(spec, all_specs, profiles)
            return
        if result.stop == Stop.AUTH:
            until_ts = now + self._policy.auth_cooldown_s
            self._store.set_cooldown(spec.id, until_ts, 1, "auth")
            self._propagate_sibling_cooldown(spec, until_ts, "auth", all_specs, profiles)
            return
        if result.stop in (Stop.QUOTA, Stop.TIMEOUT, Stop.ERROR):
            self._backend_failure(spec, result, now, sole_usable, all_specs, profiles)

    def _clear_sibling_cooldown(
        self,
        spec: ExecutorSpec,
        all_specs: list[ExecutorSpec] | None,
        profiles: dict[str, CliProfile] | None,
    ) -> None:
        if not self._policy.share_profile_cooldown or not all_specs or not spec.profile:
            return
        prof = (profiles or {}).get(spec.profile)
        if not prof or not prof.share_cooldown:
            return
        for s in all_specs:
            if s.id != spec.id and s.profile == spec.profile and s.env == spec.env:
                self._store.clear_cooldown(s.id)

    def _is_structural_error(
        self,
        spec: ExecutorSpec,
        result: TurnResult,
        profiles: dict[str, CliProfile] | None,
    ) -> bool:
        if spec.kind != "cli" or result.stop != Stop.ERROR:
            return False
        err = (result.error or "").strip().lower()
        if not err:
            return True
        prof = (profiles or {}).get(spec.profile) if spec.profile else None
        if prof and any(p.lower() in err for p in prof.unavailable_patterns):
            return True
        markers = (
            "connection refused",
            "not running",
            "cannot connect",
            "no output",
            "remote debugging",
            "cdp",
        )
        return any(m in err for m in markers)

    def _failure_delay(
        self, spec: ExecutorSpec, result: TurnResult, strikes: int, is_structural: bool
    ) -> int | None:
        if result.stop == Stop.ERROR:
            if is_structural:
                base = self._policy.error_cooldown_s
            elif self._policy.error_cooldown_s > 0:
                return 60
            else:
                return None
        elif result.stop == Stop.TIMEOUT:
            base = self._policy.timeout_cooldown_s
        else:
            base = self._policy.quota_cooldown_s
        ladder = self._policy.backoff_strikes
        ladder_delay = ladder[min(strikes - 1, len(ladder) - 1)] if ladder else base
        return max(base, ladder_delay) if base > 0 else ladder_delay

    def _propagate_sibling_cooldown(
        self,
        spec: ExecutorSpec,
        until_ts: float,
        reason: str,
        all_specs: list[ExecutorSpec] | None,
        profiles: dict[str, CliProfile] | None,
    ) -> None:
        if not self._policy.share_profile_cooldown or not all_specs or not spec.profile:
            return
        prof = (profiles or {}).get(spec.profile)
        if not prof or not prof.share_cooldown:
            return
        siblings = [
            s
            for s in all_specs
            if s.id != spec.id and s.profile == spec.profile and s.env == spec.env
        ]
        if not siblings:
            return
        usable_now = [s for s in all_specs if s.enabled and self.available(s)]
        if len(usable_now) <= len(siblings) + 1 and all(
            s.id == spec.id or s in siblings for s in usable_now
        ):
            return
        for sib in siblings:
            self._store.set_cooldown(sib.id, until_ts, 1, reason)

    def _backend_failure(
        self,
        spec: ExecutorSpec,
        result: TurnResult,
        now: float,
        sole_usable: bool,
        all_specs: list[ExecutorSpec] | None = None,
        profiles: dict[str, CliProfile] | None = None,
    ) -> None:
        existing = self._store.get_cooldown(spec.id) or {}
        if result.stop == Stop.QUOTA and result.retry_after_s is not None:
            delay_s = max(1, result.retry_after_s)
            self._store.set_cooldown(
                spec.id, now + delay_s, existing.get("strikes", 0) + 1, "quota"
            )
            return
        is_structural = self._is_structural_error(spec, result, profiles)
        if sole_usable and not is_structural:
            return
        demonstrated = existing.get("strikes", 0) >= 1
        needs_gate = self._policy.min_requests > 1 and not is_structural
        if needs_gate and not demonstrated and not self._statistically_failing(spec.id):
            return
        strikes = existing.get("strikes", 0) + 1
        delay = self._failure_delay(spec, result, strikes, is_structural)
        if delay is None or delay <= 0:
            return
        reason = "error" if result.stop == Stop.ERROR else result.stop.value
        until_ts = now + delay
        self._store.set_cooldown(spec.id, until_ts, strikes, reason)
        self._propagate_sibling_cooldown(spec, until_ts, reason, all_specs, profiles)
