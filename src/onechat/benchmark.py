"""Benchmark watchdog runner (T43): latency/health checks and dynamic circuit overrides."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel, Field

from .config import BenchmarkConfig, BenchmarkTarget
from .redact import redact
from .store import Store

ProbeFn = Callable[[str, float], tuple[bool, float, str]]
_NON_RETRY_HTTP_RE = re.compile(r"\bHTTP (?:401|403|429)\b")


class BenchmarkResult(BaseModel):
    executor_id: str
    model: str
    status: str  # "healthy" | "slow" | "error"
    latency_s: float | None = None
    enabled_override: int | None = None  # 1 = healthy, 0 = slow/error
    priority_override: int | None = None
    detail: str = "OK"


class BenchmarkReport(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    results: list[BenchmarkResult] = Field(default_factory=list)
    promoted_roles: dict[str, str] = Field(default_factory=dict)


def http_probe(
    model: str,
    timeout_s: float,
    endpoint: str = "http://127.0.0.1:8789",
) -> tuple[bool, float, str]:
    """Lightweight HTTP POST probe directly to the local Gemini proxy endpoint."""
    clean_model = model.split("/")[-1]
    url = f"{endpoint}/v1beta/models/{clean_model}:generateContent"
    payload = json.dumps(
        {
            "contents": [{"parts": [{"text": "ping"}]}],
            "generationConfig": {"maxOutputTokens": 8},
        }
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-rotator-ttfb-timeout-ms": str(int(timeout_s * 1000)),
        "x-rotator-max-attempts": "2",
    }
    req = urllib.request.Request(url, data=payload, headers=headers)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            dt = time.monotonic() - t0
            data = json.loads(raw)
            if not isinstance(data, dict) or not data.get("candidates"):
                return False, dt, "invalid response: missing candidates"
            return True, dt, "OK"
    except urllib.error.HTTPError as err:
        dt = time.monotonic() - t0
        try:
            body = err.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        short_body = body.replace("\n", " ").strip()[:140]
        detail = f"HTTP {err.code}: {err.reason}" + (f" - {short_body}" if short_body else "")
        return False, dt, detail
    except Exception as exc:
        return False, time.monotonic() - t0, str(exc)


def _is_non_retryable_error(detail: str) -> bool:
    """Check if error is definitive quota / plan / billing / auth exhaustion (non-retryable)."""
    if _NON_RETRY_HTTP_RE.search(detail):
        return True
    lowered = detail.lower()
    markers = ("quota", "usage limit", "resource_exhausted", "billing", "perday", "daily")
    return any(m in lowered for m in markers)


def _eval_target(
    target: BenchmarkTarget,
    ok: bool,
    dt: float,
    detail: str,
    max_latency_s: float,
) -> BenchmarkResult:
    if ok and dt <= max_latency_s:
        return BenchmarkResult(
            executor_id=target.executor_id,
            model=target.model,
            status="healthy",
            latency_s=round(dt, 3),
            enabled_override=1,
            priority_override=target.promoted_priority,
            detail="OK",
        )
    if ok:
        return BenchmarkResult(
            executor_id=target.executor_id,
            model=target.model,
            status="slow",
            latency_s=round(dt, 3),
            enabled_override=0,
            priority_override=None,
            detail=f"latency {dt:.2f}s > {max_latency_s}s",
        )
    return BenchmarkResult(
        executor_id=target.executor_id,
        model=target.model,
        status="error",
        latency_s=round(dt, 3) if dt > 0 else None,
        enabled_override=0,
        priority_override=None,
        detail=redact(detail),
    )


def _rank_results(results: list[BenchmarkResult], mode: str) -> None:
    if mode != "fastest":
        return
    healthy = [r for r in results if r.status == "healthy"]
    healthy.sort(key=lambda r: r.latency_s if r.latency_s is not None else 999.0)
    for idx, r in enumerate(healthy):
        r.priority_override = idx


def _probe_target(
    target: BenchmarkTarget,
    config: BenchmarkConfig,
    probe: ProbeFn,
    available_fn: Callable[[str], bool] | None = None,
) -> BenchmarkResult:
    if available_fn is not None and not available_fn(target.executor_id):
        return BenchmarkResult(
            executor_id=target.executor_id,
            model=target.model,
            status="error",
            latency_s=None,
            enabled_override=0,
            priority_override=None,
            detail="executor unavailable (binary missing)",
        )
    ok, dt, detail = probe(target.model, config.timeout_s)
    is_retryable = (not ok and not _is_non_retryable_error(detail)) or (
        ok and dt > config.max_latency_s
    )
    if is_retryable:
        ok, dt, detail = probe(target.model, config.timeout_s)
    return _eval_target(target, ok, dt, detail, config.max_latency_s)


def run_benchmark(
    config: BenchmarkConfig,
    store: Store,
    probe_fn: ProbeFn | None = None,
    available_fn: Callable[[str], bool] | None = None,
) -> BenchmarkReport:
    """Execute latency probe for all targets, persist overrides in SQLite, and return report."""
    probe = probe_fn or (lambda m, t: http_probe(m, t, endpoint=config.endpoint))
    with ThreadPoolExecutor(max_workers=max(1, len(config.targets))) as pool:
        results = list(
            pool.map(lambda t: _probe_target(t, config, probe, available_fn), config.targets)
        )
    _rank_results(results, config.selection_mode)

    for r in results:
        store.set_executor_override(
            executor_id=r.executor_id,
            status=r.status,
            enabled_override=r.enabled_override,
            priority_override=r.priority_override,
            latency_s=r.latency_s,
            detail=r.detail,
        )

    healthy = [r for r in results if r.status == "healthy" and r.priority_override is not None]
    healthy.sort(key=lambda r: r.priority_override if r.priority_override is not None else 999)

    promoted_roles: dict[str, str] = {}
    if healthy:
        top_executor = healthy[0].executor_id
        for role in config.promote_roles:
            promoted_roles[role] = top_executor

    return BenchmarkReport(
        timestamp=time.time(),
        results=results,
        promoted_roles=promoted_roles,
    )


def run_client_benchmark(client: Any) -> BenchmarkReport:
    """Run benchmark using a configured AsfClient and its executor availability."""
    return run_benchmark(
        client.config.benchmark,
        client.store,
        available_fn=lambda eid: bool(
            (ex := client.executors.get(eid)) is not None and ex.available()
        ),
    )


def print_benchmark_status(
    overrides: dict[str, Any],
    json_output: bool,
    echo: Callable[[str], None],
) -> None:
    if json_output:
        echo(json.dumps(overrides, ensure_ascii=False))
        return
    if not overrides:
        echo("no benchmark overrides recorded")
        return
    echo("executor_id\tstatus\tlatency\tpriority\toverride")
    for eid, rec in sorted(overrides.items()):
        lat = f"{rec.get('latency_s'):.2f}s" if rec.get("latency_s") is not None else "-"
        prio = str(
            rec.get("priority_override") if rec.get("priority_override") is not None else "-"
        )
        en = "enabled" if rec.get("enabled_override") == 1 else "circuited-out"
        echo(f"{eid}\t{rec.get('status')}\t{lat}\t{prio}\t{en}")


def print_benchmark_report(
    report: BenchmarkReport,
    json_output: bool,
    echo: Callable[[str], None],
) -> None:
    if json_output:
        echo(json.dumps(report.model_dump(), ensure_ascii=False))
        return
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(report.timestamp))
    echo(f"[{ts_str}] Benchmark completed:")
    for r in report.results:
        lat = f"{r.latency_s:.2f}s" if r.latency_s is not None else "-"
        prio = str(r.priority_override if r.priority_override is not None else "-")
        echo(
            f"[{r.status.upper()}] {r.executor_id} ({r.model}): "
            f"latency={lat} priority={prio} detail={r.detail}"
        )
    if report.promoted_roles:
        echo(f"promoted roles: {report.promoted_roles}")
