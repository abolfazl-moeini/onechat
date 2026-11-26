"""Tests for benchmark runner and dynamic circuit promotion (T43)."""

from __future__ import annotations

from pathlib import Path

import pytest

from onechat.benchmark import run_benchmark
from onechat.config import BenchmarkConfig, BenchmarkTarget
from onechat.store import Store


@pytest.fixture()
def store(tmp_path: Path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def test_benchmark_promotes_fast_healthy_target(store: Store) -> None:
    cfg = BenchmarkConfig(
        max_latency_s=15.0,
        targets=[
            BenchmarkTarget(
                executor_id="opencode-gemini-hydrogen",
                model="google/gemini-4-hydrogen",
                promoted_priority=0,
            )
        ],
    )

    def fake_probe(model: str, timeout_s: float):
        return True, 3.2, "OK"

    report = run_benchmark(cfg, store, probe_fn=fake_probe)
    assert len(report.results) == 1
    r0 = report.results[0]
    assert r0.status == "healthy"
    assert r0.enabled_override == 1
    assert r0.priority_override == 0
    assert r0.latency_s == 3.2
    assert report.promoted_roles["builder"] == "opencode-gemini-hydrogen"
    assert report.promoted_roles["code"] == "opencode-gemini-hydrogen"

    # Verify SQLite persistence
    overrides = store.get_executor_overrides()
    assert "opencode-gemini-hydrogen" in overrides
    assert overrides["opencode-gemini-hydrogen"]["status"] == "healthy"
    assert overrides["opencode-gemini-hydrogen"]["enabled_override"] == 1


def test_benchmark_circuits_out_slow_target(store: Store) -> None:
    cfg = BenchmarkConfig(
        max_latency_s=15.0,
        targets=[
            BenchmarkTarget(
                executor_id="opencode-gemini-flash",
                model="google/gemini-3.8-flash",
                promoted_priority=0,
            )
        ],
    )

    def fake_probe(model: str, timeout_s: float):
        return True, 22.5, "OK"

    report = run_benchmark(cfg, store, probe_fn=fake_probe)
    assert len(report.results) == 1
    r0 = report.results[0]
    assert r0.status == "slow"
    assert r0.enabled_override == 0
    assert r0.priority_override is None
    assert "latency 22.50s > 15.0s" in r0.detail
    assert report.promoted_roles == {}

    overrides = store.get_executor_overrides()
    assert overrides["opencode-gemini-flash"]["status"] == "slow"
    assert overrides["opencode-gemini-flash"]["enabled_override"] == 0


def test_benchmark_circuits_out_error_target(store: Store) -> None:
    cfg = BenchmarkConfig(
        max_latency_s=15.0,
        targets=[
            BenchmarkTarget(
                executor_id="opencode-gemini-flash",
                model="google/gemini-3.8-flash",
                promoted_priority=0,
            )
        ],
    )

    def fake_probe(model: str, timeout_s: float):
        return False, 1.1, "HTTP 503 Backend overloaded"

    report = run_benchmark(cfg, store, probe_fn=fake_probe)
    r0 = report.results[0]
    assert r0.status == "error"
    assert r0.enabled_override == 0
    assert "503" in r0.detail
    assert report.promoted_roles == {}

    overrides = store.get_executor_overrides()
    assert overrides["opencode-gemini-flash"]["status"] == "error"
    assert overrides["opencode-gemini-flash"]["enabled_override"] == 0


def test_benchmark_selection_mode_fastest(store: Store) -> None:
    cfg = BenchmarkConfig(
        max_latency_s=15.0,
        selection_mode="fastest",
        targets=[
            BenchmarkTarget(
                executor_id="opencode-gemini-hydrogen",
                model="google/gemini-4-hydrogen",
                promoted_priority=0,
            ),
            BenchmarkTarget(
                executor_id="opencode-gemini-flash",
                model="google/gemini-3.8-flash",
                promoted_priority=1,
            ),
        ],
    )

    def fake_probe(model: str, timeout_s: float):
        # flash is faster in this run (2.0s vs 6.0s)
        if "flash" in model:
            return True, 2.0, "OK"
        return True, 6.0, "OK"

    report = run_benchmark(cfg, store, probe_fn=fake_probe)
    assert report.promoted_roles["builder"] == "opencode-gemini-flash"
    overrides = store.get_executor_overrides()
    assert overrides["opencode-gemini-flash"]["priority_override"] == 0
    assert overrides["opencode-gemini-hydrogen"]["priority_override"] == 1


def test_benchmark_non_retryable_quota_error_does_not_retry(store: Store) -> None:
    calls = []

    def fake_probe(model: str, timeout_s: float):
        calls.append(model)
        return False, 0.5, "HTTP 429: You exceeded your current quota"

    cfg = BenchmarkConfig(
        targets=[
            BenchmarkTarget(
                executor_id="opencode-gemini-hydrogen",
                model="gemini-4-hydrogen",
            )
        ]
    )
    report = run_benchmark(cfg, store, probe_fn=fake_probe)
    assert len(calls) == 1  # Exactly 1 call, zero retries on 429
    assert report.results[0].status == "error"
    assert report.results[0].enabled_override == 0


def test_benchmark_retryable_503_retries_once_and_recovers(store: Store) -> None:
    calls = []

    def fake_probe(model: str, timeout_s: float):
        calls.append(model)
        if len(calls) == 1:
            return False, 1.2, "HTTP 503: The model is overloaded"
        return True, 4.5, "OK"

    cfg = BenchmarkConfig(
        max_latency_s=15.0,
        targets=[
            BenchmarkTarget(
                executor_id="opencode-gemini-hydrogen",
                model="gemini-4-hydrogen",
            )
        ],
    )
    report = run_benchmark(cfg, store, probe_fn=fake_probe)
    assert len(calls) == 2  # Retried once on transient 503
    assert report.results[0].status == "healthy"
    assert report.results[0].enabled_override == 1
    assert report.results[0].latency_s == 4.5


def test_benchmark_retryable_slow_retries_once_and_recovers(store: Store) -> None:
    calls = []

    def fake_probe(model: str, timeout_s: float):
        calls.append(model)
        if len(calls) == 1:
            return True, 16.2, "OK"
        return True, 5.1, "OK"

    cfg = BenchmarkConfig(
        max_latency_s=15.0,
        targets=[
            BenchmarkTarget(
                executor_id="opencode-gemini-hydrogen",
                model="gemini-4-hydrogen",
            )
        ],
    )
    report = run_benchmark(cfg, store, probe_fn=fake_probe)
    assert len(calls) == 2  # Retried once on marginal latency
    assert report.results[0].status == "healthy"
    assert report.results[0].latency_s == 5.1


def test_benchmark_parallel_probing_and_configurable_endpoint(store: Store) -> None:
    import time

    cfg = BenchmarkConfig(
        endpoint="http://127.0.0.1:9999",
        max_latency_s=15.0,
        timeout_s=20.0,
        targets=[
            BenchmarkTarget(executor_id="e1", model="m1", promoted_priority=0),
            BenchmarkTarget(executor_id="e2", model="m2", promoted_priority=1),
        ],
    )
    assert cfg.endpoint == "http://127.0.0.1:9999"

    def slow_parallel_probe(model: str, timeout_s: float) -> tuple[bool, float, str]:
        time.sleep(0.15)
        return True, 0.15, "OK"

    t0 = time.monotonic()
    report = run_benchmark(cfg, store, probe_fn=slow_parallel_probe)
    elapsed = time.monotonic() - t0
    assert len(report.results) == 2
    assert elapsed < 0.27  # Parallel execution finishes in ~0.15s, not 0.30s


def test_benchmark_skips_promotion_when_binary_unavailable(store: Store) -> None:
    cfg = BenchmarkConfig(
        promote_roles=["builder"],
        targets=[
            BenchmarkTarget(executor_id="opencode-gemini-flash", model="google/gemini-3.8-flash")
        ],
    )
    report = run_benchmark(
        cfg,
        store,
        probe_fn=lambda m, t: (True, 1.0, "OK"),
        available_fn=lambda eid: False,
    )
    assert report.results[0].status == "error"
    assert report.results[0].enabled_override == 0
    assert report.promoted_roles == {}


def test_benchmark_502_with_numeric_403_in_trace_id_still_retries(store: Store) -> None:
    cfg = BenchmarkConfig(
        targets=[BenchmarkTarget(executor_id="e1", model="m1")],
    )
    calls: list[int] = []

    def trace_probe(model: str, timeout_s: float) -> tuple[bool, float, str]:
        calls.append(1)
        if len(calls) == 1:
            return False, 0.5, "HTTP 502: Bad Gateway - traceId=14032"
        return True, 1.0, "OK"

    report = run_benchmark(cfg, store, probe_fn=trace_probe)
    assert len(calls) == 2
    assert report.results[0].status == "healthy"
