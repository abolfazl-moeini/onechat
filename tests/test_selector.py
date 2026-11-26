"""Plan 03 selector tests: tier resolution + candidate ordering rules."""

from __future__ import annotations

from onechat.contracts import Capability, CostClass, ExecutorSpec, Tier
from onechat.selector import Need, TierPolicy, order_candidates, resolve_tier

POLICY = TierPolicy()


def spec(
    eid: str,
    tier: int,
    cost: str = "free",
    family: str = "f",
    priority: int = 100,
    capabilities: str = "chat,code",
    kind: str = "cli",
    enabled: bool = True,
    model: str = "m",
) -> ExecutorSpec:
    if kind == "api":
        capabilities = "chat"  # contract: api executors cannot claim CODE
    caps = frozenset(Capability(c.strip()) for c in capabilities.split(","))
    return ExecutorSpec(
        id=eid,
        kind=kind,
        profile=eid if kind == "cli" else None,
        model=model,
        tier=Tier(tier),
        capabilities=caps,
        cost=CostClass(cost),
        family=family,
        priority=priority,
        enabled=enabled,
        base_url="https://x/v1" if kind == "api" else None,
        api_key_env="K" if kind == "api" else None,
    )


def usable_all(_: ExecutorSpec) -> bool:
    return True


def test_cheapest_sufficient_tier_first() -> None:
    specs = [spec("hi", 3), spec("lo", 1), spec("mid", 2)]
    ordered = order_candidates(Need(tier=2), Tier.MEDIUM, specs, usable_all, POLICY)
    assert [s.id for s in ordered] == ["mid", "hi"]  # lo filtered (tier < needed)


def test_stronger_tier_is_fallback() -> None:
    specs = [spec("max", 4), spec("mid", 2)]
    ordered = order_candidates(Need(tier=2), Tier.MEDIUM, specs, usable_all, POLICY)
    assert [s.id for s in ordered][:1] == ["mid"]
    assert ordered[1].id == "max"


def test_free_before_subscription_before_paid() -> None:
    specs = [spec("sub", 2, "subscription"), spec("paid", 2, "paid"), spec("free", 2, "free")]
    ordered = order_candidates(
        Need(tier=2), Tier.MEDIUM, specs, usable_all, TierPolicy(allow_paid=True)
    )
    assert [s.id for s in ordered] == ["free", "sub", "paid"]


def test_paid_excluded_unless_allowed() -> None:
    specs = [spec("paid", 2, "paid"), spec("free", 2, "free")]
    ordered = order_candidates(Need(tier=2), Tier.MEDIUM, specs, usable_all, POLICY)
    assert [s.id for s in ordered] == ["free"]


def test_pinned_first() -> None:
    specs = [spec("a", 2), spec("b", 1)]
    ordered = order_candidates(Need(executor_id="b"), Tier.MEDIUM, specs, usable_all, POLICY)
    assert ordered[0].id == "b"


def test_pinned_without_fallback_only() -> None:
    specs = [spec("a", 2), spec("b", 1)]
    need = Need(executor_id="b", allow_fallback=False)
    ordered = order_candidates(need, Tier.MEDIUM, specs, usable_all, POLICY)
    assert [s.id for s in ordered] == ["b"]


def test_avoid_family() -> None:
    specs = [spec("a", 2, family="openai"), spec("b", 2, family="meta")]
    ordered = order_candidates(Need(avoid_family="openai"), Tier.MEDIUM, specs, usable_all, POLICY)
    assert [s.id for s in ordered] == ["b"]


def test_capability_code_filters_api() -> None:
    specs = [spec("api", 2, kind="api"), spec("cli", 2)]
    ordered = order_candidates(
        Need(capability=Capability.CODE), Tier.MEDIUM, specs, usable_all, POLICY
    )
    assert [s.id for s in ordered] == ["cli"]


def test_unusable_skipped() -> None:
    specs = [spec("dead", 1), spec("alive", 1)]
    ordered = order_candidates(Need(tier=1), Tier.LOW, specs, lambda s: s.id != "dead", POLICY)
    assert [s.id for s in ordered] == ["alive"]


def test_auto_by_task_kind() -> None:
    policy = TierPolicy()
    assert resolve_tier(Need(task_kind="research"), "p", policy) == Tier.LOW
    assert resolve_tier(Need(task_kind="spec"), "p", policy) == Tier.HIGH


def test_auto_keywords_high() -> None:
    assert resolve_tier(Need(), "fix the race condition in the scheduler", POLICY) == Tier.HIGH


def test_auto_short_prompt_low() -> None:
    assert resolve_tier(Need(), "hi there", POLICY) == Tier.LOW


def test_auto_default_medium_for_long_prompt() -> None:
    assert resolve_tier(Need(), "x" * 250, POLICY) == Tier.MEDIUM


def test_escalation_capped_at_max() -> None:
    policy = TierPolicy(escalate_every=2)
    assert resolve_tier(Need(failures=3), "hi", policy) == Tier.LOW + 1
    assert resolve_tier(Need(failures=99), "hi", policy) == Tier.MAX


def test_tier_policy_clamps_min_and_max_tier() -> None:
    policy = TierPolicy(min_tier=Tier.MEDIUM, max_tier=Tier.MEDIUM)
    assert resolve_tier(Need(task_kind="research"), "hi", policy) == Tier.MEDIUM
    assert resolve_tier(Need(), "hi there", policy) == Tier.MEDIUM
    assert (
        resolve_tier(Need(), "fix architecture and security race condition", policy) == Tier.MEDIUM
    )
    assert resolve_tier(Need(failures=99), "hi", policy) == Tier.MEDIUM


def test_tier_policy_deepswe_and_automationbench_exceptions() -> None:
    policy = TierPolicy(min_tier=Tier.MEDIUM, max_tier=Tier.MEDIUM)
    assert resolve_tier(Need(), "Run DeepSWE benchmark task 12", policy) == Tier.HIGH
    assert resolve_tier(Need(), "automationbench agent evaluation run", policy) == Tier.HIGH
    assert resolve_tier(Need(failures=99), "DeepSWE solve bug", policy) == Tier.HIGH


# ---- T10: pinned no-fallback semantics, failure clamp, cost-order robustness -----------


def test_pinned_unavailable_with_no_fallback_returns_empty() -> None:
    """T10: pinned + allow_fallback=False + pinned unusable => [] (never other executors)."""
    specs = [spec("a", 1), spec("b", 1)]
    ordered = order_candidates(
        Need(executor_id="pinned-x", allow_fallback=False),
        Tier.MEDIUM,
        specs,
        lambda s: s.id != "pinned-x",
        POLICY,
    )
    assert ordered == []


def test_negative_failures_rejected() -> None:
    """T10: failures < 0 is a programming error — rejected at construction."""
    import pytest as _pytest
    from pydantic import ValidationError

    with _pytest.raises(ValidationError):
        Need(failures=-1)


def test_partial_cost_order_does_not_crash() -> None:
    """T10: a cost class missing from cost_order sorts last, never crashes."""
    policy = TierPolicy(cost_order=[CostClass.FREE])
    specs = [spec("free1", 1, cost="free"), spec("sub1", 1, cost="subscription")]
    ordered = order_candidates(Need(tier=1), Tier.LOW, specs, usable_all, policy)
    assert [s.id for s in ordered] == ["free1", "sub1"]


def test_pinned_executor_remains_first_even_when_model_matches_fallback() -> None:
    from onechat.contracts import Capability, Tier
    from onechat.selector import Need, TierPolicy, order_candidates

    s_pinned = spec("e_pinned", 1, model="model-a", priority=2)
    s_other = spec("e_other", 1, model="model-b", priority=1)
    need = Need(executor_id="e_pinned", model="model-b", capability=Capability.CHAT)
    cands = order_candidates(need, Tier.LOW, [s_pinned, s_other], usable_all, TierPolicy())
    assert [c.id for c in cands] == ["e_pinned", "e_other"]


def test_filter_model_case_insensitive_and_prefers_exact_match() -> None:
    from onechat.contracts import Capability, Tier
    from onechat.selector import Need, TierPolicy, order_candidates

    s_sub = spec("e_sub", 1, model="gemini-4-hydrogen-preview", priority=1)
    s_exact = spec("e_exact", 1, model="gemini-4-hydrogen", priority=2)
    s_other = spec("e_other", 1, model="claude-3-haiku", priority=1)
    need = Need(model="GEMINI-4-HYDROGEN", capability=Capability.CHAT)
    cands = order_candidates(need, Tier.LOW, [s_sub, s_exact, s_other], usable_all, TierPolicy())
    assert cands[0].id == "e_exact"
    assert cands[1].id == "e_sub"
    assert cands[2].id == "e_other"


def test_promoted_tier4_beats_tier3_on_tier3_turn() -> None:
    s_wb = spec("workbuddy", 3, cost="subscription", priority=2)
    s_hy_promoted = spec("opencode-gemini-hydrogen", 4, cost="free", priority=1)
    ordered = order_candidates(Need(), Tier.HIGH, [s_wb, s_hy_promoted], usable_all, POLICY)
    assert ordered[0].id == "opencode-gemini-hydrogen"

    # Unpromoted tier-4 still stays behind tier-3
    s_hy_unpromoted = spec("opencode-gemini-hydrogen", 4, cost="free", priority=19)
    ordered2 = order_candidates(Need(), Tier.HIGH, [s_wb, s_hy_unpromoted], usable_all, POLICY)
    assert ordered2[0].id == "workbuddy"


def test_benchmark_regex_requires_word_boundaries() -> None:
    """Benchmark check uses word boundary regex to avoid false positives."""
    from onechat.selector import is_high_effort_benchmark

    assert is_high_effort_benchmark("run deepswe eval") is True
    assert is_high_effort_benchmark("AutomationBench run") is True
    assert is_high_effort_benchmark("mydeepswehelper function") is False
    assert is_high_effort_benchmark("automationbenchmark") is False


def test_clamp_tier_enforces_min_and_max_on_explicit_tier() -> None:
    """clamp_tier clamps explicit tiers against policy min_tier and max_tier."""
    from onechat.selector import clamp_tier

    policy = TierPolicy(min_tier=Tier.MEDIUM, max_tier=Tier.MEDIUM)
    assert clamp_tier(Tier.LOW, "hello", policy) == Tier.MEDIUM
    assert clamp_tier(Tier.HIGH, "hello", policy) == Tier.MEDIUM
    assert clamp_tier(Tier.MEDIUM, "hello", policy) == Tier.MEDIUM
    assert clamp_tier(Tier.LOW, "solve DeepSWE bug", policy) == Tier.HIGH
