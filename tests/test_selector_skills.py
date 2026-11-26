from __future__ import annotations

from onechat.config import TierPolicy
from onechat.contracts import Capability, CostClass, ExecutorSpec, Tier
from onechat.scoring import SkillRoutingConfig
from onechat.selector import Need, order_candidates


def _make_spec(
    id: str,
    model: str,
    scores: dict[str, float] | None = None,
    model_key: str | None = None,
    tier: Tier = Tier.MEDIUM,
    priority: int = 100,
) -> ExecutorSpec:
    return ExecutorSpec(
        id=id,
        kind="cli",
        profile="mock",
        capabilities=frozenset({Capability.CHAT, Capability.CODE}),
        tier=tier,
        cost=CostClass.FREE,
        family="mock",
        model=model,
        model_key=model_key,
        scores=scores or {},
        priority=priority,
    )


def test_need_skills_tuple_coercion():
    need = Need(skills=["js", "react"])  # type: ignore[arg-type]
    assert need.skills == ("js", "react")


def test_order_candidates_skill_prioritization():
    s_weak_js = _make_spec("m_weak", "weak-model", scores={"js": 20.0, "rs": 90.0})
    s_strong_js = _make_spec("m_strong", "strong-model", scores={"js": 90.0, "rs": 40.0})
    s_neutral = _make_spec("m_neutral", "neutral-model", scores={})

    specs = [s_weak_js, s_neutral, s_strong_js]
    policy = TierPolicy()
    skill_cfg = SkillRoutingConfig(weak_threshold=40.0)

    # For JS task: s_strong_js should be first, s_weak_js demoted to bottom
    need_js = Need(capability=Capability.CODE, skills=("js",))
    ordered = order_candidates(
        need_js,
        Tier.MEDIUM,
        specs,
        usable=lambda _: True,
        policy=policy,
        skill_cfg=skill_cfg,
    )
    assert [s.id for s in ordered] == ["m_strong", "m_neutral", "m_weak"]

    # For Rust task: s_weak_js is strong in Rust (90), s_strong_js is neutral/borderline (40)
    need_rs = Need(capability=Capability.CODE, skills=("rs",))
    ordered_rs = order_candidates(
        need_rs,
        Tier.MEDIUM,
        specs,
        usable=lambda _: True,
        policy=policy,
        skill_cfg=skill_cfg,
    )
    assert ordered_rs[0].id == "m_weak"


def test_order_candidates_preserves_pinned_executor():
    s_weak = _make_spec("m_weak", "weak-model", scores={"js": 10.0})
    s_strong = _make_spec("m_strong", "strong-model", scores={"js": 90.0})
    specs = [s_weak, s_strong]

    policy = TierPolicy()
    skill_cfg = SkillRoutingConfig()
    need = Need(
        capability=Capability.CODE,
        executor_id="m_weak",
        skills=("js",),
        allow_fallback=True,
    )
    ordered = order_candidates(
        need,
        Tier.MEDIUM,
        specs,
        usable=lambda _: True,
        policy=policy,
        skill_cfg=skill_cfg,
    )
    assert ordered[0].id == "m_weak"


def test_order_candidates_model_key_fallback():
    s_openrouter = _make_spec(
        "ex_or",
        "openrouter/nvidia/nemotron-3:free",
        model_key="nemotron",
    )
    s_nvidia = _make_spec(
        "ex_nv",
        "nvidia/nemotron-3",
        model_key="nemotron",
    )
    specs = [s_openrouter, s_nvidia]
    policy = TierPolicy()
    skill_cfg = SkillRoutingConfig()
    model_scores = {"nemotron": {"rs": 85.0}}

    need = Need(capability=Capability.CODE, skills=("rs",))
    ordered = order_candidates(
        need,
        Tier.MEDIUM,
        specs,
        usable=lambda _: True,
        policy=policy,
        skill_cfg=skill_cfg,
        model_scores=model_scores,
    )
    assert len(ordered) == 2
    assert ordered[0].id in ("ex_or", "ex_nv")


def test_filter_model_matches_model_key():
    s = _make_spec("ex1", "openrouter/nvidia/nemotron-3:free", model_key="nemotron")
    need = Need(capability=Capability.CODE, model="nemotron")
    policy = TierPolicy()
    ordered = order_candidates(
        need,
        Tier.MEDIUM,
        [s],
        usable=lambda _: True,
        policy=policy,
    )
    assert len(ordered) == 1
    assert ordered[0].id == "ex1"


def test_weak_candidate_not_promoted_by_role_fallback():
    s_weak = _make_spec("m_weak", "weak-model", scores={"js": 20.0})
    s_strong = _make_spec("m_strong", "strong-model", scores={"js": 90.0})
    specs = [s_weak, s_strong]

    policy = TierPolicy()
    skill_cfg = SkillRoutingConfig(weak_threshold=40.0)
    roles = {"builder": "m_weak"}

    # With allow_fallback=True (default), weak model must stay below strong model
    need = Need(capability=Capability.CODE, role="builder", skills=("js",), allow_fallback=True)
    ordered = order_candidates(
        need,
        Tier.MEDIUM,
        specs,
        usable=lambda _: True,
        policy=policy,
        roles=roles,
        skill_cfg=skill_cfg,
    )
    assert [s.id for s in ordered] == ["m_strong", "m_weak"]

    # With allow_fallback=False, strictly enforce the role target
    need_strict = Need(
        capability=Capability.CODE, role="builder", skills=("js",), allow_fallback=False
    )
    ordered_strict = order_candidates(
        need_strict,
        Tier.MEDIUM,
        specs,
        usable=lambda _: True,
        policy=policy,
        roles=roles,
        skill_cfg=skill_cfg,
    )
    assert [s.id for s in ordered_strict] == ["m_weak"]


def test_skill_score_beats_priority_zero_or_one_in_strong_band():
    s_prio1 = _make_spec("ex_prio1", "model-a", scores={"rs": 45.0}, priority=1)
    s_high = _make_spec("ex_high", "model-b", scores={"rs": 95.0}, priority=10)
    s_unrated_p1 = _make_spec("ex_unrated_p1", "model-c", scores={}, priority=1)
    s_unrated_p5 = _make_spec("ex_unrated_p5", "model-d", scores={}, priority=5)

    policy = TierPolicy()
    skill_cfg = SkillRoutingConfig(weak_threshold=40.0)

    # Higher skill score (95) must beat lower skill score (45) even when lower has priority=1
    need_rs = Need(capability=Capability.CODE, skills=("rs",))
    ordered = order_candidates(
        need_rs, Tier.MEDIUM, [s_prio1, s_high], lambda _: True, policy, skill_cfg=skill_cfg
    )
    assert [s.id for s in ordered] == ["ex_high", "ex_prio1"]

    # When scores tie (both unrated at 50.0), priority <= 1 still wins tie-breaker
    need_py = Need(capability=Capability.CODE, skills=("py",))
    ordered_tie = order_candidates(
        need_py,
        Tier.MEDIUM,
        [s_unrated_p5, s_unrated_p1],
        lambda _: True,
        policy,
        skill_cfg=skill_cfg,
    )
    assert [s.id for s in ordered_tie] == ["ex_unrated_p1", "ex_unrated_p5"]


def test_need_model_strict_when_allow_fallback_false():
    s_match = _make_spec("ex1", "openrouter/nvidia/nemotron:free", model_key="nemotron")
    s_other = _make_spec("ex2", "other-model")
    need = Need(capability=Capability.CODE, model="nemotron", allow_fallback=False, skills=("rs",))
    ordered = order_candidates(
        need,
        Tier.MEDIUM,
        [s_other, s_match],
        usable=lambda _: True,
        policy=TierPolicy(),
        skill_cfg=SkillRoutingConfig(),
    )
    assert [s.id for s in ordered] == ["ex1"]
