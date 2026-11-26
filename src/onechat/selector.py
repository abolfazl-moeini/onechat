"""Selector (Plan 03 §2 / T10): tier resolution + candidate ordering. Pure functions."""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import BaseModel, Field, field_validator

from .config import TierPolicy
from .contracts import Capability, CostClass, ExecutorSpec, Tier
from .scoring import SkillRoutingConfig, score_candidate

_BENCHMARK_RE = re.compile(r"\b(?:deepswe|automationbench)\b", re.IGNORECASE)


def is_high_effort_benchmark(prompt: str) -> bool:
    """Return True if prompt contains benchmark tokens requiring high thinking effort."""
    return bool(_BENCHMARK_RE.search(prompt))


def clamp_tier(tier: Tier, prompt: str, policy: TierPolicy) -> Tier:
    """Enforce TierPolicy min_tier and max_tier bounds, with benchmark HIGH exception."""
    if is_high_effort_benchmark(prompt):
        return Tier.HIGH
    return Tier(max(policy.min_tier, min(tier, policy.max_tier)))


class Need(BaseModel):
    capability: Capability = Capability.CHAT
    tier: Tier | str = "auto"  # Tier or "auto"
    executor_id: str | None = None
    role: str | None = None
    model: str | None = None
    allow_fallback: bool = True
    avoid_family: str | None = None
    exclude: frozenset[str] = frozenset()
    task_kind: str | None = None
    skills: tuple[str, ...] = ()
    failures: int = Field(default=0, ge=0)  # negative escalation makes no sense (T10)
    timeout_s: int | None = Field(default=None, ge=1)

    @field_validator("skills", mode="before")
    @classmethod
    def _coerce_skills(cls, v: object) -> tuple[str, ...]:
        if isinstance(v, (list, tuple, set)):
            return tuple(str(x).strip().lower() for x in v if str(x).strip())
        if isinstance(v, str):
            return tuple(str(x).strip().lower() for x in v.split(",") if str(x).strip())
        return ()


__all__ = [
    "Need",
    "TierPolicy",
    "resolve_tier",
    "order_candidates",
    "is_high_effort_benchmark",
    "clamp_tier",
]


def resolve_tier(need: Need, prompt: str, policy: TierPolicy) -> Tier:
    """Auto tier: task_kind map -> keywords -> short prompt -> chat_default; escalate, cap MAX."""
    tier: Tier
    lowered = prompt.lower()
    if need.task_kind and need.task_kind in policy.by_task_kind:
        tier = policy.by_task_kind[need.task_kind]
    elif any(k in lowered for k in policy.high_keywords):
        tier = Tier.HIGH
    elif len(prompt) < 200 and "```" not in prompt:
        tier = Tier.LOW
    else:
        tier = policy.chat_default
    if need.failures and policy.escalate_every > 0:
        tier = Tier(min(tier + need.failures // policy.escalate_every, int(Tier.MAX)))
    return clamp_tier(tier, prompt, policy)


def order_candidates(
    need: Need,
    tier: Tier,
    specs: list[ExecutorSpec],
    usable: Callable[[ExecutorSpec], bool],
    policy: TierPolicy,
    roles: dict[str, str] | None = None,
    skill_cfg: SkillRoutingConfig | None = None,
    model_scores: dict[str, dict[str, float]] | None = None,
) -> list[ExecutorSpec]:
    """Keep usable+capable+family/cost/exclude filters, tier-sufficient; cheapest-smart wins.

    A pinned executor_id bypasses the tier/family/exclude filters (explicit user override)
    but must still be enabled, usable, capable and cost-allowed. With allow_fallback=False
    a pinned executor that is NOT usable yields [] — never other backends (T10).
    """
    wanted_capability = need.capability

    def cost_allowed(s: ExecutorSpec) -> bool:
        return s.cost != CostClass.PAID or policy.allow_paid

    if need.executor_id:
        pinned = [
            s
            for s in specs
            if s.id == need.executor_id
            and s.enabled
            and usable(s)
            and wanted_capability in s.capabilities
            and cost_allowed(s)
        ]
        if not need.allow_fallback:
            return pinned
        if pinned:
            rest = [s for s in _filtered(need, tier, specs, usable, policy) if s.id != pinned[0].id]
            sorted_r = _sort_candidates(rest, need, tier, policy, roles, skill_cfg, model_scores)
            return [pinned[0], *[s for s in sorted_r if s.id != pinned[0].id]]
    filtered = _filtered(need, tier, specs, usable, policy)
    return _sort_candidates(filtered, need, tier, policy, roles, skill_cfg, model_scores)


def _sort_candidates(
    cands: list[ExecutorSpec],
    need: Need,
    tier: Tier,
    policy: TierPolicy,
    roles: dict[str, str] | None,
    skill_cfg: SkillRoutingConfig | None,
    model_scores: dict[str, dict[str, float]] | None,
) -> list[ExecutorSpec]:
    if not skill_cfg or not skill_cfg.enabled or not need.skills:
        by_role = _filter_role(_sort_rest(cands, tier, policy), need, roles)
        return _filter_model(by_role, need.model)

    if not need.allow_fallback and need.role and roles and need.role in roles:
        cands = _filter_role(cands, need, roles)

    strong, weak = _split_by_skills(cands, need, tier, policy, roles, skill_cfg, model_scores)
    return _filter_model(strong, need.model, strict=not need.allow_fallback) + _filter_model(
        weak, need.model, strict=not need.allow_fallback
    )


def _score_one(
    s: ExecutorSpec,
    need: Need,
    roles: dict[str, str] | None,
    cfg: SkillRoutingConfig,
    scores: dict[str, dict[str, float]] | None,
):
    target = roles.get(need.role) if roles and need.role else None
    is_role = target is not None and (
        s.id == target or s.profile == target or s.model == target or s.family == target
    )
    return score_candidate(
        s,
        need.skills,
        cfg,
        is_role_match=is_role,
        is_benchmarked=s.priority == 0,
        model_scores=scores,
    )


def _split_by_skills(
    cands: list[ExecutorSpec],
    need: Need,
    tier: Tier,
    policy: TierPolicy,
    roles: dict[str, str] | None,
    skill_cfg: SkillRoutingConfig,
    model_scores: dict[str, dict[str, float]] | None,
) -> tuple[list[ExecutorSpec], list[ExecutorSpec]]:
    scored = [(s, _score_one(s, need, roles, skill_cfg, model_scores)) for s in cands]
    strong = [item for item in scored if not item[1].is_weak]
    weak = [item for item in scored if item[1].is_weak]

    def cost_rank(s: ExecutorSpec) -> int:
        return (
            policy.cost_order.index(s.cost)
            if s.cost in policy.cost_order
            else len(policy.cost_order)
        )

    def sort_key(item: tuple[ExecutorSpec, any]):
        s, cs = item
        return (
            -cs.final_score,
            s.priority if s.priority <= 1 else 999,
            s.tier - tier,
            cost_rank(s),
            s.priority,
            s.id,
        )

    strong_sorted = [s for s, _ in sorted(strong, key=sort_key)]
    weak_sorted = [s for s, _ in sorted(weak, key=sort_key)]
    return strong_sorted, weak_sorted


def _filter_model(
    cands: list[ExecutorSpec], model: str | None, *, strict: bool = False
) -> list[ExecutorSpec]:
    if not model:
        return cands
    m_lower = model.removesuffix(":free").lower()

    def keys(s: ExecutorSpec) -> list[str]:
        out = [s.model.lower(), s.model.removesuffix(":free").lower()]
        if s.model_key:
            out.append(s.model_key.removesuffix(":free").lower())
        return out

    exact = [s for s in cands if m_lower in keys(s)]
    sub = [s for s in cands if s not in exact and any(m_lower in k for k in keys(s))]
    matching = exact + sub
    if strict:
        return matching
    return matching + [s for s in cands if s not in matching] if matching else cands


def _filter_role(
    cands: list[ExecutorSpec], need: Need, roles: dict[str, str] | None
) -> list[ExecutorSpec]:
    if not roles or not need.role or need.role not in roles or need.executor_id:
        return cands
    target = roles[need.role]
    matching = [
        s
        for s in cands
        if s.id == target or s.profile == target or s.model == target or s.family == target
    ]
    if not need.allow_fallback:
        return matching
    return matching + [s for s in cands if s not in matching]


def _filtered(
    need: Need,
    tier: Tier,
    specs: list[ExecutorSpec],
    usable: Callable[[ExecutorSpec], bool],
    policy: TierPolicy,
) -> list[ExecutorSpec]:
    wanted_capability = need.capability
    base = [
        s
        for s in specs
        if s.enabled
        and usable(s)
        and wanted_capability in s.capabilities
        and s.id not in need.exclude
        and (s.cost != CostClass.PAID or policy.allow_paid)
        and s.tier >= tier
    ]
    if need.avoid_family is not None:
        avoided = [s for s in base if s.family != need.avoid_family]
        if avoided:
            return avoided
    return base


def _sort_rest(rest: list[ExecutorSpec], tier: Tier, policy: TierPolicy) -> list[ExecutorSpec]:
    def cost_rank(s: ExecutorSpec) -> int:
        return (
            policy.cost_order.index(s.cost)
            if s.cost in policy.cost_order
            else len(policy.cost_order)
        )

    return sorted(
        rest,
        key=lambda s: (
            s.priority if s.priority <= 1 else 999,
            s.tier - tier,
            cost_rank(s),
            s.priority,
            s.id,
        ),
    )


def coerce_tier(value: Tier | str | None) -> Tier | None:
    """Explicit tier values from CLIs/SDK: Tier passes through, strings parse, 'auto' -> None."""
    if value is None or value == "auto":
        return None
    if isinstance(value, Tier):
        return value
    text = str(value).strip().lower()
    names = {"low": Tier.LOW, "medium": Tier.MEDIUM, "high": Tier.HIGH, "max": Tier.MAX}
    if text in names:
        return names[text]
    if text.isdigit():
        return Tier(int(text))
    raise ValueError(f"unknown tier: {value!r} (use low|medium|high|max|auto or 1-4)")
