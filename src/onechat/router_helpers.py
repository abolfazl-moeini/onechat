"""Router helpers (T12a): request building + the compaction summarizer, extracted so
router.py stays within its 300-line budget. Sync or async git callables are both fine."""

from __future__ import annotations

import inspect
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import RouterConfig
from .contracts import Capability, TurnRequest
from .handoff import api_messages, delta_brief, handoff_brief
from .scoring import detect_skills
from .selector import Need, order_candidates

TURN_TIMEOUT_S = 600
_STALE_MARKERS = ("not found", "no such", "does not exist")
_CODE_PROMPT_RE = re.compile(r"\b(?:review|code|implement|build|bug|test)\b", re.IGNORECASE)


def stale_session(error: str | None) -> bool:
    """True when an ERROR text says the native session is gone (T12a: retry fresh)."""
    if not error:
        return False
    lowered = error.lower()
    return "session" in lowered and any(marker in lowered for marker in _STALE_MARKERS)


async def git_state_of(git: Callable[[Path], Any], workdir: Path) -> str:
    result = git(workdir)
    if inspect.isawaitable(result):
        result = await result
    return str(result or "")


def eligible_specs(need: Need, specs: list[Any], config: RouterConfig) -> list[Any]:
    """Capability/family/cost-filtered specs (ignores availability) — retry_at hints."""
    return [
        s
        for s in specs
        if s.enabled
        and need.capability in s.capabilities
        and (need.avoid_family is None or s.family != need.avoid_family)
        and s.id not in need.exclude
        and (s.cost.value != "paid" or config.tier_policy.allow_paid)
    ]


def bootstrap_prompt(prompt: str, workdir: Path) -> str:
    """Turn 1 bootstrap: orient assistant to workspace and key guideline files."""
    if not workdir.is_dir() or "<workspace-context>" in prompt:
        return prompt
    candidates = ("AGENTS.md", "CLAUDE.md", "SPEC.md", "README.md")
    found = [name for name in candidates if (workdir / name).is_file()]
    if not found:
        return prompt
    files_list = "\n".join(f"- {name}" for name in found)
    header = (
        f"<workspace-context>\n"
        f"Workspace root: {workdir}\n"
        f"Guideline files detected:\n{files_list}\n\n"
        f"Review relevant guideline files (especially AGENTS.md if present) to adhere "
        f"to repository rules and standards.\n"
        f"</workspace-context>\n\n"
    )
    return header + prompt


def _should_bootstrap(need: Need, prompt: str) -> bool:
    return (
        need.capability == Capability.CODE
        or need.role in ("builder", "reviewer", "auditor", "code", "code-review")
        or bool(_CODE_PROMPT_RE.search(prompt))
    )


def build_request(
    session: dict[str, Any],
    spec: Any,
    need: Need,
    prompt: str,
    turns: list[dict[str, Any]],
    git_text: str,
    interrupted: bool,
    config: RouterConfig,
    native: dict[str, Any] | None,
) -> TurnRequest:
    """Route the turn: native resume (+delta) / full handoff / fresh prompt."""
    wd = Path(session["workdir"])
    t_s, i_s = need.timeout_s or config.turn_timeout_s, config.inactivity_timeout_s
    boot = _should_bootstrap(need, prompt)
    if spec.kind == "api":
        act = prompt if turns or not boot else bootstrap_prompt(prompt, wd)
        msgs = api_messages(session, turns, act, config.handoff)
        return TurnRequest(
            prompt=act, workdir=wd, messages=msgs, timeout_s=t_s, inactivity_timeout_s=i_s
        )
    if native is not None and native["native_id"]:
        return _native_request(
            session, spec, need, prompt, turns, git_text, interrupted, config, wd, native
        )

    brief = (
        handoff_brief(session, turns, git_text, interrupted, config.handoff, prompt=prompt)
        if (turns or interrupted)
        else (bootstrap_prompt(prompt, wd) if boot else prompt)
    )
    return TurnRequest(prompt=brief, workdir=wd, timeout_s=t_s, inactivity_timeout_s=i_s)


def _native_request(
    session: dict[str, Any],
    spec: Any,
    need: Need,
    prompt: str,
    turns: list[dict[str, Any]],
    git_text: str,
    interrupted: bool,
    config: RouterConfig,
    workdir: Path,
    native: dict[str, Any],
) -> TurnRequest:
    missed = [t for t in turns if int(t["idx"]) > int(native["last_turn_idx"])]
    timeout_s = need.timeout_s or config.turn_timeout_s
    inactivity_s = config.inactivity_timeout_s
    prefix = ""
    if missed or interrupted:
        prefix = delta_brief(missed, git_text, config.handoff)
        if interrupted and need.capability == Capability.CODE:
            prefix += "The previous assistant was interrupted mid-task. Inspect workspace.\n"
    return TurnRequest(
        prompt=f"{prefix}{prompt}",
        workdir=workdir,
        native_session_id=str(native["native_id"]),
        timeout_s=timeout_s,
        inactivity_timeout_s=inactivity_s,
    )


async def _run_summarize_turn(
    spec: Any, req: Any, executors: dict[str, Any], quota: Any, config: Any, sole: bool
) -> tuple[str | None, Exception | None]:
    try:
        result = await executors[spec.id].run(req)
    except Exception as exc:
        return None, exc
    quota.record(
        spec,
        result,
        "",
        None,
        sole_usable=sole,
        all_specs=list(config.executors),
        profiles=config.profiles,
    )
    if result.stop.value != "done" or not result.text:
        return None, RuntimeError(result.error or result.stop.value)
    return result.text, None


def make_summarizer(
    config: RouterConfig,
    executors: dict[str, Any],
    quota: Any,
    usable: Callable[[Any], bool],
    workdir: Path,
) -> Callable[[str], Any]:
    """CHAT-only summarizer (T12a): records usage, tries fallbacks, raises on all failures."""
    from .contracts import Tier, TurnRequest
    from .selector import Need, order_candidates

    async def summarize(text: str) -> str:
        specs = [e.spec for e in executors.values()]
        cands = order_candidates(
            Need(task_kind="summarize"), Tier.LOW, specs, usable, config.tier_policy
        )
        if not cands:
            return text[: config.handoff.brief_max_chars]
        last_error: Exception | None = None
        req = TurnRequest(
            prompt=f"Summarize this conversation in <=500 words:\n{text}",
            workdir=workdir,
            timeout_s=120,
        )
        for spec in cands[:2]:
            text_out, err = await _run_summarize_turn(
                spec, req, executors, quota, config, len(cands) <= 1
            )
            if text_out is not None:
                return text_out
            last_error = err
        raise last_error or RuntimeError("summarizer failed")

    return summarize


def _update_spec_override(s: Any, rec: dict[str, Any], now: float, ttl_s: float | None) -> Any:
    checked_at = float(rec.get("checked_at") or 0.0)
    if ttl_s is not None and (now - checked_at) > ttl_s:
        return s
    en = bool(rec["enabled_override"]) if rec.get("enabled_override") is not None else s.enabled
    prio = rec["priority_override"] if rec.get("priority_override") is not None else s.priority
    return s.model_copy(update={"enabled": en, "priority": prio})


def apply_executor_overrides(
    specs: list[Any],
    roles: dict[str, str],
    overrides: dict[str, dict[str, Any]],
    promote_roles: list[str],
    benchmark_enabled: bool = True,
    now_ts: float | None = None,
    ttl_s: float | None = None,
    usable: Callable[[Any], bool] | None = None,
) -> tuple[list[Any], dict[str, str]]:
    """Apply dynamic circuit overrides (enabled, priority) and promote top healthy executor."""
    if not benchmark_enabled or not overrides:
        return specs, roles
    now = time.time() if now_ts is None else now_ts
    updated_specs = [
        _update_spec_override(s, overrides[s.id], now, ttl_s) if s.id in overrides else s
        for s in specs
    ]
    effective_roles = dict(roles)
    healthy = [
        s
        for s in updated_specs
        if s.id in overrides
        and overrides[s.id].get("status") == "healthy"
        and s.enabled
        and (usable is None or usable(s))
        and (ttl_s is None or (now - float(overrides[s.id].get("checked_at") or 0.0)) <= ttl_s)
    ]
    if healthy:
        healthy.sort(key=lambda s: s.priority)
        for r in promote_roles:
            effective_roles[r] = healthy[0].id
    return updated_specs, effective_roles


def advance_native(
    store: Any, session_id: str, spec: Any, result: Any, assistant_idx: int, resumed: bool
) -> None:
    if result.native_session_id:
        store.set_native(session_id, spec.id, result.native_session_id, assistant_idx)
        return
    if resumed:
        native = store.get_native(session_id, spec.id)
        if native is not None and native["native_id"]:
            store.set_native(session_id, spec.id, str(native["native_id"]), assistant_idx)


def select_candidates(
    need: Need,
    prompt: str,
    tier: Any,
    specs: list[Any],
    roles: dict[str, str],
    config: RouterConfig,
    usable: Callable[[Any], bool],
    workdir: Path | None = None,
) -> tuple[list[Any], list[Any], dict[str, str]]:
    """Select and order candidates with skill detection and circuit overrides."""
    skills = need.skills
    if not skills and config.skill_routing.auto_detect:
        skills = detect_skills(
            prompt=prompt, task_kind=need.task_kind, workdir=workdir, cfg=config.skill_routing
        )
    eff_need = need if eff_need_same(need, skills) else need.model_copy(update={"skills": skills})
    cands = order_candidates(
        eff_need,
        tier,
        specs,
        usable,
        config.tier_policy,
        roles=roles,
        skill_cfg=config.skill_routing,
        model_scores=config.model_scores,
    )
    eligible = eligible_specs(eff_need, specs, config)
    return cands, eligible, roles


def eff_need_same(need: Need, skills: tuple[str, ...]) -> bool:
    return need.skills == skills
