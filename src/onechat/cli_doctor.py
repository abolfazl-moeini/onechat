"""asf-router doctor implementation (T12b): extracted so cli.py stays lean."""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .scoring import detect_skills, score_candidate
from .sdk import AsfClient
from .selector import Need, order_candidates


def run_doctor(client: AsfClient, echo: Callable[[str], None]) -> None:
    """CLIs + versions vs docs/compat.md, key envs present, DB writable."""
    _print_cli_versions(client, echo)
    _print_api_keys(client, echo)
    _print_executors(client, echo)
    _print_benchmark_overrides(client, echo)
    echo(f"db: {client.config.db_path} (writable: {_probe_writable(client.config.db_path)})")
    echo(f"compat.md: {_compat_md_path() or 'NOT FOUND'}")


def _print_benchmark_overrides(client: AsfClient, echo: Callable[[str], None]) -> None:
    overrides = client.store.get_executor_overrides()
    if not overrides:
        return
    echo("benchmark circuit overrides")
    for eid, rec in sorted(overrides.items()):
        status = rec.get("status", "unknown")
        latency = f"{rec.get('latency_s'):.2f}s" if rec.get("latency_s") is not None else "?"
        prio = rec.get("priority_override")
        enabled = "enabled" if rec.get("enabled_override") == 1 else "circuited-out"
        echo(f"{eid}: status={status} latency={latency} priority={prio} ({enabled})")


def _print_cli_versions(client: AsfClient, echo: Callable[[str], None]) -> None:
    echo("CLI binaries")
    for profile in client.config.profiles.values():
        path = shutil.which(profile.binary)
        version = ""
        if path:
            try:
                out = subprocess.run(
                    [profile.binary, "--version"], capture_output=True, text=True, timeout=15
                )
                version = (out.stdout or out.stderr).strip().splitlines()[0][:40]
            except Exception:  # noqa: BLE001 - doctor must not crash on one CLI
                version = "?"
        echo(f"{profile.binary}: {'present' if path else 'MISSING'} {version}")


def _print_api_keys(client: AsfClient, echo: Callable[[str], None]) -> None:
    import os

    echo("api key envs")
    for spec in client.config.executors:
        if spec.kind == "api" and spec.api_key_env:
            state = "set" if os.environ.get(spec.api_key_env) else "MISSING"
            echo(f"{spec.api_key_env}: {state}")


def _print_executors(client: AsfClient, echo: Callable[[str], None]) -> None:
    echo("executors")
    for spec in client.config.executors:
        executor = client.executors.get(spec.id)
        cooldown = "yes" if _cooldown_active(client, spec.id) else "no"
        echo(
            f"{spec.id}: kind={spec.kind} tier={int(spec.tier)} "
            f"available={executor.available() if executor else False} cooldown={cooldown}"
        )


def _cooldown_active(client: AsfClient, executor_id: str) -> bool:
    cd = client.store.get_cooldown(executor_id)
    return cd is not None and cd["until_ts"] > time.time()


def _probe_writable(db_path: Path) -> bool:
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        probe = db_path.with_suffix(".doctor-probe")
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _compat_md_path() -> Path | None:
    for candidate in (Path.cwd() / "docs" / "compat.md", Path.cwd() / "compat.md"):
        if candidate.exists():
            return candidate
    return None


def _score_cand(
    s: Any, skills: tuple[str, ...], cfg: Any, roles: dict[str, str], role: str | None, scores: Any
) -> Any:
    target = roles.get(role) if role else None
    is_role = target is not None and (
        s.id == target or s.profile == target or s.model == target or s.family == target
    )
    return score_candidate(
        s, skills, cfg, is_role_match=is_role, is_benchmarked=s.priority == 0, model_scores=scores
    )


def _render_explain_text(
    prompt: str,
    skills: tuple[str, ...],
    tier: Any,
    cands: list[Any],
    roles: dict[str, str],
    role: str | None,
    cfg: Any,
    model_scores: dict[str, dict[str, float]],
    echo: Callable[[str], None],
) -> None:
    echo(f"prompt: {prompt[:60]}...")
    echo(f"detected skills: {', '.join(skills) if skills else '(none)'}")
    echo(f"resolved tier: {tier.name} ({int(tier)})")
    echo("candidate decision table:")
    for idx, s in enumerate(cands, 1):
        cs = _score_cand(s, skills, cfg, roles, role, model_scores)
        st = "WEAK" if cs.is_weak else ("RATED" if cs.is_rated else "UNRATED")
        echo(f"  {idx}. {s.id} [{st}] score={cs.final_score:.1f} model={s.model} prio={s.priority}")


def print_executors_table(client: AsfClient, echo: Callable[[str], None]) -> None:
    echo("id\tkind\ttier\tcost\tavailable\tcooldown\trequests today")
    for spec in client.config.executors:
        executor = client.executors.get(spec.id)
        available = executor.available() if executor else False
        cd = client.store.get_cooldown(spec.id)
        active = cd is not None and cd["until_ts"] > time.time()
        expiry = time.strftime("%H:%M:%S", time.localtime(cd["until_ts"])) if active else ""
        cooldown = f"{expiry} ({cd.get('reason', 'quota')})" if active else "-"
        echo(
            f"{spec.id}\t{spec.kind}\t{int(spec.tier)}\t{spec.cost.value}\t"
            f"{available}\t{cooldown}\t{client.store.usage_count(spec.id, time.time() - 86400)}"
        )


def _explain_skills(
    client: AsfClient, prompt: str, workdir: Path | None, explicit: list[str] | None
) -> tuple[tuple[str, ...], Any]:
    from .scoring import canonicalize_skill

    cfg = client.config.skill_routing
    if explicit:
        sk = tuple(sorted({canonicalize_skill(s, cfg.aliases) for s in explicit if s.strip()}))
        return sk, cfg
    return (detect_skills(prompt=prompt, workdir=workdir, cfg=cfg) if cfg.auto_detect else ()), cfg


def export_session(
    client: AsfClient, session_id: str, fmt: str, echo: Callable[[str], None]
) -> None:
    import json

    turns = client.store.turns(session_id)
    if fmt == "jsonl":
        for turn in turns:
            echo(json.dumps(turn, ensure_ascii=False))
        return
    for turn in turns:
        echo(f"## [{turn['role']}] {turn['executor_id']}\n\n{turn['text']}\n")


def _explain_specs(client: AsfClient) -> tuple[list[Any], dict[str, str]]:
    from .router_helpers import apply_executor_overrides

    bench = getattr(client.config, "benchmark", None)
    ttl = max(2 * getattr(bench, "interval_s", 1800.0), 3600.0) if bench else 3600.0
    return apply_executor_overrides(
        [e.spec for e in client.executors.values()],
        client.config.roles,
        client.store.get_executor_overrides(),
        getattr(bench, "promote_roles", []) if bench else [],
        benchmark_enabled=getattr(bench, "enabled", True) if bench else False,
        ttl_s=ttl,
        usable=client.router._usable,
    )


def _resolve_explain_cands(
    client: AsfClient,
    prompt: str,
    role: str | None,
    tier_name: str,
    workdir: Path | None,
    explicit_skills: list[str] | None = None,
    capability: str = "auto",
) -> tuple[tuple[str, ...], Any, list[Any], dict[str, str], Any]:
    from .contracts import Capability

    skills, cfg = _explain_skills(client, prompt, workdir, explicit_skills)
    auto_cap = "code" if role in ("builder", "code") else "chat"
    cap = Capability((auto_cap if capability == "auto" else capability).lower())
    need = Need(capability=cap, tier=tier_name, role=role, skills=skills)
    tier = client.router._resolve_tier(need, prompt)
    specs, roles = _explain_specs(client)
    cands = order_candidates(
        need,
        tier,
        specs,
        client.router._usable,
        client.config.tier_policy,
        roles=roles,
        skill_cfg=cfg,
        model_scores=client.config.model_scores,
    )
    return skills, tier, cands, roles, cfg


def _render_explain_json(
    prompt: str,
    skills: tuple[str, ...],
    tier: Any,
    cands: list[Any],
    roles: dict[str, str],
    role: str | None,
    cfg: Any,
    scores: Any,
    echo: Callable[[str], None],
) -> None:
    import json

    cand_list = []
    for s in cands:
        cs = _score_cand(s, skills, cfg, roles, role, scores)
        cand_list.append(
            {
                "id": s.id,
                "model": s.model,
                "model_key": s.model_key,
                "tier": int(s.tier),
                "priority": s.priority,
                "score": cs.final_score,
                "weak": cs.is_weak,
                "rated": cs.is_rated,
            }
        )
    echo(
        json.dumps(
            {"prompt": prompt, "skills": list(skills), "tier": int(tier), "candidates": cand_list},
            indent=2,
        )
    )


def explain_routing(
    client: AsfClient,
    prompt: str,
    role: str | None,
    tier_name: str,
    workdir: Path | None,
    json_output: bool,
    echo: Callable[[str], None],
    explicit_skills: list[str] | None = None,
    capability: str = "auto",
) -> None:
    skills, tier, cands, roles, cfg = _resolve_explain_cands(
        client, prompt, role, tier_name, workdir, explicit_skills, capability
    )
    scores = client.config.model_scores
    render = _render_explain_json if json_output else _render_explain_text
    render(prompt, skills, tier, cands, roles, role, cfg, scores, echo)
