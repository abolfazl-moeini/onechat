"""Config loading (Plan 02 §3-4): defaults.yaml + user overrides, CLI profile registry.

User config resolution (T03): explicit user_path wins; else $ASF_CONFIG; else
~/.asf/config.yaml when present. Nested sections (tier_policy/handoff/cooldown) deep-merge;
scalar keys: user wins; executors merge by id.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from .contracts import Capability, CostClass, ExecutorSpec, Tier
from .scoring import SkillRoutingConfig

_PACKAGE_DIR = Path(__file__).resolve().parent


class CliProfile(BaseModel):
    """How to drive one CLI family headlessly (template rules in Plan 02 §4)."""

    name: str
    binary: str
    new: list[str]
    resume: list[str]
    auto: list[str] = Field(default_factory=list)
    prompt_via: Literal["argv", "stdin"] = "argv"
    parser: str
    quota_patterns: list[str] = Field(default_factory=list)
    auth_patterns: list[str] = Field(default_factory=list)
    unavailable_patterns: list[str] = Field(default_factory=list)
    share_cooldown: bool = False


class HandoffConfig(BaseModel):
    recent_turns: int = 6
    turn_max_chars: int = 4000
    brief_max_chars: int = 24000
    compact_threshold_chars: int = 30000


class TierPolicy(BaseModel):
    by_task_kind: dict[str, Tier] = Field(
        default_factory=lambda: {
            "summarize": Tier.LOW,
            "research": Tier.LOW,
            "digest": Tier.MEDIUM,
            "fix": Tier.MEDIUM,
            "test": Tier.MEDIUM,
            "code": Tier.MEDIUM,
            "plan": Tier.HIGH,
            "spec": Tier.HIGH,
            "review": Tier.HIGH,
        }
    )
    chat_default: Tier = Tier.MEDIUM
    high_keywords: list[str] = Field(
        default_factory=lambda: [
            "architecture",
            "design",
            "security",
            "race condition",
            "migration",
            "root cause",
        ]
    )
    escalate_every: int = 2
    cost_order: list[CostClass] = Field(
        default_factory=lambda: [CostClass.FREE, CostClass.SUBSCRIPTION, CostClass.PAID]
    )
    allow_paid: bool = False
    min_tier: Tier = Tier.LOW
    max_tier: Tier = Tier.MAX


class CooldownPolicy(BaseModel):
    """Cooldown policy: immediate multi-hour failure caching + optional K1 rate gate."""

    min_requests: int = Field(default=1, ge=1)
    failure_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    window_s: int = Field(default=60, ge=1)
    quota_cooldown_s: int = Field(default=7200, ge=0)
    timeout_cooldown_s: int = Field(default=7200, ge=0)
    error_cooldown_s: int = Field(default=14400, ge=0)
    auth_cooldown_s: int = Field(default=86400, ge=0)
    backoff_strikes: list[int] = Field(default_factory=lambda: [7200, 14400, 28800, 86400])
    share_profile_cooldown: bool = True


class BenchmarkTarget(BaseModel):
    executor_id: str
    model: str
    promoted_priority: int = 0


class BenchmarkConfig(BaseModel):
    enabled: bool = True
    endpoint: str = "http://127.0.0.1:8789"
    interval_s: int = 1800
    max_latency_s: float = 15.0
    timeout_s: float = 20.0
    promote_roles: list[str] = Field(default_factory=lambda: ["builder", "code"])
    selection_mode: Literal["order", "fastest"] = "order"
    targets: list[BenchmarkTarget] = Field(default_factory=list)


class RouterConfig(BaseModel):
    db_path: Path
    executors: list[ExecutorSpec] = Field(default_factory=list)
    profiles: dict[str, CliProfile] = Field(default_factory=dict)
    handoff: HandoffConfig = Field(default_factory=HandoffConfig)
    tier_policy: TierPolicy = Field(default_factory=TierPolicy)
    cooldown: CooldownPolicy = Field(default_factory=CooldownPolicy)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    skill_routing: SkillRoutingConfig = Field(default_factory=SkillRoutingConfig)
    model_scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    roles: dict[str, str] = Field(
        default_factory=lambda: {
            "builder": "codex",
            "reviewer": "agy",
            "auditor": "workbuddy",
        }
    )
    turn_timeout_s: int = Field(default=600, ge=1)
    inactivity_timeout_s: int = Field(default=180, ge=0)

    @model_validator(mode="after")
    def _profiles_must_exist(self) -> RouterConfig:
        for spec in self.executors:
            if spec.kind == "cli" and spec.profile not in self.profiles:
                raise ValueError(
                    f"executor {spec.id!r} references unknown profile {spec.profile!r}"
                )
        return self


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """Nested dicts merge recursively; scalars and lists: override wins (T03)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_package_profiles() -> dict[str, CliProfile]:
    profiles: dict[str, CliProfile] = {}
    for path in sorted((_PACKAGE_DIR / "profiles").glob("*.yaml")):
        profile = CliProfile(**_read_yaml(path))
        profiles[profile.name] = profile
    return profiles


def _validate_parsers(profiles: dict[str, CliProfile]) -> None:
    from .executors.parsers import PARSERS  # deferred: executors import config back

    for profile in profiles.values():
        if profile.parser not in PARSERS:
            raise ValueError(
                f"profile {profile.name!r}: unknown parser {profile.parser!r} "
                f"(known: {sorted(PARSERS)})"
            )


def _merge_executors(defaults: list[ExecutorSpec], user: list[ExecutorSpec]) -> list[ExecutorSpec]:
    user_by_id = {spec.id: spec for spec in user}
    merged = [user_by_id.get(spec.id, spec) for spec in defaults]
    seen = {spec.id for spec in defaults}
    merged += [spec for spec in user if spec.id not in seen]
    return merged


def _reject_duplicate_ids(specs: list[ExecutorSpec], source: Path) -> None:
    seen: set[str] = set()
    for spec in specs:
        if spec.id in seen:
            raise ValueError(f"duplicate executor id {spec.id!r} in {source}")
        seen.add(spec.id)


def _expand_tilde(value: str, home: Path) -> str:
    """Expand only a LEADING ~ (T03): 'HEAD~1' is data, '~/acc' is a home path."""
    if value == "~":
        return str(home)
    if value.startswith("~/"):
        return str(home) + value[1:]
    return value


def _expand_env(spec: ExecutorSpec, home: Path) -> ExecutorSpec:
    data = spec.model_dump()
    data["env"] = {k: _expand_tilde(v, home) for k, v in spec.env.items()}
    data["capabilities"] = sorted(c.value for c in spec.capabilities)
    return ExecutorSpec(**data)


def _build_specs(raw: list[dict], home: Path) -> list[ExecutorSpec]:
    specs = [
        ExecutorSpec(
            **{
                **spec,
                "tier": Tier(spec["tier"]),
                "cost": CostClass(spec["cost"]),
                "capabilities": frozenset(Capability(c) for c in spec.get("capabilities", [])),
            }
        )
        for spec in raw
    ]
    return [_expand_env(spec, home) for spec in specs]


def _user_config_path(user_path: Path | None, home: Path) -> Path | None:
    """Explicit path wins; else $ONECHAT_CONFIG; else $ASF_CONFIG; else config.yaml."""
    if user_path is not None:
        return user_path
    for env_var in ("ONECHAT_CONFIG", "ASF_CONFIG"):
        env = os.environ.get(env_var)
        if env:
            return Path(env)
    for folder in (".onechat", ".asf"):
        default = home / folder / "config.yaml"
        if default.exists():
            return default
    return None


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    import re

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        match = re.match(r"^([A-Za-z_0-9]+)=(.*)$", line)
        if match:
            k, v = match.group(1), match.group(2).strip("\"'")
            if k not in os.environ:
                os.environ[k] = v


def _load_all_profiles(home: Path) -> dict[str, CliProfile]:
    profiles = _load_package_profiles()
    for folder in (".asf", ".onechat"):
        for path in sorted((home / folder / "profiles").glob("*.yaml")):
            profile = CliProfile(**_read_yaml(path))
            profiles[profile.name] = profile
    _validate_parsers(profiles)
    return profiles


def load_config(
    user_path: Path | None = None, home: Path | None = None, defaults_path: Path | None = None
) -> RouterConfig:
    """Merge package defaults.yaml with user config; user executors win by id."""
    home = home or Path.home()
    for folder in (".onechat", ".asf"):
        _load_env_file(home / folder / "env")
    defaults_file = defaults_path or (_PACKAGE_DIR / "defaults.yaml")
    defaults_data = _read_yaml(defaults_file)
    user_file = _user_config_path(user_path, home)
    user_data: dict = _read_yaml(user_file) if user_file is not None else {}
    data = _deep_merge(defaults_data, user_data)
    if (
        "db_path" not in user_data
        and (home / ".asf" / "asf.db").exists()
        and not (home / ".onechat" / "onechat.db").exists()
    ):
        data["db_path"] = "~/.asf/asf.db"
    data["db_path"] = _expand_tilde(str(data.get("db_path", "~/.onechat/onechat.db")), home)
    default_specs = _build_specs(defaults_data.get("executors", []), home)
    _reject_duplicate_ids(default_specs, defaults_file)
    if "executors" in user_data:  # Plan 02 §3: executors merge by id, not replace
        user_specs = _build_specs(user_data["executors"], home)
        _reject_duplicate_ids(user_specs, user_file or Path("<auto>"))
        executors = _merge_executors(default_specs, user_specs)
    else:
        executors = default_specs
    return RouterConfig(
        db_path=Path(data["db_path"]),
        executors=executors,
        profiles=_load_all_profiles(home),
        handoff=HandoffConfig(**(data.get("handoff") or {})),
        tier_policy=TierPolicy(**(data.get("tier_policy") or {})),
        cooldown=CooldownPolicy(**(data.get("cooldown") or {})),
        benchmark=BenchmarkConfig(**(data.get("benchmark") or {})),
        skill_routing=SkillRoutingConfig(**(data.get("skill_routing") or {})),
        model_scores=dict(data.get("model_scores") or {}),
        roles=dict(
            data.get("roles") or {"builder": "codex", "reviewer": "agy", "auditor": "workbuddy"}
        ),
        turn_timeout_s=int(data.get("turn_timeout_s", 600)),
        inactivity_timeout_s=int(data.get("inactivity_timeout_s", 180)),
    )
