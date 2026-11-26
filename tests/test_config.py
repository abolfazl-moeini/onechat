"""Plan 02 config tests: merge defaults+user, user profiles dir, tilde env, unknown profile."""

from __future__ import annotations

from pathlib import Path

import pytest
from onechat.config import load_config
from pydantic import ValidationError

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "config"


def test_defaults_load_with_all_six_profiles(tmp_path) -> None:
    cfg = load_config(home=tmp_path)
    names = set(cfg.profiles)
    assert {"claude", "codex", "gemini", "agy", "opencode", "qwen"} <= names


def test_workbuddy_profile_loads(tmp_path) -> None:
    """T03: workbuddy profile loads from package profiles with fullAccess/bypassPermissions."""
    cfg = load_config(home=tmp_path)
    assert "workbuddy" in cfg.profiles
    wb = cfg.profiles["workbuddy"]
    assert wb.name == "workbuddy"
    assert wb.binary == "workbuddy"
    assert wb.parser == "claude_stream_json"
    assert "--permission-mode" in wb.auto
    assert "bypassPermissions" in wb.auto


def test_auto_loads_home_config_yaml(tmp_path) -> None:
    """T03: user_path=None auto-loads ~/.asf/config.yaml (merged over defaults)."""
    home = tmp_path / "home"
    conf = home / ".asf"
    conf.mkdir(parents=True)
    (conf / "config.yaml").write_text(
        "executors:\n"
        "  - id: home-only\n"
        "    kind: cli\n"
        "    profile: codex\n"
        "    model: home-model\n"
        "    tier: 2\n"
        "    capabilities: [chat, code]\n"
        "    cost: free\n"
        "    family: openai\n",
        encoding="utf-8",
    )
    cfg = load_config(None, home=home)
    by_id = {e.id: e for e in cfg.executors}
    assert by_id["home-only"].model == "home-model"


def test_deep_merges_tier_policy(tmp_path) -> None:
    """T03: a partial user tier_policy must not wipe nested by_task_kind defaults."""
    user = tmp_path / "config.yaml"
    user.write_text("tier_policy:\n  by_task_kind:\n    code: 3\n", encoding="utf-8")
    cfg = load_config(user_path=user, home=tmp_path)
    assert cfg.tier_policy.by_task_kind["code"].value == 3  # user override wins
    assert "summarize" in cfg.tier_policy.by_task_kind  # nested defaults survive
    assert cfg.tier_policy.chat_default.value == 2  # untouched scalars keep defaults


def test_tilde_only_expanded_at_start(tmp_path) -> None:
    """T03: only a LEADING ~ expands; mid-string ~ is data, not a home path."""
    home = tmp_path / "home"
    home.mkdir()
    user = tmp_path / "config.yaml"
    user.write_text(
        "executors:\n"
        "  - id: tilde-a\n"
        "    kind: cli\n"
        "    profile: codex\n"
        "    model: m\n"
        "    tier: 3\n"
        "    capabilities: [chat, code]\n"
        "    cost: subscription\n"
        "    family: openai\n"
        "    env: {KEEP: HEAD~1, EXPAND: '~/acc'}\n",
        encoding="utf-8",
    )
    cfg = load_config(user_path=user, home=home)
    spec = next(e for e in cfg.executors if e.id == "tilde-a")
    assert spec.env["KEEP"] == "HEAD~1"
    assert spec.env["EXPAND"] == str(home / "acc")


def test_rejects_duplicate_executor_id_and_unknown_parser(tmp_path) -> None:
    """T03: duplicate ids inside one file and unknown parsers fail at LOAD time."""
    dup = tmp_path / "dup.yaml"
    dup.write_text(
        "executors:\n"
        "  - id: same\n"
        "    kind: cli\n"
        "    profile: codex\n"
        "    model: a\n"
        "    tier: 1\n"
        "    capabilities: [chat, code]\n"
        "    cost: free\n"
        "    family: openai\n"
        "  - id: same\n"
        "    kind: cli\n"
        "    profile: codex\n"
        "    model: b\n"
        "    tier: 1\n"
        "    capabilities: [chat, code]\n"
        "    cost: free\n"
        "    family: openai\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate executor id"):
        load_config(user_path=dup, home=tmp_path)
    home = tmp_path / "home"
    prof_dir = home / ".asf" / "profiles"
    prof_dir.mkdir(parents=True)
    (prof_dir / "bad.yaml").write_text(
        'name: bad\nbinary: bad\nnew: ["{prompt}"]\nresume: ["{prompt}"]\n'
        "auto: []\nprompt_via: argv\nparser: no_such_parser\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown parser"):
        load_config(home=home)


def test_rejects_non_positive_limits_and_empty_cli_capabilities() -> None:
    """T03: Limits must be positive; CLI executors must declare capabilities."""
    from onechat.contracts import Capability, CostClass, ExecutorSpec, Limits, Tier

    with pytest.raises(ValidationError):
        Limits(rpm=0)
    with pytest.raises(ValidationError):
        Limits(rpd=-1)
    base = dict(
        id="x",
        kind="cli",
        profile="codex",
        model="m",
        tier=Tier.LOW,
        cost=CostClass.FREE,
        family="openai",
    )
    with pytest.raises(ValidationError):
        ExecutorSpec(**base, capabilities=frozenset())
    spec = ExecutorSpec(**base, capabilities=frozenset({Capability.CHAT}))
    assert spec.stream_usage is True  # T03: pricing/stream fields exist with defaults


def test_user_config_overrides_defaults_by_id(tmp_path) -> None:
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(
        "executors:\n"
        "  - id: codex-main\n"
        "    kind: cli\n"
        "    profile: codex\n"
        "    model: gpt-5-codex\n"
        "    tier: 3\n"
        "    capabilities: [chat, code]\n"
        "    cost: subscription\n"
        "    family: openai\n"
        "  - id: gemini-cli\n"
        "    kind: cli\n"
        "    profile: gemini\n"
        "    model: gemini-2.5-flash\n"
        "    tier: 2\n"
        "    capabilities: [chat, code]\n"
        "    cost: free\n"
        "    family: google\n",
        encoding="utf-8",
    )
    user = tmp_path / "config.yaml"
    user.write_text(
        "executors:\n"
        "  - id: codex-main\n"
        "    kind: cli\n"
        "    profile: codex\n"
        "    model: gpt-6-codex\n"
        "    tier: 3\n"
        "    capabilities: [chat, code]\n"
        "    cost: subscription\n"
        "    family: openai\n"
        "  - id: extra-one\n"
        "    kind: cli\n"
        "    profile: opencode\n"
        "    model: openrouter/google/gemma-4-31b-it:free\n"
        "    tier: 1\n"
        "    capabilities: [chat, code]\n"
        "    cost: free\n"
        "    family: meta\n",
        encoding="utf-8",
    )
    cfg = load_config(user_path=user, defaults_path=defaults)
    by_id = {e.id: e for e in cfg.executors}
    assert by_id["codex-main"].model == "gpt-6-codex"  # user wins by id
    assert "extra-one" in by_id  # user adds
    assert by_id["gemini-cli"].model == "gemini-2.5-flash"  # default-only entry preserved
    assert len(cfg.executors) == 3


def test_user_profiles_dir_adds_new_cli(tmp_path) -> None:
    home = tmp_path / "home"
    prof_dir = home / ".asf" / "profiles"
    prof_dir.mkdir(parents=True)
    (prof_dir / "newcli.yaml").write_text(
        'name: newcli\nbinary: newcli\nnew: [run, "{prompt}"]\n'
        'resume: [run, "{prompt}"]\nauto: []\nprompt_via: argv\nparser: codex_jsonl\n'
        "quota_patterns: []\nauth_patterns: []\n",
        encoding="utf-8",
    )
    cfg = load_config(home=home)
    assert "newcli" in cfg.profiles


def test_env_tilde_expanded(tmp_path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    user = tmp_path / "config.yaml"
    user.write_text(
        "executors:\n"
        "  - id: codex-acc2\n"
        "    kind: cli\n"
        "    profile: codex\n"
        "    model: m\n"
        "    tier: 3\n"
        "    capabilities: [chat, code]\n"
        "    cost: subscription\n"
        "    family: openai\n"
        "    env: {CODEX_HOME: ~/.asf/accounts/codex-acc2}\n",
        encoding="utf-8",
    )
    cfg = load_config(user_path=user, home=home)
    spec = next(e for e in cfg.executors if e.id == "codex-acc2")
    assert spec.env["CODEX_HOME"] == str(home / ".asf" / "accounts" / "codex-acc2")


def test_unknown_profile_rejected(tmp_path) -> None:
    user = tmp_path / "config.yaml"
    user.write_text(
        "executors:\n"
        "  - id: bad\n"
        "    kind: cli\n"
        "    profile: nosuch\n"
        "    model: m\n"
        "    tier: 1\n"
        "    capabilities: [chat]\n"
        "    cost: free\n"
        "    family: x\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="nosuch"):
        load_config(user_path=user, home=tmp_path)


def test_example_user_config_fixture_loads() -> None:
    cfg = load_config(user_path=FIXTURES / "example_user_config.yaml")
    by_id = {e.id: e for e in cfg.executors}
    assert by_id["groq-fast"].api_key_env == "GROQ_API_KEY"
    assert by_id["codex-acc2"].env["CODEX_HOME"].endswith("codex-acc2")


def test_load_config_reads_turn_timeout_s_from_yaml(tmp_path) -> None:
    """Bug 3: turn_timeout_s in config.yaml must be wired into RouterConfig."""
    user = tmp_path / "config.yaml"
    user.write_text("turn_timeout_s: 123\n", encoding="utf-8")
    cfg = load_config(user_path=user, home=tmp_path)
    assert cfg.turn_timeout_s == 123


def test_benchmark_config_defaults(tmp_path) -> None:
    """Benchmark config loads defaults when not configured."""
    cfg = load_config(home=tmp_path)
    assert hasattr(cfg, "benchmark")
    assert cfg.benchmark.enabled is True
    assert cfg.benchmark.interval_s == 1800
    assert cfg.benchmark.max_latency_s == 15.0
    assert cfg.benchmark.timeout_s == 20.0
    assert "builder" in cfg.benchmark.promote_roles
    assert "code" in cfg.benchmark.promote_roles
    assert cfg.benchmark.selection_mode == "order"


def test_benchmark_config_user_override(tmp_path) -> None:
    """Benchmark config section deep-merges user overrides."""
    user = tmp_path / "config.yaml"
    user.write_text(
        "benchmark:\n"
        "  max_latency_s: 10.0\n"
        "  targets:\n"
        "    - executor_id: opencode-gemini-hydrogen\n"
        "      model: google/gemini-4-hydrogen\n"
        "      promoted_priority: 0\n",
        encoding="utf-8",
    )
    cfg = load_config(user_path=user, home=tmp_path)
    assert cfg.benchmark.max_latency_s == 10.0
    assert cfg.benchmark.interval_s == 1800  # default preserved
    assert len(cfg.benchmark.targets) == 1
    assert cfg.benchmark.targets[0].executor_id == "opencode-gemini-hydrogen"


def test_load_config_reads_skill_routing_and_model_scores_from_yaml(tmp_path) -> None:
    """load_config must wire skill_routing and model_scores from YAML into RouterConfig."""
    user = tmp_path / "config.yaml"
    user.write_text(
        "skill_routing:\n"
        "  weak_threshold: 35.0\n"
        "  aliases:\n"
        "    svelte: frontend\n"
        "  marker_skills:\n"
        "    Gemfile: ruby\n"
        "model_scores:\n"
        "  nemotron-3.5:\n"
        "    js: 25.0\n"
        "    rs: 90.0\n",
        encoding="utf-8",
    )
    cfg = load_config(user_path=user, home=tmp_path)
    assert cfg.skill_routing.weak_threshold == 35.0
    assert cfg.skill_routing.aliases["svelte"] == "frontend"
    assert cfg.skill_routing.marker_skills["Gemfile"] == "ruby"
    assert cfg.model_scores["nemotron-3.5"] == {"js": 25.0, "rs": 90.0}
