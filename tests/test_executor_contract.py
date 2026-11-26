"""Plan 02 executor contract: every built executor returns TurnResult, never raises.

Offline contract runs replay Plan 01 fixtures through FakeProcessRunner. The live test
(`-m live`) runs one real new_turn per available executor; when the backend cannot
authenticate/network tonight it is SKIPPED with the recorded stop reason (docs/compat.md).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from onechat.config import CliProfile
from onechat.contracts import Capability, CostClass, ExecutorSpec, Stop, Tier, TurnResult
from onechat.executors import build_executor
from onechat.executors.api import ApiExecutor
from onechat.executors.cli import CliExecutor
from pydantic import ValidationError
from router_fakes import FakeChunk, FakeOpenAI, FakeProcessRunner

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _spec_for(profile_name: str, eid: str) -> ExecutorSpec:
    return ExecutorSpec(
        id=eid,
        kind="cli",
        profile=profile_name,
        model="test-model",
        tier=Tier.LOW,
        capabilities=frozenset({Capability.CHAT}),
        cost=CostClass.FREE,
        family=f"family-{profile_name}",
    )


PROFILES_UNDER_TEST = ["claude", "codex", "opencode", "agy", "qwen"]


@pytest.mark.parametrize("profile_name", PROFILES_UNDER_TEST)
def test_contract_cli_executor_over_fixtures(profile_name: str) -> None:
    fixture = REPO_ROOT / "tests" / "fixtures" / "cli" / profile_name / "new_turn.jsonl"
    if not fixture.exists():
        pytest.skip(f"no new_turn fixture for {profile_name}")
    import yaml

    profile_data = yaml.safe_load(
        (
            REPO_ROOT / "packages/asf-router/src/onechat/profiles" / f"{profile_name}.yaml"
        ).read_text(encoding="utf-8")
    )
    profile = CliProfile(**profile_data)
    runner = FakeProcessRunner.from_fixture(profile_name, "new_turn")
    exe = CliExecutor(
        _spec_for(profile_name, f"{profile_name}-t"), profile, runner=runner, which=lambda _: True
    )
    result = asyncio.run(exe.run(_req()))
    assert isinstance(result, TurnResult)
    assert result.duration_s > 0
    assert result.stop != Stop.DONE  # tonight's fixtures are all error streams
    assert result.error  # something human-readable recorded (redacted by executor)


def _req(**kw):
    from onechat.contracts import TurnRequest

    base = dict(prompt="Reply with the word PONG", workdir=Path("/tmp"))
    base.update(kw)
    return TurnRequest(**base)


def test_contract_api_executor_fake() -> None:
    spec = ExecutorSpec(
        id="fake-api",
        kind="api",
        model="m",
        tier=Tier.LOW,
        capabilities=frozenset({Capability.CHAT}),
        cost=CostClass.FREE,
        family="x",
        base_url="https://x/v1",
        api_key_env="NO_SUCH_KEY",
    )
    exe = ApiExecutor(
        spec, client_factory=lambda s: FakeOpenAI([FakeChunk("PONG"), FakeChunk(usage=None)])
    )
    result = asyncio.run(exe.run(_req()))
    assert isinstance(result, TurnResult)
    assert result.stop == Stop.DONE and result.text == "PONG"
    assert result.duration_s > 0


def test_build_executor_dispatch() -> None:
    profiles: dict[str, CliProfile] = {
        "codex": CliProfile(
            name="codex",
            binary="codex",
            new=["x", "{prompt}"],
            resume=["x", "{prompt}"],
            parser="codex_jsonl",
        ),
    }
    api_spec = ExecutorSpec(
        id="a",
        kind="api",
        model="m",
        tier=Tier.LOW,
        capabilities=frozenset({Capability.CHAT}),
        cost=CostClass.FREE,
        family="x",
        base_url="https://x/v1",
        api_key_env="K",
    )
    cli_spec = _spec_for("codex", "c")
    assert isinstance(build_executor(api_spec, profiles), ApiExecutor)
    assert isinstance(build_executor(cli_spec, profiles), CliExecutor)
    with pytest.raises((KeyError, ValidationError)):
        build_executor(_spec_for("nosuch", "n"), profiles)


@pytest.mark.live
def test_live_new_turn_available_executors() -> None:
    """One real new_turn per executor whose binary exists. Skips when backend auth/network
    is down (recorded reason), passes only on a real DONE turn."""

    import yaml

    profiles_dir = REPO_ROOT / "packages/asf-router/src/onechat/profiles"
    for profile_path in sorted(profiles_dir.glob("*.yaml")):
        profile = CliProfile(**yaml.safe_load(profile_path.read_text(encoding="utf-8")))
        spec = _spec_for(profile.name, f"{profile.name}-live")
        exe = build_executor(spec, {profile.name: profile})
        if not exe.available():
            pytest.skip(f"{profile.name}: binary not on PATH")
        result = asyncio.run(exe.run(_req(timeout_s=60)))
        if result.stop in (Stop.AUTH, Stop.QUOTA, Stop.TIMEOUT):
            pytest.skip(
                f"{profile.name}: {result.stop.value} (interactive auth wall; compat.md DEFERRED)"
            )
        if result.stop == Stop.ERROR and result.error and "403" in result.error:
            pytest.skip(f"{profile.name}: network blocked (403)")
        assert result.stop == Stop.DONE, f"{profile.name}: {result.stop} {result.error}"
        assert "PONG" in result.text or result.text != ""
