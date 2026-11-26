"""Tests for desktop profiles, {workdir} placeholder expansion, and CLI environment variables."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml
from router_fakes import FakeProcessRunner

from onechat.config import CliProfile
from onechat.contracts import Capability, CostClass, ExecutorSpec, Stop, Tier, TurnRequest
from onechat.executors.cli import CliExecutor, build_argv


def _spec_for(profile: CliProfile) -> ExecutorSpec:
    return ExecutorSpec(
        id=f"{profile.name}-spec",
        kind="cli",
        profile=profile.name,
        model="test-model",
        tier=Tier.MEDIUM,
        capabilities=frozenset({Capability.CHAT, Capability.CODE}),
        cost=CostClass.FREE,
        family="testfam",
    )


@pytest.mark.parametrize(
    "profile_name",
    [
        "autoclaw",
        "qoder",
        "zcode",
        "workbuddy",
        "agy",
        "codex",
        "claude",
        "gemini",
        "opencode",
        "cline",
    ],
)
def test_all_profiles_expand_without_unbound_placeholders(
    tmp_path: Path, profile_name: str
) -> None:
    profiles_dir = Path(__file__).parent.parent / "src" / "onechat" / "profiles"
    prof_file = profiles_dir / f"{profile_name}.yaml"
    if not prof_file.exists():
        pytest.skip(f"Profile {profile_name} not found")

    data = yaml.safe_load(prof_file.read_text(encoding="utf-8"))
    profile = CliProfile(**data)
    spec = _spec_for(profile)
    req = TurnRequest(prompt="hello world", workdir=tmp_path, native_session_id="thread-xyz")

    argv, stdin = build_argv(profile, spec, req)
    assert argv is not None
    # Verify no unresolved curly braces remain in any argument
    for arg in argv:
        assert "{" not in arg and "}" not in arg, f"Unbound placeholder in {arg} for {profile_name}"


def test_workdir_placeholder_expansion_in_desktop_profiles(tmp_path: Path) -> None:
    profiles_dir = Path(__file__).parent.parent / "src" / "onechat" / "profiles"
    for name in ("autoclaw", "qoder", "zcode"):
        data = yaml.safe_load((profiles_dir / f"{name}.yaml").read_text(encoding="utf-8"))
        profile = CliProfile(**data)
        spec = _spec_for(profile)
        req = TurnRequest(prompt="test prompt", workdir=tmp_path)

        argv, _ = build_argv(profile, spec, req)
        idx = argv.index("-w")
        assert idx != -1
        assert argv[idx + 1] == str(tmp_path.resolve())


def test_workbuddy_argv_includes_workdir(tmp_path: Path) -> None:
    profiles_dir = Path(__file__).parent.parent / "src" / "onechat" / "profiles"
    import yaml

    data = yaml.safe_load((profiles_dir / "workbuddy.yaml").read_text(encoding="utf-8"))
    profile = CliProfile(**data)
    spec = _spec_for(profile)
    req = TurnRequest(prompt="test prompt", workdir=tmp_path)

    argv, _ = build_argv(profile, spec, req)
    idx = argv.index("--workdir")
    assert idx != -1
    assert argv[idx + 1] == str(tmp_path.resolve())


def test_cli_executor_injects_workdir_and_session_env(tmp_path: Path) -> None:
    runner = FakeProcessRunner(lines=[], exit_code=0)
    profile = CliProfile(
        name="testcli",
        binary="testcli",
        new=["run", "{prompt}"],
        resume=["resume", "{prompt}"],
        parser="codex_jsonl",
        quota_patterns=[],
        auth_patterns=[],
    )
    spec = _spec_for(profile)
    exe = CliExecutor(spec, profile, runner=runner, which=lambda _: True)

    req = TurnRequest(prompt="echo hi", workdir=tmp_path, native_session_id="native-12345")
    asyncio.run(exe.run(req))

    assert len(runner.calls) == 1
    call_env = runner.calls[0]["env"]
    assert call_env.get("ONECHAT_WORKDIR") == str(tmp_path.resolve())
    assert call_env.get("ONECHAT_SESSION_ID") == "native-12345"


def test_nonexistent_workdir_fails_fast(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does_not_exist_subfolder"
    runner = FakeProcessRunner(lines=[], exit_code=0)
    profile = CliProfile(
        name="testcli",
        binary="testcli",
        new=["run"],
        resume=["resume"],
        parser="codex_jsonl",
        quota_patterns=[],
        auth_patterns=[],
    )
    spec = _spec_for(profile)
    exe = CliExecutor(spec, profile, runner=runner, which=lambda _: True)

    req = TurnRequest(prompt="hi", workdir=nonexistent)
    result = asyncio.run(exe.run(req))
    assert result.stop == Stop.ERROR
    assert "Workdir does not exist" in (result.error or "")
    assert len(runner.calls) == 0  # Subprocess was never invoked!


def test_workdir_with_spaces_and_symlink_resolved(tmp_path: Path) -> None:
    import yaml

    real_dir = tmp_path / "my project with spaces"
    real_dir.mkdir()
    link_dir = tmp_path / "link_proj"
    link_dir.symlink_to(real_dir)

    profiles_dir = Path(__file__).parent.parent / "src" / "onechat" / "profiles"
    data = yaml.safe_load((profiles_dir / "qoder.yaml").read_text(encoding="utf-8"))
    profile = CliProfile(**data)
    spec = _spec_for(profile)

    req = TurnRequest(prompt="test", workdir=link_dir, native_session_id="q-1")
    argv, _ = build_argv(profile, spec, req)
    w_idx = argv.index("-w")
    assert argv[w_idx + 1] == str(real_dir.resolve())


@pytest.mark.parametrize("driver_name", ["qoder", "autoclaw", "zcode"])
def test_3_turn_desktop_simulation_argv_and_session(tmp_path: Path, driver_name: str) -> None:
    import json

    import yaml

    profiles_dir = Path(__file__).parent.parent / "src" / "onechat" / "profiles"
    data = yaml.safe_load((profiles_dir / f"{driver_name}.yaml").read_text(encoding="utf-8"))
    profile = CliProfile(**data)
    spec = _spec_for(profile)

    turn1_out = json.dumps(
        {"type": "result", "session_id": f"{driver_name}-sess-42", "result": "Turn 1 ok"}
    )
    turn2_out = json.dumps(
        {"type": "result", "session_id": f"{driver_name}-sess-42", "result": "Turn 2 ok"}
    )
    runner = FakeProcessRunner(lines=[turn1_out], exit_code=0)
    exe = CliExecutor(spec, profile, runner=runner, which=lambda _: True)

    res1 = asyncio.run(exe.run(TurnRequest(prompt="p1", workdir=tmp_path)))
    assert res1.stop == Stop.DONE
    assert res1.native_session_id == f"{driver_name}-sess-42"
    assert "--session" not in runner.calls[0]["argv"]
    assert str(tmp_path.resolve()) in runner.calls[0]["argv"]

    runner2 = FakeProcessRunner(lines=[turn2_out], exit_code=0)
    exe2 = CliExecutor(spec, profile, runner=runner2, which=lambda _: True)
    res2 = asyncio.run(
        exe2.run(
            TurnRequest(prompt="p2", workdir=tmp_path, native_session_id=res1.native_session_id)
        )
    )
    assert res2.stop == Stop.DONE
    assert res2.native_session_id == f"{driver_name}-sess-42"
    assert "--session" in runner2.calls[0]["argv"]
    assert f"{driver_name}-sess-42" in runner2.calls[0]["argv"]
    assert str(tmp_path.resolve()) in runner2.calls[0]["argv"]
