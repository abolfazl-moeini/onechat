"""Plan 02 CliExecutor tests: argv building, env merge, classification, event streaming."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from onechat.config import CliProfile
from onechat.contracts import Capability, CostClass, ExecutorSpec, Stop, Tier
from onechat.executors.cli import CliExecutor
from router_fakes import FakeProcessRunner

CODEX = CliProfile(
    name="codex",
    binary="codex",
    new=["exec", "--json", "-m", "{model}", "{auto}", "{prompt}"],
    resume=["exec", "resume", "{session_id}", "--json", "-m", "{model}", "{auto}", "{prompt}"],
    auto=["--sandbox", "workspace-write"],
    prompt_via="argv",
    parser="codex_jsonl",
    quota_patterns=["usage limit", "429"],
    auth_patterns=["not logged in", "401"],
)
STDIN_PROFILE = CliProfile(
    name="stdincli",
    binary="stdincli",
    new=["run", "--model", "{model}"],
    resume=["run", "--model", "{model}"],
    auto=["--yolo"],
    prompt_via="stdin",
    parser="codex_jsonl",
    quota_patterns=[],
    auth_patterns=[],
)


def spec(**kw) -> ExecutorSpec:
    base = dict(
        id="codex-main",
        kind="cli",
        profile="codex",
        model="gpt-5-codex",
        tier=Tier.MEDIUM,
        capabilities=frozenset({Capability.CHAT, Capability.CODE}),
        cost=CostClass.SUBSCRIPTION,
        family="openai",
    )
    base.update(kw)
    return ExecutorSpec(**base)


def executor(
    runner: FakeProcessRunner, sp: ExecutorSpec | None = None, profile: CliProfile = CODEX
) -> CliExecutor:
    return CliExecutor(sp or spec(), profile, runner=runner, which=lambda _: True)


def test_available_requires_binary_on_path() -> None:
    exe = CliExecutor(spec(), CODEX, runner=FakeProcessRunner(), which=lambda _: None)
    assert exe.available() is False


def test_builds_new_argv() -> None:
    runner = FakeProcessRunner(lines=["not json"], exit_code=1)
    req = _req()
    import asyncio

    asyncio.run(executor(runner).run(req))
    argv = runner.calls[0]["argv"]
    assert argv[0] == "codex"  # binary prepended by run()
    assert argv[1:3] == ["exec", "--json"]
    assert "gpt-5-codex" in argv and "Reply with the word PONG" in argv
    assert "--sandbox" in argv and "workspace-write" in argv


def _req(**kw):
    from onechat.contracts import TurnRequest

    base = dict(prompt="Reply with the word PONG", workdir=Path("/tmp/asf-spike"))
    base.update(kw)
    return TurnRequest(**base)


def test_builds_resume_argv() -> None:
    runner = FakeProcessRunner(lines=["not json"], exit_code=1)
    import asyncio

    asyncio.run(executor(runner).run(_req(native_session_id="thr-7")))
    argv = runner.calls[0]["argv"]
    assert "resume" in argv and "thr-7" in argv


def test_prompt_via_stdin() -> None:
    runner = FakeProcessRunner(lines=["not json"], exit_code=1)
    import asyncio

    asyncio.run(executor(runner, spec(profile="stdincli"), STDIN_PROFILE).run(_req()))
    call = runner.calls[0]
    assert "Reply with the word PONG" not in call["argv"]
    assert call["stdin"] == "Reply with the word PONG"


def test_auto_flags_expanded() -> None:
    runner = FakeProcessRunner(lines=["not json"], exit_code=1)
    import asyncio

    asyncio.run(executor(runner).run(_req()))
    argv = runner.calls[0]["argv"]
    assert "--sandbox" in argv and "workspace-write" in argv and "{auto}" not in argv


def test_env_merged_with_account_dir() -> None:
    runner = FakeProcessRunner(lines=["not json"], exit_code=1)
    sp = spec(env={"CODEX_HOME": "/tmp/acc2"})
    import asyncio

    asyncio.run(executor(runner, sp).run(_req()))
    assert runner.calls[0]["env"]["CODEX_HOME"] == "/tmp/acc2"
    assert "PATH" in runner.calls[0]["env"]  # os.environ merged in


def test_quota_classified_with_retry_after() -> None:
    runner = FakeProcessRunner(
        lines=[json.dumps({"type": "error", "message": "usage limit reached, try again in 5m"})],
        exit_code=1,
    )
    import asyncio

    result = asyncio.run(executor(runner).run(_req()))
    assert result.stop == Stop.QUOTA
    assert result.retry_after_s == 300


def test_auth_classified() -> None:
    runner = FakeProcessRunner(
        lines=[json.dumps({"type": "error", "message": "401 forbidden"})], exit_code=1
    )
    import asyncio

    result = asyncio.run(executor(runner).run(_req()))
    assert result.stop == Stop.AUTH


def test_timeout_classified() -> None:
    runner = FakeProcessRunner(lines=[], exit_code=0, timeout=True)
    import asyncio

    result = asyncio.run(executor(runner).run(_req()))
    assert result.stop == Stop.TIMEOUT


def test_empty_text_is_error() -> None:
    runner = FakeProcessRunner(lines=[], exit_code=0)
    import asyncio

    result = asyncio.run(executor(runner).run(_req()))
    assert result.stop == Stop.ERROR


def test_success_is_done() -> None:
    runner = FakeProcessRunner(
        lines=[
            json.dumps({"type": "thread.started", "thread_id": "thr-1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "i", "type": "agent_message", "text": "PONG"},
                }
            ),
            json.dumps(
                {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 1}}
            ),
        ],
        exit_code=0,
    )
    import asyncio

    result = asyncio.run(executor(runner).run(_req()))
    assert result.stop == Stop.DONE
    assert result.text == "PONG"
    assert result.native_session_id == "thr-1"
    assert result.usage.output_tokens == 1


def test_events_streamed_from_fixture() -> None:
    """Replay the real claude fixture (assistant text present); events stream incrementally."""
    profile = CliProfile(
        name="claude",
        binary="claude",
        new=["x", "{prompt}"],
        resume=["x", "{prompt}"],
        auto=[],
        prompt_via="argv",
        parser="claude_stream_json",
        quota_patterns=[],
        auth_patterns=["Failed to authenticate", "authentication_failed"],
    )
    runner = FakeProcessRunner.from_fixture("claude", "new_turn")
    seen: list[str] = []
    import asyncio

    result = asyncio.run(
        executor(runner, spec(id="claude-x"), profile).run(
            _req(), on_event=lambda e: seen.append(e.kind)
        )
    )
    assert result.stop == Stop.AUTH
    assert "text" in seen


def test_effort_flags_match_binary_name_and_respect_template_override() -> None:
    """_effort_flags matches binary basename, and build_argv checks template."""
    from onechat.contracts import Capability, CostClass, ExecutorSpec, Tier, TurnRequest
    from onechat.executors.cli import _effort_flags, build_argv

    prof_acc2 = CliProfile(
        name="codex-work",
        binary="/usr/local/bin/codex",
        new=["codex", "{auto}", "-p", "{prompt}"],
        resume=["codex", "{auto}", "resume", "{prompt}"],
        auto=[],
        prompt_via="argv",
        parser="codex_json",
        quota_patterns=[],
        auth_patterns=[],
    )
    assert _effort_flags(prof_acc2, "hello") == ["-c", 'model_reasoning_effort="medium"']
    assert _effort_flags(prof_acc2, "run deepswe benchmark") == [
        "-c",
        'model_reasoning_effort="high"',
    ]

    # Template already has effort override
    prof_custom = CliProfile(
        name="codex-work",
        binary="codex",
        new=["codex", "{auto}", "-c", 'model_reasoning_effort="low"', "-p", "{prompt}"],
        resume=["codex", "{auto}", "resume", "{prompt}"],
        auto=[],
        prompt_via="argv",
        parser="codex_json",
        quota_patterns=[],
        auth_patterns=[],
    )
    spec = ExecutorSpec(
        id="c1",
        kind="cli",
        profile="codex-work",
        family="codex",
        model="m",
        tier=Tier.MEDIUM,
        capabilities=frozenset({Capability.CHAT}),
        cost=CostClass.FREE,
    )
    req = TurnRequest(prompt="hello", session_id="s1", workdir=Path("/tmp"))
    argv, _ = build_argv(prof_custom, spec, req)
    # Must NOT have injected duplicate medium effort flag
    assert argv.count("-c") == 1
    assert 'model_reasoning_effort="low"' in argv
    assert 'model_reasoning_effort="medium"' not in argv


def test_cli_fixture_classification_per_profile() -> None:
    """Contract across the real error fixtures: never raises, always a TurnResult."""
    import asyncio

    for profile_name, parser, _cases in (
        ("claude", "claude_stream_json", ["new_turn"]),
        ("opencode", "opencode_json", ["new_turn"]),
        ("agy", "agy_stream_json", ["new_turn"]),
        ("qwen", "qwen_stream_json", ["new_turn"]),
    ):
        profile = CliProfile(
            name=profile_name,
            binary=profile_name,
            new=["x", "{prompt}"],
            resume=["x", "{prompt}"],
            auto=[],
            prompt_via="argv",
            parser=parser,
            quota_patterns=[],
            auth_patterns=["401", "auth", "login", "eligibility"],
        )
        runner = FakeProcessRunner.from_fixture(profile_name, "new_turn")
        result = asyncio.run(executor(runner, spec(id=f"{profile_name}-x"), profile).run(_req()))
        assert result.stop in (Stop.AUTH, Stop.QUOTA, Stop.ERROR)
        assert result.duration_s >= 0


@pytest.mark.parametrize("case", ["new_turn", "resume_turn", "edit_turn", "bad_model"])
def test_fixture_absent_is_naturally_skipped(case: str) -> None:
    from router_fakes import FIXTURE_ROOT

    if not (FIXTURE_ROOT / "codex" / f"{case}.jsonl").exists():
        pytest.skip(f"fixture {case} absent (live capture deferred)")


def test_process_runner_inactivity_timeout_kills_silent_child(tmp_path: Path) -> None:
    """AsyncProcessRunner aborts a subprocess that emits 0 bytes within inactivity_timeout_s."""
    import asyncio
    import sys

    from onechat.executors.process import AsyncProcessRunner

    runner = AsyncProcessRunner()

    # Script that sleeps silently without producing any stdout
    argv = [sys.executable, "-c", "import time; time.sleep(10)"]
    lines: list[str] = []
    res = asyncio.run(
        runner.run(
            argv,
            cwd=tmp_path,
            env=None,
            stdin=None,
            timeout_s=10,
            on_line=lines.append,
            inactivity_timeout_s=1,
        )
    )
    assert res.timed_out is True
    assert "inactivity" in res.stderr_tail.lower() or res.exit_code != 0


def test_process_runner_inactivity_timeout_resets_on_output(tmp_path: Path) -> None:
    """AsyncProcessRunner resets inactivity timer each time stdout or stderr is produced."""
    import asyncio
    import sys

    from onechat.executors.process import AsyncProcessRunner

    runner = AsyncProcessRunner()

    # Script produces output every 0.3s for 3 iterations (total ~0.9s > 0.5s inactivity timeout)
    script = (
        "import time, sys\n"
        "for i in range(3):\n"
        "    time.sleep(0.3)\n"
        "    print(f'tick {i}', flush=True)\n"
    )
    argv = [sys.executable, "-c", script]
    lines: list[str] = []
    res = asyncio.run(
        runner.run(
            argv,
            cwd=tmp_path,
            env=None,
            stdin=None,
            timeout_s=5,
            on_line=lines.append,
            inactivity_timeout_s=1,
        )
    )
    assert res.timed_out is False
    assert len(lines) == 3
    assert res.exit_code == 0


# ---- T05: real AsyncProcessRunner streaming contracts (local processes, no network) ----


async def test_handles_stdout_line_over_64kib(tmp_path: Path) -> None:
    """T05: chunked reads — one >64 KiB JSON line must arrive whole (no ValueError/deadlock)."""

    from onechat.executors.process import AsyncProcessRunner

    runner = AsyncProcessRunner()
    result = await runner.run(
        ["python3", "-c", 'import json; print(json.dumps({"x": "a" * 100000}))'],
        cwd=tmp_path,
        env=None,
        stdin=None,
        timeout_s=30,
        on_line=lambda _line: None,
    )
    assert result.exit_code == 0, result.stderr_tail
    assert len(result.stdout_lines) == 1
    assert json.loads(result.stdout_lines[0])["x"] == "a" * 100000


async def test_stdin_write_within_timeout_and_broken_pipe_safe(tmp_path: Path) -> None:
    """T05: child exits before reading stdin — feed must fail soft, exit code carried."""

    from onechat.executors.process import AsyncProcessRunner

    runner = AsyncProcessRunner()
    result = await runner.run(
        ["python3", "-c", "import sys; sys.exit(1)"],
        cwd=tmp_path,
        env=None,
        stdin="x" * 200_000,
        timeout_s=30,
        on_line=lambda _line: None,
    )
    assert result.exit_code == 1


async def test_cancelled_error_terminates_child_group(tmp_path: Path) -> None:
    """T05: cancelling run() must kill the whole process group (bash trap proves it)."""
    import asyncio
    import time as _time

    from onechat.executors.process import AsyncProcessRunner

    marker = tmp_path / "alive"
    script = f"trap 'rm -f {marker}' TERM; touch {marker}; sleep 30"
    runner = AsyncProcessRunner()
    task = asyncio.create_task(
        runner.run(
            ["bash", "-c", script],
            cwd=tmp_path,
            env=None,
            stdin=None,
            timeout_s=60,
            on_line=lambda _line: None,
        )
    )
    await asyncio.sleep(0.5)
    assert marker.exists()  # child started
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    deadline = _time.monotonic() + 6.0  # SIGTERM grace (5s) + SIGKILL slack
    while _time.monotonic() < deadline and marker.exists():
        await asyncio.sleep(0.1)
    assert not marker.exists(), "child group survived cancellation"


# ---- T07: classification order, boundary safety, linear sink, argv hardening ----------


def test_succeeded_turn_with_401_in_stderr_not_classified_as_auth() -> None:
    """T07: exit 0 + parsed text + no parsed.error => DONE even if stderr mentions 401."""
    runner = FakeProcessRunner(
        lines=[
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "i", "type": "agent_message", "text": "OK"},
                }
            )
        ],
        exit_code=0,
        stderr_tail="server on port 4010, login ok",
    )
    result = asyncio.run(executor(runner).run(_req()))
    assert result.stop == Stop.DONE
    assert result.text == "OK"


def test_tool_only_declined_codex_turn_is_done() -> None:
    """T07: a declined-only codex turn (tool event, no text) is a healthy DONE."""
    runner = FakeProcessRunner(
        lines=[
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "i1",
                        "type": "command_execution",
                        "command": "rm -rf /",
                        "status": "declined",
                    },
                }
            )
        ],
        exit_code=0,
    )
    result = asyncio.run(executor(runner).run(_req()))
    assert result.stop == Stop.DONE


def test_parse_exception_converted_to_error_value() -> None:
    """T07: any exception while parsing/classifying becomes Stop.ERROR (Rule 8)."""
    broken = CliProfile(
        name="broken",
        binary="broken",
        new=["{prompt}"],
        resume=["{prompt}"],
        auto=[],
        prompt_via="argv",
        parser="no_such_parser",
        quota_patterns=[],
        auth_patterns=[],
    )
    runner = FakeProcessRunner(lines=["whatever"], exit_code=0)
    result = asyncio.run(CliExecutor(spec(id="broken-x"), broken, runner=runner).run(_req()))
    assert result.stop == Stop.ERROR
    assert result.error  # honest detail, redacted


def test_build_argv_no_placeholder_cascade(monkeypatch) -> None:
    """T07: single-pass expansion — a {model} value containing '{prompt}' stays data."""
    runner = FakeProcessRunner(lines=["not json"], exit_code=1)
    sp = spec(model="gpt-{prompt}-x")
    asyncio.run(executor(runner, sp).run(_req(prompt="hello")))
    argv = runner.calls[0]["argv"]
    assert "gpt-{prompt}-x" in argv  # single pass: no cascaded re-expansion
    assert "gpt-hello-x" not in argv


def test_build_argv_prompt_starting_with_dash_gets_separator() -> None:
    """T07: a prompt that looks like a flag cannot be parsed as one (insert '--')."""
    runner = FakeProcessRunner(lines=["not json"], exit_code=1)
    plain = CliProfile(
        name="plain",
        binary="plain",
        new=["run", "{prompt}"],
        resume=["run", "{prompt}"],
        auto=[],
        prompt_via="argv",
        parser="codex_jsonl",
        quota_patterns=[],
        auth_patterns=[],
    )
    asyncio.run(executor(runner, spec(id="plain-x"), plain).run(_req(prompt="-Weird")))
    argv = runner.calls[0]["argv"]
    assert "--" in argv and argv.index("--") == argv.index("-Weird") - 1


def test_line_sink_is_linear(monkeypatch) -> None:
    """T07: N lines feed the parser N times — never O(N^2) re-parsing of the whole stream."""
    import onechat.executors.cli as cli_mod
    import onechat.executors.parsers as parsers_mod
    from onechat.executors.parsers import StreamParser as RealStreamParser

    total_lines = {"n": 0}
    real_parse = parsers_mod.parse

    def counting_parse(name, lines):
        total_lines["n"] += len(lines)
        return real_parse(name, lines)

    class CountingSP(RealStreamParser):
        def feed(self, line):
            total_lines["n"] += 1
            return super().feed(line)

    monkeypatch.setattr(cli_mod, "parse", counting_parse, raising=False)
    monkeypatch.setattr(cli_mod, "StreamParser", CountingSP, raising=False)
    lines = [json.dumps({"type": "agent_message", "text": f"m{i}"}) for i in range(1000)]
    runner = FakeProcessRunner(lines=lines, exit_code=0)
    asyncio.run(executor(runner).run(_req()))
    assert total_lines["n"] == 1000


def test_timeout_redacts_env_secret_and_omits_none_string(monkeypatch) -> None:
    """Audit: timeout error uses self._redact (masks spec.env) and never stringifies 'None'."""
    from onechat.executors.process import ProcResult

    class TimeoutRunner:
        async def run(self, argv, cwd, env, stdin, timeout_s, on_line):
            return ProcResult(
                exit_code=-1,
                stdout_lines=[],
                stderr_tail="leaked supersecretvalue123",
                timed_out=True,
                duration_s=1.0,
            )

    sp = spec(env={"MY_SECRET_KEY": "supersecretvalue123"})
    monkeypatch.setenv("MY_SECRET_KEY", "supersecretvalue123")
    res = asyncio.run(executor(TimeoutRunner(), sp).run(_req()))
    assert res.stop == Stop.TIMEOUT
    assert "supersecretvalue123" not in (res.error or "")
    assert not (res.error or "").startswith("None")


def test_terminate_group_ignores_process_lookup_error() -> None:
    """Audit: _terminate_group must not raise ProcessLookupError if child already exited."""
    from onechat.executors.process import AsyncProcessRunner

    class DeadProc:
        pid = 999999999

        def terminate(self) -> None:
            raise ProcessLookupError

        def kill(self) -> None:
            raise ProcessLookupError

        async def wait(self) -> int:
            return 0

    asyncio.run(AsyncProcessRunner()._terminate_group(DeadProc()))  # type: ignore[arg-type]


def test_cli_executor_redacts_spec_env_without_os_environ() -> None:
    """Bug 1: spec.env secret values must be masked even when NOT set in os.environ."""
    secret = "inline-secret-value-999"
    runner = FakeProcessRunner(lines=[], stderr_tail=f"auth failed: {secret}", exit_code=1)
    sp = spec(env={"CUSTOM_TOKEN": secret})
    res = asyncio.run(executor(runner, sp).run(_req()))
    assert res.stop == Stop.ERROR
    assert secret not in (res.error or "")


def test_cli_executor_isolates_on_event_exception() -> None:
    """Bug 2: callback errors in on_event must not abort stdout pumping or fail the turn."""
    lines = [
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "hello"}})
    ]
    runner = FakeProcessRunner(lines=lines, exit_code=0)

    def boom(_ev) -> None:
        raise RuntimeError("ui crash")

    res = asyncio.run(executor(runner).run(_req(), on_event=boom))
    assert res.stop == Stop.DONE
    assert res.text == "hello"


def test_build_argv_large_non_flag_prompt_no_spurious_double_dash() -> None:
    """Bug 2: large prompts not starting with '-' must NOT get a spurious '--'."""
    from onechat.executors.cli import build_argv

    plain = CliProfile(
        name="plain",
        binary="plain",
        new=["run", "{prompt}"],
        resume=["run", "{prompt}"],
        auto=[],
        prompt_via="argv",
        parser="codex_jsonl",
        quota_patterns=[],
        auth_patterns=[],
    )
    argv, _ = build_argv(plain, spec(id="plain-x"), _req(prompt="x" * 40000))
    assert "--" not in argv


def test_universal_quota_and_unavailable_patterns() -> None:
    profile = CliProfile(
        name="testcli",
        binary="testcli",
        new=["run"],
        resume=["resume"],
        parser="codex_jsonl",
        quota_patterns=[],
        auth_patterns=[],
        unavailable_patterns=["connection refused"],
    )
    # Number "4290" is not status 429
    runner = FakeProcessRunner(lines=[], exit_code=1, stderr_tail="process pid 4290 terminated")
    res = asyncio.run(executor(runner, profile=profile).run(_req()))
    assert res.stop == Stop.ERROR

    # "limit reached" triggers Stop.QUOTA
    runner_quota = FakeProcessRunner(
        lines=[], exit_code=1, stderr_tail="error: limit reached, wait 10s"
    )
    res_quota = asyncio.run(executor(runner_quota, profile=profile).run(_req()))
    assert res_quota.stop == Stop.QUOTA

    # "HTTP 429" triggers Stop.QUOTA
    runner_429 = FakeProcessRunner(lines=[], exit_code=1, stderr_tail="HTTP 429 Too Many Requests")
    res_429 = asyncio.run(executor(runner_429, profile=profile).run(_req()))
    assert res_429.stop == Stop.QUOTA


def test_cli_effort_flags() -> None:
    from onechat.executors.cli import _effort_flags

    assert _effort_flags(CODEX, "Write a quick script") == [
        "-c",
        'model_reasoning_effort="medium"',
    ]
    assert _effort_flags(CODEX, "Solve DeepSWE issue 402") == [
        "-c",
        'model_reasoning_effort="high"',
    ]
    assert _effort_flags(CODEX, "AutomationBench browser agent test") == [
        "-c",
        'model_reasoning_effort="high"',
    ]
