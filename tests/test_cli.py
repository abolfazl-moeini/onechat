"""Plan 03 CLI tests (typer CliRunner): run/json, executors columns, cooldown, chat /use."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from router_fakes import fake_executors_factory, write_user_config
from typer.testing import CliRunner

import onechat.cli as cli_module
from onechat.sdk import AsfClient

RUNNER = CliRunner()


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsfClient:
    client = AsfClient._for_tests(tmp_path, fake_executors_factory(tmp_path))
    monkeypatch.setattr(cli_module, "AsfClient", lambda **kw: client)
    return client


def test_run_json_output(client: AsfClient) -> None:
    result = RUNNER.invoke(cli_module.app, ["run", "hello", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["text"] == "fake answer a"
    assert payload["executor_id"] == "exec-a"
    assert payload["stop"] == "done"


def test_run_plain_output(client: AsfClient) -> None:
    result = RUNNER.invoke(cli_module.app, ["run", "hello"])
    assert result.exit_code == 0
    assert "fake answer a" in result.stdout


def test_run_accepts_role_and_model_flags(client: AsfClient) -> None:
    result = RUNNER.invoke(
        cli_module.app,
        ["run", "hello", "--role", "reviewer", "--model", "custom-model", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["stop"] == "done"


def test_run_accepts_workdir_flag(client: AsfClient, tmp_path: Path) -> None:
    target = tmp_path / "custom_workdir"
    target.mkdir()
    result = RUNNER.invoke(
        cli_module.app,
        ["run", "hello", "--workdir", str(target), "--json"],
    )
    assert result.exit_code == 0, result.output
    sessions = client.store.list_sessions()
    assert any(s["workdir"] == str(target.resolve()) for s in sessions)


def test_executors_list_columns(client: AsfClient) -> None:
    result = RUNNER.invoke(cli_module.app, ["executors", "list"])
    assert result.exit_code == 0, result.output
    for col in ("exec-a", "cli", "2", "free"):
        assert col in result.output


def test_cooldown_command(client: AsfClient) -> None:
    result = RUNNER.invoke(cli_module.app, ["executors", "cooldown", "exec-a", "120"])
    assert result.exit_code == 0, result.output
    cd = client.store.get_cooldown("exec-a")
    assert cd is not None and cd["until_ts"] > 0
    result = RUNNER.invoke(cli_module.app, ["executors", "clear", "exec-a"])
    assert result.exit_code == 0
    assert client.store.get_cooldown("exec-a") is None


def test_chat_slash_use_switches(client: AsfClient) -> None:
    client.session(workdir=str(Path("/tmp/w")), session_id="chat-test")
    script = "remember 4721\n/use exec-b\nwhat number?\n/status\n/exit\n"
    result = RUNNER.invoke(cli_module.app, ["chat", "--session", "chat-test"], input=script)
    assert result.exit_code == 0, result.output
    prompts = [r.prompt for r in client.executors["exec-b"].requests]  # type: ignore[attr-defined]
    assert any("4721" in p for p in prompts)


def test_doctor_reports(client: AsfClient, tmp_path: Path) -> None:
    result = RUNNER.invoke(cli_module.app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "exec-a" in result.output  # executor table rendered


def test_executors_test_command_runs(client: AsfClient) -> None:
    """`executors test` pins the executor with allow_fallback=False and must not crash."""
    result = RUNNER.invoke(cli_module.app, ["executors", "test", "exec-a"])
    assert result.exit_code == 0, result.output
    assert "exec-a" in result.output


def test_cli_run_emits_progress_to_stderr_and_keeps_stdout_clean(
    client: AsfClient,
) -> None:
    """asf run writes progress/attempt telemetry to stderr while stdout stays clean."""
    result = RUNNER.invoke(cli_module.app, ["run", "hello", "--executor", "exec-a"])
    assert result.exit_code == 0, result.output
    assert "fake answer a" in result.stdout
    assert "[onechat]" in result.stderr
    assert "[onechat]" not in result.stdout


def test_cli_run_emits_tool_events_to_stderr(
    client: AsfClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """onechat run forwards tool events to stderr without polluting stdout."""
    from onechat.contracts import Event, Stop, TurnResult

    class ToolEmitting:
        spec = client.executors["exec-a"].spec

        def available(self) -> bool:
            return True

        async def run(self, req, on_event=None):
            if on_event:
                on_event(Event(kind="tool", text="read_file foo.py"))
            return TurnResult(stop=Stop.DONE, text="done-tool")

    monkeypatch.setitem(client.executors, "exec-a", ToolEmitting())
    result = RUNNER.invoke(cli_module.app, ["run", "hello", "--executor", "exec-a"])
    assert result.exit_code == 0, result.output
    assert "done-tool" in result.stdout
    assert "[onechat] read_file foo.py" in result.stderr
    assert "[onechat]" not in result.stdout


def test_cli_run_quiet_suppresses_stderr(client: AsfClient) -> None:
    """onechat run --quiet suppresses stderr progress logging."""
    result = RUNNER.invoke(cli_module.app, ["run", "hello", "--executor", "exec-a", "--quiet"])
    assert result.exit_code == 0, result.output
    assert "fake answer a" in result.stdout
    assert "[onechat]" not in result.stderr


def test_chat_streams_text_events(client: AsfClient, monkeypatch) -> None:
    """Plan 03 §7: the REPL streams text events as they arrive."""

    class Emitting:
        spec = client.executors["exec-a"].spec

        def available(self) -> bool:
            return True

        async def run(self, req, on_event=None):
            if on_event:
                on_event(Event(kind="text", text="STREAM-CHUNK"))
            return TurnResult(stop=Stop.DONE, text="final")

    from onechat.contracts import Event, Stop, TurnResult

    monkeypatch.setitem(client.executors, "exec-a", Emitting())
    result = RUNNER.invoke(cli_module.app, ["chat"], input="hello\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "STREAM-CHUNK" in result.output  # streamed text is rendered, not swallowed


def test_status_shows_last_executor(client: AsfClient) -> None:
    client.session(workdir=str(Path("/tmp/w")), session_id="st")
    script = "hello\n/status\n/exit\n"
    result = RUNNER.invoke(cli_module.app, ["chat", "--session", "st"], input=script)
    assert result.exit_code == 0, result.output
    assert "exec-a" in result.output  # /status shows the executor that answered


def test_export_markdown_preserves_brackets(client: AsfClient) -> None:
    """T01: markdown export must be verbatim — no Rich markup stripping or errors."""
    sess = client.session(workdir=str(Path("/tmp/w")), session_id="md-export")
    client.store.append_turn(sess.id, 0, "user", "run arr[0] now", executor_id="", stop="done")
    client.store.append_turn(
        sess.id, 1, "assistant", "[bold]raw[/bold] tail", executor_id="exec-a", stop="done"
    )
    result = RUNNER.invoke(cli_module.app, ["sessions", "export", sess.id, "--format", "md"])
    assert result.exit_code == 0, result.output
    assert "[bold]raw[/bold]" in result.output
    assert "arr[0]" in result.output


def test_executors_list_requests_today(client: AsfClient) -> None:
    client.store.record_usage(__import__("time").time(), "exec-a", "s", None, 1, 1, 0.0, "done")
    result = RUNNER.invoke(cli_module.app, ["executors", "list"])
    assert result.exit_code == 0, result.output
    assert "requests today" in result.output


# ---- T12b: REPL hardening, cooldown filtering, config plumbing --------------------------


def test_chat_help_command_does_not_call_llm(client: AsfClient) -> None:
    """T12b: /help prints commands without an LLM turn."""
    result = RUNNER.invoke(cli_module.app, ["chat"], input="/help\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "/tier" in result.output  # the command list
    assert "fake answer" not in result.output


def test_chat_stream_does_not_double_print(client: AsfClient, monkeypatch) -> None:
    """T12b: when chunks streamed, the final reply.text is not printed again."""
    from onechat.contracts import Event, Stop, TurnResult

    class Emitting:
        spec = client.executors["exec-a"].spec

        def available(self) -> bool:
            return True

        async def run(self, req, on_event=None):
            if on_event:
                on_event(Event(kind="text", text="CHUNK-A"))
                on_event(Event(kind="text", text="CHUNK-B"))
            return TurnResult(stop=Stop.DONE, text="CHUNK-ACHUNK-B")

    monkeypatch.setitem(client.executors, "exec-a", Emitting())
    result = RUNNER.invoke(cli_module.app, ["chat"], input="hello\n/exit\n")
    assert result.exit_code == 0, result.output
    assert result.output.count("CHUNK-A") == 1  # streamed once, no final duplicate


def test_executors_list_hides_expired_cooldowns(client: AsfClient) -> None:
    """T12b: an expired cooldown shows '-' instead of a stale clock time."""
    import time as _time

    client.store.set_cooldown("exec-a", until_ts=_time.time() - 10, strikes=1, reason="quota")
    result = RUNNER.invoke(cli_module.app, ["executors", "list"])
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.startswith("exec-a\t")]
    assert lines and "\t-\t" in lines[0]  # the cooldown cell shows '-' once expired


def test_all_subcommands_accept_config_flag(client: AsfClient, tmp_path: Path) -> None:
    """T12b: run/doctor/executors/sessions all accept --config (chat already does)."""
    config_path = write_user_config(tmp_path)
    for argv in (
        ["run", "hi", "--config", str(config_path), "--json"],
        ["doctor", "--config", str(config_path)],
        ["executors", "list", "--config", str(config_path)],
        ["sessions", "list", "--config", str(config_path)],
    ):
        result = RUNNER.invoke(cli_module.app, argv)
        assert result.exit_code == 0, f"{argv[0]} rejected --config: {result.output}"


def test_chat_invalid_tier_slash_command_does_not_crash_repl(client: AsfClient) -> None:
    """Audit: /tier with an unknown tier prints [error] and stays in the REPL."""
    result = RUNNER.invoke(cli_module.app, ["chat"], input="/tier bogus\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "[error]" in result.output


def test_benchmark_cli_status_and_run(client: AsfClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI benchmark status and run commands (T43)."""
    # 1. Status when empty
    res_empty = RUNNER.invoke(cli_module.app, ["benchmark", "--status"])
    assert res_empty.exit_code == 0
    assert "no benchmark overrides recorded" in res_empty.output

    # 2. Status with overrides
    client.store.set_executor_override(
        executor_id="opencode-gemini-hydrogen",
        status="healthy",
        enabled_override=1,
        priority_override=0,
        latency_s=5.2,
    )
    res_status = RUNNER.invoke(cli_module.app, ["benchmark", "--status"])
    assert res_status.exit_code == 0
    assert "opencode-gemini-hydrogen" in res_status.output
    assert "healthy" in res_status.output

    # 3. JSON status
    res_json = RUNNER.invoke(cli_module.app, ["benchmark", "--status", "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["opencode-gemini-hydrogen"]["status"] == "healthy"

    # 4. Run benchmark with mock probe
    import onechat.benchmark as bm_mod

    monkeypatch.setattr(bm_mod, "http_probe", lambda model, timeout_s: (True, 1.5, "OK"))
    res_run = RUNNER.invoke(cli_module.app, ["benchmark", "--json"])
    assert res_run.exit_code == 0
    rep = json.loads(res_run.output)
    assert "results" in rep


def test_executors_clear_all_flag(client: AsfClient) -> None:
    import time

    client.store.set_cooldown("exec-a", time.time() + 100, 1, "quota")
    client.store.set_cooldown("exec-b", time.time() + 100, 1, "error")
    assert len(client.store.all_cooldowns()) == 2
    result = RUNNER.invoke(cli_module.app, ["executors", "clear", "--all"])
    assert result.exit_code == 0
    assert "cleared all cooldowns" in result.output
    assert len(client.store.all_cooldowns()) == 0
