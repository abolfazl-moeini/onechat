"""onechat CLI: run, sessions, executors, doctor, chat."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import typer

from .cli_repl import SLASH_COMMANDS, handle_slash, run_chat_turn
from .contracts import Event, Tier
from .sdk import AsfClient

app = typer.Typer(
    help="onechat: one chat session surviving AI backend switches", no_args_is_help=True
)
executors_app = typer.Typer(help="executor pool management", no_args_is_help=True)
sessions_app = typer.Typer(help="session inspection", no_args_is_help=True)
app.add_typer(executors_app, name="executors")
app.add_typer(sessions_app, name="sessions")

_RUN_WORKDIR_OPT = typer.Option(None, "--workdir", help="working directory")
_CHAT_WORKDIR_OPT = typer.Option(Path("."), "--workdir")
_CONFIG_OPT = typer.Option(None, "--config")
_SKILL_OPT = typer.Option(None, "--skill", help="required skill(s)")


def _client(config_path: Path | None = None) -> AsfClient:
    return AsfClient(config_path=config_path)


def _run_progress_logger(quiet: bool) -> tuple[Callable[[Event], None], Callable[[], None]]:
    if quiet:
        return lambda _e: None, lambda: None
    start = time.monotonic()
    stop_ev = threading.Event()

    def on_event(ev: Event) -> None:
        if ev.kind in ("status", "tool"):
            sys.stderr.write(f"[onechat] {ev.text}\n")
            sys.stderr.flush()

    def heartbeat() -> None:
        while not stop_ev.wait(15.0):
            elapsed = int(time.monotonic() - start)
            sys.stderr.write(f"[onechat] working... ({elapsed}s elapsed)\n")
            sys.stderr.flush()

    th = threading.Thread(target=heartbeat, daemon=True)
    th.start()

    def stop() -> None:
        stop_ev.set()
        th.join(timeout=0.5)

    return on_event, stop


@app.command()
def run(
    prompt: str,
    session: str | None = typer.Option(None, "--session", help="session id"),
    new_session: bool = typer.Option(False, "--new-session", "--fresh", help="force new session"),
    tier: str = typer.Option("auto", "--tier"),
    executor: str | None = typer.Option(None, "--executor"),
    role: str | None = typer.Option(None, "--role"),
    model: str | None = typer.Option(None, "--model"),
    skill: list[str] | None = _SKILL_OPT,
    json_output: bool = typer.Option(False, "--json"),
    workdir: Path | None = _RUN_WORKDIR_OPT,
    config_path: Path | None = _CONFIG_OPT,
    timeout: int | None = typer.Option(None, "--timeout", help="timeout in seconds"),
    no_fallback: bool = typer.Option(False, "--no-fallback", help="disable fallback"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="suppress stderr progress"),
) -> None:
    """One turn in a session (auto-continues latest for workdir); prints reply or JSON."""
    client = _client(config_path)
    target = str(workdir.resolve()) if workdir is not None else str(Path.cwd().resolve())
    on_event, stop_heartbeat = _run_progress_logger(quiet)
    try:
        sess = client.session(
            workdir=target,
            session_id=session,
            continue_latest=not new_session and session is None,
        )
        if not quiet and not json_output:
            sys.stderr.write(f"[onechat] session: {sess.id} (workdir: {target})\n")
            sys.stderr.flush()
        reply = sess.ask_sync(
            prompt,
            tier=tier,
            executor=executor,
            role=role,
            model=model,
            skills=skill,
            allow_fallback=not no_fallback,
            timeout_s=timeout,
            on_event=on_event,
        )
    finally:
        stop_heartbeat()
    typer.echo(json.dumps(reply.model_dump(), ensure_ascii=False) if json_output else reply.text)


@sessions_app.command("list")
def sessions_list(config_path: Path | None = _CONFIG_OPT) -> None:
    for row in _client(config_path).store.list_sessions():
        typer.echo(f"{row['id']}  {row.get('title') or ''}  {row.get('workdir')}")


@sessions_app.command("show")
def sessions_show(session_id: str, config_path: Path | None = _CONFIG_OPT) -> None:
    for turn in _client(config_path).store.turns(session_id):
        typer.echo(f"{turn['idx']} {turn['role']} ({turn['executor_id']})\n{turn['text'][:2000]}")


@sessions_app.command("export")
def sessions_export(
    session_id: str,
    fmt: str = typer.Option("md", "--format", help="md|jsonl"),
    config_path: Path | None = _CONFIG_OPT,
) -> None:
    from .cli_doctor import export_session

    export_session(_client(config_path), session_id, fmt, typer.echo)


@executors_app.command("list")
def executors_list(config_path: Path | None = _CONFIG_OPT) -> None:
    from .cli_doctor import print_executors_table

    print_executors_table(_client(config_path), typer.echo)


@executors_app.command("test")
def executors_test(executor_id: str, config_path: Path | None = _CONFIG_OPT) -> None:
    """Live one-turn ping."""
    reply = (
        _client(config_path)
        .session(workdir=str(Path.cwd()))
        .ask_sync("Reply with the word PONG", executor=executor_id, allow_fallback=False)
    )
    typer.echo(f"{reply.executor_id}: {reply.stop} {reply.text[:100]}")


@executors_app.command("cooldown")
def executors_cooldown(
    executor_id: str, seconds: int, config_path: Path | None = _CONFIG_OPT
) -> None:
    _client(config_path).store.set_cooldown(executor_id, time.time() + seconds, 1, "manual")
    typer.echo(f"cooldown set: {executor_id} for {seconds}s")


@executors_app.command("clear")
def executors_clear(
    executor_id: str | None = typer.Argument(None),
    all_cooldowns: bool = typer.Option(False, "--all", help="clear all cooldowns"),
    config_path: Path | None = _CONFIG_OPT,
) -> None:
    store = _client(config_path).store
    if all_cooldowns or executor_id is None:
        store.clear_cooldown(None)
        typer.echo("cleared all cooldowns")
    else:
        store.clear_cooldown(executor_id)
        typer.echo(f"cooldown cleared: {executor_id}")


@app.command()
def doctor(config_path: Path | None = _CONFIG_OPT) -> None:
    """CLIs + versions vs docs/compat.md, key envs present, DB writable."""
    from .cli_doctor import run_doctor

    run_doctor(_client(config_path), typer.echo)


@app.command()
def explain(
    prompt: str,
    role: str | None = typer.Option(None, "--role"),
    tier: str = typer.Option("auto", "--tier"),
    skill: list[str] | None = _SKILL_OPT,
    capability: str = typer.Option("auto", "--capability"),
    workdir: Path | None = _RUN_WORKDIR_OPT,
    json_output: bool = typer.Option(False, "--json"),
    config_path: Path | None = _CONFIG_OPT,
) -> None:
    """Explain candidate selection and skill detection without calling LLMs."""
    from .cli_doctor import explain_routing

    explain_routing(
        _client(config_path),
        prompt,
        role,
        tier,
        workdir,
        json_output,
        typer.echo,
        skill,
        capability,
    )


@app.command()
def benchmark(
    status_only: bool = typer.Option(False, "--status", help="show latest circuit status"),
    json_output: bool = typer.Option(False, "--json", help="output JSON"),
    config_path: Path | None = _CONFIG_OPT,
) -> None:
    """Run model health & latency benchmark, or inspect active circuit status."""
    from .benchmark import print_benchmark_report, print_benchmark_status, run_client_benchmark

    client = _client(config_path)
    if status_only:
        print_benchmark_status(client.store.get_executor_overrides(), json_output, typer.echo)
    else:
        print_benchmark_report(run_client_benchmark(client), json_output, typer.echo)


@app.command()
def chat(
    session: str | None = typer.Option(None, "--session"),
    new_session: bool = typer.Option(False, "--new-session", "--fresh", help="force new session"),
    tier: str = typer.Option("auto", "--tier"),
    executor: str | None = typer.Option(None, "--executor"),
    code: bool | None = typer.Option(None, "--code/--no-code"),
    workdir: Path = _CHAT_WORKDIR_OPT,
    config_path: Path | None = _CONFIG_OPT,
) -> None:
    """REPL: one conversation over many backends. /exit to leave."""
    client = AsfClient(config_path=config_path)
    resolved_wd = workdir.resolve()
    has_git = (resolved_wd / ".git").exists()
    capability = "code" if (code if code is not None else has_git) else "chat"
    sess = client.session(
        workdir=str(resolved_wd),
        session_id=session,
        continue_latest=not new_session and session is None,
    )
    current_tier: str | Tier = tier
    current_executor: str | None = executor
    last_executor: str | None = None
    typer.echo("onechat chat — /exit to leave, /help for commands")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt in SLASH_COMMANDS or prompt.split()[0] in SLASH_COMMANDS:
            current_tier, current_executor, sess, last_executor, done = handle_slash(
                client, sess, prompt, current_tier, current_executor, last_executor, workdir
            )
            if done:
                break
            continue
        last_executor = run_chat_turn(
            sess, prompt, current_tier, current_executor, capability, last_executor
        )


if __name__ == "__main__":
    app()
