"""onechat CLI REPL helpers: extracted from cli.py to maintain LOC budget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from .contracts import Tier
from .router import NoExecutorAvailable, SessionBusy
from .sdk import _UNSET, AsfClient
from .selector import coerce_tier

SLASH_COMMANDS = ("/tier", "/use", "/auto", "/status", "/new", "/sessions", "/help", "/exit")


def slash_command(
    client: AsfClient,
    sess: Any,
    prompt: str,
    current_tier: str | Tier,
    current_executor: str | None,
    last_executor: str | None,
    workdir: Path,
) -> tuple[str, str | Tier, str | None, Any, str | None, bool]:
    """Handle one REPL slash command. Returns (command, tier, executor, session, last, done)."""
    command, _, argument = prompt.partition(" ")
    if command == "/exit":
        return command, current_tier, current_executor, sess, last_executor, True
    if command == "/tier":
        current_tier = argument or "auto"
        typer.echo(f"tier: {current_tier}")
    elif command == "/use":
        current_executor = argument or None
        typer.echo(f"executor: {current_executor or '(auto)'}")
    elif command == "/auto":
        current_tier, current_executor = "auto", None
        typer.echo("mode: auto")
    if command == "/new":
        sess = client.session(workdir=str(workdir))
        typer.echo(f"new session: {sess.id}")
    elif command == "/sessions":
        for row in client.store.list_sessions()[:10]:
            typer.echo(f"  {row['id']} {row.get('title') or ''}")
    elif command == "/help":
        typer.echo(" ".join(SLASH_COMMANDS))
    elif command == "/status":
        typer.echo(
            f"session: {sess.id} tier: {current_tier} "
            f"executor: {current_executor or '(auto)'} last: {last_executor or '(none)'}"
        )
    return command, current_tier, current_executor, sess, last_executor, False


def handle_slash(
    client: AsfClient,
    sess: Any,
    prompt: str,
    tier: str | Tier,
    exec_id: str | None,
    last: str | None,
    workdir: Path,
) -> tuple[str | Tier, str | None, Any, str | None, bool]:
    cmd, tier, exec_id, sess, last, done = slash_command(
        client, sess, prompt, tier, exec_id, last, workdir
    )
    if cmd in ("/tier", "/use", "/auto"):
        try:
            tier_val = coerce_tier(tier) if isinstance(tier, str) else tier
            sess.pin(
                executor=exec_id if cmd != "/tier" else _UNSET,
                tier=tier_val if cmd != "/use" else _UNSET,
            )
        except ValueError as exc:
            typer.echo(f"[error] {exc}")
    return tier, exec_id, sess, last, done


def run_chat_turn(
    sess: Any,
    prompt: str,
    current_tier: str | Tier,
    current_executor: str | None,
    capability: str,
    last_executor: str | None,
) -> str | None:
    streamed = False

    def on_event(event: Any) -> None:
        nonlocal streamed
        if event.kind == "text":
            streamed = True
            typer.echo(event.text, nl=False)
        else:
            typer.echo(f"\n{event.text}")

    try:
        reply: Any = sess.ask_sync(
            prompt,
            tier=current_tier,
            executor=current_executor,
            capability=capability,
            on_event=on_event,
        )
    except (NoExecutorAvailable, SessionBusy, ValueError) as exc:
        typer.echo(f"[error] {exc}")
        return last_executor
    if streamed:
        typer.echo()
        typer.echo(f"{reply.executor_id} {reply.stop.value}")
    else:
        switched = f" (switched: {', '.join(reply.switched_from)})" if reply.switched_from else ""
        typer.echo(f"{reply.executor_id} {reply.stop.value}{switched}")
        typer.echo(reply.text)
    return str(reply.executor_id)
