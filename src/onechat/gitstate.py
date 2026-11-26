"""Git workspace state (Plan 03 §3 / T11): short status + diff stat for handoff briefs."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
from pathlib import Path

MAX_CHARS = 2000


def git_state(workdir: Path, max_chars: int = MAX_CHARS) -> str:
    """`git status --short` + `git diff --stat`, truncated. '' when not a git repo."""
    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        diff = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if status.returncode != 0:
        return ""
    parts = [status.stdout.strip()]
    if diff.returncode == 0 and diff.stdout.strip():
        parts.append(diff.stdout.strip())
    text = "\n".join(p for p in parts if p)
    return text[:max_chars]


async def async_git_state(workdir: Path, max_chars: int = MAX_CHARS) -> str:
    """Async variant (T11): staged + unstaged diff stat via create_subprocess_exec —
    the event loop never blocks on git. '' immediately when not a git repo."""
    try:
        status = await asyncio.create_subprocess_exec(
            "git",
            "status",
            "--short",
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await _communicate_or_kill(status, timeout=10)
    except (TimeoutError, OSError, subprocess.SubprocessError):
        return ""
    if status.returncode != 0:
        return ""
    parts = [out.decode("utf-8", errors="replace").strip()]
    for diff_args in (["git", "diff", "--cached", "--stat"], ["git", "diff", "--stat"]):
        text = await _git_out(diff_args, workdir)
        if text:
            parts.append(text)
    return "\n".join(p for p in parts if p)[:max_chars]


async def _git_out(args: list[str], workdir: Path) -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await _communicate_or_kill(proc, timeout=10)
    except (TimeoutError, OSError, subprocess.SubprocessError):
        return ""
    return out.decode("utf-8", errors="replace").strip() if proc.returncode == 0 else ""


async def _communicate_or_kill(
    proc: asyncio.subprocess.Process, timeout: int
) -> tuple[bytes, bytes]:
    try:
        return await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        raise
