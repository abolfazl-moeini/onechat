"""ProcessRunner protocol + asyncio implementation (Plan 02 §6.1 / T05).

Chunked stdout reads (arbitrary lines up to 4 MiB never raise ValueError), stdin fed
concurrently (broken-pipe safe when the child exits early), rolling 4 KiB stderr tail,
and full process-group termination on timeout AND cancellation (no orphan leaks).
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

STDERR_TAIL_BYTES = 4096
KILL_GRACE_S = 5.0
STDOUT_CHUNK_BYTES = 65536
MAX_LINE_BYTES = 4 * 1024 * 1024
_SOFT_PIPE_ERRORS = (BrokenPipeError, ConnectionResetError)


class ProcResult(BaseModel):
    exit_code: int
    stdout_lines: list[str]
    stderr_tail: str
    timed_out: bool


class ProcessRunner(Protocol):
    supports_inactivity_timeout: bool = False

    async def run(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str] | None,
        stdin: str | None,
        timeout_s: int,
        on_line: Callable[[str], None],
        inactivity_timeout_s: float | None = None,
    ) -> ProcResult: ...


class AsyncProcessRunner:
    """Runs one subprocess in its own process group; streams stdout lines to on_line."""

    supports_inactivity_timeout: bool = True

    async def run(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str] | None,
        stdin: str | None,
        timeout_s: int,
        on_line: Callable[[str], None],
        inactivity_timeout_s: float | None = None,
    ) -> ProcResult:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout_lines: list[str] = []
        stderr_buf = bytearray()
        timed_out = False
        try:
            await self._run_with_watchdogs(
                proc, stdin, on_line, stdout_lines, stderr_buf, timeout_s, inactivity_timeout_s
            )
        except TimeoutError:
            timed_out = True
            await self._terminate_group(proc)
        except asyncio.CancelledError:
            await self._terminate_group(proc)
            raise
        return ProcResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout_lines=stdout_lines,
            stderr_tail=bytes(stderr_buf).decode("utf-8", errors="replace"),
            timed_out=timed_out,
        )

    async def _run_with_watchdogs(
        self,
        proc: asyncio.subprocess.Process,
        stdin: str | None,
        on_line: Callable[[str], None],
        stdout_lines: list[str],
        stderr_buf: bytearray,
        timeout_s: int,
        inactivity_s: float | None,
    ) -> None:
        last_active = [time.monotonic()]
        drive_task = asyncio.ensure_future(
            self._drive(proc, stdin, on_line, stdout_lines, stderr_buf, last_active)
        )
        wall_deadline = time.monotonic() + timeout_s
        try:
            while not drive_task.done():
                now = time.monotonic()
                if now >= wall_deadline:
                    raise TimeoutError("wall-clock timeout")
                if inactivity_s and inactivity_s > 0 and (now - last_active[0]) >= inactivity_s:
                    msg = b"\n[onechat watchdog: process killed after inactivity timeout]\n"
                    stderr_buf.extend(msg)
                    raise TimeoutError("inactivity timeout")
                step = 0.5
                if inactivity_s and inactivity_s > 0:
                    step = min(step, max(0.05, inactivity_s - (now - last_active[0])))
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(drive_task), timeout=step)
            await drive_task
        finally:
            if not drive_task.done():
                drive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await drive_task

    async def _drive(
        self,
        proc: asyncio.subprocess.Process,
        stdin: str | None,
        on_line: Callable[[str], None],
        stdout_lines: list[str],
        stderr_buf: bytearray,
        last_active: list[float] | None = None,
    ) -> None:
        """stdin feed, both pumps and proc.wait() run concurrently (no pipe deadlocks)."""
        feed = asyncio.ensure_future(self._feed_stdin(proc, stdin))
        pump_out = asyncio.ensure_future(
            self._pump_lines(proc.stdout, on_line, stdout_lines, last_active)
        )
        pump_err = asyncio.ensure_future(self._pump_stderr(proc.stderr, stderr_buf, last_active))
        results = await asyncio.gather(
            pump_out, pump_err, feed, proc.wait(), return_exceptions=True
        )
        for exc in results[:3]:
            if isinstance(exc, Exception) and not isinstance(exc, _SOFT_PIPE_ERRORS):
                raise exc

    async def _feed_stdin(self, proc: asyncio.subprocess.Process, stdin_text: str | None) -> None:
        if stdin_text is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(stdin_text.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
        except _SOFT_PIPE_ERRORS:
            pass  # child exited before reading stdin; the exit code carries the outcome

    async def _pump_lines(
        self,
        stream: asyncio.StreamReader | None,
        on_line: Callable[[str], None],
        sink: list[str],
        last_active: list[float] | None = None,
    ) -> None:
        """65536-byte chunked reads: split on newlines, cap a single line at 4 MiB."""
        if stream is None:
            return
        buf = b""
        while True:
            chunk = await stream.read(STDOUT_CHUNK_BYTES)
            if not chunk:
                break
            if last_active is not None:
                last_active[0] = time.monotonic()
            buf += chunk
            while b"\n" in buf or len(buf) >= MAX_LINE_BYTES:
                if b"\n" in buf:
                    raw, _, buf = buf.partition(b"\n")
                else:  # pathological unterminated line: emit capped chunk, keep memory bounded
                    raw, buf = buf[:MAX_LINE_BYTES], buf[MAX_LINE_BYTES:]
                line = raw.decode("utf-8", errors="replace").rstrip("\r")
                sink.append(line)
                on_line(line)
        if buf:
            line = buf.decode("utf-8", errors="replace").rstrip("\r")
            sink.append(line)
            on_line(line)

    async def _pump_stderr(
        self,
        stream: asyncio.StreamReader | None,
        buf: bytearray,
        last_active: list[float] | None = None,
    ) -> None:
        while stream is not None:
            raw = await stream.read(1024)
            if not raw:
                break
            if last_active is not None:
                last_active[0] = time.monotonic()
            buf.extend(raw)
            if len(buf) > STDERR_TAIL_BYTES:
                del buf[: len(buf) - STDERR_TAIL_BYTES]

    async def _terminate_group(self, proc: asyncio.subprocess.Process) -> None:
        import contextlib

        try:
            import os

            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, AttributeError):
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=KILL_GRACE_S)
        except TimeoutError:
            try:
                import os

                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, AttributeError):
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            await proc.wait()
