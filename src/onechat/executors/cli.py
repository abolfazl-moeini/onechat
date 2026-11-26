"""CliExecutor (Plan 02 §6.2 / T07): profile-driven argv templates + incremental sink.

Classification order (T07): TIMEOUT first; exit 0 without parsed.error is DONE when the
turn produced text or tool events; quota/auth patterns match ONLY on non-zero exit or
parsed.error. Every parse/classify exception becomes an ERROR TurnResult (Rule 8).
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from ..config import CliProfile
from ..contracts import ExecutorSpec, OnEvent, Stop, TurnRequest, TurnResult
from ..quota import parse_retry_after
from ..redact import make_redactor
from ..selector import is_high_effort_benchmark
from .parsers import StreamParser
from .process import ProcessRunner, ProcResult

_PLACEHOLDER_RE = re.compile(r"\{(model|prompt|session_id)\}")
_QUOTA_RE = re.compile(
    r"\b(?:limit reach(?:ed)?|quota(?: exceeded)?|rate limit(?: exceeded)?|"
    r"http\s*429|status[:\s]*429|\b429\b)\b",
    re.I,
)
_STDIN_SWITCH_CHARS = 32768


def _effort_flags(profile: CliProfile, prompt: str) -> list[str]:
    effort = "high" if is_high_effort_benchmark(prompt) else "medium"
    bin_name = Path(profile.binary).name.lower()
    if profile.name == "codex" or bin_name == "codex":
        return ["-c", f'model_reasoning_effort="{effort}"']
    if profile.name in ("claude", "agy") or bin_name in ("claude", "agy"):
        return ["--effort", effort]
    return []


def build_argv(
    profile: CliProfile, spec: ExecutorSpec, req: TurnRequest
) -> tuple[list[str], str | None]:
    """Expand the new/resume template in ONE pass (values are never re-expanded).
    A flag-looking prompt in a positional slot gets a '--' separator (T07)."""
    template = profile.resume if req.native_session_id else profile.new
    values = {"model": spec.model, "session_id": req.native_session_id or "", "prompt": req.prompt}
    via_stdin = profile.prompt_via == "stdin"
    flag_like = req.prompt.startswith("-")
    effort_args = _effort_flags(profile, req.prompt)
    has_effort = any(
        "model_reasoning_effort" in a or "--effort" in a for a in (*template, *profile.auto)
    )
    argv: list[str] = []
    for element in template:
        if element == "{auto}":
            argv.extend(profile.auto)
            if effort_args and not has_effort:
                argv.extend(effort_args)
            continue
        if element == "{prompt}":
            if via_stdin:
                continue  # prompt rides stdin
            if flag_like and _prompt_is_positional(argv):
                argv.append("--")
        argv.append(_PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], element))
    return argv, req.prompt if via_stdin else None


def _prompt_is_positional(argv: list[str]) -> bool:
    """True unless the previous element is a short value flag like -p/-m/-o."""
    if not argv:
        return True
    prev = argv[-1]
    return not (len(prev) == 2 and prev.startswith("-") and not prev.startswith("--"))


def _elapsed(start: float) -> float:
    return max(time.monotonic() - start, 1e-6)


def _utc_now():
    return datetime.now(UTC)


class CliExecutor:
    """One CLI backend instance. run() never raises for backend problems."""

    def __init__(
        self,
        spec: ExecutorSpec,
        profile: CliProfile,
        runner: ProcessRunner | None = None,
        which: callable = shutil.which,  # type: ignore[assignment]
    ) -> None:
        self.spec = spec
        self.profile = profile
        self._runner: ProcessRunner = runner or _default_runner()
        self._which = which
        self._redact = make_redactor(
            list(self.spec.env.keys()), extra_values=self.spec.env.values()
        )

    def available(self) -> bool:
        return self._which(self.profile.binary) is not None

    async def run(self, req: TurnRequest, on_event: OnEvent | None = None) -> TurnResult:
        start = time.monotonic()
        try:
            stream = StreamParser(self.profile.parser)  # unknown parser: honest ERROR (T07)
            template_argv, stdin_prompt = build_argv(self.profile, self.spec, req)
            argv = [self.profile.binary, *template_argv]
            env = dict(os.environ)
            env.update(self.spec.env)
            kwargs = {}
            if getattr(self._runner, "supports_inactivity_timeout", False):
                kwargs["inactivity_timeout_s"] = req.inactivity_timeout_s
            proc = await self._runner.run(
                argv,
                Path(req.workdir),
                env,
                stdin_prompt,
                req.timeout_s,
                on_line=self._line_sink(stream, on_event),
                **kwargs,
            )
            parsed = stream.finalize()

            return self._classify(proc, parsed, _elapsed(start))
        except Exception as exc:  # noqa: BLE001 - boundary: backend problems are values
            return TurnResult(
                stop=Stop.ERROR,
                error=self._redact(f"{type(exc).__name__}: {exc}"),
                duration_s=_elapsed(start),
            )

    def _line_sink(self, stream: StreamParser, on_event: OnEvent | None) -> callable:  # type: ignore[annotation]
        def sink(line: str) -> None:
            if on_event is not None:
                for event in stream.feed(line):  # O(N): each line parsed exactly once (T07)
                    with contextlib.suppress(Exception):
                        on_event(event)
            else:
                stream.feed(line)

        return sink

    def _classify(self, proc: ProcResult, parsed, duration: float) -> TurnResult:
        """Order per T07: timeout -> healthy exit -> quota/auth (failure text only)."""
        if proc.timed_out:
            detail = self._redact(
                " ".join(p for p in (parsed.error, proc.stderr_tail) if p).strip()
            )
            return TurnResult(
                stop=Stop.TIMEOUT,
                text=parsed.text,
                error=detail,
                native_session_id=parsed.session_id,
                usage=parsed.usage,
                duration_s=duration,
            )
        if proc.exit_code == 0 and parsed.error is None:
            healthy = bool(parsed.text) or any(e.kind == "tool" for e in parsed.events)
            if healthy:
                return TurnResult(
                    stop=Stop.DONE,
                    text=parsed.text,
                    native_session_id=parsed.session_id,
                    usage=parsed.usage,
                    duration_s=duration,
                )
            return TurnResult(
                stop=Stop.ERROR,
                text=parsed.text,
                error=self._redact("no output"),
                native_session_id=parsed.session_id,
                usage=parsed.usage,
                duration_s=duration,
            )
        return self._classify_failure(proc, parsed, duration)

    def _classify_failure(self, proc: ProcResult, parsed, duration: float) -> TurnResult:
        error_text = " ".join(p for p in (parsed.error, proc.stderr_tail) if p).strip()
        lowered = error_text.lower()
        native = parsed.session_id
        base = dict(
            text=parsed.text, native_session_id=native, usage=parsed.usage, duration_s=duration
        )
        is_quota = any(p.lower() in lowered for p in self.profile.quota_patterns) or bool(
            _QUOTA_RE.search(lowered)
        )
        if is_quota:
            return TurnResult(
                stop=Stop.QUOTA,
                error=self._redact(error_text),
                retry_after_s=parse_retry_after(error_text, _utc_now()),
                **base,
            )
        if any(p.lower() in lowered for p in self.profile.auth_patterns):
            return TurnResult(stop=Stop.AUTH, error=self._redact(error_text), **base)
        return TurnResult(stop=Stop.ERROR, error=self._redact(error_text or "no output"), **base)


def _default_runner() -> ProcessRunner:
    from .process import AsyncProcessRunner

    return AsyncProcessRunner()
