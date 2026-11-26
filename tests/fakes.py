"""Shared fakes for Plan 02 executor tests (no network, no real processes)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from onechat.executors.process import ProcResult

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "cli"


class FakeProcessRunner:
    """Replays scripted outcomes or fixture files through the ProcessRunner protocol."""

    supports_inactivity_timeout: bool = True

    def __init__(
        self,
        result: ProcResult | None = None,
        lines: list[str] | None = None,
        exit_code: int = 0,
        stderr_tail: str = "",
        timeout: bool = False,
        hook: Callable[[list[str], dict[str, str]], None] | None = None,
        results: list[ProcResult] | None = None,
    ) -> None:
        self.result = result
        self.results_queue = list(results or [])
        self.lines = lines or []
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail
        self.timeout = timeout
        self.hook = hook
        self.calls: list[dict] = []

    async def run(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        stdin: str | None,
        timeout_s: int,
        on_line: Callable[[str], None],
        inactivity_timeout_s: int | None = None,
    ) -> ProcResult:

        self.calls.append({"argv": argv, "env": env, "cwd": cwd, "stdin": stdin})
        if self.hook:
            self.hook(argv, env)
        if self.results_queue:
            queued = self.results_queue.pop(0)
            for line in queued.stdout_lines:
                on_line(line)
            return queued
        if self.result is not None:
            for line in self.result.stdout_lines:
                on_line(line)
            return self.result
        for line in self.lines:
            on_line(line)
        return ProcResult(
            exit_code=self.exit_code,
            stdout_lines=list(self.lines),
            stderr_tail=self.stderr_tail,
            timed_out=self.timeout,
        )

    @classmethod
    def from_fixture(cls, profile: str, case: str) -> FakeProcessRunner:
        directory = FIXTURE_ROOT / profile
        lines = (directory / f"{case}.jsonl").read_text(encoding="utf-8").splitlines()
        meta = json.loads((directory / f"{case}.meta.json").read_text(encoding="utf-8"))
        return cls(lines=lines, exit_code=int(meta.get("exit_code", 0)))


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeChunk:
    def __init__(self, delta_text: str = "", usage: object = None) -> None:
        self.choices = [type("C", (), {"delta": type("D", (), {"content": delta_text})()})()]
        self.usage = usage


class FakeAsyncStream:
    """Async-iterable stream — the shape openai.AsyncOpenAI returns for stream=True."""

    def __init__(self, chunks: list[FakeChunk]) -> None:
        self._it = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


class FakeCompletions:
    """Async chat.completions.create (the ApiExecutor awaits it — code review P1-1)."""

    def __init__(self, chunks: list[FakeChunk], exc: Exception | None = None) -> None:
        self.chunks = chunks
        self.exc = exc
        self.calls: list[dict] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return FakeAsyncStream(self.chunks)


class FakeOpenAI:
    """Stands in for openai.OpenAI: one chat.completions.create behavior per instance."""

    def __init__(self, chunks: list[FakeChunk], exc: Exception | None = None) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(chunks, exc)


def make_config(tmp_path: Path, db_path: Path | None = None):
    """A RouterConfig rooted in tmp_path with two fake-ready cli executors."""
    from onechat.config import RouterConfig
    from onechat.contracts import Capability, CostClass, ExecutorSpec, Tier

    def spec(eid: str) -> ExecutorSpec:
        return ExecutorSpec(
            id=eid,
            kind="cli",
            profile=eid,
            model="m",
            tier=Tier.MEDIUM,
            capabilities=frozenset({Capability.CHAT, Capability.CODE}),
            cost=CostClass.FREE,
            family=f"family-{eid}",
        )

    return RouterConfig(
        db_path=db_path or tmp_path / "asf.db", executors=[spec("exec-a"), spec("exec-b")]
    )


def setup_session(store: object, sid: str, workdir: str = "/tmp") -> None:
    store.create_session(sid, title="t", workdir=workdir)


def write_user_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "executors:\n"
        "  - id: user-exec\n"
        "    kind: cli\n"
        "    profile: codex\n"
        "    model: m\n"
        "    tier: 2\n"
        "    capabilities: [chat]\n"
        "    cost: free\n"
        "    family: openai\n",
        encoding="utf-8",
    )
    return config_path


class FakeExecutor:
    """Scripted executor; with repeat_last=True the final scripted result repeats forever."""

    def __init__(self, spec, results, repeat_last: bool = False) -> None:
        self.spec = spec
        self.results = list(results)
        self.repeat_last = repeat_last
        self.requests: list = []
        self.last = None

    def available(self) -> bool:
        return True

    async def run(self, req, on_event=None):
        self.requests.append(req)
        if self.results:
            self.last = self.results.pop(0)
            return self.last
        if self.repeat_last and self.last is not None:
            return self.last
        raise AssertionError("FakeExecutor ran out of scripted results")


def fake_executors_factory(tmp_path: Path):
    """Builds the two default FakeExecutors from a RouterConfig (for AsfClient._for_tests)."""
    from onechat.contracts import TurnResult

    def factory(config):
        return {
            "exec-a": FakeExecutor(
                config.executors[0],
                [TurnResult(stop="done", text="fake answer a", native_session_id="nat-a")],
                repeat_last=True,
            ),
            "exec-b": FakeExecutor(
                config.executors[1],
                [TurnResult(stop="done", text="fake answer b")],
                repeat_last=True,
            ),
        }

    return factory
