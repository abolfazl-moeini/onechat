"""Plan 02 ApiExecutor tests: streaming, usage, exception mapping, availability."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import openai
from router_fakes import FakeChunk, FakeOpenAI

from onechat.contracts import ExecutorSpec, Stop, Tier, TurnRequest
from onechat.executors.api import ApiExecutor


def spec(**kw) -> ExecutorSpec:
    base = dict(
        id="groq-fast",
        kind="api",
        model="llama-3.1-8b-instant",
        tier=Tier.LOW,
        capabilities=frozenset({"chat"}),
        cost="free",
        family="meta",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    )
    base.update(kw)
    return ExecutorSpec(**base)


def req(**kw):
    from pathlib import Path

    base = dict(prompt="Reply with the word PONG", workdir=Path("/tmp"))
    base.update(kw)
    return TurnRequest(**base)


def test_streams_text_and_usage() -> None:
    chunks = [
        FakeChunk("PON"),
        FakeChunk("G"),
        FakeChunk(usage=SimpleNamespace(prompt_tokens=11, completion_tokens=2)),
    ]
    exe = ApiExecutor(spec(), client_factory=lambda s: FakeOpenAI(chunks))
    result = asyncio.run(exe.run(req()))
    print("ERROR IS:", repr(result.error))
    assert result.stop == Stop.DONE and result.text == "PONG"
    assert result.usage.input_tokens == 11 and result.usage.output_tokens == 2
    assert result.duration_s > 0


def test_rate_limit_maps_to_quota_with_retry_after() -> None:
    exc = openai.RateLimitError(
        "rate limited",
        response=SimpleNamespace(
            headers={"retry-after": "30"}, request=SimpleNamespace(headers={}), status_code=429
        ),
        body=None,
    )
    exe = ApiExecutor(spec(), client_factory=lambda s: FakeOpenAI([], exc=exc))
    result = asyncio.run(exe.run(req()))
    assert result.stop == Stop.QUOTA
    assert result.retry_after_s == 30


def test_auth_error_maps_to_auth() -> None:
    exc = openai.AuthenticationError(
        "bad key",
        response=SimpleNamespace(headers={}, request=SimpleNamespace(headers={}), status_code=401),
        body=None,
    )
    exe = ApiExecutor(spec(), client_factory=lambda s: FakeOpenAI([], exc=exc))
    result = asyncio.run(exe.run(req()))
    assert result.stop == Stop.AUTH


def test_timeout_maps_to_timeout() -> None:
    exc = openai.APITimeoutError(request=SimpleNamespace(headers={}))
    exe = ApiExecutor(spec(), client_factory=lambda s: FakeOpenAI([], exc=exc))
    result = asyncio.run(exe.run(req()))
    assert result.stop == Stop.TIMEOUT


def test_unavailable_without_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    exe = ApiExecutor(spec(), client_factory=lambda s: FakeOpenAI([]))
    assert exe.available() is False
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert exe.available() is True


def test_messages_history_used_when_given() -> None:
    from onechat.contracts import Message

    chunks = [
        FakeChunk("hi"),
        FakeChunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)),
    ]
    exe = ApiExecutor(spec(), client_factory=lambda s: FakeOpenAI(chunks))
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="u"),
        Message(role="assistant", content="a"),
        Message(role="user", content="q"),
    ]
    asyncio.run(exe.run(req(messages=messages)))
    # the fake records calls on its completions object
    # (client_factory returned it; grab via exe._client)
    assert exe._client.chat.completions.calls[0]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "q"},
    ]


def test_never_raises_for_unexpected_stream_error() -> None:
    """Plan 02 §2 rule: run() never raises, even for non-openai errors mid-stream."""

    class _BoomStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RuntimeError("stream exploded")

    async def _boom_create(**kw):
        return _BoomStream()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_boom_create)))
    exe = ApiExecutor(spec(), client_factory=lambda s: client)
    result = asyncio.run(exe.run(req()))
    assert result.stop == Stop.ERROR
    assert "stream exploded" in (result.error or "")


def test_error_redacts_api_key_env_value(monkeypatch) -> None:
    """Plan 02 §9: values of env vars named in api_key_env never survive into errors."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-real-secret-value-123")
    exc = openai.RateLimitError(
        "rate limited for key gsk-real-secret-value-123",
        response=SimpleNamespace(headers={}, request=SimpleNamespace(headers={}), status_code=429),
        body=None,
    )
    exe = ApiExecutor(spec(), client_factory=lambda s: FakeOpenAI([], exc=exc))
    result = asyncio.run(exe.run(req()))
    assert result.stop == Stop.QUOTA
    assert result.error is not None
    assert "gsk-real-secret-value-123" not in result.error


# --- Code review 2026-11-18: the executor must drive the API async (P1-1) ---


class AsyncFakeStream:
    """Async-iterable stream, the shape openai.AsyncOpenAI returns for stream=True."""

    def __init__(self, chunks: list) -> None:
        self._it = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


class AsyncFakeCompletions:
    def __init__(self, chunks: list, exc: Exception | None = None) -> None:
        self.chunks = chunks
        self.exc = exc
        self.calls: list[dict] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        await asyncio.sleep(0.1)  # simulated network latency
        if self.exc:
            raise self.exc
        return AsyncFakeStream(self.chunks)


def test_api_executor_awaits_openai_and_keeps_loop_alive() -> None:
    """P1-1: run() awaits the create call (AsyncOpenAI semantics) so concurrent tasks
    on the same loop (lease renewal, other executors) progress during the request."""
    chunks = [
        FakeChunk("PO"),
        FakeChunk("NG"),
        FakeChunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1)),
    ]
    client = SimpleNamespace(chat=SimpleNamespace(completions=AsyncFakeCompletions(chunks)))
    exe = ApiExecutor(spec(), client_factory=lambda s: client)

    async def scenario():
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        hb = asyncio.create_task(heartbeat())
        result = await exe.run(req())
        hb.cancel()
        return result, ticks

    result, ticks = asyncio.run(scenario())
    assert result.stop == Stop.DONE and result.text == "PONG"
    assert ticks >= 3, "event loop stalled during the API call"


def test_async_stream_error_maps_to_error_value() -> None:
    """Mid-stream failures on an async stream are values, never raises."""

    class BoomAsyncStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RuntimeError("stream exploded")

    client = SimpleNamespace(chat=SimpleNamespace(completions=AsyncFakeCompletions([], exc=None)))

    async def failing_create(**kw):
        return BoomAsyncStream()

    client.chat.completions.create = failing_create  # type: ignore[method-assign]
    exe = ApiExecutor(spec(), client_factory=lambda s: client)
    result = asyncio.run(exe.run(req()))
    assert result.stop == Stop.ERROR
    assert "stream exploded" in (result.error or "")


# ---- T08: loop-safe client cache, wall timeout, headers, pricing, stream_options -------


def _done_stream(chunks):
    return AsyncFakeStream(chunks)


def test_new_loop_gets_fresh_client() -> None:
    """T08: each asyncio.run() loop gets its own client (ask_sync / chat loops)."""
    loops: list = []
    chunks = [FakeChunk("PONG", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))]

    def factory(s):
        loops.append(asyncio.get_running_loop())  # strong ref: no freed-object id aliasing
        return FakeOpenAI(chunks)

    exe = ApiExecutor(spec(), client_factory=factory)
    asyncio.run(exe.run(req()))
    asyncio.run(exe.run(req()))
    assert len(loops) == 2, "client must be re-created per event loop"
    assert loops[0] is not loops[1]


def test_picks_up_rotated_env_api_key_dynamically(monkeypatch) -> None:
    """T08: a rotated key in the env is picked up on the NEXT run (key-hash cache key)."""
    import onechat.executors.api as api_module

    created: list[str] = []
    chunks = [FakeChunk("PONG")]

    class RecordingClient:
        def __init__(self, base_url, api_key, max_retries):
            created.append(api_key)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: _async_return(_done_stream(chunks)))
            )

    async def _fake_create(**kw):
        return _done_stream(chunks)

    def make_client(base_url, api_key, max_retries):
        created.append(api_key)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_create))
        )

    monkeypatch.setattr(api_module.openai, "AsyncOpenAI", make_client)
    monkeypatch.setenv("GROQ_API_KEY", "key-one-1234567")
    exe = ApiExecutor(spec())  # default factory
    asyncio.run(exe.run(req()))
    monkeypatch.setenv("GROQ_API_KEY", "key-two-7654321")
    asyncio.run(exe.run(req()))
    assert created[-1] == "key-two-7654321"


def _async_return(value):
    async def _it():
        return value

    return _it()


class SlowStream:
    """Async stream whose first chunk takes far longer than the request timeout."""

    def __init__(self, delay_s: float) -> None:
        self._delay = delay_s

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(self._delay)
        raise StopAsyncIteration


def test_wall_clock_timeout_interrupts_slow_stream() -> None:
    """T08: wall timeout covers create + stream consumption (not just the SDK call)."""
    import time as _time

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: _async_return(SlowStream(10.0)))
        )
    )
    exe = ApiExecutor(spec(), client_factory=lambda s: client)
    started = _time.monotonic()
    result = asyncio.run(exe.run(req(timeout_s=1)))
    elapsed = _time.monotonic() - started
    assert result.stop == Stop.TIMEOUT
    assert elapsed < 5.0, f"wall timeout did not interrupt the slow stream ({elapsed:.1f}s)"


def test_mid_stream_timeout_and_auth_classified_properly() -> None:
    """T08: openai errors raised MID-STREAM map exactly like pre-stream errors."""

    class ErrStream:
        def __init__(self, exc: Exception) -> None:
            self._exc = exc

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise self._exc

    timeout_exc = openai.APITimeoutError(request=SimpleNamespace(headers={}))
    auth_exc = openai.AuthenticationError(
        "bad key",
        response=SimpleNamespace(headers={}, request=SimpleNamespace(headers={}), status_code=401),
        body=None,
    )
    for exc, expected in ((timeout_exc, Stop.TIMEOUT), (auth_exc, Stop.AUTH)):
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda _exc=exc, **kw: _async_return(ErrStream(_exc))
                )
            )
        )
        exe = ApiExecutor(spec(), client_factory=lambda s, c=client: c)
        result = asyncio.run(exe.run(req()))
        assert result.stop == expected, f"{type(exc).__name__} mid-stream -> {expected}"


def test_parses_groq_duration_and_fractional_retry_after() -> None:
    """T08: Retry-After handles floats, Groq durations (1m2.5s/250ms) and HTTP-dates."""
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    future = format_datetime(datetime.now(UTC) + timedelta(hours=1))
    cases = [
        ({"retry-after": "0.5"}, 1),
        ({"x-ratelimit-reset-requests": "1m2.5s"}, 63),
        ({"x-ratelimit-reset-requests": "250ms"}, 1),
        ({"retry-after": future}, None),  # > 0 — HTTP-date path (exact value time-dependent)
    ]
    for headers, expected in cases:
        exc = openai.RateLimitError(
            "rate limited",
            response=SimpleNamespace(
                headers=headers, request=SimpleNamespace(headers={}), status_code=429
            ),
            body=None,
        )
        exe = ApiExecutor(spec(), client_factory=lambda s, e=exc: FakeOpenAI([], exc=e))
        result = asyncio.run(exe.run(req()))
        assert result.stop == Stop.QUOTA
        if expected is None:
            assert result.retry_after_s is not None and result.retry_after_s > 0
        else:
            assert result.retry_after_s == expected, f"{headers} -> {expected}"


def test_empty_stream_returns_error() -> None:
    """T08: a stream with no content chunks is an ERROR, not a silent empty DONE."""
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kw: _async_return(_done_stream([FakeChunk("")]))
            )
        )
    )
    exe = ApiExecutor(spec(), client_factory=lambda s: client)
    result = asyncio.run(exe.run(req()))
    assert result.stop == Stop.ERROR


def test_computes_cost_from_spec_pricing_when_omitted() -> None:
    """T08: cost = tokens/1M * input_cost_per_m/output_cost_per_m when the API omits it."""
    usage = SimpleNamespace(prompt_tokens=1_500_000, completion_tokens=250_000)
    chunks = [FakeChunk("hi", usage=usage)]
    exe = ApiExecutor(
        spec(input_cost_per_m=2.0, output_cost_per_m=8.0),
        client_factory=lambda s: FakeOpenAI(chunks),
    )
    result = asyncio.run(exe.run(req()))
    assert result.usage.cost_usd is not None
    assert abs(result.usage.cost_usd - 5.0) < 1e-6  # 1.5*2 + 0.25*8


def test_stream_usage_false_omits_stream_options() -> None:
    """T08: stream_usage=False sends no stream_options at all."""
    recorded: list[dict] = []

    async def create(**kw):
        recorded.append(kw)
        return _done_stream([FakeChunk("PONG")])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    exe = ApiExecutor(spec(stream_usage=False), client_factory=lambda s: client)
    result = asyncio.run(exe.run(req()))
    print("ERROR IS:", repr(result.error))
    assert result.stop == Stop.DONE
    assert "stream_options" not in recorded[0]


def test_400_retries_once_without_stream_options() -> None:
    """T08: providers rejecting stream_options get exactly ONE clean retry without them."""
    recorded: list[dict] = []

    async def create(**kw):
        recorded.append(kw)
        if "stream_options" in kw:
            raise openai.BadRequestError(
                "stream_options unsupported",
                response=SimpleNamespace(
                    headers={}, request=SimpleNamespace(headers={}), status_code=400
                ),
                body=None,
            )
        return _done_stream([FakeChunk("PONG")])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    exe = ApiExecutor(spec(), client_factory=lambda s: client)
    result = asyncio.run(exe.run(req()))
    print("ERROR IS:", repr(result.error))
    assert result.stop == Stop.DONE
    assert len(recorded) == 2
    assert "stream_options" in recorded[0] and "stream_options" not in recorded[1]


def test_stream_usage_false_does_not_retry_bad_request() -> None:
    """Audit: if stream_usage=False (never sent stream_options), 400 must not retry."""
    recorded: list[dict] = []

    async def create(**kw):
        recorded.append(kw)
        raise openai.BadRequestError(
            "bad model",
            response=SimpleNamespace(
                headers={}, request=SimpleNamespace(headers={}), status_code=400
            ),
            body=None,
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    exe = ApiExecutor(spec(stream_usage=False), client_factory=lambda s: client)
    result = asyncio.run(exe.run(req()))
    assert result.stop == Stop.ERROR
    assert len(recorded) == 1


def test_api_executor_handles_none_delta_and_on_event_error() -> None:
    """Bug 9: chunk.choices[0].delta=None and crashing on_event must not fail the turn."""
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=None)], usage=None),
        FakeChunk("PONG"),
    ]
    exe = ApiExecutor(spec(), client_factory=lambda s: FakeOpenAI(chunks))

    def bad_cb(_ev) -> None:
        raise RuntimeError("cb boom")

    result = asyncio.run(exe.run(req(), on_event=bad_cb))
    assert result.stop == Stop.DONE
    assert result.text == "PONG"


def test_api_executor_output_only_pricing() -> None:
    """Bug 9: pricing with only output_cost_per_m set still computes cost_usd."""
    import pytest

    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=500_000)
    exe = ApiExecutor(
        spec(output_cost_per_m=4.0),
        client_factory=lambda s: FakeOpenAI([FakeChunk("hi", usage=usage)]),
    )
    result = asyncio.run(exe.run(req()))
    assert result.usage.cost_usd == pytest.approx(2.0)
