"""ApiExecutor (Plan 02 §6.3 / T08): OpenAI-compatible API backends via the openai SDK.

Async end-to-end; the client cache is keyed by (running loop, key hash) so per-call
asyncio.run loops and rotated env keys always get a live client. One wall-clock
asyncio.timeout covers create + stream consumption; EVERY exception becomes a value.
"""

from __future__ import annotations

import asyncio
import contextlib
import email.utils
import hashlib
import math
import os
import re
import time
import weakref
from collections.abc import Callable
from datetime import UTC, datetime

import openai

from ..contracts import Event, ExecutorSpec, OnEvent, Stop, TurnRequest, TurnResult
from ..quota import parse_retry_after
from ..redact import make_redactor, redact

_GROQ_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(h|ms|m|s)")  # longest unit first: ms before m
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def default_openai_factory(spec: ExecutorSpec) -> object:
    # A missing key must not explode at construction: available() reports that honestly.
    # The placeholder makes any accidental call fail closed with an auth error.
    key = os.environ.get(spec.api_key_env or "", "") or "unavailable"
    return openai.AsyncOpenAI(base_url=spec.base_url, api_key=key, max_retries=0)


def _groq_duration_seconds(raw: str) -> float | None:
    """Groq-style durations: '1m2.5s', '250ms', '1h30m'. None when no unit pairs match."""
    total = 0.0
    matched = False
    for number, unit in _GROQ_DURATION_RE.findall(raw):
        total += float(number) * _UNIT_SECONDS[unit]
        matched = True
    return total if matched else None


def _duration_header_seconds(raw: str) -> float | None:
    """Retry-after header value: plain float, Groq duration, or an HTTP-date."""
    text = raw.strip()
    try:
        return float(text)
    except ValueError:
        pass
    groq = _groq_duration_seconds(text)
    if groq is not None:
        return groq
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    delta = (when - datetime.now(UTC)).total_seconds()
    return delta if delta >= 0 else None


class ApiExecutor:
    """One OpenAI-compatible API backend. run() never raises for backend problems."""

    def __init__(
        self,
        spec: ExecutorSpec,
        client_factory: Callable[[ExecutorSpec], object] = default_openai_factory,
    ) -> None:
        self.spec = spec
        self._client_factory = client_factory
        self._client: object | None = None
        self._loop_ref: weakref.ref | None = None
        self._key_hash: str = ""
        self._redact = (
            make_redactor([spec.api_key_env]) if spec.api_key_env else redact
        )  # Plan 02 §9: mask the configured key value in every error text

    def available(self) -> bool:
        return bool(os.environ.get(self.spec.api_key_env or ""))

    def _client_for_loop(self) -> object:
        """Fresh client per (live loop, key hash): async clients bind to one loop (T08).
        The loop is held by weakref identity — id() alone can alias a freed loop."""
        old_ref = self._loop_ref
        try:
            loop = asyncio.get_running_loop()
            new_ref: weakref.ref | None = weakref.ref(loop)
        except RuntimeError:
            loop, new_ref = None, None
        same_loop = loop is not None and old_ref is not None and old_ref() is loop
        key_hash = self._current_key_hash()
        if self._client is None or not same_loop or self._key_hash != key_hash:
            self._client = self._client_factory(self.spec)
        self._loop_ref = new_ref
        self._key_hash = key_hash
        return self._client

    def _current_key_hash(self) -> str:
        key = os.environ.get(self.spec.api_key_env or "") or ""
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    async def run(self, req: TurnRequest, on_event: OnEvent | None = None) -> TurnResult:
        start = time.monotonic()
        try:
            async with asyncio.timeout(req.timeout_s):  # wall clock over create+consume
                stream = await self._create_with_fallback(req)
                return await self._consume(stream, on_event, start, req)
        except openai.RateLimitError as exc:
            return self._quota(exc, start)
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            return self._auth(exc, start)
        except openai.APITimeoutError as exc:
            return self._timeout(exc, start)
        except openai.APIError as exc:
            return self._error(exc, start)
        except TimeoutError:
            return self._timeout(RuntimeError("wall clock timeout"), start)
        except Exception as exc:  # noqa: BLE001 - Plan 02 §2: run() never raises
            return self._error(exc, start)

    def _request_kwargs(self, req: TurnRequest, with_stream_options: bool) -> dict[str, object]:
        self._client_for_loop()  # ensure the loop-live client exists before creating
        kwargs: dict[str, object] = {
            "model": self.spec.model,
            "messages": self._messages(req),
            "stream": True,
            "timeout": req.timeout_s,
        }
        if with_stream_options and self.spec.stream_usage:
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    async def _create_with_fallback(self, req: TurnRequest) -> object:
        """Create the stream; a 400 with stream_options gets ONE clean retry without."""
        client = self._client_for_loop()
        completions = client.chat.completions  # type: ignore[attr-defined]
        try:
            return await completions.create(**self._request_kwargs(req, True))  # type: ignore[attr-defined]
        except openai.BadRequestError:
            if not self.spec.stream_usage:
                raise
            retry_kwargs = self._request_kwargs(req, False)
            return await completions.create(**retry_kwargs)  # type: ignore[attr-defined]

    def _messages(self, req: TurnRequest) -> list[dict[str, str]]:
        if req.messages:
            return [m.model_dump() for m in req.messages]
        return [{"role": "user", "content": req.prompt}]

    async def _consume(
        self, stream: object, on_event: OnEvent | None, start: float, req: TurnRequest
    ) -> TurnResult:
        texts: list[str] = []
        input_tokens = output_tokens = 0
        cost_usd: float | None = None
        async for chunk in stream:  # type: ignore[union-attr]
            choices = getattr(chunk, "choices", None) or []
            if choices:
                delta_obj = getattr(choices[0], "delta", None)
                delta = getattr(delta_obj, "content", None) if delta_obj is not None else None
                if delta:
                    texts.append(delta)
                    if on_event:
                        with contextlib.suppress(Exception):
                            on_event(Event(kind="text", text=delta))
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                cost_usd = getattr(usage, "cost", None)
        text = "".join(texts)
        if not text:
            return self._error(RuntimeError("empty stream: no content chunks"), start)
        return TurnResult(
            stop=Stop.DONE,
            text=text,
            usage=self._usage(input_tokens, output_tokens, cost_usd),
            duration_s=_elapsed(start),
        )

    def _usage(self, input_tokens: int, output_tokens: int, cost_usd: float | None):
        from ..contracts import Usage

        if cost_usd is None and (
            self.spec.input_cost_per_m is not None or self.spec.output_cost_per_m is not None
        ):
            cost_usd = self._priced_cost(input_tokens, output_tokens)
        return Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd)

    def _priced_cost(self, input_tokens: int, output_tokens: int) -> float:
        in_rate = self.spec.input_cost_per_m or 0.0
        out_rate = self.spec.output_cost_per_m or 0.0
        return input_tokens / 1e6 * in_rate + output_tokens / 1e6 * out_rate

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is not None and hasattr(client, "close"):
            with contextlib.suppress(Exception):
                res = client.close()  # type: ignore[attr-defined]
                if asyncio.iscoroutine(res):
                    await res

    def _quota(self, exc: Exception, start: float) -> TurnResult:
        headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
        retry_after = self._retry_after(headers) or parse_retry_after(str(exc), _utc_now())
        return TurnResult(
            stop=Stop.QUOTA,
            error=self._redact(str(exc)),
            retry_after_s=retry_after,
            duration_s=_elapsed(start),
        )

    def _retry_after(self, headers: object) -> int | None:
        get = headers.get if hasattr(headers, "get") else (lambda _name: None)
        for name in ("retry-after", "x-ratelimit-reset-requests"):
            raw = get(name)
            if not raw:
                continue
            seconds = _duration_header_seconds(str(raw))
            if seconds is not None:
                return max(1, math.ceil(seconds))
        return None

    def _auth(self, exc: Exception, start: float, text: str = "") -> TurnResult:
        return TurnResult(
            stop=Stop.AUTH, text=text, error=self._redact(str(exc)), duration_s=_elapsed(start)
        )

    def _timeout(self, exc: Exception, start: float, text: str = "") -> TurnResult:
        return TurnResult(
            stop=Stop.TIMEOUT, text=text, error=self._redact(str(exc)), duration_s=_elapsed(start)
        )

    def _error(self, exc: Exception, start: float, text: str = "") -> TurnResult:
        return TurnResult(
            stop=Stop.ERROR, text=text, error=self._redact(str(exc)), duration_s=_elapsed(start)
        )


def _elapsed(start: float) -> float:
    return max(time.monotonic() - start, 1e-6)


def _utc_now():
    return datetime.now(UTC)
