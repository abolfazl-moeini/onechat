"""Router contracts (Plan 02 §2): executor specs, turn request/result, executor protocol."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, model_validator


class Tier(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    MAX = 4


class CostClass(StrEnum):
    FREE = "free"
    SUBSCRIPTION = "subscription"
    PAID = "paid"


class Capability(StrEnum):
    CHAT = "chat"
    CODE = "code"  # edits files / runs commands in workdir


class Stop(StrEnum):
    DONE = "done"
    QUOTA = "quota"
    AUTH = "auth"
    ERROR = "error"
    TIMEOUT = "timeout"


class Limits(BaseModel):
    rpm: int | None = Field(default=None, gt=0)
    rpd: int | None = Field(default=None, gt=0)


class ExecutorSpec(BaseModel):
    """One backend instance. Validators enforce cli/api shape differences."""

    id: str
    kind: Literal["cli", "api"]
    profile: str | None = None
    model: str
    model_key: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    tier: Tier
    capabilities: frozenset[Capability] = Field(default_factory=frozenset)
    cost: CostClass
    family: str
    env: dict[str, str] = Field(default_factory=dict)
    base_url: str | None = None
    api_key_env: str | None = None
    limits: Limits = Field(default_factory=Limits)
    priority: int = 100
    enabled: bool = True
    input_cost_per_m: float | None = None  # USD per 1M input tokens (paid accounting)
    output_cost_per_m: float | None = None  # USD per 1M output tokens
    stream_usage: bool = True  # send stream_options={"include_usage": True} (T08/D14)

    @model_validator(mode="after")
    def _validate_shape(self) -> ExecutorSpec:
        if self.kind == "cli" and not self.profile:
            raise ValueError("cli executor requires a profile")
        if self.kind == "cli" and not self.capabilities:
            raise ValueError("cli executor requires at least one capability")
        if self.kind == "api":
            if not self.base_url or not self.api_key_env:
                raise ValueError("api executor requires base_url and api_key_env")
            if self.capabilities != frozenset({Capability.CHAT}):
                raise ValueError("api executor capabilities must be exactly {CHAT}")
        return self


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


class Event(BaseModel):
    kind: Literal["text", "tool", "status"]
    text: str


class TurnRequest(BaseModel):
    prompt: str
    workdir: Path
    native_session_id: str | None = None
    messages: list[Message] | None = None
    timeout_s: int = Field(default=1800, gt=0)
    inactivity_timeout_s: int | None = Field(default=None, ge=0)


class TurnResult(BaseModel):
    stop: Stop
    text: str = ""
    native_session_id: str | None = None
    usage: Usage = Field(default_factory=Usage)
    retry_after_s: int | None = None
    error: str | None = None
    duration_s: float = 0.0


OnEvent = Callable[[Event], None]


class Executor(Protocol):
    spec: ExecutorSpec

    def available(self) -> bool: ...

    async def run(self, req: TurnRequest, on_event: OnEvent | None = None) -> TurnResult: ...


class Reply(BaseModel):
    text: str
    executor_id: str
    stop: Stop
    switched_from: list[str]
    turn_idx: int


class NoExecutorAvailable(Exception):
    def __init__(self, retry_at: float | None = None) -> None:
        self.retry_at = retry_at
        super().__init__("no executor available")


class SessionBusy(Exception):
    pass
