"""Plan 02 contract tests: ExecutorSpec validation, tier ordering, TurnResult shape."""

from __future__ import annotations

from pathlib import Path

import pytest
from onechat.contracts import (
    Capability,
    CostClass,
    ExecutorSpec,
    Stop,
    Tier,
    TurnResult,
)
from pydantic import ValidationError


def cli_spec(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "codex-main",
        "kind": "cli",
        "profile": "codex",
        "model": "gpt-5-codex",
        "tier": Tier.MEDIUM,
        "capabilities": frozenset({Capability.CHAT, Capability.CODE}),
        "cost": CostClass.SUBSCRIPTION,
        "family": "openai",
    }
    base.update(overrides)
    return base


def api_spec(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "groq-fast",
        "kind": "api",
        "model": "llama-3.1-8b-instant",
        "tier": Tier.LOW,
        "capabilities": frozenset({Capability.CHAT}),
        "cost": CostClass.FREE,
        "family": "meta",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    }
    base.update(overrides)
    return base


def test_cli_spec_requires_profile() -> None:
    with pytest.raises(ValidationError, match="profile"):
        ExecutorSpec(**{k: v for k, v in cli_spec().items() if k != "profile"})


def test_api_spec_requires_base_url_and_key_env() -> None:
    with pytest.raises(ValidationError, match="base_url"):
        ExecutorSpec(**{k: v for k, v in api_spec().items() if k != "base_url"})
    with pytest.raises(ValidationError, match="api_key_env"):
        ExecutorSpec(**{k: v for k, v in api_spec().items() if k != "api_key_env"})


def test_api_spec_cannot_claim_code() -> None:
    with pytest.raises(ValidationError, match="CHAT"):
        ExecutorSpec(**api_spec(capabilities=frozenset({Capability.CHAT, Capability.CODE})))


def test_tier_ordering() -> None:
    assert Tier.LOW < Tier.MEDIUM < Tier.HIGH < Tier.MAX


def test_stop_values_are_stable() -> None:
    assert {s.value for s in Stop} == {"done", "quota", "auth", "error", "timeout"}


def test_turn_result_defaults_never_raise() -> None:
    result = TurnResult(stop=Stop.DONE)
    assert result.text == "" and result.usage.input_tokens == 0
    assert result.native_session_id is None and result.duration_s == 0.0


def test_env_tilde_placeholder_documented() -> None:
    """ExecutorSpec.env values may contain '~' (expanded at run time, not at parse)."""
    spec = ExecutorSpec(**cli_spec(id="codex-acc2", env={"CODEX_HOME": "~/.asf/accounts/codex"}))
    assert spec.env["CODEX_HOME"] == "~/.asf/accounts/codex"
    assert isinstance(spec.model, str) and isinstance(Path, type)


def test_api_spec_requires_chat_capability() -> None:
    # Plan 02 §2: api capabilities == {CHAT} exactly — an empty set is rejected.
    with pytest.raises(ValidationError, match="CHAT"):
        ExecutorSpec(**api_spec(capabilities=frozenset()))
