"""Plan 02 redaction tests: secret prefixes + configured env var values."""

from __future__ import annotations

from onechat.redact import make_redactor, redact


def test_masks_known_prefixes() -> None:
    text = "key sk-abcdefghijklmnop1234 and AIzaSyB1234567890abcdefghijklmnopqrstuvw leaked"
    out = redact(text)
    assert "sk-abcdefghijklmnop1234" not in out
    assert "AIzaSyB1234567890abcdefghijklmnopqrstuvw" not in out
    assert "[REDACTED]" in out


def test_masks_values_of_configured_env_vars(monkeypatch: object) -> None:
    import os

    os.environ["ASF_TEST_TOKEN"] = "super-secret-token-value-123"
    try:
        make = make_redactor(["ASF_TEST_TOKEN", "ABSENT_VAR"])
        out = make("error calling with super-secret-token-value-123 inside")
        assert "super-secret-token-value-123" not in out
        assert "[REDACTED]" in out
    finally:
        del os.environ["ASF_TEST_TOKEN"]


def test_masks_empty_env_value_is_noop() -> None:
    make = make_redactor(["DEFINITELY_UNSET_VAR_42"])
    assert make("plain text") == "plain text"


def test_no_false_positives_on_code_identifiers() -> None:
    """T04: snake_case identifiers that merely start with sk/ask must survive verbatim."""
    text = "skill_targets task_description skip_git_repo_check desktop_layout"
    assert redact(text) == text


def test_redacts_real_provider_tokens() -> None:
    """T04: real provider token shapes are masked (provider-prefixed, Bearer, URL userinfo)."""
    samples = [
        "sk-proj-9f8e7d6c5b4a3210fedcba9876543210ab",
        "sk-ant-api03-9f8e7d6c5b4a3210fedcba9876543210",
        "gsk_9f8e7d6c5b4a3210987654321",
        "nvapi-9f8e7d6c5b4a3210fedcba9876543210",
        "ya29.a0AfB_9f8e7d6c5b4a3210fedcba98765432",
        "ghp_9f8e7d6c5b4a32109876543210987654321ab",
        "github_pat_11ABCDEF2_9f8e7d6c5b4a32109876",
        "Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature",
        "https://user:supersecret1@api.example.com/v1",
        "AKIAIOSFODNN7EXAMPLE",
    ]
    for token in samples:
        out = redact(f"failed around {token} endpoint")
        assert token not in out, f"token survived: {token}"
        assert "[REDACTED]" in out


def test_make_redactor_ignores_short_env_values(monkeypatch) -> None:
    """T04: tiny env values would false-positive everywhere; they are skipped."""
    monkeypatch.setenv("ASF_TINY", "1")
    make = make_redactor(["ASF_TINY"])
    assert make("v1 and 1") == "v1 and 1"
