"""Secret redaction (Plan 02 §9 / T04): mask known provider token shapes and configured
env var values. Boundary-anchored patterns: snake_case identifiers that merely start
with sk/ask never match; real provider prefixes, Bearer headers and URL userinfos do."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable

SECRET_RE = re.compile(
    r"\b(?:"
    r"sk-(?:proj-|ant-|live-|test-)?[A-Za-z0-9_-]{16,}"  # openai / anthropic style
    r"|gsk_[A-Za-z0-9]{16,}"  # groq
    r"|AIza[0-9A-Za-z_-]{35}"  # google api key
    r"|(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"  # github classic
    r"|github_pat_[A-Za-z0-9_]{20,}"  # github fine-grained
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"  # slack
    r"|nvapi-[A-Za-z0-9_-]{16,}"  # nvidia nim
    r"|ya29\.[A-Za-z0-9._-]{20,}"  # google oauth access token
    r"|csk-[A-Za-z0-9_-]{16,}"  # cerebras style
    r"|hf_[A-Za-z0-9]{16,}"  # huggingface
    r"|AKIA[0-9A-Z]{16}"  # aws access key id
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"  # JWT
    r")"
)
BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE)
URL_USERINFO_RE = re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)")
MASK = "[REDACTED]"

_MIN_ENV_VALUE_LEN = 8


def redact(text: str) -> str:
    """Mask secret-shaped tokens (T04: provider-prefix exact shapes, no sk/ask false hits)."""
    out = SECRET_RE.sub(MASK, text)
    out = BEARER_RE.sub("Bearer " + MASK, out)
    return URL_USERINFO_RE.sub(MASK, out)


def make_redactor(
    api_key_envs: list[str], extra_values: Iterable[str] = ()
) -> Callable[[str], str]:
    """Build a redactor that also masks the *values* of the named env vars (if set)
    and any inline literal secret values. Values shorter than 8 chars are skipped."""
    literals = [v for v in extra_values if len(v) >= _MIN_ENV_VALUE_LEN]

    def _redact(text: str) -> str:
        out = redact(text)
        for name in api_key_envs:
            value = os.environ.get(name, "")
            if len(value) >= _MIN_ENV_VALUE_LEN:
                out = out.replace(value, MASK)
        for value in literals:
            out = out.replace(value, MASK)
        return out

    return _redact
