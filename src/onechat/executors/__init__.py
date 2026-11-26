"""Executor factory (Plan 02 §6): build_executor(spec, profiles) — a plain function."""

from __future__ import annotations

from ..config import CliProfile
from ..contracts import Executor, ExecutorSpec
from .api import ApiExecutor
from .cli import CliExecutor

__all__ = ["build_executor"]


def build_executor(spec: ExecutorSpec, profiles: dict[str, CliProfile]) -> Executor:
    """Build the right executor for a spec. Unknown cli profile is a config error (raise)."""
    if spec.kind == "api":
        return ApiExecutor(spec)
    if spec.profile not in profiles:
        raise KeyError(f"unknown profile {spec.profile!r} for executor {spec.id!r}")
    return CliExecutor(spec, profiles[spec.profile])
