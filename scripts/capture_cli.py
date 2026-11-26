#!/usr/bin/env python3
"""Capture a CLI case to sanitized fixtures (Plan 01 Step 3).

Runs the command after `--`, stores stdout as tests/fixtures/cli/<profile>/<case>.jsonl,
writes <case>.meta.json (argv, exit code, duration, stderr tail). Sanitizes secrets,
home paths and emails before anything touches disk.

Usage:
    uv run scripts/capture_cli.py --profile codex --case new_turn \
        [--env KEY=VALUE ...] [--cwd DIR] [--] codex exec --json -m model "prompt"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SECRET_RE = re.compile(r"(sk|gsk|AIza|ghp|xox)[A-Za-z0-9_\-]{10,}")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
STDERR_TAIL_LINES = 40

PLAN_FILES_ADDENDUM = (
    "tests/test_capture_cli.py: Plan 01 Step 3 requires a smoke test for this script "
    "(plan Files list did not name it; authorized per 00_MASTER §6 rule 3)."
)


def sanitize(text: str, home: str) -> str:
    """Replace secrets, home paths and emails. Applied to everything we persist."""
    text = text.replace(home, "$HOME")
    text = SECRET_RE.sub("[REDACTED]", text)
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    return text


def sanitize_obj(obj: object, home: str) -> object:
    """Recursively sanitize strings inside parsed JSON structures."""
    if isinstance(obj, str):
        return sanitize(obj, home)
    if isinstance(obj, list):
        return [sanitize_obj(item, home) for item in obj]
    if isinstance(obj, dict):
        return {key: sanitize_obj(value, home) for key, value in obj.items()}
    return obj


def sanitize_stream(raw: str, home: str) -> str:
    """Sanitize a JSONL stream line by line; keep non-JSON lines as sanitized text."""
    out: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                out.append(sanitize(line, home))
                continue
            out.append(json.dumps(sanitize_obj(parsed, home), ensure_ascii=False))
        else:
            out.append(sanitize(line, home))
    return "\n".join(out) + ("\n" if raw.endswith("\n") or out else "")


def run_case(
    cmd: list[str], cwd: Path, extra_env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(extra_env)
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def write_fixtures(
    out_dir: Path,
    profile: str,
    case: str,
    cmd: list[str],
    result: subprocess.CompletedProcess[str],
    duration_s: float,
    home: str,
    extra_env: dict[str, str] | None = None,
) -> dict[str, object]:
    target = out_dir / profile
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{case}.jsonl").write_text(sanitize_stream(result.stdout, home), encoding="utf-8")
    meta = {
        "argv": [sanitize(arg, home) for arg in cmd],
        "exit_code": result.returncode,
        "duration_s": round(duration_s, 3),
        "stderr_tail": sanitize(result.stderr[-4000:], home).splitlines()[-STDERR_TAIL_LINES:],
        "env": {key: sanitize(value, home) for key, value in sorted((extra_env or {}).items())},
    }
    (target / f"{case}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", required=True, help="CLI profile name (e.g. codex)")
    parser.add_argument("--case", required=True, help="case name (e.g. new_turn)")
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE env override")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="working dir for the command")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "cli",
        help="fixture output root",
    )
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="command after --")
    args = parser.parse_args(argv)
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        parser.error("no command given after --")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extra_env = dict(kv.split("=", 1) for kv in args.env)
    home = str(Path.home())
    start = time.monotonic()
    try:
        result = run_case(args.cmd, args.cwd, extra_env)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after 600s: {' '.join(args.cmd)}", file=sys.stderr)
        return 124
    duration = time.monotonic() - start
    meta = write_fixtures(
        args.out, args.profile, args.case, args.cmd, result, duration, home, extra_env
    )
    print(
        json.dumps(
            {
                "profile": args.profile,
                "case": args.case,
                "exit_code": meta["exit_code"],
                "duration_s": meta["duration_s"],
                "out": str(args.out / args.profile),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
