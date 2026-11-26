"""Plan 03 live switch test (@pytest.mark.live): real mid-chat backend switch.

Ladder rung L2 reproduction script. With tonight's auth/network walls (docs/compat.md) this
skips with the recorded reason; when backends are authenticated it performs the real demo:
codex-main answers a number question, gets cooled down, agy-main (or any other available)
must still know the number.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from onechat.contracts import Stop
from onechat.sdk import AsfClient


@pytest.mark.live
def test_live_switch_mid_chat() -> None:
    client = AsfClient()
    executors = {eid: ex for eid, ex in client.executors.items() if "code" in ex.spec.capabilities}
    if len(executors) < 1:
        pytest.skip("no CODE-capable executor configured")
    session = client.session(workdir=str(Path.cwd()))
    ordered = sorted(executors, key=lambda eid: executors[eid].spec.priority)
    first, second = ordered[0], (ordered[1] if len(ordered) > 1 else None)

    r1 = asyncio.run(
        session.ask("Remember the number 4721. Reply OK.", executor=first, allow_fallback=False)
    )
    if r1.stop != Stop.DONE:
        pytest.skip(f"{first}: {r1.stop.value} (compat.md DEFERRED)")
    if second is None:
        pytest.skip("need a second executor for the switch (config has one)")
    client.store.set_cooldown(first, __import__("time").time() + 600, 1, "live-demo")
    r2 = asyncio.run(
        session.ask("What number did I ask you to remember?", executor=second, allow_fallback=False)
    )
    if r2.stop != Stop.DONE:
        pytest.skip(f"{second}: {r2.stop.value} (compat.md DEFERRED)")
    assert "4721" in r2.text, f"switch lost context: {r2.text!r}"
    assert r2.switched_from == [first]
