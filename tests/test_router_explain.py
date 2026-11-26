from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import onechat.cli as cli_module
from onechat.contracts import ExecutorSpec, Stop, TurnResult
from onechat.sdk import AsfClient

runner = CliRunner()


class _FakeExec:
    def __init__(self, spec: ExecutorSpec) -> None:
        self.spec = spec
        self.calls: int = 0

    def available(self) -> bool:
        return True

    async def run(self, req, on_event=None):
        self.calls += 1
        return TurnResult(stop=Stop.DONE, text=f"from {self.spec.id}")


@pytest.fixture()
def isolated_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsfClient:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"db_path: {tmp_path / 'test.db'}\n"
        "skill_routing:\n"
        "  weak_threshold: 40.0\n"
        "  marker_skills:\n"
        "    Cargo.toml: rs\n"
        "executors:\n"
        "  - id: exec-chat\n"
        "    kind: api\n"
        "    model: m-chat\n"
        "    tier: 2\n"
        "    capabilities: [chat]\n"
        "    cost: free\n"
        "    family: api\n"
        "    base_url: https://x/v1\n"
        "    api_key_env: DUMMY_KEY\n"
        "    scores: {rs: 90.0}\n"
        "  - id: exec-weak-rs\n"
        "    kind: cli\n"
        "    profile: codex\n"
        "    model: m-weak\n"
        "    tier: 2\n"
        "    capabilities: [chat, code]\n"
        "    cost: free\n"
        "    family: a\n"
        "    priority: 1\n"
        "    scores: {rs: 20.0, js: 95.0}\n"
        "  - id: exec-strong-rs\n"
        "    kind: cli\n"
        "    profile: codex\n"
        "    model: m-strong\n"
        "    model_key: strong-rs\n"
        "    tier: 2\n"
        "    capabilities: [chat, code]\n"
        "    cost: free\n"
        "    family: b\n"
        "    priority: 10\n"
        "    scores: {rs: 92.0, js: 30.0}\n",
        encoding="utf-8",
    )
    client = AsfClient(config_path=cfg_path, home=tmp_path)
    for sid in list(client.executors):
        spec = client.executors[sid].spec
        client.executors[sid] = _FakeExec(spec)  # type: ignore[assignment]
        client.router.executors[sid] = client.executors[sid]
    monkeypatch.setattr(cli_module, "_client", lambda config_path=None: client)
    return client


def test_cli_explain_command(isolated_client: AsfClient) -> None:
    result = runner.invoke(
        cli_module.app,
        [
            "explain",
            "Write a Rust function to parse JSON with serde:\n```rust\nfn parse() {}\n```",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "detected skills: rs" in result.output
    assert "candidate decision table" in result.output


def test_cli_explain_json_output_and_flags(isolated_client: AsfClient) -> None:
    # Override priority via benchmark circuit so exec-strong-rs gets is_benchmarked bonus
    isolated_client.store.set_executor_override(
        "exec-strong-rs",
        status="healthy",
        enabled_override=1,
        priority_override=0,
        latency_s=1.0,
    )
    result = runner.invoke(
        cli_module.app,
        [
            "explain",
            "Refactor module",
            "--skill",
            "rust",
            "--role",
            "builder",
            "--capability",
            "code",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "rs" in data["skills"]
    # capability=code must exclude exec-chat (which only supports chat)
    ids = [c["id"] for c in data["candidates"]]
    assert "exec-chat" not in ids
    assert ids == ["exec-strong-rs", "exec-weak-rs"]
    # score in JSON must include benchmark_bonus (+5.0) => >= 97.0
    assert data["candidates"][0]["weak"] is False
    assert data["candidates"][1]["weak"] is True
    assert data["candidates"][0]["score"] >= 97.0


def test_cli_run_accepts_skill_flag(isolated_client: AsfClient) -> None:
    result = runner.invoke(
        cli_module.app,
        ["run", "Write helper", "--skill", "rs", "--json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    # exec-strong-rs (92) must beat exec-weak-rs (20) when --skill rs is passed
    assert data["executor_id"] == "exec-strong-rs"


@pytest.mark.asyncio
async def test_router_turn_passes_session_workdir_to_skill_detection(
    isolated_client: AsfClient, tmp_path: Path
) -> None:
    proj = tmp_path / "rust_proj"
    proj.mkdir()
    (proj / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
    sess = isolated_client.session(workdir=proj)
    reply = await sess.ask("Refactor main loop cleanly", capability="code", role="builder")
    # Prompt has no language words, so Cargo.toml in session workdir must detect 'rs'
    # and pick exec-strong-rs over exec-weak-rs
    assert reply.executor_id == "exec-strong-rs"
