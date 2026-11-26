from __future__ import annotations

from pathlib import Path

from onechat.contracts import Capability, CostClass, ExecutorSpec, Tier
from onechat.scoring import (
    SkillRoutingConfig,
    canonicalize_skill,
    detect_skills,
    resolve_model_scores,
    score_candidate,
)


def test_canonicalize_skill():
    assert canonicalize_skill("javascript") == "js"
    assert canonicalize_skill("typescript") == "ts"
    assert canonicalize_skill("python") == "py"
    assert canonicalize_skill("rust") == "rs"
    assert canonicalize_skill("review") == "code-review"
    assert canonicalize_skill("unknown_skill") == "unknown_skill"


def test_detect_skills_from_prompt():
    cfg = SkillRoutingConfig()
    prompt = "Fix the borrow checker in this code:\n```rust\nfn main() {}\n```"
    skills = detect_skills(prompt=prompt, cfg=cfg)
    assert "rs" in skills


def test_detect_skills_from_file_extensions():
    cfg = SkillRoutingConfig()
    prompt = "Please inspect src/index.ts and styles.css for bugs."
    skills = detect_skills(prompt=prompt, cfg=cfg)
    assert "ts" in skills
    assert "css" in skills


def test_detect_skills_skips_structural_tasks():
    cfg = SkillRoutingConfig()
    prompt = "Create a plan for typescript migration"
    skills = detect_skills(prompt=prompt, task_kind="plan", cfg=cfg)
    assert skills == ()


def test_detect_skills_from_workdir(tmp_path: Path):
    cfg = SkillRoutingConfig(marker_skills={"Cargo.toml": "rs", "package.json": "js"})
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'foo'\n")
    skills = detect_skills(prompt="Refactor everything", workdir=tmp_path, cfg=cfg)
    assert "rs" in skills


def test_detect_skills_ignores_monitor_dir(tmp_path: Path):
    cfg = SkillRoutingConfig(marker_skills={"package.json": "js", "Cargo.toml": "rs"})
    monitor_dir = tmp_path / "monitor"
    monitor_dir.mkdir()
    (monitor_dir / "package.json").write_text("{}")
    skills = detect_skills(prompt="Audit project", workdir=tmp_path, cfg=cfg)
    assert "js" not in skills


def test_resolve_model_scores():
    model_scores = {
        "claude-3-7-sonnet": {"rs": 90.0, "js": 95.0},
        "nemotron": {"rs": 45.0, "js": 85.0},
    }
    scores = resolve_model_scores(
        spec_model="openrouter/nvidia/nemotron-3-super:free",
        spec_model_key="nemotron",
        spec_scores={"rs": 50.0},
        model_scores=model_scores,
    )
    assert scores["rs"] == 50.0  # spec override
    assert scores["js"] == 85.0  # model_key fallback


def test_score_candidate_and_weak_threshold():
    cfg = SkillRoutingConfig(weak_threshold=40.0)
    spec = ExecutorSpec(
        id="ex1",
        kind="cli",
        profile="mock",
        capabilities=frozenset({Capability.CODE}),
        tier=Tier.MEDIUM,
        cost=CostClass.FREE,
        family="mock",
        model="mock-model",
        scores={"rs": 35.0, "js": 80.0},
    )
    # Testing weak skill rs
    res_rs = score_candidate(spec, skills=("rs",), cfg=cfg)
    assert res_rs.is_weak is True
    assert res_rs.final_score == 35.0

    # Testing strong skill js
    res_js = score_candidate(spec, skills=("js",), cfg=cfg)
    assert res_js.is_weak is False
    assert res_js.final_score == 80.0

    # Unrated skill gets default_score and is not weak
    res_py = score_candidate(spec, skills=("py",), cfg=cfg)
    assert res_py.is_weak is False
    assert res_py.final_score == cfg.default_score


def test_detect_skills_extension_boundaries_and_compound_priority() -> None:
    cfg = SkillRoutingConfig()
    # Must NOT trigger js from .json or golang from google.com
    assert detect_skills(prompt="Inspect package.json and google.com --json", cfg=cfg) == ()
    # Compound .spec.ts should yield e2e without falsely doubling ts
    assert detect_skills(prompt="Fix tests/login.spec.ts assertions", cfg=cfg) == ("e2e",)


def test_detect_skills_custom_aliases_and_ignores_plain_review() -> None:
    cfg = SkillRoutingConfig(aliases={"svelte": "frontend"})
    skills = detect_skills(prompt="Review AGENTS.md and update svelte component", cfg=cfg)
    assert "frontend" in skills
    assert "code-review" not in skills


def test_scan_workdir_collects_all_markers_and_defaults(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\n")
    (tmp_path / "package.json").write_text("{}")
    skills = detect_skills(prompt="Refactor workspace", workdir=tmp_path)
    assert "rs" in skills
    assert "js" in skills


def test_resolve_model_scores_normalizes_case_free_and_skill_aliases() -> None:
    model_scores = {
        "gpt": {"js": 40.0},
        "gpt-6-mini": {"js": 88.0},
        "NVIDIA/Nemotron-3.5:free": {"JavaScript": 25.0, "Rust": 92.0},
    }
    resolved = resolve_model_scores(
        spec_model="openrouter/NVIDIA/Nemotron-3.5:free",
        spec_model_key="nvidia/nemotron-3.5",
        spec_scores={"TypeScript": 80.0},
        model_scores=model_scores,
    )
    assert resolved == {"js": 25.0, "rs": 92.0, "ts": 80.0}

    # Longer substring key wins regardless of insertion order
    sub_resolved = resolve_model_scores(
        spec_model="openai/gpt-6-mini",
        spec_model_key=None,
        spec_scores={},
        model_scores=model_scores,
    )
    assert sub_resolved["js"] == 88.0
