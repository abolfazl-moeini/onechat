"""Skill scoring, model resolution, and prompt skill detection."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from .contracts import ExecutorSpec

CANONICAL_ALIASES: dict[str, str] = {
    "javascript": "js",
    "js": "js",
    "jsx": "js",
    "typescript": "ts",
    "ts": "ts",
    "tsx": "ts",
    "python": "py",
    "py": "py",
    "rust": "rs",
    "rs": "rs",
    "golang": "golang",
    "go": "golang",
    "bash": "sh",
    "sh": "sh",
    "shell": "sh",
    "zsh": "sh",
    "css": "css",
    "scss": "css",
    "sass": "css",
    "html": "frontend",
    "frontend": "frontend",
    "e2e": "e2e",
    "playwright": "e2e",
    "cypress": "e2e",
    "review": "code-review",
    "code-review": "code-review",
    "architecture": "architecture",
    "arch": "architecture",
}

DEFAULT_EXTENSIONS: dict[str, str] = {
    ".spec.ts": "e2e",
    ".test.ts": "ts",
    ".ts": "ts",
    ".tsx": "ts",
    ".js": "js",
    ".jsx": "js",
    ".py": "py",
    ".rs": "rs",
    ".go": "golang",
    ".sh": "sh",
    ".css": "css",
    ".scss": "css",
}

DEFAULT_MARKERS: dict[str, str] = {
    "Cargo.toml": "rs",
    "package.json": "js",
    "tsconfig.json": "ts",
    "pyproject.toml": "py",
    "go.mod": "golang",
}

STRUCTURAL_TASKS: frozenset[str] = frozenset(
    {"plan", "spec", "summarize", "digest", "research", "audit"}
)


class SkillRoutingConfig(BaseModel):
    """Configuration for skill-based routing."""

    enabled: bool = True
    default_score: float = 50.0
    weak_threshold: float = 40.0
    role_bonus: float = 5.0
    benchmark_bonus: float = 5.0
    auto_detect: bool = True
    aliases: dict[str, str] = Field(default_factory=dict)
    extension_skills: dict[str, str] = Field(default_factory=dict)
    marker_skills: dict[str, str] = Field(default_factory=dict)


class CandidateScore(BaseModel):
    """Evaluated score for an executor candidate."""

    spec: ExecutorSpec
    final_score: float
    is_weak: bool
    is_rated: bool
    min_explicit_score: float | None = None


def canonicalize_skill(skill: str, extra_aliases: dict[str, str] | None = None) -> str:
    """Normalize a skill token to its canonical form."""
    s = skill.strip().lower()
    if extra_aliases:
        low_extra = {k.strip().lower(): v.strip().lower() for k, v in extra_aliases.items()}
        if s in low_extra:
            return CANONICAL_ALIASES.get(low_extra[s], low_extra[s])
    return CANONICAL_ALIASES.get(s, s)


def _canon_score_map(
    raw: dict[str, float], aliases: dict[str, str] | None = None
) -> dict[str, float]:
    return {canonicalize_skill(k, aliases): float(v) for k, v in raw.items()}


def resolve_model_scores(
    spec_model: str,
    spec_model_key: str | None,
    spec_scores: dict[str, float],
    model_scores: dict[str, dict[str, float]],
    extra_aliases: dict[str, str] | None = None,
) -> dict[str, float]:
    """Resolve scores: spec_scores > model_key > exact/base model > substr match."""
    resolved: dict[str, float] = {}
    m_clean = spec_model.removesuffix(":free").strip().lower()
    m_base = m_clean.split("/")[-1]
    norm_scores = {
        k.removesuffix(":free").strip().lower(): _canon_score_map(sc, extra_aliases)
        for k, sc in model_scores.items()
    }

    for k in sorted(norm_scores, key=len):
        if k in m_clean or k in m_base:
            resolved.update(norm_scores[k])
    for k in (m_clean, m_base):
        if k in norm_scores:
            resolved.update(norm_scores[k])
    if spec_model_key:
        mk = spec_model_key.removesuffix(":free").strip().lower()
        if mk in norm_scores:
            resolved.update(norm_scores[mk])
    resolved.update(_canon_score_map(spec_scores, extra_aliases))
    return resolved


def detect_skills(
    prompt: str = "",
    task_kind: str | None = None,
    workdir: Path | None = None,
    cfg: SkillRoutingConfig | None = None,
) -> tuple[str, ...]:
    """Detect required skills from code blocks, extensions, keywords, and workdir markers."""
    if task_kind and task_kind.lower() in STRUCTURAL_TASKS:
        return ()
    c = cfg or SkillRoutingConfig()
    found: set[str] = set()
    for lang in re.findall(r"```([a-zA-Z0-9_\-]+)", prompt):
        found.add(canonicalize_skill(lang, c.aliases))
    _scan_extensions(prompt, c, found)
    _scan_keywords(prompt, c, found)
    if not found and workdir and workdir.exists():
        _scan_workdir(workdir, c, found)
    return tuple(sorted(found))


def _scan_extensions(prompt: str, c: SkillRoutingConfig, found: set[str]) -> None:
    ext_map = {**DEFAULT_EXTENSIONS, **c.extension_skills}
    masked = prompt
    for ext, sk in sorted(ext_map.items(), key=lambda x: -len(x[0])):
        pat = re.compile(rf"(?<!\.){re.escape(ext)}(?![A-Za-z0-9_])", re.IGNORECASE)
        if pat.search(masked):
            found.add(canonicalize_skill(sk, c.aliases))
            masked = pat.sub(" ", masked)


def _scan_keywords(prompt: str, c: SkillRoutingConfig, found: set[str]) -> None:
    words = set(re.findall(r"\b[a-zA-Z0-9_\-]+\b", prompt.lower()))
    alias_pool = {k: v for k, v in CANONICAL_ALIASES.items() if k != "review"}
    for k, v in c.aliases.items():
        alias_pool[k.strip().lower()] = v
    for alias_key, canon in alias_pool.items():
        if len(alias_key) > 2 and alias_key in words:
            found.add(canonicalize_skill(canon, c.aliases))


def _scan_workdir(workdir: Path, cfg: SkillRoutingConfig, found: set[str]) -> None:
    """Scan workdir/app then workdir markers without touching monitor/."""
    markers = {**DEFAULT_MARKERS, **cfg.marker_skills}
    for root in (workdir / "app", workdir):
        if not root.is_dir():
            continue
        matched = False
        for marker, sk in markers.items():
            if (root / marker).exists():
                found.add(canonicalize_skill(sk, cfg.aliases))
                matched = True
        if matched:
            return


def _eval_skills(
    skills: tuple[str, ...], scores: dict[str, float], c: SkillRoutingConfig
) -> tuple[float, bool, bool, float | None]:
    explicit: list[float] = []
    vals: list[float] = []
    for sk in skills:
        canon = canonicalize_skill(sk, c.aliases)
        if canon in scores:
            v = scores[canon]
            explicit.append(v)
            vals.append(v)
        else:
            vals.append(c.default_score)
    avg_score = sum(vals) / len(vals)
    min_exp = min(explicit) if explicit else None
    is_weak = min_exp is not None and min_exp < c.weak_threshold
    return avg_score, is_weak, bool(explicit), min_exp


def score_candidate(
    spec: ExecutorSpec,
    skills: tuple[str, ...],
    cfg: SkillRoutingConfig | None = None,
    is_role_match: bool = False,
    is_benchmarked: bool = False,
    model_scores: dict[str, dict[str, float]] | None = None,
) -> CandidateScore:
    """Score candidate executor against requested skills."""
    c = cfg or SkillRoutingConfig()
    scores = resolve_model_scores(
        spec.model, spec.model_key, spec.scores, model_scores or {}, c.aliases
    )
    if not skills:
        return CandidateScore(
            spec=spec, final_score=c.default_score, is_weak=False, is_rated=bool(scores)
        )
    avg, is_weak, is_rated, min_exp = _eval_skills(skills, scores, c)
    final = avg + (c.role_bonus if is_role_match else 0.0)
    final += c.benchmark_bonus if is_benchmarked else 0.0
    return CandidateScore(
        spec=spec,
        final_score=final,
        is_weak=is_weak,
        is_rated=is_rated,
        min_explicit_score=min_exp,
    )
