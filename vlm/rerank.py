"""Member A VLM candidate rerank and visual fallback.

Responsibility boundary: this file reranks existing candidates based on VLM
structured output, deprioritizes shell controls, and builds synthetic visual
candidates when VLM proposes targets not found by OCR/UIA. It does not call
VLM APIs, make planner decisions, or execute actions.

Public interfaces:
`rerank_candidates(...)`
    Reorder and annotate candidates based on VLMStructuredOutput. Returns a
    new list with VLM-recommended candidates first, shell controls demoted,
    and VLM confidence/reason attached.
`build_visual_candidates(...)`
    Convert VLM-proposed visual_candidates from VLMStructuredOutput into
    UICandidate objects with source=vlm_visual.
`is_shell_control(...)`
    Heuristic to detect Chromium/browser shell controls that should be
    deprioritized.
`SHELL_CONTROL_ROLES` / `SHELL_CONTROL_TEXTS`
    Configurable sets for shell control detection.

Data contract:
1. rerank_candidates never drops candidates; it only reorders and annotates.
2. Visual synthetic candidates get source=vlm_visual and coordinate_type.
3. Shell controls are moved to the end but never removed.
4. VLM confidence and reason are attached as vlm_reason on each candidate.

Collaboration rules:
1. Member B reads the reranked candidate list in order; first match wins.
2. Member C clicks candidate.center regardless of source.
3. When structured.uncertain is True, B should still use rule-based scoring
   but may reference reranked_candidate_ids as a hint.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from schemas import (
    COORDINATE_TYPE_SCREENSHOT_PIXEL,
    SOURCE_VLM_VISUAL,
    UICandidate,
    bbox_center,
)

from .output_schema import VLMStructuredOutput


SHELL_CONTROL_ROLES = frozenset({
    "TitleBar",
    "MenuBar",
    "ScrollBar",
    "StatusBar",
    "ToolBar",
    "Thumb",
    "SplitButton",
    "SystemMenuBar",
})

SHELL_CONTROL_TEXTS = frozenset({
    "最小化",
    "最大化",
    "关闭",
    "还原",
    "Minimize",
    "Maximize",
    "Close",
    "Restore",
    "Chrome",
    "Chromium",
    "Microsoft Edge",
    "Edge",
})

SYNTHETIC_CANDIDATE_PREFIX = "vlm_vis"


def rerank_candidates(
    candidates: Sequence[UICandidate],
    structured: VLMStructuredOutput,
    *,
    demote_shell: bool = True,
) -> list[UICandidate]:
    if not candidates or not structured.parse_ok:
        return list(candidates)

    candidate_map = {candidate.id: candidate for candidate in candidates}

    ordered: list[UICandidate] = []
    seen_ids: set[str] = set()

    target_id = structured.target_candidate_id
    if target_id and target_id in candidate_map:
        target = candidate_map[target_id]
        ordered.append(_annotate_candidate(target, structured.confidence, structured.reason))
        seen_ids.add(target_id)

    for cid in structured.reranked_candidate_ids:
        if cid in seen_ids or cid not in candidate_map:
            continue
        candidate = candidate_map[cid]
        rank_position = len(seen_ids)
        rank_confidence = max(0.1, structured.confidence - rank_position * 0.1)
        ordered.append(_annotate_candidate(candidate, rank_confidence, structured.reason))
        seen_ids.add(cid)

    remaining = [candidate for candidate in candidates if candidate.id not in seen_ids]

    if demote_shell:
        normal = [c for c in remaining if not is_shell_control(c)]
        shell = [c for c in remaining if is_shell_control(c)]
        remaining = normal + shell

    ordered.extend(remaining)
    return ordered


def build_visual_candidates(
    structured: VLMStructuredOutput,
    *,
    screenshot_width: int | None = None,
    screenshot_height: int | None = None,
) -> list[UICandidate]:
    visual_raw = structured.visual_candidates
    if not visual_raw:
        return []

    candidates: list[UICandidate] = []
    for index, raw in enumerate(visual_raw):
        candidate = _build_single_visual_candidate(
            raw=raw,
            index=index,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def is_shell_control(candidate: UICandidate | Mapping[str, Any]) -> bool:
    if isinstance(candidate, UICandidate):
        role = candidate.role
        text = candidate.text
    else:
        role = str(candidate.get("role", ""))
        text = str(candidate.get("text", ""))

    if any(shell_role in role for shell_role in SHELL_CONTROL_ROLES):
        return True
    if text.strip() in SHELL_CONTROL_TEXTS:
        return True
    return False


def _annotate_candidate(
    candidate: UICandidate,
    vlm_confidence: float,
    vlm_reason: str,
) -> UICandidate:
    return replace(
        candidate,
        confidence=max(candidate.confidence, vlm_confidence),
        vlm_reason=vlm_reason,
    )


def _build_single_visual_candidate(
    *,
    raw: Mapping[str, Any],
    index: int,
    screenshot_width: int | None,
    screenshot_height: int | None,
) -> UICandidate | None:
    bbox_raw = raw.get("bbox")
    center_raw = raw.get("center")
    if bbox_raw is None and center_raw is None:
        return None

    bbox = _resolve_bbox(raw, screenshot_width, screenshot_height)
    if bbox is None:
        return None

    center = bbox_center(bbox)
    if center_raw is not None:
        resolved_center = _resolve_center(center_raw, screenshot_width, screenshot_height)
        if resolved_center is not None:
            center = resolved_center

    candidate_id = str(raw.get("id") or f"{SYNTHETIC_CANDIDATE_PREFIX}_{index + 1:04d}")
    role = str(raw.get("role") or "unknown")
    confidence = _safe_float(raw.get("confidence"), 0.5)
    reason = str(raw.get("reason") or "")
    coordinate_type = str(raw.get("coordinate_type") or COORDINATE_TYPE_SCREENSHOT_PIXEL)

    try:
        return UICandidate(
            id=candidate_id,
            text="",
            bbox=bbox,
            center=center,
            role=role,
            clickable=True,
            editable=False,
            source=SOURCE_VLM_VISUAL,
            confidence=confidence,
            coordinate_type=coordinate_type,
            vlm_reason=reason,
        )
    except Exception:
        return None


def _resolve_bbox(
    raw: Mapping[str, Any],
    screenshot_width: int | None,
    screenshot_height: int | None,
) -> list[int] | None:
    bbox_raw = raw.get("bbox")
    if bbox_raw is not None:
        try:
            return [int(v) for v in bbox_raw]
        except (TypeError, ValueError):
            pass

    center_raw = raw.get("center")
    if center_raw is None:
        return None

    cx, cy = _resolve_center(center_raw, screenshot_width, screenshot_height) or (0, 0)
    half_w = 24
    half_h = 24
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


def _resolve_center(
    center_raw: Any,
    screenshot_width: int | None,
    screenshot_height: int | None,
) -> tuple[int, int] | None:
    if isinstance(center_raw, (list, tuple)) and len(center_raw) == 2:
        try:
            return [int(center_raw[0]), int(center_raw[1])]
        except (TypeError, ValueError):
            pass

    if isinstance(center_raw, dict):
        x = _safe_float(center_raw.get("x"), None)
        y = _safe_float(center_raw.get("y"), None)
        if x is not None and y is not None:
            return [int(x), int(y)]

    return None


def _safe_float(value: Any, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
