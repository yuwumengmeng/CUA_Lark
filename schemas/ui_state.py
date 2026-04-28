"""Member A UI state schema shared with planner and executor.

Responsibility boundary: this file defines the stable data shape returned by
`get_ui_state()` and the JSON artifact path rules. It does not capture
screenshots, run OCR/UIA, build candidates, or call VLM APIs.

Public interfaces:
`PageSummary`
    Minimal rule-based page summary with `top_texts / has_search_box /
    has_modal / candidate_count`.
`UIState`
    Top-level state with required `screenshot_path / candidates /
    page_summary` and optional `vlm_result / screen_meta`.
`ui_state_artifact_path(...)` / `save_ui_state_json(...)`
    Generate and write JSON artifacts under
    `artifacts/runs/<run_id>/ui_state_step_xxx_<phase>.json`.

Collaboration rules:
1. B/C may continue consuming only `screenshot_path/candidates/page_summary`.
2. `vlm_result` is optional and advisory. When `ok=False`, callers should use
   OCR/UIA/Candidate Builder fallback results.
3. Candidate `bbox/center` remains the executor-facing coordinate contract.
4. `screen_meta` is optional for legacy unit construction, but `get_ui_state()`
   should populate it for real capture-based flows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .page_semantics import PageSemantics
from .ui_candidate import UICandidate, UISchemaError


SUPPORTED_UI_STATE_PHASES = {"before", "after", "manual"}


@dataclass(frozen=True, slots=True)
class PageSummary:
    top_texts: list[str] = field(default_factory=list)
    has_search_box: bool = False
    has_modal: bool = False
    candidate_count: int = 0
    semantic_summary: PageSemantics | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.top_texts, list):
            raise UISchemaError("PageSummary top_texts must be a list")
        if any(not isinstance(text, str) for text in self.top_texts):
            raise UISchemaError("PageSummary top_texts items must be strings")
        if not isinstance(self.has_search_box, bool):
            raise UISchemaError("PageSummary has_search_box must be a boolean")
        if not isinstance(self.has_modal, bool):
            raise UISchemaError("PageSummary has_modal must be a boolean")
        if isinstance(self.candidate_count, bool) or not isinstance(self.candidate_count, int):
            raise UISchemaError("PageSummary candidate_count must be an integer")
        if self.candidate_count < 0:
            raise UISchemaError("PageSummary candidate_count must be non-negative")
        if self.semantic_summary is not None and not isinstance(self.semantic_summary, PageSemantics):
            raise UISchemaError("PageSummary semantic_summary must be PageSemantics when provided")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PageSummary":
        return cls(
            top_texts=list(data.get("top_texts", [])),
            has_search_box=_bool_field(data.get("has_search_box", False), "has_search_box"),
            has_modal=_bool_field(data.get("has_modal", False), "has_modal"),
            candidate_count=_non_negative_int_field(
                data.get("candidate_count", 0),
                "candidate_count",
            ),
            semantic_summary=(
                PageSemantics.from_dict(data.get("semantic_summary"))
                if data.get("semantic_summary") is not None
                else None
            ),
        )

    @classmethod
    def from_candidates(
        cls,
        candidates: list[UICandidate],
        *,
        top_text_limit: int = 20,
    ) -> "PageSummary":
        texts: list[str] = []
        for candidate in candidates:
            if candidate.text and candidate.text not in texts:
                texts.append(candidate.text)
        top_texts = texts[:top_text_limit]
        has_search_box = (
            any("搜索" in text for text in texts)
            or _contains_split_search_chars(texts)
            or _contains_search_shortcut(texts)
            or any(candidate.editable for candidate in candidates)
        )
        return cls(
            top_texts=top_texts,
            has_search_box=has_search_box,
            has_modal=False,
            candidate_count=len(candidates),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "top_texts": self.top_texts,
            "has_search_box": self.has_search_box,
            "has_modal": self.has_modal,
            "candidate_count": self.candidate_count,
        }
        if self.semantic_summary is not None:
            payload["semantic_summary"] = self.semantic_summary.to_dict()
        return payload

    def with_semantic_summary(self, semantic_summary: PageSemantics | None) -> "PageSummary":
        return PageSummary(
            top_texts=self.top_texts,
            has_search_box=self.has_search_box,
            has_modal=self.has_modal,
            candidate_count=self.candidate_count,
            semantic_summary=semantic_summary,
        )


@dataclass(frozen=True, slots=True)
class ScreenMeta:
    screenshot_size: list[int]
    screen_resolution: list[int]
    window_rect: list[int] | None
    screen_origin: list[int]
    coordinate_space: str = "screen"
    capture_target: str = "fullscreen"
    monitor_index: int = 1
    monitor_count: int = 1
    is_multi_monitor: bool = False
    dpi_scale: float | None = None
    candidate_coordinate_source_counts: dict[str, int] = field(default_factory=dict)
    negative_coordinate_candidate_count: int = 0
    out_of_screenshot_candidate_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "screenshot_size",
            _two_positive_ints(self.screenshot_size, "ScreenMeta screenshot_size"),
        )
        object.__setattr__(
            self,
            "screen_resolution",
            _two_positive_ints(self.screen_resolution, "ScreenMeta screen_resolution"),
        )
        if self.window_rect is not None:
            object.__setattr__(
                self,
                "window_rect",
                _rect_field(self.window_rect, "ScreenMeta window_rect"),
            )
        object.__setattr__(
            self,
            "screen_origin",
            _two_ints(self.screen_origin, "ScreenMeta screen_origin"),
        )
        if not self.coordinate_space.strip():
            raise UISchemaError("ScreenMeta coordinate_space cannot be empty")
        if self.capture_target not in {"fullscreen", "window"}:
            raise UISchemaError("ScreenMeta capture_target must be fullscreen or window")
        object.__setattr__(
            self,
            "monitor_index",
            _non_negative_int_field(self.monitor_index, "ScreenMeta monitor_index"),
        )
        object.__setattr__(
            self,
            "monitor_count",
            _positive_int_field(self.monitor_count, "ScreenMeta monitor_count"),
        )
        if not isinstance(self.is_multi_monitor, bool):
            raise UISchemaError("ScreenMeta is_multi_monitor must be a boolean")
        if self.dpi_scale is not None:
            object.__setattr__(self, "dpi_scale", _positive_float_field(self.dpi_scale, "ScreenMeta dpi_scale"))
        if not isinstance(self.candidate_coordinate_source_counts, dict):
            raise UISchemaError("ScreenMeta candidate_coordinate_source_counts must be a dict")
        normalized_counts: dict[str, int] = {}
        for key, value in self.candidate_coordinate_source_counts.items():
            source = str(key).strip()
            if not source:
                raise UISchemaError("ScreenMeta candidate coordinate source cannot be empty")
            normalized_counts[source] = _non_negative_int_field(
                value,
                f"ScreenMeta candidate_coordinate_source_counts[{source}]",
            )
        object.__setattr__(self, "candidate_coordinate_source_counts", normalized_counts)
        object.__setattr__(
            self,
            "negative_coordinate_candidate_count",
            _non_negative_int_field(
                self.negative_coordinate_candidate_count,
                "ScreenMeta negative_coordinate_candidate_count",
            ),
        )
        object.__setattr__(
            self,
            "out_of_screenshot_candidate_count",
            _non_negative_int_field(
                self.out_of_screenshot_candidate_count,
                "ScreenMeta out_of_screenshot_candidate_count",
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScreenMeta":
        return cls(
            screenshot_size=list(data.get("screenshot_size", [])),
            screen_resolution=list(data.get("screen_resolution", [])),
            window_rect=list(data["window_rect"]) if data.get("window_rect") is not None else None,
            screen_origin=list(data.get("screen_origin", [])),
            coordinate_space=str(data.get("coordinate_space") or "screen"),
            capture_target=str(data.get("capture_target") or "fullscreen"),
            monitor_index=_non_negative_int_field(data.get("monitor_index", 1), "monitor_index"),
            monitor_count=_positive_int_field(data.get("monitor_count", 1), "monitor_count"),
            is_multi_monitor=_bool_field(data.get("is_multi_monitor", False), "is_multi_monitor"),
            dpi_scale=(
                _positive_float_field(data.get("dpi_scale"), "dpi_scale")
                if data.get("dpi_scale") is not None
                else None
            ),
            candidate_coordinate_source_counts=dict(
                data.get("candidate_coordinate_source_counts", {})
            ),
            negative_coordinate_candidate_count=_non_negative_int_field(
                data.get("negative_coordinate_candidate_count", 0),
                "negative_coordinate_candidate_count",
            ),
            out_of_screenshot_candidate_count=_non_negative_int_field(
                data.get("out_of_screenshot_candidate_count", 0),
                "out_of_screenshot_candidate_count",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "screenshot_size": self.screenshot_size,
            "screen_resolution": self.screen_resolution,
            "window_rect": self.window_rect,
            "screen_origin": self.screen_origin,
            "coordinate_space": self.coordinate_space,
            "capture_target": self.capture_target,
            "monitor_index": self.monitor_index,
            "monitor_count": self.monitor_count,
            "is_multi_monitor": self.is_multi_monitor,
            "dpi_scale": self.dpi_scale,
            "candidate_coordinate_source_counts": dict(self.candidate_coordinate_source_counts),
            "negative_coordinate_candidate_count": self.negative_coordinate_candidate_count,
            "out_of_screenshot_candidate_count": self.out_of_screenshot_candidate_count,
        }


@dataclass(frozen=True, slots=True)
class UIState:
    screenshot_path: str
    candidates: list[UICandidate]
    page_summary: PageSummary
    vlm_result: dict[str, Any] | None = None
    screen_meta: ScreenMeta | None = None
    document_region_summary: dict[str, Any] | None = None
    perception_health: dict[str, Any] | None = None
    candidate_stats: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.screenshot_path.strip():
            raise UISchemaError("UIState screenshot_path cannot be empty")
        if not isinstance(self.candidates, list):
            raise UISchemaError("UIState candidates must be a list")
        if any(not isinstance(candidate, UICandidate) for candidate in self.candidates):
            raise UISchemaError("UIState candidates items must be UICandidate")
        if not isinstance(self.page_summary, PageSummary):
            raise UISchemaError("UIState page_summary must be PageSummary")
        if self.vlm_result is not None and not isinstance(self.vlm_result, dict):
            raise UISchemaError("UIState vlm_result must be a dict when provided")
        if self.screen_meta is not None and not isinstance(self.screen_meta, ScreenMeta):
            raise UISchemaError("UIState screen_meta must be ScreenMeta when provided")
        if self.document_region_summary is not None and not isinstance(self.document_region_summary, dict):
            raise UISchemaError("UIState document_region_summary must be a dict when provided")
        if self.perception_health is not None and not isinstance(self.perception_health, dict):
            raise UISchemaError("UIState perception_health must be a dict when provided")
        if self.candidate_stats is not None and not isinstance(self.candidate_stats, dict):
            raise UISchemaError("UIState candidate_stats must be a dict when provided")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UIState":
        if "screenshot_path" not in data:
            raise UISchemaError("UIState data is missing field: screenshot_path")
        if "candidates" not in data:
            raise UISchemaError("UIState data is missing field: candidates")
        if "page_summary" not in data:
            raise UISchemaError("UIState data is missing field: page_summary")
        candidates = [UICandidate.from_dict(candidate) for candidate in data["candidates"]]
        return cls(
            screenshot_path=str(data["screenshot_path"]),
            candidates=candidates,
            page_summary=PageSummary.from_dict(data["page_summary"]),
            vlm_result=dict(data["vlm_result"]) if data.get("vlm_result") is not None else None,
            screen_meta=(
                ScreenMeta.from_dict(data["screen_meta"])
                if data.get("screen_meta") is not None
                else None
            ),
            document_region_summary=(
                dict(data["document_region_summary"])
                if data.get("document_region_summary") is not None
                else None
            ),
            perception_health=(
                dict(data["perception_health"])
                if data.get("perception_health") is not None
                else None
            ),
            candidate_stats=(
                dict(data["candidate_stats"])
                if data.get("candidate_stats") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "screenshot_path": self.screenshot_path,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "page_summary": self.page_summary.to_dict(),
        }
        if self.vlm_result is not None:
            payload["vlm_result"] = self.vlm_result
        if self.screen_meta is not None:
            payload["screen_meta"] = self.screen_meta.to_dict()
        if self.document_region_summary is not None:
            payload["document_region_summary"] = self.document_region_summary
        if self.perception_health is not None:
            payload["perception_health"] = self.perception_health
        if self.candidate_stats is not None:
            payload["candidate_stats"] = self.candidate_stats
        return payload


def ui_state_artifact_path(
    *,
    run_id: str,
    step_idx: int,
    phase: str = "before",
    artifact_root: str | Path | None = None,
) -> Path:
    normalized_run_id = _normalize_run_id(run_id)
    normalized_step_idx = _non_negative_int_field(step_idx, "step_idx")
    normalized_phase = _normalize_phase(phase)
    root = Path(artifact_root) if artifact_root is not None else Path("artifacts") / "runs"
    return root / normalized_run_id / f"ui_state_step_{normalized_step_idx:03d}_{normalized_phase}.json"


def save_ui_state_json(
    ui_state: UIState | Mapping[str, Any],
    *,
    run_id: str,
    step_idx: int,
    phase: str = "before",
    artifact_root: str | Path | None = None,
) -> Path:
    normalized = ui_state if isinstance(ui_state, UIState) else UIState.from_dict(ui_state)
    path = ui_state_artifact_path(
        run_id=run_id,
        step_idx=step_idx,
        phase=phase,
        artifact_root=artifact_root,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalized.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def screen_meta_artifact_path(
    *,
    run_id: str,
    step_idx: int,
    phase: str = "before",
    artifact_root: str | Path | None = None,
) -> Path:
    normalized_run_id = _normalize_run_id(run_id)
    normalized_step_idx = _non_negative_int_field(step_idx, "step_idx")
    normalized_phase = _normalize_phase(phase)
    root = Path(artifact_root) if artifact_root is not None else Path("artifacts") / "runs"
    return root / normalized_run_id / f"screen_meta_step_{normalized_step_idx:03d}_{normalized_phase}.json"


def save_screen_meta_json(
    screen_meta: ScreenMeta | Mapping[str, Any],
    *,
    run_id: str,
    step_idx: int,
    phase: str = "before",
    artifact_root: str | Path | None = None,
) -> Path:
    normalized = screen_meta if isinstance(screen_meta, ScreenMeta) else ScreenMeta.from_dict(screen_meta)
    path = screen_meta_artifact_path(
        run_id=run_id,
        step_idx=step_idx,
        phase=phase,
        artifact_root=artifact_root,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalized.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _normalize_run_id(run_id: str) -> str:
    text = run_id.strip()
    if not text:
        raise UISchemaError("run_id cannot be empty")
    if any(char in text for char in '<>:"/\\|?*'):
        raise UISchemaError(f"run_id contains invalid path characters: {text}")
    return text


def _normalize_phase(phase: str) -> str:
    if phase not in SUPPORTED_UI_STATE_PHASES:
        allowed = ", ".join(sorted(SUPPORTED_UI_STATE_PHASES))
        raise UISchemaError(f"phase must be one of: {allowed}")
    return phase


def _bool_field(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise UISchemaError(f"{field_name} must be a boolean")
    return value


def _non_negative_int_field(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UISchemaError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int_field(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UISchemaError(f"{field_name} must be a positive integer")
    return value


def _positive_float_field(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise UISchemaError(f"{field_name} must be a positive number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise UISchemaError(f"{field_name} must be a positive number") from exc
    if normalized <= 0:
        raise UISchemaError(f"{field_name} must be a positive number")
    return normalized


def _two_positive_ints(value: Any, field_name: str) -> list[int]:
    result = _two_ints(value, field_name)
    if result[0] <= 0 or result[1] <= 0:
        raise UISchemaError(f"{field_name} values must be positive")
    return result


def _two_ints(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 2:
        raise UISchemaError(f"{field_name} must contain two integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise UISchemaError(f"{field_name} values must be integers")
    return [value[0], value[1]]


def _rect_field(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise UISchemaError(f"{field_name} must contain four integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise UISchemaError(f"{field_name} values must be integers")
    if value[2] <= value[0] or value[3] <= value[1]:
        raise UISchemaError(f"{field_name} must satisfy x2 > x1 and y2 > y1")
    return [value[0], value[1], value[2], value[3]]


def _contains_split_search_chars(texts: list[str]) -> bool:
    text_set = {text.strip() for text in texts if text.strip()}
    return "搜" in text_set and "索" in text_set


def _contains_search_shortcut(texts: list[str]) -> bool:
    normalized = [_normalize_search_token(text) for text in texts if text.strip()]
    has_ctrl = any("ctrl" in token for token in normalized)
    has_k = any(token == "k" for token in normalized)
    return has_ctrl and has_k


def _normalize_search_token(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum())
