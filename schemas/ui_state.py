"""成员 A 输出给 B/C 的页面状态 schema。

职责边界：
本文件只定义 `get_ui_state()` 最终返回值的数据格式和 UI state JSON
落盘路径规则。它不负责截图、不负责 OCR/UIA、不负责构建 candidates。

公开接口：
`PageSummary`
    页面摘要结构，稳定字段为 `top_texts / has_search_box / has_modal /
    candidate_count`。
`UIState`
    成员 A 对 B/C 的页面状态输出，稳定字段为 `screenshot_path /
    candidates / page_summary`。
`ui_state_artifact_path(...)`
    生成结构化页面状态 JSON 的统一路径。
`save_ui_state_json(...)`
    将 `UIState` 或同格式 dict 保存为 JSON。

路径规则：
页面状态 JSON 统一保存到：
`artifacts/runs/<run_id>/ui_state_step_xxx_<phase>.json`
例如：
`artifacts/runs/run_demo/ui_state_step_001_before.json`

协作规则：
1. B 只消费 `screenshot_path/candidates/page_summary`，不直接解析 OCR 原始结果。
2. C 只从 candidate 中消费 `bbox/center`，不自行推断点击点。
3. `page_summary` 第一周期保持最小字段，不在未对齐前随意删除或改名。
4. 真实运行产生的 `ui_state_step_xxx_<phase>.json` 属于 artifact，不提交 Git。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .ui_candidate import UICandidate, UISchemaError


SUPPORTED_UI_STATE_PHASES = {"before", "after", "manual"}


@dataclass(frozen=True, slots=True)
class PageSummary:
    top_texts: list[str] = field(default_factory=list)
    has_search_box: bool = False
    has_modal: bool = False
    candidate_count: int = 0

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
        return {
            "top_texts": self.top_texts,
            "has_search_box": self.has_search_box,
            "has_modal": self.has_modal,
            "candidate_count": self.candidate_count,
        }


@dataclass(frozen=True, slots=True)
class UIState:
    screenshot_path: str
    candidates: list[UICandidate]
    page_summary: PageSummary

    def __post_init__(self) -> None:
        if not self.screenshot_path.strip():
            raise UISchemaError("UIState screenshot_path cannot be empty")
        if not isinstance(self.candidates, list):
            raise UISchemaError("UIState candidates must be a list")
        if any(not isinstance(candidate, UICandidate) for candidate in self.candidates):
            raise UISchemaError("UIState candidates items must be UICandidate")
        if not isinstance(self.page_summary, PageSummary):
            raise UISchemaError("UIState page_summary must be PageSummary")

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
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "screenshot_path": self.screenshot_path,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "page_summary": self.page_summary.to_dict(),
        }


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
