"""项目共享 schema 包入口。

这个包负责放 A/B/C 三个成员都需要遵守的数据结构契约。当前已定义
成员 A 输出给成员 B/C 的 UI 候选元素、页面摘要、页面状态和坐标规则，
以及成员 C 输出给成员 B 的动作执行结果协议。
"""

from .action_result import (
    ACTION_CLICK,
    ACTION_DOUBLE_CLICK,
    ACTION_HOTKEY,
    ACTION_SCROLL,
    ACTION_TYPE,
    ACTION_WAIT,
    SUPPORTED_EXECUTOR_ACTION_TYPES,
    ActionResult,
    ActionResultSchemaError,
    action_result_from_exception,
    utc_now,
)
from .ui_candidate import (
    SOURCE_MERGED,
    SOURCE_OCR,
    SOURCE_UIA,
    UISchemaError,
    UICandidate,
    bbox_center,
    candidate_from_ocr,
    candidate_from_uia,
    offset_bbox,
    resolve_click_point,
)
from .ui_state import PageSummary, UIState, save_ui_state_json, ui_state_artifact_path

__all__ = [
    "ACTION_CLICK",
    "ACTION_DOUBLE_CLICK",
    "ACTION_HOTKEY",
    "ACTION_SCROLL",
    "ACTION_TYPE",
    "ACTION_WAIT",
    "SOURCE_MERGED",
    "SOURCE_OCR",
    "SOURCE_UIA",
    "SUPPORTED_EXECUTOR_ACTION_TYPES",
    "ActionResult",
    "ActionResultSchemaError",
    "PageSummary",
    "UICandidate",
    "UIState",
    "UISchemaError",
    "bbox_center",
    "candidate_from_ocr",
    "candidate_from_uia",
    "offset_bbox",
    "resolve_click_point",
    "action_result_from_exception",
    "save_ui_state_json",
    "ui_state_artifact_path",
    "utc_now",
]
