"""成员 A 输出给 B/C 的统一候选元素 schema。

职责边界：
本文件只定义候选元素的数据格式、bbox/center 坐标规则和默认点击点解析。
它不做 OCR、不做 UIA、不做候选合并排序。OCR、UIA、Candidate Builder
最终都必须把结果转换成这里的 `UICandidate`，再交给 planner/executor。

公开接口：
`UICandidate`
    统一候选元素结构，稳定字段为 `id / text / bbox / center / role /
    clickable / editable / source / confidence`。
`bbox_center(bbox)`
    根据 `[x1, y1, x2, y2]` 计算中心点 `[cx, cy]`。
`resolve_click_point(candidate)`
    第一周期默认返回 candidate 的中心点，供成员 C 执行点击。
`candidate_from_ocr(...)`
    把 OCR 文本块字段转换成首版 candidate，并把截图局部 bbox 转成屏幕绝对 bbox。
`candidate_from_uia(...)`
    把 UIA 元素字段转换成首版 candidate。UIA bbox 默认已经是屏幕绝对坐标。
`offset_bbox(...)`
    按截图 `screen_origin` 把局部 bbox 平移到屏幕绝对坐标。

坐标规则：
`UICandidate.bbox` 统一使用屏幕绝对坐标，格式为 `[x1, y1, x2, y2]`。
`center` 统一使用整数中心点，格式为 `[(x1 + x2) // 2, (y1 + y2) // 2]`。
第一周期默认点击 `center`。如果后续发现系统性偏移，再扩展
`resolve_click_point()`，不要让 B/C 自己在外部临时改坐标。

转换规则：
1. OCR 原始 bbox 来自截图局部坐标，转 candidate 时必须加上截图
   `screen_origin`。
2. UIA 原始 bbox 来自 Windows UI Automation，默认已经是屏幕绝对坐标，
   转 candidate 时不要再加 origin。
3. B/C 只消费 `UICandidate.bbox/center`，不再自行判断来源坐标系。

协作规则：
1. 对外 dict 字段名固定，不允许把 `bbox` 改成 `box/rect/bounds`。
2. `source` 只能是 `ocr`、`uia`、`merged`。
3. 每个 candidate 的 `id` 至少在当前 step 内唯一。
4. B 使用 `id/text/role/clickable/editable/confidence` 做规划。
5. C 使用 `bbox/center` 或 `resolve_click_point()` 做点击。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SOURCE_OCR = "ocr"
SOURCE_UIA = "uia"
SOURCE_MERGED = "merged"
SUPPORTED_SOURCES = {SOURCE_OCR, SOURCE_UIA, SOURCE_MERGED}


class UISchemaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class UICandidate:
    id: str
    text: str
    bbox: list[int]
    center: list[int] | None = None
    role: str = "unknown"
    clickable: bool = False
    editable: bool = False
    source: str = SOURCE_OCR
    confidence: float = 0.0

    def __post_init__(self) -> None:
        candidate_id = self.id.strip()
        if not candidate_id:
            raise UISchemaError("UICandidate id cannot be empty")
        text = self.text.strip()
        bbox = normalize_bbox(self.bbox)
        center = normalize_center(self.center) if self.center is not None else bbox_center(bbox)
        if not bbox[0] <= center[0] <= bbox[2] or not bbox[1] <= center[1] <= bbox[3]:
            raise UISchemaError("UICandidate center must be inside bbox")
        role = self.role.strip() or "unknown"
        if not isinstance(self.clickable, bool):
            raise UISchemaError("UICandidate clickable must be a boolean")
        if not isinstance(self.editable, bool):
            raise UISchemaError("UICandidate editable must be a boolean")
        if self.source not in SUPPORTED_SOURCES:
            allowed = ", ".join(sorted(SUPPORTED_SOURCES))
            raise UISchemaError(f"UICandidate source must be one of: {allowed}")
        if not 0.0 <= self.confidence <= 1.0:
            raise UISchemaError("UICandidate confidence must be between 0.0 and 1.0")
        object.__setattr__(self, "id", candidate_id)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "role", role)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UICandidate":
        required = ("id", "text", "bbox", "source", "confidence")
        missing = [field for field in required if field not in data]
        if missing:
            raise UISchemaError(f"UICandidate data is missing fields: {', '.join(missing)}")
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            bbox=list(data["bbox"]),
            center=list(data["center"]) if data.get("center") is not None else None,
            role=str(data.get("role") or "unknown"),
            clickable=_bool_field(data.get("clickable", False), "clickable"),
            editable=_bool_field(data.get("editable", False), "editable"),
            source=str(data["source"]),
            confidence=_float_field(data["confidence"], "confidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "bbox": self.bbox,
            "center": self.center,
            "role": self.role,
            "clickable": self.clickable,
            "editable": self.editable,
            "source": self.source,
            "confidence": self.confidence,
        }


def candidate_from_ocr(
    *,
    candidate_id: str,
    text: str,
    bbox: list[int],
    confidence: float,
    screen_origin: Sequence[int] | None = None,
) -> UICandidate:
    return UICandidate(
        id=candidate_id,
        text=text,
        bbox=offset_bbox(bbox, screen_origin),
        role="text",
        clickable=False,
        editable=False,
        source=SOURCE_OCR,
        confidence=confidence,
    )


def candidate_from_uia(
    *,
    candidate_id: str,
    name: str,
    role: str,
    bbox: list[int],
    enabled: bool,
    visible: bool,
    confidence: float = 0.85,
) -> UICandidate:
    normalized_role = role.strip() or "unknown"
    return UICandidate(
        id=candidate_id,
        text=name,
        bbox=bbox,
        role=normalized_role,
        clickable=enabled and visible and _uia_role_clickable(normalized_role),
        editable=enabled and visible and _uia_role_editable(normalized_role),
        source=SOURCE_UIA,
        confidence=confidence if enabled and visible else min(confidence, 0.5),
    )


def bbox_center(bbox: list[int]) -> list[int]:
    x1, y1, x2, y2 = normalize_bbox(bbox)
    return [(x1 + x2) // 2, (y1 + y2) // 2]


def resolve_click_point(candidate: UICandidate | Mapping[str, Any]) -> tuple[int, int]:
    normalized = candidate if isinstance(candidate, UICandidate) else UICandidate.from_dict(candidate)
    center = normalized.center or bbox_center(normalized.bbox)
    return center[0], center[1]


def normalize_bbox(bbox: list[int]) -> list[int]:
    if len(bbox) != 4:
        raise UISchemaError("bbox must contain four integers")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in bbox):
        raise UISchemaError("bbox values must be integers")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise UISchemaError("bbox must satisfy x2 > x1 and y2 > y1")
    return [x1, y1, x2, y2]


def offset_bbox(bbox: list[int], screen_origin: Sequence[int] | None = None) -> list[int]:
    x1, y1, x2, y2 = normalize_bbox(bbox)
    origin_x, origin_y = normalize_screen_origin(screen_origin)
    return [x1 + origin_x, y1 + origin_y, x2 + origin_x, y2 + origin_y]


def normalize_center(center: list[int]) -> list[int]:
    if len(center) != 2:
        raise UISchemaError("center must contain two integers")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in center):
        raise UISchemaError("center values must be integers")
    return [center[0], center[1]]


def normalize_screen_origin(screen_origin: Sequence[int] | None = None) -> list[int]:
    if screen_origin is None:
        return [0, 0]
    if len(screen_origin) != 2:
        raise UISchemaError("screen_origin must contain two integers")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in screen_origin):
        raise UISchemaError("screen_origin values must be integers")
    return [screen_origin[0], screen_origin[1]]


def _bool_field(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise UISchemaError(f"{field_name} must be a boolean")
    return value


def _float_field(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise UISchemaError(f"{field_name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise UISchemaError(f"{field_name} must be a number") from exc


def _uia_role_clickable(role: str) -> bool:
    clickable_keywords = (
        "Button",
        "Hyperlink",
        "MenuItem",
        "TabItem",
        "TreeItem",
        "ListItem",
        "CheckBox",
        "RadioButton",
        "Image",
    )
    return any(keyword in role for keyword in clickable_keywords)


def _uia_role_editable(role: str) -> bool:
    editable_keywords = ("Edit", "ComboBox")
    return any(keyword in role for keyword in editable_keywords)
