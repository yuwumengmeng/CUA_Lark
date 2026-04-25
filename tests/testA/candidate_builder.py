"""成员 A 4.24 首版 UI Element Candidate Builder。

职责边界：
本文件负责把 OCR 文本块和 UIA 元素统一转换为 `UICandidate`，做首版
去重合并，并生成基础页面摘要。它不负责截图采集、不负责真实 OCR/UIA
调用，也不负责 planner 决策或 executor 点击。

公开接口：
`build_ui_candidates(...)`
    输入 OCR/UIA 抽取结果，输出统一 `list[UICandidate]`。
`build_page_summary(...)`
    根据候选元素生成 `PageSummary`。
`build_ui_state_from_elements(...)`
    根据截图路径、OCR/UIA 抽取结果生成 `UIState`。

坐标规则：
`UICandidate.bbox/center` 统一使用屏幕绝对坐标。OCR 原始 bbox 来自截图
局部坐标，必须传入截图 `screen_origin` 后再转换；UIA bbox 已经是屏幕
绝对坐标，不做二次偏移。

合并规则：
第一版只做保守去重：
1. bbox IoU 高于阈值时合并。
2. 文本相同且中心点距离较近时合并。
3. 合并后 `source=merged`，confidence 取较高值，clickable/editable 取
   OR，bbox 取 union。
4. 针对飞书桌面端常见的 OCR 拆字问题，额外生成同一行的紧凑文本候选，
   并在检测到飞书界面特征时补充左侧导航、二级导航和搜索框候选。

协作规则：
1. 输出 candidate 字段必须遵守 `schemas.ui_candidate.UICandidate`。
2. 输出列表按 confidence 降序排序，并重新编号为 `elem_xxxx`。
3. C 直接消费输出中的 `center`，不再自行转换 OCR/UIA 坐标。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Sequence

from schemas import (
    SOURCE_MERGED,
    SOURCE_UIA,
    PageSummary,
    UICandidate,
    UIState,
    candidate_from_ocr,
    candidate_from_uia,
)

from .ocr import OCRTextBlock
from .uia import UIAElement


LARK_MAIN_NAV_LABELS = {
    "消息",
    "知识问答",
    "云文档",
    "推荐",
    "多维表格",
    "视频会议",
    "工作台",
    "应用中心",
    "通讯录",
    "日历",
    "飞行社",
    "更多",
}

LARK_SECONDARY_NAV_LABELS = {
    "主页",
    "云盘",
    "知识库",
    "智能纪要",
    "消息",
    "未读",
    "标记",
    "@我",
    "标签",
    "单聊",
    "群组",
    "云文档",
    "话题",
    "已完成",
}

LARK_LANDMARK_LABELS = LARK_MAIN_NAV_LABELS | LARK_SECONDARY_NAV_LABELS | {"搜索", "分组"}


@dataclass(frozen=True, slots=True)
class CandidateBuilderConfig:
    merge_iou_threshold: float = 0.65
    merge_center_distance: int = 48
    uia_confidence: float = 0.85
    enable_lark_desktop_hints: bool = True
    ocr_line_merge_max_gap: int = 36
    ocr_line_merge_max_y_delta: int = 14

    def __post_init__(self) -> None:
        if not 0.0 <= self.merge_iou_threshold <= 1.0:
            raise CandidateBuilderError("merge_iou_threshold must be between 0.0 and 1.0")
        if self.merge_center_distance < 0:
            raise CandidateBuilderError("merge_center_distance must be non-negative")
        if not 0.0 <= self.uia_confidence <= 1.0:
            raise CandidateBuilderError("uia_confidence must be between 0.0 and 1.0")
        if self.ocr_line_merge_max_gap < 0:
            raise CandidateBuilderError("ocr_line_merge_max_gap must be non-negative")
        if self.ocr_line_merge_max_y_delta < 0:
            raise CandidateBuilderError("ocr_line_merge_max_y_delta must be non-negative")


class CandidateBuilderError(ValueError):
    pass


def build_ui_candidates(
    *,
    ocr_elements: Sequence[OCRTextBlock] | None = None,
    uia_elements: Sequence[UIAElement] | None = None,
    screen_origin: Sequence[int] | None = None,
    config: CandidateBuilderConfig | None = None,
) -> list[UICandidate]:
    builder_config = config or CandidateBuilderConfig()
    ocr_candidates = _ocr_candidates(ocr_elements or [], screen_origin)
    raw_candidates = list(ocr_candidates)
    raw_candidates.extend(_compact_ocr_line_candidates(ocr_candidates, builder_config))
    raw_candidates.extend(_uia_candidates(uia_elements or [], builder_config.uia_confidence))

    merged: list[UICandidate] = []
    for candidate in raw_candidates:
        merge_index = _find_merge_target(merged, candidate, builder_config)
        if merge_index is None:
            merged.append(candidate)
        else:
            merged[merge_index] = _merge_candidates(merged[merge_index], candidate)

    if builder_config.enable_lark_desktop_hints and _looks_like_lark_desktop(merged):
        merged.extend(_lark_desktop_candidates(merged))

    return _renumber_candidates(merged)


def build_page_summary(candidates: Sequence[UICandidate]) -> PageSummary:
    return PageSummary.from_candidates(list(candidates))


def build_ui_state_from_elements(
    *,
    screenshot_path: str | Path,
    ocr_elements: Sequence[OCRTextBlock] | None = None,
    uia_elements: Sequence[UIAElement] | None = None,
    screen_origin: Sequence[int] | None = None,
    config: CandidateBuilderConfig | None = None,
) -> UIState:
    candidates = build_ui_candidates(
        ocr_elements=ocr_elements,
        uia_elements=uia_elements,
        screen_origin=screen_origin,
        config=config,
    )
    return UIState(
        screenshot_path=str(screenshot_path),
        candidates=candidates,
        page_summary=build_page_summary(candidates),
    )


def _ocr_candidates(
    ocr_elements: Sequence[OCRTextBlock],
    screen_origin: Sequence[int] | None,
) -> list[UICandidate]:
    return [
        candidate_from_ocr(
            candidate_id=f"ocr_{index:04d}",
            text=element.text,
            bbox=element.bbox,
            confidence=element.confidence,
            screen_origin=screen_origin,
        )
        for index, element in enumerate(ocr_elements, start=1)
    ]


def _uia_candidates(
    uia_elements: Sequence[UIAElement],
    uia_confidence: float,
) -> list[UICandidate]:
    return [
        candidate_from_uia(
            candidate_id=f"uia_{index:04d}",
            name=element.name,
            role=element.role,
            bbox=element.bbox,
            enabled=element.enabled,
            visible=element.visible,
            confidence=uia_confidence,
        )
        for index, element in enumerate(uia_elements, start=1)
    ]


def _compact_ocr_line_candidates(
    ocr_candidates: Sequence[UICandidate],
    config: CandidateBuilderConfig,
) -> list[UICandidate]:
    rows = _group_ocr_candidates_by_row(
        [candidate for candidate in ocr_candidates if _line_mergeable_text(candidate.text)],
        config,
    )
    line_candidates: list[UICandidate] = []
    for row in rows:
        chain: list[UICandidate] = []
        for candidate in sorted(row, key=lambda item: item.bbox[0]):
            if chain and not _should_join_ocr_tokens(chain[-1], candidate, config):
                _append_compact_line_candidate(line_candidates, chain)
                chain = []
            chain.append(candidate)
        _append_compact_line_candidate(line_candidates, chain)
    return line_candidates


def _group_ocr_candidates_by_row(
    candidates: Sequence[UICandidate],
    config: CandidateBuilderConfig,
) -> list[list[UICandidate]]:
    rows: list[list[UICandidate]] = []
    for candidate in sorted(candidates, key=lambda item: (item.center[1], item.bbox[0])):
        candidate_height = candidate.bbox[3] - candidate.bbox[1]
        placed = False
        for row in rows:
            row_center = sum(item.center[1] for item in row) / len(row)
            row_height = max(item.bbox[3] - item.bbox[1] for item in row)
            tolerance = max(config.ocr_line_merge_max_y_delta, row_height // 2, candidate_height // 2)
            if abs(candidate.center[1] - row_center) <= tolerance:
                row.append(candidate)
                placed = True
                break
        if not placed:
            rows.append([candidate])
    return rows


def _should_join_ocr_tokens(
    left: UICandidate,
    right: UICandidate,
    config: CandidateBuilderConfig,
) -> bool:
    height = max(left.bbox[3] - left.bbox[1], right.bbox[3] - right.bbox[1])
    y_delta = abs(left.center[1] - right.center[1])
    if y_delta > max(config.ocr_line_merge_max_y_delta, height // 2):
        return False

    gap = right.bbox[0] - left.bbox[2]
    if gap > max(config.ocr_line_merge_max_gap, int(height * 1.4)):
        return False
    if gap < -max(left.bbox[2] - left.bbox[0], right.bbox[2] - right.bbox[0]):
        return False

    merged_text = _join_ocr_texts([left, right])
    if len(_normalize_text_token(merged_text)) > 16:
        return False
    return _line_candidate_text_useful(merged_text)


def _append_compact_line_candidate(
    line_candidates: list[UICandidate],
    chain: Sequence[UICandidate],
) -> None:
    if len(chain) < 2:
        return
    text = _join_ocr_texts(chain)
    if not _line_candidate_text_useful(text):
        return

    bbox = chain[0].bbox
    for candidate in chain[1:]:
        bbox = union_bbox(bbox, candidate.bbox)
    confidence = sum(candidate.confidence for candidate in chain) / len(chain)
    line_candidates.append(
        UICandidate(
            id=f"ocr_line_{len(line_candidates) + 1:04d}",
            text=text,
            bbox=bbox,
            role="text",
            clickable=False,
            editable=False,
            source=SOURCE_MERGED,
            confidence=max(0.6, min(0.97, confidence)),
        )
    )


def _looks_like_lark_desktop(candidates: Sequence[UICandidate]) -> bool:
    texts = {_normalize_display_text(candidate.text) for candidate in candidates if candidate.text}
    landmark_count = sum(1 for text in texts if _lark_label_match(text, LARK_LANDMARK_LABELS) is not None)
    return landmark_count >= 3 or {"分组", "消息"}.issubset(texts)


def _lark_desktop_candidates(candidates: Sequence[UICandidate]) -> list[UICandidate]:
    if not candidates:
        return []

    left_bound = min(candidate.bbox[0] for candidate in candidates)
    generated: list[UICandidate] = []
    generated.extend(_lark_search_box_candidates(candidates, left_bound))
    main_nav_items = _lark_nav_item_candidates(
        candidates=candidates,
        left_bound=left_bound,
        labels=LARK_MAIN_NAV_LABELS,
        role="LarkMainNavItem",
        x_min=left_bound - 20,
        x_max=left_bound + 280,
        row_left=left_bound + 16,
        row_min_width=360,
        row_extra_right=220,
    )
    secondary_nav_items = _lark_nav_item_candidates(
        candidates=candidates,
        left_bound=left_bound,
        labels=LARK_SECONDARY_NAV_LABELS,
        role="LarkSecondaryNavItem",
        x_min=left_bound + 360,
        x_max=left_bound + 860,
        row_left=None,
        row_min_width=250,
        row_extra_right=180,
    )
    generated.extend(main_nav_items)
    generated.extend(secondary_nav_items)
    generated.extend(_lark_chat_list_candidates(candidates, secondary_nav_items, left_bound))
    return _dedupe_lark_candidates(generated)


def _lark_search_box_candidates(
    candidates: Sequence[UICandidate],
    left_bound: int,
) -> list[UICandidate]:
    search_candidates = [
        candidate
        for candidate in candidates
        if _normalize_display_text(candidate.text) == "搜索"
        and candidate.bbox[0] <= left_bound + 360
    ]
    if not search_candidates:
        return []

    search = min(search_candidates, key=lambda candidate: (candidate.bbox[1], candidate.bbox[0]))
    row_tokens = [
        candidate
        for candidate in candidates
        if abs(candidate.center[1] - search.center[1]) <= 18
        and search.bbox[0] - 8 <= candidate.bbox[0] <= search.bbox[0] + 260
    ]
    shortcut_text = "".join(candidate.text for candidate in sorted(row_tokens, key=lambda item: item.bbox[0]))
    text = "搜索 (Ctrl+K)" if "ctrl" in shortcut_text.lower() and "k" in shortcut_text.lower() else "搜索"
    center_y = search.center[1]
    bbox = [
        min(search.bbox[0] - 48, left_bound + 24),
        center_y - 28,
        max(search.bbox[2] + 260, left_bound + 340),
        center_y + 28,
    ]
    return [
        UICandidate(
            id="lark_search_box_0001",
            text=text,
            bbox=bbox,
            role="LarkSearchBox",
            clickable=True,
            editable=True,
            source=SOURCE_MERGED,
            confidence=0.99,
        )
    ]


def _lark_nav_item_candidates(
    *,
    candidates: Sequence[UICandidate],
    left_bound: int,
    labels: set[str],
    role: str,
    x_min: int,
    x_max: int,
    row_left: int | None,
    row_min_width: int,
    row_extra_right: int,
) -> list[UICandidate]:
    generated: list[UICandidate] = []
    for candidate in candidates:
        text = _lark_label_match(candidate.text, labels)
        if text is None:
            continue
        if not x_min <= candidate.bbox[0] <= x_max:
            continue

        center_y = candidate.center[1]
        candidate_row_left = row_left if row_left is not None else max(x_min, candidate.bbox[0] - 48)
        row_right = max(candidate.bbox[2] + row_extra_right, candidate_row_left + row_min_width)
        generated.append(
            UICandidate(
                id=f"{role.lower()}_{len(generated) + 1:04d}",
                text=text,
                bbox=[candidate_row_left, center_y - 28, row_right, center_y + 28],
                role=role,
                clickable=True,
                editable=False,
                source=SOURCE_MERGED,
                confidence=max(0.98, candidate.confidence),
            )
        )
    return generated


def _lark_label_match(text: str, labels: set[str]) -> str | None:
    normalized = _normalize_display_text(text)
    if normalized in labels:
        return normalized

    token = _normalize_text_token(normalized)
    if len(token) < 2:
        return None

    for label in sorted(labels, key=lambda value: (len(_normalize_text_token(value)), value)):
        label_token = _normalize_text_token(label)
        if not label_token or token == label_token:
            continue
        if label_token.startswith(token) and len(token) / len(label_token) >= 0.5:
            return label
    return None


def _lark_chat_list_candidates(
    candidates: Sequence[UICandidate],
    secondary_nav_items: Sequence[UICandidate],
    left_bound: int,
) -> list[UICandidate]:
    chat_left, chat_right = _resolve_lark_chat_bounds(candidates, secondary_nav_items, left_bound)
    if chat_left is None or chat_right is None:
        return []

    tokens = [
        candidate
        for candidate in candidates
        if _chat_list_token(candidate, chat_left=chat_left, chat_right=chat_right)
    ]
    if not tokens:
        return []

    generated: list[UICandidate] = []
    for row in _group_chat_list_tokens(tokens):
        row_text = _chat_row_text(row)
        if not row_text:
            continue
        center_y = int(sum(candidate.center[1] for candidate in row) / len(row))
        generated.append(
            UICandidate(
                id=f"lark_chat_list_item_{len(generated) + 1:04d}",
                text=row_text,
                bbox=[chat_left, center_y - 42, chat_right, center_y + 42],
                role="LarkChatListItem",
                clickable=True,
                editable=False,
                source=SOURCE_MERGED,
                confidence=max(0.90, min(0.97, max(candidate.confidence for candidate in row))),
            )
        )
    return generated


def _resolve_lark_chat_bounds(
    candidates: Sequence[UICandidate],
    secondary_nav_items: Sequence[UICandidate],
    left_bound: int,
) -> tuple[int | None, int | None]:
    if secondary_nav_items:
        chat_left = max(candidate.bbox[2] for candidate in secondary_nav_items) + 16
        return chat_left, chat_left + 900

    chat_headers = [
        candidate
        for candidate in candidates
        if _normalize_text_token(candidate.text) == "消息"
        and candidate.bbox[0] >= left_bound + 560
        and candidate.center[1] <= 180
    ]
    if chat_headers:
        chat_left = min(candidate.bbox[0] for candidate in chat_headers)
        return chat_left, chat_left + 900

    chat_left = left_bound + 900
    return chat_left, chat_left + 900


def _chat_list_token(candidate: UICandidate, *, chat_left: int, chat_right: int) -> bool:
    text = _normalize_display_text(candidate.text)
    if not text or not _chat_token_text_useful(text):
        return False
    center_x, center_y = candidate.center
    if not chat_left <= center_x <= chat_right:
        return False
    return center_y >= 120


def _group_chat_list_tokens(candidates: Sequence[UICandidate]) -> list[list[UICandidate]]:
    rows: list[list[UICandidate]] = []
    row_centers: list[float] = []
    for candidate in sorted(candidates, key=lambda item: (item.center[1], item.bbox[0])):
        matched_index: int | None = None
        for index, center_y in enumerate(row_centers):
            if abs(candidate.center[1] - center_y) <= 28:
                matched_index = index
                break

        if matched_index is None:
            rows.append([candidate])
            row_centers.append(float(candidate.center[1]))
            continue

        rows[matched_index].append(candidate)
        row_centers[matched_index] = sum(item.center[1] for item in rows[matched_index]) / len(
            rows[matched_index]
        )
    return [sorted(row, key=lambda item: item.bbox[0]) for row in rows]


def _chat_row_text(candidates: Sequence[UICandidate]) -> str:
    for candidate in sorted(candidates, key=lambda item: item.bbox[0]):
        text = _normalize_display_text(candidate.text)
        if _chat_token_text_useful(text):
            return text
    return ""


def _chat_token_text_useful(text: str) -> bool:
    token = _normalize_text_token(text)
    if not token:
        return False
    if len(token) > 24:
        return False
    return _contains_cjk(text) or any(char.isalnum() for char in text)


def _dedupe_lark_candidates(candidates: Sequence[UICandidate]) -> list[UICandidate]:
    seen: set[tuple[str, str, int]] = set()
    deduped: list[UICandidate] = []
    for candidate in candidates:
        key = (candidate.role, candidate.text, candidate.center[1] // 12)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _find_merge_target(
    existing: Sequence[UICandidate],
    candidate: UICandidate,
    config: CandidateBuilderConfig,
) -> int | None:
    for index, current in enumerate(existing):
        if _should_merge(current, candidate, config):
            return index
    return None


def _should_merge(
    left: UICandidate,
    right: UICandidate,
    config: CandidateBuilderConfig,
) -> bool:
    if bbox_iou(left.bbox, right.bbox) >= config.merge_iou_threshold:
        return True
    if left.text and right.text and left.text == right.text:
        return center_distance(left, right) <= config.merge_center_distance
    return False


def _merge_candidates(left: UICandidate, right: UICandidate) -> UICandidate:
    text = _prefer_text(left, right)
    role = _prefer_role(left, right)
    confidence = max(left.confidence, right.confidence)
    source = left.source if left.source == right.source else SOURCE_MERGED
    return UICandidate(
        id=left.id,
        text=text,
        bbox=union_bbox(left.bbox, right.bbox),
        role=role,
        clickable=left.clickable or right.clickable,
        editable=left.editable or right.editable,
        source=source,
        confidence=confidence,
    )


def _renumber_candidates(candidates: Sequence[UICandidate]) -> list[UICandidate]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.confidence,
            candidate.bbox[1],
            candidate.bbox[0],
            candidate.text,
        ),
    )
    return [
        UICandidate(
            id=f"elem_{index:04d}",
            text=candidate.text,
            bbox=candidate.bbox,
            role=candidate.role,
            clickable=candidate.clickable,
            editable=candidate.editable,
            source=candidate.source,
            confidence=candidate.confidence,
        )
        for index, candidate in enumerate(ordered, start=1)
    ]


def bbox_iou(left: list[int], right: list[int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def union_bbox(left: list[int], right: list[int]) -> list[int]:
    return [
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    ]


def center_distance(left: UICandidate, right: UICandidate) -> float:
    left_center = left.center or [(left.bbox[0] + left.bbox[2]) // 2, (left.bbox[1] + left.bbox[3]) // 2]
    right_center = right.center or [(right.bbox[0] + right.bbox[2]) // 2, (right.bbox[1] + right.bbox[3]) // 2]
    return hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])


def _prefer_text(left: UICandidate, right: UICandidate) -> str:
    if len(right.text) > len(left.text):
        return right.text
    return left.text


def _prefer_role(left: UICandidate, right: UICandidate) -> str:
    if left.role == "text" and right.source in {SOURCE_UIA, SOURCE_MERGED}:
        return right.role
    if left.role in {"unknown", ""}:
        return right.role
    return left.role


def _line_mergeable_text(text: str) -> bool:
    normalized = _normalize_display_text(text)
    if not normalized:
        return False
    if len(_normalize_text_token(normalized)) > 8:
        return False
    return _contains_cjk(normalized) or _looks_like_shortcut_token(normalized)


def _join_ocr_texts(candidates: Sequence[UICandidate]) -> str:
    return "".join(candidate.text.strip() for candidate in candidates if candidate.text.strip())


def _line_candidate_text_useful(text: str) -> bool:
    normalized = _normalize_display_text(text)
    token = _normalize_text_token(normalized)
    if len(token) < 2 or len(token) > 16:
        return False
    if _contains_cjk(normalized):
        return True
    return "ctrl" in token and "k" in token


def _normalize_display_text(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_text_token(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum() or "\u4e00" <= char <= "\u9fff")


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _looks_like_shortcut_token(text: str) -> bool:
    token = _normalize_text_token(text)
    return bool(token) and ("ctrl" in token or token == "k")
