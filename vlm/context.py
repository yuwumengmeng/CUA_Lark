"""VLM context budgeting and candidate filtering.

Responsibility boundary: this module builds the compact payload sent to VLM.
It limits page-summary text, trims candidate fields, and selects a step-focused
top-k candidate set. It does not call VLM APIs, parse VLM output, rerank the
final UI state, or execute actions.

The goal is to keep each VLM call narrow: one current step, a short page
summary, and only the candidates likely relevant to that step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DEFAULT_MAX_PAGE_SUMMARY_TEXTS = 12
DEFAULT_MAX_PAGE_SUMMARY_TEXT_LENGTH = 48
DEFAULT_MAX_CANDIDATE_TEXT_LENGTH = 60

FOCUS_SEARCH = "search"
FOCUS_RESULTS = "results"
FOCUS_INPUT = "input"
FOCUS_DOCUMENT_BODY = "document_body"
FOCUS_MODAL = "modal"
FOCUS_NAVIGATION = "navigation"
FOCUS_DETAIL = "detail"
FOCUS_UNKNOWN = "unknown"

SHELL_CONTROL_ROLES = (
    "titlebar",
    "menubar",
    "scrollbar",
    "statusbar",
    "toolbar",
    "thumb",
    "systemmenubar",
)

SHELL_CONTROL_TEXTS = {
    "最小化",
    "最大化",
    "关闭",
    "还原",
    "minimize",
    "maximize",
    "close",
    "restore",
    "chrome",
    "chromium",
    "microsoft edge",
    "edge",
}


@dataclass(frozen=True, slots=True)
class VLMContextPolicy:
    max_page_summary_texts: int = DEFAULT_MAX_PAGE_SUMMARY_TEXTS
    max_page_summary_text_length: int = DEFAULT_MAX_PAGE_SUMMARY_TEXT_LENGTH
    max_candidate_text_length: int = DEFAULT_MAX_CANDIDATE_TEXT_LENGTH
    include_context_meta: bool = True

    def __post_init__(self) -> None:
        if self.max_page_summary_texts <= 0:
            raise VLMContextError("max_page_summary_texts must be positive")
        if self.max_page_summary_text_length <= 0:
            raise VLMContextError("max_page_summary_text_length must be positive")
        if self.max_candidate_text_length <= 0:
            raise VLMContextError("max_candidate_text_length must be positive")


class VLMContextError(ValueError):
    pass


def build_vlm_context_payload(
    *,
    instruction: str,
    page_summary: Mapping[str, Any] | None,
    candidates: Sequence[Mapping[str, Any]] | None,
    max_candidates: int,
    step_name: str | None = None,
    step_goal: str | None = None,
    region_hint: str | None = None,
    policy: VLMContextPolicy | None = None,
) -> dict[str, Any]:
    if max_candidates <= 0:
        raise VLMContextError("max_candidates must be positive")

    context_policy = policy or VLMContextPolicy()
    raw_candidates = list(candidates or [])
    focus = infer_context_focus(
        step_name=step_name,
        step_goal=step_goal,
        region_hint=region_hint,
    )
    selected = select_candidate_context(
        raw_candidates,
        max_candidates=max_candidates,
        context_focus=focus,
        step_name=step_name,
        step_goal=step_goal,
        region_hint=region_hint,
    )
    payload: dict[str, Any] = {
        "instruction": _truncate_text(instruction, 160),
        "page_summary": compact_page_summary(page_summary or {}, policy=context_policy),
        "candidates": [
            compact_candidate(candidate, policy=context_policy)
            for candidate in selected
        ],
    }
    if step_name is not None:
        payload["step_name"] = _truncate_text(step_name, 64)
    if step_goal is not None:
        payload["step_goal"] = _truncate_text(step_goal, 160)
    if region_hint is not None:
        payload["region_hint"] = _truncate_text(region_hint, 96)
    if context_policy.include_context_meta:
        payload["context_meta"] = {
            "focus": focus,
            "candidate_count_total": len(raw_candidates),
            "candidate_count_sent": len(selected),
            "max_candidates": max_candidates,
            "summary_text_count_sent": len(payload["page_summary"]["top_texts"]),
        }
    return payload


def compact_page_summary(
    page_summary: Mapping[str, Any],
    *,
    policy: VLMContextPolicy | None = None,
) -> dict[str, Any]:
    context_policy = policy or VLMContextPolicy()
    top_texts = page_summary.get("top_texts", [])
    if not isinstance(top_texts, list):
        top_texts = []
    return {
        "top_texts": [
            _truncate_text(text, context_policy.max_page_summary_text_length)
            for text in top_texts[: context_policy.max_page_summary_texts]
            if str(text).strip()
        ],
        "has_search_box": bool(page_summary.get("has_search_box", False)),
        "has_modal": bool(page_summary.get("has_modal", False)),
        "candidate_count": _safe_int(page_summary.get("candidate_count", 0)),
    }


def compact_candidate(
    candidate: Mapping[str, Any],
    *,
    policy: VLMContextPolicy | None = None,
) -> dict[str, Any]:
    context_policy = policy or VLMContextPolicy()
    return {
        "id": candidate.get("id"),
        "text": _truncate_text(candidate.get("text", ""), context_policy.max_candidate_text_length),
        "bbox": candidate.get("bbox"),
        "center": candidate.get("center"),
        "role": candidate.get("role", "unknown"),
        "clickable": bool(candidate.get("clickable", False)),
        "editable": bool(candidate.get("editable", False)),
        "source": candidate.get("source", "unknown"),
        "confidence": _safe_float(candidate.get("confidence", 0.0)),
    }


def select_candidate_context(
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_candidates: int,
    context_focus: str | None = None,
    step_name: str | None = None,
    step_goal: str | None = None,
    region_hint: str | None = None,
) -> list[Mapping[str, Any]]:
    raw_candidates = list(candidates)
    if len(raw_candidates) <= max_candidates:
        return raw_candidates

    focus = context_focus or infer_context_focus(
        step_name=step_name,
        step_goal=step_goal,
        region_hint=region_hint,
    )
    extent = _candidate_extent(raw_candidates)
    query = " ".join(part for part in (step_name, step_goal, region_hint) if part)
    query_tokens = _query_tokens(query)

    scored: list[tuple[float, int, Mapping[str, Any]]] = []
    for index, candidate in enumerate(raw_candidates):
        score = candidate_context_score(
            candidate,
            context_focus=focus,
            query_tokens=query_tokens,
            extent=extent,
        )
        scored.append((score, index, candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in scored[:max_candidates]]


def infer_context_focus(
    *,
    step_name: str | None = None,
    step_goal: str | None = None,
    region_hint: str | None = None,
) -> str:
    text = _normalize_for_keyword(" ".join(part for part in (step_name, step_goal, region_hint) if part))

    if _contains_any(text, ("modal", "dialog", "popup", "menu", "panel", "弹窗", "菜单", "面板", "更多")):
        return FOCUS_MODAL
    if _contains_any(text, ("document_body", "body", "正文", "章节", "关键词", "可见章节")):
        return FOCUS_DOCUMENT_BODY
    if _contains_any(text, ("message_input", "input_box", "input", "type", "send", "输入框", "输入", "发送")):
        return FOCUS_INPUT
    if _contains_any(text, ("open_chat", "result", "results", "chat_list", "list", "群聊", "联系人", "搜索结果", "列表")):
        return FOCUS_RESULTS
    if _contains_any(text, ("search", "query", "keyword", "open_search", "type_keyword", "搜索", "查找", "关键词")):
        return FOCUS_SEARCH
    if _contains_any(text, ("nav", "sidebar", "入口", "导航", "侧边栏")):
        return FOCUS_NAVIGATION
    if _contains_any(text, ("detail", "conversation", "target_chat", "详情", "目标页面", "已进入")):
        return FOCUS_DETAIL
    return FOCUS_UNKNOWN


def candidate_context_score(
    candidate: Mapping[str, Any],
    *,
    context_focus: str,
    query_tokens: set[str],
    extent: tuple[int, int, int, int] | None,
) -> float:
    text = str(candidate.get("text", ""))
    role = str(candidate.get("role", ""))
    text_norm = _normalize_for_keyword(text)
    role_norm = _normalize_for_keyword(role)
    editable = bool(candidate.get("editable", False))
    clickable = bool(candidate.get("clickable", False))
    score = _safe_float(candidate.get("confidence", 0.0))

    for token in query_tokens:
        if len(token) < 2:
            continue
        if token in text_norm:
            score += 2.4
        if token in role_norm:
            score += 0.9

    score += _focus_score(
        context_focus=context_focus,
        text_norm=text_norm,
        role_norm=role_norm,
        editable=editable,
        clickable=clickable,
    )
    score += _region_score(candidate, context_focus=context_focus, extent=extent)

    if _is_shell_control(text_norm=text_norm, role_norm=role_norm):
        score -= 5.0
    return score


def _focus_score(
    *,
    context_focus: str,
    text_norm: str,
    role_norm: str,
    editable: bool,
    clickable: bool,
) -> float:
    if context_focus == FOCUS_SEARCH:
        score = 0.0
        if "search" in role_norm or "搜索" in text_norm:
            score += 4.0
        if editable:
            score += 2.5
        return score
    if context_focus == FOCUS_INPUT:
        score = 0.0
        if editable:
            score += 4.0
        if _contains_any(role_norm, ("edit", "input", "textbox", "text")):
            score += 1.4
        if _contains_any(text_norm, ("发送给", "消息", "输入", "send")):
            score += 1.2
        return score
    if context_focus == FOCUS_RESULTS:
        score = 0.0
        if _contains_any(role_norm, ("list", "item", "chat", "conversation", "result")):
            score += 3.0
        if clickable:
            score += 1.2
        if text_norm:
            score += 0.6
        if editable:
            score -= 1.4
        return score
    if context_focus == FOCUS_DOCUMENT_BODY:
        score = 0.0
        if _contains_any(role_norm, ("document", "editor", "text", "pane")):
            score += 2.2
        if _contains_any(text_norm, ("文档", "标题", "正文", "章节")):
            score += 1.2
        return score
    if context_focus == FOCUS_MODAL:
        score = 0.0
        if _contains_any(role_norm, ("menu", "dialog", "popup", "panel", "button")):
            score += 2.0
        if clickable:
            score += 0.8
        return score
    if context_focus == FOCUS_NAVIGATION:
        score = 0.0
        if _contains_any(role_norm, ("nav", "tab", "menu", "button")):
            score += 2.0
        if clickable:
            score += 0.8
        return score
    if context_focus == FOCUS_DETAIL:
        score = 0.0
        if _contains_any(role_norm, ("pane", "document", "chat", "conversation", "text")):
            score += 1.2
        return score
    return 0.0


def _region_score(
    candidate: Mapping[str, Any],
    *,
    context_focus: str,
    extent: tuple[int, int, int, int] | None,
) -> float:
    if extent is None:
        return 0.0
    normalized = _normalized_center(candidate, extent)
    if normalized is None:
        return 0.0
    x_norm, y_norm = normalized

    if context_focus == FOCUS_SEARCH:
        return 2.0 if y_norm <= 0.35 else 0.0
    if context_focus == FOCUS_INPUT:
        return 2.2 if y_norm >= 0.65 else 0.0
    if context_focus == FOCUS_RESULTS:
        return 1.8 if 0.15 <= x_norm <= 0.75 and 0.08 <= y_norm <= 0.9 else 0.0
    if context_focus == FOCUS_DOCUMENT_BODY:
        return 2.0 if x_norm >= 0.35 and 0.08 <= y_norm <= 0.95 else 0.0
    if context_focus == FOCUS_MODAL:
        return 1.7 if 0.2 <= x_norm <= 0.8 and 0.12 <= y_norm <= 0.88 else 0.0
    if context_focus == FOCUS_NAVIGATION:
        return 1.6 if x_norm <= 0.35 else 0.0
    if context_focus == FOCUS_DETAIL:
        return 1.0 if x_norm >= 0.35 else 0.0
    return 0.0


def _candidate_extent(candidates: Sequence[Mapping[str, Any]]) -> tuple[int, int, int, int] | None:
    boxes: list[list[int]] = []
    for candidate in candidates:
        bbox = candidate.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, int) for v in bbox):
            boxes.append(bbox)
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _normalized_center(
    candidate: Mapping[str, Any],
    extent: tuple[int, int, int, int],
) -> tuple[float, float] | None:
    center = candidate.get("center")
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        return None
    left, top, right, bottom = extent
    width = max(1, right - left)
    height = max(1, bottom - top)
    try:
        x_norm = (float(center[0]) - left) / width
        y_norm = (float(center[1]) - top) / height
    except (TypeError, ValueError):
        return None
    return x_norm, y_norm


def _query_tokens(text: str) -> set[str]:
    chunks = _normalized_chunks(text)
    tokens: set[str] = set(chunks)
    for chunk in chunks:
        if _contains_cjk(chunk):
            max_len = min(6, len(chunk))
            for size in range(2, max_len + 1):
                for start in range(0, len(chunk) - size + 1):
                    tokens.add(chunk[start : start + size])
    return {token for token in tokens if token}


def _normalized_chunks(text: str) -> list[str]:
    normalized = "".join(
        char.lower() if char.isalnum() or "\u4e00" <= char <= "\u9fff" else " "
        for char in str(text or "")
    )
    return [token for token in normalized.split() if token]


def _normalize_for_keyword(text: str) -> str:
    return "".join(_normalized_chunks(text))


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _is_shell_control(*, text_norm: str, role_norm: str) -> bool:
    if any(role in role_norm for role in SHELL_CONTROL_ROLES):
        return True
    return text_norm in SHELL_CONTROL_TEXTS


def _truncate_text(value: Any, max_length: int) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return text[: max_length - 3] + "..."


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
