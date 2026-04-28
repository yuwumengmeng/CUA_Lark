"""Generic page semantic summary shared by perception, planner, and validator.

This schema is intentionally product-neutral. It can describe IM, Docs,
Calendar, or later Lark product pages without hard-coding fields such as
`has_chat_list`. Product-specific signals should be represented through generic
signals, state tags, and entities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


GOAL_PROGRESS_UNKNOWN = "unknown"
GOAL_PROGRESS_NOT_STARTED = "not_started"
GOAL_PROGRESS_IN_PROGRESS = "in_progress"
GOAL_PROGRESS_NEAR_GOAL = "near_goal"
GOAL_PROGRESS_ACHIEVED = "achieved"
GOAL_PROGRESS_BLOCKED = "blocked"

SUPPORTED_GOAL_PROGRESS = frozenset({
    GOAL_PROGRESS_UNKNOWN,
    GOAL_PROGRESS_NOT_STARTED,
    GOAL_PROGRESS_IN_PROGRESS,
    GOAL_PROGRESS_NEAR_GOAL,
    GOAL_PROGRESS_ACHIEVED,
    GOAL_PROGRESS_BLOCKED,
})

DEFAULT_SEMANTIC_SIGNALS = {
    "search_available": False,
    "search_active": False,
    "results_available": False,
    "list_available": False,
    "detail_open": False,
    "input_available": False,
    "editor_available": False,
    "modal_open": False,
    "menu_open": False,
    "loading": False,
    "error": False,
    "target_visible": False,
    "target_active": False,
}

MAX_STATE_TAGS = 12
MAX_ENTITIES = 8
MAX_ACTIONS = 8
MAX_TEXT_FIELD_LENGTH = 160
MAX_TAG_LENGTH = 48


class PageSemanticsError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PageSemantics:
    domain: str = "unknown"
    view: str = "unknown"
    summary: str = ""
    state_tags: list[str] = field(default_factory=list)
    signals: dict[str, bool] = field(default_factory=dict)
    entities: list[dict[str, Any]] = field(default_factory=list)
    available_actions: list[str] = field(default_factory=list)
    goal_progress: str = GOAL_PROGRESS_UNKNOWN
    blocking_issue: str | None = None
    confidence: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _clean_token(self.domain, default="unknown"))
        object.__setattr__(self, "view", _clean_token(self.view, default="unknown"))
        object.__setattr__(self, "summary", _clean_text(self.summary))
        object.__setattr__(self, "state_tags", _clean_string_list(self.state_tags, limit=MAX_STATE_TAGS))
        object.__setattr__(self, "signals", _clean_signals(self.signals))
        object.__setattr__(self, "entities", _clean_entities(self.entities))
        object.__setattr__(
            self,
            "available_actions",
            _clean_string_list(self.available_actions, limit=MAX_ACTIONS),
        )
        progress = _clean_token(self.goal_progress, default=GOAL_PROGRESS_UNKNOWN)
        if progress not in SUPPORTED_GOAL_PROGRESS:
            progress = GOAL_PROGRESS_UNKNOWN
        object.__setattr__(self, "goal_progress", progress)
        blocking_issue = _clean_text(self.blocking_issue) if self.blocking_issue else None
        object.__setattr__(self, "blocking_issue", blocking_issue or None)
        object.__setattr__(self, "confidence", _clamp_float(self.confidence, 0.0, 1.0))
        object.__setattr__(self, "reason", _clean_text(self.reason))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PageSemantics":
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            domain=str(data.get("domain", "unknown")),
            view=str(data.get("view", "unknown")),
            summary=str(data.get("summary", "")),
            state_tags=list(data.get("state_tags") or []),
            signals=dict(data.get("signals") or {}),
            entities=list(data.get("entities") or []),
            available_actions=list(data.get("available_actions") or []),
            goal_progress=str(data.get("goal_progress", GOAL_PROGRESS_UNKNOWN)),
            blocking_issue=data.get("blocking_issue"),
            confidence=_to_float(data.get("confidence", 0.0)),
            reason=str(data.get("reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "domain": self.domain,
            "view": self.view,
            "summary": self.summary,
            "state_tags": self.state_tags,
            "signals": self.signals,
            "entities": self.entities,
            "available_actions": self.available_actions,
            "goal_progress": self.goal_progress,
            "blocking_issue": self.blocking_issue,
            "confidence": self.confidence,
            "reason": self.reason,
        }
        return payload


def _clean_token(value: Any, *, default: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return default
    allowed = "".join(char for char in text if char.isalnum() or char == "_")
    return allowed[:MAX_TAG_LENGTH] or default


def _clean_text(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:MAX_TEXT_FIELD_LENGTH]


def _clean_string_list(values: list[Any], *, limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = _clean_token(value, default="")
        if not token or token in seen:
            continue
        cleaned.append(token)
        seen.add(token)
        if len(cleaned) >= limit:
            break
    return cleaned


def _clean_signals(signals: Mapping[str, Any]) -> dict[str, bool]:
    cleaned = dict(DEFAULT_SEMANTIC_SIGNALS)
    if not isinstance(signals, Mapping):
        return cleaned
    for key, value in signals.items():
        token = _clean_token(key, default="")
        if not token:
            continue
        cleaned[token] = _to_bool(value)
    return cleaned


def _clean_entities(entities: list[Any]) -> list[dict[str, Any]]:
    if not isinstance(entities, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for raw in entities:
        if not isinstance(raw, Mapping):
            continue
        item = {
            "type": _clean_token(raw.get("type", "unknown"), default="unknown"),
            "name": _clean_text(raw.get("name", "")),
            "visible": _to_bool(raw.get("visible", True)),
            "matched_target": _to_bool(raw.get("matched_target", False)),
            "confidence": _clamp_float(_to_float(raw.get("confidence", 0.0)), 0.0, 1.0),
        }
        role = _clean_token(raw.get("role", ""), default="")
        if role:
            item["role"] = role
        if item["name"] or item["type"] != "unknown":
            cleaned.append(item)
        if len(cleaned) >= MAX_ENTITIES:
            break
    return cleaned


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    return bool(value)


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
