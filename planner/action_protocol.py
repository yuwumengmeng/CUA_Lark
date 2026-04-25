"""成员 B 4.24 planner 到 executor 的动作协议。

职责边界：
本文件只定义 `MiniPlanner` 输出给成员 C executor 的 action decision 协议，
包括稳定字段、状态枚举、动作类型和失败原因。它不负责选择 UI 元素、不负责
推进 runtime state，也不负责真实鼠标键盘执行。

公开接口：
`ActionDecision`
    单步动作决策数据结构，提供 `continue_ / success / failed` 构造方法、
    `from_dict(...)` 解析方法、`to_dict(...)` 日志字典和 `to_executor_dict(...)`
    执行器消费字典。
`action_decision_to_executor_payload(...)`
    从 action decision 中剥离 B 内部字段，得到传给 executor 的稳定 payload。

字段协议：
稳定字段为 `status / action_type / target_element_id / x / y / text / keys /
seconds / reason / failure_reason / version`。`planner_meta` 只允许 B 内部用来
推进 step 或调试，成员 C 不应依赖它。

协作规则：
1. C 侧只需要读取 `action_type / target_element_id / x / y / text / keys /
   seconds / reason` 等可执行字段。
2. 新增动作类型时，必须同时确认 `schemas/action_result.py` 的 executor 支持情况。
3. 修改字段名或状态值时，必须同步更新 `tests/testB/test_action_protocol.py`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from schemas.action_result import (
    ACTION_CLICK,
    ACTION_HOTKEY,
    ACTION_TYPE,
    ACTION_WAIT,
    SUPPORTED_EXECUTOR_ACTION_TYPES,
)


ACTION_DECISION_VERSION = "action_decision.v1"

DECISION_CONTINUE = "continue"
DECISION_SUCCESS = "success"
DECISION_FAILED = "failed"

FAILURE_TARGET_NOT_FOUND = "target_not_found"
FAILURE_PAGE_MISMATCH = "page_mismatch"
FAILURE_EXECUTOR_FAILED = "executor_failed"

SUPPORTED_DECISION_STATUSES = {
    DECISION_CONTINUE,
    DECISION_SUCCESS,
    DECISION_FAILED,
}

SUPPORTED_PLANNER_ACTION_TYPES = {
    ACTION_CLICK,
    ACTION_TYPE,
    ACTION_HOTKEY,
    ACTION_WAIT,
}

SUPPORTED_FAILURE_REASONS = {
    FAILURE_TARGET_NOT_FOUND,
    FAILURE_PAGE_MISMATCH,
    FAILURE_EXECUTOR_FAILED,
}


class ActionDecisionSchemaError(ValueError):
    """Raised when action_decision.v1 data is malformed."""


@dataclass(frozen=True, slots=True)
class ActionDecision:
    status: str
    action_type: str | None = None
    target_element_id: str | None = None
    x: int | None = None
    y: int | None = None
    text: str | None = None
    keys: list[str] | None = None
    seconds: int | float | None = None
    reason: str = ""
    failure_reason: str | None = None
    version: str = ACTION_DECISION_VERSION
    planner_meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = _status(self.status)
        action_type = _action_type(self.action_type, status)
        failure_reason = _failure_reason(self.failure_reason, status)
        x = _optional_int(self.x, "x")
        y = _optional_int(self.y, "y")
        seconds = _optional_seconds(self.seconds)
        keys = _optional_keys(self.keys)
        planner_meta = _dict_field(self.planner_meta, "planner_meta")

        _validate_action_params(action_type, x, y, self.text, keys, seconds)

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "action_type", action_type)
        object.__setattr__(self, "failure_reason", failure_reason)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "seconds", seconds)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "version", str(self.version or ACTION_DECISION_VERSION))
        object.__setattr__(self, "planner_meta", planner_meta)

    @classmethod
    def continue_(
        cls,
        *,
        action_type: str,
        target_element_id: str | None = None,
        x: int | None = None,
        y: int | None = None,
        text: str | None = None,
        keys: list[str] | None = None,
        seconds: int | float | None = None,
        reason: str = "",
        planner_meta: Mapping[str, Any] | None = None,
    ) -> "ActionDecision":
        return cls(
            status=DECISION_CONTINUE,
            action_type=action_type,
            target_element_id=target_element_id,
            x=x,
            y=y,
            text=text,
            keys=keys,
            seconds=seconds,
            reason=reason,
            failure_reason=None,
            planner_meta=dict(planner_meta or {}),
        )

    @classmethod
    def success(cls, reason: str) -> "ActionDecision":
        return cls(
            status=DECISION_SUCCESS,
            action_type=None,
            reason=reason,
            failure_reason=None,
        )

    @classmethod
    def failed(cls, failure_reason: str, reason: str) -> "ActionDecision":
        return cls(
            status=DECISION_FAILED,
            action_type=None,
            reason=reason,
            failure_reason=failure_reason,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionDecision":
        if "status" not in data:
            raise ActionDecisionSchemaError("ActionDecision is missing required field: status")
        return cls(
            status=str(data["status"]),
            action_type=_optional_str(data.get("action_type")),
            target_element_id=_optional_str(data.get("target_element_id")),
            x=data.get("x"),
            y=data.get("y"),
            text=_optional_str(data.get("text")),
            keys=data.get("keys"),
            seconds=data.get("seconds"),
            reason=str(data.get("reason") or ""),
            failure_reason=_optional_str(data.get("failure_reason")),
            version=str(data.get("version") or ACTION_DECISION_VERSION),
            planner_meta=_dict_field(data.get("planner_meta", {}), "planner_meta"),
        )

    def to_dict(self, *, include_internal: bool = True) -> dict[str, Any]:
        result = {
            "version": self.version,
            "status": self.status,
            "action_type": self.action_type,
            "target_element_id": self.target_element_id,
            "x": self.x,
            "y": self.y,
            "text": self.text,
            "keys": None if self.keys is None else list(self.keys),
            "seconds": self.seconds,
            "reason": self.reason,
            "failure_reason": self.failure_reason,
        }
        if include_internal and self.planner_meta:
            result["planner_meta"] = dict(self.planner_meta)
        return result

    def to_executor_dict(self) -> dict[str, Any]:
        return self.to_dict(include_internal=False)


def action_decision_to_executor_payload(decision: Mapping[str, Any]) -> dict[str, Any]:
    return ActionDecision.from_dict(decision).to_executor_dict()


def _status(value: str) -> str:
    text = str(value)
    if text not in SUPPORTED_DECISION_STATUSES:
        raise ActionDecisionSchemaError(f"Unsupported action decision status: {text}")
    return text


def _action_type(value: str | None, status: str) -> str | None:
    text = _optional_str(value)
    if status != DECISION_CONTINUE:
        if text is not None:
            raise ActionDecisionSchemaError("Non-continue decisions cannot include action_type")
        return None
    if text is None:
        raise ActionDecisionSchemaError("Continue decisions require action_type")
    if text not in SUPPORTED_PLANNER_ACTION_TYPES:
        raise ActionDecisionSchemaError(f"Unsupported planner action_type: {text}")
    if text not in SUPPORTED_EXECUTOR_ACTION_TYPES:
        raise ActionDecisionSchemaError(f"Executor does not support action_type: {text}")
    return text


def _failure_reason(value: str | None, status: str) -> str | None:
    text = _optional_str(value)
    if status == DECISION_FAILED:
        if text is None:
            raise ActionDecisionSchemaError("Failed decisions require failure_reason")
        if text not in SUPPORTED_FAILURE_REASONS:
            raise ActionDecisionSchemaError(f"Unsupported failure_reason: {text}")
        return text
    if text is not None:
        raise ActionDecisionSchemaError("Only failed decisions can include failure_reason")
    return None


def _validate_action_params(
    action_type: str | None,
    x: int | None,
    y: int | None,
    text: str | None,
    keys: list[str] | None,
    seconds: int | float | None,
) -> None:
    if action_type == ACTION_CLICK and (x is None or y is None):
        raise ActionDecisionSchemaError("Click decisions require x and y")
    if action_type == ACTION_TYPE and text is None:
        raise ActionDecisionSchemaError("Type decisions require text")
    if action_type == ACTION_HOTKEY and not keys:
        raise ActionDecisionSchemaError("Hotkey decisions require keys")
    if action_type == ACTION_WAIT and seconds is None:
        raise ActionDecisionSchemaError("Wait decisions require seconds")


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActionDecisionSchemaError(f"ActionDecision {field_name} must be an integer")
    return value


def _optional_seconds(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ActionDecisionSchemaError("ActionDecision seconds must be a non-negative number")
    return value


def _optional_keys(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ActionDecisionSchemaError("ActionDecision keys must be a list")
    keys = []
    for item in value:
        text = str(item).strip()
        if not text:
            raise ActionDecisionSchemaError("ActionDecision keys cannot include empty items")
        keys.append(text)
    return keys


def _dict_field(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionDecisionSchemaError(f"ActionDecision {field_name} must be an object")
    return dict(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = [
    "ACTION_CLICK",
    "ACTION_DECISION_VERSION",
    "ACTION_HOTKEY",
    "ACTION_TYPE",
    "ACTION_WAIT",
    "DECISION_CONTINUE",
    "DECISION_FAILED",
    "DECISION_SUCCESS",
    "FAILURE_EXECUTOR_FAILED",
    "FAILURE_PAGE_MISMATCH",
    "FAILURE_TARGET_NOT_FOUND",
    "ActionDecision",
    "ActionDecisionSchemaError",
    "SUPPORTED_PLANNER_ACTION_TYPES",
    "action_decision_to_executor_payload",
]
