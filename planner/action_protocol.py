"""成员 B planner 到 executor 的动作协议。

职责边界：
本文件只定义成员 B 输出给成员 C executor 的 action decision 协议，包括稳定字段、
状态枚举、动作类型和失败原因。它不负责选择 UI 元素、不负责推进 runtime state，
也不负责真实鼠标键盘执行。

公开接口：
`ActionDecision`
    第一周期 action_decision.v1 数据结构，提供 `continue_ / success / failed` 构造方法、
    `from_dict(...)` 解析方法、`to_dict(...)` 日志字典和 `to_executor_dict(...)`
    执行器消费字典。
`action_decision_to_executor_payload(...)`
    从 action decision 中剥离 B 内部字段，得到传给 executor 的稳定 payload。
`ActionDecisionV2`
    第二周期 action_decision.v2 数据结构，固定 `step_name / target_candidate_id /
    retry_count / expected_after_state` 等字段，支持 click、type、hotkey、wait、scroll；
    press enter 暂时用 `action_type=hotkey, keys=["enter"]` 表达。
`action_decision_v2_to_executor_payload(...)`
    将 action_decision.v2 转成 C 侧可执行 payload。

字段协议：
v1 稳定字段为 `status / action_type / target_element_id / x / y / text / keys /
repeat / seconds / reason / failure_reason / version`。v2 稳定字段为 `action_type /
step_name / target_candidate_id / x / y / text / keys / reason / retry_count /
expected_after_state / version`。`planner_meta` 只允许 B 内部使用，成员 C 不应依赖它。

协作规则：
1. C 侧只需要读取 `action_type / target_element_id / x / y / text / keys /
   repeat / seconds / reason` 等可执行字段。
2. 新增动作类型时，必须同时确认 `schemas/action_result.py` 的 executor 支持情况。
3. 修改字段名或状态值时，必须同步更新 `tests/testB/test_action_protocol.py`
   和 `tests/testB/test_action_protocol_v2.py`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from schemas.action_result import (
    ACTION_CLICK,
    ACTION_HOTKEY,
    ACTION_SCROLL,
    ACTION_TYPE,
    ACTION_WAIT,
    SUPPORTED_EXECUTOR_ACTION_TYPES,
)


ACTION_DECISION_VERSION = "action_decision.v1"
ACTION_DECISION_V2_VERSION = "action_decision.v2"

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

SUPPORTED_PLANNER_ACTION_TYPES_V2 = {
    ACTION_CLICK,
    ACTION_TYPE,
    ACTION_HOTKEY,
    ACTION_WAIT,
    ACTION_SCROLL,
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
    repeat: int | None = None
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
        repeat = _optional_repeat(self.repeat)
        planner_meta = _dict_field(self.planner_meta, "planner_meta")

        _validate_action_params(action_type, x, y, self.text, keys, repeat, seconds)

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "action_type", action_type)
        object.__setattr__(self, "failure_reason", failure_reason)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "seconds", seconds)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "repeat", repeat)
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
        repeat: int | None = None,
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
            repeat=repeat,
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
            repeat=data.get("repeat"),
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
            "repeat": self.repeat,
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


class ActionDecisionV2SchemaError(ValueError):
    """Raised when action_decision.v2 data is malformed."""


@dataclass(frozen=True, slots=True)
class ActionDecisionV2:
    """Second-cycle action protocol produced by member B for member C."""

    status: str
    step_name: str
    action_type: str | None = None
    target_candidate_id: str | None = None
    x: int | None = None
    y: int | None = None
    text: str | None = None
    keys: list[str] | None = None
    repeat: int | None = None
    seconds: int | float | None = None
    direction: str | None = None
    amount: int | None = None
    reason: str = ""
    retry_count: int = 0
    expected_after_state: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    version: str = ACTION_DECISION_V2_VERSION

    def __post_init__(self) -> None:
        status = _status_v2(self.status)
        step_name = _required_text_v2(self.step_name, "step_name")
        action_type = _action_type_v2(self.action_type, status)
        failure_reason = _failure_reason_v2(self.failure_reason, status)
        x = _optional_int_v2(self.x, "x")
        y = _optional_int_v2(self.y, "y")
        seconds = _optional_seconds_v2(self.seconds)
        keys = _optional_keys_v2(self.keys)
        repeat = _optional_repeat_v2(self.repeat)
        retry_count = _non_negative_int_v2(self.retry_count, "retry_count")
        expected_after_state = _dict_field_v2(
            self.expected_after_state,
            "expected_after_state",
        )
        direction = _optional_str_v2(self.direction)
        amount = _optional_int_v2(self.amount, "amount")

        _validate_action_params_v2(action_type, x, y, self.text, keys, repeat, seconds, amount)

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "step_name", step_name)
        object.__setattr__(self, "action_type", action_type)
        object.__setattr__(self, "target_candidate_id", _optional_str_v2(self.target_candidate_id))
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "repeat", repeat)
        object.__setattr__(self, "seconds", seconds)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "retry_count", retry_count)
        object.__setattr__(self, "expected_after_state", expected_after_state)
        object.__setattr__(self, "failure_reason", failure_reason)
        object.__setattr__(self, "version", str(self.version or ACTION_DECISION_V2_VERSION))

    @classmethod
    def continue_(
        cls,
        *,
        step_name: str,
        action_type: str,
        target_candidate_id: str | None = None,
        x: int | None = None,
        y: int | None = None,
        text: str | None = None,
        keys: list[str] | None = None,
        repeat: int | None = None,
        seconds: int | float | None = None,
        direction: str | None = None,
        amount: int | None = None,
        reason: str = "",
        retry_count: int = 0,
        expected_after_state: Mapping[str, Any] | None = None,
    ) -> "ActionDecisionV2":
        return cls(
            status=DECISION_CONTINUE,
            step_name=step_name,
            action_type=action_type,
            target_candidate_id=target_candidate_id,
            x=x,
            y=y,
            text=text,
            keys=keys,
            repeat=repeat,
            seconds=seconds,
            direction=direction,
            amount=amount,
            reason=reason,
            retry_count=retry_count,
            expected_after_state=dict(expected_after_state or {}),
        )

    @classmethod
    def click_candidate(
        cls,
        *,
        step_name: str,
        target_candidate_id: str,
        x: int,
        y: int,
        reason: str = "",
        retry_count: int = 0,
        expected_after_state: Mapping[str, Any] | None = None,
    ) -> "ActionDecisionV2":
        return cls.continue_(
            step_name=step_name,
            action_type=ACTION_CLICK,
            target_candidate_id=target_candidate_id,
            x=x,
            y=y,
            reason=reason,
            retry_count=retry_count,
            expected_after_state=expected_after_state,
        )

    @classmethod
    def click_point(
        cls,
        *,
        step_name: str,
        x: int,
        y: int,
        reason: str = "",
        retry_count: int = 0,
        expected_after_state: Mapping[str, Any] | None = None,
    ) -> "ActionDecisionV2":
        return cls.continue_(
            step_name=step_name,
            action_type=ACTION_CLICK,
            x=x,
            y=y,
            reason=reason,
            retry_count=retry_count,
            expected_after_state=expected_after_state,
        )

    @classmethod
    def type_text(
        cls,
        *,
        step_name: str,
        text: str,
        target_candidate_id: str | None = None,
        x: int | None = None,
        y: int | None = None,
        reason: str = "",
        retry_count: int = 0,
        expected_after_state: Mapping[str, Any] | None = None,
    ) -> "ActionDecisionV2":
        return cls.continue_(
            step_name=step_name,
            action_type=ACTION_TYPE,
            target_candidate_id=target_candidate_id,
            x=x,
            y=y,
            text=text,
            reason=reason,
            retry_count=retry_count,
            expected_after_state=expected_after_state,
        )

    @classmethod
    def hotkey(
        cls,
        *,
        step_name: str,
        keys: list[str],
        repeat: int | None = None,
        reason: str = "",
        retry_count: int = 0,
        expected_after_state: Mapping[str, Any] | None = None,
    ) -> "ActionDecisionV2":
        return cls.continue_(
            step_name=step_name,
            action_type=ACTION_HOTKEY,
            keys=keys,
            repeat=repeat,
            reason=reason,
            retry_count=retry_count,
            expected_after_state=expected_after_state,
        )

    @classmethod
    def press_enter(
        cls,
        *,
        step_name: str,
        reason: str = "",
        retry_count: int = 0,
        expected_after_state: Mapping[str, Any] | None = None,
    ) -> "ActionDecisionV2":
        return cls.hotkey(
            step_name=step_name,
            keys=["enter"],
            reason=reason or "press enter",
            retry_count=retry_count,
            expected_after_state=expected_after_state,
        )

    @classmethod
    def wait(
        cls,
        *,
        step_name: str,
        seconds: int | float,
        reason: str = "",
        retry_count: int = 0,
        expected_after_state: Mapping[str, Any] | None = None,
    ) -> "ActionDecisionV2":
        return cls.continue_(
            step_name=step_name,
            action_type=ACTION_WAIT,
            seconds=seconds,
            reason=reason,
            retry_count=retry_count,
            expected_after_state=expected_after_state,
        )

    @classmethod
    def scroll(
        cls,
        *,
        step_name: str,
        direction: str = "down",
        amount: int | None = None,
        x: int | None = None,
        y: int | None = None,
        reason: str = "",
        retry_count: int = 0,
        expected_after_state: Mapping[str, Any] | None = None,
    ) -> "ActionDecisionV2":
        return cls.continue_(
            step_name=step_name,
            action_type=ACTION_SCROLL,
            x=x,
            y=y,
            direction=direction,
            amount=amount,
            reason=reason,
            retry_count=retry_count,
            expected_after_state=expected_after_state,
        )

    @classmethod
    def success(cls, *, step_name: str, reason: str) -> "ActionDecisionV2":
        return cls(status=DECISION_SUCCESS, step_name=step_name, reason=reason)

    @classmethod
    def failed(
        cls,
        failure_reason: str,
        reason: str,
        *,
        step_name: str,
        retry_count: int = 0,
    ) -> "ActionDecisionV2":
        return cls(
            status=DECISION_FAILED,
            step_name=step_name,
            reason=reason,
            failure_reason=failure_reason,
            retry_count=retry_count,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionDecisionV2":
        if "status" not in data:
            raise ActionDecisionV2SchemaError("ActionDecisionV2 is missing required field: status")
        return cls(
            status=str(data["status"]),
            step_name=str(data.get("step_name") or ""),
            action_type=_optional_str_v2(data.get("action_type")),
            target_candidate_id=_optional_str_v2(data.get("target_candidate_id")),
            x=data.get("x"),
            y=data.get("y"),
            text=_optional_str_v2(data.get("text")),
            keys=data.get("keys"),
            repeat=data.get("repeat"),
            seconds=data.get("seconds"),
            direction=_optional_str_v2(data.get("direction")),
            amount=data.get("amount"),
            reason=str(data.get("reason") or ""),
            retry_count=_non_negative_int_v2(data.get("retry_count", 0), "retry_count"),
            expected_after_state=_dict_field_v2(
                data.get("expected_after_state", {}),
                "expected_after_state",
            ),
            failure_reason=_optional_str_v2(data.get("failure_reason")),
            version=str(data.get("version") or ACTION_DECISION_V2_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status,
            "action_type": self.action_type,
            "step_name": self.step_name,
            "target_candidate_id": self.target_candidate_id,
            "x": self.x,
            "y": self.y,
            "text": self.text,
            "keys": None if self.keys is None else list(self.keys),
            "repeat": self.repeat,
            "seconds": self.seconds,
            "direction": self.direction,
            "amount": self.amount,
            "reason": self.reason,
            "retry_count": self.retry_count,
            "expected_after_state": dict(self.expected_after_state),
            "failure_reason": self.failure_reason,
        }

    def to_executor_dict(self) -> dict[str, Any]:
        result = self.to_dict()
        result["target_element_id"] = self.target_candidate_id
        result.pop("version", None)
        result.pop("status", None)
        result.pop("step_name", None)
        result.pop("retry_count", None)
        result.pop("expected_after_state", None)
        result.pop("failure_reason", None)
        return result


def action_decision_v2_to_executor_payload(decision: Mapping[str, Any]) -> dict[str, Any]:
    return ActionDecisionV2.from_dict(decision).to_executor_dict()


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
    repeat: int | None,
    seconds: int | float | None,
) -> None:
    if action_type == ACTION_CLICK and (x is None or y is None):
        raise ActionDecisionSchemaError("Click decisions require x and y")
    if action_type == ACTION_TYPE and text is None:
        raise ActionDecisionSchemaError("Type decisions require text")
    if action_type == ACTION_HOTKEY and not keys:
        raise ActionDecisionSchemaError("Hotkey decisions require keys")
    if repeat is not None and action_type != ACTION_HOTKEY:
        raise ActionDecisionSchemaError("Only hotkey decisions can include repeat")
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


def _optional_repeat(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ActionDecisionSchemaError("ActionDecision repeat must be a positive integer")
    return value


def _dict_field(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionDecisionSchemaError(f"ActionDecision {field_name} must be an object")
    return dict(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _status_v2(value: str) -> str:
    text = str(value)
    if text not in SUPPORTED_DECISION_STATUSES:
        raise ActionDecisionV2SchemaError(f"Unsupported action_decision.v2 status: {text}")
    return text


def _action_type_v2(value: str | None, status: str) -> str | None:
    text = _optional_str_v2(value)
    if status != DECISION_CONTINUE:
        if text is not None:
            raise ActionDecisionV2SchemaError("Non-continue decisions cannot include action_type")
        return None
    if text is None:
        raise ActionDecisionV2SchemaError("Continue decisions require action_type")
    if text not in SUPPORTED_PLANNER_ACTION_TYPES_V2:
        raise ActionDecisionV2SchemaError(f"Unsupported action_decision.v2 action_type: {text}")
    if text not in SUPPORTED_EXECUTOR_ACTION_TYPES:
        raise ActionDecisionV2SchemaError(f"Executor does not support action_type: {text}")
    return text


def _failure_reason_v2(value: str | None, status: str) -> str | None:
    text = _optional_str_v2(value)
    if status == DECISION_FAILED:
        if text is None:
            raise ActionDecisionV2SchemaError("Failed decisions require failure_reason")
        return text
    if text is not None:
        raise ActionDecisionV2SchemaError("Only failed decisions can include failure_reason")
    return None


def _validate_action_params_v2(
    action_type: str | None,
    x: int | None,
    y: int | None,
    text: str | None,
    keys: list[str] | None,
    repeat: int | None,
    seconds: int | float | None,
    amount: int | None,
) -> None:
    if action_type == ACTION_CLICK and (x is None or y is None):
        raise ActionDecisionV2SchemaError("Click decisions require x and y")
    if action_type == ACTION_TYPE and not str(text or "").strip():
        raise ActionDecisionV2SchemaError("Type decisions require non-empty text")
    if action_type == ACTION_HOTKEY and not keys:
        raise ActionDecisionV2SchemaError("Hotkey decisions require keys")
    if action_type != ACTION_HOTKEY and repeat is not None:
        raise ActionDecisionV2SchemaError("Only hotkey decisions can include repeat")
    if action_type == ACTION_WAIT and seconds is None:
        raise ActionDecisionV2SchemaError("Wait decisions require seconds")
    if action_type == ACTION_SCROLL and amount == 0:
        raise ActionDecisionV2SchemaError("Scroll amount cannot be zero")


def _required_text_v2(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ActionDecisionV2SchemaError(f"ActionDecisionV2 {field_name} cannot be empty")
    return text


def _optional_str_v2(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int_v2(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActionDecisionV2SchemaError(f"ActionDecisionV2 {field_name} must be an integer")
    return value


def _optional_seconds_v2(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ActionDecisionV2SchemaError("ActionDecisionV2 seconds must be a non-negative number")
    return value


def _optional_keys_v2(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ActionDecisionV2SchemaError("ActionDecisionV2 keys must be a list")
    keys = [str(item).strip() for item in value if str(item).strip()]
    if len(keys) != len(value):
        raise ActionDecisionV2SchemaError("ActionDecisionV2 keys cannot include empty items")
    return keys


def _optional_repeat_v2(value: Any) -> int | None:
    if value is None:
        return None
    repeat = _non_negative_int_v2(value, "repeat")
    if repeat <= 0:
        raise ActionDecisionV2SchemaError("ActionDecisionV2 repeat must be positive")
    return repeat


def _non_negative_int_v2(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ActionDecisionV2SchemaError(
            f"ActionDecisionV2 {field_name} must be a non-negative integer"
        )
    return value


def _dict_field_v2(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionDecisionV2SchemaError(f"ActionDecisionV2 {field_name} must be an object")
    return dict(value)


__all__ = [
    "ACTION_CLICK",
    "ACTION_DECISION_VERSION",
    "ACTION_DECISION_V2_VERSION",
    "ACTION_HOTKEY",
    "ACTION_SCROLL",
    "ACTION_TYPE",
    "ACTION_WAIT",
    "DECISION_CONTINUE",
    "DECISION_FAILED",
    "DECISION_SUCCESS",
    "FAILURE_EXECUTOR_FAILED",
    "FAILURE_PAGE_MISMATCH",
    "FAILURE_TARGET_NOT_FOUND",
    "ActionDecision",
    "ActionDecisionV2",
    "ActionDecisionV2SchemaError",
    "ActionDecisionSchemaError",
    "SUPPORTED_PLANNER_ACTION_TYPES",
    "SUPPORTED_PLANNER_ACTION_TYPES_V2",
    "action_decision_v2_to_executor_payload",
    "action_decision_to_executor_payload",
]
