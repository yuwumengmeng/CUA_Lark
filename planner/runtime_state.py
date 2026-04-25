"""运行时状态 schema 脚本。

这个脚本完成成员 B 4.23 第二项：定义 planner loop 运行时要维护的
`RuntimeState`。它只描述和更新状态字段，包括当前任务、当前 step、
页面摘要、候选元素、上一步动作、运行状态、执行历史和失败原因；
真正的动作选择逻辑会放到后续 `mini_planner.py`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from .task_schema import TaskSchemaError, TaskSpec, parse_task


RUNTIME_STATE_SCHEMA_VERSION = "runtime_state.v1"


class RuntimeStateError(ValueError):
    """Raised when runtime state data does not match runtime_state.v1."""


class RuntimeStatus(StrEnum):
    """Planner loop status values for the first-week runtime."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(slots=True)
class RuntimeHistoryItem:
    """One observed step in the planner loop history."""

    step_idx: int
    page_summary: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] | None = None
    action_result: dict[str, Any] | None = None
    status: str = RuntimeStatus.RUNNING.value
    failure_reason: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeHistoryItem":
        return cls(
            step_idx=_require_non_negative_int(data.get("step_idx"), "step_idx"),
            page_summary=_dict_field(data.get("page_summary"), "page_summary"),
            action=_optional_dict_field(data.get("action"), "action"),
            action_result=_optional_dict_field(data.get("action_result"), "action_result"),
            status=_normalize_status(data.get("status", RuntimeStatus.RUNNING.value)),
            failure_reason=_optional_str(data.get("failure_reason")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "page_summary": self.page_summary,
            "action": self.action,
            "action_result": self.action_result,
            "status": self.status,
            "failure_reason": self.failure_reason,
        }


@dataclass(slots=True)
class RuntimeState:
    """Current state consumed and updated by the future MiniPlanner."""

    task: TaskSpec
    step_idx: int = 0
    page_summary: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    last_action: dict[str, Any] | None = None
    status: str = RuntimeStatus.RUNNING.value
    history: list[RuntimeHistoryItem] = field(default_factory=list)
    failure_reason: str | None = None
    version: str = RUNTIME_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.step_idx = _require_non_negative_int(self.step_idx, "step_idx")
        self.status = _normalize_status(self.status)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeState":
        if "task" not in data:
            raise RuntimeStateError("RuntimeState is missing required field: task")

        raw_history = data.get("history", [])
        if not isinstance(raw_history, list):
            raise RuntimeStateError("RuntimeState field history must be a list")

        return cls(
            task=_task_from_value(data["task"]),
            step_idx=_require_non_negative_int(data.get("step_idx", 0), "step_idx"),
            page_summary=_dict_field(data.get("page_summary"), "page_summary"),
            candidates=_candidate_list_field(data.get("candidates"), "candidates"),
            last_action=_optional_dict_field(data.get("last_action"), "last_action"),
            status=_normalize_status(data.get("status", RuntimeStatus.RUNNING.value)),
            history=[RuntimeHistoryItem.from_dict(item) for item in raw_history],
            failure_reason=_optional_str(data.get("failure_reason")),
            version=str(data.get("version") or RUNTIME_STATE_SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "task": self.task.to_dict(),
            "step_idx": self.step_idx,
            "page_summary": self.page_summary,
            "candidates": self.candidates,
            "last_action": self.last_action,
            "status": self.status,
            "history": [item.to_dict() for item in self.history],
            "failure_reason": self.failure_reason,
        }

    def update_ui_state(self, ui_state: Mapping[str, Any]) -> None:
        """Store the latest perception output fields used by the planner."""

        self.page_summary = _dict_field(ui_state.get("page_summary"), "page_summary")
        self.candidates = _candidate_list_field(ui_state.get("candidates"), "candidates")

    def record_step(
        self,
        action: Mapping[str, Any] | None,
        action_result: Mapping[str, Any] | None = None,
        *,
        status: str = RuntimeStatus.RUNNING.value,
        failure_reason: str | None = None,
        advance_step: bool = False,
    ) -> None:
        """Append one history item and update top-level status fields."""

        normalized_status = _normalize_status(status)
        action_dict = dict(action) if action is not None else None
        result_dict = dict(action_result) if action_result is not None else None

        self.last_action = action_dict
        self.status = normalized_status
        self.failure_reason = failure_reason
        self.history.append(
            RuntimeHistoryItem(
                step_idx=self.step_idx,
                page_summary=dict(self.page_summary),
                action=action_dict,
                action_result=result_dict,
                status=normalized_status,
                failure_reason=failure_reason,
            )
        )
        if advance_step:
            self.step_idx += 1


def create_runtime_state(task_input: str | Mapping[str, Any] | TaskSpec) -> RuntimeState:
    """Create a fresh runtime_state.v1 object from a task input."""

    return RuntimeState(task=_task_from_value(task_input))


def _task_from_value(value: str | Mapping[str, Any] | TaskSpec) -> TaskSpec:
    if isinstance(value, TaskSpec):
        return value
    if isinstance(value, (str, Mapping)):
        return parse_task(value)
    raise RuntimeStateError("RuntimeState field task must be a TaskSpec, string, or object")


def _normalize_status(value: Any) -> str:
    text = str(value)
    allowed = {status.value for status in RuntimeStatus}
    if text not in allowed:
        raise RuntimeStateError(f"Unsupported runtime status: {text}")
    return text


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeStateError(f"{field_name} must be a non-negative integer")
    return value


def _dict_field(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeStateError(f"{field_name} must be an object")
    return dict(value)


def _optional_dict_field(value: Any, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _dict_field(value, field_name)


def _candidate_list_field(value: Any, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeStateError(f"{field_name} must be a list")
    candidates: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise RuntimeStateError(f"{field_name} items must be objects")
        candidates.append(dict(item))
    return candidates


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
