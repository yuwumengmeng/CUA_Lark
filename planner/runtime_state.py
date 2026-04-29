"""成员 B planner 运行时状态 schema。

职责边界：
本文件只定义 planner 在运行过程中需要保存的状态字段；不负责选择 UI 候选、
不调用 VLM、不调用 executor，也不实现第二周期 Planner Loop v2。

公开接口：
`RuntimeState / create_runtime_state`
    第一周期单步 runtime_state.v1，供现有 MiniPlanner 和测试继续使用。
`RuntimeStateV2 / create_runtime_state_v2 / parse_runtime_state_v2`
    第二周期多步 runtime_state.v2，围绕 workflow 记录当前 step、A 侧 ui_state
    语义字段、C 侧 action_result、retry、failure_reason、history 和 conflict。

稳定规则：
runtime_state.v2 字段名固定为 `workflow / task / step_idx / current_ui_state /
current_page_summary / current_vlm_result / current_document_region_summary /
current_screen_meta / last_action_decision / last_action_result / retry_count /
status / failure_reason / history / conflicts / version`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from .task_schema import TaskSpec, parse_task
from .workflow_schema import WorkflowSpec, WorkflowStep, parse_workflow


RUNTIME_STATE_SCHEMA_VERSION = "runtime_state.v1"
RUNTIME_STATE_V2_SCHEMA_VERSION = "runtime_state.v2"


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


@dataclass(slots=True)
class RuntimeConflictItemV2:
    step_idx: int
    step_name: str | None = None
    rule_candidate_id: str | None = None
    vlm_candidate_id: str | None = None
    reason: str = ""
    vlm_confidence: int | float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeConflictItemV2":
        return cls(
            step_idx=_require_non_negative_int(data.get("step_idx"), "step_idx"),
            step_name=_optional_str(data.get("step_name")),
            rule_candidate_id=_optional_str(data.get("rule_candidate_id")),
            vlm_candidate_id=_optional_str(data.get("vlm_candidate_id")),
            reason=str(data.get("reason") or ""),
            vlm_confidence=_optional_number(data.get("vlm_confidence"), "vlm_confidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "step_name": self.step_name,
            "rule_candidate_id": self.rule_candidate_id,
            "vlm_candidate_id": self.vlm_candidate_id,
            "reason": self.reason,
            "vlm_confidence": self.vlm_confidence,
        }


@dataclass(slots=True)
class RuntimeHistoryItemV2:
    step_idx: int
    step_name: str | None = None
    ui_state: dict[str, Any] = field(default_factory=dict)
    page_summary: dict[str, Any] = field(default_factory=dict)
    vlm_result: dict[str, Any] = field(default_factory=dict)
    document_region_summary: dict[str, Any] = field(default_factory=dict)
    screen_meta: dict[str, Any] = field(default_factory=dict)
    action_decision: dict[str, Any] | None = None
    action_result: dict[str, Any] | None = None
    retry_count: int = 0
    status: str = RuntimeStatus.RUNNING.value
    failure_reason: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeHistoryItemV2":
        return cls(
            step_idx=_require_non_negative_int(data.get("step_idx"), "step_idx"),
            step_name=_optional_str(data.get("step_name")),
            ui_state=_dict_field(data.get("ui_state"), "ui_state"),
            page_summary=_dict_field(data.get("page_summary"), "page_summary"),
            vlm_result=_dict_field(data.get("vlm_result"), "vlm_result"),
            document_region_summary=_dict_field(
                data.get("document_region_summary"),
                "document_region_summary",
            ),
            screen_meta=_dict_field(data.get("screen_meta"), "screen_meta"),
            action_decision=_optional_dict_field(data.get("action_decision"), "action_decision"),
            action_result=_optional_dict_field(data.get("action_result"), "action_result"),
            retry_count=_require_non_negative_int(data.get("retry_count", 0), "retry_count"),
            status=_normalize_status(data.get("status", RuntimeStatus.RUNNING.value)),
            failure_reason=_optional_str(data.get("failure_reason")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "step_name": self.step_name,
            "ui_state": self.ui_state,
            "page_summary": self.page_summary,
            "vlm_result": self.vlm_result,
            "document_region_summary": self.document_region_summary,
            "screen_meta": self.screen_meta,
            "action_decision": self.action_decision,
            "action_result": self.action_result,
            "retry_count": self.retry_count,
            "status": self.status,
            "failure_reason": self.failure_reason,
        }


@dataclass(slots=True)
class RuntimeStateV2:
    workflow: WorkflowSpec
    task: dict[str, Any] = field(default_factory=dict)
    step_idx: int = 0
    current_ui_state: dict[str, Any] = field(default_factory=dict)
    current_page_summary: dict[str, Any] = field(default_factory=dict)
    current_vlm_result: dict[str, Any] = field(default_factory=dict)
    current_document_region_summary: dict[str, Any] = field(default_factory=dict)
    current_screen_meta: dict[str, Any] = field(default_factory=dict)
    last_action_decision: dict[str, Any] | None = None
    last_action_result: dict[str, Any] | None = None
    retry_count: int = 0
    status: str = RuntimeStatus.RUNNING.value
    failure_reason: str | None = None
    history: list[RuntimeHistoryItemV2] = field(default_factory=list)
    conflicts: list[RuntimeConflictItemV2] = field(default_factory=list)
    version: str = RUNTIME_STATE_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.step_idx = _require_non_negative_int(self.step_idx, "step_idx")
        self.retry_count = _require_non_negative_int(self.retry_count, "retry_count")
        self.status = _normalize_status(self.status)
        self.task = _dict_field(self.task, "task")
        self.current_ui_state = _dict_field(self.current_ui_state, "current_ui_state")
        self.current_page_summary = _dict_field(
            self.current_page_summary,
            "current_page_summary",
        )
        self.current_vlm_result = _dict_field(self.current_vlm_result, "current_vlm_result")
        self.current_document_region_summary = _dict_field(
            self.current_document_region_summary,
            "current_document_region_summary",
        )
        self.current_screen_meta = _dict_field(self.current_screen_meta, "current_screen_meta")
        self.last_action_decision = _optional_dict_field(
            self.last_action_decision,
            "last_action_decision",
        )
        self.last_action_result = _optional_dict_field(
            self.last_action_result,
            "last_action_result",
        )

    @property
    def current_step(self) -> WorkflowStep | None:
        if self.step_idx >= len(self.workflow.steps):
            return None
        return self.workflow.steps[self.step_idx]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeStateV2":
        if "workflow" not in data:
            raise RuntimeStateError("RuntimeStateV2 is missing required field: workflow")

        raw_history = data.get("history", [])
        if not isinstance(raw_history, list):
            raise RuntimeStateError("RuntimeStateV2 field history must be a list")
        raw_conflicts = data.get("conflicts", [])
        if not isinstance(raw_conflicts, list):
            raise RuntimeStateError("RuntimeStateV2 field conflicts must be a list")

        return cls(
            workflow=parse_workflow(data["workflow"]),
            task=_dict_field(data.get("task"), "task"),
            step_idx=_require_non_negative_int(data.get("step_idx", 0), "step_idx"),
            current_ui_state=_dict_field(data.get("current_ui_state"), "current_ui_state"),
            current_page_summary=_dict_field(
                data.get("current_page_summary"),
                "current_page_summary",
            ),
            current_vlm_result=_dict_field(data.get("current_vlm_result"), "current_vlm_result"),
            current_document_region_summary=_dict_field(
                data.get("current_document_region_summary"),
                "current_document_region_summary",
            ),
            current_screen_meta=_dict_field(data.get("current_screen_meta"), "current_screen_meta"),
            last_action_decision=_optional_dict_field(
                data.get("last_action_decision"),
                "last_action_decision",
            ),
            last_action_result=_optional_dict_field(
                data.get("last_action_result"),
                "last_action_result",
            ),
            retry_count=_require_non_negative_int(data.get("retry_count", 0), "retry_count"),
            status=_normalize_status(data.get("status", RuntimeStatus.RUNNING.value)),
            failure_reason=_optional_str(data.get("failure_reason")),
            history=[RuntimeHistoryItemV2.from_dict(item) for item in raw_history],
            conflicts=[RuntimeConflictItemV2.from_dict(item) for item in raw_conflicts],
            version=str(data.get("version") or RUNTIME_STATE_V2_SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        current_step = self.current_step
        return {
            "version": self.version,
            "workflow": self.workflow.to_dict(),
            "task": dict(self.task),
            "step_idx": self.step_idx,
            "current_step": None if current_step is None else current_step.to_dict(),
            "current_ui_state": self.current_ui_state,
            "current_page_summary": self.current_page_summary,
            "current_vlm_result": self.current_vlm_result,
            "current_document_region_summary": self.current_document_region_summary,
            "current_screen_meta": self.current_screen_meta,
            "last_action_decision": self.last_action_decision,
            "last_action_result": self.last_action_result,
            "retry_count": self.retry_count,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "history": [item.to_dict() for item in self.history],
            "conflicts": [item.to_dict() for item in self.conflicts],
        }

    def update_ui_state(self, ui_state: Mapping[str, Any]) -> None:
        self.current_ui_state = _dict_field(ui_state, "ui_state")
        self.current_page_summary = _dict_field(ui_state.get("page_summary"), "page_summary")
        self.current_vlm_result = _dict_field(ui_state.get("vlm_result"), "vlm_result")
        self.current_document_region_summary = _dict_field(
            ui_state.get("document_region_summary"),
            "document_region_summary",
        )
        self.current_screen_meta = _dict_field(ui_state.get("screen_meta"), "screen_meta")

    def record_action(
        self,
        action_decision: Mapping[str, Any],
        action_result: Mapping[str, Any] | None = None,
        *,
        status: str = RuntimeStatus.RUNNING.value,
        failure_reason: str | None = None,
    ) -> None:
        normalized_status = _normalize_status(status)
        self.last_action_decision = dict(action_decision)
        self.last_action_result = None if action_result is None else dict(action_result)
        self.status = normalized_status
        self.failure_reason = failure_reason
        self.history.append(
            RuntimeHistoryItemV2(
                step_idx=self.step_idx,
                step_name=None if self.current_step is None else self.current_step.step_name,
                ui_state=dict(self.current_ui_state),
                page_summary=dict(self.current_page_summary),
                vlm_result=dict(self.current_vlm_result),
                document_region_summary=dict(self.current_document_region_summary),
                screen_meta=dict(self.current_screen_meta),
                action_decision=self.last_action_decision,
                action_result=self.last_action_result,
                retry_count=self.retry_count,
                status=normalized_status,
                failure_reason=failure_reason,
            )
        )

    def advance_step(self) -> None:
        self.step_idx += 1
        self.retry_count = 0
        if self.step_idx >= len(self.workflow.steps):
            self.status = RuntimeStatus.SUCCESS.value

    def increment_retry(self) -> int:
        self.retry_count += 1
        return self.retry_count

    def mark_failed(self, failure_reason: str) -> None:
        self.status = RuntimeStatus.FAILED.value
        self.failure_reason = str(failure_reason)

    def record_conflict(
        self,
        *,
        rule_candidate_id: str | None,
        vlm_candidate_id: str | None,
        reason: str,
        vlm_confidence: int | float | None = None,
    ) -> None:
        self.conflicts.append(
            RuntimeConflictItemV2(
                step_idx=self.step_idx,
                step_name=None if self.current_step is None else self.current_step.step_name,
                rule_candidate_id=rule_candidate_id,
                vlm_candidate_id=vlm_candidate_id,
                reason=reason,
                vlm_confidence=vlm_confidence,
            )
        )


def create_runtime_state(task_input: str | Mapping[str, Any] | TaskSpec) -> RuntimeState:
    """Create a fresh runtime_state.v1 object from a task input."""

    return RuntimeState(task=_task_from_value(task_input))


def create_runtime_state_v2(
    workflow_input: Mapping[str, Any] | WorkflowSpec,
    *,
    task: Mapping[str, Any] | None = None,
) -> RuntimeStateV2:
    """Create a fresh runtime_state.v2 object from a workflow."""

    workflow = parse_workflow(workflow_input)
    task_payload = dict(task or {"instruction": workflow.instruction, "params": workflow.params})
    return RuntimeStateV2(workflow=workflow, task=task_payload)


def parse_runtime_state_v2(state_input: Mapping[str, Any] | RuntimeStateV2) -> RuntimeStateV2:
    """Parse runtime_state.v2 from a dict or pass through an existing object."""

    if isinstance(state_input, RuntimeStateV2):
        return state_input
    if isinstance(state_input, Mapping):
        return RuntimeStateV2.from_dict(state_input)
    raise RuntimeStateError("RuntimeStateV2 input must be a RuntimeStateV2 or object")


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


def _optional_number(value: Any, field_name: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeStateError(f"{field_name} must be a number")
    return value
