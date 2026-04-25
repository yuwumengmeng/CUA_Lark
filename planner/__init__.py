"""planner 包入口。

职责边界：
这个文件只负责把外部常用的成员 B 模块统一导出，方便后面用
`from planner import ...` 直接拿到任务表示、运行时状态和最小 planner。
它不放业务逻辑，也不定义新的字段协议。

公开接口：
`TaskSpec / TaskStep / parse_task`
    4.23 首版任务表示。
`RuntimeState / RuntimeStatus / create_runtime_state`
    4.23 runtime state schema。
`ActionDecision`
    4.24 planner 到 executor 的动作协议。
`MiniPlanner / plan_next_action`
    4.24/4.25 最小 planner loop 和下一步动作决策入口。
`run_project_once`
    项目内接入 A 的 `get_ui_state()` 和 C 的 `execute_action()` 的便捷入口。
"""

from .action_protocol import (
    ACTION_CLICK,
    ACTION_DECISION_VERSION,
    ACTION_HOTKEY,
    ACTION_TYPE,
    ACTION_WAIT,
    DECISION_CONTINUE,
    DECISION_FAILED,
    DECISION_SUCCESS,
    FAILURE_EXECUTOR_FAILED,
    FAILURE_PAGE_MISMATCH,
    FAILURE_TARGET_NOT_FOUND,
    ActionDecision,
    action_decision_to_executor_payload,
)
from .mini_planner import MiniPlanner, plan_next_action, run_project_once
from .task_schema import TaskSpec, TaskStep, parse_task
from .runtime_state import RuntimeState, RuntimeStatus, create_runtime_state

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
    "MiniPlanner",
    "RuntimeState",
    "RuntimeStatus",
    "TaskSpec",
    "TaskStep",
    "action_decision_to_executor_payload",
    "create_runtime_state",
    "plan_next_action",
    "parse_task",
    "run_project_once",
]
