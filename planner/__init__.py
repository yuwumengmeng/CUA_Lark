"""planner 包入口。

职责边界：
这个文件只负责把外部常用的成员 B 模块统一导出，方便后面用
`from planner import ...` 直接拿到任务表示、运行时状态、动作协议和最小 planner。
它不放业务逻辑，也不在这里定义新的字段协议。

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
`WorkflowSpec / WorkflowStep / build_im_message_workflow`
    第二周期 workflow.v2 流程表示。
`RuntimeStateV2 / create_runtime_state_v2 / parse_runtime_state_v2`
    第二周期 runtime_state.v2 多步流程运行态。
`ActionDecisionV2 / action_decision_v2_to_executor_payload`
    第二周期 action_decision.v2 动作协议。
`PlannerLoopV2 / run_workflow_once / run_workflow_until_done`
    第二周期 Planner Loop v2 多步 workflow 推进入口，包含基础 fallback 和 step 成功判断。
"""

from .action_protocol import (
    ACTION_CLICK,
    ACTION_DECISION_VERSION,
    ACTION_DECISION_V2_VERSION,
    ACTION_HOTKEY,
    ACTION_SCROLL,
    ACTION_TYPE,
    ACTION_WAIT,
    DECISION_CONTINUE,
    DECISION_FAILED,
    DECISION_SUCCESS,
    FAILURE_EXECUTOR_FAILED,
    FAILURE_PAGE_MISMATCH,
    FAILURE_TARGET_NOT_FOUND,
    ActionDecision,
    ActionDecisionV2,
    action_decision_v2_to_executor_payload,
    action_decision_to_executor_payload,
)
from .mini_planner import MiniPlanner, plan_next_action, run_project_once
from .planner_loop_v2 import (
    FAILURE_CANDIDATE_CONFLICT,
    FAILURE_CANDIDATE_INVALID,
    FAILURE_COORDINATE_INVALID,
    FAILURE_FALLBACK_FAILED,
    FAILURE_MAX_RETRY_EXCEEDED,
    FAILURE_PAGE_NOT_CHANGED,
    FAILURE_VALIDATION_FAILED,
    PlannerLoopV2,
    PlannerLoopV2Result,
    FAILURE_VLM_UNCERTAIN,
    plan_next_workflow_action,
    run_workflow_once,
    run_workflow_until_done,
)
from .task_schema import TaskSpec, TaskStep, parse_task
from .runtime_state import (
    RUNTIME_STATE_V2_SCHEMA_VERSION,
    RuntimeConflictItemV2,
    RuntimeHistoryItemV2,
    RuntimeState,
    RuntimeStateV2,
    RuntimeStatus,
    create_runtime_state,
    create_runtime_state_v2,
    parse_runtime_state_v2,
)
from .workflow_schema import (
    STEP_FOCUS_MESSAGE_INPUT,
    STEP_OPEN_CHAT,
    STEP_OPEN_SEARCH,
    STEP_SEND_MESSAGE,
    STEP_TYPE_KEYWORD,
    STEP_VERIFY_MESSAGE_SENT,
    WORKFLOW_SCHEMA_VERSION,
    WORKFLOW_TYPE_IM_MESSAGE,
    WorkflowSchemaError,
    WorkflowSpec,
    WorkflowStep,
    build_im_message_workflow,
    build_sample_im_workflows,
    parse_workflow,
)

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
    "FAILURE_CANDIDATE_CONFLICT",
    "FAILURE_CANDIDATE_INVALID",
    "FAILURE_COORDINATE_INVALID",
    "FAILURE_PAGE_MISMATCH",
    "FAILURE_PAGE_NOT_CHANGED",
    "FAILURE_FALLBACK_FAILED",
    "FAILURE_MAX_RETRY_EXCEEDED",
    "FAILURE_TARGET_NOT_FOUND",
    "FAILURE_VALIDATION_FAILED",
    "FAILURE_VLM_UNCERTAIN",
    "ActionDecision",
    "ActionDecisionV2",
    "MiniPlanner",
    "PlannerLoopV2",
    "PlannerLoopV2Result",
    "RUNTIME_STATE_V2_SCHEMA_VERSION",
    "RuntimeConflictItemV2",
    "RuntimeHistoryItemV2",
    "RuntimeState",
    "RuntimeStateV2",
    "RuntimeStatus",
    "STEP_FOCUS_MESSAGE_INPUT",
    "STEP_OPEN_CHAT",
    "STEP_OPEN_SEARCH",
    "STEP_SEND_MESSAGE",
    "STEP_TYPE_KEYWORD",
    "STEP_VERIFY_MESSAGE_SENT",
    "TaskSpec",
    "TaskStep",
    "WORKFLOW_SCHEMA_VERSION",
    "WORKFLOW_TYPE_IM_MESSAGE",
    "WorkflowSchemaError",
    "WorkflowSpec",
    "WorkflowStep",
    "action_decision_v2_to_executor_payload",
    "action_decision_to_executor_payload",
    "build_im_message_workflow",
    "build_sample_im_workflows",
    "create_runtime_state",
    "create_runtime_state_v2",
    "plan_next_action",
    "plan_next_workflow_action",
    "parse_runtime_state_v2",
    "parse_workflow",
    "parse_task",
    "run_project_once",
    "run_workflow_once",
    "run_workflow_until_done",
]
