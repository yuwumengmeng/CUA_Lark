"""成员 C executor 包入口。

这个包负责把成员 B 输出的动作协议落到真实桌面执行，并把每一步执行结果
和运行工件统一记录下来。当前导出首版基础执行器入口；recorder 和 runner
会继续复用同一套 `ActionResult` 协议。
"""

from .base_executor import (
    ActionDecisionError,
    BaseExecutor,
    DesktopActionBackend,
    ExecutorConfig,
    PyAutoGUIDesktopBackend,
    execute_action,
    resolve_action_point,
)
from .im_skills import (
    IMExecutionSkills,
    click_candidate,
    focus_and_type,
    open_search_by_hotkey,
    press_enter_to_send,
    scroll_result_list,
    wait_for_page_change,
)
from .action_runner import ExecutorWithRecorder, StepExecutionResult, run_step, run_steps
from .recorder import RunArtifactPaths, RunRecorder, RunRecorderError
from .validator import (
    ValidatorError,
    ValidatorResult,
    validate_candidates_contain_text,
    validate_chat_message_visible,
    validate_expected_after_state,
    validate_page_changed,
    validate_text_visible,
    validate_vlm_semantic_pass,
)

__all__ = [
    "ActionDecisionError",
    "BaseExecutor",
    "DesktopActionBackend",
    "ExecutorWithRecorder",
    "ExecutorConfig",
    "IMExecutionSkills",
    "PyAutoGUIDesktopBackend",
    "RunArtifactPaths",
    "RunRecorder",
    "RunRecorderError",
    "StepExecutionResult",
    "ValidatorError",
    "ValidatorResult",
    "click_candidate",
    "execute_action",
    "focus_and_type",
    "open_search_by_hotkey",
    "press_enter_to_send",
    "resolve_action_point",
    "run_step",
    "run_steps",
    "scroll_result_list",
    "validate_candidates_contain_text",
    "validate_chat_message_visible",
    "validate_expected_after_state",
    "validate_page_changed",
    "validate_text_visible",
    "validate_vlm_semantic_pass",
    "wait_for_page_change",
]
