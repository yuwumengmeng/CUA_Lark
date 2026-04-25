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
__all__ = [
    "ActionDecisionError",
    "BaseExecutor",
    "DesktopActionBackend",
    "ExecutorConfig",
    "PyAutoGUIDesktopBackend",
    "execute_action",
    "resolve_action_point",
]
