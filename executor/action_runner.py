"""成员 C 4.24/4.25 Executor + Recorder 串联入口。

职责边界：
本文件负责把单步动作执行和运行工件记录绑在一起：before 截图、执行动作、
after 截图、写入 action/result/screenshot 日志并更新 summary。它不负责
planner 业务语义，也不做 OCR/UIA 感知。

公开接口：
`ExecutorWithRecorder`
    首版统一 runner，提供 `run_step(action_decision, step_idx=...)`。
`StepExecutionResult`
    单步执行返回结构，包含 `action_result`、before/after 截图信息和 summary。
`run_step(...)`
    便捷函数，适合脚本或成员 B 临时联调。

artifact 规则：
runner 使用 `RunRecorder` 的目录规则：
`artifacts/runs/<run_id>/task.json`
`artifacts/runs/<run_id>/actions.jsonl`
`artifacts/runs/<run_id>/results.jsonl`
`artifacts/runs/<run_id>/screenshot_log.jsonl`
`artifacts/runs/<run_id>/screenshots/step_xxx_before.png`
`artifacts/runs/<run_id>/screenshots/step_xxx_after.png`
`artifacts/runs/<run_id>/summary.json`

稳定性规则：
1. before 截图失败时不执行动作，直接记录失败结果。
2. after 截图失败时把本步结果标记失败，避免 artifact 不完整却被认为成功。
3. executor 内部异常仍由 `BaseExecutor` 转换为 `ActionResult`。

真实环境依赖：
默认截图器会延迟导入 `capture.ScreenshotCapturer`，真实运行需要截图依赖和
桌面权限。单元测试注入 fake capturer，不访问真实桌面。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from schemas import ActionResult, action_result_from_exception, utc_now

from .base_executor import BaseExecutor, DesktopActionBackend, ExecutorConfig
from .recorder import RunRecorder


class StepScreenshotCapturer(Protocol):
    def capture_before(self, *, run_id: str | None = None, step_idx: int = 0) -> Any:
        ...

    def capture_after(self, *, run_id: str | None = None, step_idx: int = 0) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class StepExecutionResult:
    run_id: str
    step_idx: int
    action_result: ActionResult
    before_screenshot: dict[str, Any] | None
    after_screenshot: dict[str, Any] | None
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step_idx": self.step_idx,
            "action_result": self.action_result.to_dict(),
            "before_screenshot": self.before_screenshot,
            "after_screenshot": self.after_screenshot,
            "summary": self.summary,
        }


class ExecutorWithRecorder:
    def __init__(
        self,
        *,
        run_id: str | None = None,
        task: Mapping[str, Any] | Any | None = None,
        artifact_root: str | Path | None = None,
        executor: BaseExecutor | None = None,
        recorder: RunRecorder | None = None,
        screenshot_capturer: StepScreenshotCapturer | None = None,
        backend: DesktopActionBackend | None = None,
        executor_config: ExecutorConfig | None = None,
        monitor_index: int = 1,
        window_title: str | None = None,
        focus_window: bool = True,
        focus_delay_seconds: float = 0.5,
    ) -> None:
        self.recorder = recorder or RunRecorder(
            run_id=run_id,
            task=task,
            artifact_root=artifact_root,
        )
        self.executor = executor or BaseExecutor(backend=backend, config=executor_config)
        self.screenshot_capturer = screenshot_capturer or _default_screenshot_capturer(
            artifact_root=self.recorder.paths.run_root.parent,
            monitor_index=monitor_index,
            window_title=window_title,
            focus_window=focus_window,
            focus_delay_seconds=focus_delay_seconds,
        )

    @property
    def run_id(self) -> str:
        return self.recorder.run_id

    def run_step(
        self,
        action_decision: Mapping[str, Any],
        *,
        step_idx: int,
    ) -> StepExecutionResult:
        before_screenshot: Any | None = None
        after_screenshot: Any | None = None
        action_result: ActionResult

        try:
            before_screenshot = self.screenshot_capturer.capture_before(
                run_id=self.run_id,
                step_idx=step_idx,
            )
        except Exception as exc:
            action_result = _failure_result(action_decision, f"before_screenshot_failed: {exc}")
            summary = self.recorder.record_step(
                step_idx=step_idx,
                action_decision=action_decision,
                action_result=action_result,
            )
            return StepExecutionResult(
                run_id=self.run_id,
                step_idx=step_idx,
                action_result=action_result,
                before_screenshot=None,
                after_screenshot=None,
                summary=summary,
            )

        action_result = self.executor.execute_action(action_decision)

        try:
            after_screenshot = self.screenshot_capturer.capture_after(
                run_id=self.run_id,
                step_idx=step_idx,
            )
        except Exception as exc:
            action_result = _failure_result(action_decision, f"after_screenshot_failed: {exc}")

        summary = self.recorder.record_step(
            step_idx=step_idx,
            action_decision=action_decision,
            action_result=action_result,
            before_screenshot=before_screenshot,
            after_screenshot=after_screenshot,
        )
        return StepExecutionResult(
            run_id=self.run_id,
            step_idx=step_idx,
            action_result=action_result,
            before_screenshot=_optional_jsonable(before_screenshot),
            after_screenshot=_optional_jsonable(after_screenshot),
            summary=summary,
        )

    def finalize(
        self,
        *,
        status: str | None = None,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        return self.recorder.finalize(status=status, failure_reason=failure_reason)


def run_step(
    action_decision: Mapping[str, Any],
    *,
    step_idx: int,
    run_id: str | None = None,
    task: Mapping[str, Any] | Any | None = None,
    artifact_root: str | Path | None = None,
    backend: DesktopActionBackend | None = None,
    screenshot_capturer: StepScreenshotCapturer | None = None,
) -> dict[str, Any]:
    runner = ExecutorWithRecorder(
        run_id=run_id,
        task=task,
        artifact_root=artifact_root,
        backend=backend,
        screenshot_capturer=screenshot_capturer,
    )
    return runner.run_step(action_decision, step_idx=step_idx).to_dict()


def _default_screenshot_capturer(
    *,
    artifact_root: Path,
    monitor_index: int,
    window_title: str | None,
    focus_window: bool,
    focus_delay_seconds: float,
) -> StepScreenshotCapturer:
    from capture import ScreenshotCaptureConfig, ScreenshotCapturer

    return ScreenshotCapturer(
        config=ScreenshotCaptureConfig(
            artifact_root=artifact_root,
            monitor_index=monitor_index,
            window_title=window_title,
            focus_window=focus_window,
            focus_delay_seconds=focus_delay_seconds,
        )
    )


def _failure_result(action_decision: Mapping[str, Any], error: str) -> ActionResult:
    start_ts = utc_now()
    return action_result_from_exception(
        action_type=_action_type(action_decision),
        start_ts=start_ts,
        end_ts=utc_now(),
        error=error,
    )


def _action_type(action_decision: Mapping[str, Any]) -> str:
    action_type = str(action_decision.get("action_type") or "").strip()
    return action_type or "unknown"


def _optional_jsonable(value: Mapping[str, Any] | Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, Mapping):
            return dict(data)
    return {"repr": repr(value)}
