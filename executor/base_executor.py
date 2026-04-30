"""成员 C 4.23 首版桌面执行器。

职责边界：
本文件只负责解释 planner 下发的单步动作协议，并调用桌面 backend 执行
click / double_click / scroll / type / hotkey / wait 六类基础动作。它不做
业务语义判断、不读取飞书页面状态、不生成 run artifact；这些由 planner、
perception 和 recorder 分别负责。

公开接口：
`BaseExecutor.execute_action(...)`
    执行一个 action_decision，返回强类型 `ActionResult`。
`execute_action(...)`
    给成员 B 调用的统一入口，返回可 JSON 序列化 dict。
`resolve_action_point(...)`
    将动作里的 `x/y`、`center` 或 candidate 转成屏幕绝对点击点。
`PyAutoGUIDesktopBackend`
    真实桌面 backend，默认使用 pyautogui 执行鼠标键盘动作。

动作协议：
`action_type`
    必填，首版支持 click / double_click / scroll / type / hotkey / wait。
`x / y`
    可选屏幕绝对坐标。click、double_click 优先使用它们。
`center` 或 `target_candidate`
    可选候选元素点击点。candidate 字段遵守 `schemas.UICandidate`。
`text`
    type 动作必填文本。
`keys`
    hotkey 动作必填按键列表，例如 `["alt", "left"]`。
`repeat`
    hotkey 动作可选重复次数，例如搜索框里 `Esc` 第一次清空文本、第二次关闭弹窗。
`seconds`
    wait 动作等待秒数；也兼容 `duration_seconds` 和 `duration_ms`。

稳定性规则：
1. click / double_click 前默认等待 80ms，给截图和窗口焦点一点稳定时间。
2. type 动作如果带坐标或 candidate，会先单击聚焦，再等待 80ms 后输入。
3. hotkey 动作如果带坐标或 candidate，会先单击聚焦，再发送快捷键；需要多次
   快捷键时使用 `repeat`。关闭弹窗、返回等动作不要裸发快捷键，避免焦点留在
   终端或其他窗口。
4. scroll 支持 `direction=up/down`，向下滚动会转换成 pyautogui 负 clicks。
5. 所有异常都会被转换为失败 `ActionResult`，不向 planner 泄漏 traceback。

真实环境依赖：
真实 backend 需要 `pyautogui`；文本输入默认走剪贴板粘贴，需要 `pyperclip`，
并会按平台选择 `ctrl+v` 或 macOS 的 `command+v`。
单元测试应注入 fake backend，不访问真实桌面。
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from time import sleep
from typing import Any, Mapping, Protocol, Sequence

from schemas import (
    ACTION_CLICK,
    ACTION_DOUBLE_CLICK,
    ACTION_HOTKEY,
    ACTION_SCROLL,
    ACTION_TYPE,
    ACTION_WAIT,
    SUPPORTED_EXECUTOR_ACTION_TYPES,
    ActionResult,
    action_result_from_exception,
    resolve_click_point,
    utc_now,
)


DEFAULT_SCROLL_AMOUNT = 5
UNKNOWN_ACTION_TYPE = "unknown"


class ActionDecisionError(ValueError):
    """Raised when planner action_decision data cannot be executed."""


class DesktopActionBackend(Protocol):
    def click(self, x: int, y: int) -> None:
        ...

    def double_click(self, x: int, y: int) -> None:
        ...

    def scroll(self, amount: int, x: int | None = None, y: int | None = None) -> None:
        ...

    def type_text(self, text: str, *, interval: float = 0.0) -> None:
        ...

    def hotkey(self, keys: Sequence[str]) -> None:
        ...

    def wait(self, seconds: float) -> None:
        ...


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    click_stability_delay_seconds: float = 0.08
    type_focus_delay_seconds: float = 0.08
    hotkey_focus_delay_seconds: float = 0.08
    hotkey_repeat_delay_seconds: float = 0.08
    type_interval_seconds: float = 0.0
    default_scroll_amount: int = DEFAULT_SCROLL_AMOUNT
    max_wait_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.click_stability_delay_seconds < 0:
            raise ActionDecisionError("click_stability_delay_seconds must be non-negative")
        if self.type_focus_delay_seconds < 0:
            raise ActionDecisionError("type_focus_delay_seconds must be non-negative")
        if self.hotkey_focus_delay_seconds < 0:
            raise ActionDecisionError("hotkey_focus_delay_seconds must be non-negative")
        if self.hotkey_repeat_delay_seconds < 0:
            raise ActionDecisionError("hotkey_repeat_delay_seconds must be non-negative")
        if self.type_interval_seconds < 0:
            raise ActionDecisionError("type_interval_seconds must be non-negative")
        if self.default_scroll_amount <= 0:
            raise ActionDecisionError("default_scroll_amount must be positive")
        if self.max_wait_seconds <= 0:
            raise ActionDecisionError("max_wait_seconds must be positive")


class PyAutoGUIDesktopBackend:
    def __init__(self, *, use_clipboard_for_text: bool = True) -> None:
        self.use_clipboard_for_text = use_clipboard_for_text

    def click(self, x: int, y: int) -> None:
        pyautogui = self._pyautogui()
        pyautogui.click(x=x, y=y)

    def double_click(self, x: int, y: int) -> None:
        pyautogui = self._pyautogui()
        pyautogui.doubleClick(x=x, y=y)

    def scroll(self, amount: int, x: int | None = None, y: int | None = None) -> None:
        pyautogui = self._pyautogui()
        if x is not None and y is not None:
            pyautogui.moveTo(x=x, y=y)
        pyautogui.scroll(amount)

    def type_text(self, text: str, *, interval: float = 0.0) -> None:
        if self.use_clipboard_for_text:
            self._paste_text(text)
            return
        pyautogui = self._pyautogui()
        pyautogui.write(text, interval=interval)

    def hotkey(self, keys: Sequence[str]) -> None:
        pyautogui = self._pyautogui()
        pyautogui.hotkey(*keys)

    def wait(self, seconds: float) -> None:
        sleep(seconds)

    def _paste_text(self, text: str) -> None:
        pyautogui = self._pyautogui()
        try:
            import pyperclip
        except ImportError as exc:
            raise ActionDecisionError("pyperclip is required for clipboard text input") from exc

        previous_text: str | None = None
        try:
            previous_text = pyperclip.paste()
        except Exception:
            previous_text = None

        pyperclip.copy(text)
        pyautogui.hotkey(_paste_modifier_key(), "v")
        if previous_text is not None:
            pyperclip.copy(previous_text)

    def _pyautogui(self) -> Any:
        try:
            import pyautogui
        except ImportError as exc:
            raise ActionDecisionError("pyautogui is not installed") from exc
        return pyautogui


class BaseExecutor:
    def __init__(
        self,
        *,
        backend: DesktopActionBackend | None = None,
        config: ExecutorConfig | None = None,
    ) -> None:
        self.backend = backend or PyAutoGUIDesktopBackend()
        self.config = config or ExecutorConfig()

    def execute_action(self, action_decision: Mapping[str, Any]) -> ActionResult:
        start_ts = utc_now()
        action_type = _action_type_for_result(action_decision)
        target_candidate_id = _target_candidate_id(action_decision)
        planned_click_point = _safe_planned_click_point(action_decision)

        try:
            normalized_action_type = _normalize_action_type(action_decision)
            actual_click_point = self._execute_normalized(normalized_action_type, action_decision)
            return ActionResult.success(
                action_type=normalized_action_type,
                start_ts=start_ts,
                end_ts=utc_now(),
                target_candidate_id=target_candidate_id,
                planned_click_point=planned_click_point,
                actual_click_point=actual_click_point,
                executor_message=f"executed {normalized_action_type}",
            )
        except Exception as exc:
            return action_result_from_exception(
                action_type=action_type,
                start_ts=start_ts,
                end_ts=utc_now(),
                error=exc,
                target_candidate_id=target_candidate_id,
                planned_click_point=planned_click_point,
                executor_message=f"failed {action_type}",
            )

    def _execute_normalized(
        self,
        action_type: str,
        action_decision: Mapping[str, Any],
    ) -> list[int] | None:
        if action_type == ACTION_CLICK:
            x, y = resolve_action_point(action_decision)
            self.backend.wait(self.config.click_stability_delay_seconds)
            self.backend.click(x, y)
            return [x, y]

        if action_type == ACTION_DOUBLE_CLICK:
            x, y = resolve_action_point(action_decision)
            self.backend.wait(self.config.click_stability_delay_seconds)
            self.backend.double_click(x, y)
            return [x, y]

        if action_type == ACTION_SCROLL:
            x, y = resolve_optional_action_point(action_decision)
            self.backend.scroll(_scroll_amount(action_decision, self.config), x=x, y=y)
            return None

        if action_type == ACTION_TYPE:
            text = _required_text(action_decision)
            point = resolve_optional_action_point(action_decision)
            actual_click_point: list[int] | None = None
            if point != (None, None):
                x, y = point
                if x is not None and y is not None:
                    self.backend.click(x, y)
                    actual_click_point = [x, y]
                    self.backend.wait(self.config.type_focus_delay_seconds)
            self.backend.type_text(text, interval=self.config.type_interval_seconds)
            return actual_click_point

        if action_type == ACTION_HOTKEY:
            keys = _required_keys(action_decision)
            repeat = _hotkey_repeat_count(action_decision)
            point = resolve_optional_action_point(action_decision)
            actual_click_point = None
            if point != (None, None):
                x, y = point
                if x is not None and y is not None:
                    self.backend.click(x, y)
                    actual_click_point = [x, y]
                    self.backend.wait(self.config.hotkey_focus_delay_seconds)
            for index in range(repeat):
                if index > 0:
                    self.backend.wait(self.config.hotkey_repeat_delay_seconds)
                self.backend.hotkey(keys)
            return actual_click_point

        if action_type == ACTION_WAIT:
            self.backend.wait(_wait_seconds(action_decision, self.config))
            return None

        raise ActionDecisionError(f"unsupported action_type: {action_type}")


def execute_action(
    action_decision: Mapping[str, Any],
    *,
    backend: DesktopActionBackend | None = None,
    config: ExecutorConfig | None = None,
) -> dict[str, Any]:
    return BaseExecutor(backend=backend, config=config).execute_action(action_decision).to_dict()


def resolve_action_point(action_decision: Mapping[str, Any]) -> tuple[int, int]:
    point = resolve_optional_action_point(action_decision)
    if point == (None, None):
        raise ActionDecisionError("action requires x/y, center, or target_candidate")
    x, y = point
    if x is None or y is None:
        raise ActionDecisionError("action point requires both x and y")
    return x, y


def resolve_optional_action_point(action_decision: Mapping[str, Any]) -> tuple[int | None, int | None]:
    x = action_decision.get("x")
    y = action_decision.get("y")
    if x is not None or y is not None:
        return _int_field(x, "x"), _int_field(y, "y")

    center = action_decision.get("center")
    if center is not None:
        return _point_from_sequence(center, "center")

    for field_name in ("target_candidate", "candidate", "target_element", "element"):
        candidate = action_decision.get(field_name)
        if isinstance(candidate, Mapping):
            return resolve_click_point(candidate)

    return None, None


def _target_candidate_id(action_decision: Mapping[str, Any]) -> str | None:
    for field_name in ("target_candidate_id", "target_element_id"):
        value = action_decision.get(field_name)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    for field_name in ("target_candidate", "candidate", "target_element", "element"):
        candidate = action_decision.get(field_name)
        if isinstance(candidate, Mapping):
            candidate_id = candidate.get("id")
            if candidate_id is not None:
                text = str(candidate_id).strip()
                if text:
                    return text
    return None


def _safe_planned_click_point(action_decision: Mapping[str, Any]) -> list[int] | None:
    try:
        x, y = resolve_optional_action_point(action_decision)
    except Exception:
        return None
    if x is None or y is None:
        return None
    return [x, y]


def _normalize_action_type(action_decision: Mapping[str, Any]) -> str:
    action_type = _action_type_for_result(action_decision)
    if action_type not in SUPPORTED_EXECUTOR_ACTION_TYPES:
        raise ActionDecisionError(f"unsupported action_type: {action_type}")
    return action_type


def _action_type_for_result(action_decision: Mapping[str, Any]) -> str:
    raw_action_type = action_decision.get("action_type")
    if raw_action_type is None:
        return UNKNOWN_ACTION_TYPE
    action_type = str(raw_action_type).strip()
    return action_type or UNKNOWN_ACTION_TYPE


def _required_text(action_decision: Mapping[str, Any]) -> str:
    text = str(action_decision.get("text") or "")
    if not text:
        raise ActionDecisionError("type action requires non-empty text")
    return text


def _required_keys(action_decision: Mapping[str, Any]) -> list[str]:
    raw_keys = action_decision.get("keys")
    if raw_keys is None:
        raw_key = action_decision.get("key")
        raw_keys = [raw_key] if raw_key is not None else None
    if isinstance(raw_keys, (str, bytes)) or not isinstance(raw_keys, Sequence):
        raise ActionDecisionError("hotkey action requires keys list")

    keys = [str(key).strip() for key in raw_keys if str(key).strip()]
    if not keys:
        raise ActionDecisionError("hotkey action requires at least one key")
    return keys


def _paste_modifier_key() -> str:
    return "command" if platform.system() == "Darwin" else "ctrl"


def _hotkey_repeat_count(action_decision: Mapping[str, Any]) -> int:
    raw_repeat = action_decision.get("repeat")
    if raw_repeat is None:
        raw_repeat = action_decision.get("presses")
    if raw_repeat is None:
        return 1
    repeat = _int_field(raw_repeat, "repeat")
    if repeat <= 0:
        raise ActionDecisionError("hotkey repeat must be positive")
    return repeat


def _scroll_amount(action_decision: Mapping[str, Any], config: ExecutorConfig) -> int:
    for field_name in ("amount", "scroll_amount", "clicks"):
        value = action_decision.get(field_name)
        if value is not None:
            amount = _int_field(value, field_name)
            if amount == 0:
                raise ActionDecisionError("scroll amount cannot be zero")
            return amount

    direction = str(action_decision.get("direction") or "").strip().lower()
    if direction in {"down", "向下"}:
        return -config.default_scroll_amount
    if direction in {"up", "向上"}:
        return config.default_scroll_amount
    return -config.default_scroll_amount


def _wait_seconds(action_decision: Mapping[str, Any], config: ExecutorConfig) -> float:
    if action_decision.get("duration_ms") is not None:
        seconds = _float_field(action_decision["duration_ms"], "duration_ms") / 1000
    else:
        raw_seconds = (
            action_decision.get("seconds")
            if action_decision.get("seconds") is not None
            else action_decision.get("duration_seconds")
        )
        seconds = _float_field(raw_seconds if raw_seconds is not None else 1.0, "seconds")
    if seconds < 0:
        raise ActionDecisionError("wait seconds must be non-negative")
    if seconds > config.max_wait_seconds:
        raise ActionDecisionError("wait seconds exceeds max_wait_seconds")
    return seconds


def _point_from_sequence(value: Any, field_name: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ActionDecisionError(f"{field_name} must be a two-item coordinate list")
    return _int_field(value[0], f"{field_name}[0]"), _int_field(value[1], f"{field_name}[1]")


def _int_field(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ActionDecisionError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ActionDecisionError(f"{field_name} must be an integer") from exc


def _float_field(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ActionDecisionError(f"{field_name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ActionDecisionError(f"{field_name} must be a number") from exc
