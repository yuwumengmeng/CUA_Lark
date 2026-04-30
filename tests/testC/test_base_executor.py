"""成员 C 基础执行器测试。

测试目标：
本文件验证 `executor/base_executor.py` 是否能解释成员 B 的 action_decision
协议，并把 click / double_click / scroll / type / hotkey / wait 六类动作
下发给桌面 backend，同时把异常统一转换为 `ActionResult`。

重要约定：
这里使用 fake backend，不访问真实桌面、不移动鼠标、不输入文本。真实飞书
执行验证应放到手动验收脚本或 recorder runner 中。

运行方式：
```powershell
python -m pytest tests\testC\test_base_executor.py
python tests\testC\test_base_executor.py
```
"""

import unittest
from unittest.mock import patch

import executor.base_executor as base_executor
from executor import (
    BaseExecutor,
    ExecutorConfig,
    PyAutoGUIDesktopBackend,
    execute_action,
    resolve_action_point,
)
from planner.action_protocol import ActionDecisionV2


class FakeDesktopBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def click(self, x: int, y: int) -> None:
        self.calls.append(("click", (x, y)))

    def double_click(self, x: int, y: int) -> None:
        self.calls.append(("double_click", (x, y)))

    def scroll(self, amount: int, x: int | None = None, y: int | None = None) -> None:
        self.calls.append(("scroll", (amount, x, y)))

    def type_text(self, text: str, *, interval: float = 0.0) -> None:
        self.calls.append(("type_text", (text, interval)))

    def hotkey(self, keys: list[str]) -> None:
        self.calls.append(("hotkey", tuple(keys)))

    def wait(self, seconds: float) -> None:
        self.calls.append(("wait", seconds))


class FailingBackend(FakeDesktopBackend):
    def click(self, x: int, y: int) -> None:
        raise RuntimeError("desktop refused click")


class BaseExecutorTests(unittest.TestCase):
    def test_click_uses_absolute_point_with_stability_wait(self):
        backend = FakeDesktopBackend()
        executor = BaseExecutor(
            backend=backend,
            config=ExecutorConfig(click_stability_delay_seconds=0.05),
        )

        result = executor.execute_action({"action_type": "click", "x": 120, "y": 240})

        self.assertTrue(result.ok)
        self.assertEqual(result.action_type, "click")
        self.assertEqual(result.planned_click_point, [120, 240])
        self.assertEqual(result.actual_click_point, [120, 240])
        self.assertEqual(result.click_offset, [0, 0])
        self.assertEqual(backend.calls, [("wait", 0.05), ("click", (120, 240))])

    def test_full_action_decision_v2_click_is_executed_with_diagnostics(self):
        backend = FakeDesktopBackend()
        executor = BaseExecutor(backend=backend)
        decision = ActionDecisionV2.click_candidate(
            step_name="open_search",
            target_candidate_id="elem_search",
            x=120,
            y=240,
            reason="open search box",
            retry_count=1,
            expected_after_state={"search_active": True},
        )

        result = executor.execute_action(decision.to_dict())

        self.assertTrue(result.ok)
        self.assertEqual(result.action_type, "click")
        self.assertEqual(result.target_candidate_id, "elem_search")
        self.assertEqual(result.planned_click_point, [120, 240])
        self.assertEqual(result.actual_click_point, [120, 240])
        self.assertEqual(result.click_offset, [0, 0])
        self.assertEqual(result.executor_message, "executed click")
        self.assertEqual(backend.calls, [("wait", 0.08), ("click", (120, 240))])

    def test_double_click_resolves_target_candidate_center(self):
        backend = FakeDesktopBackend()
        executor = BaseExecutor(backend=backend)

        result = executor.execute_action(
            {
                "action_type": "double_click",
                "target_candidate": {
                    "id": "elem_0001",
                    "text": "测试群",
                    "bbox": [10, 20, 50, 60],
                    "center": [30, 40],
                    "role": "ListItemControl",
                    "clickable": True,
                    "editable": False,
                    "source": "uia",
                    "confidence": 0.9,
                },
            }
        )

        self.assertTrue(result.ok)
        self.assertEqual(backend.calls, [("wait", 0.08), ("double_click", (30, 40))])

    def test_scroll_defaults_direction_down(self):
        backend = FakeDesktopBackend()
        executor = BaseExecutor(
            backend=backend,
            config=ExecutorConfig(default_scroll_amount=7),
        )

        result = executor.execute_action({"action_type": "scroll", "direction": "down"})

        self.assertTrue(result.ok)
        self.assertEqual(backend.calls, [("scroll", (-7, None, None))])

    def test_type_focuses_target_before_input(self):
        backend = FakeDesktopBackend()
        executor = BaseExecutor(
            backend=backend,
            config=ExecutorConfig(type_focus_delay_seconds=0.03, type_interval_seconds=0.01),
        )

        result = executor.execute_action(
            {
                "action_type": "type",
                "text": "Hello World",
                "target_candidate_id": "elem_input",
                "center": [10, 20],
            }
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.target_candidate_id, "elem_input")
        self.assertEqual(result.planned_click_point, [10, 20])
        self.assertEqual(result.actual_click_point, [10, 20])
        self.assertEqual(
            backend.calls,
            [
                ("click", (10, 20)),
                ("wait", 0.03),
                ("type_text", ("Hello World", 0.01)),
            ],
        )

    def test_hotkey_and_wait_are_dispatched(self):
        backend = FakeDesktopBackend()
        executor = BaseExecutor(backend=backend)

        hotkey_result = executor.execute_action(
            {"action_type": "hotkey", "keys": ["alt", "left"]}
        )
        wait_result = executor.execute_action({"action_type": "wait", "duration_ms": 250})

        self.assertTrue(hotkey_result.ok)
        self.assertTrue(wait_result.ok)
        self.assertEqual(backend.calls, [("hotkey", ("alt", "left")), ("wait", 0.25)])

    def test_action_decision_v2_press_enter_uses_hotkey_enter(self):
        backend = FakeDesktopBackend()
        executor = BaseExecutor(backend=backend)

        result = executor.execute_action(
            ActionDecisionV2.press_enter(step_name="send_message").to_dict()
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.action_type, "hotkey")
        self.assertEqual(backend.calls, [("hotkey", ("enter",))])

    def test_hotkey_can_focus_point_before_dispatch(self):
        backend = FakeDesktopBackend()
        executor = BaseExecutor(
            backend=backend,
            config=ExecutorConfig(hotkey_focus_delay_seconds=0.04),
        )

        result = executor.execute_action(
            {"action_type": "hotkey", "keys": ["esc"], "x": 1000, "y": 120}
        )

        self.assertTrue(result.ok)
        self.assertEqual(
            backend.calls,
            [("click", (1000, 120)), ("wait", 0.04), ("hotkey", ("esc",))],
        )

    def test_hotkey_can_repeat_with_delay(self):
        backend = FakeDesktopBackend()
        executor = BaseExecutor(
            backend=backend,
            config=ExecutorConfig(hotkey_repeat_delay_seconds=0.03),
        )

        result = executor.execute_action(
            {"action_type": "hotkey", "keys": ["esc"], "repeat": 2}
        )

        self.assertTrue(result.ok)
        self.assertEqual(
            backend.calls,
            [("hotkey", ("esc",)), ("wait", 0.03), ("hotkey", ("esc",))],
        )

    def test_execute_action_module_entry_returns_dict(self):
        backend = FakeDesktopBackend()

        payload = execute_action(
            {"action_type": "click", "x": 1, "y": 2},
            backend=backend,
        )

        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["action_type"], "click")
        self.assertEqual(payload["error"], None)

    def test_invalid_or_failing_actions_return_failure_result(self):
        missing_point = BaseExecutor(backend=FakeDesktopBackend()).execute_action(
            {"action_type": "click", "target_element_id": "elem_0001"}
        )
        unsupported = BaseExecutor(backend=FakeDesktopBackend()).execute_action(
            {"action_type": "drag", "x": 1, "y": 2}
        )
        backend_failure = BaseExecutor(backend=FailingBackend()).execute_action(
            {"action_type": "click", "x": 1, "y": 2}
        )

        self.assertFalse(missing_point.ok)
        self.assertIn("requires x/y", missing_point.error)
        self.assertFalse(unsupported.ok)
        self.assertEqual(unsupported.action_type, "drag")
        self.assertIn("unsupported action_type", unsupported.error)
        self.assertFalse(backend_failure.ok)
        self.assertEqual(backend_failure.action_type, "click")
        self.assertEqual(backend_failure.error, "desktop refused click")
        self.assertEqual(backend_failure.error_type, "RuntimeError")
        self.assertEqual(backend_failure.planned_click_point, [1, 2])
        self.assertIsNone(backend_failure.actual_click_point)

    def test_resolve_action_point_accepts_direct_center(self):
        self.assertEqual(resolve_action_point({"center": [3, 4]}), (3, 4))

    def test_clipboard_paste_uses_platform_shortcut(self):
        fake_pyautogui = FakePyAutoGUI()
        fake_pyperclip = FakePyperclip()
        backend = PyAutoGUIDesktopBackend()

        with patch.object(backend, "_pyautogui", return_value=fake_pyautogui):
            with patch.dict("sys.modules", {"pyperclip": fake_pyperclip}):
                with patch.object(base_executor.platform, "system", return_value="Darwin"):
                    backend.type_text("CUA-Lark")

        self.assertEqual(fake_pyperclip.copied_texts, ["CUA-Lark", "previous"])
        self.assertEqual(fake_pyautogui.hotkeys, [("command", "v")])


class FakePyAutoGUI:
    def __init__(self) -> None:
        self.hotkeys: list[tuple[str, ...]] = []

    def hotkey(self, *keys: str) -> None:
        self.hotkeys.append(tuple(keys))


class FakePyperclip:
    def __init__(self) -> None:
        self.copied_texts: list[str] = []

    def paste(self) -> str:
        return "previous"

    def copy(self, text: str) -> None:
        self.copied_texts.append(text)


if __name__ == "__main__":
    unittest.main()
