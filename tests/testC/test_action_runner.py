"""成员 C ExecutorWithRecorder 串联测试。

测试目标：
本文件验证 `executor/action_runner.py` 是否能把 before 截图、动作执行、
after 截图和 recorder 落盘串成单步闭环。

重要约定：
这里使用 fake screenshot capturer 和 fake desktop backend，不访问真实桌面。

运行方式：
```powershell
python -m pytest tests\testC\test_action_runner.py
python tests\testC\test_action_runner.py
```
"""

import json
import tempfile
import unittest

from executor import ExecutorWithRecorder, RunRecorder, run_step, run_steps
from planner.action_protocol import ActionDecisionV2


class FakeScreenshot:
    def __init__(self, run_id: str, step_idx: int, phase: str) -> None:
        self.run_id = run_id
        self.step_idx = step_idx
        self.phase = phase

    def to_dict(self) -> dict[str, object]:
        return {
            "screenshot_path": f"artifacts/runs/{self.run_id}/screenshots/step_{self.step_idx:03d}_{self.phase}.png",
            "run_id": self.run_id,
            "step_idx": self.step_idx,
            "phase": self.phase,
            "bbox": [0, 0, 100, 100],
            "screen_origin": [0, 0],
        }


class FakeScreenshotCapturer:
    def __init__(self, *, fail_before: bool = False, fail_after: bool = False) -> None:
        self.fail_before = fail_before
        self.fail_after = fail_after
        self.calls: list[tuple[str, str | None, int]] = []

    def capture_before(self, *, run_id: str | None = None, step_idx: int = 0) -> FakeScreenshot:
        self.calls.append(("before", run_id, step_idx))
        if self.fail_before:
            raise RuntimeError("before unavailable")
        return FakeScreenshot(run_id or "run_missing", step_idx, "before")

    def capture_after(self, *, run_id: str | None = None, step_idx: int = 0) -> FakeScreenshot:
        self.calls.append(("after", run_id, step_idx))
        if self.fail_after:
            raise RuntimeError("after unavailable")
        return FakeScreenshot(run_id or "run_missing", step_idx, "after")


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


class ExecutorWithRecorderTests(unittest.TestCase):
    def test_run_step_records_screenshots_action_and_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FakeDesktopBackend()
            capturer = FakeScreenshotCapturer()
            runner = ExecutorWithRecorder(
                run_id="run_test",
                task={"goal": "click_point"},
                artifact_root=temp_dir,
                backend=backend,
                screenshot_capturer=capturer,
            )

            step_result = runner.run_step(
                {"action_type": "click", "step_name": "open_search", "x": 10, "y": 20},
                step_idx=1,
                after_ui_state={
                    "screenshot_path": "after.png",
                    "screen_meta": {"dpi_scale": 1.5},
                },
                validator_result={"passed": True, "check_type": "page_changed"},
            )
            final_summary = runner.finalize()

            self.assertTrue(step_result.action_result.ok)
            self.assertEqual(step_result.action_result.before_screenshot["phase"], "before")
            self.assertEqual(step_result.action_result.after_screenshot["phase"], "after")
            self.assertEqual(step_result.action_result.screen_meta_snapshot["dpi_scale"], 1.5)
            self.assertEqual(step_result.before_screenshot["phase"], "before")
            self.assertEqual(step_result.after_screenshot["phase"], "after")
            self.assertEqual(capturer.calls, [("before", "run_test", 1), ("after", "run_test", 1)])
            self.assertEqual(backend.calls, [("wait", 0.08), ("click", (10, 20))])
            self.assertEqual(final_summary["status"], "success")

            actions = read_jsonl(runner.recorder.paths.actions_path)
            results = read_jsonl(runner.recorder.paths.results_path)
            screenshots = read_jsonl(runner.recorder.paths.screenshot_log_path)
            validators = read_jsonl(runner.recorder.paths.validator_log_path)
            ui_states = read_jsonl(runner.recorder.paths.ui_state_log_path)
            self.assertEqual(actions[0]["action"]["action_type"], "click")
            self.assertEqual(actions[0]["step_name"], "open_search")
            self.assertEqual(results[0]["result"]["ok"], True)
            self.assertEqual(results[0]["result"]["before_screenshot"]["phase"], "before")
            self.assertEqual(screenshots[0]["after"]["phase"], "after")
            self.assertEqual(validators[0]["validator_result"]["passed"], True)
            self.assertEqual(ui_states[0]["after_ui_state"]["screenshot_path"], "after.png")

    def test_before_screenshot_failure_prevents_action(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FakeDesktopBackend()
            runner = ExecutorWithRecorder(
                run_id="run_test",
                artifact_root=temp_dir,
                backend=backend,
                screenshot_capturer=FakeScreenshotCapturer(fail_before=True),
            )

            step_result = runner.run_step({"action_type": "click", "x": 10, "y": 20}, step_idx=1)

            self.assertFalse(step_result.action_result.ok)
            self.assertIn("before_screenshot_failed", step_result.action_result.error)
            self.assertEqual(backend.calls, [])

    def test_after_screenshot_failure_marks_step_failed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FakeDesktopBackend()
            runner = ExecutorWithRecorder(
                run_id="run_test",
                artifact_root=temp_dir,
                backend=backend,
                screenshot_capturer=FakeScreenshotCapturer(fail_after=True),
            )

            step_result = runner.run_step({"action_type": "wait", "seconds": 0.01}, step_idx=1)

            self.assertFalse(step_result.action_result.ok)
            self.assertIn("after_screenshot_failed", step_result.action_result.error)
            self.assertEqual(backend.calls, [("wait", 0.01)])

    def test_module_run_step_returns_dict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = run_step(
                {"action_type": "hotkey", "keys": ["alt", "left"]},
                step_idx=3,
                run_id="run_test",
                artifact_root=temp_dir,
                backend=FakeDesktopBackend(),
                screenshot_capturer=FakeScreenshotCapturer(),
            )

            self.assertEqual(payload["run_id"], "run_test")
            self.assertEqual(payload["step_idx"], 3)
            self.assertEqual(payload["action_result"]["ok"], True)

    def test_runner_can_use_existing_recorder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = RunRecorder(run_id="run_existing", artifact_root=temp_dir)
            runner = ExecutorWithRecorder(
                recorder=recorder,
                backend=FakeDesktopBackend(),
                screenshot_capturer=FakeScreenshotCapturer(),
            )

            self.assertEqual(runner.run_id, "run_existing")

    def test_run_steps_executes_multi_step_action_decision_v2_sequence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FakeDesktopBackend()
            runner = ExecutorWithRecorder(
                run_id="run_v2_sequence",
                task={"goal": "execute B multi-step decisions"},
                artifact_root=temp_dir,
                backend=backend,
                screenshot_capturer=FakeScreenshotCapturer(),
            )
            decisions = [
                ActionDecisionV2.click_candidate(
                    step_name="open_chat",
                    target_candidate_id="elem_chat",
                    x=120,
                    y=240,
                    expected_after_state={"target_text": "测试群"},
                ).to_dict(),
                ActionDecisionV2.type_text(
                    step_name="type_message",
                    target_candidate_id="elem_input",
                    x=300,
                    y=500,
                    text="Hello CUA-Lark",
                    expected_after_state={"message_text": "Hello CUA-Lark"},
                ).to_dict(),
                ActionDecisionV2.press_enter(
                    step_name="send_message",
                    expected_after_state={"message_text": "Hello CUA-Lark"},
                ).to_dict(),
                ActionDecisionV2.scroll(
                    step_name="scroll_result_list",
                    direction="down",
                    amount=-3,
                ).to_dict(),
            ]

            step_results = runner.run_steps(decisions)
            summary = runner.finalize()

            self.assertEqual([result.step_idx for result in step_results], [1, 2, 3, 4])
            self.assertTrue(all(result.action_result.ok for result in step_results))
            self.assertEqual(summary["step_count"], 4)
            self.assertEqual(summary["success_count"], 4)
            self.assertEqual(summary["status"], "success")
            self.assertEqual(
                backend.calls,
                [
                    ("wait", 0.08),
                    ("click", (120, 240)),
                    ("click", (300, 500)),
                    ("wait", 0.08),
                    ("type_text", ("Hello CUA-Lark", 0.0)),
                    ("hotkey", ("enter",)),
                    ("scroll", (-3, None, None)),
                ],
            )

            actions = read_jsonl(runner.recorder.paths.actions_path)
            self.assertEqual([action["step_name"] for action in actions], [
                "open_chat",
                "type_message",
                "send_message",
                "scroll_result_list",
            ])
            self.assertEqual(
                actions[1]["action"]["expected_after_state"],
                {"message_text": "Hello CUA-Lark"},
            )

    def test_module_run_steps_returns_steps_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = run_steps(
                [
                    ActionDecisionV2.hotkey(
                        step_name="open_search",
                        keys=["ctrl", "k"],
                    ).to_dict(),
                    ActionDecisionV2.wait(
                        step_name="wait_search",
                        seconds=0.1,
                    ).to_dict(),
                ],
                run_id="run_module_steps",
                artifact_root=temp_dir,
                backend=FakeDesktopBackend(),
                screenshot_capturer=FakeScreenshotCapturer(),
            )

            self.assertEqual(payload["run_id"], "run_module_steps")
            self.assertEqual(len(payload["steps"]), 2)
            self.assertEqual(payload["summary"]["status"], "success")
            self.assertEqual(payload["summary"]["action_stats"]["hotkey"]["count"], 1)
            self.assertEqual(payload["summary"]["action_stats"]["wait"]["count"], 1)


def read_jsonl(path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
