"""成员 C run artifact recorder 测试。

测试目标：
本文件验证 `executor/recorder.py` 是否能为一次 run 生成统一目录结构，
落盘 `task.json / actions.jsonl / results.jsonl / screenshot_log.jsonl /
summary.json`，并统计每类动作执行次数、失败次数和平均耗时。

重要约定：
这里只写入临时目录，不访问真实桌面、不截图、不执行鼠标键盘动作。

运行方式：
```powershell
python -m pytest tests\testC\test_recorder.py
python tests\testC\test_recorder.py
```
"""

import json
import tempfile
import unittest
from pathlib import Path

from executor import RunRecorder, RunRecorderError
from schemas import ActionResult


class FakeScreenshot:
    def __init__(self, path: str, phase: str) -> None:
        self.path = path
        self.phase = phase

    def to_dict(self) -> dict[str, object]:
        return {
            "screenshot_path": self.path,
            "phase": self.phase,
            "bbox": [0, 0, 100, 100],
            "screen_origin": [0, 0],
        }


class RunRecorderTests(unittest.TestCase):
    def test_recorder_creates_artifacts_and_summary_stats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = RunRecorder(
                run_id="run_test",
                task={"goal": "open_docs", "params": {}},
                artifact_root=temp_dir,
            )

            self.assertTrue(recorder.paths.run_root.exists())
            self.assertTrue(recorder.paths.screenshots_dir.exists())
            self.assertTrue(recorder.paths.task_path.exists())

            recorder.record_step(
                step_idx=1,
                action_decision={"action_type": "click", "x": 10, "y": 20},
                action_result=ActionResult.success(
                    action_type="click",
                    start_ts="2026-04-24T10:00:00.000Z",
                    end_ts="2026-04-24T10:00:00.100Z",
                ),
                before_screenshot=FakeScreenshot("before.png", "before"),
                after_screenshot=FakeScreenshot("after.png", "after"),
            )
            summary = recorder.record_step(
                step_idx=2,
                action_decision={"action_type": "click", "x": 11, "y": 21},
                action_result=ActionResult.failure(
                    action_type="click",
                    start_ts="2026-04-24T10:00:01.000Z",
                    end_ts="2026-04-24T10:00:01.050Z",
                    error="target_not_found",
                ),
            )

            self.assertEqual(summary["step_count"], 2)
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(summary["failure_count"], 1)
            self.assertEqual(
                summary["action_stats"]["click"],
                {"count": 2, "failure_count": 1, "avg_duration_ms": 75},
            )

            action_lines = read_jsonl(recorder.paths.actions_path)
            result_lines = read_jsonl(recorder.paths.results_path)
            screenshot_lines = read_jsonl(recorder.paths.screenshot_log_path)
            saved_summary = json.loads(recorder.paths.summary_path.read_text(encoding="utf-8"))

            self.assertEqual(len(action_lines), 2)
            self.assertEqual(action_lines[0]["action"]["x"], 10)
            self.assertEqual(result_lines[1]["result"]["error"], "target_not_found")
            self.assertEqual(screenshot_lines[0]["before"]["phase"], "before")
            self.assertEqual(saved_summary["artifact_paths"]["summary_path"], str(recorder.paths.summary_path))

    def test_finalize_marks_success_or_failed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            success_recorder = RunRecorder(run_id="run_success", artifact_root=temp_dir)
            success_payload = success_recorder.finalize()

            failed_recorder = RunRecorder(run_id="run_failed", artifact_root=temp_dir)
            failed_recorder.record_step(
                step_idx=1,
                action_decision={"action_type": "hotkey", "keys": ["alt", "left"]},
                action_result={
                    "ok": False,
                    "action_type": "hotkey",
                    "start_ts": "2026-04-24T10:00:00.000Z",
                    "end_ts": "2026-04-24T10:00:00.010Z",
                    "error": "shortcut_failed",
                },
            )
            failed_payload = failed_recorder.finalize(failure_reason="executor_failed")

            self.assertEqual(success_payload["status"], "success")
            self.assertEqual(failed_payload["status"], "failed")
            self.assertEqual(failed_payload["failure_reason"], "executor_failed")

    def test_invalid_recorder_inputs_are_rejected(self):
        with self.assertRaises(RunRecorderError):
            RunRecorder(run_id="bad/run")

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = RunRecorder(run_id="run_test", artifact_root=temp_dir)
            with self.assertRaises(RunRecorderError):
                recorder.record_step(
                    step_idx=-1,
                    action_decision={"action_type": "wait"},
                    action_result=ActionResult.success(
                        action_type="wait",
                        start_ts="2026-04-24T10:00:00.000Z",
                        end_ts="2026-04-24T10:00:01.000Z",
                    ),
                )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
