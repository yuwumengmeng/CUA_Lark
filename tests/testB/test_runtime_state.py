"""运行时状态测试脚本。

这个脚本验证成员 B 4.23 第二项的 `runtime_state.py`：能够创建初始
runtime state、接收 A 的 ui_state 字段、记录动作历史，并拒绝非法状态。
"""

import unittest

from planner.runtime_state import (
    RuntimeState,
    RuntimeStateError,
    RuntimeStatus,
    create_runtime_state,
)


class RuntimeStateTests(unittest.TestCase):
    def test_create_runtime_state_from_instruction(self):
        state = create_runtime_state("打开聊天列表里的测试群")

        self.assertEqual(state.step_idx, 0)
        self.assertEqual(state.status, RuntimeStatus.RUNNING.value)
        self.assertEqual(state.failure_reason, None)
        self.assertEqual(state.task.params["target_name"], "测试群")
        self.assertEqual(state.to_dict()["history"], [])

    def test_update_ui_state_keeps_planner_fields(self):
        state = create_runtime_state({"goal": "open_docs", "params": {}})
        state.update_ui_state(
            {
                "screenshot_path": "artifacts/runs/run_001/screenshots/step_001.png",
                "page_summary": {"has_search_box": True, "candidate_count": 1},
                "candidates": [{"id": "elem_001", "text": "文档"}],
            }
        )

        self.assertEqual(state.page_summary["candidate_count"], 1)
        self.assertEqual(state.candidates[0]["id"], "elem_001")

    def test_record_step_updates_history_and_last_action(self):
        state = create_runtime_state({"goal": "open_docs", "params": {}})
        action = {"action_type": "click", "target_element_id": "elem_001"}
        result = {"ok": True, "action_type": "click"}

        state.record_step(action, result, advance_step=True)

        self.assertEqual(state.step_idx, 1)
        self.assertEqual(state.last_action, action)
        self.assertEqual(len(state.history), 1)
        self.assertEqual(state.history[0].action_result, result)

    def test_from_dict_roundtrip(self):
        state = create_runtime_state("点击文档入口")
        state.record_step(
            {"action_type": "click", "target_element_id": "elem_001"},
            {"ok": False},
            status=RuntimeStatus.FAILED.value,
            failure_reason="executor_failed",
        )

        restored = RuntimeState.from_dict(state.to_dict())

        self.assertEqual(restored.status, RuntimeStatus.FAILED.value)
        self.assertEqual(restored.failure_reason, "executor_failed")
        self.assertEqual(restored.history[0].failure_reason, "executor_failed")

    def test_invalid_status_is_rejected(self):
        with self.assertRaises(RuntimeStateError):
            create_runtime_state("点击文档入口").record_step(
                {"action_type": "click"},
                status="unknown",
            )


if __name__ == "__main__":
    unittest.main()
