"""成员 B 第二周期 ActionDecision v2 协议测试。

测试目标：
验证 `planner/action_protocol.py` 新增的 action_decision.v2 是否能稳定表达
click candidate、click absolute point、type text、hotkey、wait、scroll 和 press enter。

依赖约定：
本文件只检查协议对象和 dict 字段，不访问真实桌面，不调用真实 executor。
press enter 在本周期先表达为 `action_type=hotkey, keys=["enter"]`。

运行方式：
python -m unittest tests\testB\test_action_protocol_v2.py
python -m unittest discover -s tests\testB
"""

import unittest

from planner.action_protocol import (
    ACTION_CLICK,
    ACTION_DECISION_V2_VERSION,
    ACTION_HOTKEY,
    ACTION_SCROLL,
    ACTION_TYPE,
    ACTION_WAIT,
    DECISION_FAILED,
    ActionDecisionV2,
    ActionDecisionV2SchemaError,
    action_decision_v2_to_executor_payload,
)


class ActionDecisionV2Tests(unittest.TestCase):
    def test_click_candidate_outputs_target_candidate_and_point(self):
        decision = ActionDecisionV2.click_candidate(
            step_name="open_chat",
            target_candidate_id="elem_chat",
            x=120,
            y=240,
            reason="vlm and rule matched target chat",
            retry_count=1,
            expected_after_state={"chat_name_visible": "测试群"},
        )

        payload = decision.to_dict()

        self.assertEqual(payload["version"], ACTION_DECISION_V2_VERSION)
        self.assertEqual(payload["action_type"], ACTION_CLICK)
        self.assertEqual(payload["step_name"], "open_chat")
        self.assertEqual(payload["target_candidate_id"], "elem_chat")
        self.assertEqual(payload["x"], 120)
        self.assertEqual(payload["y"], 240)
        self.assertEqual(payload["retry_count"], 1)
        self.assertEqual(payload["expected_after_state"], {"chat_name_visible": "测试群"})

    def test_click_absolute_point_does_not_require_candidate_id(self):
        decision = ActionDecisionV2.click_point(
            step_name="click_safe_point",
            x=10,
            y=20,
            reason="absolute point fallback",
        )

        self.assertIsNone(decision.to_dict()["target_candidate_id"])
        self.assertEqual(action_decision_v2_to_executor_payload(decision.to_dict())["x"], 10)

    def test_type_hotkey_wait_scroll_and_press_enter_are_executable(self):
        type_decision = ActionDecisionV2.type_text(
            step_name="send_message",
            text="Hello CUA-Lark",
            target_candidate_id="elem_input",
            x=300,
            y=500,
        )
        hotkey_decision = ActionDecisionV2.hotkey(step_name="open_search", keys=["ctrl", "k"])
        wait_decision = ActionDecisionV2.wait(step_name="wait_loading", seconds=0.5)
        scroll_decision = ActionDecisionV2.scroll(
            step_name="scroll_results",
            direction="down",
            amount=-5,
        )
        enter_decision = ActionDecisionV2.press_enter(step_name="send_message")

        self.assertEqual(type_decision.to_dict()["action_type"], ACTION_TYPE)
        self.assertEqual(type_decision.to_dict()["text"], "Hello CUA-Lark")
        self.assertEqual(hotkey_decision.to_dict()["action_type"], ACTION_HOTKEY)
        self.assertEqual(hotkey_decision.to_dict()["keys"], ["ctrl", "k"])
        self.assertEqual(wait_decision.to_dict()["action_type"], ACTION_WAIT)
        self.assertEqual(wait_decision.to_dict()["seconds"], 0.5)
        self.assertEqual(scroll_decision.to_dict()["action_type"], ACTION_SCROLL)
        self.assertEqual(scroll_decision.to_executor_dict()["amount"], -5)
        self.assertEqual(enter_decision.to_dict()["action_type"], ACTION_HOTKEY)
        self.assertEqual(enter_decision.to_dict()["keys"], ["enter"])

    def test_dict_roundtrip_and_executor_payload_are_stable(self):
        raw = ActionDecisionV2.click_candidate(
            step_name="open_chat",
            target_candidate_id="elem_chat",
            x=120,
            y=240,
            reason="matched",
            expected_after_state={"chat_name_visible": "测试群"},
        ).to_dict()

        restored = ActionDecisionV2.from_dict(raw)
        executor_payload = restored.to_executor_dict()

        self.assertEqual(restored.to_dict(), raw)
        self.assertNotIn("step_name", executor_payload)
        self.assertNotIn("expected_after_state", executor_payload)
        self.assertEqual(executor_payload["target_element_id"], "elem_chat")
        self.assertEqual(executor_payload["action_type"], ACTION_CLICK)

    def test_failed_decision_requires_failure_reason(self):
        decision = ActionDecisionV2.failed(
            "candidate_conflict",
            "vlm target conflicts with rule target",
            step_name="open_chat",
            retry_count=2,
        )

        self.assertEqual(decision.to_dict()["status"], DECISION_FAILED)
        self.assertEqual(decision.to_dict()["failure_reason"], "candidate_conflict")
        self.assertEqual(decision.to_dict()["retry_count"], 2)

        with self.assertRaises(ActionDecisionV2SchemaError):
            ActionDecisionV2(status=DECISION_FAILED, step_name="open_chat")

    def test_invalid_v2_fields_are_rejected(self):
        with self.assertRaises(ActionDecisionV2SchemaError):
            ActionDecisionV2.click_candidate(
                step_name="open_chat",
                target_candidate_id="elem_chat",
                x=120,
                y=True,
            )
        with self.assertRaises(ActionDecisionV2SchemaError):
            ActionDecisionV2.hotkey(step_name="open_search", keys=[])
        with self.assertRaises(ActionDecisionV2SchemaError):
            ActionDecisionV2.wait(step_name="wait_loading", seconds=-1)


if __name__ == "__main__":
    unittest.main()
