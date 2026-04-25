"""成员 B planner 到 executor 动作协议测试。

测试目标：
本文件验证 `planner/action_protocol.py` 的 action decision 协议是否稳定，包括
click / type / hotkey / wait 四类基础动作、失败决策、日志字段和 executor payload。

重要约定：
这里不访问真实桌面、不执行鼠标键盘动作，只验证协议对象和 dict 字段。

运行方式：
```powershell
python -m unittest tests\testB\test_action_protocol.py
python -m unittest discover -s tests\testB
```

修改规则：
1. 新增 action decision 字段时，必须补充稳定字段断言。
2. 新增动作类型时，必须确认 executor 是否支持该 action_type。
3. 不允许把真实 GUI 依赖放进这个协议测试。
"""

import unittest

from planner.action_protocol import (
    ACTION_CLICK,
    ACTION_DECISION_VERSION,
    ACTION_HOTKEY,
    ACTION_TYPE,
    ACTION_WAIT,
    DECISION_CONTINUE,
    DECISION_FAILED,
    FAILURE_TARGET_NOT_FOUND,
    ActionDecision,
    ActionDecisionSchemaError,
    action_decision_to_executor_payload,
)


class ActionDecisionTests(unittest.TestCase):
    def test_click_decision_outputs_stable_fields(self):
        decision = ActionDecision.continue_(
            action_type=ACTION_CLICK,
            target_element_id="elem_001",
            x=412,
            y=238,
            reason="matched target_text=测试群",
            planner_meta={"advance_steps": 2},
        )

        self.assertEqual(
            decision.to_dict(),
            {
                "version": ACTION_DECISION_VERSION,
                "status": DECISION_CONTINUE,
                "action_type": ACTION_CLICK,
                "target_element_id": "elem_001",
                "x": 412,
                "y": 238,
                "text": None,
                "keys": None,
                "seconds": None,
                "reason": "matched target_text=测试群",
                "failure_reason": None,
                "planner_meta": {"advance_steps": 2},
            },
        )

        executor_payload = decision.to_executor_dict()
        self.assertNotIn("planner_meta", executor_payload)
        self.assertEqual(executor_payload["action_type"], ACTION_CLICK)
        self.assertEqual(executor_payload["target_element_id"], "elem_001")

    def test_type_decision_outputs_text(self):
        decision = ActionDecision.continue_(
            action_type=ACTION_TYPE,
            text="项目周报",
            reason="type text for step_idx=1",
        )

        self.assertEqual(decision.to_dict()["text"], "项目周报")
        self.assertEqual(decision.to_dict()["action_type"], ACTION_TYPE)

    def test_hotkey_decision_outputs_keys(self):
        decision = ActionDecision.continue_(
            action_type=ACTION_HOTKEY,
            keys=["alt", "left"],
            reason="hotkey back",
        )

        self.assertEqual(decision.to_dict()["keys"], ["alt", "left"])

    def test_wait_decision_outputs_seconds(self):
        decision = ActionDecision.continue_(
            action_type=ACTION_WAIT,
            seconds=2,
            reason="wait 2 seconds",
        )

        self.assertEqual(decision.to_dict()["seconds"], 2)

    def test_failed_decision_requires_failure_reason(self):
        decision = ActionDecision.failed(
            FAILURE_TARGET_NOT_FOUND,
            "target candidate not found: 测试群",
        )

        self.assertEqual(decision.to_dict()["status"], DECISION_FAILED)
        self.assertEqual(decision.to_dict()["failure_reason"], FAILURE_TARGET_NOT_FOUND)

        with self.assertRaises(ActionDecisionSchemaError):
            ActionDecision(status=DECISION_FAILED)

    def test_dict_roundtrip_and_executor_payload_are_stable(self):
        raw = {
            "version": ACTION_DECISION_VERSION,
            "status": DECISION_CONTINUE,
            "action_type": ACTION_CLICK,
            "target_element_id": "elem_docs",
            "x": 120,
            "y": 40,
            "text": None,
            "keys": None,
            "seconds": None,
            "reason": "matched target_text=文档",
            "failure_reason": None,
            "planner_meta": {"advance_steps": 1},
        }

        restored = ActionDecision.from_dict(raw)
        payload = action_decision_to_executor_payload(raw)

        self.assertEqual(restored.to_dict(), raw)
        self.assertNotIn("planner_meta", payload)
        self.assertEqual(payload["action_type"], ACTION_CLICK)
        self.assertEqual(payload["x"], 120)
        self.assertEqual(payload["y"], 40)


if __name__ == "__main__":
    unittest.main()
