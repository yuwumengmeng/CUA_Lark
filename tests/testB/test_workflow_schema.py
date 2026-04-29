"""成员 B 第二周期 workflow.v2 流程表示测试。

测试目标：
本文件验证 `planner/workflow_schema.py` 是否能表达第二周期 IM 主流程，包括
默认 6 个 step、固定 step 字段、半结构化 dict roundtrip，以及非法字段拦截。

重要约定：
这里不访问真实飞书窗口、不调用 VLM、不执行鼠标键盘动作，只测试流程 schema。

运行方式：
```powershell
python -m unittest tests\testB\test_workflow_schema.py
python -m unittest discover -s tests\testB
```

修改规则：
1. 新增 workflow step 字段时，必须同步补充字段断言。
2. 新增 step_type 时，必须同步更新默认流程和非法类型测试。
3. 不允许把 Planner Loop v2 执行逻辑放进这个 schema 测试。
"""

from __future__ import annotations

import unittest

from planner.workflow_schema import (
    STEP_FOCUS_MESSAGE_INPUT,
    STEP_OPEN_CHAT,
    STEP_OPEN_SEARCH,
    STEP_SEND_MESSAGE,
    STEP_TYPE_KEYWORD,
    STEP_VERIFY_MESSAGE_SENT,
    WORKFLOW_SCHEMA_VERSION,
    WORKFLOW_TYPE_IM_MESSAGE,
    WorkflowSchemaError,
    WorkflowStep,
    build_im_message_workflow,
    build_sample_im_workflows,
    parse_workflow,
)


class WorkflowSchemaTests(unittest.TestCase):
    def test_default_im_workflow_outputs_six_steps(self):
        workflow = build_im_message_workflow(
            chat_name="测试群",
            message_text="Hello CUA-Lark",
        )

        self.assertEqual(workflow.version, WORKFLOW_SCHEMA_VERSION)
        self.assertEqual(workflow.workflow_type, WORKFLOW_TYPE_IM_MESSAGE)
        self.assertEqual(workflow.params["chat_name"], "测试群")
        self.assertEqual(workflow.params["message_text"], "Hello CUA-Lark")
        self.assertEqual(
            [step.step_type for step in workflow.steps],
            [
                STEP_OPEN_SEARCH,
                STEP_TYPE_KEYWORD,
                STEP_OPEN_CHAT,
                STEP_FOCUS_MESSAGE_INPUT,
                STEP_SEND_MESSAGE,
                STEP_VERIFY_MESSAGE_SENT,
            ],
        )

    def test_each_step_has_required_workflow_fields(self):
        workflow = build_im_message_workflow(
            chat_name="测试群",
            message_text="Hello CUA-Lark",
        )

        for step in workflow.steps:
            step_dict = step.to_dict()
            self.assertIn("step_name", step_dict)
            self.assertIn("step_type", step_dict)
            self.assertIn("instruction", step_dict)
            self.assertIn("target", step_dict)
            self.assertIn("input_text", step_dict)
            self.assertIn("success_condition", step_dict)
            self.assertIn("fallback", step_dict)
            self.assertIn("max_retry", step_dict)
            self.assertIsInstance(step.success_condition, dict)
            self.assertIsInstance(step.fallback, list)
            self.assertGreaterEqual(step.max_retry, 0)

    def test_structured_workflow_roundtrip(self):
        workflow = build_im_message_workflow(
            chat_name="测试群",
            message_text="第二周期 smoke test",
            workflow_id="workflow_demo",
        )

        restored = parse_workflow(workflow.to_dict())

        self.assertEqual(restored.to_dict(), workflow.to_dict())

    def test_invalid_step_type_is_rejected(self):
        with self.assertRaises(WorkflowSchemaError):
            WorkflowStep(
                step_name="bad_step",
                step_type="bad_type",
                instruction="非法步骤",
            )

    def test_invalid_max_retry_is_rejected(self):
        with self.assertRaises(WorkflowSchemaError):
            WorkflowStep(
                step_name=STEP_OPEN_SEARCH,
                step_type=STEP_OPEN_SEARCH,
                instruction="打开搜索",
                max_retry=-1,
            )

    def test_sample_im_workflows_prepare_three_tasks(self):
        samples = build_sample_im_workflows()

        self.assertEqual(len(samples), 3)
        self.assertTrue(all(sample.workflow_type == WORKFLOW_TYPE_IM_MESSAGE for sample in samples))
        self.assertEqual([len(sample.steps) for sample in samples], [6, 6, 6])


if __name__ == "__main__":
    unittest.main()