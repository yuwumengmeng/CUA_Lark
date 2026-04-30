"""第二周期 B/C 协议联调测试。

测试目标：
验证成员 B 的 `ActionDecisionV2` 能稳定转成成员 C 可执行 payload，
成员 C 的 `ActionResult` v2 字段能被 B 和 recorder 消费，并确认 C 产出的
validator 结果字段如何影响 Planner 状态推进。

依赖约定：
本文件只使用 fake ui_state、fake executor 和项目内 `artifacts/tmp` 临时目录；
不访问真实飞书窗口、不截图、不移动鼠标、不调用真实 VLM。

运行方式：
python -m unittest tests\testABC\test_second_cycle_contracts.py
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from executor import RunRecorder
from planner.action_protocol import (
    ACTION_CLICK,
    ACTION_TYPE,
    ActionDecisionV2,
    action_decision_v2_to_executor_payload,
)
from planner.planner_loop_v2 import FAILURE_VALIDATION_FAILED, PlannerLoopV2
from planner.runtime_state import RuntimeStatus, create_runtime_state_v2
from planner.workflow_schema import build_im_message_workflow
from schemas import ActionResult


ARTIFACT_ROOT = Path("artifacts") / "tmp" / "test_second_cycle_contracts"


class SecondCycleContractTests(unittest.TestCase):
    def setUp(self) -> None:
        shutil.rmtree(ARTIFACT_ROOT, ignore_errors=True)
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(ARTIFACT_ROOT, ignore_errors=True)

    def test_action_decision_v2_payload_is_stable_for_c(self):
        decision = ActionDecisionV2.click_candidate(
            step_name="open_chat",
            target_candidate_id="candidate_chat_001",
            x=320,
            y=240,
            reason="rule and vlm selected target chat",
            retry_count=1,
            expected_after_state={"chat_name_visible": "测试群"},
        ).to_dict()

        payload = action_decision_v2_to_executor_payload(decision)

        for field_name in (
            "action_type",
            "target_candidate_id",
            "x",
            "y",
            "text",
            "keys",
            "reason",
            "step_name",
            "retry_count",
            "expected_after_state",
        ):
            self.assertIn(field_name, decision)
        self.assertEqual(payload["action_type"], ACTION_CLICK)
        self.assertEqual(payload["target_candidate_id"], "candidate_chat_001")
        self.assertEqual(payload["target_element_id"], "candidate_chat_001")
        self.assertNotIn("step_name", payload)
        self.assertNotIn("retry_count", payload)
        self.assertNotIn("expected_after_state", payload)

    def test_action_result_v2_payload_is_stable_for_b(self):
        result = ActionResult.success(
            action_type=ACTION_CLICK,
            start_ts="2026-04-24T10:00:00.000Z",
            end_ts="2026-04-24T10:00:00.120Z",
            warning="minor offset",
            target_candidate_id="candidate_chat_001",
            planned_click_point=[320, 240],
            actual_click_point=[322, 241],
            before_screenshot={"screenshot_path": "before.png"},
            after_screenshot={"screenshot_path": "after.png"},
            screen_meta_snapshot={"screenshot_size": [1920, 1080]},
        ).to_dict()

        for field_name in (
            "ok",
            "action_type",
            "target_candidate_id",
            "planned_click_point",
            "actual_click_point",
            "click_offset",
            "before_screenshot",
            "after_screenshot",
            "duration_ms",
            "error",
            "warning",
            "screen_meta_snapshot",
        ):
            self.assertIn(field_name, result)
        self.assertTrue(result["ok"])
        self.assertEqual(result["click_offset"], [2, 1])
        self.assertEqual(result["duration_ms"], 120)

    def test_c_validator_result_advances_planner_state(self):
        workflow = build_im_message_workflow(chat_name="测试群", message_text="Hello CUA-Lark")
        state = create_runtime_state_v2(workflow)
        state.step_idx = 4

        result = PlannerLoopV2().run_once(
            workflow,
            message_input_ui_state,
            validator_passing_executor,
            state,
            get_after_ui_state=message_input_ui_state,
        )

        self.assertEqual(result.step_outcome, RuntimeStatus.RUNNING.value)
        self.assertEqual(state.step_idx, 5)
        self.assertEqual(state.history[-1].status, RuntimeStatus.SUCCESS.value)
        self.assertEqual(
            state.history[-1].action_result["validator_result"]["check_type"],
            "text_visible",
        )

    def test_planner_failure_reason_is_written_to_run_summary(self):
        recorder = RunRecorder(
            run_id="second_cycle_failure_summary",
            artifact_root=ARTIFACT_ROOT,
        )
        summary = recorder.record_step(
            step_idx=1,
            step_name="verify_message_sent",
            action_decision={
                "action_type": ACTION_TYPE,
                "step_name": "verify_message_sent",
                "reason": "validator failed",
            },
            action_result=ActionResult.failure(
                action_type=ACTION_TYPE,
                start_ts="2026-04-24T10:00:00.000Z",
                end_ts="2026-04-24T10:00:00.010Z",
                error=FAILURE_VALIDATION_FAILED,
            ),
            validator_result={
                "passed": False,
                "check_type": "text_visible",
                "failure_reason": FAILURE_VALIDATION_FAILED,
                "evidence": {"target_text": "Hello CUA-Lark"},
            },
            failure_reason=FAILURE_VALIDATION_FAILED,
        )

        self.assertEqual(summary["failure_reasons"], {FAILURE_VALIDATION_FAILED: 1})
        self.assertEqual(summary["validator_count"], 1)
        self.assertEqual(summary["validator_failure_count"], 1)


def message_input_ui_state() -> dict:
    return {
        "screenshot_path": "fake/im-input.png",
        "candidates": [
            {
                "id": "candidate_message_input",
                "text": "",
                "bbox": [100, 500, 800, 560],
                "center": [450, 530],
                "role": "Edit",
                "editable": True,
                "clickable": True,
                "candidate_type": "message_input",
            }
        ],
        "page_summary": {
            "top_texts": ["测试群"],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": 1,
        },
        "vlm_result": {"structured": {}},
        "document_region_summary": {},
        "screen_meta": {"screenshot_size": [1920, 1080]},
    }


def validator_passing_executor(action_payload: dict) -> dict:
    result = ActionResult.success(
        action_type=str(action_payload["action_type"]),
        start_ts="2026-04-24T10:00:00.000Z",
        end_ts="2026-04-24T10:00:00.020Z",
        target_candidate_id=action_payload.get("target_candidate_id"),
    ).to_dict()
    result["validator_result"] = {
        "passed": True,
        "check_type": "text_visible",
        "failure_reason": None,
        "evidence": {"source": "C validator"},
    }
    return result


if __name__ == "__main__":
    unittest.main()
