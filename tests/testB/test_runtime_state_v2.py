"""成员 B 第二周期 Runtime State v2 测试。

测试目标：
验证 `planner/runtime_state.py` 新增的 runtime_state.v2 是否能围绕 workflow
记录多步流程状态、A 侧 ui_state 语义字段、C 侧 action_result、retry、失败原因、
history 以及 VLM 与规则候选冲突。

依赖约定：
本文件使用 fake workflow、fake ui_state 和 fake action/result，不访问真实飞书窗口、
不截图、不调用真实 executor。

运行方式：
python -m unittest tests\testB\test_runtime_state_v2.py
python -m unittest discover -s tests\testB
"""

import unittest

from planner.runtime_state import (
    RUNTIME_STATE_V2_SCHEMA_VERSION,
    RuntimeStateError,
    RuntimeStateV2,
    RuntimeStatus,
    create_runtime_state_v2,
    parse_runtime_state_v2,
)
from planner.workflow_schema import STEP_OPEN_SEARCH, STEP_TYPE_KEYWORD, build_im_message_workflow


class RuntimeStateV2Tests(unittest.TestCase):
    def test_workflow_initialization_starts_at_first_step(self):
        state = create_runtime_state_v2(sample_workflow())

        self.assertEqual(state.version, RUNTIME_STATE_V2_SCHEMA_VERSION)
        self.assertEqual(state.status, RuntimeStatus.RUNNING.value)
        self.assertEqual(state.step_idx, 0)
        self.assertEqual(state.current_step.step_name, STEP_OPEN_SEARCH)
        self.assertEqual(state.to_dict()["current_step"]["step_name"], STEP_OPEN_SEARCH)

    def test_update_ui_state_records_a_side_semantic_fields(self):
        state = create_runtime_state_v2(sample_workflow())
        state.update_ui_state(fake_ui_state())

        self.assertEqual(state.current_ui_state["screenshot_path"], "run/step_001.png")
        self.assertEqual(state.current_page_summary["top_texts"], ["搜索", "测试群"])
        self.assertEqual(state.current_vlm_result["structured"]["target_candidate_id"], "elem_search")
        self.assertEqual(state.current_document_region_summary["visible_title"], "无")
        self.assertEqual(state.current_screen_meta["dpi_scale"], 1.25)

    def test_record_action_result_updates_last_fields_and_history(self):
        state = create_runtime_state_v2(sample_workflow())
        state.update_ui_state(fake_ui_state())
        action = {
            "version": "action_decision.v2",
            "action_type": "click",
            "step_name": STEP_OPEN_SEARCH,
            "target_candidate_id": "elem_search",
            "x": 120,
            "y": 40,
        }
        result = {"ok": True, "action_type": "click", "error": None}

        state.record_action(action, result)

        self.assertEqual(state.last_action_decision, action)
        self.assertEqual(state.last_action_result, result)
        self.assertEqual(len(state.history), 1)
        self.assertEqual(state.history[0].step_name, STEP_OPEN_SEARCH)
        self.assertEqual(state.history[0].page_summary["candidate_count"], 1)

    def test_retry_advance_and_failure_fields_are_stable(self):
        state = create_runtime_state_v2(sample_workflow())

        self.assertEqual(state.increment_retry(), 1)
        self.assertEqual(state.retry_count, 1)
        state.advance_step()
        self.assertEqual(state.step_idx, 1)
        self.assertEqual(state.current_step.step_name, STEP_TYPE_KEYWORD)
        self.assertEqual(state.retry_count, 0)

        state.mark_failed("validation_failed")
        self.assertEqual(state.status, RuntimeStatus.FAILED.value)
        self.assertEqual(state.failure_reason, "validation_failed")

    def test_conflict_records_vlm_and_rule_candidate_ids(self):
        state = create_runtime_state_v2(sample_workflow())

        state.record_conflict(
            rule_candidate_id="elem_rule",
            vlm_candidate_id="elem_vlm",
            reason="rule and vlm selected different search boxes",
            vlm_confidence=0.61,
        )

        self.assertEqual(len(state.conflicts), 1)
        self.assertEqual(state.conflicts[0].rule_candidate_id, "elem_rule")
        self.assertEqual(state.to_dict()["conflicts"][0]["vlm_candidate_id"], "elem_vlm")

    def test_roundtrip_keeps_runtime_state_v2_fields(self):
        state = create_runtime_state_v2(sample_workflow())
        state.update_ui_state(fake_ui_state())
        state.record_conflict(
            rule_candidate_id="elem_rule",
            vlm_candidate_id="elem_vlm",
            reason="candidate conflict",
        )

        restored = parse_runtime_state_v2(state.to_dict())

        self.assertEqual(restored.to_dict(), state.to_dict())
        self.assertEqual(restored.current_step.step_name, STEP_OPEN_SEARCH)

    def test_invalid_status_and_retry_count_are_rejected(self):
        with self.assertRaises(RuntimeStateError):
            RuntimeStateV2(workflow=sample_workflow(), status="unknown")
        with self.assertRaises(RuntimeStateError):
            RuntimeStateV2(workflow=sample_workflow(), retry_count=-1)


def sample_workflow():
    return build_im_message_workflow(chat_name="测试群", message_text="Hello CUA-Lark")


def fake_ui_state():
    return {
        "screenshot_path": "run/step_001.png",
        "candidates": [
            {
                "id": "elem_search",
                "text": "搜索",
                "center": [120, 40],
                "bbox": [20, 20, 220, 60],
            }
        ],
        "page_summary": {
            "top_texts": ["搜索", "测试群"],
            "candidate_count": 1,
            "has_search_box": True,
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": "elem_search",
                "confidence": 0.88,
                "uncertain": False,
            }
        },
        "document_region_summary": {"visible_title": "无"},
        "screen_meta": {"width": 2560, "height": 1440, "dpi_scale": 1.25},
    }


if __name__ == "__main__":
    unittest.main()
