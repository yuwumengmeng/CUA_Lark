"""成员 B 第二周期 Planner Loop v2 测试。

测试目标：
验证 `planner/planner_loop_v2.py` 能把 workflow.v2、runtime_state.v2 和
action_decision.v2 串成多步 planner 闭环，覆盖 IM 主流程、retry、失败原因、
VLM/规则融合、candidate 异常分类、多候选消歧、文档摘要判断和 history 回放。

依赖约定：
本文件全部使用 fake `ui_state` 与 fake executor，不访问真实飞书窗口、不截图、
不移动鼠标，也不调用真实 VLM。

运行方式：
python -m unittest tests\testB\test_planner_loop_v2.py
python -m unittest discover -s tests\testB
"""

import unittest

from planner.action_protocol import ACTION_CLICK, ACTION_HOTKEY, ACTION_TYPE
from planner.planner_loop_v2 import (
    FAILURE_CANDIDATE_CONFLICT,
    FAILURE_CANDIDATE_INVALID,
    FAILURE_COORDINATE_INVALID,
    FAILURE_FALLBACK_FAILED,
    FAILURE_MAX_RETRY_EXCEEDED,
    FAILURE_PAGE_MISMATCH,
    FAILURE_PAGE_NOT_CHANGED,
    FAILURE_TARGET_NOT_FOUND,
    FAILURE_VALIDATION_FAILED,
    FAILURE_VLM_UNCERTAIN,
    PlannerLoopV2,
    plan_next_workflow_action,
    run_workflow_until_done,
)
from planner.runtime_state import RuntimeStatus, create_runtime_state_v2
from planner.workflow_schema import (
    STEP_FOCUS_MESSAGE_INPUT,
    STEP_OPEN_CHAT,
    STEP_OPEN_SEARCH,
    STEP_SEND_MESSAGE,
    STEP_TYPE_KEYWORD,
    STEP_VERIFY_MESSAGE_SENT,
    build_im_message_workflow,
)


class PlannerLoopV2Tests(unittest.TestCase):
    def test_plan_next_workflow_action_reads_current_step(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)

        decision = plan_next_workflow_action(state, ui_state_for_step(STEP_OPEN_SEARCH))

        self.assertEqual(state.current_step.step_name, STEP_OPEN_SEARCH)
        self.assertEqual(decision["action_type"], ACTION_CLICK)
        self.assertEqual(decision["step_name"], STEP_OPEN_SEARCH)
        self.assertEqual(decision["target_candidate_id"], "elem_search")

    def test_run_once_consumes_a_side_fields_and_generates_executor_payload(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        calls: list[dict] = []

        result = PlannerLoopV2().run_once(
            workflow,
            lambda: ui_state_for_step(STEP_OPEN_SEARCH),
            executor_spy(calls),
            state,
            get_after_ui_state=lambda: ui_state_for_step(STEP_TYPE_KEYWORD),
        )

        self.assertEqual(result.runtime_state.step_idx, 1)
        self.assertEqual(result.runtime_state.status, RuntimeStatus.RUNNING.value)
        self.assertEqual(state.current_page_summary["has_search_box"], True)
        self.assertEqual(state.current_vlm_result["structured"]["target_candidate_id"], "elem_search")
        self.assertEqual(state.current_document_region_summary["visible_title"], "无")
        self.assertEqual(state.current_screen_meta["dpi_scale"], 1.25)
        self.assertEqual(calls[0]["action_type"], ACTION_CLICK)
        self.assertEqual(calls[0]["target_element_id"], "elem_search")

    def test_open_search_can_fallback_to_hotkey(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        calls: list[dict] = []

        result = PlannerLoopV2().run_once(
            workflow,
            lambda: ui_state_for_step("empty"),
            executor_spy(calls),
            state,
            get_after_ui_state=lambda: ui_state_for_step(STEP_TYPE_KEYWORD),
        )

        self.assertEqual(result.decision["action_type"], ACTION_HOTKEY)
        self.assertEqual(result.decision["keys"], ["ctrl", "k"])
        self.assertEqual(calls[0]["keys"], ["ctrl", "k"])
        self.assertEqual(state.step_idx, 1)

    def test_three_continuous_steps_advance(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        provider = SequenceProvider(
            [
                ui_state_for_step(STEP_OPEN_SEARCH),
                ui_state_for_step(STEP_TYPE_KEYWORD),
                ui_state_for_step(STEP_TYPE_KEYWORD),
                ui_state_for_step(STEP_OPEN_CHAT),
                ui_state_for_step(STEP_OPEN_CHAT),
                ui_state_for_step("chat_open"),
            ]
        )

        PlannerLoopV2().run_once(workflow, provider.next, success_executor, state, get_after_ui_state=provider.next)
        PlannerLoopV2().run_once(workflow, provider.next, success_executor, state, get_after_ui_state=provider.next)
        PlannerLoopV2().run_once(workflow, provider.next, success_executor, state, get_after_ui_state=provider.next)

        self.assertEqual(state.step_idx, 3)
        self.assertEqual(state.current_step.step_name, "focus_message_input")
        self.assertEqual(len(state.history), 3)
        self.assertEqual([item.status for item in state.history], [RuntimeStatus.SUCCESS.value] * 3)

    def test_fake_im_workflow_runs_to_success(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        before = SequenceProvider(
            [
                ui_state_for_step(STEP_OPEN_SEARCH),
                ui_state_for_step(STEP_TYPE_KEYWORD),
                ui_state_for_step(STEP_OPEN_CHAT),
                ui_state_for_step("message_input"),
                ui_state_for_step("message_input"),
                ui_state_for_step("message_input_with_text"),
                ui_state_for_step(STEP_VERIFY_MESSAGE_SENT),
            ]
        )
        after = SequenceProvider(
            [
                ui_state_for_step(STEP_TYPE_KEYWORD),
                ui_state_for_step(STEP_OPEN_CHAT),
                ui_state_for_step("chat_open"),
                ui_state_for_step("message_input"),
                ui_state_for_step("message_input_with_text"),
                ui_state_for_step(STEP_VERIFY_MESSAGE_SENT),
                ui_state_for_step(STEP_VERIFY_MESSAGE_SENT),
            ]
        )

        results = run_workflow_until_done(
            workflow,
            before.next,
            success_executor,
            state,
            get_after_ui_state=after.next,
            max_steps=10,
        )

        self.assertEqual(state.status, RuntimeStatus.SUCCESS.value)
        self.assertEqual(state.step_idx, 6)
        self.assertGreaterEqual(len(results), 6)
        self.assertEqual(state.history[-1].step_name, STEP_VERIFY_MESSAGE_SENT)
        self.assertEqual(state.history[-1].status, RuntimeStatus.SUCCESS.value)

    def test_executor_failure_retries_then_max_retry_failed(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        loop = PlannerLoopV2()

        first = loop.run_once(
            workflow,
            lambda: ui_state_for_step(STEP_OPEN_SEARCH),
            failing_executor,
            state,
        )
        self.assertEqual(first.runtime_state.status, RuntimeStatus.RUNNING.value)
        self.assertEqual(first.runtime_state.retry_count, 1)

        second = loop.run_once(
            workflow,
            lambda: ui_state_for_step(STEP_OPEN_SEARCH),
            failing_executor,
            state,
        )

        self.assertEqual(second.runtime_state.status, RuntimeStatus.FAILED.value)
        self.assertEqual(second.runtime_state.failure_reason, FAILURE_MAX_RETRY_EXCEEDED)
        self.assertEqual(state.history[-1].failure_reason, FAILURE_MAX_RETRY_EXCEEDED)

    def test_missing_chat_target_enters_retry_without_executor_call(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 2
        calls: list[dict] = []

        result = PlannerLoopV2().run_once(
            workflow,
            lambda: ui_state_for_step("empty"),
            executor_spy(calls),
            state,
        )

        self.assertEqual(calls, [])
        self.assertEqual(result.runtime_state.status, RuntimeStatus.RUNNING.value)
        self.assertEqual(result.runtime_state.retry_count, 1)
        self.assertEqual(result.decision["failure_reason"], FAILURE_TARGET_NOT_FOUND)

    def test_missing_chat_target_next_action_reopens_search_as_fallback(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 2
        state.failure_reason = FAILURE_TARGET_NOT_FOUND
        state.retry_count = 1

        decision = plan_next_workflow_action(state, ui_state_for_step("empty"))

        self.assertEqual(decision["action_type"], ACTION_HOTKEY)
        self.assertEqual(decision["keys"], ["ctrl", "k"])
        self.assertIn("fallback", decision["reason"])

    def test_open_chat_clicks_first_search_result_when_text_not_recalled(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 2

        decision = plan_next_workflow_action(state, search_results_without_target_candidate_state())

        self.assertEqual(decision["action_type"], ACTION_CLICK)
        self.assertEqual(decision["target_candidate_id"], None)
        self.assertEqual(decision["x"], 4861)
        self.assertEqual(decision["y"], 914)
        self.assertIn("fallback", decision["reason"])

    def test_vlm_rule_conflict_is_recorded(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)

        result = PlannerLoopV2().run_once(
            workflow,
            lambda: conflict_ui_state(),
            success_executor,
            state,
            get_after_ui_state=lambda: ui_state_for_step(STEP_TYPE_KEYWORD),
        )

        self.assertEqual(result.decision["target_candidate_id"], "elem_vlm_search")
        self.assertEqual(len(state.conflicts), 1)
        self.assertEqual(state.conflicts[0].rule_candidate_id, "elem_rule_search")
        self.assertEqual(state.conflicts[0].vlm_candidate_id, "elem_vlm_search")
        self.assertEqual(state.conflicts[0].reason, FAILURE_CANDIDATE_CONFLICT)

    def test_open_search_rejects_vlm_non_search_candidate(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)

        decision = plan_next_workflow_action(state, vlm_wrong_search_kind_state())

        self.assertEqual(decision["action_type"], ACTION_CLICK)
        self.assertEqual(decision["target_candidate_id"], "elem_rule_search")
        self.assertEqual(state.conflicts[0].reason, FAILURE_CANDIDATE_CONFLICT)

    def test_vlm_uncertain_falls_back_to_rule_candidate(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)

        decision = plan_next_workflow_action(state, uncertain_vlm_ui_state())

        self.assertEqual(decision["target_candidate_id"], "elem_rule_search")
        self.assertEqual(state.conflicts[0].reason, FAILURE_VLM_UNCERTAIN)
        self.assertEqual(state.conflicts[0].rule_candidate_id, "elem_rule_search")

    def test_vlm_uncertain_without_rule_candidate_fails_with_reason(self):
        workflow = build_im_message_workflow(chat_name="team", message_text="hello")
        state = create_runtime_state_v2(workflow)
        state.step_idx = 2

        decision = plan_next_workflow_action(state, uncertain_without_rule_ui_state())

        self.assertEqual(decision["status"], "failed")
        self.assertEqual(decision["failure_reason"], FAILURE_VLM_UNCERTAIN)

    def test_invalid_vlm_candidate_is_rejected(self):
        workflow = build_im_message_workflow(chat_name="team", message_text="hello")
        state = create_runtime_state_v2(workflow)
        state.step_idx = 2
        calls: list[dict] = []

        result = PlannerLoopV2().run_once(
            workflow,
            lambda: invalid_vlm_candidate_ui_state(),
            executor_spy(calls),
            state,
        )

        self.assertEqual(calls, [])
        self.assertEqual(result.decision["status"], "failed")
        self.assertEqual(result.decision["failure_reason"], FAILURE_CANDIDATE_INVALID)
        self.assertEqual(state.conflicts[0].reason, FAILURE_CANDIDATE_INVALID)

    def test_reranked_candidate_ids_influence_rule_scoring(self):
        workflow = build_im_message_workflow(chat_name="team", message_text="hello")
        state = create_runtime_state_v2(workflow)
        state.step_idx = 2

        decision = plan_next_workflow_action(state, reranked_ui_state())

        self.assertEqual(decision["target_candidate_id"], "elem_second_team")

    def test_document_summary_fields_satisfy_content_condition(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 5

        decision = plan_next_workflow_action(state, document_summary_ui_state())

        self.assertEqual(decision["status"], "success")

    def test_document_truth_from_allowed_source_satisfies_content_condition(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 5

        decision = plan_next_workflow_action(state, document_truth_ui_state())

        self.assertEqual(decision["status"], "success")

    def test_document_truth_overrides_weak_page_summary(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 5

        decision = plan_next_workflow_action(state, conflicting_document_truth_ui_state())

        self.assertEqual(decision["action_type"], "wait")

    def test_ocr_candidate_body_is_not_document_success_condition(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 5

        decision = plan_next_workflow_action(state, ocr_only_ui_state())

        self.assertEqual(decision["action_type"], "wait")

    def test_open_search_ignores_browser_shell_search_box(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)

        decision = plan_next_workflow_action(state, browser_shell_search_state())

        self.assertEqual(decision["target_candidate_id"], "elem_lark_search")
        self.assertEqual(state.conflicts[0].reason, FAILURE_CANDIDATE_CONFLICT)

    def test_open_chat_prefers_chat_list_item_over_plain_text(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 2

        decision = plan_next_workflow_action(state, chat_list_vs_plain_text_state())

        self.assertEqual(decision["target_candidate_id"], "elem_chat_row")

    def test_open_chat_prefers_exact_target_over_similar_chat(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 2

        decision = plan_next_workflow_action(state, similar_chat_candidates_state())

        self.assertEqual(decision["target_candidate_id"], "elem_target_chat")

    def test_open_chat_rejects_vlm_search_box_candidate(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 2

        decision = plan_next_workflow_action(state, open_chat_vlm_search_box_state())

        self.assertEqual(decision["target_candidate_id"], "elem_target_chat")
        self.assertEqual(state.conflicts[0].reason, FAILURE_CANDIDATE_CONFLICT)

    def test_message_input_prefers_editable_input_over_plain_text_area(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 3

        decision = plan_next_workflow_action(state, message_input_disambiguation_state())

        self.assertEqual(decision["target_candidate_id"], "elem_message_input")

    def test_message_input_rejects_search_box_candidate(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 3

        decision = plan_next_workflow_action(state, search_box_only_state())

        self.assertEqual(decision["status"], "failed")
        self.assertEqual(decision["failure_reason"], FAILURE_TARGET_NOT_FOUND)

    def test_focus_message_input_uses_bottom_input_fallback_after_chat_open(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 2
        state.record_action(
            {"step_name": STEP_OPEN_CHAT, "action_type": ACTION_CLICK},
            {"ok": True},
            status=RuntimeStatus.SUCCESS.value,
        )
        state.advance_step()

        decision = plan_next_workflow_action(state, chat_without_input_candidate_state())

        self.assertEqual(decision["action_type"], ACTION_CLICK)
        self.assertIsNone(decision["target_candidate_id"])
        self.assertEqual(decision["x"], 5120)
        self.assertEqual(decision["y"], 1999)
        self.assertIn("bottom input", decision["reason"])

    def test_send_message_uses_bottom_input_fallback_after_focus(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 2
        state.record_action(
            {"step_name": STEP_OPEN_CHAT, "action_type": ACTION_CLICK},
            {"ok": True},
            status=RuntimeStatus.SUCCESS.value,
        )
        state.advance_step()
        state.record_action(
            {"step_name": STEP_FOCUS_MESSAGE_INPUT, "action_type": ACTION_CLICK},
            {"ok": True},
            status=RuntimeStatus.SUCCESS.value,
        )
        state.advance_step()

        decision = plan_next_workflow_action(state, chat_without_input_candidate_state())

        self.assertEqual(decision["action_type"], ACTION_TYPE)
        self.assertIsNone(decision["target_candidate_id"])
        self.assertEqual(decision["x"], 5120)
        self.assertEqual(decision["y"], 1999)
        self.assertEqual(decision["text"], "Hello CUA-Lark")
        self.assertIn("bottom input", decision["reason"])

    def test_send_message_retry_presses_enter_after_type_attempt(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 4
        state.retry_count = 1
        state.failure_reason = FAILURE_VALIDATION_FAILED
        state.last_action_decision = {
            "step_name": STEP_SEND_MESSAGE,
            "action_type": ACTION_TYPE,
            "text": "Hello CUA-Lark",
        }

        decision = plan_next_workflow_action(state, chat_without_input_candidate_state())

        self.assertEqual(decision["action_type"], ACTION_HOTKEY)
        self.assertEqual(decision["keys"], ["enter"])
        self.assertIn("enter", decision["reason"])

    def test_open_chat_wrong_after_page_retries_with_page_mismatch(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 2

        result = PlannerLoopV2().run_once(
            workflow,
            lambda: ui_state_for_step(STEP_OPEN_CHAT),
            success_executor,
            state,
            get_after_ui_state=wrong_chat_after_state,
        )

        self.assertEqual(result.runtime_state.status, RuntimeStatus.RUNNING.value)
        self.assertEqual(result.runtime_state.retry_count, 1)
        self.assertEqual(result.runtime_state.failure_reason, FAILURE_PAGE_MISMATCH)
        self.assertEqual(result.runtime_state.history[-1].failure_reason, FAILURE_PAGE_MISMATCH)

    def test_wrong_chat_next_action_goes_back_as_fallback(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 2
        state.retry_count = 1
        state.failure_reason = FAILURE_PAGE_MISMATCH

        decision = plan_next_workflow_action(state, wrong_chat_after_state())

        self.assertEqual(decision["action_type"], ACTION_HOTKEY)
        self.assertEqual(decision["keys"], ["alt", "left"])
        self.assertIn("fallback", decision["reason"])

    def test_click_page_not_changed_waits_before_retry(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 2

        result = PlannerLoopV2().run_once(
            workflow,
            lambda: ui_state_for_step(STEP_OPEN_CHAT),
            success_executor,
            state,
            get_after_ui_state=lambda: ui_state_for_step(STEP_OPEN_CHAT),
        )
        self.assertEqual(result.runtime_state.failure_reason, FAILURE_PAGE_NOT_CHANGED)
        self.assertEqual(result.runtime_state.retry_count, 1)

        decision = plan_next_workflow_action(state, ui_state_for_step(STEP_OPEN_CHAT))
        self.assertEqual(decision["action_type"], "wait")
        self.assertIn("fallback", decision["reason"])

    def test_type_keyword_validation_failure_refocuses_search_box(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 1

        result = PlannerLoopV2().run_once(
            workflow,
            lambda: ui_state_for_step(STEP_TYPE_KEYWORD),
            success_executor,
            state,
            get_after_ui_state=lambda: input_failed_state(),
        )
        self.assertEqual(result.runtime_state.failure_reason, FAILURE_VALIDATION_FAILED)
        self.assertEqual(result.runtime_state.retry_count, 1)

        decision = plan_next_workflow_action(state, ui_state_for_step(STEP_TYPE_KEYWORD))
        self.assertEqual(decision["action_type"], ACTION_CLICK)
        self.assertIn("refocus", decision["reason"])

    def test_type_keyword_success_can_use_changed_search_results(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 1

        result = PlannerLoopV2().run_once(
            workflow,
            lambda: ui_state_for_step(STEP_TYPE_KEYWORD),
            success_executor,
            state,
            get_after_ui_state=lambda: changed_without_keyword_state(),
        )

        self.assertEqual(result.runtime_state.step_idx, 2)
        self.assertEqual(result.runtime_state.current_step.step_name, STEP_OPEN_CHAT)

    def test_coordinate_invalid_candidate_switches_to_valid_candidate(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)

        decision = plan_next_workflow_action(state, coordinate_invalid_with_backup_state())

        self.assertEqual(decision["target_candidate_id"], "elem_valid_search")
        self.assertEqual(state.conflicts[0].reason, FAILURE_COORDINATE_INVALID)

    def test_absolute_screen_coordinates_use_screen_origin_bounds(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 1

        decision = plan_next_workflow_action(state, absolute_screen_search_state())

        self.assertEqual(decision["action_type"], ACTION_TYPE)
        self.assertEqual(decision["target_candidate_id"], "elem_abs_search")
        self.assertIsNone(decision["x"])
        self.assertIsNone(decision["y"])

    def test_fallback_executor_failure_records_fallback_failed(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 2
        state.retry_count = 1
        state.failure_reason = FAILURE_PAGE_MISMATCH

        result = PlannerLoopV2().run_once(
            workflow,
            wrong_chat_after_state,
            failing_executor,
            state,
        )

        self.assertEqual(result.runtime_state.failure_reason, FAILURE_FALLBACK_FAILED)
        self.assertEqual(result.runtime_state.history[-1].failure_reason, FAILURE_FALLBACK_FAILED)

    def test_open_search_success_can_use_vlm_semantic_state(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)

        result = PlannerLoopV2().run_once(
            workflow,
            lambda: ui_state_for_step("empty"),
            success_executor,
            state,
            get_after_ui_state=lambda: semantic_search_state(),
        )

        self.assertEqual(result.runtime_state.step_idx, 1)
        self.assertEqual(result.runtime_state.status, RuntimeStatus.RUNNING.value)

    def test_send_message_success_can_use_validator_result(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)
        state.step_idx = 4
        state.record_action(
            {
                "status": "continue",
                "action_type": ACTION_TYPE,
                "step_name": STEP_SEND_MESSAGE,
                "text": "Hello CUA-Lark",
            },
            {"ok": True, "action_type": ACTION_TYPE},
        )

        result = PlannerLoopV2().run_once(
            workflow,
            lambda: ui_state_for_step("message_input_with_text"),
            validator_success_executor,
            state,
            get_after_ui_state=lambda: changed_without_keyword_state(),
        )

        self.assertEqual(result.runtime_state.step_idx, 5)
        self.assertEqual(result.runtime_state.current_step.step_name, STEP_VERIFY_MESSAGE_SENT)

    def test_history_is_replayable(self):
        workflow = sample_workflow()
        state = create_runtime_state_v2(workflow)

        PlannerLoopV2().run_once(
            workflow,
            lambda: ui_state_for_step(STEP_OPEN_SEARCH),
            success_executor,
            state,
            get_after_ui_state=lambda: ui_state_for_step(STEP_TYPE_KEYWORD),
        )

        item = state.to_dict()["history"][0]
        self.assertEqual(item["step_name"], STEP_OPEN_SEARCH)
        self.assertIn("page_summary", item)
        self.assertEqual(item["action_decision"]["step_name"], STEP_OPEN_SEARCH)
        self.assertEqual(item["action_result"]["ok"], True)
        self.assertEqual(item["status"], RuntimeStatus.SUCCESS.value)


class SequenceProvider:
    def __init__(self, values: list[dict]) -> None:
        self.values = list(values)

    def next(self) -> dict:
        if len(self.values) == 1:
            return self.values[0]
        return self.values.pop(0)


def sample_workflow():
    return build_im_message_workflow(chat_name="测试群", message_text="Hello CUA-Lark")


def executor_spy(calls: list[dict]):
    def execute(action: dict) -> dict:
        calls.append(dict(action))
        return success_executor(action)

    return execute


def success_executor(action: dict) -> dict:
    return {"ok": True, "action_type": action["action_type"], "error": None}


def failing_executor(action: dict) -> dict:
    return {"ok": False, "action_type": action["action_type"], "error": "boom"}


def validator_success_executor(action: dict) -> dict:
    return {
        "ok": True,
        "action_type": action["action_type"],
        "error": None,
        "validator_result": {"passed": True, "message": "message visible"},
    }


def ui_state_for_step(step_name: str) -> dict:
    candidates: list[dict] = []
    top_texts: list[str] = []
    has_search_box = False

    if step_name == STEP_OPEN_SEARCH:
        candidates = [candidate("elem_search", "搜索", editable=True)]
        top_texts = ["首页", "搜索"]
        has_search_box = True
    elif step_name == STEP_TYPE_KEYWORD:
        candidates = [candidate("elem_search", "", editable=True)]
        top_texts = ["搜索"]
        has_search_box = True
    elif step_name == STEP_OPEN_CHAT:
        candidates = [candidate("elem_chat", "测试群")]
        top_texts = ["测试群"]
        has_search_box = True
    elif step_name == "chat_open":
        candidates = [candidate("elem_message_input", "", editable=True)]
        top_texts = ["测试群"]
    elif step_name == "message_input":
        candidates = [candidate("elem_message_input", "", editable=True)]
        top_texts = ["测试群"]
    elif step_name == "message_input_with_text":
        candidates = [candidate("elem_message_input", "Hello CUA-Lark", editable=True)]
        top_texts = ["测试群", "Hello CUA-Lark"]
    elif step_name == STEP_VERIFY_MESSAGE_SENT:
        candidates = [candidate("elem_message", "Hello CUA-Lark")]
        top_texts = ["测试群", "Hello CUA-Lark"]

    return {
        "screenshot_path": f"run/{step_name}.png",
        "candidates": candidates,
        "page_summary": {
            "top_texts": top_texts,
            "has_search_box": has_search_box,
            "has_modal": False,
            "candidate_count": len(candidates),
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": candidates[0]["id"] if candidates else None,
                "reranked_candidate_ids": [item["id"] for item in candidates],
                "confidence": 0.88,
                "uncertain": False,
            }
        },
        "document_region_summary": {"visible_title": "无"},
        "screen_meta": {"width": 2560, "height": 1440, "dpi_scale": 1.25},
    }


def conflict_ui_state() -> dict:
    candidates = [
        candidate("elem_rule_search", "搜索", editable=True, confidence=0.95),
        candidate("elem_vlm_search", "搜索", editable=True, confidence=0.3, x=180),
    ]
    return {
        "screenshot_path": "run/conflict.png",
        "candidates": candidates,
        "page_summary": {
            "top_texts": ["搜索"],
            "has_search_box": True,
            "has_modal": False,
            "candidate_count": len(candidates),
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": "elem_vlm_search",
                "reranked_candidate_ids": ["elem_vlm_search", "elem_rule_search"],
                "confidence": 0.9,
                "uncertain": False,
            }
        },
        "document_region_summary": {},
        "screen_meta": {},
    }


def search_results_without_target_candidate_state() -> dict:
    return {
        "screenshot_path": "run/search-results-without-target-candidate.png",
        "candidates": [
            candidate(
                "elem_search",
                "搜索 (Ctrl+K)",
                editable=True,
                role="LarkSearchBox",
            ),
            candidate("elem_result_row", "HAA", role="LarkChatListItem", x=1300),
        ],
        "page_summary": {
            "top_texts": ["搜索 (Ctrl+K)", "联系人群组"],
            "has_search_box": True,
            "has_modal": False,
            "candidate_count": 2,
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": None,
                "reranked_candidate_ids": [],
                "confidence": 0.0,
                "uncertain": True,
            }
        },
        "document_region_summary": {},
        "screen_meta": {
            "screenshot_size": [2582, 1550],
            "screen_origin": [3829, 542],
            "screen_resolution": [2560, 1600],
        },
    }


def vlm_wrong_search_kind_state() -> dict:
    state = conflict_ui_state()
    state["candidates"] = [
        candidate("elem_rule_search", "搜索", editable=True, confidence=0.95),
        candidate("elem_vlm_messages", "消息", role="LarkSecondaryNavItem", confidence=0.99),
    ]
    state["vlm_result"]["structured"] = {
        "target_candidate_id": "elem_vlm_messages",
        "reranked_candidate_ids": ["elem_vlm_messages", "elem_rule_search"],
        "confidence": 0.9,
        "uncertain": False,
    }
    return state


def uncertain_vlm_ui_state() -> dict:
    state = conflict_ui_state()
    state["candidates"] = [candidate("elem_rule_search", "搜索", editable=True, confidence=0.95)]
    state["page_summary"]["candidate_count"] = 1
    state["vlm_result"]["structured"] = {
        "target_candidate_id": None,
        "reranked_candidate_ids": [],
        "confidence": 0.2,
        "uncertain": True,
    }
    return state


def uncertain_without_rule_ui_state() -> dict:
    return {
        "screenshot_path": "run/uncertain-without-rule.png",
        "candidates": [candidate("elem_other", "other", confidence=0.4)],
        "page_summary": {
            "top_texts": ["other"],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": 1,
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": None,
                "reranked_candidate_ids": [],
                "confidence": 0.1,
                "uncertain": True,
            }
        },
        "document_region_summary": {},
        "screen_meta": {},
    }


def invalid_vlm_candidate_ui_state() -> dict:
    return {
        "screenshot_path": "run/invalid.png",
        "candidates": [
            {
                "id": "elem_invalid_team",
                "text": "team",
                "bbox": [100, 20, 90, 60],
                "role": "Button",
                "clickable": True,
                "editable": False,
                "source": "uia",
                "confidence": 0.95,
            }
        ],
        "page_summary": {
            "top_texts": ["team"],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": 1,
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": "elem_invalid_team",
                "reranked_candidate_ids": ["elem_invalid_team"],
                "confidence": 0.92,
                "uncertain": False,
            }
        },
        "document_region_summary": {},
        "screen_meta": {},
    }


def reranked_ui_state() -> dict:
    return {
        "screenshot_path": "run/rerank.png",
        "candidates": [
            candidate("elem_first_team", "team", confidence=0.95, x=100),
            candidate("elem_second_team", "team", confidence=0.1, x=220),
        ],
        "page_summary": {
            "top_texts": ["team"],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": 2,
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": None,
                "reranked_candidate_ids": ["elem_second_team", "elem_first_team"],
                "confidence": 0.6,
                "uncertain": False,
            }
        },
        "document_region_summary": {},
        "screen_meta": {},
    }


def document_summary_ui_state() -> dict:
    return {
        "screenshot_path": "run/document-summary.png",
        "candidates": [],
        "page_summary": {
            "top_texts": [],
            "semantic_summary": {"summary": "latest visible message exists"},
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": 0,
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": None,
                "reranked_candidate_ids": [],
                "confidence": 0.8,
                "uncertain": True,
                "page_semantics": {
                    "domain": "docs",
                    "summary": "Hello CUA-Lark is visible in the document region",
                },
            }
        },
        "document_region_summary": {
            "visible_title": "send result",
            "visible_texts": ["Hello CUA-Lark"],
        },
        "screen_meta": {},
    }


def document_truth_ui_state() -> dict:
    state = document_summary_ui_state()
    state["page_summary"]["semantic_summary"] = {}
    state["document_region_summary"] = {}
    state["document_truth"] = {
        "source": "clipboard",
        "content": "Hello CUA-Lark",
    }
    return state


def conflicting_document_truth_ui_state() -> dict:
    state = document_summary_ui_state()
    state["document_truth"] = {
        "source": "openapi",
        "content": "The authoritative document text does not contain the target message.",
    }
    return state


def ocr_only_ui_state() -> dict:
    return {
        "screenshot_path": "run/ocr-only.png",
        "candidates": [
            {
                "id": "ocr_body",
                "text": "Hello CUA-Lark",
                "bbox": [10, 10, 500, 500],
                "center": [255, 255],
                "role": "Text",
                "clickable": False,
                "editable": False,
                "source": "ocr",
                "confidence": 0.7,
            }
        ],
        "page_summary": {
            "top_texts": [],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": 1,
        },
        "vlm_result": {"structured": {}},
        "document_region_summary": {},
        "screen_meta": {},
    }


def browser_shell_search_state() -> dict:
    candidates = [
        candidate(
            "browser_address_search",
            "搜索或输入网址",
            editable=True,
            confidence=0.99,
            role="Edit",
            x=10,
        ),
        candidate(
            "elem_lark_search",
            "搜索",
            editable=True,
            confidence=0.4,
            role="LarkSearchBox",
            x=220,
        ),
    ]
    return {
        "screenshot_path": "run/browser-shell-search.png",
        "candidates": candidates,
        "page_summary": {
            "top_texts": ["搜索"],
            "has_search_box": True,
            "has_modal": False,
            "candidate_count": len(candidates),
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": "browser_address_search",
                "reranked_candidate_ids": ["browser_address_search", "elem_lark_search"],
                "confidence": 0.92,
                "uncertain": False,
            }
        },
        "document_region_summary": {},
        "screen_meta": {},
    }


def chat_list_vs_plain_text_state() -> dict:
    candidates = [
        candidate(
            "elem_plain_chat_text",
            "测试群",
            confidence=0.99,
            role="Text",
            clickable=False,
            source="ocr",
        ),
        candidate(
            "elem_chat_row",
            "测试群",
            confidence=0.4,
            role="LarkChatListItem",
            x=240,
        ),
    ]
    return chat_ui_state("chat-disambiguation", candidates)


def similar_chat_candidates_state() -> dict:
    candidates = [
        candidate("elem_similar_chat", "测试群2", confidence=0.99, role="LarkChatListItem"),
        candidate("elem_target_chat", "测试群", confidence=0.3, role="LarkChatListItem", x=240),
    ]
    return chat_ui_state("similar-chats", candidates)


def open_chat_vlm_search_box_state() -> dict:
    candidates = [
        candidate(
            "elem_search",
            "搜索 (Ctrl+K)",
            editable=True,
            confidence=0.99,
            role="LarkSearchBox",
        ),
        candidate("elem_target_chat", "测试群", confidence=0.3, role="LarkChatListItem", x=240),
    ]
    return chat_ui_state("open-chat-vlm-search-box", candidates)


def message_input_disambiguation_state() -> dict:
    candidates = [
        candidate(
            "elem_plain_message_text",
            "输入消息",
            confidence=0.98,
            role="Text",
            clickable=False,
            source="ocr",
        ),
        candidate(
            "elem_message_input",
            "",
            editable=True,
            confidence=0.3,
            role="Edit",
            x=240,
        ),
    ]
    return {
        "screenshot_path": "run/message-input-disambiguation.png",
        "candidates": candidates,
        "page_summary": {
            "top_texts": ["输入消息"],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": len(candidates),
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": "elem_plain_message_text",
                "reranked_candidate_ids": ["elem_plain_message_text", "elem_message_input"],
                "confidence": 0.9,
                "uncertain": False,
            }
        },
        "document_region_summary": {},
        "screen_meta": {},
    }


def search_box_only_state() -> dict:
    return {
        "screenshot_path": "run/search-box-only.png",
        "candidates": [
            candidate(
                "elem_search",
                "搜索 (Ctrl+K)",
                editable=True,
                role="LarkSearchBox",
            )
        ],
        "page_summary": {
            "top_texts": ["搜索 (Ctrl+K)"],
            "has_search_box": True,
            "has_modal": False,
            "candidate_count": 1,
        },
        "vlm_result": {},
        "document_region_summary": {},
        "screen_meta": {},
    }


def chat_without_input_candidate_state() -> dict:
    return {
        "screenshot_path": "run/chat-without-input-candidate.png",
        "candidates": [
            candidate(
                "elem_chat_row",
                "测试群",
                role="LarkChatListItem",
                editable=False,
                x=900,
            )
        ],
        "page_summary": {
            "top_texts": ["消息", "测试群", "Aa@"],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": 1,
        },
        "vlm_result": {},
        "document_region_summary": {},
        "screen_meta": {
            "screenshot_size": [2582, 1550],
            "screen_origin": [3829, 542],
            "screen_resolution": [2560, 1600],
            "coordinate_space": "screen",
        },
    }


def wrong_chat_after_state() -> dict:
    return {
        "screenshot_path": "run/wrong-chat.png",
        "candidates": [candidate("elem_message_input", "", editable=True)],
        "page_summary": {
            "top_texts": ["其他群"],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": 1,
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": "elem_message_input",
                "reranked_candidate_ids": ["elem_message_input"],
                "confidence": 0.8,
                "uncertain": False,
            }
        },
        "document_region_summary": {},
        "screen_meta": {},
    }


def changed_without_keyword_state() -> dict:
    return {
        "screenshot_path": "run/changed-without-keyword.png",
        "candidates": [candidate("elem_other", "其他结果")],
        "page_summary": {
            "top_texts": ["其他结果"],
            "has_search_box": True,
            "has_modal": False,
            "candidate_count": 1,
        },
        "vlm_result": {"structured": {}},
        "document_region_summary": {},
        "screen_meta": {},
    }


def input_failed_state() -> dict:
    return {
        "screenshot_path": "run/input-failed.png",
        "candidates": [],
        "page_summary": {
            "top_texts": [],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": 0,
        },
        "vlm_result": {"structured": {}},
        "document_region_summary": {},
        "screen_meta": {},
    }


def coordinate_invalid_with_backup_state() -> dict:
    candidates = [
        candidate("elem_bad_search", "搜索", editable=True, confidence=0.99, x=3000),
        candidate("elem_valid_search", "搜索", editable=True, confidence=0.2, x=120),
    ]
    return {
        "screenshot_path": "run/coordinate-invalid.png",
        "candidates": candidates,
        "page_summary": {
            "top_texts": ["搜索"],
            "has_search_box": True,
            "has_modal": False,
            "candidate_count": len(candidates),
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": "elem_bad_search",
                "reranked_candidate_ids": ["elem_bad_search", "elem_valid_search"],
                "confidence": 0.9,
                "uncertain": False,
            }
        },
        "document_region_summary": {},
        "screen_meta": {"screenshot_size": [1920, 1080]},
    }


def absolute_screen_search_state() -> dict:
    return {
        "screenshot_path": "run/absolute-screen-search.png",
        "candidates": [
            {
                "id": "elem_abs_search",
                "text": "搜索 (Ctrl+K)",
                "bbox": [3853, 635, 4198, 691],
                "center": [4025, 663],
                "role": "LarkSearchBox",
                "clickable": True,
                "editable": True,
                "source": "merged",
                "confidence": 0.99,
            }
        ],
        "page_summary": {
            "top_texts": ["搜索 (Ctrl+K)"],
            "has_search_box": True,
            "has_modal": False,
            "candidate_count": 1,
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": None,
                "reranked_candidate_ids": [],
                "confidence": 0.0,
                "uncertain": True,
            }
        },
        "document_region_summary": {},
        "screen_meta": {
            "screenshot_size": [2582, 1550],
            "screen_origin": [3829, 542],
            "screen_resolution": [2560, 1600],
            "coordinate_space": "screen",
        },
    }


def semantic_search_state() -> dict:
    return {
        "screenshot_path": "run/semantic-search.png",
        "candidates": [],
        "page_summary": {
            "top_texts": [],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": 0,
            "semantic_summary": {
                "domain": "im",
                "view": "search",
                "signals": {"search_active": True},
            },
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": None,
                "reranked_candidate_ids": [],
                "confidence": 0.8,
                "uncertain": True,
                "page_semantics": {
                    "domain": "im",
                    "view": "search",
                    "signals": {"search_active": True},
                },
            }
        },
        "document_region_summary": {},
        "screen_meta": {},
    }


def chat_ui_state(name: str, candidates: list[dict]) -> dict:
    return {
        "screenshot_path": f"run/{name}.png",
        "candidates": candidates,
        "page_summary": {
            "top_texts": [str(candidate["text"]) for candidate in candidates if candidate.get("text")],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": len(candidates),
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": candidates[0]["id"] if candidates else None,
                "reranked_candidate_ids": [candidate["id"] for candidate in candidates],
                "confidence": 0.88,
                "uncertain": False,
            }
        },
        "document_region_summary": {},
        "screen_meta": {},
    }


def candidate(
    candidate_id: str,
    text: str,
    *,
    editable: bool = False,
    confidence: float = 0.9,
    x: int = 100,
    role: str | None = None,
    clickable: bool = True,
    source: str = "uia",
) -> dict:
    return {
        "id": candidate_id,
        "text": text,
        "bbox": [x, 20, x + 80, 60],
        "center": [x + 40, 40],
        "role": role or ("Edit" if editable else "Button"),
        "clickable": clickable,
        "editable": editable,
        "source": source,
        "confidence": confidence,
    }


if __name__ == "__main__":
    unittest.main()
