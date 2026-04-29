"""成员 B 可视化检查脚本。

使用场景：
这个脚本不是单元测试，而是给人看的控制台验证。它会把任务解析、
runtime state 更新、动作历史记录、4.24 最小 planner loop 和规则型决策
打印出来，并用 `[PASS]` 标明每一步检查成功。4.25 的检查会额外展示
fake A/C 闭环、step-level 成功判断、5 个单步操作和失败原因分类。
第二周期检查会展示 workflow.v2、runtime_state.v2、action_decision.v2、Planner Loop v2、
VLM/规则候选融合、多候选消歧、基础 fallback、step-level 成功判断、
文档真值优先通道、B/C 协议对齐和 run summary 失败原因汇总。

输出含义：
`[任务结构]` 展示自然语言任务被拆成什么 goal 和 steps。
`[写入 ui_state 后]` 展示成员 A 输出给 B 的页面状态被保存到了哪里。
`[最小 planner loop]` 展示 B 如何选择动作、调用 fake executor 并推进 state。
`[规则型决策]` 展示文本命中点击、输入框定位输入、返回/等待动作。
`[动作协议]` 展示 planner 输出字段稳定，并且传给 executor 时不包含 B 内部字段。

重要约定：
本脚本使用 fake `ui_state` 和 fake executor，不访问真实飞书窗口、不截图、
不移动鼠标。真实桌面验证应放到成员 A/C 的验收脚本里。

运行方式：
```powershell
python tests\testB\run_visual_checks.py
```
"""

from __future__ import annotations

# ruff: noqa: E402

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from executor import RunRecorder
from planner.action_protocol import (
    ACTION_DECISION_V2_VERSION,
    ACTION_DECISION_VERSION,
    ActionDecisionV2,
    action_decision_to_executor_payload,
    action_decision_v2_to_executor_payload,
)
from planner.mini_planner import (
    ACTION_CLICK,
    ACTION_HOTKEY,
    ACTION_TYPE,
    ACTION_WAIT,
    FAILURE_EXECUTOR_FAILED,
    FAILURE_PAGE_MISMATCH,
    FAILURE_TARGET_NOT_FOUND,
    MiniPlanner,
)
from planner.planner_loop_v2 import (
    FAILURE_CANDIDATE_CONFLICT,
    FAILURE_CANDIDATE_INVALID,
    FAILURE_COORDINATE_INVALID,
    FAILURE_PAGE_NOT_CHANGED,
    FAILURE_VALIDATION_FAILED,
    FAILURE_VLM_UNCERTAIN,
    PlannerLoopV2,
    plan_next_workflow_action,
    run_workflow_until_done,
)
from planner.runtime_state import (
    RUNTIME_STATE_V2_SCHEMA_VERSION,
    RuntimeStateError,
    RuntimeStatus,
    create_runtime_state,
    create_runtime_state_v2,
)
from planner.task_schema import parse_task
from planner.workflow_schema import (
    STEP_VERIFY_MESSAGE_SENT,
    build_im_message_workflow,
    build_sample_im_workflows,
)
from schemas import ActionResult


def main() -> int:
    print("\n=== CUA-Lark 成员B 可视化检查 ===\n")

    task = parse_task("打开聊天列表里的测试群")
    require(task.goal == "open_chat", "自然语言任务解析成功")
    print_task(task.to_dict())

    workflow = build_im_message_workflow(chat_name="测试群", message_text="Hello CUA-Lark")
    require(
        len(workflow.steps) == 6
        and workflow.steps[-1].step_type == STEP_VERIFY_MESSAGE_SENT
        and len(build_sample_im_workflows()) == 3,
        "第二周期 workflow schema 成功",
    )
    print_workflow(workflow.to_dict())

    runtime_v2 = create_runtime_state_v2(workflow)
    runtime_v2.update_ui_state(fake_second_cycle_ui_state())
    runtime_v2.record_action(
        {
            "version": ACTION_DECISION_V2_VERSION,
            "action_type": ACTION_CLICK,
            "step_name": workflow.steps[0].step_name,
            "target_candidate_id": "elem_search",
            "x": 110,
            "y": 26,
        },
        {"ok": True, "action_type": ACTION_CLICK, "error": None},
    )
    runtime_v2.record_conflict(
        rule_candidate_id="elem_search",
        vlm_candidate_id="elem_search_vlm",
        reason="visual check conflict sample",
        vlm_confidence=0.62,
    )
    require(
        runtime_v2.version == RUNTIME_STATE_V2_SCHEMA_VERSION
        and runtime_v2.current_step.step_name == workflow.steps[0].step_name
        and runtime_v2.current_vlm_result["structured"]["target_candidate_id"] == "elem_search"
        and runtime_v2.history[0].action_result["ok"] is True
        and runtime_v2.conflicts[0].vlm_candidate_id == "elem_search_vlm",
        "第二周期 runtime state v2 成功",
    )
    print_runtime_v2(runtime_v2.to_dict())

    action_v2 = ActionDecisionV2.click_candidate(
        step_name=workflow.steps[0].step_name,
        target_candidate_id="elem_search",
        x=110,
        y=26,
        reason="visual check action_decision.v2",
        expected_after_state={"has_search_box": True},
    )
    enter_v2 = ActionDecisionV2.press_enter(step_name=workflow.steps[4].step_name)
    scroll_v2 = ActionDecisionV2.scroll(step_name="scroll_results", direction="down", amount=-5)
    require(
        action_v2.to_dict()["version"] == ACTION_DECISION_V2_VERSION
        and action_v2.to_dict()["target_candidate_id"] == "elem_search"
        and enter_v2.to_dict()["keys"] == ["enter"]
        and scroll_v2.to_dict()["action_type"] == "scroll",
        "第二周期 ActionDecision v2 成功",
    )
    print_action_v2(action_v2.to_dict(), enter_v2.to_dict(), scroll_v2.to_dict())

    loop_v2_state = create_runtime_state_v2(workflow)
    before_provider = SequenceProvider(fake_workflow_before_states())
    after_provider = SequenceProvider(fake_workflow_after_states())
    loop_v2_results = run_workflow_until_done(
        workflow,
        before_provider.next,
        fake_quiet_execute_action,
        loop_v2_state,
        get_after_ui_state=after_provider.next,
        max_steps=10,
    )
    require(
        loop_v2_state.status == RuntimeStatus.SUCCESS.value
        and loop_v2_state.step_idx == len(workflow.steps)
        and len(loop_v2_state.history) >= 6
        and any(
            (item.action_decision or {}).get("keys") == ["enter"]
            for item in loop_v2_state.history
        ),
        "第二周期 Planner Loop v2 成功",
    )
    print_planner_loop_v2(loop_v2_state.to_dict(), [item.to_dict() for item in loop_v2_results])

    require(check_second_cycle_vlm_rule_fusion(), "第二周期 VLM 与规则融合成功")
    require(check_second_cycle_candidate_disambiguation(), "第二周期多候选消歧成功")
    require(check_second_cycle_fallback_and_step_validation(), "第二周期基础 fallback 与 step 成功判断成功")
    require(check_second_cycle_document_truth_priority(), "第二周期文档真值优先通道成功")
    require(check_second_cycle_bc_contracts(), "第二周期 B/C 协议对齐成功")
    require(check_run_summary_failure_reason(), "第二周期 run summary 失败原因记录成功")

    state = create_runtime_state(task)
    require(state.status == RuntimeStatus.RUNNING.value, "初始 RuntimeState 创建成功")
    print_state_summary(state.to_dict())

    ui_state = {
        "screenshot_path": "artifacts/runs/run_001/screenshots/step_001.png",
        "page_summary": {
            "top_texts": ["搜索", "测试群", "文档"],
            "has_search_box": True,
            "has_modal": False,
            "candidate_count": 2,
        },
        "candidates": [
            {"id": "elem_001", "text": "测试群", "role": "list_item"},
            {"id": "elem_002", "text": "搜索", "role": "textbox"},
        ],
    }
    state.update_ui_state(ui_state)
    require(state.page_summary["candidate_count"] == 2, "ui_state 写入 RuntimeState 成功")
    print_ui_state(state.to_dict())

    action = {
        "action_type": "click",
        "target_element_id": "elem_001",
        "x": 412,
        "y": 238,
        "reason": "matched target_text=测试群",
    }
    action_result = {"ok": True, "action_type": "click", "error": None}
    state.record_step(action, action_result, advance_step=True)
    require(state.step_idx == 1 and len(state.history) == 1, "动作历史记录成功")
    print_history(state.to_dict())

    try:
        state.record_step({"action_type": "click"}, status="unknown")
    except RuntimeStateError as exc:
        require("Unsupported runtime status" in str(exc), "非法 runtime status 拦截成功")
    else:
        raise AssertionError("非法 runtime status 没有被拦截")

    planner = MiniPlanner()
    loop_state = create_runtime_state("点击文档入口")
    loop_result = planner.run_once(
        "点击文档入口",
        fake_get_ui_state,
        fake_execute_action,
        loop_state,
    )
    require(loop_result.decision["action_type"] == ACTION_CLICK, "规则型文本点击成功")
    require(loop_result.runtime_state.status == RuntimeStatus.SUCCESS.value, "最小 planner loop 更新状态成功")
    require(_looks_like_a_ui_state(fake_get_ui_state()), "接入 A 感知输出成功")
    print_planner_loop(loop_result.to_dict())
    protocol_payload = action_decision_to_executor_payload(loop_result.decision)
    require(
        loop_result.decision["version"] == ACTION_DECISION_VERSION
        and protocol_payload["action_type"] == ACTION_CLICK
        and protocol_payload["target_element_id"] == "elem_docs"
        and "planner_meta" not in protocol_payload,
        "动作协议字段统一成功",
    )
    print_action_protocol(loop_result.decision, protocol_payload)

    input_state = create_runtime_state("在搜索框输入项目周报")
    focus_result = planner.run_once(
        "在搜索框输入项目周报",
        fake_input_only_ui_state,
        fake_execute_action,
        input_state,
    )
    type_result = planner.run_once(
        "在搜索框输入项目周报",
        fake_input_only_ui_state,
        fake_execute_action,
        input_state,
    )
    require(
        focus_result.decision["action_type"] == ACTION_CLICK
        and focus_result.decision["target_element_id"] == "elem_input"
        and type_result.decision["action_type"] == ACTION_TYPE
        and type_result.decision["text"] == "项目周报",
        "输入框定位型输入成功",
    )
    print_rule_decision("输入框定位型输入", focus_result.decision)
    print_rule_decision("输入文本动作", type_result.decision)

    validated_state = create_runtime_state("在搜索框输入项目周报")
    planner.run_once(
        "在搜索框输入项目周报",
        fake_input_only_ui_state,
        fake_execute_action,
        validated_state,
        get_after_ui_state=fake_input_only_ui_state,
    )
    validated_type = planner.run_once(
        "在搜索框输入项目周报",
        fake_input_only_ui_state,
        fake_execute_action,
        validated_state,
        get_after_ui_state=lambda: fake_text_ui_state("项目周报", editable=True),
    )
    require(
        validated_type.runtime_state.status == RuntimeStatus.SUCCESS.value
        and validated_type.runtime_state.page_summary["top_texts"] == ["项目周报"],
        "step-level 成功判断成功",
    )

    back_decision = planner.plan_next_action(
        "返回上一级",
        fake_get_ui_state(),
        create_runtime_state("返回上一级"),
    )
    wait_state = create_runtime_state(
        {
            "goal": "wait_demo",
            "steps": [{"action_hint": "wait", "params": {"seconds": 2}}],
        }
    )
    wait_decision = planner.plan_next_action(wait_state.task, fake_get_ui_state(), wait_state)
    require(
        back_decision["action_type"] == ACTION_HOTKEY
        and wait_decision["action_type"] == ACTION_WAIT,
        "返回/等待类动作成功",
    )
    print_rule_decision("返回动作", back_decision)
    print_rule_decision("等待动作", wait_decision)

    require(run_five_single_step_scenarios(planner), "5 个单步操作 planner 闭环成功")
    require(check_failure_reasons(planner), "失败原因分类成功")

    print("\n=== ALL VISUAL CHECKS PASSED ===")
    print(
        "成员B 4.23、4.24 第 3/4/5 项、4.25 第 6/7/8/9 项，"
        "以及第二周期 2.1-2.8 的核心数据链路都能正常工作。\n"
    )
    return 0


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"[FAIL] {message}")
        raise AssertionError(message)
    print(f"[PASS] {message}")


def _looks_like_a_ui_state(value: Mapping[str, Any]) -> bool:
    return (
        isinstance(value.get("screenshot_path"), str)
        and isinstance(value.get("candidates"), list)
        and isinstance(value.get("page_summary"), Mapping)
    )


def run_five_single_step_scenarios(planner: MiniPlanner) -> bool:
    sidebar_state = create_runtime_state(
        {
            "goal": "open_sidebar_module",
            "steps": [{"action_hint": "find_sidebar_module", "target_text": "日历"}],
        }
    )
    sidebar = planner.run_once(
        sidebar_state.task,
        lambda: fake_text_ui_state("日历"),
        fake_execute_action,
        sidebar_state,
        get_after_ui_state=lambda: fake_text_ui_state("日历"),
    )

    search_state = create_runtime_state("在搜索框输入项目周报")
    planner.run_once(
        search_state.task,
        fake_input_only_ui_state,
        fake_execute_action,
        search_state,
        get_after_ui_state=fake_input_only_ui_state,
    )
    search = planner.run_once(
        search_state.task,
        fake_input_only_ui_state,
        fake_execute_action,
        search_state,
        get_after_ui_state=lambda: fake_text_ui_state("项目周报", editable=True),
    )

    chat = planner.run_once(
        "打开聊天列表里的测试群",
        lambda: fake_text_ui_state("测试群"),
        fake_execute_action,
        create_runtime_state("打开聊天列表里的测试群"),
        get_after_ui_state=lambda: fake_text_ui_state("测试群"),
    )
    docs = planner.run_once(
        "点击文档入口",
        lambda: fake_text_ui_state("文档"),
        fake_execute_action,
        create_runtime_state("点击文档入口"),
        get_after_ui_state=lambda: fake_text_ui_state("新建文档"),
    )
    back = planner.run_once(
        "返回上一级",
        lambda: fake_text_ui_state("文档"),
        fake_execute_action,
        create_runtime_state("返回上一级"),
        get_after_ui_state=lambda: fake_text_ui_state("聊天"),
    )

    return all(
        result.runtime_state.status == RuntimeStatus.SUCCESS.value
        for result in (sidebar, search, chat, docs, back)
    )


def check_failure_reasons(planner: MiniPlanner) -> bool:
    missing = planner.run_once(
        "点击文档入口",
        lambda: fake_text_ui_state("通讯录"),
        fake_execute_action,
        create_runtime_state("点击文档入口"),
    )

    mismatch_state = create_runtime_state("在搜索框输入项目周报")
    mismatch_state.step_idx = 1
    mismatch = planner.run_once(
        "在搜索框输入项目周报",
        fake_input_only_ui_state,
        fake_execute_action,
        mismatch_state,
        get_after_ui_state=fake_input_only_ui_state,
    )

    executor_failed = planner.run_once(
        "点击文档入口",
        lambda: fake_text_ui_state("文档"),
        fake_failed_executor,
        create_runtime_state("点击文档入口"),
    )

    return (
        missing.runtime_state.failure_reason == FAILURE_TARGET_NOT_FOUND
        and mismatch.runtime_state.failure_reason == FAILURE_PAGE_MISMATCH
        and executor_failed.runtime_state.failure_reason == FAILURE_EXECUTOR_FAILED
    )


def check_second_cycle_vlm_rule_fusion() -> bool:
    workflow = build_im_message_workflow(chat_name="team", message_text="Hello CUA-Lark")
    conflict_state = create_runtime_state_v2(workflow)
    conflict_decision = plan_next_workflow_action(conflict_state, fake_vlm_conflict_state())

    uncertain_state = create_runtime_state_v2(workflow)
    uncertain_decision = plan_next_workflow_action(uncertain_state, fake_vlm_uncertain_state())

    invalid_state = create_runtime_state_v2(workflow)
    invalid_state.step_idx = 2
    invalid_decision = plan_next_workflow_action(invalid_state, fake_invalid_candidate_state())

    document_state = create_runtime_state_v2(workflow)
    document_state.step_idx = 5
    document_decision = plan_next_workflow_action(document_state, fake_document_summary_state())

    ocr_state = create_runtime_state_v2(workflow)
    ocr_state.step_idx = 5
    ocr_decision = plan_next_workflow_action(ocr_state, fake_ocr_only_state())

    print("\n[第二周期 VLM 与规则融合]")
    print(f"high_confidence_decision: {to_json(conflict_decision)}")
    print(f"uncertain_fallback_decision: {to_json(uncertain_decision)}")
    print(f"invalid_candidate_decision: {to_json(invalid_decision)}")
    print(f"document_summary_decision: {to_json(document_decision)}")
    print(f"ocr_only_decision: {to_json(ocr_decision)}")
    print(f"conflicts: {to_json([item.to_dict() for item in conflict_state.conflicts])}")

    return (
        conflict_decision["target_candidate_id"] == "elem_vlm_search"
        and conflict_state.conflicts[0].reason == FAILURE_CANDIDATE_CONFLICT
        and uncertain_decision["target_candidate_id"] == "elem_rule_search"
        and uncertain_state.conflicts[0].reason == FAILURE_VLM_UNCERTAIN
        and invalid_decision["failure_reason"] == FAILURE_CANDIDATE_INVALID
        and invalid_state.conflicts[0].reason == FAILURE_CANDIDATE_INVALID
        and document_decision["status"] == "success"
        and ocr_decision["action_type"] == ACTION_WAIT
    )


def check_second_cycle_document_truth_priority() -> bool:
    workflow = build_im_message_workflow(chat_name="team", message_text="Hello CUA-Lark")

    truth_state = create_runtime_state_v2(workflow)
    truth_state.step_idx = 5
    truth_decision = plan_next_workflow_action(truth_state, fake_document_truth_state())

    conflict_state = create_runtime_state_v2(workflow)
    conflict_state.step_idx = 5
    conflict_decision = plan_next_workflow_action(conflict_state, fake_conflicting_document_truth_state())

    print("\n[第二周期文档真值优先通道]")
    print(f"truth_decision: {to_json(truth_decision)}")
    print(f"conflict_decision: {to_json(conflict_decision)}")

    return truth_decision["status"] == "success" and conflict_decision["action_type"] == ACTION_WAIT


def check_second_cycle_bc_contracts() -> bool:
    workflow = build_im_message_workflow(chat_name="测试群", message_text="Hello CUA-Lark")
    decision = ActionDecisionV2.click_candidate(
        step_name="open_chat",
        target_candidate_id="candidate_chat_001",
        x=320,
        y=240,
        reason="visual check B/C contract",
        retry_count=1,
        expected_after_state={"chat_name_visible": "测试群"},
    ).to_dict()
    executor_payload = action_decision_v2_to_executor_payload(decision)
    action_result = ActionResult.success(
        action_type=ACTION_CLICK,
        start_ts="2026-04-24T10:00:00.000Z",
        end_ts="2026-04-24T10:00:00.120Z",
        target_candidate_id="candidate_chat_001",
        planned_click_point=[320, 240],
        actual_click_point=[322, 241],
        before_screenshot={"screenshot_path": "before.png"},
        after_screenshot={"screenshot_path": "after.png"},
        screen_meta_snapshot={"screenshot_size": [1920, 1080]},
    ).to_dict()

    validator_state = create_runtime_state_v2(workflow)
    validator_state.step_idx = 4
    PlannerLoopV2().run_once(
        workflow,
        fake_message_input_state,
        fake_validator_passing_execute_action,
        validator_state,
        get_after_ui_state=fake_message_input_state,
    )

    print("\n[第二周期 B/C 协议对齐]")
    print(f"action_decision_v2: {to_json(decision)}")
    print(f"executor_payload: {to_json(executor_payload)}")
    print(f"action_result_v2: {to_json(action_result)}")
    print(f"validator_history: {to_json(validator_state.history[-1].to_dict())}")

    return (
        executor_payload["target_element_id"] == "candidate_chat_001"
        and "step_name" not in executor_payload
        and action_result["click_offset"] == [2, 1]
        and validator_state.step_idx == 5
        and validator_state.history[-1].action_result["validator_result"]["passed"] is True
    )


def check_run_summary_failure_reason() -> bool:
    artifact_root = ROOT / "artifacts" / "tmp" / "testB_visual_checks"
    shutil.rmtree(artifact_root, ignore_errors=True)
    try:
        recorder = RunRecorder(
            run_id="visual_failure_summary",
            artifact_root=artifact_root,
        )
        summary = recorder.record_step(
            step_idx=1,
            step_name="verify_message_sent",
            action_decision={
                "action_type": ACTION_TYPE,
                "step_name": "verify_message_sent",
                "reason": "visual validator failed",
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
        print("\n[第二周期 run summary 失败原因]")
        print(f"summary: {to_json(summary)}")
        return (
            summary["failure_reasons"] == {FAILURE_VALIDATION_FAILED: 1}
            and summary["validator_failure_count"] == 1
        )
    finally:
        shutil.rmtree(artifact_root, ignore_errors=True)


def check_second_cycle_candidate_disambiguation() -> bool:
    workflow = build_im_message_workflow(chat_name="测试群", message_text="Hello CUA-Lark")

    search_state = create_runtime_state_v2(workflow)
    search_decision = plan_next_workflow_action(search_state, fake_browser_shell_search_state())

    chat_state = create_runtime_state_v2(workflow)
    chat_state.step_idx = 2
    chat_decision = plan_next_workflow_action(chat_state, fake_chat_disambiguation_state())

    similar_state = create_runtime_state_v2(workflow)
    similar_state.step_idx = 2
    similar_decision = plan_next_workflow_action(similar_state, fake_similar_chat_state())

    input_state = create_runtime_state_v2(workflow)
    input_state.step_idx = 3
    input_decision = plan_next_workflow_action(input_state, fake_message_input_disambiguation_state())

    retry_state = create_runtime_state_v2(workflow)
    retry_state.step_idx = 2
    retry_result = PlannerLoopV2().run_once(
        workflow,
        fake_chat_candidate_state,
        fake_quiet_execute_action,
        retry_state,
        get_after_ui_state=fake_wrong_chat_after_state,
    )

    docs_decision = MiniPlanner().plan_next_action(
        "点击文档入口",
        fake_docs_entry_disambiguation_state(),
        create_runtime_state("点击文档入口"),
    )

    print("\n[第二周期多候选消歧]")
    print(f"search_decision: {to_json(search_decision)}")
    print(f"chat_decision: {to_json(chat_decision)}")
    print(f"similar_chat_decision: {to_json(similar_decision)}")
    print(f"message_input_decision: {to_json(input_decision)}")
    print(f"wrong_chat_retry_status: {retry_result.runtime_state.status}")
    print(f"docs_entry_decision: {to_json(docs_decision)}")

    return (
        search_decision["target_candidate_id"] == "elem_lark_search"
        and chat_decision["target_candidate_id"] == "elem_chat_row"
        and similar_decision["target_candidate_id"] == "elem_target_chat"
        and input_decision["target_candidate_id"] == "elem_message_input"
        and retry_result.runtime_state.status == RuntimeStatus.RUNNING.value
        and retry_result.runtime_state.failure_reason == "page_mismatch"
        and docs_decision["target_element_id"] == "elem_docs_entry"
    )


def check_second_cycle_fallback_and_step_validation() -> bool:
    workflow = build_im_message_workflow(chat_name="测试群", message_text="Hello CUA-Lark")
    loop = PlannerLoopV2()

    missing_chat_state = create_runtime_state_v2(workflow)
    missing_chat_state.step_idx = 2
    missing_result = loop.run_once(
        workflow,
        fake_no_chat_result_state,
        fake_quiet_execute_action,
        missing_chat_state,
    )
    retry_search_decision = plan_next_workflow_action(missing_chat_state, fake_no_chat_result_state())

    unchanged_state = create_runtime_state_v2(workflow)
    unchanged_state.step_idx = 2
    unchanged_result = loop.run_once(
        workflow,
        fake_chat_candidate_state,
        fake_quiet_execute_action,
        unchanged_state,
        get_after_ui_state=fake_chat_candidate_state,
    )
    wait_decision = plan_next_workflow_action(unchanged_state, fake_chat_candidate_state())

    wrong_page_state = create_runtime_state_v2(workflow)
    wrong_page_state.step_idx = 2
    wrong_page_state.retry_count = 1
    wrong_page_state.failure_reason = FAILURE_PAGE_MISMATCH
    back_decision = plan_next_workflow_action(wrong_page_state, fake_wrong_chat_after_state())

    coordinate_state = create_runtime_state_v2(workflow)
    coordinate_decision = plan_next_workflow_action(coordinate_state, fake_coordinate_invalid_backup_state())

    type_state = create_runtime_state_v2(workflow)
    type_state.step_idx = 1
    type_failed = loop.run_once(
        workflow,
        fake_search_box_state,
        fake_quiet_execute_action,
        type_state,
        get_after_ui_state=fake_input_failed_state,
    )
    refocus_decision = plan_next_workflow_action(type_state, fake_search_box_state())

    semantic_state = create_runtime_state_v2(workflow)
    semantic_result = loop.run_once(
        workflow,
        fake_no_chat_result_state,
        fake_quiet_execute_action,
        semantic_state,
        get_after_ui_state=fake_semantic_search_state,
    )

    validator_state = create_runtime_state_v2(workflow)
    validator_state.step_idx = 4
    validator_state.record_action(
        {
            "status": "continue",
            "action_type": ACTION_TYPE,
            "step_name": "send_message",
            "text": "Hello CUA-Lark",
        },
        {"ok": True, "action_type": ACTION_TYPE},
    )
    validator_result = loop.run_once(
        workflow,
        fake_message_input_with_text_state,
        fake_validator_success_executor,
        validator_state,
        get_after_ui_state=fake_input_failed_state,
    )

    print("\n[第二周期 fallback 与 step 成功判断]")
    print(f"missing_chat_failure: {missing_result.runtime_state.failure_reason}")
    print(f"retry_search_decision: {to_json(retry_search_decision)}")
    print(f"page_not_changed_failure: {unchanged_result.runtime_state.failure_reason}")
    print(f"wait_decision: {to_json(wait_decision)}")
    print(f"wrong_page_back_decision: {to_json(back_decision)}")
    print(f"coordinate_decision: {to_json(coordinate_decision)}")
    print(f"type_validation_failure: {type_failed.runtime_state.failure_reason}")
    print(f"refocus_decision: {to_json(refocus_decision)}")
    print(f"semantic_open_search_step_idx: {semantic_result.runtime_state.step_idx}")
    print(f"validator_send_message_step_idx: {validator_result.runtime_state.step_idx}")

    return (
        missing_result.runtime_state.failure_reason == FAILURE_TARGET_NOT_FOUND
        and retry_search_decision["keys"] == ["ctrl", "k"]
        and unchanged_result.runtime_state.failure_reason == FAILURE_PAGE_NOT_CHANGED
        and wait_decision["action_type"] == ACTION_WAIT
        and back_decision["keys"] == ["alt", "left"]
        and coordinate_decision["target_candidate_id"] == "elem_valid_search"
        and coordinate_state.conflicts[0].reason == FAILURE_COORDINATE_INVALID
        and type_failed.runtime_state.failure_reason == FAILURE_VALIDATION_FAILED
        and refocus_decision["action_type"] == ACTION_CLICK
        and semantic_result.runtime_state.step_idx == 1
        and validator_result.runtime_state.step_idx == 5
    )


def print_task(task: dict[str, Any]) -> None:
    print("\n[任务结构]")
    print(f"task_id: {task['task_id']}")
    print(f"instruction: {task['instruction']}")
    print(f"goal: {task['goal']}")
    print(f"params: {to_json(task['params'])}")
    print("steps:")
    for index, step in enumerate(task["steps"], start=1):
        print(
            f"  {index}. action_hint={step['action_hint']}, "
            f"target_text={step['target_text']}, input_text={step['input_text']}"
        )


def print_workflow(workflow: dict[str, Any]) -> None:
    print("\n[第二周期 workflow.v2]")
    print(f"workflow_id: {workflow['workflow_id']}")
    print(f"instruction: {workflow['instruction']}")
    print(f"workflow_type: {workflow['workflow_type']}")
    print(f"params: {to_json(workflow['params'])}")
    print("steps:")
    for index, step in enumerate(workflow["steps"], start=1):
        print(
            f"  {index}. step_name={step['step_name']}, "
            f"step_type={step['step_type']}, target={step['target']}, "
            f"max_retry={step['max_retry']}"
        )


def print_runtime_v2(state: dict[str, Any]) -> None:
    print("\n[第二周期 runtime_state.v2]")
    print(f"version: {state['version']}")
    print(f"step_idx: {state['step_idx']}")
    print(f"current_step: {state['current_step']['step_name']}")
    print(f"page_summary: {to_json(state['current_page_summary'])}")
    print(f"vlm_result: {to_json(state['current_vlm_result'])}")
    print(f"screen_meta: {to_json(state['current_screen_meta'])}")
    print(f"history_length: {len(state['history'])}")
    print(f"conflicts: {to_json(state['conflicts'])}")


def print_action_v2(
    click_decision: Mapping[str, Any],
    enter_decision: Mapping[str, Any],
    scroll_decision: Mapping[str, Any],
) -> None:
    print("\n[第二周期 action_decision.v2]")
    print(f"click_candidate: {to_json(dict(click_decision))}")
    print(f"press_enter: {to_json(dict(enter_decision))}")
    print(f"scroll: {to_json(dict(scroll_decision))}")


def print_planner_loop_v2(state: dict[str, Any], results: list[dict[str, Any]]) -> None:
    print("\n[第二周期 Planner Loop v2]")
    print(f"status: {state['status']}")
    print(f"step_idx: {state['step_idx']}")
    print(f"history_length: {len(state['history'])}")
    print(f"result_count: {len(results)}")
    for item in state["history"]:
        decision = item.get("action_decision") or {}
        print(
            "  - "
            f"step={item.get('step_name')}, "
            f"action={decision.get('action_type')}, "
            f"status={item.get('status')}, "
            f"retry={item.get('retry_count')}"
        )


def print_state_summary(state: dict[str, Any]) -> None:
    print("\n[初始运行状态]")
    print(f"status: {state['status']}")
    print(f"step_idx: {state['step_idx']}")
    print(f"failure_reason: {state['failure_reason']}")
    print(f"history_length: {len(state['history'])}")


def print_ui_state(state: dict[str, Any]) -> None:
    print("\n[写入 ui_state 后]")
    print(f"page_summary: {to_json(state['page_summary'])}")
    print("candidates:")
    for candidate in state["candidates"]:
        print(
            f"  - id={candidate.get('id')}, "
            f"text={candidate.get('text')}, role={candidate.get('role')}"
        )


def print_history(state: dict[str, Any]) -> None:
    print("\n[记录动作后]")
    print(f"step_idx: {state['step_idx']}")
    print(f"last_action: {to_json(state['last_action'])}")
    print(f"history_length: {len(state['history'])}")
    print(f"history[0]: {to_json(state['history'][0])}")


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class SequenceProvider:
    def __init__(self, values: list[dict[str, Any]]) -> None:
        self.values = list(values)

    def next(self) -> dict[str, Any]:
        if len(self.values) == 1:
            return self.values[0]
        return self.values.pop(0)


def fake_workflow_before_states() -> list[dict[str, Any]]:
    return [
        fake_second_cycle_ui_state(),
        fake_search_box_state(),
        fake_chat_candidate_state(),
        fake_message_input_state(),
        fake_message_input_state(),
        fake_message_input_with_text_state(),
        fake_message_visible_state(),
    ]


def fake_workflow_after_states() -> list[dict[str, Any]]:
    return [
        fake_search_box_state(),
        fake_chat_candidate_state(),
        fake_chat_open_state(),
        fake_message_input_state(),
        fake_message_input_with_text_state(),
        fake_message_visible_state(),
    ]


def fake_second_cycle_ui_state() -> dict[str, Any]:
    return {
        "screenshot_path": "artifacts/runs/run_002/screenshots/step_001.png",
        "page_summary": {
            "top_texts": ["搜索", "测试群"],
            "has_search_box": True,
            "candidate_count": 1,
        },
        "candidates": [
            {
                "id": "elem_search",
                "text": "搜索",
                "bbox": [10, 10, 210, 42],
                "center": [110, 26],
                "role": "Edit",
                "clickable": True,
                "editable": True,
            }
        ],
        "vlm_result": {
            "structured": {
                "target_candidate_id": "elem_search",
                "reranked_candidate_ids": ["elem_search"],
                "confidence": 0.88,
                "uncertain": False,
            }
        },
        "document_region_summary": {"visible_title": "无"},
        "screen_meta": {"width": 2560, "height": 1440, "dpi_scale": 1.25},
    }


def fake_vlm_conflict_state() -> dict[str, Any]:
    candidates = [
        fake_candidate("elem_rule_search", "搜索", editable=True),
        fake_candidate("elem_vlm_search", "搜索", editable=True),
    ]
    candidates[0]["confidence"] = 0.95
    candidates[1]["confidence"] = 0.2
    return _fake_vlm_state(
        "vlm_conflict",
        candidates,
        ["搜索"],
        target_candidate_id="elem_vlm_search",
        reranked_candidate_ids=["elem_vlm_search", "elem_rule_search"],
        confidence=0.92,
        uncertain=False,
        has_search_box=True,
    )


def fake_vlm_uncertain_state() -> dict[str, Any]:
    return _fake_vlm_state(
        "vlm_uncertain",
        [fake_candidate("elem_rule_search", "搜索", editable=True)],
        ["搜索"],
        target_candidate_id=None,
        reranked_candidate_ids=[],
        confidence=0.2,
        uncertain=True,
        has_search_box=True,
    )


def fake_invalid_candidate_state() -> dict[str, Any]:
    return _fake_vlm_state(
        "invalid_candidate",
        [
            {
                "id": "elem_invalid_team",
                "text": "team",
                "bbox": [120, 30, 110, 70],
                "role": "Button",
                "clickable": True,
                "editable": False,
                "source": "uia",
                "confidence": 0.95,
            }
        ],
        ["team"],
        target_candidate_id="elem_invalid_team",
        reranked_candidate_ids=["elem_invalid_team"],
        confidence=0.94,
        uncertain=False,
    )


def fake_document_summary_state() -> dict[str, Any]:
    state = _fake_vlm_state(
        "document_summary",
        [],
        [],
        target_candidate_id=None,
        reranked_candidate_ids=[],
        confidence=0.7,
        uncertain=True,
    )
    state["page_summary"]["semantic_summary"] = {"summary": "message visible"}
    state["vlm_result"]["structured"]["page_semantics"] = {
        "domain": "docs",
        "summary": "Hello CUA-Lark appears in document area",
    }
    state["document_region_summary"] = {"visible_texts": ["Hello CUA-Lark"]}
    return state


def fake_document_truth_state() -> dict[str, Any]:
    state = fake_document_summary_state()
    state["page_summary"]["semantic_summary"] = {}
    state["vlm_result"]["structured"]["page_semantics"] = {}
    state["document_region_summary"] = {}
    state["document_truth"] = {"source": "mcp", "content": "Hello CUA-Lark"}
    return state


def fake_conflicting_document_truth_state() -> dict[str, Any]:
    state = fake_document_summary_state()
    state["document_truth"] = {
        "source": "openapi",
        "content": "Authoritative document content does not include the expected message.",
    }
    return state


def fake_ocr_only_state() -> dict[str, Any]:
    return _fake_vlm_state(
        "ocr_only",
        [
            {
                "id": "ocr_body",
                "text": "Hello CUA-Lark",
                "bbox": [10, 10, 500, 500],
                "center": [255, 255],
                "role": "Text",
                "clickable": False,
                "editable": False,
                "source": "ocr",
                "confidence": 0.8,
            }
        ],
        [],
        target_candidate_id=None,
        reranked_candidate_ids=[],
        confidence=0.1,
        uncertain=True,
    )


def fake_browser_shell_search_state() -> dict[str, Any]:
    candidates = [
        fake_candidate(
            "browser_address_search",
            "搜索或输入网址",
            editable=True,
            role="Edit",
            confidence=0.99,
        ),
        fake_candidate(
            "elem_lark_search",
            "搜索",
            editable=True,
            role="LarkSearchBox",
            confidence=0.4,
        ),
    ]
    return _fake_vlm_state(
        "browser_shell_search",
        candidates,
        ["搜索"],
        target_candidate_id="browser_address_search",
        reranked_candidate_ids=["browser_address_search", "elem_lark_search"],
        confidence=0.92,
        uncertain=False,
        has_search_box=True,
    )


def fake_chat_disambiguation_state() -> dict[str, Any]:
    return _fake_vlm_state(
        "chat_disambiguation",
        [
            fake_candidate(
                "elem_plain_chat_text",
                "测试群",
                role="Text",
                clickable=False,
                source="ocr",
                confidence=0.99,
            ),
            fake_candidate(
                "elem_chat_row",
                "测试群",
                role="LarkChatListItem",
                confidence=0.4,
            ),
        ],
        ["测试群"],
        target_candidate_id="elem_plain_chat_text",
        reranked_candidate_ids=["elem_plain_chat_text", "elem_chat_row"],
        confidence=0.9,
        uncertain=False,
    )


def fake_similar_chat_state() -> dict[str, Any]:
    return _fake_vlm_state(
        "similar_chat",
        [
            fake_candidate("elem_similar_chat", "测试群2", role="LarkChatListItem", confidence=0.99),
            fake_candidate("elem_target_chat", "测试群", role="LarkChatListItem", confidence=0.3),
        ],
        ["测试群2", "测试群"],
        target_candidate_id="elem_similar_chat",
        reranked_candidate_ids=["elem_similar_chat", "elem_target_chat"],
        confidence=0.9,
        uncertain=False,
    )


def fake_message_input_disambiguation_state() -> dict[str, Any]:
    return _fake_vlm_state(
        "message_input_disambiguation",
        [
            fake_candidate(
                "elem_plain_message_text",
                "输入消息",
                role="Text",
                clickable=False,
                source="ocr",
                confidence=0.98,
            ),
            fake_candidate("elem_message_input", "", editable=True, role="Edit", confidence=0.3),
        ],
        ["输入消息"],
        target_candidate_id="elem_plain_message_text",
        reranked_candidate_ids=["elem_plain_message_text", "elem_message_input"],
        confidence=0.9,
        uncertain=False,
    )


def fake_wrong_chat_after_state() -> dict[str, Any]:
    return _fake_vlm_state(
        "wrong_chat",
        [fake_candidate("elem_message_input", "", editable=True)],
        ["其他群"],
        target_candidate_id="elem_message_input",
        reranked_candidate_ids=["elem_message_input"],
        confidence=0.8,
        uncertain=False,
    )


def fake_no_chat_result_state() -> dict[str, Any]:
    candidates = [fake_candidate("elem_other", "其他结果", role="LarkChatListItem")]
    return {
        "screenshot_path": "artifacts/runs/run_002/screenshots/no_chat_result.png",
        "candidates": candidates,
        "page_summary": {
            "top_texts": ["其他结果"],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": len(candidates),
        },
        "vlm_result": {},
        "document_region_summary": {},
        "screen_meta": {"width": 2560, "height": 1440, "dpi_scale": 1.25},
    }


def fake_coordinate_invalid_backup_state() -> dict[str, Any]:
    candidates = [
        {
            "id": "elem_bad_search",
            "text": "搜索",
            "bbox": [3000, 20, 3100, 60],
            "center": [3050, 40],
            "role": "LarkSearchBox",
            "clickable": True,
            "editable": True,
            "source": "uia",
            "confidence": 0.99,
        },
        {
            "id": "elem_valid_search",
            "text": "搜索",
            "bbox": [80, 20, 180, 60],
            "center": [130, 40],
            "role": "LarkSearchBox",
            "clickable": True,
            "editable": True,
            "source": "uia",
            "confidence": 0.2,
        },
    ]
    state = _fake_vlm_state(
        "coordinate_invalid_backup",
        candidates,
        ["搜索"],
        target_candidate_id="elem_bad_search",
        reranked_candidate_ids=["elem_bad_search", "elem_valid_search"],
        confidence=0.92,
        uncertain=False,
        has_search_box=True,
    )
    state["screen_meta"] = {"screenshot_size": [1920, 1080]}
    return state


def fake_input_failed_state() -> dict[str, Any]:
    return _fake_vlm_state(
        "input_failed",
        [],
        [],
        target_candidate_id=None,
        reranked_candidate_ids=[],
        confidence=0.1,
        uncertain=True,
    )


def fake_semantic_search_state() -> dict[str, Any]:
    state = _fake_vlm_state(
        "semantic_search",
        [],
        [],
        target_candidate_id=None,
        reranked_candidate_ids=[],
        confidence=0.8,
        uncertain=True,
    )
    state["page_summary"]["semantic_summary"] = {
        "domain": "im",
        "view": "search",
        "signals": {"search_active": True},
    }
    state["vlm_result"]["structured"]["page_semantics"] = {
        "domain": "im",
        "view": "search",
        "signals": {"search_active": True},
    }
    return state


def fake_docs_entry_disambiguation_state() -> dict[str, Any]:
    candidates = [
        {
            "id": "elem_existing_doc",
            "text": "文档",
            "bbox": [380, 200, 760, 252],
            "center": [570, 226],
            "role": "LarkDocumentListItem",
            "clickable": True,
            "editable": False,
            "source": "merged",
            "confidence": 0.99,
        },
        {
            "id": "elem_docs_entry",
            "text": "文档",
            "bbox": [80, 20, 160, 60],
            "center": [120, 40],
            "role": "LarkMainNavItem",
            "clickable": True,
            "editable": False,
            "source": "merged",
            "confidence": 0.4,
        },
    ]
    return {
        "screenshot_path": "artifacts/runs/run_002/screenshots/docs_entry_disambiguation.png",
        "candidates": candidates,
        "page_summary": {
            "top_texts": ["文档"],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": len(candidates),
        },
    }


def _fake_vlm_state(
    name: str,
    candidates: list[dict[str, Any]],
    top_texts: list[str],
    *,
    target_candidate_id: str | None,
    reranked_candidate_ids: list[str],
    confidence: float,
    uncertain: bool,
    has_search_box: bool = False,
) -> dict[str, Any]:
    return {
        "screenshot_path": f"artifacts/runs/run_002/screenshots/{name}.png",
        "candidates": candidates,
        "page_summary": {
            "top_texts": top_texts,
            "has_search_box": has_search_box,
            "has_modal": False,
            "candidate_count": len(candidates),
        },
        "vlm_result": {
            "structured": {
                "target_candidate_id": target_candidate_id,
                "reranked_candidate_ids": reranked_candidate_ids,
                "confidence": confidence,
                "uncertain": uncertain,
            }
        },
        "document_region_summary": {},
        "screen_meta": {"width": 2560, "height": 1440, "dpi_scale": 1.25},
    }


def fake_search_box_state() -> dict[str, Any]:
    return _fake_loop_v2_state(
        "search_box",
        [fake_candidate("elem_search", "", editable=True)],
        ["搜索"],
        has_search_box=True,
    )


def fake_chat_candidate_state() -> dict[str, Any]:
    return _fake_loop_v2_state(
        "chat_candidate",
        [fake_candidate("elem_chat", "测试群")],
        ["测试群"],
    )


def fake_chat_open_state() -> dict[str, Any]:
    return _fake_loop_v2_state(
        "chat_open",
        [fake_candidate("elem_message_input", "", editable=True)],
        ["测试群"],
    )


def fake_message_input_state() -> dict[str, Any]:
    return _fake_loop_v2_state(
        "message_input",
        [fake_candidate("elem_message_input", "", editable=True)],
        ["测试群"],
    )


def fake_message_input_with_text_state() -> dict[str, Any]:
    return _fake_loop_v2_state(
        "message_input_with_text",
        [fake_candidate("elem_message_input", "Hello CUA-Lark", editable=True)],
        ["测试群", "Hello CUA-Lark"],
    )


def fake_message_visible_state() -> dict[str, Any]:
    return _fake_loop_v2_state(
        "message_visible",
        [fake_candidate("elem_message", "Hello CUA-Lark")],
        ["测试群", "Hello CUA-Lark"],
    )


def _fake_loop_v2_state(
    name: str,
    candidates: list[dict[str, Any]],
    top_texts: list[str],
    *,
    has_search_box: bool = False,
) -> dict[str, Any]:
    return {
        "screenshot_path": f"artifacts/runs/run_002/screenshots/{name}.png",
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
                "reranked_candidate_ids": [candidate["id"] for candidate in candidates],
                "confidence": 0.88,
                "uncertain": False,
            }
        },
        "document_region_summary": {"visible_title": "无"},
        "screen_meta": {"width": 2560, "height": 1440, "dpi_scale": 1.25},
    }


def fake_candidate(
    candidate_id: str,
    text: str,
    *,
    editable: bool = False,
    role: str | None = None,
    clickable: bool = True,
    source: str = "uia",
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "text": text,
        "bbox": [80, 20, 180, 60],
        "center": [130, 40],
        "role": role or ("Edit" if editable else "Button"),
        "clickable": clickable,
        "editable": editable,
        "source": source,
        "confidence": confidence,
    }


def fake_get_ui_state() -> dict[str, Any]:
    return {
        "screenshot_path": "artifacts/runs/run_001/screenshots/step_001.png",
        "page_summary": {
            "top_texts": ["搜索", "测试群", "文档"],
            "has_search_box": True,
            "has_modal": False,
            "candidate_count": 2,
        },
        "candidates": [
            {
                "id": "elem_docs",
                "text": "文档",
                "bbox": [80, 20, 160, 60],
                "center": [120, 40],
                "role": "Button",
                "clickable": True,
                "editable": False,
                "source": "uia",
                "confidence": 0.9,
            },
            {
                "id": "elem_search",
                "text": "搜索",
                "bbox": [10, 10, 210, 42],
                "center": [110, 26],
                "role": "Edit",
                "clickable": True,
                "editable": True,
                "source": "uia",
                "confidence": 0.95,
            },
        ],
    }


def fake_input_only_ui_state() -> dict[str, Any]:
    return {
        "screenshot_path": "artifacts/runs/run_001/screenshots/step_001.png",
        "page_summary": {
            "top_texts": ["搜索"],
            "has_search_box": True,
            "has_modal": False,
            "candidate_count": 2,
        },
        "candidates": [
            {
                "id": "elem_search_label",
                "text": "搜索",
                "bbox": [10, 10, 80, 42],
                "center": [45, 26],
                "role": "Text",
                "clickable": False,
                "editable": False,
                "source": "ocr",
                "confidence": 0.85,
            },
            {
                "id": "elem_input",
                "text": "",
                "bbox": [90, 10, 260, 42],
                "center": [175, 26],
                "role": "Edit",
                "clickable": True,
                "editable": True,
                "source": "uia",
                "confidence": 0.95,
            },
        ],
    }


def fake_text_ui_state(text: str, *, editable: bool = False) -> dict[str, Any]:
    return {
        "screenshot_path": "artifacts/runs/run_001/screenshots/step_001.png",
        "page_summary": {
            "top_texts": [text],
            "has_search_box": editable,
            "has_modal": False,
            "candidate_count": 1,
        },
        "candidates": [
            {
                "id": "elem_search" if editable else f"elem_{text}",
                "text": text,
                "bbox": [80, 20, 160, 60],
                "center": [120, 40],
                "role": "Edit" if editable else "Button",
                "clickable": True,
                "editable": editable,
                "source": "uia",
                "confidence": 0.95,
            }
        ],
    }


def fake_execute_action(action: Mapping[str, Any]) -> dict[str, Any]:
    print("\n[fake executor 收到动作]")
    print(to_json(action))
    return {"ok": True, "action_type": str(action["action_type"]), "error": None}


def fake_quiet_execute_action(action: Mapping[str, Any]) -> dict[str, Any]:
    return {"ok": True, "action_type": str(action["action_type"]), "error": None}


def fake_validator_success_executor(action: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "action_type": str(action["action_type"]),
        "error": None,
        "validator_result": {"passed": True},
    }


def fake_validator_passing_execute_action(action: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "action_type": str(action["action_type"]),
        "error": None,
        "validator_result": {
            "passed": True,
            "check_type": "text_visible",
            "failure_reason": None,
            "evidence": {"source": "C validator"},
        },
    }


def fake_failed_executor(action: Mapping[str, Any]) -> dict[str, Any]:
    return {"ok": False, "action_type": str(action["action_type"]), "error": "boom"}


def print_planner_loop(result: dict[str, Any]) -> None:
    print("\n[最小 planner loop]")
    print(f"decision: {to_json(result['decision'])}")
    print(f"action_result: {to_json(result['action_result'])}")
    runtime_state = result["runtime_state"]
    print(f"runtime_status: {runtime_state['status']}")
    print(f"runtime_step_idx: {runtime_state['step_idx']}")
    print(f"history_length: {len(runtime_state['history'])}")


def print_action_protocol(decision: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    print("\n[动作协议]")
    print(f"decision: {to_json(dict(decision))}")
    print(f"executor_payload: {to_json(dict(payload))}")


def print_rule_decision(title: str, decision: Mapping[str, Any]) -> None:
    print(f"\n[规则型决策] {title}")
    print(to_json(dict(decision)))


if __name__ == "__main__":
    raise SystemExit(main())
