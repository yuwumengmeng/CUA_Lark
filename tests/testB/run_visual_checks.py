"""成员 B 可视化检查脚本。

使用场景：
这个脚本不是单元测试，而是给人看的控制台验证。它会把任务解析、
runtime state 更新、动作历史记录、4.24 最小 planner loop 和规则型决策
打印出来，并用 `[PASS]` 标明每一步检查成功。4.25 的检查会额外展示
fake A/C 闭环、step-level 成功判断、5 个单步操作和失败原因分类。
第二周期检查会展示 workflow.v2 默认 IM 短流程 schema。

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

import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planner.action_protocol import ACTION_DECISION_VERSION, action_decision_to_executor_payload
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
from planner.runtime_state import RuntimeStateError, RuntimeStatus, create_runtime_state
from planner.task_schema import parse_task
from planner.workflow_schema import (
    STEP_VERIFY_MESSAGE_SENT,
    build_im_message_workflow,
    build_sample_im_workflows,
)


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
    print("成员B 4.23、4.24 第 3/4/5 项和 4.25 第 6/7/8/9 项的核心数据链路都能正常工作。\n")
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
