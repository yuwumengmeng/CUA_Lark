"""成员 B 4.24 最小 planner loop 与规则型决策测试。

测试目标：
验证 `planner/mini_planner.py` 能用 fake `ui_state` 和 fake executor 跑通
最小闭环：读取任务、选择下一步动作、调用执行器、更新 runtime state。
同时覆盖 4.24 第 4 项的三类规则型决策：文本命中型点击、输入框定位型
输入、返回 / 等待类动作。

重要约定：
本测试不访问真实飞书窗口、不截图、不移动鼠标，所有页面状态和执行结果
都在测试内构造。这样测试稳定，也不会影响本机桌面。

运行方式：
```powershell
python -m unittest tests\testB\test_mini_planner.py
python -m unittest discover -s tests\testB
```

修改规则：
1. 如果 action decision 字段变化，必须同步修改这些断言。
2. 如果 `RuntimeState` 的推进规则变化，必须补充成功和失败分支测试。
3. 不允许把真实 GUI 依赖塞进这个单元测试。
"""

from __future__ import annotations

import unittest

from planner.mini_planner import (
    ACTION_CLICK,
    ACTION_HOTKEY,
    ACTION_TYPE,
    ACTION_WAIT,
    DECISION_FAILED,
    FAILURE_EXECUTOR_FAILED,
    FAILURE_TARGET_NOT_FOUND,
    MiniPlanner,
)
from planner.runtime_state import RuntimeStatus, create_runtime_state


class MiniPlannerTests(unittest.TestCase):
    def test_plan_next_action_clicks_matching_docs_candidate(self):
        planner = MiniPlanner()
        state = create_runtime_state("点击文档入口")

        decision = planner.plan_next_action("点击文档入口", fake_ui_state("文档"), state)

        self.assertEqual(decision["status"], "continue")
        self.assertEqual(decision["action_type"], ACTION_CLICK)
        self.assertEqual(decision["target_element_id"], "elem_docs")
        self.assertEqual(decision["x"], 120)
        self.assertEqual(decision["y"], 40)
        self.assertEqual(state.page_summary["candidate_count"], 2)

    def test_run_once_completes_single_click_task(self):
        planner = MiniPlanner()
        state = create_runtime_state("点击文档入口")

        result = planner.run_once(
            "点击文档入口",
            lambda: fake_ui_state("文档"),
            fake_success_executor,
            state,
        )

        self.assertEqual(result.runtime_state.status, RuntimeStatus.SUCCESS.value)
        self.assertEqual(result.runtime_state.step_idx, 2)
        self.assertEqual(len(result.runtime_state.history), 1)
        self.assertEqual(result.action_result, {"ok": True, "action_type": ACTION_CLICK})

    def test_run_once_search_task_clicks_then_types(self):
        planner = MiniPlanner()
        state = create_runtime_state("在搜索框输入项目周报")

        first = planner.run_once(
            "在搜索框输入项目周报",
            fake_input_only_ui_state,
            fake_success_executor,
            state,
        )
        second = planner.run_once(
            "在搜索框输入项目周报",
            fake_input_only_ui_state,
            fake_success_executor,
            state,
        )

        self.assertEqual(first.decision["action_type"], ACTION_CLICK)
        self.assertEqual(first.decision["target_element_id"], "elem_input")
        self.assertEqual(second.decision["action_type"], ACTION_TYPE)
        self.assertEqual(second.decision["text"], "项目周报")
        self.assertEqual(state.status, RuntimeStatus.SUCCESS.value)
        self.assertEqual(state.step_idx, 2)

    def test_back_task_outputs_hotkey_action(self):
        planner = MiniPlanner()
        state = create_runtime_state("返回上一级")

        decision = planner.plan_next_action("返回上一级", fake_ui_state("文档"), state)

        self.assertEqual(decision["action_type"], ACTION_HOTKEY)
        self.assertEqual(decision["keys"], ["alt", "left"])

    def test_structured_wait_task_outputs_wait_action(self):
        planner = MiniPlanner()
        state = create_runtime_state(
            {
                "goal": "wait_demo",
                "steps": [{"action_hint": "wait", "params": {"seconds": 2}}],
            }
        )

        decision = planner.plan_next_action(state.task, fake_ui_state("文档"), state)

        self.assertEqual(decision["action_type"], ACTION_WAIT)
        self.assertEqual(decision["seconds"], 2)

    def test_missing_target_marks_state_failed_without_executor_call(self):
        planner = MiniPlanner()
        state = create_runtime_state("点击文档入口")
        called = False

        def executor(_action):
            nonlocal called
            called = True
            return {"ok": True, "action_type": ACTION_CLICK}

        result = planner.run_once(
            "点击文档入口",
            lambda: fake_ui_state("通讯录"),
            executor,
            state,
        )

        self.assertFalse(called)
        self.assertEqual(result.decision["status"], DECISION_FAILED)
        self.assertEqual(result.decision["failure_reason"], FAILURE_TARGET_NOT_FOUND)
        self.assertEqual(state.status, RuntimeStatus.FAILED.value)
        self.assertEqual(state.failure_reason, FAILURE_TARGET_NOT_FOUND)

    def test_executor_failure_marks_state_failed(self):
        planner = MiniPlanner()
        state = create_runtime_state("点击文档入口")

        result = planner.run_once(
            "点击文档入口",
            lambda: fake_ui_state("文档"),
            lambda _action: {"ok": False, "action_type": ACTION_CLICK, "error": "boom"},
            state,
        )

        self.assertEqual(result.runtime_state.status, RuntimeStatus.FAILED.value)
        self.assertEqual(result.runtime_state.failure_reason, FAILURE_EXECUTOR_FAILED)
        self.assertEqual(result.action_result["error"], "boom")

    def test_chat_candidate_ignores_single_character_fragment(self):
        planner = MiniPlanner()
        task = {
            "goal": "open_chat",
            "params": {"target_name": "测试群"},
            "steps": [{"action_hint": "find_chat", "target_text": "测试群"}],
        }
        state = create_runtime_state(task)

        decision = planner.plan_next_action(task, single_character_chat_fragment_state(), state)

        self.assertEqual(decision["status"], DECISION_FAILED)
        self.assertEqual(decision["failure_reason"], FAILURE_TARGET_NOT_FOUND)

    def test_chat_candidate_prefers_meaningful_fragment(self):
        planner = MiniPlanner()
        task = {
            "goal": "open_chat",
            "params": {"target_name": "测试群"},
            "steps": [{"action_hint": "find_chat", "target_text": "测试群"}],
        }
        state = create_runtime_state(task)

        decision = planner.plan_next_action(task, meaningful_chat_fragment_state(), state)

        self.assertEqual(decision["status"], "continue")
        self.assertEqual(decision["action_type"], ACTION_CLICK)
        self.assertEqual(decision["target_element_id"], "elem_chat_title")
        self.assertEqual(decision["x"], 1052)
        self.assertEqual(decision["y"], 333)

    def test_chat_candidate_uses_unique_chat_row_suffix_fallback(self):
        planner = MiniPlanner()
        task = {
            "goal": "open_chat",
            "params": {"target_name": "测试群"},
            "steps": [
                {"action_hint": "find_chat", "target_text": "测试群"},
                {"action_hint": "click_target", "target_text": "测试群"},
            ],
        }
        state = create_runtime_state(task)

        decision = planner.plan_next_action(task, unique_chat_row_suffix_state(), state)

        self.assertEqual(decision["status"], "continue")
        self.assertEqual(decision["action_type"], ACTION_CLICK)
        self.assertEqual(decision["target_element_id"], "elem_chat_row")
        self.assertEqual(decision["x"], 1350)
        self.assertEqual(decision["y"], 360)
        self.assertEqual(decision["planner_meta"]["matched_text"], "群")
        self.assertEqual(decision["planner_meta"]["fallback"], "unique_chat_row_suffix")
        self.assertIn("unique chat row suffix fallback", decision["reason"])

    def test_chat_candidate_rejects_multiple_chat_row_suffix_fallbacks(self):
        planner = MiniPlanner()
        task = {
            "goal": "open_chat",
            "params": {"target_name": "测试群"},
            "steps": [{"action_hint": "find_chat", "target_text": "测试群"}],
        }
        state = create_runtime_state(task)

        decision = planner.plan_next_action(task, multiple_chat_row_suffix_state(), state)

        self.assertEqual(decision["status"], DECISION_FAILED)
        self.assertEqual(decision["failure_reason"], FAILURE_TARGET_NOT_FOUND)

    def test_chat_candidate_prefers_text_match_over_chat_row_suffix_fallback(self):
        planner = MiniPlanner()
        task = {
            "goal": "open_chat",
            "params": {"target_name": "测试群"},
            "steps": [{"action_hint": "find_chat", "target_text": "测试群"}],
        }
        state = create_runtime_state(task)

        decision = planner.plan_next_action(task, text_match_with_chat_row_suffix_state(), state)

        self.assertEqual(decision["status"], "continue")
        self.assertEqual(decision["target_element_id"], "elem_chat_title")
        self.assertNotIn("fallback", decision["planner_meta"])


def fake_ui_state(primary_text: str) -> dict:
    candidates = [
        {
            "id": "elem_docs" if primary_text == "文档" else "elem_other",
            "text": primary_text,
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
    ]
    return {
        "screenshot_path": "artifacts/runs/run_test/screenshots/step_001_before.png",
        "candidates": candidates,
        "page_summary": {
            "top_texts": [candidate["text"] for candidate in candidates],
            "has_search_box": True,
            "has_modal": False,
            "candidate_count": len(candidates),
        },
    }


def fake_input_only_ui_state() -> dict:
    candidates = [
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
    ]
    return {
        "screenshot_path": "artifacts/runs/run_test/screenshots/step_001_before.png",
        "candidates": candidates,
        "page_summary": {
            "top_texts": ["搜索"],
            "has_search_box": True,
            "has_modal": False,
            "candidate_count": len(candidates),
        },
    }


def single_character_chat_fragment_state() -> dict:
    candidates = [
        {
            "id": "elem_single_group_char",
            "text": "群",
            "bbox": [2385, 368, 2409, 413],
            "center": [2397, 390],
            "role": "text",
            "clickable": False,
            "editable": False,
            "source": "ocr",
            "confidence": 0.93,
        }
    ]
    return {
        "screenshot_path": "artifacts/runs/run_test/screenshots/step_001_before.png",
        "candidates": candidates,
        "page_summary": {
            "top_texts": ["欢迎加入群组"],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": len(candidates),
        },
    }


def meaningful_chat_fragment_state() -> dict:
    candidates = [
        {
            "id": "elem_single_group_char",
            "text": "群",
            "bbox": [2385, 368, 2409, 413],
            "center": [2397, 390],
            "role": "text",
            "clickable": False,
            "editable": False,
            "source": "ocr",
            "confidence": 0.93,
        },
        {
            "id": "elem_chat_title",
            "text": "测试",
            "bbox": [1011, 320, 1094, 347],
            "center": [1052, 333],
            "role": "text",
            "clickable": False,
            "editable": False,
            "source": "ocr",
            "confidence": 0.87,
        },
    ]
    return {
        "screenshot_path": "artifacts/runs/run_test/screenshots/step_001_before.png",
        "candidates": candidates,
        "page_summary": {
            "top_texts": ["测试", "欢迎加入群组"],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": len(candidates),
        },
    }


def unique_chat_row_suffix_state() -> dict:
    candidates = [
        {
            "id": "elem_regular_group_char",
            "text": "群",
            "bbox": [2385, 368, 2409, 413],
            "center": [2397, 390],
            "role": "text",
            "clickable": False,
            "editable": False,
            "source": "ocr",
            "confidence": 0.93,
        },
        {
            "id": "elem_chat_row",
            "text": "群",
            "bbox": [900, 318, 1800, 402],
            "center": [1350, 360],
            "role": "LarkChatListItem",
            "clickable": True,
            "editable": False,
            "source": "merged",
            "confidence": 0.9,
        },
    ]
    return {
        "screenshot_path": "artifacts/runs/run_test/screenshots/step_001_before.png",
        "candidates": candidates,
        "page_summary": {
            "top_texts": ["群"],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": len(candidates),
        },
    }


def multiple_chat_row_suffix_state() -> dict:
    state = unique_chat_row_suffix_state()
    state["candidates"].append(
        {
            "id": "elem_second_chat_row",
            "text": "群",
            "bbox": [900, 438, 1800, 522],
            "center": [1350, 480],
            "role": "LarkChatListItem",
            "clickable": True,
            "editable": False,
            "source": "merged",
            "confidence": 0.9,
        }
    )
    state["page_summary"]["candidate_count"] = len(state["candidates"])
    return state


def text_match_with_chat_row_suffix_state() -> dict:
    state = unique_chat_row_suffix_state()
    state["candidates"].append(
        {
            "id": "elem_chat_title",
            "text": "测试",
            "bbox": [1011, 320, 1094, 347],
            "center": [1052, 333],
            "role": "text",
            "clickable": False,
            "editable": False,
            "source": "ocr",
            "confidence": 0.87,
        }
    )
    state["page_summary"]["top_texts"].append("测试")
    state["page_summary"]["candidate_count"] = len(state["candidates"])
    return state


def fake_success_executor(action: dict) -> dict:
    return {"ok": True, "action_type": action["action_type"]}


if __name__ == "__main__":
    unittest.main()
