"""成员 B 4.25 planner 闭环集成测试。

测试目标：
本文件验证成员 B 编号 6-9：能直接消费成员 A 风格 `ui_state`，能接收成员 C
风格 `action_result`，能做首版 step-level 成功判断，能覆盖 5 个单步操作，
并能稳定输出 `target_not_found / page_mismatch / executor_failed` 三类失败原因。

重要约定：
这里全部使用 fake `ui_state` 和 fake executor，不访问真实飞书窗口、不截图、
不移动鼠标。真实桌面联调应放到 A/C 的验收脚本中。

运行方式：
```powershell
python -m unittest tests\testB\test_planner_loop_integration.py
python -m unittest discover -s tests\testB
```

修改规则：
1. 修改 A 的 `ui_state` 字段协议时，必须同步更新 fake 数据和断言。
2. 修改 C 的 `action_result` 字段协议时，必须同步更新 fake executor。
3. 新增成功判断规则时，必须补充 after 页面断言。
"""

from __future__ import annotations

import unittest

from planner.mini_planner import (
    ACTION_CLICK,
    ACTION_HOTKEY,
    ACTION_TYPE,
    DECISION_FAILED,
    FAILURE_EXECUTOR_FAILED,
    FAILURE_PAGE_MISMATCH,
    FAILURE_TARGET_NOT_FOUND,
    MiniPlanner,
)
from planner.runtime_state import RuntimeStatus, create_runtime_state


class PlannerLoopIntegrationTests(unittest.TestCase):
    def test_run_once_consumes_a_ui_state_and_c_action_result(self):
        planner = MiniPlanner()
        state = create_runtime_state("在搜索框输入项目周报")
        calls: list[dict] = []

        def executor(action: dict) -> dict:
            calls.append(action)
            return {"ok": True, "action_type": action["action_type"], "error": None}

        first = planner.run_once(
            "在搜索框输入项目周报",
            lambda: ui_state("搜索", editable=True),
            executor,
            state,
            get_after_ui_state=lambda: ui_state("搜索", editable=True),
        )
        second = planner.run_once(
            "在搜索框输入项目周报",
            lambda: ui_state("搜索", editable=True),
            executor,
            state,
            get_after_ui_state=lambda: ui_state("项目周报", editable=True),
        )

        self.assertEqual(first.decision["action_type"], ACTION_CLICK)
        self.assertEqual(second.decision["action_type"], ACTION_TYPE)
        self.assertEqual(calls[0]["target_element_id"], "elem_search")
        self.assertEqual(calls[1]["text"], "项目周报")
        self.assertEqual(state.status, RuntimeStatus.SUCCESS.value)
        self.assertEqual(state.page_summary["top_texts"], ["项目周报"])

    def test_after_page_mismatch_marks_step_failed(self):
        planner = MiniPlanner()
        state = create_runtime_state("在搜索框输入项目周报")
        state.step_idx = 1

        result = planner.run_once(
            "在搜索框输入项目周报",
            lambda: ui_state("搜索", editable=True),
            success_executor,
            state,
            get_after_ui_state=lambda: ui_state("搜索", editable=True),
        )

        self.assertEqual(result.decision["action_type"], ACTION_TYPE)
        self.assertEqual(result.runtime_state.status, RuntimeStatus.FAILED.value)
        self.assertEqual(result.runtime_state.failure_reason, FAILURE_PAGE_MISMATCH)
        self.assertEqual(result.action_result["ok"], True)

    def test_failure_reason_categories_are_stable(self):
        planner = MiniPlanner()

        missing = planner.run_once(
            "点击文档入口",
            lambda: ui_state("通讯录"),
            success_executor,
            create_runtime_state("点击文档入口"),
        )
        self.assertEqual(missing.decision["status"], DECISION_FAILED)
        self.assertEqual(missing.decision["failure_reason"], FAILURE_TARGET_NOT_FOUND)

        page_mismatch = planner.run_once(
            "在搜索框输入项目周报",
            lambda: ui_state("搜索", editable=True),
            success_executor,
            create_type_step_state(),
            get_after_ui_state=lambda: ui_state("搜索", editable=True),
        )
        self.assertEqual(page_mismatch.runtime_state.failure_reason, FAILURE_PAGE_MISMATCH)

        executor_failed = planner.run_once(
            "点击文档入口",
            lambda: ui_state("文档"),
            lambda action: {"ok": False, "action_type": action["action_type"], "error": "boom"},
            create_runtime_state("点击文档入口"),
        )
        self.assertEqual(executor_failed.runtime_state.failure_reason, FAILURE_EXECUTOR_FAILED)

    def test_open_chat_strict_check_accepts_chat_surface_when_title_ocr_is_missing(self):
        planner = MiniPlanner()
        task = {
            "goal": "open_chat",
            "params": {"target_name": "\u6d4b\u8bd5\u7fa4"},
            "steps": [
                {"action_hint": "find_chat", "target_text": "\u6d4b\u8bd5\u7fa4"},
            ],
        }

        result = planner.run_once(
            task,
            lambda: ui_state("\u6d4b\u8bd5\u7fa4"),
            success_executor,
            create_runtime_state(task),
            get_after_ui_state=lambda: ui_state("\u65b0\u6d88\u606f"),
        )

        self.assertEqual(result.runtime_state.status, RuntimeStatus.SUCCESS.value)

    def test_five_single_step_scenarios_drive_planner_loop(self):
        planner = MiniPlanner()

        sidebar_state = create_runtime_state(
            {
                "goal": "open_sidebar_module",
                "steps": [{"action_hint": "find_sidebar_module", "target_text": "日历"}],
            }
        )
        sidebar = planner.run_once(
            sidebar_state.task,
            lambda: ui_state("日历"),
            success_executor,
            sidebar_state,
            get_after_ui_state=lambda: ui_state("日历"),
        )

        search_state = create_runtime_state("在搜索框输入项目周报")
        planner.run_once(
            search_state.task,
            lambda: ui_state("搜索", editable=True),
            success_executor,
            search_state,
            get_after_ui_state=lambda: ui_state("搜索", editable=True),
        )
        search = planner.run_once(
            search_state.task,
            lambda: ui_state("搜索", editable=True),
            success_executor,
            search_state,
            get_after_ui_state=lambda: ui_state("项目周报", editable=True),
        )

        chat = planner.run_once(
            "打开聊天列表里的测试群",
            lambda: ui_state("测试群"),
            success_executor,
            create_runtime_state("打开聊天列表里的测试群"),
            get_after_ui_state=lambda: ui_state("测试群"),
        )
        docs = planner.run_once(
            "点击文档入口",
            lambda: ui_state("文档"),
            success_executor,
            create_runtime_state("点击文档入口"),
            get_after_ui_state=lambda: ui_state("新建文档"),
        )
        back = planner.run_once(
            "返回上一级",
            lambda: ui_state("文档"),
            success_executor,
            create_runtime_state("返回上一级"),
            get_after_ui_state=lambda: ui_state("聊天"),
        )

        for result in (sidebar, search, chat, docs, back):
            self.assertEqual(result.runtime_state.status, RuntimeStatus.SUCCESS.value)
        self.assertEqual(back.decision["action_type"], ACTION_HOTKEY)


def create_type_step_state():
    state = create_runtime_state("在搜索框输入项目周报")
    state.step_idx = 1
    return state


def success_executor(action: dict) -> dict:
    return {"ok": True, "action_type": action["action_type"], "error": None}


def ui_state(
    text: str,
    *,
    editable: bool = False,
    role: str | None = None,
) -> dict:
    role_text = role or ("Edit" if editable else "Button")
    candidate_id = "elem_search" if editable else f"elem_{text}"
    candidates = [
        {
            "id": candidate_id,
            "text": text,
            "bbox": [80, 20, 160, 60],
            "center": [120, 40],
            "role": role_text,
            "clickable": True,
            "editable": editable,
            "source": "uia",
            "confidence": 0.95,
        }
    ]
    return {
        "screenshot_path": "artifacts/runs/run_test/screenshots/step_001_before.png",
        "candidates": candidates,
        "page_summary": {
            "top_texts": [text],
            "has_search_box": editable,
            "has_modal": False,
            "candidate_count": len(candidates),
        },
    }


if __name__ == "__main__":
    unittest.main()
