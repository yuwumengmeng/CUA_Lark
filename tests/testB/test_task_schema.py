"""任务表示测试脚本。

这个脚本验证成员 B 4.23 第一项的 `task_schema.py`：自然语言任务能被
解析成固定 `goal / params / steps`，半结构化输入也能生成默认步骤。
"""

import unittest

from planner.task_schema import (
    GOAL_INPUT_SEARCH,
    GOAL_OPEN_CHAT,
    GOAL_OPEN_DOCS,
    ACTION_CLICK_TARGET,
    ACTION_FIND_CHAT,
    ACTION_FIND_SEARCH_BOX,
    ACTION_FIND_SIDEBAR_MODULE,
    ACTION_TYPE_TEXT,
    TaskSchemaError,
    parse_task,
)


class TaskSchemaTests(unittest.TestCase):
    def test_parse_open_chat_instruction(self):
        task = parse_task("打开聊天列表里的测试群")

        self.assertEqual(task.goal, GOAL_OPEN_CHAT)
        self.assertEqual(task.params["target_name"], "测试群")
        self.assertEqual(
            [step.action_hint for step in task.steps],
            [ACTION_FIND_CHAT, ACTION_CLICK_TARGET],
        )
        self.assertEqual(task.to_dict()["steps"][0]["target_text"], "测试群")

    def test_parse_input_search_instruction(self):
        task = parse_task("在搜索框输入项目周报")

        self.assertEqual(task.goal, GOAL_INPUT_SEARCH)
        self.assertEqual(task.params["text"], "项目周报")
        self.assertEqual(
            [step.action_hint for step in task.steps],
            [ACTION_FIND_SEARCH_BOX, ACTION_TYPE_TEXT],
        )
        self.assertEqual(task.steps[1].input_text, "项目周报")

    def test_parse_open_docs_instruction(self):
        task = parse_task("点击文档入口")

        self.assertEqual(task.goal, GOAL_OPEN_DOCS)
        self.assertEqual(
            [step.action_hint for step in task.steps],
            [ACTION_FIND_SIDEBAR_MODULE, ACTION_CLICK_TARGET],
        )
        self.assertEqual(task.steps[0].target_text, "文档")

    def test_parse_structured_task_without_steps(self):
        task = parse_task({"goal": "open_chat", "params": {"target_name": "测试群"}})

        self.assertEqual(task.goal, GOAL_OPEN_CHAT)
        self.assertEqual(task.params["target_name"], "测试群")
        self.assertEqual(len(task.steps), 2)

    def test_unknown_natural_language_requires_structured_input(self):
        with self.assertRaises(TaskSchemaError):
            parse_task("帮我做一个很复杂的飞书任务")


if __name__ == "__main__":
    unittest.main()
