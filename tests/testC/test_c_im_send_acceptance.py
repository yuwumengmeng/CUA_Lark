"""Unit checks for the member C IM send acceptance runner.

These tests inspect plans and geometry only. They never touch the real desktop
and never send a Feishu message.
"""

import argparse
import unittest

from capture.screenshot import ScreenBounds
from tests.testC.run_c_im_send_acceptance import (
    build_open_result_actions,
    build_search_actions,
    build_send_actions,
    resolve_first_search_result_point,
    resolve_message_input_point,
)


class CImSendAcceptanceTests(unittest.TestCase):
    def test_search_actions_reset_overlay_and_wait_for_results(self):
        args = argparse.Namespace(
            search_hotkey=["ctrl", "k"],
            chat_name="测试群",
            wait_seconds=1.2,
        )

        actions = build_search_actions(args)

        self.assertEqual(
            [action["step_name"] for action in actions],
            [
                "cleanup_close_search",
                "open_search_by_hotkey",
                "type_search_keyword",
                "wait_for_search_results",
            ],
        )
        self.assertEqual(actions[0]["keys"], ["esc"])
        self.assertEqual(actions[0]["repeat"], 2)
        self.assertEqual(actions[2]["text"], "测试群")

    def test_open_result_actions_click_first_result_then_wait(self):
        args = argparse.Namespace(chat_name="测试群", wait_seconds=1.2)

        actions = build_open_result_actions(args, (500, 600))

        self.assertEqual([action["step_name"] for action in actions], [
            "click_first_search_result",
            "wait_for_chat_page",
        ])
        self.assertEqual(actions[0]["x"], 500)
        self.assertEqual(actions[0]["y"], 600)

    def test_send_actions_type_then_press_enter(self):
        actions = build_send_actions("Hello CUA-Lark C run_001", (120, 340))

        self.assertEqual([action["step_name"] for action in actions], [
            "type_message",
            "press_enter_to_send",
            "wait_for_message_sent",
        ])
        self.assertEqual(actions[0]["text"], "Hello CUA-Lark C run_001")
        self.assertEqual(actions[0]["x"], 120)
        self.assertEqual(actions[0]["y"], 340)
        self.assertEqual(actions[1]["keys"], ["enter"])

    def test_resolve_message_input_point_uses_window_bounds(self):
        def fake_finder(title: str):
            return 1, title, ScreenBounds(x=1000, y=200, width=800, height=600)

        point = resolve_message_input_point(
            "飞书",
            x_ratio=0.5,
            bottom_offset=80,
            finder=fake_finder,
        )

        self.assertEqual(point, (1400, 720))

    def test_resolve_first_search_result_point_uses_window_bounds(self):
        def fake_finder(title: str):
            return 1, title, ScreenBounds(x=1000, y=200, width=800, height=600)

        point = resolve_first_search_result_point(
            "飞书",
            x_ratio=0.4,
            y_ratio=0.25,
            finder=fake_finder,
        )

        self.assertEqual(point, (1320, 350))


if __name__ == "__main__":
    unittest.main()
