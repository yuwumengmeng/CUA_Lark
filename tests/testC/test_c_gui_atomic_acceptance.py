"""Unit checks for the member C GUI atomic acceptance runner.

These tests only inspect planned actions and report snapshots. They never touch
the real desktop.
"""

import argparse
import unittest

from tests.testC.run_c_gui_atomic_acceptance import build_actions, _state_snapshot


class CGuiAtomicAcceptanceTests(unittest.TestCase):
    def test_build_actions_never_sends_message(self):
        args = argparse.Namespace(
            search_hotkey=["ctrl", "k"],
            chat_name="测试群",
            wait_seconds=1.2,
        )

        actions = build_actions(args)

        self.assertEqual([action["step_name"] for action in actions], [
            "open_search_by_hotkey",
            "type_search_keyword",
            "wait_for_search_results",
        ])
        self.assertEqual(actions[1]["text"], "测试群")
        self.assertNotIn({"keys": ["enter"]}, actions)
        self.assertTrue(all(action["action_type"] != "click" for action in actions))

    def test_state_snapshot_is_compact(self):
        snapshot = _state_snapshot(
            {
                "screenshot_path": "after.png",
                "candidates": [{"id": "a"}, {"id": "b"}],
                "page_summary": {"top_texts": [str(index) for index in range(20)]},
                "screen_meta": {"dpi_scale": 1.5},
            }
        )

        self.assertEqual(snapshot["screenshot_path"], "after.png")
        self.assertEqual(snapshot["candidate_count"], 2)
        self.assertEqual(len(snapshot["top_texts"]), 10)
        self.assertEqual(snapshot["screen_meta"]["dpi_scale"], 1.5)


if __name__ == "__main__":
    unittest.main()
