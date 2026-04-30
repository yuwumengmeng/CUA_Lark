"""Member C validator tests.

These tests use plain ui_state dictionaries and do not touch the real desktop.
"""

import unittest

from executor import (
    ValidatorResult,
    validate_candidates_contain_text,
    validate_chat_message_visible,
    validate_expected_after_state,
    validate_page_changed,
    validate_text_visible,
    validate_vlm_semantic_pass,
)
from executor.validator import (
    VALIDATION_EXPECTATION_UNSUPPORTED,
    VALIDATION_PAGE_NOT_CHANGED,
    VALIDATION_TARGET_TEXT_NOT_FOUND,
    VALIDATION_VLM_SEMANTIC_FAILED,
)


class ValidatorTests(unittest.TestCase):
    def test_validator_result_roundtrip(self):
        result = ValidatorResult.pass_(
            check_type="target_text_visible",
            target_text="Hello",
            evidence={"source": "candidates"},
        )

        restored = ValidatorResult.from_dict(result.to_dict())

        self.assertTrue(restored.passed)
        self.assertEqual(restored.check_type, "target_text_visible")
        self.assertEqual(restored.target_text, "Hello")
        self.assertEqual(restored.evidence["source"], "candidates")

    def test_text_visible_checks_summary_candidates_document_and_semantics(self):
        result = validate_text_visible(sample_after_state(), "Hello CUA-Lark")

        self.assertTrue(result.passed)
        sources = {match["source"] for match in result.evidence["matches"]}
        self.assertIn("page_summary.top_texts", sources)
        self.assertIn("candidates", sources)
        self.assertIn("document_region_summary.visible_texts", sources)
        self.assertIn("page_semantics.entities", sources)

    def test_text_visible_failure_is_diagnostic(self):
        result = validate_text_visible(sample_after_state(), "missing text")

        self.assertFalse(result.passed)
        self.assertEqual(result.failure_reason, VALIDATION_TARGET_TEXT_NOT_FOUND)
        self.assertEqual(result.evidence["source_count"], 0)

    def test_candidates_contain_text_and_chat_message_visible(self):
        candidates = {"candidates": [{"id": "elem_msg", "text": "Hello CUA-Lark"}]}

        self.assertTrue(validate_candidates_contain_text(candidates, "Hello").passed)
        self.assertTrue(validate_chat_message_visible(sample_after_state(), "Hello CUA-Lark").passed)

    def test_page_changed_uses_material_ui_state_signature(self):
        unchanged = validate_page_changed(sample_before_state(), sample_before_state())
        changed = validate_page_changed(sample_before_state(), sample_after_state())

        self.assertFalse(unchanged.passed)
        self.assertEqual(unchanged.failure_reason, VALIDATION_PAGE_NOT_CHANGED)
        self.assertTrue(changed.passed)
        self.assertTrue(changed.evidence["screenshot_path_changed"])

    def test_vlm_semantic_signal_and_goal_progress(self):
        signal_result = validate_vlm_semantic_pass(
            sample_after_state(),
            expected_signal="target_visible",
        )
        goal_result = validate_vlm_semantic_pass(
            sample_after_state(),
            expected_goal_progress="achieved",
        )
        failed = validate_vlm_semantic_pass(
            sample_after_state(),
            expected_signal="search_active",
        )

        self.assertTrue(signal_result.passed)
        self.assertTrue(goal_result.passed)
        self.assertFalse(failed.passed)
        self.assertEqual(failed.failure_reason, VALIDATION_VLM_SEMANTIC_FAILED)

    def test_expected_after_state_combines_supported_checks(self):
        result = validate_expected_after_state(
            {
                "page_changed": True,
                "message_text": "Hello CUA-Lark",
                "vlm_signal": "target_visible",
            },
            before_ui_state=sample_before_state(),
            after_ui_state=sample_after_state(),
        )

        self.assertTrue(result.passed)
        self.assertEqual(len(result.evidence["results"]), 3)

    def test_expected_after_state_rejects_unsupported_keys(self):
        result = validate_expected_after_state(
            {"unknown": True},
            before_ui_state=sample_before_state(),
            after_ui_state=sample_after_state(),
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.failure_reason, VALIDATION_EXPECTATION_UNSUPPORTED)


def sample_before_state() -> dict:
    return {
        "screenshot_path": "before.png",
        "candidates": [{"id": "elem_input", "text": "", "center": [100, 200]}],
        "page_summary": {
            "top_texts": ["测试群"],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": 1,
        },
        "vlm_result": {"structured": {}},
    }


def sample_after_state() -> dict:
    return {
        "screenshot_path": "after.png",
        "candidates": [
            {"id": "elem_input", "text": "", "center": [100, 200]},
            {"id": "elem_msg", "text": "Hello CUA-Lark", "center": [300, 400]},
        ],
        "page_summary": {
            "top_texts": ["测试群", "Hello CUA-Lark"],
            "has_search_box": False,
            "has_modal": False,
            "candidate_count": 2,
        },
        "document_region_summary": {
            "visible_texts": ["Hello CUA-Lark"],
        },
        "vlm_result": {
            "structured": {
                "page_semantics": {
                    "summary": "The sent message is visible in the chat area.",
                    "signals": {"target_visible": True, "search_active": False},
                    "goal_progress": "achieved",
                    "entities": [
                        {
                            "type": "other",
                            "name": "Hello CUA-Lark",
                            "visible": True,
                            "matched_target": True,
                        }
                    ],
                }
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
