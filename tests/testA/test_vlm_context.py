"""成员 A 2.5 VLM Context 控制策略测试。"""

import sys
import unittest
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

context = import_module("vlm.context")

FOCUS_DOCUMENT_BODY = context.FOCUS_DOCUMENT_BODY
FOCUS_INPUT = context.FOCUS_INPUT
FOCUS_MODAL = context.FOCUS_MODAL
FOCUS_RESULTS = context.FOCUS_RESULTS
FOCUS_SEARCH = context.FOCUS_SEARCH
VLMContextPolicy = context.VLMContextPolicy
build_vlm_context_payload = context.build_vlm_context_payload
compact_candidate = context.compact_candidate
compact_page_summary = context.compact_page_summary
infer_context_focus = context.infer_context_focus
select_candidate_context = context.select_candidate_context


def candidate(
    cid,
    text,
    center,
    *,
    role="Text",
    bbox=None,
    clickable=False,
    editable=False,
    confidence=0.5,
):
    x, y = center
    return {
        "id": cid,
        "text": text,
        "bbox": bbox or [x - 10, y - 10, x + 10, y + 10],
        "center": [x, y],
        "role": role,
        "clickable": clickable,
        "editable": editable,
        "source": "merged",
        "confidence": confidence,
        "debug_raw": "must not be sent",
    }


class VLMContextFocusTests(unittest.TestCase):
    def test_infer_search_focus(self):
        self.assertEqual(
            infer_context_focus(step_name="open_search", step_goal="点击搜索框"),
            FOCUS_SEARCH,
        )

    def test_infer_results_focus_for_open_chat(self):
        self.assertEqual(
            infer_context_focus(step_name="open_chat", step_goal="打开测试群搜索结果"),
            FOCUS_RESULTS,
        )

    def test_infer_input_focus(self):
        self.assertEqual(
            infer_context_focus(step_name="send_message", step_goal="在输入框输入消息"),
            FOCUS_INPUT,
        )

    def test_infer_modal_focus(self):
        self.assertEqual(
            infer_context_focus(region_hint="center modal menu panel"),
            FOCUS_MODAL,
        )

    def test_infer_document_body_focus(self):
        self.assertEqual(
            infer_context_focus(step_goal="判断文档正文是否包含目标关键词"),
            FOCUS_DOCUMENT_BODY,
        )


class VLMContextCompactionTests(unittest.TestCase):
    def test_compact_page_summary_limits_text_count_and_length(self):
        summary = {
            "top_texts": ["很长的页面文本" * 20 for _ in range(30)],
            "has_search_box": True,
            "has_modal": False,
            "candidate_count": 80,
            "semantic_summary": {"should": "not be forwarded"},
        }

        compacted = compact_page_summary(summary)

        self.assertEqual(len(compacted["top_texts"]), 12)
        self.assertTrue(all(len(text) <= 48 for text in compacted["top_texts"]))
        self.assertNotIn("semantic_summary", compacted)

    def test_compact_candidate_removes_extra_fields_and_truncates_text(self):
        raw = candidate("elem_0001", "测试文本" * 40, [100, 100], editable=True)

        compacted = compact_candidate(raw, policy=VLMContextPolicy(max_candidate_text_length=24))

        self.assertEqual(compacted["id"], "elem_0001")
        self.assertLessEqual(len(compacted["text"]), 24)
        self.assertNotIn("debug_raw", compacted)
        self.assertTrue(compacted["editable"])


class VLMContextSelectionTests(unittest.TestCase):
    def test_search_context_prioritizes_search_candidate(self):
        candidates = [
            candidate("elem_0001", "普通文本", [1200, 900], role="Text"),
            candidate("elem_0002", "关闭", [2500, 20], role="TitleBar", clickable=True),
            candidate("elem_0003", "搜索", [120, 70], role="LarkSearchBox", editable=True),
        ]

        selected = select_candidate_context(
            candidates,
            max_candidates=1,
            step_name="open_search",
            step_goal="点击搜索框",
            region_hint="top search area",
        )

        self.assertEqual(selected[0]["id"], "elem_0003")

    def test_open_chat_context_prioritizes_target_result_with_cjk_ngram(self):
        candidates = [
            candidate("elem_0001", "测试", [1000, 200], role="Text", clickable=True),
            candidate("elem_0002", "测试群(3)", [1000, 300], role="LarkChatListItem", clickable=True),
            candidate("elem_0003", "搜索", [200, 70], role="LarkSearchBox", editable=True),
        ]

        selected = select_candidate_context(
            candidates,
            max_candidates=1,
            step_name="open_chat",
            step_goal="打开测试群",
            region_hint="search results list",
        )

        self.assertEqual(selected[0]["id"], "elem_0002")

    def test_input_context_prioritizes_bottom_editable(self):
        candidates = [
            candidate("elem_0001", "搜索", [200, 70], role="LarkSearchBox", editable=True),
            candidate("elem_0002", "发送给 测试群", [1800, 1320], role="EditControl", editable=True),
            candidate("elem_0003", "测试群", [1200, 100], role="Text"),
        ]

        selected = select_candidate_context(
            candidates,
            max_candidates=1,
            step_name="focus_message_input",
            step_goal="定位消息输入框",
            region_hint="bottom message input area",
        )

        self.assertEqual(selected[0]["id"], "elem_0002")

    def test_document_body_context_prioritizes_right_body_region(self):
        candidates = [
            candidate("elem_0001", "云文档", [90, 220], role="LarkMainNavItem", clickable=True),
            candidate("elem_0002", "项目方案正文内容", [1800, 600], role="DocumentText"),
            candidate("elem_0003", "标题", [500, 120], role="Text"),
        ]

        selected = select_candidate_context(
            candidates,
            max_candidates=1,
            step_goal="判断文档正文是否包含项目方案关键词",
            region_hint="right document body",
        )

        self.assertEqual(selected[0]["id"], "elem_0002")

    def test_build_payload_includes_small_context_meta(self):
        candidates = [
            candidate(f"elem_{index:04d}", f"候选{index}", [index * 10, index * 10])
            for index in range(40)
        ]
        payload = build_vlm_context_payload(
            instruction="选择当前步骤目标",
            page_summary={"top_texts": [f"文本{index}" for index in range(20)], "candidate_count": 40},
            candidates=candidates,
            max_candidates=5,
            step_name="open_search",
            step_goal="点击搜索框",
            region_hint="top search area",
        )

        self.assertEqual(len(payload["candidates"]), 5)
        self.assertEqual(payload["context_meta"]["candidate_count_total"], 40)
        self.assertEqual(payload["context_meta"]["candidate_count_sent"], 5)
        self.assertEqual(payload["context_meta"]["focus"], FOCUS_SEARCH)


if __name__ == "__main__":
    unittest.main()
