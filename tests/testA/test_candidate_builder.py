"""成员 A 4.24 首版 Candidate Builder 测试。

测试目标：
本文件验证 `perception/candidate_builder.py` 是否能把 OCR/UIA 抽取结果
统一转换为 `UICandidate`，并遵守屏幕绝对坐标协议。

重要约定：
这里不调用真实 OCR、真实 UIA，也不访问桌面窗口。测试直接构造
`OCRTextBlock` 和 `UIAElement`，只验证转换、合并、排序和页面摘要。

直接运行：
```powershell
python tests\testA\test_candidate_builder.py
```

通过 pytest 运行：
```powershell
python -m pytest tests\testA\test_candidate_builder.py
```
"""

import sys
import unittest
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

builder_module = import_module("perception.candidate_builder")
ocr_module = import_module("perception.ocr")
uia_module = import_module("perception.uia")

CandidateBuilderConfig = builder_module.CandidateBuilderConfig
build_ui_candidates = builder_module.build_ui_candidates
build_ui_state_from_elements = builder_module.build_ui_state_from_elements
OCRTextBlock = ocr_module.OCRTextBlock
UIAElement = uia_module.UIAElement


class CandidateBuilderTests(unittest.TestCase):
    def test_ocr_bbox_is_offset_to_screen_coordinates(self):
        candidates = build_ui_candidates(
            ocr_elements=[
                OCRTextBlock(text="消息", bbox=[40, 50, 90, 80], confidence=0.92),
            ],
            screen_origin=[4000, -20],
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].id, "elem_0001")
        self.assertEqual(candidates[0].bbox, [4040, 30, 4090, 60])
        self.assertEqual(candidates[0].center, [4065, 45])
        self.assertEqual(candidates[0].source, "ocr")

    def test_uia_bbox_is_kept_as_screen_coordinates(self):
        candidates = build_ui_candidates(
            uia_elements=[
                UIAElement(
                    name="搜索",
                    role="EditControl",
                    bbox=[4200, 80, 4480, 128],
                    enabled=True,
                    visible=True,
                )
            ]
        )

        self.assertEqual(candidates[0].bbox, [4200, 80, 4480, 128])
        self.assertEqual(candidates[0].center, [4340, 104])
        self.assertTrue(candidates[0].editable)

    def test_ocr_and_uia_are_merged_when_text_and_position_match(self):
        candidates = build_ui_candidates(
            ocr_elements=[
                OCRTextBlock(text="搜索", bbox=[180, 100, 230, 130], confidence=0.91),
            ],
            uia_elements=[
                UIAElement(
                    name="搜索",
                    role="EditControl",
                    bbox=[4180, 95, 4450, 135],
                    enabled=True,
                    visible=True,
                )
            ],
            screen_origin=[4000, 0],
            config=CandidateBuilderConfig(merge_center_distance=260),
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source, "merged")
        self.assertEqual(candidates[0].text, "搜索")
        self.assertEqual(candidates[0].role, "EditControl")
        self.assertTrue(candidates[0].editable)
        self.assertEqual(candidates[0].bbox, [4180, 95, 4450, 135])

    def test_build_ui_state_from_elements_outputs_summary(self):
        ui_state = build_ui_state_from_elements(
            screenshot_path="artifacts/runs/run_demo/screenshots/step_001_before.png",
            ocr_elements=[
                OCRTextBlock(text="消息", bbox=[40, 50, 90, 80], confidence=0.92),
            ],
            uia_elements=[
                UIAElement(
                    name="搜索",
                    role="EditControl",
                    bbox=[4200, 80, 4480, 128],
                    enabled=True,
                    visible=True,
                )
            ],
            screen_origin=[4000, -20],
        )

        payload = ui_state.to_dict()

        self.assertEqual(set(payload.keys()), {"screenshot_path", "candidates", "page_summary"})
        self.assertEqual(payload["page_summary"]["candidate_count"], 2)
        self.assertTrue(payload["page_summary"]["has_search_box"])
        self.assertEqual(payload["candidates"][0]["id"], "elem_0001")

    def test_compacts_split_lark_ocr_label(self):
        candidates = build_ui_candidates(
            ocr_elements=[
                OCRTextBlock(text="云", bbox=[90, 360, 116, 386], confidence=0.93),
                OCRTextBlock(text="文", bbox=[128, 360, 154, 386], confidence=0.91),
                OCRTextBlock(text="档", bbox=[156, 360, 178, 386], confidence=0.89),
            ]
        )

        merged_label = next(candidate for candidate in candidates if candidate.text == "云文档")
        self.assertEqual(merged_label.source, "merged")
        self.assertEqual(merged_label.role, "text")
        self.assertFalse(merged_label.clickable)
        self.assertEqual(merged_label.bbox, [90, 360, 178, 386])

    def test_lark_desktop_hints_add_clickable_main_nav_rows(self):
        candidates = build_ui_candidates(
            ocr_elements=[
                OCRTextBlock(text="消息", bbox=[90, 150, 140, 178], confidence=0.96),
                OCRTextBlock(text="知识", bbox=[90, 220, 145, 248], confidence=0.95),
                OCRTextBlock(text="问答", bbox=[154, 220, 200, 248], confidence=0.96),
                OCRTextBlock(text="云", bbox=[90, 290, 116, 316], confidence=0.93),
                OCRTextBlock(text="文", bbox=[128, 290, 154, 316], confidence=0.91),
                OCRTextBlock(text="档", bbox=[156, 290, 178, 316], confidence=0.89),
                OCRTextBlock(text="工作", bbox=[90, 360, 150, 386], confidence=0.93),
                OCRTextBlock(text="台", bbox=[154, 360, 180, 386], confidence=0.91),
            ]
        )

        nav_items = {
            candidate.text: candidate
            for candidate in candidates
            if candidate.role == "LarkMainNavItem"
        }

        self.assertIn("云文档", nav_items)
        self.assertIn("工作台", nav_items)
        self.assertTrue(nav_items["云文档"].clickable)
        self.assertGreater(nav_items["云文档"].bbox[2] - nav_items["云文档"].bbox[0], 300)

    def test_lark_desktop_hints_promote_meaningful_nav_prefix(self):
        cloud_docs = "\u4e91\u6587\u6863"
        candidates = build_ui_candidates(
            ocr_elements=[
                OCRTextBlock(text="\u641c\u7d22", bbox=[90, 80, 140, 108], confidence=0.92),
                OCRTextBlock(text="\u6d88\u606f", bbox=[90, 150, 140, 178], confidence=0.96),
                OCRTextBlock(text="\u4e91\u6587", bbox=[90, 290, 150, 316], confidence=0.89),
                OCRTextBlock(text="\u5de5\u4f5c\u53f0", bbox=[90, 360, 180, 386], confidence=0.93),
            ]
        )

        nav_item = next(
            candidate
            for candidate in candidates
            if candidate.role == "LarkMainNavItem" and candidate.text == cloud_docs
        )

        self.assertTrue(nav_item.clickable)
        self.assertGreater(nav_item.bbox[2] - nav_item.bbox[0], 300)

    def test_lark_desktop_hints_do_not_promote_single_char_nav_prefix(self):
        cloud_docs = "\u4e91\u6587\u6863"
        candidates = build_ui_candidates(
            ocr_elements=[
                OCRTextBlock(text="\u641c\u7d22", bbox=[90, 80, 140, 108], confidence=0.92),
                OCRTextBlock(text="\u6d88\u606f", bbox=[90, 150, 140, 178], confidence=0.96),
                OCRTextBlock(text="\u6587", bbox=[90, 290, 118, 316], confidence=0.89),
                OCRTextBlock(text="\u5de5\u4f5c\u53f0", bbox=[90, 360, 180, 386], confidence=0.93),
            ]
        )

        self.assertFalse(
            any(
                candidate.role == "LarkMainNavItem" and candidate.text == cloud_docs
                for candidate in candidates
            )
        )

    def test_lark_desktop_hints_add_editable_search_box(self):
        candidates = build_ui_candidates(
            ocr_elements=[
                OCRTextBlock(text="搜索", bbox=[90, 80, 140, 108], confidence=0.92),
                OCRTextBlock(text="(Ctrl", bbox=[150, 82, 195, 108], confidence=0.80),
                OCRTextBlock(text="K)", bbox=[220, 82, 245, 108], confidence=0.90),
                OCRTextBlock(text="消息", bbox=[90, 150, 140, 178], confidence=0.96),
                OCRTextBlock(text="知识", bbox=[90, 220, 145, 248], confidence=0.95),
                OCRTextBlock(text="问答", bbox=[154, 220, 200, 248], confidence=0.96),
                OCRTextBlock(text="云", bbox=[90, 290, 116, 316], confidence=0.93),
                OCRTextBlock(text="文", bbox=[128, 290, 154, 316], confidence=0.91),
                OCRTextBlock(text="档", bbox=[156, 290, 178, 316], confidence=0.89),
            ]
        )

        search_box = next(candidate for candidate in candidates if candidate.role == "LarkSearchBox")
        self.assertEqual(search_box.text, "搜索 (Ctrl+K)")
        self.assertTrue(search_box.clickable)
        self.assertTrue(search_box.editable)
        self.assertGreater(search_box.bbox[2] - search_box.bbox[0], 250)

    def test_lark_desktop_hints_add_chat_list_row_for_group_avatar_text(self):
        candidates = build_ui_candidates(
            ocr_elements=[
                OCRTextBlock(text="分组", bbox=[450, 80, 510, 110], confidence=0.95),
                OCRTextBlock(text="消息", bbox=[450, 150, 505, 178], confidence=0.96),
                OCRTextBlock(text="群组", bbox=[450, 300, 505, 328], confidence=0.96),
                OCRTextBlock(text="云文档", bbox=[90, 290, 178, 316], confidence=0.93),
                OCRTextBlock(text="群", bbox=[710, 320, 735, 350], confidence=0.87),
                OCRTextBlock(text="谢智华", bbox=[780, 324, 845, 350], confidence=0.83),
            ]
        )

        chat_items = [candidate for candidate in candidates if candidate.role == "LarkChatListItem"]

        self.assertEqual(len(chat_items), 1)
        self.assertEqual(chat_items[0].text, "群")
        self.assertTrue(chat_items[0].clickable)
        self.assertEqual(chat_items[0].bbox[0], 716)
        self.assertEqual(chat_items[0].bbox[2], 1616)

    def test_lark_desktop_hints_keep_multiple_group_chat_rows_separate(self):
        candidates = build_ui_candidates(
            ocr_elements=[
                OCRTextBlock(text="分组", bbox=[450, 80, 510, 110], confidence=0.95),
                OCRTextBlock(text="消息", bbox=[450, 150, 505, 178], confidence=0.96),
                OCRTextBlock(text="群组", bbox=[450, 300, 505, 328], confidence=0.96),
                OCRTextBlock(text="云文档", bbox=[90, 290, 178, 316], confidence=0.93),
                OCRTextBlock(text="群", bbox=[710, 320, 735, 350], confidence=0.87),
                OCRTextBlock(text="群", bbox=[710, 440, 735, 470], confidence=0.86),
            ]
        )

        chat_items = [candidate for candidate in candidates if candidate.role == "LarkChatListItem"]

        self.assertEqual(len(chat_items), 2)
        self.assertEqual([candidate.text for candidate in chat_items], ["群", "群"])

    def test_lark_desktop_hints_infer_chat_list_from_message_header(self):
        candidates = build_ui_candidates(
            ocr_elements=[
                OCRTextBlock(text="搜索", bbox=[90, 80, 140, 108], confidence=0.92),
                OCRTextBlock(text="云文档", bbox=[90, 290, 178, 316], confidence=0.93),
                OCRTextBlock(text="多维表格", bbox=[90, 430, 200, 456], confidence=0.93),
                OCRTextBlock(text="分组", bbox=[640, 80, 710, 112], confidence=0.88),
                OCRTextBlock(text="”消息", bbox=[906, 84, 970, 123], confidence=0.85),
                OCRTextBlock(text="群", bbox=[940, 357, 960, 377], confidence=0.87),
                OCRTextBlock(text="谢智华", bbox=[1014, 368, 1081, 389], confidence=0.83),
            ]
        )

        chat_items = [candidate for candidate in candidates if candidate.role == "LarkChatListItem"]

        self.assertEqual(len(chat_items), 1)
        self.assertEqual(chat_items[0].text, "群")
        self.assertEqual(chat_items[0].bbox[0], 906)
        self.assertEqual(chat_items[0].bbox[2], 1806)


if __name__ == "__main__":
    unittest.main()
