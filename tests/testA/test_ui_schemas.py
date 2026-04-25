"""成员 A 第 0 步接口与数据格式对齐测试。

测试目标：
本文件验证 A 输出给 B/C 的稳定数据契约，包括：
1. `UICandidate` 固定字段、bbox/center 坐标规则和点击点解析。
2. OCR 文本块字段能转换为首版 candidate。
3. `PageSummary` 固定字段和基础摘要生成规则。
4. `UIState` 顶层接口固定为 `screenshot_path / candidates / page_summary`。
5. UI state JSON 按统一 artifact 路径保存。

直接运行：
```powershell
python tests\testA\test_ui_schemas.py
```

通过 pytest 运行：
```powershell
python -m pytest tests\testA\test_ui_schemas.py
```
"""

import json
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

schemas_module = import_module("schemas")
SOURCE_OCR = schemas_module.SOURCE_OCR
UICandidate = schemas_module.UICandidate
UIState = schemas_module.UIState
PageSummary = schemas_module.PageSummary
UISchemaError = schemas_module.UISchemaError
bbox_center = schemas_module.bbox_center
candidate_from_ocr = schemas_module.candidate_from_ocr
candidate_from_uia = schemas_module.candidate_from_uia
offset_bbox = schemas_module.offset_bbox
resolve_click_point = schemas_module.resolve_click_point
save_ui_state_json = schemas_module.save_ui_state_json
ui_state_artifact_path = schemas_module.ui_state_artifact_path


class UISchemaTests(unittest.TestCase):
    def test_ui_candidate_outputs_stable_fields_and_center(self):
        candidate = UICandidate(
            id="elem_001",
            text="搜索",
            bbox=[10, 20, 50, 40],
            role="text",
            clickable=True,
            editable=False,
            source=SOURCE_OCR,
            confidence=0.91,
        )

        self.assertEqual(candidate.center, [30, 30])
        self.assertEqual(resolve_click_point(candidate), (30, 30))
        self.assertEqual(
            candidate.to_dict(),
            {
                "id": "elem_001",
                "text": "搜索",
                "bbox": [10, 20, 50, 40],
                "center": [30, 30],
                "role": "text",
                "clickable": True,
                "editable": False,
                "source": "ocr",
                "confidence": 0.91,
            },
        )

    def test_candidate_from_ocr_uses_expected_defaults(self):
        candidate = candidate_from_ocr(
            candidate_id="ocr_001",
            text="测试群",
            bbox=[60, 20, 120, 44],
            confidence=0.88,
        )

        self.assertEqual(candidate.role, "text")
        self.assertEqual(candidate.source, "ocr")
        self.assertEqual(candidate.center, [90, 32])
        self.assertFalse(candidate.clickable)
        self.assertFalse(candidate.editable)

    def test_candidate_from_ocr_offsets_bbox_to_screen_coordinates(self):
        candidate = candidate_from_ocr(
            candidate_id="ocr_001",
            text="搜索",
            bbox=[10, 20, 50, 40],
            confidence=0.88,
            screen_origin=[4000, -20],
        )

        self.assertEqual(candidate.bbox, [4010, 0, 4050, 20])
        self.assertEqual(candidate.center, [4030, 10])
        self.assertEqual(resolve_click_point(candidate), (4030, 10))
        self.assertEqual(offset_bbox([10, 20, 50, 40], [4000, -20]), [4010, 0, 4050, 20])

    def test_candidate_from_uia_uses_screen_coordinates_without_offset(self):
        candidate = candidate_from_uia(
            candidate_id="uia_001",
            name="搜索",
            role="EditControl",
            bbox=[4200, 80, 4480, 128],
            enabled=True,
            visible=True,
        )

        self.assertEqual(candidate.text, "搜索")
        self.assertEqual(candidate.bbox, [4200, 80, 4480, 128])
        self.assertEqual(candidate.center, [4340, 104])
        self.assertEqual(candidate.source, "uia")
        self.assertTrue(candidate.editable)
        self.assertFalse(candidate.clickable)

    def test_invalid_candidate_data_is_rejected(self):
        with self.assertRaises(UISchemaError):
            UICandidate(id="", text="搜索", bbox=[0, 0, 10, 10])

        with self.assertRaises(UISchemaError):
            UICandidate(id="elem_001", text="搜索", bbox=[10, 10, 5, 20])

        with self.assertRaises(UISchemaError):
            UICandidate(id="elem_001", text="搜索", bbox=[0, 0, 10, 10], source="raw")

        with self.assertRaises(UISchemaError):
            UICandidate(id="elem_001", text="搜索", bbox=[0, 0, 10, 10], confidence=1.5)

    def test_bbox_center_uses_integer_center_rule(self):
        self.assertEqual(bbox_center([0, 0, 11, 21]), [5, 10])

    def test_page_summary_and_ui_state_roundtrip(self):
        candidates = [
            UICandidate(id="elem_001", text="搜索", bbox=[10, 20, 50, 40], confidence=0.9),
            UICandidate(id="elem_002", text="测试群", bbox=[60, 20, 120, 44], confidence=0.8),
        ]
        summary = PageSummary.from_candidates(candidates)
        ui_state = UIState(
            screenshot_path="artifacts/runs/run_demo/screenshots/step_001_before.png",
            candidates=candidates,
            page_summary=summary,
        )
        restored = UIState.from_dict(ui_state.to_dict())

        self.assertTrue(summary.has_search_box)
        self.assertEqual(summary.candidate_count, 2)
        self.assertEqual(set(ui_state.to_dict().keys()), {"screenshot_path", "candidates", "page_summary"})
        self.assertEqual(restored.candidates[1].text, "测试群")

    def test_page_summary_detects_search_box_from_ctrl_k_tokens(self):
        candidates = [
            UICandidate(id="elem_001", text="(Ctrl", bbox=[10, 20, 50, 40], confidence=0.9),
            UICandidate(id="elem_002", text="K)", bbox=[60, 20, 120, 44], confidence=0.8),
        ]

        summary = PageSummary.from_candidates(candidates)

        self.assertTrue(summary.has_search_box)

    def test_page_summary_detects_search_box_from_split_chars(self):
        candidates = [
            UICandidate(id="elem_001", text="搜", bbox=[10, 20, 30, 40], confidence=0.9),
            UICandidate(id="elem_002", text="索", bbox=[34, 20, 54, 40], confidence=0.8),
        ]

        summary = PageSummary.from_candidates(candidates)

        self.assertTrue(summary.has_search_box)

    def test_ui_state_artifact_path_and_save_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = UICandidate(
                id="elem_001",
                text="搜索",
                bbox=[10, 20, 50, 40],
                confidence=0.9,
            )
            ui_state = UIState(
                screenshot_path="screenshots/step_001_before.png",
                candidates=[candidate],
                page_summary=PageSummary.from_candidates([candidate]),
            )

            expected_path = ui_state_artifact_path(
                run_id="run_demo",
                step_idx=1,
                phase="before",
                artifact_root=temp_dir,
            )
            saved_path = save_ui_state_json(
                ui_state,
                run_id="run_demo",
                step_idx=1,
                phase="before",
                artifact_root=temp_dir,
            )

            self.assertEqual(saved_path, expected_path)
            self.assertEqual(saved_path.name, "ui_state_step_001_before.json")
            payload = json.loads(saved_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["candidates"][0]["center"], [30, 30])


if __name__ == "__main__":
    unittest.main()
