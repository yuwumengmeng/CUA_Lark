"""ABC acceptance annotation artifact tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tests.testABC.run_joint_abc_acceptance import (
    StepRecord,
    draw_candidate_annotations,
    render_markdown,
    selected_candidate_summary,
)


class JointABCAnnotationTests(unittest.TestCase):
    def test_draw_candidate_annotations_highlights_selected_candidate_and_click_point(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            screenshot_path = root / "before.png"
            annotated_path = root / "annotations" / "step_001_candidates.png"
            Image.new("RGB", (100, 80), color=(255, 255, 255)).save(screenshot_path)
            ui_state = {
                "screenshot_path": str(screenshot_path),
                "page_summary": {"candidate_count": 2},
                "candidates": [
                    {
                        "id": "elem_001",
                        "text": "first",
                        "role": "text",
                        "source": "ocr",
                        "confidence": 0.95,
                        "bbox": [4010, 510, 4030, 530],
                        "center": [4020, 520],
                    },
                    {
                        "id": "elem_002",
                        "text": "target",
                        "role": "button",
                        "source": "merged",
                        "confidence": 0.91,
                        "bbox": [4050, 540, 4080, 570],
                        "center": [4065, 555],
                    },
                ],
            }

            result = draw_candidate_annotations(
                ui_state=ui_state,
                action={
                    "action_type": "click",
                    "target_element_id": "elem_002",
                    "x": 4065,
                    "y": 555,
                },
                annotated_path=annotated_path,
                top_n=1,
                screenshot_metadata={"screen_origin": [4000, 500]},
            )

            self.assertEqual(result, annotated_path)
            self.assertTrue(annotated_path.exists())
            with Image.open(annotated_path) as image:
                self.assertEqual(image.getpixel((50, 40)), (255, 0, 0))
                self.assertEqual(image.getpixel((65, 55)), (0, 255, 255))

    def test_step_record_and_markdown_include_selected_candidate_and_annotation(self):
        record = StepRecord(
            scenario_key="demo",
            scenario_title="Demo",
            step_idx=1,
            planner_step_idx=0,
        )
        record.annotation_path = "artifacts/runs/demo/annotations/step_001_candidates.png"
        record.selected_candidate = {
            "id": "elem_002",
            "text": "target",
            "role": "button",
            "source": "merged",
            "confidence": 0.91,
            "bbox": [50, 40, 80, 70],
            "center": [65, 55],
        }
        payload = record.to_dict()

        self.assertEqual(payload["annotation_path"], record.annotation_path)
        self.assertEqual(payload["selected_candidate"]["id"], "elem_002")

        markdown = render_markdown(
            {
                "run_id": "demo",
                "status": "success",
                "real_gui": False,
                "strict_after_checks": False,
                "passed_count": 1,
                "total_count": 1,
                "metrics": {},
                "results": [
                    {
                        "title": "Demo",
                        "status": "success",
                        "failure_reason": None,
                        "steps": [payload],
                    }
                ],
            }
        )

        self.assertIn("annotation", markdown)
        self.assertIn("selected id", markdown)
        self.assertIn("elem_002", markdown)

    def test_selected_candidate_summary_uses_target_element_id(self):
        ui_state = {
            "candidates": [
                {"id": "elem_001", "text": "first"},
                {
                    "id": "elem_002",
                    "text": "target",
                    "role": "button",
                    "source": "merged",
                    "confidence": 0.91,
                    "bbox": [50, 40, 80, 70],
                    "center": [65, 55],
                },
            ]
        }

        summary = selected_candidate_summary(ui_state, "elem_002")

        self.assertIsNotNone(summary)
        self.assertEqual(summary["text"], "target")
        self.assertEqual(summary["bbox"], [50, 40, 80, 70])


if __name__ == "__main__":
    unittest.main()
