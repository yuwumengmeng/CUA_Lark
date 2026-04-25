"""成员 A 第 10 项验收脚本匹配逻辑测试。"""

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

schemas_module = __import__("schemas", fromlist=["UICandidate", "UIState", "PageSummary"])
UICandidate = schemas_module.UICandidate
UIState = schemas_module.UIState
PageSummary = schemas_module.PageSummary
ocr_module = __import__("perception.ocr", fromlist=["OCRTextBlock"])
OCRTextBlock = ocr_module.OCRTextBlock

acceptance_path = ROOT / "tests" / "testA" / "run_perception_acceptance.py"
spec = importlib.util.spec_from_file_location("testA_run_perception_acceptance", acceptance_path)
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to load run_perception_acceptance module")
acceptance_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = acceptance_module
spec.loader.exec_module(acceptance_module)

ScenarioResult = acceptance_module.ScenarioResult
SCENARIOS = acceptance_module.SCENARIOS
match_ocr_elements = acceptance_module.match_ocr_elements
match_candidates = acceptance_module.match_candidates
text_matches = acceptance_module.text_matches


class PerceptionAcceptanceLogicTests(unittest.TestCase):
    def test_single_letter_keyword_requires_exact_token(self):
        self.assertTrue(text_matches("K)", ("K",)))
        self.assertFalse(text_matches("KB", ("K",)))
        self.assertFalse(text_matches("云", ("云文档",)))

    def test_left_nav_allows_split_ocr_keywords(self):
        left_nav_scenario = next(scenario for scenario in SCENARIOS if scenario.key == "left_nav")
        ocr_elements = [
            OCRTextBlock(text="云", bbox=[10, 10, 20, 30], confidence=0.95),
            OCRTextBlock(text="文", bbox=[22, 10, 32, 30], confidence=0.95),
            OCRTextBlock(text="档", bbox=[34, 10, 44, 30], confidence=0.95),
        ]

        hits = match_ocr_elements(left_nav_scenario, ocr_elements)

        self.assertEqual([hit["text"] for hit in hits], ["云", "文", "档"])

    def test_left_nav_allows_locatable_ocr_text_without_clickable_flag(self):
        left_nav_scenario = next(scenario for scenario in SCENARIOS if scenario.key == "left_nav")
        candidates = [
            UICandidate(
                id="elem_0001",
                text="云",
                bbox=[10, 10, 20, 30],
                role="text",
                clickable=False,
                editable=False,
                source="ocr",
                confidence=0.95,
            )
        ]
        ui_state = UIState(
            screenshot_path="artifacts/runs/run_demo/screenshots/step_001_before.png",
            candidates=candidates,
            page_summary=PageSummary.from_candidates(candidates),
        )

        hits = match_candidates(
            left_nav_scenario,
            ui_state,
            screen_origin=[0, 0],
            screenshot_width=800,
            screenshot_height=600,
        )

        self.assertEqual([candidate["text"] for candidate in hits], ["云"])

    def test_docs_entry_allows_locatable_ocr_text_without_clickable_flag(self):
        docs_scenario = next(scenario for scenario in SCENARIOS if scenario.key == "docs_entry")
        candidates = [
            UICandidate(
                id="elem_0001",
                text="文档",
                bbox=[10, 10, 60, 30],
                role="text",
                clickable=False,
                editable=False,
                source="ocr",
                confidence=0.95,
            )
        ]
        ui_state = UIState(
            screenshot_path="artifacts/runs/run_demo/screenshots/step_004_before.png",
            candidates=candidates,
            page_summary=PageSummary.from_candidates(candidates),
        )

        hits = match_candidates(
            docs_scenario,
            ui_state,
            screen_origin=[0, 0],
            screenshot_width=800,
            screenshot_height=600,
        )

        self.assertEqual([candidate["text"] for candidate in hits], ["文档"])

    def test_search_scenario_allows_text_fallback_without_editable_flag(self):
        search_scenario = next(scenario for scenario in SCENARIOS if scenario.key == "search_box")
        candidates = [
            UICandidate(
                id="elem_0001",
                text="KB",
                bbox=[10, 10, 50, 30],
                role="text",
                clickable=False,
                editable=False,
                source="ocr",
                confidence=0.95,
            ),
            UICandidate(
                id="elem_0002",
                text="搜索",
                bbox=[60, 10, 120, 30],
                role="text",
                clickable=False,
                editable=False,
                source="ocr",
                confidence=0.95,
            ),
        ]
        ui_state = UIState(
            screenshot_path="artifacts/runs/run_demo/screenshots/step_001_before.png",
            candidates=candidates,
            page_summary=PageSummary.from_candidates(candidates),
        )

        hits = match_candidates(
            search_scenario,
            ui_state,
            screen_origin=[0, 0],
            screenshot_width=800,
            screenshot_height=600,
        )

        hit_texts = [candidate["text"] for candidate in hits]
        self.assertIn("搜索", hit_texts)
        self.assertNotIn("KB", hit_texts)

    def test_scenario_result_passes_when_only_one_modality_hits(self):
        scenario = next(s for s in SCENARIOS if s.key == "back_button")
        result = ScenarioResult(
            scenario=scenario,
            step_idx=1,
            screenshot_path="a.png",
            ui_state_path="a.json",
            annotated_path="a_annotated.png",
            screenshot_ok=True,
            ocr_ok=False,
            uia_ok=True,
            candidates_ok=True,
            page_summary_ok=True,
            clickable_center_ok=True,
            ocr_hits=[],
            uia_hits=[{"role": "ButtonControl"}],
            candidate_hits=[{"center": [1, 1], "bbox": [0, 0, 2, 2]}],
            page_summary={"candidate_count": 1},
        )

        self.assertTrue(result.modality_ok)
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
