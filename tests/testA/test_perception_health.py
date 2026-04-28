"""Member A Task 2.7 perception health assessment tests.

Tests: PerceptionHealth schema, assess_perception_health for various
failure modes, VLM instruction enrichment, UIState integration.
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

health_module = import_module("perception.health")
schemas_health = import_module("schemas.perception_health")
ui_state_module = import_module("perception.ui_state")

assess_perception_health = health_module.assess_perception_health
build_perception_health_summary = health_module.build_perception_health_summary
PerceptionHealth = schemas_health.PerceptionHealth

# Constants
HEALTH_HEALTHY = schemas_health.HEALTH_HEALTHY
HEALTH_DEGRADED = schemas_health.HEALTH_DEGRADED
HEALTH_UNHEALTHY = schemas_health.HEALTH_UNHEALTHY
QUALITY_GOOD = schemas_health.QUALITY_GOOD
QUALITY_DEGRADED = schemas_health.QUALITY_DEGRADED
QUALITY_POOR = schemas_health.QUALITY_POOR
COVERAGE_ADEQUATE = schemas_health.COVERAGE_ADEQUATE
COVERAGE_NONE = schemas_health.COVERAGE_NONE
COVERAGE_PARTIAL = schemas_health.COVERAGE_PARTIAL
FLAG_OCR_SPARSE = schemas_health.FLAG_OCR_SPARSE
FLAG_OCR_LOW_CONFIDENCE = schemas_health.FLAG_OCR_LOW_CONFIDENCE
FLAG_UIA_SHELL_DOMINATED = schemas_health.FLAG_UIA_SHELL_DOMINATED
FLAG_UIA_EMPTY = schemas_health.FLAG_UIA_EMPTY
FLAG_NO_SEARCH_TARGET = schemas_health.FLAG_NO_SEARCH_TARGET
FLAG_NO_INPUT_TARGET = schemas_health.FLAG_NO_INPUT_TARGET
FLAG_NO_CLICKABLE = schemas_health.FLAG_NO_CLICKABLE
FLAG_ALL_SHELL = schemas_health.FLAG_ALL_SHELL
FALLBACK_NONE = schemas_health.FALLBACK_NONE
FALLBACK_USE_VLM_VISUAL = schemas_health.FALLBACK_USE_VLM_VISUAL
FALLBACK_USE_UIA_ONLY = schemas_health.FALLBACK_USE_UIA_ONLY
FALLBACK_USE_OCR_ONLY = schemas_health.FALLBACK_USE_OCR_ONLY


# ── Fake element classes ────────────────────────────────────────────────────

class FakeOCRTextBlock:
    def __init__(self, text: str, bbox: list[int], confidence: float = 0.8):
        self.text = text
        self.bbox = bbox
        self.confidence = confidence


class FakeUIAElement:
    def __init__(self, name: str, role: str, bbox: list[int],
                 enabled: bool = True, visible: bool = True):
        self.name = name
        self.role = role
        self.bbox = bbox
        self.enabled = enabled
        self.visible = visible


class FakeUICandidate:
    def __init__(self, id: str, text: str = "", role: str = "",
                 clickable: bool = False, editable: bool = False,
                 source: str = "merged", bbox: list[int] | None = None,
                 center: list[int] | None = None):
        self.id = id
        self.text = text
        self.role = role
        self.clickable = clickable
        self.editable = editable
        self.source = source
        self.bbox = bbox or [0, 0, 100, 100]
        self.center = center or [50, 50]


# ── PerceptionHealth Schema Tests ────────────────────────────────────────────


class TestPerceptionHealthSchema(unittest.TestCase):

    def test_default_is_healthy(self):
        health = PerceptionHealth()
        self.assertEqual(health.overall_health, HEALTH_HEALTHY)
        self.assertEqual(health.ocr_quality, QUALITY_GOOD)
        self.assertEqual(health.failure_flags, [])
        self.assertEqual(health.suggested_fallback, FALLBACK_NONE)

    def test_to_dict_roundtrip(self):
        health = PerceptionHealth(
            ocr_element_count=10,
            ocr_quality=QUALITY_GOOD,
            uia_element_count=5,
            uia_quality=QUALITY_DEGRADED,
            shell_control_count=3,
            shell_control_ratio=0.6,
            merged_candidate_count=12,
            clickable_count=5,
            editable_count=2,
            textless_candidate_count=1,
            search_target_coverage=COVERAGE_ADEQUATE,
            input_target_coverage=COVERAGE_PARTIAL,
            overall_health=HEALTH_DEGRADED,
            failure_flags=["ocr_sparse", "no_input_target"],
            suggested_fallback=FALLBACK_USE_VLM_VISUAL,
            summary="test summary",
        )
        d = health.to_dict()
        self.assertEqual(d["ocr_element_count"], 10)
        self.assertEqual(d["ocr_quality"], QUALITY_GOOD)
        self.assertEqual(d["failure_flags"], ["ocr_sparse", "no_input_target"])
        self.assertEqual(d["suggested_fallback"], FALLBACK_USE_VLM_VISUAL)

        restored = PerceptionHealth.from_dict(d)
        self.assertEqual(restored.ocr_element_count, 10)
        self.assertEqual(restored.overall_health, HEALTH_DEGRADED)
        self.assertEqual(restored.failure_flags, ["ocr_sparse", "no_input_target"])

    def test_from_dict_partial(self):
        health = PerceptionHealth.from_dict({})
        self.assertEqual(health.ocr_element_count, 0)
        self.assertEqual(health.overall_health, HEALTH_HEALTHY)

    def test_clamps_shell_ratio(self):
        health = PerceptionHealth(shell_control_ratio=1.5)
        self.assertEqual(health.shell_control_ratio, 1.0)
        health = PerceptionHealth(shell_control_ratio=-0.5)
        self.assertEqual(health.shell_control_ratio, 0.0)

    def test_rejects_invalid_quality_levels(self):
        health = PerceptionHealth(ocr_quality="excellent", uia_quality="bad")
        self.assertEqual(health.ocr_quality, QUALITY_GOOD)
        self.assertEqual(health.uia_quality, QUALITY_GOOD)

    def test_filters_invalid_failure_flags(self):
        health = PerceptionHealth(
            failure_flags=["ocr_sparse", "made_up_flag", "no_clickable"],
        )
        self.assertEqual(health.failure_flags, ["ocr_sparse", "no_clickable"])

    def test_rejects_invalid_fallback(self):
        health = PerceptionHealth(suggested_fallback="do_something_crazy")
        self.assertEqual(health.suggested_fallback, FALLBACK_NONE)

    def test_clamps_negative_counts(self):
        health = PerceptionHealth(
            ocr_element_count=-5,
            clickable_count=-3,
            merged_candidate_count=-1,
        )
        self.assertEqual(health.ocr_element_count, 0)
        self.assertEqual(health.clickable_count, 0)
        self.assertEqual(health.merged_candidate_count, 0)


# ── Health Assessment Tests ──────────────────────────────────────────────────


class TestAssessPerceptionHealth(unittest.TestCase):

    def test_healthy_with_good_ocr_and_uia(self):
        ocr = [FakeOCRTextBlock(f"text{i}", [0, 0, 100, 20]) for i in range(10)]
        uia = [FakeUIAElement(f"elem{i}", "Button", [0, 0, 100, 20]) for i in range(8)]
        candidates = [
            FakeUICandidate("c1", text="搜索", role="EditControl", clickable=True, editable=True),
            FakeUICandidate("c2", text="消息输入", role="EditControl", clickable=True, editable=True),
            FakeUICandidate("c3", text="发送", role="Button", clickable=True),
        ]

        health = assess_perception_health(
            ocr_elements=ocr, uia_elements=uia, candidates=candidates,
        )
        self.assertEqual(health.overall_health, HEALTH_HEALTHY)
        self.assertEqual(health.ocr_quality, QUALITY_GOOD)
        self.assertEqual(health.uia_quality, QUALITY_GOOD)
        self.assertEqual(health.failure_flags, [])
        self.assertEqual(health.suggested_fallback, FALLBACK_NONE)
        self.assertEqual(health.search_target_coverage, COVERAGE_ADEQUATE)
        self.assertEqual(health.input_target_coverage, COVERAGE_ADEQUATE)

    def test_ocr_sparse_detected(self):
        ocr = [FakeOCRTextBlock("only one", [0, 0, 100, 20], confidence=0.3)]
        uia = [FakeUIAElement("btn", "Button", [0, 0, 100, 20])]
        candidates = [
            FakeUICandidate("c1", text="only one", role="Text", clickable=False),
        ]

        health = assess_perception_health(
            ocr_elements=ocr, uia_elements=uia, candidates=candidates,
        )
        self.assertEqual(health.ocr_quality, QUALITY_POOR)
        self.assertIn(FLAG_OCR_SPARSE, health.failure_flags)
        self.assertIn(FLAG_NO_CLICKABLE, health.failure_flags)
        self.assertIn(FLAG_NO_SEARCH_TARGET, health.failure_flags)
        self.assertEqual(health.overall_health, HEALTH_UNHEALTHY)

    def test_ocr_empty_detected(self):
        uia = [FakeUIAElement("btn", "Button", [0, 0, 100, 20])]
        candidates = [
            FakeUICandidate("c1", text="btn", role="Button", clickable=True),
        ]

        health = assess_perception_health(
            ocr_elements=[], uia_elements=uia, candidates=candidates,
        )
        self.assertEqual(health.ocr_quality, QUALITY_POOR)
        self.assertEqual(health.ocr_element_count, 0)
        self.assertIn(FLAG_OCR_SPARSE, health.failure_flags)

    def test_uia_shell_dominated_detected(self):
        ocr = [FakeOCRTextBlock("text", [0, 0, 100, 20])]
        # Mostly shell controls
        uia = [
            FakeUIAElement("Title", "TitleBar", [0, 0, 1000, 30]),
            FakeUIAElement("Menu", "MenuBar", [0, 30, 1000, 60]),
            FakeUIAElement("Scroll", "ScrollBar", [980, 0, 1000, 800]),
            FakeUIAElement("bar", "ToolBar", [0, 60, 1000, 100]),
        ]
        candidates = [
            FakeUICandidate("c1", text="text", role="Text", clickable=False),
        ]

        health = assess_perception_health(
            ocr_elements=ocr, uia_elements=uia, candidates=candidates,
        )
        self.assertEqual(health.uia_quality, QUALITY_POOR)
        self.assertEqual(health.shell_control_count, 4)
        self.assertEqual(health.shell_control_ratio, 1.0)
        self.assertIn(FLAG_UIA_SHELL_DOMINATED, health.failure_flags)
        self.assertEqual(health.suggested_fallback, FALLBACK_USE_VLM_VISUAL)

    def test_uia_empty_detected(self):
        ocr = [FakeOCRTextBlock(f"text{i}", [0, 0, 100, 20]) for i in range(10)]
        candidates = [
            FakeUICandidate("c1", text="搜索", role="EditControl", clickable=True, editable=True),
            FakeUICandidate("c2", text="发送", role="Button", clickable=True),
        ]

        health = assess_perception_health(
            ocr_elements=ocr, uia_elements=[], candidates=candidates,
        )
        self.assertIn(FLAG_UIA_EMPTY, health.failure_flags)
        self.assertEqual(health.uia_quality, QUALITY_POOR)

    def test_all_shell_candidates_detected(self):
        ocr = [FakeOCRTextBlock("text", [0, 0, 100, 20])]
        uia = []
        # All clickable candidates are shell controls
        candidates = [
            FakeUICandidate("c1", text="", role="TitleBar", clickable=True),
            FakeUICandidate("c2", text="", role="ScrollBar", clickable=True),
        ]

        health = assess_perception_health(
            ocr_elements=ocr, uia_elements=uia, candidates=candidates,
        )
        self.assertIn(FLAG_ALL_SHELL, health.failure_flags)
        # Shell controls are still clickable, so no_clickable doesn't fire
        self.assertEqual(health.clickable_count, 2)

    def test_no_search_target_detected(self):
        ocr = [FakeOCRTextBlock(f"text{i}", [0, 0, 100, 20]) for i in range(10)]
        uia = [FakeUIAElement("btn", "Button", [0, 0, 100, 20])]
        candidates = [
            FakeUICandidate("c1", text="普通文本", role="Text", clickable=False),
            FakeUICandidate("c2", text="按钮", role="Button", clickable=True),
        ]

        health = assess_perception_health(
            ocr_elements=ocr, uia_elements=uia, candidates=candidates,
        )
        self.assertEqual(health.search_target_coverage, COVERAGE_NONE)
        self.assertIn(FLAG_NO_SEARCH_TARGET, health.failure_flags)

    def test_no_input_target_detected(self):
        ocr = [FakeOCRTextBlock(f"text{i}", [0, 0, 100, 20]) for i in range(10)]
        uia = [FakeUIAElement("btn", "Button", [0, 0, 100, 20])]
        candidates = [
            FakeUICandidate("c1", text="标题", role="Text", clickable=False),
        ]

        health = assess_perception_health(
            ocr_elements=ocr, uia_elements=uia, candidates=candidates,
        )
        self.assertEqual(health.input_target_coverage, COVERAGE_NONE)
        self.assertIn(FLAG_NO_INPUT_TARGET, health.failure_flags)

    def test_partial_search_coverage(self):
        ocr = [FakeOCRTextBlock("搜", [0, 0, 100, 20])]
        uia = []
        candidates = [
            FakeUICandidate("c1", text="搜", role="Text", clickable=True, editable=False),
        ]

        health = assess_perception_health(
            ocr_elements=ocr, uia_elements=uia, candidates=candidates,
        )
        self.assertEqual(health.search_target_coverage, COVERAGE_PARTIAL)

    def test_fallback_ocr_only_when_uia_shell_dominated(self):
        ocr = [FakeOCRTextBlock(f"t{i}", [0, 0, 100, 20], confidence=0.8) for i in range(10)]
        uia = [
            FakeUIAElement("TitleBar", "TitleBar", [0, 0, 1000, 30]),
            FakeUIAElement("ScrollBar", "ScrollBar", [980, 0, 1000, 800]),
        ]
        candidates = [
            FakeUICandidate("c1", text="搜索", role="EditControl", clickable=True, editable=True),
        ]

        health = assess_perception_health(
            ocr_elements=ocr, uia_elements=uia, candidates=candidates,
        )
        self.assertEqual(health.suggested_fallback, FALLBACK_USE_OCR_ONLY)

    def test_fallback_uia_only_when_ocr_sparse(self):
        ocr = []
        uia = [FakeUIAElement("btn", "Button", [0, 0, 100, 20]) for _ in range(5)]
        candidates = [
            FakeUICandidate("c1", text="btn", role="Button", clickable=True),
        ]

        health = assess_perception_health(
            ocr_elements=ocr, uia_elements=uia, candidates=candidates,
        )
        self.assertEqual(health.suggested_fallback, FALLBACK_USE_UIA_ONLY)

    def test_fallback_vlm_visual_when_both_degraded_and_no_targets(self):
        ocr = [FakeOCRTextBlock("x", [0, 0, 10, 10], confidence=0.2)]
        uia = [
            FakeUIAElement("TitleBar", "TitleBar", [0, 0, 1000, 30]),
            FakeUIAElement("ScrollBar", "ScrollBar", [980, 0, 1000, 800]),
        ]
        candidates = []

        health = assess_perception_health(
            ocr_elements=ocr, uia_elements=uia, candidates=candidates,
        )
        self.assertEqual(health.suggested_fallback, FALLBACK_USE_VLM_VISUAL)
        self.assertIn(FLAG_NO_SEARCH_TARGET, health.failure_flags)
        self.assertIn(FLAG_NO_INPUT_TARGET, health.failure_flags)

    def test_textless_candidates_counted(self):
        ocr = [FakeOCRTextBlock(f"t{i}", [0, 0, 100, 20]) for i in range(10)]
        uia = []
        candidates = [
            FakeUICandidate("c1", text="", role="Button", clickable=True),
            FakeUICandidate("c2", text="", role="Button", clickable=True),
            FakeUICandidate("c3", text="has text", role="Text"),
        ]

        health = assess_perception_health(
            ocr_elements=ocr, uia_elements=uia, candidates=candidates,
        )
        self.assertEqual(health.textless_candidate_count, 2)

    def test_ocr_low_confidence_flag(self):
        ocr = [FakeOCRTextBlock(f"t{i}", [0, 0, 100, 20], confidence=0.3) for i in range(6)]
        uia = [FakeUIAElement("btn", "Button", [0, 0, 100, 20])]
        candidates = [
            FakeUICandidate("c1", text="btn", role="Button", clickable=True),
        ]

        health = assess_perception_health(
            ocr_elements=ocr, uia_elements=uia, candidates=candidates,
        )
        self.assertIn(FLAG_OCR_LOW_CONFIDENCE, health.failure_flags)
        self.assertEqual(health.ocr_quality, QUALITY_DEGRADED)

    def test_summary_generated(self):
        health = assess_perception_health(
            ocr_elements=[FakeOCRTextBlock("t", [0, 0, 100, 20])],
            uia_elements=[FakeUIAElement("b", "Button", [0, 0, 100, 20])],
            candidates=[FakeUICandidate("c1", text="t", role="Text", clickable=False)],
        )
        self.assertIsInstance(health.summary, str)
        self.assertIn("OCR:", health.summary)
        self.assertIn("UIA:", health.summary)
        self.assertIn("health:", health.summary)

    def test_all_none_inputs_handled(self):
        health = assess_perception_health()
        self.assertIsInstance(health, PerceptionHealth)
        self.assertEqual(health.ocr_element_count, 0)
        self.assertEqual(health.uia_element_count, 0)
        self.assertEqual(health.merged_candidate_count, 0)


# ── VLM Instruction Enrichment Tests ─────────────────────────────────────────


class TestEnrichVLMInstruction(unittest.TestCase):

    def test_no_enrichment_when_healthy(self):
        health = PerceptionHealth(overall_health=HEALTH_HEALTHY)
        result = ui_state_module._enrich_vlm_instruction_with_health(
            base_instruction="Find the search box.",
            perception_health=health,
        )
        self.assertEqual(result, "Find the search box.")

    def test_enriches_with_health_warning(self):
        health = PerceptionHealth(
            overall_health=HEALTH_DEGRADED,
            failure_flags=[FLAG_OCR_SPARSE, FLAG_NO_SEARCH_TARGET],
        )
        result = ui_state_module._enrich_vlm_instruction_with_health(
            base_instruction="Find the search box.",
            perception_health=health,
        )
        self.assertIn("Perception health is degraded", result)
        self.assertIn("OCR returned very few text elements", result)
        self.assertIn("propose it as a visual_candidate", result)
        # Original instruction preserved
        self.assertIn("Find the search box.", result)

    def test_enriches_with_shell_warning(self):
        health = PerceptionHealth(
            overall_health=HEALTH_UNHEALTHY,
            failure_flags=[FLAG_UIA_SHELL_DOMINATED, FLAG_NO_CLICKABLE],
        )
        result = ui_state_module._enrich_vlm_instruction_with_health(
            base_instruction="Click the send button.",
            perception_health=health,
        )
        self.assertIn("Perception health is unhealthy", result)
        self.assertIn("browser shell controls", result)
        self.assertIn("No clickable candidates", result)

    def test_enriches_with_no_input_warning(self):
        health = PerceptionHealth(
            overall_health=HEALTH_DEGRADED,
            failure_flags=[FLAG_NO_INPUT_TARGET],
        )
        result = ui_state_module._enrich_vlm_instruction_with_health(
            base_instruction="Type a message.",
            perception_health=health,
        )
        self.assertIn("No text input field", result)


# ── UIState PerceptionHealth Integration Tests ───────────────────────────────


class TestUIStatePerceptionHealth(unittest.TestCase):

    def test_ui_state_serializes_perception_health(self):
        UIState = import_module("schemas.ui_state").UIState
        PageSummary = import_module("schemas.ui_state").PageSummary

        ui_state = UIState(
            screenshot_path="/tmp/test.png",
            candidates=[],
            page_summary=PageSummary(candidate_count=0),
            perception_health={
                "ocr_element_count": 10,
                "ocr_quality": "good",
                "overall_health": "healthy",
                "failure_flags": [],
            },
        )
        d = ui_state.to_dict()
        self.assertIn("perception_health", d)
        self.assertEqual(d["perception_health"]["ocr_element_count"], 10)
        self.assertEqual(d["perception_health"]["overall_health"], "healthy")

    def test_ui_state_omits_perception_health_when_none(self):
        UIState = import_module("schemas.ui_state").UIState
        PageSummary = import_module("schemas.ui_state").PageSummary

        ui_state = UIState(
            screenshot_path="/tmp/test.png",
            candidates=[],
            page_summary=PageSummary(candidate_count=0),
        )
        self.assertNotIn("perception_health", ui_state.to_dict())

    def test_ui_state_validates_perception_health_type(self):
        UIState = import_module("schemas.ui_state").UIState
        PageSummary = import_module("schemas.ui_state").PageSummary

        with self.assertRaises(Exception):
            UIState(
                screenshot_path="/tmp/test.png",
                candidates=[],
                page_summary=PageSummary(candidate_count=0),
                perception_health="not a dict",
            )


if __name__ == "__main__":
    unittest.main()
