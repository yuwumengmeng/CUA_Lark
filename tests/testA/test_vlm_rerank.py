"""成员 A 2.3 VLM candidate rerank 与视觉候选兜底测试。"""

import json
import unittest

from schemas import (
    COORDINATE_TYPE_SCREENSHOT_PIXEL,
    SOURCE_OCR,
    SOURCE_VLM_VISUAL,
    UICandidate,
)
from vlm import (
    SHELL_CONTROL_ROLES,
    SHELL_CONTROL_TEXTS,
    VLMStructuredOutput,
    build_visual_candidates,
    is_shell_control,
    rerank_candidates,
)


def _make_candidate(
    candidate_id: str = "elem_0001",
    text: str = "搜索",
    bbox: list[int] | None = None,
    role: str = "text",
    clickable: bool = False,
    editable: bool = False,
    source: str = SOURCE_OCR,
    confidence: float = 0.9,
    coordinate_type: str | None = None,
    vlm_reason: str | None = None,
) -> UICandidate:
    if bbox is None:
        bbox = [100, 100, 200, 140]
    return UICandidate(
        id=candidate_id,
        text=text,
        bbox=bbox,
        role=role,
        clickable=clickable,
        editable=editable,
        source=source,
        confidence=confidence,
        coordinate_type=coordinate_type,
        vlm_reason=vlm_reason,
    )


class RerankCandidatesTests(unittest.TestCase):
    def test_rerank_puts_target_first(self):
        c1 = _make_candidate("elem_0001", "搜索框", role="LarkSearchBox", editable=True)
        c2 = _make_candidate("elem_0002", "消息", bbox=[100, 200, 200, 240])
        c3 = _make_candidate("elem_0003", "设置", bbox=[100, 300, 200, 340])
        structured = VLMStructuredOutput(
            page_type="im",
            target_candidate_id="elem_0003",
            reranked_candidate_ids=["elem_0003", "elem_0001"],
            confidence=0.92,
            reason="设置按钮匹配 step_goal",
            uncertain=False,
            parse_ok=True,
        )
        result = rerank_candidates([c1, c2, c3], structured)
        self.assertEqual(result[0].id, "elem_0003")
        self.assertEqual(result[1].id, "elem_0001")
        self.assertEqual(result[2].id, "elem_0002")

    def test_rerank_attaches_vlm_reason(self):
        c1 = _make_candidate("elem_0001", "搜索框")
        structured = VLMStructuredOutput(
            target_candidate_id="elem_0001",
            reranked_candidate_ids=["elem_0001"],
            confidence=0.85,
            reason="搜索框匹配目标",
            uncertain=False,
            parse_ok=True,
        )
        result = rerank_candidates([c1], structured)
        self.assertEqual(result[0].vlm_reason, "搜索框匹配目标")

    def test_rerank_no_change_when_parse_not_ok(self):
        c1 = _make_candidate("elem_0001", "搜索框")
        c2 = _make_candidate("elem_0002", "消息", bbox=[100, 200, 200, 240])
        structured = VLMStructuredOutput(parse_ok=False)
        result = rerank_candidates([c1, c2], structured)
        self.assertEqual(result[0].id, "elem_0001")
        self.assertEqual(result[1].id, "elem_0002")

    def test_rerank_empty_candidates(self):
        structured = VLMStructuredOutput(parse_ok=True, target_candidate_id="elem_0001")
        result = rerank_candidates([], structured)
        self.assertEqual(result, [])

    def test_rerank_demotes_shell_controls(self):
        c1 = _make_candidate("elem_0001", "关闭", role="Button")
        c2 = _make_candidate("elem_0002", "搜索框", editable=True)
        c3 = _make_candidate("elem_0003", "最小化", role="Button", bbox=[100, 200, 200, 240])
        structured = VLMStructuredOutput(
            target_candidate_id="elem_0002",
            reranked_candidate_ids=["elem_0002"],
            confidence=0.9,
            reason="搜索框",
            uncertain=False,
            parse_ok=True,
        )
        result = rerank_candidates([c1, c2, c3], structured, demote_shell=True)
        ids = [c.id for c in result]
        self.assertEqual(ids[0], "elem_0002")
        self.assertIn("elem_0001", ids[-2:])
        self.assertIn("elem_0003", ids[-2:])

    def test_rerank_no_demote_when_flag_false(self):
        c1 = _make_candidate("elem_0001", "关闭", role="Button")
        c2 = _make_candidate("elem_0002", "搜索框", editable=True)
        structured = VLMStructuredOutput(
            target_candidate_id="elem_0002",
            reranked_candidate_ids=["elem_0002"],
            confidence=0.9,
            reason="搜索框",
            uncertain=False,
            parse_ok=True,
        )
        result = rerank_candidates([c1, c2], structured, demote_shell=False)
        self.assertEqual(result[0].id, "elem_0002")
        self.assertEqual(result[1].id, "elem_0001")

    def test_rerank_preserves_all_candidates(self):
        candidates = [_make_candidate(f"elem_{i:04d}", f"text_{i}", bbox=[100, i * 50, 200, i * 50 + 40]) for i in range(5)]
        structured = VLMStructuredOutput(
            target_candidate_id="elem_0002",
            reranked_candidate_ids=["elem_0002", "elem_0004"],
            confidence=0.8,
            reason="test",
            uncertain=False,
            parse_ok=True,
        )
        result = rerank_candidates(candidates, structured)
        result_ids = [c.id for c in result]
        self.assertEqual(len(result_ids), 5)
        for candidate in candidates:
            self.assertIn(candidate.id, result_ids)

    def test_rerank_confidence_decreases_with_rank(self):
        candidates = [
            _make_candidate("elem_0001", "a", confidence=0.5),
            _make_candidate("elem_0002", "b", bbox=[100, 200, 200, 240], confidence=0.5),
            _make_candidate("elem_0003", "c", bbox=[100, 300, 200, 340], confidence=0.5),
        ]
        structured = VLMStructuredOutput(
            reranked_candidate_ids=["elem_0001", "elem_0002", "elem_0003"],
            confidence=0.9,
            reason="test",
            uncertain=False,
            parse_ok=True,
        )
        result = rerank_candidates(candidates, structured)
        self.assertGreaterEqual(result[0].confidence, result[1].confidence)
        self.assertGreaterEqual(result[1].confidence, result[2].confidence)


class IsShellControlTests(unittest.TestCase):
    def test_shell_role_detected(self):
        c = _make_candidate(role="TitleBar")
        self.assertTrue(is_shell_control(c))

    def test_shell_text_detected(self):
        c = _make_candidate(text="最小化")
        self.assertTrue(is_shell_control(c))

    def test_normal_candidate_not_shell(self):
        c = _make_candidate(text="搜索框", role="LarkSearchBox")
        self.assertFalse(is_shell_control(c))

    def test_dict_input_shell_role(self):
        self.assertTrue(is_shell_control({"role": "ScrollBar", "text": ""}))

    def test_dict_input_shell_text(self):
        self.assertTrue(is_shell_control({"role": "Button", "text": "关闭"}))

    def test_dict_input_normal(self):
        self.assertFalse(is_shell_control({"role": "Button", "text": "发送"}))

    def test_all_shell_roles_recognized(self):
        for role in SHELL_CONTROL_ROLES:
            c = _make_candidate(role=role)
            self.assertTrue(is_shell_control(c), f"Role '{role}' should be shell")

    def test_all_shell_texts_recognized(self):
        for text in SHELL_CONTROL_TEXTS:
            c = _make_candidate(text=text)
            self.assertTrue(is_shell_control(c), f"Text '{text}' should be shell")


class BuildVisualCandidatesTests(unittest.TestCase):
    def test_build_from_bbox(self):
        structured = VLMStructuredOutput(
            parse_ok=True,
            visual_candidates=[
                {
                    "id": "vlm_vis_0001",
                    "role": "icon",
                    "bbox": [300, 100, 350, 150],
                    "confidence": 0.7,
                    "reason": "search icon",
                    "coordinate_type": "screenshot_pixel",
                }
            ],
        )
        result = build_visual_candidates(structured)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source, SOURCE_VLM_VISUAL)
        self.assertEqual(result[0].role, "icon")
        self.assertEqual(result[0].bbox, [300, 100, 350, 150])
        self.assertEqual(result[0].coordinate_type, COORDINATE_TYPE_SCREENSHOT_PIXEL)
        self.assertEqual(result[0].vlm_reason, "search icon")
        self.assertTrue(result[0].clickable)

    def test_build_from_center_only(self):
        structured = VLMStructuredOutput(
            parse_ok=True,
            visual_candidates=[
                {
                    "role": "button",
                    "center": [400, 200],
                    "confidence": 0.6,
                    "reason": "back arrow",
                }
            ],
        )
        result = build_visual_candidates(structured)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source, SOURCE_VLM_VISUAL)
        self.assertEqual(result[0].center, [400, 200])

    def test_build_empty_visual_candidates(self):
        structured = VLMStructuredOutput(parse_ok=True)
        result = build_visual_candidates(structured)
        self.assertEqual(result, [])

    def test_build_skips_invalid_no_bbox_no_center(self):
        structured = VLMStructuredOutput(
            parse_ok=True,
            visual_candidates=[
                {"role": "icon", "confidence": 0.5, "reason": "no coords"},
            ],
        )
        result = build_visual_candidates(structured)
        self.assertEqual(result, [])

    def test_build_multiple_visual_candidates(self):
        structured = VLMStructuredOutput(
            parse_ok=True,
            visual_candidates=[
                {"bbox": [100, 100, 140, 140], "role": "icon", "confidence": 0.7, "reason": "search"},
                {"bbox": [200, 200, 240, 240], "role": "button", "confidence": 0.6, "reason": "back"},
            ],
        )
        result = build_visual_candidates(structured)
        self.assertEqual(len(result), 2)

    def test_build_auto_generates_id(self):
        structured = VLMStructuredOutput(
            parse_ok=True,
            visual_candidates=[
                {"bbox": [100, 100, 140, 140], "role": "icon", "confidence": 0.7, "reason": "test"},
            ],
        )
        result = build_visual_candidates(structured)
        self.assertTrue(result[0].id.startswith("vlm_vis_"))

    def test_build_default_coordinate_type(self):
        structured = VLMStructuredOutput(
            parse_ok=True,
            visual_candidates=[
                {"bbox": [100, 100, 140, 140], "role": "icon", "confidence": 0.7, "reason": "test"},
            ],
        )
        result = build_visual_candidates(structured)
        self.assertEqual(result[0].coordinate_type, COORDINATE_TYPE_SCREENSHOT_PIXEL)


class VLMStructuredOutputVisualCandidatesTests(unittest.TestCase):
    def test_visual_candidates_parsed_from_json(self):
        text = json.dumps({
            "page_type": "im",
            "target_candidate_id": None,
            "confidence": 0.3,
            "reason": "no text match",
            "uncertain": True,
            "suggested_fallback": "rule_fallback",
            "visual_candidates": [
                {
                    "id": "vlm_vis_0001",
                    "role": "icon",
                    "bbox": [300, 100, 350, 150],
                    "confidence": 0.7,
                    "reason": "search icon",
                    "coordinate_type": "screenshot_pixel",
                }
            ],
        })
        from vlm import parse_vlm_content
        result = parse_vlm_content(text)
        self.assertTrue(result.parse_ok)
        self.assertEqual(len(result.visual_candidates), 1)
        self.assertEqual(result.visual_candidates[0]["role"], "icon")

    def test_visual_candidates_empty_when_absent(self):
        text = json.dumps({
            "page_type": "im",
            "target_candidate_id": "elem_0001",
            "confidence": 0.9,
            "reason": "match",
            "uncertain": False,
            "suggested_fallback": "rule_fallback",
        })
        from vlm import parse_vlm_content
        result = parse_vlm_content(text)
        self.assertEqual(result.visual_candidates, [])

    def test_visual_candidates_filtered_no_coords(self):
        structured = VLMStructuredOutput(
            parse_ok=True,
            visual_candidates=[
                {"role": "icon", "confidence": 0.7, "reason": "no coords"},
                {"bbox": [100, 100, 140, 140], "role": "button", "confidence": 0.6, "reason": "has bbox"},
            ],
        )
        self.assertEqual(len(structured.visual_candidates), 1)
        self.assertEqual(structured.visual_candidates[0]["role"], "button")

    def test_visual_candidates_in_to_dict(self):
        structured = VLMStructuredOutput(
            parse_ok=True,
            visual_candidates=[
                {"bbox": [100, 100, 140, 140], "role": "icon", "confidence": 0.7, "reason": "test"},
            ],
        )
        d = structured.to_dict()
        self.assertIn("visual_candidates", d)
        self.assertEqual(len(d["visual_candidates"]), 1)

    def test_visual_candidates_absent_in_to_dict_when_empty(self):
        structured = VLMStructuredOutput(parse_ok=True)
        d = structured.to_dict()
        self.assertNotIn("visual_candidates", d)


class UICandidateVLMVisualSourceTests(unittest.TestCase):
    def test_vlm_visual_source_accepted(self):
        c = _make_candidate(source=SOURCE_VLM_VISUAL, coordinate_type=COORDINATE_TYPE_SCREENSHOT_PIXEL)
        self.assertEqual(c.source, SOURCE_VLM_VISUAL)
        self.assertEqual(c.coordinate_type, COORDINATE_TYPE_SCREENSHOT_PIXEL)

    def test_coordinate_type_screenshot_pixel(self):
        c = _make_candidate(coordinate_type=COORDINATE_TYPE_SCREENSHOT_PIXEL)
        self.assertEqual(c.coordinate_type, "screenshot_pixel")

    def test_coordinate_type_normalized(self):
        c = _make_candidate(coordinate_type="normalized")
        self.assertEqual(c.coordinate_type, "normalized")

    def test_invalid_coordinate_type_rejected(self):
        with self.assertRaises(Exception):
            _make_candidate(coordinate_type="invalid_type")

    def test_vlm_reason_field(self):
        c = _make_candidate(vlm_reason="VLM selected as search box")
        self.assertEqual(c.vlm_reason, "VLM selected as search box")

    def test_vlm_reason_none_by_default(self):
        c = _make_candidate()
        self.assertIsNone(c.vlm_reason)

    def test_coordinate_type_none_by_default(self):
        c = _make_candidate()
        self.assertIsNone(c.coordinate_type)

    def test_to_dict_includes_vlm_fields(self):
        c = _make_candidate(
            source=SOURCE_VLM_VISUAL,
            coordinate_type=COORDINATE_TYPE_SCREENSHOT_PIXEL,
            vlm_reason="search icon",
        )
        d = c.to_dict()
        self.assertEqual(d["source"], SOURCE_VLM_VISUAL)
        self.assertEqual(d["coordinate_type"], COORDINATE_TYPE_SCREENSHOT_PIXEL)
        self.assertEqual(d["vlm_reason"], "search icon")

    def test_to_dict_omits_vlm_fields_when_none(self):
        c = _make_candidate()
        d = c.to_dict()
        self.assertNotIn("coordinate_type", d)
        self.assertNotIn("vlm_reason", d)

    def test_from_dict_roundtrip(self):
        c = _make_candidate(
            source=SOURCE_VLM_VISUAL,
            coordinate_type=COORDINATE_TYPE_SCREENSHOT_PIXEL,
            vlm_reason="test reason",
        )
        d = c.to_dict()
        restored = UICandidate.from_dict(d)
        self.assertEqual(restored.source, SOURCE_VLM_VISUAL)
        self.assertEqual(restored.coordinate_type, COORDINATE_TYPE_SCREENSHOT_PIXEL)
        self.assertEqual(restored.vlm_reason, "test reason")


class RerankIntegrationTests(unittest.TestCase):
    def test_full_rerank_with_shell_demote_and_visual(self):
        c1 = _make_candidate("elem_0001", "搜索框", role="LarkSearchBox", editable=True, confidence=0.95)
        c2 = _make_candidate("elem_0002", "关闭", role="Button", confidence=0.9)
        c3 = _make_candidate("elem_0003", "消息", bbox=[100, 200, 200, 240], confidence=0.85)
        c4 = _make_candidate("elem_0004", "最小化", role="Button", bbox=[100, 300, 200, 340], confidence=0.8)

        structured = VLMStructuredOutput(
            page_type="im",
            target_candidate_id="elem_0001",
            reranked_candidate_ids=["elem_0001", "elem_0003"],
            confidence=0.92,
            reason="搜索框匹配 step_goal",
            uncertain=False,
            parse_ok=True,
            visual_candidates=[
                {
                    "id": "vlm_vis_0001",
                    "role": "icon",
                    "bbox": [500, 100, 540, 140],
                    "confidence": 0.6,
                    "reason": "settings gear icon",
                    "coordinate_type": "screenshot_pixel",
                }
            ],
        )

        reranked = rerank_candidates([c1, c2, c3, c4], structured)
        visual = build_visual_candidates(structured)
        all_candidates = reranked + visual

        self.assertEqual(all_candidates[0].id, "elem_0001")
        self.assertEqual(all_candidates[0].vlm_reason, "搜索框匹配 step_goal")
        self.assertEqual(all_candidates[1].id, "elem_0003")

        shell_ids = [c.id for c in all_candidates if is_shell_control(c)]
        for shell_id in shell_ids:
            self.assertGreater(
                all_candidates.index(next(c for c in all_candidates if c.id == shell_id)),
                all_candidates.index(next(c for c in all_candidates if c.id == "elem_0001")),
            )

        visual_candidate = next((c for c in all_candidates if c.source == SOURCE_VLM_VISUAL), None)
        self.assertIsNotNone(visual_candidate)
        self.assertEqual(visual_candidate.role, "icon")
        self.assertEqual(visual_candidate.coordinate_type, COORDINATE_TYPE_SCREENSHOT_PIXEL)

    def test_rerank_uncertain_keeps_all_candidates(self):
        c1 = _make_candidate("elem_0001", "搜索框", editable=True)
        c2 = _make_candidate("elem_0002", "消息", bbox=[100, 200, 200, 240])
        structured = VLMStructuredOutput(
            page_type="im",
            target_candidate_id=None,
            reranked_candidate_ids=[],
            confidence=0.2,
            reason="无法确定目标",
            uncertain=True,
            parse_ok=True,
        )
        result = rerank_candidates([c1, c2], structured)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
