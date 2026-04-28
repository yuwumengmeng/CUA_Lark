"""成员 A 2.2 VLM 输入输出格式测试。

测试目标：
验证 VLMStructuredOutput 数据类、parse_vlm_content() 解析器、
VLMRequest step 字段、以及 ui_state 集成输出是否符合 2.2 规范。

重要约定：
测试不调用真实 VLM API，只验证数据结构和解析逻辑。

直接运行：
```powershell
python tests\testA\test_vlm_output_schema.py
```

通过 pytest 运行：
```powershell
python -m pytest tests\testA\test_vlm_output_schema.py
```
"""

import json
import sys
import unittest
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

output_schema = import_module("vlm.output_schema")
provider_module = import_module("vlm.provider")

VLMStructuredOutput = output_schema.VLMStructuredOutput
parse_vlm_content = output_schema.parse_vlm_content
VLMOutputError = output_schema.VLMOutputError
VLM_PAGE_TYPES = output_schema.VLM_PAGE_TYPES
FALLBACK_RULE = output_schema.FALLBACK_RULE
FALLBACK_RETRY = output_schema.FALLBACK_RETRY
FALLBACK_WAIT = output_schema.FALLBACK_WAIT
FALLBACK_ABORT = output_schema.FALLBACK_ABORT

VLMRequest = provider_module.VLMRequest
PageSemantics = import_module("schemas").PageSemantics


class VLMStructuredOutputBasicTests(unittest.TestCase):
    def test_default_values(self):
        out = VLMStructuredOutput()
        self.assertEqual(out.page_type, "unknown")
        self.assertEqual(out.page_summary_enhancement, "")
        self.assertIsNone(out.target_candidate_id)
        self.assertEqual(out.reranked_candidate_ids, [])
        self.assertEqual(out.confidence, 0.0)
        self.assertEqual(out.reason, "")
        self.assertTrue(out.uncertain)
        self.assertEqual(out.suggested_fallback, FALLBACK_RULE)
        self.assertFalse(out.parse_ok)
        self.assertIsNone(out.parse_error)
        self.assertEqual(out.page_semantics.domain, "unknown")
        self.assertFalse(out.page_semantics.signals["search_active"])

    def test_frozen(self):
        out = VLMStructuredOutput()
        with self.assertRaises(AttributeError):
            out.page_type = "im"

    def test_page_type_normalization(self):
        out = VLMStructuredOutput(page_type="IM")
        self.assertEqual(out.page_type, "im")

    def test_page_type_unknown_for_invalid(self):
        out = VLMStructuredOutput(page_type="foobar")
        self.assertEqual(out.page_type, "unknown")

    def test_page_type_empty_becomes_unknown(self):
        out = VLMStructuredOutput(page_type="")
        self.assertEqual(out.page_type, "unknown")

    def test_confidence_clamped_high(self):
        out = VLMStructuredOutput(confidence=1.5)
        self.assertEqual(out.confidence, 1.0)

    def test_confidence_clamped_low(self):
        out = VLMStructuredOutput(confidence=-0.3)
        self.assertEqual(out.confidence, 0.0)

    def test_suggested_fallback_invalid_becomes_rule(self):
        out = VLMStructuredOutput(suggested_fallback="unknown_strategy")
        self.assertEqual(out.suggested_fallback, FALLBACK_RULE)

    def test_suggested_fallback_valid(self):
        for fallback in (FALLBACK_RULE, FALLBACK_RETRY, FALLBACK_WAIT, FALLBACK_ABORT):
            out = VLMStructuredOutput(suggested_fallback=fallback)
            self.assertEqual(out.suggested_fallback, fallback)

    def test_reranked_ids_stripped_and_stringified(self):
        out = VLMStructuredOutput(
            reranked_candidate_ids=["elem_001", "  ", "elem_002", 3],
        )
        self.assertEqual(out.reranked_candidate_ids, ["elem_001", "elem_002", "3"])

    def test_reranked_ids_empty_strings_removed(self):
        out = VLMStructuredOutput(reranked_candidate_ids=["", "elem_001", "   "])
        self.assertEqual(out.reranked_candidate_ids, ["elem_001"])

    def test_target_candidate_id_stripped(self):
        out = VLMStructuredOutput(target_candidate_id="  elem_003  ")
        self.assertEqual(out.target_candidate_id, "elem_003")

    def test_target_candidate_id_empty_becomes_none(self):
        out = VLMStructuredOutput(target_candidate_id="   ")
        self.assertIsNone(out.target_candidate_id)

    def test_target_candidate_id_none_stays_none(self):
        out = VLMStructuredOutput(target_candidate_id=None)
        self.assertIsNone(out.target_candidate_id)


class VLMStructuredOutputSerializationTests(unittest.TestCase):
    def test_to_dict_roundtrip(self):
        out = VLMStructuredOutput(
            page_type="im",
            page_summary_enhancement="Chat list page",
            page_semantics=PageSemantics(
                domain="im",
                view="list",
                signals={"list_available": True, "search_available": True},
                goal_progress="near_goal",
                confidence=0.82,
            ),
            target_candidate_id="elem_0003",
            reranked_candidate_ids=["elem_0003", "elem_0001"],
            confidence=0.92,
            reason="search box matches step goal",
            uncertain=False,
            suggested_fallback=FALLBACK_RULE,
            parse_ok=True,
        )
        d = out.to_dict()
        restored = VLMStructuredOutput.from_dict(d)
        self.assertEqual(restored.page_type, out.page_type)
        self.assertEqual(restored.target_candidate_id, out.target_candidate_id)
        self.assertEqual(restored.reranked_candidate_ids, out.reranked_candidate_ids)
        self.assertEqual(restored.confidence, out.confidence)
        self.assertEqual(restored.reason, out.reason)
        self.assertEqual(restored.uncertain, out.uncertain)
        self.assertEqual(restored.parse_ok, out.parse_ok)
        self.assertTrue(restored.page_semantics.signals["list_available"])
        self.assertEqual(restored.page_semantics.goal_progress, "near_goal")

    def test_to_dict_includes_parse_error_when_set(self):
        out = VLMStructuredOutput(parse_error="something went wrong")
        d = out.to_dict()
        self.assertIn("parse_error", d)
        self.assertEqual(d["parse_error"], "something went wrong")

    def test_to_dict_omits_parse_error_when_none(self):
        out = VLMStructuredOutput(parse_error=None)
        d = out.to_dict()
        self.assertNotIn("parse_error", d)

    def test_from_dict_handles_missing_fields(self):
        d = {"page_type": "im"}
        out = VLMStructuredOutput.from_dict(d)
        self.assertEqual(out.page_type, "im")
        self.assertEqual(out.page_semantics.domain, "im")
        self.assertIsNone(out.target_candidate_id)
        self.assertTrue(out.uncertain)
        self.assertFalse(out.parse_ok)

    def test_from_dict_parses_page_semantics(self):
        d = {
            "page_type": "docs",
            "page_semantics": {
                "domain": "docs",
                "view": "editor",
                "signals": {"editor_available": True, "target_visible": True},
                "entities": [{"type": "document", "name": "方案", "matched_target": True}],
                "goal_progress": "achieved",
                "confidence": 0.88,
            },
        }
        out = VLMStructuredOutput.from_dict(d)

        self.assertEqual(out.page_semantics.domain, "docs")
        self.assertEqual(out.page_semantics.view, "editor")
        self.assertTrue(out.page_semantics.signals["editor_available"])
        self.assertEqual(out.page_semantics.entities[0]["type"], "document")
        self.assertEqual(out.page_semantics.goal_progress, "achieved")

    def test_from_dict_handles_bool_strings(self):
        d = {"uncertain": "true", "parse_ok": "1"}
        out = VLMStructuredOutput.from_dict(d)
        self.assertTrue(out.uncertain)
        self.assertTrue(out.parse_ok)


class VLMStructuredOutputValidationTests(unittest.TestCase):
    def test_validate_candidate_ids_target_valid(self):
        out = VLMStructuredOutput(
            target_candidate_id="elem_0003",
            reranked_candidate_ids=["elem_0003", "elem_0001"],
            confidence=0.9,
            uncertain=False,
            parse_ok=True,
        )
        validated = out.validate_candidate_ids({"elem_0001", "elem_0003", "elem_0005"})
        self.assertEqual(validated.target_candidate_id, "elem_0003")
        self.assertFalse(validated.uncertain)

    def test_validate_candidate_ids_target_invalid(self):
        out = VLMStructuredOutput(
            target_candidate_id="elem_9999",
            reranked_candidate_ids=["elem_9999", "elem_0001"],
            confidence=0.9,
            uncertain=False,
            parse_ok=True,
        )
        validated = out.validate_candidate_ids({"elem_0001", "elem_0003"})
        self.assertIsNone(validated.target_candidate_id)
        self.assertTrue(validated.uncertain)
        self.assertLessEqual(validated.confidence, 0.3)
        self.assertIn("elem_9999", validated.parse_error)

    def test_validate_candidate_ids_filters_reranked(self):
        out = VLMStructuredOutput(
            target_candidate_id="elem_0001",
            reranked_candidate_ids=["elem_0001", "elem_9999", "elem_0003"],
            confidence=0.8,
            uncertain=False,
            parse_ok=True,
        )
        validated = out.validate_candidate_ids({"elem_0001", "elem_0003"})
        self.assertEqual(validated.reranked_candidate_ids, ["elem_0001", "elem_0003"])

    def test_validate_candidate_ids_empty_valid_ids_returns_self(self):
        out = VLMStructuredOutput(target_candidate_id="elem_0001")
        validated = out.validate_candidate_ids(set())
        self.assertIs(validated, out)


class ParseVLMContentNormalTests(unittest.TestCase):
    def test_parse_valid_json(self):
        text = json.dumps({
            "page_type": "im",
            "page_summary_enhancement": "Chat list",
            "page_semantics": {
                "domain": "im",
                "view": "search_results",
                "signals": {"search_active": True, "results_available": True},
                "goal_progress": "near_goal",
                "confidence": 0.87,
            },
            "target_candidate_id": "elem_0003",
            "reranked_candidate_ids": ["elem_0003", "elem_0001"],
            "confidence": 0.92,
            "reason": "search box",
            "uncertain": False,
            "suggested_fallback": "rule_fallback",
        })
        result = parse_vlm_content(text, valid_candidate_ids=["elem_0001", "elem_0003"])
        self.assertTrue(result.parse_ok)
        self.assertEqual(result.page_type, "im")
        self.assertEqual(result.target_candidate_id, "elem_0003")
        self.assertEqual(result.reranked_candidate_ids, ["elem_0003", "elem_0001"])
        self.assertTrue(result.page_semantics.signals["search_active"])
        self.assertEqual(result.page_semantics.goal_progress, "near_goal")
        self.assertFalse(result.uncertain)
        self.assertIsNone(result.parse_error)

    def test_parse_sets_parse_ok_true(self):
        result = parse_vlm_content('{"page_type":"im"}')
        self.assertTrue(result.parse_ok)

    def test_parse_with_valid_candidate_ids(self):
        result = parse_vlm_content(
            '{"target_candidate_id":"elem_0001"}',
            valid_candidate_ids=["elem_0001"],
        )
        self.assertEqual(result.target_candidate_id, "elem_0001")

    def test_parse_without_valid_candidate_ids(self):
        result = parse_vlm_content('{"target_candidate_id":"elem_0001"}')
        self.assertEqual(result.target_candidate_id, "elem_0001")


class ParseVLMContentMarkdownFenceTests(unittest.TestCase):
    def test_parse_json_in_markdown_fence(self):
        text = '```json\n{"page_type":"calendar","confidence":0.7,"uncertain":true}\n```'
        result = parse_vlm_content(text)
        self.assertTrue(result.parse_ok)
        self.assertEqual(result.page_type, "calendar")

    def test_parse_json_in_plain_fence(self):
        text = '```\n{"page_type":"docs"}\n```'
        result = parse_vlm_content(text)
        self.assertTrue(result.parse_ok)
        self.assertEqual(result.page_type, "docs")

    def test_parse_json_with_surrounding_text(self):
        text = 'Here is my analysis:\n{"page_type":"im","confidence":0.8}\nEnd of response.'
        result = parse_vlm_content(text)
        self.assertTrue(result.parse_ok)
        self.assertEqual(result.page_type, "im")


class ParseVLMContentFailureTests(unittest.TestCase):
    def test_parse_empty_string(self):
        result = parse_vlm_content("")
        self.assertFalse(result.parse_ok)
        self.assertTrue(result.uncertain)
        self.assertEqual(result.page_type, "unknown")
        self.assertIsNotNone(result.parse_error)

    def test_parse_whitespace_only(self):
        result = parse_vlm_content("   \n  ")
        self.assertFalse(result.parse_ok)

    def test_parse_no_json(self):
        result = parse_vlm_content("I cannot determine the page type.")
        self.assertFalse(result.parse_ok)
        self.assertIn("No JSON object", result.parse_error)

    def test_parse_invalid_json(self):
        result = parse_vlm_content('{"page_type": broken json}')
        self.assertFalse(result.parse_ok)
        self.assertIn("parse error", result.parse_error.lower())

    def test_parse_json_array_not_object(self):
        result = parse_vlm_content('[1, 2, 3]')
        self.assertFalse(result.parse_ok)
        self.assertIn("no json object", result.parse_error.lower())

    def test_parse_invalid_candidate_id_with_validation(self):
        text = json.dumps({"target_candidate_id": "elem_9999", "uncertain": False})
        result = parse_vlm_content(text, valid_candidate_ids=["elem_0001", "elem_0003"])
        self.assertTrue(result.parse_ok)
        self.assertIsNone(result.target_candidate_id)
        self.assertTrue(result.uncertain)


class ParseVLMContentEdgeCaseTests(unittest.TestCase):
    def test_parse_nested_json_object(self):
        text = json.dumps({
            "page_type": "im",
            "extra": {"nested": True},
            "confidence": 0.5,
        })
        result = parse_vlm_content(text)
        self.assertTrue(result.parse_ok)
        self.assertEqual(result.page_type, "im")

    def test_parse_partial_fields(self):
        result = parse_vlm_content('{"page_type":"search"}')
        self.assertTrue(result.parse_ok)
        self.assertEqual(result.page_type, "search")
        self.assertIsNone(result.target_candidate_id)
        self.assertEqual(result.reranked_candidate_ids, [])

    def test_parse_never_raises(self):
        cases = [None, 0, False, [], {}, Exception("boom")]
        for case in cases:
            try:
                parse_vlm_content(str(case))
            except Exception:
                self.fail(f"parse_vlm_content raised for input: {case!r}")


class VLMRequestStepFieldsTests(unittest.TestCase):
    def test_step_name_in_compact_payload(self):
        req = VLMRequest(instruction="find search", step_name="open_search")
        payload = req.compact_payload(default_max_candidates=30)
        self.assertEqual(payload["step_name"], "open_search")

    def test_step_goal_in_compact_payload(self):
        req = VLMRequest(instruction="find search", step_goal="locate search box")
        payload = req.compact_payload(default_max_candidates=30)
        self.assertEqual(payload["step_goal"], "locate search box")

    def test_region_hint_in_compact_payload(self):
        req = VLMRequest(instruction="find search", region_hint="top-left area")
        payload = req.compact_payload(default_max_candidates=30)
        self.assertEqual(payload["region_hint"], "top-left area")

    def test_step_fields_omitted_when_none(self):
        req = VLMRequest(instruction="find search")
        payload = req.compact_payload(default_max_candidates=30)
        self.assertNotIn("step_name", payload)
        self.assertNotIn("step_goal", payload)
        self.assertNotIn("region_hint", payload)

    def test_all_step_fields_together(self):
        req = VLMRequest(
            instruction="open chat",
            step_name="open_chat",
            step_goal="click target group chat",
            region_hint="center chat list",
        )
        payload = req.compact_payload(default_max_candidates=30)
        self.assertEqual(payload["step_name"], "open_chat")
        self.assertEqual(payload["step_goal"], "click target group chat")
        self.assertEqual(payload["region_hint"], "center chat list")

    def test_max_candidates_limits_candidates(self):
        candidates = [{"id": f"elem_{i:04d}"} for i in range(50)]
        req = VLMRequest(instruction="test", candidates=candidates, max_candidates=10)
        payload = req.compact_payload(default_max_candidates=30)
        self.assertEqual(len(payload["candidates"]), 10)

    def test_max_candidates_defaults_to_config(self):
        candidates = [{"id": f"elem_{i:04d}"} for i in range(50)]
        req = VLMRequest(instruction="test", candidates=candidates)
        payload = req.compact_payload(default_max_candidates=20)
        self.assertEqual(len(payload["candidates"]), 20)

    def test_compact_payload_keeps_context_small(self):
        candidates = [
            {
                "id": f"elem_{i:04d}",
                "text": "搜索" if i == 45 else f"普通候选{i}",
                "bbox": [i * 10, i * 10, i * 10 + 20, i * 10 + 20],
                "center": [i * 10 + 10, i * 10 + 10],
                "role": "EditControl" if i == 45 else "Text",
                "editable": i == 45,
            }
            for i in range(60)
        ]
        req = VLMRequest(
            instruction="find search",
            candidates=candidates,
            max_candidates=8,
            step_name="open_search",
            step_goal="locate search box",
            region_hint="top search area",
            page_summary={"top_texts": [f"文本{i}" for i in range(40)], "candidate_count": 60},
        )
        payload = req.compact_payload(default_max_candidates=30)

        self.assertEqual(len(payload["candidates"]), 8)
        self.assertLessEqual(len(payload["page_summary"]["top_texts"]), 12)
        self.assertIn("elem_0045", {candidate["id"] for candidate in payload["candidates"]})


class VLMRequestCompactCandidateTests(unittest.TestCase):
    def test_compact_candidate_fields(self):
        req = VLMRequest(
            instruction="test",
            candidates=[{
                "id": "elem_0001",
                "text": "搜索",
                "bbox": [10, 20, 50, 40],
                "center": [30, 30],
                "role": "EditControl",
                "clickable": True,
                "editable": True,
                "source": "merged",
                "extra_field": "should_be_removed",
            }],
        )
        payload = req.compact_payload(default_max_candidates=30)
        candidate = payload["candidates"][0]
        self.assertEqual(candidate["id"], "elem_0001")
        self.assertEqual(candidate["text"], "搜索")
        self.assertEqual(candidate["bbox"], [10, 20, 50, 40])
        self.assertEqual(candidate["center"], [30, 30])
        self.assertEqual(candidate["role"], "EditControl")
        self.assertTrue(candidate["clickable"])
        self.assertTrue(candidate["editable"])
        self.assertEqual(candidate["source"], "merged")
        self.assertNotIn("extra_field", candidate)


if __name__ == "__main__":
    unittest.main()
