"""Member A Task 2.6 document region reading tests.

Tests: DocumentRegionSummary schema, estimate_doc_body_bbox heuristics,
read_document_region with fake VLM provider, _should_read_document_region
logic, and integration with collect_ui_state().

Run:
    python -m pytest tests/testA/test_document_region.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

schemas_doc = import_module("schemas.document_region")
region_reader = import_module("vlm.region_reader")
ui_state_module = import_module("perception.ui_state")
collect_ui_state = ui_state_module.collect_ui_state
get_ui_state = ui_state_module.get_ui_state

DocumentRegionSummary = schemas_doc.DocumentRegionSummary
estimate_doc_body_bbox = region_reader.estimate_doc_body_bbox
read_document_region = region_reader.read_document_region
UIState = import_module("schemas.ui_state").UIState
PageSummary = import_module("schemas.ui_state").PageSummary
ScreenMeta = import_module("schemas.ui_state").ScreenMeta
UICandidate = import_module("schemas.ui_candidate").UICandidate


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_screenshot_png(path: Path, width: int = 1920, height: int = 1080) -> Path:
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    img.save(str(path), "PNG")
    return path


def _make_doc_ui_state(screenshot_path: str) -> UIState:
    """Build a minimal UIState that looks like a docs page."""
    return UIState(
        screenshot_path=screenshot_path,
        candidates=[
            UICandidate(
                id="c1",
                text="文档正文",
                bbox=[400, 100, 1800, 1000],
                center=[1100, 550],
                role="document",
                clickable=False,
                editable=False,
                source="ocr",
            )
        ],
        page_summary=PageSummary(
            top_texts=["知识库", "项目文档", "正文"],
            has_search_box=False,
            has_modal=False,
            candidate_count=1,
        ),
        vlm_result={
            "ok": True,
            "content": "{}",
            "structured": {
                "page_type": "docs",
                "page_semantics": {"domain": "docs", "view": "detail"},
            },
        },
        screen_meta=ScreenMeta(
            screenshot_size=[1920, 1080],
            screen_resolution=[1920, 1080],
            window_rect=None,
            screen_origin=[0, 0],
        ),
    )


class FakeOkVLMProvider:
    """Returns a successful VLM response with provided content."""

    def __init__(self, response_content: str) -> None:
        self.response_content = response_content
        self.config = None

    def analyze(self, request):
        result_module = import_module("vlm.provider")
        return result_module.VLMCallResult(
            ok=True,
            provider="fake",
            model="fake-model",
            api_base="fake",
            content=self.response_content,
        )


class FakeFailVLMProvider:
    """Returns a failed VLM response."""

    def analyze(self, request):
        result_module = import_module("vlm.provider")
        return result_module.VLMCallResult(
            ok=False,
            provider="fake",
            model="fake-model",
            api_base="fake",
            error="VLM not available",
        )


class FakeExceptionVLMProvider:
    """Raises an exception during analyze."""

    def analyze(self, request):
        raise RuntimeError("VLM connection refused")


_DOC_REGION_VLM_JSON = json.dumps({
    "visible_title": "项目攻坚周会纪要",
    "visible_sections": ["一、项目背景", "二、当前进度", "三、风险与对策"],
    "visible_texts": [
        "本项目旨在完成飞书桌面端的CUA测试系统",
        "当前已完成第一周期感知闭环",
        "第二周期目标是跑通IM短流程",
    ],
    "target_keywords_found": ["CUA", "感知闭环"],
    "is_edit_mode": False,
    "is_target_document": True,
    "body_content_summary": "文档记录项目周会内容，包含项目背景、当前进度和风险对策三个章节。",
    "confidence": 0.9,
})


# ── DocumentRegionSummary Schema Tests ──────────────────────────────────────


class TestDocumentRegionSummary(unittest.TestCase):

    def test_default_construction(self):
        summary = DocumentRegionSummary()
        self.assertEqual(summary.visible_title, "")
        self.assertEqual(summary.visible_sections, [])
        self.assertEqual(summary.visible_texts, [])
        self.assertEqual(summary.target_keywords_found, [])
        self.assertFalse(summary.is_edit_mode)
        self.assertFalse(summary.is_target_document)
        self.assertEqual(summary.body_content_summary, "")
        self.assertEqual(summary.confidence, 0.0)
        self.assertIsNone(summary.error)

    def test_full_construction(self):
        summary = DocumentRegionSummary(
            visible_title="测试文档",
            visible_sections=["第一节", "第二节"],
            visible_texts=["文本一", "文本二"],
            target_keywords_found=["关键词"],
            is_edit_mode=True,
            is_target_document=True,
            body_content_summary="这是测试文档。",
            confidence=0.85,
        )
        self.assertEqual(summary.visible_title, "测试文档")
        self.assertEqual(summary.visible_sections, ["第一节", "第二节"])
        self.assertEqual(summary.visible_texts, ["文本一", "文本二"])
        self.assertEqual(summary.target_keywords_found, ["关键词"])
        self.assertTrue(summary.is_edit_mode)
        self.assertTrue(summary.is_target_document)
        self.assertEqual(summary.body_content_summary, "这是测试文档。")
        self.assertEqual(summary.confidence, 0.85)
        self.assertIsNone(summary.error)

    def test_confidence_clamped(self):
        summary = DocumentRegionSummary(confidence=1.5)
        self.assertEqual(summary.confidence, 1.0)
        summary = DocumentRegionSummary(confidence=-0.5)
        self.assertEqual(summary.confidence, 0.0)

    def test_empty_factory(self):
        summary = DocumentRegionSummary.empty(error="VLM disconnected")
        self.assertEqual(summary.visible_title, "")
        self.assertEqual(summary.error, "VLM disconnected")
        self.assertEqual(summary.confidence, 0.0)

        summary_no_error = DocumentRegionSummary.empty()
        self.assertIsNone(summary_no_error.error)

    def test_to_dict_roundtrip(self):
        original = DocumentRegionSummary(
            visible_title="测试文档",
            visible_sections=["一", "二"],
            visible_texts=["文本"],
            target_keywords_found=["关键"],
            is_edit_mode=True,
            is_target_document=True,
            body_content_summary="摘要",
            confidence=0.9,
        )
        d = original.to_dict()
        self.assertEqual(d["visible_title"], "测试文档")
        self.assertEqual(d["visible_sections"], ["一", "二"])
        self.assertEqual(d["visible_texts"], ["文本"])
        self.assertEqual(d["confidence"], 0.9)
        self.assertTrue(d["is_edit_mode"])

        restored = DocumentRegionSummary.from_dict(d)
        self.assertEqual(restored.visible_title, original.visible_title)
        self.assertEqual(restored.visible_sections, original.visible_sections)
        self.assertEqual(restored.visible_texts, original.visible_texts)
        self.assertEqual(restored.confidence, original.confidence)
        self.assertEqual(restored.is_edit_mode, original.is_edit_mode)
        self.assertEqual(restored.is_target_document, original.is_target_document)

    def test_to_dict_includes_error(self):
        summary = DocumentRegionSummary.empty(error="test error")
        d = summary.to_dict()
        self.assertEqual(d["error"], "test error")
        self.assertEqual(d["confidence"], 0.0)

    def test_from_dict_partial(self):
        summary = DocumentRegionSummary.from_dict({"visible_title": "标题"})
        self.assertEqual(summary.visible_title, "标题")
        self.assertEqual(summary.visible_sections, [])
        self.assertEqual(summary.confidence, 0.0)

    def test_from_dict_filters_empty_strings(self):
        summary = DocumentRegionSummary.from_dict({
            "visible_sections": ["", "有效章节", "  ", "另一章"],
            "visible_texts": [None, "文本", ""],
        })
        self.assertEqual(summary.visible_sections, ["有效章节", "另一章"])
        self.assertEqual(summary.visible_texts, ["文本"])


# ── estimate_doc_body_bbox Tests ────────────────────────────────────────────


class TestEstimateDocBodyBbox(unittest.TestCase):

    def test_standard_1920x1080(self):
        bbox = estimate_doc_body_bbox(1920, 1080)
        self.assertIsNotNone(bbox)
        x1, y1, x2, y2 = bbox
        self.assertEqual(x1, 364)  # 64 + 300
        self.assertEqual(y1, 56)
        self.assertEqual(x2, 1904)  # 1920 - 16
        self.assertEqual(y2, 1080)

    def test_small_screen_rejected(self):
        bbox = estimate_doc_body_bbox(300, 200)
        self.assertIsNone(bbox)

    def test_narrow_but_tall_rejected(self):
        bbox = estimate_doc_body_bbox(400, 2000)
        self.assertIsNone(bbox)  # width < 64 + 300 + 200 = 564

    def test_minimum_valid_size(self):
        bbox = estimate_doc_body_bbox(600, 300)
        self.assertIsNotNone(bbox)
        x1, y1, x2, y2 = bbox
        self.assertEqual(x1, 364)
        self.assertEqual(y1, 56)
        self.assertEqual(x2, 584)  # 600 - 16
        self.assertEqual(y2, 300)


# ── read_document_region Tests ──────────────────────────────────────────────


class TestReadDocumentRegion(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.screenshot_path = Path(self.tmpdir) / "test_screenshot.png"
        _make_screenshot_png(self.screenshot_path, 1920, 1080)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_empty_when_screenshot_missing(self):
        result = read_document_region(
            "/nonexistent/path.png",
            vlm_provider=FakeOkVLMProvider("{}"),
        )
        self.assertIsNotNone(result.error)
        self.assertIn("not found", result.error)

    def test_returns_empty_when_vlm_fails(self):
        result = read_document_region(
            self.screenshot_path,
            vlm_provider=FakeFailVLMProvider(),
        )
        self.assertIsNotNone(result.error)
        self.assertIn("VLM call failed", result.error)

    def test_returns_empty_when_vlm_raises(self):
        result = read_document_region(
            self.screenshot_path,
            vlm_provider=FakeExceptionVLMProvider(),
        )
        self.assertIsNotNone(result.error)

    def test_returns_empty_when_vlm_content_has_no_json(self):
        result = read_document_region(
            self.screenshot_path,
            vlm_provider=FakeOkVLMProvider("just some text, no json here"),
        )
        self.assertIsNotNone(result.error)
        self.assertIn("No JSON", result.error)

    def test_parses_valid_vlm_response(self):
        result = read_document_region(
            self.screenshot_path,
            vlm_provider=FakeOkVLMProvider(_DOC_REGION_VLM_JSON),
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.visible_title, "项目攻坚周会纪要")
        self.assertEqual(len(result.visible_sections), 3)
        self.assertIn("一、项目背景", result.visible_sections)
        self.assertEqual(len(result.visible_texts), 3)
        self.assertEqual(result.target_keywords_found, ["CUA", "感知闭环"])
        self.assertFalse(result.is_edit_mode)
        self.assertTrue(result.is_target_document)
        self.assertIn("周会", result.body_content_summary)
        self.assertEqual(result.confidence, 0.9)

    def test_parses_json_with_markdown_fence(self):
        content = f"```json\n{_DOC_REGION_VLM_JSON}\n```"
        result = read_document_region(
            self.screenshot_path,
            vlm_provider=FakeOkVLMProvider(content),
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.visible_title, "项目攻坚周会纪要")

    def test_parses_json_embedded_in_text(self):
        content = f"Here is the analysis:\n\n{_DOC_REGION_VLM_JSON}\n\nHope this helps."
        result = read_document_region(
            self.screenshot_path,
            vlm_provider=FakeOkVLMProvider(content),
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.visible_title, "项目攻坚周会纪要")

    def test_handles_empty_json_object(self):
        result = read_document_region(
            self.screenshot_path,
            vlm_provider=FakeOkVLMProvider("{}"),
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.visible_title, "")
        self.assertEqual(result.visible_sections, [])

    def test_handles_invalid_json(self):
        result = read_document_region(
            self.screenshot_path,
            vlm_provider=FakeOkVLMProvider("{not valid json"),
        )
        self.assertIsNotNone(result.error)
        self.assertIn("JSON", result.error)


# ── _should_read_document_region Tests ──────────────────────────────────────


class TestShouldReadDocumentRegion(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.screenshot_path = Path(self.tmpdir) / "test.png"
        _make_screenshot_png(self.screenshot_path, 1920, 1080)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_true_for_docs_domain_in_vlm_semantics(self):
        ui_state = _make_doc_ui_state(str(self.screenshot_path))
        result = ui_state_module._should_read_document_region(ui_state)
        self.assertTrue(result)

    def test_returns_false_for_non_docs_domain(self):
        ui_state = UIState(
            screenshot_path=str(self.screenshot_path),
            candidates=[],
            page_summary=PageSummary(top_texts=[], candidate_count=0),
            vlm_result={
                "structured": {
                    "page_semantics": {"domain": "im", "view": "home"}
                }
            },
        )
        result = ui_state_module._should_read_document_region(ui_state)
        self.assertFalse(result)

    def test_returns_true_for_docs_keywords_in_candidates(self):
        ui_state = UIState(
            screenshot_path=str(self.screenshot_path),
            candidates=[
                UICandidate(
                    id="c1", text="打开文档", bbox=[0, 0, 100, 100],
                    center=[50, 50], role="button", clickable=True,
                    editable=False, source="ocr",
                )
            ],
            page_summary=PageSummary(top_texts=["文档标题"], candidate_count=1),
            vlm_result=None,
        )
        result = ui_state_module._should_read_document_region(ui_state)
        self.assertTrue(result)

    def test_returns_false_for_search_page(self):
        ui_state = UIState(
            screenshot_path=str(self.screenshot_path),
            candidates=[],
            page_summary=PageSummary(
                top_texts=["搜索框"],
                has_search_box=True,
                candidate_count=1,
            ),
            vlm_result={
                "structured": {
                    "page_semantics": {"domain": "unknown", "view": "search"}
                }
            },
        )
        result = ui_state_module._should_read_document_region(ui_state)
        self.assertFalse(result)

    def test_returns_false_when_no_docs_signals(self):
        ui_state = UIState(
            screenshot_path=str(self.screenshot_path),
            candidates=[
                UICandidate(
                    id="c1", text="聊天列表", bbox=[0, 0, 100, 100],
                    center=[50, 50], role="list_item", clickable=True,
                    editable=False, source="uia",
                )
            ],
            page_summary=PageSummary(
                top_texts=["群聊", "消息"],
                candidate_count=1,
            ),
            vlm_result=None,
        )
        result = ui_state_module._should_read_document_region(ui_state)
        self.assertFalse(result)


# ── UIState document_region_summary integration Tests ───────────────────────


class TestUIStateDocumentRegionSummary(unittest.TestCase):

    def test_ui_state_serializes_document_region_summary(self):
        ui_state = UIState(
            screenshot_path="/tmp/test.png",
            candidates=[],
            page_summary=PageSummary(candidate_count=0),
            document_region_summary={
                "visible_title": "测试文档",
                "visible_texts": ["文本一"],
                "confidence": 0.8,
            },
        )
        d = ui_state.to_dict()
        self.assertIn("document_region_summary", d)
        self.assertEqual(d["document_region_summary"]["visible_title"], "测试文档")
        self.assertEqual(d["document_region_summary"]["visible_texts"], ["文本一"])

    def test_ui_state_omits_document_region_summary_when_none(self):
        ui_state = UIState(
            screenshot_path="/tmp/test.png",
            candidates=[],
            page_summary=PageSummary(candidate_count=0),
        )
        d = ui_state.to_dict()
        self.assertNotIn("document_region_summary", d)

    def test_ui_state_from_dict_restores_document_region_summary(self):
        data = {
            "screenshot_path": "/tmp/test.png",
            "candidates": [],
            "page_summary": {"candidate_count": 0},
            "document_region_summary": {
                "visible_title": "标题",
                "visible_texts": ["内容"],
                "confidence": 0.7,
            },
        }
        ui_state = UIState.from_dict(data)
        self.assertIsNotNone(ui_state.document_region_summary)
        self.assertEqual(
            ui_state.document_region_summary["visible_title"], "标题"
        )

    def test_ui_state_validates_document_region_summary_type(self):
        with self.assertRaises(Exception):
            UIState(
                screenshot_path="/tmp/test.png",
                candidates=[],
                page_summary=PageSummary(candidate_count=0),
                document_region_summary="not a dict",
            )


# ── collect_ui_state with document_region integration Tests ──────────────────


class TestCollectUIStateWithDocumentRegion(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.artifact_root = Path(self.tmpdir)
        self.screenshot_path = self.artifact_root / "test.png"
        _make_screenshot_png(self.screenshot_path, 1920, 1080)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_fake_capturer(self):
        capture_module = import_module("capture.screenshot")
        CapturedImage = capture_module.CapturedImage
        ScreenBounds = capture_module.ScreenBounds

        class FakeBackend:
            def grab_fullscreen(self, monitor_index):
                return CapturedImage(
                    image=Image.new("RGB", (1920, 1080)),
                    bounds=ScreenBounds(x=0, y=0, width=1920, height=1080),
                )

        return capture_module.ScreenshotCapturer(
            config=capture_module.ScreenshotCaptureConfig(
                artifact_root=str(self.artifact_root),
                monitor_index=1,
                window_title=None,
                focus_window=False,
                maximize_window=False,
            ),
            backend=FakeBackend(),
        )

    def _make_fake_ocr(self):
        ocr_module = import_module("perception.ocr")
        OCRTextBlock = ocr_module.OCRTextBlock

        class FakeOCRBackend:
            def extract(self, image, config):
                return [
                    OCRTextBlock(
                        text="文档正文", bbox=[400, 100, 1800, 1000],
                        confidence=0.9,
                    ),
                ]
        return FakeOCRBackend()

    def _make_fake_uia(self):
        uia_module = import_module("perception.uia")
        UIAElement = uia_module.UIAElement

        class FakeUIABackend:
            def extract(self, window, config):
                return [
                    UIAElement(
                        name="文档编辑区",
                        role="document",
                        bbox=[400, 100, 1800, 1000],
                        enabled=True,
                        visible=True,
                    ),
                ]
        return FakeUIABackend()

    def test_collect_ui_state_with_document_region_enabled_and_docs_page(self):
        capturer = self._make_fake_capturer()
        # Create a fake VLM provider that returns a docs page
        vlm_json = json.dumps({
            "page_type": "docs",
            "page_summary_enhancement": "docs page",
            "page_semantics": {"domain": "docs", "view": "detail"},
            "target_candidate_id": "ocr_c1_strict",
            "reranked_candidate_ids": [],
            "confidence": 0.8,
            "reason": "test",
            "uncertain": False,
            "suggested_fallback": "rule_fallback",
        })

        class DocsVLMProvider:
            def analyze(self, request):
                result_module = import_module("vlm.provider")
                return result_module.VLMCallResult(
                    ok=True,
                    provider="fake",
                    model="fake",
                    api_base="fake",
                    content=vlm_json,
                )

        doc_vlm_json = json.dumps({
            "visible_title": "项目文档",
            "visible_sections": ["第一章"],
            "visible_texts": ["文档内容"],
            "target_keywords_found": [],
            "is_edit_mode": False,
            "is_target_document": True,
            "body_content_summary": "项目文档的内容摘要。",
            "confidence": 0.85,
        })

        class DocRegionVLMProvider:
            def analyze(self, request):
                result_module = import_module("vlm.provider")
                return result_module.VLMCallResult(
                    ok=True,
                    provider="fake",
                    model="fake",
                    api_base="fake",
                    content=doc_vlm_json,
                )

        # We need two separate VLM providers: one for the main VLM call,
        # one for the document region call. The main VLM call uses the
        # vlm_provider parameter; the document region call also uses it.
        # Use the DocRegionVLMProvider for both - it returns docs semantics
        # and also doc content.
        class CombinedVLMProvider:
            def __init__(self):
                self.call_count = 0

            def analyze(self, request):
                self.call_count += 1
                result_module = import_module("vlm.provider")
                if self.call_count == 1:
                    # First call: main VLM analysis (page_semantics)
                    return result_module.VLMCallResult(
                        ok=True, provider="fake", model="fake",
                        api_base="fake", content=vlm_json,
                    )
                else:
                    # Second call: document region reading
                    return result_module.VLMCallResult(
                        ok=True, provider="fake", model="fake",
                        api_base="fake", content=doc_vlm_json,
                    )

        combined_provider = CombinedVLMProvider()

        ui_state = collect_ui_state(
            run_id="test_doc_region",
            step_idx=0,
            phase="before",
            artifact_root=str(self.artifact_root),
            monitor_index=1,
            capture_window_title=None,
            maximize_capture_window=False,
            focus_capture_window=False,
            screenshot_capturer=capturer,
            ocr_backend=self._make_fake_ocr(),
            uia_backend=self._make_fake_uia(),
            enable_vlm=True,
            vlm_provider=combined_provider,
            enable_document_region=True,
            target_document_title="项目文档",
            window=None,
            window_title=None,
            save_json=False,
        )
        self.assertIsNotNone(ui_state.document_region_summary)
        doc = ui_state.document_region_summary
        self.assertEqual(doc["visible_title"], "项目文档")
        self.assertIn("文档内容", doc["visible_texts"])
        self.assertTrue(doc["is_target_document"])
        self.assertEqual(doc["confidence"], 0.85)

    def test_collect_ui_state_with_document_region_disabled(self):
        capturer = self._make_fake_capturer()

        ui_state = collect_ui_state(
            run_id="test_no_doc",
            step_idx=0,
            phase="before",
            artifact_root=str(self.artifact_root),
            monitor_index=1,
            capture_window_title=None,
            maximize_capture_window=False,
            focus_capture_window=False,
            screenshot_capturer=capturer,
            ocr_backend=self._make_fake_ocr(),
            uia_backend=self._make_fake_uia(),
            enable_vlm=False,
            enable_document_region=False,
            window=None,
            window_title=None,
            save_json=False,
        )
        self.assertIsNone(ui_state.document_region_summary)

    def test_get_ui_state_passes_through_document_region_params(self):
        capturer = self._make_fake_capturer()

        result = get_ui_state(
            run_id="test_get",
            step_idx=0,
            phase="before",
            artifact_root=str(self.artifact_root),
            monitor_index=1,
            capture_window_title=None,
            maximize_capture_window=False,
            focus_capture_window=False,
            screenshot_capturer=capturer,
            ocr_backend=self._make_fake_ocr(),
            uia_backend=self._make_fake_uia(),
            enable_vlm=False,
            enable_document_region=True,
            target_document_title="测试标题",
            window=None,
            window_title=None,
            save_json=False,
        )
        # When enable_document_region=True and a docs-like page is detected,
        # document_region_summary is populated (even if empty/error).
        self.assertIn("document_region_summary", result)
        self.assertIsInstance(result["document_region_summary"], dict)


if __name__ == "__main__":
    unittest.main()
