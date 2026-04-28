"""成员 A 4.25 `get_ui_state()` 总接口测试。

测试目标：
本文件验证 `perception/ui_state.py` 是否能把截图、OCR、UIA、Candidate
Builder 和 UI state JSON 落盘串成一个稳定总入口。

重要约定：
测试使用 fake screenshot backend、fake OCR backend 和 fake UIA backend，
不会访问真实桌面、真实飞书窗口或真实 Tesseract。

直接运行：
```powershell
python tests\testA\test_get_ui_state.py
```

通过 pytest 运行：
```powershell
python -m pytest tests\testA\test_get_ui_state.py
```
"""

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

capture_module = import_module("capture.screenshot")
ocr_module = import_module("perception.ocr")
ui_state_module = import_module("perception.ui_state")
uia_module = import_module("perception.uia")

CapturedImage = capture_module.CapturedImage
ScreenBounds = capture_module.ScreenBounds
ScreenshotCaptureConfig = capture_module.ScreenshotCaptureConfig
ScreenshotCapturer = capture_module.ScreenshotCapturer
OCRConfig = ocr_module.OCRConfig
OCRTextBlock = ocr_module.OCRTextBlock
UIAConfig = uia_module.UIAConfig
UIAElement = uia_module.UIAElement
collect_ui_state = ui_state_module.collect_ui_state
get_ui_state = ui_state_module.get_ui_state
VLMRequest = import_module("vlm").VLMRequest


class FakeScreenshotBackend:
    def __init__(self) -> None:
        self.monitor_indexes: list[int] = []

    def grab_fullscreen(self, monitor_index: int) -> CapturedImage:
        self.monitor_indexes.append(monitor_index)
        image = Image.new("RGB", (120, 80), color=(255, 255, 255))
        return CapturedImage(
            image=image,
            bounds=ScreenBounds(x=4000, y=-20, width=120, height=80),
        )


class FakeOCRBackend:
    def __init__(self) -> None:
        self.seen_sizes: list[tuple[int, int]] = []

    def extract(self, image: Image.Image, config: OCRConfig) -> list[OCRTextBlock]:
        self.seen_sizes.append(image.size)
        return [
            OCRTextBlock(text="消息", bbox=[10, 20, 50, 40], confidence=0.92),
        ]


class FakeUIABackend:
    def __init__(self) -> None:
        self.seen_window_titles: list[str | None] = []

    def extract(self, window: object, config: UIAConfig) -> list[UIAElement]:
        self.seen_window_titles.append(config.window_title)
        return [
            UIAElement(
                name="搜索",
                role="EditControl",
                bbox=[4200, 80, 4480, 128],
                enabled=True,
                visible=True,
            )
        ]


class FailingVLMProvider:
    def __init__(self) -> None:
        self.requests: list[VLMRequest] = []

    def analyze(self, request: VLMRequest):
        self.requests.append(request)
        raise RuntimeError("vlm unavailable")


class FakeVLMProvider:
    def __init__(self, content: str = '{"page_type":"im","target_candidate_id":"elem_0001","confidence":0.9,"uncertain":false}') -> None:
        self.requests: list[VLMRequest] = []
        self.content = content

    def analyze(self, request: VLMRequest):
        self.requests.append(request)
        VLMCallResult = import_module("vlm.provider").VLMCallResult
        return VLMCallResult(
            ok=True,
            provider="fake",
            model="fake-model",
            api_base="http://fake",
            content=self.content,
            attempts=1,
            fallback_used=False,
        )


class GetUIStateTests(unittest.TestCase):
    def test_get_ui_state_returns_dict_and_saves_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot_backend = FakeScreenshotBackend()
            ocr_backend = FakeOCRBackend()
            uia_backend = FakeUIABackend()
            capturer = ScreenshotCapturer(
                config=ScreenshotCaptureConfig(
                    artifact_root=Path(temp_dir),
                    monitor_index=1,
                ),
                backend=screenshot_backend,
            )

            payload = get_ui_state(
                run_id="run_test",
                step_idx=1,
                phase="before",
                screenshot_capturer=capturer,
                ocr_backend=ocr_backend,
                uia_backend=uia_backend,
                enable_vlm=False,
            )

            self.assertEqual(
                set(payload.keys()),
                {"screenshot_path", "candidates", "page_summary", "screen_meta", "perception_health", "candidate_stats"},
            )
            self.assertTrue(Path(payload["screenshot_path"]).exists())
            self.assertEqual(payload["page_summary"]["candidate_count"], 2)
            self.assertTrue(payload["page_summary"]["has_search_box"])
            self.assertEqual(payload["screen_meta"]["screenshot_size"], [120, 80])
            self.assertEqual(payload["screen_meta"]["screen_resolution"], [120, 80])
            self.assertIsNone(payload["screen_meta"]["window_rect"])
            self.assertEqual(payload["screen_meta"]["screen_origin"], [4000, -20])
            self.assertEqual(payload["screen_meta"]["capture_target"], "fullscreen")
            self.assertEqual(payload["screen_meta"]["monitor_index"], 1)
            self.assertEqual(payload["screen_meta"]["monitor_count"], 1)
            self.assertFalse(payload["screen_meta"]["is_multi_monitor"])
            self.assertEqual(
                payload["screen_meta"]["candidate_coordinate_source_counts"],
                {"ocr": 1, "uia": 1},
            )
            self.assertEqual(payload["screen_meta"]["negative_coordinate_candidate_count"], 0)
            self.assertEqual(payload["screen_meta"]["out_of_screenshot_candidate_count"], 1)
            message_candidate = next(
                candidate for candidate in payload["candidates"] if candidate["text"] == "消息"
            )
            self.assertEqual(message_candidate["bbox"], [4010, 0, 4050, 20])
            self.assertEqual(message_candidate["center"], [4030, 10])
            self.assertEqual(ocr_backend.seen_sizes, [(120, 80)])
            self.assertEqual(uia_backend.seen_window_titles, ["飞书"])

            saved_path = Path(temp_dir) / "run_test" / "ui_state_step_001_before.json"
            self.assertTrue(saved_path.exists())
            saved_payload = json.loads(saved_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_payload, payload)
            screen_meta_path = Path(temp_dir) / "run_test" / "screen_meta_step_001_before.json"
            self.assertTrue(screen_meta_path.exists())
            saved_screen_meta = json.loads(screen_meta_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_screen_meta, payload["screen_meta"])

    def test_collect_ui_state_can_skip_json_save(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            capturer = ScreenshotCapturer(
                config=ScreenshotCaptureConfig(artifact_root=Path(temp_dir)),
                backend=FakeScreenshotBackend(),
            )

            ui_state = collect_ui_state(
                run_id="run_test",
                step_idx=2,
                phase="after",
                save_json=False,
                screenshot_capturer=capturer,
                ocr_backend=FakeOCRBackend(),
                uia_backend=FakeUIABackend(),
                enable_vlm=False,
            )

            self.assertEqual(ui_state.page_summary.candidate_count, 2)
            self.assertIsNotNone(ui_state.screen_meta)
            self.assertFalse((Path(temp_dir) / "run_test" / "ui_state_step_002_after.json").exists())
            self.assertFalse((Path(temp_dir) / "run_test" / "screen_meta_step_002_after.json").exists())

    def test_get_ui_state_vlm_failure_keeps_original_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot_backend = FakeScreenshotBackend()
            vlm_provider = FailingVLMProvider()
            capturer = ScreenshotCapturer(
                config=ScreenshotCaptureConfig(
                    artifact_root=Path(temp_dir),
                    monitor_index=1,
                ),
                backend=screenshot_backend,
            )

            payload = get_ui_state(
                run_id="run_test",
                step_idx=3,
                phase="before",
                screenshot_capturer=capturer,
                ocr_backend=FakeOCRBackend(),
                uia_backend=FakeUIABackend(),
                enable_vlm=True,
                vlm_provider=vlm_provider,
            )

            self.assertEqual(payload["page_summary"]["candidate_count"], 2)
            self.assertEqual(len(payload["candidates"]), 2)
            self.assertIn("vlm_result", payload)
            self.assertFalse(payload["vlm_result"]["ok"])
            self.assertTrue(payload["vlm_result"]["fallback_used"])
            self.assertEqual(payload["vlm_result"]["error_type"], "RuntimeError")
            self.assertEqual(len(vlm_provider.requests), 1)

    def test_vlm_failure_includes_structured_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vlm_provider = FailingVLMProvider()
            capturer = ScreenshotCapturer(
                config=ScreenshotCaptureConfig(artifact_root=Path(temp_dir)),
                backend=FakeScreenshotBackend(),
            )

            payload = get_ui_state(
                run_id="run_test",
                step_idx=4,
                phase="before",
                screenshot_capturer=capturer,
                ocr_backend=FakeOCRBackend(),
                uia_backend=FakeUIABackend(),
                enable_vlm=True,
                vlm_provider=vlm_provider,
            )

            self.assertIn("structured", payload["vlm_result"])
            structured = payload["vlm_result"]["structured"]
            self.assertFalse(structured["parse_ok"])
            self.assertTrue(structured["uncertain"])
            self.assertEqual(structured["page_type"], "unknown")
            self.assertIsNone(structured["target_candidate_id"])
            self.assertEqual(structured["suggested_fallback"], "rule_fallback")

    def test_vlm_success_includes_structured_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vlm_provider = FakeVLMProvider(
                content=json.dumps({
                    "page_type": "im",
                    "page_summary_enhancement": "Chat list with search box",
                    "page_semantics": {
                        "domain": "im",
                        "view": "list",
                        "summary": "Chat list with search box",
                        "signals": {
                            "search_available": True,
                            "list_available": True,
                            "target_visible": False,
                        },
                        "goal_progress": "in_progress",
                        "confidence": 0.86,
                    },
                    "target_candidate_id": "elem_0001",
                    "reranked_candidate_ids": ["elem_0001", "elem_0002"],
                    "confidence": 0.92,
                    "reason": "search box at top",
                    "uncertain": False,
                    "suggested_fallback": "rule_fallback",
                }),
            )
            capturer = ScreenshotCapturer(
                config=ScreenshotCaptureConfig(artifact_root=Path(temp_dir)),
                backend=FakeScreenshotBackend(),
            )

            payload = get_ui_state(
                run_id="run_test",
                step_idx=5,
                phase="before",
                screenshot_capturer=capturer,
                ocr_backend=FakeOCRBackend(),
                uia_backend=FakeUIABackend(),
                enable_vlm=True,
                vlm_provider=vlm_provider,
            )

            self.assertIn("structured", payload["vlm_result"])
            structured = payload["vlm_result"]["structured"]
            self.assertTrue(structured["parse_ok"])
            self.assertEqual(structured["page_type"], "im")
            self.assertEqual(structured["target_candidate_id"], "elem_0001")
            self.assertEqual(structured["reranked_candidate_ids"], ["elem_0001", "elem_0002"])
            self.assertAlmostEqual(structured["confidence"], 0.92)
            self.assertFalse(structured["uncertain"])
            self.assertTrue(structured["page_semantics"]["signals"]["search_available"])
            self.assertIn("semantic_summary", payload["page_summary"])
            self.assertTrue(payload["page_summary"]["semantic_summary"]["signals"]["list_available"])

    def test_vlm_step_fields_passed_to_provider(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vlm_provider = FakeVLMProvider()
            capturer = ScreenshotCapturer(
                config=ScreenshotCaptureConfig(artifact_root=Path(temp_dir)),
                backend=FakeScreenshotBackend(),
            )

            get_ui_state(
                run_id="run_test",
                step_idx=6,
                phase="before",
                screenshot_capturer=capturer,
                ocr_backend=FakeOCRBackend(),
                uia_backend=FakeUIABackend(),
                enable_vlm=True,
                vlm_provider=vlm_provider,
                step_name="open_search",
                step_goal="locate and click the search box",
                region_hint="top-left area",
            )

            self.assertEqual(len(vlm_provider.requests), 1)
            req = vlm_provider.requests[0]
            self.assertEqual(req.step_name, "open_search")
            self.assertEqual(req.step_goal, "locate and click the search box")
            self.assertEqual(req.region_hint, "top-left area")

    def test_vlm_invalid_candidate_id_demoted_in_structured(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vlm_provider = FakeVLMProvider(
                content=json.dumps({
                    "page_type": "im",
                    "target_candidate_id": "elem_9999",
                    "confidence": 0.8,
                    "uncertain": False,
                }),
            )
            capturer = ScreenshotCapturer(
                config=ScreenshotCaptureConfig(artifact_root=Path(temp_dir)),
                backend=FakeScreenshotBackend(),
            )

            payload = get_ui_state(
                run_id="run_test",
                step_idx=7,
                phase="before",
                screenshot_capturer=capturer,
                ocr_backend=FakeOCRBackend(),
                uia_backend=FakeUIABackend(),
                enable_vlm=True,
                vlm_provider=vlm_provider,
            )

            structured = payload["vlm_result"]["structured"]
            self.assertTrue(structured["parse_ok"])
            self.assertIsNone(structured["target_candidate_id"])
            self.assertTrue(structured["uncertain"])


if __name__ == "__main__":
    unittest.main()
