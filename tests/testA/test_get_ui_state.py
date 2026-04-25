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
            )

            self.assertEqual(set(payload.keys()), {"screenshot_path", "candidates", "page_summary"})
            self.assertTrue(Path(payload["screenshot_path"]).exists())
            self.assertEqual(payload["page_summary"]["candidate_count"], 2)
            self.assertTrue(payload["page_summary"]["has_search_box"])
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
            )

            self.assertEqual(ui_state.page_summary.candidate_count, 2)
            self.assertFalse((Path(temp_dir) / "run_test" / "ui_state_step_002_after.json").exists())


if __name__ == "__main__":
    unittest.main()
