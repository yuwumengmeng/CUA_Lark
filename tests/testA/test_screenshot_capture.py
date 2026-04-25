"""成员 A 4.23 截图采集链路测试。

测试目标：
本文件验证 `capture/screenshot.py` 的首版截图协议是否稳定，包括：
1. before/after 截图是否按统一目录和命名规则落盘。
2. `ScreenshotResult` 是否返回固定字段和屏幕坐标元信息。
3. `run_id`、`step_idx`、`phase` 等非法输入是否会被拒绝。
4. 测试脚本是否可以被 `pytest` 收集，也可以被 `python 文件路径`
   直接运行，方便成员 A 单独自测。

重要约定：
这里不截真实桌面，不会向 `artifacts/` 写入真实屏幕图片。测试使用
`FakeScreenshotBackend` 生成固定尺寸的假图片，所以测试只覆盖“链路、
路径、命名、metadata、参数校验”，不覆盖操作系统真实截图能力。

直接运行：
```powershell
python tests\testA\test_screenshot_capture.py
```

通过 pytest 运行：
```powershell
python -m pytest tests\testA\test_screenshot_capture.py
```

真实截图手动验证请用：
```powershell
python -c "from capture import capture_before; r = capture_before(run_id='run_demo', step_idx=1); print(r.screenshot_path)"
```

新增或修改截图链路时的测试规则：
1. 单元测试继续使用 fake backend，不依赖当前电脑屏幕状态。
2. 如果新增字段，必须同时断言 `ScreenshotResult` 的属性和 `to_dict()` 输出。
3. 如果修改文件命名或目录结构，必须同步更新这里的路径断言。
4. 如果支持窗口截图或区域截图，必须新增对应 backend 测试，不能破坏
   现有全屏截图测试。
"""

import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

screenshot_module = import_module("capture.screenshot")
CapturedImage = screenshot_module.CapturedImage
CaptureError = screenshot_module.CaptureError
ScreenBounds = screenshot_module.ScreenBounds
ScreenshotCaptureConfig = screenshot_module.ScreenshotCaptureConfig
ScreenshotCapturer = screenshot_module.ScreenshotCapturer


class FakeScreenshotBackend:
    def __init__(
        self,
        *,
        size: tuple[int, int] = (80, 60),
        origin: tuple[int, int] = (0, 0),
    ) -> None:
        self.size = size
        self.origin = origin
        self.monitor_indexes: list[int] = []
        self.window_capture_calls: list[tuple[str, bool, bool, float]] = []

    def grab_fullscreen(self, monitor_index: int) -> CapturedImage:
        self.monitor_indexes.append(monitor_index)
        image = Image.new("RGB", self.size, color=(12, 34, 56))
        bounds = ScreenBounds(
            x=self.origin[0],
            y=self.origin[1],
            width=self.size[0],
            height=self.size[1],
        )
        return CapturedImage(image=image, bounds=bounds)

    def grab_window(
        self,
        window_title: str,
        *,
        focus_window: bool = True,
        maximize_window: bool = False,
        focus_delay_seconds: float = 0.5,
    ) -> CapturedImage:
        self.window_capture_calls.append(
            (window_title, focus_window, maximize_window, focus_delay_seconds)
        )
        image = Image.new("RGB", self.size, color=(45, 67, 89))
        bounds = ScreenBounds(
            x=self.origin[0],
            y=self.origin[1],
            width=self.size[0],
            height=self.size[1],
        )
        return CapturedImage(
            image=image,
            bounds=bounds,
            capture_target="window",
            window_title=window_title,
        )


class ScreenshotCaptureTests(unittest.TestCase):
    def test_before_and_after_screenshots_are_saved_with_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FakeScreenshotBackend(size=(64, 32), origin=(10, 20))
            capturer = ScreenshotCapturer(
                config=ScreenshotCaptureConfig(
                    artifact_root=Path(temp_dir),
                    monitor_index=1,
                ),
                backend=backend,
            )

            before = capturer.capture_before(run_id="run_test", step_idx=1)
            after = capturer.capture_after(run_id="run_test", step_idx=1)

            self.assertTrue(Path(before.screenshot_path).exists())
            self.assertTrue(Path(after.screenshot_path).exists())
            self.assertEqual(Path(before.screenshot_path).name, "step_001_before.png")
            self.assertEqual(Path(after.screenshot_path).name, "step_001_after.png")
            self.assertEqual(before.width, 64)
            self.assertEqual(before.height, 32)
            self.assertEqual(before.bbox, [10, 20, 74, 52])
            self.assertEqual(before.screen_origin, [10, 20])
            self.assertEqual(before.coordinate_space, "screen")
            self.assertEqual(before.monitor_index, 1)
            self.assertEqual(before.capture_target, "fullscreen")
            self.assertIsNone(before.window_title)
            self.assertEqual(backend.monitor_indexes, [1, 1])

            with Image.open(before.screenshot_path) as saved_image:
                self.assertEqual(saved_image.size, (64, 32))

    def test_capture_screenshot_creates_run_id_when_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            capturer = ScreenshotCapturer(
                config=ScreenshotCaptureConfig(artifact_root=Path(temp_dir)),
                backend=FakeScreenshotBackend(),
            )

            result = capturer.capture_screenshot(step_idx=0, phase="manual")

            self.assertTrue(result.run_id.startswith("run_"))
            self.assertEqual(Path(result.screenshot_path).name, "step_000_manual.png")
            self.assertTrue(Path(result.screenshot_path).exists())

    def test_window_screenshot_is_saved_with_window_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FakeScreenshotBackend(size=(100, 70), origin=(4000, -20))
            capturer = ScreenshotCapturer(
                config=ScreenshotCaptureConfig(
                    artifact_root=Path(temp_dir),
                    window_title="飞书",
                    focus_window=True,
                    maximize_window=True,
                    focus_delay_seconds=0.1,
                ),
                backend=backend,
            )

            result = capturer.capture_before(run_id="run_window", step_idx=1)

            self.assertTrue(Path(result.screenshot_path).exists())
            self.assertEqual(result.width, 100)
            self.assertEqual(result.height, 70)
            self.assertEqual(result.bbox, [4000, -20, 4100, 50])
            self.assertEqual(result.screen_origin, [4000, -20])
            self.assertEqual(result.capture_target, "window")
            self.assertEqual(result.window_title, "飞书")
            self.assertEqual(result.to_dict()["capture_target"], "window")
            self.assertEqual(result.to_dict()["window_title"], "飞书")
            self.assertEqual(backend.window_capture_calls, [("飞书", True, True, 0.1)])

    def test_invalid_step_idx_is_rejected(self):
        capturer = ScreenshotCapturer(backend=FakeScreenshotBackend())

        with self.assertRaises(CaptureError):
            capturer.capture_screenshot(run_id="run_test", step_idx=-1)

    def test_invalid_phase_is_rejected(self):
        capturer = ScreenshotCapturer(backend=FakeScreenshotBackend())

        with self.assertRaises(CaptureError):
            capturer.capture_screenshot(run_id="run_test", phase="middle")

    def test_invalid_run_id_is_rejected(self):
        capturer = ScreenshotCapturer(backend=FakeScreenshotBackend())

        with self.assertRaises(CaptureError):
            capturer.capture_screenshot(run_id="bad/run", step_idx=1)


if __name__ == "__main__":
    unittest.main()
