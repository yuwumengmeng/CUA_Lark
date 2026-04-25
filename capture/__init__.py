"""截图采集包入口。

这个包负责成员 A 4.23 第一项的截图采集链路：稳定抓取屏幕画面，
按 run/step/phase 统一落盘，并返回后续 perception、planner、executor
可消费的截图路径与坐标元信息。
"""

from .screenshot import (
    CapturedImage,
    CaptureError,
    MssScreenshotBackend,
    ScreenBounds,
    ScreenshotCaptureConfig,
    ScreenshotCapturer,
    ScreenshotResult,
    capture_after,
    capture_before,
    capture_screenshot,
    create_run_id,
)

__all__ = [
    "CapturedImage",
    "CaptureError",
    "MssScreenshotBackend",
    "ScreenBounds",
    "ScreenshotCaptureConfig",
    "ScreenshotCapturer",
    "ScreenshotResult",
    "capture_after",
    "capture_before",
    "capture_screenshot",
    "create_run_id",
]
