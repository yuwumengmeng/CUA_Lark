"""成员 A 4.23 截图采集链路。

职责边界：
本文件只负责“把当前屏幕画面稳定保存成 PNG，并返回截图元信息”。
它不做 OCR、不做 UIA、不做候选元素合并，也不负责点击执行。后续
`perception/ocr.py`、`perception/uia.py`、`candidate_builder.py`
可以消费这里产出的 `ScreenshotResult`。

首版方案：
默认使用 `mss` 做截图，`monitor_index=1` 表示主屏。未指定窗口标题时
截取主屏全屏；指定 `window_title` 时只截取匹配标题的桌面窗口区域。
两种模式都返回屏幕绝对 `bbox/screen_origin`，OCR 局部坐标可以据此转换
为执行器可点击的屏幕坐标。

公开接口：
`capture_screenshot(...)`
    通用截图入口。默认 phase 为 `manual`。
`capture_before(...)`
    执行动作前截图，phase 固定为 `before`。
`capture_after(...)`
    执行动作后截图，phase 固定为 `after`。
`ScreenshotCapturer`
    可注入配置和 backend 的类，测试或高级调用时使用。
`ScreenshotResult.to_dict()`
    给 planner / recorder / 日志系统使用的稳定 dict 输出。

保存规则：
所有截图统一落到：
`artifacts/runs/<run_id>/screenshots/step_xxx_<phase>.png`
例如：
`artifacts/runs/run_demo/screenshots/step_001_before.png`
`artifacts/runs/run_demo/screenshots/step_001_after.png`

返回字段规则：
`screenshot_path`
    PNG 文件路径。
`run_id`
    一次任务运行的唯一 ID。同一次 run 的所有步骤必须复用同一个值。
`step_idx`
    非负整数。建议从 1 开始对应真实动作步骤，0 可用于手动调试。
`phase`
    只能是 `before`、`after`、`manual`。
`bbox`
    截图覆盖的屏幕区域，格式为 `[x1, y1, x2, y2]`。
`screen_origin`
    截图坐标原点，格式为 `[x, y]`。
`coordinate_space`
    当前固定为 `screen`，表示屏幕绝对坐标。
`capture_target`
    截图目标，取值为 `fullscreen` 或 `window`。
`window_title`
    窗口截图时使用的标题关键字；全屏截图时为 `None`。
`focus_window`
    窗口截图前是否自动激活目标窗口。默认 `True`，用于避免按 Enter 后
    VSCode/终端遮挡飞书窗口。

使用示例：
```python
from capture import capture_before, capture_after

run_id = "run_demo"
before = capture_before(run_id=run_id, step_idx=1)
after = capture_after(run_id=run_id, step_idx=1)

print(before.screenshot_path)
print(before.to_dict())
```

命令行快速验证真实截图：
```powershell
python -c "from capture import capture_before; r = capture_before(run_id='run_demo', step_idx=1); print(r.screenshot_path)"
```

协作规则：
1. A/B/C 共用 `screenshot_path`、`bbox`、`screen_origin`、`coordinate_space`
   这些字段，不从文件名反推坐标或步骤信息。
2. C 的 executor 做动作前调用 `capture_before`，动作后调用 `capture_after`。
3. 同一个动作步骤的 before/after 必须使用相同的 `run_id` 和 `step_idx`。
4. 窗口截图失败时默认抛出 `CaptureError`，避免误把整个桌面当作飞书窗口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any, Literal, Protocol
from uuid import uuid4

from PIL import Image


SCREENSHOT_SCHEMA_VERSION = "screenshot.v1"
DEFAULT_COORDINATE_SPACE = "screen"
SUPPORTED_PHASES = {"before", "after", "manual"}
ScreenshotPhase = Literal["before", "after", "manual"]


class CaptureError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ScreenBounds:
    x: int
    y: int
    width: int
    height: int

    @property
    def bbox(self) -> list[int]:
        return [self.x, self.y, self.x + self.width, self.y + self.height]

    @property
    def origin(self) -> list[int]:
        return [self.x, self.y]


@dataclass(frozen=True, slots=True)
class CapturedImage:
    image: Image.Image
    bounds: ScreenBounds
    capture_target: str = "fullscreen"
    window_title: str | None = None


@dataclass(frozen=True, slots=True)
class ScreenshotResult:
    screenshot_path: str
    run_id: str
    step_idx: int
    phase: str
    width: int
    height: int
    bbox: list[int]
    coordinate_space: str
    screen_origin: list[int]
    monitor_index: int
    created_at: str
    capture_target: str = "fullscreen"
    window_title: str | None = None
    version: str = SCREENSHOT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "screenshot_path": self.screenshot_path,
            "run_id": self.run_id,
            "step_idx": self.step_idx,
            "phase": self.phase,
            "width": self.width,
            "height": self.height,
            "bbox": self.bbox,
            "coordinate_space": self.coordinate_space,
            "screen_origin": self.screen_origin,
            "monitor_index": self.monitor_index,
            "created_at": self.created_at,
            "capture_target": self.capture_target,
            "window_title": self.window_title,
        }


class ScreenshotBackend(Protocol):
    def grab_fullscreen(self, monitor_index: int) -> CapturedImage:
        ...


class MssScreenshotBackend:
    def grab_fullscreen(self, monitor_index: int) -> CapturedImage:
        import mss

        with mss.mss() as screen_capture:
            if monitor_index < 0 or monitor_index >= len(screen_capture.monitors):
                raise CaptureError(f"Unsupported monitor_index: {monitor_index}")

            monitor = screen_capture.monitors[monitor_index]
            raw_image = screen_capture.grab(monitor)
            image = Image.frombytes("RGB", raw_image.size, raw_image.bgra, "raw", "BGRX")
            bounds = ScreenBounds(
                x=int(monitor["left"]),
                y=int(monitor["top"]),
                width=int(monitor["width"]),
                height=int(monitor["height"]),
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
        import mss

        hwnd, _, bounds = _find_window(window_title)
        if focus_window or maximize_window:
            _activate_window(
                hwnd,
                focus_window=focus_window,
                maximize_window=maximize_window,
            )
            if focus_delay_seconds > 0:
                sleep(focus_delay_seconds)
            bounds = _window_bounds(hwnd)
        monitor = {
            "left": bounds.x,
            "top": bounds.y,
            "width": bounds.width,
            "height": bounds.height,
        }
        with mss.mss() as screen_capture:
            raw_image = screen_capture.grab(monitor)
            image = Image.frombytes("RGB", raw_image.size, raw_image.bgra, "raw", "BGRX")
        return CapturedImage(
            image=image,
            bounds=bounds,
            capture_target="window",
            window_title=window_title,
        )


@dataclass(slots=True)
class ScreenshotCaptureConfig:
    artifact_root: Path = field(default_factory=lambda: Path("artifacts") / "runs")
    monitor_index: int = 1
    coordinate_space: str = DEFAULT_COORDINATE_SPACE
    window_title: str | None = None
    focus_window: bool = True
    maximize_window: bool = True
    focus_delay_seconds: float = 0.5

    def __post_init__(self) -> None:
        self.artifact_root = Path(self.artifact_root)
        if self.monitor_index < 0:
            raise CaptureError("monitor_index must be a non-negative integer")
        if not self.coordinate_space:
            raise CaptureError("coordinate_space cannot be empty")
        if self.window_title is not None and not self.window_title.strip():
            raise CaptureError("window_title cannot be empty")
        if not isinstance(self.focus_window, bool):
            raise CaptureError("focus_window must be a boolean")
        if not isinstance(self.maximize_window, bool):
            raise CaptureError("maximize_window must be a boolean")
        if self.focus_delay_seconds < 0:
            raise CaptureError("focus_delay_seconds must be non-negative")


class ScreenshotCapturer:
    def __init__(
        self,
        config: ScreenshotCaptureConfig | None = None,
        backend: ScreenshotBackend | None = None,
    ) -> None:
        self.config = config or ScreenshotCaptureConfig()
        self.backend = backend or MssScreenshotBackend()

    def capture_screenshot(
        self,
        *,
        run_id: str | None = None,
        step_idx: int = 0,
        phase: ScreenshotPhase = "manual",
    ) -> ScreenshotResult:
        normalized_run_id = _normalize_run_id(run_id)
        normalized_step_idx = _normalize_step_idx(step_idx)
        normalized_phase = _normalize_phase(phase)
        captured = self._grab()
        screenshot_path = self._screenshot_path(
            normalized_run_id,
            normalized_step_idx,
            normalized_phase,
        )
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        captured.image.save(screenshot_path, format="PNG")
        width, height = captured.image.size
        return ScreenshotResult(
            screenshot_path=str(screenshot_path),
            run_id=normalized_run_id,
            step_idx=normalized_step_idx,
            phase=normalized_phase,
            width=width,
            height=height,
            bbox=captured.bounds.bbox,
            coordinate_space=self.config.coordinate_space,
            screen_origin=captured.bounds.origin,
            monitor_index=self.config.monitor_index,
            created_at=_utc_now(),
            capture_target=captured.capture_target,
            window_title=captured.window_title,
        )

    def capture_before(self, *, run_id: str | None = None, step_idx: int = 0) -> ScreenshotResult:
        return self.capture_screenshot(run_id=run_id, step_idx=step_idx, phase="before")

    def capture_after(self, *, run_id: str | None = None, step_idx: int = 0) -> ScreenshotResult:
        return self.capture_screenshot(run_id=run_id, step_idx=step_idx, phase="after")

    def _screenshot_path(self, run_id: str, step_idx: int, phase: str) -> Path:
        return (
            self.config.artifact_root
            / run_id
            / "screenshots"
            / f"step_{step_idx:03d}_{phase}.png"
        )

    def _grab(self) -> CapturedImage:
        if self.config.window_title:
            grab_window = getattr(self.backend, "grab_window", None)
            if grab_window is None:
                raise CaptureError("Configured screenshot backend does not support window capture")
            return grab_window(
                self.config.window_title,
                focus_window=self.config.focus_window,
                maximize_window=self.config.maximize_window,
                focus_delay_seconds=self.config.focus_delay_seconds,
            )
        return self.backend.grab_fullscreen(self.config.monitor_index)


def capture_screenshot(
    *,
    run_id: str | None = None,
    step_idx: int = 0,
    phase: ScreenshotPhase = "manual",
    artifact_root: str | Path | None = None,
    monitor_index: int = 1,
    window_title: str | None = None,
    focus_window: bool = True,
    maximize_window: bool | None = None,
    focus_delay_seconds: float = 0.5,
) -> ScreenshotResult:
    resolved_maximize_window = _resolve_maximize_window(maximize_window, window_title)
    config = ScreenshotCaptureConfig(
        artifact_root=Path(artifact_root) if artifact_root is not None else Path("artifacts") / "runs",
        monitor_index=monitor_index,
        window_title=window_title,
        focus_window=focus_window,
        maximize_window=resolved_maximize_window,
        focus_delay_seconds=focus_delay_seconds,
    )
    return ScreenshotCapturer(config=config).capture_screenshot(
        run_id=run_id,
        step_idx=step_idx,
        phase=phase,
    )


def capture_before(
    *,
    run_id: str | None = None,
    step_idx: int = 0,
    artifact_root: str | Path | None = None,
    monitor_index: int = 1,
    window_title: str | None = None,
    focus_window: bool = True,
    maximize_window: bool | None = None,
    focus_delay_seconds: float = 0.5,
) -> ScreenshotResult:
    return capture_screenshot(
        run_id=run_id,
        step_idx=step_idx,
        phase="before",
        artifact_root=artifact_root,
        monitor_index=monitor_index,
        window_title=window_title,
        focus_window=focus_window,
        maximize_window=maximize_window,
        focus_delay_seconds=focus_delay_seconds,
    )


def capture_after(
    *,
    run_id: str | None = None,
    step_idx: int = 0,
    artifact_root: str | Path | None = None,
    monitor_index: int = 1,
    window_title: str | None = None,
    focus_window: bool = True,
    maximize_window: bool | None = None,
    focus_delay_seconds: float = 0.5,
) -> ScreenshotResult:
    return capture_screenshot(
        run_id=run_id,
        step_idx=step_idx,
        phase="after",
        artifact_root=artifact_root,
        monitor_index=monitor_index,
        window_title=window_title,
        focus_window=focus_window,
        maximize_window=maximize_window,
        focus_delay_seconds=focus_delay_seconds,
    )


def create_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"


def _normalize_run_id(run_id: str | None) -> str:
    text = (run_id or create_run_id()).strip()
    if not text:
        raise CaptureError("run_id cannot be empty")
    if any(char in text for char in '<>:"/\\|?*'):
        raise CaptureError(f"run_id contains invalid path characters: {text}")
    return text


def _normalize_step_idx(step_idx: int) -> int:
    if isinstance(step_idx, bool) or not isinstance(step_idx, int) or step_idx < 0:
        raise CaptureError("step_idx must be a non-negative integer")
    return step_idx


def _normalize_phase(phase: str) -> str:
    if phase not in SUPPORTED_PHASES:
        allowed = ", ".join(sorted(SUPPORTED_PHASES))
        raise CaptureError(f"phase must be one of: {allowed}")
    return phase


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_maximize_window(maximize_window: bool | None, window_title: str | None) -> bool:
    if maximize_window is None:
        return window_title is not None
    if not isinstance(maximize_window, bool):
        raise CaptureError("maximize_window must be a boolean or None")
    return maximize_window


def _find_window_bounds(window_title: str) -> ScreenBounds:
    _, _, bounds = _find_window(window_title)
    return bounds


def _find_window(window_title: str) -> tuple[int, str, ScreenBounds]:
    try:
        import win32gui
    except ImportError as exc:
        raise CaptureError("pywin32 is required for window capture") from exc

    title_keyword = window_title.strip()
    matches: list[tuple[int, str, ScreenBounds]] = []

    def collect_window(hwnd: int, _: object) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd).strip()
        if title_keyword not in title:
            return
        try:
            bounds = _window_bounds(hwnd)
        except CaptureError:
            return
        matches.append((hwnd, title, bounds))

    win32gui.EnumWindows(collect_window, None)
    if not matches:
        raise CaptureError(f"Window was not found by title: {window_title}")

    return max(matches, key=lambda item: item[2].width * item[2].height)


def _window_bounds(hwnd: int) -> ScreenBounds:
    try:
        import win32gui
    except ImportError as exc:
        raise CaptureError("pywin32 is required for window capture") from exc

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    if right <= left or bottom <= top:
        raise CaptureError(f"Window has invalid bounds: {(left, top, right, bottom)}")
    return ScreenBounds(
        x=int(left),
        y=int(top),
        width=int(right - left),
        height=int(bottom - top),
    )


def _activate_window(
    hwnd: int,
    *,
    focus_window: bool = True,
    maximize_window: bool = False,
) -> None:
    try:
        import win32con
        import win32api
        import win32gui
    except ImportError as exc:
        raise CaptureError("pywin32 is required for window activation") from exc

    show_command = _resolve_show_command(
        hwnd,
        maximize_window=maximize_window,
        win32con_module=win32con,
        win32gui_module=win32gui,
    )

    try:
        win32gui.ShowWindow(hwnd, show_command)
    except Exception as exc:
        raise CaptureError("Failed to prepare target window before screenshot") from exc

    if not focus_window:
        return

    if _try_focus_window(hwnd, win32gui_module=win32gui, win32con_module=win32con):
        return

    flags = win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW
    try:
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, flags)
        win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, flags)
        if _try_focus_window(hwnd, win32gui_module=win32gui, win32con_module=win32con):
            return
    except Exception:
        pass

    try:
        # ALT key hack helps bypass Windows foreground lock in terminal-driven flows.
        win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
        win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
        if _try_focus_window(hwnd, win32gui_module=win32gui, win32con_module=win32con):
            return
    except Exception:
        pass

    raise CaptureError("Failed to activate target window before screenshot")


def _resolve_show_command(
    hwnd: int,
    *,
    maximize_window: bool,
    win32con_module: Any,
    win32gui_module: Any,
) -> int:
    if maximize_window:
        return int(win32con_module.SW_MAXIMIZE)
    try:
        if win32gui_module.IsIconic(hwnd):
            return int(win32con_module.SW_RESTORE)
    except Exception:
        pass
    return int(win32con_module.SW_SHOW)


def _try_focus_window(hwnd: int, *, win32gui_module: Any, win32con_module: Any) -> bool:
    for operation_name in ("BringWindowToTop", "SetForegroundWindow", "SetActiveWindow"):
        operation = getattr(win32gui_module, operation_name, None)
        if operation is None:
            continue
        try:
            operation(hwnd)
        except Exception:
            continue
    return _is_foreground_window(
        hwnd,
        win32gui_module=win32gui_module,
        win32con_module=win32con_module,
    )


def _is_foreground_window(hwnd: int, *, win32gui_module: Any, win32con_module: Any) -> bool:
    try:
        foreground_hwnd = win32gui_module.GetForegroundWindow()
    except Exception:
        return False

    if foreground_hwnd == hwnd:
        return True
    if not foreground_hwnd:
        return False

    try:
        foreground_root = win32gui_module.GetAncestor(foreground_hwnd, win32con_module.GA_ROOT)
        target_root = win32gui_module.GetAncestor(hwnd, win32con_module.GA_ROOT)
        return foreground_root == target_root
    except Exception:
        return False
