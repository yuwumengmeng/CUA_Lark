"""成员 A 4.25 `get_ui_state()` 总接口。

职责边界：
本文件负责把截图采集、OCR、UIA、Candidate Builder 和 UI state JSON
落盘串成一个稳定入口。它不负责 planner 决策、不负责 executor 点击，
也不负责 OCR/UIA 算法本身的识别质量优化。

公开接口：
`get_ui_state(...)`
    成员 A 对成员 B / C 的总入口。返回普通 dict，稳定字段为
    `screenshot_path / candidates / page_summary`。
`collect_ui_state(...)`
    与 `get_ui_state()` 相同流程，但返回强类型 `UIState`，便于内部测试
    或后续扩展。

执行流程：
```text
capture screenshot
    -> extract_ocr_elements
    -> extract_uia_elements
    -> build_ui_state_from_elements
    -> save_ui_state_json
    -> return ui_state
```

坐标规则：
截图模块返回的 `screen_origin` 会传给 Candidate Builder。OCR bbox 会从
截图局部坐标转换成屏幕绝对坐标；UIA bbox 保持屏幕绝对坐标。最终输出
给 B/C 的 candidate `bbox/center` 都可以直接给执行器点击使用。
默认会按 `capture_window_title="飞书"` 进行窗口截图，并尝试最大化和激活
窗口后再采集，避免按 Enter 后被 VSCode/终端遮挡。`screen_origin` 对应
飞书窗口左上角。

使用示例：
```python
from perception.ui_state import get_ui_state

ui_state = get_ui_state(run_id="run_demo", step_idx=1, phase="before")
print(ui_state["screenshot_path"])
print(ui_state["page_summary"])
```

协作规则：
1. 对外只承诺 `screenshot_path / candidates / page_summary` 三个顶层字段。
2. candidates 内部字段遵守 `schemas.UICandidate`。
3. 真实运行产生的 `ui_state_step_xxx_<phase>.json` 属于 artifact，不提交 Git。
4. 单元测试必须注入 fake screenshot/OCR/UIA backend，不访问真实桌面。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from capture import ScreenshotCaptureConfig, ScreenshotCapturer
from schemas import UIState, save_ui_state_json

from .candidate_builder import CandidateBuilderConfig, build_ui_state_from_elements
from .ocr import OCRBackend, OCRConfig, extract_ocr_elements
from .uia import UIABackend, UIAConfig, extract_uia_elements


class UIStateCollectionError(ValueError):
    pass


def get_ui_state(
    *,
    run_id: str,
    step_idx: int,
    phase: str = "before",
    artifact_root: str | Path | None = None,
    monitor_index: int = 1,
    capture_window_title: str | None = "飞书",
    maximize_capture_window: bool = True,
    focus_capture_window: bool = True,
    capture_focus_delay_seconds: float = 0.8,
    window: Any = None,
    window_title: str | None = "飞书",
    save_json: bool = True,
    screenshot_capturer: ScreenshotCapturer | None = None,
    ocr_config: OCRConfig | None = None,
    ocr_backend: OCRBackend | None = None,
    uia_config: UIAConfig | None = None,
    uia_backend: UIABackend | None = None,
    candidate_config: CandidateBuilderConfig | None = None,
) -> dict[str, Any]:
    ui_state = collect_ui_state(
        run_id=run_id,
        step_idx=step_idx,
        phase=phase,
        artifact_root=artifact_root,
        monitor_index=monitor_index,
        capture_window_title=capture_window_title,
        maximize_capture_window=maximize_capture_window,
        focus_capture_window=focus_capture_window,
        capture_focus_delay_seconds=capture_focus_delay_seconds,
        window=window,
        window_title=window_title,
        save_json=save_json,
        screenshot_capturer=screenshot_capturer,
        ocr_config=ocr_config,
        ocr_backend=ocr_backend,
        uia_config=uia_config,
        uia_backend=uia_backend,
        candidate_config=candidate_config,
    )
    return ui_state.to_dict()


def collect_ui_state(
    *,
    run_id: str,
    step_idx: int,
    phase: str = "before",
    artifact_root: str | Path | None = None,
    monitor_index: int = 1,
    capture_window_title: str | None = "飞书",
    maximize_capture_window: bool = True,
    focus_capture_window: bool = True,
    capture_focus_delay_seconds: float = 0.8,
    window: Any = None,
    window_title: str | None = "飞书",
    save_json: bool = True,
    screenshot_capturer: ScreenshotCapturer | None = None,
    ocr_config: OCRConfig | None = None,
    ocr_backend: OCRBackend | None = None,
    uia_config: UIAConfig | None = None,
    uia_backend: UIABackend | None = None,
    candidate_config: CandidateBuilderConfig | None = None,
) -> UIState:
    capturer = screenshot_capturer or ScreenshotCapturer(
        config=ScreenshotCaptureConfig(
            artifact_root=_artifact_root(artifact_root),
            monitor_index=monitor_index,
            window_title=capture_window_title,
            focus_window=focus_capture_window,
            maximize_window=maximize_capture_window,
            focus_delay_seconds=capture_focus_delay_seconds,
        )
    )
    screenshot = capturer.capture_screenshot(
        run_id=run_id,
        step_idx=step_idx,
        phase=phase,  # type: ignore[arg-type]
    )

    ocr_elements = extract_ocr_elements(
        screenshot.screenshot_path,
        config=ocr_config,
        backend=ocr_backend,
    )
    uia_elements = extract_uia_elements(
        window,
        config=uia_config or _default_uia_config(window_title),
        backend=uia_backend,
    )
    ui_state = build_ui_state_from_elements(
        screenshot_path=screenshot.screenshot_path,
        ocr_elements=ocr_elements,
        uia_elements=uia_elements,
        screen_origin=screenshot.screen_origin,
        config=candidate_config,
    )

    if save_json:
        save_ui_state_json(
            ui_state,
            run_id=screenshot.run_id,
            step_idx=screenshot.step_idx,
            phase=screenshot.phase,
            artifact_root=_artifact_root(artifact_root, capturer),
        )

    return ui_state


def _default_uia_config(window_title: str | None) -> UIAConfig:
    if window_title is None:
        return UIAConfig()
    return UIAConfig(window_title=window_title)


def _artifact_root(
    artifact_root: str | Path | None,
    capturer: ScreenshotCapturer | None = None,
) -> Path:
    if artifact_root is not None:
        return Path(artifact_root)
    if capturer is not None:
        return Path(capturer.config.artifact_root)
    return Path("artifacts") / "runs"
