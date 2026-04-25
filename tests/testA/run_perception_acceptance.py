"""成员 A 第 10 项真实感知验收脚本。

用途：
这个脚本不是 pytest 单元测试，而是给成员 A 在真实飞书桌面端上验证
“首批 5 个单步场景感知支撑”的人工验收工具。它会逐个提示你把飞书切到
指定场景，按 Enter 后执行截图、OCR、UIA、Candidate Builder、page_summary
生成，并输出 Markdown 报告、结构化 JSON 和标注图。

覆盖场景：
1. 左侧导航
2. 搜索框
3. 聊天列表
4. 文档入口
5. 返回类按钮

每个场景检查：
1. 是否能截图。
2. OCR 是否抽到关键文本。
3. UIA 是否抽到部分元素。
4. Candidate Builder 是否生成匹配候选。
5. 是否生成 page_summary。
6. 匹配候选是否包含可供 C 点击的 center。

运行方式：
```powershell
python tests\testA\run_perception_acceptance.py
```

常用参数：
```powershell
python tests\testA\run_perception_acceptance.py --window-title 飞书 --monitor-index 1
python tests\testA\run_perception_acceptance.py --non-interactive
python tests\testA\run_perception_acceptance.py --fullscreen
python tests\testA\run_perception_acceptance.py --no-focus-window
python tests\testA\run_perception_acceptance.py --no-maximize-window
```

重要约定：
1. 这个脚本会访问真实屏幕、真实 OCR、真实 Windows UIA。
2. 运行前需要打开飞书桌面端。
3. 交互模式下，每个场景开始前按提示切换页面，再按 Enter；脚本会自动
    激活并默认最大化 `--window-title` 对应窗口后再采集。
4. 返回类按钮在很多页面没有文字，本脚本会用 “Button/Image + 左上区域”
   作为弱匹配规则，最终需要看标注图确认候选是否合理。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capture import ScreenshotCaptureConfig, ScreenshotCapturer  # noqa: E402
from perception.candidate_builder import build_ui_state_from_elements  # noqa: E402
from perception.ocr import OCRTextBlock, extract_ocr_elements  # noqa: E402
from perception.uia import UIAConfig, UIAElement, extract_uia_elements  # noqa: E402
from schemas import UIState, save_ui_state_json  # noqa: E402


@dataclass(frozen=True, slots=True)
class Scenario:
    key: str
    title: str
    instruction: str
    keywords: tuple[str, ...]
    role_keywords: tuple[str, ...] = ()
    require_editable: bool = False
    require_clickable: bool = False
    prefer_top_left_button: bool = False
    allow_text_fallback_when_not_editable: bool = False
    allow_locatable_text_fallback: bool = False
    allow_split_ocr_keywords: bool = False


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    scenario: Scenario
    step_idx: int
    screenshot_path: str
    ui_state_path: str
    annotated_path: str
    screenshot_ok: bool
    ocr_ok: bool
    uia_ok: bool
    candidates_ok: bool
    page_summary_ok: bool
    clickable_center_ok: bool
    ocr_hits: list[dict[str, Any]]
    uia_hits: list[dict[str, Any]]
    candidate_hits: list[dict[str, Any]]
    page_summary: dict[str, Any]
    error: str | None = None

    @property
    def modality_ok(self) -> bool:
        return self.ocr_ok or self.uia_ok

    @property
    def passed(self) -> bool:
        return all(
            [
                self.screenshot_ok,
                self.modality_ok,
                self.candidates_ok,
                self.page_summary_ok,
                self.clickable_center_ok,
            ]
        )


SCENARIOS = (
    Scenario(
        key="left_nav",
        title="左侧导航",
        instruction="确认飞书主窗口可见，左侧主导航栏显示消息、云文档、工作台、通讯录等入口。",
        keywords=("消息", "云文档", "工作台", "通讯录", "日历", "应用中心", "知识问答"),
        role_keywords=("Button", "Text", "Image"),
        require_clickable=True,
        allow_locatable_text_fallback=True,
        allow_split_ocr_keywords=True,
    ),
    Scenario(
        key="search_box",
        title="搜索框",
        instruction="确认搜索框可见，通常显示“搜索”或“Ctrl + K”。",
        keywords=("搜索", "Ctrl", "K"),
        role_keywords=("Edit", "Text"),
        require_editable=True,
        allow_text_fallback_when_not_editable=True,
    ),
    Scenario(
        key="chat_list",
        title="聊天列表",
        instruction="确认聊天列表或消息列表可见，例如分组、单聊、群组、未读、标记等。",
        keywords=("分组", "单聊", "群组", "未读", "标记", "@我", "消息"),
        role_keywords=("Text", "Document", "Button"),
        require_clickable=True,
    ),
    Scenario(
        key="docs_entry",
        title="文档入口",
        instruction="确认云文档入口或云文档页面可见。",
        keywords=("云文档", "文档", "多维表格", "知识库"),
        role_keywords=("Text", "Button", "Document"),
        require_clickable=True,
        allow_locatable_text_fallback=True,
        allow_split_ocr_keywords=True,
    ),
    Scenario(
        key="back_button",
        title="返回类按钮",
        instruction="请进入一个有返回、关闭、上一页或面包屑返回按钮的页面。若按钮无文字，确保它在窗口左上区域可见。",
        keywords=("返回", "关闭", "上一页", "Back", "back"),
        role_keywords=("Button", "Image"),
        require_clickable=True,
        prefer_top_left_button=True,
    ),
)


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"run_perception_acceptance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifact_root = Path(args.artifact_root)
    run_root = artifact_root / run_id
    annotated_dir = run_root / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    maximize_window = not args.fullscreen and not args.no_maximize_window

    print("\n=== 成员 A 第 10 项真实感知验收 ===")
    print(f"run_id: {run_id}")
    print(f"artifact_root: {artifact_root}")
    print(f"window_title: {args.window_title}")
    print(f"monitor_index: {args.monitor_index}")
    print(f"capture_target: {'fullscreen' if args.fullscreen else 'window'}")
    print(f"focus_window: {not args.no_focus_window}")
    print(f"maximize_window: {maximize_window}")
    print("建议：优先使用窗口截图模式（不要传 --fullscreen），避免把 VSCode 终端一起截进去。")
    print("提示：这个脚本会真实截图、真实 OCR、真实读取 UIA。\n")

    capturer = ScreenshotCapturer(
        config=ScreenshotCaptureConfig(
            artifact_root=artifact_root,
            monitor_index=args.monitor_index,
            window_title=None if args.fullscreen else args.window_title,
            focus_window=not args.no_focus_window,
            maximize_window=maximize_window,
            focus_delay_seconds=args.focus_delay_seconds,
        )
    )
    results: list[ScenarioResult] = []

    for step_idx, scenario in enumerate(SCENARIOS, start=1):
        print(f"\n--- 场景 {step_idx}: {scenario.title} ---")
        print(scenario.instruction)
        if not args.non_interactive:
            input("准备好飞书界面后按 Enter 采集；脚本会自动切回飞书后截图。Ctrl+C 可中止。")

        result = run_scenario(
            scenario=scenario,
            step_idx=step_idx,
            run_id=run_id,
            capturer=capturer,
            artifact_root=artifact_root,
            annotated_dir=annotated_dir,
            window_title=args.window_title,
        )
        results.append(result)
        print_scenario_result(result)

    summary_path = run_root / "acceptance_summary.json"
    report_path = run_root / "acceptance_report.md"
    summary_path.write_text(
        json.dumps(summary_payload(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(render_markdown_report(results), encoding="utf-8")

    passed_count = sum(result.passed for result in results)
    print("\n=== 验收输出 ===")
    print(f"Markdown 报告: {report_path}")
    print(f"结构化汇总: {summary_path}")
    print(f"标注图目录: {annotated_dir}")
    print(f"通过场景: {passed_count}/{len(results)}")

    if passed_count < len(results):
        print("\n存在未通过场景。打开 Markdown 报告和标注图，确认是页面没切到位、OCR/UIA 没召回，还是候选规则需要调整。")
        return 1

    print("\n=== ALL ACCEPTANCE SCENARIOS PASSED ===")
    return 0


def run_scenario(
    *,
    scenario: Scenario,
    step_idx: int,
    run_id: str,
    capturer: ScreenshotCapturer,
    artifact_root: Path,
    annotated_dir: Path,
    window_title: str,
) -> ScenarioResult:
    try:
        screenshot = capturer.capture_before(run_id=run_id, step_idx=step_idx)
        ocr_elements = extract_ocr_elements(screenshot.screenshot_path)
        uia_elements = extract_uia_elements(config=UIAConfig(window_title=window_title))
        ui_state = build_ui_state_from_elements(
            screenshot_path=screenshot.screenshot_path,
            ocr_elements=ocr_elements,
            uia_elements=uia_elements,
            screen_origin=screenshot.screen_origin,
        )
        ui_state_path = save_ui_state_json(
            ui_state,
            run_id=run_id,
            step_idx=step_idx,
            phase="before",
            artifact_root=artifact_root,
        )

        ocr_hits = match_ocr_elements(scenario, ocr_elements)
        uia_hits = match_uia_elements(scenario, uia_elements)
        candidate_hits = match_candidates(scenario, ui_state, screenshot.screen_origin, screenshot.width, screenshot.height)
        annotated_path = annotated_dir / f"step_{step_idx:03d}_{scenario.key}.png"
        draw_annotations(
            screenshot_path=Path(screenshot.screenshot_path),
            annotated_path=annotated_path,
            screen_origin=screenshot.screen_origin,
            candidates=candidate_hits,
        )

        return ScenarioResult(
            scenario=scenario,
            step_idx=step_idx,
            screenshot_path=screenshot.screenshot_path,
            ui_state_path=str(ui_state_path),
            annotated_path=str(annotated_path),
            screenshot_ok=Path(screenshot.screenshot_path).exists(),
            ocr_ok=bool(ocr_hits),
            uia_ok=bool(uia_hits),
            candidates_ok=bool(candidate_hits),
            page_summary_ok=ui_state.page_summary.candidate_count == len(ui_state.candidates),
            clickable_center_ok=any(has_clickable_center(candidate) for candidate in candidate_hits),
            ocr_hits=ocr_hits[:20],
            uia_hits=uia_hits[:20],
            candidate_hits=candidate_hits[:20],
            page_summary=ui_state.page_summary.to_dict(),
        )
    except Exception as exc:
        return ScenarioResult(
            scenario=scenario,
            step_idx=step_idx,
            screenshot_path="",
            ui_state_path="",
            annotated_path="",
            screenshot_ok=False,
            ocr_ok=False,
            uia_ok=False,
            candidates_ok=False,
            page_summary_ok=False,
            clickable_center_ok=False,
            ocr_hits=[],
            uia_hits=[],
            candidate_hits=[],
            page_summary={},
            error=repr(exc),
        )


def match_ocr_elements(scenario: Scenario, ocr_elements: list[OCRTextBlock]) -> list[dict[str, Any]]:
    return [
        element.to_dict()
        for element in ocr_elements
        if text_matches(element.text, scenario.keywords)
        or (
            scenario.allow_split_ocr_keywords
            and keyword_fragment_matches(element.text, scenario.keywords)
        )
    ]


def match_uia_elements(scenario: Scenario, uia_elements: list[UIAElement]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for element in uia_elements:
        if text_matches(element.name, scenario.keywords) or role_matches(element.role, scenario.role_keywords):
            hits.append(element.to_dict())
    return hits


def match_candidates(
    scenario: Scenario,
    ui_state: UIState,
    screen_origin: list[int],
    screenshot_width: int,
    screenshot_height: int,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for candidate in ui_state.to_dict()["candidates"]:
        text_hit = candidate_text_matches(scenario, candidate)
        role_hit = role_matches(candidate["role"], scenario.role_keywords)
        top_left_hit = scenario.prefer_top_left_button and is_top_left_button(
            candidate,
            screen_origin=screen_origin,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
        )
        locatable_text_hit = (
            scenario.allow_locatable_text_fallback
            and text_hit
            and str(candidate.get("source", "")) in {"ocr", "merged"}
            and has_clickable_center(candidate)
        )

        if not (text_hit or role_hit or top_left_hit):
            continue
        if scenario.require_editable and not candidate["editable"]:
            if not (scenario.allow_text_fallback_when_not_editable and text_hit):
                continue
        if scenario.require_clickable and not (
            candidate["clickable"]
            or candidate["editable"]
            or top_left_hit
            or locatable_text_hit
        ):
            continue
        hits.append(candidate)
    return hits


def candidate_text_matches(scenario: Scenario, candidate: dict[str, Any]) -> bool:
    text = str(candidate.get("text", ""))
    if text_matches(text, scenario.keywords):
        return True
    return (
        scenario.allow_split_ocr_keywords
        and str(candidate.get("source", "")) in {"ocr", "merged"}
        and keyword_fragment_matches(text, scenario.keywords)
    )


def text_matches(text: str, keywords: tuple[str, ...]) -> bool:
    normalized = text.strip()
    if not normalized:
        return False

    lowered = normalized.lower()
    token = _normalize_token(normalized)
    for keyword in keywords:
        keyword_text = keyword.strip()
        if not keyword_text:
            continue
        if _is_single_ascii_letter(keyword_text):
            if token == keyword_text.lower():
                return True
            continue
        if keyword_text.lower() in lowered:
            return True
    return False


def keyword_fragment_matches(text: str, keywords: tuple[str, ...]) -> bool:
    token = _normalize_token(text)
    if not token:
        return False

    for keyword in keywords:
        keyword_token = _normalize_token(keyword)
        if len(keyword_token) < 2:
            continue
        if token == keyword_token:
            return True
        if _is_cjk_token(token) and token in keyword_token:
            return True
    return False


def role_matches(role: str, role_keywords: tuple[str, ...]) -> bool:
    return any(keyword in role for keyword in role_keywords)


def has_clickable_center(candidate: dict[str, Any]) -> bool:
    center = candidate.get("center")
    bbox = candidate.get("bbox")
    return (
        isinstance(center, list)
        and len(center) == 2
        and all(isinstance(value, int) for value in center)
        and isinstance(bbox, list)
        and len(bbox) == 4
    )


def is_top_left_button(
    candidate: dict[str, Any],
    *,
    screen_origin: list[int],
    screenshot_width: int,
    screenshot_height: int,
) -> bool:
    role = str(candidate.get("role", ""))
    if "Button" not in role and "Image" not in role:
        return False
    center = candidate.get("center")
    if not isinstance(center, list) or len(center) != 2:
        return False
    local_x = center[0] - screen_origin[0]
    local_y = center[1] - screen_origin[1]
    return 0 <= local_x <= screenshot_width * 0.35 and 0 <= local_y <= screenshot_height * 0.25


def draw_annotations(
    *,
    screenshot_path: Path,
    annotated_path: Path,
    screen_origin: list[int],
    candidates: list[dict[str, Any]],
) -> None:
    with Image.open(screenshot_path) as image:
        annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    for index, candidate in enumerate(candidates[:30], start=1):
        bbox = candidate["bbox"]
        local_bbox = [
            bbox[0] - screen_origin[0],
            bbox[1] - screen_origin[1],
            bbox[2] - screen_origin[0],
            bbox[3] - screen_origin[1],
        ]
        if local_bbox[2] < 0 or local_bbox[3] < 0 or local_bbox[0] > width or local_bbox[1] > height:
            continue
        local_bbox = [
            max(0, min(width, local_bbox[0])),
            max(0, min(height, local_bbox[1])),
            max(0, min(width, local_bbox[2])),
            max(0, min(height, local_bbox[3])),
        ]
        color = "red" if index == 1 else "yellow"
        draw.rectangle(local_bbox, outline=color, width=3)
        label = f"{index}:{candidate['id']} {candidate['role']} {candidate['confidence']:.2f}"
        draw.text((local_bbox[0] + 2, max(0, local_bbox[1] - 14)), label, fill=color)

    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(annotated_path)


def print_scenario_result(result: ScenarioResult) -> None:
    if result.error:
        print(f"[FAIL] {result.scenario.title}: {result.error}")
        return
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.scenario.title}")
    print(f"  screenshot: {mark(result.screenshot_ok)} {result.screenshot_path}")
    print(f"  OCR hit: {mark(result.ocr_ok)} count={len(result.ocr_hits)}")
    print(f"  UIA hit: {mark(result.uia_ok)} count={len(result.uia_hits)}")
    print(f"  modality(ocr/uia): {mark(result.modality_ok)}")
    print(f"  candidate hit: {mark(result.candidates_ok)} count={len(result.candidate_hits)}")
    print(f"  page_summary: {mark(result.page_summary_ok)} {json.dumps(result.page_summary, ensure_ascii=False)}")
    print(f"  clickable center: {mark(result.clickable_center_ok)}")
    print(f"  annotated: {result.annotated_path}")


def mark(value: bool) -> str:
    return "[PASS]" if value else "[FAIL]"


def summary_payload(results: list[ScenarioResult]) -> dict[str, Any]:
    return {
        "passed_count": sum(result.passed for result in results),
        "total_count": len(results),
        "results": [
            {
                "scenario": result.scenario.title,
                "passed": result.passed,
                "screenshot_path": result.screenshot_path,
                "ui_state_path": result.ui_state_path,
                "annotated_path": result.annotated_path,
                "checks": {
                    "screenshot": result.screenshot_ok,
                    "ocr": result.ocr_ok,
                    "uia": result.uia_ok,
                    "modality": result.modality_ok,
                    "candidates": result.candidates_ok,
                    "page_summary": result.page_summary_ok,
                    "clickable_center": result.clickable_center_ok,
                },
                "ocr_hits": result.ocr_hits,
                "uia_hits": result.uia_hits,
                "candidate_hits": result.candidate_hits,
                "page_summary": result.page_summary,
                "error": result.error,
            }
            for result in results
        ],
    }


def render_markdown_report(results: list[ScenarioResult]) -> str:
    lines = [
        "# 成员 A 第 10 项真实感知验收报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "| 场景 | 截图 | OCR | UIA | Candidates | Page Summary | Center | 结论 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.scenario.title,
                    pass_fail(result.screenshot_ok),
                    pass_fail(result.ocr_ok),
                    pass_fail(result.uia_ok),
                    pass_fail(result.candidates_ok),
                    pass_fail(result.page_summary_ok),
                    pass_fail(result.clickable_center_ok),
                    pass_fail(result.passed),
                ]
            )
            + " |"
        )

    for result in results:
        lines.extend(
            [
                "",
                f"## {result.scenario.title}",
                "",
                f"- 说明：{result.scenario.instruction}",
                f"- 截图：`{result.screenshot_path}`",
                f"- UI State：`{result.ui_state_path}`",
                f"- 标注图：`{result.annotated_path}`",
                f"- Page Summary：`{json.dumps(result.page_summary, ensure_ascii=False)}`",
                f"- 错误：`{result.error}`" if result.error else "- 错误：无",
                "",
                "### 命中 Candidates",
                "",
            ]
        )
        if result.candidate_hits:
            lines.append("| id | text | role | clickable | editable | center | confidence | source |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for candidate in result.candidate_hits[:10]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(candidate.get("id", "")),
                            str(candidate.get("text", "")),
                            str(candidate.get("role", "")),
                            str(candidate.get("clickable", "")),
                            str(candidate.get("editable", "")),
                            str(candidate.get("center", "")),
                            f"{float(candidate.get('confidence', 0.0)):.2f}",
                            str(candidate.get("source", "")),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("无匹配候选。")
    lines.append("")
    return "\n".join(lines)


def pass_fail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _normalize_token(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum())


def _is_single_ascii_letter(text: str) -> bool:
    return len(text) == 1 and text.isascii() and text.isalpha()


def _is_cjk_token(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="成员 A 第 10 项真实感知验收脚本")
    parser.add_argument("--run-id", default=None, help="本次验收 run_id，默认自动生成")
    parser.add_argument("--artifact-root", default="artifacts/runs", help="artifact 根目录")
    parser.add_argument("--window-title", default="飞书", help="UIA 查找窗口标题")
    parser.add_argument("--monitor-index", type=int, default=1, help="mss monitor index，默认主屏 1")
    parser.add_argument("--fullscreen", action="store_true", help="回退为全屏截图，不裁剪窗口")
    parser.add_argument("--no-focus-window", action="store_true", help="窗口截图前不自动激活目标窗口")
    parser.add_argument("--no-maximize-window", action="store_true", help="窗口截图前不自动最大化目标窗口")
    parser.add_argument("--focus-delay-seconds", type=float, default=0.8, help="激活窗口后等待多久再截图")
    parser.add_argument("--non-interactive", action="store_true", help="不逐场景等待 Enter，直接连续采集")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
