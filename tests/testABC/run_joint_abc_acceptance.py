"""ABC joint real-GUI acceptance runner for the first-cycle 4 single-step cases.

This is an operator script, not a pytest test. It guides a human through the
four first-cycle acceptance cases, then runs the real A -> B -> C chain:

1. A captures the current Lark/Feishu UI and writes ui_state JSON.
2. B consumes that ui_state and emits one action_decision.
3. C executes the action through ExecutorWithRecorder and writes run artifacts.

Safety:
By default the script only previews planner decisions. Pass --real-gui to
execute mouse/keyboard actions. In interactive real-GUI mode every action is
shown before execution; for click actions you can press Enter to use B's point,
type "m" to use the current mouse position, "s" to skip/fail the step, or "q"
to abort the run.

Example:
    python tests\\testABC\\run_joint_abc_acceptance.py --real-gui --window-title 飞书
    python tests\\testABC\\run_joint_abc_acceptance.py --real-gui --yes --window-title 飞书
    python tests\\testABC\\run_joint_abc_acceptance.py --real-gui --yes --window-title 飞书 --chat-name 测试群 --search-text 项目周报


Useful options:
    --chat-name 测试群
    --search-text 项目周报
    --docs-entry-text 文档
    --sidebar-text 云文档
    --yes
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from executor import ExecutorWithRecorder  # noqa: E402
from planner import MiniPlanner  # noqa: E402
from planner.action_protocol import DECISION_FAILED, DECISION_SUCCESS  # noqa: E402
from planner.runtime_state import RuntimeStatus, create_runtime_state  # noqa: E402
from perception.ui_state import get_ui_state  # noqa: E402
from schemas import ACTION_CLICK, ACTION_TYPE, utc_now  # noqa: E402


DEFAULT_SIDEBAR_TEXT = "云文档"
DEFAULT_SEARCH_TEXT = "项目周报"
DEFAULT_CHAT_NAME = "测试群"
DEFAULT_DOCS_ENTRY_TEXT = "文档"


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    key: str
    title: str
    operator_instruction: str
    task: dict[str, Any]
    max_planner_steps: int = 2


@dataclass(slots=True)
class StepRecord:
    scenario_key: str
    scenario_title: str
    step_idx: int
    planner_step_idx: int
    before_ui_state_path: str | None = None
    after_ui_state_path: str | None = None
    annotation_path: str | None = None
    selected_candidate: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    planned_action: dict[str, Any] | None = None
    executed_action: dict[str, Any] | None = None
    action_result: dict[str, Any] | None = None
    runner_result: dict[str, Any] | None = None
    status: str = "running"
    failure_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_key": self.scenario_key,
            "scenario_title": self.scenario_title,
            "step_idx": self.step_idx,
            "planner_step_idx": self.planner_step_idx,
            "before_ui_state_path": self.before_ui_state_path,
            "after_ui_state_path": self.after_ui_state_path,
            "annotation_path": self.annotation_path,
            "selected_candidate": self.selected_candidate,
            "decision": self.decision,
            "planned_action": self.planned_action,
            "executed_action": self.executed_action,
            "action_result": self.action_result,
            "runner_result": self.runner_result,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "notes": self.notes,
        }


@dataclass(slots=True)
class ScenarioRunResult:
    spec: ScenarioSpec
    status: str
    planner_state: dict[str, Any]
    steps: list[StepRecord]
    failure_reason: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == RuntimeStatus.SUCCESS.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.spec.key,
            "title": self.spec.title,
            "status": self.status,
            "passed": self.passed,
            "failure_reason": self.failure_reason,
            "planner_state": self.planner_state,
            "steps": [step.to_dict() for step in self.steps],
        }


class OperatorAbort(RuntimeError):
    pass


class OperatorSkip(RuntimeError):
    pass


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"run_joint_abc_acceptance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifact_root = Path(args.artifact_root)
    run_root = artifact_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    specs = build_scenarios(args)
    print_header(args, run_id, artifact_root, specs)

    if args.real_gui and not args.yes:
        input("确认飞书测试环境已准备好，即将进入 ABC 真实联合验收；按 Enter 继续，Ctrl+C 取消。")
    elif args.real_gui and args.yes:
        print("自动动作模式：每个场景开始前按一次 Enter；场景内直接使用 B 规划出的动作和坐标。")
    else:
        print("未传入 --real-gui：只采集 A 的真实 ui_state 并预览 B 的动作，不执行 C 的桌面动作。")

    runner = ExecutorWithRecorder(
        run_id=run_id,
        task={
            "name": "abc_joint_acceptance",
            "scenario_count": len(specs),
            "real_gui": args.real_gui,
            "scenarios": [spec.task for spec in specs],
        },
        artifact_root=artifact_root,
        window_title=None if args.fullscreen else args.window_title,
        monitor_index=args.monitor_index,
        focus_window=not args.no_focus_window,
        focus_delay_seconds=args.focus_delay_seconds,
    )
    planner = MiniPlanner()

    scenario_results: list[ScenarioRunResult] = []
    step_counter = 1
    try:
        for spec in specs:
            result, step_counter = run_scenario(
                spec=spec,
                planner=planner,
                runner=runner,
                args=args,
                run_id=run_id,
                artifact_root=artifact_root,
                first_step_idx=step_counter,
            )
            scenario_results.append(result)
            print_scenario_result(result)
    except KeyboardInterrupt:
        print("\n用户中止运行。")
        summary = write_reports(
            run_id=run_id,
            artifact_root=artifact_root,
            scenario_results=scenario_results,
            recorder_summary=runner.finalize(status="failed", failure_reason="keyboard_interrupt"),
            status="failed",
            failure_reason="keyboard_interrupt",
            args=args,
        )
        print_report_paths(summary)
        return 130
    except OperatorAbort:
        print("\n用户中止运行。")
        summary = write_reports(
            run_id=run_id,
            artifact_root=artifact_root,
            scenario_results=scenario_results,
            recorder_summary=runner.finalize(status="failed", failure_reason="operator_abort"),
            status="failed",
            failure_reason="operator_abort",
            args=args,
        )
        print_report_paths(summary)
        return 130

    all_ok = all(result.passed for result in scenario_results)
    recorder_summary = runner.finalize(status="success" if all_ok else "failed")
    summary = write_reports(
        run_id=run_id,
        artifact_root=artifact_root,
        scenario_results=scenario_results,
        recorder_summary=recorder_summary,
        status="success" if all_ok else "failed",
        failure_reason=None if all_ok else "scenario_failed",
        args=args,
    )
    print_report_paths(summary)
    return 0 if all_ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABC 四场景联合真实 GUI 验收脚本")
    parser.add_argument("--real-gui", action="store_true", help="确认执行真实鼠标键盘动作")
    parser.add_argument("--yes", action="store_true", help="跳过动作级确认；仍保留每个场景开始前一次 Enter")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/runs")
    parser.add_argument("--window-title", default="飞书")
    parser.add_argument("--fullscreen", action="store_true", help="A/C 都使用全屏截图，而不是窗口截图")
    parser.add_argument("--monitor-index", type=int, default=1)
    parser.add_argument("--no-focus-window", action="store_true")
    parser.add_argument("--no-maximize-window", action="store_true")
    parser.add_argument("--focus-delay-seconds", type=float, default=0.35)
    parser.add_argument("--after-action-delay-seconds", type=float, default=0.25)
    parser.add_argument(
        "--strict-after-checks",
        action="store_true",
        help="每步执行后再跑一次 A 的 OCR/UIA 感知做业务校验；更严格但明显更慢",
    )
    parser.add_argument("--sidebar-text", default=DEFAULT_SIDEBAR_TEXT)
    parser.add_argument("--search-text", default=DEFAULT_SEARCH_TEXT)
    parser.add_argument("--chat-name", default=DEFAULT_CHAT_NAME)
    parser.add_argument("--docs-entry-text", default=DEFAULT_DOCS_ENTRY_TEXT)
    parser.add_argument("--annotation-top-n", type=int, default=30)
    parser.add_argument("--skip", action="append", default=[], help="跳过场景 key，可重复传入")
    return parser.parse_args()


def build_scenarios(args: argparse.Namespace) -> list[ScenarioSpec]:
    all_specs = [
        ScenarioSpec(
            key="sidebar_docs",
            title="点击侧边栏模块的云文档按钮",
            operator_instruction=(
                f"请让飞书左侧导航可见，并确认“{args.sidebar_text}”入口在屏幕上。"
            ),
            task={
                "goal": "open_sidebar_module",
                "params": {"target_text": args.sidebar_text},
                "steps": [
                    {
                        "action_hint": "find_sidebar_module",
                        "target_text": args.sidebar_text,
                    }
                ],
            },
            max_planner_steps=1,
        ),
        ScenarioSpec(
            key="search_input",
            title="搜索框输入关键词",
            operator_instruction=(
                f"请让飞书搜索框可见。脚本会先点击搜索框，再输入“{args.search_text}”。"
            ),
            task={
                "goal": "input_search",
                "params": {"text": args.search_text},
                "steps": [
                    {
                        "action_hint": "find_search_box",
                        "target_text": "搜索",
                        "target_role": "textbox",
                    },
                    {"action_hint": "type_text", "input_text": args.search_text},
                ],
            },
            max_planner_steps=2,
        ),
        ScenarioSpec(
            key="click_group_chat",
            title="点击群聊",
            operator_instruction=(
                f"请让聊天列表可见，并确认群聊“{args.chat_name}”在列表中可见。"
            ),
            task={
                "goal": "open_chat",
                "params": {"target_name": args.chat_name},
                "steps": [
                    {"action_hint": "find_chat", "target_text": args.chat_name},
                    {"action_hint": "click_target", "target_text": args.chat_name},
                ],
            },
            max_planner_steps=1,
        ),
        ScenarioSpec(
            key="open_docs_entry",
            title="打开文档入口",
            operator_instruction=(
                f"请让文档入口或云文档页面可见，并确认“{args.docs_entry_text}”相关入口可见。"
            ),
            task={
                "goal": "open_docs_entry",
                "params": {"target_text": args.docs_entry_text},
                "steps": [
                    {"action_hint": "click_target", "target_text": args.docs_entry_text}
                ],
            },
            max_planner_steps=1,
        ),
    ]
    skip = {str(key).strip() for key in args.skip if str(key).strip()}
    return [spec for spec in all_specs if spec.key not in skip]


def run_scenario(
    *,
    spec: ScenarioSpec,
    planner: MiniPlanner,
    runner: ExecutorWithRecorder,
    args: argparse.Namespace,
    run_id: str,
    artifact_root: Path,
    first_step_idx: int,
) -> tuple[ScenarioRunResult, int]:
    print(f"\n=== 场景：{spec.title} ===")
    print(spec.operator_instruction)
    input("准备好飞书界面后按 Enter 开始本场景；Ctrl+C 可中止。")

    state = create_runtime_state(spec.task)
    records: list[StepRecord] = []
    step_idx = first_step_idx

    for _ in range(spec.max_planner_steps):
        if state.status == RuntimeStatus.SUCCESS.value:
            break
        record = StepRecord(
            scenario_key=spec.key,
            scenario_title=spec.title,
            step_idx=step_idx,
            planner_step_idx=state.step_idx,
        )
        records.append(record)

        latest_before: dict[str, Any] = {}
        latest_executed_action: dict[str, Any] | None = None
        latest_runner_result: dict[str, Any] | None = None

        def get_before() -> Mapping[str, Any]:
            ui_state = collect_ui_state_for_step(
                args=args,
                run_id=run_id,
                step_idx=step_idx,
                phase="before",
                artifact_root=artifact_root,
            )
            latest_before.clear()
            latest_before.update(ui_state)
            record.before_ui_state_path = ui_state_path(
                artifact_root=artifact_root,
                run_id=run_id,
                step_idx=step_idx,
                phase="before",
            )
            return ui_state

        def execute(action: Mapping[str, Any]) -> Mapping[str, Any]:
            nonlocal latest_executed_action, latest_runner_result
            planned_action = dict(action)
            record.planned_action = planned_action
            executed_action = prepare_action_for_execution(
                action=planned_action,
                ui_state=latest_before,
                args=args,
                scenario=spec,
                step_idx=step_idx,
            )
            latest_executed_action = executed_action
            record.executed_action = executed_action
            record.selected_candidate = selected_candidate_summary(
                latest_before,
                executed_action.get("target_element_id"),
            )

            if not args.real_gui:
                write_step_annotation(
                    record=record,
                    ui_state=latest_before,
                    action=executed_action,
                    artifact_root=artifact_root,
                    run_id=run_id,
                    top_n=args.annotation_top_n,
                )
                return fake_success_result(executed_action)

            runner_result = runner.run_step(executed_action, step_idx=step_idx).to_dict()
            latest_runner_result = runner_result
            record.runner_result = runner_result
            write_step_annotation(
                record=record,
                ui_state=latest_before,
                action=executed_action,
                artifact_root=artifact_root,
                run_id=run_id,
                top_n=args.annotation_top_n,
                screenshot_metadata=runner_result.get("before_screenshot"),
            )
            return dict(runner_result["action_result"])

        def get_after() -> Mapping[str, Any]:
            if args.after_action_delay_seconds > 0:
                sleep(args.after_action_delay_seconds)
            ui_state = collect_ui_state_for_step(
                args=args,
                run_id=run_id,
                step_idx=step_idx,
                phase="after",
                artifact_root=artifact_root,
            )
            record.after_ui_state_path = ui_state_path(
                artifact_root=artifact_root,
                run_id=run_id,
                step_idx=step_idx,
                phase="after",
            )
            return ui_state

        try:
            run_result = planner.run_once(
                spec.task,
                get_before,
                execute,
                state,
                get_after_ui_state=get_after if args.real_gui and args.strict_after_checks else None,
            )
        except OperatorSkip as exc:
            record.status = RuntimeStatus.FAILED.value
            record.failure_reason = "operator_skip"
            record.notes.append(str(exc))
            state.status = RuntimeStatus.FAILED.value
            state.failure_reason = "operator_skip"
            step_idx += 1
            break

        state = run_result.runtime_state
        record.decision = dict(run_result.decision)
        record.action_result = dict(run_result.action_result or {})
        if latest_executed_action is not None:
            record.executed_action = latest_executed_action
        if latest_runner_result is not None:
            record.runner_result = latest_runner_result

        decision_status = run_result.decision.get("status")
        if decision_status == DECISION_FAILED:
            record.status = RuntimeStatus.FAILED.value
            record.failure_reason = str(run_result.decision.get("failure_reason") or "decision_failed")
            step_idx += 1
            break
        if decision_status == DECISION_SUCCESS:
            record.status = RuntimeStatus.SUCCESS.value
            step_idx += 1
            break

        record.status = state.status
        record.failure_reason = state.failure_reason
        step_idx += 1

    scenario_status = state.status
    failure_reason = state.failure_reason
    if scenario_status == RuntimeStatus.RUNNING.value:
        scenario_status = RuntimeStatus.FAILED.value
        failure_reason = "max_planner_steps_exhausted"

    return (
        ScenarioRunResult(
            spec=spec,
            status=scenario_status,
            failure_reason=failure_reason,
            planner_state=state.to_dict(),
            steps=records,
        ),
        step_idx,
    )


def collect_ui_state_for_step(
    *,
    args: argparse.Namespace,
    run_id: str,
    step_idx: int,
    phase: str,
    artifact_root: Path,
) -> dict[str, Any]:
    return get_ui_state(
        run_id=run_id,
        step_idx=step_idx,
        phase=phase,
        artifact_root=artifact_root,
        monitor_index=args.monitor_index,
        capture_window_title=None if args.fullscreen else args.window_title,
        maximize_capture_window=not args.fullscreen and not args.no_maximize_window,
        focus_capture_window=not args.no_focus_window,
        capture_focus_delay_seconds=args.focus_delay_seconds,
        window_title=args.window_title,
    )


def prepare_action_for_execution(
    *,
    action: Mapping[str, Any],
    ui_state: Mapping[str, Any],
    args: argparse.Namespace,
    scenario: ScenarioSpec,
    step_idx: int,
) -> dict[str, Any]:
    payload = dict(action)
    print_action_preview(
        action=payload,
        ui_state=ui_state,
        scenario=scenario,
        step_idx=step_idx,
        real_gui=args.real_gui,
    )

    if not args.real_gui or args.yes:
        return payload

    if payload.get("action_type") == ACTION_TYPE:
        response = input(
            "按 Enter 直接输入；输入 m 先点击当前鼠标位置再输入；输入 s 跳过本步；输入 q 中止："
        ).strip().lower()
        if response == "q":
            raise OperatorAbort()
        if response == "s":
            raise OperatorSkip("operator skipped type action")
        if response == "m":
            x, y = current_mouse_position()
            payload["x"] = x
            payload["y"] = y
            payload["operator_override"] = "current_mouse_position"
            payload["reason"] = f"{payload.get('reason') or ''} | operator_override=type_focus".strip()
            print(f"输入前先点击当前鼠标位置聚焦: x={x}, y={y}")
        return payload

    if payload.get("action_type") != ACTION_CLICK:
        input("按 Enter 执行该动作，Ctrl+C 取消。")
        return payload

    response = input(
        "按 Enter 使用 B 规划点位；输入 m 使用当前鼠标位置；输入 s 跳过本步；输入 q 中止："
    ).strip().lower()
    if response == "q":
        raise OperatorAbort()
    if response == "s":
        raise OperatorSkip("operator skipped click action")
    if response == "m":
        x, y = current_mouse_position()
        payload["x"] = x
        payload["y"] = y
        payload["operator_override"] = "current_mouse_position"
        payload["reason"] = f"{payload.get('reason') or ''} | operator_override=mouse_position".strip()
        print(f"使用当前鼠标位置执行 click: x={x}, y={y}")
    return payload


def print_action_preview(
    *,
    action: Mapping[str, Any],
    ui_state: Mapping[str, Any],
    scenario: ScenarioSpec,
    step_idx: int,
    real_gui: bool,
) -> None:
    print(f"\n--- Step {step_idx}: {scenario.title} ---")
    print("B 输出 action:")
    print(json.dumps(dict(action), ensure_ascii=False, indent=2))
    candidate = candidate_by_id(ui_state, action.get("target_element_id"))
    if candidate:
        print("命中候选:")
        print(
            json.dumps(
                {
                    "id": candidate.get("id"),
                    "text": candidate.get("text"),
                    "role": candidate.get("role"),
                    "clickable": candidate.get("clickable"),
                    "editable": candidate.get("editable"),
                    "center": candidate.get("center"),
                    "confidence": candidate.get("confidence"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if action.get("action_type") == ACTION_CLICK and candidate.get("clickable") is not True:
            print("提示：该候选 clickable=false，真实点击前建议核对点位，必要时输入 m 使用当前鼠标位置。")
    if not real_gui:
        print("dry-run：不会执行 C 的桌面动作。")


def current_mouse_position() -> tuple[int, int]:
    try:
        import pyautogui
    except ImportError as exc:
        raise RuntimeError("pyautogui is required to read current mouse position") from exc
    pos = pyautogui.position()
    return int(pos.x), int(pos.y)


def fake_success_result(action: Mapping[str, Any]) -> dict[str, Any]:
    now = utc_now()
    return {
        "ok": True,
        "action_type": str(action.get("action_type") or "unknown"),
        "start_ts": now,
        "end_ts": now,
        "duration_ms": 0,
        "error": None,
    }


def ui_state_path(*, artifact_root: Path, run_id: str, step_idx: int, phase: str) -> str:
    return str(artifact_root / run_id / f"ui_state_step_{step_idx:03d}_{phase}.json")


def annotation_path(*, artifact_root: Path, run_id: str, step_idx: int) -> Path:
    return artifact_root / run_id / "annotations" / f"step_{step_idx:03d}_candidates.png"


def selected_candidate_summary(
    ui_state: Mapping[str, Any],
    candidate_id: Any,
) -> dict[str, Any] | None:
    candidate = candidate_by_id(ui_state, candidate_id)
    if candidate is None:
        return None
    return {
        "id": candidate.get("id"),
        "text": candidate.get("text"),
        "role": candidate.get("role"),
        "source": candidate.get("source"),
        "clickable": candidate.get("clickable"),
        "editable": candidate.get("editable"),
        "confidence": candidate.get("confidence"),
        "bbox": candidate.get("bbox"),
        "center": candidate.get("center"),
    }


def write_step_annotation(
    *,
    record: StepRecord,
    ui_state: Mapping[str, Any],
    action: Mapping[str, Any],
    artifact_root: Path,
    run_id: str,
    top_n: int,
    screenshot_metadata: Mapping[str, Any] | Any | None = None,
) -> None:
    if top_n <= 0:
        return
    path = annotation_path(artifact_root=artifact_root, run_id=run_id, step_idx=record.step_idx)
    try:
        draw_candidate_annotations(
            ui_state=ui_state,
            action=action,
            annotated_path=path,
            top_n=top_n,
            screenshot_metadata=screenshot_metadata,
        )
    except Exception as exc:
        record.notes.append(f"annotation_failed: {exc}")
        return
    record.annotation_path = str(path)


def draw_candidate_annotations(
    *,
    ui_state: Mapping[str, Any],
    action: Mapping[str, Any],
    annotated_path: Path,
    top_n: int = 30,
    screenshot_metadata: Mapping[str, Any] | Any | None = None,
) -> Path:
    screenshot_path = _resolve_existing_path(str(ui_state.get("screenshot_path") or ""))
    if screenshot_path is None:
        raise ValueError("ui_state screenshot_path does not exist")

    candidates = annotation_candidates(
        ui_state=ui_state,
        selected_candidate_id=action.get("target_element_id"),
        top_n=top_n,
    )
    screen_origin = annotation_screen_origin(screenshot_metadata)
    with Image.open(screenshot_path) as image:
        annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = annotation_font()
    width, height = annotated.size

    for index, candidate in enumerate(candidates, start=1):
        local_bbox = local_candidate_bbox(candidate, screen_origin, width, height)
        if local_bbox is None:
            continue
        draw.rectangle(local_bbox, outline="yellow", width=2)
        draw_label(draw, local_bbox, candidate_label(index, candidate), fill="yellow", font=font)

    selected = candidate_by_id(ui_state, action.get("target_element_id"))
    if selected is not None:
        local_bbox = local_candidate_bbox(selected, screen_origin, width, height)
        if local_bbox is not None:
            draw.rectangle(local_bbox, outline="red", width=5)
            draw_label(draw, local_bbox, "SELECTED " + candidate_label(0, selected), fill="red", font=font)

    draw_click_point(draw, action, screen_origin, width, height)
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(annotated_path)
    return annotated_path


def annotation_candidates(
    *,
    ui_state: Mapping[str, Any],
    selected_candidate_id: Any,
    top_n: int,
) -> list[dict[str, Any]]:
    raw_candidates = ui_state.get("candidates", [])
    if not isinstance(raw_candidates, list):
        return []
    candidates = [dict(candidate) for candidate in raw_candidates if isinstance(candidate, Mapping)]
    selected_id = str(selected_candidate_id) if selected_candidate_id is not None else None
    selected = None
    if selected_id is not None:
        selected = next((candidate for candidate in candidates if str(candidate.get("id")) == selected_id), None)

    selected_top = candidates[: max(0, top_n)]
    if selected is not None and all(str(candidate.get("id")) != selected_id for candidate in selected_top):
        selected_top.append(selected)
    return selected_top


def annotation_screen_origin(screenshot_metadata: Mapping[str, Any] | Any | None) -> list[int]:
    metadata = _optional_mapping(screenshot_metadata)
    if metadata is not None:
        origin = metadata.get("screen_origin")
        if _is_int_list(origin, 2):
            return [int(origin[0]), int(origin[1])]
        bbox = metadata.get("bbox")
        if _is_int_list(bbox, 4):
            return [int(bbox[0]), int(bbox[1])]
    return [0, 0]


def local_candidate_bbox(
    candidate: Mapping[str, Any],
    screen_origin: list[int],
    width: int,
    height: int,
) -> list[int] | None:
    bbox = candidate.get("bbox")
    if not _is_int_list(bbox, 4):
        return None
    local_bbox = [
        int(bbox[0]) - screen_origin[0],
        int(bbox[1]) - screen_origin[1],
        int(bbox[2]) - screen_origin[0],
        int(bbox[3]) - screen_origin[1],
    ]
    if local_bbox[2] < 0 or local_bbox[3] < 0 or local_bbox[0] > width or local_bbox[1] > height:
        return None
    return [
        max(0, min(width, local_bbox[0])),
        max(0, min(height, local_bbox[1])),
        max(0, min(width, local_bbox[2])),
        max(0, min(height, local_bbox[3])),
    ]


def draw_click_point(
    draw: ImageDraw.ImageDraw,
    action: Mapping[str, Any],
    screen_origin: list[int],
    width: int,
    height: int,
) -> None:
    x = action.get("x")
    y = action.get("y")
    if not _is_int_list([x, y], 2):
        return
    local_x = int(x) - screen_origin[0]
    local_y = int(y) - screen_origin[1]
    if not 0 <= local_x <= width or not 0 <= local_y <= height:
        return
    radius = 8
    draw.ellipse(
        [local_x - radius, local_y - radius, local_x + radius, local_y + radius],
        outline="cyan",
        width=3,
    )
    draw.line([local_x - 14, local_y, local_x + 14, local_y], fill="cyan", width=3)
    draw.line([local_x, local_y - 14, local_x, local_y + 14], fill="cyan", width=3)


def candidate_label(index: int, candidate: Mapping[str, Any]) -> str:
    confidence = _float_value(candidate.get("confidence"))
    prefix = "selected" if index <= 0 else str(index)
    text = str(candidate.get("text") or "").strip().replace("|", "/")
    if len(text) > 12:
        text = text[:12] + "..."
    return "{prefix}:{id} {role} {confidence:.2f} {text}".format(
        prefix=prefix,
        id=candidate.get("id") or "",
        role=candidate.get("role") or "",
        confidence=confidence,
        text=text,
    )


def draw_label(
    draw: ImageDraw.ImageDraw,
    bbox: list[int],
    label: str,
    *,
    fill: str,
    font: ImageFont.ImageFont,
) -> None:
    xy = (bbox[0] + 3, max(0, bbox[1] - 18))
    try:
        draw.text(xy, label, fill=fill, font=font)
    except UnicodeError:
        draw.text(xy, label.encode("ascii", "replace").decode("ascii"), fill=fill, font=font)


def annotation_font() -> ImageFont.ImageFont:
    for font_path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=14)
    return ImageFont.load_default()


def _resolve_existing_path(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return None


def _optional_mapping(value: Mapping[str, Any] | Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, Mapping):
            return dict(data)
    return None


def _is_int_list(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def candidate_by_id(ui_state: Mapping[str, Any], candidate_id: Any) -> dict[str, Any] | None:
    if candidate_id is None:
        return None
    raw_candidates = ui_state.get("candidates", [])
    if not isinstance(raw_candidates, list):
        return None
    for candidate in raw_candidates:
        if isinstance(candidate, Mapping) and str(candidate.get("id")) == str(candidate_id):
            return dict(candidate)
    return None


def print_header(
    args: argparse.Namespace,
    run_id: str,
    artifact_root: Path,
    specs: list[ScenarioSpec],
) -> None:
    print("\n=== ABC 第一周期 4 单步联合验收 ===")
    print(f"run_id: {run_id}")
    print(f"artifact_root: {artifact_root}")
    print(f"real_gui: {args.real_gui}")
    print(f"strict_after_checks: {args.strict_after_checks}")
    print(f"window_title: {None if args.fullscreen else args.window_title}")
    print(f"monitor_index: {args.monitor_index}")
    print("场景:")
    for index, spec in enumerate(specs, start=1):
        print(f"  {index}. {spec.title} ({spec.key})")


def print_scenario_result(result: ScenarioRunResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"\n[{status}] {result.spec.title}")
    if result.failure_reason:
        print(f"failure_reason: {result.failure_reason}")
    for step in result.steps:
        action_type = (step.executed_action or step.planned_action or {}).get("action_type")
        ok = (step.action_result or {}).get("ok")
        print(
            f"  step {step.step_idx}: action={action_type}, ok={ok}, "
            f"status={step.status}, annotation={step.annotation_path}"
        )


def write_reports(
    *,
    run_id: str,
    artifact_root: Path,
    scenario_results: list[ScenarioRunResult],
    recorder_summary: dict[str, Any],
    status: str,
    failure_reason: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_root = artifact_root / run_id
    summary_path = run_root / "abc_acceptance_summary.json"
    report_path = run_root / "abc_acceptance_report.md"
    passed_count = sum(result.passed for result in scenario_results)
    summary = {
        "version": "abc_joint_acceptance.v1",
        "run_id": run_id,
        "status": status,
        "failure_reason": failure_reason,
        "real_gui": args.real_gui,
        "strict_after_checks": args.strict_after_checks,
        "passed_count": passed_count,
        "total_count": len(scenario_results),
        "generated_at": utc_now(),
        "recorder_summary": recorder_summary,
        "metrics": aggregate_acceptance_metrics(
            artifact_root=artifact_root,
            run_id=run_id,
            scenario_results=scenario_results,
            recorder_summary=recorder_summary,
        ),
        "results": [result.to_dict() for result in scenario_results],
        "artifact_paths": {
            "summary_path": str(summary_path),
            "report_path": str(report_path),
            "run_root": str(run_root),
            "annotations_dir": str(run_root / "annotations"),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    metrics = summary.get("metrics", {})
    lines = [
        "# ABC 第一周期 4 单步联合验收报告",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- status: `{summary['status']}`",
        f"- real_gui: `{summary['real_gui']}`",
        f"- strict_after_checks: `{summary.get('strict_after_checks')}`",
        f"- passed: `{summary['passed_count']}/{summary['total_count']}`",
        f"- scenario_success_rate: `{metrics.get('scenario_success_rate', '')}`",
        f"- action_type_count: `{metrics.get('action_type_count', '')}`",
        f"- avg_before_candidate_count: `{metrics.get('avg_before_candidate_count', '')}`",
        "",
        "| 场景 | 状态 | 动作 | 结果 | 失败原因 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in summary.get("results", []):
        steps = result.get("steps", [])
        action_text = "<br>".join(
            str((step.get("executed_action") or step.get("planned_action") or {}).get("action_type") or "")
            for step in steps
        )
        ok_text = "<br>".join(str((step.get("action_result") or {}).get("ok")) for step in steps)
        lines.append(
            "| {title} | {status} | {actions} | {ok} | {failure} |".format(
                title=result.get("title", ""),
                status=result.get("status", ""),
                actions=action_text,
                ok=ok_text,
                failure=result.get("failure_reason") or "",
            )
        )

    lines.extend(["", "## 步骤明细", ""])
    for result in summary.get("results", []):
        lines.append(f"### {result.get('title', '')}")
        lines.append("")
        for step in result.get("steps", []):
            lines.append(f"- step `{step.get('step_idx')}` / planner_step `{step.get('planner_step_idx')}`")
            lines.append(f"  - before_ui_state: `{step.get('before_ui_state_path')}`")
            lines.append(f"  - after_ui_state: `{step.get('after_ui_state_path')}`")
            lines.append(f"  - status: `{step.get('status')}`")
            lines.append(f"  - failure_reason: `{step.get('failure_reason') or ''}`")
            action = step.get("executed_action") or step.get("planned_action") or {}
            lines.append(f"  - action: `{json.dumps(action, ensure_ascii=False)}`")
            annotation = step.get("annotation_path")
            if annotation:
                lines.append(f"  - annotation: `{annotation}`")
                lines.append(f"  - ![step {step.get('step_idx')} candidates]({markdown_path(annotation)})")
            selected = step.get("selected_candidate") or {}
            if selected:
                lines.extend(
                    [
                        "",
                        "| selected id | text | role | source | confidence | bbox | center |",
                        "| --- | --- | --- | --- | --- | --- | --- |",
                        "| {id} | {text} | {role} | {source} | {confidence} | {bbox} | {center} |".format(
                            id=selected.get("id", ""),
                            text=_markdown_cell(selected.get("text", "")),
                            role=_markdown_cell(selected.get("role", "")),
                            source=_markdown_cell(selected.get("source", "")),
                            confidence=selected.get("confidence", ""),
                            bbox=selected.get("bbox", ""),
                            center=selected.get("center", ""),
                        ),
                        "",
                    ]
                )
        lines.append("")
    return "\n".join(lines)


def markdown_path(path: Any) -> str:
    return str(path).replace("\\", "/")


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


def aggregate_acceptance_metrics(
    *,
    artifact_root: Path,
    run_id: str,
    scenario_results: list[ScenarioRunResult],
    recorder_summary: Mapping[str, Any],
) -> dict[str, Any]:
    total = len(scenario_results)
    passed = sum(result.passed for result in scenario_results)
    before_candidate_counts: list[int] = []
    for result in scenario_results:
        for step in result.steps:
            if not step.before_ui_state_path:
                continue
            path = Path(step.before_ui_state_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            candidate_count = payload.get("page_summary", {}).get("candidate_count")
            if isinstance(candidate_count, int) and not isinstance(candidate_count, bool):
                before_candidate_counts.append(candidate_count)

    action_stats = recorder_summary.get("action_stats", {})
    if not isinstance(action_stats, Mapping):
        action_stats = {}
    failure_action_types = [
        str(action_type)
        for action_type, stats in action_stats.items()
        if isinstance(stats, Mapping) and int(stats.get("failure_count") or 0) > 0
    ]
    return {
        "scenario_success_rate": round(passed / total, 4) if total else 0,
        "step_count": recorder_summary.get("step_count", 0),
        "action_type_count": len(action_stats),
        "action_stats": dict(action_stats),
        "before_candidate_counts": before_candidate_counts,
        "avg_before_candidate_count": (
            round(sum(before_candidate_counts) / len(before_candidate_counts), 2)
            if before_candidate_counts
            else 0
        ),
        "min_before_candidate_count": min(before_candidate_counts) if before_candidate_counts else 0,
        "max_before_candidate_count": max(before_candidate_counts) if before_candidate_counts else 0,
        "failure_action_types": failure_action_types,
        "run_root": str(artifact_root / run_id),
    }


def print_report_paths(summary: Mapping[str, Any]) -> None:
    paths = summary["artifact_paths"]
    print("\n=== ABC 联合验收输出 ===")
    print(f"summary: {paths['summary_path']}")
    print(f"report: {paths['report_path']}")
    print(f"run_root: {paths['run_root']}")
    print(f"status: {summary['status']}")
    print(f"passed: {summary['passed_count']}/{summary['total_count']}")


if __name__ == "__main__":
    raise SystemExit(main())
