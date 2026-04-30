"""Member C real-GUI IM send acceptance runner.

This operator script validates the C-side executor path for a safe IM send
flow:

1. open Feishu search
2. type the target chat name
3. click the first search result
4. click the message input area
5. type a unique test message
6. press Enter to send
7. validate that the sent text is visible in the after ui_state

It sends a real message only when both --real-gui and --yes are provided.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capture.screenshot import ScreenBounds, _find_window  # noqa: E402
from executor import ExecutorWithRecorder, validate_expected_after_state  # noqa: E402
from perception.ui_state import get_ui_state  # noqa: E402


DEFAULT_WINDOW_TITLE = "\u98de\u4e66"
DEFAULT_CHAT_NAME = "\u6d4b\u8bd5\u7fa4"
DEFAULT_MESSAGE_PREFIX = "Hello CUA-Lark C"
DEFAULT_SEARCH_RESULT_X_RATIO = 0.40
DEFAULT_SEARCH_RESULT_Y_RATIO = 0.24

WindowFinder = Callable[[str], tuple[int, str, ScreenBounds]]


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"run_c_im_send_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    message = args.message or f"{DEFAULT_MESSAGE_PREFIX} {run_id}"
    assert_text = args.assert_text or message
    print_header(args=args, run_id=run_id, message=message, assert_text=assert_text)

    if not args.real_gui:
        print("Dry run only. Add --real-gui --yes to send the message.")
        return 0
    if not args.yes:
        input("This will send a real message to the Feishu test chat. Press Enter to start.")

    runner = ExecutorWithRecorder(
        run_id=run_id,
        task={
            "name": "c_im_send_acceptance",
            "target": args.chat_name,
            "message": message,
            "sends_message": True,
        },
        artifact_root=args.artifact_root,
        window_title=args.window_title,
        monitor_index=args.monitor_index,
        focus_window=not args.no_focus_window,
        focus_delay_seconds=args.focus_delay_seconds,
    )

    before_ui_state = collect_ui_state(args=args, run_id=run_id, step_idx=0, phase="before")
    step_results: list[dict[str, Any]] = []
    all_ok = True

    next_step_idx = 1
    for action in build_search_actions(args):
        print(f"Step {next_step_idx}: {json.dumps(action, ensure_ascii=False)}")
        step = runner.run_step(action, step_idx=next_step_idx)
        step_results.append(step.to_dict())
        all_ok = all_ok and step.action_result.ok
        next_step_idx += 1

    search_result_point = resolve_first_search_result_point(
        args.window_title,
        x_ratio=args.search_result_x_ratio,
        y_ratio=args.search_result_y_ratio,
    )
    for action in build_open_result_actions(args, search_result_point):
        print(f"Step {next_step_idx}: {json.dumps(action, ensure_ascii=False)}")
        step = runner.run_step(action, step_idx=next_step_idx)
        step_results.append(step.to_dict())
        all_ok = all_ok and step.action_result.ok
        next_step_idx += 1

    chat_ui_state = collect_ui_state(
        args=args,
        run_id=run_id,
        step_idx=next_step_idx,
        phase="manual",
    )
    input_point = resolve_message_input_point(
        args.window_title,
        x_ratio=args.input_x_ratio,
        bottom_offset=args.input_bottom_offset,
    )

    for action in build_send_actions(message, input_point):
        print(f"Step {next_step_idx}: {json.dumps(action, ensure_ascii=False)}")
        step = runner.run_step(action, step_idx=next_step_idx)
        step_results.append(step.to_dict())
        all_ok = all_ok and step.action_result.ok
        next_step_idx += 1

    after_ui_state = collect_ui_state(
        args=args,
        run_id=run_id,
        step_idx=next_step_idx,
        phase="after",
    )
    validator_result = validate_expected_after_state(
        {"message_text": assert_text},
        before_ui_state=chat_ui_state,
        after_ui_state=after_ui_state,
    )
    runner.recorder.record_step(
        step_idx=next_step_idx,
        action_decision={
            "action_type": "wait",
            "step_name": "validate_message_sent",
            "seconds": 0,
            "expected_after_state": {"message_text": assert_text},
        },
        action_result={
            "ok": validator_result.passed,
            "action_type": "wait",
            "start_ts": _utc_now(),
            "end_ts": _utc_now(),
            "error": None if validator_result.passed else validator_result.failure_reason,
            "duration_ms": 0,
            "executor_message": "validator-only step",
        },
        after_ui_state=after_ui_state,
        validator_result=validator_result.to_dict(),
        failure_reason=validator_result.failure_reason,
    )

    all_ok = all_ok and validator_result.passed
    summary = runner.finalize(status="success" if all_ok else "failed")
    report = write_acceptance_report(
        args=args,
        run_id=run_id,
        message=message,
        assert_text=assert_text,
        search_result_point=search_result_point,
        input_point=input_point,
        summary=summary,
        before_ui_state=before_ui_state,
        chat_ui_state=chat_ui_state,
        after_ui_state=after_ui_state,
        validator_result=validator_result.to_dict(),
        step_results=step_results,
    )
    print_result(summary=summary, report=report, validator_result=validator_result.to_dict())
    return 0 if all_ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Member C real GUI IM send acceptance")
    parser.add_argument("--real-gui", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/runs")
    parser.add_argument("--window-title", default=DEFAULT_WINDOW_TITLE)
    parser.add_argument("--monitor-index", type=int, default=1)
    parser.add_argument("--no-focus-window", action="store_true")
    parser.add_argument("--focus-delay-seconds", type=float, default=0.5)
    parser.add_argument("--chat-name", default=DEFAULT_CHAT_NAME)
    parser.add_argument("--message", default=None)
    parser.add_argument("--assert-text", default=None)
    parser.add_argument("--search-hotkey", nargs="+", default=["ctrl", "k"])
    parser.add_argument("--wait-seconds", type=float, default=1.2)
    parser.add_argument("--search-result-x-ratio", type=float, default=DEFAULT_SEARCH_RESULT_X_RATIO)
    parser.add_argument("--search-result-y-ratio", type=float, default=DEFAULT_SEARCH_RESULT_Y_RATIO)
    parser.add_argument("--input-x-ratio", type=float, default=0.55)
    parser.add_argument("--input-bottom-offset", type=int, default=95)
    parser.add_argument("--enable-vlm", action="store_true")
    return parser.parse_args()


def build_search_actions(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "action_type": "hotkey",
            "step_name": "cleanup_close_search",
            "keys": ["esc"],
            "repeat": 2,
            "reason": "close any previous search overlay",
        },
        {
            "action_type": "hotkey",
            "step_name": "open_search_by_hotkey",
            "keys": list(args.search_hotkey),
            "reason": "open Feishu search",
            "expected_after_state": {"page_changed": True},
        },
        {
            "action_type": "type",
            "step_name": "type_search_keyword",
            "text": args.chat_name,
            "reason": "type target chat name into search",
            "expected_after_state": {"target_text": args.chat_name},
        },
        {
            "action_type": "wait",
            "step_name": "wait_for_search_results",
            "seconds": args.wait_seconds,
            "reason": "wait for search results",
            "expected_after_state": {"target_text": args.chat_name},
        },
    ]


def build_open_result_actions(
    args: argparse.Namespace,
    search_result_point: tuple[int, int],
) -> list[dict[str, Any]]:
    x, y = search_result_point
    return [
        {
            "action_type": "click",
            "step_name": "click_first_search_result",
            "x": x,
            "y": y,
            "reason": "open first matching search result",
            "expected_after_state": {"target_text": args.chat_name, "page_changed": True},
        },
        {
            "action_type": "wait",
            "step_name": "wait_for_chat_page",
            "seconds": args.wait_seconds,
            "reason": "wait for chat page to load",
            "expected_after_state": {"target_text": args.chat_name},
        },
    ]


def build_send_actions(message: str, input_point: tuple[int, int]) -> list[dict[str, Any]]:
    x, y = input_point
    return [
        {
            "action_type": "type",
            "step_name": "type_message",
            "text": message,
            "x": x,
            "y": y,
            "reason": "focus message input and type test message",
            "expected_after_state": {"message_text": message},
        },
        {
            "action_type": "hotkey",
            "step_name": "press_enter_to_send",
            "keys": ["enter"],
            "reason": "send message with Enter",
            "expected_after_state": {"message_text": message, "page_changed": True},
        },
        {
            "action_type": "wait",
            "step_name": "wait_for_message_sent",
            "seconds": 1.0,
            "reason": "wait for sent message to appear",
            "expected_after_state": {"message_text": message},
        },
    ]


def resolve_message_input_point(
    window_title: str,
    *,
    x_ratio: float = 0.55,
    bottom_offset: int = 95,
    finder: WindowFinder | None = None,
) -> tuple[int, int]:
    if not 0 < x_ratio < 1:
        raise ValueError("x_ratio must be between 0 and 1")
    if bottom_offset <= 0:
        raise ValueError("bottom_offset must be positive")
    window_finder = finder or _find_window
    _, _, bounds = window_finder(window_title)
    return (
        bounds.x + int(bounds.width * x_ratio),
        bounds.y + bounds.height - int(bottom_offset),
    )


def resolve_first_search_result_point(
    window_title: str,
    *,
    x_ratio: float = DEFAULT_SEARCH_RESULT_X_RATIO,
    y_ratio: float = DEFAULT_SEARCH_RESULT_Y_RATIO,
    finder: WindowFinder | None = None,
) -> tuple[int, int]:
    if not 0 < x_ratio < 1:
        raise ValueError("x_ratio must be between 0 and 1")
    if not 0 < y_ratio < 1:
        raise ValueError("y_ratio must be between 0 and 1")
    window_finder = finder or _find_window
    _, _, bounds = window_finder(window_title)
    return (
        bounds.x + int(bounds.width * x_ratio),
        bounds.y + int(bounds.height * y_ratio),
    )


def collect_ui_state(
    *,
    args: argparse.Namespace,
    run_id: str,
    step_idx: int,
    phase: str,
) -> dict[str, Any]:
    return get_ui_state(
        run_id=run_id,
        step_idx=step_idx,
        phase=phase,
        artifact_root=args.artifact_root,
        monitor_index=args.monitor_index,
        capture_window_title=args.window_title,
        window_title=args.window_title,
        enable_vlm=args.enable_vlm,
        capture_focus_delay_seconds=args.focus_delay_seconds,
    )


def write_acceptance_report(
    *,
    args: argparse.Namespace,
    run_id: str,
    message: str,
    assert_text: str,
    search_result_point: tuple[int, int],
    input_point: tuple[int, int],
    summary: Mapping[str, Any],
    before_ui_state: Mapping[str, Any],
    chat_ui_state: Mapping[str, Any],
    after_ui_state: Mapping[str, Any],
    validator_result: Mapping[str, Any],
    step_results: list[Mapping[str, Any]],
) -> Path:
    run_root = Path(args.artifact_root) / run_id
    report_path = run_root / "c_im_send_acceptance.json"
    report = {
        "version": "c_im_send_acceptance.v1",
        "run_id": run_id,
        "window_title": args.window_title,
        "chat_name": args.chat_name,
        "message": message,
        "assert_text": assert_text,
        "sends_message": True,
        "search_result_point": list(search_result_point),
        "input_point": list(input_point),
        "summary": dict(summary),
        "validator_result": dict(validator_result),
        "before": _state_snapshot(before_ui_state),
        "chat": _state_snapshot(chat_ui_state),
        "after": _state_snapshot(after_ui_state),
        "steps": [dict(step) for step in step_results],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def _state_snapshot(ui_state: Mapping[str, Any]) -> dict[str, Any]:
    page_summary = ui_state.get("page_summary") if isinstance(ui_state.get("page_summary"), Mapping) else {}
    candidates = ui_state.get("candidates") if isinstance(ui_state.get("candidates"), list) else []
    return {
        "screenshot_path": ui_state.get("screenshot_path"),
        "candidate_count": len(candidates),
        "top_texts": list(page_summary.get("top_texts", []))[:10],
        "screen_meta": ui_state.get("screen_meta"),
    }


def print_header(
    *,
    args: argparse.Namespace,
    run_id: str,
    message: str,
    assert_text: str,
) -> None:
    print("=== Member C IM send acceptance ===")
    print(f"run_id: {run_id}")
    print(f"window_title: {args.window_title}")
    print(f"target chat: {args.chat_name}")
    print(f"message: {message}")
    print(f"assert_text: {assert_text}")
    print("This runner sends a real message only in --real-gui --yes mode.")


def print_result(
    *,
    summary: Mapping[str, Any],
    report: Path,
    validator_result: Mapping[str, Any],
) -> None:
    print("=== Member C IM send acceptance result ===")
    print(f"status: {summary.get('status')}")
    print(f"validator_passed: {validator_result.get('passed')}")
    print(f"validator_failure_reason: {validator_result.get('failure_reason')}")
    artifact_paths = summary.get("artifact_paths") if isinstance(summary.get("artifact_paths"), Mapping) else {}
    print(f"summary: {Path(artifact_paths.get('summary_path', ''))}")
    print(f"report: {report}")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
