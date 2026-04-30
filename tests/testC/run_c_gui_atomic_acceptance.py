"""Member C second-cycle real-GUI atomic acceptance runner.

This is an operator script, not a pytest case. It validates C-side GUI atoms
that do not require final A/B integration:

1. collect before ui_state from the Feishu window
2. execute Ctrl+K search hotkey through ExecutorWithRecorder
3. type a safe search keyword through ExecutorWithRecorder
4. wait for results
5. collect after ui_state
6. validate that the keyword appears and page state changed
7. persist action/result/screenshot/validator/ui_state logs and a summary

The script never sends a chat message. Pass --real-gui to execute desktop
keyboard actions. Without --real-gui it only prints the planned steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from executor import ExecutorWithRecorder, validate_expected_after_state  # noqa: E402
from perception.ui_state import get_ui_state  # noqa: E402


DEFAULT_WINDOW_TITLE = "飞书"
DEFAULT_CHAT_NAME = "测试群"


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"run_c_gui_atomic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    actions = build_actions(args)
    print_header(args=args, run_id=run_id, actions=actions)

    if not args.real_gui:
        print("Dry run only. Add --real-gui to execute desktop actions.")
        return 0
    if not args.yes:
        input("Feishu must be open and safe for search input. Press Enter to start, Ctrl+C to abort.")

    runner = ExecutorWithRecorder(
        run_id=run_id,
        task={
            "name": "c_gui_atomic_acceptance",
            "target": args.chat_name,
            "actions": actions,
            "sends_message": False,
        },
        artifact_root=args.artifact_root,
        window_title=args.window_title,
        monitor_index=args.monitor_index,
        focus_window=not args.no_focus_window,
        focus_delay_seconds=args.focus_delay_seconds,
    )

    before_ui_state = collect_ui_state(
        args=args,
        run_id=run_id,
        step_idx=0,
        phase="before",
    )
    step_results: list[dict[str, Any]] = []
    all_ok = True

    for step_idx, action in enumerate(actions, start=1):
        after_ui_state = None
        validator_result = None
        if step_idx == len(actions):
            # The final wait step gets the validator result after we collect
            # the visible state below. It is recorded with the run summary.
            pass
        print(f"Step {step_idx}: {json.dumps(action, ensure_ascii=False)}")
        step = runner.run_step(
            action,
            step_idx=step_idx,
            after_ui_state=after_ui_state,
            validator_result=validator_result,
        )
        payload = step.to_dict()
        step_results.append(payload)
        all_ok = all_ok and step.action_result.ok

    after_ui_state = collect_ui_state(
        args=args,
        run_id=run_id,
        step_idx=len(actions) + 1,
        phase="after",
    )
    validator_result = validate_expected_after_state(
        {
            "page_changed": True,
            "target_text": args.chat_name,
        },
        before_ui_state=before_ui_state,
        after_ui_state=after_ui_state,
    )
    runner.recorder.record_step(
        step_idx=len(actions) + 1,
        action_decision={
            "action_type": "wait",
            "step_name": "validate_search_results",
            "seconds": 0,
            "expected_after_state": {"target_text": args.chat_name, "page_changed": True},
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

    if args.cleanup_esc:
        cleanup_action = {
            "action_type": "hotkey",
            "step_name": "cleanup_close_search",
            "keys": ["esc"],
            "reason": "close search overlay",
        }
        step = runner.run_step(cleanup_action, step_idx=len(actions) + 2)
        step_results.append(step.to_dict())

    all_ok = all_ok and validator_result.passed
    summary = runner.finalize(status="success" if all_ok else "failed")
    report = write_acceptance_report(
        args=args,
        run_id=run_id,
        summary=summary,
        before_ui_state=before_ui_state,
        after_ui_state=after_ui_state,
        validator_result=validator_result.to_dict(),
        step_results=step_results,
    )
    print_result(summary=summary, report=report, validator_result=validator_result.to_dict())
    return 0 if all_ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Member C real GUI atomic acceptance")
    parser.add_argument("--real-gui", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/runs")
    parser.add_argument("--window-title", default=DEFAULT_WINDOW_TITLE)
    parser.add_argument("--monitor-index", type=int, default=1)
    parser.add_argument("--no-focus-window", action="store_true")
    parser.add_argument("--focus-delay-seconds", type=float, default=0.5)
    parser.add_argument("--chat-name", default=DEFAULT_CHAT_NAME)
    parser.add_argument("--search-hotkey", nargs="+", default=["ctrl", "k"])
    parser.add_argument("--wait-seconds", type=float, default=1.2)
    parser.add_argument("--cleanup-esc", action="store_true")
    parser.add_argument("--enable-vlm", action="store_true")
    return parser.parse_args()


def build_actions(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
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
            "expected_after_state": {"target_text": args.chat_name, "page_changed": True},
        },
    ]


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
    summary: Mapping[str, Any],
    before_ui_state: Mapping[str, Any],
    after_ui_state: Mapping[str, Any],
    validator_result: Mapping[str, Any],
    step_results: list[Mapping[str, Any]],
) -> Path:
    run_root = Path(args.artifact_root) / run_id
    report_path = run_root / "c_gui_atomic_acceptance.json"
    report = {
        "version": "c_gui_atomic_acceptance.v1",
        "run_id": run_id,
        "window_title": args.window_title,
        "chat_name": args.chat_name,
        "sends_message": False,
        "summary": dict(summary),
        "validator_result": dict(validator_result),
        "before": _state_snapshot(before_ui_state),
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


def print_header(*, args: argparse.Namespace, run_id: str, actions: list[Mapping[str, Any]]) -> None:
    print("=== Member C GUI atomic acceptance ===")
    print(f"run_id: {run_id}")
    print(f"window_title: {args.window_title}")
    print(f"target search text: {args.chat_name}")
    print("This runner does not send chat messages.")
    for index, action in enumerate(actions, start=1):
        print(f"  {index}. {json.dumps(action, ensure_ascii=False)}")


def print_result(
    *,
    summary: Mapping[str, Any],
    report: Path,
    validator_result: Mapping[str, Any],
) -> None:
    print("=== Member C GUI atomic acceptance result ===")
    print(f"status: {summary.get('status')}")
    print(f"validator_passed: {validator_result.get('passed')}")
    print(f"validator_failure_reason: {validator_result.get('failure_reason')}")
    print(f"summary: {summary.get('artifact_paths', {}).get('summary_path')}")
    print(f"report: {report}")


def _utc_now() -> str:
    from schemas import utc_now

    return utc_now()


if __name__ == "__main__":
    raise SystemExit(main())
