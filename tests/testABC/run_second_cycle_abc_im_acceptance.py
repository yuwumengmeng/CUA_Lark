"""Second-cycle ABC real-GUI IM workflow acceptance runner.

This operator script is intentionally separate from the first-cycle ABC runner.
It validates the second-cycle chain:

    A get_ui_state() -> B PlannerLoopV2 -> C Executor/Validator/Recorder

The script sends a real message only when both --real-gui and --yes are passed.
Use --enable-vlm for the final second-cycle run so A's VLM result participates
in B's candidate selection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from executor import ExecutorWithRecorder, validate_expected_after_state  # noqa: E402
from planner.action_protocol import (  # noqa: E402
    ACTION_TYPE,
    DECISION_CONTINUE,
    DECISION_FAILED,
    DECISION_SUCCESS,
    action_decision_v2_to_executor_payload,
)
from planner.planner_loop_v2 import FAILURE_EXECUTOR_FAILED, PlannerLoopV2  # noqa: E402
from planner.runtime_state import RuntimeStateV2, RuntimeStatus, create_runtime_state_v2  # noqa: E402
from planner.workflow_schema import (  # noqa: E402
    STEP_OPEN_CHAT,
    STEP_SEND_MESSAGE,
    STEP_TYPE_KEYWORD,
    build_im_message_workflow,
)
from perception.ui_state import get_ui_state  # noqa: E402
from schemas import utc_now  # noqa: E402


DEFAULT_WINDOW_TITLE = "飞书"
DEFAULT_CHAT_NAME = "测试群"
DEFAULT_MESSAGE_PREFIX = "Hello CUA-Lark ABC"


@dataclass(slots=True)
class WorkflowStepRecord:
    loop_idx: int
    step_idx: int
    step_name: str | None
    decision: dict[str, Any]
    action_payload: dict[str, Any] | None = None
    action_result: dict[str, Any] | None = None
    validator_result: dict[str, Any] | None = None
    before_ui_state: dict[str, Any] = field(default_factory=dict)
    after_ui_state: dict[str, Any] = field(default_factory=dict)
    outcome: str = RuntimeStatus.RUNNING.value
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "loop_idx": self.loop_idx,
            "step_idx": self.step_idx,
            "step_name": self.step_name,
            "decision": self.decision,
            "action_payload": self.action_payload,
            "action_result": self.action_result,
            "validator_result": self.validator_result,
            "before": _state_snapshot(self.before_ui_state),
            "after": _state_snapshot(self.after_ui_state),
            "outcome": self.outcome,
            "failure_reason": self.failure_reason,
        }


def main() -> int:
    args = parse_args()
    token = datetime.now().strftime("%H%M%S")
    run_id = args.run_id or f"run_second_cycle_abc_im_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    message = args.message or f"{DEFAULT_MESSAGE_PREFIX} PASS{token}"
    assert_text = args.assert_text or f"PASS{token}"
    print_header(args=args, run_id=run_id, message=message, assert_text=assert_text)

    if not args.real_gui:
        print("Dry run only. Add --real-gui --yes to execute the ABC IM workflow.")
        return 0
    if not args.yes:
        input("This will send a real message to the Feishu test chat. Press Enter to start.")

    workflow = build_im_message_workflow(chat_name=args.chat_name, message_text=message)
    state = create_runtime_state_v2(
        workflow,
        task={
            "name": "second_cycle_abc_im_acceptance",
            "chat_name": args.chat_name,
            "message": message,
            "assert_text": assert_text,
            "real_gui": args.real_gui,
            "enable_vlm": args.enable_vlm,
            "sends_message": True,
        },
    )
    planner = PlannerLoopV2()
    runner = ExecutorWithRecorder(
        run_id=run_id,
        task=state.task,
        artifact_root=args.artifact_root,
        window_title=args.window_title,
        monitor_index=args.monitor_index,
        focus_window=not args.no_focus_window,
        focus_delay_seconds=args.focus_delay_seconds,
    )

    records: list[WorkflowStepRecord] = []
    try:
        status = run_workflow(
            args=args,
            run_id=run_id,
            assert_text=assert_text,
            planner=planner,
            runner=runner,
            state=state,
            records=records,
        )
    except Exception as exc:
        summary = runner.finalize(status="failed", failure_reason=str(exc))
        report = write_report(
            args=args,
            run_id=run_id,
            message=message,
            assert_text=assert_text,
            state=state,
            records=records,
            recorder_summary=summary,
            status="failed",
            failure_reason=str(exc),
        )
        print_result(status="failed", report=report, summary=summary, failure_reason=str(exc))
        raise

    final_status = "success" if status == RuntimeStatus.SUCCESS.value else "failed"
    final_failure = None if final_status == "success" else state.failure_reason or status
    summary = runner.finalize(status=final_status, failure_reason=final_failure)
    report = write_report(
        args=args,
        run_id=run_id,
        message=message,
        assert_text=assert_text,
        state=state,
        records=records,
        recorder_summary=summary,
        status=final_status,
        failure_reason=final_failure,
    )
    print_result(status=final_status, report=report, summary=summary, failure_reason=final_failure)
    return 0 if final_status == "success" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Second-cycle ABC IM real-GUI acceptance")
    parser.add_argument("--real-gui", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--enable-vlm", action="store_true")
    parser.add_argument("--after-vlm", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/runs")
    parser.add_argument("--window-title", default=DEFAULT_WINDOW_TITLE)
    parser.add_argument("--monitor-index", type=int, default=1)
    parser.add_argument("--no-focus-window", action="store_true")
    parser.add_argument("--focus-delay-seconds", type=float, default=0.5)
    parser.add_argument("--after-action-delay-seconds", type=float, default=1.2)
    parser.add_argument("--chat-name", default=DEFAULT_CHAT_NAME)
    parser.add_argument("--message", default=None)
    parser.add_argument("--assert-text", default=None)
    parser.add_argument("--max-steps", type=int, default=12)
    return parser.parse_args()


def run_workflow(
    *,
    args: argparse.Namespace,
    run_id: str,
    assert_text: str,
    planner: PlannerLoopV2,
    runner: ExecutorWithRecorder,
    state: RuntimeStateV2,
    records: list[WorkflowStepRecord],
) -> str:
    for loop_idx in range(1, args.max_steps + 1):
        if state.status in {RuntimeStatus.SUCCESS.value, RuntimeStatus.FAILED.value}:
            break

        current_step = state.current_step
        before_ui_state = collect_ui_state(
            args=args,
            run_id=run_id,
            step_idx=loop_idx,
            phase="before",
            enable_vlm=args.enable_vlm,
        )
        decision = planner.plan_next_action(state, before_ui_state)
        print(f"Loop {loop_idx}: {json.dumps(decision, ensure_ascii=False)}")

        if decision["status"] == DECISION_SUCCESS:
            validator_result = _validator_for_success_decision(
                decision,
                before_ui_state=before_ui_state,
                assert_text=assert_text,
            )
            action_result = _virtual_action_result(decision, validator_result=validator_result)
            state.record_action(decision, action_result, status=RuntimeStatus.SUCCESS.value)
            state.advance_step()
            runner.recorder.record_step(
                step_idx=loop_idx,
                step_name=decision.get("step_name"),
                action_decision=decision,
                action_result=action_result,
                after_ui_state=before_ui_state,
                validator_result=validator_result,
            )
            records.append(
                WorkflowStepRecord(
                    loop_idx=loop_idx,
                    step_idx=state.step_idx,
                    step_name=decision.get("step_name"),
                    decision=decision,
                    action_result=action_result,
                    validator_result=validator_result,
                    before_ui_state=before_ui_state,
                    after_ui_state=before_ui_state,
                    outcome=RuntimeStatus.SUCCESS.value,
                )
            )
            continue

        if decision["status"] == DECISION_FAILED:
            state.record_action(
                decision,
                None,
                status=RuntimeStatus.FAILED.value,
                failure_reason=decision.get("failure_reason"),
            )
            records.append(
                WorkflowStepRecord(
                    loop_idx=loop_idx,
                    step_idx=state.step_idx,
                    step_name=decision.get("step_name"),
                    decision=decision,
                    before_ui_state=before_ui_state,
                    outcome=RuntimeStatus.FAILED.value,
                    failure_reason=decision.get("failure_reason"),
                )
            )
            break

        action_payload = action_decision_v2_to_executor_payload(decision)
        action_result, after_ui_state, validator_result = execute_and_validate(
            args=args,
            run_id=run_id,
            loop_idx=loop_idx,
            runner=runner,
            decision=decision,
            action_payload=action_payload,
            before_ui_state=before_ui_state,
        )

        if action_result.get("ok") is not True:
            planner._handle_retry_or_failure(  # noqa: SLF001
                state,
                decision,
                action_result,
                failure_reason=FAILURE_EXECUTOR_FAILED,
            )
            outcome = state.status
            failure_reason = state.failure_reason
        else:
            outcome, failure_reason = planner._step_outcome(  # noqa: SLF001
                state,
                decision,
                action_result,
                before_ui_state,
                after_ui_state,
            )
            state.update_ui_state(after_ui_state)
            if outcome == RuntimeStatus.SUCCESS.value:
                state.record_action(decision, action_result, status=RuntimeStatus.SUCCESS.value)
                state.advance_step()
                if state.current_step is not None:
                    state.status = RuntimeStatus.RUNNING.value
            elif outcome == DECISION_CONTINUE:
                state.record_action(decision, action_result, status=RuntimeStatus.RUNNING.value)
            else:
                planner._handle_retry_or_failure(  # noqa: SLF001
                    state,
                    decision,
                    action_result,
                    failure_reason=failure_reason or "validation_failed",
                )
                outcome = state.status
                failure_reason = state.failure_reason

        records.append(
            WorkflowStepRecord(
                loop_idx=loop_idx,
                step_idx=state.step_idx,
                step_name=None if current_step is None else current_step.step_name,
                decision=decision,
                action_payload=action_payload,
                action_result=action_result,
                validator_result=validator_result,
                before_ui_state=before_ui_state,
                after_ui_state=after_ui_state,
                outcome=outcome,
                failure_reason=failure_reason,
            )
        )

    if state.status == RuntimeStatus.RUNNING.value:
        state.mark_failed("max_steps_exceeded")
    return state.status


def execute_and_validate(
    *,
    args: argparse.Namespace,
    run_id: str,
    loop_idx: int,
    runner: ExecutorWithRecorder,
    decision: Mapping[str, Any],
    action_payload: Mapping[str, Any],
    before_ui_state: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    before_screenshot = runner.screenshot_capturer.capture_before(run_id=run_id, step_idx=loop_idx)
    action_result_obj = runner.executor.execute_action(action_payload)
    time.sleep(args.after_action_delay_seconds)
    after_screenshot = runner.screenshot_capturer.capture_after(run_id=run_id, step_idx=loop_idx)
    after_ui_state = collect_ui_state(
        args=args,
        run_id=run_id,
        step_idx=loop_idx,
        phase="after",
        enable_vlm=args.after_vlm,
    )
    validator_result = _validator_for_decision(
        decision,
        before_ui_state=before_ui_state,
        after_ui_state=after_ui_state,
    )
    action_result = action_result_obj.to_dict()
    action_result["before_screenshot"] = _jsonable(before_screenshot)
    action_result["after_screenshot"] = _jsonable(after_screenshot)
    action_result["screen_meta_snapshot"] = after_ui_state.get("screen_meta")
    if _should_feed_validator_to_planner(decision, validator_result):
        action_result["validator_result"] = validator_result
        action_result["validation_passed"] = validator_result.get("passed")

    runner.recorder.record_step(
        step_idx=loop_idx,
        step_name=decision.get("step_name"),
        action_decision=decision,
        executed_action=action_payload,
        action_result=action_result,
        before_screenshot=before_screenshot,
        after_screenshot=after_screenshot,
        after_ui_state=after_ui_state,
        validator_result=validator_result,
        failure_reason=validator_result.get("failure_reason") if validator_result else None,
    )
    return action_result, dict(after_ui_state), validator_result


def collect_ui_state(
    *,
    args: argparse.Namespace,
    run_id: str,
    step_idx: int,
    phase: str,
    enable_vlm: bool,
) -> dict[str, Any]:
    return get_ui_state(
        run_id=run_id,
        step_idx=step_idx,
        phase=phase,
        artifact_root=args.artifact_root,
        monitor_index=args.monitor_index,
        capture_window_title=args.window_title,
        window_title=args.window_title,
        enable_vlm=enable_vlm,
        capture_focus_delay_seconds=args.focus_delay_seconds,
    )


def _validator_for_decision(
    decision: Mapping[str, Any],
    *,
    before_ui_state: Mapping[str, Any],
    after_ui_state: Mapping[str, Any],
) -> dict[str, Any] | None:
    expectations = _validator_expectations(decision.get("expected_after_state"))
    if not expectations:
        return None
    return validate_expected_after_state(
        expectations,
        before_ui_state=before_ui_state,
        after_ui_state=after_ui_state,
    ).to_dict()


def _validator_for_success_decision(
    decision: Mapping[str, Any],
    *,
    before_ui_state: Mapping[str, Any],
    assert_text: str,
) -> dict[str, Any]:
    expectations = _validator_expectations(decision.get("expected_after_state")) or {
        "message_text": assert_text
    }
    return validate_expected_after_state(
        expectations,
        before_ui_state=before_ui_state,
        after_ui_state=before_ui_state,
    ).to_dict()


def _validator_expectations(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    if value.get("text_present") is not None:
        return {"target_text": value["text_present"]}
    if value.get("chat_name_visible") is not None:
        return {"target_text": value["chat_name_visible"]}
    if value.get("message_visible") is not None:
        return {"message_text": value["message_visible"]}
    if value.get("message_sent") is not None:
        return {"message_text": value["message_sent"]}
    return {}


def _should_feed_validator_to_planner(
    decision: Mapping[str, Any],
    validator_result: Mapping[str, Any] | None,
) -> bool:
    if validator_result is None or validator_result.get("passed") is not True:
        return False
    if decision.get("step_name") in {STEP_TYPE_KEYWORD, STEP_OPEN_CHAT}:
        return False
    return not (
        decision.get("step_name") == STEP_SEND_MESSAGE
        and decision.get("action_type") == ACTION_TYPE
    )


def _virtual_action_result(
    decision: Mapping[str, Any],
    *,
    validator_result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "ok": validator_result.get("passed") is True,
        "action_type": decision.get("action_type") or "none",
        "start_ts": utc_now(),
        "end_ts": utc_now(),
        "duration_ms": 0,
        "error": None if validator_result.get("passed") else validator_result.get("failure_reason"),
        "validator_result": dict(validator_result),
        "validation_passed": validator_result.get("passed"),
        "executor_message": "planner success / validator-only step",
    }


def write_report(
    *,
    args: argparse.Namespace,
    run_id: str,
    message: str,
    assert_text: str,
    state: RuntimeStateV2,
    records: list[WorkflowStepRecord],
    recorder_summary: Mapping[str, Any],
    status: str,
    failure_reason: str | None,
) -> Path:
    run_root = Path(args.artifact_root) / run_id
    report_path = run_root / "abc_second_cycle_im_acceptance.json"
    report = {
        "version": "abc_second_cycle_im_acceptance.v1",
        "run_id": run_id,
        "status": status,
        "passed": status == "success",
        "failure_reason": failure_reason,
        "window_title": args.window_title,
        "chat_name": args.chat_name,
        "message": message,
        "assert_text": assert_text,
        "real_gui": args.real_gui,
        "enable_vlm": args.enable_vlm,
        "after_vlm": args.after_vlm,
        "abc_chain": {
            "A": "perception.ui_state.get_ui_state",
            "B": "planner.planner_loop_v2.PlannerLoopV2",
            "C": "executor.ExecutorWithRecorder + executor.validator",
        },
        "vlm_usage": _vlm_usage(records),
        "runtime_state": state.to_dict(),
        "recorder_summary": dict(recorder_summary),
        "records": [record.to_dict() for record in records],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def _vlm_usage(records: list[WorkflowStepRecord]) -> dict[str, Any]:
    states: list[Mapping[str, Any]] = []
    for record in records:
        if record.before_ui_state:
            states.append(record.before_ui_state)
        if record.after_ui_state:
            states.append(record.after_ui_state)

    attempted = 0
    ok = 0
    fallback = 0
    targets: list[str] = []
    reasons: list[str] = []
    for state in states:
        vlm_result = state.get("vlm_result")
        if not isinstance(vlm_result, Mapping) or not vlm_result:
            continue
        attempted += 1
        if vlm_result.get("ok") is True:
            ok += 1
        if vlm_result.get("fallback_used") is True:
            fallback += 1
        structured = vlm_result.get("structured")
        if isinstance(structured, Mapping):
            target = str(structured.get("target_candidate_id") or "").strip()
            if target:
                targets.append(target)
            reason = str(structured.get("reason") or "").strip()
            if reason:
                reasons.append(reason)
    return {
        "attempted_state_count": attempted,
        "ok_state_count": ok,
        "fallback_state_count": fallback,
        "target_candidate_ids": targets,
        "sample_reasons": reasons[:3],
        "participated": ok > 0,
    }


def _state_snapshot(ui_state: Mapping[str, Any]) -> dict[str, Any]:
    page_summary = ui_state.get("page_summary") if isinstance(ui_state.get("page_summary"), Mapping) else {}
    candidates = ui_state.get("candidates") if isinstance(ui_state.get("candidates"), list) else []
    vlm_result = ui_state.get("vlm_result") if isinstance(ui_state.get("vlm_result"), Mapping) else {}
    structured = vlm_result.get("structured") if isinstance(vlm_result.get("structured"), Mapping) else {}
    return {
        "screenshot_path": ui_state.get("screenshot_path"),
        "candidate_count": len(candidates),
        "top_texts": list(page_summary.get("top_texts", []))[:10],
        "screen_meta": ui_state.get("screen_meta"),
        "vlm_ok": vlm_result.get("ok"),
        "vlm_target_candidate_id": structured.get("target_candidate_id"),
    }


def _jsonable(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, Mapping):
            return dict(data)
    return {"repr": repr(value)}


def print_header(
    *,
    args: argparse.Namespace,
    run_id: str,
    message: str,
    assert_text: str,
) -> None:
    print("=== ABC second-cycle IM acceptance ===")
    print(f"run_id: {run_id}")
    print(f"window_title: {args.window_title}")
    print(f"target chat: {args.chat_name}")
    print(f"message: {message}")
    print(f"assert_text: {assert_text}")
    print(f"enable_vlm: {args.enable_vlm}")
    print("This runner sends a real message only in --real-gui --yes mode.")


def print_result(
    *,
    status: str,
    report: Path,
    summary: Mapping[str, Any],
    failure_reason: str | None,
) -> None:
    print("=== ABC second-cycle IM acceptance result ===")
    print(f"status: {status}")
    print(f"failure_reason: {failure_reason}")
    print(f"summary_status: {summary.get('status')}")
    print(f"step_count: {summary.get('step_count')}")
    print(f"success_rate: {summary.get('success_rate')}")
    print(f"report: {report}")


if __name__ == "__main__":
    raise SystemExit(main())
