"""成员 B 第二周期 Planner Loop v2。

职责边界：
本文件把 `workflow.v2 / runtime_state.v2 / action_decision.v2` 串成可连续推进的
B 侧短流程闭环；不访问真实飞书窗口、不调用真实 VLM、不直接执行鼠标键盘。

公开接口：
`PlannerLoopV2`
    多步 workflow planner，提供 plan/run once/run until done 三类入口。
`plan_next_workflow_action(...)`
    只根据当前 step 和 ui_state 生成下一步 ActionDecisionV2。
`run_workflow_once(...)`
    执行一次 get_ui_state -> plan -> execute_action -> after ui_state -> 更新 runtime。
`run_workflow_until_done(...)`
    使用 fake 或真实 A/C 回调连续推进，直到 success/failed 或达到 max_steps。

协作规则：
1. A 侧输入按顶层 `candidates / page_summary / vlm_result /
   document_region_summary / screen_meta / document_truth` 读取；后三者可选，缺失不阻断主链路。
2. C 侧只消费 `ActionDecisionV2.to_executor_dict()` 输出，不需要理解 workflow 语义。
3. VLM 高置信候选可优先，VLM 不确定时回退规则打分；候选冲突、坐标异常和候选类型误选必须记录原因。
4. fallback 只做第二周期轻量策略：热键打开搜索、等待后重试、重新聚焦、返回错误页；
   不做复杂 backtrack 树、不绕过 GUI。
5. step 成功判断只读 after `ui_state`、VLM semantic、C validator 结果和可选
   `document_truth`；文档真值只接受 openapi / mcp / clipboard / export 来源，不把完整 OCR 正文当 truth。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .action_protocol import (
    ACTION_TYPE,
    DECISION_CONTINUE,
    DECISION_FAILED,
    DECISION_SUCCESS,
    ActionDecisionV2,
    action_decision_v2_to_executor_payload,
)
from .runtime_state import (
    RuntimeStateV2,
    RuntimeStatus,
    create_runtime_state_v2,
    parse_runtime_state_v2,
)
from .workflow_schema import (
    STEP_FOCUS_MESSAGE_INPUT,
    STEP_OPEN_CHAT,
    STEP_OPEN_SEARCH,
    STEP_SEND_MESSAGE,
    STEP_TYPE_KEYWORD,
    STEP_VERIFY_MESSAGE_SENT,
    WorkflowSpec,
    WorkflowStep,
    parse_workflow,
)


GetUIStateV2 = Callable[[], Mapping[str, Any]]
ExecuteActionV2 = Callable[[Mapping[str, Any]], Mapping[str, Any]]

FAILURE_TARGET_NOT_FOUND = "target_not_found"
FAILURE_EXECUTOR_FAILED = "executor_failed"
FAILURE_PAGE_MISMATCH = "page_mismatch"
FAILURE_MAX_RETRY_EXCEEDED = "max_retry_exceeded"
FAILURE_UNSUPPORTED_STEP = "unsupported_step"
FAILURE_VLM_UNCERTAIN = "vlm_uncertain"
FAILURE_CANDIDATE_INVALID = "candidate_invalid"
FAILURE_CANDIDATE_CONFLICT = "candidate_conflict"
FAILURE_COORDINATE_INVALID = "coordinate_invalid"
FAILURE_PAGE_NOT_CHANGED = "page_not_changed"
FAILURE_VALIDATION_FAILED = "validation_failed"
FAILURE_FALLBACK_FAILED = "fallback_failed"
DOCUMENT_TRUTH_SOURCES = {"openapi", "mcp", "clipboard", "export"}
DOCUMENT_TRUTH_TEXT_FIELDS = (
    "text",
    "content",
    "visible_text",
    "raw_text",
    "body_content",
    "body_content_summary",
    "summary",
)
DOCUMENT_TRUTH_LIST_FIELDS = ("texts", "visible_texts", "paragraphs", "blocks")
VLM_HIGH_CONFIDENCE_THRESHOLD = 0.75
CANDIDATE_KIND_SEARCH_INPUT = "search_input"
CANDIDATE_KIND_CHAT_LIST_ITEM = "chat_list_item"
CANDIDATE_KIND_MESSAGE_INPUT = "message_input"
CANDIDATE_KIND_REJECT_SCORE = -100.0


@dataclass(slots=True)
class CandidateSelection:
    candidate: dict[str, Any] | None
    failure_reason: str | None = None


@dataclass(slots=True)
class PlannerLoopV2Result:
    decision: dict[str, Any]
    runtime_state: RuntimeStateV2
    action_result: dict[str, Any] | None = None
    before_ui_state: dict[str, Any] | None = None
    after_ui_state: dict[str, Any] | None = None
    step_outcome: str = RuntimeStatus.RUNNING.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "action_result": self.action_result,
            "before_ui_state": self.before_ui_state,
            "after_ui_state": self.after_ui_state,
            "step_outcome": self.step_outcome,
            "runtime_state": self.runtime_state.to_dict(),
        }


class PlannerLoopV2:
    """Planner Loop v2 for deterministic IM workflow execution."""

    def plan_next_action(
        self,
        runtime_state: RuntimeStateV2 | Mapping[str, Any],
        ui_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        state = _state_from_value(runtime_state)
        state.update_ui_state(ui_state)
        step = state.current_step
        if step is None:
            return ActionDecisionV2.success(
                step_name="workflow_complete",
                reason="all workflow steps are complete",
            ).to_dict()
        return self._decision_for_step(state, step, ui_state).to_dict()

    def run_once(
        self,
        workflow: WorkflowSpec | Mapping[str, Any],
        get_ui_state: GetUIStateV2,
        execute_action: ExecuteActionV2,
        runtime_state: RuntimeStateV2 | Mapping[str, Any] | None = None,
        *,
        get_after_ui_state: GetUIStateV2 | None = None,
    ) -> PlannerLoopV2Result:
        state = _state_or_new(runtime_state, workflow)
        before_ui_state = dict(get_ui_state())
        decision = self.plan_next_action(state, before_ui_state)

        if decision["status"] == DECISION_SUCCESS:
            state.record_action(decision, None, status=RuntimeStatus.SUCCESS.value)
            state.advance_step()
            if state.current_step is not None:
                state.status = RuntimeStatus.RUNNING.value
            return PlannerLoopV2Result(
                decision=decision,
                runtime_state=state,
                before_ui_state=before_ui_state,
                step_outcome=state.status,
            )

        if decision["status"] == DECISION_FAILED:
            if self._handle_planning_retry_if_possible(state, decision):
                return PlannerLoopV2Result(
                    decision=decision,
                    runtime_state=state,
                    before_ui_state=before_ui_state,
                    step_outcome=RuntimeStatus.RUNNING.value,
                )
            state.record_action(
                decision,
                None,
                status=RuntimeStatus.FAILED.value,
                failure_reason=decision.get("failure_reason"),
            )
            return PlannerLoopV2Result(
                decision=decision,
                runtime_state=state,
                before_ui_state=before_ui_state,
                step_outcome=RuntimeStatus.FAILED.value,
            )

        action_payload = action_decision_v2_to_executor_payload(decision)
        action_result = dict(execute_action(action_payload))
        if action_result.get("ok") is not True:
            failure_reason = (
                FAILURE_FALLBACK_FAILED if _is_fallback_decision(decision) else FAILURE_EXECUTOR_FAILED
            )
            self._handle_retry_or_failure(
                state,
                decision,
                action_result,
                failure_reason=failure_reason,
            )
            return PlannerLoopV2Result(
                decision=decision,
                action_result=action_result,
                runtime_state=state,
                before_ui_state=before_ui_state,
                step_outcome=state.status,
            )

        after_ui_state = dict(get_after_ui_state()) if get_after_ui_state is not None else None
        outcome, failure_reason = self._step_outcome(
            state,
            decision,
            action_result,
            before_ui_state,
            after_ui_state,
        )
        if after_ui_state is not None:
            state.update_ui_state(after_ui_state)

        if outcome == RuntimeStatus.SUCCESS.value:
            state.record_action(
                decision,
                action_result,
                status=RuntimeStatus.SUCCESS.value,
            )
            state.advance_step()
            if state.current_step is not None:
                state.status = RuntimeStatus.RUNNING.value
        elif outcome == DECISION_CONTINUE:
            state.record_action(
                decision,
                action_result,
                status=RuntimeStatus.RUNNING.value,
            )
        else:
            self._handle_retry_or_failure(
                state,
                decision,
                action_result,
                failure_reason=failure_reason or FAILURE_PAGE_MISMATCH,
            )

        return PlannerLoopV2Result(
            decision=decision,
            action_result=action_result,
            runtime_state=state,
            before_ui_state=before_ui_state,
            after_ui_state=after_ui_state,
            step_outcome=state.status,
        )

    def run_until_done(
        self,
        workflow: WorkflowSpec | Mapping[str, Any],
        get_ui_state: GetUIStateV2,
        execute_action: ExecuteActionV2,
        runtime_state: RuntimeStateV2 | Mapping[str, Any] | None = None,
        *,
        get_after_ui_state: GetUIStateV2 | None = None,
        max_steps: int = 20,
    ) -> list[PlannerLoopV2Result]:
        state = _state_or_new(runtime_state, workflow)
        results: list[PlannerLoopV2Result] = []
        for _index in range(max_steps):
            if state.status in {RuntimeStatus.SUCCESS.value, RuntimeStatus.FAILED.value}:
                break
            result = self.run_once(
                workflow,
                get_ui_state,
                execute_action,
                state,
                get_after_ui_state=get_after_ui_state,
            )
            results.append(result)
        if state.status == RuntimeStatus.RUNNING.value and len(results) >= max_steps:
            state.mark_failed(FAILURE_MAX_RETRY_EXCEEDED)
        return results

    def _decision_for_step(
        self,
        state: RuntimeStateV2,
        step: WorkflowStep,
        ui_state: Mapping[str, Any],
    ) -> ActionDecisionV2:
        fallback_decision = self._fallback_decision_for_step(state, step, ui_state)
        if fallback_decision is not None:
            return fallback_decision

        if step.step_type == STEP_OPEN_SEARCH:
            selection = self._select_candidate(
                state,
                ui_state,
                target_text="搜索",
                prefer_editable=True,
                expected_kind=CANDIDATE_KIND_SEARCH_INPUT,
            )
            if selection.candidate is not None:
                return _click_candidate_decision(
                    step,
                    selection.candidate,
                    state.retry_count,
                    "open search",
                )
            if selection.failure_reason == FAILURE_CANDIDATE_INVALID:
                return _failed(
                    step,
                    FAILURE_CANDIDATE_INVALID,
                    "search candidate coordinate invalid",
                    state,
                )
            return ActionDecisionV2.hotkey(
                step_name=step.step_name,
                keys=["ctrl", "k"],
                reason="open search by hotkey fallback",
                retry_count=state.retry_count,
                expected_after_state=dict(step.success_condition),
            )

        if step.step_type == STEP_TYPE_KEYWORD:
            selection = self._select_candidate(
                state,
                ui_state,
                target_text=step.target or "search_box",
                prefer_editable=True,
                expected_kind=CANDIDATE_KIND_SEARCH_INPUT,
            )
            candidate = selection.candidate
            if candidate is None:
                return _failed(
                    step,
                    selection.failure_reason or FAILURE_TARGET_NOT_FOUND,
                    "search box candidate not found",
                    state,
                )
            point = _candidate_point(candidate)
            if point is None:
                return _failed(step, FAILURE_CANDIDATE_INVALID, "search box coordinate invalid", state)
            return ActionDecisionV2.type_text(
                step_name=step.step_name,
                target_candidate_id=_candidate_id(candidate),
                text=step.input_text or "",
                reason="type search keyword",
                retry_count=state.retry_count,
                expected_after_state=dict(step.success_condition),
            )

        if step.step_type == STEP_OPEN_CHAT:
            selection = self._select_candidate(
                state,
                ui_state,
                target_text=step.target,
                expected_kind=CANDIDATE_KIND_CHAT_LIST_ITEM,
            )
            candidate = selection.candidate
            if candidate is None:
                search_result_point = _first_search_result_point(ui_state)
                if search_result_point is not None:
                    x, y = search_result_point
                    return ActionDecisionV2.click_point(
                        step_name=step.step_name,
                        x=x,
                        y=y,
                        reason="open first search result fallback",
                        retry_count=state.retry_count,
                        expected_after_state=dict(step.success_condition),
                    )
                return _failed(
                    step,
                    selection.failure_reason or FAILURE_TARGET_NOT_FOUND,
                    "chat candidate not found",
                    state,
                )
            return _click_candidate_decision(step, candidate, state.retry_count, "open target chat")

        if step.step_type == STEP_FOCUS_MESSAGE_INPUT:
            selection = self._select_candidate(
                state,
                ui_state,
                target_text=step.target or "message_input",
                prefer_editable=True,
                expected_kind=CANDIDATE_KIND_MESSAGE_INPUT,
            )
            candidate = selection.candidate
            if candidate is None:
                point = _message_input_fallback_point(state, ui_state)
                if point is not None:
                    x, y = point
                    return ActionDecisionV2.click_point(
                        step_name=step.step_name,
                        x=x,
                        y=y,
                        reason="click bottom input area fallback",
                        retry_count=state.retry_count,
                        expected_after_state=dict(step.success_condition),
                    )
                return _failed(
                    step,
                    selection.failure_reason or FAILURE_TARGET_NOT_FOUND,
                    "message input candidate not found",
                    state,
                )
            return _click_candidate_decision(step, candidate, state.retry_count, "focus message input")

        if step.step_type == STEP_SEND_MESSAGE:
            if _last_decision_was_type_for_step(state, step.step_name):
                return ActionDecisionV2.press_enter(
                    step_name=step.step_name,
                    reason="press enter to send message",
                    retry_count=state.retry_count,
                    expected_after_state=dict(step.success_condition),
                )
            selection = self._select_candidate(
                state,
                ui_state,
                target_text=step.target or "message_input",
                prefer_editable=True,
                expected_kind=CANDIDATE_KIND_MESSAGE_INPUT,
            )
            candidate = selection.candidate
            if candidate is None:
                point = _message_input_fallback_point(state, ui_state)
                if point is not None:
                    x, y = point
                    return ActionDecisionV2.type_text(
                        step_name=step.step_name,
                        x=x,
                        y=y,
                        text=step.input_text or "",
                        reason="type message text with bottom input fallback",
                        retry_count=state.retry_count,
                        expected_after_state=dict(step.success_condition),
                    )
                return _failed(
                    step,
                    selection.failure_reason or FAILURE_TARGET_NOT_FOUND,
                    "message input candidate not found",
                    state,
                )
            point = _candidate_point(candidate)
            if point is None:
                return _failed(step, FAILURE_CANDIDATE_INVALID, "message input coordinate invalid", state)
            x, y = point
            return ActionDecisionV2.type_text(
                step_name=step.step_name,
                target_candidate_id=_candidate_id(candidate),
                x=x,
                y=y,
                text=step.input_text or "",
                reason="type message text",
                retry_count=state.retry_count,
                expected_after_state=dict(step.success_condition),
            )

        if step.step_type == STEP_VERIFY_MESSAGE_SENT:
            if _condition_satisfied(step, ui_state):
                return ActionDecisionV2.success(
                    step_name=step.step_name,
                    reason="message is already visible",
                )
            return ActionDecisionV2.wait(
                step_name=step.step_name,
                seconds=1,
                reason="wait then verify message sent",
                retry_count=state.retry_count,
                expected_after_state=dict(step.success_condition),
            )

        return _failed(step, FAILURE_UNSUPPORTED_STEP, f"unsupported step_type: {step.step_type}", state)

    def _fallback_decision_for_step(
        self,
        state: RuntimeStateV2,
        step: WorkflowStep,
        ui_state: Mapping[str, Any],
    ) -> ActionDecisionV2 | None:
        if state.retry_count <= 0 or not state.failure_reason:
            return None

        if state.failure_reason == FAILURE_PAGE_NOT_CHANGED:
            return ActionDecisionV2.wait(
                step_name=step.step_name,
                seconds=1,
                reason="wait after page unchanged fallback",
                retry_count=state.retry_count,
                expected_after_state=dict(step.success_condition),
            )

        if step.step_type == STEP_TYPE_KEYWORD and state.failure_reason in {
            FAILURE_VALIDATION_FAILED,
            FAILURE_PAGE_MISMATCH,
            FAILURE_TARGET_NOT_FOUND,
        }:
            selection = self._select_candidate(
                state,
                ui_state,
                target_text=step.target or "search_box",
                prefer_editable=True,
                expected_kind=CANDIDATE_KIND_SEARCH_INPUT,
            )
            if selection.candidate is not None:
                return _click_candidate_decision(
                    step,
                    selection.candidate,
                    state.retry_count,
                    "refocus search box fallback",
                )
            return ActionDecisionV2.hotkey(
                step_name=step.step_name,
                keys=["ctrl", "k"],
                reason="reopen search by hotkey fallback",
                retry_count=state.retry_count,
                expected_after_state=dict(step.success_condition),
            )

        if step.step_type == STEP_OPEN_CHAT and state.failure_reason == FAILURE_TARGET_NOT_FOUND:
            return ActionDecisionV2.hotkey(
                step_name=step.step_name,
                keys=["ctrl", "k"],
                reason="retry search fallback",
                retry_count=state.retry_count,
                expected_after_state=dict(step.success_condition),
            )

        if step.step_type == STEP_OPEN_CHAT and state.failure_reason == FAILURE_PAGE_MISMATCH:
            return ActionDecisionV2.hotkey(
                step_name=step.step_name,
                keys=["alt", "left"],
                reason="wrong page back fallback",
                retry_count=state.retry_count,
                expected_after_state=dict(step.success_condition),
            )

        if step.step_type == STEP_FOCUS_MESSAGE_INPUT and state.failure_reason in {
            FAILURE_TARGET_NOT_FOUND,
            FAILURE_VALIDATION_FAILED,
        }:
            point = _message_input_fallback_point(state, ui_state)
            if point is None:
                return ActionDecisionV2.wait(
                    step_name=step.step_name,
                    seconds=1,
                    reason="wait before focus message input fallback",
                    retry_count=state.retry_count,
                    expected_after_state=dict(step.success_condition),
                )
            x, y = point
            return ActionDecisionV2.click_point(
                step_name=step.step_name,
                x=x,
                y=y,
                reason="click bottom input area fallback",
                retry_count=state.retry_count,
                expected_after_state=dict(step.success_condition),
            )

        if step.step_type == STEP_SEND_MESSAGE and state.failure_reason in {
            FAILURE_TARGET_NOT_FOUND,
            FAILURE_VALIDATION_FAILED,
            FAILURE_PAGE_MISMATCH,
        }:
            if _last_decision_was_type_for_step(state, step.step_name):
                return ActionDecisionV2.press_enter(
                    step_name=step.step_name,
                    reason="press enter to send fallback",
                    retry_count=state.retry_count,
                    expected_after_state=dict(step.success_condition),
                )
            selection = self._select_candidate(
                state,
                ui_state,
                target_text=step.target or "message_input",
                prefer_editable=True,
                expected_kind=CANDIDATE_KIND_MESSAGE_INPUT,
            )
            if selection.candidate is not None:
                return _click_candidate_decision(
                    step,
                    selection.candidate,
                    state.retry_count,
                    "refocus message input fallback",
                )
            point = _message_input_fallback_point(state, ui_state)
            if point is not None:
                x, y = point
                return ActionDecisionV2.type_text(
                    step_name=step.step_name,
                    x=x,
                    y=y,
                    text=step.input_text or "",
                    reason="type message text with bottom input fallback",
                    retry_count=state.retry_count,
                    expected_after_state=dict(step.success_condition),
                )
            return ActionDecisionV2.press_enter(
                step_name=step.step_name,
                reason="press enter to send fallback",
                retry_count=state.retry_count,
                expected_after_state=dict(step.success_condition),
            )

        if step.step_type == STEP_VERIFY_MESSAGE_SENT:
            return ActionDecisionV2.wait(
                step_name=step.step_name,
                seconds=1,
                reason="wait then check again fallback",
                retry_count=state.retry_count,
                expected_after_state=dict(step.success_condition),
            )

        return None

    def _select_candidate(
        self,
        state: RuntimeStateV2,
        ui_state: Mapping[str, Any],
        *,
        target_text: str | None,
        prefer_editable: bool = False,
        expected_kind: str | None = None,
    ) -> CandidateSelection:
        candidates = _candidate_list(ui_state)
        if not candidates:
            return CandidateSelection(None, FAILURE_TARGET_NOT_FOUND)

        rule_candidate = _best_candidate(
            candidates,
            ui_state=ui_state,
            target_text=target_text,
            prefer_editable=prefer_editable,
            expected_kind=expected_kind,
        )
        structured = _vlm_structured(ui_state)
        confidence = _vlm_confidence(ui_state)
        if structured is None:
            return CandidateSelection(
                rule_candidate,
                None if rule_candidate is not None else FAILURE_TARGET_NOT_FOUND,
            )

        target_candidate_id = _vlm_target_candidate_id(structured)
        reranked_candidate_ids = _vlm_reranked_candidate_ids(structured)
        if _vlm_is_uncertain(structured):
            _record_candidate_note(
                state,
                rule_candidate=rule_candidate,
                vlm_candidate_id=target_candidate_id,
                reason=FAILURE_VLM_UNCERTAIN,
                vlm_confidence=confidence,
            )
            return CandidateSelection(
                rule_candidate,
                None if rule_candidate is not None else FAILURE_VLM_UNCERTAIN,
            )

        candidate_by_id = {
            candidate_id: candidate
            for candidate in candidates
            if (candidate_id := _candidate_id(candidate))
        }
        vlm_candidate = candidate_by_id.get(target_candidate_id) if target_candidate_id else None
        if target_candidate_id and vlm_candidate is None and rule_candidate is None:
            return CandidateSelection(None, FAILURE_TARGET_NOT_FOUND)

        if vlm_candidate is not None and _candidate_point(vlm_candidate) is None:
            _record_candidate_note(
                state,
                rule_candidate=rule_candidate,
                vlm_candidate_id=target_candidate_id,
                reason=FAILURE_CANDIDATE_INVALID,
                vlm_confidence=confidence,
            )
            return CandidateSelection(
                rule_candidate,
                None if rule_candidate is not None else FAILURE_CANDIDATE_INVALID,
            )

        if vlm_candidate is not None and not _candidate_coordinate_is_usable(vlm_candidate, ui_state):
            _record_candidate_note(
                state,
                rule_candidate=rule_candidate,
                vlm_candidate_id=target_candidate_id,
                reason=FAILURE_COORDINATE_INVALID,
                vlm_confidence=confidence,
            )
            return CandidateSelection(
                rule_candidate,
                None if rule_candidate is not None else FAILURE_COORDINATE_INVALID,
            )

        if vlm_candidate is not None and _candidate_kind_score(vlm_candidate, expected_kind) == CANDIDATE_KIND_REJECT_SCORE:
            _record_candidate_note(
                state,
                rule_candidate=rule_candidate,
                vlm_candidate_id=target_candidate_id,
                reason=FAILURE_CANDIDATE_CONFLICT,
                vlm_confidence=confidence,
            )
            return CandidateSelection(
                rule_candidate,
                None if rule_candidate is not None else FAILURE_TARGET_NOT_FOUND,
            )

        if (
            expected_kind == CANDIDATE_KIND_CHAT_LIST_ITEM
            and vlm_candidate is not None
            and rule_candidate is not None
            and _candidate_exact_text_matches(rule_candidate, target_text)
            and not _candidate_exact_text_matches(vlm_candidate, target_text)
        ):
            _record_candidate_note(
                state,
                rule_candidate=rule_candidate,
                vlm_candidate_id=target_candidate_id,
                reason=FAILURE_CANDIDATE_CONFLICT,
                vlm_confidence=confidence,
            )
            return CandidateSelection(rule_candidate)

        if (
            vlm_candidate is not None
            and _is_high_confidence(confidence)
        ):
            if _candidate_id(rule_candidate) and _candidate_id(rule_candidate) != target_candidate_id:
                _record_candidate_note(
                    state,
                    rule_candidate=rule_candidate,
                    vlm_candidate_id=target_candidate_id,
                    reason=FAILURE_CANDIDATE_CONFLICT,
                    vlm_confidence=confidence,
                )
            return CandidateSelection(vlm_candidate)

        selected = _best_candidate(
            candidates,
            ui_state=ui_state,
            target_text=target_text,
            prefer_editable=prefer_editable,
            expected_kind=expected_kind,
            vlm_candidate_id=target_candidate_id if vlm_candidate is not None else None,
            vlm_reranked_candidate_ids=reranked_candidate_ids,
        )
        if vlm_candidate is not None and _candidate_id(rule_candidate) != _candidate_id(vlm_candidate):
            _record_candidate_note(
                state,
                rule_candidate=rule_candidate,
                vlm_candidate_id=target_candidate_id,
                reason=FAILURE_CANDIDATE_CONFLICT,
                vlm_confidence=confidence,
            )
        return CandidateSelection(
            selected,
            None if selected is not None else FAILURE_TARGET_NOT_FOUND,
        )

    def _step_outcome(
        self,
        state: RuntimeStateV2,
        decision: Mapping[str, Any],
        action_result: Mapping[str, Any],
        before_ui_state: Mapping[str, Any],
        after_ui_state: Mapping[str, Any] | None,
    ) -> tuple[str, str | None]:
        step = state.current_step
        if step is None:
            return RuntimeStatus.SUCCESS.value, None
        if after_ui_state is None:
            return RuntimeStatus.SUCCESS.value, None

        if step.step_type == STEP_SEND_MESSAGE and decision.get("action_type") == ACTION_TYPE:
            if _validator_satisfied(action_result):
                return RuntimeStatus.SUCCESS.value, None
            if _contains_text(after_ui_state, step.input_text):
                return DECISION_CONTINUE, None
            return RuntimeStatus.FAILED.value, _failure_for_unsatisfied_step(
                step,
                before_ui_state,
                after_ui_state,
            )

        if step.step_type == STEP_TYPE_KEYWORD:
            if _type_keyword_satisfied(step, before_ui_state, after_ui_state, action_result):
                return RuntimeStatus.SUCCESS.value, None
            return RuntimeStatus.FAILED.value, _failure_for_unsatisfied_step(
                step,
                before_ui_state,
                after_ui_state,
            )

        if _condition_satisfied(step, after_ui_state, action_result):
            return RuntimeStatus.SUCCESS.value, None

        if _is_fallback_decision(decision):
            if _ui_state_looks_same(before_ui_state, after_ui_state):
                return RuntimeStatus.FAILED.value, FAILURE_FALLBACK_FAILED
            if _fallback_can_continue_without_step_success(step, decision):
                return DECISION_CONTINUE, None

        return RuntimeStatus.FAILED.value, _failure_for_unsatisfied_step(
            step,
            before_ui_state,
            after_ui_state,
        )

    def _handle_planning_retry_if_possible(
        self,
        state: RuntimeStateV2,
        decision: Mapping[str, Any],
    ) -> bool:
        step = state.current_step
        if step is None:
            return False
        failure_reason = str(decision.get("failure_reason") or "")
        retryable = (
            step.step_type == STEP_OPEN_CHAT
            and failure_reason == FAILURE_TARGET_NOT_FOUND
            and "retry_search" in step.fallback
        )
        if not retryable or state.retry_count >= step.max_retry:
            return False
        state.record_action(
            decision,
            None,
            status=RuntimeStatus.RUNNING.value,
            failure_reason=failure_reason,
        )
        state.increment_retry()
        return True

    def _handle_retry_or_failure(
        self,
        state: RuntimeStateV2,
        decision: Mapping[str, Any],
        action_result: Mapping[str, Any] | None,
        *,
        failure_reason: str,
    ) -> None:
        step = state.current_step
        max_retry = step.max_retry if step is not None else 0
        if state.retry_count < max_retry:
            state.record_action(
                decision,
                action_result,
                status=RuntimeStatus.RUNNING.value,
                failure_reason=failure_reason,
            )
            state.increment_retry()
            return
        state.record_action(
            decision,
            action_result,
            status=RuntimeStatus.FAILED.value,
            failure_reason=FAILURE_MAX_RETRY_EXCEEDED,
        )
        state.mark_failed(FAILURE_MAX_RETRY_EXCEEDED)


def plan_next_workflow_action(
    runtime_state: RuntimeStateV2 | Mapping[str, Any],
    ui_state: Mapping[str, Any],
) -> dict[str, Any]:
    return PlannerLoopV2().plan_next_action(runtime_state, ui_state)


def run_workflow_once(
    workflow: WorkflowSpec | Mapping[str, Any],
    get_ui_state: GetUIStateV2,
    execute_action: ExecuteActionV2,
    runtime_state: RuntimeStateV2 | Mapping[str, Any] | None = None,
    *,
    get_after_ui_state: GetUIStateV2 | None = None,
) -> PlannerLoopV2Result:
    return PlannerLoopV2().run_once(
        workflow,
        get_ui_state,
        execute_action,
        runtime_state,
        get_after_ui_state=get_after_ui_state,
    )


def run_workflow_until_done(
    workflow: WorkflowSpec | Mapping[str, Any],
    get_ui_state: GetUIStateV2,
    execute_action: ExecuteActionV2,
    runtime_state: RuntimeStateV2 | Mapping[str, Any] | None = None,
    *,
    get_after_ui_state: GetUIStateV2 | None = None,
    max_steps: int = 20,
) -> list[PlannerLoopV2Result]:
    return PlannerLoopV2().run_until_done(
        workflow,
        get_ui_state,
        execute_action,
        runtime_state,
        get_after_ui_state=get_after_ui_state,
        max_steps=max_steps,
    )


def _state_from_value(value: RuntimeStateV2 | Mapping[str, Any]) -> RuntimeStateV2:
    return parse_runtime_state_v2(value)


def _state_or_new(
    value: RuntimeStateV2 | Mapping[str, Any] | None,
    workflow: WorkflowSpec | Mapping[str, Any],
) -> RuntimeStateV2:
    if value is not None:
        return _state_from_value(value)
    return create_runtime_state_v2(parse_workflow(workflow))


def _candidate_list(ui_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = ui_state.get("candidates", [])
    if not isinstance(raw_candidates, list):
        return []
    return [dict(candidate) for candidate in raw_candidates if isinstance(candidate, Mapping)]


def _best_candidate(
    candidates: list[dict[str, Any]],
    *,
    ui_state: Mapping[str, Any],
    target_text: str | None,
    prefer_editable: bool,
    expected_kind: str | None = None,
    vlm_candidate_id: str | None = None,
    vlm_reranked_candidate_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    rerank_boosts = {
        candidate_id: max(1.0, 6.0 - index)
        for index, candidate_id in enumerate(vlm_reranked_candidate_ids or [])
        if candidate_id
    }
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        if _candidate_point(candidate) is None:
            continue
        if not _candidate_coordinate_is_usable(candidate, ui_state):
            continue
        score = _candidate_score(
            candidate,
            target_text=target_text,
            prefer_editable=prefer_editable,
            expected_kind=expected_kind,
        )
        if vlm_candidate_id and _candidate_id(candidate) == vlm_candidate_id:
            score += 8.0
        score += rerank_boosts.get(_candidate_id(candidate), 0.0)
        if score > 0:
            scored.append((score, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _candidate_score(
    candidate: Mapping[str, Any],
    *,
    target_text: str | None,
    prefer_editable: bool,
    expected_kind: str | None = None,
) -> float:
    score = float(candidate.get("confidence") or 0.0)
    text = str(candidate.get("text") or "")
    role = str(candidate.get("role") or "")
    candidate_id = str(candidate.get("id") or "")
    normalized_target = _normalize_text(target_text)
    normalized_text = _normalize_text(text)
    normalized_role = role.casefold()
    normalized_id = candidate_id.casefold()
    kind_score = _candidate_kind_score(candidate, expected_kind)
    if kind_score == CANDIDATE_KIND_REJECT_SCORE:
        return CANDIDATE_KIND_REJECT_SCORE

    if normalized_target:
        if normalized_text == normalized_target:
            score += 20
        elif normalized_target in normalized_text:
            score += 14
        elif normalized_text and len(normalized_text) >= 2 and normalized_text in normalized_target:
            score += 6
        elif expected_kind == CANDIDATE_KIND_CHAT_LIST_ITEM:
            return CANDIDATE_KIND_REJECT_SCORE
        elif prefer_editable:
            score += 1
        else:
            score -= 5

    if prefer_editable and _is_editable(candidate):
        score += 12
    if any(keyword in normalized_role for keyword in ("edit", "textbox", "input")):
        score += 3
    if any(keyword in normalized_id for keyword in ("search", "input", "message")):
        score += 2
    if candidate.get("clickable") is True:
        score += 1
    score += kind_score
    return score


def _candidate_kind_score(candidate: Mapping[str, Any], expected_kind: str | None) -> float:
    if expected_kind is None:
        return 0.0
    if expected_kind == CANDIDATE_KIND_SEARCH_INPUT:
        if _is_browser_shell_candidate(candidate):
            return CANDIDATE_KIND_REJECT_SCORE
        if _is_search_input_candidate(candidate):
            return 12.0
        return CANDIDATE_KIND_REJECT_SCORE
    if expected_kind == CANDIDATE_KIND_CHAT_LIST_ITEM:
        if (
            _is_browser_shell_candidate(candidate)
            or _is_plain_text_candidate(candidate)
            or _is_search_input_candidate(candidate)
        ):
            return CANDIDATE_KIND_REJECT_SCORE
        if _is_chat_list_candidate(candidate):
            return 14.0
        if candidate.get("clickable") is True and not _is_editable(candidate):
            return 3.0
        return -12.0
    if expected_kind == CANDIDATE_KIND_MESSAGE_INPUT:
        if _is_browser_shell_candidate(candidate) or _is_plain_text_candidate(candidate):
            return CANDIDATE_KIND_REJECT_SCORE
        if _is_message_input_candidate(candidate):
            return 14.0
        if _is_editable(candidate) and not _is_search_input_candidate(candidate):
            return 6.0
        return CANDIDATE_KIND_REJECT_SCORE
    return 0.0


def _candidate_exact_text_matches(candidate: Mapping[str, Any], target_text: str | None) -> bool:
    target = _normalize_text(target_text)
    text = _normalize_text(candidate.get("text"))
    return bool(target and text == target)


def _is_browser_shell_candidate(candidate: Mapping[str, Any]) -> bool:
    text = _candidate_blob(candidate)
    shell_keywords = (
        "browser",
        "chrome",
        "chromium",
        "edge",
        "address",
        "omnibox",
        "url",
        "titlebar",
        "toolbar",
        "tabbar",
        "tab bar",
        "minimize",
        "maximize",
        "close",
        "关闭",
        "最小化",
        "最大化",
        "地址栏",
    )
    return any(keyword in text for keyword in shell_keywords)


def _is_search_input_candidate(candidate: Mapping[str, Any]) -> bool:
    text = _candidate_blob(candidate)
    if _is_browser_shell_candidate(candidate):
        return False
    return (
        "larksearchbox" in text
        or "searchbox" in text
        or "global_search" in text
        or "search" in text
        or "搜索" in text
        or (_is_editable(candidate) and "message" not in text and "chat" not in text)
    )


def _is_chat_list_candidate(candidate: Mapping[str, Any]) -> bool:
    text = _candidate_blob(candidate)
    return any(
        keyword in text
        for keyword in (
            "larkchatlistitem",
            "chatlist",
            "conversation",
            "session",
            "chat_row",
            "chatrow",
            "群聊",
        )
    )


def _is_message_input_candidate(candidate: Mapping[str, Any]) -> bool:
    text = _candidate_blob(candidate)
    return _is_editable(candidate) and any(
        keyword in text
        for keyword in (
            "message",
            "chat_input",
            "chatinput",
            "composer",
            "editor",
            "input",
            "消息",
            "输入",
        )
    )


def _is_plain_text_candidate(candidate: Mapping[str, Any]) -> bool:
    role = str(candidate.get("role") or "").casefold()
    source = str(candidate.get("source") or "").casefold()
    return (
        not candidate.get("clickable")
        and not candidate.get("editable")
        and (source == "ocr" or role in {"text", "statictext", "label"})
    )


def _vlm_candidate(
    ui_state: Mapping[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    structured = _vlm_structured(ui_state)
    if not structured or structured.get("uncertain") is True:
        return None
    target_id = str(structured.get("target_candidate_id") or "").strip()
    if not target_id:
        return None
    for candidate in candidates:
        if _candidate_id(candidate) == target_id:
            return candidate
    return None


def _vlm_structured(ui_state: Mapping[str, Any]) -> Mapping[str, Any] | None:
    vlm_result = ui_state.get("vlm_result")
    if not isinstance(vlm_result, Mapping):
        return None
    structured = vlm_result.get("structured")
    if not isinstance(structured, Mapping):
        return None
    return structured


def _vlm_target_candidate_id(structured: Mapping[str, Any]) -> str:
    return str(structured.get("target_candidate_id") or "").strip()


def _vlm_reranked_candidate_ids(structured: Mapping[str, Any]) -> list[str]:
    raw_ids = structured.get("reranked_candidate_ids", [])
    if not isinstance(raw_ids, list):
        return []
    return [str(candidate_id).strip() for candidate_id in raw_ids if str(candidate_id).strip()]


def _vlm_is_uncertain(structured: Mapping[str, Any]) -> bool:
    if structured.get("uncertain") is True:
        return True
    return not _vlm_target_candidate_id(structured) and not _vlm_reranked_candidate_ids(structured)


def _vlm_confidence(ui_state: Mapping[str, Any]) -> int | float | None:
    structured = _vlm_structured(ui_state)
    if structured is None:
        return None
    value = structured.get("confidence")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _is_high_confidence(value: int | float | None) -> bool:
    return value is not None and value >= VLM_HIGH_CONFIDENCE_THRESHOLD


def _record_candidate_note(
    state: RuntimeStateV2,
    *,
    rule_candidate: Mapping[str, Any] | None,
    vlm_candidate_id: str | None,
    reason: str,
    vlm_confidence: int | float | None,
) -> None:
    state.record_conflict(
        rule_candidate_id=_candidate_id(rule_candidate),
        vlm_candidate_id=vlm_candidate_id or None,
        reason=reason,
        vlm_confidence=vlm_confidence,
    )


def _click_candidate_decision(
    step: WorkflowStep,
    candidate: Mapping[str, Any],
    retry_count: int,
    reason: str,
) -> ActionDecisionV2:
    point = _candidate_point(candidate)
    if point is None:
        raise ValueError("candidate must have a valid point before creating click decision")
    x, y = point
    return ActionDecisionV2.click_candidate(
        step_name=step.step_name,
        target_candidate_id=_candidate_id(candidate),
        x=x,
        y=y,
        reason=reason,
        retry_count=retry_count,
        expected_after_state=dict(step.success_condition),
    )


def _failed(
    step: WorkflowStep,
    failure_reason: str,
    reason: str,
    state: RuntimeStateV2,
) -> ActionDecisionV2:
    return ActionDecisionV2.failed(
        failure_reason,
        reason,
        step_name=step.step_name,
        retry_count=state.retry_count,
    )


def _condition_satisfied(
    step: WorkflowStep,
    ui_state: Mapping[str, Any],
    action_result: Mapping[str, Any] | None = None,
) -> bool:
    condition = step.success_condition
    if not condition:
        return True
    if _validator_satisfied(action_result):
        return True
    if condition.get("has_search_box") is True and (
        _has_search_box(ui_state)
        or _semantic_signal(ui_state, "search_active", "search_available")
        or _semantic_view_is(ui_state, "search", "results")
    ):
        return True
    chat_expected = condition.get("chat_name_visible")
    if chat_expected is not None:
        if (
            _contains_text(ui_state, chat_expected)
            and (
                _has_editable_candidate(ui_state)
                or _semantic_signal(ui_state, "target_active", "input_available", "detail_open")
                or _semantic_view_is(ui_state, "detail", "chat")
            )
        ):
            return True
        if _semantic_entity_matches(ui_state, chat_expected) or _semantic_goal_satisfied(
            ui_state,
            "chat_name_visible",
        ):
            return True
    for field_name in (
        "text_present",
        "message_sent",
        "message_visible",
    ):
        expected = condition.get(field_name)
        if expected is not None and (
            _contains_text(ui_state, expected)
            or _semantic_entity_matches(ui_state, expected)
            or _semantic_goal_satisfied(ui_state, field_name)
        ):
            return True
    if condition.get("message_input_focused") is True and (
        _has_editable_candidate(ui_state) or _semantic_signal(ui_state, "input_available", "target_active")
    ):
        return True
    if condition.get("message_input_focused") is True and _action_result_ok(action_result):
        return _chat_context_active(ui_state)
    return False


def _type_keyword_satisfied(
    step: WorkflowStep,
    before_ui_state: Mapping[str, Any],
    after_ui_state: Mapping[str, Any],
    action_result: Mapping[str, Any] | None = None,
) -> bool:
    if _validator_satisfied(action_result) and _search_context_active(after_ui_state):
        return True

    expected = step.success_condition.get("text_present") or step.input_text
    if not _search_context_active(after_ui_state):
        return False
    if expected is not None and _contains_text(after_ui_state, expected):
        return True
    return _search_results_changed(before_ui_state, after_ui_state)


def _search_context_active(ui_state: Mapping[str, Any]) -> bool:
    return (
        _has_search_box(ui_state)
        or _semantic_signal(ui_state, "search_active", "results_available")
        or _semantic_view_is(ui_state, "search", "results")
    )


def _validator_satisfied(action_result: Mapping[str, Any] | None) -> bool:
    if not isinstance(action_result, Mapping):
        return False
    if action_result.get("validated") is True or action_result.get("validation_passed") is True:
        return True
    for field_name in ("validator_result", "validation_result", "validator"):
        value = action_result.get(field_name)
        if not isinstance(value, Mapping):
            continue
        if value.get("ok") is True or value.get("passed") is True or value.get("success") is True:
            return True
    return False


def _action_result_ok(action_result: Mapping[str, Any] | None) -> bool:
    return isinstance(action_result, Mapping) and action_result.get("ok") is True


def _semantic_signal(ui_state: Mapping[str, Any], *signal_names: str) -> bool:
    for semantic in _semantic_mappings(ui_state):
        signals = semantic.get("signals")
        if isinstance(signals, Mapping):
            for signal_name in signal_names:
                if signals.get(signal_name) is True:
                    return True
        for signal_name in signal_names:
            if semantic.get(signal_name) is True:
                return True
    return False


def _semantic_view_is(ui_state: Mapping[str, Any], *view_names: str) -> bool:
    expected = {name.casefold() for name in view_names}
    for semantic in _semantic_mappings(ui_state):
        view = str(semantic.get("view") or "").casefold()
        domain = str(semantic.get("domain") or "").casefold()
        if view in expected or domain in expected:
            return True
    return False


def _semantic_entity_matches(ui_state: Mapping[str, Any], expected_text: Any) -> bool:
    needle = _normalize_text(expected_text)
    if not needle:
        return False
    for semantic in _semantic_mappings(ui_state):
        entities = semantic.get("entities")
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if not isinstance(entity, Mapping):
                continue
            if entity.get("matched_target") is True:
                return True
            if needle in _normalize_text(entity.get("name")) or needle in _normalize_text(entity.get("text")):
                return True
    return False


def _semantic_goal_satisfied(ui_state: Mapping[str, Any], field_name: str) -> bool:
    for semantic in _semantic_mappings(ui_state):
        goal_progress = str(semantic.get("goal_progress") or "").casefold()
        if goal_progress == "achieved":
            return True
        signals = semantic.get("signals")
        if isinstance(signals, Mapping):
            if field_name == "chat_name_visible" and signals.get("target_active") is True:
                return True
            if field_name in {"message_sent", "message_visible"} and (
                signals.get("target_visible") is True
                or signals.get("message_visible") is True
                or signals.get("message_sent") is True
            ):
                return True
        if field_name == "chat_name_visible" and semantic.get("target_active") is True:
            return True
    return False


def _semantic_mappings(ui_state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    page_summary = ui_state.get("page_summary")
    if isinstance(page_summary, Mapping):
        semantic_summary = page_summary.get("semantic_summary")
        if isinstance(semantic_summary, Mapping):
            values.append(semantic_summary)
    structured = _vlm_structured(ui_state)
    if isinstance(structured, Mapping):
        page_semantics = structured.get("page_semantics")
        if isinstance(page_semantics, Mapping):
            values.append(page_semantics)
    return values


def _search_results_changed(
    before_ui_state: Mapping[str, Any],
    after_ui_state: Mapping[str, Any],
) -> bool:
    before_summary = before_ui_state.get("page_summary")
    after_summary = after_ui_state.get("page_summary")
    if not isinstance(before_summary, Mapping) or not isinstance(after_summary, Mapping):
        return False
    before_texts = tuple(str(text) for text in before_summary.get("top_texts", []) if str(text).strip())
    after_texts = tuple(str(text) for text in after_summary.get("top_texts", []) if str(text).strip())
    before_count = before_summary.get("candidate_count")
    after_count = after_summary.get("candidate_count")
    return bool(after_texts and (after_texts != before_texts or after_count != before_count))


def _failure_for_unsatisfied_step(
    step: WorkflowStep,
    before_ui_state: Mapping[str, Any],
    after_ui_state: Mapping[str, Any],
) -> str:
    if _ui_state_looks_same(before_ui_state, after_ui_state):
        return FAILURE_PAGE_NOT_CHANGED
    if step.step_type == STEP_OPEN_CHAT:
        return FAILURE_PAGE_MISMATCH
    return FAILURE_VALIDATION_FAILED


def _is_fallback_decision(decision: Mapping[str, Any]) -> bool:
    return "fallback" in str(decision.get("reason") or "").casefold()


def _fallback_can_continue_without_step_success(
    step: WorkflowStep,
    decision: Mapping[str, Any],
) -> bool:
    reason = str(decision.get("reason") or "").casefold()
    if step.step_type == STEP_OPEN_SEARCH:
        return False
    return any(
        marker in reason
        for marker in (
            "refocus",
            "retry search",
            "wrong page back",
            "bottom input",
            "wait after page unchanged",
            "wait then check",
        )
    )


def _ui_state_looks_same(
    before_ui_state: Mapping[str, Any],
    after_ui_state: Mapping[str, Any],
) -> bool:
    return _ui_state_signature(before_ui_state) == _ui_state_signature(after_ui_state)


def _ui_state_signature(ui_state: Mapping[str, Any]) -> tuple[Any, ...]:
    page_summary = ui_state.get("page_summary")
    if not isinstance(page_summary, Mapping):
        page_summary = {}
    candidates = []
    for candidate in _candidate_list(ui_state):
        candidates.append(
            (
                _candidate_id(candidate),
                _normalize_text(candidate.get("text")),
                str(candidate.get("role") or "").casefold(),
                bool(candidate.get("editable")),
                bool(candidate.get("clickable")),
            )
        )
    return (
        tuple(str(text) for text in page_summary.get("top_texts", []) if str(text).strip()),
        bool(page_summary.get("has_search_box")),
        bool(page_summary.get("has_modal")),
        tuple(candidates),
    )


def _has_search_box(ui_state: Mapping[str, Any]) -> bool:
    page_summary = ui_state.get("page_summary", {})
    if isinstance(page_summary, Mapping) and page_summary.get("has_search_box") is True:
        return True
    return any(_is_editable(candidate) for candidate in _candidate_list(ui_state))


def _has_editable_candidate(ui_state: Mapping[str, Any]) -> bool:
    return any(_is_editable(candidate) for candidate in _candidate_list(ui_state))


def _contains_text(ui_state: Mapping[str, Any], target_text: Any) -> bool:
    needle = _normalize_text(target_text)
    if not needle:
        return False
    truth_texts = _document_truth_texts(ui_state)
    if truth_texts:
        return any(needle in _normalize_text(text) for text in truth_texts)
    for text in _visible_texts(ui_state):
        if needle in _normalize_text(text):
            return True
    return False


def _visible_texts(ui_state: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    texts.extend(_document_truth_texts(ui_state))
    page_summary = ui_state.get("page_summary")
    if isinstance(page_summary, Mapping):
        top_texts = page_summary.get("top_texts", [])
        if isinstance(top_texts, list):
            texts.extend(str(text).strip() for text in top_texts if str(text).strip())
        semantic_summary = page_summary.get("semantic_summary")
        if isinstance(semantic_summary, Mapping):
            _extend_semantic_texts(texts, semantic_summary)
    structured = _vlm_structured(ui_state)
    if isinstance(structured, Mapping):
        page_semantics = structured.get("page_semantics")
        if isinstance(page_semantics, Mapping):
            _extend_semantic_texts(texts, page_semantics)
    document_region_summary = ui_state.get("document_region_summary")
    if isinstance(document_region_summary, Mapping):
        _extend_semantic_texts(texts, document_region_summary)
    for candidate in _candidate_list(ui_state):
        if str(candidate.get("source") or "").casefold() == "ocr":
            continue
        text = str(candidate.get("text") or "").strip()
        if text:
            texts.append(text)
    return texts


def _document_truth_texts(ui_state: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    for item in _document_truth_items(ui_state.get("document_truth")):
        source = str(item.get("source") or "").strip().casefold()
        if source not in DOCUMENT_TRUTH_SOURCES:
            continue
        for field_name in DOCUMENT_TRUTH_TEXT_FIELDS:
            _extend_semantic_texts(texts, item.get(field_name))
        for field_name in DOCUMENT_TRUTH_LIST_FIELDS:
            _extend_semantic_texts(texts, item.get(field_name))
        data = item.get("data")
        if isinstance(data, Mapping):
            for field_name in DOCUMENT_TRUTH_TEXT_FIELDS + DOCUMENT_TRUTH_LIST_FIELDS:
                _extend_semantic_texts(texts, data.get(field_name))
    return texts


def _document_truth_items(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        items = value.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, Mapping)]
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _extend_semantic_texts(texts: list[str], value: Any) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _extend_semantic_texts(texts, child)
        return
    if isinstance(value, list):
        for child in value:
            _extend_semantic_texts(texts, child)
        return
    text = str(value).strip()
    if text:
        texts.append(text)


def _last_decision_was_type_for_step(state: RuntimeStateV2, step_name: str) -> bool:
    decision = state.last_action_decision
    if not isinstance(decision, Mapping):
        return False
    return decision.get("step_name") == step_name and decision.get("action_type") == ACTION_TYPE


def _candidate_point(candidate: Mapping[str, Any]) -> tuple[int, int] | None:
    center = candidate.get("center")
    if _is_int_pair(center):
        return int(center[0]), int(center[1])
    bbox = candidate.get("bbox")
    if (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, int) and not isinstance(value, bool) for value in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    ):
        return (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
    return None


def _is_int_pair(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _is_editable(candidate: Mapping[str, Any]) -> bool:
    role = str(candidate.get("role") or "").casefold()
    return candidate.get("editable") is True or any(
        keyword in role for keyword in ("edit", "textbox", "input")
    )


def _candidate_coordinate_is_usable(
    candidate: Mapping[str, Any],
    ui_state: Mapping[str, Any],
) -> bool:
    point = _candidate_point(candidate)
    if point is None:
        return False
    bounds = _coordinate_bounds(ui_state)
    if bounds is None:
        return point[0] >= 0 and point[1] >= 0
    left, top, right, bottom = bounds
    return left <= point[0] <= right and top <= point[1] <= bottom


def _coordinate_bounds(ui_state: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    screen_meta = ui_state.get("screen_meta")
    if not isinstance(screen_meta, Mapping):
        return None

    screenshot_size = screen_meta.get("screenshot_size")
    screen_origin = screen_meta.get("screen_origin")
    if _is_int_pair(screenshot_size) and _is_int_pair(screen_origin):
        left, top = int(screen_origin[0]), int(screen_origin[1])
        return left, top, left + int(screenshot_size[0]), top + int(screenshot_size[1])

    window_rect = screen_meta.get("window_rect")
    if (
        isinstance(window_rect, list)
        and len(window_rect) == 4
        and all(isinstance(value, int) and not isinstance(value, bool) for value in window_rect)
        and window_rect[2] > window_rect[0]
        and window_rect[3] > window_rect[1]
    ):
        return int(window_rect[0]), int(window_rect[1]), int(window_rect[2]), int(window_rect[3])

    if _is_int_pair(screenshot_size):
        return 0, 0, int(screenshot_size[0]), int(screenshot_size[1])

    screen_resolution = screen_meta.get("screen_resolution")
    if _is_int_pair(screen_resolution) and _is_int_pair(screen_origin):
        left, top = int(screen_origin[0]), int(screen_origin[1])
        return left, top, left + int(screen_resolution[0]), top + int(screen_resolution[1])

    if _is_int_pair(screen_resolution):
        return 0, 0, int(screen_resolution[0]), int(screen_resolution[1])

    width = screen_meta.get("width")
    height = screen_meta.get("height")
    if (
        isinstance(width, int)
        and not isinstance(width, bool)
        and isinstance(height, int)
        and not isinstance(height, bool)
        and width > 0
        and height > 0
    ):
        return 0, 0, width, height
    return None


def _screen_size(ui_state: Mapping[str, Any]) -> tuple[int | None, int | None]:
    screen_meta = ui_state.get("screen_meta")
    if not isinstance(screen_meta, Mapping):
        return None, None

    screenshot_size = screen_meta.get("screenshot_size")
    if _is_int_pair(screenshot_size):
        return int(screenshot_size[0]), int(screenshot_size[1])

    screen_resolution = screen_meta.get("screen_resolution")
    if _is_int_pair(screen_resolution):
        return int(screen_resolution[0]), int(screen_resolution[1])

    width = screen_meta.get("width")
    height = screen_meta.get("height")
    if (
        isinstance(width, int)
        and not isinstance(width, bool)
        and isinstance(height, int)
        and not isinstance(height, bool)
        and width > 0
        and height > 0
    ):
        return width, height
    return None, None


def _bottom_input_point(ui_state: Mapping[str, Any]) -> tuple[int, int] | None:
    bounds = _coordinate_bounds(ui_state)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top
    return left + width // 2, top + int(height * 0.94)


def _message_input_fallback_point(
    state: RuntimeStateV2,
    ui_state: Mapping[str, Any],
) -> tuple[int, int] | None:
    if not (
        _prior_step_succeeded(state, STEP_OPEN_CHAT)
        or _prior_step_succeeded(state, STEP_FOCUS_MESSAGE_INPUT)
        or _chat_context_active(ui_state)
    ):
        return None
    return _bottom_input_point(ui_state)


def _prior_step_succeeded(state: RuntimeStateV2, step_name: str) -> bool:
    return any(
        item.step_name == step_name and item.status == RuntimeStatus.SUCCESS.value
        for item in state.history
    )


def _chat_context_active(ui_state: Mapping[str, Any]) -> bool:
    if _semantic_signal(ui_state, "input_available", "target_active", "detail_open"):
        return True
    if _semantic_view_is(ui_state, "chat", "detail"):
        return True

    for candidate in _candidate_list(ui_state):
        role = str(candidate.get("role") or "").casefold()
        if "chatlistitem" in role or "chatinput" in role or ("message" in role and "input" in role):
            return True

    markers = ("message", "chat", "aa@", "\u53d1\u9001\u7ed9", "\u6d88\u606f")
    for text in _visible_texts(ui_state):
        normalized = _normalize_text(text)
        if any(_normalize_text(marker) in normalized for marker in markers):
            return True
    return False


def _first_search_result_point(ui_state: Mapping[str, Any]) -> tuple[int, int] | None:
    if not _search_context_active(ui_state):
        return None
    bounds = _coordinate_bounds(ui_state)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top
    return left + int(width * 0.40), top + int(height * 0.24)


def _candidate_blob(candidate: Mapping[str, Any]) -> str:
    return " ".join(
        str(candidate.get(field_name) or "").casefold()
        for field_name in (
            "id",
            "text",
            "role",
            "source",
            "candidate_type",
            "semantic_type",
            "type",
            "class_name",
        )
    )


def _candidate_id(candidate: Mapping[str, Any] | None) -> str:
    if candidate is None:
        return ""
    return str(candidate.get("id") or "").strip()


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().casefold()


__all__ = [
    "FAILURE_EXECUTOR_FAILED",
    "FAILURE_CANDIDATE_CONFLICT",
    "FAILURE_CANDIDATE_INVALID",
    "FAILURE_COORDINATE_INVALID",
    "FAILURE_MAX_RETRY_EXCEEDED",
    "FAILURE_FALLBACK_FAILED",
    "FAILURE_PAGE_MISMATCH",
    "FAILURE_PAGE_NOT_CHANGED",
    "FAILURE_TARGET_NOT_FOUND",
    "FAILURE_VALIDATION_FAILED",
    "FAILURE_VLM_UNCERTAIN",
    "PlannerLoopV2",
    "PlannerLoopV2Result",
    "plan_next_workflow_action",
    "run_workflow_once",
    "run_workflow_until_done",
]
