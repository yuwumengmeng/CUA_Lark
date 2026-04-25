"""成员 B 4.24/4.25 最小 Planner Loop、规则型决策与闭环校验。

职责边界：
本文件负责把 `TaskSpec + ui_state + RuntimeState` 串成第一版可运行的
planner loop，并实现第一周期的确定性规则决策、A/C 接口接入和轻量
step-level 成功判断。它不负责真实截图、不负责 OCR/UIA，也不负责真实
鼠标键盘执行；这些分别由成员 A 和成员 C 的模块提供。

公开接口：
`MiniPlanner`
    最小 planner 类。`plan_next_action(...)` 只做下一步决策；
    `run_once(...)` 串起一次 get_ui_state -> plan -> execute -> update state。
`plan_next_action(...)`
    模块级快捷入口，方便外部按文档约定直接调用。
`run_project_once(...)`
    项目内便捷入口，延迟接入 A 的 `get_ui_state()` 和 C 的 `execute_action()`。

输入输出规则：
`ui_state` 必须使用成员 A 的顶层字段：`screenshot_path / candidates /
page_summary`。candidate 第一周期至少依赖 `id / text / bbox / center /
role / clickable / editable / confidence`。

动作决策规则：
本文件通过 `planner/action_protocol.py` 生成 dict 形式的 action decision。
稳定字段为 `status / action_type / target_element_id / x / y / text / keys /
seconds / reason / failure_reason / version`。`planner_meta` 只用于 B 内部推进
step，不属于 C 必须依赖的执行协议。

协作规则：
1. B 只根据 A 的 `ui_state` 做选择，不直接解析 OCR/UIA 原始结果。
2. C 只消费 `action_type / target_element_id / x / y / text / keys / seconds /
   reason` 这些可执行字段。
3. `planner_meta` 只给 B 自己更新 runtime state 使用，不要写入 executor 依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .action_protocol import (
    ACTION_CLICK,
    ACTION_HOTKEY,
    ACTION_TYPE,
    ACTION_WAIT,
    DECISION_FAILED,
    DECISION_SUCCESS,
    FAILURE_EXECUTOR_FAILED,
    FAILURE_PAGE_MISMATCH,
    FAILURE_TARGET_NOT_FOUND,
    ActionDecision,
    action_decision_to_executor_payload,
)
from .runtime_state import RuntimeState, RuntimeStatus, create_runtime_state
from .task_schema import (
    ACTION_CLICK_TARGET,
    ACTION_FIND_CHAT,
    ACTION_FIND_SEARCH_BOX,
    ACTION_FIND_SIDEBAR_MODULE,
    ACTION_HOTKEY_BACK,
    ACTION_TYPE_TEXT,
    ACTION_WAIT as STEP_WAIT,
    GOAL_OPEN_CHAT,
    GOAL_OPEN_DOCS,
    TaskSpec,
    TaskStep,
    parse_task,
)


GetUIState = Callable[[], Mapping[str, Any]]
ExecuteAction = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(slots=True)
class PlannerRunResult:
    decision: dict[str, Any]
    runtime_state: RuntimeState
    action_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "action_result": self.action_result,
            "runtime_state": self.runtime_state.to_dict(),
        }


class MiniPlanner:
    """First runnable planner loop with deterministic rule-based decisions."""

    def plan_next_action(
        self,
        task: str | Mapping[str, Any] | TaskSpec,
        ui_state: Mapping[str, Any],
        runtime_state: RuntimeState | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        task_spec = _task_from_value(task)
        state = _state_from_value(runtime_state, task_spec)
        state.update_ui_state(ui_state)

        if state.step_idx >= len(task_spec.steps):
            return _decision_success("all task steps are already complete")

        step = task_spec.steps[state.step_idx]
        candidates = _candidate_list(ui_state)
        return self._decision_for_step(task_spec, step, candidates, state.step_idx)

    def run_once(
        self,
        task: str | Mapping[str, Any] | TaskSpec,
        get_ui_state: GetUIState,
        execute_action: ExecuteAction,
        runtime_state: RuntimeState | Mapping[str, Any] | None = None,
        *,
        get_after_ui_state: GetUIState | None = None,
    ) -> PlannerRunResult:
        task_spec = _task_from_value(task)
        state = _state_from_value(runtime_state, task_spec)
        step_idx = state.step_idx
        step = task_spec.steps[step_idx] if step_idx < len(task_spec.steps) else None
        ui_state = get_ui_state()
        decision = self.plan_next_action(task_spec, ui_state, state)

        if decision["status"] == DECISION_SUCCESS:
            state.status = RuntimeStatus.SUCCESS.value
            return PlannerRunResult(decision=decision, runtime_state=state)

        if decision["status"] == DECISION_FAILED:
            state.record_step(
                decision,
                None,
                status=RuntimeStatus.FAILED.value,
                failure_reason=decision.get("failure_reason"),
            )
            return PlannerRunResult(decision=decision, runtime_state=state)

        action_payload = action_decision_to_executor_payload(decision)
        action_result = dict(execute_action(action_payload))
        if not action_result.get("ok", False):
            state.record_step(
                decision,
                action_result,
                status=RuntimeStatus.FAILED.value,
                failure_reason=FAILURE_EXECUTOR_FAILED,
            )
            return PlannerRunResult(
                decision=decision,
                action_result=action_result,
                runtime_state=state,
            )

        after_ui_state = get_after_ui_state() if get_after_ui_state is not None else None
        if after_ui_state is not None and step is not None:
            success, _reason = _step_succeeded(task_spec, step, decision, after_ui_state)
            if not success:
                state.record_step(
                    decision,
                    action_result,
                    status=RuntimeStatus.FAILED.value,
                    failure_reason=FAILURE_PAGE_MISMATCH,
                )
                state.update_ui_state(after_ui_state)
                state.status = RuntimeStatus.FAILED.value
                state.failure_reason = FAILURE_PAGE_MISMATCH
                return PlannerRunResult(
                    decision=decision,
                    action_result=action_result,
                    runtime_state=state,
                )

        advance_steps = _advance_steps(decision)
        next_step_idx = state.step_idx + advance_steps
        next_status = (
            RuntimeStatus.SUCCESS.value
            if next_step_idx >= len(task_spec.steps)
            else RuntimeStatus.RUNNING.value
        )
        state.record_step(
            decision,
            action_result,
            status=next_status,
            advance_step=True,
        )
        if advance_steps > 1:
            state.step_idx += advance_steps - 1
        if after_ui_state is not None:
            state.update_ui_state(after_ui_state)
        state.status = next_status
        return PlannerRunResult(
            decision=decision,
            action_result=action_result,
            runtime_state=state,
        )

    def _decision_for_step(
        self,
        task: TaskSpec,
        step: TaskStep,
        candidates: list[dict[str, Any]],
        step_idx: int,
    ) -> dict[str, Any]:
        if step.action_hint == ACTION_TYPE_TEXT:
            if not step.input_text:
                return _decision_failed(FAILURE_PAGE_MISMATCH, "type_text step has no input_text")
            return _decision_continue(
                action_type=ACTION_TYPE,
                text=step.input_text,
                reason=f"type text for step_idx={step_idx}",
            )

        if step.action_hint == ACTION_HOTKEY_BACK:
            keys = list(step.params.get("keys", ["alt", "left"]))
            return _decision_continue(
                action_type=ACTION_HOTKEY,
                keys=keys,
                reason=f"hotkey back for step_idx={step_idx}",
            )

        if step.action_hint == STEP_WAIT:
            seconds = _wait_seconds(step)
            if seconds is None:
                return _decision_failed(FAILURE_PAGE_MISMATCH, "wait step has invalid seconds")
            return _decision_continue(
                action_type=ACTION_WAIT,
                seconds=seconds,
                reason=f"wait {seconds} seconds for step_idx={step_idx}",
            )

        if step.action_hint == ACTION_FIND_SEARCH_BOX:
            candidate = _best_candidate(
                candidates,
                target_text=step.target_text or "搜索",
                target_role=step.target_role,
                prefer_editable=True,
            )
            if candidate is None:
                return _decision_failed(FAILURE_TARGET_NOT_FOUND, "search box candidate not found")
            return _click_decision(candidate, f"focus search box for step_idx={step_idx}")

        if step.action_hint in {ACTION_FIND_CHAT, ACTION_FIND_SIDEBAR_MODULE, ACTION_CLICK_TARGET}:
            target_text = _target_text_for_step(task, step)
            candidate = _best_candidate(candidates, target_text=target_text)
            if candidate is None:
                candidate = _unique_chat_row_suffix_candidate(task, candidates, target_text)
                if candidate is None:
                    return _decision_failed(
                        FAILURE_TARGET_NOT_FOUND,
                        f"target candidate not found: {target_text}",
                    )

            advance_steps = 1
            if step.action_hint in {ACTION_FIND_CHAT, ACTION_FIND_SIDEBAR_MODULE}:
                next_step = _next_step(task, step_idx)
                if next_step is not None and next_step.action_hint == ACTION_CLICK_TARGET:
                    advance_steps = 2
            matched_text = str(candidate.get("text") or "")
            fallback = str(candidate.get("_planner_fallback") or "")
            reason = f"matched target_text={target_text}"
            extra_meta = None
            if fallback:
                reason = (
                    f"unique chat row suffix fallback target_text={target_text} "
                    f"matched_text={matched_text}"
                )
                extra_meta = {"fallback": fallback}
            return _click_decision(
                candidate,
                reason,
                advance_steps=advance_steps,
                matched_text=matched_text,
                extra_meta=extra_meta,
            )

        return _decision_failed(
            FAILURE_PAGE_MISMATCH,
            f"unsupported step action_hint: {step.action_hint}",
        )


def plan_next_action(
    task: str | Mapping[str, Any] | TaskSpec,
    ui_state: Mapping[str, Any],
    runtime_state: RuntimeState | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return MiniPlanner().plan_next_action(task, ui_state, runtime_state)


def run_project_once(
    task: str | Mapping[str, Any] | TaskSpec,
    runtime_state: RuntimeState | Mapping[str, Any] | None = None,
    *,
    run_id: str = "run_planner",
    step_idx: int | None = None,
    artifact_root: str | None = None,
) -> PlannerRunResult:
    task_spec = _task_from_value(task)
    state = _state_from_value(runtime_state, task_spec)
    capture_step_idx = step_idx if step_idx is not None else state.step_idx + 1

    from executor.base_executor import execute_action
    from perception.ui_state import get_ui_state

    def get_before_ui_state() -> Mapping[str, Any]:
        return get_ui_state(
            run_id=run_id,
            step_idx=capture_step_idx,
            phase="before",
            artifact_root=artifact_root,
        )

    def get_after_ui_state() -> Mapping[str, Any]:
        return get_ui_state(
            run_id=run_id,
            step_idx=capture_step_idx,
            phase="after",
            artifact_root=artifact_root,
        )

    return MiniPlanner().run_once(
        task_spec,
        get_before_ui_state,
        execute_action,
        state,
        get_after_ui_state=get_after_ui_state,
    )


def _task_from_value(value: str | Mapping[str, Any] | TaskSpec) -> TaskSpec:
    if isinstance(value, TaskSpec):
        return value
    return parse_task(value)


def _state_from_value(
    value: RuntimeState | Mapping[str, Any] | None,
    task: TaskSpec,
) -> RuntimeState:
    if value is None:
        return create_runtime_state(task)
    if isinstance(value, RuntimeState):
        return value
    return RuntimeState.from_dict(value)


def _candidate_list(ui_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = ui_state.get("candidates", [])
    if not isinstance(raw_candidates, list):
        return []
    return [dict(candidate) for candidate in raw_candidates if isinstance(candidate, Mapping)]


def _step_succeeded(
    task: TaskSpec,
    step: TaskStep,
    decision: Mapping[str, Any],
    after_ui_state: Mapping[str, Any],
) -> tuple[bool, str]:
    if decision.get("action_type") in {ACTION_HOTKEY, ACTION_WAIT}:
        return True, ""

    if step.action_hint == ACTION_TYPE_TEXT:
        target_text = step.input_text or decision.get("text")
        if _contains_text(after_ui_state, target_text):
            return True, ""
        return False, f"after page does not contain typed text: {target_text}"

    if step.action_hint == ACTION_FIND_SEARCH_BOX:
        if _has_search_box(after_ui_state) or _contains_text(after_ui_state, step.target_text):
            return True, ""
        return False, "after page does not look like a search box is available"

    if task.goal == GOAL_OPEN_DOCS and _contains_any_text(after_ui_state, ("文档", "新建文档", "新建")):
        return True, ""

    if step.action_hint in {ACTION_FIND_CHAT, ACTION_FIND_SIDEBAR_MODULE, ACTION_CLICK_TARGET}:
        target_text = _target_text_for_step(task, step)
        if not target_text or _contains_text(after_ui_state, target_text):
            return True, ""
        if task.goal == GOAL_OPEN_CHAT:
            matched_text = _decision_matched_text(decision)
            if _is_meaningful_partial_match(matched_text, target_text) and (
                _contains_text(after_ui_state, matched_text)
                or _looks_like_open_chat(after_ui_state)
            ):
                return True, ""
        return False, f"after page does not contain target_text: {target_text}"

    return False, f"unsupported success check for action_hint: {step.action_hint}"


def _has_search_box(ui_state: Mapping[str, Any]) -> bool:
    page_summary = ui_state.get("page_summary", {})
    if isinstance(page_summary, Mapping) and page_summary.get("has_search_box") is True:
        return True
    for candidate in _candidate_list(ui_state):
        if candidate.get("editable") is True or _role_matches(str(candidate.get("role") or ""), "edit"):
            return True
    return False


def _contains_any_text(ui_state: Mapping[str, Any], targets: tuple[str, ...]) -> bool:
    return any(_contains_text(ui_state, target) for target in targets)


def _looks_like_open_chat(ui_state: Mapping[str, Any]) -> bool:
    chat_surface_markers = (
        "\u53d1\u9001\u7ed9",
        "\u65b0\u6d88\u606f",
        "\u9080\u8bf7",
        "\u52a0\u5165\u6b64\u7fa4",
        "\u64a4\u56de\u4e86\u4e00\u6761\u6d88\u606f",
    )
    return _contains_any_text(ui_state, chat_surface_markers)


def _contains_text(ui_state: Mapping[str, Any], target_text: Any) -> bool:
    if target_text is None:
        return False
    needle = str(target_text).strip().casefold()
    if not needle:
        return False
    for text in _visible_texts(ui_state):
        haystack = text.casefold()
        if needle in haystack:
            return True
        if _ocr_tolerant_text_match(text, target_text):
            return True
    return False


def _ocr_tolerant_text_match(text: Any, target_text: Any) -> bool:
    candidate = _normalize_match_text(text)
    target = _normalize_match_text(target_text)
    if not candidate or not target:
        return False
    if candidate == target or target in candidate:
        return True
    if candidate in target:
        return (
            _is_meaningful_partial_match(candidate, target)
            and len(candidate) / len(target) >= 0.75
        )

    # OCR can drop or substitute one Chinese character in short search terms
    # such as "项目周报" -> "项目报". Keep this conservative so one-character
    # fragments and unrelated short labels do not pass business validation.
    if len(target) < 4 or len(candidate) < 3:
        return False
    common = _longest_common_subsequence_length(candidate, target)
    return common >= len(target) - 1 and len(candidate) >= len(target) - 1


def _longest_common_subsequence_length(left: str, right: str) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _visible_texts(ui_state: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    page_summary = ui_state.get("page_summary", {})
    if isinstance(page_summary, Mapping):
        top_texts = page_summary.get("top_texts", [])
        if isinstance(top_texts, list):
            texts.extend(str(text).strip() for text in top_texts if str(text).strip())
    for candidate in _candidate_list(ui_state):
        text = str(candidate.get("text") or "").strip()
        if text:
            texts.append(text)
    return texts


def _best_candidate(
    candidates: list[dict[str, Any]],
    *,
    target_text: str | None = None,
    target_role: str | None = None,
    prefer_editable: bool = False,
) -> dict[str, Any] | None:
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        score = _candidate_score(
            candidate,
            target_text=target_text,
            target_role=target_role,
            prefer_editable=prefer_editable,
        )
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
    target_role: str | None,
    prefer_editable: bool,
) -> float:
    text = str(candidate.get("text") or "")
    role = str(candidate.get("role") or "")
    score = float(candidate.get("confidence") or 0.0)
    preferred_input = prefer_editable and (
        candidate.get("editable") is True or _role_matches(role, "edit")
    )

    if target_text:
        match_score = _candidate_text_match_score(text, target_text)
        if match_score is not None:
            score += match_score
        elif preferred_input:
            score += 1.0
        else:
            score -= 10.0

    if _role_matches(role, target_role):
        score += 3.0

    if preferred_input:
        score += 8.0

    if candidate.get("clickable") is True:
        score += 1.0

    if _candidate_point(candidate) is None:
        score -= 20.0

    return score


def _role_matches(role: str, target_role: str | None) -> bool:
    if not target_role:
        return False
    role_text = role.lower()
    target_text = target_role.lower()
    if target_text in role_text:
        return True
    if target_text in {"textbox", "input", "edit"}:
        return any(keyword in role_text for keyword in ("textbox", "input", "edit"))
    return False


def _wait_seconds(step: TaskStep) -> int | float | None:
    value = step.params.get("seconds", 1)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return value


def _click_decision(
    candidate: Mapping[str, Any],
    reason: str,
    *,
    advance_steps: int = 1,
    matched_text: str | None = None,
    extra_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    point = _candidate_point(candidate)
    if point is None:
        return _decision_failed(
            FAILURE_TARGET_NOT_FOUND,
            f"candidate has no usable center or bbox: {candidate.get('id')}",
        )
    planner_meta: dict[str, Any] = {"advance_steps": advance_steps}
    if matched_text:
        planner_meta["matched_text"] = matched_text
    if extra_meta:
        planner_meta.update(dict(extra_meta))
    return _decision_continue(
        action_type=ACTION_CLICK,
        target_element_id=str(candidate.get("id") or ""),
        x=point[0],
        y=point[1],
        reason=reason,
        planner_meta=planner_meta,
    )


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


def _target_text_for_step(task: TaskSpec, step: TaskStep) -> str:
    if step.target_text:
        return step.target_text
    for field_name in ("target_text", "target_name", "text"):
        value = task.params.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _next_step(task: TaskSpec, step_idx: int) -> TaskStep | None:
    next_index = step_idx + 1
    if next_index >= len(task.steps):
        return None
    return task.steps[next_index]


def _advance_steps(decision: Mapping[str, Any]) -> int:
    planner_meta = decision.get("planner_meta", {})
    if not isinstance(planner_meta, Mapping):
        return 1
    value = planner_meta.get("advance_steps", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 1
    return value


def _decision_continue(**fields: Any) -> dict[str, Any]:
    return ActionDecision.continue_(**fields).to_dict()


def _decision_success(reason: str) -> dict[str, Any]:
    return ActionDecision.success(reason).to_dict()


def _decision_failed(failure_reason: str, reason: str) -> dict[str, Any]:
    return ActionDecision.failed(failure_reason, reason).to_dict()


def _unique_chat_row_suffix_candidate(
    task: TaskSpec,
    candidates: Sequence[Mapping[str, Any]],
    target_text: str,
) -> dict[str, Any] | None:
    if task.goal != GOAL_OPEN_CHAT:
        return None
    target = _normalize_match_text(target_text)
    if not target:
        return None

    matches: list[Mapping[str, Any]] = []
    for candidate in candidates:
        if str(candidate.get("role") or "") != "LarkChatListItem":
            continue
        text = _normalize_match_text(candidate.get("text"))
        if not text or text == target:
            continue
        if len(text) < len(target) and target.endswith(text):
            matches.append(candidate)

    if len(matches) != 1:
        return None

    result = dict(matches[0])
    result["_planner_fallback"] = "unique_chat_row_suffix"
    return result


def _candidate_text_match_score(text: str, target_text: str) -> float | None:
    candidate = _normalize_match_text(text)
    target = _normalize_match_text(target_text)
    if not candidate or not target:
        return None

    if target in candidate:
        if target == candidate:
            return 12.0
        # Longer candidates like "我的文档库" are still valid, but exact
        # visible labels should win over broad container text.
        return 10.0 + min(len(target) / max(len(candidate), 1), 1.0)

    if candidate in target:
        if not _is_meaningful_partial_match(candidate, target):
            return None
        ratio = len(candidate) / len(target)
        return 4.0 + ratio * 3.0

    return None


def _is_meaningful_partial_match(text: Any, target_text: Any) -> bool:
    candidate = _normalize_match_text(text)
    target = _normalize_match_text(target_text)
    if not candidate or not target or candidate not in target:
        return False
    if candidate == target:
        return True
    # Avoid picking single OCR fragments such as "群" for target "测试群".
    if len(candidate) < 2 and len(target) > 1:
        return False
    return len(candidate) / len(target) >= 0.5


def _normalize_match_text(value: Any) -> str:
    text = str(value or "").strip().casefold()
    return "".join(char for char in text if not char.isspace() and char not in "，。,.；;：:（）()[]【】<>《》\"'`")


def _decision_matched_text(decision: Mapping[str, Any]) -> str:
    planner_meta = decision.get("planner_meta", {})
    if not isinstance(planner_meta, Mapping):
        return ""
    return str(planner_meta.get("matched_text") or "")
