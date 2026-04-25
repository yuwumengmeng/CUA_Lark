"""首版任务表示脚本。

职责边界：
本文件负责把自然语言或半结构化任务统一成 `TaskSpec`，并把任务拆成
固定字段的 `TaskStep` 列表。它定义 `goal / params / steps` 这套 mini
DSL，不负责 runtime state、planner loop，也不负责真实 GUI 执行。

公开接口：
`TaskSpec / TaskStep`
    成员 B 第一周期使用的任务与 step 结构。
`parse_task(...)`
    接收自然语言字符串或半结构化 dict，返回 `TaskSpec`。
`default_steps_for_goal(...)`
    根据已知 goal 生成默认 steps。

字段规则：
`TaskStep.action_hint` 是 planner 内部 step 提示，不等同于最终 executor
动作协议。当前支持 find_chat / find_search_box / find_sidebar_module /
click_target / type_text / hotkey_back / wait。

协作规则：
1. B 后续 planner 只能依赖这里稳定输出的 `goal / params / steps`。
2. 新增 action_hint 时必须同步更新 `SUPPORTED_ACTION_HINTS` 和测试。
3. 正式 action decision 协议不在本文件定义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from uuid import uuid4


TASK_SCHEMA_VERSION = "task.v1"

GOAL_OPEN_CHAT = "open_chat"
GOAL_INPUT_SEARCH = "input_search"
GOAL_OPEN_DOCS = "open_docs"
GOAL_GO_BACK = "go_back"

ACTION_FIND_CHAT = "find_chat"
ACTION_FIND_SEARCH_BOX = "find_search_box"
ACTION_FIND_SIDEBAR_MODULE = "find_sidebar_module"
ACTION_CLICK_TARGET = "click_target"
ACTION_TYPE_TEXT = "type_text"
ACTION_HOTKEY_BACK = "hotkey_back"
ACTION_WAIT = "wait"

SUPPORTED_ACTION_HINTS = {
    ACTION_FIND_CHAT,
    ACTION_FIND_SEARCH_BOX,
    ACTION_FIND_SIDEBAR_MODULE,
    ACTION_CLICK_TARGET,
    ACTION_TYPE_TEXT,
    ACTION_HOTKEY_BACK,
    ACTION_WAIT,
}


class TaskSchemaError(ValueError):
    """Raised when an input cannot be represented as a v1 task."""


@dataclass(slots=True)
class TaskStep:
    """A single mini-DSL step consumed by the future MiniPlanner."""

    action_hint: str
    target_text: str | None = None
    input_text: str | None = None
    target_role: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_hint not in SUPPORTED_ACTION_HINTS:
            raise TaskSchemaError(f"Unsupported action_hint: {self.action_hint}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskStep":
        if "action_hint" not in data:
            raise TaskSchemaError("Step is missing required field: action_hint")
        params = data.get("params", {})
        if not isinstance(params, Mapping):
            raise TaskSchemaError("Step field params must be an object")
        return cls(
            action_hint=str(data["action_hint"]),
            target_text=_optional_str(data.get("target_text")),
            input_text=_optional_str(data.get("input_text")),
            target_role=_optional_str(data.get("target_role")),
            params=dict(params),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_hint": self.action_hint,
            "target_text": self.target_text,
            "input_text": self.input_text,
            "target_role": self.target_role,
            "params": self.params,
        }


@dataclass(slots=True)
class TaskSpec:
    """Fixed v1 task shape.

    `goal` stays as a plain string so new goals can be added without changing
    the wire format. The first rule-based parser only emits the goals above.
    """

    task_id: str
    instruction: str
    goal: str
    params: dict[str, Any] = field(default_factory=dict)
    steps: list[TaskStep] = field(default_factory=list)
    version: str = TASK_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskSpec":
        if "goal" not in data:
            raise TaskSchemaError("Task is missing required field: goal")

        params = data.get("params", {})
        if not isinstance(params, Mapping):
            raise TaskSchemaError("Task field params must be an object")

        raw_steps = data.get("steps")
        if raw_steps is None:
            steps = default_steps_for_goal(str(data["goal"]), dict(params))
        else:
            if isinstance(raw_steps, (str, bytes)) or not isinstance(raw_steps, Sequence):
                raise TaskSchemaError("Task field steps must be a list")
            steps = [TaskStep.from_dict(step) for step in raw_steps]

        return cls(
            task_id=str(data.get("task_id") or _new_task_id()),
            instruction=str(data.get("instruction") or ""),
            goal=str(data["goal"]),
            params=dict(params),
            steps=steps,
            version=str(data.get("version") or TASK_SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "goal": self.goal,
            "params": self.params,
            "steps": [step.to_dict() for step in self.steps],
        }


def parse_task(task_input: str | Mapping[str, Any]) -> TaskSpec:
    """Normalize natural-language or semi-structured input into a TaskSpec."""

    if isinstance(task_input, str):
        return _parse_instruction(task_input)
    if isinstance(task_input, Mapping):
        return TaskSpec.from_dict(task_input)
    raise TaskSchemaError("Task input must be a string or object")


def default_steps_for_goal(goal: str, params: Mapping[str, Any]) -> list[TaskStep]:
    """Build the default first-week mini DSL steps for a known goal."""

    if goal == GOAL_OPEN_CHAT:
        target_name = _require_text_param(params, "target_name")
        return [
            TaskStep(action_hint=ACTION_FIND_CHAT, target_text=target_name),
            TaskStep(action_hint=ACTION_CLICK_TARGET, target_text=target_name),
        ]

    if goal == GOAL_INPUT_SEARCH:
        text = _require_text_param(params, "text", "keyword", "query")
        return [
            TaskStep(
                action_hint=ACTION_FIND_SEARCH_BOX,
                target_text="搜索",
                target_role="textbox",
            ),
            TaskStep(action_hint=ACTION_TYPE_TEXT, input_text=text),
        ]

    if goal == GOAL_OPEN_DOCS:
        target_text = str(params.get("target_text") or "文档")
        return [
            TaskStep(
                action_hint=ACTION_FIND_SIDEBAR_MODULE,
                target_text=target_text,
            ),
            TaskStep(action_hint=ACTION_CLICK_TARGET, target_text=target_text),
        ]

    if goal == GOAL_GO_BACK:
        return [TaskStep(action_hint=ACTION_HOTKEY_BACK, params={"keys": ["alt", "left"]})]

    raise TaskSchemaError(f"No default steps for goal: {goal}")


def _parse_instruction(instruction: str) -> TaskSpec:
    normalized = instruction.strip()
    if not normalized:
        raise TaskSchemaError("Instruction cannot be empty")

    if "返回" in normalized or "上一级" in normalized:
        return _task_from_goal(normalized, GOAL_GO_BACK, {})

    if "搜索框" in normalized and ("输入" in normalized or "搜索" in normalized):
        text = _extract_after_keyword(normalized, "输入")
        if not text:
            text = _extract_after_keyword(normalized, "搜索")
        return _task_from_goal(normalized, GOAL_INPUT_SEARCH, {"text": text})

    if "文档" in normalized and ("点击" in normalized or "打开" in normalized):
        return _task_from_goal(normalized, GOAL_OPEN_DOCS, {"target_text": "文档"})

    if "打开" in normalized and ("聊天" in normalized or "群" in normalized):
        target_name = _extract_open_target(normalized)
        return _task_from_goal(normalized, GOAL_OPEN_CHAT, {"target_name": target_name})

    raise TaskSchemaError(
        "Cannot parse instruction into task.v1. Provide a structured task object instead."
    )


def _task_from_goal(
    instruction: str,
    goal: str,
    params: Mapping[str, Any],
) -> TaskSpec:
    return TaskSpec(
        task_id=_new_task_id(),
        instruction=instruction,
        goal=goal,
        params=dict(params),
        steps=default_steps_for_goal(goal, params),
    )


def _extract_after_keyword(text: str, keyword: str) -> str:
    if keyword not in text:
        return ""
    value = text.split(keyword, 1)[1].strip()
    return _clean_value(_cut_at_stop(value))


def _extract_open_target(text: str) -> str:
    value = text.split("打开", 1)[1].strip()
    for marker in ("里的", "里", "中的", "中"):
        if marker in value:
            value = value.rsplit(marker, 1)[-1]
            break
    value = _clean_value(_cut_at_stop(value))
    if not value:
        raise TaskSchemaError("open_chat instruction is missing target name")
    return value


def _cut_at_stop(value: str) -> str:
    stops = "，。,.；;！!？?\n"
    for index, char in enumerate(value):
        if char in stops:
            return value[:index]
    return value


def _clean_value(value: str) -> str:
    cleaned = value.strip().strip("“”\"'` ")
    for prefix in ("的", "：", ":"):
        cleaned = cleaned.removeprefix(prefix).strip()
    return cleaned


def _require_text_param(params: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = params.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    joined = " / ".join(names)
    raise TaskSchemaError(f"Missing required text param: {joined}")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _new_task_id() -> str:
    return f"task_{uuid4().hex[:8]}"
