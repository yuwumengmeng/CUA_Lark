"""成员 B 第二周期 workflow.v2 流程表示。

职责边界：
本文件只定义第二周期 Planner Loop v2 使用的轻量 workflow schema，用来把 IM
短流程拆成固定 step 列表。它不负责运行时状态、不负责动作决策、不接入 VLM，
也不调用 executor；这些会在 Runtime State v2、ActionDecision v2 和 Planner
Loop v2 中继续实现。

公开接口：
`WorkflowStep`
    单个流程步骤，稳定字段为 `step_name / step_type / instruction / target /
    input_text / success_condition / fallback / max_retry`。
`WorkflowSpec`
    一个完整 workflow，包含任务说明、参数和 step 列表。
`build_im_message_workflow(...)`
    生成第二周期默认 IM 短流程：打开搜索、输入关键词、打开群聊、聚焦输入框、
    发送消息、验证发送成功。
`parse_workflow(...)`
    解析半结构化 workflow dict，方便测试和后续 Planner Loop v2 消费。

字段规则：
`step_type` 当前只支持 open_search / type_keyword / open_chat /
focus_message_input / send_message / verify_message_sent。`success_condition`
固定为 dict，`fallback` 固定为字符串列表，`max_retry` 固定为非负整数。

协作规则：
1. 新增 step_type 时必须同步更新 `SUPPORTED_WORKFLOW_STEP_TYPES` 和测试。
2. 本文件不定义 executor 动作字段，不能让成员 C 直接消费 workflow step。
3. workflow.v2 字段变化时必须同步更新 `tests/testB/test_workflow_schema.py`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from uuid import uuid4


WORKFLOW_SCHEMA_VERSION = "workflow.v2"
WORKFLOW_TYPE_IM_MESSAGE = "im_message"

STEP_OPEN_SEARCH = "open_search"
STEP_TYPE_KEYWORD = "type_keyword"
STEP_OPEN_CHAT = "open_chat"
STEP_FOCUS_MESSAGE_INPUT = "focus_message_input"
STEP_SEND_MESSAGE = "send_message"
STEP_VERIFY_MESSAGE_SENT = "verify_message_sent"

SUPPORTED_WORKFLOW_STEP_TYPES = {
    STEP_OPEN_SEARCH,
    STEP_TYPE_KEYWORD,
    STEP_OPEN_CHAT,
    STEP_FOCUS_MESSAGE_INPUT,
    STEP_SEND_MESSAGE,
    STEP_VERIFY_MESSAGE_SENT,
}


class WorkflowSchemaError(ValueError):
    """Raised when workflow.v2 data is malformed."""


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    step_name: str
    step_type: str
    instruction: str
    target: str | None = None
    input_text: str | None = None
    success_condition: dict[str, Any] = field(default_factory=dict)
    fallback: list[str] = field(default_factory=list)
    max_retry: int = 1

    def __post_init__(self) -> None:
        step_name = _required_text(self.step_name, "step_name")
        step_type = _required_text(self.step_type, "step_type")
        if step_type not in SUPPORTED_WORKFLOW_STEP_TYPES:
            raise WorkflowSchemaError(f"Unsupported workflow step_type: {step_type}")

        instruction = _required_text(self.instruction, "instruction")
        max_retry = _non_negative_int(self.max_retry, "max_retry")
        success_condition = _dict_field(self.success_condition, "success_condition")
        fallback = _fallback_list(self.fallback)

        object.__setattr__(self, "step_name", step_name)
        object.__setattr__(self, "step_type", step_type)
        object.__setattr__(self, "instruction", instruction)
        object.__setattr__(self, "target", _optional_text(self.target))
        object.__setattr__(self, "input_text", _optional_text(self.input_text))
        object.__setattr__(self, "success_condition", success_condition)
        object.__setattr__(self, "fallback", fallback)
        object.__setattr__(self, "max_retry", max_retry)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowStep":
        return cls(
            step_name=str(data.get("step_name") or ""),
            step_type=str(data.get("step_type") or ""),
            instruction=str(data.get("instruction") or ""),
            target=_optional_text(data.get("target")),
            input_text=_optional_text(data.get("input_text")),
            success_condition=_dict_field(
                data.get("success_condition", {}),
                "success_condition",
            ),
            fallback=_fallback_list(data.get("fallback", [])),
            max_retry=_non_negative_int(data.get("max_retry", 1), "max_retry"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "step_type": self.step_type,
            "instruction": self.instruction,
            "target": self.target,
            "input_text": self.input_text,
            "success_condition": dict(self.success_condition),
            "fallback": list(self.fallback),
            "max_retry": self.max_retry,
        }


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    workflow_id: str
    instruction: str
    workflow_type: str
    params: dict[str, Any] = field(default_factory=dict)
    steps: list[WorkflowStep] = field(default_factory=list)
    version: str = WORKFLOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        workflow_id = _required_text(self.workflow_id, "workflow_id")
        instruction = _required_text(self.instruction, "instruction")
        workflow_type = _required_text(self.workflow_type, "workflow_type")
        params = _dict_field(self.params, "params")
        if not self.steps:
            raise WorkflowSchemaError("WorkflowSpec steps cannot be empty")

        object.__setattr__(self, "workflow_id", workflow_id)
        object.__setattr__(self, "instruction", instruction)
        object.__setattr__(self, "workflow_type", workflow_type)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "steps", list(self.steps))
        object.__setattr__(self, "version", str(self.version or WORKFLOW_SCHEMA_VERSION))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowSpec":
        raw_steps = data.get("steps")
        if isinstance(raw_steps, (str, bytes)) or not isinstance(raw_steps, Sequence):
            raise WorkflowSchemaError("WorkflowSpec field steps must be a list")
        params = _dict_field(data.get("params", {}), "params")
        return cls(
            workflow_id=str(data.get("workflow_id") or _new_workflow_id()),
            instruction=str(data.get("instruction") or ""),
            workflow_type=str(data.get("workflow_type") or WORKFLOW_TYPE_IM_MESSAGE),
            params=params,
            steps=[WorkflowStep.from_dict(step) for step in raw_steps],
            version=str(data.get("version") or WORKFLOW_SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "workflow_id": self.workflow_id,
            "instruction": self.instruction,
            "workflow_type": self.workflow_type,
            "params": dict(self.params),
            "steps": [step.to_dict() for step in self.steps],
        }


def build_im_message_workflow(
    *,
    chat_name: str,
    message_text: str,
    search_keyword: str | None = None,
    instruction: str | None = None,
    workflow_id: str | None = None,
) -> WorkflowSpec:
    chat = _required_text(chat_name, "chat_name")
    message = _required_text(message_text, "message_text")
    keyword = _required_text(search_keyword or chat, "search_keyword")
    workflow_instruction = instruction or f"搜索并打开{chat}，发送消息：{message}"

    return WorkflowSpec(
        workflow_id=workflow_id or _new_workflow_id(),
        instruction=workflow_instruction,
        workflow_type=WORKFLOW_TYPE_IM_MESSAGE,
        params={
            "chat_name": chat,
            "search_keyword": keyword,
            "message_text": message,
        },
        steps=[
            WorkflowStep(
                step_name=STEP_OPEN_SEARCH,
                step_type=STEP_OPEN_SEARCH,
                instruction="打开飞书搜索",
                target="global_search",
                success_condition={"has_search_box": True},
                fallback=["open_search_by_hotkey"],
                max_retry=1,
            ),
            WorkflowStep(
                step_name=STEP_TYPE_KEYWORD,
                step_type=STEP_TYPE_KEYWORD,
                instruction=f"在搜索框输入：{keyword}",
                target="search_box",
                input_text=keyword,
                success_condition={"text_present": keyword},
                fallback=["refocus_search_box"],
                max_retry=1,
            ),
            WorkflowStep(
                step_name=STEP_OPEN_CHAT,
                step_type=STEP_OPEN_CHAT,
                instruction=f"打开目标群聊：{chat}",
                target=chat,
                success_condition={"chat_name_visible": chat},
                fallback=["retry_search"],
                max_retry=2,
            ),
            WorkflowStep(
                step_name=STEP_FOCUS_MESSAGE_INPUT,
                step_type=STEP_FOCUS_MESSAGE_INPUT,
                instruction="聚焦消息输入框",
                target="message_input",
                success_condition={"message_input_focused": True},
                fallback=["click_bottom_input_area"],
                max_retry=1,
            ),
            WorkflowStep(
                step_name=STEP_SEND_MESSAGE,
                step_type=STEP_SEND_MESSAGE,
                instruction=f"输入并发送消息：{message}",
                target="message_input",
                input_text=message,
                success_condition={"message_sent": message},
                fallback=["press_enter_to_send"],
                max_retry=2,
            ),
            WorkflowStep(
                step_name=STEP_VERIFY_MESSAGE_SENT,
                step_type=STEP_VERIFY_MESSAGE_SENT,
                instruction=f"验证聊天区出现消息：{message}",
                target=chat,
                input_text=message,
                success_condition={"message_visible": message},
                fallback=["wait_then_check_again"],
                max_retry=1,
            ),
        ],
    )


def parse_workflow(workflow_input: Mapping[str, Any] | WorkflowSpec) -> WorkflowSpec:
    if isinstance(workflow_input, WorkflowSpec):
        return workflow_input
    if isinstance(workflow_input, Mapping):
        return WorkflowSpec.from_dict(workflow_input)
    raise WorkflowSchemaError("Workflow input must be a WorkflowSpec or object")


def build_sample_im_workflows() -> list[WorkflowSpec]:
    return [
        build_im_message_workflow(chat_name="测试群", message_text="Hello CUA-Lark"),
        build_im_message_workflow(chat_name="测试群", message_text="第二周期 smoke test"),
        build_im_message_workflow(
            chat_name="项目群",
            search_keyword="项目群",
            message_text="CUA-Lark workflow.v2",
        ),
    ]


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise WorkflowSchemaError(f"Workflow field {field_name} cannot be empty")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dict_field(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkflowSchemaError(f"Workflow field {field_name} must be an object")
    return dict(value)


def _fallback_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WorkflowSchemaError("Workflow field fallback must be a list")
    return [str(item).strip() for item in value if str(item).strip()]


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkflowSchemaError(f"Workflow field {field_name} must be a non-negative integer")
    return value


def _new_workflow_id() -> str:
    return f"workflow_{uuid4().hex[:8]}"


__all__ = [
    "STEP_FOCUS_MESSAGE_INPUT",
    "STEP_OPEN_CHAT",
    "STEP_OPEN_SEARCH",
    "STEP_SEND_MESSAGE",
    "STEP_TYPE_KEYWORD",
    "STEP_VERIFY_MESSAGE_SENT",
    "SUPPORTED_WORKFLOW_STEP_TYPES",
    "WORKFLOW_SCHEMA_VERSION",
    "WORKFLOW_TYPE_IM_MESSAGE",
    "WorkflowSchemaError",
    "WorkflowSpec",
    "WorkflowStep",
    "build_im_message_workflow",
    "build_sample_im_workflows",
    "parse_workflow",
]
