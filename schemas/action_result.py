"""成员 C 输出给 B 的动作执行结果 schema。

职责边界：
本文件只定义 executor 执行单步动作后的结果协议、动作类型常量和 JSONL
可序列化字段。它不负责真实鼠标键盘执行、不负责截图，也不负责写日志；
`executor/base_executor.py` 和 `executor/recorder.py` 会消费这里的结构。

公开接口：
`ActionResult`
    单步动作执行结果，稳定字段为 `ok / action_type / start_ts / end_ts /
    error`，并额外提供 `duration_ms` 方便第一周期统计。
`action_result_from_exception(...)`
    将执行异常转换为失败结果，保证 executor 不向 planner 泄漏原始异常。

字段规则：
`ok`
    动作是否执行成功。
`action_type`
    planner 下发的动作类型，第一周期主要为 click / double_click / scroll /
    type / hotkey / wait。
`start_ts`、`end_ts`
    UTC ISO-8601 字符串，用于 run artifact 复盘和统计耗时。
`error`
    成功时固定为 None，失败时为可读错误文本。
`duration_ms`
    根据 start_ts/end_ts 计算的非负整数毫秒数。

协作规则：
1. B 只依赖 `ok / action_type / start_ts / end_ts / error` 判断成功或失败。
2. C 可以在 recorder 中使用 `duration_ms` 做动作耗时统计。
3. 失败结果必须保留原始 `action_type`，即使该类型暂不支持，方便定位协议问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping


ACTION_RESULT_SCHEMA_VERSION = "action_result.v1"

ACTION_CLICK = "click"
ACTION_DOUBLE_CLICK = "double_click"
ACTION_SCROLL = "scroll"
ACTION_TYPE = "type"
ACTION_HOTKEY = "hotkey"
ACTION_WAIT = "wait"

SUPPORTED_EXECUTOR_ACTION_TYPES = {
    ACTION_CLICK,
    ACTION_DOUBLE_CLICK,
    ACTION_SCROLL,
    ACTION_TYPE,
    ACTION_HOTKEY,
    ACTION_WAIT,
}


class ActionResultSchemaError(ValueError):
    """Raised when action_result.v1 data is malformed."""


@dataclass(frozen=True, slots=True)
class ActionResult:
    ok: bool
    action_type: str
    start_ts: str
    end_ts: str
    error: str | None = None
    duration_ms: int | None = None
    version: str = ACTION_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise ActionResultSchemaError("ActionResult ok must be a boolean")

        action_type = self.action_type.strip()
        if not action_type:
            raise ActionResultSchemaError("ActionResult action_type cannot be empty")

        start = _parse_timestamp(self.start_ts, "start_ts")
        end = _parse_timestamp(self.end_ts, "end_ts")
        if end < start:
            raise ActionResultSchemaError("ActionResult end_ts cannot be earlier than start_ts")

        error = _normalize_error(self.error)
        if self.ok and error is not None:
            raise ActionResultSchemaError("ActionResult error must be None when ok is true")
        if not self.ok and error is None:
            raise ActionResultSchemaError("ActionResult error is required when ok is false")

        duration_ms = self.duration_ms
        if duration_ms is None:
            duration_ms = int((end - start).total_seconds() * 1000)
        if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms < 0:
            raise ActionResultSchemaError("ActionResult duration_ms must be a non-negative integer")

        object.__setattr__(self, "action_type", action_type)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "duration_ms", duration_ms)

    @classmethod
    def success(
        cls,
        *,
        action_type: str,
        start_ts: str,
        end_ts: str | None = None,
    ) -> "ActionResult":
        return cls(
            ok=True,
            action_type=action_type,
            start_ts=start_ts,
            end_ts=end_ts or utc_now(),
            error=None,
        )

    @classmethod
    def failure(
        cls,
        *,
        action_type: str,
        start_ts: str,
        error: str,
        end_ts: str | None = None,
    ) -> "ActionResult":
        return cls(
            ok=False,
            action_type=action_type,
            start_ts=start_ts,
            end_ts=end_ts or utc_now(),
            error=error,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionResult":
        required = ("ok", "action_type", "start_ts", "end_ts", "error")
        missing = [field for field in required if field not in data]
        if missing:
            raise ActionResultSchemaError(
                f"ActionResult data is missing fields: {', '.join(missing)}"
            )
        return cls(
            ok=_bool_field(data["ok"], "ok"),
            action_type=str(data["action_type"]),
            start_ts=str(data["start_ts"]),
            end_ts=str(data["end_ts"]),
            error=None if data["error"] is None else str(data["error"]),
            duration_ms=_optional_non_negative_int(data.get("duration_ms"), "duration_ms"),
            version=str(data.get("version") or ACTION_RESULT_SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "ok": self.ok,
            "action_type": self.action_type,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


def action_result_from_exception(
    *,
    action_type: str,
    start_ts: str,
    error: Exception | str,
    end_ts: str | None = None,
) -> ActionResult:
    return ActionResult.failure(
        action_type=action_type,
        start_ts=start_ts,
        end_ts=end_ts,
        error=str(error),
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str, field_name: str) -> datetime:
    text = value.strip()
    if not text:
        raise ActionResultSchemaError(f"ActionResult {field_name} cannot be empty")
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ActionResultSchemaError(
            f"ActionResult {field_name} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _normalize_error(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool_field(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ActionResultSchemaError(f"ActionResult {field_name} must be a boolean")
    return value


def _optional_non_negative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ActionResultSchemaError(
            f"ActionResult {field_name} must be a non-negative integer"
        )
    return value
