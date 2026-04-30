"""成员 C 4.24 运行工件 recorder。

职责边界：
本文件负责为一次 run 建立 artifact 目录结构，并把任务输入、每步动作、
每步执行结果、前后截图索引和运行汇总落盘。它不负责真实鼠标键盘执行，
不负责截图采集本身，也不判断业务成功；`action_runner.py` 会把 executor
和 recorder 串起来。

公开接口：
`RunRecorder`
    管理 `artifacts/runs/<run_id>/` 下的 task/actions/results/screenshots/
    summary 等文件。
`record_step(...)`
    写入单步 action、result、screenshot log，并更新 summary 统计。
`finalize(...)`
    写入最终 summary，返回可 JSON 序列化 dict。

路径规则：
```text
artifacts/runs/<run_id>/
  task.json
  actions.jsonl
  results.jsonl
  screenshots/
  screenshot_log.jsonl
  summary.json
```

统计规则：
summary 中按 `action_type` 统计：
1. `count`：该动作执行次数。
2. `failure_count`：失败次数。
3. `avg_duration_ms`：平均执行耗时，来自 `ActionResult.duration_ms`。

协作规则：
1. JSONL 每行都是独立 JSON，方便后续 replay / report 增量读取。
2. `action_result` 必须遵守 `schemas.ActionResult`。
3. recorder 只记录动作协议和执行结果，不写业务解释。
4. 真实运行产生的 artifacts 不提交 Git。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from schemas import ActionResult


RUN_SUMMARY_SCHEMA_VERSION = "run_summary.v1"
ACTION_LOG_SCHEMA_VERSION = "action_log.v1"
RESULT_LOG_SCHEMA_VERSION = "result_log.v1"
SCREENSHOT_LOG_SCHEMA_VERSION = "screenshot_log.v1"
VALIDATOR_LOG_SCHEMA_VERSION = "validator_log.v1"
UI_STATE_LOG_SCHEMA_VERSION = "ui_state_log.v1"


class RunRecorderError(ValueError):
    """Raised when run artifact data or paths are invalid."""


@dataclass(frozen=True, slots=True)
class RunArtifactPaths:
    run_id: str
    run_root: Path
    task_path: Path
    actions_path: Path
    results_path: Path
    screenshot_log_path: Path
    validator_log_path: Path
    ui_state_log_path: Path
    screenshots_dir: Path
    summary_path: Path

    @classmethod
    def from_root(cls, *, run_id: str, artifact_root: str | Path | None = None) -> "RunArtifactPaths":
        normalized_run_id = normalize_run_id(run_id)
        root = Path(artifact_root) if artifact_root is not None else Path("artifacts") / "runs"
        run_root = root / normalized_run_id
        return cls(
            run_id=normalized_run_id,
            run_root=run_root,
            task_path=run_root / "task.json",
            actions_path=run_root / "actions.jsonl",
            results_path=run_root / "results.jsonl",
            screenshot_log_path=run_root / "screenshot_log.jsonl",
            validator_log_path=run_root / "validator_log.jsonl",
            ui_state_log_path=run_root / "ui_state_log.jsonl",
            screenshots_dir=run_root / "screenshots",
            summary_path=run_root / "summary.json",
        )

    def ensure_dirs(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, str]:
        return {
            "run_root": str(self.run_root),
            "task_path": str(self.task_path),
            "actions_path": str(self.actions_path),
            "results_path": str(self.results_path),
            "screenshot_log_path": str(self.screenshot_log_path),
            "validator_log_path": str(self.validator_log_path),
            "ui_state_log_path": str(self.ui_state_log_path),
            "screenshots_dir": str(self.screenshots_dir),
            "summary_path": str(self.summary_path),
        }


class RunRecorder:
    def __init__(
        self,
        *,
        run_id: str | None = None,
        task: Mapping[str, Any] | Any | None = None,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.run_id = normalize_run_id(run_id or create_run_id())
        self.paths = RunArtifactPaths.from_root(run_id=self.run_id, artifact_root=artifact_root)
        self.paths.ensure_dirs()
        self.started_at = utc_now()
        self.updated_at = self.started_at
        self.step_count = 0
        self.success_count = 0
        self.failure_count = 0
        self._stats: dict[str, dict[str, int]] = {}
        self._failure_reasons: dict[str, int] = {}
        self._validator_count = 0
        self._validator_failure_count = 0
        self._task_saved = False
        self._status = "running"
        self._failure_reason: str | None = None

        if task is not None:
            self.save_task(task)

    def save_task(self, task: Mapping[str, Any] | Any) -> Path:
        payload = _jsonable(task)
        self.paths.task_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._task_saved = True
        return self.paths.task_path

    def record_step(
        self,
        *,
        step_idx: int,
        action_decision: Mapping[str, Any],
        action_result: ActionResult | Mapping[str, Any],
        before_screenshot: Mapping[str, Any] | Any | None = None,
        after_screenshot: Mapping[str, Any] | Any | None = None,
        step_name: str | None = None,
        executed_action: Mapping[str, Any] | None = None,
        after_ui_state: Mapping[str, Any] | Any | None = None,
        validator_result: Mapping[str, Any] | Any | None = None,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        normalized_step_idx = _non_negative_int(step_idx, "step_idx")
        action_payload = _mapping_field(action_decision, "action_decision")
        result = _action_result(action_result)
        recorded_at = utc_now()
        normalized_step_name = _optional_str(step_name) or _optional_str(action_payload.get("step_name"))
        normalized_failure_reason = _optional_str(failure_reason) or _optional_str(
            result.error if not result.ok else None
        )
        validator_payload = _optional_jsonable(validator_result)
        after_ui_state_payload = _optional_jsonable(after_ui_state)

        action_record = {
            "version": ACTION_LOG_SCHEMA_VERSION,
            "run_id": self.run_id,
            "step_idx": normalized_step_idx,
            "step_name": normalized_step_name,
            "recorded_at": recorded_at,
            "action": action_payload,
            "executed_action": dict(executed_action) if executed_action is not None else action_payload,
        }
        result_record = {
            "version": RESULT_LOG_SCHEMA_VERSION,
            "run_id": self.run_id,
            "step_idx": normalized_step_idx,
            "step_name": normalized_step_name,
            "recorded_at": recorded_at,
            "result": result.to_dict(),
            "failure_reason": normalized_failure_reason,
        }
        screenshot_record = {
            "version": SCREENSHOT_LOG_SCHEMA_VERSION,
            "run_id": self.run_id,
            "step_idx": normalized_step_idx,
            "step_name": normalized_step_name,
            "recorded_at": recorded_at,
            "before": _optional_jsonable(before_screenshot),
            "after": _optional_jsonable(after_screenshot),
        }
        validator_record = {
            "version": VALIDATOR_LOG_SCHEMA_VERSION,
            "run_id": self.run_id,
            "step_idx": normalized_step_idx,
            "step_name": normalized_step_name,
            "recorded_at": recorded_at,
            "validator_result": validator_payload,
        }
        ui_state_record = {
            "version": UI_STATE_LOG_SCHEMA_VERSION,
            "run_id": self.run_id,
            "step_idx": normalized_step_idx,
            "step_name": normalized_step_name,
            "recorded_at": recorded_at,
            "after_ui_state": after_ui_state_payload,
        }

        append_jsonl(self.paths.actions_path, action_record)
        append_jsonl(self.paths.results_path, result_record)
        append_jsonl(self.paths.screenshot_log_path, screenshot_record)
        append_jsonl(self.paths.validator_log_path, validator_record)
        append_jsonl(self.paths.ui_state_log_path, ui_state_record)
        self._update_stats(
            result,
            failure_reason=normalized_failure_reason,
            validator_result=validator_payload,
        )
        return self.write_summary()

    def finalize(
        self,
        *,
        status: str | None = None,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        if status is not None:
            self._status = _status_field(status)
        elif self.failure_count:
            self._status = "failed"
        else:
            self._status = "success"
        self._failure_reason = _optional_str(failure_reason)
        return self.write_summary()

    def write_summary(self) -> dict[str, Any]:
        self.updated_at = utc_now()
        payload = self.summary_payload()
        self.paths.summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload

    def summary_payload(self) -> dict[str, Any]:
        return {
            "version": RUN_SUMMARY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": self._status,
            "failure_reason": self._failure_reason,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "task_saved": self._task_saved,
            "step_count": self.step_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self._success_rate(),
            "avg_step_duration_ms": self._avg_step_duration_ms(),
            "validator_count": self._validator_count,
            "validator_failure_count": self._validator_failure_count,
            "failure_reasons": dict(sorted(self._failure_reasons.items())),
            "action_stats": self._action_stats_payload(),
            "artifact_paths": self.paths.to_dict(),
        }

    def _success_rate(self) -> float:
        if self.step_count == 0:
            return 0.0
        return round(self.success_count / self.step_count, 4)

    def _avg_step_duration_ms(self) -> int:
        if self.step_count == 0:
            return 0
        total_duration = sum(stats["total_duration_ms"] for stats in self._stats.values())
        return total_duration // self.step_count

    def _update_stats(
        self,
        result: ActionResult,
        *,
        failure_reason: str | None,
        validator_result: Mapping[str, Any] | None,
    ) -> None:
        self.step_count += 1
        if result.ok:
            self.success_count += 1
        else:
            self.failure_count += 1
        if failure_reason is not None:
            self._failure_reasons[failure_reason] = self._failure_reasons.get(failure_reason, 0) + 1
        if validator_result is not None:
            self._validator_count += 1
            if validator_result.get("passed") is False:
                self._validator_failure_count += 1

        stats = self._stats.setdefault(
            result.action_type,
            {
                "count": 0,
                "failure_count": 0,
                "total_duration_ms": 0,
                "click_offset_count": 0,
                "total_abs_click_offset_x": 0,
                "total_abs_click_offset_y": 0,
            },
        )
        stats["count"] += 1
        if not result.ok:
            stats["failure_count"] += 1
        stats["total_duration_ms"] += result.duration_ms or 0
        if result.click_offset is not None:
            stats["click_offset_count"] += 1
            stats["total_abs_click_offset_x"] += abs(result.click_offset[0])
            stats["total_abs_click_offset_y"] += abs(result.click_offset[1])

    def _action_stats_payload(self) -> dict[str, dict[str, int]]:
        payload: dict[str, dict[str, int]] = {}
        for action_type, stats in sorted(self._stats.items()):
            count = stats["count"]
            total_duration = stats["total_duration_ms"]
            payload[action_type] = {
                "count": count,
                "failure_count": stats["failure_count"],
                "avg_duration_ms": total_duration // count if count else 0,
            }
            click_offset_count = stats.get("click_offset_count", 0)
            if click_offset_count:
                payload[action_type]["avg_abs_click_offset_x"] = (
                    stats["total_abs_click_offset_x"] // click_offset_count
                )
                payload[action_type]["avg_abs_click_offset_y"] = (
                    stats["total_abs_click_offset_y"] // click_offset_count
                )
        return payload


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True))
        file.write("\n")


def create_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"


def normalize_run_id(run_id: str) -> str:
    text = str(run_id).strip()
    if not text:
        raise RunRecorderError("run_id cannot be empty")
    if any(char in text for char in '<>:"/\\|?*'):
        raise RunRecorderError(f"run_id contains invalid path characters: {text}")
    return text


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _action_result(value: ActionResult | Mapping[str, Any]) -> ActionResult:
    if isinstance(value, ActionResult):
        return value
    if isinstance(value, Mapping):
        return ActionResult.from_dict(value)
    raise RunRecorderError("action_result must be ActionResult or mapping")


def _jsonable(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, Mapping):
            return dict(data)
    raise RunRecorderError("value must be a mapping or expose to_dict()")


def _optional_jsonable(value: Mapping[str, Any] | Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return _jsonable(value)


def _mapping_field(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RunRecorderError(f"{field_name} must be a mapping")
    return dict(value)


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunRecorderError(f"{field_name} must be a non-negative integer")
    return value


def _status_field(value: str) -> str:
    text = value.strip()
    if text not in {"running", "success", "failed"}:
        raise RunRecorderError("status must be running, success, or failed")
    return text


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
