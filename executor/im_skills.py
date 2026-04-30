"""Member C IM-side GUI action skills for second-cycle workflows.

Responsibility boundary: these helpers compose executor actions for common IM
steps. They still operate through the GUI executor and never call Feishu APIs or
planner internals.

Public interfaces:
`IMExecutionSkills`
    Small wrapper around BaseExecutor for click candidate, focus/type, Enter
    send, search hotkey, wait, and result-list scroll actions.
Module-level functions
    Convenience wrappers for scripts and tests that do not need to keep a
    skills instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from schemas import ACTION_CLICK, ACTION_HOTKEY, ACTION_SCROLL, ACTION_TYPE, ACTION_WAIT, ActionResult

from .base_executor import BaseExecutor, DesktopActionBackend, ExecutorConfig


DEFAULT_SEARCH_HOTKEY = ["ctrl", "k"]


@dataclass(slots=True)
class IMExecutionSkills:
    executor: BaseExecutor

    @classmethod
    def create(
        cls,
        *,
        backend: DesktopActionBackend | None = None,
        config: ExecutorConfig | None = None,
    ) -> "IMExecutionSkills":
        return cls(executor=BaseExecutor(backend=backend, config=config))

    def click_candidate(
        self,
        candidate: Mapping[str, Any] | Any,
        *,
        step_name: str = "click_candidate",
        reason: str = "",
    ) -> ActionResult:
        payload = _candidate_payload(candidate)
        return self.executor.execute_action(
            {
                "action_type": ACTION_CLICK,
                "step_name": step_name,
                "target_candidate_id": payload.get("id"),
                "target_candidate": payload,
                "reason": reason or "click candidate",
            }
        )

    def focus_and_type(
        self,
        text: str,
        *,
        candidate: Mapping[str, Any] | Any | None = None,
        x: int | None = None,
        y: int | None = None,
        step_name: str = "focus_and_type",
        reason: str = "",
    ) -> ActionResult:
        action: dict[str, Any] = {
            "action_type": ACTION_TYPE,
            "step_name": step_name,
            "text": text,
            "reason": reason or "focus target and type text",
        }
        if candidate is not None:
            payload = _candidate_payload(candidate)
            action["target_candidate_id"] = payload.get("id")
            action["target_candidate"] = payload
        elif x is not None or y is not None:
            action["x"] = x
            action["y"] = y
        return self.executor.execute_action(action)

    def press_enter_to_send(
        self,
        *,
        step_name: str = "press_enter_to_send",
        reason: str = "",
    ) -> ActionResult:
        return self.executor.execute_action(
            {
                "action_type": ACTION_HOTKEY,
                "step_name": step_name,
                "keys": ["enter"],
                "reason": reason or "press Enter to send message",
            }
        )

    def open_search_by_hotkey(
        self,
        *,
        keys: Sequence[str] = DEFAULT_SEARCH_HOTKEY,
        step_name: str = "open_search_by_hotkey",
        reason: str = "",
    ) -> ActionResult:
        return self.executor.execute_action(
            {
                "action_type": ACTION_HOTKEY,
                "step_name": step_name,
                "keys": [str(key) for key in keys],
                "reason": reason or "open search by hotkey",
            }
        )

    def wait_for_page_change(
        self,
        *,
        seconds: int | float = 1.0,
        step_name: str = "wait_for_page_change",
        reason: str = "",
    ) -> ActionResult:
        return self.executor.execute_action(
            {
                "action_type": ACTION_WAIT,
                "step_name": step_name,
                "seconds": seconds,
                "reason": reason or "wait for page change",
            }
        )

    def scroll_result_list(
        self,
        *,
        direction: str = "down",
        amount: int | None = None,
        x: int | None = None,
        y: int | None = None,
        step_name: str = "scroll_result_list",
        reason: str = "",
    ) -> ActionResult:
        action: dict[str, Any] = {
            "action_type": ACTION_SCROLL,
            "step_name": step_name,
            "direction": direction,
            "reason": reason or "scroll result list",
        }
        if amount is not None:
            action["amount"] = amount
        if x is not None or y is not None:
            action["x"] = x
            action["y"] = y
        return self.executor.execute_action(action)


def click_candidate(
    candidate: Mapping[str, Any] | Any,
    *,
    backend: DesktopActionBackend | None = None,
    config: ExecutorConfig | None = None,
) -> ActionResult:
    return IMExecutionSkills.create(backend=backend, config=config).click_candidate(candidate)


def focus_and_type(
    text: str,
    *,
    candidate: Mapping[str, Any] | Any | None = None,
    x: int | None = None,
    y: int | None = None,
    backend: DesktopActionBackend | None = None,
    config: ExecutorConfig | None = None,
) -> ActionResult:
    return IMExecutionSkills.create(backend=backend, config=config).focus_and_type(
        text,
        candidate=candidate,
        x=x,
        y=y,
    )


def press_enter_to_send(
    *,
    backend: DesktopActionBackend | None = None,
    config: ExecutorConfig | None = None,
) -> ActionResult:
    return IMExecutionSkills.create(backend=backend, config=config).press_enter_to_send()


def open_search_by_hotkey(
    *,
    backend: DesktopActionBackend | None = None,
    config: ExecutorConfig | None = None,
) -> ActionResult:
    return IMExecutionSkills.create(backend=backend, config=config).open_search_by_hotkey()


def wait_for_page_change(
    *,
    seconds: int | float = 1.0,
    backend: DesktopActionBackend | None = None,
    config: ExecutorConfig | None = None,
) -> ActionResult:
    return IMExecutionSkills.create(backend=backend, config=config).wait_for_page_change(
        seconds=seconds
    )


def scroll_result_list(
    *,
    direction: str = "down",
    amount: int | None = None,
    x: int | None = None,
    y: int | None = None,
    backend: DesktopActionBackend | None = None,
    config: ExecutorConfig | None = None,
) -> ActionResult:
    return IMExecutionSkills.create(backend=backend, config=config).scroll_result_list(
        direction=direction,
        amount=amount,
        x=x,
        y=y,
    )


def _candidate_payload(candidate: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(candidate, Mapping):
        return dict(candidate)
    to_dict = getattr(candidate, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise ValueError("candidate must be a mapping or expose to_dict()")


__all__ = [
    "DEFAULT_SEARCH_HOTKEY",
    "IMExecutionSkills",
    "click_candidate",
    "focus_and_type",
    "open_search_by_hotkey",
    "press_enter_to_send",
    "scroll_result_list",
    "wait_for_page_change",
]
