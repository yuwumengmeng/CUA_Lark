"""Member C lightweight validation layer for second-cycle workflow steps.

Responsibility boundary: this module validates post-action state from existing
artifacts and ui_state dictionaries. It does not capture screenshots, execute
desktop actions, call VLM APIs, or advance planner state.

Public interfaces:
`ValidatorResult`
    Stable validator result dict for recorder and planner consumption. C 侧产出
    `passed / check_type / failure_reason / evidence / version`，B 侧只消费
    `passed` 判断状态推进，失败时读取 `failure_reason`。
`validate_text_visible(...)`
    Check whether target text appears in page summary, candidates, document
    summary, or VLM semantics.
`validate_page_changed(...)`
    Compare before/after ui_state signatures without relying only on screenshot
    file names.
`validate_vlm_semantic_pass(...)`
    Check VLM semantic signals and goal progress when available.
`validate_expected_after_state(...)`
    Small dispatcher for ActionDecision.expected_after_state-style dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


VALIDATOR_RESULT_SCHEMA_VERSION = "validator_result.v1"

VALIDATION_TARGET_TEXT_NOT_FOUND = "target_text_not_found"
VALIDATION_PAGE_NOT_CHANGED = "page_not_changed"
VALIDATION_VLM_SEMANTIC_FAILED = "vlm_semantic_failed"
VALIDATION_EXPECTATION_UNSUPPORTED = "expectation_unsupported"

GOAL_PROGRESS_PASS_VALUES = {"achieved", "near_goal"}


@dataclass(frozen=True, slots=True)
class ValidatorResult:
    passed: bool
    check_type: str
    target_text: str | None = None
    message: str = ""
    failure_reason: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    version: str = VALIDATOR_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise ValidatorError("ValidatorResult passed must be a boolean")
        check_type = str(self.check_type).strip()
        if not check_type:
            raise ValidatorError("ValidatorResult check_type cannot be empty")
        target_text = _optional_text(self.target_text)
        failure_reason = _optional_text(self.failure_reason)
        if self.passed and failure_reason is not None:
            raise ValidatorError("ValidatorResult failure_reason must be None when passed")
        if not self.passed and failure_reason is None:
            raise ValidatorError("ValidatorResult failure_reason is required when failed")
        if not isinstance(self.evidence, Mapping):
            raise ValidatorError("ValidatorResult evidence must be a mapping")
        object.__setattr__(self, "check_type", check_type)
        object.__setattr__(self, "target_text", target_text)
        object.__setattr__(self, "message", str(self.message or ""))
        object.__setattr__(self, "failure_reason", failure_reason)
        object.__setattr__(self, "evidence", dict(self.evidence))

    @classmethod
    def pass_(
        cls,
        *,
        check_type: str,
        target_text: str | None = None,
        message: str = "",
        evidence: Mapping[str, Any] | None = None,
    ) -> "ValidatorResult":
        return cls(
            passed=True,
            check_type=check_type,
            target_text=target_text,
            message=message,
            failure_reason=None,
            evidence=dict(evidence or {}),
        )

    @classmethod
    def fail(
        cls,
        *,
        check_type: str,
        failure_reason: str,
        target_text: str | None = None,
        message: str = "",
        evidence: Mapping[str, Any] | None = None,
    ) -> "ValidatorResult":
        return cls(
            passed=False,
            check_type=check_type,
            target_text=target_text,
            message=message,
            failure_reason=failure_reason,
            evidence=dict(evidence or {}),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidatorResult":
        return cls(
            passed=_bool_field(data.get("passed"), "passed"),
            check_type=str(data.get("check_type") or ""),
            target_text=_optional_text(data.get("target_text")),
            message=str(data.get("message") or ""),
            failure_reason=_optional_text(data.get("failure_reason")),
            evidence=dict(data.get("evidence", {})),
            version=str(data.get("version") or VALIDATOR_RESULT_SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "passed": self.passed,
            "check_type": self.check_type,
            "target_text": self.target_text,
            "message": self.message,
            "failure_reason": self.failure_reason,
            "evidence": dict(self.evidence),
        }


class ValidatorError(ValueError):
    """Raised when validator inputs or results are malformed."""


def validate_text_visible(
    ui_state: Mapping[str, Any],
    target_text: str,
    *,
    check_type: str = "target_text_visible",
) -> ValidatorResult:
    target = _required_text(target_text, "target_text")
    matches = _find_text_matches(ui_state, target)
    evidence = {"matches": matches, "source_count": len(matches)}
    if matches:
        return ValidatorResult.pass_(
            check_type=check_type,
            target_text=target,
            message="target text is visible",
            evidence=evidence,
        )
    return ValidatorResult.fail(
        check_type=check_type,
        target_text=target,
        failure_reason=VALIDATION_TARGET_TEXT_NOT_FOUND,
        message="target text was not found in after ui_state",
        evidence=evidence,
    )


def validate_candidates_contain_text(
    ui_state: Mapping[str, Any],
    target_text: str,
) -> ValidatorResult:
    return validate_text_visible(
        {"candidates": list(ui_state.get("candidates", []))},
        target_text,
        check_type="candidates_contain_text",
    )


def validate_chat_message_visible(
    ui_state: Mapping[str, Any],
    message_text: str,
) -> ValidatorResult:
    return validate_text_visible(
        ui_state,
        message_text,
        check_type="chat_message_visible",
    )


def validate_page_changed(
    before_ui_state: Mapping[str, Any] | None,
    after_ui_state: Mapping[str, Any] | None,
) -> ValidatorResult:
    before_signature = _state_signature(before_ui_state or {})
    after_signature = _state_signature(after_ui_state or {})
    screenshot_path_changed = str((before_ui_state or {}).get("screenshot_path") or "") != str(
        (after_ui_state or {}).get("screenshot_path") or ""
    )
    evidence = {
        "before_signature": before_signature,
        "after_signature": after_signature,
        "screenshot_path_changed": screenshot_path_changed,
    }
    if before_signature != after_signature:
        return ValidatorResult.pass_(
            check_type="page_changed",
            message="after ui_state differs from before ui_state",
            evidence=evidence,
        )
    return ValidatorResult.fail(
        check_type="page_changed",
        failure_reason=VALIDATION_PAGE_NOT_CHANGED,
        message="after ui_state did not materially change",
        evidence=evidence,
    )


def validate_vlm_semantic_pass(
    ui_state: Mapping[str, Any],
    *,
    expected_signal: str | None = None,
    expected_goal_progress: str | None = None,
    target_text: str | None = None,
) -> ValidatorResult:
    semantics = _page_semantics(ui_state)
    signals = semantics.get("signals") if isinstance(semantics.get("signals"), Mapping) else {}
    goal_progress = _optional_text(semantics.get("goal_progress"))
    expected_signal = _optional_text(expected_signal)
    expected_goal_progress = _optional_text(expected_goal_progress)
    target_text = _optional_text(target_text)
    evidence = {
        "signals": dict(signals),
        "goal_progress": goal_progress,
        "summary": semantics.get("summary"),
        "entities": list(semantics.get("entities", []))
        if isinstance(semantics.get("entities"), list)
        else [],
    }

    checks: list[bool] = []
    if expected_signal is not None:
        checks.append(bool(signals.get(expected_signal)))
    if expected_goal_progress is not None:
        checks.append(goal_progress == expected_goal_progress)
    elif expected_signal is None and target_text is None:
        checks.append(goal_progress in GOAL_PROGRESS_PASS_VALUES)
    if target_text is not None:
        checks.append(bool(_find_text_matches(ui_state, target_text)))

    passed = bool(checks) and all(checks)
    if passed:
        return ValidatorResult.pass_(
            check_type="vlm_semantic_pass",
            target_text=target_text,
            message="VLM semantic expectation passed",
            evidence=evidence,
        )
    return ValidatorResult.fail(
        check_type="vlm_semantic_pass",
        target_text=target_text,
        failure_reason=VALIDATION_VLM_SEMANTIC_FAILED,
        message="VLM semantic expectation failed",
        evidence=evidence,
    )


def validate_expected_after_state(
    expected_after_state: Mapping[str, Any],
    *,
    before_ui_state: Mapping[str, Any] | None = None,
    after_ui_state: Mapping[str, Any] | None = None,
) -> ValidatorResult:
    expectations = dict(expected_after_state)
    after = after_ui_state or {}
    results: list[ValidatorResult] = []

    if expectations.get("page_changed") is True:
        results.append(validate_page_changed(before_ui_state, after))
    if expectations.get("target_text") is not None:
        results.append(validate_text_visible(after, str(expectations["target_text"])))
    if expectations.get("message_text") is not None:
        results.append(validate_chat_message_visible(after, str(expectations["message_text"])))
    if expectations.get("vlm_signal") is not None:
        results.append(
            validate_vlm_semantic_pass(after, expected_signal=str(expectations["vlm_signal"]))
        )
    if expectations.get("goal_progress") is not None:
        results.append(
            validate_vlm_semantic_pass(
                after,
                expected_goal_progress=str(expectations["goal_progress"]),
            )
        )

    if not results:
        return ValidatorResult.fail(
            check_type="expected_after_state",
            failure_reason=VALIDATION_EXPECTATION_UNSUPPORTED,
            message="expected_after_state contains no supported validator keys",
            evidence={"expected_after_state": expectations},
        )

    evidence = {
        "results": [result.to_dict() for result in results],
        "expected_after_state": expectations,
    }
    failed = [result for result in results if not result.passed]
    if failed:
        return ValidatorResult.fail(
            check_type="expected_after_state",
            failure_reason=failed[0].failure_reason or VALIDATION_EXPECTATION_UNSUPPORTED,
            message="one or more expected_after_state checks failed",
            evidence=evidence,
        )
    return ValidatorResult.pass_(
        check_type="expected_after_state",
        message="all expected_after_state checks passed",
        evidence=evidence,
    )


def _find_text_matches(ui_state: Mapping[str, Any], target_text: str) -> list[dict[str, Any]]:
    target = _normalize_text(target_text)
    matches: list[dict[str, Any]] = []

    page_summary = ui_state.get("page_summary")
    if isinstance(page_summary, Mapping):
        for index, text in enumerate(_strings(page_summary.get("top_texts", []))):
            if target in _normalize_text(text):
                matches.append({"source": "page_summary.top_texts", "index": index, "text": text})
        semantic_summary = page_summary.get("semantic_summary")
        if isinstance(semantic_summary, Mapping):
            for key in ("summary", "reason"):
                text = str(semantic_summary.get(key) or "")
                if target in _normalize_text(text):
                    matches.append({"source": f"page_summary.semantic_summary.{key}", "text": text})

    for index, candidate in enumerate(_mappings(ui_state.get("candidates", []))):
        text = str(candidate.get("text") or "")
        if target in _normalize_text(text):
            matches.append(
                {
                    "source": "candidates",
                    "index": index,
                    "id": candidate.get("id"),
                    "text": text,
                }
            )

    document_summary = ui_state.get("document_region_summary")
    if isinstance(document_summary, Mapping):
        for key in ("visible_title", "summary", "visible_text", "content"):
            text = str(document_summary.get(key) or "")
            if target in _normalize_text(text):
                matches.append({"source": f"document_region_summary.{key}", "text": text})
        for index, text in enumerate(_strings(document_summary.get("visible_texts", []))):
            if target in _normalize_text(text):
                matches.append(
                    {"source": "document_region_summary.visible_texts", "index": index, "text": text}
                )

    semantics = _page_semantics(ui_state)
    for key in ("summary", "reason"):
        text = str(semantics.get(key) or "")
        if target in _normalize_text(text):
            matches.append({"source": f"page_semantics.{key}", "text": text})
    for index, entity in enumerate(_mappings(semantics.get("entities", []))):
        text = str(entity.get("name") or "")
        if target in _normalize_text(text):
            matches.append({"source": "page_semantics.entities", "index": index, "text": text})

    return matches


def _page_semantics(ui_state: Mapping[str, Any]) -> dict[str, Any]:
    page_summary = ui_state.get("page_summary")
    if isinstance(page_summary, Mapping) and isinstance(page_summary.get("semantic_summary"), Mapping):
        return dict(page_summary["semantic_summary"])
    vlm_result = ui_state.get("vlm_result")
    if isinstance(vlm_result, Mapping):
        structured = vlm_result.get("structured")
        if isinstance(structured, Mapping):
            semantics = structured.get("page_semantics")
            if isinstance(semantics, Mapping):
                return dict(semantics)
    return {}


def _state_signature(ui_state: Mapping[str, Any]) -> dict[str, Any]:
    page_summary = ui_state.get("page_summary")
    candidates = list(_mappings(ui_state.get("candidates", [])))
    return {
        "top_texts": tuple(_strings(page_summary.get("top_texts", [])) if isinstance(page_summary, Mapping) else []),
        "candidate_count": len(candidates),
        "candidates": tuple(
            (
                str(candidate.get("id") or ""),
                str(candidate.get("text") or ""),
                tuple(candidate.get("center") or []),
            )
            for candidate in candidates[:30]
        ),
        "semantic_summary": (
            page_summary.get("semantic_summary") if isinstance(page_summary, Mapping) else None
        ),
    }


def _required_text(value: Any, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValidatorError(f"{field_name} cannot be empty")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_text(value: str) -> str:
    return "".join(str(value).lower().split())


def _strings(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _bool_field(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidatorError(f"{field_name} must be a boolean")
    return value


__all__ = [
    "VALIDATION_EXPECTATION_UNSUPPORTED",
    "VALIDATION_PAGE_NOT_CHANGED",
    "VALIDATION_TARGET_TEXT_NOT_FOUND",
    "VALIDATION_VLM_SEMANTIC_FAILED",
    "VALIDATOR_RESULT_SCHEMA_VERSION",
    "ValidatorError",
    "ValidatorResult",
    "validate_candidates_contain_text",
    "validate_chat_message_visible",
    "validate_expected_after_state",
    "validate_page_changed",
    "validate_text_visible",
    "validate_vlm_semantic_pass",
]
