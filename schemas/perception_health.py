"""Member A perception health diagnostics schema (Task 2.7).

Responsibility boundary: this file defines the structured diagnostics that
assess OCR/UIA/candidate quality and flag perception failure modes. It does
not run OCR/UIA, build candidates, or make planner decisions.

Public interfaces:
`PerceptionHealth`
    Structured assessment of perception pipeline health, including OCR/UIA
    quality, candidate coverage, failure flags, and fallback suggestions.

Data contract:
1. All fields have sensible defaults for healthy perception.
2. `failure_flags` is empty when perception is healthy.
3. `overall_health` summarizes the worst-case status across all checks.

Collaboration rules:
1. Produced by `perception/health.py` and surfaced via `get_ui_state()`.
2. B reads `perception_health` to decide whether to trust candidates or
   request VLM visual fallback.
3. When `overall_health` is "degraded" or "unhealthy", B should prefer
   VLM visual candidates over raw OCR/UIA candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


HEALTH_HEALTHY = "healthy"
HEALTH_DEGRADED = "degraded"
HEALTH_UNHEALTHY = "unhealthy"
HEALTH_LEVELS = frozenset({HEALTH_HEALTHY, HEALTH_DEGRADED, HEALTH_UNHEALTHY})

QUALITY_GOOD = "good"
QUALITY_DEGRADED = "degraded"
QUALITY_POOR = "poor"
QUALITY_LEVELS = frozenset({QUALITY_GOOD, QUALITY_DEGRADED, QUALITY_POOR})

COVERAGE_NONE = "none"
COVERAGE_PARTIAL = "partial"
COVERAGE_ADEQUATE = "adequate"
COVERAGE_LEVELS = frozenset({COVERAGE_NONE, COVERAGE_PARTIAL, COVERAGE_ADEQUATE})

# Failure flag constants
FLAG_OCR_SPARSE = "ocr_sparse"
FLAG_OCR_LOW_CONFIDENCE = "ocr_low_confidence"
FLAG_UIA_SHELL_DOMINATED = "uia_shell_dominated"
FLAG_UIA_EMPTY = "uia_empty"
FLAG_NO_SEARCH_TARGET = "no_search_target"
FLAG_NO_INPUT_TARGET = "no_input_target"
FLAG_NO_CLICKABLE = "no_clickable"
FLAG_ALL_SHELL = "all_shell"
FLAG_CANDIDATE_GAP = "candidate_gap"

FALLBACK_USE_VLM_VISUAL = "use_vlm_visual"
FALLBACK_USE_UIA_ONLY = "use_uia_only"
FALLBACK_USE_OCR_ONLY = "use_ocr_only"
FALLBACK_RETRY_WITH_REFOCUS = "retry_with_refocus"
FALLBACK_NONE = "none"


@dataclass(frozen=True, slots=True)
class PerceptionHealth:
    ocr_element_count: int = 0
    ocr_quality: str = QUALITY_GOOD
    uia_element_count: int = 0
    uia_quality: str = QUALITY_GOOD
    shell_control_count: int = 0
    shell_control_ratio: float = 0.0
    merged_candidate_count: int = 0
    clickable_count: int = 0
    editable_count: int = 0
    textless_candidate_count: int = 0
    search_target_coverage: str = COVERAGE_NONE
    input_target_coverage: str = COVERAGE_NONE
    overall_health: str = HEALTH_HEALTHY
    failure_flags: list[str] = field(default_factory=list)
    suggested_fallback: str = FALLBACK_NONE
    summary: str = ""

    def __post_init__(self) -> None:
        if self.ocr_quality not in QUALITY_LEVELS:
            object.__setattr__(self, "ocr_quality", QUALITY_GOOD)
        if self.uia_quality not in QUALITY_LEVELS:
            object.__setattr__(self, "uia_quality", QUALITY_GOOD)
        if self.search_target_coverage not in COVERAGE_LEVELS:
            object.__setattr__(self, "search_target_coverage", COVERAGE_NONE)
        if self.input_target_coverage not in COVERAGE_LEVELS:
            object.__setattr__(self, "input_target_coverage", COVERAGE_NONE)
        if self.overall_health not in HEALTH_LEVELS:
            object.__setattr__(self, "overall_health", HEALTH_HEALTHY)

        # Clamp ratios
        ratio = max(0.0, min(1.0, float(self.shell_control_ratio)))
        object.__setattr__(self, "shell_control_ratio", ratio)

        # Validate counts
        for field_name in (
            "ocr_element_count", "uia_element_count", "shell_control_count",
            "merged_candidate_count", "clickable_count", "editable_count",
            "textless_candidate_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                object.__setattr__(self, field_name, 0)

        # Validate failure flags
        valid_flags = {
            FLAG_OCR_SPARSE, FLAG_OCR_LOW_CONFIDENCE,
            FLAG_UIA_SHELL_DOMINATED, FLAG_UIA_EMPTY,
            FLAG_NO_SEARCH_TARGET, FLAG_NO_INPUT_TARGET,
            FLAG_NO_CLICKABLE, FLAG_ALL_SHELL, FLAG_CANDIDATE_GAP,
        }
        flags = [f for f in self.failure_flags if f in valid_flags]
        object.__setattr__(self, "failure_flags", flags)

        if self.suggested_fallback not in {
            FALLBACK_USE_VLM_VISUAL, FALLBACK_USE_UIA_ONLY,
            FALLBACK_USE_OCR_ONLY, FALLBACK_RETRY_WITH_REFOCUS, FALLBACK_NONE,
        }:
            object.__setattr__(self, "suggested_fallback", FALLBACK_NONE)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PerceptionHealth":
        return cls(
            ocr_element_count=_int_field(data.get("ocr_element_count", 0)),
            ocr_quality=str(data.get("ocr_quality", QUALITY_GOOD)),
            uia_element_count=_int_field(data.get("uia_element_count", 0)),
            uia_quality=str(data.get("uia_quality", QUALITY_GOOD)),
            shell_control_count=_int_field(data.get("shell_control_count", 0)),
            shell_control_ratio=float(data.get("shell_control_ratio", 0.0)),
            merged_candidate_count=_int_field(data.get("merged_candidate_count", 0)),
            clickable_count=_int_field(data.get("clickable_count", 0)),
            editable_count=_int_field(data.get("editable_count", 0)),
            textless_candidate_count=_int_field(data.get("textless_candidate_count", 0)),
            search_target_coverage=str(data.get("search_target_coverage", COVERAGE_NONE)),
            input_target_coverage=str(data.get("input_target_coverage", COVERAGE_NONE)),
            overall_health=str(data.get("overall_health", HEALTH_HEALTHY)),
            failure_flags=list(data.get("failure_flags", [])),
            suggested_fallback=str(data.get("suggested_fallback", FALLBACK_NONE)),
            summary=str(data.get("summary", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ocr_element_count": self.ocr_element_count,
            "ocr_quality": self.ocr_quality,
            "uia_element_count": self.uia_element_count,
            "uia_quality": self.uia_quality,
            "shell_control_count": self.shell_control_count,
            "shell_control_ratio": self.shell_control_ratio,
            "merged_candidate_count": self.merged_candidate_count,
            "clickable_count": self.clickable_count,
            "editable_count": self.editable_count,
            "textless_candidate_count": self.textless_candidate_count,
            "search_target_coverage": self.search_target_coverage,
            "input_target_coverage": self.input_target_coverage,
            "overall_health": self.overall_health,
            "failure_flags": list(self.failure_flags),
            "suggested_fallback": self.suggested_fallback,
            "summary": self.summary,
        }


def _int_field(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)
