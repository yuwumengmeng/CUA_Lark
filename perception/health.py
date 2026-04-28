"""Member A perception health assessment (Task 2.7).

Responsibility boundary: this module analyzes OCR/UIA/candidate quality and
flags perception failure modes so B/C can decide whether to trust candidates
or fall back to VLM visual search. It does not capture screenshots, run
OCR/UIA, build candidates, or call VLM APIs.

Public interfaces:
`assess_perception_health(...)`
    Main entry point. Takes raw OCR/UIA elements and built candidates,
    returns a PerceptionHealth with quality assessments, failure flags,
    and fallback suggestions.

`build_perception_health_summary(...)`
    Generate a human-readable summary string from PerceptionHealth.
"""

from __future__ import annotations

from typing import Any, Sequence

from schemas.perception_health import (
    COVERAGE_ADEQUATE,
    COVERAGE_NONE,
    COVERAGE_PARTIAL,
    FALLBACK_NONE,
    FALLBACK_RETRY_WITH_REFOCUS,
    FALLBACK_USE_OCR_ONLY,
    FALLBACK_USE_UIA_ONLY,
    FALLBACK_USE_VLM_VISUAL,
    FLAG_ALL_SHELL,
    FLAG_NO_CLICKABLE,
    FLAG_NO_INPUT_TARGET,
    FLAG_NO_SEARCH_TARGET,
    FLAG_OCR_LOW_CONFIDENCE,
    FLAG_OCR_SPARSE,
    FLAG_UIA_EMPTY,
    FLAG_UIA_SHELL_DOMINATED,
    HEALTH_DEGRADED,
    HEALTH_HEALTHY,
    HEALTH_UNHEALTHY,
    QUALITY_DEGRADED,
    QUALITY_GOOD,
    QUALITY_POOR,
    PerceptionHealth,
)
from vlm.rerank import SHELL_CONTROL_ROLES, SHELL_CONTROL_TEXTS


# ── Thresholds ─────────────────────────────────────────────────────────────

MIN_OCR_ELEMENTS_FOR_GOOD = 5
MIN_OCR_ELEMENTS_FOR_DEGRADED = 2
MIN_OCR_CONFIDENCE_FOR_GOOD = 0.5
MAX_SHELL_RATIO_FOR_GOOD = 0.3
MAX_SHELL_RATIO_FOR_DEGRADED = 0.6
MIN_CLICKABLE_FOR_ADEQUATE = 2
MIN_CLICKABLE_FOR_PARTIAL = 1


# ── Public API ─────────────────────────────────────────────────────────────

def assess_perception_health(
    *,
    ocr_elements: Sequence[Any] | None = None,
    uia_elements: Sequence[Any] | None = None,
    candidates: Sequence[Any] | None = None,
) -> PerceptionHealth:
    """Assess perception pipeline health from raw elements and candidates.

    Args:
        ocr_elements: Raw OCRTextBlock elements from OCR extraction.
        uia_elements: Raw UIAElement elements from UIA extraction.
        candidates: Built UICandidate list from Candidate Builder.

    Returns:
        PerceptionHealth with quality assessments and fallback suggestions.
    """
    ocr_list = list(ocr_elements or [])
    uia_list = list(uia_elements or [])
    candidate_list = list(candidates or [])

    # OCR quality
    ocr_count = len(ocr_list)
    ocr_avg_conf = _ocr_avg_confidence(ocr_list)
    ocr_quality = _assess_ocr_quality(ocr_count, ocr_avg_conf)

    # UIA quality
    uia_count = len(uia_list)
    shell_count = _count_shell_controls(uia_list)
    shell_ratio = shell_count / uia_count if uia_count > 0 else 0.0
    uia_quality = _assess_uia_quality(uia_count, shell_ratio)

    # Candidate coverage
    clickable_count = sum(1 for c in candidate_list if _attr(c, "clickable"))
    editable_count = sum(1 for c in candidate_list if _attr(c, "editable"))
    textless_count = sum(1 for c in candidate_list if not _attr(c, "text"))

    search_coverage = _assess_search_coverage(candidate_list)
    input_coverage = _assess_input_coverage(candidate_list)
    is_all_shell = _check_all_shell(candidate_list)

    # Failure flags
    failure_flags: list[str] = []

    if ocr_quality == QUALITY_POOR:
        failure_flags.append(FLAG_OCR_SPARSE)
    if ocr_avg_conf < MIN_OCR_CONFIDENCE_FOR_GOOD and ocr_count > 0:
        failure_flags.append(FLAG_OCR_LOW_CONFIDENCE)

    if uia_count == 0:
        failure_flags.append(FLAG_UIA_EMPTY)
    elif shell_ratio > MAX_SHELL_RATIO_FOR_DEGRADED:
        failure_flags.append(FLAG_UIA_SHELL_DOMINATED)

    if is_all_shell:
        failure_flags.append(FLAG_ALL_SHELL)
    if search_coverage == COVERAGE_NONE:
        failure_flags.append(FLAG_NO_SEARCH_TARGET)
    if input_coverage == COVERAGE_NONE:
        failure_flags.append(FLAG_NO_INPUT_TARGET)
    if clickable_count == 0:
        failure_flags.append(FLAG_NO_CLICKABLE)

    # Overall health
    overall = _compute_overall_health(
        ocr_quality=ocr_quality,
        uia_quality=uia_quality,
        clickable_count=clickable_count,
        failure_flags=failure_flags,
    )

    # Fallback suggestion
    fallback = _suggest_fallback(
        overall_health=overall,
        ocr_quality=ocr_quality,
        uia_quality=uia_quality,
        shell_ratio=shell_ratio,
        failure_flags=failure_flags,
    )

    # Summary
    summary = build_perception_health_summary(
        ocr_count=ocr_count,
        ocr_quality=ocr_quality,
        uia_count=uia_count,
        uia_quality=uia_quality,
        shell_count=shell_count,
        shell_ratio=shell_ratio,
        candidate_count=len(candidate_list),
        clickable_count=clickable_count,
        editable_count=editable_count,
        search_coverage=search_coverage,
        input_coverage=input_coverage,
        overall=overall,
        failure_flags=failure_flags,
        fallback=fallback,
    )

    return PerceptionHealth(
        ocr_element_count=ocr_count,
        ocr_quality=ocr_quality,
        uia_element_count=uia_count,
        uia_quality=uia_quality,
        shell_control_count=shell_count,
        shell_control_ratio=round(shell_ratio, 4),
        merged_candidate_count=len(candidate_list),
        clickable_count=clickable_count,
        editable_count=editable_count,
        textless_candidate_count=textless_count,
        search_target_coverage=search_coverage,
        input_target_coverage=input_coverage,
        overall_health=overall,
        failure_flags=failure_flags,
        suggested_fallback=fallback,
        summary=summary,
    )


def build_perception_health_summary(
    ocr_count: int,
    ocr_quality: str,
    uia_count: int,
    uia_quality: str,
    shell_count: int,
    shell_ratio: float,
    candidate_count: int,
    clickable_count: int,
    editable_count: int,
    search_coverage: str,
    input_coverage: str,
    overall: str,
    failure_flags: list[str],
    fallback: str,
) -> str:
    """Generate a human-readable perception health summary."""
    parts = [
        f"OCR:{ocr_count}e/{ocr_quality}",
        f"UIA:{uia_count}e/{uia_quality}",
        f"shell:{shell_count}({shell_ratio:.0%})",
        f"candidates:{candidate_count}(clickable:{clickable_count},editable:{editable_count})",
        f"search:{search_coverage}",
        f"input:{input_coverage}",
        f"health:{overall}",
    ]
    if failure_flags:
        parts.append(f"flags:[{','.join(failure_flags)}]")
    if fallback and fallback != FALLBACK_NONE:
        parts.append(f"fallback:{fallback}")
    return " ".join(parts)


# ── Internal assessment helpers ────────────────────────────────────────────

def _ocr_avg_confidence(ocr_elements: list[Any]) -> float:
    if not ocr_elements:
        return 0.0
    confidences = [getattr(e, "confidence", 0.0) for e in ocr_elements]
    valid = [c for c in confidences if isinstance(c, (int, float))]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def _assess_ocr_quality(count: int, avg_confidence: float) -> str:
    if count == 0:
        return QUALITY_POOR
    if count < MIN_OCR_ELEMENTS_FOR_DEGRADED:
        return QUALITY_POOR
    if count < MIN_OCR_ELEMENTS_FOR_GOOD:
        return QUALITY_DEGRADED
    if avg_confidence < MIN_OCR_CONFIDENCE_FOR_GOOD:
        return QUALITY_DEGRADED
    return QUALITY_GOOD


def _assess_uia_quality(count: int, shell_ratio: float) -> str:
    if count == 0:
        return QUALITY_POOR
    if shell_ratio > MAX_SHELL_RATIO_FOR_DEGRADED:
        return QUALITY_POOR
    if shell_ratio > MAX_SHELL_RATIO_FOR_GOOD:
        return QUALITY_DEGRADED
    return QUALITY_GOOD


def _count_shell_controls(uia_elements: list[Any]) -> int:
    return sum(1 for e in uia_elements if _is_uia_shell_control(e))


def _is_uia_shell_control(element: Any) -> bool:
    role = str(getattr(element, "role", "") or "").strip()
    if role in SHELL_CONTROL_ROLES:
        return True
    name = str(getattr(element, "name", "") or "").strip().lower()
    for token in SHELL_CONTROL_TEXTS:
        if token.lower() in name:
            return True
    return False


def _assess_search_coverage(candidates: list[Any]) -> str:
    search_signals = 0
    for c in candidates:
        text = str(_attr(c, "text") or "").lower()
        role = str(_attr(c, "role") or "").lower()
        editable = _attr(c, "editable")
        if editable:
            search_signals += 2
        if any(kw in text for kw in ("搜索", "search", "搜", "索")):
            search_signals += 2
        if any(kw in role for kw in ("edit", "search", "input")):
            search_signals += 1
    if search_signals >= 3:
        return COVERAGE_ADEQUATE
    if search_signals >= 1:
        return COVERAGE_PARTIAL
    return COVERAGE_NONE


def _assess_input_coverage(candidates: list[Any]) -> str:
    input_signals = 0
    for c in candidates:
        text = str(_attr(c, "text") or "").lower()
        role = str(_attr(c, "role") or "").lower()
        editable = _attr(c, "editable")
        if editable:
            input_signals += 2
        if any(kw in text for kw in ("输入", "input", "消息", "message", "发送")):
            input_signals += 1
        # Only match specific input-related roles, not generic "Text"
        if any(kw in role for kw in ("edit", "input", "textbox", "richedit")):
            input_signals += 1
    if input_signals >= 3:
        return COVERAGE_ADEQUATE
    if input_signals >= 1:
        return COVERAGE_PARTIAL
    return COVERAGE_NONE


def _check_all_shell(candidates: list[Any]) -> bool:
    if not candidates:
        return False
    clickable = [c for c in candidates if _attr(c, "clickable")]
    if not clickable:
        return True
    # Check if all clickable candidates are shell controls
    shell_clickable = sum(1 for c in clickable if _is_candidate_shell(c))
    return shell_clickable > 0 and shell_clickable == len(clickable)


def _is_candidate_shell(candidate: Any) -> bool:
    role = str(_attr(candidate, "role") or "").strip()
    if role in SHELL_CONTROL_ROLES:
        return True
    text = str(_attr(candidate, "text") or "").strip().lower()
    if text:
        for token in SHELL_CONTROL_TEXTS:
            if token.lower() in text:
                return True
    return False


def _compute_overall_health(
    ocr_quality: str,
    uia_quality: str,
    clickable_count: int,
    failure_flags: list[str],
) -> str:
    # Both poor = unhealthy
    if ocr_quality == QUALITY_POOR and uia_quality == QUALITY_POOR:
        return HEALTH_UNHEALTHY
    # Either poor with multiple failures = unhealthy
    if QUALITY_POOR in (ocr_quality, uia_quality) and len(failure_flags) >= 3:
        return HEALTH_UNHEALTHY
    # Either poor = degraded
    if QUALITY_POOR in (ocr_quality, uia_quality):
        return HEALTH_DEGRADED
    # Multiple failures = degraded
    if len(failure_flags) >= 2:
        return HEALTH_DEGRADED
    # Either degraded = degraded
    if QUALITY_DEGRADED in (ocr_quality, uia_quality):
        return HEALTH_DEGRADED
    # No clickable candidates = degraded
    if clickable_count == 0:
        return HEALTH_DEGRADED
    return HEALTH_HEALTHY


def _suggest_fallback(
    overall_health: str,
    ocr_quality: str,
    uia_quality: str,
    shell_ratio: float,
    failure_flags: list[str],
) -> str:
    if overall_health == HEALTH_HEALTHY:
        return FALLBACK_NONE

    # If UIA is all shell controls but OCR is good, prefer OCR
    if FLAG_UIA_SHELL_DOMINATED in failure_flags and ocr_quality == QUALITY_GOOD:
        return FALLBACK_USE_OCR_ONLY

    # If OCR is sparse but UIA is good, prefer UIA
    if FLAG_OCR_SPARSE in failure_flags and uia_quality == QUALITY_GOOD:
        return FALLBACK_USE_UIA_ONLY

    # If both are degraded, use VLM visual
    if overall_health in (HEALTH_DEGRADED, HEALTH_UNHEALTHY):
        if FLAG_NO_SEARCH_TARGET in failure_flags or FLAG_NO_INPUT_TARGET in failure_flags:
            return FALLBACK_USE_VLM_VISUAL
        if FLAG_NO_CLICKABLE in failure_flags or FLAG_ALL_SHELL in failure_flags:
            return FALLBACK_USE_VLM_VISUAL
        return FALLBACK_RETRY_WITH_REFOCUS

    return FALLBACK_USE_VLM_VISUAL


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Safe attribute access for dicts and objects."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
