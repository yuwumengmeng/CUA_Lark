"""Member A VLM structured output schema and content parser.

Responsibility boundary: this file defines the stable structured output that
Member B can directly consume from VLM analysis results, and provides a parser
that extracts structured JSON from raw VLM text responses. It does not call VLM
APIs, make planner decisions, or execute actions.

Public interfaces:
`VLMStructuredOutput`
    Stable structured output with page_type, target_candidate_id,
    reranked_candidate_ids, confidence, reason, uncertain, and
    suggested_fallback. Member B reads these fields directly instead of
    parsing raw VLM text.
`parse_vlm_content(...)`
    Extract structured JSON from VLM response text. Returns a
    VLMStructuredOutput with sensible defaults when parsing fails.
`VLM_PAGE_TYPES`
    Allowed page_type values.

Data contract:
1. When `uncertain=True`, Member B should ignore `target_candidate_id` and
   fall back to rule-based candidate scoring.
2. `reranked_candidate_ids` may be empty when VLM cannot determine relevance.
3. `target_candidate_id` is always one of the input candidate IDs; if VLM
   returns an unknown ID, it is discarded and `uncertain` is set to True.
4. `confidence` is clamped to [0.0, 1.0].
5. `suggested_fallback` tells B what to do next when uncertain.

Collaboration rules:
1. Member B reads `vlm_result["structured"]` for the parsed output.
2. When `structured.uncertain` is True, B must use OCR/UIA/rule fallback.
3. The parser never raises; all errors produce a fallback-safe output.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from schemas import PageSemantics


VLM_PAGE_TYPES = frozenset({
    "im",
    "calendar",
    "docs",
    "base",
    "vc",
    "mail",
    "search",
    "settings",
    "unknown",
})

FALLBACK_RULE = "rule_fallback"
FALLBACK_RETRY = "retry"
FALLBACK_WAIT = "wait_and_recheck"
FALLBACK_ABORT = "abort"
SUPPORTED_FALLBACKS = frozenset({
    FALLBACK_RULE,
    FALLBACK_RETRY,
    FALLBACK_WAIT,
    FALLBACK_ABORT,
})


class VLMOutputError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class VLMStructuredOutput:
    page_type: str = "unknown"
    page_summary_enhancement: str = ""
    target_candidate_id: str | None = None
    reranked_candidate_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    uncertain: bool = True
    suggested_fallback: str = FALLBACK_RULE
    parse_ok: bool = False
    parse_error: str | None = None
    page_semantics: PageSemantics = field(default_factory=PageSemantics)
    visual_candidates: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        page_type = self.page_type.strip().lower() if self.page_type else "unknown"
        if page_type not in VLM_PAGE_TYPES:
            page_type = "unknown"
        object.__setattr__(self, "page_type", page_type)

        confidence = _clamp_float(self.confidence, 0.0, 1.0)
        object.__setattr__(self, "confidence", confidence)

        fallback = self.suggested_fallback.strip() if self.suggested_fallback else FALLBACK_RULE
        if fallback not in SUPPORTED_FALLBACKS:
            fallback = FALLBACK_RULE
        object.__setattr__(self, "suggested_fallback", fallback)

        reranked = [str(cid) for cid in self.reranked_candidate_ids if str(cid).strip()]
        object.__setattr__(self, "reranked_candidate_ids", reranked)

        if self.target_candidate_id is not None:
            tid = str(self.target_candidate_id).strip()
            object.__setattr__(self, "target_candidate_id", tid if tid else None)

        semantics = _normalize_page_semantics(
            self.page_semantics,
            page_type=page_type,
            summary=self.page_summary_enhancement,
            confidence=confidence,
            reason=self.reason,
        )
        object.__setattr__(self, "page_semantics", semantics)

        visual = [
            item for item in self.visual_candidates
            if isinstance(item, dict) and (item.get("bbox") is not None or item.get("center") is not None)
        ]
        object.__setattr__(self, "visual_candidates", visual)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VLMStructuredOutput":
        return cls(
            page_type=str(data.get("page_type", "unknown")),
            page_summary_enhancement=str(data.get("page_summary_enhancement", "")),
            target_candidate_id=data.get("target_candidate_id"),
            reranked_candidate_ids=list(data.get("reranked_candidate_ids") or []),
            confidence=_to_float(data.get("confidence", 0.0), "confidence"),
            reason=str(data.get("reason", "")),
            uncertain=_to_bool(data.get("uncertain", True), "uncertain"),
            suggested_fallback=str(data.get("suggested_fallback", FALLBACK_RULE)),
            parse_ok=_to_bool(data.get("parse_ok", False), "parse_ok"),
            parse_error=data.get("parse_error"),
            page_semantics=PageSemantics.from_dict(data.get("page_semantics")),
            visual_candidates=list(data.get("visual_candidates") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "page_type": self.page_type,
            "page_summary_enhancement": self.page_summary_enhancement,
            "target_candidate_id": self.target_candidate_id,
            "reranked_candidate_ids": self.reranked_candidate_ids,
            "confidence": self.confidence,
            "reason": self.reason,
            "uncertain": self.uncertain,
            "suggested_fallback": self.suggested_fallback,
            "parse_ok": self.parse_ok,
            "page_semantics": self.page_semantics.to_dict(),
        }
        if self.parse_error is not None:
            payload["parse_error"] = self.parse_error
        if self.visual_candidates:
            payload["visual_candidates"] = self.visual_candidates
        return payload

    def validate_candidate_ids(self, valid_ids: set[str]) -> "VLMStructuredOutput":
        if not valid_ids:
            return self

        target_id = self.target_candidate_id
        if target_id is not None and target_id not in valid_ids:
            return VLMStructuredOutput(
                page_type=self.page_type,
                page_summary_enhancement=self.page_summary_enhancement,
                target_candidate_id=None,
                reranked_candidate_ids=[
                    cid for cid in self.reranked_candidate_ids if cid in valid_ids
                ],
                confidence=min(self.confidence, 0.3),
                reason=self.reason,
                uncertain=True,
                suggested_fallback=FALLBACK_RULE,
                parse_ok=self.parse_ok,
                parse_error=f"target_candidate_id '{target_id}' not in valid candidate IDs",
                page_semantics=self.page_semantics,
                visual_candidates=self.visual_candidates,
            )

        reranked = [cid for cid in self.reranked_candidate_ids if cid in valid_ids]
        return VLMStructuredOutput(
            page_type=self.page_type,
            page_summary_enhancement=self.page_summary_enhancement,
            target_candidate_id=self.target_candidate_id,
            reranked_candidate_ids=reranked,
            confidence=self.confidence,
            reason=self.reason,
            uncertain=self.uncertain,
            suggested_fallback=self.suggested_fallback,
            parse_ok=self.parse_ok,
            parse_error=self.parse_error,
            page_semantics=self.page_semantics,
            visual_candidates=self.visual_candidates,
        )


def parse_vlm_content(
    content: str,
    *,
    valid_candidate_ids: Sequence[str] | None = None,
) -> VLMStructuredOutput:
    if not content or not content.strip():
        return _fallback_output("VLM response content is empty")

    json_text = _extract_json_block(content)
    if json_text is None:
        return _fallback_output("No JSON object found in VLM response content")

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return _fallback_output(f"VLM response JSON parse error: {exc}")

    if not isinstance(parsed, dict):
        return _fallback_output("VLM response JSON is not an object")

    output = VLMStructuredOutput.from_dict(parsed)
    output = VLMStructuredOutput(
        page_type=output.page_type,
        page_summary_enhancement=output.page_summary_enhancement,
        target_candidate_id=output.target_candidate_id,
        reranked_candidate_ids=output.reranked_candidate_ids,
        confidence=output.confidence,
        reason=output.reason,
        uncertain=output.uncertain,
        suggested_fallback=output.suggested_fallback,
        parse_ok=True,
        parse_error=None,
        page_semantics=output.page_semantics,
        visual_candidates=output.visual_candidates,
    )

    if valid_candidate_ids is not None:
        output = output.validate_candidate_ids(set(valid_candidate_ids))

    return output


def _extract_json_block(text: str) -> str | None:
    stripped = text.strip()

    if stripped.startswith("{"):
        depth = 0
        for index, char in enumerate(stripped):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return stripped[: index + 1]
        return stripped

    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    first_brace = text.find("{")
    if first_brace >= 0:
        remainder = text[first_brace:]
        depth = 0
        for index, char in enumerate(remainder):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return remainder[: index + 1]

    return None


def _fallback_output(reason: str) -> VLMStructuredOutput:
    return VLMStructuredOutput(
        page_type="unknown",
        page_summary_enhancement="",
        target_candidate_id=None,
        reranked_candidate_ids=[],
        confidence=0.0,
        reason=reason,
        uncertain=True,
        suggested_fallback=FALLBACK_RULE,
        parse_ok=False,
        parse_error=reason,
        page_semantics=PageSemantics(reason=reason),
    )


def _normalize_page_semantics(
    value: PageSemantics | Mapping[str, Any],
    *,
    page_type: str,
    summary: str,
    confidence: float,
    reason: str,
) -> PageSemantics:
    semantics = value if isinstance(value, PageSemantics) else PageSemantics.from_dict(value)
    if (
        semantics.domain != "unknown"
        and semantics.summary
        and semantics.confidence > 0
        and semantics.reason
    ):
        return semantics
    return PageSemantics(
        domain=semantics.domain if semantics.domain != "unknown" else page_type,
        view=semantics.view,
        summary=semantics.summary or summary,
        state_tags=semantics.state_tags,
        signals=semantics.signals,
        entities=semantics.entities,
        available_actions=semantics.available_actions,
        goal_progress=semantics.goal_progress,
        blocking_issue=semantics.blocking_issue,
        confidence=semantics.confidence if semantics.confidence > 0 else confidence,
        reason=semantics.reason or reason,
    )


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _to_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)
