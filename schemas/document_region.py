"""Member A document body region VLM reading schema.

Responsibility boundary: this file defines the stable structured output of
VLM-based document body region reading (Task 2.6). It does not perform
screenshot cropping, VLM calls, or any other IO.

Public interfaces:
`DocumentRegionSummary`
    Structured summary of a document's body region, produced by cropping
    the body area from the full screenshot and sending it to a VLM for
    semantic understanding.

Data contract:
1. All text fields default to empty; callers must check `error` first.
2. `visible_texts` is the primary field consumed by B's _extend_semantic_texts
   and C's validator for text visibility checks.
3. When `error` is non-null, all other fields should be treated as unreliable.

Collaboration rules:
1. Produced by `vlm/region_reader.py` and integrated into `get_ui_state()` output.
2. B reads `ui_state["document_region_summary"]` via `_visible_texts()` and `update_ui_state()`.
3. C's validator reads specific keys: visible_title, summary, visible_text, content, visible_texts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .ui_candidate import UISchemaError


@dataclass(frozen=True, slots=True)
class DocumentRegionSummary:
    visible_title: str = ""
    visible_sections: list[str] = field(default_factory=list)
    visible_texts: list[str] = field(default_factory=list)
    target_keywords_found: list[str] = field(default_factory=list)
    is_edit_mode: bool = False
    is_target_document: bool = False
    body_content_summary: str = ""
    confidence: float = 0.0
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.visible_sections, list):
            raise UISchemaError("DocumentRegionSummary visible_sections must be a list")
        if not isinstance(self.visible_texts, list):
            raise UISchemaError("DocumentRegionSummary visible_texts must be a list")
        if not isinstance(self.target_keywords_found, list):
            raise UISchemaError("DocumentRegionSummary target_keywords_found must be a list")
        if not isinstance(self.is_edit_mode, bool):
            raise UISchemaError("DocumentRegionSummary is_edit_mode must be a boolean")
        if not isinstance(self.is_target_document, bool):
            raise UISchemaError("DocumentRegionSummary is_target_document must be a boolean")
        confidence = float(self.confidence)
        if confidence < 0.0 or confidence > 1.0:
            confidence = max(0.0, min(1.0, confidence))
        object.__setattr__(self, "confidence", confidence)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DocumentRegionSummary":
        return cls(
            visible_title=str(data.get("visible_title", "")),
            visible_sections=_str_list(data.get("visible_sections", [])),
            visible_texts=_str_list(data.get("visible_texts", [])),
            target_keywords_found=_str_list(data.get("target_keywords_found", [])),
            is_edit_mode=bool(data.get("is_edit_mode", False)),
            is_target_document=bool(data.get("is_target_document", False)),
            body_content_summary=str(data.get("body_content_summary", "")),
            confidence=float(data.get("confidence", 0.0)),
            error=data.get("error") if data.get("error") else None,
        )

    @classmethod
    def empty(cls, *, error: str | None = None) -> "DocumentRegionSummary":
        return cls(error=error)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "visible_title": self.visible_title,
            "visible_sections": list(self.visible_sections),
            "visible_texts": list(self.visible_texts),
            "target_keywords_found": list(self.target_keywords_found),
            "is_edit_mode": self.is_edit_mode,
            "is_target_document": self.is_target_document,
            "body_content_summary": self.body_content_summary,
            "confidence": self.confidence,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            result.append(text)
    return result
