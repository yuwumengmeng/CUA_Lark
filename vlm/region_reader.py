"""Member A document body region VLM reader (Task 2.6).

Responsibility boundary: this module handles rough region segmentation of a
screenshot, cropping the document body area, and sending it to a VLM for
semantic understanding. It does not capture screenshots, run OCR/UIA, or
make planner decisions.

Public interfaces:
`read_document_region(...)`
    Main entry point. Estimates the document body region, crops it from the
    screenshot, sends the cropped image to VLM, and returns a structured
    DocumentRegionSummary.
`estimate_doc_body_bbox(...)`
    Heuristic-based estimation of the document body region from screenshot
    dimensions. Returns (x1, y1, x2, y2) or None.
`crop_screenshot(...)`
    Crop a region from a screenshot image and save to a new file.

Approach:
1. Estimate the body region using known Feishu Docs layout heuristics.
2. Crop the body region from the screenshot.
3. Send the cropped image to a VLM with a focused prompt about document content.
4. Parse the VLM response into a structured DocumentRegionSummary.
5. Fall back gracefully on any failure — document_region_summary is optional.

Region estimation heuristics (Feishu Docs in standard window):
- Left navigation sidebar: ~64px
- Middle doc list / TOC panel: ~300px
- Right body area: remaining width
- Top toolbar: ~56px
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from schemas.document_region import DocumentRegionSummary
from vlm.config import VLMConfig, load_vlm_config
from vlm.provider import (
    VLMProvider,
    VLMRequest,
    build_default_vlm_provider,
    safe_analyze,
)

# ── Heuristic layout constants (Feishu Docs in standard 1920×1080) ──────────

LEFT_NAV_WIDTH = 64
MIDDLE_PANEL_WIDTH = 300
TOP_TOOLBAR_HEIGHT = 56

# Minimum crop dimensions to be considered valid
MIN_CROP_WIDTH = 200
MIN_CROP_HEIGHT = 200

# Right margin to exclude scrollbar area
RIGHT_MARGIN = 16

# ── VLM prompt ─────────────────────────────────────────────────────────────

_DOC_REGION_SYSTEM_PROMPT = """You are a document body region reader for a desktop automation system.
Your task is to analyze a cropped screenshot of a document body area from Feishu (Lark) Docs.

Read the visible document content and return a single JSON object. Do NOT include markdown fences, commentary, or text outside the JSON.

Required JSON schema:
{
  "visible_title": "the document title visible in the body area",
  "visible_sections": ["section 1 heading", "section 2 heading"],
  "visible_texts": ["representative text snippet 1", "snippet 2"],
  "target_keywords_found": ["keyword1", "keyword2"],
  "is_edit_mode": true|false,
  "is_target_document": true|false,
  "body_content_summary": "brief summary of what the visible body area shows",
  "confidence": 0.0-1.0,
  "error": null
}

Rules:
- visible_title: extract the main document title if visible in the body area.
- visible_sections: list visible chapter/section headings (not full text).
- visible_texts: list 5-10 representative text snippets visible in the body area, each under 120 chars.
- target_keywords_found: only include keywords that are actually visible in the body text.
- is_edit_mode: true if the document appears to be in edit/editing mode (cursor, toolbar active, etc.).
- is_target_document: true if the visible content suggests this is the document being searched for.
- body_content_summary: a 1-2 sentence description of what the body area currently displays.
- confidence: how confident you are in this assessment (0.0-1.0).
- error: null unless you cannot read the content at all; then set to a brief reason and set confidence to 0.
"""


def read_document_region(
    screenshot_path: str | Path,
    *,
    vlm_config: VLMConfig | None = None,
    vlm_provider: VLMProvider | None = None,
    step_goal: str | None = None,
    target_document_title: str | None = None,
    artifact_dir: str | Path | None = None,
) -> DocumentRegionSummary:
    """Read the document body region from a screenshot using VLM.

    Estimates the body region using heuristics, crops it, and sends the
    cropped image to a VLM for semantic understanding.

    Args:
        screenshot_path: Path to the full screenshot image.
        vlm_config: Optional VLM configuration. Loaded from env if omitted.
        vlm_provider: Optional pre-built VLM provider. Built from config if omitted.
        step_goal: Current step goal for context (e.g. "verify document content").
        target_document_title: Optional target doc title for keyword matching.
        artifact_dir: Optional directory to save the cropped screenshot.

    Returns:
        DocumentRegionSummary with VLM reading results. Returns empty
        summary with error field set on any failure.
    """
    screenshot_path = Path(screenshot_path)

    if not screenshot_path.exists():
        return DocumentRegionSummary.empty(
            error=f"Screenshot not found: {screenshot_path}"
        )

    try:
        if vlm_config is not None:
            resolved_config = vlm_config
        elif vlm_provider is not None:
            resolved_config = VLMConfig(enabled=True, api_key="provided-provider")
        else:
            resolved_config = load_vlm_config()
    except Exception as exc:
        return DocumentRegionSummary.empty(error=f"VLM config load failed: {exc}")

    if vlm_provider is None and (
        not resolved_config.enabled or not resolved_config.api_key
    ):
        return DocumentRegionSummary.empty(error="VLM is disabled or API key missing")

    try:
        provider = vlm_provider or build_default_vlm_provider(resolved_config)
    except Exception as exc:
        return DocumentRegionSummary.empty(error=f"VLM provider build failed: {exc}")

    image_width, image_height = _read_image_size(screenshot_path)
    if image_width is None:
        return DocumentRegionSummary.empty(error="Cannot read screenshot dimensions")

    bbox = estimate_doc_body_bbox(image_width, image_height)
    if bbox is None:
        return DocumentRegionSummary.empty(
            error=f"Screenshot too small for body region estimation ({image_width}x{image_height})"
        )

    cropped_path = _crop_image(screenshot_path, bbox, artifact_dir)

    instruction = _build_region_instruction(
        step_goal=step_goal,
        target_document_title=target_document_title,
    )

    try:
        result = safe_analyze(
            provider,
            VLMRequest(
                instruction=instruction,
                screenshot_path=cropped_path,
            ),
        )
    except Exception as exc:
        return DocumentRegionSummary.empty(error=f"VLM call failed: {exc}")

    if not result.ok:
        return DocumentRegionSummary.empty(
            error=f"VLM call failed: {result.error or 'unknown error'}"
        )

    return _parse_region_result(result.content)


def estimate_doc_body_bbox(
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Estimate the document body region bbox from screenshot dimensions.

    Uses heuristics tuned for Feishu Docs layout:
    - Left nav (~64px) + middle panel (~300px) = left gutter of ~364px
    - Top toolbar: ~56px
    - Right margin: ~16px (scrollbar)

    Returns (x1, y1, x2, y2) in pixel coordinates, or None if the image
    is too small for reliable estimation.
    """
    if image_width < LEFT_NAV_WIDTH + MIDDLE_PANEL_WIDTH + MIN_CROP_WIDTH:
        return None
    if image_height < TOP_TOOLBAR_HEIGHT + MIN_CROP_HEIGHT:
        return None

    x1 = LEFT_NAV_WIDTH + MIDDLE_PANEL_WIDTH
    y1 = TOP_TOOLBAR_HEIGHT
    x2 = image_width - RIGHT_MARGIN
    y2 = image_height

    return (x1, y1, x2, y2)


# ── Internal helpers ───────────────────────────────────────────────────────

def _read_image_size(path: Path) -> tuple[int | None, int | None]:
    """Read image dimensions without heavy dependencies.

    Uses Pillow if available, otherwise falls back to a minimal PNG header
    parser. Returns (width, height) or (None, None) on failure.
    """
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        pass

    try:
        return _read_png_size(path)
    except Exception:
        return None, None


def _read_png_size(path: Path) -> tuple[int, int]:
    """Read PNG IHDR chunk to extract dimensions."""
    with open(path, "rb") as fh:
        header = fh.read(8)
        if header[:4] != b"\x89PNG":
            raise ValueError("Not a PNG file")
        fh.read(4)  # chunk length
        fh.read(4)  # IHDR
        width = int.from_bytes(fh.read(4), "big")
        height = int.from_bytes(fh.read(4), "big")
        return width, height


def _crop_image(
    screenshot_path: Path,
    bbox: tuple[int, int, int, int],
    artifact_dir: str | Path | None,
) -> Path:
    """Crop the image to the given bbox and save to artifact_dir.

    If artifact_dir is None, saves next to the original with a _crop suffix.
    Falls back to the original path if cropping fails.
    """
    x1, y1, x2, y2 = bbox
    try:
        from PIL import Image

        with Image.open(screenshot_path) as img:
            cropped = img.crop((x1, y1, x2, y2))
    except Exception:
        return screenshot_path

    if artifact_dir is not None:
        out_dir = Path(artifact_dir)
    else:
        out_dir = screenshot_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = screenshot_path.stem
    out_path = out_dir / f"{stem}_doc_body_crop.png"
    try:
        cropped.save(str(out_path), "PNG")
        return out_path
    except Exception:
        return screenshot_path


def _build_region_instruction(
    step_goal: str | None,
    target_document_title: str | None,
) -> str:
    parts = ["Analyze the document body area in this cropped screenshot."]
    if target_document_title:
        parts.append(f"Target document title: \"{target_document_title}\".")
        parts.append(
            "Check if the visible title or content matches this target."
        )
    if step_goal:
        parts.append(f"Current step goal: {step_goal}.")
    parts.append(
        "Return a single JSON object describing the visible document content."
    )
    return " ".join(parts)


def _parse_region_result(content: str) -> DocumentRegionSummary:
    """Parse VLM response into DocumentRegionSummary. Never raises."""
    if not content or not content.strip():
        return DocumentRegionSummary.empty(error="VLM returned empty content")

    json_text = _extract_json(content)
    if json_text is None:
        return DocumentRegionSummary.empty(error="No JSON found in VLM response")

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return DocumentRegionSummary.empty(error=f"JSON parse error: {exc}")

    if not isinstance(parsed, dict):
        return DocumentRegionSummary.empty(error="VLM response is not a JSON object")

    vlm_error = parsed.get("error")
    if vlm_error and isinstance(vlm_error, str) and vlm_error.strip():
        return DocumentRegionSummary(
            visible_title=str(parsed.get("visible_title", "")),
            visible_sections=_parse_str_list(parsed.get("visible_sections")),
            visible_texts=_parse_str_list(parsed.get("visible_texts")),
            target_keywords_found=_parse_str_list(parsed.get("target_keywords_found")),
            is_edit_mode=bool(parsed.get("is_edit_mode")),
            is_target_document=bool(parsed.get("is_target_document")),
            body_content_summary=str(parsed.get("body_content_summary", "")),
            confidence=float(parsed.get("confidence", 0.0)),
            error=None,  # VLM error is informational, not a system error
        )

    try:
        return DocumentRegionSummary.from_dict(parsed)
    except Exception as exc:
        return DocumentRegionSummary.empty(error=f"Schema validation error: {exc}")


def _parse_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _extract_json(text: str) -> str | None:
    """Extract JSON object from text. Same logic as vlm/output_schema.py."""
    stripped = text.strip()
    if stripped.startswith("{"):
        depth = 0
        for i, ch in enumerate(stripped):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return stripped[: i + 1]
        return stripped

    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    first_brace = text.find("{")
    if first_brace >= 0:
        remainder = text[first_brace:]
        depth = 0
        for i, ch in enumerate(remainder):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return remainder[: i + 1]

    return None


def _build_doc_region_messages(
    screenshot_path: str | Path,
    instruction: str,
) -> list[dict[str, Any]]:
    """Build messages for the document region VLM call.

    Exported for use by custom provider implementations that need to
    construct messages manually.
    """
    data_url = _image_path_to_data_url(screenshot_path)
    return [
        {"role": "system", "content": _DOC_REGION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": instruction},
            ],
        },
    ]


def _image_path_to_data_url(path: str | Path) -> str:
    """Convert a local image path to a base64 data URL."""
    path = Path(path)
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    suffix = path.suffix.lower().lstrip(".")
    mime = "png" if suffix in ("png", "") else suffix
    return f"data:image/{mime};base64,{encoded}"
