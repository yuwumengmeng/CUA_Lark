"""Member A VLM provider base layer.

Responsibility boundary: this file defines the provider protocol, a disabled
fallback provider, and an OpenAI-compatible HTTP implementation for VLM calls.
It does not choose UI actions, move the mouse, or mutate OCR/UIA candidates.

Public interfaces:
`VLMRequest`
    Stable input envelope containing instruction, screenshot path, summary, and
    top candidates.
`VLMCallResult`
    Stable output envelope. `ok=False` means callers should use OCR/UIA/
    Candidate Builder fallback results.
`OpenAICompatibleVLMProvider`
    Sends chat completions requests with optional screenshot `image_url` blocks.
`build_default_vlm_provider(...)`
    Creates a configured provider or a disabled provider.
`safe_analyze(...)`
    Calls any provider and converts exceptions into fallback-safe results.

Data contract:
No provider returns API keys. Raw API responses are preserved only in
`raw_response` for debugging and later artifact storage.

Collaboration rules:
1. Provider failures must never block perception's main OCR/UIA/candidate path.
2. Later vendors should implement `analyze(request)` and return `VLMCallResult`.
3. Callers should pass top-k candidates, not the full raw UI state.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .context import build_vlm_context_payload
from .config import VLMConfig, load_vlm_config


class VLMProviderError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class VLMRequest:
    instruction: str
    screenshot_path: str | Path | None = None
    page_summary: Mapping[str, Any] | None = None
    candidates: Sequence[Mapping[str, Any]] | None = None
    max_candidates: int | None = None
    step_name: str | None = None
    step_goal: str | None = None
    region_hint: str | None = None

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            raise VLMProviderError("VLMRequest instruction cannot be empty")
        if self.max_candidates is not None and self.max_candidates <= 0:
            raise VLMProviderError("VLMRequest max_candidates must be positive")

    def compact_payload(self, *, default_max_candidates: int) -> dict[str, Any]:
        max_candidates = self.max_candidates or default_max_candidates
        return build_vlm_context_payload(
            instruction=self.instruction,
            page_summary=self.page_summary or {},
            candidates=self.candidates or [],
            max_candidates=max_candidates,
            step_name=self.step_name,
            step_goal=self.step_goal,
            region_hint=self.region_hint,
        )


@dataclass(frozen=True, slots=True)
class VLMCallResult:
    ok: bool
    provider: str
    model: str
    api_base: str
    content: str = ""
    raw_response: Mapping[str, Any] | None = None
    usage: Mapping[str, Any] | None = None
    finish_reason: str | None = None
    error: str | None = None
    error_type: str | None = None
    latency_ms: int | None = None
    attempts: int = 0
    fallback_used: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "content": self.content,
            "usage": dict(self.usage or {}),
            "finish_reason": self.finish_reason,
            "error": self.error,
            "error_type": self.error_type,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "fallback_used": self.fallback_used,
            "created_at": self.created_at,
        }
        if self.raw_response is not None:
            payload["raw_response"] = dict(self.raw_response)
        return payload


class VLMProvider(Protocol):
    def analyze(self, request: VLMRequest) -> VLMCallResult:
        ...


class DisabledVLMProvider:
    def __init__(
        self,
        *,
        reason: str = "VLM is disabled",
        config: VLMConfig | None = None,
    ) -> None:
        self.config = config or VLMConfig(enabled=False)
        self.reason = reason

    def analyze(self, request: VLMRequest) -> VLMCallResult:
        return VLMCallResult(
            ok=False,
            provider=self.config.provider,
            model=self.config.model,
            api_base=self.config.api_base,
            error=self.reason,
            error_type="disabled",
            attempts=0,
            fallback_used=True,
        )


class OpenAICompatibleVLMProvider:
    def __init__(self, config: VLMConfig | None = None) -> None:
        self.config = config or load_vlm_config()
        if not self.config.api_key:
            raise VLMProviderError("VLM API key is not configured")

    def analyze(self, request: VLMRequest) -> VLMCallResult:
        start = time.monotonic()
        attempts = 0
        last_error: str | None = None
        last_error_type: str | None = None
        max_attempts = self.config.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            try:
                data = self._post_chat_completions(request)
                latency_ms = int((time.monotonic() - start) * 1000)
                content, finish_reason, usage = _extract_completion(data)
                return VLMCallResult(
                    ok=True,
                    provider=self.config.provider,
                    model=self.config.model,
                    api_base=self.config.api_base,
                    content=content,
                    raw_response=data,
                    usage=usage,
                    finish_reason=finish_reason,
                    latency_ms=latency_ms,
                    attempts=attempts,
                    fallback_used=False,
                )
            except Exception as exc:
                last_error = str(exc)
                last_error_type = type(exc).__name__
                if attempt >= max_attempts or not _is_retryable_exception(exc):
                    break
                time.sleep(min(0.5 * attempt, 2.0))

        return VLMCallResult(
            ok=False,
            provider=self.config.provider,
            model=self.config.model,
            api_base=self.config.api_base,
            error=last_error or "VLM call failed",
            error_type=last_error_type or "VLMProviderError",
            latency_ms=int((time.monotonic() - start) * 1000),
            attempts=attempts,
            fallback_used=True,
        )

    def _post_chat_completions(self, request: VLMRequest) -> dict[str, Any]:
        body = {
            "model": self.config.model,
            "messages": _build_messages(request, self.config),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        http_request = Request(
            f"{self.config.api_base}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(http_request, timeout=self.config.timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
        parsed = json.loads(response_body)
        if not isinstance(parsed, dict):
            raise VLMProviderError("VLM response must be a JSON object")
        return parsed


def build_default_vlm_provider(config: VLMConfig | None = None) -> VLMProvider:
    resolved = config or load_vlm_config()
    if not resolved.enabled:
        return DisabledVLMProvider(reason="VLM is disabled by VLM_ENABLED", config=resolved)
    if not resolved.api_key:
        return DisabledVLMProvider(reason="VLM API key is not configured", config=resolved)
    if resolved.provider == "openai_compatible":
        return OpenAICompatibleVLMProvider(resolved)
    if resolved.provider == "disabled":
        return DisabledVLMProvider(reason="VLM provider is disabled", config=resolved)
    return DisabledVLMProvider(
        reason=f"Unsupported VLM provider: {resolved.provider}",
        config=resolved,
    )


def safe_analyze(provider: VLMProvider, request: VLMRequest) -> VLMCallResult:
    try:
        return provider.analyze(request)
    except Exception as exc:
        return VLMCallResult(
            ok=False,
            provider=provider.__class__.__name__,
            model="unknown",
            api_base="unknown",
            error=str(exc),
            error_type=type(exc).__name__,
            attempts=0,
            fallback_used=True,
        )


def _build_messages(request: VLMRequest, config: VLMConfig) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = []
    if request.screenshot_path is not None:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_path_to_data_url(request.screenshot_path)},
            }
        )
    user_content.append(
        {
            "type": "text",
            "text": _build_prompt_text(request, config),
        }
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a VLM perception helper for a desktop automation system "
                "that operates Feishu (Lark) desktop client. Your job is to:\n"
                "1. Identify the current page type.\n"
                "2. Determine which UI candidate best matches the current step goal.\n"
                "3. Rerank candidates by relevance to the step goal.\n"
                "4. Express uncertainty when you cannot confidently select a target.\n"
                "5. When no existing candidate matches, propose visual synthetic candidates.\n\n"
                "You MUST respond with a single JSON object and nothing else. "
                "Do NOT include markdown fences, commentary, or text outside the JSON.\n\n"
                "Required JSON schema:\n"
                "{\n"
                '  "page_type": "im|calendar|docs|base|vc|mail|search|settings|unknown",\n'
                '  "page_summary_enhancement": "brief description of what the current page shows",\n'
                '  "page_semantics": {\n'
                '    "domain": "im|calendar|docs|base|vc|mail|search|settings|unknown",\n'
                '    "view": "short generic view state, e.g. home/search/results/detail/editor/dialog",\n'
                '    "summary": "brief semantic summary",\n'
                '    "state_tags": ["search_active", "list_visible", "target_active"],\n'
                '    "signals": {\n'
                '      "search_available": true|false,\n'
                '      "search_active": true|false,\n'
                '      "results_available": true|false,\n'
                '      "list_available": true|false,\n'
                '      "detail_open": true|false,\n'
                '      "input_available": true|false,\n'
                '      "editor_available": true|false,\n'
                '      "modal_open": true|false,\n'
                '      "menu_open": true|false,\n'
                '      "loading": true|false,\n'
                '      "error": true|false,\n'
                '      "target_visible": true|false,\n'
                '      "target_active": true|false\n'
                '    },\n'
                '    "entities": [\n'
                '      {"type": "conversation|document|calendar_event|user|control|other", '
                '"name": "visible name", "visible": true, "matched_target": false, '
                '"confidence": 0.0-1.0}\n'
                '    ],\n'
                '    "available_actions": ["search", "open_result", "type", "send"],\n'
                '    "goal_progress": "unknown|not_started|in_progress|near_goal|achieved|blocked",\n'
                '    "blocking_issue": "modal|loading|error|null",\n'
                '    "confidence": 0.0-1.0,\n'
                '    "reason": "short reason"\n'
                '  },\n'
                '  "target_candidate_id": "elem_xxxx or null",\n'
                '  "reranked_candidate_ids": ["elem_xxxx", ...],\n'
                '  "confidence": 0.0-1.0,\n'
                '  "reason": "why this candidate was selected",\n'
                '  "uncertain": true|false,\n'
                '  "suggested_fallback": "rule_fallback|retry|wait_and_recheck|abort",\n'
                '  "visual_candidates": [\n'
                '    {\n'
                '      "id": "vlm_vis_0001",\n'
                '      "role": "icon|button|link|input|other",\n'
                '      "bbox": [x1, y1, x2, y2],\n'
                '      "center": [cx, cy],\n'
                '      "confidence": 0.0-1.0,\n'
                '      "reason": "why this visual element was proposed",\n'
                '      "coordinate_type": "screenshot_pixel"\n'
                '    }\n'
                '  ]\n'
                "}\n\n"
                "Rules:\n"
                "- target_candidate_id must be one of the provided candidate IDs, or null.\n"
                "- reranked_candidate_ids should list candidate IDs ordered by relevance.\n"
                "- Set uncertain=true when no candidate clearly matches the step goal.\n"
                "- Set confidence=0.0 when uncertain; set high confidence only when sure.\n"
                "- suggested_fallback tells the planner what to do when uncertain.\n"
                "- page_type must be one of: im, calendar, docs, base, vc, mail, "
                "search, settings, unknown.\n\n"
                "Page semantics rules:\n"
                "- Keep page_semantics generic and reusable across Lark products.\n"
                "- Use signals and state_tags instead of product-specific hard-coded fields.\n"
                "- Keep summary/reason short. Do not transcribe the whole page.\n"
                "- Include at most 5 entities and 6 available_actions.\n\n"
                "Visual candidate rules:\n"
                "- Only propose visual_candidates when no existing candidate matches the step goal.\n"
                "- visual_candidates are for icons/buttons without text (search icon, back arrow, "
                "settings gear, plus button, send button, more menu, etc.).\n"
                "- bbox coordinates are relative to the screenshot top-left corner (screenshot_pixel).\n"
                "- Keep visual_candidates minimal: at most 2 proposals per response.\n"
                "- Deprioritize browser shell controls (minimize/maximize/close, tab bar, address bar). "
                "These are NOT valid targets.\n"
                "- Do NOT propose visual_candidates for high-risk actions (delete, confirm, submit, "
                "permission changes). Only for low-risk navigation (search, back, more, cancel popup)."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _build_prompt_text(request: VLMRequest, config: VLMConfig) -> str:
    payload = request.compact_payload(default_max_candidates=config.max_candidates)
    parts = ["Analyze the screenshot and UI candidates for the current step."]
    if request.step_name:
        parts.append(f"Step name: {request.step_name}")
    if request.step_goal:
        parts.append(f"Step goal: {request.step_goal}")
    if request.region_hint:
        parts.append(f"Region hint: {request.region_hint}")
    parts.append(
        "If uncertain, set uncertain=true and suggest a fallback.\n"
        "Respond with ONLY the JSON object.\n"
        "Input JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return "\n".join(parts)


def _image_path_to_data_url(path: str | Path) -> str:
    image_path = Path(path)
    if not image_path.exists():
        raise VLMProviderError(f"Screenshot does not exist: {image_path}")
    mime_type = _mime_type_for_image(image_path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _mime_type_for_image(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _extract_completion(
    data: Mapping[str, Any],
) -> tuple[str, str | None, Mapping[str, Any] | None]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VLMProviderError("VLM response is missing choices")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise VLMProviderError("VLM choice must be a JSON object")
    message = first_choice.get("message", {})
    if not isinstance(message, Mapping):
        raise VLMProviderError("VLM response message must be a JSON object")
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
        ]
        content_text = "\n".join(part for part in text_parts if part)
    else:
        content_text = str(content or "")
    usage = data.get("usage") if isinstance(data.get("usage"), Mapping) else None
    finish_reason = first_choice.get("finish_reason")
    return content_text, str(finish_reason) if finish_reason is not None else None, usage


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, (TimeoutError, URLError))
