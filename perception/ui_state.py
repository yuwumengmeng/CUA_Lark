"""Member A get_ui_state perception entrypoint.

Responsibility boundary: this file chains screenshot capture, OCR, UIA,
Candidate Builder, optional VLM enrichment, and UI state JSON persistence into
one stable entrypoint. It does not make planner decisions, execute clicks, or
optimize OCR/UIA recognition quality.

Public interfaces:
`get_ui_state(...)`
    Return a plain dict for B/C. Required top-level fields remain
    `screenshot_path / candidates / page_summary`; `vlm_result` is optional.
`collect_ui_state(...)`
    Same flow, but returns typed `UIState` for tests and later extensions.

Execution flow:
```text
capture screenshot
    -> extract_ocr_elements
    -> extract_uia_elements
    -> build_ui_state_from_elements
    -> optionally safe VLM analyze
    -> save_ui_state_json
    -> return ui_state
```

Fallback contract: VLM is opt-in. If VLM is disabled, missing an API key, or
raises an error, OCR/UIA/Candidate Builder results are still returned. When
present, `vlm_result.ok == False` and `fallback_used == True` is the signal for
B/C to ignore VLM output and keep using the original candidates.

Coordinate rule: candidate `bbox/center` remains screen absolute coordinates
and can be passed directly to the executor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from capture import ScreenshotCaptureConfig, ScreenshotCapturer
from schemas import ScreenMeta, UICandidate, UIState, save_screen_meta_json, save_ui_state_json
from schemas.document_region import DocumentRegionSummary
from schemas.perception_health import PerceptionHealth
from vlm import (
    VLMConfig,
    VLMProvider,
    VLMRequest,
    VLMStructuredOutput,
    build_default_vlm_provider,
    build_visual_candidates,
    load_vlm_config,
    parse_vlm_content,
    rerank_candidates,
    safe_analyze,
)
from vlm.region_reader import read_document_region

from .candidate_builder import CandidateBuilderConfig, build_ui_state_from_elements
from .ocr import OCRBackend, OCRConfig, extract_ocr_elements
from .uia import UIABackend, UIAConfig, extract_uia_elements


class UIStateCollectionError(ValueError):
    pass


DEFAULT_VLM_UI_STATE_INSTRUCTION = (
    "Identify the current page type and select the best UI candidate "
    "for the current step goal. Return structured JSON."
)


def get_ui_state(
    *,
    run_id: str,
    step_idx: int,
    phase: str = "before",
    artifact_root: str | Path | None = None,
    monitor_index: int = 1,
    capture_window_title: str | None = "飞书",
    maximize_capture_window: bool = True,
    focus_capture_window: bool = True,
    capture_focus_delay_seconds: float = 0.8,
    window: Any = None,
    window_title: str | None = "飞书",
    save_json: bool = True,
    screenshot_capturer: ScreenshotCapturer | None = None,
    ocr_config: OCRConfig | None = None,
    ocr_backend: OCRBackend | None = None,
    uia_config: UIAConfig | None = None,
    uia_backend: UIABackend | None = None,
    candidate_config: CandidateBuilderConfig | None = None,
    enable_vlm: bool | None = None,
    vlm_instruction: str = DEFAULT_VLM_UI_STATE_INSTRUCTION,
    vlm_config: VLMConfig | None = None,
    vlm_provider: VLMProvider | None = None,
    step_name: str | None = None,
    step_goal: str | None = None,
    region_hint: str | None = None,
    enable_document_region: bool = False,
    target_document_title: str | None = None,
) -> dict[str, Any]:
    ui_state = collect_ui_state(
        run_id=run_id,
        step_idx=step_idx,
        phase=phase,
        artifact_root=artifact_root,
        monitor_index=monitor_index,
        capture_window_title=capture_window_title,
        maximize_capture_window=maximize_capture_window,
        focus_capture_window=focus_capture_window,
        capture_focus_delay_seconds=capture_focus_delay_seconds,
        window=window,
        window_title=window_title,
        save_json=save_json,
        screenshot_capturer=screenshot_capturer,
        ocr_config=ocr_config,
        ocr_backend=ocr_backend,
        uia_config=uia_config,
        uia_backend=uia_backend,
        candidate_config=candidate_config,
        enable_vlm=enable_vlm,
        vlm_instruction=vlm_instruction,
        vlm_config=vlm_config,
        vlm_provider=vlm_provider,
        step_name=step_name,
        step_goal=step_goal,
        region_hint=region_hint,
        enable_document_region=enable_document_region,
        target_document_title=target_document_title,
    )
    return ui_state.to_dict()


def collect_ui_state(
    *,
    run_id: str,
    step_idx: int,
    phase: str = "before",
    artifact_root: str | Path | None = None,
    monitor_index: int = 1,
    capture_window_title: str | None = "飞书",
    maximize_capture_window: bool = True,
    focus_capture_window: bool = True,
    capture_focus_delay_seconds: float = 0.8,
    window: Any = None,
    window_title: str | None = "飞书",
    save_json: bool = True,
    screenshot_capturer: ScreenshotCapturer | None = None,
    ocr_config: OCRConfig | None = None,
    ocr_backend: OCRBackend | None = None,
    uia_config: UIAConfig | None = None,
    uia_backend: UIABackend | None = None,
    candidate_config: CandidateBuilderConfig | None = None,
    enable_vlm: bool | None = None,
    vlm_instruction: str = DEFAULT_VLM_UI_STATE_INSTRUCTION,
    vlm_config: VLMConfig | None = None,
    vlm_provider: VLMProvider | None = None,
    step_name: str | None = None,
    step_goal: str | None = None,
    region_hint: str | None = None,
    enable_document_region: bool = False,
    target_document_title: str | None = None,
) -> UIState:
    capturer = screenshot_capturer or ScreenshotCapturer(
        config=ScreenshotCaptureConfig(
            artifact_root=_artifact_root(artifact_root),
            monitor_index=monitor_index,
            window_title=capture_window_title,
            focus_window=focus_capture_window,
            maximize_window=maximize_capture_window,
            focus_delay_seconds=capture_focus_delay_seconds,
        )
    )
    screenshot = capturer.capture_screenshot(
        run_id=run_id,
        step_idx=step_idx,
        phase=phase,  # type: ignore[arg-type]
    )

    ocr_elements = extract_ocr_elements(
        screenshot.screenshot_path,
        config=ocr_config,
        backend=ocr_backend,
    )
    uia_elements = extract_uia_elements(
        window,
        config=uia_config or _default_uia_config(window_title),
        backend=uia_backend,
    )
    ui_state = build_ui_state_from_elements(
        screenshot_path=screenshot.screenshot_path,
        ocr_elements=ocr_elements,
        uia_elements=uia_elements,
        screen_origin=screenshot.screen_origin,
        config=candidate_config,
    )
    ui_state = _with_screen_meta(ui_state, screenshot)

    perception_health = _assess_perception_health(
        ocr_elements=ocr_elements,
        uia_elements=uia_elements,
        candidates=ui_state.candidates,
    )
    candidate_stats = _build_candidate_stats(perception_health, ui_state.candidates)
    ui_state = UIState(
        screenshot_path=ui_state.screenshot_path,
        candidates=ui_state.candidates,
        page_summary=ui_state.page_summary,
        vlm_result=ui_state.vlm_result,
        screen_meta=ui_state.screen_meta,
        perception_health=perception_health.to_dict(),
        candidate_stats=candidate_stats,
    )

    resolved_vlm_config, vlm_config_error = _resolve_vlm_config(vlm_config)
    if _should_collect_vlm_result(
        enable_vlm=enable_vlm,
        vlm_provider=vlm_provider,
        vlm_config=resolved_vlm_config,
    ):
        if vlm_config_error is not None and vlm_provider is None:
            vlm_result = _vlm_error_result(
                error=vlm_config_error,
                vlm_config=resolved_vlm_config,
            )
        else:
            vlm_instruction = _enrich_vlm_instruction_with_health(
                base_instruction=vlm_instruction,
                perception_health=perception_health,
            )
            vlm_result = _collect_vlm_result(
                ui_state=ui_state,
                instruction=vlm_instruction,
                vlm_provider=vlm_provider,
                vlm_config=resolved_vlm_config,
                step_name=step_name,
                step_goal=step_goal,
                region_hint=region_hint,
            )
        ui_state = UIState(
            screenshot_path=ui_state.screenshot_path,
            candidates=ui_state.candidates,
            page_summary=ui_state.page_summary,
            vlm_result=vlm_result,
            screen_meta=ui_state.screen_meta,
            perception_health=ui_state.perception_health,
            candidate_stats=ui_state.candidate_stats,
        )
        ui_state = _apply_vlm_rerank(ui_state, vlm_result, screenshot=screenshot)

    if enable_document_region and _should_read_document_region(ui_state):
        doc_summary = _collect_document_region_summary(
            ui_state=ui_state,
            vlm_config=resolved_vlm_config,
            vlm_provider=vlm_provider,
            step_goal=step_goal,
            target_document_title=target_document_title,
            artifact_root=_artifact_root(artifact_root, capturer) if save_json else None,
        )
        ui_state = UIState(
            screenshot_path=ui_state.screenshot_path,
            candidates=ui_state.candidates,
            page_summary=ui_state.page_summary,
            vlm_result=ui_state.vlm_result,
            screen_meta=ui_state.screen_meta,
            document_region_summary=doc_summary.to_dict() if doc_summary is not None else None,
            perception_health=ui_state.perception_health,
            candidate_stats=ui_state.candidate_stats,
        )

    if save_json:
        artifact_root_path = _artifact_root(artifact_root, capturer)
        save_ui_state_json(
            ui_state,
            run_id=screenshot.run_id,
            step_idx=screenshot.step_idx,
            phase=screenshot.phase,
            artifact_root=artifact_root_path,
        )
        if ui_state.screen_meta is not None:
            save_screen_meta_json(
                ui_state.screen_meta,
                run_id=screenshot.run_id,
                step_idx=screenshot.step_idx,
                phase=screenshot.phase,
                artifact_root=artifact_root_path,
            )

    return ui_state


def _default_uia_config(window_title: str | None) -> UIAConfig:
    if window_title is None:
        return UIAConfig()
    return UIAConfig(window_title=window_title)


def _artifact_root(
    artifact_root: str | Path | None,
    capturer: ScreenshotCapturer | None = None,
) -> Path:
    if artifact_root is not None:
        return Path(artifact_root)
    if capturer is not None:
        return Path(capturer.config.artifact_root)
    return Path("artifacts") / "runs"


def _resolve_vlm_config(vlm_config: VLMConfig | None) -> tuple[VLMConfig, Exception | None]:
    if vlm_config is not None:
        return vlm_config, None
    try:
        return load_vlm_config(), None
    except Exception as exc:
        return VLMConfig(enabled=False), exc


def _should_collect_vlm_result(
    *,
    enable_vlm: bool | None,
    vlm_provider: VLMProvider | None,
    vlm_config: VLMConfig,
) -> bool:
    if enable_vlm is not None:
        return enable_vlm
    if vlm_provider is not None:
        return True
    return vlm_config.enabled


def _collect_vlm_result(
    *,
    ui_state: UIState,
    instruction: str,
    vlm_provider: VLMProvider | None,
    vlm_config: VLMConfig,
    step_name: str | None = None,
    step_goal: str | None = None,
    region_hint: str | None = None,
) -> dict[str, Any]:
    try:
        provider = vlm_provider or build_default_vlm_provider(vlm_config)
        candidate_dicts = [candidate.to_dict() for candidate in ui_state.candidates]
        result = safe_analyze(
            provider,
            VLMRequest(
                instruction=instruction,
                screenshot_path=ui_state.screenshot_path,
                page_summary=ui_state.page_summary.to_dict(),
                candidates=candidate_dicts,
                max_candidates=vlm_config.max_candidates,
                step_name=step_name,
                step_goal=step_goal,
                region_hint=region_hint,
            ),
        )
        result_dict = result.to_dict()
        if result.ok and result.content:
            valid_ids = [candidate.get("id", "") for candidate in candidate_dicts]
            structured = parse_vlm_content(
                result.content,
                valid_candidate_ids=valid_ids,
            )
            result_dict["structured"] = structured.to_dict()
        else:
            result_dict["structured"] = _fallback_structured(result_dict).to_dict()
        return result_dict
    except Exception as exc:
        error_dict = _vlm_error_result(error=exc, vlm_config=vlm_config)
        error_dict["structured"] = _fallback_structured(error_dict).to_dict()
        return error_dict


def _vlm_error_result(*, error: Exception, vlm_config: VLMConfig) -> dict[str, Any]:
    return {
        "ok": False,
        "provider": vlm_config.provider,
        "model": vlm_config.model,
        "api_base": vlm_config.api_base,
        "content": "",
        "usage": {},
        "finish_reason": None,
        "error": str(error),
        "error_type": type(error).__name__,
        "latency_ms": None,
        "attempts": 0,
        "fallback_used": True,
    }


def _fallback_structured(vlm_result: dict[str, Any]) -> Any:
    from vlm import VLMStructuredOutput

    error_msg = vlm_result.get("error") or "VLM call failed or returned no content"
    return VLMStructuredOutput(
        page_type="unknown",
        page_summary_enhancement="",
        target_candidate_id=None,
        reranked_candidate_ids=[],
        confidence=0.0,
        reason=error_msg,
        uncertain=True,
        suggested_fallback="rule_fallback",
        parse_ok=False,
        parse_error=error_msg,
    )


def _apply_vlm_rerank(ui_state: UIState, vlm_result: dict[str, Any], *, screenshot: Any) -> UIState:
    structured_raw = vlm_result.get("structured")
    if not structured_raw or not isinstance(structured_raw, dict):
        return ui_state

    structured = VLMStructuredOutput.from_dict(structured_raw)
    if not structured.parse_ok:
        return ui_state

    reranked = rerank_candidates(ui_state.candidates, structured)
    visual = build_visual_candidates(structured)
    merged_candidates = reranked + visual
    page_summary = ui_state.page_summary.with_semantic_summary(structured.page_semantics)

    return UIState(
        screenshot_path=ui_state.screenshot_path,
        candidates=merged_candidates,
        page_summary=page_summary,
        vlm_result=vlm_result,
        screen_meta=_build_screen_meta(screenshot, merged_candidates),
        document_region_summary=ui_state.document_region_summary,
        perception_health=ui_state.perception_health,
        candidate_stats=ui_state.candidate_stats,
    )


def _with_screen_meta(ui_state: UIState, screenshot: Any) -> UIState:
    return UIState(
        screenshot_path=ui_state.screenshot_path,
        candidates=ui_state.candidates,
        page_summary=ui_state.page_summary,
        vlm_result=ui_state.vlm_result,
        screen_meta=_build_screen_meta(screenshot, ui_state.candidates),
        perception_health=ui_state.perception_health,
        candidate_stats=ui_state.candidate_stats,
    )


def _build_screen_meta(screenshot: Any, candidates: list[UICandidate]) -> ScreenMeta:
    screenshot_bbox = list(getattr(screenshot, "bbox", []))
    capture_target = str(getattr(screenshot, "capture_target", "fullscreen") or "fullscreen")
    window_rect = screenshot_bbox if capture_target == "window" else None
    monitor_count = int(getattr(screenshot, "monitor_count", 1) or 1)
    return ScreenMeta(
        screenshot_size=[
            int(getattr(screenshot, "width")),
            int(getattr(screenshot, "height")),
        ],
        screen_resolution=list(
            getattr(
                screenshot,
                "screen_resolution",
                [int(getattr(screenshot, "width")), int(getattr(screenshot, "height"))],
            )
        ),
        window_rect=window_rect,
        screen_origin=list(getattr(screenshot, "screen_origin")),
        coordinate_space=str(getattr(screenshot, "coordinate_space", "screen") or "screen"),
        capture_target=capture_target,
        monitor_index=int(getattr(screenshot, "monitor_index", 1)),
        monitor_count=monitor_count,
        is_multi_monitor=monitor_count > 1,
        dpi_scale=_current_dpi_scale(),
        candidate_coordinate_source_counts=_candidate_source_counts(candidates),
        negative_coordinate_candidate_count=_negative_coordinate_count(candidates),
        out_of_screenshot_candidate_count=_out_of_screenshot_count(candidates, screenshot_bbox),
    )


def _candidate_source_counts(candidates: list[UICandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.source] = counts.get(candidate.source, 0) + 1
    return counts


def _negative_coordinate_count(candidates: list[UICandidate]) -> int:
    return sum(1 for candidate in candidates if any(value < 0 for value in candidate.bbox))


def _out_of_screenshot_count(candidates: list[UICandidate], screenshot_bbox: list[int]) -> int:
    if len(screenshot_bbox) != 4:
        return 0
    sx1, sy1, sx2, sy2 = screenshot_bbox
    return sum(
        1
        for candidate in candidates
        if (
            candidate.bbox[0] < sx1
            or candidate.bbox[1] < sy1
            or candidate.bbox[2] > sx2
            or candidate.bbox[3] > sy2
        )
    )


def _current_dpi_scale() -> float | None:
    try:
        import ctypes

        dpi = ctypes.windll.user32.GetDpiForSystem()
    except Exception:
        return None
    if not dpi:
        return None
    return round(float(dpi) / 96.0, 4)


def _assess_perception_health(
    *,
    ocr_elements: list[Any],
    uia_elements: list[Any],
    candidates: list[UICandidate],
) -> "PerceptionHealth":
    from perception.health import assess_perception_health

    return assess_perception_health(
        ocr_elements=ocr_elements,
        uia_elements=uia_elements,
        candidates=candidates,
    )


def _build_candidate_stats(
    health: PerceptionHealth,
    candidates: list[UICandidate],
) -> dict[str, Any]:
    """Build a candidate_stats summary dict for B/C consumption."""
    source_counts: dict[str, int] = {}
    for c in candidates:
        source = str(getattr(c, "source", "unknown") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
    return {
        "ocr_source_count": health.ocr_element_count,
        "uia_source_count": health.uia_element_count,
        "merged_candidate_count": health.merged_candidate_count,
        "clickable_count": health.clickable_count,
        "editable_count": health.editable_count,
        "textless_count": health.textless_candidate_count,
        "shell_control_count": health.shell_control_count,
        "by_source": source_counts,
    }


def _enrich_vlm_instruction_with_health(
    *,
    base_instruction: str,
    perception_health: PerceptionHealth,
) -> str:
    """Append perception health context to VLM instruction when degraded."""
    if perception_health.overall_health not in ("degraded", "unhealthy"):
        return base_instruction

    flags = perception_health.failure_flags
    hints: list[str] = []
    hints.append(
        "Perception health is {}. Please look more carefully at the screenshot "
        "and consider proposing visual_candidates for any clickable targets "
        "that OCR/UIA may have missed.".format(perception_health.overall_health)
    )
    if "ocr_sparse" in flags:
        hints.append(
            "OCR returned very few text elements. Many UI targets may not have "
            "associated text. Pay extra attention to icons, buttons, and visual "
            "controls in the screenshot."
        )
    if "uia_shell_dominated" in flags or "all_shell" in flags:
        hints.append(
            "Most UIA elements are browser shell controls (title bars, scroll bars, "
            "menus). Ignore these and focus on the actual application content "
            "visible in the screenshot."
        )
    if "no_search_target" in flags:
        hints.append(
            "No search box was found by OCR/UIA. If you see a search box or "
            "search icon in the screenshot, propose it as a visual_candidate."
        )
    if "no_input_target" in flags:
        hints.append(
            "No text input field was found by OCR/UIA. If you see a message "
            "input area or text editor in the screenshot, propose it as a "
            "visual_candidate."
        )
    if "no_clickable" in flags:
        hints.append(
            "No clickable candidates were identified. Propose visual_candidates "
            "for any buttons, list items, or interactive elements you can see."
        )
    return "{} {}".format(base_instruction, " ".join(hints))


def _should_read_document_region(ui_state: UIState) -> bool:
    """Determine whether document region reading is useful for this page.

    Checks page_semantics from the VLM result first. Falls back to keyword
    matching on candidate texts if VLM semantics are unavailable.
    """
    # VLM page_semantics indicates a docs page
    vlm_result = ui_state.vlm_result
    if vlm_result and isinstance(vlm_result, dict):
        structured = vlm_result.get("structured")
        if isinstance(structured, dict):
            page_semantics = structured.get("page_semantics")
            if isinstance(page_semantics, dict):
                domain = str(page_semantics.get("domain", "")).strip().lower()
                if domain == "docs":
                    return True
                if domain == "unknown":
                    pass  # fall through to heuristic check
                else:
                    return False  # non-docs domain, skip

    # Heuristic: check if page_summary or candidates suggest a docs page
    page_summary = ui_state.page_summary
    if page_summary and page_summary.has_search_box:
        return False  # search page, not a doc body

    doc_keywords = {"文档", "Doc", "知识库", "wiki", "文档标题", "正文"}
    for candidate in ui_state.candidates:
        text = str(candidate.text or "")
        if any(kw in text for kw in doc_keywords):
            return True

    for text in page_summary.top_texts:
        if any(kw in text for kw in doc_keywords):
            return True

    return False


def _collect_document_region_summary(
    *,
    ui_state: UIState,
    vlm_config: VLMConfig | None = None,
    vlm_provider: VLMProvider | None = None,
    step_goal: str | None = None,
    target_document_title: str | None = None,
    artifact_root: Path | None = None,
) -> DocumentRegionSummary | None:
    """Attempt to read document body region via VLM.

    Returns None when VLM is unavailable. Returns a DocumentRegionSummary
    (possibly with error set) on any failure during the read.
    """
    try:
        return read_document_region(
            screenshot_path=ui_state.screenshot_path,
            vlm_config=vlm_config,
            vlm_provider=vlm_provider,
            step_goal=step_goal,
            target_document_title=target_document_title,
            artifact_dir=artifact_root,
        )
    except Exception as exc:
        return DocumentRegionSummary.empty(error=str(exc))
