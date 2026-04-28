"""Member A VLM provider package.

Responsibility boundary: this package owns VLM API configuration, provider
creation, safe invocation, structured output parsing, and a small smoke-test
entry point. It does not own OCR, UIA, candidate building, planner decisions,
or executor actions.

Public interfaces:
`VLMConfig` / `load_vlm_config(...)`
    Read VLM settings from environment variables and optional `.env` files.
`OpenAICompatibleVLMProvider`
    Calls VLM services that expose the OpenAI chat completions API shape.
`DisabledVLMProvider` / `build_default_vlm_provider(...)`
    Provide a no-op fallback when VLM is disabled or not configured.
`VLMRequest` / `VLMCallResult` / `safe_analyze(...)`
    Stable request/result contracts for perception and later planner code.
`VLMStructuredOutput` / `parse_vlm_content(...)`
    Structured output that Member B can directly consume. Parsed from VLM
    response text into stable fields: page_type, target_candidate_id,
    reranked_candidate_ids, confidence, reason, uncertain, suggested_fallback.

Collaboration rules:
1. Callers must treat `VLMCallResult.ok == False` as a fallback signal, not as a
   hard perception failure.
2. API keys are never serialized into artifacts or returned by public dicts.
3. New model vendors should implement the provider protocol without changing
   OCR/UIA/Candidate Builder contracts.
4. Member B reads `vlm_result["structured"]` for the parsed VLM output.
   When `structured.uncertain` is True, B must fall back to rule-based scoring.
"""

from .config import VLMConfig, VLMConfigError, load_dotenv_values, load_vlm_config
from .context import (
    FOCUS_DETAIL,
    FOCUS_DOCUMENT_BODY,
    FOCUS_INPUT,
    FOCUS_MODAL,
    FOCUS_NAVIGATION,
    FOCUS_RESULTS,
    FOCUS_SEARCH,
    FOCUS_UNKNOWN,
    VLMContextError,
    VLMContextPolicy,
    build_vlm_context_payload,
    compact_candidate,
    compact_page_summary,
    infer_context_focus,
    select_candidate_context,
)
from .output_schema import (
    FALLBACK_ABORT,
    FALLBACK_RETRY,
    FALLBACK_RULE,
    FALLBACK_WAIT,
    VLMOutputError,
    VLMStructuredOutput,
    parse_vlm_content,
)
from .provider import (
    DisabledVLMProvider,
    OpenAICompatibleVLMProvider,
    VLMCallResult,
    VLMProvider,
    VLMProviderError,
    VLMRequest,
    build_default_vlm_provider,
    safe_analyze,
)
from .rerank import (
    SHELL_CONTROL_ROLES,
    SHELL_CONTROL_TEXTS,
    SYNTHETIC_CANDIDATE_PREFIX,
    build_visual_candidates,
    is_shell_control,
    rerank_candidates,
)
from .region_reader import (
    estimate_doc_body_bbox,
    read_document_region,
)

__all__ = [
    "DisabledVLMProvider",
    "FALLBACK_ABORT",
    "FALLBACK_RETRY",
    "FALLBACK_RULE",
    "FALLBACK_WAIT",
    "FOCUS_DETAIL",
    "FOCUS_DOCUMENT_BODY",
    "FOCUS_INPUT",
    "FOCUS_MODAL",
    "FOCUS_NAVIGATION",
    "FOCUS_RESULTS",
    "FOCUS_SEARCH",
    "FOCUS_UNKNOWN",
    "OpenAICompatibleVLMProvider",
    "SHELL_CONTROL_ROLES",
    "SHELL_CONTROL_TEXTS",
    "SYNTHETIC_CANDIDATE_PREFIX",
    "VLMCallResult",
    "VLMConfig",
    "VLMConfigError",
    "VLMContextError",
    "VLMContextPolicy",
    "VLMOutputError",
    "VLMProvider",
    "VLMProviderError",
    "VLMRequest",
    "VLMStructuredOutput",
    "build_default_vlm_provider",
    "build_vlm_context_payload",
    "build_visual_candidates",
    "compact_candidate",
    "compact_page_summary",
    "infer_context_focus",
    "is_shell_control",
    "load_dotenv_values",
    "load_vlm_config",
    "parse_vlm_content",
    "rerank_candidates",
    "safe_analyze",
    "select_candidate_context",
    "estimate_doc_body_bbox",
    "read_document_region",
]
