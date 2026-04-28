"""视觉感知包入口。

这个包放成员 A 的 perception 层能力：OCR、UIA、候选元素构建和页面摘要。
当前导出 OCR、Windows UIA / Accessibility 和 Candidate Builder 首版入口。
"""

from .candidate_builder import (
    CandidateBuilderConfig,
    CandidateBuilderError,
    build_page_summary,
    build_ui_candidates,
    build_ui_state_from_elements,
)
from .ocr import (
    OCRConfig,
    OCRError,
    OCRTextBlock,
    PytesseractOCRBackend,
    extract_ocr_elements,
)
from .health import assess_perception_health, build_perception_health_summary
from .ui_state import UIStateCollectionError, collect_ui_state, get_ui_state
from .uia import (
    UIAConfig,
    UIAElement,
    UIAError,
    UIAutomationBackend,
    extract_uia_elements,
)

__all__ = [
    "CandidateBuilderConfig",
    "CandidateBuilderError",
    "OCRConfig",
    "OCRError",
    "OCRTextBlock",
    "PytesseractOCRBackend",
    "UIAConfig",
    "UIAElement",
    "UIAError",
    "UIStateCollectionError",
    "UIAutomationBackend",
    "build_page_summary",
    "build_ui_candidates",
    "build_ui_state_from_elements",
    "extract_ocr_elements",
    "extract_uia_elements",
    "assess_perception_health",
    "build_perception_health_summary",
    "collect_ui_state",
    "get_ui_state",
]
