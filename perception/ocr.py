"""成员 A 4.23 初版 OCR 接入。

职责边界：
本文件只负责从截图图片中抽取可见文本，并输出统一的 OCR 文本块结构。
它不负责截图采集、不负责 UIA/Accessibility、不负责 OCR+UIA 合并，也不
负责判断元素是否 clickable。后续 candidate builder 应该消费这里的
`OCRTextBlock`，再和 UIA 结果融合成统一候选元素。

公开接口：
`extract_ocr_elements(image, config=None, backend=None)`
    首版 OCR 入口。输入可以是图片路径、`Path` 或 `PIL.Image.Image`。
    输出 `list[OCRTextBlock]`。
`OCRTextBlock.to_dict()`
    返回稳定 dict，字段为 `text / bbox / confidence`。
`PytesseractOCRBackend`
    默认真实 OCR 后端。使用 Python 包 `pytesseract` 调用系统 Tesseract。

输出结构：
`text`
    OCR 识别出的非空文本。
`bbox`
    文本块在截图坐标系中的位置，格式为 `[x1, y1, x2, y2]`。
`confidence`
    归一化置信度，范围为 `0.0 ~ 1.0`。
    默认 `min_confidence=0.6`，用于过滤图标、头像、系统栏等低置信度噪声。

真实 OCR 环境要求：
1. Python 依赖需要安装 `pytesseract`。
2. 系统需要安装 Tesseract OCR 可执行文件，并能在 PATH 中找到。
3. 中文界面建议安装 `chi_sim` 语言包。默认语言为 `chi_sim+eng`，
   如果语言包缺失，会尝试回退到 `eng`。

使用示例：
```python
from perception.ocr import extract_ocr_elements

blocks = extract_ocr_elements("artifacts/runs/run_demo/screenshots/step_001_before.png")
for block in blocks:
    print(block.to_dict())
```

协作规则：
1. OCR 输出字段名固定为 `text / bbox / confidence`，不要改成 `box`、
   `score` 等别名。
2. `bbox` 坐标必须沿用截图坐标系，不做窗口坐标、比例坐标或中心点转换。
3. 单元测试使用 fake backend，不依赖真实 Tesseract；真实 OCR 验证应
   单独用手动命令或可视化检查脚本完成。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageEnhance, ImageFilter


OCR_SCHEMA_VERSION = "ocr_text_block.v1"


class OCRError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OCRTextBlock:
    text: str
    bbox: list[int]
    confidence: float
    version: str = OCR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        text = self.text.strip()
        if not text:
            raise OCRError("OCRTextBlock text cannot be empty")
        if len(self.bbox) != 4:
            raise OCRError("OCRTextBlock bbox must contain four integers")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.bbox):
            raise OCRError("OCRTextBlock bbox values must be integers")
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise OCRError("OCRTextBlock bbox must satisfy x2 > x1 and y2 > y1")
        if not 0.0 <= self.confidence <= 1.0:
            raise OCRError("OCRTextBlock confidence must be between 0.0 and 1.0")
        object.__setattr__(self, "text", text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "text": self.text,
            "bbox": self.bbox,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class OCRConfig:
    lang: str = "chi_sim+eng"
    fallback_lang: str | None = "eng"
    min_confidence: float = 0.45
    tesseract_config: str = "--oem 3 --psm 11 -c preserve_interword_spaces=1"
    preprocess_image: bool = True
    grayscale: bool = True
    scale_factor: float = 2.0
    contrast_factor: float = 1.6
    sharpen: bool = True

    def __post_init__(self) -> None:
        if not self.lang:
            raise OCRError("OCRConfig lang cannot be empty")
        if self.fallback_lang is not None and not self.fallback_lang:
            raise OCRError("OCRConfig fallback_lang cannot be empty")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise OCRError("OCRConfig min_confidence must be between 0.0 and 1.0")
        if self.tesseract_config is None:
            raise OCRError("OCRConfig tesseract_config cannot be None")
        if not isinstance(self.preprocess_image, bool):
            raise OCRError("OCRConfig preprocess_image must be a boolean")
        if not isinstance(self.grayscale, bool):
            raise OCRError("OCRConfig grayscale must be a boolean")
        if (
            isinstance(self.scale_factor, bool)
            or not isinstance(self.scale_factor, (int, float))
            or self.scale_factor <= 0
        ):
            raise OCRError("OCRConfig scale_factor must be positive")
        if (
            isinstance(self.contrast_factor, bool)
            or not isinstance(self.contrast_factor, (int, float))
            or self.contrast_factor <= 0
        ):
            raise OCRError("OCRConfig contrast_factor must be positive")
        if not isinstance(self.sharpen, bool):
            raise OCRError("OCRConfig sharpen must be a boolean")


class OCRBackend(Protocol):
    def extract(self, image: Image.Image, config: OCRConfig) -> list[OCRTextBlock]:
        ...


class PytesseractOCRBackend:
    def extract(self, image: Image.Image, config: OCRConfig) -> list[OCRTextBlock]:
        try:
            import pytesseract
        except ImportError as exc:
            raise OCRError(
                "pytesseract is not installed. Install dependencies from requirements.txt."
            ) from exc

        original_size = image.size
        ocr_image = _prepare_ocr_image(image, config)
        data = self._image_to_data(ocr_image, config, pytesseract)
        blocks = _blocks_from_tesseract_data(data, config.min_confidence)
        return _scale_blocks_to_original(blocks, ocr_image.size, original_size)

    def _image_to_data(
        self,
        image: Image.Image,
        config: OCRConfig,
        pytesseract_module: Any,
    ) -> dict[str, list[Any]]:
        try:
            return pytesseract_module.image_to_data(
                image,
                lang=config.lang,
                config=config.tesseract_config,
                output_type=pytesseract_module.Output.DICT,
            )
        except pytesseract_module.TesseractNotFoundError as exc:
            raise OCRError(
                "Tesseract executable was not found. Install Tesseract OCR and add it to PATH."
            ) from exc
        except pytesseract_module.TesseractError as exc:
            if config.fallback_lang and config.fallback_lang != config.lang:
                try:
                    return pytesseract_module.image_to_data(
                        image,
                        lang=config.fallback_lang,
                        config=config.tesseract_config,
                        output_type=pytesseract_module.Output.DICT,
                    )
                except pytesseract_module.TesseractError:
                    pass
            raise OCRError(f"Tesseract OCR failed: {exc}") from exc


def extract_ocr_elements(
    image: str | Path | Image.Image,
    *,
    config: OCRConfig | None = None,
    backend: OCRBackend | None = None,
) -> list[OCRTextBlock]:
    ocr_config = config or OCRConfig()
    ocr_backend = backend or PytesseractOCRBackend()
    loaded_image = _load_image(image)
    return ocr_backend.extract(loaded_image, ocr_config)


def _load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()

    path = Path(image)
    if not path.exists():
        raise OCRError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise OCRError(f"Image path is not a file: {path}")

    with Image.open(path) as opened:
        return opened.convert("RGB").copy()


def _prepare_ocr_image(image: Image.Image, config: OCRConfig) -> Image.Image:
    prepared = image.convert("RGB")
    if not config.preprocess_image:
        return prepared

    if config.grayscale:
        prepared = prepared.convert("L")
    if config.contrast_factor != 1.0:
        prepared = ImageEnhance.Contrast(prepared).enhance(config.contrast_factor)
    if config.sharpen:
        prepared = prepared.filter(ImageFilter.SHARPEN)
    if config.scale_factor != 1.0:
        width, height = prepared.size
        scaled_size = (
            max(1, int(round(width * config.scale_factor))),
            max(1, int(round(height * config.scale_factor))),
        )
        prepared = prepared.resize(scaled_size, Image.Resampling.LANCZOS)
    return prepared


def _scale_blocks_to_original(
    blocks: list[OCRTextBlock],
    ocr_size: tuple[int, int],
    original_size: tuple[int, int],
) -> list[OCRTextBlock]:
    if not blocks or ocr_size == original_size:
        return blocks

    x_scale = original_size[0] / ocr_size[0]
    y_scale = original_size[1] / ocr_size[1]
    scaled_blocks: list[OCRTextBlock] = []
    for block in blocks:
        x1, y1, x2, y2 = block.bbox
        bbox = [
            _scale_coordinate(x1, x_scale, original_size[0]),
            _scale_coordinate(y1, y_scale, original_size[1]),
            _scale_coordinate(x2, x_scale, original_size[0]),
            _scale_coordinate(y2, y_scale, original_size[1]),
        ]
        bbox[0] = min(bbox[0], max(0, original_size[0] - 1))
        bbox[1] = min(bbox[1], max(0, original_size[1] - 1))
        if bbox[2] <= bbox[0]:
            bbox[2] = min(original_size[0], bbox[0] + 1)
        if bbox[3] <= bbox[1]:
            bbox[3] = min(original_size[1], bbox[1] + 1)
        scaled_blocks.append(
            OCRTextBlock(
                text=block.text,
                bbox=bbox,
                confidence=block.confidence,
                version=block.version,
            )
        )
    return scaled_blocks


def _scale_coordinate(value: int, scale: float, limit: int) -> int:
    return max(0, min(limit, int(round(value * scale))))


def _blocks_from_tesseract_data(
    data: dict[str, list[Any]],
    min_confidence: float,
) -> list[OCRTextBlock]:
    required = ("text", "left", "top", "width", "height", "conf")
    missing = [field for field in required if field not in data]
    if missing:
        raise OCRError(f"Tesseract data is missing fields: {', '.join(missing)}")

    blocks: list[OCRTextBlock] = []
    for index, raw_text in enumerate(data["text"]):
        text = str(raw_text).strip()
        confidence = _normalize_confidence(data["conf"][index])
        if not text or confidence is None or confidence < min_confidence:
            continue

        left = _require_int(data["left"][index], "left")
        top = _require_int(data["top"][index], "top")
        width = _require_int(data["width"][index], "width")
        height = _require_int(data["height"][index], "height")
        if width <= 0 or height <= 0:
            continue

        blocks.append(
            OCRTextBlock(
                text=text,
                bbox=[left, top, left + width, top + height],
                confidence=confidence,
            )
        )
    return blocks


def _normalize_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence < 0:
        return None
    if confidence > 1:
        confidence = confidence / 100
    return max(0.0, min(confidence, 1.0))


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise OCRError(f"Tesseract field {field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise OCRError(f"Tesseract field {field_name} must be an integer") from exc
