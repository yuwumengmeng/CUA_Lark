"""成员 A 4.23 初版 OCR 接入测试。

测试目标：
本文件验证 `perception/ocr.py` 的 OCR 输出协议是否稳定，包括：
1. `extract_ocr_elements` 能接收图片路径和 `PIL.Image.Image`。
2. OCR 输出字段固定为 `text / bbox / confidence`。
3. Tesseract 风格原始结果能转换为统一 `OCRTextBlock`。
4. 空文本、负置信度、非法 bbox、缺失图片路径会被正确处理或拒绝。

重要约定：
这里不调用真实 Tesseract OCR，不依赖系统是否安装 Tesseract，也不依赖
当前电脑屏幕内容。测试使用 fake backend 和固定图片，只验证接口协议和
数据转换逻辑。

直接运行：
```powershell
python tests\testA\test_ocr.py
```

通过 pytest 运行：
```powershell
python -m pytest tests\testA\test_ocr.py
```

真实 OCR 手动验证请先安装系统 Tesseract，然后运行：
```powershell
python -c "from perception.ocr import extract_ocr_elements; print([b.to_dict() for b in extract_ocr_elements('artifacts/runs/run_demo/screenshots/step_001_before.png')])"
```
"""

import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ocr_module = import_module("perception.ocr")
OCRConfig = ocr_module.OCRConfig
OCRError = ocr_module.OCRError
OCRTextBlock = ocr_module.OCRTextBlock
PytesseractOCRBackend = ocr_module.PytesseractOCRBackend
extract_ocr_elements = ocr_module.extract_ocr_elements
blocks_from_tesseract_data = ocr_module._blocks_from_tesseract_data


class FakeOCRBackend:
    def __init__(self) -> None:
        self.seen_sizes: list[tuple[int, int]] = []
        self.seen_langs: list[str] = []

    def extract(self, image: Image.Image, config: OCRConfig) -> list[OCRTextBlock]:
        self.seen_sizes.append(image.size)
        self.seen_langs.append(config.lang)
        return [
            OCRTextBlock(text="搜索", bbox=[10, 20, 50, 40], confidence=0.91),
            OCRTextBlock(text="测试群", bbox=[60, 20, 120, 45], confidence=0.87),
        ]


class FakePytesseractModule:
    class Output:
        DICT = "dict"

    class TesseractNotFoundError(Exception):
        pass

    class TesseractError(Exception):
        pass

    def __init__(self, data):
        self.data = data
        self.calls = []

    def image_to_data(self, image, *, lang, config, output_type):
        self.calls.append(
            {
                "size": image.size,
                "mode": image.mode,
                "lang": lang,
                "config": config,
                "output_type": output_type,
            }
        )
        return self.data


class OCRTests(unittest.TestCase):
    def test_extract_ocr_elements_accepts_pil_image(self):
        backend = FakeOCRBackend()
        image = Image.new("RGB", (160, 90), color=(255, 255, 255))

        blocks = extract_ocr_elements(image, backend=backend)

        self.assertEqual([block.text for block in blocks], ["搜索", "测试群"])
        self.assertEqual(blocks[0].to_dict()["bbox"], [10, 20, 50, 40])
        self.assertEqual(blocks[0].to_dict()["confidence"], 0.91)
        self.assertEqual(backend.seen_sizes, [(160, 90)])
        self.assertEqual(backend.seen_langs, ["chi_sim+eng"])

    def test_default_min_confidence_filters_low_confidence_noise(self):
        config = OCRConfig()

        self.assertEqual(config.min_confidence, 0.45)
        self.assertEqual(config.scale_factor, 2.0)
        self.assertIn("--psm 11", config.tesseract_config)
        self.assertTrue(config.preprocess_image)

    def test_pytesseract_backend_preprocesses_and_rescales_bbox(self):
        data = {
            "text": ["鎼滅储"],
            "left": [20],
            "top": [10],
            "width": [40],
            "height": [20],
            "conf": ["90"],
        }
        fake_pytesseract = FakePytesseractModule(data)
        old_module = sys.modules.get("pytesseract")
        sys.modules["pytesseract"] = fake_pytesseract
        try:
            blocks = PytesseractOCRBackend().extract(
                Image.new("RGB", (100, 50), color=(255, 255, 255)),
                OCRConfig(),
            )
        finally:
            if old_module is None:
                sys.modules.pop("pytesseract", None)
            else:
                sys.modules["pytesseract"] = old_module

        self.assertEqual(fake_pytesseract.calls[0]["size"], (200, 100))
        self.assertEqual(fake_pytesseract.calls[0]["lang"], "chi_sim+eng")
        self.assertIn("--oem 3", fake_pytesseract.calls[0]["config"])
        self.assertEqual(blocks[0].bbox, [10, 5, 30, 15])

    def test_extract_ocr_elements_accepts_image_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            Image.new("RGB", (80, 40), color=(255, 255, 255)).save(image_path)
            backend = FakeOCRBackend()

            blocks = extract_ocr_elements(image_path, backend=backend)

            self.assertEqual(len(blocks), 2)
            self.assertEqual(backend.seen_sizes, [(80, 40)])

    def test_tesseract_data_is_converted_to_text_blocks(self):
        data = {
            "text": ["", "搜索", "低置信度", "测试群"],
            "left": [0, 10, 20, 60],
            "top": [0, 20, 20, 30],
            "width": [0, 40, 30, 60],
            "height": [0, 20, 20, 25],
            "conf": ["-1", "91", "10", "87.5"],
        }

        blocks = blocks_from_tesseract_data(data, min_confidence=0.5)

        self.assertEqual([block.text for block in blocks], ["搜索", "测试群"])
        self.assertEqual(blocks[0].bbox, [10, 20, 50, 40])
        self.assertEqual(blocks[0].confidence, 0.91)
        self.assertEqual(blocks[1].confidence, 0.875)

    def test_missing_image_path_is_rejected(self):
        with self.assertRaises(OCRError):
            extract_ocr_elements("missing.png", backend=FakeOCRBackend())

    def test_invalid_ocr_text_block_is_rejected(self):
        with self.assertRaises(OCRError):
            OCRTextBlock(text="", bbox=[0, 0, 10, 10], confidence=0.5)

        with self.assertRaises(OCRError):
            OCRTextBlock(text="搜索", bbox=[10, 10, 5, 20], confidence=0.5)

        with self.assertRaises(OCRError):
            OCRTextBlock(text="搜索", bbox=[0, 0, 10, 10], confidence=1.5)


if __name__ == "__main__":
    unittest.main()
