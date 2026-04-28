"""Member A VLM provider base tests.

Test goals: verify provider fallback behavior and request/result contracts for
the VLM base layer without touching external network services.

Important constraints: tests use fake providers and disabled providers only.
They do not read real API keys, send screenshots, or call real VLM APIs.

Run:
```powershell
python -m pytest tests\testA\test_vlm_provider.py
python tests\testA\test_vlm_provider.py
```
"""

import sys
import unittest
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

config_module = import_module("vlm.config")
provider_module = import_module("vlm.provider")

VLMConfig = config_module.VLMConfig
DisabledVLMProvider = provider_module.DisabledVLMProvider
VLMCallResult = provider_module.VLMCallResult
VLMRequest = provider_module.VLMRequest
build_default_vlm_provider = provider_module.build_default_vlm_provider
safe_analyze = provider_module.safe_analyze


class RaisingProvider:
    def analyze(self, request: VLMRequest) -> VLMCallResult:
        raise RuntimeError("simulated outage")


class VLMProviderTests(unittest.TestCase):
    def test_disabled_provider_returns_fallback_result(self):
        config = VLMConfig(enabled=False, api_key=None)
        provider = build_default_vlm_provider(config)

        result = provider.analyze(VLMRequest(instruction="describe screen"))

        self.assertIsInstance(provider, DisabledVLMProvider)
        self.assertFalse(result.ok)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.error_type, "disabled")

    def test_safe_analyze_converts_exceptions_to_fallback(self):
        result = safe_analyze(RaisingProvider(), VLMRequest(instruction="describe screen"))

        self.assertFalse(result.ok)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertIn("simulated outage", result.error)

    def test_vlm_request_compacts_top_candidates(self):
        request = VLMRequest(
            instruction="choose target",
            candidates=[
                {"id": "elem_0001", "text": "A", "extra": "ignored"},
                {"id": "elem_0002", "text": "B"},
            ],
            max_candidates=1,
        )

        payload = request.compact_payload(default_max_candidates=30)

        self.assertEqual(len(payload["candidates"]), 1)
        self.assertEqual(payload["candidates"][0]["id"], "elem_0001")
        self.assertNotIn("extra", payload["candidates"][0])


if __name__ == "__main__":
    unittest.main()
