"""Member A VLM configuration tests.

Test goals: verify that `vlm.config` reads `.env` style files, respects real
environment override precedence, normalizes DashScope/OpenAI-compatible
defaults, and never exposes the raw API key in public config output.

Important constraints: tests use temporary files and injected environment maps.
They do not read the developer's real `.env`, do not access network, and do not
call any real VLM API.

Run:
```powershell
python -m pytest tests\testA\test_vlm_config.py
python tests\testA\test_vlm_config.py
```
"""

import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

config_module = import_module("vlm.config")

DEFAULT_API_BASE = config_module.DEFAULT_API_BASE
DEFAULT_MODEL = config_module.DEFAULT_MODEL
VLMConfigError = config_module.VLMConfigError
load_vlm_config = config_module.load_vlm_config


class VLMConfigTests(unittest.TestCase):
    def test_loads_defaults_without_api_key(self):
        config = load_vlm_config(env={}, load_env_file=False)

        self.assertFalse(config.enabled)
        self.assertFalse(config.is_configured)
        self.assertEqual(config.api_base, DEFAULT_API_BASE)
        self.assertEqual(config.model, DEFAULT_MODEL)

    def test_loads_dotenv_and_dashscope_key_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "VLM_ENABLED=true",
                        "DASHSCOPE_API_KEY=sk-from-dashscope",
                        "VLM_MODEL=qwen-vl-plus",
                        "VLM_TIMEOUT_SECONDS=12.5",
                        "VLM_MAX_RETRIES=3",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_vlm_config(env={}, env_file=env_path)

            self.assertTrue(config.enabled)
            self.assertTrue(config.is_configured)
            self.assertEqual(config.api_key, "sk-from-dashscope")
            self.assertEqual(config.model, "qwen-vl-plus")
            self.assertEqual(config.timeout_seconds, 12.5)
            self.assertEqual(config.max_retries, 3)

    def test_environment_overrides_dotenv_and_masks_public_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "VLM_API_KEY=sk-from-file\nVLM_MODEL=qwen-vl-plus\n",
                encoding="utf-8",
            )

            config = load_vlm_config(
                env={
                    "VLM_API_KEY": "sk-from-env-123456",
                    "VLM_MODEL": "qwen3.6-plus",
                },
                env_file=env_path,
            )

            public = config.to_public_dict()
            self.assertEqual(config.api_key, "sk-from-env-123456")
            self.assertEqual(config.model, "qwen3.6-plus")
            self.assertTrue(public["api_key_configured"])
            self.assertNotIn("sk-from-env-123456", str(public))

    def test_rejects_invalid_numeric_values(self):
        with self.assertRaises(VLMConfigError):
            load_vlm_config(
                env={"VLM_TIMEOUT_SECONDS": "slow"},
                load_env_file=False,
            )


if __name__ == "__main__":
    unittest.main()
