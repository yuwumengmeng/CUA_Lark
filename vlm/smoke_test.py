"""Manual VLM smoke-test entry point.

Responsibility boundary: this module provides one manually invoked connectivity
check for the configured VLM provider. It does not run as part of unit tests and
does not capture screenshots by itself.

Public interfaces:
`run_smoke_test(...)`
    Send a supplied screenshot plus a short prompt through the configured VLM.
Command line:
```powershell
python -m vlm.smoke_test --screenshot artifacts\\runs\\run_demo\\screenshots\\step_001_before.png
```

Collaboration rules:
1. This command reads `.env` and environment variables but never prints API key
   values.
2. A non-zero process exit means the VLM layer fell back; OCR/UIA perception is
   still expected to remain usable.
3. This script is for manual API connectivity checks, not CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_vlm_config
from .provider import (
    VLMCallResult,
    VLMRequest,
    build_default_vlm_provider,
    safe_analyze,
)


DEFAULT_SMOKE_INSTRUCTION = "请确认你能看到这张截图，并用一句话概括画面。"


def run_smoke_test(
    *,
    screenshot_path: str | Path,
    instruction: str = DEFAULT_SMOKE_INSTRUCTION,
) -> VLMCallResult:
    config = load_vlm_config()
    provider = build_default_vlm_provider(config)
    return safe_analyze(
        provider,
        VLMRequest(
            instruction=instruction,
            screenshot_path=screenshot_path,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a manual VLM screenshot smoke test."
    )
    parser.add_argument(
        "--screenshot",
        required=True,
        help="Path to a PNG/JPEG/WebP screenshot.",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_SMOKE_INSTRUCTION,
        help="Prompt sent with the screenshot.",
    )
    args = parser.parse_args(argv)

    result = run_smoke_test(
        screenshot_path=args.screenshot,
        instruction=args.instruction,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
