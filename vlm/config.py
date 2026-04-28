"""Member A VLM configuration loader.

Responsibility boundary: this file reads VLM settings from process environment
variables and optional `.env` files, validates basic values, and exposes a
sanitized public config dict. It does not call remote APIs and does not mutate
OCR/UIA/candidate output.

Public interfaces:
`load_vlm_config(...)`
    Build a `VLMConfig` from environment variables plus optional `.env` values.
`load_dotenv_values(...)`
    Minimal `.env` parser used to avoid adding a dependency for this layer.
`VLMConfig.to_public_dict()`
    Return configuration suitable for logs and artifacts without leaking keys.

Environment contract:
`VLM_ENABLED`
    Explicitly enables VLM calls when set to true/1/yes/on. Defaults to false.
`VLM_API_KEY` or `DASHSCOPE_API_KEY`
    API key. `VLM_API_KEY` wins when both are present.
`VLM_PROVIDER`, `VLM_MODEL`, `VLM_API_BASE`, `VLM_TIMEOUT_SECONDS`,
`VLM_MAX_RETRIES`, `VLM_MAX_CANDIDATES`
    Provider and request tuning knobs.

Collaboration rules:
1. Missing API keys do not raise during config loading; providers convert that
   into a disabled/fallback result.
2. Real environment variables override `.env` values.
3. Public dicts must never include the full API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_PROVIDER = "openai_compatible"
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.6-plus"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_CANDIDATES = 30
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off", ""}


class VLMConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class VLMConfig:
    enabled: bool = False
    provider: str = DEFAULT_PROVIDER
    api_key: str | None = None
    api_base: str = DEFAULT_API_BASE
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise VLMConfigError("VLM provider cannot be empty")
        if not self.api_base.strip():
            raise VLMConfigError("VLM API base cannot be empty")
        if not self.model.strip():
            raise VLMConfigError("VLM model cannot be empty")
        if self.timeout_seconds <= 0:
            raise VLMConfigError("VLM timeout_seconds must be positive")
        if self.max_retries < 0:
            raise VLMConfigError("VLM max_retries must be non-negative")
        if self.max_candidates <= 0:
            raise VLMConfigError("VLM max_candidates must be positive")
        if self.max_tokens <= 0:
            raise VLMConfigError("VLM max_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise VLMConfigError("VLM temperature must be between 0.0 and 2.0")

        object.__setattr__(self, "provider", self.provider.strip())
        object.__setattr__(
            self,
            "api_key",
            self.api_key.strip() if self.api_key else None,
        )
        object.__setattr__(self, "api_base", self.api_base.rstrip("/"))
        object.__setattr__(self, "model", self.model.strip())

    @property
    def is_configured(self) -> bool:
        return bool(self.enabled and self.api_key)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "api_key_configured": bool(self.api_key),
            "api_key_hint": _mask_secret(self.api_key),
            "api_base": self.api_base,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_candidates": self.max_candidates,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


def load_vlm_config(
    *,
    env: Mapping[str, str] | None = None,
    env_file: str | Path | None = None,
    load_env_file: bool = True,
) -> VLMConfig:
    base_env = dict(os.environ if env is None else env)
    file_values = _resolve_env_file_values(env_file, load_env_file=load_env_file)
    effective_env = {**file_values, **base_env}

    return VLMConfig(
        enabled=_bool_env(effective_env.get("VLM_ENABLED"), default=False),
        provider=effective_env.get("VLM_PROVIDER", DEFAULT_PROVIDER),
        api_key=effective_env.get("VLM_API_KEY") or effective_env.get("DASHSCOPE_API_KEY"),
        api_base=effective_env.get("VLM_API_BASE", DEFAULT_API_BASE),
        model=effective_env.get("VLM_MODEL", DEFAULT_MODEL),
        timeout_seconds=_float_env(
            effective_env.get("VLM_TIMEOUT_SECONDS"),
            DEFAULT_TIMEOUT_SECONDS,
            "VLM_TIMEOUT_SECONDS",
        ),
        max_retries=_int_env(
            effective_env.get("VLM_MAX_RETRIES"),
            DEFAULT_MAX_RETRIES,
            "VLM_MAX_RETRIES",
        ),
        max_candidates=_int_env(
            effective_env.get("VLM_MAX_CANDIDATES"),
            DEFAULT_MAX_CANDIDATES,
            "VLM_MAX_CANDIDATES",
        ),
        temperature=_float_env(
            effective_env.get("VLM_TEMPERATURE"),
            DEFAULT_TEMPERATURE,
            "VLM_TEMPERATURE",
        ),
        max_tokens=_int_env(
            effective_env.get("VLM_MAX_TOKENS"),
            DEFAULT_MAX_TOKENS,
            "VLM_MAX_TOKENS",
        ),
    )


def load_dotenv_values(path: str | Path) -> dict[str, str]:
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name:
            continue
        values[name] = _strip_env_value(value.strip())
    return values


def _resolve_env_file_values(
    env_file: str | Path | None,
    *,
    load_env_file: bool,
) -> dict[str, str]:
    if not load_env_file:
        return {}
    if env_file is not None:
        return load_dotenv_values(env_file)
    return load_dotenv_values(".env")


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    marker = " #"
    if marker in value:
        return value.split(marker, 1)[0].rstrip()
    return value


def _bool_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise VLMConfigError(f"Invalid boolean environment value: {value}")


def _float_env(value: str | None, default: float, name: str) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise VLMConfigError(f"{name} must be a number") from exc


def _int_env(value: str | None, default: int, name: str) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise VLMConfigError(f"{name} must be an integer") from exc


def _mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"
