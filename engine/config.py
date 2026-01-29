from dataclasses import dataclass
from typing import Optional

import yaml

from engine.sampler import SamplingConfig


@dataclass
class ModelConfig:
    model_name: str
    quantization: str = "4bit"
    max_tokens: Optional[int] = None
    device_map: str = "cuda"
    dtype: str = "float16"


@dataclass
class EngineConfig:
    draft: ModelConfig
    verify: ModelConfig
    sampling: SamplingConfig


def load_engine_config(path: str) -> EngineConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if "draft" not in data or "verify" not in data:
        raise ValueError("Config must include 'draft' and 'verify' sections.")

    draft = ModelConfig(**data["draft"])
    verify = ModelConfig(**data["verify"])

    sampling_data = data.get("sampling", {})
    sampling = SamplingConfig(**sampling_data)

    return EngineConfig(draft=draft, verify=verify, sampling=sampling)
