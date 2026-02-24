"""Configuration parsing and model loading utilities."""
from dataclasses import dataclass
from typing import Optional

import yaml

from engine.sampler import SamplingConfig


@dataclass
class ModelConfig:
    """Configuration for a single model (draft or verify)."""
    model_name: str                    # HuggingFace model ID (e.g., "meta-llama/Llama-2-7b")
    quantization: str = "4bit"        # "4bit", "8bit", "none", "fp16", "fp32"
    max_tokens: Optional[int] = None  # Inference max_new_tokens override
    device_map: str = "cuda"          # "cuda", "cpu", or device mapping dict
    dtype: str = "float16"            # Model precision: "float16", "bfloat16", "float32"


@dataclass
class EngineConfig:
    """Top-level configuration for the speculative decoding engine."""
    draft: ModelConfig     # Small, fast model for token proposals
    verify: ModelConfig    # Large, accurate model for verification
    sampling: SamplingConfig  # Temperature, top-k, top-p parameters


def load_engine_config(path: str) -> EngineConfig:
    """
    Load engine configuration from YAML file.
    
    YAML structure:
        draft:
            model_name: "model/id"
            quantization: "4bit"
            dtype: "float16"
        verify:
            model_name: "model/id"
            quantization: "4bit"
            dtype: "float16"
        sampling:
            temperature: 0.8
            top_k: 40
            top_p: 0.95
            min_p: 0.05
    
    Args:
        path: Path to YAML config file
        
    Returns:
        EngineConfig with parsed draft, verify, and sampling configs
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if "draft" not in data or "verify" not in data:
        raise ValueError("Config must include 'draft' and 'verify' sections.")

    draft = ModelConfig(**data["draft"])
    verify = ModelConfig(**data["verify"])

    sampling_data = data.get("sampling", {})
    sampling = SamplingConfig(**sampling_data)

    return EngineConfig(draft=draft, verify=verify, sampling=sampling)
