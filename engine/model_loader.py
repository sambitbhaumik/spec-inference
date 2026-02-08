"""Model loading with support for quantization and device placement."""
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from engine.config import ModelConfig


@dataclass
class ModelBundle:
    """Loaded model with tokenizer and device information."""
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device


def _resolve_dtype(dtype: str) -> torch.dtype:
    """Map dtype string to torch.dtype."""
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _quant_config(quantization: str) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytes quantization config for specified bit-width."""
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization in ("none", "fp16", "fp32"):
        return None
    raise ValueError(f"Unsupported quantization: {quantization}")


def load_model_bundle(config: ModelConfig) -> ModelBundle:
    """
    Load pretrained LLM with tokenizer, applying quantization and device placement.
    
    Args:
        config: ModelConfig with model_name, quantization, dtype, device_map
        
    Returns:
        ModelBundle with loaded model, tokenizer, and device
    """
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    # Set pad token to EOS if not defined
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve dtype and quantization
    dtype = _resolve_dtype(config.dtype)
    quant_config = _quant_config(config.quantization)

    # Build model kwargs
    model_kwargs = {
        "device_map": config.device_map,
        "trust_remote_code": True,
    }

    if quant_config is not None:
        # Quantization overrides explicit dtype
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = dtype

    # Load model from HuggingFace hub
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs,
    )

    # Set to eval mode (no dropout, batch norm in eval mode)
    model.eval()

    # Determine actual device for inference
    if config.device_map == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Move model if not quantized and using CUDA (quantized models handle placement via device_map)
    if quant_config is None and config.device_map == "cuda":
        model.to(device)

    return ModelBundle(model=model, tokenizer=tokenizer, device=device)
