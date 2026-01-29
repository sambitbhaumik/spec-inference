from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from engine.config import ModelConfig


@dataclass
class ModelBundle:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _quant_config(quantization: str) -> Optional[BitsAndBytesConfig]:
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
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _resolve_dtype(config.dtype)
    quant_config = _quant_config(config.quantization)

    model_kwargs = {
        "device_map": config.device_map,
        "trust_remote_code": True,
    }

    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs,
    )

    model.eval()

    if config.device_map == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if quant_config is None and config.device_map == "cuda":
        model.to(device)

    return ModelBundle(model=model, tokenizer=tokenizer, device=device)
