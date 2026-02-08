"""
Benchmarking utility: compares naive autoregressive generation vs speculative decoding.
Measures tokens/sec and acceptance rates to validate speedup from speculation.
"""
import argparse
import time

import torch

from engine.config import load_engine_config
from engine.kv_cache import KVCache
from engine.model_loader import load_model_bundle
from engine.sampler import sample_next_token
from speculative.spec_decoder import SpeculativeDecoder
from speculative.stats import SpeculativeStats


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding")
    parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate from")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of new tokens")
    parser.add_argument("--draft_k", type=int, default=None, help="Draft tokens per step")
    return parser.parse_args()


def naive_generate(model_bundle, sampling, prompt: str, max_tokens: int):
    """
    Baseline autoregressive generation: generate one token per forward pass.
    
    Args:
        model_bundle: ModelBundle with model, tokenizer, device
        sampling: SamplingConfig for temperature/top-k/top-p
        prompt: Initial text prompt (str)
        max_tokens: Max new tokens to generate (int)
        
    Returns:
        (decoded_text, num_tokens_generated)
    """
    tokenizer = model_bundle.tokenizer
    device = model_bundle.device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    with torch.no_grad():
        # Pre-fill cache with prompt
        outputs = model_bundle.model(input_ids=input_ids, use_cache=True)
        cache = KVCache.from_past_key_values(outputs.past_key_values)
        last_token = input_ids[:, -1:]
        generated = []

        for _ in range(max_tokens):
            # One token per step (standard autoregressive)
            out = model_bundle.model(
                input_ids=last_token,
                past_key_values=cache.past_key_values,
                use_cache=True,
            )
            next_token = sample_next_token(out.logits[:, -1, :], sampling)
            cache = KVCache.from_past_key_values(out.past_key_values)
            token_id = int(next_token.item())
            generated.append(token_id)
            last_token = next_token.unsqueeze(1)
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated, skip_special_tokens=True), len(generated)


def main() -> None:
    """Run benchmark: naive vs speculative generation with timing."""
    args = parse_args()
    config = load_engine_config(args.config)

    draft_k = args.draft_k if args.draft_k is not None else (config.draft.max_tokens or 5)

    # Load models
    verify_bundle = load_model_bundle(config.verify)
    draft_bundle = load_model_bundle(config.draft)

    # Benchmark naive (baseline)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    _, naive_tokens = naive_generate(verify_bundle, config.sampling, args.prompt, args.max_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    naive_time = time.perf_counter() - start

    # Benchmark speculative decoding
    decoder = SpeculativeDecoder(draft_bundle, verify_bundle, config.sampling, draft_k=draft_k)
    stats = SpeculativeStats()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    _, spec_stats = decoder.generate(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        stats=stats,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    spec_time = time.perf_counter() - start

    # Compute throughput: tokens / second
    naive_tps = naive_tokens / max(naive_time, 1e-6)
    spec_tps = spec_stats.total_emitted / max(spec_time, 1e-6)

    print(f"Naive tokens: {naive_tokens}, time: {naive_time:.2f}s, tokens/sec: {naive_tps:.2f}")
    print(f"Spec tokens: {spec_stats.total_emitted}, time: {spec_time:.2f}s, tokens/sec: {spec_tps:.2f}")
    print(f"Acceptance rate: {spec_stats.acceptance_rate:.2f}")


if __name__ == "__main__":
    main()
