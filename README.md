# Speculative Decoding Inference Engine

A high-performance LLM inference engine demonstrating **speculative decoding**, an acceleration technique that significantly improves token generation throughput while maintaining accuracy.

### Key Efficiency Techniques

- **KV-Cache Reuse**: Past key/values from both models are cached and reused, avoiding redundant computation
- **Batch Processing**: Verify model processes k draft tokens in parallel (batch size = k) rather than sequentially
- **Prefix Matching**: On mismatch, caches are trimmed and repositioned to the agreed-upon prefix
- **Quantization Support**: 4-bit/8-bit quantization reduces memory footprint for larger models
- **Optional CUDA Kernels**: Templated support for fused operations (temperature scaling, top-k/top-p filtering)

## Project Structure

```
inference-engine/
├── main.py                 # Entry point for interactive generation with visualization
├── benchmark.py            # Benchmarking: naive vs speculative throughput comparison
├── configs/
│   └── models.yaml        # Model configuration (draft/verify models, sampling params)
├── engine/                # Core inference engine components
│   ├── config.py          # Configuration loading (YAML → dataclasses)
│   ├── model_loader.py    # Model loading with quantization support
│   ├── kv_cache.py        # KV cache management and trimming logic
│   └── sampler.py         # Token sampling: temperature, top-k, top-p
├── speculative/           # Speculative decoding algorithm
│   ├── spec_decoder.py    # Core: draft tokens → verify → accept/reject logic
│   └── stats.py           # Acceptance rate, speedup, and throughput metrics
├── visualization/         # Real-time terminal UI
│   └── terminal_viz.py    # Live stats display during generation
├── cuda_ops/              # Optional CUDA kernels (not included; template structure)
│   ├── kv_ops.cu
│   └── sampling_kernel.cu
└── requirements.txt       # Python dependencies
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Build CUDA kernels (if available)
cd cuda_ops && python setup.py build_ext --inplace
```

### Configuration

Create `configs/models.yaml`:

### Generate Text

```bash
python main.py \
  --config configs/models.yaml \
  --prompt "Explain Speculative Decoding" \
  --max_tokens 200 \
  --draft_k 5
```

**Output**: Live terminal UI showing:
- **Acceptance Bar**: Visual feedback on draft-verify agreement
- **Stats Panel**: Steps, acceptance rate, avg tokens/step, speedup, memory usage
- **Generated Text**: Streaming text output

### Benchmark

Compare naive autoregressive vs speculative decoding:

```bash
python benchmark.py \
  --config configs/models.yaml \
  --prompt "The future of AI is" \
  --max_tokens 200 \
  --draft_k 5
```

**Output**:
```
Naive tokens: 50, time: 12.34s, tokens/sec: 4.05
Spec tokens: 50, time: 4.56s, tokens/sec: 10.97
Acceptance rate: 0.72
```

## Data Dimensions Reference

Understanding tensor shapes throughout the pipeline is critical for debugging and optimization:

### Model Outputs & Key Variables

- **Prompt tokens** after encoding: `(batch=1, seq_len=prompt_length)`
- **Draft tokens** generated per step: `(batch=1, k)` where k is typically 5-10
- **Verify logits** from draft tokens: `(batch=1, seq_len=k, vocab_size)`
- **KV cache per layer**: 
  - 4D format: `(batch=1, num_heads, seq_len, head_dim)`
  - 3D format: `(batch=1, seq_len, hidden_dim)`
- **Generated token list**: `[]` accumulates integers (token IDs), max size = `max_tokens`

### Stats Tracked

- `total_proposed`: Sum of all draft tokens proposed (= steps × draft_k)
- `total_accepted`: Sum of draft tokens accepted by verifier
- `total_emitted`: Total output tokens = accepted + 1 (for mismatches)
- `acceptance_rate`: `total_accepted / total_proposed` (target: 0.7-0.9)
- `avg_tokens_per_step`: `total_emitted / steps` (typical speedup multiplier)

## Troubleshooting

- **Out of Memory**: Reduce quantization precision or draft_k
- **Low Acceptance Rate**: Increase temperature or reduce top_k/top_p for more conservative draft model
- **Slow Generation**: Verify draft model is significantly smaller/faster than verify; profile with `benchmark.py`

## References

- **Speculative Decoding Paper**: ["Accelerating Large Language Model Decoding with Speculative Execution"](https://arxiv.org/abs/2211.17192)
- **KV-Cache Optimization**: ["Efficient Transformers with KV-Cache"](https://arxiv.org/abs/2305.18323)
- **Quantization (4-bit)**: BitsAndBytes library

---
