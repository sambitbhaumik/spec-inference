import argparse

from engine.config import load_engine_config
from engine.model_loader import load_model_bundle
from speculative.spec_decoder import SpeculativeDecoder
from speculative.stats import SpeculativeStats
from visualization.terminal_viz import TerminalVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speculative decoding inference engine")
    parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate from")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of new tokens")
    parser.add_argument("--draft_k", type=int, default=None, help="Draft tokens per step")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_engine_config(args.config)

    draft_k = args.draft_k if args.draft_k is not None else (config.draft.max_tokens or 5)

    draft_bundle = load_model_bundle(config.draft)
    verify_bundle = load_model_bundle(config.verify)

    decoder = SpeculativeDecoder(
        draft=draft_bundle,
        verify=verify_bundle,
        sampling=config.sampling,
        draft_k=draft_k,
    )

    stats = SpeculativeStats()

    with TerminalVisualizer() as viz:
        text, _ = decoder.generate(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            stats=stats,
            step_callback=viz.update,
        )

    print(text)


if __name__ == "__main__":
    main()
