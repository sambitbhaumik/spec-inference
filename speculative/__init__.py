"""Speculative decoding: Fast inference by predicting multiple tokens with a draft model and verifying with a larger model."""
from speculative.spec_decoder import SpeculativeDecoder
from speculative.stats import SpeculativeStats

__all__ = [
    "SpeculativeDecoder",
    "SpeculativeStats",
]
