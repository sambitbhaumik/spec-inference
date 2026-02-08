"""
Core speculative decoding implementation.
Generates tokens efficiently by: (1) draft model generates k candidate tokens,
(2) verify model validates them in parallel, (3) accept matching tokens, reject mismatches.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from engine.kv_cache import KVCache
from engine.model_loader import ModelBundle
from engine.sampler import SamplingConfig, sample_next_token
from speculative.stats import SpeculativeStats


class SpeculativeDecoder:
    def __init__(
        self,
        draft: ModelBundle,
        verify: ModelBundle,
        sampling: SamplingConfig,
        draft_k: int = 5,
    ) -> None:
        """
        Initialize decoder with draft (small/fast) and verify (large/accurate) models.
        
        Args:
            draft: Small model for fast token proposals (batch=1, seq_len varies)
            verify: Large model for verification (batch=1, seq_len varies)
            sampling: Temperature, top-k, top-p configuration
            draft_k: Number of tokens to speculate per step (typically 5-10)
        """
        self.draft = draft
        self.verify = verify
        self.sampling = sampling
        self.draft_k = draft_k

        # Ensure both models share the same vocabulary for valid comparison
        if self.draft.tokenizer.get_vocab() != self.verify.tokenizer.get_vocab():
            raise ValueError("Draft and verify tokenizers must share the same vocabulary.")

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        stats: Optional[SpeculativeStats] = None,
        eos_token_id: Optional[int] = None,
        step_callback=None,
    ) -> Tuple[str, SpeculativeStats]:
        """
        Generate text using speculative decoding.
        
        Args:
            prompt: Initial text prompt (str)
            max_tokens: Max new tokens to generate (int)
            stats: Stats tracker for acceptance rate and speedup
            eos_token_id: End-of-sequence token ID
            step_callback: Callback(stats, new_text) called after each step
            
        Returns:
            (generated_text, stats) where generated_text is str
        """
        if stats is None:
            stats = SpeculativeStats()

        tokenizer = self.verify.tokenizer
        device = self.verify.device

        # Encode prompt: shape (1, prompt_len)
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)

        eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

        with torch.no_grad():
            # Pre-fill both caches with the prompt
            draft_cache = self._prime_cache(self.draft.model, input_ids)
            verify_cache = self._prime_cache(self.verify.model, input_ids)

            generated: List[int] = []
            last_token = input_ids[:, -1:]  # shape (1, 1)

            while len(generated) < max_tokens:
                prefix_len = verify_cache.seq_len

                # Step 1: Draft model generates k candidate tokens
                # draft_tokens: shape (1, draft_k)
                draft_tokens, draft_cache = self._draft_tokens(
                    last_token,
                    draft_cache,
                    self.draft_k,
                )

                # Step 2: Verify model validates draft tokens in parallel
                # verify_output.logits: shape (1, draft_k, vocab_size)
                verify_output = self.verify.model(
                    input_ids=draft_tokens,
                    past_key_values=verify_cache.past_key_values,
                    use_cache=True,
                )
                # Get argmax predictions: shape (1, draft_k)
                verify_tokens = torch.argmax(verify_output.logits, dim=-1)

                # Step 3: Find longest prefix match between draft and verify
                accept_len = self._accept_length(draft_tokens, verify_tokens)
                emitted_tokens: List[int] = []

                if accept_len == self.draft_k:
                    # All k tokens match! Update cache and emit all
                    verify_cache = KVCache.from_past_key_values(verify_output.past_key_values)
                    emitted_tokens = draft_tokens.squeeze(0).tolist()
                else:
                    # Partial match: accept accept_len tokens, replace with verify prediction
                    accepted = draft_tokens[:, :accept_len]
                    mismatch_token = verify_tokens[:, accept_len]
                    emitted_tokens = accepted.squeeze(0).tolist() if accept_len > 0 else []
                    emitted_tokens.append(int(mismatch_token.item()))

                    # Trim caches to accepted length and append verifier's token
                    verify_trim = KVCache.from_past_key_values(verify_output.past_key_values).trim(
                        prefix_len + accept_len
                    )
                    verify_cache = self._append_token(self.verify.model, verify_trim, mismatch_token)

                    draft_cache = draft_cache.trim(prefix_len + accept_len)
                    draft_cache = self._append_token(self.draft.model, draft_cache, mismatch_token)

                # Process emitted tokens and update stats
                actual_emitted: List[int] = []
                for token_id in emitted_tokens:
                    generated.append(token_id)
                    actual_emitted.append(token_id)
                    last_token = torch.tensor([[token_id]], device=device)
                    
                    # Check for EOS token
                    if eos_token_id is not None and token_id == eos_token_id:
                        stats.update(self.draft_k, accept_len, len(actual_emitted))
                        if step_callback is not None:
                            new_text = tokenizer.decode(actual_emitted, skip_special_tokens=True)
                            step_callback(stats, new_text)
                        text = tokenizer.decode(generated, skip_special_tokens=True)
                        return text, stats
                    
                    if len(generated) >= max_tokens:
                        break

                # Update stats and invoke callback
                stats.update(self.draft_k, accept_len, len(actual_emitted))
                if actual_emitted and step_callback is not None:
                    new_text = tokenizer.decode(actual_emitted, skip_special_tokens=True)
                    step_callback(stats, new_text)

        text = tokenizer.decode(generated, skip_special_tokens=True)
        return text, stats

    def _prime_cache(self, model, input_ids: torch.Tensor) -> KVCache:
        """
        Pre-fill KV cache with initial prompt.
        
        Args:
            model: Transformer model
            input_ids: Prompt tokens, shape (1, prompt_len)
            
        Returns:
            KVCache with cached key/values after processing prompt
        """
        outputs = model(input_ids=input_ids, use_cache=True)
        return KVCache.from_past_key_values(outputs.past_key_values)

    def _draft_tokens(
        self,
        last_token: torch.Tensor,
        cache: KVCache,
        k: int,
    ) -> Tuple[torch.Tensor, KVCache]:
        """
        Generate k candidate tokens from draft model autoregressively.
        
        Args:
            last_token: Previous token, shape (1, 1)
            cache: KV cache for draft model
            k: Number of tokens to generate
            
        Returns:
            (draft_tokens, updated_cache) where draft_tokens shape is (1, k)
        """
        tokens: List[torch.Tensor] = []
        current_token = last_token
        current_cache = cache

        for _ in range(k):
            outputs = self.draft.model(
                input_ids=current_token,
                past_key_values=current_cache.past_key_values,
                use_cache=True,
            )
            # Sample token from draft logits, shape (1, 1)
            next_token = sample_next_token(outputs.logits[:, -1, :], self.sampling)
            tokens.append(next_token)
            current_cache = KVCache.from_past_key_values(outputs.past_key_values)
            current_token = next_token.unsqueeze(1)

        # Stack all tokens: (1, k)
        token_tensor = torch.stack(tokens, dim=1)
        return token_tensor, current_cache

    @staticmethod
    def _append_token(model, cache: KVCache, token: torch.Tensor) -> KVCache:
        """
        Append single token to cache by running model once.
        
        Args:
            model: Transformer model
            cache: KV cache before appending
            token: Token ID, shape (1,)
            
        Returns:
            Updated KV cache
        """
        outputs = model(input_ids=token.unsqueeze(1), past_key_values=cache.past_key_values, use_cache=True)
        return KVCache.from_past_key_values(outputs.past_key_values)

    @staticmethod
    def _accept_length(draft_tokens: torch.Tensor, verify_tokens: torch.Tensor) -> int:
        """
        Find length of longest matching prefix between draft and verify predictions.
        
        Args:
            draft_tokens: Shape (1, k)
            verify_tokens: Shape (1, k)
            
        Returns:
            Number of matching tokens from start (0 to k)
        """
        draft_list = draft_tokens.squeeze(0).tolist()
        verify_list = verify_tokens.squeeze(0).tolist()
        accept_len = 0
        for draft_token, verify_token in zip(draft_list, verify_list):
            if draft_token == verify_token:
                accept_len += 1
            else:
                break
        return accept_len
