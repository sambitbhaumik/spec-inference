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
        self.draft = draft
        self.verify = verify
        self.sampling = sampling
        self.draft_k = draft_k

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
        if stats is None:
            stats = SpeculativeStats()

        tokenizer = self.verify.tokenizer
        device = self.verify.device

        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)

        eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

        with torch.no_grad():
            draft_cache = self._prime_cache(self.draft.model, input_ids)
            verify_cache = self._prime_cache(self.verify.model, input_ids)

            generated: List[int] = []
            last_token = input_ids[:, -1:]

            while len(generated) < max_tokens:
                prefix_len = verify_cache.seq_len

                draft_tokens, draft_cache = self._draft_tokens(
                    last_token,
                    draft_cache,
                    self.draft_k,
                )

                verify_output = self.verify.model(
                    input_ids=draft_tokens,
                    past_key_values=verify_cache.past_key_values,
                    use_cache=True,
                )
                verify_tokens = torch.argmax(verify_output.logits, dim=-1)

                accept_len = self._accept_length(draft_tokens, verify_tokens)
                emitted_tokens: List[int] = []

                if accept_len == self.draft_k:
                    verify_cache = KVCache.from_past_key_values(verify_output.past_key_values)
                    emitted_tokens = draft_tokens.squeeze(0).tolist()
                else:
                    accepted = draft_tokens[:, :accept_len]
                    mismatch_token = verify_tokens[:, accept_len]
                    emitted_tokens = accepted.squeeze(0).tolist() if accept_len > 0 else []
                    emitted_tokens.append(int(mismatch_token.item()))

                    verify_trim = KVCache.from_past_key_values(verify_output.past_key_values).trim(
                        prefix_len + accept_len
                    )
                    verify_cache = self._append_token(self.verify.model, verify_trim, mismatch_token)

                    draft_cache = draft_cache.trim(prefix_len + accept_len)
                    draft_cache = self._append_token(self.draft.model, draft_cache, mismatch_token)

                actual_emitted: List[int] = []
                for token_id in emitted_tokens:
                    generated.append(token_id)
                    actual_emitted.append(token_id)
                    last_token = torch.tensor([[token_id]], device=device)
                    if eos_token_id is not None and token_id == eos_token_id:
                        stats.update(self.draft_k, accept_len, len(actual_emitted))
                        if step_callback is not None:
                            new_text = tokenizer.decode(actual_emitted, skip_special_tokens=True)
                            step_callback(stats, new_text)
                        text = tokenizer.decode(generated, skip_special_tokens=True)
                        return text, stats
                    if len(generated) >= max_tokens:
                        break

                stats.update(self.draft_k, accept_len, len(actual_emitted))
                if actual_emitted and step_callback is not None:
                    new_text = tokenizer.decode(actual_emitted, skip_special_tokens=True)
                    step_callback(stats, new_text)

        text = tokenizer.decode(generated, skip_special_tokens=True)
        return text, stats

    def _prime_cache(self, model, input_ids: torch.Tensor) -> KVCache:
        outputs = model(input_ids=input_ids, use_cache=True)
        return KVCache.from_past_key_values(outputs.past_key_values)

    def _draft_tokens(
        self,
        last_token: torch.Tensor,
        cache: KVCache,
        k: int,
    ) -> Tuple[torch.Tensor, KVCache]:
        tokens: List[torch.Tensor] = []
        current_token = last_token
        current_cache = cache

        for _ in range(k):
            outputs = self.draft.model(
                input_ids=current_token,
                past_key_values=current_cache.past_key_values,
                use_cache=True,
            )
            next_token = sample_next_token(outputs.logits[:, -1, :], self.sampling)
            tokens.append(next_token)
            current_cache = KVCache.from_past_key_values(outputs.past_key_values)
            current_token = next_token.unsqueeze(1)

        token_tensor = torch.stack(tokens, dim=1)
        return token_tensor, current_cache

    @staticmethod
    def _append_token(model, cache: KVCache, token: torch.Tensor) -> KVCache:
        outputs = model(input_ids=token.unsqueeze(1), past_key_values=cache.past_key_values, use_cache=True)
        return KVCache.from_past_key_values(outputs.past_key_values)

    @staticmethod
    def _accept_length(draft_tokens: torch.Tensor, verify_tokens: torch.Tensor) -> int:
        draft_list = draft_tokens.squeeze(0).tolist()
        verify_list = verify_tokens.squeeze(0).tolist()
        accept_len = 0
        for draft_token, verify_token in zip(draft_list, verify_list):
            if draft_token == verify_token:
                accept_len += 1
            else:
                break
        return accept_len
