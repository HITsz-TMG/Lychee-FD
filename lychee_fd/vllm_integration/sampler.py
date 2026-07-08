"""
Multi-head sampler for the Lychee-FD full-duplex model.

The default vLLM sampler consumes a single logits tensor. This module keeps
sampling explicit for the model's three output heads:
  1. Accept text, stoken, and control logits from model forward.
  2. Apply per-head LogitsProcessor lists.
  3. Sample each head independently:
       - text:    greedy argmax by default
       - stoken:  nucleus/top-k/temperature sampling by default
       - control: greedy argmax by default
  4. Return all three sampled tokens in MultiHeadSamplerOutput.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

@dataclass
class MultiHeadSamplingParams:
    """Per-head sampling configuration."""
    text_do_sample: bool = False
    text_temperature: float = 1.0
    text_top_k: int = 0
    text_top_p: float = 1.0

    stoken_do_sample: bool = True
    stoken_temperature: float = 0.7
    stoken_top_k: int = 0
    stoken_top_p: float = 1.0

    control_do_sample: bool = False
    control_temperature: float = 1.0
    control_top_k: int = 0
    control_top_p: float = 1.0

    @classmethod
    def from_generation_kwargs(
        cls,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        text_do_sample: bool = False,
        stoken_do_sample: Optional[bool] = None,
        control_do_sample: bool = False,
    ) -> "MultiHeadSamplingParams":
        """
        Build per-head sampling params from a HF-style generation call.

        The duplex framework mainly passes shared kwargs (`temperature`, `top_k`,
        `top_p`) and expects text/control to stay deterministic while stoken can
        sample. This helper keeps that behavior explicit.
        """
        if stoken_do_sample is None:
            stoken_do_sample = temperature != 0

        safe_temp = float(temperature)
        safe_top_k = int(top_k) if top_k is not None else 0
        safe_top_p = float(top_p) if top_p is not None else 1.0

        return cls(
            text_do_sample=text_do_sample,
            text_temperature=safe_temp if text_do_sample else 1.0,
            text_top_k=safe_top_k if text_do_sample else 0,
            text_top_p=safe_top_p if text_do_sample else 1.0,
            stoken_do_sample=stoken_do_sample,
            stoken_temperature=safe_temp,
            stoken_top_k=safe_top_k,
            stoken_top_p=safe_top_p,
            control_do_sample=control_do_sample,
            control_temperature=safe_temp if control_do_sample else 1.0,
            control_top_k=safe_top_k if control_do_sample else 0,
            control_top_p=safe_top_p if control_do_sample else 1.0,
        )


@dataclass
class MultiHeadSamplerOutput:
    """Output from multi-head sampling containing all 3 token selections."""
    text_token: int
    stoken_token: int
    control_token: int
    text_logprobs: Optional[Dict[int, float]] = None
    stoken_logprobs: Optional[Dict[int, float]] = None
    control_logprobs: Optional[Dict[int, float]] = None


class MultiHeadSampler:
    """
    Stateless sampler operating on the 3-tuple output from
    LycheeDuplexForVLLM.forward().

    Usage:
        sampler = MultiHeadSampler()
        output = sampler.sample(
            text_logits, stoken_logits, control_logits,
            params=MultiHeadSamplingParams(...),
            text_processors=[...],
            stoken_processors=[...],
            control_processors=[...],
            text_input_ids=..., stoken_input_ids=..., control_input_ids=...,
        )
    """

    @staticmethod
    def _apply_processors(logits, input_ids, processors):
        if processors is None:
            return logits
        for proc in processors:
            logits = proc(input_ids, logits)
        return logits

    @staticmethod
    def _sample_single(
        logits: torch.Tensor,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[int, Optional[Dict[int, float]]]:
        """
        Sample a single token from a [1, vocab_size] or [vocab_size] logits tensor.
        Returns (token_id, logprobs_dict_or_None).
        """
        if logits.dim() == 2:
            logits = logits[0]

        if not do_sample or temperature == 0:
            token_id = torch.argmax(logits).item()
            return token_id, None

        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            vals, _ = torch.topk(logits, top_k)
            pivot = vals[-1]
            logits = torch.where(logits < pivot, torch.tensor(float('-inf'), device=logits.device), logits)

        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[1:] = sorted_mask[:-1].clone()
            sorted_mask[0] = False
            remove_mask = sorted_mask.scatter(0, sorted_indices, sorted_mask)
            logits = logits.masked_fill(remove_mask, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        if torch.isnan(probs).any() or probs.sum() == 0:
            probs = torch.ones_like(probs) / probs.shape[-1]

        token_id = torch.multinomial(probs, num_samples=1).item()

        top5_vals, top5_idx = torch.topk(probs, min(5, probs.shape[-1]))
        logprobs = {idx.item(): float(torch.log(val + 1e-10)) for idx, val in zip(top5_idx, top5_vals)}

        return token_id, logprobs

    def sample(
        self,
        text_logits: torch.Tensor,
        stoken_logits: torch.Tensor,
        control_logits: torch.Tensor,
        params: Optional[MultiHeadSamplingParams] = None,
        text_processors: Optional[List] = None,
        stoken_processors: Optional[List] = None,
        control_processors: Optional[List] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        stoken_input_ids: Optional[torch.Tensor] = None,
        control_input_ids: Optional[torch.Tensor] = None,
    ) -> MultiHeadSamplerOutput:
        """
        Sample from all 3 heads independently.

        Args:
            text_logits:    [batch=1, vocab_size] logits for the text head
            stoken_logits:  [batch=1, vocab_size] logits for the stoken head
            control_logits: [batch=1, vocab_size] logits for the control head
            params:         per-head sampling configuration
            *_processors:   list of LogitsProcessor callables for each head
            *_input_ids:    [batch=1, seq_len] historical tokens for processors

        Returns:
            MultiHeadSamplerOutput with the 3 sampled token IDs
        """
        if params is None:
            params = MultiHeadSamplingParams()

        if text_input_ids is not None:
            text_logits = self._apply_processors(text_logits, text_input_ids, text_processors)
        if stoken_input_ids is not None:
            stoken_logits = self._apply_processors(stoken_logits, stoken_input_ids, stoken_processors)
        if control_input_ids is not None:
            control_logits = self._apply_processors(control_logits, control_input_ids, control_processors)

        text_token, text_lp = self._sample_single(
            text_logits, params.text_do_sample,
            params.text_temperature, params.text_top_k, params.text_top_p,
        )
        stoken_token, stoken_lp = self._sample_single(
            stoken_logits, params.stoken_do_sample,
            params.stoken_temperature, params.stoken_top_k, params.stoken_top_p,
        )
        control_token, control_lp = self._sample_single(
            control_logits, params.control_do_sample,
            params.control_temperature, params.control_top_k, params.control_top_p,
        )

        return MultiHeadSamplerOutput(
            text_token=text_token,
            stoken_token=stoken_token,
            control_token=control_token,
            text_logprobs=text_lp,
            stoken_logprobs=stoken_lp,
            control_logprobs=control_lp,
        )


class BatchMultiHeadSampler(MultiHeadSampler):
    """
    Extension that can handle batched requests when vLLM continuous batching
    is enabled. Each request in the batch may have different sampling params.
    """

    def sample_batch(
        self,
        text_logits: torch.Tensor,
        stoken_logits: torch.Tensor,
        control_logits: torch.Tensor,
        params_list: List[MultiHeadSamplingParams],
        text_processors_list: Optional[List[Optional[List]]] = None,
        stoken_processors_list: Optional[List[Optional[List]]] = None,
        control_processors_list: Optional[List[Optional[List]]] = None,
        text_input_ids_list: Optional[List[Optional[torch.Tensor]]] = None,
        stoken_input_ids_list: Optional[List[Optional[torch.Tensor]]] = None,
        control_input_ids_list: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> List[MultiHeadSamplerOutput]:
        """
        Sample for a batch of requests.
        logits shape: [batch_size, vocab_size]
        """
        batch_size = text_logits.shape[0]
        results = []

        for i in range(batch_size):
            t_log = text_logits[i:i+1]
            s_log = stoken_logits[i:i+1]
            c_log = control_logits[i:i+1]

            t_proc = text_processors_list[i] if text_processors_list else None
            s_proc = stoken_processors_list[i] if stoken_processors_list else None
            c_proc = control_processors_list[i] if control_processors_list else None

            t_ids = text_input_ids_list[i] if text_input_ids_list else None
            s_ids = stoken_input_ids_list[i] if stoken_input_ids_list else None
            c_ids = control_input_ids_list[i] if control_input_ids_list else None

            output = self.sample(
                t_log, s_log, c_log,
                params=params_list[i],
                text_processors=t_proc,
                stoken_processors=s_proc,
                control_processors=c_proc,
                text_input_ids=t_ids,
                stoken_input_ids=s_ids,
                control_input_ids=c_ids,
            )
            results.append(output)

        return results
