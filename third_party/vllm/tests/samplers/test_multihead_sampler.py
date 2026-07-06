import torch

from vllm.model_executor.layers.sampler import (SampleResultArgsType,
                                                _sample_multihead_token_ids)
from vllm.model_executor.sampling_metadata import (SamplingMetadata,
                                                   SequenceGroupToSample)
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SequenceData


def _build_sampling_metadata(
    sampling_params: SamplingParams,
    sample_indices: list[int],
) -> SamplingMetadata:
    seq_group = SequenceGroupToSample(
        seq_ids=[0],
        sampling_params=sampling_params,
        seq_data={0: SequenceData.from_seqs([1, 2, 3])},
        seq_len=None,
        query_len=None,
        generator=None,
        is_prompt=False,
        prompt_logprob_indices=[],
        sample_indices=sample_indices,
    )
    categorized_indices = {
        sampling_type: torch.empty((0, ), dtype=torch.int64)
        for sampling_type in SamplingType
    }
    return SamplingMetadata(
        seq_groups=[seq_group],
        selected_token_indices=torch.tensor(sample_indices, dtype=torch.long),
        categorized_sample_indices=categorized_indices,
        num_prompts=0,
    )


def test_multihead_sampler_deterministic_fixed_logits() -> None:
    logits = torch.tensor([[0.1, 1.2, -2.0, 3.4]], dtype=torch.float)
    sampling_params = SamplingParams(
        temperature=0.0,
        multihead_sampling={
            "stoken": {
                "do_sample": False,
                "temperature": 1.0,
                "top_k": -1,
                "top_p": 1.0,
            },
            "control": {
                "do_sample": False,
                "temperature": 1.0,
                "top_k": 1,
                "top_p": 1.0,
            },
        },
    )
    sampling_metadata = _build_sampling_metadata(sampling_params, [0])

    stoken_token_ids, control_token_ids = _sample_multihead_token_ids(
        logits=logits,
        sampling_metadata=sampling_metadata,
        maybe_deferred_sample_results=[([3], [0])],
    )

    assert stoken_token_ids == [[3]]
    assert control_token_ids == [[3]]


def test_multihead_sampler_deferred_path_uses_first_parent() -> None:
    logits = torch.tensor([[1.0, 0.5, 2.0]], dtype=torch.float)
    sampling_params = SamplingParams(
        temperature=0.0,
        multihead_sampling={
            "stoken": {
                "do_sample": False,
                "temperature": 1.0,
                "top_k": -1,
                "top_p": 1.0,
            },
        },
    )
    sampling_metadata = _build_sampling_metadata(sampling_params, [0])
    deferred_args = SampleResultArgsType(
        sample_metadata={},
        multinomial_samples={},
        sample_results_dict={},
        sampling_metadata=sampling_metadata,
        greedy_samples=None,
        beam_search_logprobs=None,
    )

    stoken_token_ids, control_token_ids = _sample_multihead_token_ids(
        logits=logits,
        sampling_metadata=sampling_metadata,
        maybe_deferred_sample_results=deferred_args,
    )

    assert stoken_token_ids == [[2]]
    assert control_token_ids is None
