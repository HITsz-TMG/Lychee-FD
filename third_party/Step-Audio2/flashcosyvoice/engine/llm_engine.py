import atexit
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from flashcosyvoice.config import Config, SamplingParams
from flashcosyvoice.engine.model_runner import ModelRunner
from flashcosyvoice.engine.scheduler import Scheduler
from flashcosyvoice.engine.sequence import Sequence


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        assert config.tensor_parallel_size == 1, "NOTE(xcsong): Currently only support tp=1"
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        if hasattr(config.hf_config, "speech_vocab_size"):
            # NOTE: non-chat model, all these special tokens keep randomly initialized.
            special_tokens = {
                'eos_token': '<|endoftext|>',
                'pad_token': '<|endoftext|>',
                'additional_special_tokens': [
                    '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                    '[breath]', '<strong>', '</strong>', '[noise]',
                    '[laughter]', '[cough]', '[clucking]', '[accent]',
                    '[quick_breath]',
                    "<laughter>", "</laughter>",
                    "[hissing]", "[sigh]", "[vocalized-noise]",
                    "[lipsmack]", "[mn]"
                ]
            }
            self.tokenizer = AutoTokenizer.from_pretrained(f"{config.model}/CosyVoice-BlankEN")
            self.tokenizer.add_special_tokens(special_tokens)
            self.skip_special_tokens = True
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        if hasattr(config.hf_config, "eos_token_id"):
            config.eos = config.hf_config.eos_token_id
        else:
            config.eos = self.tokenizer.eos_token_id
        self.model_runner = ModelRunner(config, config.rank, self.events)
        self.scheduler = Scheduler(config)
        self.config = config
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating tokens (LLM)", leave=False,
                        dynamic_ncols=True, position=self.config.rank + 1)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = instant_decode_throughput = 0.
        total_decode_tokens = 0
        total_decode_time = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            step_time = perf_counter() - t
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / step_time
                else:
                    instant_decode_throughput = -num_tokens / step_time
                    total_decode_tokens += -num_tokens
                    total_decode_time += step_time
                    decode_throughput = total_decode_tokens / total_decode_time if total_decode_time > 0 else 0
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "AvgDecode": f"{int(decode_throughput)}tok/s",
                    "InstDecode": f"{int(instant_decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
