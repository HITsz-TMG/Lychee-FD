# vLLM Realtime Memory Growth Analysis

Date: 2026-07-09

## Summary

The current evidence does **not** point to residual SoulX truncation logic as the direct cause of vLLM memory growth. The current codebase no longer has an active `STEPAUDIO_SOULX_*` / SoulX idle-compaction path in the realtime inference loop.

The most likely cause is the realtime vLLM branch's cumulative audio-side cache in `IncrementalChunkStreamSession`. Each incremental audio window is converted into audio token ids, waveform features, and precomputed audio embeddings. These are retained for the lifetime of the realtime session, and the cached audio context is rebuilt on GPU every round.

This cache lives outside vLLM's internal KV cache budget. Therefore, `gpu_memory_utilization=0.50` or `0.90` can still show increasing `nvidia-smi` memory if Python/PyTorch allocations grow outside the vLLM block manager.

## Current Cleanup Instrumentation

The realtime worker now explicitly closes the incremental stream session when a call finishes or exits with an error. The close path aborts the kept-alive vLLM request, clears per-call audio/text/control caches, runs Python garbage collection, and logs CUDA memory counters.

The important distinction is:

- `cuda_after_gc.allocated_mib`: live CUDA tensor memory still referenced by the process.
- `cuda_after_gc.reserved_mib`: memory reserved by the CUDA caching allocator or vLLM runtime for reuse.
- `nvidia-smi`: process-level driver memory, which usually follows reserved memory rather than live tensor memory.

If `allocated_mib` drops after close but `reserved_mib` and `nvidia-smi` stay high, the session tensors were released and the remaining memory is allocator/runtime reservation. For a validation-only run, set `STEPAUDIO_CUDA_EMPTY_CACHE_ON_SESSION_CLOSE=1` to call `torch.cuda.empty_cache()` after session close. This should not be the default serving behavior because it can reduce reuse efficiency and increase latency on the next call.

## Observed Measurements

From the manual runs discussed during debugging:

| Backend | Start memory | Later memory | Duration | Increment |
| --- | ---: | ---: | ---: | ---: |
| HF online | ~22 GB | ~40 GB | 6 min 15 s | ~18 GB |
| vLLM online, `gpu_memory_utilization=0.50` | 42239 MiB | 57049 MiB | 6 min 30 s | 14810 MiB |

Duration-normalized, the vLLM run still grows, but its incremental growth rate is roughly 23% lower than the HF run:

- HF: ~18 GB / 6.25 min = ~2.88 GB/min
- vLLM: ~14.46 GiB / 6.5 min = ~2.22 GiB/min

This supports the conclusion that vLLM reduces part of the long-session memory pressure, but there is still a separate accumulation path in the realtime code.

## Why It Is Probably Not SoulX

Static search shows no active SoulX-specific configuration path in the current realtime code. The remaining truncation-related code is vLLM tail-truncation infrastructure:

- `lychee_fd/runtime/vllm_generation.py` defines `truncate_active_request_to_prefix(...)`.
- `lychee_fd/vllm_integration/engine.py` defines `truncate_active_request_to_sequences(...)`.
- The realtime worker does not currently call `truncate_active_request_to_prefix(...)`.

So the current behavior is closer to: truncation support exists, but no periodic realtime compaction is driving it.

## Primary Suspect: Incremental Audio Cache

`IncrementalChunkStreamSession` initializes cumulative audio caches:

- `lychee_fd/runtime/vllm_generation.py:2337` stores `_audio_tail`.
- `lychee_fd/runtime/vllm_generation.py:2338` stores `_audio_input_id_list`.
- `lychee_fd/runtime/vllm_generation.py:2340` stores `_audio_feats_cache`.
- `lychee_fd/runtime/vllm_generation.py:2342` stores `_audio_embed_seq_cache`.

During every incremental window, `_append_incremental_audio(...)` appends new cached data:

- `lychee_fd/runtime/vllm_generation.py:2732` appends per-window audio token lengths.
- `lychee_fd/runtime/vllm_generation.py:2734` extends cumulative audio token ids.
- `lychee_fd/runtime/vllm_generation.py:2736` extends cumulative audio feature tensors.
- `lychee_fd/runtime/vllm_generation.py:2738` computes and caches precomputed audio embeddings.

The strongest GPU-memory candidate is `_cache_precomputed_audio(...)`:

- `lychee_fd/runtime/vllm_generation.py:2684` pads waveform features and moves them to `self.fw.device`.
- `lychee_fd/runtime/vllm_generation.py:2692` calls vLLM-side audio precompute.
- `lychee_fd/runtime/vllm_generation.py:2710` appends `out[i, :li, :].detach()` into `_audio_embed_seq_cache`.

`detach()` only removes autograd tracking. It does not move the tensor to CPU. If `out` is a CUDA tensor, every cached audio embedding remains on GPU until the session is released or the cache is explicitly pruned.

Then every round rebuilds the full cached audio context:

- `lychee_fd/runtime/vllm_generation.py:2745` creates cumulative `audio_input_ids` on GPU.
- `lychee_fd/runtime/vllm_generation.py:2753` repacks all cached waveform features and moves them to GPU.
- `lychee_fd/runtime/vllm_generation.py:2763` repacks all cached audio embeddings.
- `lychee_fd/runtime/vllm_generation.py:2772` passes the packed audio embeddings to the vLLM backend.

This explains the observed pattern where memory can remain flat for a while and then continue growing: PyTorch's CUDA allocator may reuse reserved blocks until a larger cumulative packed tensor is needed; then it reserves more memory, and `nvidia-smi` increases.

## vLLM KV Cache Is A Different Budget

The vLLM logs show the internal budget clearly:

```text
total_gpu_memory x gpu_memory_utilization = vLLM usable memory
model weights take ...
the rest of the memory reserved for KV Cache is ...
# GPU blocks: ...
Maximum concurrency for 16384 tokens per request: ...
```

That budget is for vLLM-managed KV blocks. It does not cap external allocations such as:

- cached audio embeddings held in `IncrementalChunkStreamSession`;
- temporary packed tensors created before each vLLM call;
- PyTorch tensors in the app/session layer;
- Token2Wav memory if Token2Wav runs on the same GPU.

If external allocations keep growing, vLLM does not "borrow more KV cache" in a controlled way. PyTorch/vLLM can still allocate additional GPU memory outside the preprofiled block cache until the device runs out, at which point the likely failure mode is CUDA OOM.

## Secondary Memory Risks

These are probably not the main vLLM GPU issue, but they can affect long runs or repeated sessions.

### Session Audio Retention

`RealtimeSessionState` stores all received audio and pending audio:

- `lychee_fd/app.py:2567` defines `all_audio_chunks`.
- `lychee_fd/app.py:2568` defines `pending_audio_chunks`.
- `lychee_fd/app.py:4293` appends every incoming chunk to `all_audio_chunks`.
- `lychee_fd/app.py:4294` appends every incoming chunk to `pending_audio_chunks`.
- `lychee_fd/app.py:3516` only drops processed `pending_audio_chunks`.

`all_audio_chunks` is never pruned during the session. This is CPU memory, not direct GPU memory, but it increases process memory with session length.

### Trace Retention

The session also retains traces:

- `lychee_fd/app.py:2595` defines `control_prob_trace_records`.
- `lychee_fd/app.py:2597` defines `trace_rounds`.
- `lychee_fd/app.py:2598` defines `trace_state_changes`.
- `lychee_fd/app.py:2599` defines `trace_events`.
- `lychee_fd/app.py:3632` appends every `round_trace`.
- `lychee_fd/app.py:3652` appends state changes.
- `lychee_fd/app.py:3659` appends trace events.

This is useful for debug/save-aligned workflows, but it is another unbounded per-session retention path.

### Session Dictionary Retention

Realtime sessions are stored globally:

- `lychee_fd/app.py:2609` defines `_realtime_sessions`.
- `lychee_fd/app.py:4212` inserts new sessions.
- `lychee_fd/app.py:4374` stop only sets `stop_requested`.

The worker sets `finished=True` and releases `incremental_stream_session` / `tts_bridge`, but there is no obvious automatic `_realtime_sessions.pop(session_id)` cleanup in the inspected route. This is probably intentional for post-session aligned-audio save, but repeated tests can retain completed session metadata.

### Remote Token2Wav State

The remote Token2Wav service keeps per-stream state:

- `lychee_fd/token2wav_server.py:41` defines `_stream_states`.
- `lychee_fd/token2wav_server.py:78` saves stream state.
- `lychee_fd/token2wav_server.py:203` closes a stream by popping it.

The backend does call remote close on abort and worker shutdown:

- `lychee_fd/app.py:971` closes the old stream on event abort.
- `lychee_fd/app.py:1021` closes the current stream in worker `finally`.

So Token2Wav is a secondary suspect only if the frontend disconnects without clean session stop, if the backend worker does not reach `finally`, or if Token2Wav runs on the same GPU and its stream count grows.

## Verification Plan

The fastest way to confirm the main suspect is to add one diagnostic log around each realtime round:

- `len(_audio_input_id_list)`
- `len(_audio_feats_cache)`
- `len(_audio_embed_seq_cache)`
- total elements / estimated bytes in `_audio_embed_seq_cache`
- `torch.cuda.memory_allocated()`
- `torch.cuda.memory_reserved()`
- vLLM active request length and whether request reuse happened

Expected result if the diagnosis is correct:

- `_audio_embed_seq_cache` length grows roughly with audio windows.
- `torch.cuda.memory_reserved()` grows stepwise.
- `nvidia-smi` follows `reserved`, not just `allocated`.

Useful runtime checks:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv -l 5
```

```bash
curl -s http://127.0.0.1:8091/v1/token2wav/health -X POST | python -m json.tool
```

The Token2Wav health output includes `stream_count`; it should return to zero after session stop.

## Ablation Tests

Run these one at a time to isolate the source:

1. Disable vLLM keep-alive:
   - `STEPAUDIO_VLLM_KEEP_ALIVE_LISTENING=0`
   - `STEPAUDIO_VLLM_KEEP_ALIVE_SPEAKING=0`

   If memory still grows at a similar rate, the vLLM KV keep-alive request is not the primary cause.

2. Keep vLLM but disable/reduce incremental audio cache experimentally.

   Expected result: if `_audio_embed_seq_cache` is the source, GPU memory growth should drop sharply.

3. Run Token2Wav on another GPU or disable remote Token2Wav temporarily.

   This separates backend vLLM growth from Token2Wav stream cache growth.

4. Force a short session stop and check whether:
   - backend worker exits;
   - `incremental_stream_session` is released;
   - Token2Wav `stream_count` returns to zero;
   - GPU memory stabilizes after a short allocator delay.

## Fix Directions

Do not blindly delete cached audio state. The stream has coupled text, stoken, control, and audio timelines, and `truncate_active_request_to_prefix(...)` already guards against misaligned audio cache truncation.

Recommended safe direction:

1. Add diagnostic logging first.
2. Add a bounded audio-cache policy:
   - keep only a recent token/window budget; or
   - move old `_audio_embed_seq_cache` tensors to CPU; or
   - periodically call `truncate_active_request_to_prefix(...)` with a token-aligned prefix.
3. Make session cleanup explicit after save-aligned workflows no longer need the old session.
4. Add Token2Wav stream-count checks during stop/abort tests.

The highest-impact code area to inspect next is:

- `lychee_fd/runtime/vllm_generation.py`
  - `_append_incremental_audio(...)`
  - `_cache_precomputed_audio(...)`
  - `_build_cached_audio_context(...)`
  - `truncate_active_request_to_prefix(...)`

## Current Conclusion

The vLLM branch is still growing because the realtime wrapper keeps cumulative audio-side tensors and rebuilds cumulative GPU context each round. This is separate from vLLM's own KV cache reservation. The SoulX path does not appear to be active in the current code; the more relevant issue is that no active periodic truncation/compaction path is currently connected to the realtime worker.
