# Lychee-FD

Full-duplex realtime speech interaction demo.

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow?logo=huggingface)](https://huggingface.co/PLACEHOLDER/Lychee-FD)
[![Demo](https://img.shields.io/badge/Demo-Website-blue?logo=googlechrome)](https://PLACEHOLDER.demo)
[![Paper](https://img.shields.io/badge/Paper-ACL%202026-red?logo=adobeacrobatreader)](https://aclanthology.org/2026.acl-long.419.pdf)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)

This repository provides the Lychee-FD source code, frontend demo, and Docker Compose configuration. The Docker image contains the runtime environment and code, but does not include model weights.

## System Architecture

<p align="center">
  <img src="docs/assets/vllm_optimized_online_pipeline_detail.svg" alt="vLLM-optimized online full-duplex inference pipeline" width="100%">
</p>

The online pipeline separates realtime audio ingestion, state-aware full-duplex inference, vLLM-optimized response generation, and streaming token-to-waveform synthesis.

## Online Serving Performance

We compare the Hugging Face online backend with the vLLM online backend on an evaluation subset. Each online round consumes a fixed 400 ms audio window, so the key metric is whether backend computation finishes before the next window arrives. The main gain is in response-generation rounds: vLLM substantially improves the probability that speaking rounds finish within the realtime budget.

| Scope | HF rounds | vLLM rounds | HF mean compute / window | vLLM mean compute / window | Speedup | HF window RTF | vLLM window RTF | HF within window | vLLM within window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All rounds | 15,191 | 15,191 | 288.49 ms | 249.30 ms | 1.16x | 0.721 | 0.623 | 70.8% | 99.5% |
| Speaking rounds | 4,470 | 4,183 | 777.50 ms | 261.50 ms | 2.97x | 1.944 | 0.654 | 0.6% | 98.5% |
| Listening rounds | 10,701 | 11,003 | 84.70 ms | 244.77 ms | 0.35x | 0.212 | 0.612 | 100.0% | 99.9% |

The speaking-round result is the most relevant realtime serving signal: it includes response generation work, where the HF backend frequently exceeds the online window, while the vLLM backend keeps 98.5% of speaking rounds within the realtime budget.

Listening rounds are input-gated: once the backend reliably consumes each 400 ms audio window before the next window arrives, additional compute-time reduction does not directly reduce user-perceived latency. Both backends satisfy this condition for listening rounds, so they are effectively equivalent in the listening state despite different raw compute times.

`window RTF = round_compute_ms / 400 ms`; values below 1.0 indicate that the backend finishes processing the current audio window before the next window arrives.

## Docker Quick Start

Clone the repository:

```bash
git clone https://github.com/HITsz-TMG/Lychee-FD.git
cd Lychee-FD
```

Create a local environment file:

```bash
cp .env.docker.example .env
```

Edit `.env` and set the model paths:

```dotenv
LYCHEE_FD_IMAGE=ghcr.io/idealistxy/lychee-fd:latest

HOST_MODEL_ROOT=/path/to/model-root
STEPAUDIO_MODEL_PATH=/models/path/to/your-lychee-fd-checkpoint
STEPAUDIO_T2W_MODEL_PATH=/models/token2wav
```

`HOST_MODEL_ROOT` is the model directory on your host machine. It is mounted into the container as `/models`.

Pull the prebuilt image and start the demo:

```bash
docker compose pull
docker compose up
```

Open:

```text
http://127.0.0.1:8084
```

For a remote server, replace `127.0.0.1` with the server IP.

## Model Presets

The frontend model list is loaded from:

```text
model_presets_dev.json
```

Update the preset path to the container-side model path:

```json
{
  "name": "my-lychee-fd-checkpoint",
  "model_path": "/models/path/to/your-lychee-fd-checkpoint",
  "backend_type": "vllm",
  "mode": "stable"
}
```

After editing presets:

```bash
docker compose restart frontend
```

## GPU Settings

By default, token2wav and the main backend use separate GPUs:

```dotenv
TOKEN2WAV_CUDA_VISIBLE_DEVICES=0
BACKEND_CUDA_VISIBLE_DEVICES=1
```

For a single-GPU machine:

```dotenv
TOKEN2WAV_CUDA_VISIBLE_DEVICES=0
BACKEND_CUDA_VISIBLE_DEVICES=0
```

Check Docker GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Common Commands

Run in background:

```bash
docker compose up -d
```

View logs:

```bash
docker compose logs -f
```

Stop:

```bash
docker compose down
```

Pull the latest image:

```bash
docker compose pull
docker compose up -d
```

## Third-Party Code

This repository vendors selected third-party components for the demo and online
serving pipeline. See [third_party/THIRD_PARTY_NOTICES.md](third_party/THIRD_PARTY_NOTICES.md)
for upstream sources, license notices, and local integration notes.

## License

Lychee-FD is released under the [Apache License 2.0](LICENSE).

Copyright 2026 HITsz-TMG and Lychee-FD authors.

## Detailed Guide

See [启动指南.md](启动指南.md) for more startup details and optional source-based launch commands.
