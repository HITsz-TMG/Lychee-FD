# Lychee-FD

Full-duplex realtime speech interaction demo.

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow?logo=huggingface)](https://huggingface.co/PLACEHOLDER/Lychee-FD)
[![Demo](https://img.shields.io/badge/Demo-Website-blue?logo=googlechrome)](https://PLACEHOLDER.demo)
[![Paper](https://img.shields.io/badge/Paper-ACL%202026-red?logo=adobeacrobatreader)](https://aclanthology.org/2026.acl-long.419.pdf)

This repository provides the Lychee-FD source code, frontend demo, and Docker Compose configuration. The Docker image contains the runtime environment and code, but does not include model weights.

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

## Detailed Guide

See [启动指南.md](启动指南.md) for more startup details and optional source-based launch commands.
