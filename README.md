# Lychee-FD

<p align="center">
  <img src="docs/assets/images/logo/Lychee-FD-logo.png" width="260"/>
</p>

<h4 align="center">
Lychee-FD is a full-duplex realtime speech interaction system designed for low-latency spoken dialogue.
</h4>

<p align="center">
  <a href="https://huggingface.co/PLACEHOLDER/Lychee-FD">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow" alt="Hugging Face">
  </a>
  <a href="https://hitsz-tmg.github.io/Lychee-FD/">
    <img src="https://img.shields.io/badge/Demo-Website-blue?logo=googlechrome" alt="Demo">
  </a>
  <a href="https://arxiv.org/pdf/2607.06540">
    <img src="https://img.shields.io/badge/arXiv-2607.06540-b31b1b?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
</p>

<h4 align="center">
If you appreciate our project, please consider giving us a star ⭐ on GitHub to stay updated with the latest developments.
</h4>

## 🔥 News

- [2026/07/10] 🎉 We release the Lychee-FD codebase, paper, and web demo.
- [2026/07/07] 🏆 Our paper has been selected as an [**Outstanding Paper**](https://aclanthology.org/2026.acl-long.419) at **ACL 2026**!


## 📌 Introduction

**Lychee-FD** is a full-duplex realtime speech interaction system designed for low-latency spoken dialogue. Unlike conventional turn-based speech systems, Lychee-FD supports simultaneous listening and speaking, enabling more natural interactive behaviors such as realtime response generation, interruption handling, and streaming speech output.

This repository provides:

- Source code for the Lychee-FD online serving pipeline.
- A browser-based realtime speech interaction frontend.
- Docker Compose configuration for quick deployment.
- vLLM-optimized backend support for low-latency online inference.
- Runtime integration notes and third-party license notices.

## 📀 Demo Video

[**todo**] 这里放web demo + 数字人 + 机器人 各10秒演示的视频

## 🌟 Model Structure

[**todo**] 按照推文的来改:

这种设计与前面的科学洞察一一对应：既然冲突主要发生在深层，就不再强迫语义和声学在深层共享同一套参数；既然文本语义容易被高频语音信号稀释，就引入密集语义对齐通道，让模型在生成语音的同时保留清晰、连续的“内部语义线索”。

因此，Lychee-FD 的全双工能力不是外挂的打断模块，也不是级联系统里的流程调度，而是被内化到模型架构中的原生交互能力。
它让模型能够在连续语音流中协同处理语义理解、语音生成和节奏控制，从而在保持推理效率的同时，兼顾语音智能与交互流畅度。
实验结果也验证了这一点：Lychee-FD 在 Spoken QA 任务上平均提升 7.4%，在 FullDuplexBench 1.5 上提升 28.5%，在多个全双工语音交互基准上达到当前领先水平。

3. 工程实现：从论文模型到可在线交互系统
真正的全双工交互，最终必须落到实时系统里。
Lychee-FD 的架构同时生成语义、声学和控制信号，这对推理引擎提出了新的挑战。传统大语言模型推理框架通常面向单一路径、单一输出流设计，如果直接套用到多通道全双工模型上，多个专门通道会被顺序执行，带来额外延迟，影响实时对话的流畅度。
为此，团队围绕 Lychee-FD 的架构特点，开发了实时并行多流 vLLM 推理框架：在共享主干完成计算后，将中间表示分发到语义、声学和控制通道，让多个通道并行执行、独立管理KV cache，从而显著减少多流生成带来的推理延迟。
同时，团队进一步提出控制头早退策略。由于打断、停说、转入倾听等行为首先依赖控制信号，系统不必等待完整语音和文本生成结束，而是让控制 Token 更早产出，为打断响应提供一条“快速通道”。
并行多流推理解决了“跑得慢”的问题，控制头早退解决了“反应慢”的问题。 两者共同把 Lychee-FD 从论文中的模型框架，推进到可以真实交互的数字人与机器人系统。

<p align="center">
  <img src="docs/assets/vllm_optimized_online_pipeline_detail.svg" alt="vLLM-optimized online full-duplex inference pipeline" width="100%">
</p>

The online pipeline separates realtime audio ingestion, state-aware full-duplex inference, vLLM-optimized response generation, and streaming token-to-waveform synthesis.

## Online Serving 


Performance

We compare the Hugging Face online backend with the vLLM online backend on an evaluation subset. Each online round consumes a fixed 400 ms audio window, so the key metric is whether backend computation finishes before the next window arrives.

| Scope | HF rounds | vLLM rounds | HF mean compute / window | vLLM mean compute / window | Speedup | HF window RTF | vLLM window RTF | HF within window | vLLM within window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All rounds | 15,191 | 15,191 | 288.49 ms | 249.30 ms | 1.16x | 0.721 | 0.623 | 70.8% | **99.5%** |
| Speaking rounds | 4,470 | 4,183 | 777.50 ms | **261.50 ms** | **2.97x** | 1.944 | **0.654** | 0.6% | **98.5%** |
| Listening rounds | 10,701 | 11,003 | 84.70 ms | 244.77 ms | 0.35x | 0.212 | 0.612 | **100.0%** | **99.9%** |

**Key points:**

- **Speaking rounds are the critical online workload:** vLLM reduces mean compute from 777.50 ms to 261.50 ms and raises within-window completion from 0.6% to 98.5%.
- **Overall realtime stability improves:** across all rounds, vLLM finishes 99.5% of windows within the 400 ms budget.
- **Listening-round raw speed is not the bottleneck:** HF is faster in listening-only compute, but this does not reduce user latency because the system must wait for the next 400 ms input window; both backends already finish listening rounds in time.

`window RTF = round_compute_ms / 400 ms`; values below 1.0 indicate that the backend finishes processing the current audio window before the next window arrives.

## 🚀 Docker Quick Start

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
