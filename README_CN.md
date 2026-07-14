# Lychee-FD

<p align="center">
  <img src="docs/assets/images/logo/Lychee-FD-logo.png" width="260"/>
</p>

<h4 align="center">
Lychee-FD 是一个面向低延迟语音对话的全双工实时语音交互系统。
</h4>

<p align="center">
  <a href="https://huggingface.co/HIT-TMG/Lychee-FD">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow" alt="Hugging Face">
  </a>
  <a href="https://hitsz-tmg.github.io/Lychee-FD/lychee-fd-showcase/">
    <img src="https://img.shields.io/badge/Demo-Website-blue?logo=googlechrome" alt="Demo">
  </a>
  <a href="https://arxiv.org/pdf/2607.06540">
    <img src="https://img.shields.io/badge/arXiv-2607.06540-b31b1b?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
</p>

<h4 align="center">
如果这个项目对你有帮助，欢迎在 GitHub 上给我们一个 star，以便获取后续更新。
</h4>

<p align="center">
  <a href="README.md">English README</a>
</p>

## 🔥 最新动态

- [2026/07/10] 🎉 我们发布 Lychee-FD 代码、论文和在线演示网站。
- [2026/07/07] 🏆 我们的论文入选 **ACL 2026** [**Outstanding Paper**](https://2026.aclweb.org/program/best_papers/#outstanding-papers)。

## 📌 简介

**Lychee-FD** 是一个面向低延迟语音对话的全双工实时语音交互系统。不同于传统的轮次式语音系统，Lychee-FD 支持同时听和说，能够实现实时回复生成、用户打断处理和流式语音输出等更自然的交互行为。

本仓库包含：

- Lychee-FD 在线服务 pipeline 的源代码。
- 浏览器端实时语音交互前端。
- 用于快速部署的 Docker Compose 配置。
- 面向低延迟在线推理的 vLLM 优化后端。
- 运行时集成说明和第三方许可证说明。

## 🌟 模型结构

Lychee-FD 是一个原生端到端全双工语音语言模型，面向实时语音交互设计。它不依赖级联的 ASR、LLM、TTS 和轮次判定模块，而是在端到端多流架构中联合建模听、理解、说和交互控制。

这一架构来自我们对原生全双工语音建模优化动态的观察。在更深层中，声学生成和语义推理会对共享参数施加逐渐分化的优化目标。同时，高频语音 token 也可能稀释稀疏文本监督，从而削弱语音生成过程中的语义一致性。

<table>
  <tr>
    <td width="50%" align="center">
      <img src="docs/assets/images/paper/acoustic_semantic_optimization.png" alt="Lychee-FD acoustic-semantic optimization dynamics" width="100%">
    </td>
    <td width="50%" align="center">
      <img src="docs/assets/images/paper/modality_layer_conflict.png" alt="Lychee-FD layer-wise acoustic-semantic conflict" width="100%">
    </td>
  </tr>
</table>

Lychee-FD 通过层次化声学-语义建模解决这一冲突，而不是依赖外部调度。低层共享以从连续音频流中学习通用语音-语言表示；高层则解耦为语义、声学和对话控制三条流。这样的设计使语义推理能够保持语言理解和知识能力，声学建模专注于自然语音 token 生成，对话控制则负责决定何时说话、停止、聆听或响应用户打断。

<p align="center">
  <img src="docs/assets/images/architecture/lychee_fd_architecture.png" alt="Lychee-FD hierarchical acoustic-semantic model architecture" width="100%">
</p>

## ⚙️ 工程实现

真正的全双工交互必须以在线系统形式运行。因此，Lychee-FD 针对层次化多通道架构定制了 vLLM：在共享 backbone 计算后，后端会将中间状态分发到语义、声学和对话控制通道，同时维护多流解码所需的生成状态和 KV cache。

这一设计避免将所有专门通道强行压到单一串行推理路径。控制头还使用 early-exit 路径，使打断、停止说话、聆听/回复等决策可以在完整语音生成完成前产生。在我们的在线评估中，这一 vLLM 优化的多流服务 pipeline 在说话轮次中取得约 **2.96x** 加速，并在长会话测试中将增量显存增长降低约 **23%**。

<p align="center">
  <img src="docs/assets/images/architecture/online_multistream_inference.png" alt="Lychee-FD online multi-stream inference framework" width="100%">
</p>

## 模型权重

Docker 镜像包含运行环境和 demo 代码，但不包含模型权重。启动 demo 前需要先下载所需 checkpoint。

| 组件 | 来源 | 模型根目录下的预期目录 |
| --- | --- | --- |
| Lychee-FD full-duplex model | [HIT-TMG/Lychee-FD](https://huggingface.co/HIT-TMG/Lychee-FD)，目录 `lychee_full_duplex/` | `lychee_full_duplex/` |
| Token2Wav vocoder | [stepfun-ai/Step-Audio-2-mini](https://huggingface.co/stepfun-ai/Step-Audio-2-mini)，目录 `token2wav/` | `token2wav/` |

创建一个本地模型根目录：

```text
/path/to/model-root/
  lychee_full_duplex/
  token2wav/
```

下载 Lychee-FD checkpoint：

```bash
huggingface-cli download HIT-TMG/Lychee-FD \
  --include "lychee_full_duplex/*" \
  --local-dir /path/to/model-root
```

下载 Step-Audio-2-mini 的 Token2Wav：

```bash
huggingface-cli download stepfun-ai/Step-Audio-2-mini \
  --include "token2wav/*" \
  --local-dir /path/to/model-root
```

## 🚀 Docker 快速启动

克隆仓库：

```bash
git clone https://github.com/HITsz-TMG/Lychee-FD.git
cd Lychee-FD
```

创建本地环境配置：

```bash
cp .env.docker.example .env
```

编辑 `.env`，设置模型路径：

```dotenv
LYCHEE_FD_IMAGE=ghcr.io/hitsz-tmg/lychee-fd:latest

HOST_MODEL_ROOT=/path/to/model-root
LYCHEEFD_MODEL_PATH=/models/lychee_full_duplex
LYCHEEFD_T2W_MODEL_PATH=/models/token2wav
```

`HOST_MODEL_ROOT` 是宿主机上的模型目录，它会被挂载到容器内的 `/models`。

拉取预构建镜像并启动 demo：

```bash
docker compose pull
docker compose up
```

打开页面：

```text
http://127.0.0.1:8084
```

如果部署在远程服务器上，请打开 `http://<server-ip>:8084`。浏览器前端会连接 `http://<server-ip>:7860` 上的后端 API，因此浏览器必须能访问 `8084` 和 `7860` 两个端口。如果通过 SSH 端口转发访问，需要同时转发两个端口，例如：

```bash
ssh -L 8084:127.0.0.1:8084 -L 7860:127.0.0.1:7860 user@server
```

## 模型 Preset

前端模型列表来自：

```text
model_presets_dev.json
```

请把 preset 路径更新为容器内模型路径：

```json
{
  "name": "lychee_full_duplex",
  "model_path": "/models/lychee_full_duplex",
  "backend_type": "vllm",
  "mode": "stable"
}
```

修改 preset 后重启前端服务：

```bash
docker compose restart frontend
```

## GPU 设置

默认情况下，Token2Wav 和主后端使用不同 GPU：

```dotenv
TOKEN2WAV_CUDA_VISIBLE_DEVICES=0
BACKEND_CUDA_VISIBLE_DEVICES=1
```

如果机器只有一张 GPU：

```dotenv
TOKEN2WAV_CUDA_VISIBLE_DEVICES=0
BACKEND_CUDA_VISIBLE_DEVICES=0
```

如果出现 CUDA OOM，尤其是 Token2Wav 和主后端共用一张 GPU 时，可以降低 vLLM KV cache 的显存预算：

```dotenv
LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION=0.70
```

默认值是 `0.90`。调低后会给 Token2Wav、CUDA kernels 和临时 activations 留出更多显存，但会减少 vLLM 可用 KV cache 容量。显存受限时，也可以将 `LYCHEEFD_VLLM_MAX_MODEL_LEN` 从 `16384` 降到 `8192`。

检查 Docker GPU 访问：

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 常用命令

后台启动：

```bash
docker compose up -d
```

查看日志：

```bash
docker compose logs -f
```

停止服务：

```bash
docker compose down
```

拉取最新镜像：

```bash
docker compose pull
docker compose up -d
```

## 源码安装

Docker 是推荐且可复现的部署方式。源码安装主要用于开发或调试，因为 vLLM 和 FlashAttention wheel 必须匹配本机 CUDA/PyTorch 环境。

使用仓库提供的环境文件创建后端 Python 环境：

```bash
conda env create -f environment.yml
conda activate lychee-fd
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation "flash-attn==2.8.2"
```

这些环境文件用于复现发布 Docker 镜像中的后端栈，包括 Python 3.10、PyTorch 2.5.1、CUDA 12.x runtime packages、vLLM 0.6.5 和其他服务依赖。`flash-attn` 单独安装，因为它对本机 CUDA/PyTorch 构建较敏感；如果源码编译太慢或不可用，可以安装与机器环境匹配的预编译 wheel。

在线 vLLM 后端需要同时满足两点：

- conda 环境中已安装 vLLM wheel，用于提供 `_C.abi3.so`、`_moe_C.abi3.so` 和 `vllm_flash_attn_c.abi3.so` 等 native libraries；
- `third_party/vllm` 中的 patched source tree，用于实现 Lychee-FD 多流服务路径。

源码启动前，请把脚本指向 conda 环境和 patched vLLM source tree：

```bash
export LYCHEEFD_CONDA_ENV_PATH="${CONDA_PREFIX}"
export STEPAUDIO2_SOURCE_DIR="${PWD}/third_party/Step-Audio2"
export LYCHEEFD_VLLM_SOURCE_DIR="${PWD}/third_party/vllm"
export LYCHEEFD_VLLM_SYNC_FLASH_ATTN=1
export LYCHEEFD_VLLM_FORCE_SYNC_FLASH_ATTN=1
```

`scripts/start_backend.sh` 会把 `LYCHEEFD_VLLM_SOURCE_DIR` 放到 `PYTHONPATH` 最前面，并将已安装 vLLM/FlashAttention wheel 中的 native artifacts 同步到 patched source tree。这一步是必需的；直接导入 site-packages 中的原始 vLLM 不会使用 Lychee-FD 的服务实现。

首次启动前安装前端依赖：

```bash
cd frontend
npm ci
cd ..
```

启动 Token2Wav sidecar：

```bash
CUDA_VISIBLE_DEVICES=0 \
LYCHEEFD_T2W_MODEL_PATH=/path/to/token2wav \
./scripts/start_token2wav_server.sh
```

启动前端和实时后端 controller：

```bash
CUDA_VISIBLE_DEVICES=1 \
LYCHEEFD_CONDA_ENV_PATH=${CONDA_PREFIX} \
STEPAUDIO2_SOURCE_DIR=${PWD}/third_party/Step-Audio2 \
LYCHEEFD_VLLM_SOURCE_DIR=${PWD}/third_party/vllm \
LYCHEEFD_VLLM_SYNC_FLASH_ATTN=1 \
LYCHEEFD_VLLM_FORCE_SYNC_FLASH_ATTN=1 \
ALLOWED_MODEL_ROOT=/path/to/model/root \
AUTO_LOAD_DEFAULT=0 \
LYCHEEFD_REALTIME_STRICT_INFER_WINDOW=1 \
LYCHEEFD_STOKEN_DELAY_NUM=10 \
LYCHEEFD_TTS_VOCODER_HOP_SIZE=10 \
LYCHEEFD_T2W_STREAM_LOOKAHEAD_LEN=3 \
LYCHEEFD_T2W_REMOTE_ENABLED=1 \
LYCHEEFD_T2W_REMOTE_URL=http://127.0.0.1:8091 \
LYCHEEFD_T2W_REMOTE_FALLBACK=0 \
LYCHEEFD_USE_VLLM=1 \
LYCHEEFD_VLLM_MAX_MODEL_LEN=16384 \
LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION=0.90 \
./scripts/start_frontend_dev.sh prod public
```

两个服务都启动后，打开：

```text
http://127.0.0.1:8084
```

## 第三方代码

本仓库为 demo 和在线服务 pipeline vendored 了部分第三方组件。上游来源、许可证声明和本地集成说明见 [third_party/THIRD_PARTY_NOTICES.md](third_party/THIRD_PARTY_NOTICES.md)。

## 许可证

Lychee-FD 基于 [Apache License 2.0](LICENSE) 发布。

Copyright 2026 HITsz-TMG and Lychee-FD authors.
