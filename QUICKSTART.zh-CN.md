# Lychee-FD 启动指南

本项目提供预构建 Docker 镜像，推荐直接拉取镜像启动。镜像包含运行环境和代码，但不包含模型权重。

预构建镜像：

```text
ghcr.io/idealistxy/lychee-fd:latest
```

## 准备模型

启动前需要在本机准备模型目录，至少包含：

```text
model-root/
  lychee_full_duplex_v1.5/
  token2wav/
```

其中：

- `lychee_full_duplex_v1.5/` 是 [HIT-TMG/Lychee-FD](https://huggingface.co/HIT-TMG/Lychee-FD) 提供的 Lychee-FD full-duplex checkpoint。
- `token2wav/` 是 Step-Audio-2-mini 的 token2wav 权重目录。

下载 Lychee-FD checkpoint：

```bash
huggingface-cli download HIT-TMG/Lychee-FD \
  --include "lychee_full_duplex_v1.5/*" \
  --local-dir /path/to/model-root
```

下载 Token2Wav：

```bash
huggingface-cli download stepfun-ai/Step-Audio-2-mini \
  --include "token2wav/*" \
  --local-dir /path/to/model-root
```

容器内统一把模型根目录挂载为：

```text
/models
```

例如宿主机模型目录是：

```text
/home/user/models
```

容器内对应是：

```text
/models
```

## Docker 快速启动

克隆代码：

```bash
git clone https://github.com/HITsz-TMG/Lychee-FD.git
cd Lychee-FD
```

创建配置文件：

```bash
cp .env.docker.example .env
```

编辑 `.env`，至少修改这几项：

```dotenv
LYCHEE_FD_IMAGE=ghcr.io/idealistxy/lychee-fd:latest

HOST_MODEL_ROOT=/path/to/model-root
LYCHEEFD_MODEL_PATH=/models/lychee_full_duplex_v1.5
LYCHEEFD_T2W_MODEL_PATH=/models/token2wav
```

说明：

- `HOST_MODEL_ROOT` 是宿主机真实模型根目录。
- `LYCHEEFD_MODEL_PATH` 是容器内的 Lychee-FD checkpoint 路径。
- `LYCHEEFD_T2W_MODEL_PATH` 是容器内的 token2wav 路径。

拉取镜像：

```bash
docker compose pull
```

启动服务：

```bash
docker compose up
```

打开页面：

```text
http://127.0.0.1:8084
```

如果在远程服务器上运行，请打开 `http://<server-ip>:8084`。前端页面会自动请求
`http://<server-ip>:7860` 上的后端 API，所以浏览器必须同时能访问 `8084` 和
`7860` 两个端口。如果通过 SSH 端口转发访问，需要同时转发两个端口：

```bash
ssh -L 8084:127.0.0.1:8084 -L 7860:127.0.0.1:7860 user@server
```

## GPU 设置

默认配置使用：

```dotenv
TOKEN2WAV_CUDA_VISIBLE_DEVICES=0
BACKEND_CUDA_VISIBLE_DEVICES=1
```

如果机器只有一张 GPU，可以改成：

```dotenv
TOKEN2WAV_CUDA_VISIBLE_DEVICES=0
BACKEND_CUDA_VISIBLE_DEVICES=0
```

如果出现 CUDA OOM，尤其是 token2wav 和主后端共用一张 GPU 时，可以降低 vLLM
预分配给 KV cache 的显存占比：

```dotenv
LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION=0.70
```

默认值是 `0.90`。调低后会给 token2wav、CUDA kernel 和临时 activation 留出更多显存，但会降低 vLLM 可用 KV cache 容量。显存仍然紧张时，也可以把
`LYCHEEFD_VLLM_MAX_MODEL_LEN` 从 `16384` 降到 `8192`。

启动前请确认 Docker 能访问 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 模型 Preset

前端模型下拉列表来自：

```text
model_presets_dev.json
```

请把里面的示例路径替换成容器内路径，例如：

```json
{
  "name": "lychee_full_duplex_v1.5",
  "model_path": "/models/lychee_full_duplex_v1.5",
  "backend_type": "vllm",
  "mode": "stable"
}
```

如果修改了 `model_presets_dev.json`，重启 `frontend` 服务：

```bash
docker compose restart frontend
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

查看服务状态：

```bash
docker compose ps
```

停止服务：

```bash
docker compose down
```

重新拉取最新镜像：

```bash
docker compose pull
docker compose up -d
```

## 可选：源码方式启动

如果不使用 Docker，也可以手动启动两个服务。

先按仓库提供的环境文件创建后端 conda 环境：

```bash
conda env create -f environment.yml
conda activate lychee-fd
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation "flash-attn==2.8.2"
```

`requirements.txt` 不直接安装 `flash-attn`，因为它通常需要匹配本机 CUDA/PyTorch 版本并使用 `--no-build-isolation`。如果源码编译太慢，可以改用与机器环境匹配的预编译 wheel。

启动 token2wav：

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
LYCHEEFD_REALTIME_DEBUG=0 \
VUE_APP_REALTIME_DEBUG=0 \
./scripts/start_frontend_dev.sh prod public
```

源码方式使用 vLLM 后端时，需要同时满足两点：conda 环境中已安装兼容的 vLLM wheel，用于提供 CUDA native libraries；`LYCHEEFD_VLLM_SOURCE_DIR` 指向本仓库的 `third_party/vllm` patched source tree。启动脚本会把 patched source tree 放到 `PYTHONPATH` 前面，并把已安装 vLLM wheel 中的 native artifacts 同步到该源码树。
