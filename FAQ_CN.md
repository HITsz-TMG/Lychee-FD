# Lychee-FD 常见问题与排障指南

[English](FAQ.md)

本文面向**源码部署**的 Lychee-FD。请先按照 [README_CN.md 的源码安装说明](README_CN.md#源码安装)完成环境和服务启动。Docker 部署不在本文的排障范围内。

## 1. 应该使用哪个页面？为什么不建议直接使用后端 Gradio 页面？

进行实时交互时，建议使用我们提供的 Lychee-FD 前端；后端 Gradio 页面更适合作为模型加载和底层诊断入口：

```text
http://127.0.0.1:8084
```

后端的 Gradio 页面主要用于模型加载、单次推理和底层诊断。受 Gradio 组件及事件生命周期限制，它不能完整复现 Lychee-FD 的长期实时会话体验，尤其是：

- 持续的麦克风音频分片上传；
- 同一会话中的连续上下文和全双工状态；
- 服务端事件流和增量 PCM 音频播放；
- 用户说话时的打断处理；
- 连续音频块的缓冲、调度和播放衔接。

我们的前端使用 Lychee-FD 的 realtime session API，并针对实时语音额外实现了播放缓冲、平滑调速、音频块衔接、打断处理和交互历史。因此，建议以我们前端的效果判断系统是否正常；Gradio 单次推理页面更适合底层诊断，不能代表完整的实时对话流畅度。

<p align="center">
  <img src="docs/assets/images/FAQ/frontend_UI.png" alt="我们提供的 Lychee-FD 实时交互前端" width="100%">
</p>

基本使用步骤：

1. 在顶部选择模型 preset、`vllm` 和 `stable`。
2. 点击 **Load / Switch**，等待状态变为 `ready`。
3. 选择音色，点击 **Start Call**。
4. 首次使用时允许浏览器访问麦克风。
5. 结束后点击 **End Call**；输入和输出音频会出现在右侧 Interaction History。

`ready` 表示 controller 已经检测到后端服务可访问。是否完整加载了 vLLM、模型权重和 Token2Wav，还应结合第 3 节中的日志标志判断。

## 2. 如何让模型更快开始回答？

打开前端底部的 **Settings**，逐步提高 **Start factor**。

<p align="center">
  <img src="docs/assets/images/FAQ/quick_start_talking.png" alt="调整 Start factor 让模型更倾向提前回答" width="720">
</p>

`Start factor` 会调整 S-S（Start Speaking）控制 token 的倾向。数值越高，模型通常越容易从聆听状态切换到说话状态，因此可能更早开始回答。

建议从默认值 `1.2` 开始，每次只增加 `0.1` 后重新测试。设置过高可能导致模型在用户尚未说完时提前回答或增加误触发。该参数改变的是“何时开始说话”的决策倾向，并不会缩短一次模型 forward 的实际计算时间。

**Playback speed** 只改变浏览器播放速度，也不会降低模型推理延迟。

## 3. 如何确认 Token2Wav、vLLM 和主模型启动成功？

controller 启动的后端日志位于：

```text
runtime_logs/controller/backend_dev_*.log
```

查看最新日志中的关键标志：

```bash
latest_log="$(ls -t runtime_logs/controller/backend_dev_*.log 2>/dev/null | head -n 1)"

grep -nE \
'resolved_vllm=|Remote Token2Wav health|Initializing an LLM engine|load_weights summary|Initialized with vLLM backend|Model loaded in|Running on local URL|ERROR|Traceback' \
"${latest_log}"
```

正常启动应依次出现类似内容：

```text
[backend] resolved_vllm=.../third_party/vllm/vllm/__init__.py
Remote Token2Wav health: {'ok': True, ...}
Initializing an LLM engine (v0.6.5)
Lychee-FD load_weights summary: ... unmatched=0
[VLLMGenerationFramework] Initialized with vLLM backend
Model loaded in ...s
Running on local URL: http://0.0.0.0:7860
```

检查要点：

- `resolved_vllm` 必须指向当前仓库的 `third_party/vllm`。如果指向普通 `site-packages/vllm`，说明没有使用 Lychee-FD 的 patched vLLM 服务实现。
- `Remote Token2Wav health` 应包含 `ok: True`。
- `load_weights summary` 中应为 `unmatched=0`。
- 最后应出现 `Initialized with vLLM backend`、`Model loaded in` 和 `Running on local URL`。

日志中单独出现一次 `Traceback` 并不一定意味着启动失败。某些配置加载失败后会自动 fallback；建议继续查看日志末尾是否出现完整的成功标志，以及进程是否仍然存活。

## 4. 点击 Start Call 后一直停在 Connecting 怎么办？

前端启动通话时会依次创建 realtime session、连接事件流、申请麦克风权限并初始化音频设备。任一步骤失败都会影响连接。

首先检查：

1. 前端顶部模型状态是否为 `ready`。
2. 第 3 节中的 vLLM、模型和 Token2Wav 成功标志是否完整。
3. 浏览器是否允许当前页面使用麦克风。
4. 浏览器是否可以访问后端 `7860` 端口。

在浏览器开发者工具的 **Network** 面板中检查：

- `POST /api/realtime/session/start` 应返回 HTTP 200 和 `session_id`。
- `GET /api/realtime/session/<session_id>/events` 是持续的 SSE 连接，保持 pending 属于正常现象。
- `session/start` 返回 503 通常表示主模型或 Token2Wav 尚未就绪。
- 请求一直 pending 时，应检查后端日志是否仍在推理或发生阻塞。
- `Failed to fetch` 通常表示端口不可达、代理、CORS 或 HTTPS 页面访问 HTTP API 被浏览器拦截。

远程服务器建议同时转发前端和后端端口：

```bash
ssh -L 8084:127.0.0.1:8084 \
    -L 7860:127.0.0.1:7860 \
    user@server
```

然后访问：

```text
http://127.0.0.1:8084
```

浏览器通常只允许在 HTTPS 或 localhost 等安全上下文中使用麦克风。直接打开远程服务器的普通 HTTP 地址时，麦克风权限可能被阻止。

## 5. 如何查看 RTF？什么结果表示能够实时运行？

打开主前端右侧的 **Debug** 抽屉，可以看到当前会话的 Round RTF：

```text
Round RTF
Last / Mean / Max
RTF ≤ 1
Mean Total Round
```

<p align="center">
  <img src="docs/assets/images/FAQ/debug_info.png" alt="Lychee-FD Debug 抽屉中的实时性能信息" width="520">
</p>

Round RTF 的定义为：

```text
RTF = total_round_ms / actual_input_duration_ms
```

- `RTF ≤ 1` 表示该轮计算速度能够跟上输入音频。
- Debug 中的 `RTF ≤ 1` 以“通过轮次 / 总轮次（占比）”显示。
- 输入窗口为 400 ms 时，`RTF ≤ 1` 等价于 `total_round ≤ 400 ms`。
- 最后一轮可能不足 400 ms，因此实现使用每轮实际输入时长作为分母。

实际 RTF 会随 GPU、CPU、驱动、模型配置和系统负载等硬件与运行条件变化。经过多轮对话验证后，如果 `RTF ≤ 1` 的轮次占比超过 90%，通常就能够获得正常、连贯的实时交互体验；占比越高，系统的实时余量越充足。

固定音频测试页面位于：

```text
http://127.0.0.1:8084/weight-test.html
```

其中显示的 **Token2Wav RTF** 只衡量声码器合成速度，不等同于完整 Round RTF。判断主链路是否能够实时运行时，建议优先查看 Debug 抽屉中的 Round RTF 和 `RTF ≤ 1` 占比。

## 6. 为什么音频仍然断续或回答很慢？

即使 GPU 型号较高，以下问题仍可能导致卡顿：

- 实际导入的是原始 `site-packages/vllm`，而不是 `third_party/vllm`。
- 后端误用了 `hf`，而不是低延迟在线路径所需的 `vllm`。
- 主后端 Round RTF 大于 1，输入队列持续积压。
- Token2Wav 合成或远程调用跟不上音频播放速度。
- 主模型和 Token2Wav 共用一张显存紧张的 GPU。
- 浏览器播放队列、网络或代理引入额外抖动。

排查时依次检查：

1. Debug 中 `RTF ≤ 1` 的占比是否超过 90%。
2. `Queue Before` 和 `Consumed / Remaining` 是否持续增长。
3. `Playback Backlog` 是否持续增长。
4. `/weight-test.html` 中 Token2Wav RTF 是否频繁大于 1。
5. 日志中的 `resolved_vllm` 和 backend 类型是否正确。

建议每次只调整一个推理参数并记录结果，这样更容易判断具体是哪项变化产生了影响。

## 7. CUDA OOM 或显存不足怎么处理？

如果有两张 GPU，建议让 Token2Wav 和主后端使用不同 GPU：

```bash
# Token2Wav 进程
CUDA_VISIBLE_DEVICES=0 \
LYCHEEFD_T2W_MODEL_PATH=/path/to/token2wav \
./scripts/start_token2wav_server.sh

# 前端 controller 和主后端进程
CUDA_VISIBLE_DEVICES=1 \
./scripts/start_frontend_dev.sh prod public
```

如果只能共用一张 GPU，可以尝试降低 vLLM KV cache 预算和最大上下文长度：

```bash
export LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION=0.70
export LYCHEEFD_VLLM_MAX_MODEL_LEN=8192
```

降低这些值会减少显存占用，但也会减少 KV cache 容量。当前公开在线 pipeline 主要采用单卡主后端加独立 Token2Wav 的部署方式；仅设置多个可见 GPU 并不会自动启用模型张量并行。

## 8. 提交 Issue 时的信息整理建议

以下清单仅用于帮助整理问题描述，可根据实际情况选取相关内容：

- 当前 Git commit：`git rev-parse HEAD`；
- 操作系统、CPU 架构、GPU、驱动及 CUDA 信息；
- Python、PyTorch、vLLM 和 Transformers 版本；
- 实际导入的 `vllm.__file__`；
- 相关启动命令及关键环境变量；
- 最新 `runtime_logs/controller/backend_dev_*.log` 的启动部分和错误末尾；
- 问题发生步骤、预期结果和实际结果；
- Connecting 问题对应的浏览器 Network/Console 截图；
- 性能问题对应的 Round RTF、`RTF ≤ 1` 占比、GPU 型号和推理窗口。

可以使用下面的命令收集 Python 组件信息：

```bash
python - <<'PY'
import platform
import torch
import transformers
import vllm

print("platform:", platform.platform())
print("python:", platform.python_version())
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("vllm:", getattr(vllm, "__version__", "unknown"))
print("vllm file:", vllm.__file__)
PY

nvidia-smi
```
