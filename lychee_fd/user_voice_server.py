import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ASR_MODEL_DIR = str(PROJECT_DIR / "models" / "paraformer-zh")
ASR_MODEL_DIR = Path(os.environ.get("USER_VOICE_ASR_MODEL_DIR", DEFAULT_ASR_MODEL_DIR)).resolve()
CLONE_PROMPT_DIR = Path(
    os.environ.get(
        "LYCHEEFD_CLONE_PROMPT_DIR",
        str(PROJECT_DIR / "frontend" / "public" / "clone_24k_mono"),
    )
).resolve()
TMP_DIR = Path(os.environ.get("USER_VOICE_TMP_DIR", "/tmp/lychee_user_voice_uploads")).resolve()
USER_VOICE_WAV = os.environ.get("USER_VOICE_WAV_NAME", "user_voice.wav")
USER_VOICE_TEXT = os.environ.get("USER_VOICE_TEXT_NAME", "user_voice.txt")
MAX_UPLOAD_BYTES = int(os.environ.get("USER_VOICE_MAX_UPLOAD_BYTES", str(80 * 1024 * 1024)))

TARGET_SAMPLE_RATE = 24000
ASR_SAMPLE_RATE = 16000

TMP_DIR.mkdir(parents=True, exist_ok=True)
CLONE_PROMPT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="lychee user voice upload", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_asr_model = None


def _html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>录制用户音色</title>
  <style>
    :root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f4f7fb; color: #111827; }
    main { width: min(820px, calc(100vw - 32px)); margin: 42px auto; }
    h1 { margin: 0 0 8px; font-size: 28px; }
    p { color: #475569; line-height: 1.6; }
    .panel { background: #fff; border: 1px solid #d8e0ea; border-radius: 10px; padding: 20px; box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08); }
    .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin: 14px 0; }
    button { border: 0; border-radius: 8px; padding: 10px 16px; font-weight: 700; cursor: pointer; background: #0284c7; color: white; }
    button.secondary { background: #e2e8f0; color: #0f172a; }
    button:disabled { cursor: not-allowed; opacity: 0.55; }
    textarea { width: 100%; min-height: 150px; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; font-size: 15px; line-height: 1.6; resize: vertical; }
    audio { width: 100%; margin-top: 10px; }
    .status { margin-top: 12px; padding: 10px 12px; border-radius: 8px; background: #f1f5f9; color: #334155; white-space: pre-wrap; }
    .status.error { background: #fef2f2; color: #991b1b; }
    .status.ok { background: #ecfdf5; color: #166534; }
    .meta { font-size: 13px; color: #64748b; }
  </style>
</head>
<body>
  <main>
    <h1>录制用户音色</h1>
    <p>只保留一个用户音色。每次录音并提交都会覆盖上一次的音频和文本，主页面参考音色列表会显示“用户音色”。</p>
    <section class="panel">
      <div class="row">
        <button id="recordStart">开始录音</button>
        <button id="recordStop" class="secondary" disabled>停止并转写</button>
        <button id="reload" class="secondary">读取当前已保存音色</button>
      </div>
      <p class="meta">请先允许麦克风权限，对着麦克风说话，停止后会自动转写。</p>
      <div id="audioWrap"></div>
      <label class="meta" for="text">转写文本，可编辑修正后提交</label>
      <textarea id="text" placeholder="录音后这里会出现 paraformer 转写结果"></textarea>
      <div class="row">
        <button id="commit" disabled>提交为用户音色</button>
      </div>
      <div id="status" class="status">等待录音。</div>
    </section>
  </main>
  <script>
    const recordStartBtn = document.getElementById('recordStart');
    const recordStopBtn = document.getElementById('recordStop');
    const textArea = document.getElementById('text');
    const statusEl = document.getElementById('status');
    const audioWrap = document.getElementById('audioWrap');
    const commitBtn = document.getElementById('commit');
    const reloadBtn = document.getElementById('reload');
    let uploadId = '';
    let previewObjectUrl = '';
    let recordingActive = false;
    let recordingStream = null;
    let recordingContext = null;
    let recordingSource = null;
    let recordingProcessor = null;
    let recordingSilentGain = null;
    let recordingChunks = [];

    function status(text, kind = '') {
      statusEl.textContent = text;
      statusEl.className = 'status' + (kind ? ' ' + kind : '');
    }
    function showAudio(url, isLocal = false) {
      if (!url) {
        if (previewObjectUrl) {
          URL.revokeObjectURL(previewObjectUrl);
          previewObjectUrl = '';
        }
        audioWrap.innerHTML = '';
        return;
      }
      if (isLocal) {
        if (previewObjectUrl) {
          URL.revokeObjectURL(previewObjectUrl);
        }
        previewObjectUrl = url;
      } else if (previewObjectUrl) {
        URL.revokeObjectURL(previewObjectUrl);
        previewObjectUrl = '';
      }
      audioWrap.innerHTML = `<audio src="${url}" controls preload="metadata"></audio>`;
    }
    function setRecordingButtons(active) {
      recordStartBtn.disabled = active;
      recordStopBtn.disabled = !active;
      reloadBtn.disabled = active;
    }
    function cleanupRecordingResources() {
      const context = recordingContext;
      const stream = recordingStream;
      const source = recordingSource;
      const processor = recordingProcessor;
      const silentGain = recordingSilentGain;
      recordingContext = null;
      recordingStream = null;
      recordingSource = null;
      recordingProcessor = null;
      recordingSilentGain = null;
      if (processor) {
        processor.onaudioprocess = null;
        try { processor.disconnect(); } catch (_err) {}
      }
      if (source) {
        try { source.disconnect(); } catch (_err) {}
      }
      if (silentGain) {
        try { silentGain.disconnect(); } catch (_err) {}
      }
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      if (context && context.state !== 'closed') {
        context.close().catch(() => {});
      }
    }
    function mergeChunks(chunks) {
      const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const merged = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }
      return merged;
    }
    function writeString(view, offset, text) {
      for (let i = 0; i < text.length; i += 1) {
        view.setUint8(offset + i, text.charCodeAt(i));
      }
    }
    function encodeWavBlob(samples, sampleRate) {
      const pcm16 = new Int16Array(samples.length);
      for (let i = 0; i < samples.length; i += 1) {
        const sample = Math.max(-1, Math.min(1, samples[i]));
        pcm16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      }
      const buffer = new ArrayBuffer(44 + pcm16.length * 2);
      const view = new DataView(buffer);
      writeString(view, 0, 'RIFF');
      view.setUint32(4, 36 + pcm16.length * 2, true);
      writeString(view, 8, 'WAVE');
      writeString(view, 12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeString(view, 36, 'data');
      view.setUint32(40, pcm16.length * 2, true);
      new Uint8Array(buffer, 44).set(new Uint8Array(pcm16.buffer));
      return new Blob([buffer], { type: 'audio/wav' });
    }
    async function transcribeBlob(blob) {
      uploadId = '';
      commitBtn.disabled = true;
      status('录音结束，正在转写。首次加载 paraformer 会慢一些...');
      const form = new FormData();
      form.append('audio', blob, 'user_voice.wav');
      const resp = await fetch('/api/transcribe', { method: 'POST', body: form });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || data.error || resp.statusText);
      uploadId = data.upload_id;
      textArea.value = data.text || '';
      showAudio(data.preview_url + '?t=' + Date.now());
      commitBtn.disabled = false;
      status('转写完成。请检查文本，必要时修改后提交。', 'ok');
    }
    async function startRecording() {
      if (recordingActive) return;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('当前浏览器不支持麦克风录音');
      }
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) {
        throw new Error('当前浏览器不支持音频上下文');
      }
      uploadId = '';
      commitBtn.disabled = true;
      recordingChunks = [];
      showAudio('');
      status('正在请求麦克风权限...');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const context = new AudioCtx();
      if (context.state === 'suspended') {
        await context.resume();
      }
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      const silentGain = context.createGain();
      silentGain.gain.value = 0;
      recordingActive = true;
      recordingStream = stream;
      recordingContext = context;
      recordingSource = source;
      recordingProcessor = processor;
      recordingSilentGain = silentGain;
      processor.onaudioprocess = (event) => {
        if (!recordingActive) return;
        const input = event.inputBuffer.getChannelData(0);
        recordingChunks.push(new Float32Array(input));
      };
      source.connect(processor);
      processor.connect(silentGain);
      silentGain.connect(context.destination);
      setRecordingButtons(true);
      status('录音中，请开始说话。', 'ok');
    }
    async function stopRecordingAndTranscribe() {
      if (!recordingActive) return;
      recordingActive = false;
      setRecordingButtons(false);
      status('正在停止录音并整理音频...');
      const sampleRate = recordingContext ? recordingContext.sampleRate : 48000;
      const samples = mergeChunks(recordingChunks);
      cleanupRecordingResources();
      if (!samples.length) {
        throw new Error('录音为空，请重新录制');
      }
      const blob = encodeWavBlob(samples, sampleRate);
      showAudio(URL.createObjectURL(blob), true);
      await transcribeBlob(blob);
    }
    async function loadSaved() {
      uploadId = '';
      commitBtn.disabled = true;
      const resp = await fetch('/api/user_voice');
      const data = await resp.json();
      if (data.exists) {
        textArea.value = data.text || '';
        showAudio(data.audio_url + '?t=' + Date.now());
        status('当前已保存用户音色。需要替换时重新录音并提交。', 'ok');
      } else {
        textArea.value = '';
        showAudio('');
        status('当前还没有保存用户音色。');
      }
    }
    recordStartBtn.onclick = async () => {
      try {
        await startRecording();
      } catch (err) {
        cleanupRecordingResources();
        setRecordingButtons(false);
        status(String(err.message || err), 'error');
      }
    };
    recordStopBtn.onclick = async () => {
      try {
        await stopRecordingAndTranscribe();
      } catch (err) {
        cleanupRecordingResources();
        setRecordingButtons(false);
        status(String(err.message || err), 'error');
      }
    };
    commitBtn.onclick = async () => {
      if (!uploadId) {
        status('没有可提交的录音，请先录音并转写。', 'error');
        return;
      }
      const text = textArea.value.trim();
      if (!text) {
        status('文本不能为空。', 'error');
        return;
      }
      status('提交中...');
      try {
        const resp = await fetch('/api/commit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ upload_id: uploadId, text })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || data.error || resp.statusText);
        uploadId = '';
        commitBtn.disabled = true;
        showAudio(data.audio_url + '?t=' + Date.now());
        status('已提交并覆盖用户音色。切回主页面后参考音色列表会刷新。', 'ok');
      } catch (err) {
        status(String(err.message || err), 'error');
      }
    };
    reloadBtn.onclick = () => loadSaved().catch(err => status(String(err.message || err), 'error'));
    loadSaved().catch(err => status(String(err.message || err), 'error'));
  </script>
</body>
</html>"""


def _json_error(message: str, status_code: int = 500) -> HTTPException:
    return HTTPException(status_code=status_code, detail=message)


def _target_paths() -> Dict[str, Path]:
    return {
        "wav": CLONE_PROMPT_DIR / USER_VOICE_WAV,
        "text": CLONE_PROMPT_DIR / USER_VOICE_TEXT,
    }


def _load_asr_model():
    global _asr_model
    if _asr_model is not None:
        return _asr_model
    try:
        from funasr import AutoModel
    except Exception as exc:
        raise RuntimeError(f"当前 Python 环境缺少 funasr，无法转写: {exc}") from exc
    if not ASR_MODEL_DIR.exists():
        raise RuntimeError(f"paraformer 模型目录不存在: {ASR_MODEL_DIR}")
    _asr_model = AutoModel(model=str(ASR_MODEL_DIR), disable_update=True)
    return _asr_model


def _safe_text_from_asr_result(result: Any) -> str:
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, dict):
                text = item.get("text") or item.get("sentence")
                if isinstance(text, str):
                    parts.append(text.strip())
        return "".join(parts).strip()
    if isinstance(result, dict):
        text = result.get("text") or result.get("sentence")
        if isinstance(text, str):
            return text.strip()
    return str(result or "").strip()


def _write_mono_wav(src_path: Path, dst_path: Path, sample_rate: int) -> None:
    audio, _sr = librosa.load(str(src_path), sr=sample_rate, mono=True)
    if audio.size == 0:
        raise RuntimeError("音频为空或解码失败")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio = audio / peak * 0.98
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), audio, sample_rate, subtype="PCM_16")


def _upload_paths(upload_id: str) -> Dict[str, Path]:
    safe_id = "".join(ch for ch in str(upload_id) if ch.isalnum() or ch in {"-", "_"})
    if not safe_id:
        raise ValueError("invalid upload_id")
    base = TMP_DIR / safe_id
    return {
        "dir": base,
        "original": base / "original_audio",
        "asr": base / "asr_16k.wav",
        "clone": base / "clone_24k.wav",
    }


def _public_audio_url() -> str:
    return "/api/user_voice/audio"


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_html())


@app.get("/api/user_voice")
async def get_user_voice():
    paths = _target_paths()
    exists = paths["wav"].is_file() and paths["text"].is_file()
    text = paths["text"].read_text(encoding="utf-8") if exists else ""
    updated_at = max(paths["wav"].stat().st_mtime, paths["text"].stat().st_mtime) if exists else None
    return JSONResponse(
        {
            "exists": exists,
            "id": "user_voice",
            "label": "用户音色",
            "text": text,
            "audio_url": _public_audio_url() if exists else "",
            "updated_at": updated_at,
            "clone_prompt_dir": str(CLONE_PROMPT_DIR),
        }
    )


@app.get("/api/user_voice/audio")
async def get_user_voice_audio():
    from fastapi.responses import FileResponse

    wav_path = _target_paths()["wav"]
    if not wav_path.is_file():
        raise _json_error("用户音色音频不存在", 404)
    return FileResponse(str(wav_path), media_type="audio/wav", filename=USER_VOICE_WAV)


@app.get("/api/preview/{upload_id}")
async def get_preview_audio(upload_id: str):
    from fastapi.responses import FileResponse

    try:
        clone_path = _upload_paths(upload_id)["clone"]
    except ValueError:
        raise _json_error("invalid upload_id", 400)
    if not clone_path.is_file():
        raise _json_error("预览音频不存在或已过期", 404)
    return FileResponse(str(clone_path), media_type="audio/wav", filename="preview_user_voice.wav")


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    raw = await audio.read()
    if not raw:
        raise _json_error("录音为空", 400)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise _json_error(f"录音过大，最大 {MAX_UPLOAD_BYTES // 1024 // 1024} MB", 413)

    upload_id = uuid.uuid4().hex
    paths = _upload_paths(upload_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    suffix = Path(audio.filename or "audio").suffix or ".audio"
    original = paths["dir"] / f"original{suffix}"
    original.write_bytes(raw)

    try:
        _write_mono_wav(original, paths["asr"], ASR_SAMPLE_RATE)
        _write_mono_wav(original, paths["clone"], TARGET_SAMPLE_RATE)
        model = _load_asr_model()
        result = model.generate(input=str(paths["asr"]), batch_size_s=300)
        text = _safe_text_from_asr_result(result)
    except Exception as exc:
        shutil.rmtree(paths["dir"], ignore_errors=True)
        raise _json_error(f"转写失败: {exc}", 500)

    return JSONResponse(
        {
            "upload_id": upload_id,
            "text": text,
            "preview_url": f"/api/preview/{upload_id}",
            "sample_rate": TARGET_SAMPLE_RATE,
        }
    )


class CommitPayload(BaseModel):
    upload_id: str
    text: str


@app.post("/api/commit")
async def commit(payload: CommitPayload):
    text = (payload.text or "").strip()
    if not text:
        raise _json_error("文本不能为空", 400)
    try:
        paths = _upload_paths(payload.upload_id)
    except ValueError:
        raise _json_error("invalid upload_id", 400)
    if not paths["clone"].is_file():
        raise _json_error("录音不存在或已过期，请重新录制", 404)

    target = _target_paths()
    tmp_wav = target["wav"].with_suffix(".wav.tmp")
    tmp_txt = target["text"].with_suffix(".txt.tmp")
    shutil.copyfile(paths["clone"], tmp_wav)
    tmp_txt.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp_wav, target["wav"])
    os.replace(tmp_txt, target["text"])
    shutil.rmtree(paths["dir"], ignore_errors=True)

    return JSONResponse(
        {
            "ok": True,
            "id": "user_voice",
            "label": "用户音色",
            "text": text,
            "audio_url": _public_audio_url(),
            "updated_at": time.time(),
            "wav": str(target["wav"]),
            "text_path": str(target["text"]),
        }
    )
