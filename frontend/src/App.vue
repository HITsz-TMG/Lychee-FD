<template>
  <div class="rt-app-container">
    <header class="rt-glass-header">
      <div class="rt-header-content">
        <img src="@/assets/hetao.png" class="rt-logo-img" alt="logo" />
        <span class="rt-header-title">lychee-FD <span class="rt-badge">Real-time Audio Call</span></span>
      </div>
    </header>
    <div v-if="startTalkErrorSummary" class="rt-global-error-strip">
      <div class="rt-global-error-content">
        <div class="rt-global-error-title">最近一次失败</div>
        <div class="rt-global-error-text">{{ startTalkErrorSummary }}</div>
        <div v-if="startTalkErrorHint" class="rt-global-error-hint">{{ startTalkErrorHint }}</div>
      </div>
      <button class="rt-global-error-dismiss" @click="dismissStartTalkError">清除</button>
    </div>

    <main class="rt-main-workspace">
      <transition name="fade">
        <div v-if="!isTalking && !aiReplyText" class="rt-welcome-screen">
          <div class="rt-welcome-card">
            <div class="rt-icon-pulse">🎙️</div>
            <h1>纯语音流式交互</h1>
            <p>实时上传 + 实时回复，端到端双工通话链路</p>
            <button class="rt-btn-primary rt-large-btn" @click="startTalk">
              <span class="rt-icon">📞</span> 立即开始通话
            </button>
          </div>
        </div>
      </transition>

      <transition name="fade">
        <div v-if="isTalking || aiReplyText || alignedSessionHistory.length > 0" class="rt-active-workspace">
          <div class="rt-panel rt-audio-panel">
            <div class="rt-panel-header">
              <span class="rt-panel-title">🎙️ 音频采集区</span>
              <div class="rt-status-badge" :class="connectionStatusClass">
                <span class="rt-dot"></span>
                <span class="rt-status-text">{{ connectionStatus || '初始化中...' }}</span>
              </div>
            </div>

            <div class="rt-audio-container">
              <div class="rt-mic-wrapper" :class="{ 'is-active': isTalking }">
                <div class="rt-mic-icon">🎤</div>
                <div class="rt-ripple" v-if="isTalking"></div>
                <div class="rt-ripple rt-ripple-2" v-if="isTalking"></div>
              </div>

              <div class="rt-audio-info">
                <div class="rt-audio-info-head">
                  <span>输入采样率: {{ captureSampleRateDisplay }} Hz</span>
                  <button
                    type="button"
                    class="rt-metrics-toggle"
                    @click="runtimeMetricsExpanded = !runtimeMetricsExpanded"
                  >
                    {{ runtimeMetricsExpanded ? '收起指标' : '展开指标' }}
                  </button>
                </div>
                <template v-if="runtimeMetricsExpanded">
                  <span class="rt-debug-flag">运行指标已开启</span>
                  <span>上传队列: {{ segmentQueueDepth }}</span>
                  <span>优先队列: {{ prioritySegmentQueueDepth }}</span>
                  <span>已发送分片: {{ segmentsSentCount }}</span>
                  <span>分片长度: {{ streamChunkMs }} ms / chunk</span>
                  <span v-if="priorityUploadInProgress">插队发送中: {{ priorityUploadSourceName || '测试音频' }}</span>
                  <span>延迟样本: {{ realtimeLatencySampleCount }}</span>
                  <span>前端送片→后端emit: {{ formatLatencyTriplet(realtimePreEmitClientLastMs, realtimePreEmitClientAvgMs, realtimePreEmitClientP95Ms) }}</span>
                  <span>后端收片→后端emit: {{ formatLatencyTriplet(realtimePreEmitServerLastMs, realtimePreEmitServerAvgMs, realtimePreEmitServerP95Ms) }}</span>
                  <span>后端队列延迟: {{ formatLatencyTriplet(realtimeServerQueueDelayLastMs, realtimeServerQueueDelayAvgMs, realtimeServerQueueDelayP95Ms) }}</span>
                  <span>后端emit→前端收包: {{ formatLatencyTriplet(realtimeLatencyReceiveLastMs, realtimeLatencyReceiveAvgMs, realtimeLatencyReceiveP95Ms) }}</span>
                  <span>后端emit→预计开播: {{ formatLatencyTriplet(realtimeLatencyAudibleLastMs, realtimeLatencyAudibleAvgMs, realtimeLatencyAudibleP95Ms) }}</span>
                  <span>播放排队积压: {{ formatLatencyTriplet(realtimePlaybackBacklogLastMs, realtimePlaybackBacklogAvgMs, realtimePlaybackBacklogP95Ms) }}</span>
                  <div class="rt-prob-row">
                    <span class="rt-prob-tag">S-L: {{ formatProbability(slProbability) }}</span>
                    <span class="rt-prob-tag">S-S: {{ formatProbability(ssProbability) }}</span>
                  </div>
                </template>
                <span v-else class="rt-audio-info-muted">运行指标已收起（点击“展开指标”查看）</span>
              </div>

              <div class="rt-playback-controls">
                <label for="playbackRate">播放速度</label>
                <input
                  id="playbackRate"
                  type="range"
                  min="0.5"
                  max="1.8"
                  step="0.1"
                  v-model.number="playbackRate"
                />
                <span>{{ playbackRate.toFixed(1) }}x</span>
              </div>
            </div>
          </div>

          <div class="rt-panel rt-text-panel">
            <div class="rt-panel-header">
              <span class="rt-panel-title">🤖 智能体响应</span>
            </div>
            <div class="rt-text-content" ref="chatMessagesRef">
              <div v-if="aiReplyText" class="markdown-body" v-html="renderMarkdown(aiReplyText)"></div>
              <div v-else class="rt-waiting-text">
                <span class="rt-typing-indicator">等待处理与响应<span>.</span><span>.</span><span>.</span></span>
              </div>
            </div>
          </div>

          <div class="rt-panel rt-history-panel">
            <div class="rt-panel-header">
              <span class="rt-panel-title">🗂️ 交互历史（未返回首页）</span>
            </div>
            <div class="rt-history-content">
              <template v-if="alignedSessionHistory.length > 0">
                <div
                  v-for="item in alignedSessionHistory"
                  :key="item.id"
                  class="rt-history-entry"
                >
                  <div class="rt-history-meta">
                    <div>会话: {{ item.sessionId || 'unknown' }}</div>
                    <div>时间: {{ item.startedAt }}</div>
                    <div>采样率: {{ item.sampleRate }} Hz</div>
                  </div>
                  <div class="rt-history-audio-block">
                    <div class="rt-history-label">输入对齐音频（{{ item.inputSec }}s）</div>
                    <audio :src="item.inputUrl" controls preload="metadata"></audio>
                  </div>
                  <div class="rt-history-audio-block">
                    <div class="rt-history-label">输出对齐音频（{{ item.outputSec }}s）</div>
                    <audio :src="item.outputUrl" controls preload="metadata"></audio>
                  </div>
                  <div v-if="item.rawOutputUrl" class="rt-history-audio-block">
                    <div class="rt-history-label">输出原始块音频（{{ item.rawOutputSec }}s）</div>
                    <audio :src="item.rawOutputUrl" controls preload="metadata"></audio>
                  </div>
                </div>
              </template>
              <div v-else class="rt-waiting-text">
                暂无对齐音频历史
              </div>
            </div>
          </div>
        </div>
      </transition>
    </main>

    <footer class="rt-bottom-control-bar" v-show="isTalking || aiReplyText || alignedSessionHistory.length > 0">
      <input
        id="hidden-realtime-audio-inject-input"
        type="file"
        accept="audio/*"
        @change="onRealtimeAudioFileChange"
        hidden
      />
      <div class="rt-control-group">
        <template v-if="!isTalking">
          <button class="rt-btn-primary" @click="startTalk">
            📞 重新连接
          </button>
          <button class="rt-btn-secondary" @click="returnToHome">
            🏠 返回首页
          </button>
        </template>
        <button v-else class="rt-btn-danger rt-hover-shake" @click="endTalk">
          ☎️ 挂断通话
        </button>
        <button v-if="isTalking" class="rt-btn-secondary" @click="clickRealtimeAudioInjectButton">
          🎵 发送测试音频
        </button>
      </div>
    </footer>
  </div>
</template>




<script setup>
import { ref, watch, nextTick, onMounted, onBeforeUnmount } from 'vue'
import hljs from 'highlight.js'
import 'highlight.js/styles/atom-one-dark.css' // 可选样式，比如 GitHub 风格
import newFavicon from '@/assets/hetao.png'  // 引入新的 favicon 图标
import MarkdownIt from 'markdown-it'         // 引入 Markdown 解析库
import WAVEncoder from 'wav-encoder'         // 引入 WAV 编码器
import { ElNotification } from 'element-plus'  // 引入 Element Plus 的通知组件


// ==========================================
// 实时通话 (Function 6) 
// ==========================================
const isTalking = ref(false);
const mediaStream = ref(null);
const captureAudioContext = ref(null);
const connectionStatus = ref('');
const connectionStatusClass = ref('');
const aiReplyText = ref(''); // 用于在页面上流式展示 AI 的文字回复
const startTalkErrorSummary = ref('');
const startTalkErrorHint = ref('');
const playbackRate = ref(1.0);
const alignedSessionHistory = ref([]);
const captureSampleRateDisplay = ref('-');
const segmentQueueDepth = ref(0);
const prioritySegmentQueueDepth = ref(0);
const segmentsSentCount = ref(0);
const priorityUploadInProgress = ref(false);
const priorityUploadSourceName = ref('');
const streamChunkMs = 200;
const runtimeMetricsExpanded = ref(true);
const realtimeBackendHint = (() => {
  try {
    const v = new URLSearchParams(window.location.search).get('rtBackend');
    return (v || '').trim().toLowerCase();
  } catch (_err) {
    return '';
  }
})();
const slProbability = ref(null);
const ssProbability = ref(null);
const START_TALK_ERROR_STORAGE_KEY = 'fd_realtime_start_error_v1';

const TARGET_SAMPLE_RATE = 16000;
const MIN_SEGMENT_SAMPLES = 800;
const MAX_UPLOAD_QUEUE_DEPTH = 8;

let captureSourceNode = null;
let captureProcessorNode = null;
let captureSilentGainNode = null;
let uploadInterval = null;

let pendingCaptureChunks = [];
let pendingCaptureSamples = 0;

let uploadQueue = [];
let priorityUploadQueue = [];
let queueProcessing = false;
let nextUploadAllowedAtMs = 0;
let segmentSeqId = 0;
let currentAlignedArchive = null;
const realtimeSessionId = ref('');
const realtimeStoppingExpected = ref(false);

const activeRequestControllers = new Set();
const gradioAudioQueue = [];
let gradioAudioPlaying = false;
let currentGradioAudio = null;
let lastQueuedAudioUrl = '';
let realtimePlaybackContext = null;
let realtimePlaybackGainNode = null;
let realtimePlaybackWorkletNode = null;
let realtimePlaybackWorkletBlobUrl = null;
let realtimePlaybackBufferLevelFrames = 0;
let realtimePlaybackIsPlaying = false;
let realtimePlaybackAutoRate = 1.0;
let realtimePlaybackManualRate = 1.0;
let realtimePlaybackSupportsWorklet = false;
// Fallback timeline scheduler state for browsers without AudioWorklet.
let realtimePlaybackTimelineStart = 0;
let realtimePlaybackScheduledSec = 0;
let realtimePlaybackLastSource = null;
const REALTIME_PLAYBACK_MIN_LEAD_SEC = 0.015;
const REALTIME_PLAYBACK_MAX_LAG_SEC = 0.08;
const REALTIME_PLAYBACK_SMOOTHING = {
  ringBufferSec: 12,
  startWarmupSec: 0.55,
  lowWaterSec: 0.28,
  highWaterSec: 1.20,
  lowWaterRate: 0.985,
  highWaterRate: 1.055,
  levelSmoothKeep: 0.95,
  levelSmoothUpdate: 0.05,
  autoRateSmoothKeep: 0.985,
  autoRateSmoothUpdate: 0.015
};
const REALTIME_PLAYBACK_START_WARMUP_SEC = REALTIME_PLAYBACK_SMOOTHING.startWarmupSec;
const REALTIME_LATENCY_HISTORY_MAX = 200;
let realtimeLatencyReceiveHistory = [];
let realtimeLatencyAudibleHistory = [];
let realtimeLatencyBacklogHistory = [];
let realtimePreEmitClientHistory = [];
let realtimePreEmitServerHistory = [];
let realtimeServerQueueDelayHistory = [];
const realtimeLatencySampleCount = ref(0);
const realtimePreEmitClientLastMs = ref(null);
const realtimePreEmitClientAvgMs = ref(null);
const realtimePreEmitClientP95Ms = ref(null);
const realtimePreEmitServerLastMs = ref(null);
const realtimePreEmitServerAvgMs = ref(null);
const realtimePreEmitServerP95Ms = ref(null);
const realtimeServerQueueDelayLastMs = ref(null);
const realtimeServerQueueDelayAvgMs = ref(null);
const realtimeServerQueueDelayP95Ms = ref(null);
const realtimeLatencyReceiveLastMs = ref(null);
const realtimeLatencyReceiveAvgMs = ref(null);
const realtimeLatencyReceiveP95Ms = ref(null);
const realtimeLatencyAudibleLastMs = ref(null);
const realtimeLatencyAudibleAvgMs = ref(null);
const realtimeLatencyAudibleP95Ms = ref(null);
const realtimePlaybackBacklogLastMs = ref(null);
const realtimePlaybackBacklogAvgMs = ref(null);
const realtimePlaybackBacklogP95Ms = ref(null);

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const toFiniteNumber = (v) => {
  if (v && typeof v === 'object' && Object.prototype.hasOwnProperty.call(v, 'value')) {
    v = v.value;
  }
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

const formatLatencyMs = (v) => {
  const n = toFiniteNumber(v);
  if (n === null) return '--';
  return `${n.toFixed(1)}ms`;
};

const formatLatencyTriplet = (lastMs, avgMs, p95Ms) => {
  return `last ${formatLatencyMs(lastMs)} | avg ${formatLatencyMs(avgMs)} | p95 ${formatLatencyMs(p95Ms)}`;
};

const calcPercentile = (arr, p) => {
  if (!Array.isArray(arr) || arr.length === 0) return null;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
};

const updateLatencySeries = (history, value, lastRef, avgRef, p95Ref) => {
  const n = toFiniteNumber(value);
  if (n === null) return;
  const clamped = Math.max(0, n);
  history.push(clamped);
  if (history.length > REALTIME_LATENCY_HISTORY_MAX) {
    history.splice(0, history.length - REALTIME_LATENCY_HISTORY_MAX);
  }
  lastRef.value = clamped;
  avgRef.value = history.reduce((acc, x) => acc + x, 0) / history.length;
  p95Ref.value = calcPercentile(history, 95);
  realtimeLatencySampleCount.value = Math.max(
    realtimePreEmitClientHistory.length,
    realtimePreEmitServerHistory.length,
    realtimeServerQueueDelayHistory.length,
    realtimeLatencyReceiveHistory.length,
    realtimeLatencyAudibleHistory.length,
    realtimeLatencyBacklogHistory.length
  );
};

const resetRealtimeLatencyStats = () => {
  realtimePreEmitClientHistory = [];
  realtimePreEmitServerHistory = [];
  realtimeServerQueueDelayHistory = [];
  realtimeLatencyReceiveHistory = [];
  realtimeLatencyAudibleHistory = [];
  realtimeLatencyBacklogHistory = [];
  realtimeLatencySampleCount.value = 0;
  realtimePreEmitClientLastMs.value = null;
  realtimePreEmitClientAvgMs.value = null;
  realtimePreEmitClientP95Ms.value = null;
  realtimePreEmitServerLastMs.value = null;
  realtimePreEmitServerAvgMs.value = null;
  realtimePreEmitServerP95Ms.value = null;
  realtimeServerQueueDelayLastMs.value = null;
  realtimeServerQueueDelayAvgMs.value = null;
  realtimeServerQueueDelayP95Ms.value = null;
  realtimeLatencyReceiveLastMs.value = null;
  realtimeLatencyReceiveAvgMs.value = null;
  realtimeLatencyReceiveP95Ms.value = null;
  realtimeLatencyAudibleLastMs.value = null;
  realtimeLatencyAudibleAvgMs.value = null;
  realtimeLatencyAudibleP95Ms.value = null;
  realtimePlaybackBacklogLastMs.value = null;
  realtimePlaybackBacklogAvgMs.value = null;
  realtimePlaybackBacklogP95Ms.value = null;
};

const updateRealtimeAudioLatencyMetrics = ({
  backendEmitEpochMs,
  clientReceiveEpochMs,
  scheduleMetrics,
  preEmitClientMs,
  preEmitServerMs,
  serverQueueDelayMs
}) => {
  const preClient = toFiniteNumber(preEmitClientMs);
  if (preClient !== null) {
    updateLatencySeries(
      realtimePreEmitClientHistory,
      preClient,
      realtimePreEmitClientLastMs,
      realtimePreEmitClientAvgMs,
      realtimePreEmitClientP95Ms
    );
  }

  const preServer = toFiniteNumber(preEmitServerMs);
  if (preServer !== null) {
    updateLatencySeries(
      realtimePreEmitServerHistory,
      preServer,
      realtimePreEmitServerLastMs,
      realtimePreEmitServerAvgMs,
      realtimePreEmitServerP95Ms
    );
  }

  const queueDelay = toFiniteNumber(serverQueueDelayMs);
  if (queueDelay !== null) {
    updateLatencySeries(
      realtimeServerQueueDelayHistory,
      queueDelay,
      realtimeServerQueueDelayLastMs,
      realtimeServerQueueDelayAvgMs,
      realtimeServerQueueDelayP95Ms
    );
  }

  const emitMs = toFiniteNumber(backendEmitEpochMs);
  const recvMs = toFiniteNumber(clientReceiveEpochMs);
  if (emitMs !== null && recvMs !== null) {
    updateLatencySeries(
      realtimeLatencyReceiveHistory,
      recvMs - emitMs,
      realtimeLatencyReceiveLastMs,
      realtimeLatencyReceiveAvgMs,
      realtimeLatencyReceiveP95Ms
    );
  }

  const startLeadMs = toFiniteNumber(scheduleMetrics?.startLeadMs);
  const backlogMs = toFiniteNumber(scheduleMetrics?.backlogMs);
  if (backlogMs !== null) {
    updateLatencySeries(
      realtimeLatencyBacklogHistory,
      backlogMs,
      realtimePlaybackBacklogLastMs,
      realtimePlaybackBacklogAvgMs,
      realtimePlaybackBacklogP95Ms
    );
  }

  if (emitMs !== null && recvMs !== null && startLeadMs !== null) {
    updateLatencySeries(
      realtimeLatencyAudibleHistory,
      (recvMs + startLeadMs) - emitMs,
      realtimeLatencyAudibleLastMs,
      realtimeLatencyAudibleAvgMs,
      realtimeLatencyAudibleP95Ms
    );
  }
};

const derivePreEmitMs = (explicitMs, emitEpochMs, priorEpochMs) => {
  const explicit = toFiniteNumber(explicitMs);
  if (explicit !== null) {
    return explicit;
  }
  const emitMs = toFiniteNumber(emitEpochMs);
  const priorMs = toFiniteNumber(priorEpochMs);
  if (emitMs === null || priorMs === null) {
    return null;
  }
  return Math.max(0, emitMs - priorMs);
};

const deriveServerQueueDelayMs = (explicitMs, sseSendEpochMs, emitEpochMs) => {
  const explicit = toFiniteNumber(explicitMs);
  if (explicit !== null) {
    return Math.max(0, explicit);
  }
  const sendMs = toFiniteNumber(sseSendEpochMs);
  const emitMs = toFiniteNumber(emitEpochMs);
  if (sendMs === null || emitMs === null) {
    return null;
  }
  return Math.max(0, sendMs - emitMs);
};

const normalizeRealtimeEventPayload = (payload, sseEventType = 'message') => {
  if (!payload || typeof payload !== 'object') {
    return null;
  }

  const rawType = typeof payload.type === 'string' ? payload.type : '';
  const unifiedType = typeof payload.event_type === 'string' ? payload.event_type : '';
  const fallbackType = typeof sseEventType === 'string' && sseEventType ? sseEventType : 'message';
  const mergedType = unifiedType || rawType || fallbackType;
  const normalized = { ...payload, type: mergedType };

  if (mergedType === 'status' && typeof normalized.status !== 'string' && typeof payload.frame_status === 'string') {
    normalized.status = payload.frame_status;
  }
  if (mergedType === 'error' && typeof normalized.error !== 'string' && typeof payload.frame_error === 'string') {
    normalized.error = payload.frame_error;
  }

  const frameText = payload.frame_text && typeof payload.frame_text === 'object'
    ? payload.frame_text
    : null;
  if (mergedType === 'assistant_text' && typeof normalized.text !== 'string' && frameText) {
    if (typeof frameText.delta === 'string' && frameText.delta) {
      normalized.text = frameText.delta;
    } else if (typeof frameText.snapshot === 'string' && frameText.snapshot) {
      normalized.text = frameText.snapshot;
    }
  }
  if (mergedType === 'assistant_text' && frameText) {
    if (typeof normalized.snapshot !== 'string' && typeof frameText.snapshot === 'string') {
      normalized.snapshot = frameText.snapshot;
    }
    if (typeof normalized.event_id !== 'string' && typeof frameText.event_id === 'string') {
      normalized.event_id = frameText.event_id;
    }
    if (typeof normalized.event_kind !== 'string' && typeof frameText.event_kind === 'string') {
      normalized.event_kind = frameText.event_kind;
    }
    if (typeof normalized.seq === 'undefined' && typeof frameText.seq !== 'undefined') {
      normalized.seq = frameText.seq;
    }
    if (typeof normalized.is_final === 'undefined' && typeof frameText.is_final !== 'undefined') {
      normalized.is_final = frameText.is_final;
    }
  }

  const frameAudio = payload.frame_audio && typeof payload.frame_audio === 'object'
    ? payload.frame_audio
    : null;
  if (frameAudio && (mergedType === 'audio_chunk_pcm' || mergedType === 'audio_chunk')) {
    const audioFormat = String(frameAudio.format || '').toLowerCase();
    if (audioFormat === 'pcm_s16le' || (typeof frameAudio.pcm_b64 === 'string' && frameAudio.pcm_b64)) {
      normalized.type = 'audio_chunk_pcm';
      if (typeof normalized.pcm_b64 !== 'string' || !normalized.pcm_b64) {
        normalized.pcm_b64 = frameAudio.pcm_b64;
      }
      if (toFiniteNumber(normalized.sample_rate) === null) {
        normalized.sample_rate = frameAudio.sample_rate;
      }
      if (toFiniteNumber(normalized.num_channels) === null) {
        normalized.num_channels = frameAudio.num_channels;
      }
      if (toFiniteNumber(normalized.num_samples) === null) {
        normalized.num_samples = frameAudio.num_samples;
      }
      if (typeof normalized.pcm_format !== 'string' || !normalized.pcm_format) {
        normalized.pcm_format = 's16le';
      }
    } else if (audioFormat === 'wav_b64' || (typeof frameAudio.wav_b64 === 'string' && frameAudio.wav_b64)) {
      normalized.type = 'audio_chunk';
      if (typeof normalized.wav_b64 !== 'string' || !normalized.wav_b64) {
        normalized.wav_b64 = frameAudio.wav_b64;
      }
      if (toFiniteNumber(normalized.sample_rate) === null) {
        normalized.sample_rate = frameAudio.sample_rate;
      }
      if (toFiniteNumber(normalized.num_channels) === null) {
        normalized.num_channels = frameAudio.num_channels;
      }
      if (toFiniteNumber(normalized.num_samples) === null) {
        normalized.num_samples = frameAudio.num_samples;
      }
    }
  }

  return normalized;
};

const formatProbability = (v) => {
  const n = toFiniteNumber(v);
  if (n === null) return '--';
  return `${(Math.max(0, Math.min(1, n)) * 100).toFixed(1)}%`;
};

const resetRealtimeProbabilities = () => {
  slProbability.value = null;
  ssProbability.value = null;
};

const softmax2 = (a, b) => {
  const va = toFiniteNumber(a);
  const vb = toFiniteNumber(b);
  if (va === null || vb === null) return null;
  const m = Math.max(va, vb);
  const ea = Math.exp(va - m);
  const eb = Math.exp(vb - m);
  const s = ea + eb;
  if (!Number.isFinite(s) || s <= 0) return null;
  return { sl: ea / s, ss: eb / s };
};

const normalizeProbLike = (v) => {
  const n = toFiniteNumber(v);
  if (n === null) return null;
  if (n > 1 && n <= 100) return n / 100;
  if (n >= 0 && n <= 1) return n;
  return null;
};

const extractReadableErrorMessage = (err) => {
  if (!err) return '未知错误';
  if (typeof err === 'string' && err.trim()) return err.trim();
  if (typeof err?.message === 'string' && err.message.trim()) return err.message.trim();
  try {
    const asStr = String(err);
    if (asStr && asStr !== '[object Object]') {
      return asStr;
    }
  } catch (_err) {
    // ignore
  }
  try {
    return JSON.stringify(err);
  } catch (_err) {
    return '未知错误';
  }
};

const isAbortLikeError = (err) => {
  const name = typeof err?.name === 'string' ? err.name.trim() : '';
  const message = extractReadableErrorMessage(err);
  const merged = `${name} ${message}`.toLowerCase();
  if (name === 'AbortError') {
    return true;
  }
  return (
    merged.includes('aborterror') ||
    merged.includes('aborted') ||
    merged.includes('bodystreambuffer was aborted')
  );
};

const classifyStartTalkError = (err) => {
  const name = typeof err?.name === 'string' ? err.name.trim() : '';
  const message = extractReadableErrorMessage(err);
  const raw = `${name} ${message}`.toLowerCase();
  let hint = '请打开浏览器控制台查看详细报错。';
  if (name === 'NotAllowedError' || /permission|denied|notallowed/.test(raw)) {
    hint = '麦克风权限被拒绝，请在浏览器站点权限中允许麦克风后重试。';
  } else if (name === 'NotFoundError' || /notfound|device.*not found|no input device/.test(raw)) {
    hint = '未检测到可用麦克风设备，请检查系统录音设备。';
  } else if (name === 'NotReadableError' || /notreadable|device in use|hardware|could not start audio source/.test(raw)) {
    hint = '麦克风可能被其他程序占用，请关闭占用后重试。';
  } else if (name === 'SecurityError' || /insecure|secure context/.test(raw)) {
    hint = '当前页面不在安全上下文，请优先使用 localhost 或 https 访问。';
  } else if (/failed to fetch|networkerror|load failed|err_connection|cors|502|504/.test(raw)) {
    hint = '前后端请求失败，常见原因是代理劫持、跨域或端口未通。';
  }
  const summary = name && !message.toLowerCase().startsWith(`${name.toLowerCase()}:`)
    ? `${name}: ${message}`
    : message;
  return { summary, hint };
};

const persistStartTalkError = (summary, hint) => {
  if (typeof window === 'undefined' || !window.localStorage) return;
  try {
    window.localStorage.setItem(
      START_TALK_ERROR_STORAGE_KEY,
      JSON.stringify({
        summary: String(summary || ''),
        hint: String(hint || ''),
        ts: Date.now()
      })
    );
  } catch (_err) {
    // Ignore quota or privacy mode failures.
  }
};

const restorePersistedStartTalkError = () => {
  if (typeof window === 'undefined' || !window.localStorage) return;
  try {
    const raw = window.localStorage.getItem(START_TALK_ERROR_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    const summary = typeof parsed?.summary === 'string' ? parsed.summary.trim() : '';
    const hint = typeof parsed?.hint === 'string' ? parsed.hint.trim() : '';
    if (!summary) return;
    startTalkErrorSummary.value = summary;
    startTalkErrorHint.value = hint;
  } catch (_err) {
    // Ignore parse errors.
  }
};

const clearPersistedStartTalkError = () => {
  if (typeof window === 'undefined' || !window.localStorage) return;
  try {
    window.localStorage.removeItem(START_TALK_ERROR_STORAGE_KEY);
  } catch (_err) {
    // Ignore storage failures.
  }
};

const clearStartTalkError = ({ clearStorage = true } = {}) => {
  startTalkErrorSummary.value = '';
  startTalkErrorHint.value = '';
  if (clearStorage) {
    clearPersistedStartTalkError();
  }
};

const dismissStartTalkError = () => {
  clearStartTalkError({ clearStorage: true });
};

const setStartTalkError = (
  summary,
  hint = '',
  {
    toast = false,
    title = '实时通话错误',
    duration = 9000,
    forceToast = false
  } = {}
) => {
  const normalizedSummary = extractReadableErrorMessage(summary);
  const normalizedHint = typeof hint === 'string' ? hint.trim() : '';
  const sameAsCurrent =
    normalizedSummary === startTalkErrorSummary.value &&
    normalizedHint === startTalkErrorHint.value;

  startTalkErrorSummary.value = normalizedSummary;
  startTalkErrorHint.value = normalizedHint;
  persistStartTalkError(normalizedSummary, normalizedHint);

  if (toast && (!sameAsCurrent || forceToast)) {
    ElNotification({
      title,
      message: normalizedHint
        ? `${normalizedSummary} ${normalizedHint}`
        : normalizedSummary,
      type: 'error',
      duration,
      showClose: true
    });
  }
};

const realtimeCommittedReply = ref('');
let realtimeLiveReply = '';
let realtimeLastCommittedReply = '';
let realtimeLastEventSignature = '';
let realtimeLastState = null;
let realtimeCurrentEventText = '';
let realtimePendingSsLineBreak = false;
let realtimeStructuredTextEvents = [];

const normalizeRealtimeText = (text) => {
  if (typeof text !== 'string') return '';
  return text.replace(/\r/g, '').trim();
};

const renderRealtimeReply = () => {
  let structured = '';
  for (const item of realtimeStructuredTextEvents) {
    const text = normalizeRealtimeText(item?.text);
    if (!text) continue;
    if (!structured) {
      structured = text;
    } else {
      structured = `${structured}${text}`;
    }
  }
  if (structured) {
    aiReplyText.value = structured;
    return;
  }
  const committed = normalizeRealtimeText(realtimeCommittedReply.value);
  const live = normalizeRealtimeText(realtimeLiveReply);
  if (committed && live) {
    aiReplyText.value = `${committed}\n\n${live}`;
  } else {
    aiReplyText.value = committed || live || '';
  }
};

const handleStructuredRealtimeText = (payload) => {
  if (!payload || typeof payload !== 'object') return false;
  const frameText = payload.frame_text && typeof payload.frame_text === 'object'
    ? payload.frame_text
    : null;
  const eventIdRaw = payload.event_id || frameText?.event_id;
  const snapshotRaw = typeof payload.snapshot === 'string'
    ? payload.snapshot
    : (typeof frameText?.snapshot === 'string' ? frameText.snapshot : '');
  const eventId = normalizeRealtimeText(String(eventIdRaw || ''));
  const snapshot = normalizeRealtimeText(snapshotRaw);
  if (
    !eventId ||
    !snapshot ||
    /^\(no text\)$/i.test(snapshot) ||
    /^\*\*\[[^\]]+\]\*\*\s*generating/i.test(snapshot) ||
    /^State:\s*/i.test(snapshot) ||
    /^\*\*\[[^\]]+\]\*\*/.test(snapshot)
  ) {
    return false;
  }

  const seqRaw = typeof payload.seq !== 'undefined' ? payload.seq : frameText?.seq;
  const seq = Number.isFinite(Number(seqRaw)) ? Number(seqRaw) : 0;
  const eventKind = normalizeRealtimeText(payload.event_kind || frameText?.event_kind || 'response').toLowerCase();
  const isFinal = !!(payload.is_final ?? frameText?.is_final);
  const resumed = !!(payload.resumed ?? frameText?.resumed);
  const existingIdx = realtimeStructuredTextEvents.findIndex(item => item.id === eventId);
  if (existingIdx >= 0) {
    const existing = realtimeStructuredTextEvents[existingIdx];
    if (Number.isFinite(existing.seq) && seq > 0 && existing.seq > seq) {
      return true;
    }
    realtimeStructuredTextEvents[existingIdx] = {
      ...existing,
      kind: eventKind || existing.kind,
      text: snapshot,
      seq: Math.max(Number(existing.seq) || 0, seq),
      isFinal: isFinal || !!existing.isFinal,
      resumed: resumed || !!existing.resumed,
      breakBefore: false
    };
  } else {
    realtimeStructuredTextEvents.push({
      id: eventId,
      kind: eventKind || 'response',
      text: snapshot,
      seq,
      isFinal,
      resumed,
      breakBefore: false
    });
  }
  realtimeLiveReply = '';
  realtimeCurrentEventText = snapshot;
  realtimePendingSsLineBreak = false;
  renderRealtimeReply();
  return true;
};

const appendRealtimeCommittedChunk = (chunk, options = {}) => {
  const newParagraph = options?.newParagraph === true;
  const prependNewline = options?.prependNewline === true;
  if (typeof chunk !== 'string') return;
  const text = chunk.replace(/\r/g, '');
  if (!text) return;
  if (!realtimeCommittedReply.value) {
    realtimeCommittedReply.value = text;
  } else if (newParagraph) {
    realtimeCommittedReply.value = `${realtimeCommittedReply.value}\n\n${text}`;
  } else if (prependNewline) {
    realtimeCommittedReply.value = `${realtimeCommittedReply.value}\n${text}`;
  } else {
    realtimeCommittedReply.value = `${realtimeCommittedReply.value}${text}`;
  }
  realtimeLiveReply = '';
  renderRealtimeReply();
};

const appendRealtimeIncrementalEventText = (fullText) => {
  const next = normalizeRealtimeText(fullText);
  if (!next) return;
  const prev = normalizeRealtimeText(realtimeCurrentEventText);
  const needSsLineBreak = !!realtimePendingSsLineBreak && !!realtimeCommittedReply.value;
  if (!prev) {
    if (needSsLineBreak) {
      appendRealtimeCommittedChunk(next, { prependNewline: true });
    } else {
      appendRealtimeCommittedChunk(next, { newParagraph: !!realtimeCommittedReply.value });
    }
    realtimePendingSsLineBreak = false;
    realtimeCurrentEventText = next;
    return;
  }
  if (needSsLineBreak) {
    appendRealtimeCommittedChunk(next, { prependNewline: true });
    realtimePendingSsLineBreak = false;
    realtimeCurrentEventText = next;
    return;
  }
  if (next.startsWith(prev)) {
    const delta = next.slice(prev.length);
    if (delta) {
      appendRealtimeCommittedChunk(delta, { newParagraph: false });
    }
    realtimeCurrentEventText = next;
    return;
  }
  if (prev.startsWith(next)) {
    // Ignore temporary rollbacks; keep waiting for a longer continuation.
    realtimeCurrentEventText = next;
    return;
  }
  appendRealtimeCommittedChunk(next, { newParagraph: true });
  realtimeCurrentEventText = next;
};

const finalizeRealtimeEventText = (eventText) => {
  const payload = normalizeRealtimeText(eventText);
  if (!payload) {
    realtimeCurrentEventText = '';
    return;
  }
  const current = normalizeRealtimeText(realtimeCurrentEventText);
  if (!current) {
    appendRealtimeCommittedChunk(payload, { newParagraph: !!realtimeCommittedReply.value });
    realtimeCurrentEventText = '';
    return;
  }
  if (payload.startsWith(current)) {
    const delta = payload.slice(current.length);
    if (delta) {
      appendRealtimeCommittedChunk(delta, { newParagraph: false });
    }
    realtimeCurrentEventText = '';
    return;
  }
  if (current.startsWith(payload)) {
    realtimeCurrentEventText = '';
    return;
  }
  appendRealtimeCommittedChunk(payload, { newParagraph: true });
  realtimeCurrentEventText = '';
};

const appendRealtimeEventText = (eventText, signature = '', options = {}) => {
  const allowImmediateDuplicate = options?.allowImmediateDuplicate === true;
  const clean = normalizeRealtimeText(eventText);
  if (!clean) return;
  const dedupeSig = normalizeRealtimeText(signature || clean);
  if (dedupeSig && dedupeSig === realtimeLastEventSignature) {
    return;
  }
  if (!allowImmediateDuplicate && clean === realtimeLastCommittedReply) {
    return;
  }
  if (realtimeCommittedReply.value) {
    realtimeCommittedReply.value = `${realtimeCommittedReply.value}\n\n${clean}`;
  } else {
    realtimeCommittedReply.value = clean;
  }
  realtimeLastEventSignature = dedupeSig;
  realtimeLastCommittedReply = clean;
  realtimeLiveReply = '';
  renderRealtimeReply();
};

const commitRealtimeLiveReply = () => {
  if (realtimeStructuredTextEvents.length > 0) {
    realtimeLiveReply = '';
    realtimeCurrentEventText = '';
    realtimePendingSsLineBreak = false;
    renderRealtimeReply();
    return;
  }
  const live = normalizeRealtimeText(realtimeLiveReply);
  if (live) {
    appendRealtimeCommittedChunk(live, { newParagraph: !!realtimeCommittedReply.value });
  }
  realtimeLiveReply = '';
  realtimeCurrentEventText = '';
  realtimePendingSsLineBreak = false;
};

const resetRealtimeReplyState = () => {
  realtimeCommittedReply.value = '';
  realtimeLiveReply = '';
  realtimeLastCommittedReply = '';
  realtimeLastEventSignature = '';
  realtimeLastState = null;
  realtimeCurrentEventText = '';
  realtimePendingSsLineBreak = false;
  realtimeStructuredTextEvents = [];
  aiReplyText.value = '';
};

const parseStateFromStatus = (statusText) => {
  if (typeof statusText !== 'string' || !statusText) return null;
  const match = statusText.match(/\bstate\s*=\s*([lsb])\b/i);
  if (!match) return null;
  return String(match[1]).toLowerCase();
};

const parseStateChangeFromAssistant = (assistantText) => {
  const text = normalizeRealtimeText(assistantText);
  if (!text) return null;
  const match = text.match(/State:\s*([LSB])\s*->\s*([LSB])([\s\S]*)$/i);
  if (!match) return null;
  return {
    from: String(match[1]).toLowerCase(),
    to: String(match[2]).toLowerCase()
  };
};

const isAssistantNoiseLine = (assistantText) => {
  const text = normalizeRealtimeText(assistantText);
  if (!text) return true;
  if (/^---\s*$/.test(text)) return true;
  if (/^###\s*Summary/i.test(text)) return true;
  if (/^- Input:\s*/im.test(text)) return true;
  if (/^- Events:\s*/im.test(text)) return true;
  if (/^\*\*\[[^\]]+\]\*\*\s*generating/i.test(text)) return true;
  if (/^Audio synthesis complete:/i.test(text)) return true;
  if (/^No audio output/i.test(text)) return true;
  return false;
};

const extractEventTextsFromAssistant = (assistantText) => {
  const text = normalizeRealtimeText(assistantText);
  if (!text || isAssistantNoiseLine(text)) return [];

  const results = [];
  const eventPattern = /\*\*\[([^\]]+)\]\*\*\s*([\s\S]*?)(?=(?:\*\*\[[^\]]+\]\*\*)|$)/gi;
  let matched = false;
  let match = null;

  while ((match = eventPattern.exec(text)) !== null) {
    matched = true;
    const eventKind = normalizeRealtimeText(match[1]).toLowerCase();
    let payload = normalizeRealtimeText(match[2]);
    if (!payload) continue;
    if (/^\s*generating/i.test(payload)) continue;
    payload = normalizeRealtimeText(payload.replace(/\s*Audio:\s*[0-9]+[\s\S]*$/i, ''));
    if (!payload || /^\(no text\)$/i.test(payload)) continue;
    results.push({
      text: payload,
      signature: `${eventKind}|${payload}`
    });
  }

  if (!matched) return [];
  return results;
};

const handleAssistantRealtimeText = (assistantText) => {
  const raw = normalizeRealtimeText(assistantText);
  if (!raw) return;

  const stateChange = parseStateChangeFromAssistant(raw);
  if (stateChange) {
    const fromState = stateChange.from || realtimeLastState;
    if (fromState && fromState !== 'l' && stateChange.to === 'l') {
      commitRealtimeLiveReply();
    }
    if (stateChange.to === 'l') {
      realtimeCurrentEventText = '';
    }
    realtimeLastState = stateChange.to;
    return;
  }

  const textDelta = raw.match(/^\*\*Text\*\*:\s*([\s\S]*)$/i);
  if (textDelta) {
    appendRealtimeIncrementalEventText(textDelta[1]);
    return;
  }

  const eventPayloads = extractEventTextsFromAssistant(raw);
  if (eventPayloads.length > 0) {
    for (const item of eventPayloads) {
      finalizeRealtimeEventText(item.text);
      const dedupeSig = normalizeRealtimeText(item.signature || item.text);
      if (dedupeSig) {
        realtimeLastEventSignature = dedupeSig;
      }
      realtimeLastCommittedReply = normalizeRealtimeText(item.text);
    }
    return;
  }

  if (/\*\*\[[^\]]+\]\*\*/.test(raw)) {
    realtimeCurrentEventText = '';
    return;
  }

  if (!isAssistantNoiseLine(raw) && !/^State:\s*/i.test(raw)) {
    appendRealtimeCommittedChunk(raw, { newParagraph: !!realtimeCommittedReply.value });
    realtimeCurrentEventText = '';
  }
};

const updateRealtimeProbabilities = ({
  sl = null,
  ss = null,
  treatAsLogits = false,
  slProvided = false,
  ssProvided = false
} = {}) => {
  if (treatAsLogits) {
    const probs = softmax2(sl, ss);
    if (probs) {
      slProbability.value = probs.sl;
      ssProbability.value = probs.ss;
      return true;
    }
  }
  let updated = false;
  const slNorm = normalizeProbLike(sl);
  const ssNorm = normalizeProbLike(ss);
  if (slNorm !== null) {
    slProbability.value = slNorm;
    updated = true;
  } else if (slProvided && (sl === null || /^(none|null|nan)$/i.test(String(sl).trim()))) {
    slProbability.value = 0;
    updated = true;
  }
  if (ssNorm !== null) {
    ssProbability.value = ssNorm;
    updated = true;
  } else if (ssProvided && (ss === null || /^(none|null|nan)$/i.test(String(ss).trim()))) {
    ssProbability.value = 0;
    updated = true;
  }
  return updated;
};

const extractRealtimeProbabilitiesFromPayload = (payload) => {
  if (!payload || typeof payload !== 'object') return null;
  const directCandidates = [
    { sl: payload.sl_prob, ss: payload.ss_prob },
    { sl: payload.s_l_prob, ss: payload.s_s_prob },
    { sl: payload.prob_sl, ss: payload.prob_ss },
    { sl: payload.probability_sl, ss: payload.probability_ss },
    { sl: payload?.state_probs?.sl, ss: payload?.state_probs?.ss },
    { sl: payload?.control_probs?.sl, ss: payload?.control_probs?.ss }
  ];
  for (const item of directCandidates) {
    if (item.sl !== undefined || item.ss !== undefined) {
      return {
        sl: item.sl,
        ss: item.ss,
        treatAsLogits: false,
        slProvided: item.sl !== undefined,
        ssProvided: item.ss !== undefined
      };
    }
  }
  if (payload.control_tokens && typeof payload.control_tokens === 'object') {
    const slLogit = payload.control_tokens.sl;
    const ssLogit = payload.control_tokens.ss;
    if (slLogit !== undefined || ssLogit !== undefined) {
      return { sl: slLogit, ss: ssLogit, treatAsLogits: true };
    }
  }
  return null;
};

const extractRealtimeProbabilitiesFromText = (text) => {
  if (typeof text !== 'string' || !text) return null;
  const readOne = (patterns) => {
    for (const pattern of patterns) {
      const m = text.match(pattern);
      if (m && m[1] !== undefined) {
        return {
          present: true,
          raw: String(m[1]).trim()
        };
      }
    }
    return {
      present: false,
      raw: null
    };
  };
  const normalizeRaw = (raw) => {
    if (raw === null || raw === undefined) return null;
    if (/^(none|null|nan)$/i.test(String(raw).trim())) {
      return 0;
    }
    const val = toFiniteNumber(raw);
    return val === null ? null : val;
  };
  const sl = readOne([
    /\bS[-_\s]?L\b\s*[:=]\s*([^,\s]+)/i,
    /\bSL\b\s*[:=]\s*([^,\s]+)/i
  ]);
  const ss = readOne([
    /\bS[-_\s]?S\b\s*[:=]\s*([^,\s]+)/i,
    /\bSS\b\s*[:=]\s*([^,\s]+)/i
  ]);
  if (!sl.present && !ss.present) return null;
  return {
    sl: normalizeRaw(sl.raw),
    ss: normalizeRaw(ss.raw),
    treatAsLogits: false,
    slProvided: sl.present,
    ssProvided: ss.present
  };
};

const returnToHome = () => {
  resetRealtimeReplyState();
  connectionStatus.value = '';
  connectionStatusClass.value = '';
  resetRealtimeProbabilities();
  clearStartTalkError({ clearStorage: true });
  clearAlignedHistory();
};

const formatSessionTime = (tsMs) => {
  const dt = new Date(tsMs);
  const pad2 = (v) => String(v).padStart(2, '0');
  return `${pad2(dt.getHours())}:${pad2(dt.getMinutes())}:${pad2(dt.getSeconds())}`;
};

const padNum = (v, width = 2) => String(v).padStart(width, '0');

const formatTraceFilenameTimestamp = (tsMs) => {
  const dt = new Date(tsMs);
  return `${dt.getFullYear()}${padNum(dt.getMonth() + 1)}${padNum(dt.getDate())}_${padNum(dt.getHours())}${padNum(dt.getMinutes())}${padNum(dt.getSeconds())}_${padNum(dt.getMilliseconds(), 3)}`;
};

const sanitizeSessionIdForFilename = (sessionId = '') => {
  const cleaned = String(sessionId)
    .replace(/[^0-9A-Za-z_-]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return cleaned ? cleaned.slice(0, 64) : 'nosession';
};

const triggerJsonDownload = (payload, filename) => {
  try {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.style.display = 'none';
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  } catch (err) {
    console.warn('导出实时对齐日志失败:', err);
  }
};

const revokeAlignedUrlsForSession = (session) => {
  if (!session) return;
  try {
    URL.revokeObjectURL(session.inputUrl);
    URL.revokeObjectURL(session.outputUrl);
    if (session.rawOutputUrl) {
      URL.revokeObjectURL(session.rawOutputUrl);
    }
  } catch (err) {
    console.warn('释放对齐音频 URL 失败:', err);
  }
};

const clearAlignedHistory = () => {
  for (const item of alignedSessionHistory.value) {
    revokeAlignedUrlsForSession(item);
  }
  alignedSessionHistory.value = [];
};

const syncQueueDepth = () => {
  prioritySegmentQueueDepth.value = priorityUploadQueue.length;
  segmentQueueDepth.value = priorityUploadQueue.length + uploadQueue.length;
};

const concatFloat32 = (chunks, totalLen) => {
  const out = new Float32Array(totalLen);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
};

const resampleLinear = (input, srcRate, dstRate) => {
  if (srcRate === dstRate) {
    return input;
  }
  const ratio = srcRate / dstRate;
  const outLen = Math.max(1, Math.round(input.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i += 1) {
    const srcPos = i * ratio;
    const left = Math.floor(srcPos);
    const right = Math.min(left + 1, input.length - 1);
    const frac = srcPos - left;
    out[i] = input[left] * (1 - frac) + input[right] * frac;
  }
  return out;
};

const resampleLinearToLength = (input, outLen) => {
  const targetLen = Math.max(1, Math.floor(Number(outLen) || 0));
  if (!input || input.length === 0) {
    return new Float32Array(0);
  }
  if (input.length === targetLen) {
    return input;
  }
  if (targetLen === 1) {
    return new Float32Array([input[0]]);
  }
  const ratio = input.length / targetLen;
  const out = new Float32Array(targetLen);
  for (let i = 0; i < targetLen; i += 1) {
    const srcPos = i * ratio;
    const left = Math.floor(srcPos);
    const right = Math.min(left + 1, input.length - 1);
    const frac = srcPos - left;
    out[i] = input[left] * (1 - frac) + input[right] * frac;
  }
  return out;
};

const takePendingCaptureSamples = (sampleCount) => {
  const target = Math.max(0, Math.floor(Number(sampleCount) || 0));
  const out = new Float32Array(target);
  let offset = 0;
  let remaining = target;

  while (remaining > 0 && pendingCaptureChunks.length > 0) {
    const head = pendingCaptureChunks[0];
    if (!head || head.length === 0) {
      pendingCaptureChunks.shift();
      continue;
    }
    if (head.length <= remaining) {
      out.set(head, offset);
      offset += head.length;
      remaining -= head.length;
      pendingCaptureChunks.shift();
    } else {
      out.set(head.subarray(0, remaining), offset);
      pendingCaptureChunks[0] = head.slice(remaining);
      offset += remaining;
      remaining = 0;
    }
  }

  const taken = target - remaining;
  pendingCaptureSamples = Math.max(0, pendingCaptureSamples - taken);
  return taken === target ? out : out.slice(0, taken);
};

const float32ToInt16 = (samples) => {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    const s = clamp(samples[i], -1, 1);
    out[i] = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
  }
  return out;
};

const wavBlobFromFloat32 = (samples, sampleRate) => {
  const pcm16 = float32ToInt16(samples);
  const dataSize = pcm16.length * 2;
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);

  const writeStr = (off, str) => {
    for (let i = 0; i < str.length; i += 1) {
      view.setUint8(off + i, str.charCodeAt(i));
    }
  };

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < pcm16.length; i += 1) {
    view.setInt16(offset, pcm16[i], true);
    offset += 2;
  }
  return new Blob([buf], { type: 'audio/wav' });
};

const createCurrentAlignedArchive = (sessionId = '') => {
  const startEpochMs = Date.now();
  return {
    sessionId,
    startEpochMs,
    startPerfMs: performance.now(),
    inputChunks: [],
    outputChunks: [],
    rawOutputChunks: [],
    inputSamples: 0,
    outputSamples: 0,
    rawOutputSamples: 0,
    finalized: false,
    outputTraceEvents: [],
    outputTraceSummary: null
  };
};

const exportAlignedArchiveTrace = (archive) => {
  if (!archive) {
    return;
  }
  const exportedAtMs = Date.now();
  const payload = {
    kind: 'realtime_audio_alignment_trace',
    version: 1,
    sessionId: archive.sessionId || '',
    startEpochMs: archive.startEpochMs,
    startIso: new Date(archive.startEpochMs).toISOString(),
    exportedAtEpochMs: exportedAtMs,
    exportedAtIso: new Date(exportedAtMs).toISOString(),
    targetSampleRate: TARGET_SAMPLE_RATE,
    streamChunkMs,
    totals: {
      inputSamples: archive.inputSamples,
      outputSamples: archive.outputSamples,
      rawOutputSamples: archive.rawOutputSamples || 0,
      inputMs: Number(((archive.inputSamples / TARGET_SAMPLE_RATE) * 1000).toFixed(3)),
      outputMs: Number(((archive.outputSamples / TARGET_SAMPLE_RATE) * 1000).toFixed(3)),
      rawOutputMs: Number((((archive.rawOutputSamples || 0) / TARGET_SAMPLE_RATE) * 1000).toFixed(3))
    },
    finalize: archive.outputTraceSummary || null,
    outputChunkTrace: archive.outputTraceEvents
  };
  const tsPart = formatTraceFilenameTimestamp(exportedAtMs);
  const sessionPart = sanitizeSessionIdForFilename(archive.sessionId || '');
  const filename = `realtime_audio_align_trace_${tsPart}_${sessionPart}.json`;
  triggerJsonDownload(payload, filename);
  console.info(`[RealtimeAlignTrace] exported ${filename}`);
};

const startCurrentAlignedArchive = (sessionId = '') => {
  currentAlignedArchive = createCurrentAlignedArchive(sessionId);
};

const recordRealtimeInputSamples = (samples) => {
  if (!currentAlignedArchive || !samples || samples.length === 0) {
    return;
  }
  const chunk = new Float32Array(samples);
  currentAlignedArchive.inputChunks.push(chunk);
  currentAlignedArchive.inputSamples += chunk.length;
};

const decodePcm16WavBase64 = (wavB64) => {
  const byteChars = atob(wavB64);
  const byteArray = new Uint8Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i += 1) {
    byteArray[i] = byteChars.charCodeAt(i);
  }
  if (byteArray.length < 44) {
    throw new Error('WAV 数据长度不足');
  }
  const view = new DataView(byteArray.buffer, byteArray.byteOffset, byteArray.byteLength);
  const readTag = (offset) => String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3)
  );
  if (readTag(0) !== 'RIFF' || readTag(8) !== 'WAVE') {
    throw new Error('不是标准 WAV');
  }

  let fmtOffset = -1;
  let fmtSize = 0;
  let dataOffset = -1;
  let dataSize = 0;
  let offset = 12;
  while (offset + 8 <= view.byteLength) {
    const tag = readTag(offset);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (tag === 'fmt ') {
      fmtOffset = body;
      fmtSize = size;
    } else if (tag === 'data') {
      dataOffset = body;
      dataSize = size;
      break;
    }
    offset = body + size + (size % 2);
  }
  if (fmtOffset < 0 || dataOffset < 0 || fmtSize < 16) {
    throw new Error('WAV chunk 不完整');
  }
  const audioFormat = view.getUint16(fmtOffset, true);
  const channels = view.getUint16(fmtOffset + 2, true);
  const sampleRate = view.getUint32(fmtOffset + 4, true);
  const bitsPerSample = view.getUint16(fmtOffset + 14, true);
  if (audioFormat !== 1 || bitsPerSample !== 16) {
    throw new Error('仅支持 PCM16');
  }

  const frameCount = Math.floor(dataSize / 2 / Math.max(1, channels));
  const mono = new Float32Array(frameCount);
  let ptr = dataOffset;
  for (let i = 0; i < frameCount; i += 1) {
    let sum = 0;
    for (let ch = 0; ch < channels; ch += 1) {
      const s = view.getInt16(ptr, true);
      ptr += 2;
      sum += s < 0 ? s / 32768 : s / 32767;
    }
    mono[i] = sum / Math.max(1, channels);
  }
  return { samples: mono, sampleRate };
};

const decodePcm16Base64 = (pcmB64) => {
  const byteChars = atob(pcmB64);
  const byteLen = byteChars.length - (byteChars.length % 2);
  if (byteLen <= 0) {
    return new Float32Array(0);
  }
  const byteArray = new Uint8Array(byteLen);
  for (let i = 0; i < byteLen; i += 1) {
    byteArray[i] = byteChars.charCodeAt(i);
  }
  const view = new DataView(byteArray.buffer, byteArray.byteOffset, byteArray.byteLength);
  const frameCount = Math.floor(byteLen / 2);
  const mono = new Float32Array(frameCount);
  for (let i = 0, offset = 0; i < frameCount; i += 1, offset += 2) {
    const s = view.getInt16(offset, true);
    mono[i] = s < 0 ? s / 32768 : s / 32767;
  }
  return mono;
};

const recordRealtimeOutputSamples = (samples, sampleRate = TARGET_SAMPLE_RATE) => {
  if (!currentAlignedArchive || !samples || samples.length === 0) {
    return;
  }
  const sourceSamples = samples.length;
  const normalized = sampleRate === TARGET_SAMPLE_RATE
    ? samples
    : resampleLinear(samples, sampleRate, TARGET_SAMPLE_RATE);
  const arrivalPerfMs = performance.now();
  const arrivalEpochMs = Date.now();
  const elapsedMs = Math.max(0, arrivalPerfMs - currentAlignedArchive.startPerfMs);
  const elapsedSamples = Math.max(
    0,
    Math.round((elapsedMs / 1000) * TARGET_SAMPLE_RATE)
  );
  const outputSamplesBefore = currentAlignedArchive.outputSamples;
  let silenceLen = 0;
  const rawChunk = new Float32Array(normalized);
  currentAlignedArchive.rawOutputChunks.push(rawChunk);
  currentAlignedArchive.rawOutputSamples += rawChunk.length;
  if (elapsedSamples > currentAlignedArchive.outputSamples) {
    silenceLen = elapsedSamples - currentAlignedArchive.outputSamples;
    currentAlignedArchive.outputChunks.push(new Float32Array(silenceLen));
    currentAlignedArchive.outputSamples += silenceLen;
  }
  const chunk = new Float32Array(normalized);
  currentAlignedArchive.outputChunks.push(chunk);
  currentAlignedArchive.outputSamples += chunk.length;
  currentAlignedArchive.outputTraceEvents.push({
    idx: currentAlignedArchive.outputTraceEvents.length + 1,
    arrivalEpochMs,
    arrivalIso: new Date(arrivalEpochMs).toISOString(),
    elapsedMs: Number(elapsedMs.toFixed(3)),
    sourceSampleRate: sampleRate,
    sourceSamples,
    normalizedSamples: chunk.length,
    chunkMs: Number(((chunk.length / TARGET_SAMPLE_RATE) * 1000).toFixed(3)),
    rawOutputSamplesBefore: currentAlignedArchive.rawOutputSamples - rawChunk.length,
    rawOutputSamplesAfter: currentAlignedArchive.rawOutputSamples,
    elapsedSamples,
    outputSamplesBefore,
    silenceFilledSamples: silenceLen,
    silenceFilledMs: Number(((silenceLen / TARGET_SAMPLE_RATE) * 1000).toFixed(3)),
    outputSamplesAfter: currentAlignedArchive.outputSamples
  });
};

const finalizeCurrentAlignedArchive = () => {
  if (!currentAlignedArchive || currentAlignedArchive.finalized) {
    return;
  }
  currentAlignedArchive.finalized = true;

  if (currentAlignedArchive.inputSamples <= 0 && currentAlignedArchive.outputSamples <= 0) {
    currentAlignedArchive.outputTraceSummary = {
      inputTailPadSamples: 0,
      outputTailPadSamples: 0,
      alignedMaxSamples: 0
    };
    exportAlignedArchiveTrace(currentAlignedArchive);
    currentAlignedArchive = null;
    return;
  }

  let inputTailPadSamples = 0;
  let outputTailPadSamples = 0;
  const maxLen = Math.max(currentAlignedArchive.inputSamples, currentAlignedArchive.outputSamples);
  if (maxLen > currentAlignedArchive.inputSamples) {
    inputTailPadSamples = maxLen - currentAlignedArchive.inputSamples;
    currentAlignedArchive.inputChunks.push(new Float32Array(inputTailPadSamples));
    currentAlignedArchive.inputSamples = maxLen;
  }
  if (maxLen > currentAlignedArchive.outputSamples) {
    outputTailPadSamples = maxLen - currentAlignedArchive.outputSamples;
    currentAlignedArchive.outputChunks.push(new Float32Array(outputTailPadSamples));
    currentAlignedArchive.outputSamples = maxLen;
  }

  const inputSamples = concatFloat32(currentAlignedArchive.inputChunks, currentAlignedArchive.inputSamples);
  const outputSamples = concatFloat32(currentAlignedArchive.outputChunks, currentAlignedArchive.outputSamples);
  const rawOutputSamples = concatFloat32(
    currentAlignedArchive.rawOutputChunks || [],
    currentAlignedArchive.rawOutputSamples || 0
  );
  const inputBlob = wavBlobFromFloat32(inputSamples, TARGET_SAMPLE_RATE);
  const outputBlob = wavBlobFromFloat32(outputSamples, TARGET_SAMPLE_RATE);
  const rawOutputBlob = wavBlobFromFloat32(rawOutputSamples, TARGET_SAMPLE_RATE);
  const inputUrl = URL.createObjectURL(inputBlob);
  const outputUrl = URL.createObjectURL(outputBlob);
  const rawOutputUrl = URL.createObjectURL(rawOutputBlob);

  alignedSessionHistory.value.unshift({
    id: `${currentAlignedArchive.startEpochMs}_${Math.random().toString(16).slice(2, 10)}`,
    sessionId: currentAlignedArchive.sessionId || '',
    startedAt: formatSessionTime(currentAlignedArchive.startEpochMs),
    sampleRate: TARGET_SAMPLE_RATE,
    inputSec: (currentAlignedArchive.inputSamples / TARGET_SAMPLE_RATE).toFixed(2),
    outputSec: (currentAlignedArchive.outputSamples / TARGET_SAMPLE_RATE).toFixed(2),
    rawOutputSec: ((currentAlignedArchive.rawOutputSamples || 0) / TARGET_SAMPLE_RATE).toFixed(2),
    inputUrl,
    outputUrl,
    rawOutputUrl
  });

  currentAlignedArchive.outputTraceSummary = {
    inputTailPadSamples,
    inputTailPadMs: Number(((inputTailPadSamples / TARGET_SAMPLE_RATE) * 1000).toFixed(3)),
    outputTailPadSamples,
    outputTailPadMs: Number(((outputTailPadSamples / TARGET_SAMPLE_RATE) * 1000).toFixed(3)),
    alignedMaxSamples: maxLen,
    alignedMaxMs: Number(((maxLen / TARGET_SAMPLE_RATE) * 1000).toFixed(3))
  };
  exportAlignedArchiveTrace(currentAlignedArchive);
  currentAlignedArchive = null;
};

const decodeAudioFileToMonoFloat32 = async (file) => {
  const arrayBuffer = await file.arrayBuffer();
  const decodeContext = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer.slice(0));
    const channelCount = Math.max(1, audioBuffer.numberOfChannels || 1);
    const totalSamples = audioBuffer.length || 0;
    if (totalSamples <= 0) {
      return {
        samples: new Float32Array(0),
        sampleRate: audioBuffer.sampleRate || TARGET_SAMPLE_RATE
      };
    }
    if (channelCount === 1) {
      return {
        samples: new Float32Array(audioBuffer.getChannelData(0)),
        sampleRate: audioBuffer.sampleRate || TARGET_SAMPLE_RATE
      };
    }
    const mixed = new Float32Array(totalSamples);
    for (let ch = 0; ch < channelCount; ch += 1) {
      const channelData = audioBuffer.getChannelData(ch);
      for (let i = 0; i < totalSamples; i += 1) {
        mixed[i] += channelData[i];
      }
    }
    const scale = 1 / channelCount;
    for (let i = 0; i < totalSamples; i += 1) {
      mixed[i] *= scale;
    }
    return {
      samples: mixed,
      sampleRate: audioBuffer.sampleRate || TARGET_SAMPLE_RATE
    };
  } finally {
    decodeContext.close().catch(() => {});
  }
};

const splitSamplesToRealtimeSegments = (samples, sampleRate, chunkMs) => {
  const out = [];
  if (!samples || samples.length === 0) {
    return out;
  }
  const segmentSamples = Math.max(
    MIN_SEGMENT_SAMPLES,
    Math.floor((sampleRate * chunkMs) / 1000)
  );
  for (let start = 0; start < samples.length; start += segmentSamples) {
    const end = Math.min(start + segmentSamples, samples.length);
    const piece = samples.slice(start, end);
    if (piece.length < MIN_SEGMENT_SAMPLES && out.length > 0) {
      const last = out.pop();
      const merged = new Float32Array(last.length + piece.length);
      merged.set(last, 0);
      merged.set(piece, last.length);
      out.push(merged);
      continue;
    }
    out.push(piece);
  }
  if (out.length === 1 && out[0].length < MIN_SEGMENT_SAMPLES) {
    return [];
  }
  return out;
};

const enqueuePriorityAudioFile = async (file) => {
  if (!isTalking.value || !realtimeSessionId.value) {
    throw new Error('请先开始通话后再发送测试音频');
  }
  const decoded = await decodeAudioFileToMonoFloat32(file);
  const inputSamples = decoded.samples;
  if (!inputSamples || inputSamples.length === 0) {
    throw new Error('音频文件为空或解码失败');
  }
  const srcRate = decoded.sampleRate || TARGET_SAMPLE_RATE;
  const samples16k = resampleLinear(inputSamples, srcRate, TARGET_SAMPLE_RATE);
  const segments = splitSamplesToRealtimeSegments(samples16k, TARGET_SAMPLE_RATE, streamChunkMs);
  if (segments.length === 0) {
    throw new Error('音频过短，无法切分为可发送分片');
  }
  const fileSegments = segments.map((samples) => ({
    id: ++segmentSeqId,
    samples,
    source: 'file'
  }));
  priorityUploadQueue = priorityUploadQueue.concat(fileSegments);
  priorityUploadInProgress.value = true;
  priorityUploadSourceName.value = file.name || '';
  syncQueueDepth();
  processUploadQueue().catch((err) => {
    console.error('处理优先音频队列失败:', err);
  });
  return fileSegments.length;
};

const buildGradioApiBase = () => {
  const explicitBase = window.__UNIMOE_GRADIO_API_BASE__;
  if (typeof explicitBase === 'string' && explicitBase.trim()) {
    return explicitBase.replace(/\/+$/, '');
  }
  const configuredPort = window.__UNIMOE_GRADIO_PORT__;
  if (!configuredPort) {
    const host = window.location.hostname || 'localhost';
    if (window.location.port === '7860') {
      return '';
    }
    return `${window.location.protocol}//${host}:7860`;
  }
  const host = window.__UNIMOE_GRADIO_HOST__ || window.location.hostname || 'localhost';
  return `${window.location.protocol}//${host}:${configuredPort}`;
};

const buildGradioUploadUrl = () => `${buildGradioApiBase()}/gradio_api/upload`;
const buildGradioCallPostUrl = () => `${buildGradioApiBase()}/gradio_api/call/run_chunk_dialogue_inference`;
const buildGradioCallStreamUrl = (eventId) => `${buildGradioApiBase()}/gradio_api/call/run_chunk_dialogue_inference/${eventId}`;
const buildRealtimeSessionStartUrl = () => `${buildGradioApiBase()}/api/realtime/session/start`;
const buildRealtimeSessionChunkUrl = (sessionId) => `${buildGradioApiBase()}/api/realtime/session/${encodeURIComponent(sessionId)}/chunk`;
const buildRealtimeSessionEventsUrl = (sessionId) => `${buildGradioApiBase()}/api/realtime/session/${encodeURIComponent(sessionId)}/events`;
const buildRealtimeSessionStopUrl = (sessionId) => `${buildGradioApiBase()}/api/realtime/session/${encodeURIComponent(sessionId)}/stop`;

const fetchWithRuntimeContext = async (url, options = {}, stage = '请求') => {
  try {
    return await fetch(url, options);
  } catch (err) {
    const message = extractReadableErrorMessage(err);
    let pageOrigin = '';
    let pageProtocol = '';
    try {
      pageOrigin = window?.location?.origin || '';
      pageProtocol = window?.location?.protocol || '';
    } catch (_err) {
      pageOrigin = '';
      pageProtocol = '';
    }
    const apiBase = buildGradioApiBase() || pageOrigin || '';
    let extraHint = '请检查端口可达性、浏览器代理绕过、以及 CORS/同源策略。';
    if (pageProtocol === 'https:' && String(url).startsWith('http://')) {
      extraHint = '当前页面是 HTTPS，但请求是 HTTP，浏览器可能拦截混合内容。';
    }
    throw new Error(
      `${stage}网络请求失败: ${message}; url=${url}; page_origin=${pageOrigin}; api_base=${apiBase}; hint=${extraHint}`
    );
  }
};

const resolveGradioFileUrl = (fileInfo) => {
  if (!fileInfo) return '';
  const base = buildGradioApiBase() || window.location.origin;
  if (fileInfo.url) {
    try {
      return new URL(fileInfo.url, base).toString();
    } catch (err) {
      return fileInfo.url;
    }
  }
  if (fileInfo.path) {
    return `${base}/gradio_api/file=${encodeURIComponent(fileInfo.path)}`;
  }
  return '';
};

const realtimePlaybackWorkletCode = `
class RealtimePCMPlaybackWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sampleRateHz = sampleRate;
    this.bufferSize = Math.max(2048, Math.floor(this.sampleRateHz * ${REALTIME_PLAYBACK_SMOOTHING.ringBufferSec}));
    this.ringBuffer = new Float32Array(this.bufferSize);
    this.writePtr = 0;
    this.readPtr = 0;
    this.isPlaying = false;
    this.lastOut = 0.0;
    this.fadeVolume = 0.0;
    this.smoothedLevel = 0.0;
    this.autoRate = 1.0;
    this.manualRate = 1.0;
    this.tickCount = 0;
    this.startThreshold = Math.floor(this.sampleRateHz * ${REALTIME_PLAYBACK_SMOOTHING.startWarmupSec});
    this.lowWater = Math.floor(this.sampleRateHz * ${REALTIME_PLAYBACK_SMOOTHING.lowWaterSec});
    this.highWater = Math.floor(this.sampleRateHz * ${REALTIME_PLAYBACK_SMOOTHING.highWaterSec});
    this.port.onmessage = (e) => {
      const data = e.data;
      if (!data) return;
      if (data.type === 'set_manual_rate') {
        const v = Number(data.value);
        if (Number.isFinite(v)) {
          this.manualRate = Math.max(0.5, Math.min(1.8, v));
        }
        return;
      }
      if (data.type === 'flush') {
        this.writePtr = 0;
        this.readPtr = 0;
        this.isPlaying = false;
        this.lastOut = 0.0;
        this.fadeVolume = 0.0;
        this.smoothedLevel = 0.0;
        this.autoRate = 1.0;
        this._postMetrics();
        return;
      }
      if (data.type === 'metrics') {
        this._postMetrics();
        return;
      }
      if (data instanceof Float32Array) {
        for (let i = 0; i < data.length; i++) {
          this.ringBuffer[this.writePtr] = data[i];
          this.writePtr = (this.writePtr + 1) % this.bufferSize;
          if (this.writePtr === Math.floor(this.readPtr)) {
            this.readPtr = (this.readPtr + 1) % this.bufferSize;
          }
        }
      }
    };
  }
  _availableFrames() {
    let available = this.writePtr - this.readPtr;
    if (available < 0) available += this.bufferSize;
    return available;
  }
  _postMetrics() {
    this.port.postMessage({
      type: 'metrics',
      bufferLevelFrames: this._availableFrames(),
      isPlaying: this.isPlaying,
      autoRate: this.autoRate,
      manualRate: this.manualRate,
      sampleRate: this.sampleRateHz
    });
  }
  process(_inputs, outputs) {
    const output = outputs[0];
    const channel = output[0];
    let availableFrames = this._availableFrames();
    if (this.smoothedLevel === 0) {
      this.smoothedLevel = availableFrames;
    } else {
      this.smoothedLevel = this.smoothedLevel * ${REALTIME_PLAYBACK_SMOOTHING.levelSmoothKeep} + availableFrames * ${REALTIME_PLAYBACK_SMOOTHING.levelSmoothUpdate};
    }
    if (!this.isPlaying && availableFrames >= this.startThreshold) {
      this.isPlaying = true;
    }
    let targetAutoRate = 1.0;
    if (this.isPlaying) {
      if (this.smoothedLevel < this.lowWater) {
        targetAutoRate = ${REALTIME_PLAYBACK_SMOOTHING.lowWaterRate};
      } else if (this.smoothedLevel > this.highWater) {
        targetAutoRate = ${REALTIME_PLAYBACK_SMOOTHING.highWaterRate};
      }
    }
    this.autoRate = this.autoRate * ${REALTIME_PLAYBACK_SMOOTHING.autoRateSmoothKeep} + targetAutoRate * ${REALTIME_PLAYBACK_SMOOTHING.autoRateSmoothUpdate};
    const speed = Math.max(0.5, Math.min(1.8, this.manualRate * this.autoRate));
    for (let i = 0; i < channel.length; i++) {
      if (this.isPlaying && availableFrames >= speed) {
        if (this.fadeVolume < 1.0) {
          this.fadeVolume = Math.min(1.0, this.fadeVolume + 0.0025);
        }
        const readIdx = Math.floor(this.readPtr);
        const nextIdx = (readIdx + 1) % this.bufferSize;
        const frac = this.readPtr - readIdx;
        const rawSample = this.ringBuffer[readIdx] + (this.ringBuffer[nextIdx] - this.ringBuffer[readIdx]) * frac;
        channel[i] = rawSample * this.fadeVolume;
        this.lastOut = channel[i];
        this.readPtr = (this.readPtr + speed) % this.bufferSize;
        availableFrames -= speed;
      } else {
        if (this.isPlaying) {
          this.isPlaying = false;
          this.fadeVolume = 0.0;
        }
        this.lastOut *= 0.9;
        channel[i] = this.lastOut;
      }
    }
    this.tickCount += 1;
    if ((this.tickCount % 10) === 0) {
      this._postMetrics();
    }
    return true;
  }
}
registerProcessor('realtime-pcm-playback-worklet', RealtimePCMPlaybackWorklet);
`;

const getRealtimePlaybackMetricsSnapshot = (sampleRate) => {
  const sr = Math.max(1, Number(sampleRate) || TARGET_SAMPLE_RATE);
  const bufferFrames = Math.max(0, Number(realtimePlaybackBufferLevelFrames) || 0);
  const manualRate = clamp(Number(realtimePlaybackManualRate) || 1.0, 0.5, 1.8);
  const autoRate = clamp(Number(realtimePlaybackAutoRate) || 1.0, 0.8, 1.2);
  const effectiveRate = clamp(manualRate * autoRate, 0.5, 1.8);
  return {
    sampleRate: sr,
    bufferFrames,
    isPlaying: !!realtimePlaybackIsPlaying,
    manualRate,
    autoRate,
    effectiveRate,
    startThresholdFrames: Math.floor(sr * REALTIME_PLAYBACK_START_WARMUP_SEC)
  };
};

const ensureRealtimePlaybackContext = async () => {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) {
    throw new Error('当前浏览器不支持 AudioContext');
  }
  if (!realtimePlaybackContext || realtimePlaybackContext.state === 'closed') {
    realtimePlaybackContext = new AudioCtx({ latencyHint: 'interactive' });
    realtimePlaybackGainNode = realtimePlaybackContext.createGain();
    realtimePlaybackGainNode.gain.value = 1.0;
    realtimePlaybackGainNode.connect(realtimePlaybackContext.destination);
    realtimePlaybackSupportsWorklet = !!(
      realtimePlaybackContext.audioWorklet && typeof AudioWorkletNode !== 'undefined'
    );
    realtimePlaybackBufferLevelFrames = 0;
    realtimePlaybackIsPlaying = false;
    realtimePlaybackAutoRate = 1.0;
    realtimePlaybackManualRate = clamp(Number(playbackRate.value) || 1.0, 0.5, 1.8);
    realtimePlaybackTimelineStart = 0;
    realtimePlaybackScheduledSec = 0;
    realtimePlaybackLastSource = null;
  }
  if (realtimePlaybackContext.state === 'suspended') {
    await realtimePlaybackContext.resume();
  }
  if (realtimePlaybackSupportsWorklet && !realtimePlaybackWorkletNode) {
    try {
      if (!realtimePlaybackWorkletBlobUrl) {
        const blob = new Blob([realtimePlaybackWorkletCode], { type: 'application/javascript' });
        realtimePlaybackWorkletBlobUrl = URL.createObjectURL(blob);
      }
      await realtimePlaybackContext.audioWorklet.addModule(realtimePlaybackWorkletBlobUrl);
      realtimePlaybackWorkletNode = new AudioWorkletNode(
        realtimePlaybackContext,
        'realtime-pcm-playback-worklet'
      );
      realtimePlaybackWorkletNode.connect(realtimePlaybackGainNode);
      realtimePlaybackWorkletNode.port.onmessage = (e) => {
        const payload = e.data || {};
        if (payload.type !== 'metrics') return;
        const frames = Number(payload.bufferLevelFrames);
        if (Number.isFinite(frames)) {
          realtimePlaybackBufferLevelFrames = Math.max(0, frames);
        }
        realtimePlaybackIsPlaying = !!payload.isPlaying;
        const autoRate = Number(payload.autoRate);
        if (Number.isFinite(autoRate)) {
          realtimePlaybackAutoRate = clamp(autoRate, 0.8, 1.2);
        }
      };
      realtimePlaybackWorkletNode.port.postMessage({
        type: 'set_manual_rate',
        value: realtimePlaybackManualRate
      });
      realtimePlaybackWorkletNode.port.postMessage({ type: 'metrics' });
    } catch (err) {
      console.warn('AudioWorklet 初始化失败，回退到时间轴调度播放:', err);
      realtimePlaybackSupportsWorklet = false;
      if (realtimePlaybackWorkletNode) {
        try {
          realtimePlaybackWorkletNode.disconnect();
        } catch (_err) {
          // ignore
        }
        realtimePlaybackWorkletNode = null;
      }
      if (realtimePlaybackWorkletBlobUrl) {
        URL.revokeObjectURL(realtimePlaybackWorkletBlobUrl);
        realtimePlaybackWorkletBlobUrl = null;
      }
    }
  }
  return realtimePlaybackContext;
};

const resetRealtimePlaybackScheduler = async (closeContext = false) => {
  if (realtimePlaybackWorkletNode) {
    try {
      realtimePlaybackWorkletNode.port.postMessage({ type: 'flush' });
    } catch (_err) {
      // ignore
    }
    try {
      realtimePlaybackWorkletNode.disconnect();
    } catch (_err) {
      // ignore
    }
    realtimePlaybackWorkletNode = null;
  }
  if (realtimePlaybackWorkletBlobUrl) {
    URL.revokeObjectURL(realtimePlaybackWorkletBlobUrl);
    realtimePlaybackWorkletBlobUrl = null;
  }
  if (realtimePlaybackLastSource) {
    try {
      realtimePlaybackLastSource.stop();
    } catch (_err) {
      // ignore
    }
    try {
      realtimePlaybackLastSource.disconnect();
    } catch (_err) {
      // ignore
    }
    realtimePlaybackLastSource = null;
  }
  realtimePlaybackBufferLevelFrames = 0;
  realtimePlaybackIsPlaying = false;
  realtimePlaybackAutoRate = 1.0;
  realtimePlaybackManualRate = clamp(Number(playbackRate.value) || 1.0, 0.5, 1.8);
  realtimePlaybackTimelineStart = 0;
  realtimePlaybackScheduledSec = 0;
  if (!closeContext || !realtimePlaybackContext) {
    return;
  }
  try {
    if (realtimePlaybackGainNode) {
      realtimePlaybackGainNode.disconnect();
    }
  } catch (_err) {
    // ignore
  }
  realtimePlaybackGainNode = null;
  try {
    await realtimePlaybackContext.close();
  } catch (_err) {
    // ignore
  }
  realtimePlaybackContext = null;
  realtimePlaybackSupportsWorklet = false;
};

const scheduleRealtimePcmSamples = async (samples, sampleRate) => {
  if (!samples || samples.length === 0) {
    return { startLeadMs: null, backlogMs: null };
  }
  const ctx = await ensureRealtimePlaybackContext();
  const rate = clamp(Number(playbackRate.value) || 1.0, 0.5, 1.8);
  realtimePlaybackManualRate = rate;
  const normalized = samples instanceof Float32Array ? samples : new Float32Array(samples);
  const targetRate = Math.max(1, Number(sampleRate) || TARGET_SAMPLE_RATE);
  if (realtimePlaybackWorkletNode) {
    try {
      realtimePlaybackWorkletNode.port.postMessage({
        type: 'set_manual_rate',
        value: realtimePlaybackManualRate
      });
      let chunk = normalized;
      const workletRate = Math.max(1, Number(ctx.sampleRate) || TARGET_SAMPLE_RATE);
      if (targetRate !== workletRate) {
        chunk = resampleLinear(chunk, targetRate, workletRate);
      }
      if (!chunk || chunk.length === 0) {
        return { startLeadMs: null, backlogMs: null };
      }
      const snapshot = getRealtimePlaybackMetricsSnapshot(workletRate);
      const expectedFrames = snapshot.bufferFrames + chunk.length;
      const startLeadMs = snapshot.isPlaying
        ? 0
        : Math.max(0, ((snapshot.startThresholdFrames - snapshot.bufferFrames) / workletRate) * 1000);
      const backlogMs = Math.max(0, (expectedFrames / (workletRate * snapshot.effectiveRate)) * 1000);
      realtimePlaybackWorkletNode.port.postMessage(chunk, [chunk.buffer]);
      realtimePlaybackWorkletNode.port.postMessage({ type: 'metrics' });
      return { startLeadMs, backlogMs };
    } catch (err) {
      console.warn('Worklet 推流失败，回退到时间轴调度播放:', err);
      if (realtimePlaybackWorkletNode) {
        try {
          realtimePlaybackWorkletNode.disconnect();
        } catch (_err) {
          // ignore
        }
        realtimePlaybackWorkletNode = null;
      }
      if (realtimePlaybackWorkletBlobUrl) {
        URL.revokeObjectURL(realtimePlaybackWorkletBlobUrl);
        realtimePlaybackWorkletBlobUrl = null;
      }
      realtimePlaybackSupportsWorklet = false;
    }
  }

  // Fallback: legacy timeline scheduler.
  const buf = ctx.createBuffer(1, normalized.length, targetRate);
  buf.copyToChannel(normalized, 0);
  const source = ctx.createBufferSource();
  source.buffer = buf;
  source.playbackRate.value = rate;
  if (realtimePlaybackGainNode) {
    source.connect(realtimePlaybackGainNode);
  } else {
    source.connect(ctx.destination);
  }

  const now = ctx.currentTime;
  if (realtimePlaybackTimelineStart <= 0) {
    realtimePlaybackTimelineStart = now + REALTIME_PLAYBACK_MIN_LEAD_SEC;
    realtimePlaybackScheduledSec = 0;
  }
  const scheduledEnd = realtimePlaybackTimelineStart + realtimePlaybackScheduledSec;
  if (now - scheduledEnd > REALTIME_PLAYBACK_MAX_LAG_SEC) {
    realtimePlaybackTimelineStart = now + REALTIME_PLAYBACK_MIN_LEAD_SEC;
    realtimePlaybackScheduledSec = 0;
  }
  const backlogMs = Math.max(0, (realtimePlaybackTimelineStart + realtimePlaybackScheduledSec - now) * 1000);
  const startAt = Math.max(
    now + REALTIME_PLAYBACK_MIN_LEAD_SEC,
    realtimePlaybackTimelineStart + realtimePlaybackScheduledSec
  );
  const startLeadMs = Math.max(0, (startAt - now) * 1000);
  source.start(startAt);
  realtimePlaybackScheduledSec = (startAt - realtimePlaybackTimelineStart) + (buf.duration / rate);
  realtimePlaybackLastSource = source;
  source.onended = () => {
    if (realtimePlaybackLastSource === source) {
      realtimePlaybackLastSource = null;
    }
    try {
      source.disconnect();
    } catch (_err) {
      // ignore
    }
  };
  return { startLeadMs, backlogMs };
};

const playNextGradioAudio = async () => {
  if (gradioAudioPlaying || gradioAudioQueue.length === 0) {
    return;
  }
  gradioAudioPlaying = true;
  const url = gradioAudioQueue.shift();
  try {
    const audio = new Audio(url);
    audio.defaultPlaybackRate = playbackRate.value;
    audio.playbackRate = playbackRate.value;
    currentGradioAudio = audio;
    await audio.play();
    audio.onended = () => {
      currentGradioAudio = null;
      gradioAudioPlaying = false;
      playNextGradioAudio().catch(() => {});
    };
    audio.onerror = () => {
      currentGradioAudio = null;
      gradioAudioPlaying = false;
      playNextGradioAudio().catch(() => {});
    };
  } catch (err) {
    currentGradioAudio = null;
    gradioAudioPlaying = false;
    playNextGradioAudio().catch(() => {});
  }
};

watch(playbackRate, (newRate) => {
  const rate = clamp(Number(newRate) || 1.0, 0.5, 1.8);
  playbackRate.value = rate;
  realtimePlaybackManualRate = rate;
  if (currentGradioAudio) {
    currentGradioAudio.playbackRate = rate;
  }
  if (realtimePlaybackWorkletNode) {
    try {
      realtimePlaybackWorkletNode.port.postMessage({
        type: 'set_manual_rate',
        value: rate
      });
      realtimePlaybackWorkletNode.port.postMessage({ type: 'metrics' });
    } catch (_err) {
      // ignore
    }
  }
  if (realtimePlaybackLastSource) {
    try {
      realtimePlaybackLastSource.playbackRate.value = rate;
    } catch (_err) {
      // ignore
    }
  }
});

const enqueueGradioAudio = (fileInfoOrUrl) => {
  const url = typeof fileInfoOrUrl === 'string'
    ? fileInfoOrUrl
    : resolveGradioFileUrl(fileInfoOrUrl);
  if (!url) {
    return;
  }
  if (url === lastQueuedAudioUrl) {
    return;
  }
  lastQueuedAudioUrl = url;
  gradioAudioQueue.push(url);
  playNextGradioAudio().catch(() => {});
};

const extractAssistantTextFromChatbotData = (chatbotData) => {
  if (!Array.isArray(chatbotData)) {
    return '';
  }
  for (let i = chatbotData.length - 1; i >= 0; i -= 1) {
    const item = chatbotData[i];
    if (!item || item.role !== 'assistant') {
      continue;
    }
    const content = item.content;
    if (typeof content === 'string') {
      return content;
    }
    if (Array.isArray(content)) {
      const textList = [];
      for (const part of content) {
        if (part && part.type === 'text' && typeof part.text === 'string') {
          textList.push(part.text);
        }
      }
      return textList.join('\n');
    }
  }
  return '';
};

const consumeGradioSse = async (eventId) => {
  const controller = new AbortController();
  activeRequestControllers.add(controller);

  try {
    const resp = await fetch(buildGradioCallStreamUrl(eventId), {
      method: 'GET',
      signal: controller.signal
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`SSE HTTP ${resp.status}: ${text}`);
    }
    if (!resp.body) {
      throw new Error('SSE 响应缺少 body');
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let pending = '';

    const processSseBlock = (block) => {
      const lines = block.split('\n');
      let eventType = 'message';
      const dataLines = [];
      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) continue;
        if (line.startsWith('event:')) {
          eventType = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).trim());
        }
      }

      const rawData = dataLines.join('\n');
      if (!rawData) {
        return;
      }

      if (eventType === 'error') {
        throw new Error(rawData);
      }
      if (eventType === 'heartbeat') {
        return;
      }

      let parsed = null;
      try {
        parsed = JSON.parse(rawData);
      } catch (err) {
        return;
      }

      if (!Array.isArray(parsed)) {
        return;
      }

      const chatbotData = parsed[0];
      const audioData = parsed[1];
      const statusText = parsed[2];

      if (typeof statusText === 'string' && statusText) {
        connectionStatus.value = statusText;
        connectionStatusClass.value = 'connected';
        const probsFromStatus = extractRealtimeProbabilitiesFromText(statusText);
        if (probsFromStatus) {
          updateRealtimeProbabilities(probsFromStatus);
        }
        const nextState = parseStateFromStatus(statusText);
        if (nextState) {
          if (realtimeLastState && realtimeLastState !== 'l' && nextState === 'l') {
            commitRealtimeLiveReply();
          }
          realtimeLastState = nextState;
        }
      }

      const probsFromPayload = extractRealtimeProbabilitiesFromPayload(parsed[3]);
      if (probsFromPayload) {
        updateRealtimeProbabilities(probsFromPayload);
      }

      const assistantText = extractAssistantTextFromChatbotData(chatbotData);
      if (assistantText) {
        handleAssistantRealtimeText(assistantText);
      }

      if (audioData && typeof audioData === 'object') {
        enqueueGradioAudio(audioData);
      }
    };

    while (true) {
      const result = await reader.read();
      if (result.done) {
        break;
      }
      pending += decoder.decode(result.value, { stream: true }).replace(/\r/g, '');
      let sepIdx = pending.indexOf('\n\n');
      while (sepIdx >= 0) {
        const block = pending.slice(0, sepIdx);
        pending = pending.slice(sepIdx + 2);
        processSseBlock(block);
        sepIdx = pending.indexOf('\n\n');
      }
    }
    if (pending.trim()) {
      processSseBlock(pending);
    }
  } catch (err) {
    if (controller.signal.aborted || isAbortLikeError(err)) {
      return;
    }
    throw err;
  } finally {
    activeRequestControllers.delete(controller);
  }
};

const createRealtimeSession = async () => {
  const startPayload = {
    start_speak_factor: 1.2,
    start_listen_factor: 1.2,
    end_speak_factor: 1.0,
    prompt_voice: '男声',
    tts_chunk_size: 1,
    infer_window_ms: 400,
    stage_timing_log: true
  };
  if (realtimeBackendHint === 'hf') {
    startPayload.incremental_backend = 'hf';
    startPayload.infer_window_ms = 400;
  }
  const startUrl = buildRealtimeSessionStartUrl();
  const resp = await fetchWithRuntimeContext(startUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(startPayload)
  }, '创建实时会话');
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`创建实时会话失败 ${resp.status}: ${text}`);
  }
  const result = await resp.json();
  if (!result || typeof result.session_id !== 'string' || !result.session_id) {
    throw new Error('创建实时会话返回格式异常');
  }
  return result.session_id;
};

const consumeRealtimeSessionSse = async (sessionId) => {
  const controller = new AbortController();
  activeRequestControllers.add(controller);

  try {
    const eventsUrl = buildRealtimeSessionEventsUrl(sessionId);
    const resp = await fetchWithRuntimeContext(eventsUrl, {
      method: 'GET',
      signal: controller.signal
    }, '连接实时事件流');
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`实时事件流 HTTP ${resp.status}: ${text}`);
    }
    if (!resp.body) {
      throw new Error('实时事件流响应缺少 body');
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let pending = '';

    const processSseBlock = (block) => {
      const lines = block.split('\n');
      let eventType = 'message';
      const dataLines = [];
      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) continue;
        if (line.startsWith('event:')) {
          eventType = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).trim());
        }
      }
      const rawData = dataLines.join('\n');
      if (!rawData) return;
      if (eventType === 'heartbeat') return;

      let parsed = null;
      try {
        parsed = JSON.parse(rawData);
      } catch (err) {
        return;
      }
      if (!parsed || typeof parsed !== 'object') return;
      const eventPayload = normalizeRealtimeEventPayload(parsed, eventType);
      if (!eventPayload || typeof eventPayload !== 'object') return;

      const probsFromPayload = extractRealtimeProbabilitiesFromPayload(eventPayload);
      if (probsFromPayload) {
        updateRealtimeProbabilities(probsFromPayload);
      }

      if (eventType === 'error' || eventPayload.type === 'error') {
        const errMsg = eventPayload.error || rawData;
        throw new Error(String(errMsg));
      }

      if (eventPayload.type === 'status' && typeof eventPayload.status === 'string') {
        connectionStatus.value = eventPayload.status;
        connectionStatusClass.value = 'connected';
        const probsFromStatus = extractRealtimeProbabilitiesFromText(eventPayload.status);
        if (probsFromStatus) {
          updateRealtimeProbabilities(probsFromStatus);
        }
        const nextState = parseStateFromStatus(eventPayload.status);
        if (nextState) {
          if (realtimeLastState && realtimeLastState !== 'l' && nextState === 'l') {
            commitRealtimeLiveReply();
          }
          realtimeLastState = nextState;
        }
      }

      if (eventPayload.type === 'state_change') {
        const toState = typeof eventPayload.to === 'string' ? eventPayload.to.toLowerCase() : '';
        const fromState = typeof eventPayload.from === 'string'
          ? eventPayload.from.toLowerCase()
          : realtimeLastState;
        if (toState) {
          if (fromState && fromState !== 'l' && toState === 'l') {
            commitRealtimeLiveReply();
          }
          realtimeLastState = toState;
        }
      }

      if (eventPayload.type === 'assistant_text' && typeof eventPayload.text === 'string') {
        if (!handleStructuredRealtimeText(eventPayload)) {
          handleAssistantRealtimeText(eventPayload.text);
        }
      }

      if (eventPayload.type === 'audio_chunk_pcm' && typeof eventPayload.pcm_b64 === 'string' && eventPayload.pcm_b64) {
        const clientReceiveEpochMs = Date.now();
        try {
          const sampleRate = Math.max(1, Number(eventPayload.sample_rate) || 24000);
          const pcmSamples = decodePcm16Base64(eventPayload.pcm_b64);
          if (pcmSamples.length > 0) {
            recordRealtimeOutputSamples(pcmSamples, sampleRate);
            scheduleRealtimePcmSamples(pcmSamples, sampleRate)
              .then((scheduleMetrics) => {
                const preEmitClientMs = derivePreEmitMs(
                  eventPayload.client_chunk_sent_to_emit_ms,
                  eventPayload.server_audio_emit_epoch_ms,
                  eventPayload.client_latest_chunk_sent_epoch_ms
                );
                const preEmitServerMs = derivePreEmitMs(
                  eventPayload.server_chunk_recv_to_emit_ms,
                  eventPayload.server_audio_emit_epoch_ms,
                  eventPayload.server_latest_chunk_recv_epoch_ms
                );
                const serverQueueDelayMs = deriveServerQueueDelayMs(
                  eventPayload.server_queue_delay_ms,
                  eventPayload.server_sse_send_epoch_ms,
                  eventPayload.server_audio_emit_epoch_ms
                );
                updateRealtimeAudioLatencyMetrics({
                  backendEmitEpochMs: eventPayload.server_audio_emit_epoch_ms,
                  clientReceiveEpochMs,
                  scheduleMetrics,
                  preEmitClientMs,
                  preEmitServerMs,
                  serverQueueDelayMs
                });
              })
              .catch((err) => {
                console.warn('播放实时PCM音频失败:', err);
              });
          }
        } catch (err) {
          console.warn('解析实时PCM音频失败，跳过对齐采样:', err);
        }
      } else if (eventPayload.type === 'audio_chunk' && typeof eventPayload.wav_b64 === 'string' && eventPayload.wav_b64) {
        const clientReceiveEpochMs = Date.now();
        try {
          const decoded = decodePcm16WavBase64(eventPayload.wav_b64);
          recordRealtimeOutputSamples(decoded.samples, decoded.sampleRate);
          scheduleRealtimePcmSamples(decoded.samples, decoded.sampleRate)
            .then((scheduleMetrics) => {
              const preEmitClientMs = derivePreEmitMs(
                eventPayload.client_chunk_sent_to_emit_ms,
                eventPayload.server_audio_emit_epoch_ms,
                eventPayload.client_latest_chunk_sent_epoch_ms
              );
              const preEmitServerMs = derivePreEmitMs(
                eventPayload.server_chunk_recv_to_emit_ms,
                eventPayload.server_audio_emit_epoch_ms,
                eventPayload.server_latest_chunk_recv_epoch_ms
              );
              const serverQueueDelayMs = deriveServerQueueDelayMs(
                eventPayload.server_queue_delay_ms,
                eventPayload.server_sse_send_epoch_ms,
                eventPayload.server_audio_emit_epoch_ms
              );
              updateRealtimeAudioLatencyMetrics({
                backendEmitEpochMs: eventPayload.server_audio_emit_epoch_ms,
                clientReceiveEpochMs,
                scheduleMetrics,
                preEmitClientMs,
                preEmitServerMs,
                serverQueueDelayMs
              });
            })
            .catch((err) => {
              console.warn('播放实时输出音频失败:', err);
            });
        } catch (err) {
          console.warn('解析实时输出音频失败，跳过对齐采样:', err);
        }
      }

      if (eventPayload.type === 'done') {
        commitRealtimeLiveReply();
        renderRealtimeReply();
        connectionStatus.value = '实时会话结束';
        connectionStatusClass.value = 'disconnected';
      }
    };

    while (true) {
      const result = await reader.read();
      if (result.done) break;
      pending += decoder.decode(result.value, { stream: true }).replace(/\r/g, '');
      let sepIdx = pending.indexOf('\n\n');
      while (sepIdx >= 0) {
        const block = pending.slice(0, sepIdx);
        pending = pending.slice(sepIdx + 2);
        processSseBlock(block);
        sepIdx = pending.indexOf('\n\n');
      }
    }
    if (pending.trim()) {
      processSseBlock(pending);
    }
  } finally {
    activeRequestControllers.delete(controller);
  }
};

const stopRealtimeSession = async (sessionId) => {
  if (!sessionId) return;
  try {
    await fetch(buildRealtimeSessionStopUrl(sessionId), {
      method: 'POST'
    });
  } catch (err) {
    console.warn('停止实时会话失败:', err);
  }
};

const uploadWavChunkToGradio = async (wavBlob, segmentId) => {
  const formData = new FormData();
  formData.append('files', wavBlob, `realtime_seg_${segmentId}.wav`);
  const resp = await fetch(buildGradioUploadUrl(), {
    method: 'POST',
    body: formData
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`上传失败 ${resp.status}: ${text}`);
  }
  const files = await resp.json();
  if (!Array.isArray(files) || files.length === 0 || typeof files[0] !== 'string') {
    throw new Error('上传返回格式异常');
  }
  return {
    path: files[0],
    meta: { _type: 'gradio.FileData' }
  };
};

const flushCaptureBufferToQueue = (force = false) => {
  if (!captureAudioContext.value || pendingCaptureSamples <= 0) {
    return;
  }
  const sourceRate = Math.max(1, Number(captureAudioContext.value.sampleRate) || TARGET_SAMPLE_RATE);
  const chunkMs = Math.max(1, Number(streamChunkMs) || 200);
  const sourceSamplesPerChunk = Math.max(1, Math.round(sourceRate * chunkMs / 1000));
  const targetSamplesPerChunk = Math.max(
    MIN_SEGMENT_SAMPLES,
    Math.round(TARGET_SAMPLE_RATE * chunkMs / 1000)
  );
  if (!force && pendingCaptureSamples < sourceSamplesPerChunk) {
    return;
  }

  let queuedSegments = 0;
  while (pendingCaptureSamples >= sourceSamplesPerChunk) {
    const sourceSegment = takePendingCaptureSamples(sourceSamplesPerChunk);
    if (sourceSegment.length !== sourceSamplesPerChunk) {
      break;
    }
    const samples16k = resampleLinearToLength(sourceSegment, targetSamplesPerChunk);
    if (samples16k.length !== targetSamplesPerChunk) {
      continue;
    }
    if (uploadQueue.length >= MAX_UPLOAD_QUEUE_DEPTH) {
      uploadQueue.shift();
    }
    uploadQueue.push({
      id: ++segmentSeqId,
      samples: samples16k
    });
    queuedSegments += 1;
  }

  if (queuedSegments <= 0) {
    return;
  }

  syncQueueDepth();

  processUploadQueue().catch((err) => {
    console.error('处理实时上传队列失败:', err);
  });
};

const sendOneSegment = async (segment) => {
  if (!realtimeSessionId.value) {
    throw new Error('实时会话不存在');
  }
  const wavBlob = wavBlobFromFloat32(segment.samples, TARGET_SAMPLE_RATE);
  const clientChunkSentEpochMs = Date.now();
  const controller = new AbortController();
  activeRequestControllers.add(controller);

  try {
    const chunkUrl = buildRealtimeSessionChunkUrl(realtimeSessionId.value);
    const resp = await fetchWithRuntimeContext(chunkUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'audio/wav',
        'X-Client-Chunk-Sent-Epoch-Ms': String(clientChunkSentEpochMs)
      },
      body: wavBlob,
      signal: controller.signal
    }, '发送实时分片');

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`上传分片失败 ${resp.status}: ${text}`);
    }
    if (segment && segment.samples && segment.samples.length > 0) {
      recordRealtimeInputSamples(segment.samples);
    }
  } finally {
    activeRequestControllers.delete(controller);
  }
};

const processUploadQueue = async () => {
  if (queueProcessing) {
    return;
  }
  queueProcessing = true;
  try {
    while (isTalking.value && (priorityUploadQueue.length > 0 || uploadQueue.length > 0)) {
      const segment = priorityUploadQueue.length > 0
        ? priorityUploadQueue.shift()
        : uploadQueue.shift();
      syncQueueDepth();
      if (segment?.source === 'file') {
        priorityUploadInProgress.value = true;
      }
      await sendOneSegment(segment);
      segmentsSentCount.value += 1;
      nextUploadAllowedAtMs = 0;
      if (priorityUploadQueue.length === 0) {
        priorityUploadInProgress.value = false;
        priorityUploadSourceName.value = '';
      }
    }
  } catch (err) {
    if (isTalking.value) {
      console.error('实时分片发送失败:', err);
      const failure = classifyStartTalkError(err);
      setStartTalkError(
        `发送失败: ${failure.summary}`,
        failure.hint,
        {
          toast: true,
          title: '实时分片发送失败',
          duration: 7000
        }
      );
      connectionStatus.value = `发送失败: ${failure.summary}`;
      connectionStatusClass.value = 'disconnected';
    }
  } finally {
    queueProcessing = false;
    syncQueueDepth();
  }
};

const releaseRealtimeResources = () => {
  if (uploadInterval) {
    clearInterval(uploadInterval);
    uploadInterval = null;
  }

  for (const controller of activeRequestControllers) {
    controller.abort();
  }
  activeRequestControllers.clear();

  try {
    if (captureProcessorNode) {
      captureProcessorNode.disconnect();
    }
    if (captureSourceNode) {
      captureSourceNode.disconnect();
    }
    if (captureSilentGainNode) {
      captureSilentGainNode.disconnect();
    }
  } catch (err) {
    console.warn('断开音频节点失败:', err);
  }
  captureProcessorNode = null;
  captureSourceNode = null;
  captureSilentGainNode = null;

  if (mediaStream.value) {
    mediaStream.value.getTracks().forEach(track => track.stop());
    mediaStream.value = null;
  }

  if (captureAudioContext.value) {
    captureAudioContext.value.close().catch(() => {});
    captureAudioContext.value = null;
  }

  resetRealtimePlaybackScheduler(true).catch(() => {});

  if (currentGradioAudio) {
    currentGradioAudio.pause();
    currentGradioAudio.src = '';
    currentGradioAudio = null;
  }
  gradioAudioQueue.length = 0;
  gradioAudioPlaying = false;
  lastQueuedAudioUrl = '';
  captureSampleRateDisplay.value = '-';

  pendingCaptureChunks = [];
  pendingCaptureSamples = 0;
  uploadQueue = [];
  priorityUploadQueue = [];
  prioritySegmentQueueDepth.value = 0;
  priorityUploadInProgress.value = false;
  priorityUploadSourceName.value = '';
  segmentQueueDepth.value = 0;
  queueProcessing = false;
  nextUploadAllowedAtMs = 0;
  resetRealtimeLatencyStats();
};

const startTalk = async () => {
  if (isTalking.value) {
    return;
  }

  realtimeStoppingExpected.value = false;
  connectionStatus.value = '连接中...';
  connectionStatusClass.value = 'waiting';
  resetRealtimeReplyState();
  resetRealtimeProbabilities();
  segmentSeqId = 0;
  segmentsSentCount.value = 0;
  pendingCaptureChunks = [];
  pendingCaptureSamples = 0;
  uploadQueue = [];
  priorityUploadQueue = [];
  prioritySegmentQueueDepth.value = 0;
  priorityUploadInProgress.value = false;
  priorityUploadSourceName.value = '';
  segmentQueueDepth.value = 0;
  nextUploadAllowedAtMs = 0;
  resetRealtimeLatencyStats();

  try {
    try {
      await ensureRealtimePlaybackContext();
    } catch (err) {
      console.warn('初始化实时播放上下文失败:', err);
    }
    const sessionId = await createRealtimeSession();
    realtimeSessionId.value = sessionId;
    startCurrentAlignedArchive(sessionId);
    consumeRealtimeSessionSse(sessionId).catch((err) => {
      if (isAbortLikeError(err) || realtimeStoppingExpected.value) {
        return;
      }
      console.error('实时事件流失败:', err);
      const failure = classifyStartTalkError(err);
      setStartTalkError(
        `实时事件流异常: ${failure.summary}`,
        failure.hint,
        {
          toast: true,
          title: '实时事件流失败',
          duration: 7000
        }
      );
      connectionStatus.value = `实时事件流异常: ${failure.summary}`;
      connectionStatusClass.value = 'disconnected';
    });

    mediaStream.value = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      },
      video: false
    });

    captureAudioContext.value = new (window.AudioContext || window.webkitAudioContext)();
    captureSourceNode = captureAudioContext.value.createMediaStreamSource(mediaStream.value);
    captureProcessorNode = captureAudioContext.value.createScriptProcessor(2048, 1, 1);
    captureSilentGainNode = captureAudioContext.value.createGain();
    captureSilentGainNode.gain.value = 0;

    captureSourceNode.connect(captureProcessorNode);
    captureProcessorNode.connect(captureSilentGainNode);
    captureSilentGainNode.connect(captureAudioContext.value.destination);

    captureSampleRateDisplay.value = String(captureAudioContext.value.sampleRate || '-');
    isTalking.value = true;

    captureProcessorNode.onaudioprocess = (event) => {
      if (!isTalking.value) {
        return;
      }
      const frame = new Float32Array(event.inputBuffer.getChannelData(0));
      pendingCaptureChunks.push(frame);
      pendingCaptureSamples += frame.length;
      flushCaptureBufferToQueue(false);
    };

    uploadInterval = setInterval(() => {
      flushCaptureBufferToQueue(false);
    }, streamChunkMs);

    clearStartTalkError({ clearStorage: true });
    connectionStatus.value = '已连接（实时会话），持续发送 200ms 麦克风分片中...';
    connectionStatusClass.value = 'connected';
  } catch (err) {
    console.error('无法启动实时通话:', err);
    const sessionId = realtimeSessionId.value;
    realtimeSessionId.value = '';
    if (sessionId) {
      stopRealtimeSession(sessionId).catch(() => {});
    }
    currentAlignedArchive = null;
    releaseRealtimeResources();
    isTalking.value = false;
    const failure = classifyStartTalkError(err);
    setStartTalkError(
      failure.summary,
      failure.hint,
      {
        toast: true,
        title: '实时通话启动失败',
        duration: 9000
      }
    );
    connectionStatus.value = `启动失败: ${failure.summary}`;
    connectionStatusClass.value = 'disconnected';
  }
};

const endTalk = () => {
  if (!isTalking.value && !captureAudioContext.value && !mediaStream.value) {
    return;
  }
  realtimeStoppingExpected.value = true;
  const sessionId = realtimeSessionId.value;
  realtimeSessionId.value = '';
  isTalking.value = false;
  if (sessionId) {
    stopRealtimeSession(sessionId).catch(() => {});
  }
  releaseRealtimeResources();
  finalizeCurrentAlignedArchive();
  connectionStatus.value = '已挂断';
  connectionStatusClass.value = 'disconnected';
  resetRealtimeProbabilities();
};


// 初始化 Markdown 渲染器
const md = new MarkdownIt({
  highlight: function (str, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try {
        // 使用 highlight.js 高亮代码
        return `<pre class="hljs"><code>${hljs.highlight(str, { language: lang }).value}</code></pre>`
      } catch (e) { console.error(e) }
    }
    return `<pre class="hljs"><code>${md.utils.escapeHtml(str)}</code></pre>`
  }
})


// 页面加载完成后，设置页面标题
onMounted(() => {
  document.title = 'Uni-MoE 2.0'
  function_value.value = 'function_6'
  restorePersistedStartTalkError();
})

onBeforeUnmount(() => {
  endTalk()
  clearAlignedHistory()
})


// 修改页面图标
const link = document.querySelector("link[rel~='icon']")
  || document.createElement('link')
link.rel = 'icon'
link.href = newFavicon
document.head.appendChild(link)


// 初始消息内容，展示给用户的介绍信息
const initialMessages = [
  { id: 1, role: 'bot', text: "你好，我将提供以下功能：\n1.音色克隆：上传用于提取音色的语音文件，并输入需要朗读的文本。" +
    "\n2.音乐生成：我可以根据你输入的文本描述生成对应的音乐！\n3.语音对话：我可以模仿你的声音与你对话！" +
    "\n4.多模态交互：你可以输入语音、图像、文本、视频四种数据进行对话，注意：文本框输入不是必须的，但本功能需要在文本框内按下enter键触发请求！" +
    "\n5.图像生成：我可以生成符合你描述的图像或在给定图像基础上按你的要求编辑图像！\n6.实时通话：实时语音通话，流式输入输出！" }
]


// 用于管理历史消息
const history = ref([])


// 功能选项列表
const fuction_options = [
  { value: 'function_1', label: '音色克隆' },
  { value: 'function_2', label: '音乐生成' },
  { value: 'function_3', label: '语音对话' },
  { value: 'function_4', label: '多模态交互' },
  { value: 'function_5', label: '图像生成' },
  { value: 'function_6', label: '实时通话' }
]


// 当前选择的功能，初始为'function_1'
const function_value = ref("function_1")  


// 音乐选项，仅用于function_2
const music_options = [
  { value: 'music/1_20251206181633.txt', label: '打击乐' },
  { value: 'music/2_20251206181635.txt', label: '摇滚' },
  { value: 'music/3_20251206181551.txt', label: '电子贝斯' },
  { value: 'music/4_20251206181629.txt', label: '节奏鼓点' },
  { value: 'music/5_20251207151005.txt', label: '世界打击乐' },
  { value: 'music/6_20251207151006.txt', label: '放克摇滚' },
  { value: 'music/7_20251207151005.txt', label: '流行电子乐' }
]
// 当前选择的音乐类型
const music_value = ref('music/1_20251206181633.txt')


// 音色选项，包含不同的音色和相应的音频文件
const yinse_options=[
  {value: '1', label: '雷军',audio_file:'clone/leijun_voice.mp3',content_file:'clone/text.txt'},
  {value: '2', label: '郭德纲',audio_file:'clone/gudegang_voice.wav',content_file:'clone/guodegang.txt'},
  {value: '3', label: '周杰伦',audio_file:'clone/jay.wav',content_file:'clone/jay.txt'},
  {value: '4', label: '胡歌',audio_file:'clone/huge.mp3',content_file:'clone/huge.txt'},
  {value: '5', label: '韩红',audio_file:'clone/hanhong.wav',content_file:'clone/hanhong.txt'},
  {value: '6', label: '奶龙',audio_file:'clone/nailong.wav',content_file:'clone/nailong.txt'},
  {value: '7', label: '柯南',audio_file:'clone/kenan.wav',content_file:'clone/kenan.txt'},
  {value: '8', label: '海绵宝宝',audio_file:'clone/haimian.wav',content_file:'clone/haimian.txt'},
  {value: '9', label: '邓紫棋',audio_file:'clone/dengziqi.wav',content_file:'clone/dengziqi.txt'},
  {value: '10', label: '李云龙',audio_file:'clone/liyunlong.wav',content_file:'clone/liyunlong.txt'},
  {value: '11', label: '清纯女声',audio_file:'clone/new_female_voice.mp3',content_file:'clone/text.txt'},
  {value: '12', label: '阳光女声',audio_file:'clone/female_voice.mp3',content_file:'clone/text.txt'},
  {value: '13', label: '播音男声',audio_file:'clone/news_male_voice.mp3',content_file:'clone/text.txt'}
]
// 当前选择的音色
const yinse_value=ref('')


// 消息列表，用户和机器人之间的对话记录
const messages = ref([...initialMessages])


// 录音相关变量
const isRecording = ref(false)  // 是否正在录音
let mediaRecorder = null  // MediaRecorder 实例
let audioChunks = []  // 存储音频数据
let audioBlob = null  // 存储音频生成的 Blob 文件


// 用于控制文件上传的显示状态
const micro_visible = ref(false)
const audioupload_visible = ref(false)
const textupload_visible = ref(false)
const imageupload_visible = ref(false)
const videoupload_visible = ref(false)


// 用于文件内容的存储
const fileContent = ref('')

// 是否正在发送消息
const isSending = ref(false)


// 图像合成功能是否开启think模式
const img_think = ref(false)

// 引用 chatMessagesRef，用于控制聊天消息区域的滚动
const chatMessagesRef = ref(null)

// 用户输入框的内容
const userInput = ref('')

// 存储用户选择的文件
const audioFile = ref(null)  // 存储音频文件
const textFile = ref(null)   // 存储文本文件
const imageFile = ref(null)  // 存储图片文件
const videoFile = ref(null)  // 存储视频文件

// 文件标识符
let file_id = 1111




// 将 WebM 格式音频转换为 WAV 格式
function convertWebmToWavUsingEncoder(webmBlob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = async function () {
      const arrayBuffer = reader.result;
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();

      // 解码音频数据
      audioContext.decodeAudioData(arrayBuffer, (audioBuffer) => {
        const pcmData = audioBuffer.getChannelData(0); // 获取音频的 PCM 数据（假设是单声道）

        const wavData = {
          sampleRate: audioBuffer.sampleRate,
          channelData: [pcmData],  // 假设是单声道数据
        };

        // 使用 WAVEncoder 编码为 WAV 格式
        WAVEncoder.encode(wavData).then((encodedWav) => {
          const wavBlob = new Blob([encodedWav], { type: 'audio/wav' });
          resolve(wavBlob);
        }).catch(reject);
      }, reject);
    };

    reader.onerror = reject;
    reader.readAsArrayBuffer(webmBlob);  // 将 WebM 文件读取为 ArrayBuffer
  });
}



// 启动录音功能
function startRecording() {
  if (isRecording.value) return;  // 如果已经在录音中，直接返回
  isRecording.value = true;
  audioChunks = [];  // 清空之前的录音数据

  // 检查浏览器是否支持录音功能
  if (typeof window !== 'undefined' && navigator.mediaDevices) {
    console.log('浏览器支持录音');
  } else {
    console.log('当前环境不支持录音');
  }

  // 请求麦克风权限
  window.navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      // 创建 MediaRecorder 实例
      mediaRecorder = new MediaRecorder(stream);

      // 收集录音数据块
      mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
      };

      // 录音停止后处理录音数据
      mediaRecorder.onstop = async () => {
        audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });  // 将录音数据转换为 Blob
        audioFile.value = new File([audioBlob], 'recorded_audio.webm', { type: mediaRecorder.mimeType });  // 创建文件对象

        // 将 WebM 格式音频转换为 WAV 格式
        const wavBlob = await convertWebmToWavUsingEncoder(audioFile.value);
        const wavFile = new File([wavBlob], 'recorded_audio.wav', { type: 'audio/wav' });
        audioFile.value = wavFile;

        // 根据功能值处理后续操作
        if (function_value.value == "function_3") {
          audioChat();  // 语音对话功能
        } else {
          open1(audioFile.value.name);  // 上传成功通知
        }

        // 上传音频数据
        if (function_value.value == 'function_4' || function_value.value == 'function_6' || function_value.value == 'function_7') {
          const fileToSend = audioFile.value;
          const localAudioUrl = URL.createObjectURL(fileToSend);
          messages.value.push({
            id: Date.now() + 1,
            role: 'user',
            text: '',
            audioUrl: localAudioUrl
          });
          scrollToBottom();  // 滚动到消息底部
        }

        // 发送消息
        if (function_value.value == 'function_6' || function_value.value == 'function_7') {
          sendMessage();
        }
      };

      mediaRecorder.start();  // 开始录音
    })
    .catch(error => {
      console.error('无法访问麦克风:', error);
      ElNotification({
        title: '录音失败',
        message: '无法访问麦克风，请检查麦克风权限设置。',
        type: 'error'
      });
    });
}


// 停止录音
function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop()  // 停止录音
    isRecording.value = false;  // 录音停止
    micro_visible.value=false;
  }
  
}

// 语音对话功能，处理用户上传的音频
async function audioChat() {
  if (isSending.value) return;
  if (function_value.value == "function_3" && !audioFile.value) return;  // 如果功能为语音对话且没有上传音频，直接返回

  isSending.value = true;
  let text = '';  // 初始化文本内容

  const fileToSend = audioFile.value;
  const localAudioUrl = URL.createObjectURL(fileToSend);
  messages.value.push({
    id: Date.now(),
    role: 'user',
    text: '',
    audioUrl: localAudioUrl
  });

  scrollToBottom();  // 滚动到消息底部

  // 清空输入框和音频文件
  userInput.value = '';
  audioFile.value = null;
  messages.value.push({
    id: Date.now() + 1,
    role: 'bot',
    text: '正在生成中，请稍后...',
    audioUrl: null
  });

  try {
    // 将音频文件转换为 Base64 格式
    const dataUrl = await readFileAsBase64(fileToSend);
    let payload = {
      model: 'UniMoE-Audio',
      text,
      prompt_text: text,
      prompt_audio: dataUrl,  // 这里放 Base64 编码后的音频
      return_base64: true,
      function: "voice_call",
    };

    // 发送请求至后台进行处理
    const resp = await fetch('http://219.223.251.156:8085/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) {
      throw new Error('处理语音失败');
    }

    const data = await resp.json();
    const choice = data.choices?.[0];
    const contentStr = choice?.message?.content || '{}';
    let contentObj = {};

    try {
      contentObj = JSON.parse(contentStr);
    } catch (e) {
      console.error('解析后端 content JSON 失败：', e);
      contentObj = {};
    }

    const blob = base64ToBlob(contentObj.audio_base64, 'audio/wav');
    const processedAudioUrl = URL.createObjectURL(blob);

    let audio_content = "";
    if (contentObj.reply_text != "No Reply") {
      audio_content = contentObj.reply_text;
    }

    messages.value.pop();  // 移除"正在生成中..."的提示消息

    // 将处理后的语音（机器人回复）显示到对话区
    messages.value.push({
      id: Date.now() + 1,
      role: 'bot',
      text: audio_content,
      audioUrl: processedAudioUrl
    });

    scrollToBottom();
  } catch (error) {
    // 出现错误时，返回错误消息
    messages.value.pop();
    messages.value.push({
      id: Date.now() + 2,
      role: 'bot',
      text: '抱歉，处理语音时出现了问题，请稍后重试。'
    });
    scrollToBottom();
  } finally {
    isSending.value = false;
  }
}


// 将实时文本转换为 HTML，同时保留不同说话段之间的换行。
function renderMarkdown(text) {
  if (typeof text !== 'string') return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\r/g, '')
    .replace(/\n/g, '<br>');
  // return md.render(text)
}


// 重新发起对话
function restartConversation() {
  messages.value = [...initialMessages]
  userInput.value = ''
  audioFile.value = null
  textFile.value = null
  imageFile.value = null
  videoFile.value = null
}

// 手动触发上传按钮
function clickMultiButton() {
  const fileInput = document.querySelector('#hidden-audio-input');
  if (fileInput) {
    fileInput.click();  // 触发音频文件上传
  }
}

function clickRealtimeAudioInjectButton() {
  if (!isTalking.value) {
    ElNotification({
      title: '提示',
      message: '请先开始通话后再发送测试音频',
      type: 'warning'
    });
    return;
  }
  const fileInput = document.querySelector('#hidden-realtime-audio-inject-input');
  if (fileInput) {
    fileInput.click();
  }
}

// 上传文本文件
function clickMultiButtonText() {
  const fileInput = document.querySelector('#hidden-text-input');
  if (fileInput) {
    fileInput.click();  // 触发文本文件上传
  }
}

// 上传图片文件
function clickMultiButtonImage() {
  const fileInput = document.querySelector('#hidden-image-input');
  if (fileInput) {
    fileInput.click();  // 触发图片文件上传
  }
}

// 上传视频文件
function clickMultiButtonVideo() {
  const fileInput = document.querySelector('#hidden-video-input');
  if (fileInput) {
    fileInput.click();  // 触发视频文件上传
  }
}

// 上传成功通知
function open1(msg) {
  ElNotification({
    title: '上传成功',
    message: msg,
    type: 'success'
  });
}

// 上传失败通知
function open2() {
  ElNotification({
    title: '上传失败',
    message: '',
    type: 'error'
  });
}



// 选择语音文件
function onFileChangeAudio(e) {
  const file = e.target.files[0]  // 获取用户选择的第一个文件
  if (file) {
    audioFile.value = file  // 将选择的音频文件存储到 audioFile 变量中
    open1(`已选择音频文件：${audioFile.value.name}`)  // 弹出提示，显示已选择的文件名
  } else {
    audioFile.value = null  // 如果没有文件选择，清空 audioFile
    open2()  // 弹出提示，显示选择失败
  }

  // 如果当前功能为多模态交互或实时通话，发送音频文件
  if (function_value.value == "function_4" || function_value.value == "function_6" || function_value.value == "function_7") {
    const fileToSend = audioFile.value
    const localImageUrl = URL.createObjectURL(fileToSend)  // 创建文件的本地 URL
    messages.value.push({
      id: Date.now() + 1,
      role: 'user',
      text: '',
      audioUrl: localImageUrl  // 在消息中附加音频文件的 URL
    })
    scrollToBottom()  // 滚动到消息的底部
  }
}

async function onRealtimeAudioFileChange(e) {
  const file = e.target?.files?.[0];
  if (e.target) {
    e.target.value = '';
  }
  if (!file) {
    return;
  }

  try {
    const localAudioUrl = URL.createObjectURL(file);
    messages.value.push({
      id: Date.now() + 1,
      role: 'user',
      text: '[实时测试音频：优先发送]',
      audioUrl: localAudioUrl
    });
    scrollToBottom();

    const segmentCount = await enqueuePriorityAudioFile(file);
    ElNotification({
      title: '已加入优先队列',
      message: `${file.name}，共 ${segmentCount} 个分片，将优先于麦克风分片发送`,
      type: 'success'
    });
  } catch (err) {
    console.error('插队发送测试音频失败:', err);
    ElNotification({
      title: '发送失败',
      message: String(err),
      type: 'error'
    });
  }
}

// 选择语音文本文件
function onFileChangeText(e) {
  const file = e.target.files[0];  // 获取用户上传的第一个文本文件
  
  if (file) {  // 确保文件存在
    const reader = new FileReader();  // 创建文件读取器
    
    reader.onload = function(e) {
      fileContent.value = e.target.result;  // 文件读取成功，将内容存储到 fileContent 中
      textFile.value = file  // 将文件对象存储到 textFile 中
      open1(`已选择文本文件：${textFile.value.name}`)  // 弹出提示，显示已选择的文件名
    };
    
    reader.onerror = function(error) {
      console.error("文件读取失败", error);  // 读取失败时，打印错误
    };
    
    reader.readAsText(file);  // 读取文本文件的内容
  } else {
    fileContent.value = null  // 如果没有文件选择，清空文本内容
    textFile.value = null  // 清空文本文件
    open2()  // 弹出提示，显示选择失败
  }
}


// 选择图像文件
function onFileChangeImage(e) {
  const file = e.target.files[0]  // 获取用户选择的第一个图像文件
  if (file) {
    imageFile.value = file  // 将图像文件存储到 imageFile 变量中
    open1(`已选择图片文件：${imageFile.value.name}`)  // 弹出提示，显示已选择的图像文件名
  } else {
    imageFile.value = null  // 如果没有文件选择，清空图像文件
    open2()  // 弹出提示，显示选择失败
  }

  // 如果当前功能为图像生成或多模态交互，发送图像文件
  if (function_value.value == "function_4" || function_value.value == "function_5") {
    const fileToSend = imageFile.value
    const localImageUrl = URL.createObjectURL(fileToSend)  // 创建文件的本地 URL
    messages.value.push({
      id: Date.now() + 1,
      role: 'user',
      text: '',
      imageUrl: localImageUrl  // 在消息中附加图像文件的 URL
    })
    scrollToBottom()  // 滚动到消息的底部
  }
}

// 选择视频文件
function onFileChangeVideo(e) {
  const file = e.target.files[0]  // 获取用户选择的第一个视频文件
  if (file) {
    videoFile.value = file  // 将视频文件存储到 videoFile 变量中
    open1(`已选择视频文件：${videoFile.value.name}`)  // 弹出提示，显示已选择的视频文件名
  } else {
    videoFile.value = null  // 如果没有文件选择，清空视频文件
    open2()  // 弹出提示，显示选择失败
  }

  // 如果当前功能为多模态交互，发送视频文件
  if (function_value.value == "function_4") {
    const fileToSend = videoFile.value
    const localImageUrl = URL.createObjectURL(fileToSend)  // 创建文件的本地 URL
    messages.value.push({
      id: Date.now() + 1,
      role: 'user',
      text: '',
      videoUrl: localImageUrl  // 在消息中附加视频文件的 URL
    })
    scrollToBottom()  // 滚动到消息的底部
  }
}

// 获得文件格式（例如音频、图片等的文件后缀名）
function getExtension(filename) {
  return filename.split('.').pop().toLowerCase()  // 获取文件的扩展名并转换为小写
}



// 发送消息的函数
async function sendMessage() {
  // 防止重复发送消息
  if (isSending.value) return

  // 获取用户输入的文本和文件内容
  let text = userInput.value.trim()
  const prompt_text = fileContent.value.trim()

  // 校验各个功能所需的条件
  if (function_value.value == "function_2" && !text) return  // 音乐生成功能需要文本
  if ((function_value.value == "function_1") && (!text || !audioFile.value)) return  // 音色克隆需要文本和音频文件
  if ((function_value.value == "function_4") && (!text && !audioFile.value && !imageFile.value && !videoFile.value)) return  // 多模态交互需要至少一种输入
  if ((function_value.value == "function_5") && (!text)) return  // 图像生成功能需要文本
  if ((function_value.value == "function_6" || function_value.value == "function_7") && (!audioFile.value)) return  // 语音对话需要音频文件

  isSending.value = true  // 标记正在发送消息

  const fileToSend = audioFile.value  // 获取要发送的音频文件

  // 1. 根据不同功能将“用户消息”放到对话区
  if (function_value.value == "function_1") {
    // 音色克隆功能
    const localAudioUrl = URL.createObjectURL(fileToSend)  // 创建音频文件的 URL
    messages.value.push({
      id: Date.now(),
      role: 'user',
      text,
      audioUrl: localAudioUrl  // 发送带有音频的消息
    })
  }

  if (function_value.value == "function_2") {
    // 音乐生成功能
    messages.value.push({
      id: Date.now(),
      role: 'user',
      text,
      audioUrl: null  // 发送文本消息，音频 URL 为空
    })
  }

  if (function_value.value == "function_4" && text) {
    // 多模态交互功能
    messages.value.push({
      id: Date.now(),
      role: 'user',
      text  // 发送文本消息
    })
  }

  if (function_value.value == "function_5" && text) {
    // 图像生成功能
    messages.value.push({
      id: Date.now(),
      role: 'user',
      text
    })
    if (img_think.value) {
      // 如果启用了图像思考模式，修改文本内容
      text = "You should first think step by step about how to construct the image, including background, objects, colors, lighting, and style. \nThe reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively."
    }
  }

  // 滚动到消息底部
  scrollToBottom()

  // 发送消息后，清空输入框内容
  userInput.value = ''
  messages.value.push({
    id: Date.now() + 1,
    role: 'bot',
    text: '正在生成中，请稍后...',
    audioUrl: null
  })

  try {
    let payload = {}

    // 根据功能构建请求的 payload
    if (function_value.value == "function_1") {
      // 音色克隆功能
      const dataUrl = await readFileAsBase64(fileToSend)  // 将音频文件转为 Base64
      payload = {
        model: 'UniMoE-Audio',
        text,
        prompt_text: prompt_text,
        prompt_audio: dataUrl,  // 发送 Base64 编码的音频
        return_base64: true,
        function: "text_to_speech",  // 功能：文本转语音
      }
    }

    if (function_value.value == "function_2") {
      // 音乐生成功能
      payload = {
        model: 'UniMoE-Audio',
        text,
        return_base64: true,
        function: "text_to_music",  // 功能：文本转音乐
      }
    }

    if (function_value.value == "function_4" || function_value.value == "function_5" || function_value.value == "function_6" || function_value.value == "function_7") {
      // 多模态交互、图像生成、语音对话功能
      let content = []
      if (text) content.push({ type: "text", text })  // 发送文本消息

      // 处理音频文件
      if (audioFile.value) {
        let dataUrl = await readFileAsBase64(audioFile.value)
        content.push({ type: "audio", file_type: `${file_id}.${getExtension(audioFile.value.name)}`, audio: dataUrl })
        file_id++
      }

      // 处理图像文件
      if (imageFile.value) {
        let dataUrl = await readFileAsBase64(imageFile.value)
        content.push({ type: "image", file_type: `${file_id}.${getExtension(imageFile.value.name)}`, image: dataUrl })
        file_id++
      }

      // 处理视频文件
      if (videoFile.value) {
        let dataUrl = await readFileAsBase64(videoFile.value)
        content.push({ type: "video", file_type: `${file_id}.${getExtension(videoFile.value.name)}`, video: dataUrl })
        file_id++
      }

      history.value.push({ role: 'user', content })
      
      // 设置不同的 payload
      if (function_value.value == "function_4") {
        payload = { function: "文本交互", history: history.value }
      }
      if (function_value.value == "function_5") {
        payload = { function: "图像生成", history: history.value }
      }
      if (function_value.value == "function_6") {
        payload = { function: "中文语音对话", history: history.value }
      }
      if (function_value.value == "function_7") {
        payload = { function: "英文语音对话", history: history.value }
      }
    }

    // 发送请求到后端生成内容
    const url = function_value.value == "function_5" ? 'http://10.249.45.38:6778/gen' : 'http://10.249.45.38:6777/gen'
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })

    if (!resp.ok) throw new Error('生成失败')

    const data = await resp.json()
    const reply = data.reply
    const contents = reply.content
    history.value.push(reply)
    messages.value.pop()

    // 处理返回的内容（音频、图像、文本）
    for (const item of contents) {
      if (item.type !== "text") {
        let mimeType = `${item.type}/${getExtension(item.file_type)}`
        let processedUrl = URL.createObjectURL(base64ToBlob(item[item.type], mimeType))
        
        // 根据内容类型更新消息
        if (item.type === "audio") {
          messages.value.push({ id: Date.now() + 1, role: 'bot', audioUrl: processedUrl })
        }
        if (item.type === "image") {
          messages.value.push({ id: Date.now() + 1, role: 'bot', imageUrl: processedUrl })
        }
        if (item.type === "video") {
          messages.value.push({ id: Date.now() + 1, role: 'bot', videoUrl: processedUrl })
        }
        
        scrollToBottom()
      } else {
        let pre = ""
        if (img_think.value && function_value.value == "function_5") {
          pre = "思考过程：\n\n"
        }
        let t = item.text.replace(/<answer>.*?<\/answer>/s, "")
        messages.value.push({ id: Date.now() + 1, role: 'bot', text: pre + t })
      }
    }

  } catch (error) {
    // 错误处理，展示错误消息
    messages.value.pop()
    messages.value.push({
      id: Date.now() + 2,
      role: 'bot',
      text: '抱歉，生成时出现了问题，请稍后重试。'
    })
    scrollToBottom()
  } finally {
    isSending.value = false  // 发送完成，恢复发送状态
  }
}

// 把 File 读成 base64（dataURL）
function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}


// 把纯 base64 字符串转成 Blob
function base64ToBlob(base64, mimeType = 'audio/wav') {
  const byteChars = atob(base64)
  const byteNumbers = new Array(byteChars.length)
  for (let i = 0; i < byteChars.length; i++) {
    byteNumbers[i] = byteChars.charCodeAt(i)
  }
  const byteArray = new Uint8Array(byteNumbers)
  return new Blob([byteArray], { type: mimeType })
}


// 滚动到消息底部的函数
function scrollToBottom() {
  nextTick(() => {
    const el = chatMessagesRef.value
    if (el) {
      // 设置滚动条的位置为消息区域的最底部
      el.scrollTop = el.scrollHeight
    }
  })
}


// 监听功能模式的变化（切换不同功能时的响应）
watch(function_value, (newVal, oldVal) => {
  if (oldVal === 'function_6' && newVal !== 'function_6' && isTalking.value) {
    endTalk()
  }

  // 每次功能切换时，重置消息历史
  messages.value = [initialMessages[0]]

  // 音色克隆模式切换：清空输入框和音色
  if (newVal === 'function_1' && oldVal !== 'function_1') {
    userInput.value = ""
    yinse_value.value = ""
  }

  // 音乐生成模式切换：设置默认音乐和输入示例
  if (newVal === 'function_2' && oldVal !== 'function_2') {
    music_value.value = "music/1_20251206181633.txt"
    userInput.value = "This song contains a digital drum playing a simple pattern with a kick and a snare sound. Synthesizers are playing a repeating melody in the higher register..."
  }

  // 多模态交互模式切换：清空相关文件和历史记录
  if (newVal === 'function_4' && oldVal !== 'function_4') {
    audioFile.value = null
    imageFile.value = null
    videoFile.value = null
    history.value = []
    userInput.value = ""
  }

  // 图像生成模式切换：清空图像文件和历史记录
  if (newVal === 'function_5' && oldVal !== 'function_5') {
    imageFile.value = null
    history.value = []
    userInput.value = ""
  }

  // 中文语音对话模式切换：清空音频文件和历史记录
  if (newVal === 'function_6' && oldVal !== 'function_6') {
    audioFile.value = null
    history.value = []
  }

  // 英文语音对话模式切换：清空音频文件和历史记录
  if (newVal === 'function_7' && oldVal !== 'function_7') {
    audioFile.value = null
    history.value = []
  }
})

// 监听音色选择的变化，并更新对应的文本和音频文件
watch(yinse_value, (newVal, oldVal) => {
  let index = parseInt(newVal) - 1

  if (function_value.value == "function_1") {
    // 加载音色对应的文本文件
    fetch(yinse_options[index].content_file)
      .then(res => res.text())  // 获取文本内容
      .then(text => {
        const textBlob = new Blob([text], { type: 'text/plain' })
        const textFileObj = new File([textBlob], 'content.txt', { type: 'text/plain' })
        textFile.value = textFileObj
        fileContent.value = text
      })
      .catch(err => {
        console.error('读取 txt 失败: ', err)
        textFile.value = null
      })

    // 加载音色对应的音频文件
    fetch(yinse_options[index].audio_file)
      .then(res => res.blob())  // 获取音频文件的 Blob（二进制文件）
      .then(blob => {
        const fileType = blob.type
        const fileExtension = fileType.split('/')[1]
        const audioFileObj = new File([blob], `audio_file.${fileExtension}`, { type: fileType })
        audioFile.value = audioFileObj
      })
      .catch(error => {
        console.error('加载音频失败: ', error)
        audioFile.value = null
      })
  }
})

// 监听音乐选择的变化，并更新用户输入框的内容
watch(music_value, (newVal, oldVal) => {
  if (function_value.value == "function_2") {
    // 加载选中的音乐描述文件
    fetch(newVal)
      .then(res => res.text())
      .then(text => {
        userInput.value = text  // 更新输入框内容为音乐描述
      })
      .catch(err => {
        console.error('读取 txt 失败: ', err)
      })
  }
})

// 监听消息列表的变化，自动滚动到底部
watch(
  () => messages.value.length,
  () => {
    scrollToBottom()  // 每当消息列表变化时，自动滚动到底部
  }
)
</script>

<style scoped>
.rt-app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  width: 100%;
  color: #1f2a37;
  background:
    radial-gradient(circle at 12% 10%, rgba(66, 153, 225, 0.12), transparent 38%),
    radial-gradient(circle at 90% 86%, rgba(16, 185, 129, 0.1), transparent 40%),
    linear-gradient(120deg, #f4f8fb 0%, #ecf3f9 100%);
}

.rt-glass-header {
  height: 64px;
  display: flex;
  align-items: center;
  padding: 0 20px;
  background: rgba(255, 255, 255, 0.75);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(15, 23, 42, 0.08);
}

.rt-header-content {
  display: flex;
  align-items: center;
  gap: 10px;
}

.rt-logo-img {
  width: 38px;
  height: 38px;
}

.rt-header-title {
  font-size: 20px;
  font-weight: 700;
  color: #0f172a;
}

.rt-badge {
  margin-left: 8px;
  font-size: 12px;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 999px;
  color: #0b7285;
  background: rgba(13, 148, 136, 0.12);
}

.rt-main-workspace {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}

.rt-global-error-strip {
  width: min(1180px, 96vw);
  margin: 12px auto 0;
  border-radius: 12px;
  border: 1px solid rgba(220, 38, 38, 0.25);
  background: rgba(254, 242, 242, 0.92);
  padding: 10px 12px;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.rt-global-error-content {
  min-width: 0;
}

.rt-global-error-title {
  font-size: 12px;
  font-weight: 700;
  color: #991b1b;
  margin-bottom: 4px;
}

.rt-global-error-text {
  font-size: 13px;
  color: #7f1d1d;
  line-height: 1.5;
  word-break: break-word;
}

.rt-global-error-hint {
  margin-top: 4px;
  font-size: 12px;
  color: #92400e;
  line-height: 1.5;
  word-break: break-word;
}

.rt-global-error-dismiss {
  border: none;
  background: rgba(127, 29, 29, 0.12);
  color: #7f1d1d;
  border-radius: 8px;
  padding: 4px 8px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
  flex-shrink: 0;
}

.rt-global-error-dismiss:hover {
  background: rgba(127, 29, 29, 0.2);
}

.rt-welcome-screen {
  width: min(760px, 94vw);
}

.rt-welcome-card {
  padding: 42px 36px;
  text-align: center;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.85);
  border: 1px solid rgba(15, 23, 42, 0.08);
  box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
}

.rt-icon-pulse {
  font-size: 52px;
  margin-bottom: 10px;
  animation: rt-pulse 1.8s infinite ease-in-out;
}

.rt-welcome-card h1 {
  margin: 0;
  font-size: 30px;
  color: #0f172a;
}

.rt-welcome-card p {
  margin: 10px 0 24px;
  font-size: 16px;
  color: #475569;
}

.rt-active-workspace {
  width: min(1180px, 96vw);
  display: grid;
  grid-template-columns: minmax(280px, 360px) minmax(340px, 1fr) minmax(300px, 360px);
  gap: 16px;
}

.rt-panel {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
  min-height: 420px;
}

.rt-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
  gap: 10px;
}

.rt-panel-title {
  font-size: 16px;
  font-weight: 700;
  color: #0f172a;
}

.rt-status-badge {
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: start;
  column-gap: 8px;
  font-size: 12px;
  font-weight: 600;
  border-radius: 12px;
  padding: 6px 10px;
  background: rgba(100, 116, 139, 0.14);
  color: #334155;
  min-width: 320px;
  max-width: 320px;
  line-height: 1.3;
  overflow: hidden;
}

.rt-status-text {
  min-width: 0;
  white-space: normal;
  word-break: break-word;
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  overflow: hidden;
  max-height: calc(1.3em * 2);
}

.rt-status-badge.connected {
  color: #166534;
  background: rgba(34, 197, 94, 0.16);
}

.rt-status-badge.disconnected {
  color: #b91c1c;
  background: rgba(239, 68, 68, 0.16);
}

.rt-status-badge.waiting {
  color: #9a3412;
  background: rgba(251, 146, 60, 0.16);
}

.rt-dot {
  width: 8px;
  height: 8px;
  margin-top: 4px;
  border-radius: 50%;
  background: currentColor;
}

.rt-audio-container {
  height: calc(100% - 48px);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 18px;
}

.rt-mic-wrapper {
  margin-top: 12px;
  width: 150px;
  height: 150px;
  border-radius: 50%;
  background: linear-gradient(155deg, #dbeafe, #dcfce7);
  border: 1px solid rgba(15, 23, 42, 0.08);
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}

.rt-mic-icon {
  font-size: 46px;
  z-index: 2;
}

.rt-ripple {
  position: absolute;
  width: 150px;
  height: 150px;
  border-radius: 50%;
  border: 2px solid rgba(59, 130, 246, 0.3);
  animation: rt-ripple 1.8s infinite ease-out;
}

.rt-ripple-2 {
  animation-delay: 0.8s;
}

.rt-audio-info {
  width: 100%;
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
  font-size: 13px;
  color: #334155;
}

.rt-audio-info > span {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-variant-numeric: tabular-nums;
}

.rt-audio-info-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-width: 0;
}

.rt-audio-info-head > span {
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-variant-numeric: tabular-nums;
}

.rt-audio-info-muted {
  color: #64748b;
}

.rt-debug-flag {
  color: #0f766e;
  font-weight: 700;
}

.rt-metrics-toggle {
  border: 1px solid rgba(15, 23, 42, 0.14);
  border-radius: 999px;
  background: #ffffff;
  color: #0f172a;
  font-size: 12px;
  font-weight: 600;
  padding: 4px 10px;
  cursor: pointer;
  white-space: nowrap;
  transition: background-color 0.2s ease, border-color 0.2s ease;
}

.rt-metrics-toggle:hover {
  background: #f8fafc;
  border-color: rgba(14, 116, 144, 0.38);
}

.rt-prob-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 2px;
}

.rt-prob-tag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 104px;
  padding: 3px 10px;
  border-radius: 999px;
  border: 1px solid rgba(14, 116, 144, 0.25);
  background: rgba(14, 116, 144, 0.1);
  color: #0f172a;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.2px;
  font-variant-numeric: tabular-nums;
}

.rt-playback-controls {
  width: 100%;
  margin-top: auto;
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  color: #334155;
}

.rt-playback-controls input[type="range"] {
  width: 100%;
}

.rt-text-content {
  height: calc(100% - 48px);
  overflow: auto;
  border-radius: 12px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  background: #ffffff;
  padding: 14px;
}

.rt-history-content {
  height: calc(100% - 48px);
  overflow: auto;
  border-radius: 12px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  background: #ffffff;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.rt-history-entry {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-bottom: 12px;
  border-bottom: 1px dashed rgba(15, 23, 42, 0.12);
}

.rt-history-entry:last-child {
  border-bottom: none;
  padding-bottom: 0;
}

.rt-history-meta {
  font-size: 12px;
  color: #475569;
  line-height: 1.6;
  padding-bottom: 8px;
  border-bottom: 1px dashed rgba(15, 23, 42, 0.12);
}

.rt-history-audio-block {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.rt-history-label {
  font-size: 13px;
  color: #1f2937;
  font-weight: 600;
}

.rt-history-audio-block audio {
  width: 100%;
}

.rt-waiting-text {
  color: #64748b;
  font-size: 14px;
}

.rt-typing-indicator span {
  animation: rt-blink 1.5s infinite;
}

.rt-typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.rt-typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

.rt-bottom-control-bar {
  height: 74px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 16px;
  background: rgba(255, 255, 255, 0.82);
  border-top: 1px solid rgba(15, 23, 42, 0.08);
}

.rt-control-group {
  display: flex;
  gap: 12px;
}

.rt-btn-primary,
.rt-btn-secondary,
.rt-btn-danger {
  border: none;
  border-radius: 12px;
  padding: 10px 16px;
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
  transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
}

.rt-btn-primary {
  background: linear-gradient(130deg, #0ea5e9, #0284c7);
  color: #fff;
  box-shadow: 0 8px 18px rgba(2, 132, 199, 0.28);
}

.rt-btn-secondary {
  background: #e2e8f0;
  color: #1e293b;
}

.rt-btn-danger {
  background: linear-gradient(130deg, #ef4444, #dc2626);
  color: #fff;
  box-shadow: 0 8px 18px rgba(220, 38, 38, 0.28);
}

.rt-btn-primary:hover,
.rt-btn-secondary:hover,
.rt-btn-danger:hover {
  transform: translateY(-1px);
}

.rt-large-btn {
  padding: 12px 22px;
  font-size: 16px;
}

.rt-hover-shake:hover {
  animation: rt-shake 0.36s ease-in-out;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

@media (max-width: 960px) {
  .rt-main-workspace {
    padding: 14px;
  }

  .rt-global-error-strip {
    width: calc(100% - 20px);
    margin: 10px 10px 0;
  }

  .rt-active-workspace {
    width: 100%;
    grid-template-columns: 1fr;
  }

  .rt-panel {
    min-height: 300px;
  }

  .rt-status-badge {
    min-width: 0;
    max-width: 100%;
  }

  .rt-audio-info-head {
    align-items: flex-start;
    flex-direction: column;
  }

  .rt-metrics-toggle {
    align-self: stretch;
  }
}

@keyframes rt-pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.08); }
  100% { transform: scale(1); }
}

@keyframes rt-ripple {
  from {
    transform: scale(1);
    opacity: 0.7;
  }
  to {
    transform: scale(1.35);
    opacity: 0;
  }
}

@keyframes rt-blink {
  0%, 80%, 100% { opacity: 0.2; }
  40% { opacity: 1; }
}

@keyframes rt-shake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-2px); }
  75% { transform: translateX(2px); }
}
</style>
