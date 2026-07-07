<template>
  <div class="wt-page">
    <header class="wt-header">
      <div>
        <h1>权重批量对比测试</h1>
        <p>文件输入固定切片发送，复用 realtime session 接口，不启用麦克风采集。</p>
      </div>
      <a class="wt-link" href="/">实时通话页</a>
    </header>

    <main class="wt-layout">
      <section class="wt-panel wt-control-panel">
        <div class="wt-section-title">输入</div>
        <label class="wt-field">
          <span>测试音频</span>
          <input type="file" accept="audio/*" :disabled="running" @change="onFileChange" />
        </label>

        <label class="wt-field">
          <span>参考音色</span>
          <select v-model="selectedVoice" :disabled="running || voiceOptions.length === 0">
            <option v-for="voice in voiceOptions" :key="voice.id" :value="voice.id">
              {{ voice.label }}
            </option>
          </select>
        </label>

        <div class="wt-grid-2">
          <label class="wt-field">
            <span>发送分片(ms)</span>
            <input v-model.number="chunkMs" type="number" min="20" max="2000" step="10" :disabled="running" />
          </label>
          <label class="wt-field">
            <span>推理窗口(ms)</span>
            <input v-model.number="inferWindowMs" type="number" min="160" max="2000" step="10" :disabled="running" />
          </label>
        </div>

        <div class="wt-grid-2">
          <label class="wt-field">
            <span>说话起始因子</span>
            <input v-model.number="startSpeakFactor" type="number" min="0.1" max="5" step="0.05" :disabled="running" />
          </label>
          <label class="wt-field">
            <span>监听起始因子</span>
            <input v-model.number="startListenFactor" type="number" min="0.1" max="5" step="0.05" :disabled="running" />
          </label>
        </div>
        <label class="wt-field">
          <span>说话结束因子</span>
          <input v-model.number="endSpeakFactor" type="number" min="0.1" max="5" step="0.05" :disabled="running" />
        </label>

        <label class="wt-field">
          <span>卡顿阈值</span>
          <input v-model.number="stutterThresholdMs" type="number" min="20" max="1000" step="10" :disabled="running" />
        </label>

        <label class="wt-field">
          <span>尾部静音</span>
          <input v-model.number="tailPaddingSec" type="number" min="0" max="120" step="1" :disabled="running" />
        </label>

        <div class="wt-actions">
          <button class="wt-primary" :disabled="running || !selectedFile" @click="runTest">
            {{ running ? '测试中...' : '发送文件并测试' }}
          </button>
          <button class="wt-secondary" :disabled="!running" @click="stopTest">停止</button>
        </div>

        <div class="wt-status" :class="statusClass">{{ statusText }}</div>

        <div class="wt-file-meta" v-if="fileMeta.name">
          <div>文件: {{ fileMeta.name }}</div>
          <div>输入: {{ fileMeta.durationSec }}s / {{ fileMeta.sourceRate }} Hz -> 16000 Hz</div>
          <div>实际发送: {{ fileMeta.sentDurationSec }}s（含 {{ fileMeta.paddingSec }}s 尾部静音）</div>
          <div>分片: {{ fileMeta.chunkCount }} 个 / {{ chunkMs }} ms</div>
          <div>实际发送 PCM SHA-1: <code>{{ inputSha1 || '计算中' }}</code></div>
        </div>
      </section>

      <section class="wt-panel wt-stats-panel">
        <div class="wt-section-title">统计</div>
        <div class="wt-stat-grid">
          <div class="wt-stat">
            <span>已发送分片</span>
            <strong>{{ sentChunks }} / {{ totalChunks }}</strong>
          </div>
          <div class="wt-stat">
            <span>响应音频块</span>
            <strong>{{ outputChunkCount }}</strong>
          </div>
          <div class="wt-stat">
            <span>卡顿次数</span>
            <strong>{{ stutterCount }}</strong>
          </div>
          <div class="wt-stat">
            <span>最大卡顿</span>
            <strong>{{ maxStutterMs.toFixed(1) }} ms</strong>
          </div>
          <div class="wt-stat">
            <span>原始输出</span>
            <strong>{{ rawOutputSec }}s</strong>
          </div>
          <div class="wt-stat">
            <span>对齐长度</span>
            <strong>{{ alignedSec }}s</strong>
          </div>
        </div>

        <div class="wt-subtitle">卡顿明细</div>
        <div class="wt-stutter-list">
          <div v-if="stutterEvents.length === 0" class="wt-muted">暂无卡顿</div>
          <div v-for="item in stutterEvents" :key="item.index" class="wt-stutter-row">
            #{{ item.index }} <strong>@+{{ item.wallRelS.toFixed(2) }}s</strong>
            gap={{ formatMs(item.gapMs) }}
            interval={{ formatMs(item.arrivalDeltaMs) }}
            align={{ formatMs(item.alignGapMs) }}
            <span class="wt-t2w-hint">前次合成 dur={{ fmt(item.nearT2wDur) }} rtf={{ fmtRtf(item.nearT2wRtf) }}</span>
          </div>
        </div>
      </section>

      <section class="wt-panel wt-stage-panel">
        <div class="wt-section-title">各阶段用时</div>
        <table class="wt-stage-table" v-if="lastStageTiming">
          <thead>
            <tr>
              <th>阶段</th>
              <th>最近一轮</th>
              <th>累计平均</th>
              <th>说话平均</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>输入队列等待</td>
              <td>{{ fmt(lastStageTiming.queueWait) }}</td>
              <td>{{ fmt(avgStageTiming && avgStageTiming.queueWait) }}</td>
              <td>{{ fmt(avgSpeakingStageTiming && avgSpeakingStageTiming.queueWait) }}</td>
            </tr>
            <tr>
              <td>预处理</td>
              <td>{{ fmt(lastStageTiming.preprocess) }}</td>
              <td>{{ fmt(avgStageTiming && avgStageTiming.preprocess) }}</td>
              <td>{{ fmt(avgSpeakingStageTiming && avgSpeakingStageTiming.preprocess) }}</td>
            </tr>
            <tr>
              <td>编码器</td>
              <td>{{ fmt(lastStageTiming.encoder) }}</td>
              <td>{{ fmt(avgStageTiming && avgStageTiming.encoder) }}</td>
              <td>{{ fmt(avgSpeakingStageTiming && avgSpeakingStageTiming.encoder) }}</td>
            </tr>
            <tr>
              <td>Transformer</td>
              <td>{{ fmt(lastStageTiming.transformer) }}</td>
              <td>{{ fmt(avgStageTiming && avgStageTiming.transformer) }}</td>
              <td>{{ fmt(avgSpeakingStageTiming && avgSpeakingStageTiming.transformer) }}</td>
            </tr>
            <tr>
              <td>Token2Wav</td>
              <td>{{ fmt(lastStageTiming.token2wav) }}</td>
              <td>{{ fmt(avgStageTiming && avgStageTiming.token2wav) }}</td>
              <td>{{ fmt(avgSpeakingStageTiming && avgSpeakingStageTiming.token2wav) }}</td>
            </tr>
            <tr>
              <td>首包延迟(后端)</td>
              <td>{{ fmt(lastStageTiming.firstPcm) }}</td>
              <td>{{ fmt(avgStageTiming && avgStageTiming.firstPcm) }}</td>
              <td>{{ fmt(avgSpeakingStageTiming && avgSpeakingStageTiming.firstPcm) }}</td>
            </tr>
            <tr>
              <td>整轮耗时</td>
              <td>{{ fmt(lastStageTiming.totalRound) }}</td>
              <td>{{ fmt(avgStageTiming && avgStageTiming.totalRound) }}</td>
              <td>{{ fmt(avgSpeakingStageTiming && avgSpeakingStageTiming.totalRound) }}</td>
            </tr>
            <tr>
              <td>队列积压(取材料前)</td>
              <td>{{ fmt(lastStageTiming.qBefore) }}</td>
              <td>{{ fmt(avgStageTiming && avgStageTiming.qBefore) }}</td>
              <td>{{ fmt(avgSpeakingStageTiming && avgSpeakingStageTiming.qBefore) }}</td>
            </tr>
            <tr>
              <td>队列消费(本轮)</td>
              <td>{{ fmt(lastStageTiming.qConsumed) }}</td>
              <td>{{ fmt(avgStageTiming && avgStageTiming.qConsumed) }}</td>
              <td>{{ fmt(avgSpeakingStageTiming && avgSpeakingStageTiming.qConsumed) }}</td>
            </tr>
            <tr>
              <td>队列剩余(消费后)</td>
              <td>{{ fmt(lastStageTiming.qAfter) }}</td>
              <td>{{ fmt(avgStageTiming && avgStageTiming.qAfter) }}</td>
              <td>{{ fmt(avgSpeakingStageTiming && avgSpeakingStageTiming.qAfter) }}</td>
            </tr>
            <tr>
              <td>后端推理窗口</td>
              <td>{{ fmt(lastStageTiming.qWindow) }}</td>
              <td>{{ fmt(avgStageTiming && avgStageTiming.qWindow) }}</td>
              <td>{{ fmt(avgSpeakingStageTiming && avgSpeakingStageTiming.qWindow) }}</td>
            </tr>
            <tr>
              <td>首包延迟(端到端)</td>
              <td>{{ fmt(firstPcmE2eMs) }}</td>
              <td>—</td>
              <td>—</td>
            </tr>
          </tbody>
        </table>
        <div v-else class="wt-muted">暂无阶段数据</div>
        <div class="wt-stage-meta" v-if="lastStageTiming">
          累计 {{ stageRounds.length }} 轮，其中说话 {{ speakingRounds.length }} 轮
        </div>

        <div class="wt-subtitle">逐轮明细</div>
        <div class="wt-stutter-list">
          <div v-if="stageRounds.length === 0" class="wt-muted">暂无</div>
          <div v-for="r in stageRounds" :key="r.round" class="wt-stutter-row">
            #{{ r.round }}<span v-if="r.state==='speaking'" class="wt-tag-spk">说话</span>
            <span class="wt-t2w-hint">主干 {{ fmtRel(r.startedAtMs) }}→{{ fmtRel(r.completedAtMs) }}</span>
            qw={{ fmt(r.queueWait) }} pre={{ fmt(r.preprocess) }}
            enc={{ fmt(r.encoder) }} tf={{ fmt(r.transformer) }} t2w={{ fmt(r.token2wav) }}
            fp={{ fmt(r.firstPcm) }} total={{ fmt(r.totalRound) }}
            <span class="wt-t2w-hint">队列 {{ fmt(r.qBefore) }} -{{ fmt(r.qConsumed) }} → {{ fmt(r.qAfter) }} / 窗口 {{ fmt(r.qWindow) }}</span>
          </div>
        </div>

        <div class="wt-subtitle">Token2Wav 活动轨迹</div>
        <div class="wt-stage-meta" v-if="t2wSummary">
          合成 {{ t2wSummary.count }} 次，耗时均 {{ fmt(t2wSummary.durAvg) }} / max {{ fmt(t2wSummary.durMax) }}，
          RTF 均 {{ fmtRtf(t2wSummary.rtfAvg) }} / max {{ fmtRtf(t2wSummary.rtfMax) }}，
          <strong>RTF&gt;1 共 {{ t2wSummary.rtfOver1 }} 次</strong>
        </div>
        <div class="wt-stutter-list">
          <div v-if="t2wTrace.length === 0" class="wt-muted">暂无</div>
          <div v-for="t in t2wTrace" :key="t.idx" class="wt-stutter-row"
               :class="{ 'wt-t2w-slow': t.rtf != null && t.rtf > 1 }">
            #{{ t.idx }} start={{ fmtRel(t.startMs) }} dur={{ fmt(t.durMs) }}
            tok={{ t.tokens }} adv={{ t.advance }} audio={{ fmt(t.audioMs) }}
            rtf={{ fmtRtf(t.rtf) }}<span v-if="t.roundtripMs != null"> rt={{ fmt(t.roundtripMs) }}</span>
          </div>
        </div>
      </section>

      <section class="wt-panel wt-audio-panel">
        <div class="wt-section-title">输出音频</div>
        <div class="wt-audio-block">
          <div class="wt-audio-title">对齐双声道音频（左: 输入，右: 输出）</div>
          <audio v-if="alignedStereoUrl" :src="alignedStereoUrl" controls preload="metadata"></audio>
          <button class="wt-secondary" :disabled="!alignedStereoUrl" @click="downloadUrl(alignedStereoUrl, alignedStereoFilename)">
            下载对齐双声道
          </button>
        </div>
        <div class="wt-audio-block">
          <div class="wt-audio-title">原始输出音频</div>
          <audio v-if="rawOutputUrl" :src="rawOutputUrl" controls preload="metadata"></audio>
          <button class="wt-secondary" :disabled="!rawOutputUrl" @click="downloadUrl(rawOutputUrl, rawOutputFilename)">
            下载原始输出
          </button>
        </div>
      </section>

      <section class="wt-panel wt-text-panel">
        <div class="wt-section-title">响应文本</div>
        <pre class="wt-response-text">{{ responseText || '暂无响应文本' }}</pre>
      </section>
    </main>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue';

const TARGET_SAMPLE_RATE = 16000;
const TTS_SAMPLE_RATE = 24000;
const MIN_SEGMENT_SAMPLES = 800;
const DEFAULT_VOICE = 'default_female';

const fallbackVoiceOptions = [
  { id: 'default_female', label: '默认女声' },
  { id: 'default_male', label: '默认男声' },
  { id: 'leijun', label: '雷军' },
  { id: 'guodegang', label: '郭德纲' },
  { id: 'jay', label: '周杰伦' },
  { id: 'huge', label: '胡歌' },
  { id: 'hanhong', label: '韩红' },
  { id: 'nailong', label: '奶龙' },
  { id: 'kenan', label: '柯南' },
  { id: 'haimian', label: '海绵宝宝' },
  { id: 'dengziqi', label: '邓紫棋' },
  { id: 'liyunlong', label: '李云龙' },
  { id: 'new_female', label: '清纯女声' },
  { id: 'female', label: '阳光女声' },
  { id: 'news_male', label: '播音男声' }
];

const selectedFile = ref(null);
const selectedVoice = ref(DEFAULT_VOICE);
const voiceOptions = ref([...fallbackVoiceOptions]);
const chunkMs = ref(200);
const inferWindowMs = ref(400);
const startSpeakFactor = ref(1.2);
const startListenFactor = ref(1.2);
const endSpeakFactor = ref(1.0);
const stutterThresholdMs = ref(160);
const tailPaddingSec = ref(30);
const running = ref(false);
const statusText = ref('等待选择音频文件');
const statusClass = ref('idle');
const responseText = ref('');
const sentChunks = ref(0);
const totalChunks = ref(0);
const outputChunkCount = ref(0);
const stutterCount = ref(0);
const maxStutterMs = ref(0);
const stutterEvents = ref([]);
const inputSha1 = ref('');
const alignedStereoUrl = ref('');
const rawOutputUrl = ref('');
const alignedStereoFilename = ref('aligned_stereo.wav');
const rawOutputFilename = ref('raw_output.wav');
const rawOutputSec = ref('0.00');
const alignedSec = ref('0.00');
// ---- 各阶段用时（来自后端 stage_timing 事件）----
const lastStageTiming = ref(null);   // 最近一轮
const stageRounds = ref([]);         // 每轮一条
const firstPcmE2eMs = ref(null);     // 端到端首包延迟（前端测量）
let firstSentEpochMs = null;         // 首片发送时刻（= X-Client-Chunk-Sent-Epoch-Ms）
let firstPcmRecorded = false;
const STAGE_KEYS = [
  'queueWait',
  'preprocess',
  'encoder',
  'transformer',
  'token2wav',
  'firstPcm',
  'totalRound',
  'qBefore',
  'qConsumed',
  'qAfter',
  'qWindow'
];
const averageRows = (rows) => {
  if (!rows.length) return null;
  const out = {};
  for (const k of STAGE_KEYS) {
    const vals = rows.map((r) => r[k]).filter((v) => v != null && Number.isFinite(v));
    out[k] = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  }
  return out;
};
const avgStageTiming = computed(() => averageRows(stageRounds.value));
const speakingRounds = computed(() => stageRounds.value.filter((r) => r.state === 'speaking'));
const avgSpeakingStageTiming = computed(() => averageRows(speakingRounds.value));

// ---- token2wav 活动轨迹（解耦异步合成，用于和卡顿对照）----
const TTS_TOKEN_HZ = 25;             // 25Hz：advance_tokens / 25 * 1000 = 合成音频长度(ms)
const t2wTrace = ref([]);            // 每次合成一条
const t2wSummary = computed(() => {
  const rows = t2wTrace.value;
  if (!rows.length) return null;
  const durs = rows.map((r) => r.durMs).filter((v) => v != null && Number.isFinite(v));
  const rtfs = rows.map((r) => r.rtf).filter((v) => v != null && Number.isFinite(v));
  const avg = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
  return {
    count: rows.length,
    durAvg: avg(durs),
    durMax: durs.length ? Math.max(...durs) : null,
    rtfAvg: avg(rtfs),
    rtfMax: rtfs.length ? Math.max(...rtfs) : null,
    rtfOver1: rtfs.filter((v) => v > 1).length,   // 合成跟不上的次数 → 卡顿风险
  };
});
const fileMeta = reactive({
  name: '',
  durationSec: '0.00',
  sentDurationSec: '0.00',
  paddingSec: '30.00',
  sourceRate: '-',
  chunkCount: 0
});

let activeSessionId = '';
let stopRequested = false;
let eventsController = null;
let chunkControllers = [];
let archive = null;
let currentTextEventId = '';
let currentTextEventSnapshot = '';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, Math.max(0, ms)));

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

const realtimeStartUrl = () => `${buildGradioApiBase()}/api/realtime/session/start`;
const realtimeVoicesUrl = () => `${buildGradioApiBase()}/api/realtime/voices`;
const realtimeChunkUrl = (sessionId) => `${buildGradioApiBase()}/api/realtime/session/${encodeURIComponent(sessionId)}/chunk`;
const realtimeEventsUrl = (sessionId) => `${buildGradioApiBase()}/api/realtime/session/${encodeURIComponent(sessionId)}/events`;
const realtimeStopUrl = (sessionId) => `${buildGradioApiBase()}/api/realtime/session/${encodeURIComponent(sessionId)}/stop`;

const setStatus = (text, cls = 'idle') => {
  statusText.value = text;
  statusClass.value = cls;
};

const formatMs = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? `${n.toFixed(1)}ms` : '--';
};

// 阶段值可能缺失（如 token2wav），null 显示「—」
const fmt = (value) => (value == null ? '—' : formatMs(value));
// 相对时刻：epoch 毫秒 − 音频开始发送的 startEpochMs，显示成 +N.Ns（相对发送开始）
const fmtRel = (epochMs) => {
  if (epochMs == null || !archive || !archive.startEpochMs) return '—';
  const rel = epochMs - archive.startEpochMs;
  return `+${(rel / 1000).toFixed(2)}s`;
};
const fmtRtf = (v) => (v == null || !Number.isFinite(v) ? '—' : v.toFixed(2));
const normalizeNumber = (value, fallback, min, max, digits = 3) => {
  const n = Number(value);
  const bounded = clamp(Number.isFinite(n) ? n : fallback, min, max);
  const scale = 10 ** digits;
  return Math.round(bounded * scale) / scale;
};
const normalizeInt = (value, fallback, min, max) => {
  const n = Number(value);
  return Math.round(clamp(Number.isFinite(n) ? n : fallback, min, max));
};
const normalizeTestConfig = () => {
  const cfg = {
    chunkMs: normalizeInt(chunkMs.value, 200, 20, 2000),
    inferWindowMs: normalizeInt(inferWindowMs.value, 400, 160, 2000),
    startSpeakFactor: normalizeNumber(startSpeakFactor.value, 1.2, 0.1, 5),
    startListenFactor: normalizeNumber(startListenFactor.value, 1.2, 0.1, 5),
    endSpeakFactor: normalizeNumber(endSpeakFactor.value, 1.0, 0.1, 5),
  };
  chunkMs.value = cfg.chunkMs;
  inferWindowMs.value = cfg.inferWindowMs;
  startSpeakFactor.value = cfg.startSpeakFactor;
  startListenFactor.value = cfg.startListenFactor;
  endSpeakFactor.value = cfg.endSpeakFactor;
  return cfg;
};

// 记录 token2wav 一次合成的活动轨迹（来自 audio_chunk_pcm 透传的 t2w_* 字段）
const recordT2wTrace = (payload) => {
  if (payload.t2w_synth_start_epoch_ms == null && payload.t2w_synth_duration_ms == null) return;
  const advance = Number(payload.t2w_advance_tokens) || 0;
  const audioMs = advance > 0 ? (advance / TTS_TOKEN_HZ) * 1000 : null;
  const durMs = payload.t2w_synth_duration_ms != null ? Number(payload.t2w_synth_duration_ms) : null;
  t2wTrace.value.push({
    idx: t2wTrace.value.length + 1,
    startMs: payload.t2w_synth_start_epoch_ms != null ? Number(payload.t2w_synth_start_epoch_ms) : null,
    endMs: payload.t2w_synth_end_epoch_ms != null ? Number(payload.t2w_synth_end_epoch_ms) : null,
    durMs,
    tokens: payload.t2w_tokens != null ? Number(payload.t2w_tokens) : null,
    advance,
    audioMs,
    rtf: (durMs != null && audioMs && audioMs > 0) ? durMs / audioMs : null,
    roundtripMs: payload.t2w_remote_roundtrip_ms != null ? Number(payload.t2w_remote_roundtrip_ms) : null,
    backend: payload.t2w_backend || null,
    recvMs: Date.now(),
  });
};

// 解析后端 stage_timing 事件
const applyStageTiming = (payload) => {
  const stages = payload.stages || {};
  const latency = payload.latency || {};
  const row = {
    round: payload.round_id,
    state: payload.state || null,
    inputSec: payload.input_duration_sec,
    startedAtMs: payload.round_started_at_epoch_ms != null ? Number(payload.round_started_at_epoch_ms) : null,
    completedAtMs: payload.round_completed_at_epoch_ms != null ? Number(payload.round_completed_at_epoch_ms) : null,
    queueWait: stages.input_queue_wait,
    preprocess: stages.audio_preprocess,
    encoder: stages.audio_encoder,
    transformer: stages.transformer,
    token2wav: stages.token2wav,
    firstPcm: latency.first_pcm_out_ms,
    totalRound: latency.total_round_ms,
    qBefore: payload.queue ? payload.queue.pending_before_ms : null,   // 取材料前队列积压(ms)
    qConsumed: payload.queue ? payload.queue.consumed_ms : null,       // 本轮消费(ms)
    qAfter: payload.queue ? payload.queue.pending_after_ms : null,     // 消费后剩余(ms)
    qWindow: payload.queue ? payload.queue.infer_window_ms : null,     // 后端实际推理窗口(ms)
  };
  lastStageTiming.value = row;
  stageRounds.value.push(row);
};

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

const concatFloat32 = (chunks, totalLen) => {
  const out = new Float32Array(Math.max(0, totalLen));
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
};

const resampleLinear = (input, srcRate, dstRate) => {
  if (!input || input.length === 0 || srcRate === dstRate) {
    return input || new Float32Array(0);
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

const float32ToInt16 = (samples) => {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    const s = clamp(samples[i], -1, 1);
    out[i] = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
  }
  return out;
};

const wavBlobFromMono = (samples, sampleRate) => {
  const pcm16 = float32ToInt16(samples);
  const dataSize = pcm16.length * 2;
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);
  writeWavHeader(view, dataSize, sampleRate, 1);
  let offset = 44;
  for (let i = 0; i < pcm16.length; i += 1) {
    view.setInt16(offset, pcm16[i], true);
    offset += 2;
  }
  return new Blob([buf], { type: 'audio/wav' });
};

const wavBlobFromStereo = (left, right, sampleRate) => {
  const frameCount = Math.max(left.length, right.length);
  const dataSize = frameCount * 4;
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);
  writeWavHeader(view, dataSize, sampleRate, 2);
  let offset = 44;
  for (let i = 0; i < frameCount; i += 1) {
    const l = clamp(left[i] || 0, -1, 1);
    const r = clamp(right[i] || 0, -1, 1);
    view.setInt16(offset, l < 0 ? Math.round(l * 32768) : Math.round(l * 32767), true);
    view.setInt16(offset + 2, r < 0 ? Math.round(r * 32768) : Math.round(r * 32767), true);
    offset += 4;
  }
  return new Blob([buf], { type: 'audio/wav' });
};

const writeWavHeader = (view, dataSize, sampleRate, channels) => {
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
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, dataSize, true);
};

const sha1Hex = async (bytes) => {
  const digest = await crypto.subtle.digest('SHA-1', bytes);
  return Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, '0')).join('');
};

const decodeAudioFileToMono16k = async (file) => {
  const arrayBuffer = await file.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
    const channelCount = Math.max(1, audioBuffer.numberOfChannels || 1);
    const mixed = new Float32Array(audioBuffer.length || 0);
    for (let ch = 0; ch < channelCount; ch += 1) {
      const data = audioBuffer.getChannelData(ch);
      for (let i = 0; i < mixed.length; i += 1) {
        mixed[i] += data[i] / channelCount;
      }
    }
    const samples16k = resampleLinear(mixed, audioBuffer.sampleRate || TARGET_SAMPLE_RATE, TARGET_SAMPLE_RATE);
    return {
      samples: samples16k,
      sourceRate: audioBuffer.sampleRate || TARGET_SAMPLE_RATE
    };
  } finally {
    ctx.close().catch(() => {});
  }
};

const splitSamples = (samples, currentChunkMs) => {
  const segmentSamples = Math.max(MIN_SEGMENT_SAMPLES, Math.floor(TARGET_SAMPLE_RATE * currentChunkMs / 1000));
  const out = [];
  for (let start = 0; start < samples.length; start += segmentSamples) {
    const end = Math.min(samples.length, start + segmentSamples);
    const piece = samples.slice(start, end);
    if (piece.length < MIN_SEGMENT_SAMPLES && out.length > 0) {
      const prev = out.pop();
      const merged = new Float32Array(prev.length + piece.length);
      merged.set(prev, 0);
      merged.set(piece, prev.length);
      out.push(merged);
    } else {
      out.push(piece);
    }
  }
  return out.filter((item) => item.length >= MIN_SEGMENT_SAMPLES);
};

const appendTailSilence = (samples, silenceSec) => {
  const padSamples = Math.max(0, Math.round((Number(silenceSec) || 0) * TARGET_SAMPLE_RATE));
  if (padSamples <= 0) {
    return samples;
  }
  const out = new Float32Array(samples.length + padSamples);
  out.set(samples, 0);
  return out;
};

const base64ToBytes = (value) => {
  const raw = atob(value);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) {
    bytes[i] = raw.charCodeAt(i);
  }
  return bytes;
};

const decodePcm16Base64 = (pcmB64, channels = 1) => {
  const bytes = base64ToBytes(pcmB64);
  const usableLen = bytes.length - (bytes.length % 2);
  const view = new DataView(bytes.buffer, bytes.byteOffset, usableLen);
  const channelCount = Math.max(1, Number(channels) || 1);
  const frameCount = Math.floor(usableLen / 2 / channelCount);
  const mono = new Float32Array(frameCount);
  let ptr = 0;
  for (let i = 0; i < frameCount; i += 1) {
    let sum = 0;
    for (let ch = 0; ch < channelCount; ch += 1) {
      const s = view.getInt16(ptr, true);
      ptr += 2;
      sum += s < 0 ? s / 32768 : s / 32767;
    }
    mono[i] = sum / channelCount;
  }
  return mono;
};

const decodeWavBase64 = (wavB64) => {
  const bytes = base64ToBytes(wavB64);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (offset) => String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1), view.getUint8(offset + 2), view.getUint8(offset + 3));
  if (bytes.length < 44 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') {
    throw new Error('Invalid WAV payload');
  }
  let fmt = -1;
  let data = -1;
  let dataSize = 0;
  for (let offset = 12; offset + 8 <= view.byteLength;) {
    const chunkTag = tag(offset);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (chunkTag === 'fmt ') fmt = body;
    if (chunkTag === 'data') {
      data = body;
      dataSize = size;
      break;
    }
    offset = body + size + (size % 2);
  }
  if (fmt < 0 || data < 0) throw new Error('Missing WAV chunks');
  const format = view.getUint16(fmt, true);
  const channels = view.getUint16(fmt + 2, true);
  const sampleRate = view.getUint32(fmt + 4, true);
  const bits = view.getUint16(fmt + 14, true);
  if (format !== 1 || bits !== 16) throw new Error('Only PCM16 WAV is supported');
  const frameCount = Math.floor(dataSize / 2 / Math.max(1, channels));
  const mono = new Float32Array(frameCount);
  let ptr = data;
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

const resetOutputs = () => {
  responseText.value = '';
  currentTextEventId = '';
  currentTextEventSnapshot = '';
  sentChunks.value = 0;
  totalChunks.value = 0;
  outputChunkCount.value = 0;
  stutterCount.value = 0;
  maxStutterMs.value = 0;
  stutterEvents.value = [];
  inputSha1.value = '';
  rawOutputSec.value = '0.00';
  alignedSec.value = '0.00';
  lastStageTiming.value = null;
  stageRounds.value = [];
  firstPcmE2eMs.value = null;
  firstSentEpochMs = null;
  firstPcmRecorded = false;
  t2wTrace.value = [];
  revokeOutputUrls();
};

const revokeOutputUrls = () => {
  if (alignedStereoUrl.value) URL.revokeObjectURL(alignedStereoUrl.value);
  if (rawOutputUrl.value) URL.revokeObjectURL(rawOutputUrl.value);
  alignedStereoUrl.value = '';
  rawOutputUrl.value = '';
};

const createArchive = (inputSamples) => ({
  startPerfMs: 0,
  startEpochMs: 0,
  inputSamples,
  outputChunks: [],
  rawOutputChunks: [],
  outputSamples: 0,
  rawOutputSamples: 0,
  alignedSamples: inputSamples.length,
  hasOutput: false,
  lastAudioArrivalPerfMs: null,
  lastAudioChunkDurationMs: 0
});

const recordOutputChunk = (samples, sampleRate) => {
  if (!archive || !samples || samples.length === 0) return;
  // 端到端首包延迟：首个 PCM 到达时刻 - 首片发送时刻（与后端 first_pcm 口径互补）
  if (!firstPcmRecorded) {
    firstPcmRecorded = true;
    if (firstSentEpochMs != null) {
      firstPcmE2eMs.value = Date.now() - firstSentEpochMs;
    }
  }
  const arrivalPerfMs = performance.now();
  const normalized = sampleRate === TARGET_SAMPLE_RATE
    ? samples
    : resampleLinear(samples, sampleRate, TARGET_SAMPLE_RATE);
  const chunkDurationMs = (normalized.length / TARGET_SAMPLE_RATE) * 1000;
  const rawChunk = new Float32Array(normalized);
  archive.rawOutputChunks.push(rawChunk);
  archive.rawOutputSamples += rawChunk.length;
  outputChunkCount.value += 1;

  const elapsedSamples = Math.max(0, Math.round(((arrivalPerfMs - archive.startPerfMs) / 1000) * TARGET_SAMPLE_RATE));
  const gapSamples = Math.max(0, elapsedSamples - archive.outputSamples);
  let stutterGapMs = 0;
  let arrivalDeltaMs = null;
  if (archive.hasOutput && archive.lastAudioArrivalPerfMs !== null) {
    arrivalDeltaMs = arrivalPerfMs - archive.lastAudioArrivalPerfMs;
    stutterGapMs = Math.max(0, arrivalDeltaMs - archive.lastAudioChunkDurationMs);
  }
  if (gapSamples > 0) {
    const alignGapMs = (gapSamples / TARGET_SAMPLE_RATE) * 1000;
    if (stutterGapMs >= Number(stutterThresholdMs.value || 0)) {
      const event = {
        index: stutterEvents.value.length + 1,
        gapMs: stutterGapMs,
        alignGapMs,
        arrivalDeltaMs,
        prevChunkMs: archive.lastAudioChunkDurationMs,
        atMs: (archive.outputSamples / TARGET_SAMPLE_RATE) * 1000,
        wallRelS: ((Date.now() - (archive.startEpochMs || Date.now())) / 1000),  // 相对发送开始(秒)
        // 此卡顿前最近一次 token2wav 合成的指标，便于判断是否其抖动导致
        nearT2wDur: (t2wTrace.value.length ? t2wTrace.value[t2wTrace.value.length - 1].durMs : null),
        nearT2wRtf: (t2wTrace.value.length ? t2wTrace.value[t2wTrace.value.length - 1].rtf : null)
      };
      stutterEvents.value.push(event);
      stutterCount.value = stutterEvents.value.length;
      maxStutterMs.value = Math.max(maxStutterMs.value, stutterGapMs);
    }
    archive.outputChunks.push(new Float32Array(gapSamples));
    archive.outputSamples += gapSamples;
  }
  archive.outputChunks.push(rawChunk);
  archive.outputSamples += rawChunk.length;
  archive.alignedSamples = Math.max(archive.alignedSamples, archive.outputSamples);
  rawOutputSec.value = (archive.rawOutputSamples / TARGET_SAMPLE_RATE).toFixed(2);
  alignedSec.value = (archive.alignedSamples / TARGET_SAMPLE_RATE).toFixed(2);
  archive.hasOutput = true;
  archive.lastAudioArrivalPerfMs = arrivalPerfMs;
  archive.lastAudioChunkDurationMs = chunkDurationMs;
};

const finalizeArchive = () => {
  if (!archive) return;
  const alignedLen = Math.max(archive.inputSamples.length, archive.outputSamples);
  const left = new Float32Array(alignedLen);
  left.set(archive.inputSamples.slice(0, alignedLen));
  const right = concatFloat32(archive.outputChunks, archive.outputSamples);
  const rightAligned = new Float32Array(alignedLen);
  rightAligned.set(right.slice(0, alignedLen));
  const raw = concatFloat32(archive.rawOutputChunks, archive.rawOutputSamples);

  const timestamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+$/, '').replace('T', '_');
  alignedStereoFilename.value = `weight_test_aligned_${timestamp}.wav`;
  rawOutputFilename.value = `weight_test_raw_output_${timestamp}.wav`;
  alignedStereoUrl.value = URL.createObjectURL(wavBlobFromStereo(left, rightAligned, TARGET_SAMPLE_RATE));
  rawOutputUrl.value = URL.createObjectURL(wavBlobFromMono(raw, TARGET_SAMPLE_RATE));
  archive.alignedSamples = alignedLen;
  rawOutputSec.value = (archive.rawOutputSamples / TARGET_SAMPLE_RATE).toFixed(2);
  alignedSec.value = (archive.alignedSamples / TARGET_SAMPLE_RATE).toFixed(2);
};

const normalizeEventPayload = (payload, sseEventType = 'message') => {
  if (!payload || typeof payload !== 'object') return null;
  const rawType = typeof payload.type === 'string' ? payload.type : '';
  const unifiedType = typeof payload.event_type === 'string' ? payload.event_type : '';
  const type = unifiedType || rawType || sseEventType || 'message';
  const normalized = { ...payload, type };

  const frameText = payload.frame_text && typeof payload.frame_text === 'object' ? payload.frame_text : null;
  if (type === 'assistant_text' && frameText) {
    normalized.text = normalized.text || frameText.delta || frameText.snapshot || '';
    normalized.snapshot = normalized.snapshot || frameText.snapshot || '';
    normalized.delta = normalized.delta || frameText.delta || '';
    normalized.event_id = normalized.event_id || frameText.event_id || '';
    if (typeof normalized.is_final === 'undefined') {
      normalized.is_final = frameText.is_final;
    }
  }

  const frameAudio = payload.frame_audio && typeof payload.frame_audio === 'object' ? payload.frame_audio : null;
  if (frameAudio && (type === 'audio_chunk_pcm' || type === 'audio_chunk')) {
    const fmt = String(frameAudio.format || '').toLowerCase();
    if (fmt === 'pcm_s16le' || frameAudio.pcm_b64) {
      normalized.type = 'audio_chunk_pcm';
      normalized.pcm_b64 = normalized.pcm_b64 || frameAudio.pcm_b64;
      normalized.sample_rate = normalized.sample_rate || frameAudio.sample_rate;
      normalized.num_channels = normalized.num_channels || frameAudio.num_channels;
    } else if (fmt === 'wav_b64' || frameAudio.wav_b64) {
      normalized.type = 'audio_chunk';
      normalized.wav_b64 = normalized.wav_b64 || frameAudio.wav_b64;
      normalized.sample_rate = normalized.sample_rate || frameAudio.sample_rate;
    }
  }
  return normalized;
};

const applyTextEvent = (event) => {
  const delta = typeof event.delta === 'string'
    ? event.delta
    : (typeof event.text === 'string' ? event.text : '');
  const snapshot = typeof event.snapshot === 'string' ? event.snapshot : '';
  const eventId = typeof event.event_id === 'string' && event.event_id
    ? event.event_id
    : 'default';
  const current = String(responseText.value || '').replace(/\r/g, '');

  if (snapshot) {
    const next = snapshot.replace(/\r/g, '').trim();
    const sameEvent = currentTextEventId === eventId;
    const prev = sameEvent ? currentTextEventSnapshot : '';
    if (!next) return;
    if (sameEvent && prev && next.startsWith(prev)) {
      const suffix = next.slice(prev.length);
      if (suffix) {
        responseText.value = `${current}${suffix}`;
      }
      currentTextEventSnapshot = next;
      return;
    }
    if (sameEvent && prev && prev.startsWith(next)) {
      return;
    }
    responseText.value = current ? `${current}\n\n${next}` : next;
    currentTextEventId = eventId;
    currentTextEventSnapshot = next;
    return;
  }

  if (delta) {
    responseText.value = `${current}${delta.replace(/\r/g, '')}`;
    currentTextEventId = eventId;
    currentTextEventSnapshot = `${currentTextEventSnapshot}${delta.replace(/\r/g, '')}`;
  }
};

const consumeEvents = async (sessionId) => {
  eventsController = new AbortController();
  const resp = await fetch(realtimeEventsUrl(sessionId), {
    method: 'GET',
    signal: eventsController.signal
  });
  if (!resp.ok || !resp.body) {
    throw new Error(`事件流连接失败: ${resp.status}`);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let pending = '';

  const processBlock = (block) => {
    let eventType = 'message';
    const data = [];
    for (const raw of block.split('\n')) {
      const line = raw.trim();
      if (line.startsWith('event:')) eventType = line.slice(6).trim();
      if (line.startsWith('data:')) data.push(line.slice(5).trim());
    }
    const rawData = data.join('\n');
    if (!rawData || eventType === 'heartbeat') return false;
    const rawPayload = JSON.parse(rawData);
    // stage_timing carries custom fields that normalizeEventPayload would drop,
    // so handle it directly off the raw payload.
    if (rawPayload && rawPayload.type === 'stage_timing') {
      applyStageTiming(rawPayload);
      return false;
    }
    const payload = normalizeEventPayload(rawPayload, eventType);
    if (!payload) return false;
    if (payload.type === 'error') throw new Error(String(payload.error || rawData));
    if (payload.type === 'status' && typeof payload.status === 'string') {
      setStatus(payload.status, 'running');
    }
    if (payload.type === 'assistant_text') {
      applyTextEvent(payload);
    }
    if (payload.type === 'audio_chunk_pcm' && payload.pcm_b64) {
      recordT2wTrace(payload);
      const samples = decodePcm16Base64(payload.pcm_b64, payload.num_channels || 1);
      recordOutputChunk(samples, Number(payload.sample_rate) || TTS_SAMPLE_RATE);
    }
    if (payload.type === 'audio_chunk' && payload.wav_b64) {
      const decoded = decodeWavBase64(payload.wav_b64);
      recordOutputChunk(decoded.samples, decoded.sampleRate);
    }
    return payload.type === 'done';
  };

  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    pending += decoder.decode(chunk.value, { stream: true }).replace(/\r/g, '');
    let idx = pending.indexOf('\n\n');
    while (idx >= 0) {
      const block = pending.slice(0, idx);
      pending = pending.slice(idx + 2);
      if (processBlock(block)) return;
      idx = pending.indexOf('\n\n');
    }
  }
  if (pending.trim()) processBlock(pending);
};

const createSession = async () => {
  const cfg = normalizeTestConfig();
  const resp = await fetch(realtimeStartUrl(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      start_speak_factor: cfg.startSpeakFactor,
      start_listen_factor: cfg.startListenFactor,
      end_speak_factor: cfg.endSpeakFactor,
      prompt_voice: selectedVoice.value || DEFAULT_VOICE,
      tts_chunk_size: 1,
      infer_window_ms: cfg.inferWindowMs,
      stage_timing_log: false
    })
  });
  if (!resp.ok) {
    throw new Error(`创建 session 失败 ${resp.status}: ${await resp.text()}`);
  }
  const payload = await resp.json();
  if (!payload.session_id) throw new Error('创建 session 返回缺少 session_id');
  return payload.session_id;
};

const sendChunk = async (sessionId, samples, index) => {
  const controller = new AbortController();
  chunkControllers.push(controller);
  try {
    const blob = wavBlobFromMono(samples, TARGET_SAMPLE_RATE);
    const sentEpochMs = Date.now();
    if (index === 1) firstSentEpochMs = sentEpochMs;
    const resp = await fetch(realtimeChunkUrl(sessionId), {
      method: 'POST',
      headers: {
        'Content-Type': 'audio/wav',
        'X-Client-Chunk-Sent-Epoch-Ms': String(sentEpochMs)
      },
      body: blob,
      signal: controller.signal
    });
    if (!resp.ok) {
      throw new Error(`发送分片 ${index} 失败 ${resp.status}: ${await resp.text()}`);
    }
  } finally {
    chunkControllers = chunkControllers.filter((item) => item !== controller);
  }
};

const stopSession = async (sessionId) => {
  if (!sessionId) return;
  await fetch(realtimeStopUrl(sessionId), { method: 'POST' }).catch(() => {});
};

const runTest = async () => {
  if (!selectedFile.value || running.value) return;
  running.value = true;
  stopRequested = false;
  activeSessionId = '';
  resetOutputs();
  setStatus('解码输入音频...', 'running');

  try {
    const decoded = await decodeAudioFileToMono16k(selectedFile.value);
    const inputSamples = decoded.samples;
    if (!inputSamples || inputSamples.length < MIN_SEGMENT_SAMPLES) {
      throw new Error('音频过短或解码失败');
    }
    const cfg = normalizeTestConfig();
    const effectivePaddingSec = Math.max(0, Number(tailPaddingSec.value) || 0);
    const sentSamples = appendTailSilence(inputSamples, effectivePaddingSec);
    const segments = splitSamples(sentSamples, cfg.chunkMs);
    if (segments.length === 0) throw new Error('无法切分为可发送分片');
    totalChunks.value = segments.length;
    fileMeta.durationSec = (inputSamples.length / TARGET_SAMPLE_RATE).toFixed(2);
    fileMeta.sentDurationSec = (sentSamples.length / TARGET_SAMPLE_RATE).toFixed(2);
    fileMeta.paddingSec = effectivePaddingSec.toFixed(2);
    fileMeta.sourceRate = String(decoded.sourceRate);
    fileMeta.chunkCount = segments.length;
    const inputWav = wavBlobFromMono(sentSamples, TARGET_SAMPLE_RATE);
    inputSha1.value = await sha1Hex(await inputWav.arrayBuffer());

    archive = createArchive(sentSamples);
    activeSessionId = await createSession();
    archive.startPerfMs = performance.now();
    archive.startEpochMs = Date.now();   // 相对时刻零点（音频开始发送）
    setStatus(`session=${activeSessionId.slice(0, 8)} 已创建，开始发送`, 'running');
    const eventTask = consumeEvents(activeSessionId);

    for (let i = 0; i < segments.length; i += 1) {
      if (stopRequested) break;
      const plannedAt = archive.startPerfMs + i * cfg.chunkMs;
      const waitMs = plannedAt - performance.now();
      if (waitMs > 0) await sleep(waitMs);
      await sendChunk(activeSessionId, segments[i], i + 1);
      sentChunks.value = i + 1;
    }

    setStatus('输入发送完成，等待后端 flush...', 'running');
    await stopSession(activeSessionId);
    await eventTask;
    finalizeArchive();
    setStatus(stopRequested ? '已停止并生成当前结果' : '测试完成', stopRequested ? 'idle' : 'done');
  } catch (err) {
    console.error(err);
    setStatus(err?.message || String(err), 'error');
    if (activeSessionId) {
      await stopSession(activeSessionId);
    }
  } finally {
    running.value = false;
    activeSessionId = '';
    eventsController = null;
    chunkControllers = [];
  }
};

const stopTest = async () => {
  stopRequested = true;
  setStatus('停止中...', 'running');
  for (const controller of chunkControllers) controller.abort();
  await stopSession(activeSessionId);
};

const onFileChange = (event) => {
  const file = event.target.files && event.target.files[0] ? event.target.files[0] : null;
  selectedFile.value = file;
  fileMeta.name = file ? file.name : '';
  fileMeta.durationSec = '0.00';
  fileMeta.sentDurationSec = '0.00';
  fileMeta.paddingSec = (Math.max(0, Number(tailPaddingSec.value) || 0)).toFixed(2);
  fileMeta.sourceRate = '-';
  fileMeta.chunkCount = 0;
  resetOutputs();
  setStatus(file ? '文件已选择，等待发送' : '等待选择音频文件', 'idle');
};

const loadVoices = async () => {
  try {
    const resp = await fetch(realtimeVoicesUrl());
    if (!resp.ok) return;
    const payload = await resp.json();
    const voices = Array.isArray(payload.voices)
      ? payload.voices.filter((item) => item?.id && item?.label).map((item) => ({ id: item.id, label: item.label }))
      : [];
    if (voices.length > 0) {
      voiceOptions.value = voices;
      const backendDefault = typeof payload.default_voice === 'string' ? payload.default_voice : DEFAULT_VOICE;
      if (!voices.some((item) => item.id === selectedVoice.value)) {
        selectedVoice.value = voices.some((item) => item.id === backendDefault) ? backendDefault : voices[0].id;
      }
    }
  } catch (err) {
    console.warn('加载参考音色失败，使用本地列表:', err);
  }
};

const downloadUrl = (url, filename) => {
  if (!url) return;
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
};

onMounted(() => {
  document.title = '权重批量对比测试';
  loadVoices();
});

onBeforeUnmount(() => {
  stopRequested = true;
  for (const controller of chunkControllers) controller.abort();
  if (eventsController) eventsController.abort();
  revokeOutputUrls();
});
</script>

<style scoped>
.wt-page {
  min-height: 100vh;
  background: #f6f8fb;
  color: #0f172a;
  padding: 24px;
}

.wt-header {
  max-width: 1280px;   /* ≈34cm 宽，方便固定尺寸展示 */
  margin: 0 auto 18px;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
}

.wt-header h1 {
  margin: 0;
  font-size: 28px;
}

.wt-header p {
  margin: 8px 0 0;
  color: #475569;
}

.wt-link {
  color: #0369a1;
  font-weight: 700;
  text-decoration: none;
}

.wt-layout {
  max-width: 1280px;   /* ≈34cm 宽，整体固定不再铺满超宽屏 */
  margin: 0 auto;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  align-items: start;
  gap: 16px;
}

.wt-panel {
  background: #ffffff;
  border: 1px solid rgba(15, 23, 42, 0.1);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
  min-width: 0;
  align-self: start;
}

/* 布局：上排 3 列（控制 / 统计 / 各阶段用时），下排 2 列（响应文本 / 输出音频）。
   整页自然滚动，不限制 panel 高度。 */
.wt-control-panel { grid-column: 1; grid-row: 1; }
.wt-stats-panel   { grid-column: 2; grid-row: 1; }
.wt-stage-panel   { grid-column: 3; grid-row: 1; }
.wt-text-panel    { grid-column: 1 / 3; grid-row: 2; }   /* 下排左：跨前两列 */
.wt-audio-panel   { grid-column: 3; grid-row: 2; }       /* 下排右 */

/* 各阶段用时三列表格：每行一个阶段，列 = 最近一轮 / 累计平均 / 说话平均 */
.wt-stage-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  margin-top: 4px;
}
.wt-stage-table th,
.wt-stage-table td {
  padding: 5px 6px;
  text-align: right;
  border-bottom: 1px solid rgba(15, 23, 42, 0.06);
  white-space: nowrap;
}
.wt-stage-table th {
  font-weight: 700;
  color: #64748b;
  border-bottom: 1px solid rgba(15, 23, 42, 0.15);
}
.wt-stage-table th:first-child,
.wt-stage-table td:first-child {
  text-align: left;
  color: #334155;
  font-weight: 600;
}
.wt-stage-table td {
  font-variant-numeric: tabular-nums;
  color: #0f172a;
}
.wt-stage-meta {
  margin-top: 6px;
  font-size: 11px;
  color: #94a3b8;
}
.wt-tag-spk {
  display: inline-block;
  margin: 0 4px;
  padding: 0 5px;
  border-radius: 6px;
  background: #fde68a;
  color: #92400e;
  font-size: 10px;
}

.wt-t2w-hint {
  color: #64748b;
  margin-left: 4px;
}

/* token2wav 单次合成 RTF>1（合成跟不上）高亮，便于和卡顿对照 */
.wt-t2w-slow {
  color: #b91c1c;
  font-weight: 600;
}

.wt-section-title {
  font-size: 16px;
  font-weight: 800;
  margin-bottom: 12px;
}

.wt-subtitle,
.wt-audio-title {
  font-size: 13px;
  font-weight: 800;
  color: #334155;
  margin-bottom: 8px;
}

.wt-field {
  display: grid;
  grid-template-columns: 92px 1fr;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
  font-size: 13px;
  font-weight: 700;
  color: #334155;
}

.wt-field input,
.wt-field select {
  min-width: 0;
  height: 38px;
  border: 1px solid rgba(15, 23, 42, 0.16);
  border-radius: 8px;
  background: #fff;
  padding: 0 10px;
}

.wt-field input[type="file"] {
  height: auto;
  padding: 8px;
}

.wt-grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

.wt-grid-2 .wt-field {
  grid-template-columns: 1fr;
  gap: 6px;
}

.wt-actions {
  display: flex;
  gap: 10px;
  margin: 14px 0;
}

.wt-primary,
.wt-secondary {
  border: none;
  border-radius: 8px;
  padding: 10px 14px;
  font-weight: 800;
  cursor: pointer;
}

.wt-primary {
  background: #0284c7;
  color: #fff;
}

.wt-secondary {
  background: #e2e8f0;
  color: #0f172a;
}

.wt-primary:disabled,
.wt-secondary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.wt-status {
  border-radius: 8px;
  padding: 10px;
  font-size: 13px;
  font-weight: 700;
  background: #f1f5f9;
  color: #334155;
  word-break: break-word;
}

.wt-status.running {
  background: #dbeafe;
  color: #1d4ed8;
}

.wt-status.done {
  background: #dcfce7;
  color: #166534;
}

.wt-status.error {
  background: #fee2e2;
  color: #b91c1c;
}

.wt-file-meta {
  margin-top: 12px;
  display: grid;
  gap: 6px;
  color: #475569;
  font-size: 12px;
  line-height: 1.5;
}

.wt-file-meta code {
  font-size: 11px;
  word-break: break-all;
}

.wt-stat-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}

.wt-stat {
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 10px;
  padding: 12px;
  background: #f8fafc;
  display: grid;
  gap: 6px;
}

.wt-stat span {
  font-size: 12px;
  color: #64748b;
  font-weight: 700;
}

.wt-stat strong {
  font-size: 20px;
}

.wt-stutter-list {
  max-height: 130px;
  overflow: auto;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 8px;
  padding: 8px;
  background: #f8fafc;
}

.wt-stutter-row,
.wt-muted {
  font-size: 12px;
  color: #475569;
  line-height: 1.6;
}

.wt-response-text {
  min-height: 220px;
  max-height: 380px;
  margin: 0;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13px;
  line-height: 1.6;
  color: #0f172a;
  background: #f8fafc;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 8px;
  padding: 12px;
}

.wt-audio-block {
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
}

.wt-audio-block audio {
  grid-column: 1 / -1;
  width: 100%;
}

/* 响应式：超宽屏 4 列，逐级降到 1 列；整页滚动，不限制 panel 高度 */
/* 中等屏：收起为单列，按 DOM 顺序自然堆叠，整页滚动 */
@media (max-width: 1280px) {
  .wt-layout {
    grid-template-columns: 1fr;
  }

  .wt-control-panel,
  .wt-stats-panel,
  .wt-stage-panel,
  .wt-text-panel,
  .wt-audio-panel {
    grid-column: auto;
    grid-row: auto;
  }

  .wt-header {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
