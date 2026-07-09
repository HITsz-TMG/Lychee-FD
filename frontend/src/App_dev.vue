<template>
  <div class="rt-app-container">
    <header class="rt-glass-header">
      <div class="rt-header-content">
        <img src="@/assets/hit.png" class="rt-logo-img" alt="logo" />
        <span class="rt-header-title">lychee-FD</span>
        <span class="rt-badge">Real-time audio call</span>
      </div>
      <nav class="rt-header-links">
        <a class="rt-header-link" :href="EXTERNAL_LINKS.github" target="_blank" rel="noopener noreferrer">GitHub</a>
        <a class="rt-header-link" :href="EXTERNAL_LINKS.paper" target="_blank" rel="noopener noreferrer">Paper</a>
        <a class="rt-header-link" :href="EXTERNAL_LINKS.docs" target="_blank" rel="noopener noreferrer">Project</a>
      </nav>
    </header>
    <!-- ===== 模型加载栏 ===== -->
    <section class="rt-model-bar">
      <div class="rt-model-row">
        <span class="rt-model-label">Model</span>
        <select class="rt-model-select" v-model="devSelectedPresetIdx" :disabled="devLoading">
          <option v-for="(p, idx) in devPresets" :key="idx" :value="idx">
            {{ p.name }}
          </option>
          <option :value="-1">Custom path</option>
        </select>
        <input
          v-if="devSelectedPresetIdx === -1"
          class="rt-model-input"
          v-model="devCustomModelPath"
          placeholder="/abs/path/to/checkpoint"
          :disabled="devLoading"
        />
        <select class="rt-model-select rt-model-select-sm" v-model="devCustomBackendType" :disabled="devLoading">
          <option value="vllm">vllm</option>
          <option value="hf">hf</option>
        </select>
        <select class="rt-model-select rt-model-select-sm" v-model="devCustomMode" :disabled="devLoading">
          <option value="stable">stable</option>
          <option value="aggressive">aggressive</option>
        </select>
        <button class="rt-model-btn rt-model-btn-primary" :disabled="devLoading" @click="devLoadSelectedModel">
          {{ devLoading ? 'Loading...' : 'Load / Switch' }}
        </button>
        <button class="rt-model-btn rt-model-btn-ghost" :disabled="devLoading" @click="devRefreshStatus">
          Refresh
        </button>
        <span class="rt-model-state" :class="'rt-model-state-' + (devStatus.state || 'idle')">
          {{ devStatus.state || 'idle' }}
        </span>
        <button class="rt-model-meta-toggle" @click="modelMetaExpanded = !modelMetaExpanded">
          {{ modelMetaExpanded ? 'Hide details' : 'Details' }}
        </button>
      </div>
      <div v-if="modelMetaExpanded" class="rt-model-meta">
        <span v-if="devStatus.alive">pid={{ devStatus.pid }} · backend={{ devStatus.backend_type }} · model={{ devStatus.model_path || '-' }}</span>
        <span v-else>Backend offline</span>
        <span v-if="devStatus.last_error" class="rt-model-meta-error">⚠ {{ devStatus.last_error }}</span>
      </div>
    </section>
    <!-- ===== /模型加载栏 ===== -->
    <div v-if="startTalkErrorSummary" class="rt-global-error-strip">
      <div class="rt-global-error-content">
        <div class="rt-global-error-title">Latest failure</div>
        <div class="rt-global-error-text">{{ startTalkErrorSummary }}</div>
        <div v-if="startTalkErrorHint" class="rt-global-error-hint">{{ startTalkErrorHint }}</div>
      </div>
      <button class="rt-global-error-dismiss" @click="dismissStartTalkError">Dismiss</button>
    </div>

    <main class="rt-main-workspace">
      <div class="rt-workspace-grid">
        <!-- 左侧：智能体响应区 -->
        <section class="rt-panel rt-agent-panel">
          <div class="rt-panel-header">
            <span class="rt-panel-title">Agent Response</span>
          </div>
          <div class="rt-orb-stage">
            <span class="rt-call-timer">{{ callTimerDisplay }}</span>
            <div class="rt-orb" :class="'rt-orb-' + callState">
              <span class="rt-orb-core"></span>
              <span class="rt-orb-ring"></span>
            </div>
            <div class="rt-orb-status">{{ callStateText }}</div>
          </div>
          <div class="rt-subtitle-panel" ref="chatMessagesRef">
            <template v-if="aiReplyText || realtimeTextAudioGatePlaceholderVisible">
              <div
                v-if="aiReplyText"
                class="markdown-body rt-subtitle-text"
                :class="{ 'is-streaming': isAiSpeaking }"
                v-html="renderMarkdown(aiReplyText)"
              ></div>
              <div v-else-if="realtimeTextAudioGatePlaceholderVisible" class="rt-subtitle-thinking">
                <span class="rt-typing-indicator">Processing<span>.</span><span>.</span><span>.</span></span>
              </div>
            </template>
            <div v-else class="rt-subtitle-idle">
              <template v-if="isTalking">
                <span class="rt-typing-indicator">Waiting for response<span>.</span><span>.</span><span>.</span></span>
              </template>
              <template v-else>
                <p class="rt-welcome-title">👋 Welcome to Lychee-FD</p>
                <p class="rt-welcome-desc">Start a realtime full-duplex voice conversation with natural turn-taking and interruption support.</p>
                <div v-if="randomShowcaseCases.length > 0" class="rt-welcome-suggestions">
                  <div class="rt-welcome-suggestions-row">
                    <button
                      v-for="item in randomShowcaseCases"
                      :key="item.id"
                      type="button"
                      class="rt-suggestion-chip"
                      :disabled="showcaseStarting"
                      @click="startShowcaseCase(item)"
                    >
                      <span class="rt-suggestion-chip-icon">💬</span>
                      <span class="rt-suggestion-chip-text">{{ item.label }}</span>
                    </button>
                  </div>
                  <button
                    type="button"
                    class="rt-suggestion-refresh"
                    :disabled="showcaseStarting"
                    @click="refreshRandomShowcase"
                  >
                    <span class="rt-suggestion-refresh-icon">↻</span>
                    <span>Shuffle</span>
                  </button>
                </div>
              </template>
            </div>
          </div>
        </section>

        <!-- 右侧：交互历史区 -->
        <aside class="rt-panel rt-history-panel">
          <div class="rt-panel-header">
            <span class="rt-panel-title">Interaction History</span>
          </div>
          <div class="rt-history-content">
            <template v-if="alignedSessionHistory.length > 0">
              <div
                v-for="(item, idx) in alignedSessionHistory"
                :key="item.id"
                class="rt-history-entry"
                :class="{ 'is-latest': idx === 0 }"
              >
                <div class="rt-history-head">
                  <span class="rt-history-round">#{{ alignedSessionHistory.length - idx }}</span>
                  <span v-if="idx === 0" class="rt-history-latest-tag">Latest</span>
                  <span class="rt-history-time">{{ item.startedAt }}</span>
                </div>
                <div class="rt-history-meta">
                  <span>Session {{ item.sessionId || 'unknown' }}</span>
                  <span>{{ item.sampleRate }} Hz</span>
                </div>
                <div class="rt-history-audio-block">
                  <div class="rt-history-label">Aligned input audio<span class="rt-history-dur">{{ item.inputSec }}s</span></div>
                  <audio :src="item.inputUrl" controls preload="metadata"></audio>
                </div>
                <div class="rt-history-audio-block">
                  <div class="rt-history-label">Aligned output audio<span class="rt-history-dur">{{ item.outputSec }}s</span></div>
                  <audio :src="item.outputUrl" controls preload="metadata"></audio>
                </div>
                <div v-if="item.rawOutputUrl" class="rt-history-audio-block">
                  <div class="rt-history-label">Raw output chunks<span class="rt-history-dur">{{ item.rawOutputSec }}s</span></div>
                  <audio :src="item.rawOutputUrl" controls preload="metadata"></audio>
                </div>
              </div>
            </template>
            <div v-else class="rt-history-empty">
              <span class="rt-history-empty-icon">🗂️</span>
              <span>No interaction history yet</span>
              <span class="rt-history-empty-hint">Input and output audio for each turn will appear here after a call.</span>
            </div>
          </div>
        </aside>

        <!-- 最右：Debug 抽屉 -->
        <aside v-if="DEBUG_DRAWER_VISIBLE" class="rt-debug-drawer" :class="{ 'is-open': debugDrawerOpen }">
          <button
            class="rt-debug-rail"
            :class="{ 'has-error': startTalkErrorSummary || devStatus.last_error }"
            @click="debugDrawerOpen = !debugDrawerOpen"
          >
            <span class="rt-debug-rail-arrow">{{ debugDrawerOpen ? '›' : '‹' }}</span>
            <span class="rt-debug-rail-text">Debug</span>
            <span v-if="startTalkErrorSummary || devStatus.last_error" class="rt-debug-rail-dot"></span>
          </button>
          <div v-if="debugDrawerOpen" class="rt-debug-body">
            <div class="rt-debug-toolbar">
              <span class="rt-debug-toolbar-title">Debug</span>
              <button type="button" class="rt-debug-copy" @click="copyDebugInfo">复制</button>
            </div>

            <div v-if="startTalkErrorSummary" class="rt-debug-alert">
              <div class="rt-debug-alert-title">最近一次失败</div>
              <div class="rt-debug-alert-text">{{ startTalkErrorSummary }}</div>
              <div v-if="startTalkErrorHint" class="rt-debug-alert-hint">{{ startTalkErrorHint }}</div>
            </div>

            <div class="rt-debug-group">
              <div class="rt-debug-group-title">连接 / 状态</div>
              <div class="rt-debug-row"><span>通话状态</span><span>{{ callState }}</span></div>
              <div class="rt-debug-row"><span>连接</span><span>{{ connectionStatus || '—' }}</span></div>
              <div class="rt-debug-row"><span>输入采样率</span><span>{{ captureSampleRateDisplay }} Hz</span></div>
            </div>

            <div class="rt-debug-group">
              <div class="rt-debug-group-title">队列</div>
              <div class="rt-debug-row"><span>上传 / 优先</span><span>{{ segmentQueueDepth }} / {{ prioritySegmentQueueDepth }}</span></div>
              <div class="rt-debug-row"><span>已发送分片</span><span>{{ segmentsSentCount }}</span></div>
              <div class="rt-debug-row"><span>后端轮次 / 窗口</span><span>#{{ backendQueueRoundId || '--' }} / {{ formatNullableMs(backendQueueInferWindowMs) }}</span></div>
              <div class="rt-debug-row"><span>取前队列</span><span>{{ formatNullableMs(backendQueueBeforeMs) }}</span></div>
              <div class="rt-debug-row"><span>消费 / 剩余</span><span>{{ formatNullableMs(backendQueueConsumedMs) }} / {{ formatNullableMs(backendQueueAfterMs) }}</span></div>
              <div v-if="priorityUploadInProgress" class="rt-debug-row"><span>插队发送中</span><span>{{ priorityUploadSourceName || '测试音频' }}</span></div>
            </div>

            <div class="rt-debug-group">
              <div class="rt-debug-group-title">延迟（样本 {{ realtimeLatencySampleCount }}）</div>
              <div class="rt-debug-metrics">
                <span>前端送片→后端emit: {{ formatLatencyTriplet(realtimePreEmitClientLastMs, realtimePreEmitClientAvgMs, realtimePreEmitClientP95Ms) }}</span>
                <span>后端收片→后端emit: {{ formatLatencyTriplet(realtimePreEmitServerLastMs, realtimePreEmitServerAvgMs, realtimePreEmitServerP95Ms) }}</span>
                <span>后端队列延迟: {{ formatLatencyTriplet(realtimeServerQueueDelayLastMs, realtimeServerQueueDelayAvgMs, realtimeServerQueueDelayP95Ms) }}</span>
                <span>后端emit→前端收包: {{ formatLatencyTriplet(realtimeLatencyReceiveLastMs, realtimeLatencyReceiveAvgMs, realtimeLatencyReceiveP95Ms) }}</span>
                <span>后端emit→预计开播: {{ formatLatencyTriplet(realtimeLatencyAudibleLastMs, realtimeLatencyAudibleAvgMs, realtimeLatencyAudibleP95Ms) }}</span>
                <span>播放排队积压: {{ formatLatencyTriplet(realtimePlaybackBacklogLastMs, realtimePlaybackBacklogAvgMs, realtimePlaybackBacklogP95Ms) }}</span>
              </div>
            </div>

            <div class="rt-debug-group">
              <div class="rt-debug-group-title">Duplex / VAD</div>
              <div class="rt-debug-row"><span>控制参数</span><span>SS={{ startSpeakFactor }} / SL={{ startListenFactor }} / END={{ endSpeakFactor }}</span></div>
              <div class="rt-debug-prob-row">
                <span class="rt-prob-tag">S-L: {{ formatProbability(slProbability) }}</span>
                <span class="rt-prob-tag">S-S: {{ formatProbability(ssProbability) }}</span>
              </div>
              <div v-if="lsStartEvents.length > 0" class="rt-transition-list">
                <span v-for="item in lsStartEvents" :key="item.index" class="rt-transition-tag">
                  #{{ item.index }} L-&gt;S {{ formatAudioRelMs(item.audioMs) }}
                  <template v-if="item.roundId !== null">round={{ item.roundId }}</template>
                  <template v-if="item.chunkIdx !== null">chunk={{ item.chunkIdx }}</template>
                </span>
              </div>
            </div>

            <div class="rt-debug-group">
              <div class="rt-debug-group-title">模型</div>
              <div class="rt-debug-row"><span>state</span><span>{{ devStatus.state || 'idle' }}</span></div>
              <div class="rt-debug-row"><span>pid</span><span>{{ devStatus.pid || '—' }}</span></div>
              <div class="rt-debug-row"><span>backend</span><span>{{ devStatus.backend_type || '—' }}</span></div>
              <div class="rt-debug-row rt-debug-row-path"><span>model</span><span>{{ devStatus.model_path || '—' }}</span></div>
              <div v-if="devStatus.last_error" class="rt-debug-row rt-debug-row-error"><span>error</span><span>{{ devStatus.last_error }}</span></div>
            </div>

            <div class="rt-debug-group">
              <div class="rt-debug-group-title">调试操作</div>
              <button
                class="rt-debug-action"
                :disabled="!(isTalking && realtimeInputMode === 'mic')"
                @click="clickRealtimeAudioInjectButton"
              >🎵 发送测试音频</button>
            </div>
          </div>
        </aside>
      </div>
    </main>

    <footer class="rt-toolbar">
      <input
        id="hidden-realtime-audio-inject-input"
        type="file"
        accept="audio/*"
        @change="onRealtimeAudioFileChange"
        hidden
      />

      <!-- 弹层点击外部关闭背板 -->
      <div
        v-if="voicePopoverOpen || settingsPopoverOpen"
        class="rt-popover-backdrop"
        @click="closeAllPopovers"
      ></div>

      <!-- 音色设置 -->
      <div class="rt-tool-slot">
        <transition name="rt-pop">
          <div v-if="voicePopoverOpen" class="rt-popover rt-voice-popover">
            <div class="rt-popover-title">Voice Settings</div>
            <div v-if="isTalking" class="rt-popover-hint">Voice changes take effect on the next turn.</div>
            <div class="rt-voice-upload-row">
              <span class="rt-voice-upload-text">Upload custom voice</span>
              <button type="button" class="rt-voice-upload-btn" @click="openUserVoiceUpload">Select file</button>
            </div>
            <div class="rt-voice-list">
              <div
                v-for="(v, idx) in voiceOptions"
                :key="v.id"
                class="rt-voice-item"
                :class="{ 'is-selected': selectedVoice === v.id }"
              >
                <button
                  type="button"
                  class="rt-voice-select"
                  :disabled="isTalking"
                  @click="selectedVoice = v.id"
                >
                  <span class="rt-voice-icon" :style="voiceIconStyle(idx)">{{ voiceInitial(v.label) }}</span>
                  <span class="rt-voice-name">{{ v.label }}</span>
                  <span v-if="selectedVoice === v.id" class="rt-voice-check">✓</span>
                </button>
                <button
                  type="button"
                  class="rt-voice-preview"
                  :class="{ 'is-playing': voicePreviewId === v.id }"
                  @click="previewVoice(v)"
                >
                  {{ voicePreviewId === v.id ? 'Stop' : 'Preview' }}
                </button>
              </div>
            </div>
          </div>
        </transition>
        <button
          type="button"
          class="rt-tool-btn rt-tool-btn-voice"
          :class="{ 'is-active': voicePopoverOpen }"
          @click="toggleVoicePopover"
        >
          + Voice
        </button>
      </div>

      <!-- 开始 / 挂断 -->
      <button
        type="button"
        class="rt-call-btn"
        :class="isTalking ? 'is-hangup' : 'is-start'"
        :disabled="showcaseStarting"
        @click="isTalking ? endTalk() : startTalk()"
      >
        {{ isTalking ? 'End Call' : (showcaseStarting ? 'Connecting...' : 'Start Call') }}
      </button>

      <!-- 对话设置 -->
      <div class="rt-tool-slot">
        <transition name="rt-pop">
          <div v-if="settingsPopoverOpen" class="rt-popover rt-settings-popover">
            <div class="rt-popover-title">Conversation Settings</div>
            <label class="rt-setting-row">
              <span>Start factor</span>
              <input type="number" min="0.1" max="5" step="0.05" v-model.number="startSpeakFactor" :disabled="isTalking" />
            </label>
            <label class="rt-setting-row">
              <span>Listen factor</span>
              <input type="number" min="0.1" max="5" step="0.05" v-model.number="startListenFactor" :disabled="isTalking" />
            </label>
            <label class="rt-setting-row">
              <span>End factor</span>
              <input type="number" min="0.1" max="5" step="0.05" v-model.number="endSpeakFactor" :disabled="isTalking" />
            </label>
            <label class="rt-setting-row">
              <span>Playback speed</span>
              <span class="rt-setting-range">
                <input type="range" min="0.5" max="1.8" step="0.1" v-model.number="playbackRate" />
                <span class="rt-setting-range-val">{{ playbackRate.toFixed(1) }}x</span>
              </span>
            </label>
          </div>
        </transition>
        <button
          type="button"
          class="rt-tool-btn rt-tool-btn-settings"
          :class="{ 'is-active': settingsPopoverOpen }"
          @click="toggleSettingsPopover"
        >
          Settings
        </button>
      </div>
    </footer>
  </div>
</template>




<script setup>
import { computed, ref, watch, nextTick, onMounted, onBeforeUnmount } from 'vue'
import hljs from 'highlight.js'
import 'highlight.js/styles/atom-one-dark.css' // 可选样式，比如 GitHub 风格
import newFavicon from '@/assets/hit.png'  // 引入新的 favicon 图标
import MarkdownIt from 'markdown-it'         // 引入 Markdown 解析库
import WAVEncoder from 'wav-encoder'         // 引入 WAV 编码器
import { ElNotification } from 'element-plus'  // 引入 Element Plus 的通知组件


// =========================================================================
// REDESIGN: 单页实时通话工作台 —— 新增的 UI 状态/常量（不改动实时引擎逻辑）
// =========================================================================
// External resource links.
const EXTERNAL_LINKS = {
  github: 'https://github.com/HITsz-TMG/Lychee-FD',
  paper: 'https://arxiv.org/pdf/2607.06540',
  docs: 'https://hitsz-tmg.github.io/Lychee-FD/',
};

// 底部弹层 / Debug 抽屉 / 模型详情 显隐状态
const voicePopoverOpen = ref(false);
const settingsPopoverOpen = ref(false);
const debugDrawerOpen = ref(false);
const DEBUG_DRAWER_VISIBLE = false;
const modelMetaExpanded = ref(false);

const toggleVoicePopover = () => {
  voicePopoverOpen.value = !voicePopoverOpen.value;
  if (voicePopoverOpen.value) {
    settingsPopoverOpen.value = false;
  } else {
    stopVoicePreview();
  }
};
const toggleSettingsPopover = () => {
  settingsPopoverOpen.value = !settingsPopoverOpen.value;
  if (settingsPopoverOpen.value) voicePopoverOpen.value = false;
};
const closeAllPopovers = () => {
  voicePopoverOpen.value = false;
  settingsPopoverOpen.value = false;
  stopVoicePreview();
};

// 音色图标占位：彩色圆点 + 首字母（后续可替换为真实音色 icon）
const VOICE_ICON_PALETTE = [
  '#378ADD', '#1D9E75', '#7C6FF0', '#E08A3C', '#D94F8C',
  '#3CA9B8', '#C24B4A', '#5B8DEF', '#2FA36B', '#9B6BD6',
];
const voiceIconStyle = (idx) => {
  const color = VOICE_ICON_PALETTE[idx % VOICE_ICON_PALETTE.length];
  return { background: `${color}22`, color };
};
const voiceInitial = (label) => {
  const s = String(label || '').trim();
  return s ? s.slice(0, 1).toUpperCase() : '?';
};

// 音色试听（占位：映射到本地参考音色样本目录，后续可替换为真实试听样本）
// TODO: 替换为真实试听样本——可改 VOICE_PREVIEW_BASE / VOICE_PREVIEW_MAP，
// 或在后端 /api/realtime/voices 的返回项中提供 preview_url / sample 字段（优先生效）。
const VOICE_PREVIEW_BASE = '/clone_24k_mono/';
const VOICE_PREVIEW_MAP = {
  default_female: 'default_female.wav',
  default_male: 'default_male.wav',
  leijun: 'leijun_voice.wav',
  guodegang: 'gudegang_voice.wav',
  jay: 'jay.wav',
  huge: 'huge.wav',
  hanhong: 'hanhong.wav',
  nailong: 'nailong.wav',
  kenan: 'kenan.wav',
  haimian: 'haimian.wav',
  dengziqi: 'dengziqi.wav',
  liyunlong: 'liyunlong.wav',
  new_female: 'new_female_voice.wav',
  female: 'female_voice.wav',
  news_male: 'news_male_voice.wav',
};
const voicePreviewId = ref('');
let voicePreviewAudio = null;
const resolveVoicePreviewUrl = (voice) => {
  if (!voice) return '';
  if (voice.preview_url) return voice.preview_url;
  if (voice.sample) return voice.sample;
  const file = VOICE_PREVIEW_MAP[voice.id];
  return file ? `${VOICE_PREVIEW_BASE}${file}` : '';
};
const stopVoicePreview = () => {
  if (voicePreviewAudio) {
    try { voicePreviewAudio.pause(); } catch (_e) { /* ignore */ }
    voicePreviewAudio = null;
  }
  voicePreviewId.value = '';
};
const previewVoice = (voice) => {
  if (!voice) return;
  if (voicePreviewId.value === voice.id) {
    stopVoicePreview();
    return;
  }
  stopVoicePreview();
  const url = resolveVoicePreviewUrl(voice);
  if (!url) {
    if (typeof ElNotification === 'function') {
      ElNotification({ type: 'info', title: 'No preview sample', message: `Voice "${voice.label}" has no preview sample.`, duration: 2500 });
    }
    return;
  }
  const audio = new Audio(url);
  voicePreviewAudio = audio;
  voicePreviewId.value = voice.id;
  audio.onended = () => { if (voicePreviewId.value === voice.id) stopVoicePreview(); };
  audio.onerror = () => {
    if (typeof ElNotification === 'function') {
      ElNotification({ type: 'warning', title: 'Preview failed', message: `Unable to load sample: ${url}`, duration: 3000 });
    }
    stopVoicePreview();
  };
  audio.play().catch((err) => { console.warn('Voice preview playback failed:', err); stopVoicePreview(); });
};

// 通话计时器（实际启停在 startTalk/endTalk 中接入，见后续任务）
const callElapsedSec = ref(0);
let callTimerInterval = null;
const callTimerDisplay = computed(() => {
  const total = Math.max(0, Math.floor(callElapsedSec.value));
  const mm = String(Math.floor(total / 60)).padStart(2, '0');
  const ss = String(total % 60).padStart(2, '0');
  return `${mm}:${ss}`;
});
const stopCallTimer = () => {
  if (callTimerInterval) {
    clearInterval(callTimerInterval);
    callTimerInterval = null;
  }
};
const startCallTimer = () => {
  stopCallTimer();
  callElapsedSec.value = 0;
  callTimerInterval = setInterval(() => {
    callElapsedSec.value += 1;
  }, 1000);
};
const resetCallTimer = () => {
  stopCallTimer();
  callElapsedSec.value = 0;
};

// AI 是否正在播放（响应式镜像，赋值接入在播放 worklet 回调/调度复位处）
const isAiSpeaking = ref(false);

// 派生通话状态，驱动状态球与文案
const callState = computed(() => {
  if (typeof isTalking === 'undefined' || !isTalking.value) {
    const hasHistory = (alignedSessionHistory.value && alignedSessionHistory.value.length > 0) || !!aiReplyText.value;
    return hasHistory ? 'ended' : 'idle';
  }
  if (showcaseStarting.value) return 'connecting';
  if (connectionStatusClass.value === 'disconnected') return 'error';
  if (isAiSpeaking.value) return 'ai_speaking';
  if (realtimeTextAudioGatePlaceholderVisible.value) return 'thinking';
  return 'listening';
});
const CALL_STATE_TEXT = {
  idle: 'Idle',
  connecting: 'Connecting...',
  listening: 'Listening',
  user_speaking: 'Processing speech',
  thinking: 'Processing',
  ai_speaking: 'Responding',
  ended: 'Call ended',
  error: 'Connection error. Please retry.',
};
const callStateText = computed(() => CALL_STATE_TEXT[callState.value] || '');

// 复制 Debug 抽屉中的调试信息快照到剪贴板
const copyDebugInfo = async () => {
  const snapshot = {
    时间: new Date().toISOString(),
    通话状态: callState.value,
    连接: connectionStatus.value,
    输入采样率: captureSampleRateDisplay.value,
    队列: {
      上传队列: segmentQueueDepth.value,
      优先队列: prioritySegmentQueueDepth.value,
      已发送分片: segmentsSentCount.value,
      后端轮次: backendQueueRoundId.value,
      推理窗口ms: backendQueueInferWindowMs.value,
      取前队列ms: backendQueueBeforeMs.value,
      消费ms: backendQueueConsumedMs.value,
      剩余ms: backendQueueAfterMs.value,
    },
    延迟样本: realtimeLatencySampleCount.value,
    概率: { 'S-L': slProbability.value, 'S-S': ssProbability.value },
    控制参数: { SS: startSpeakFactor.value, SL: startListenFactor.value, END: endSpeakFactor.value },
    LS事件: lsStartEvents.value,
    模型: {
      state: devStatus.value.state,
      pid: devStatus.value.pid,
      backend: devStatus.value.backend_type,
      model: devStatus.value.model_path,
      last_error: devStatus.value.last_error || null,
    },
    最近失败: startTalkErrorSummary.value || null,
  };
  const text = JSON.stringify(snapshot, null, 2);
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    if (typeof ElNotification === 'function') {
      ElNotification({ type: 'success', title: '已复制调试信息', duration: 2000 });
    }
  } catch (err) {
    console.error('复制调试信息失败:', err);
    if (typeof ElNotification === 'function') {
      ElNotification({ type: 'warning', title: '复制失败', message: String(err && err.message || err) });
    }
  }
};


// =========================================================================
// DEV: lychee_fd.controller / 模型切换
// =========================================================================
const buildAdminBase = () => {
  const explicit = window.__UNIMOE_ADMIN_BASE__;
  if (typeof explicit === 'string' && explicit.trim()) {
    return explicit.replace(/\/+$/, '');
  }
  return `${window.location.protocol}//${window.location.host}`;
};
const buildAdminUrl = (path) => `${buildAdminBase()}${path}`;
const adminToken = () => (window.__UNIMOE_ADMIN_TOKEN__ || '');

const devFetchAdmin = async (path, options = {}) => {
  const headers = Object.assign({}, options.headers || {});
  if (adminToken()) headers['x-admin-token'] = adminToken();
  if (options.body && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }
  const resp = await fetch(buildAdminUrl(path), { ...options, headers });
  let body;
  try { body = await resp.json(); } catch (_e) { body = null; }
  if (!resp.ok) {
    const detail = (body && (body.detail || body.error)) || resp.statusText;
    throw new Error(`admin ${path} ${resp.status}: ${detail}`);
  }
  return body;
};

const devPresets = ref([]);
const devSelectedPresetIdx = ref(0);
const devCustomModelPath = ref('');
const devCustomBackendType = ref('vllm');
const devCustomMode = ref('stable');
const devLoading = ref(false);
const devStatus = ref({ state: 'idle', alive: false, pid: null, model_path: null });
let devStatusTimer = null;
const streamChunkMs = ref(200);
const inferWindowMs = ref(400);

const parsePresetMs = (value) => {
  if (value === undefined || value === null || String(value).trim() === '') {
    return null;
  }
  const n = Number(value);
  return Number.isFinite(n) ? Math.round(n) : null;
};

const clampPresetMs = (value, min = 50, max = 2000) => {
  const n = parsePresetMs(value);
  if (n === null) return null;
  return Math.max(min, Math.min(max, n));
};

const applyPresetFrontendOverrides = (preset) => {
  if (!preset || typeof preset !== 'object') {
    return;
  }
  const extraEnv = preset.extra_env && typeof preset.extra_env === 'object'
    ? preset.extra_env
    : {};
  const frontendConfig = preset.frontend_config && typeof preset.frontend_config === 'object'
    ? preset.frontend_config
    : {};

  const inferMs = clampPresetMs(
    frontendConfig.infer_window_ms
      ?? extraEnv.LYCHEEFD_REALTIME_INFER_WINDOW_MS
      ?? extraEnv.REALTIME_INFER_WINDOW_MS,
    160,
    2000
  );
  if (inferMs !== null) {
    inferWindowMs.value = inferMs;
  }

  const uploadMs = clampPresetMs(
    frontendConfig.upload_chunk_ms
      ?? frontendConfig.stream_chunk_ms
      ?? extraEnv.LYCHEEFD_REALTIME_UPLOAD_CHUNK_MS
      ?? extraEnv.LYCHEEFD_FRONTEND_UPLOAD_CHUNK_MS
      ?? extraEnv.LYCHEEFD_STREAM_CHUNK_MS
      ?? extraEnv.VUE_APP_REALTIME_STREAM_CHUNK_MS,
    50,
    2000
  );
  if (uploadMs !== null) {
    streamChunkMs.value = uploadMs;
  }
};

const devRefreshStatus = async () => {
  try {
    const s = await devFetchAdmin('/admin/status');
    devStatus.value = s || {};
    if (s && s.model_path && Array.isArray(devPresets.value)) {
      const idx = devPresets.value.findIndex(p => p.model_path === s.model_path);
      if (idx >= 0) {
        devSelectedPresetIdx.value = idx;
        applyPresetFrontendOverrides(devPresets.value[idx]);
      }
      if (s.backend_type) devCustomBackendType.value = s.backend_type;
      if (s.mode) devCustomMode.value = s.mode;
    }
  } catch (e) {
    devStatus.value = { state: 'error', alive: false, last_error: String(e.message || e) };
  }
};

const devLoadPresets = async () => {
  try {
    const data = await devFetchAdmin('/admin/presets');
    devPresets.value = (data && data.presets) || [];
    if (devSelectedPresetIdx.value >= 0) {
      applyPresetFrontendOverrides(devPresets.value[devSelectedPresetIdx.value]);
    }
  } catch (e) {
    if (typeof ElNotification === 'function') {
      ElNotification({ type: 'warning', title: 'Failed to load presets', message: String(e.message || e) });
    }
  }
};

const devLoadSelectedModel = async () => {
  if (devLoading.value) return;
  let payload;
  if (devSelectedPresetIdx.value === -1) {
    if (!devCustomModelPath.value) {
      if (typeof ElNotification === 'function') {
        ElNotification({ type: 'warning', title: 'Enter a model path' });
      }
      return;
    }
    payload = {
      model_path: devCustomModelPath.value.trim(),
      backend_type: devCustomBackendType.value,
      mode: devCustomMode.value,
      wait_ready: true,
    };
  } else {
    const p = devPresets.value[devSelectedPresetIdx.value];
    if (!p) return;
    applyPresetFrontendOverrides(p);
    payload = {
      model_path: p.model_path,
      backend_type: devCustomBackendType.value || p.backend_type || 'vllm',
      mode: devCustomMode.value || p.mode || 'stable',
      extra_env: p.extra_env || {},
      wait_ready: true,
    };
  }

  try {
    if (typeof endTalk === 'function' && typeof isTalking !== 'undefined' && isTalking.value) {
      endTalk();
    }
  } catch (_e) { /* ignore */ }

  devLoading.value = true;
  devStatus.value = { ...devStatus.value, state: 'starting' };
  try {
    const resp = await devFetchAdmin('/admin/restart', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    if (resp && resp.ok) {
      if (typeof ElNotification === 'function') {
        ElNotification({ type: 'success', title: 'Model loaded', message: payload.model_path });
      }
    } else {
      if (typeof ElNotification === 'function') {
        ElNotification({
          type: 'error', title: 'Model load failed',
          message: (resp && resp.error) || 'Unknown error',
        });
      }
    }
  } catch (e) {
    if (typeof ElNotification === 'function') {
      ElNotification({ type: 'error', title: 'Model load failed', message: String(e.message || e) });
    }
  } finally {
    devLoading.value = false;
    await devRefreshStatus();
  }
};

watch(devSelectedPresetIdx, (idx) => {
  if (idx >= 0) {
    applyPresetFrontendOverrides(devPresets.value[idx]);
  }
});

const refreshVoicesOnWindowFocus = () => {
  loadVoices();
};

const refreshVoicesOnVisibilityChange = () => {
  if (typeof document !== 'undefined' && document.visibilityState === 'visible') {
    loadVoices();
  }
};

onMounted(async () => {
  await devLoadPresets();
  await devRefreshStatus();
  devStatusTimer = setInterval(devRefreshStatus, 5000);
  loadVoices();
  if (typeof window !== 'undefined') {
    window.addEventListener('focus', refreshVoicesOnWindowFocus);
  }
  if (typeof document !== 'undefined') {
    document.addEventListener('visibilitychange', refreshVoicesOnVisibilityChange);
  }
});
onBeforeUnmount(() => {
  if (devStatusTimer) { clearInterval(devStatusTimer); devStatusTimer = null; }
  if (typeof window !== 'undefined') {
    window.removeEventListener('focus', refreshVoicesOnWindowFocus);
  }
  if (typeof document !== 'undefined') {
    document.removeEventListener('visibilitychange', refreshVoicesOnVisibilityChange);
  }
});
// =========================================================================
// /DEV
// =========================================================================



// ==========================================
// 实时通话 (Function 6) 
// ==========================================
const isTalking = ref(false);
const mediaStream = ref(null);
const captureAudioContext = ref(null);
const connectionStatus = ref('');
const connectionStatusClass = ref('');
const aiReplyText = ref(''); // 用于在页面上流式展示 AI 的文字回复
const realtimeTextAudioGatePlaceholderVisible = ref(false);
const startTalkErrorSummary = ref('');
const startTalkErrorHint = ref('');
const playbackRate = ref(1.0);
const alignedSessionHistory = ref([]);
const showActiveWorkspace = computed(() => (
  isTalking.value || !!aiReplyText.value || alignedSessionHistory.value.length > 0
));
const realtimeInputMode = ref('mic');
const showcaseStarting = ref(false);
const selectedShowcaseCaseId = ref('');
const captureSampleRateDisplay = ref('-');
const segmentQueueDepth = ref(0);
const prioritySegmentQueueDepth = ref(0);
const segmentsSentCount = ref(0);
const priorityUploadInProgress = ref(false);
const priorityUploadSourceName = ref('');
// 可调推理参数
const startSpeakFactor = ref(1.2);
const startListenFactor = ref(1.0);
const endSpeakFactor = ref(1.0);
const backendQueueBeforeMs = ref(null);   // 后端取材料前队列积压(ms)
const backendQueueAfterMs = ref(null);    // 消费后剩余(ms)
const backendQueueConsumedMs = ref(null); // 本轮消费(ms)
const backendQueueInferWindowMs = ref(null);
const backendQueueRoundId = ref(null);
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
const lsStartEvents = ref([]);
const START_TALK_ERROR_STORAGE_KEY = 'fd_realtime_start_error_v1';

const TARGET_SAMPLE_RATE = 16000;
const CONTROL_TOKEN_HZ = 25;
const MIN_SEGMENT_SAMPLES = 800;
const MAX_UPLOAD_QUEUE_DEPTH = 8;
const REALTIME_TEXT_AUDIO_GATE_FALLBACK_MS = 3500;
const REALTIME_TEXT_TYPEWRITER_INTERVAL_MS = 28;
const REALTIME_TEXT_TYPEWRITER_CHARS_PER_TICK = 1;
const REALTIME_FILLER_WORD_AUDIO_BASE_PATH = '/filler_words/';
const REALTIME_FILLER_WORD_AUDIO_MANIFEST_PATH = `${REALTIME_FILLER_WORD_AUDIO_BASE_PATH}manifest.json`;
const SHOWCASE_CASE_BASE_PATH = '/input_cases/showcase/';
const SHOWCASE_TAIL_SILENCE_SEC = 15;
const showcaseCases = [
  { id: 'spoken', label: '口语', filename: 'spoken.m4a' },
  { id: 'schedule', label: '安排时间', filename: 'schedule.m4a' },
  { id: 'flight', label: '航班', filename: 'flight.m4a' },
  { id: 'shandong_food', label: '山东菜推荐', filename: 'shandong_food.m4a' },
  { id: 'family_blessing', label: '亲戚祝福', filename: 'family_blessing.m4a' },
  { id: 'weekend_relax', label: '周末放松', filename: 'weekend_relax.m4a' },
  { id: 'info_entropy', label: '信息论熵的解释', filename: '信息论熵的解释.wav' },
  { id: 'make_cake', label: '做蛋糕', filename: '做蛋糕.wav' },
  { id: 'write_email', label: '写邮件', filename: '写邮件.wav' },
  { id: 'travel_plan', label: '旅游规划', filename: '旅游规划.wav' },
  { id: 'shenzhen_travel', label: '深圳旅游', filename: '深圳旅游.wav' },
  { id: 'continuous_interrupt', label: '连续打断输入', filename: '连续打断输入.wav' },
  { id: 'hotel_recommendation', label: '酒店推荐', filename: '酒店推荐.wav' }
];

// 欢迎页随机展示 3 个样例（点击直接开始；通话开始后整块自然隐藏）
const RANDOM_SHOWCASE_DISPLAY_COUNT = 3;
const pickRandomShowcaseCases = () => {
  const pool = showcaseCases.slice();
  if (pool.length <= RANDOM_SHOWCASE_DISPLAY_COUNT) return pool;
  // Fisher-Yates 取前 N
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, RANDOM_SHOWCASE_DISPLAY_COUNT);
};
const randomShowcaseCases = ref(pickRandomShowcaseCases());
const refreshRandomShowcase = () => {
  // 至少保证和上一组不完全一致
  if (showcaseCases.length > RANDOM_SHOWCASE_DISPLAY_COUNT) {
    const prevIds = randomShowcaseCases.value.map((item) => item.id).join('|');
    for (let attempt = 0; attempt < 5; attempt++) {
      const next = pickRandomShowcaseCases();
      if (next.map((item) => item.id).join('|') !== prevIds) {
        randomShowcaseCases.value = next;
        return;
      }
    }
  }
  randomShowcaseCases.value = pickRandomShowcaseCases();
};
const isEnabledFlag = (value) => /^(1|true|yes|on)$/i.test(String(value || '').trim());
const REALTIME_ALIGN_TRACE_DOWNLOAD_ENABLED = (() => {
  const runtimeFlag = window.__UNIMOE_REALTIME_ALIGN_TRACE_DOWNLOAD__;
  if (runtimeFlag !== undefined && runtimeFlag !== null && String(runtimeFlag).trim() !== '') {
    return isEnabledFlag(runtimeFlag);
  }
  const envFlag = typeof process !== 'undefined' && process.env
    ? process.env.VUE_APP_REALTIME_ALIGN_TRACE_DOWNLOAD
    : '';
  return isEnabledFlag(envFlag);
})();

let captureSourceNode = null;
let captureProcessorNode = null;
let captureSilentGainNode = null;
let uploadInterval = null;

let pendingCaptureChunks = [];
let pendingCaptureSamples = 0;

let uploadQueue = [];
let priorityUploadQueue = [];
let queueProcessing = false;
let lsStartEventFingerprints = new Set();
let nextUploadAllowedAtMs = 0;
let segmentSeqId = 0;
let currentAlignedArchive = null;
const realtimeSessionId = ref('');
const realtimeStoppingExpected = ref(false);

const activeRequestControllers = new Set();
const gradioAudioQueue = [];
let gradioAudioPlaying = false;
let currentGradioAudio = null;
let currentShowcaseInputAudio = null;
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

const formatNullableMs = (v) => formatLatencyMs(v);

const formatLatencyTriplet = (lastMs, avgMs, p95Ms) => {
  return `last ${formatLatencyMs(lastMs)} | avg ${formatLatencyMs(avgMs)} | p95 ${formatLatencyMs(p95Ms)}`;
};

const normalizeRealtimeNumber = (value, fallback, min, max, digits = 3) => {
  const n = toFiniteNumber(value);
  const bounded = clamp(n === null ? fallback : n, min, max);
  const scale = 10 ** digits;
  return Math.round(bounded * scale) / scale;
};

const normalizeRealtimeInt = (value, fallback, min, max) => {
  const n = toFiniteNumber(value);
  return Math.round(clamp(n === null ? fallback : n, min, max));
};

const normalizeRealtimeConfig = () => {
  const config = {
    startSpeakFactor: normalizeRealtimeNumber(startSpeakFactor.value, 1.2, 0.1, 5),
    startListenFactor: normalizeRealtimeNumber(startListenFactor.value, 1.0, 0.1, 5),
    endSpeakFactor: normalizeRealtimeNumber(endSpeakFactor.value, 1.0, 0.1, 5),
    streamChunkMs: normalizeRealtimeInt(streamChunkMs.value, 200, 50, 2000),
    inferWindowMs: normalizeRealtimeInt(inferWindowMs.value, 400, 160, 2000)
  };
  startSpeakFactor.value = config.startSpeakFactor;
  startListenFactor.value = config.startListenFactor;
  endSpeakFactor.value = config.endSpeakFactor;
  streamChunkMs.value = config.streamChunkMs;
  inferWindowMs.value = config.inferWindowMs;
  return config;
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
  backendQueueBeforeMs.value = null;
  backendQueueAfterMs.value = null;
  backendQueueConsumedMs.value = null;
  backendQueueInferWindowMs.value = null;
  backendQueueRoundId.value = null;
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

const updateBackendQueueMetricsFromStageTiming = (payload) => {
  if (!payload || payload.type !== 'stage_timing') {
    return;
  }
  const roundId = toFiniteNumber(payload.round_id);
  if (roundId !== null) {
    backendQueueRoundId.value = Math.round(roundId);
  }
  const q = payload.queue && typeof payload.queue === 'object' ? payload.queue : null;
  if (!q) {
    return;
  }
  const beforeMs = toFiniteNumber(q.pending_before_ms);
  const afterMs = toFiniteNumber(q.pending_after_ms);
  const consumedMs = toFiniteNumber(q.consumed_ms);
  const inferMs = toFiniteNumber(q.infer_window_ms);
  if (beforeMs !== null) backendQueueBeforeMs.value = beforeMs;
  if (afterMs !== null) backendQueueAfterMs.value = afterMs;
  if (consumedMs !== null) backendQueueConsumedMs.value = consumedMs;
  if (inferMs !== null) backendQueueInferWindowMs.value = inferMs;
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

const formatAudioRelMs = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? `+${(n / 1000).toFixed(2)}s` : '--';
};

const resetRealtimeProbabilities = () => {
  slProbability.value = null;
  ssProbability.value = null;
};

// Listening 阶段后端默认不会推送 S-L / S-S，导致 Debug 面板在模型说话前一直显示 "--"。
// 这里在进入通话时给出 listening 状态的先验值（继续监听=1, 开始说=0），
// 让用户连接成功就能看到非空数字；后端推送首个真值会立刻覆盖。
const primeRealtimeProbabilitiesIfEmpty = () => {
  if (slProbability.value === null || slProbability.value === undefined) {
    slProbability.value = 1;
  }
  if (ssProbability.value === null || ssProbability.value === undefined) {
    ssProbability.value = 0;
  }
};

const resetRealtimeTransitionTimings = () => {
  lsStartEvents.value = [];
  lsStartEventFingerprints = new Set();
};

const normalizeStateName = (value) => {
  if (typeof value !== 'string') return '';
  return value.trim().toLowerCase();
};

const stateEventAudioMs = (event) => {
  if (!event || typeof event !== 'object') return null;
  const explicitFields = [
    event.aligned_audio_ms,
    event.processed_audio_ms,
    event.audio_ms,
    event.audio_time_ms,
    event.relative_audio_ms,
    event.audio_rel_ms
  ];
  for (const value of explicitFields) {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  const tokenFields = [event.pos, event.chunk, event.chunk_idx, event.chunk_pos];
  for (const value of tokenFields) {
    const n = Number(value);
    if (Number.isFinite(n)) return (n / CONTROL_TOKEN_HZ) * 1000;
  }
  return null;
};

const stateEventNumberField = (event, fields) => {
  if (!event || typeof event !== 'object') return null;
  for (const field of fields) {
    const n = Number(event[field]);
    if (Number.isFinite(n)) return n;
  }
  return null;
};

const recordStateTransitionTiming = (event) => {
  const fromState = normalizeStateName(event?.from);
  const toState = normalizeStateName(event?.to);
  if (fromState !== 'l' || toState !== 's') return;

  const audioMs = stateEventAudioMs(event);
  const roundId = stateEventNumberField(event, ['round_id', 'infer_round']);
  const chunkIdx = stateEventNumberField(event, ['chunk_idx', 'chunk', 'chunk_pos', 'pos']);
  const fingerprint = [
    roundId ?? '',
    chunkIdx ?? '',
    Number.isFinite(Number(audioMs)) ? Number(audioMs).toFixed(3) : ''
  ].join(':');
  if (lsStartEventFingerprints.has(fingerprint)) return;
  lsStartEventFingerprints.add(fingerprint);
  lsStartEvents.value.push({
    index: lsStartEvents.value.length + 1,
    audioMs,
    roundId,
    chunkIdx
  });
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
    return 'Unknown error';
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
  let hint = 'Open the browser console for detailed error output.';
  if (name === 'NotAllowedError' || /permission|denied|notallowed/.test(raw)) {
    hint = 'Microphone permission was denied. Allow microphone access in site permissions and retry.';
  } else if (name === 'NotFoundError' || /notfound|device.*not found|no input device/.test(raw)) {
    hint = 'No microphone device was detected. Check the system recording device.';
  } else if (name === 'NotReadableError' || /notreadable|device in use|hardware|could not start audio source/.test(raw)) {
    hint = 'The microphone may be used by another application. Close it and retry.';
  } else if (name === 'SecurityError' || /insecure|secure context/.test(raw)) {
    hint = 'The page is not running in a secure context. Use localhost or HTTPS.';
  } else if (/failed to fetch|networkerror|load failed|err_connection|cors|502|504/.test(raw)) {
    hint = 'Frontend-backend request failed. Common causes are proxy interception, CORS, or an unreachable port.';
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
let realtimeStructuredTextEvents = [];
let realtimeReplyStartMarkerPending = false;
const REALTIME_REPLY_START_MARKER = '💬';

const normalizeRealtimeText = (text) => {
  if (typeof text !== 'string') return '';
  return text.replace(/\r/g, '').trim();
};

const normalizeRealtimeDisplayText = (text) => {
  if (typeof text !== 'string') return '';
  return text.replace(/\r/g, '');
};

const formatRealtimeReplyStart = (text, options = {}) => {
  const clean = typeof text === 'string' ? text : '';
  if (!clean) return '';
  if (options?.marker !== true) {
    return clean;
  }
  return clean.startsWith(REALTIME_REPLY_START_MARKER)
    ? clean
    : `${REALTIME_REPLY_START_MARKER} ${clean}`;
};

const markRealtimeReplyStartIfListenToSpeak = (fromState, toState) => {
  const from = typeof fromState === 'string' ? fromState.toLowerCase() : '';
  const to = typeof toState === 'string' ? toState.toLowerCase() : '';
  if (to === 's' && from === 'l') {
    realtimeReplyStartMarkerPending = true;
  }
};

const takeRealtimeReplyStartMarker = () => {
  if (!realtimeReplyStartMarkerPending) return false;
  realtimeReplyStartMarkerPending = false;
  return true;
};

let realtimeTextAudioGateTimer = null;
let realtimeTextAudioGate = {
  active: false,
  opened: true,
  key: '',
  roundId: null,
  visibleBeforeGate: '',
  hasPendingText: false
};
let realtimeTextTypewriterTimer = null;
let realtimeTextTypewriterTarget = '';

const getRealtimePayloadRoundId = (payload) => {
  if (!payload || typeof payload !== 'object') return null;
  const direct = toFiniteNumber(payload.round_id);
  if (direct !== null) return Number(direct);
  const inferRound = toFiniteNumber(payload.infer_round);
  if (inferRound !== null) return Number(inferRound);
  const eventId = normalizeRealtimeText(String(payload.event_id || payload?.frame_text?.event_id || ''));
  const match = eventId.match(/^(\d+):/);
  if (!match) return null;
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) ? parsed : null;
};

const getRealtimeTextGateKey = (payload) => {
  if (!payload || typeof payload !== 'object') return '';
  const eventId = normalizeRealtimeText(String(payload.event_id || payload?.frame_text?.event_id || ''));
  if (eventId) return `event:${eventId}`;
  const roundId = getRealtimePayloadRoundId(payload);
  if (roundId !== null) return `round:${roundId}`;
  return '';
};

const clearRealtimeTextAudioGateTimer = () => {
  if (realtimeTextAudioGateTimer) {
    clearTimeout(realtimeTextAudioGateTimer);
    realtimeTextAudioGateTimer = null;
  }
};

const resetRealtimeTextAudioGate = () => {
  clearRealtimeTextAudioGateTimer();
  realtimeTextAudioGate = {
    active: false,
    opened: true,
    key: '',
    roundId: null,
    visibleBeforeGate: '',
    hasPendingText: false
  };
  realtimeTextAudioGatePlaceholderVisible.value = false;
};

const composeRealtimeReply = () => {
  let structured = '';
  for (const item of realtimeStructuredTextEvents) {
    const text = normalizeRealtimeDisplayText(item?.text);
    if (!text.trim()) continue;
    const part = formatRealtimeReplyStart(text, { marker: !!item.startMarker });
    structured = structured
      ? `${structured}${item.startMarker ? '\n\n' : ''}${part}`
      : part;
  }
  if (structured) {
    return structured;
  }
  const committed = normalizeRealtimeDisplayText(realtimeCommittedReply.value);
  const live = normalizeRealtimeDisplayText(realtimeLiveReply);
  const hasCommitted = !!committed.trim();
  const hasLive = !!live.trim();
  if (hasCommitted && hasLive) {
    return `${committed}${live}`;
  }
  return hasCommitted ? committed : (hasLive ? live : '');
};

const clearRealtimeTextTypewriter = () => {
  if (realtimeTextTypewriterTimer) {
    clearTimeout(realtimeTextTypewriterTimer);
    realtimeTextTypewriterTimer = null;
  }
};

const setRealtimeReplyImmediate = (text) => {
  const next = typeof text === 'string' ? text : '';
  clearRealtimeTextTypewriter();
  realtimeTextTypewriterTarget = next;
  aiReplyText.value = next;
};

const stepRealtimeTextTypewriter = () => {
  realtimeTextTypewriterTimer = null;
  const target = realtimeTextTypewriterTarget || '';
  const current = aiReplyText.value || '';
  if (current === target) {
    return;
  }
  if (!target.startsWith(current)) {
    setRealtimeReplyImmediate(target);
    return;
  }
  const targetChars = Array.from(target);
  const currentLen = Array.from(current).length;
  const nextLen = Math.min(
    targetChars.length,
    currentLen + REALTIME_TEXT_TYPEWRITER_CHARS_PER_TICK
  );
  aiReplyText.value = targetChars.slice(0, nextLen).join('');
  if (nextLen < targetChars.length) {
    realtimeTextTypewriterTimer = setTimeout(
      stepRealtimeTextTypewriter,
      REALTIME_TEXT_TYPEWRITER_INTERVAL_MS
    );
  }
};

const animateRealtimeReplyTo = (text) => {
  const target = typeof text === 'string' ? text : '';
  const current = aiReplyText.value || '';
  if (!target || !target.startsWith(current) || current.length > target.length) {
    setRealtimeReplyImmediate(target);
    return;
  }
  realtimeTextTypewriterTarget = target;
  if (current === target || realtimeTextTypewriterTimer) {
    return;
  }
  realtimeTextTypewriterTimer = setTimeout(
    stepRealtimeTextTypewriter,
    REALTIME_TEXT_TYPEWRITER_INTERVAL_MS
  );
};

const renderRealtimeReply = () => {
  const nextReply = composeRealtimeReply();
  const visibleBeforeGate = normalizeRealtimeText(realtimeTextAudioGate.visibleBeforeGate);
  const nextVisibleReply = normalizeRealtimeText(nextReply);
  if (
    realtimeTextAudioGate.active &&
    !realtimeTextAudioGate.opened &&
    realtimeTextAudioGate.hasPendingText &&
    nextReply &&
    nextVisibleReply !== visibleBeforeGate
  ) {
    setRealtimeReplyImmediate(realtimeTextAudioGate.visibleBeforeGate || '');
    realtimeTextAudioGatePlaceholderVisible.value = true;
    return;
  }
  animateRealtimeReplyTo(nextReply);
  realtimeTextAudioGatePlaceholderVisible.value = false;
};

const openRealtimeTextAudioGate = () => {
  if (!realtimeTextAudioGate.active && realtimeTextAudioGate.opened) {
    realtimeTextAudioGatePlaceholderVisible.value = false;
    return;
  }
  clearRealtimeTextAudioGateTimer();
  realtimeTextAudioGate = {
    ...realtimeTextAudioGate,
    active: false,
    opened: true,
    visibleBeforeGate: '',
    hasPendingText: false
  };
  realtimeTextAudioGatePlaceholderVisible.value = false;
  renderRealtimeReply();
};

const scheduleRealtimeTextAudioGateFallback = () => {
  clearRealtimeTextAudioGateTimer();
  realtimeTextAudioGateTimer = setTimeout(() => {
    openRealtimeTextAudioGate();
  }, REALTIME_TEXT_AUDIO_GATE_FALLBACK_MS);
};

const beginRealtimeTextAudioGate = (payload = {}, { force = false } = {}) => {
  const key = getRealtimeTextGateKey(payload);
  const roundId = getRealtimePayloadRoundId(payload);
  const sameKey = !!key && key === realtimeTextAudioGate.key;
  const sameRound = roundId !== null && roundId === realtimeTextAudioGate.roundId;

  if (!force && realtimeTextAudioGate.opened && (sameKey || sameRound)) {
    return;
  }

  if (!force && realtimeTextAudioGate.active && !realtimeTextAudioGate.opened && (sameKey || sameRound)) {
    if (roundId !== null && realtimeTextAudioGate.roundId === null) {
      realtimeTextAudioGate.roundId = roundId;
    }
    return;
  }

  clearRealtimeTextAudioGateTimer();
  realtimeTextAudioGate = {
    active: true,
    opened: false,
    key: key || `speech:${Date.now()}`,
    roundId,
    visibleBeforeGate: aiReplyText.value || '',
    hasPendingText: false
  };
  realtimeTextAudioGatePlaceholderVisible.value = false;
};

const markRealtimeTextPendingBeforeAudio = (payload = {}) => {
  const key = getRealtimeTextGateKey(payload);
  const roundId = getRealtimePayloadRoundId(payload);
  const sameOpenedKey = !!key && key === realtimeTextAudioGate.key && realtimeTextAudioGate.opened;
  const sameOpenedRound = roundId !== null && roundId === realtimeTextAudioGate.roundId && realtimeTextAudioGate.opened;

  if (!realtimeTextAudioGate.active || realtimeTextAudioGate.opened) {
    if (realtimeLastState === 's' || realtimeLastState === 'b') {
      return;
    }
    beginRealtimeTextAudioGate(payload);
  }

  if (!realtimeTextAudioGate.active || realtimeTextAudioGate.opened) {
    return;
  }

  realtimeTextAudioGate.hasPendingText = true;
  scheduleRealtimeTextAudioGateFallback();
};

const openRealtimeTextAudioGateOnAudio = (payload = {}) => {
  if (!realtimeTextAudioGate.active || realtimeTextAudioGate.opened) {
    return;
  }
  const gateRound = realtimeTextAudioGate.roundId;
  const audioRound = getRealtimePayloadRoundId(payload);
  if (gateRound !== null && audioRound !== null && audioRound < gateRound) {
    return;
  }
  openRealtimeTextAudioGate();
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
  const snapshot = normalizeRealtimeDisplayText(snapshotRaw);
  const snapshotForChecks = normalizeRealtimeText(snapshotRaw);
  if (
    !eventId ||
    !snapshotForChecks ||
    /^\(no text\)$/i.test(snapshotForChecks) ||
    /^\*\*\[[^\]]+\]\*\*\s*generating/i.test(snapshotForChecks) ||
    /^State:\s*/i.test(snapshotForChecks) ||
    /^\*\*\[[^\]]+\]\*\*/.test(snapshotForChecks)
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
      breakBefore: false,
      startMarker: !!existing.startMarker
    };
  } else {
    realtimeStructuredTextEvents.push({
      id: eventId,
      kind: eventKind || 'response',
      text: snapshot,
      seq,
      isFinal,
      resumed,
      breakBefore: false,
      startMarker: takeRealtimeReplyStartMarker()
    });
  }
  realtimeLiveReply = '';
  realtimeCurrentEventText = snapshot;
  renderRealtimeReply();
  return true;
};

const appendRealtimeCommittedChunk = (chunk) => {
  if (typeof chunk !== 'string') return;
  const text = chunk.replace(/\r/g, '');
  if (!text) return;
  const shouldMarkStart = takeRealtimeReplyStartMarker();
  const outputText = formatRealtimeReplyStart(text, { marker: shouldMarkStart });
  const separator = shouldMarkStart && realtimeCommittedReply.value ? '\n\n' : '';
  if (!realtimeCommittedReply.value) {
    realtimeCommittedReply.value = outputText;
  } else {
    realtimeCommittedReply.value = `${realtimeCommittedReply.value}${separator}${outputText}`;
  }
  realtimeLiveReply = '';
  renderRealtimeReply();
};

const appendRealtimeIncrementalEventText = (fullText) => {
  const next = normalizeRealtimeDisplayText(fullText);
  if (!next.trim()) return;
  const prev = normalizeRealtimeDisplayText(realtimeCurrentEventText);
  if (!prev.trim()) {
    appendRealtimeCommittedChunk(next);
    realtimeCurrentEventText = next;
    return;
  }
  if (next.startsWith(prev)) {
    const delta = next.slice(prev.length);
    if (delta) {
      appendRealtimeCommittedChunk(delta);
    }
    realtimeCurrentEventText = next;
    return;
  }
  if (prev.startsWith(next)) {
    // Ignore temporary rollbacks; keep waiting for a longer continuation.
    realtimeCurrentEventText = next;
    return;
  }
  appendRealtimeCommittedChunk(next);
  realtimeCurrentEventText = next;
};

const finalizeRealtimeEventText = (eventText) => {
  const payload = normalizeRealtimeDisplayText(eventText);
  if (!payload.trim()) {
    realtimeCurrentEventText = '';
    return;
  }
  const current = normalizeRealtimeDisplayText(realtimeCurrentEventText);
  if (!current.trim()) {
    appendRealtimeCommittedChunk(payload);
    realtimeCurrentEventText = '';
    return;
  }
  if (payload.startsWith(current)) {
    const delta = payload.slice(current.length);
    if (delta) {
      appendRealtimeCommittedChunk(delta);
    }
    realtimeCurrentEventText = '';
    return;
  }
  if (current.startsWith(payload)) {
    realtimeCurrentEventText = '';
    return;
  }
  appendRealtimeCommittedChunk(payload);
  realtimeCurrentEventText = '';
};

const appendRealtimeEventText = (eventText, signature = '', options = {}) => {
  const allowImmediateDuplicate = options?.allowImmediateDuplicate === true;
  const clean = normalizeRealtimeDisplayText(eventText);
  const cleanKey = normalizeRealtimeText(clean);
  if (!cleanKey) return;
  const dedupeSig = normalizeRealtimeText(signature || clean);
  if (dedupeSig && dedupeSig === realtimeLastEventSignature) {
    return;
  }
  if (!allowImmediateDuplicate && cleanKey === normalizeRealtimeText(realtimeLastCommittedReply)) {
    return;
  }
  const shouldMarkStart = takeRealtimeReplyStartMarker();
  const outputText = formatRealtimeReplyStart(clean, { marker: shouldMarkStart });
  if (realtimeCommittedReply.value) {
    const separator = shouldMarkStart ? '\n\n' : '';
    realtimeCommittedReply.value = `${realtimeCommittedReply.value}${separator}${outputText}`;
  } else {
    realtimeCommittedReply.value = outputText;
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
    renderRealtimeReply();
    return;
  }
  const live = normalizeRealtimeDisplayText(realtimeLiveReply);
  if (live.trim()) {
    appendRealtimeCommittedChunk(live);
  }
  realtimeLiveReply = '';
  realtimeCurrentEventText = '';
};

const resetRealtimeReplyState = () => {
  realtimeCommittedReply.value = '';
  realtimeLiveReply = '';
  realtimeLastCommittedReply = '';
  realtimeLastEventSignature = '';
  realtimeLastState = null;
  realtimeCurrentEventText = '';
  realtimeReplyStartMarkerPending = false;
  realtimeStructuredTextEvents = [];
  setRealtimeReplyImmediate('');
  resetRealtimeTextAudioGate();
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
    let payload = normalizeRealtimeDisplayText(match[2]);
    if (!payload.trim()) continue;
    if (/^\s*generating/i.test(payload)) continue;
    payload = normalizeRealtimeDisplayText(payload.replace(/\s*Audio:\s*[0-9]+[\s\S]*$/i, ''));
    const payloadKey = normalizeRealtimeText(payload);
    if (!payloadKey || /^\(no text\)$/i.test(payloadKey)) continue;
    results.push({
      text: payload,
      signature: `${eventKind}|${payloadKey}`
    });
  }

  if (!matched) return [];
  return results;
};

const handleAssistantRealtimeText = (assistantText) => {
  const raw = normalizeRealtimeDisplayText(assistantText);
  const rawForChecks = normalizeRealtimeText(raw);
  if (!rawForChecks) return;

  const stateChange = parseStateChangeFromAssistant(rawForChecks);
  if (stateChange) {
    const fromState = stateChange.from || realtimeLastState;
    if ((stateChange.to === 's' || stateChange.to === 'b') && (!fromState || fromState === 'l')) {
      beginRealtimeTextAudioGate({ type: 'state_change', from: fromState, to: stateChange.to }, { force: true });
    }
    markRealtimeReplyStartIfListenToSpeak(fromState, stateChange.to);
    if (fromState && fromState !== 'l' && stateChange.to === 'l') {
      openRealtimeTextAudioGate();
      commitRealtimeLiveReply();
    }
    if (stateChange.to === 'l') {
      realtimeCurrentEventText = '';
      realtimeReplyStartMarkerPending = false;
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
      realtimeLastCommittedReply = item.text;
    }
    return;
  }

  if (/\*\*\[[^\]]+\]\*\*/.test(rawForChecks)) {
    realtimeCurrentEventText = '';
    return;
  }

  if (!isAssistantNoiseLine(rawForChecks) && !/^State:\s*/i.test(rawForChecks)) {
    appendRealtimeCommittedChunk(raw);
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
  realtimeInputMode.value = 'mic';
  selectedShowcaseCaseId.value = '';
  showcaseStarting.value = false;
  stopShowcaseInputAudio();
  resetRealtimeProbabilities();
  resetRealtimeTransitionTimings();
  clearStartTalkError({ clearStorage: true });
  clearAlignedHistory();
  resetCallTimer();
  isAiSpeaking.value = false;
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

const renderSmoothedRealtimeOutput = ({
  rawOutputChunks = [],
  outputTraceEvents = [],
  sampleRate = TARGET_SAMPLE_RATE,
  manualRate = 1.0,
  smoothing = REALTIME_PLAYBACK_SMOOTHING
} = {}) => {
  const chunks = Array.isArray(rawOutputChunks) ? rawOutputChunks : [];
  const events = Array.isArray(outputTraceEvents) ? outputTraceEvents : [];
  const sr = Math.max(1, Math.round(Number(sampleRate) || TARGET_SAMPLE_RATE));
  const rateManual = clamp(Number(manualRate) || 1.0, 0.5, 1.8);
  const bufferSize = Math.max(2048, Math.floor(sr * Number(smoothing.ringBufferSec || 12)));
  const ringBuffer = new Float32Array(bufferSize);
  let writePtr = 0;
  let readPtr = 0;
  let isPlaying = false;
  let lastOut = 0.0;
  let fadeVolume = 0.0;
  let smoothedLevel = 0.0;
  let autoRate = 1.0;
  let underrunFrames = 0;
  let peakBufferFrames = 0;
  const autoRateTrace = [];

  const startThreshold = Math.floor(sr * Number(smoothing.startWarmupSec || 0.55));
  const lowWater = Math.floor(sr * Number(smoothing.lowWaterSec || 0.28));
  const highWater = Math.floor(sr * Number(smoothing.highWaterSec || 1.2));
  const lowWaterRate = Number(smoothing.lowWaterRate || 0.985);
  const highWaterRate = Number(smoothing.highWaterRate || 1.055);
  const levelSmoothKeep = Number(smoothing.levelSmoothKeep || 0.95);
  const levelSmoothUpdate = Number(smoothing.levelSmoothUpdate || 0.05);
  const autoRateSmoothKeep = Number(smoothing.autoRateSmoothKeep || 0.985);
  const autoRateSmoothUpdate = Number(smoothing.autoRateSmoothUpdate || 0.015);

  const normalizedChunks = chunks
    .map((chunk) => (chunk instanceof Float32Array ? chunk : new Float32Array(chunk || 0)))
    .filter((chunk) => chunk.length > 0);
  if (!normalizedChunks.length) {
    return {
      samples: new Float32Array(0),
      trace: {
        renderedSamples: 0,
        underrunFrames: 0,
        peakBufferFrames: 0,
        autoRateMin: 1.0,
        autoRateMax: 1.0,
        manualRate: rateManual
      }
    };
  }

  const arrivals = normalizedChunks.map((chunk, idx) => {
    const event = events[idx] || {};
    const n = Number(event.elapsedSamples);
    return {
      chunk,
      elapsedSamples: Math.max(0, Number.isFinite(n) ? Math.round(n) : 0)
    };
  }).sort((a, b) => a.elapsedSamples - b.elapsedSamples);

  const availableFrames = () => {
    let available = writePtr - readPtr;
    if (available < 0) available += bufferSize;
    return available;
  };

  const pushChunk = (chunk) => {
    for (let i = 0; i < chunk.length; i += 1) {
      ringBuffer[writePtr] = chunk[i];
      writePtr = (writePtr + 1) % bufferSize;
      if (writePtr === Math.floor(readPtr)) {
        readPtr = (readPtr + 1) % bufferSize;
      }
    }
    peakBufferFrames = Math.max(peakBufferFrames, availableFrames());
  };

  const readOneFrame = () => {
    const available = availableFrames();
    if (smoothedLevel === 0) {
      smoothedLevel = available;
    } else {
      smoothedLevel = smoothedLevel * levelSmoothKeep + available * levelSmoothUpdate;
    }
    if (!isPlaying && available >= startThreshold) {
      isPlaying = true;
    }
    let targetAutoRate = 1.0;
    if (isPlaying) {
      if (smoothedLevel < lowWater) {
        targetAutoRate = lowWaterRate;
      } else if (smoothedLevel > highWater) {
        targetAutoRate = highWaterRate;
      }
    }
    autoRate = autoRate * autoRateSmoothKeep + targetAutoRate * autoRateSmoothUpdate;
    const speed = clamp(rateManual * autoRate, 0.5, 1.8);

    let out = 0.0;
    if (isPlaying && available >= speed) {
      if (fadeVolume < 1.0) {
        fadeVolume = Math.min(1.0, fadeVolume + 0.0025);
      }
      const readIdx = Math.floor(readPtr);
      const nextIdx = (readIdx + 1) % bufferSize;
      const frac = readPtr - readIdx;
      const rawSample = ringBuffer[readIdx] + (ringBuffer[nextIdx] - ringBuffer[readIdx]) * frac;
      out = rawSample * fadeVolume;
      lastOut = out;
      readPtr = (readPtr + speed) % bufferSize;
    } else {
      if (isPlaying) {
        isPlaying = false;
        fadeVolume = 0.0;
        underrunFrames += 1;
      }
      lastOut *= 0.9;
      out = lastOut;
    }
    autoRateTrace.push(autoRate);
    return out;
  };

  const lastArrival = arrivals.reduce((acc, item) => {
    return Math.max(acc, item.elapsedSamples + item.chunk.length);
  }, 0);
  const totalInputSamples = normalizedChunks.reduce((acc, chunk) => acc + chunk.length, 0);
  const tailSamples = Math.max(startThreshold, Math.floor(sr * 0.5));
  const maxRenderSamples = Math.max(
    lastArrival + tailSamples + startThreshold,
    Math.ceil(totalInputSamples / Math.max(0.5, rateManual * 0.95)) + tailSamples + startThreshold
  );
  const out = new Float32Array(maxRenderSamples);
  let outLen = 0;
  let arrivalIdx = 0;

  for (let renderPos = 0; renderPos < maxRenderSamples; renderPos += 1) {
    while (arrivalIdx < arrivals.length && arrivals[arrivalIdx].elapsedSamples <= renderPos) {
      pushChunk(arrivals[arrivalIdx].chunk);
      arrivalIdx += 1;
    }
    out[outLen] = readOneFrame();
    outLen += 1;
    if (
      arrivalIdx >= arrivals.length &&
      availableFrames() <= 1 &&
      Math.abs(lastOut) < 1e-5 &&
      renderPos > lastArrival + Math.floor(sr * 0.1)
    ) {
      break;
    }
  }

  const minTail = Math.floor(sr * 0.08);
  let trimLen = outLen;
  while (trimLen > minTail && Math.abs(out[trimLen - 1]) < 1e-5) {
    trimLen -= 1;
  }
  trimLen = Math.min(outLen, trimLen + minTail);

  let autoRateMin = 1.0;
  let autoRateMax = 1.0;
  for (const value of autoRateTrace) {
    autoRateMin = Math.min(autoRateMin, value);
    autoRateMax = Math.max(autoRateMax, value);
  }

  return {
    samples: out.slice(0, trimLen),
    trace: {
      renderedSamples: trimLen,
      renderedMs: Number(((trimLen / sr) * 1000).toFixed(3)),
      inputChunkCount: normalizedChunks.length,
      totalInputSamples,
      lastArrivalSamples: lastArrival,
      underrunFrames,
      peakBufferFrames,
      manualRate: rateManual,
      autoRateMin: Number(autoRateMin.toFixed(6)),
      autoRateMax: Number(autoRateMax.toFixed(6)),
      sampleRate: sr
    }
  };
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
    outputTraceSummary: null,
    smoothedOutputTrace: null
  };
};

const exportAlignedArchiveTrace = (archive) => {
  if (!archive) {
    return;
  }
  if (!REALTIME_ALIGN_TRACE_DOWNLOAD_ENABLED) {
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
    streamChunkMs: streamChunkMs.value,
    totals: {
      inputSamples: archive.inputSamples,
      outputSamples: archive.outputSamples,
      rawOutputSamples: archive.rawOutputSamples || 0,
      smoothedOutputSamples: Number(archive.smoothedOutputSamples || 0),
      inputMs: Number(((archive.inputSamples / TARGET_SAMPLE_RATE) * 1000).toFixed(3)),
      outputMs: Number(((archive.outputSamples / TARGET_SAMPLE_RATE) * 1000).toFixed(3)),
      rawOutputMs: Number((((archive.rawOutputSamples || 0) / TARGET_SAMPLE_RATE) * 1000).toFixed(3)),
      smoothedOutputMs: Number((((archive.smoothedOutputSamples || 0) / TARGET_SAMPLE_RATE) * 1000).toFixed(3))
    },
    finalize: archive.outputTraceSummary || null,
    smoothedPlayback: archive.smoothedOutputTrace || null,
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
  const smoothedOutput = renderSmoothedRealtimeOutput({
    rawOutputChunks: currentAlignedArchive.rawOutputChunks || [],
    outputTraceEvents: currentAlignedArchive.outputTraceEvents || [],
    sampleRate: TARGET_SAMPLE_RATE,
    manualRate: playbackRate.value,
    smoothing: REALTIME_PLAYBACK_SMOOTHING
  });
  currentAlignedArchive.smoothedOutputSamples = smoothedOutput.samples.length;
  currentAlignedArchive.smoothedOutputTrace = smoothedOutput.trace;
  const inputBlob = wavBlobFromFloat32(inputSamples, TARGET_SAMPLE_RATE);
  const outputBlob = wavBlobFromFloat32(smoothedOutput.samples, TARGET_SAMPLE_RATE);
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
    outputSec: (smoothedOutput.samples.length / TARGET_SAMPLE_RATE).toFixed(2),
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

const appendTailSilence = (samples, sampleRate, silenceSec) => {
  const source = samples instanceof Float32Array ? samples : new Float32Array(samples || 0);
  const silenceSamples = Math.max(0, Math.round((Number(silenceSec) || 0) * Math.max(1, Number(sampleRate) || TARGET_SAMPLE_RATE)));
  if (silenceSamples <= 0) {
    return source;
  }
  const out = new Float32Array(source.length + silenceSamples);
  out.set(source, 0);
  return out;
};

const buildRealtimeAudioFileSegments = async (file, options = {}) => {
  const decoded = await decodeAudioFileToMonoFloat32(file);
  const inputSamples = decoded.samples;
  if (!inputSamples || inputSamples.length === 0) {
    throw new Error('Audio file is empty or failed to decode.');
  }
  const srcRate = decoded.sampleRate || TARGET_SAMPLE_RATE;
  let samples16k = resampleLinear(inputSamples, srcRate, TARGET_SAMPLE_RATE);
  samples16k = appendTailSilence(samples16k, TARGET_SAMPLE_RATE, options.tailSilenceSec || 0);
  const segments = splitSamplesToRealtimeSegments(samples16k, TARGET_SAMPLE_RATE, streamChunkMs.value);
  if (segments.length === 0) {
    throw new Error('Audio is too short to split into realtime chunks.');
  }
  const paceMs = options.paced
    ? Math.max(1, Number(streamChunkMs.value) || 200)
    : 0;
  return segments.map((samples) => ({
    id: ++segmentSeqId,
    samples,
    source: options.source || 'file',
    paceMs
  }));
};

const enqueueRealtimeAudioSegments = (segments, displayName = '') => {
  if (!Array.isArray(segments) || segments.length === 0) {
    throw new Error('No audio chunks are available to send.');
  }
  if (!isTalking.value || !realtimeSessionId.value) {
    throw new Error('Start a call before sending test audio.');
  }
  priorityUploadQueue = priorityUploadQueue.concat(segments);
  priorityUploadInProgress.value = true;
  priorityUploadSourceName.value = displayName || '';
  syncQueueDepth();
  processUploadQueue().catch((err) => {
    console.error('Priority audio queue failed:', err);
  });
  return segments.length;
};

const enqueuePriorityAudioFile = async (file, options = {}) => {
  if (!isTalking.value || !realtimeSessionId.value) {
    throw new Error('Start a call before sending test audio.');
  }
  const fileSegments = await buildRealtimeAudioFileSegments(file, options);
  return enqueueRealtimeAudioSegments(fileSegments, options.displayName || file.name || '');
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
const buildRealtimeVoicesUrl = () => `${buildGradioApiBase()}/api/realtime/voices`;
const buildUserVoiceUploadUrl = () => {
  const explicit = window.__UNIMOE_USER_VOICE_UPLOAD_URL__;
  if (typeof explicit === 'string' && explicit.trim()) {
    return explicit.trim();
  }
  const port = window.__UNIMOE_USER_VOICE_UPLOAD_PORT__ || '18092';
  const host = window.__UNIMOE_USER_VOICE_UPLOAD_HOST__ || window.location.hostname || 'localhost';
  return `${window.location.protocol}//${host}:${port}/`;
};

const openUserVoiceUpload = () => {
  window.open(buildUserVoiceUploadUrl(), '_blank', 'noopener,noreferrer');
};

// ---- 参考音色选择 ----
const DEFAULT_VOICE = 'default_female';
const VOICE_LABEL_EN = {
  default_female: 'Default Female',
  default_male: 'Default Male',
  leijun: '雷军',
  guodegang: '郭德纲',
  jay: '周杰伦',
  huge: '胡歌',
  hanhong: '韩红',
  nailong: '奶龙',
  kenan: '柯南',
  haimian: '海绵宝宝',
  dengziqi: '邓紫棋',
  liyunlong: '李云龙',
  new_female: 'Clear Female',
  female: 'Bright Female',
  news_male: 'News Male',
  默认女声: 'Default Female',
  默认男声: 'Default Male',
  雷军: '雷军',
  郭德纲: '郭德纲',
  周杰伦: '周杰伦',
  胡歌: '胡歌',
  韩红: '韩红',
  奶龙: '奶龙',
  柯南: '柯南',
  海绵宝宝: '海绵宝宝',
  邓紫棋: '邓紫棋',
  李云龙: '李云龙',
  清纯女声: 'Clear Female',
  阳光女声: 'Bright Female',
  播音男声: 'News Male',
};
const normalizeVoiceLabel = (voice) => ({
  ...voice,
  label: VOICE_LABEL_EN[voice.id] || VOICE_LABEL_EN[voice.label] || voice.label,
});
const fallbackVoiceOptions = [
  { id: 'default_female', label: 'Default Female' },
  { id: 'default_male', label: 'Default Male' },
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
  { id: 'new_female', label: 'Clear Female' },
  { id: 'female', label: 'Bright Female' },
  { id: 'news_male', label: 'News Male' },
].map(normalizeVoiceLabel);
const voiceOptions = ref([...fallbackVoiceOptions]);
const selectedVoice = ref(DEFAULT_VOICE);

const loadVoices = async () => {
  try {
    const resp = await fetch(buildRealtimeVoicesUrl());
    if (!resp.ok) return;
    const payload = await resp.json();
    const voices = Array.isArray(payload.voices)
      ? payload.voices.filter((v) => v && v.id && v.label).map(normalizeVoiceLabel)
      : [];
    if (voices.length) {
      voiceOptions.value = voices;
      const bd = typeof payload.default_voice === 'string' ? payload.default_voice : DEFAULT_VOICE;
      if (!voices.some((v) => v.id === selectedVoice.value)) {
        selectedVoice.value = voices.some((v) => v.id === bd) ? bd : voices[0].id;
      }
    }
  } catch (e) {
    console.warn('Failed to load voice presets, using local fallback:', e);
  }
};

const fetchWithRuntimeContext = async (url, options = {}, stage = 'request') => {
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
    let extraHint = 'Check port reachability, browser proxy bypass rules, and CORS/same-origin policy.';
    if (pageProtocol === 'https:' && String(url).startsWith('http://')) {
      extraHint = 'The current page uses HTTPS but the request uses HTTP. The browser may block mixed content.';
    }
    throw new Error(
      `${stage} network request failed: ${message}; url=${url}; page_origin=${pageOrigin}; api_base=${apiBase}; hint=${extraHint}`
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
    isAiSpeaking.value = false;
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
        isAiSpeaking.value = realtimePlaybackIsPlaying;
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
  isAiSpeaking.value = false;
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

const flushRealtimePlaybackBuffer = async () => {
  if (realtimePlaybackWorkletNode) {
    try {
      realtimePlaybackWorkletNode.port.postMessage({ type: 'flush' });
    } catch (_err) {
      // ignore
    }
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
  isAiSpeaking.value = false;
  realtimePlaybackAutoRate = 1.0;
  realtimePlaybackManualRate = clamp(Number(playbackRate.value) || 1.0, 0.5, 1.8);
  realtimePlaybackTimelineStart = 0;
  realtimePlaybackScheduledSec = 0;
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
            realtimeReplyStartMarkerPending = false;
          }
          markRealtimeReplyStartIfListenToSpeak(realtimeLastState, nextState);
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
  const realtimeConfig = normalizeRealtimeConfig();
  const startPayload = {
    start_speak_factor: realtimeConfig.startSpeakFactor,
    start_listen_factor: realtimeConfig.startListenFactor,
    end_speak_factor: realtimeConfig.endSpeakFactor,
    prompt_voice: selectedVoice.value,
    tts_chunk_size: 1,
    infer_window_ms: realtimeConfig.inferWindowMs,
    client_upload_chunk_ms: realtimeConfig.streamChunkMs,
    stage_timing_log: true
  };
  if (realtimeBackendHint === 'hf') {
    startPayload.incremental_backend = 'hf';
  }
  const startUrl = buildRealtimeSessionStartUrl();
  const resp = await fetchWithRuntimeContext(startUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(startPayload)
  }, 'create realtime session');
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Failed to create realtime session ${resp.status}: ${text}`);
  }
  const result = await resp.json();
  if (!result || typeof result.session_id !== 'string' || !result.session_id) {
    throw new Error('Realtime session response has an invalid format.');
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
    }, 'connect realtime event stream');
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Realtime event stream HTTP ${resp.status}: ${text}`);
    }
    if (!resp.body) {
      throw new Error('Realtime event stream response has no body.');
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
          if ((nextState === 's' || nextState === 'b') && (!realtimeLastState || realtimeLastState === 'l')) {
            beginRealtimeTextAudioGate({ type: 'status_state', to: nextState }, { force: true });
          }
          markRealtimeReplyStartIfListenToSpeak(realtimeLastState, nextState);
          if (realtimeLastState && realtimeLastState !== 'l' && nextState === 'l') {
            openRealtimeTextAudioGate();
            commitRealtimeLiveReply();
            realtimeReplyStartMarkerPending = false;
          }
          realtimeLastState = nextState;
        }
      }

      if (eventPayload.type === 'state_change') {
        recordStateTransitionTiming(eventPayload);
        const toState = typeof eventPayload.to === 'string' ? eventPayload.to.toLowerCase() : '';
        const fromState = typeof eventPayload.from === 'string'
          ? eventPayload.from.toLowerCase()
          : realtimeLastState;
        if (toState) {
          if ((toState === 's' || toState === 'b') && (!fromState || fromState === 'l')) {
            beginRealtimeTextAudioGate(eventPayload, { force: true });
          }
          markRealtimeReplyStartIfListenToSpeak(fromState, toState);
          if (fromState && fromState !== 'l' && toState === 'l') {
            openRealtimeTextAudioGate();
            commitRealtimeLiveReply();
            realtimeReplyStartMarkerPending = false;
          }
          realtimeLastState = toState;
        }
        if (eventPayload.interrupt === true) {
          flushRealtimePlaybackBuffer().catch((err) => {
            console.warn('清理实时播放缓冲失败:', err);
          });
        }
      }

      if (eventPayload.type === 'audio_interrupt') {
        flushRealtimePlaybackBuffer().catch((err) => {
          console.warn('处理实时音频中断失败:', err);
        });
      }

      if (eventPayload.type === 'assistant_text' && typeof eventPayload.text === 'string') {
        markRealtimeTextPendingBeforeAudio(eventPayload);
        if (!handleStructuredRealtimeText(eventPayload)) {
          handleAssistantRealtimeText(eventPayload.text);
        }
      }

      if (eventPayload.type === 'stage_timing') {
        updateBackendQueueMetricsFromStageTiming(eventPayload);
      }

      if (eventPayload.type === 'audio_chunk_pcm' && typeof eventPayload.pcm_b64 === 'string' && eventPayload.pcm_b64) {
        const clientReceiveEpochMs = Date.now();
        try {
          const sampleRate = Math.max(1, Number(eventPayload.sample_rate) || 24000);
          const pcmSamples = decodePcm16Base64(eventPayload.pcm_b64);
          if (pcmSamples.length > 0) {
            openRealtimeTextAudioGateOnAudio(eventPayload);
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
          if (decoded.samples && decoded.samples.length > 0) {
            openRealtimeTextAudioGateOnAudio(eventPayload);
          }
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
        openRealtimeTextAudioGate();
        commitRealtimeLiveReply();
        renderRealtimeReply();
        connectionStatus.value = 'Realtime session ended';
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
  const chunkMs = Math.max(1, Number(streamChunkMs.value) || 200);
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
    }, 'send realtime chunk');

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Chunk upload failed ${resp.status}: ${text}`);
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
      const paceMs = Math.max(0, Number(segment?.paceMs) || 0);
      if (paceMs > 0) {
        const waitMs = nextUploadAllowedAtMs - performance.now();
        if (waitMs > 0) {
          await sleep(waitMs);
        }
        if (!isTalking.value) {
          break;
        }
        nextUploadAllowedAtMs = performance.now() + paceMs;
      }
      await sendOneSegment(segment);
      segmentsSentCount.value += 1;
      if (paceMs <= 0) {
        nextUploadAllowedAtMs = 0;
      }
      if (priorityUploadQueue.length === 0) {
        priorityUploadInProgress.value = false;
        priorityUploadSourceName.value = '';
      }
    }
  } catch (err) {
    if (isTalking.value) {
      console.error('Realtime chunk upload failed:', err);
      const failure = classifyStartTalkError(err);
      setStartTalkError(
        `Send failed: ${failure.summary}`,
        failure.hint,
        {
          toast: true,
          title: 'Realtime chunk upload failed',
          duration: 7000
        }
      );
      connectionStatus.value = `Send failed: ${failure.summary}`;
      connectionStatusClass.value = 'disconnected';
    }
  } finally {
    queueProcessing = false;
    syncQueueDepth();
  }
};

const buildShowcaseCaseUrl = (caseItem) => (
  `${SHOWCASE_CASE_BASE_PATH}${encodeURIComponent(caseItem.filename)}`
);

const fetchShowcaseCaseFile = async (caseItem) => {
  const url = buildShowcaseCaseUrl(caseItem);
  const resp = await fetch(url);
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`Failed to load sample audio ${resp.status}: ${text || resp.statusText}`);
  }
  const blob = await resp.blob();
  const file = new File([blob], caseItem.filename, {
    type: blob.type || 'audio/mp4'
  });
  return { file, url };
};

const stopShowcaseInputAudio = () => {
  if (!currentShowcaseInputAudio) {
    return;
  }
  try {
    currentShowcaseInputAudio.pause();
    currentShowcaseInputAudio.src = '';
  } catch (_err) {
    // Ignore playback cleanup failures.
  }
  currentShowcaseInputAudio = null;
};

const playShowcaseInputAudio = async (url) => {
  stopShowcaseInputAudio();
  const audio = new Audio(url);
  audio.preload = 'auto';
  audio.playbackRate = 1.0;
  currentShowcaseInputAudio = audio;
  audio.onended = () => {
    if (currentShowcaseInputAudio === audio) {
      currentShowcaseInputAudio = null;
    }
  };
  audio.onerror = () => {
    if (currentShowcaseInputAudio === audio) {
      currentShowcaseInputAudio = null;
    }
  };
  await audio.play();
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
  stopShowcaseInputAudio();
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

const resetRealtimeStartState = () => {
  realtimeStoppingExpected.value = false;
  connectionStatus.value = 'Connecting...';
  connectionStatusClass.value = 'waiting';
  resetRealtimeReplyState();
  resetRealtimeProbabilities();
  resetRealtimeTransitionTimings();
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
};

const attachRealtimeSessionEventStream = (sessionId) => {
  consumeRealtimeSessionSse(sessionId).catch((err) => {
    if (isAbortLikeError(err) || realtimeStoppingExpected.value) {
      return;
    }
    console.error('Realtime event stream failed:', err);
    const failure = classifyStartTalkError(err);
    setStartTalkError(
      `Realtime event stream error: ${failure.summary}`,
      failure.hint,
      {
        toast: true,
        title: 'Realtime event stream failed',
        duration: 7000
      }
    );
    connectionStatus.value = `Realtime event stream error: ${failure.summary}`;
    connectionStatusClass.value = 'disconnected';
  });
};

const startShowcaseCase = async (caseItem) => {
  if (!caseItem || isTalking.value || showcaseStarting.value) {
    return;
  }

  showcaseStarting.value = true;
  realtimeInputMode.value = 'showcase';
  resetRealtimeStartState();

  try {
    try {
      await ensureRealtimePlaybackContext();
    } catch (err) {
      console.warn('Failed to initialize realtime playback context:', err);
    }

    const { file, url } = await fetchShowcaseCaseFile(caseItem);
    const fileSegments = await buildRealtimeAudioFileSegments(file, {
      source: 'showcase',
      displayName: caseItem.label,
      tailSilenceSec: SHOWCASE_TAIL_SILENCE_SEC,
      paced: true
    });

    const sessionId = await createRealtimeSession();
    realtimeSessionId.value = sessionId;
    startCurrentAlignedArchive(sessionId);
    attachRealtimeSessionEventStream(sessionId);

    isTalking.value = true;
    captureSampleRateDisplay.value = `${TARGET_SAMPLE_RATE} Hz (sample file)`;
    primeRealtimeProbabilitiesIfEmpty();
    clearStartTalkError({ clearStorage: true });
    connectionStatus.value = `Playing sample: ${caseItem.label} (mic off, ${SHOWCASE_TAIL_SILENCE_SEC}s tail silence)`;
    connectionStatusClass.value = 'connected';

    playShowcaseInputAudio(url).catch((err) => {
      console.warn('Sample input playback failed:', err);
      ElNotification({
        title: 'Sample playback failed',
        message: 'The browser may have blocked autoplay, but the audio will still be sent to the model.',
        type: 'warning',
        duration: 5000
      });
    });
    enqueueRealtimeAudioSegments(
      fileSegments,
      `${caseItem.label} (${SHOWCASE_TAIL_SILENCE_SEC}s tail silence)`
    );
  } catch (err) {
    console.error('Failed to start sample case:', err);
    const sessionId = realtimeSessionId.value;
    realtimeSessionId.value = '';
    if (sessionId) {
      stopRealtimeSession(sessionId).catch(() => {});
    }
    currentAlignedArchive = null;
    releaseRealtimeResources();
    isTalking.value = false;
    realtimeInputMode.value = 'mic';
    const failure = classifyStartTalkError(err);
    setStartTalkError(
      failure.summary,
      failure.hint,
      {
        toast: true,
        title: 'Sample start failed',
        duration: 9000
      }
    );
    connectionStatus.value = `Sample start failed: ${failure.summary}`;
    connectionStatusClass.value = 'disconnected';
  } finally {
    showcaseStarting.value = false;
    selectedShowcaseCaseId.value = '';
  }
};


const startTalk = async () => {
  if (isTalking.value || showcaseStarting.value) {
    return;
  }

  realtimeInputMode.value = 'mic';
  resetRealtimeStartState();

  try {
    try {
      await ensureRealtimePlaybackContext();
    } catch (err) {
      console.warn('初始化实时播放上下文失败:', err);
    }
    const sessionId = await createRealtimeSession();
    realtimeSessionId.value = sessionId;
    startCurrentAlignedArchive(sessionId);
    attachRealtimeSessionEventStream(sessionId);

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
    startCallTimer();
    // 通话开始：用 listening 状态的先验填充 S-L/S-S，避免在后端推送首个 token 前显示 "--"
    // 后端收到第一组真值后会立即被 updateRealtimeProbabilities 覆盖
    primeRealtimeProbabilitiesIfEmpty();

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
    }, streamChunkMs.value);

    clearStartTalkError({ clearStorage: true });
    connectionStatus.value = 'Connected. Streaming microphone chunks...';
    connectionStatusClass.value = 'connected';
  } catch (err) {
    console.error('Failed to start realtime call:', err);
    const sessionId = realtimeSessionId.value;
    realtimeSessionId.value = '';
    if (sessionId) {
      stopRealtimeSession(sessionId).catch(() => {});
    }
    currentAlignedArchive = null;
    releaseRealtimeResources();
    isTalking.value = false;
    realtimeInputMode.value = 'mic';
    const failure = classifyStartTalkError(err);
    setStartTalkError(
      failure.summary,
      failure.hint,
      {
        toast: true,
        title: 'Realtime call start failed',
        duration: 9000
      }
    );
    connectionStatus.value = `Start failed: ${failure.summary}`;
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
  openRealtimeTextAudioGate();
  releaseRealtimeResources();
  finalizeCurrentAlignedArchive();
  connectionStatus.value = 'Call ended';
  connectionStatusClass.value = 'disconnected';
  resetRealtimeProbabilities();
  stopCallTimer();
  isAiSpeaking.value = false;
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
  document.title = 'lychee-FD'
  function_value.value = 'function_6'
  restorePersistedStartTalkError();
})

onBeforeUnmount(() => {
  endTalk()
  clearRealtimeTextTypewriter()
  clearRealtimeTextAudioGateTimer()
  clearAlignedHistory()
  stopVoicePreview()
  stopCallTimer()
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

watch(aiReplyText, () => {
  scrollToBottom()
})


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
/* ===== DEV: model bar ===== */
.dev-model-bar {
  width: 100%;
  flex-shrink: 0;
  padding: 10px 18px;
  background: rgba(20, 22, 30, 0.55);
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  color: #e6e6e6;
  font-size: 13px;
  display: flex; flex-direction: column; gap: 6px;
}
.dev-model-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.dev-model-label { color: #aaa; min-width: 36px; }
.dev-model-select,
.dev-model-input {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.15);
  color: #f0f0f0;
  padding: 6px 8px; border-radius: 6px; font-size: 13px; min-width: 220px;
}
.dev-model-select-sm { min-width: 96px; }
.dev-model-input { flex: 1 1 320px; }
.dev-btn-primary, .dev-btn-ghost {
  border: none; border-radius: 6px; padding: 6px 14px;
  font-size: 13px; cursor: pointer; transition: opacity .2s;
}
.dev-btn-primary { background: #4a8cff; color: #fff; }
.dev-btn-primary[disabled] { opacity: .5; cursor: not-allowed; }
.dev-btn-ghost { background: transparent; color: #ccc; border: 1px solid rgba(255,255,255,0.2); }
.dev-model-status { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; font-size: 12px; }
.dev-tag {
  padding: 2px 8px; border-radius: 10px;
  background: rgba(255,255,255,0.1); color: #ddd; font-weight: 600;
}
.dev-tag-ready    { background: #1e7e34; color: #fff; }
.dev-tag-starting { background: #b58900; color: #fff; }
.dev-tag-stopping { background: #b58900; color: #fff; }
.dev-tag-error    { background: #b22222; color: #fff; }
.dev-tag-idle     { background: #555; color: #ddd; }
.dev-status-detail { color: #bbb; }
.dev-status-error { color: #ff8a8a; }
/* ===== /DEV ===== */

.rt-app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  min-height: 100vh;
  width: 100%;
  overflow: hidden;
  color: #1f2a37;
  background:
    radial-gradient(circle at 12% 10%, rgba(66, 153, 225, 0.12), transparent 38%),
    radial-gradient(circle at 90% 86%, rgba(16, 185, 129, 0.1), transparent 40%),
    linear-gradient(120deg, #f4f8fb 0%, #ecf3f9 100%);
}

.rt-glass-header {
  height: 64px;
  flex-shrink: 0;
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
  min-height: 0;
  display: flex;
  align-items: stretch;
  justify-content: center;
  padding: 20px 24px 12px;
  overflow: hidden;
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
  width: min(980px, 96vw);
}

.rt-welcome-card {
  padding: 46px 56px;
  text-align: center;
  border-radius: 20px;
  font-family: "MiSans", "HarmonyOS Sans SC", "Alibaba PuHuiTi", "Noto Sans SC", "Source Han Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
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
  font-size: clamp(25px, 3.1vw, 32px);
  line-height: 1.28;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: #0f172a;
}

.rt-welcome-card p {
  margin: 10px 0 24px;
  font-size: 16px;
  font-weight: 400;
  color: #475569;
}

.rt-welcome-copy {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-width: 820px;
  margin: 14px auto 24px;
}

.rt-welcome-copy p {
  margin: 0;
  line-height: 1.75;
  font-size: 16px;
  font-weight: 400;
  letter-spacing: 0.01em;
  text-align: left;
}

.rt-welcome-copy .rt-welcome-cta {
  font-weight: 400;
  text-align: center;
  color: #0f172a;
}

.rt-voice-picker {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  margin-bottom: 20px;
}

.rt-voice-picker label {
  font-size: 14px;
  color: #000;
}

.rt-voice-picker select {
  min-width: 160px;
  background: #fff;
  border-color: rgba(15, 23, 42, 0.18);
  color: #000;
}

.rt-showcase-picker {
  width: min(360px, 100%);
  margin: 14px auto 0;
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  align-items: center;
  gap: 10px;
}

.rt-showcase-picker label {
  font-size: 14px;
  color: #0f172a;
  font-weight: 600;
}

.rt-showcase-select {
  width: 100%;
  min-width: 0;
  border: 1px solid rgba(15, 23, 42, 0.16);
  border-radius: 8px;
  background: #ffffff;
  color: #0f172a;
  font-size: 14px;
  padding: 9px 10px;
  outline: none;
}

.rt-showcase-select:focus {
  border-color: rgba(14, 116, 144, 0.55);
  box-shadow: 0 0 0 3px rgba(14, 116, 144, 0.12);
}

.rt-session-config {
  width: min(560px, 100%);
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin: 0 auto 20px;
}

.rt-session-config label {
  display: grid;
  gap: 4px;
  text-align: left;
  font-size: 12px;
  color: #475569;
}

.rt-session-config input {
  width: 100%;
  min-width: 0;
  border: 1px solid rgba(15, 23, 42, 0.14);
  border-radius: 8px;
  background: #ffffff;
  color: #0f172a;
  font-size: 13px;
  padding: 7px 9px;
  font-variant-numeric: tabular-nums;
  outline: none;
}

.rt-session-config input:focus {
  border-color: rgba(14, 116, 144, 0.55);
  box-shadow: 0 0 0 3px rgba(14, 116, 144, 0.12);
}

.rt-active-workspace {
  width: min(1180px, 96vw);
  height: 100%;
  max-height: min(860px, calc(100vh - 132px));
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(280px, 360px) minmax(340px, 1fr) minmax(300px, 360px);
  gap: 16px;
}

.rt-panel {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 16px;
  padding: 16px 16px 20px;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
  min-height: 0;
  flex: 1 1 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.rt-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
  gap: 10px;
  flex-shrink: 0;
  min-width: 0;
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
  flex: 1 1 auto;
  min-height: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 18px;
  overflow: hidden;
}

.rt-mic-wrapper {
  margin-top: 12px;
  width: 150px;
  height: 150px;
  flex: 0 0 150px;
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
  flex: 0 0 430px;
  min-height: 430px;
  max-height: 430px;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  gap: 8px;
  font-size: 12px;
  line-height: 1.35;
  color: #334155;
  overflow: visible;
}

.rt-audio-info > span {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-variant-numeric: tabular-nums;
}

.rt-audio-info-metrics {
  min-height: 0;
  overflow: visible;
  display: grid;
  grid-template-columns: 1fr;
  gap: 4px;
}

.rt-audio-info-metrics > span {
  display: block;
  min-width: 0;
  min-height: 18px;
  line-height: 18px;
  white-space: normal;
  overflow: visible;
  text-overflow: clip;
  word-break: break-word;
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

/* 事件记录：竖向滚动列表，一行一条；横向溢出由每条 tag 内部省略号处理 */
.rt-transition-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-top: 4px;
  min-height: 56px;
  max-height: 140px;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 6px 8px;
  border-radius: 8px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  background: rgba(248, 250, 252, 0.82);
  scrollbar-gutter: stable;
}

.rt-transition-tag {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  min-height: 22px;
  padding: 3px 8px;
  border-radius: 6px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  background: rgba(15, 23, 42, 0.05);
  color: #334155;
  font-size: 11px;
  line-height: 1.3;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.rt-playback-controls {
  width: 100%;
  flex-shrink: 0;
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
  flex: 1 1 auto;
  min-height: 0;
  overflow: auto;
  border-radius: 12px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  background: #ffffff;
  padding: 14px;
}

.markdown-body {
  max-width: 100%;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.markdown-body :deep(pre) {
  max-width: 100%;
  overflow: auto;
}

.markdown-body :deep(code) {
  white-space: pre-wrap;
}

.markdown-body :deep(table) {
  display: block;
  max-width: 100%;
  overflow-x: auto;
}

.rt-history-content {
  flex: 1 1 auto;
  min-height: 0;
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
  flex-shrink: 0;
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

.rt-btn-primary[disabled],
.rt-btn-secondary[disabled],
.rt-btn-danger[disabled],
.rt-showcase-select[disabled] {
  cursor: not-allowed;
  opacity: 0.62;
  transform: none;
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
    overflow: auto;
    align-items: flex-start;
  }

  .rt-global-error-strip {
    width: calc(100% - 20px);
    margin: 10px 10px 0;
  }

  .rt-welcome-screen {
    width: 100%;
  }

  .rt-welcome-card {
    padding: 34px 22px;
  }

  .rt-active-workspace {
    width: 100%;
    height: auto;
    max-height: none;
    grid-template-columns: 1fr;
  }

  .rt-session-config {
    grid-template-columns: 1fr;
  }

  .rt-showcase-picker {
    grid-template-columns: 1fr;
    text-align: left;
  }

  .rt-panel {
    min-height: 300px;
    height: min(520px, 72vh);
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

/* =======================================================================
   REDESIGN: 单页实时通话工作台（基础样式，后续 style-system 任务继续打磨）
   ======================================================================= */
.rt-app-container {
  background:
    radial-gradient(circle at 12% 8%, rgba(55, 138, 221, 0.10), transparent 40%),
    radial-gradient(circle at 92% 90%, rgba(29, 158, 117, 0.08), transparent 42%),
    linear-gradient(120deg, #f6fafc 0%, #eef6f8 100%);
  font-family: "PingFang SC", "MiSans", "HarmonyOS Sans SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
}

/* ---- Header ---- */
.rt-glass-header {
  height: 56px;
  justify-content: space-between;
}
.rt-header-content { gap: 8px; }
.rt-logo-img { width: 38px; height: 30px; }
.rt-header-title { font-size: 18px; }
.rt-badge {
  margin-left: 6px;
  color: #1d9e75;
  background: rgba(29, 158, 117, 0.12);
}
.rt-header-links { display: flex; align-items: center; gap: 8px; }
.rt-header-link {
  font-size: 13px;
  color: #5f5e5a;
  text-decoration: none;
  padding: 6px 14px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.05);
  transition: background .18s, color .18s;
}
.rt-header-link:hover { background: rgba(55, 138, 221, 0.12); color: #378add; }

/* ---- 模型栏 ---- */
.rt-model-bar {
  width: 100%;
  flex-shrink: 0;
  padding: 10px 20px;
  background: rgba(255, 255, 255, 0.7);
  border-bottom: 1px solid #e4eaee;
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: #0f172a;
  font-size: 13px;
}
.rt-model-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.rt-model-label { color: #5f5e5a; font-weight: 600; }
.rt-model-select, .rt-model-input {
  background: #fff;
  border: 1px solid #d3d1c7;
  color: #0f172a;
  padding: 6px 10px;
  border-radius: 8px;
  font-size: 13px;
  min-width: 200px;
  outline: none;
}
.rt-model-select-sm { min-width: 100px; }
.rt-model-input { flex: 1 1 280px; }
.rt-model-btn {
  border: none; border-radius: 999px; padding: 6px 16px;
  font-size: 13px; cursor: pointer; transition: opacity .18s, background .18s;
}
.rt-model-btn-primary { background: #378add; color: #fff; }
.rt-model-btn-primary:hover { background: #2f7bc9; }
.rt-model-btn-primary[disabled] { opacity: .5; cursor: not-allowed; }
.rt-model-btn-ghost { background: rgba(15,23,42,0.05); color: #5f5e5a; }
.rt-model-btn-ghost:hover { background: rgba(15,23,42,0.1); }
.rt-model-state {
  padding: 3px 12px; border-radius: 999px; font-weight: 600; font-size: 12px;
  background: #eceae1; color: #5f5e5a;
}
.rt-model-state-ready { background: rgba(29,158,117,0.15); color: #1d9e75; }
.rt-model-state-starting, .rt-model-state-loading { background: rgba(55,138,221,0.15); color: #378add; }
.rt-model-state-error { background: rgba(226,75,74,0.15); color: #e24b4a; }
.rt-model-meta-toggle {
  border: none; background: transparent; color: #888780;
  font-size: 12px; cursor: pointer; text-decoration: underline; padding: 2px 4px;
}
.rt-model-meta {
  font-size: 12px; color: #888780; display: flex; flex-wrap: wrap; gap: 12px;
  word-break: break-all;
}
.rt-model-meta-error { color: #e24b4a; }

/* ---- 主工作区 ---- */
.rt-main-workspace { align-items: stretch; padding: 16px 20px 10px; min-height: 0; }
.rt-workspace-grid {
  width: 100%;
  flex: 1 1 0;
  min-height: 0;
  display: flex;
  align-items: stretch;
  gap: 16px;
}
.rt-agent-panel { flex: 1.7 1 0; min-width: 0; min-height: 0; }
.rt-history-panel { flex: 1 1 0; min-width: 280px; max-width: 380px; min-height: 0; }

/* ---- 智能体响应区 ---- */
.rt-call-timer {
  display: block;
  text-align: center;
  font-size: 22px;
  font-weight: 700;
  color: #0f172a;
  letter-spacing: 0.5px;
  font-variant-numeric: tabular-nums;
  margin-bottom: -4px;
}
.rt-orb-stage {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 12px; padding: 18px 0 8px;
  flex-shrink: 0;
}
.rt-orb {
  position: relative; width: 116px; height: 116px; border-radius: 50%;
  display: grid; place-items: center;
}
.rt-orb-core {
  width: 92px; height: 92px; border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, #bcd6f7, #9fc0ec 45%, #7fb6d8 100%);
  box-shadow: 0 8px 28px rgba(55,138,221,0.25);
  transition: transform .4s ease, background .4s ease, box-shadow .4s ease;
}
.rt-orb-ring {
  position: absolute; inset: 0; border-radius: 50%;
  border: 2px solid rgba(55,138,221,0.35); opacity: 0;
}
.rt-orb-status { font-size: 15px; font-weight: 600; color: #0f172a; }
/* 状态色相（基础版，动画在 style-system 完善） */
.rt-orb-idle .rt-orb-core { filter: grayscale(0.5) brightness(0.98); box-shadow: none; }
.rt-orb-listening .rt-orb-core { background: radial-gradient(circle at 35% 30%, #b6e7d4, #7fd0b4 50%, #1d9e75 110%); animation: rt-orb-breathe 2.4s ease-in-out infinite; }
.rt-orb-thinking .rt-orb-core { background: radial-gradient(circle at 35% 30%, #cdc7f6, #9b8ef0 50%, #6f5fe0 110%); animation: rt-orb-pulse 1.4s ease-in-out infinite; }
.rt-orb-ai_speaking .rt-orb-core { background: radial-gradient(circle at 35% 30%, #b6e7df, #6fd0c4 50%, #1d9eb0 110%); animation: rt-orb-pulse 0.9s ease-in-out infinite; }
.rt-orb-ai_speaking .rt-orb-ring { opacity: 1; animation: rt-orb-spread 1.4s ease-out infinite; }
.rt-orb-connecting .rt-orb-core { animation: rt-orb-pulse 1s ease-in-out infinite; }
.rt-orb-error .rt-orb-core { background: radial-gradient(circle at 35% 30%, #f3c2c1, #e24b4a 110%); }
.rt-orb-ended .rt-orb-core { filter: grayscale(0.3); }
@keyframes rt-orb-breathe { 0%,100% { transform: scale(1); } 50% { transform: scale(1.06); } }
@keyframes rt-orb-pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.09); } }
@keyframes rt-orb-spread { 0% { transform: scale(0.85); opacity: 0.6; } 100% { transform: scale(1.35); opacity: 0; } }

.rt-subtitle-panel {
  flex: 1; min-height: 0; overflow-y: auto;
  background: #f3f8fb;
  border: 1px solid #e4eaee;
  border-radius: 12px;
  padding: 14px 16px 24px;
  font-size: 14px; line-height: 1.6; color: #0f172a;
  scroll-padding-bottom: 24px;
}
.rt-subtitle-thinking, .rt-subtitle-idle { color: #888780; }
.rt-subtitle-idle p { margin: 0 0 8px; line-height: 1.7; }
.rt-welcome-title { font-size: 15px; color: #0f172a; font-weight: 600; }
.rt-welcome-desc { font-size: 13px; color: #5f5e5a; }

/* 欢迎页浮动样例：通话开始后自然隐藏（外层 v-else 由 isTalking/aiReplyText 控制） */
.rt-welcome-suggestions {
  margin-top: 14px;
  display: flex; flex-direction: column; align-items: center; gap: 10px;
  animation: rt-welcome-float-in .4s ease both;
}
.rt-welcome-suggestions-row {
  display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;
  width: 100%;
}
.rt-suggestion-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 8px 16px;
  border-radius: 999px;
  border: 1px solid rgba(55, 138, 221, 0.25);
  background: #fff;
  color: #0f172a;
  font-size: 13px; font-weight: 500;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease;
}
.rt-suggestion-chip:hover:not(:disabled) {
  transform: translateY(-2px);
  border-color: rgba(55, 138, 221, 0.55);
  box-shadow: 0 6px 18px rgba(55, 138, 221, 0.18);
  background: rgba(55, 138, 221, 0.05);
}
.rt-suggestion-chip:disabled { opacity: 0.55; cursor: not-allowed; }
.rt-suggestion-chip-icon { font-size: 14px; }
.rt-suggestion-chip-text { line-height: 1.2; }
.rt-suggestion-refresh {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 4px 12px;
  border-radius: 999px;
  border: 1px dashed rgba(15, 23, 42, 0.18);
  background: transparent;
  color: #888780;
  font-size: 12px;
  cursor: pointer;
  transition: color .18s ease, border-color .18s ease;
}
.rt-suggestion-refresh:hover:not(:disabled) {
  color: #378add;
  border-color: rgba(55, 138, 221, 0.45);
}
.rt-suggestion-refresh:disabled { opacity: 0.55; cursor: not-allowed; }
.rt-suggestion-refresh-icon {
  display: inline-block; font-size: 13px; line-height: 1;
  transition: transform .35s ease;
}
.rt-suggestion-refresh:hover:not(:disabled) .rt-suggestion-refresh-icon {
  transform: rotate(180deg);
}
@keyframes rt-welcome-float-in {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
/* 打字机光标：AI 回应中显示，停止即消失 */
.rt-subtitle-text.is-streaming::after {
  content: '▋';
  display: inline-block;
  margin-left: 2px;
  color: #1d9e75;
  animation: rt-caret-blink 0.9s steps(1) infinite;
}
@keyframes rt-caret-blink { 0%, 50% { opacity: 1; } 50.01%, 100% { opacity: 0; } }

/* ---- 交互历史区 ---- */
.rt-history-content {
  flex: 1; min-height: 0; overflow-y: auto;
  display: flex; flex-direction: column; gap: 12px;
  /* 底部充分留白，避免最后一个 entry 内的 audio 控件视觉上贴住 panel 边界 */
  padding: 2px 6px 28px 0;
  scroll-padding-bottom: 28px;
}
.rt-history-content > .rt-history-entry:last-child { margin-bottom: 12px; }
.rt-history-entry {
  background: #fff; border: 1px solid #e4eaee; border-radius: 12px;
  padding: 12px; display: flex; flex-direction: column; gap: 8px;
  transition: border-color .2s, box-shadow .2s;
}
.rt-history-entry.is-latest {
  border-color: rgba(55, 138, 221, 0.4);
  box-shadow: 0 6px 18px rgba(55, 138, 221, 0.1);
}
.rt-history-head { display: flex; align-items: center; gap: 8px; }
.rt-history-round { font-size: 13px; font-weight: 700; color: #0f172a; }
.rt-history-latest-tag {
  font-size: 10px; font-weight: 600; color: #378add;
  background: rgba(55, 138, 221, 0.12); border-radius: 999px; padding: 1px 8px;
}
.rt-history-time { margin-left: auto; font-size: 11px; color: #888780; font-variant-numeric: tabular-nums; }
.rt-history-meta { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11px; color: #888780; word-break: break-all; }
.rt-history-label {
  font-size: 12px; color: #5f5e5a; margin-bottom: 4px;
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
}
.rt-history-dur { font-size: 11px; color: #888780; font-variant-numeric: tabular-nums; }
.rt-history-audio-block audio { width: 100%; height: 34px; }
.rt-history-empty {
  flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 6px; color: #888780; font-size: 13px; text-align: center; padding: 20px 10px;
}
.rt-history-empty-icon { font-size: 28px; opacity: 0.7; }
.rt-history-empty-hint { font-size: 11px; color: #b3b1a8; line-height: 1.5; }

/* ---- Debug 抽屉 ---- */
.rt-debug-drawer {
  flex-shrink: 0; width: 44px;
  display: flex; align-items: stretch;
  background: rgba(255,255,255,0.9);
  border: 1px solid #e4eaee; border-radius: 16px;
  overflow: hidden; transition: width .25s ease;
}
.rt-debug-drawer.is-open { width: 320px; }
.rt-debug-rail {
  flex-shrink: 0; width: 44px; border: none; cursor: pointer;
  background: transparent; color: #5f5e5a;
  display: flex; flex-direction: column; align-items: center; justify-content: space-between;
  padding: 12px 0; position: relative;
}
.rt-debug-rail-text { writing-mode: vertical-rl; letter-spacing: 2px; font-size: 12px; font-weight: 600; }
.rt-debug-rail-arrow { font-size: 16px; }
.rt-debug-rail:hover { color: #378add; }
.rt-debug-rail.has-error { color: #e24b4a; }
.rt-debug-rail-dot {
  width: 8px; height: 8px; border-radius: 50%; background: #e24b4a;
  animation: rt-debug-dot-pulse 1.4s ease-in-out infinite;
}
@keyframes rt-debug-dot-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
.rt-debug-toolbar {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 4px;
}
.rt-debug-toolbar-title { font-size: 13px; font-weight: 700; color: #0f172a; }
.rt-debug-copy {
  border: 1px solid #d3d1c7; background: #fff; color: #5f5e5a;
  border-radius: 6px; padding: 3px 10px; font-size: 11px; cursor: pointer;
}
.rt-debug-copy:hover { border-color: #378add; color: #378add; }
.rt-debug-alert {
  background: rgba(226, 75, 74, 0.08); border: 1px solid rgba(226, 75, 74, 0.25);
  border-radius: 8px; padding: 8px 10px;
}
.rt-debug-alert-title { font-size: 11px; font-weight: 700; color: #e24b4a; margin-bottom: 3px; }
.rt-debug-alert-text { font-size: 11px; color: #b3322f; line-height: 1.5; word-break: break-word; }
.rt-debug-alert-hint { font-size: 10px; color: #92400e; margin-top: 3px; line-height: 1.5; }
.rt-debug-prob-row { display: flex; gap: 8px; }
.rt-prob-tag {
  font-size: 11px; color: #5f5e5a; background: #f3f8fb;
  border-radius: 6px; padding: 2px 8px;
}
/* （冗余规则已移除；事件记录列表样式见上文 .rt-transition-list / .rt-transition-tag） */
.rt-debug-body {
  flex: 1; min-width: 0; overflow-y: auto;
  border-left: 1px solid #e4eaee; padding: 12px 12px 20px;
  display: flex; flex-direction: column; gap: 14px;
  scroll-padding-bottom: 20px;
}
.rt-debug-group { display: flex; flex-direction: column; gap: 6px; }
.rt-debug-group-title { font-size: 12px; font-weight: 700; color: #378add; }
.rt-debug-row { display: flex; justify-content: space-between; gap: 8px; font-size: 11px; color: #5f5e5a; }
.rt-debug-row span:last-child { color: #0f172a; word-break: break-all; text-align: right; }
.rt-debug-row-path span:last-child, .rt-debug-row-error span:last-child { font-size: 10px; }
.rt-debug-row-error span:last-child { color: #e24b4a; }
.rt-debug-metrics { display: flex; flex-direction: column; gap: 3px; font-size: 11px; color: #5f5e5a; }
.rt-debug-action {
  border: 1px solid #d3d1c7; background: #fff; color: #0f172a;
  border-radius: 8px; padding: 6px 10px; font-size: 12px; cursor: pointer;
}
.rt-debug-action[disabled] { opacity: .5; cursor: not-allowed; }

/* ---- 底部工具栏 ----
   注意：主工作区为 flex:1，与 toolbar 是 flex sibling，本不会重叠；
   但视觉上两个面板内容可滚动，紧贴 toolbar 边缘会"压迫"。
   这里给 toolbar 加上下足够呼吸空间，同时面板内滚动区底部也留白。 */
.rt-toolbar {
  flex-shrink: 0;
  display: flex; align-items: center; justify-content: center; gap: 16px;
  padding: 16px 20px 22px;
  margin-top: 4px;
}
.rt-tool-slot { position: relative; }
.rt-tool-btn {
  border: none; border-radius: 999px; cursor: pointer;
  height: 44px; padding: 0 22px; font-size: 14px; font-weight: 600;
  transition: background .18s, box-shadow .18s;
}
.rt-tool-btn-voice { background: rgba(55,138,221,0.12); color: #378add; }
.rt-tool-btn-voice.is-active, .rt-tool-btn-voice:hover { background: rgba(55,138,221,0.2); }
.rt-tool-btn-settings { background: #efece4; color: #5f5e5a; }
.rt-tool-btn-settings.is-active, .rt-tool-btn-settings:hover { background: #e4e0d5; }
.rt-call-btn {
  border: none; border-radius: 999px; cursor: pointer;
  height: 48px; padding: 0 36px; font-size: 15px; font-weight: 700; color: #fff;
  transition: background .18s, box-shadow .18s, transform .12s;
}
.rt-call-btn.is-start { background: #0f172a; box-shadow: 0 8px 22px rgba(15,23,42,0.22); }
.rt-call-btn.is-start:hover { background: #1e293b; }
.rt-call-btn.is-hangup { background: #e24b4a; box-shadow: 0 8px 22px rgba(226,75,74,0.3); }
.rt-call-btn.is-hangup:hover { background: #cf3f3e; }
.rt-call-btn[disabled] { opacity: .6; cursor: not-allowed; }

/* ---- 弹层 ---- */
.rt-popover {
  position: absolute; bottom: calc(100% + 12px); left: 50%; transform: translateX(-50%);
  width: 300px; background: #fff; border: 1px solid #e4eaee; border-radius: 14px;
  box-shadow: 0 16px 40px rgba(15,23,42,0.16); padding: 14px; z-index: 30;
}
.rt-popover-title { font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 10px; }
.rt-popover-hint {
  font-size: 11px; color: #92400e; background: rgba(245, 158, 11, 0.1);
  border-radius: 6px; padding: 5px 8px; margin-bottom: 10px;
}
.rt-pop-enter-active, .rt-pop-leave-active { transition: opacity .18s ease, transform .18s ease; }
.rt-pop-enter-from, .rt-pop-leave-to { opacity: 0; transform: translateX(-50%) translateY(8px); }
.rt-popover-backdrop { position: fixed; inset: 0; z-index: 20; background: transparent; }

.rt-voice-upload-row {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
  background: #f3f8fb; border-radius: 10px; padding: 10px 12px; margin-bottom: 10px;
}
.rt-voice-upload-text { font-size: 13px; color: #5f5e5a; }
.rt-voice-upload-btn {
  border: none; background: #378add; color: #fff; border-radius: 999px;
  padding: 5px 14px; font-size: 12px; cursor: pointer; white-space: nowrap;
}
.rt-voice-list { display: flex; flex-direction: column; gap: 4px; max-height: 280px; overflow-y: auto; }
.rt-voice-item {
  display: flex; align-items: center; gap: 6px; width: 100%;
  border: 1px solid transparent; border-radius: 10px;
  padding: 4px 6px; transition: background .15s;
}
.rt-voice-item:hover { background: #f3f8fb; }
.rt-voice-item.is-selected { background: rgba(55,138,221,0.1); border-color: rgba(55,138,221,0.25); }
.rt-voice-select {
  flex: 1; min-width: 0; display: flex; align-items: center; gap: 10px;
  border: none; background: transparent; cursor: pointer; text-align: left;
  padding: 4px 4px;
}
.rt-voice-select:disabled { opacity: .55; cursor: not-allowed; }
.rt-voice-icon {
  flex-shrink: 0; width: 28px; height: 28px; border-radius: 50%;
  display: grid; place-items: center; font-size: 13px; font-weight: 700;
}
.rt-voice-name { flex: 1; font-size: 13px; color: #0f172a; }
.rt-voice-check { color: #378add; font-weight: 700; }
.rt-voice-preview {
  flex-shrink: 0; border: 1px solid #d3d1c7; background: #fff; color: #5f5e5a;
  border-radius: 999px; padding: 3px 12px; font-size: 12px; cursor: pointer;
  transition: background .15s, color .15s, border-color .15s;
}
.rt-voice-preview:hover { border-color: #378add; color: #378add; }
.rt-voice-preview.is-playing { background: #378add; border-color: #378add; color: #fff; }

.rt-setting-row {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  margin-bottom: 10px; font-size: 13px; color: #5f5e5a;
}
.rt-setting-row input[type="number"] {
  width: 90px; border: 1px solid #d3d1c7; border-radius: 8px;
  padding: 6px 8px; font-size: 13px; outline: none; color: #0f172a;
}
.rt-setting-range { display: flex; align-items: center; gap: 8px; }
.rt-setting-range-val { font-size: 12px; color: #888780; min-width: 30px; }

/* ---- 滚动条美化 ---- */
.rt-subtitle-panel::-webkit-scrollbar,
.rt-history-content::-webkit-scrollbar,
.rt-debug-body::-webkit-scrollbar,
.rt-voice-list::-webkit-scrollbar { width: 6px; }
.rt-subtitle-panel::-webkit-scrollbar-thumb,
.rt-history-content::-webkit-scrollbar-thumb,
.rt-debug-body::-webkit-scrollbar-thumb,
.rt-voice-list::-webkit-scrollbar-thumb { background: rgba(15, 23, 42, 0.16); border-radius: 999px; }
.rt-subtitle-panel::-webkit-scrollbar-track,
.rt-history-content::-webkit-scrollbar-track,
.rt-debug-body::-webkit-scrollbar-track,
.rt-voice-list::-webkit-scrollbar-track { background: transparent; }

/* ---- 面板标题统一（覆盖旧的 emoji 风格，落地设计配色） ---- */
.rt-panel-title { color: #0f172a; font-weight: 700; font-size: 16px; }
.rt-panel { box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06); border-color: #e4eaee; }

/* ---- 响应式：窄屏堆叠 ---- */
@media (max-width: 1080px) {
  .rt-workspace-grid { flex-direction: column; overflow-y: auto; }
  .rt-agent-panel, .rt-history-panel { flex: none; width: 100%; max-width: none; min-width: 0; }
  .rt-agent-panel { min-height: 360px; }
  .rt-history-panel { min-height: 260px; }
  .rt-debug-drawer {
    width: 100%; flex-direction: column;
  }
  .rt-debug-drawer.is-open { width: 100%; }
  .rt-debug-rail {
    width: 100%; flex-direction: row; justify-content: center; gap: 8px; padding: 10px 0;
  }
  .rt-debug-rail-text { writing-mode: horizontal-tb; }
  .rt-debug-rail-dot { position: static; }
  .rt-debug-body { border-left: none; border-top: 1px solid #e4eaee; }
}

@media (max-width: 720px) {
  .rt-glass-header { height: auto; flex-direction: column; align-items: flex-start; gap: 6px; padding: 10px 16px; }
  .rt-header-links { flex-wrap: wrap; }
  .rt-model-row { gap: 6px; }
  .rt-model-select { min-width: 140px; }
  .rt-toolbar { flex-wrap: wrap; gap: 10px; }
  .rt-popover { width: min(86vw, 300px); }
  .rt-orb { width: 96px; height: 96px; }
  .rt-orb-core { width: 76px; height: 76px; }
}
</style>
