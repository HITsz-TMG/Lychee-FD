/*
  Lychee-FD 中文技术展示页
  后续替换素材时，优先修改下面的配置区域即可。
*/

const METRICS = [
  { value: "7.4%", label: "语音问答准确率提升" },
  { value: "28.5%", label: "打断与接话指标提升" },
  { value: "570ms", label: "中断停止延迟约" },
  { value: "4.50", label: "语音自然度 UTMOS" },
];

const HIGHLIGHTS = [
  {
    icon: "▦",
    title: "原生端到端全双工架构",
    desc: "摒弃冗余级联设计，统一建模语音、文本和控制信号，实现更自然的实时对话体验。",
  },
  {
    icon: "⌁",
    title: "层次化语义-声学参数分离",
    desc: "物理级解耦深层语义和声学参数，缓解模态梯度冲突，兼顾语义能力和语音质量。",
  },
  {
    icon: "⌖",
    title: "密集语义对齐通道",
    desc: "引入显式语义锚点，对抗语义稀释，确保回复逻辑连贯、内容准确可信。",
  },
  {
    icon: "⚡",
    title: "实时推理优化",
    desc: "适配多通道流式生成，结合缓存复用和上下文压缩，降低全双工交互延迟。",
  },
];

const DEMO_TABS = ["自然打断", "附和反馈", "长语义理解", "具身陪伴"];

const DEMO_CASES = [
  {
    tab: "自然打断",
    scene: "旅行规划",
    title: "用户中途改需求，系统立即调整计划",
    summary: "系统正在规划杭州行程时，用户突然补充“下午才到”，模型立即停止当前回答并更新计划。",
    ability: ["用户可打断", "停止并改口", "上下文更新"],
    value: "用户不必听完冗长回答，可以像真人沟通一样随时修正需求。",
    script: [
      "用户：帮我规划一个周末杭州行程，预算不要太高，最好轻松一点。",
      "系统：可以，第一天上午可以先去西湖，下午去灵隐寺……",
      "用户打断：等等，我上午可能到不了，下午才到。",
      "系统：明白，那我把行程改成下午抵达后从西湖周边开始，第一天不安排太满。",
    ],
    turn: "系统开始回答后，用户中途打断并补充新条件。",
    materials: ["用户音频", "系统音频", "web 录屏", "输出文本"],
    score: "★★★★☆",
    usage: "首页展示 / 宣传视频 / web demo",
    tech: "控制通道 / 语义对齐通道 / 全双工打断处理",
  },
  {
    tab: "自然打断",
    scene: "产品问答",
    title: "系统长回答时，用户要求变短",
    summary: "系统正在展开解释，用户直接要求“简单说”，模型停止长回答并切换为简洁版本。",
    ability: ["长回答截断", "表达风格切换", "即时改写"],
    value: "用户可以控制回答长度，减少等待与信息负担。",
    script: [
      "用户：帮我解释一下这个功能的优势。",
      "系统：这个功能主要体现在三个层面，首先是……",
      "用户打断：简单说，别太长。",
      "系统：好，简单说就是：更快响应、更自然交流、更适合数字人展示。",
    ],
    turn: "用户不改变主题，只改变表达约束。",
    materials: ["用户音频", "系统音频", "web 录屏", "输出文本"],
    score: "★★★★☆",
    usage: "产品宣传 / 首页展示",
    tech: "控制通道 / 上下文约束更新",
  },
  {
    tab: "自然打断",
    scene: "半双工对照",
    title: "半双工等待 vs 全双工自然插话",
    summary: "同一段需求在半双工和 Lychee-FD 中对照展示，让普通用户直观看懂差异。",
    ability: ["对照展示", "用户插话", "停止响应"],
    value: "适合解释 full-duplex 的用户价值，不需要懂论文也能看懂。",
    script: [
      "Before：用户只能等待系统说完。",
      "After：用户中途补充新条件，系统立即停止并接住新需求。",
    ],
    turn: "同一问题用两种交互范式展示体验差异。",
    materials: ["对照录屏", "用户音频", "系统音频", "说明字幕"],
    score: "★★★★★",
    usage: "宣传视频 / 科普讲解 / web demo",
    tech: "半双工对照 / 全双工交互",
  },
  {
    tab: "附和反馈",
    scene: "用户附和",
    title: "用户说“嗯”“对”，系统不误停",
    summary: "用户边听系统讲话边附和，模型识别为短反馈而非真实打断，继续自然表达。",
    ability: ["短反馈识别", "不误停", "继续表达"],
    value: "用户可以像真人聊天一样给反馈，不会把系统打断。",
    script: [
      "系统：我建议你先把周末安排分成轻松和备选两层……",
      "用户：嗯，对。",
      "系统：然后第一天不要排太满，第二天再安排核心景点。",
    ],
    turn: "用户有声音输入，但语义上只是附和。",
    materials: ["用户音频", "系统音频", "web 录屏", "输出文本"],
    score: "★★★★☆",
    usage: "web demo / 技术解释",
    tech: "控制通道 / Backchannel 判断",
  },
  {
    tab: "附和反馈",
    scene: "模型附和",
    title: "用户长段表达时，模型用短反馈降低等待焦虑",
    summary: "用户连续讲述需求或情绪，模型用“嗯”“我在听”等短反馈表示倾听，但不抢走话轮。",
    ability: ["模型附和", "低打扰反馈", "不抢话"],
    value: "让用户感到被倾听，适合陪伴、咨询、会议记录等场景。",
    script: [
      "用户：我最近有点累，事情很多，又不知道怎么安排……",
      "系统短反馈：嗯，我在听。",
      "用户：主要是每天都被临时任务打断。",
      "系统：明白，我们可以先把任务分成必须做和可以延后的两类。",
    ],
    turn: "模型在用户未结束时只做短反馈，不提前给长答案。",
    materials: ["用户音频", "系统音频", "时间轴标注", "输出文本"],
    score: "★★★★☆",
    usage: "数字人展示 / 情感陪伴 / 宣传视频",
    tech: "Backchannel 生成 / 控制通道 / 语义对齐",
  },
  {
    tab: "附和反馈",
    scene: "Backchannel 四分类",
    title: "倾听、理解、赞同、情感四类短反馈",
    summary: "展示“嗯哼 / 明白 / 是的是的 / 啊？”等不同短反馈在语境中的作用。",
    ability: ["倾听反馈", "理解反馈", "赞同反馈", "情感反馈"],
    value: "让数字人和语音助手更像真人，而不是冷冰冰地等待。",
    script: [
      "倾听：嗯哼 / 嗯 / 是",
      "理解：明白 / 哦 / 原来",
      "赞同：是的是的 / 对",
      "情感：啊？ / 唉",
    ],
    turn: "不同短反馈需要结合语境使用，不能只靠声音长度判断。",
    materials: ["四类音频", "字幕文本", "web 录屏", "分类说明"],
    score: "★★★★☆",
    usage: "技术博客 / demo 集合页",
    tech: "Backchannel 分类 / 反馈生成",
  },
  {
    tab: "长语义理解",
    scene: "犹豫停顿",
    title: "用户犹豫停顿，系统不抢话",
    summary: "用户中途思考、停顿、补充条件，系统保持倾听，不把短暂停顿误判为结束。",
    ability: ["自然等待", "不抢话", "恢复上下文"],
    value: "用户可以自然组织语言，不用担心系统抢答。",
    script: [
      "用户：我想做一个技术展示页，嗯……可能要偏官网一点。",
      "系统短反馈：嗯。",
      "用户：但是也不能太营销，要有论文深度。",
      "系统：明白，这个页面需要同时兼顾科研可信和产品展示感。",
    ],
    turn: "停顿并不等于话轮结束，模型需要持续判断意图。",
    materials: ["用户音频", "系统音频", "时间轴标注", "输出文本"],
    score: "★★★★★",
    usage: "宣传视频 / 技术解释 / web demo",
    tech: "话轮控制 / 语义对齐通道",
  },
  {
    tab: "长语义理解",
    scene: "会议助理",
    title: "长段记录中只短反馈，不打断用户",
    summary: "用户连续描述会议内容，系统只给低打扰反馈，并在结束后整理结构化摘要。",
    ability: ["长语义保持", "短反馈", "结构化整理"],
    value: "适合会议记录、采访记录、口述整理等真实长语音场景。",
    script: [
      "用户：今天会议主要有三件事，第一是数据采集规范……",
      "系统短反馈：明白。",
      "用户：第二是每个 case 都要有音频、录屏和元信息。",
      "系统：我整理成三部分：采集内容、命名规范、质量检查。",
    ],
    turn: "用户长段表达期间，模型要收集约束而不是抢答。",
    materials: ["用户音频", "系统音频", "输出文本", "结构化摘要"],
    score: "★★★★☆",
    usage: "会议助理 demo / 官网案例",
    tech: "长语义理解 / 语义锚点",
  },
  {
    tab: "长语义理解",
    scene: "复杂任务改写",
    title: "用户补充条件，系统边听边整理",
    summary: "用户不断追加格式、语气和展示要求，系统保持语义一致并更新输出方案。",
    ability: ["实时约束收集", "上下文更新", "复杂任务改写"],
    value: "适合文案、策划、代码需求沟通等复杂协作任务。",
    script: [
      "用户：页面要中文优先，非必要英文去掉。",
      "系统短反馈：好。",
      "用户：还要有 demo、指标、数字人展示，不要像普通论文笔记。",
      "系统：明白，我会按技术发布页结构来组织。",
    ],
    turn: "多个约束分散出现，模型需要持续更新任务状态。",
    materials: ["用户音频", "系统音频", "web 录屏", "输出文本"],
    score: "★★★★☆",
    usage: "web demo / 产品案例",
    tech: "语义通道 / 上下文压缩",
  },
  {
    tab: "具身陪伴",
    scene: "情感陪聊 / 哄睡",
    title: "温柔短反馈，让数字人更像在陪伴",
    summary: "用户表达疲惫或情绪时，模型用低打扰短反馈和温柔语音维持陪伴感。",
    ability: ["温柔音色", "情绪感知", "自然等待", "低打扰短反馈"],
    value: "适合数字人、陪伴型助手、睡前聊天等展示。",
    script: [
      "用户：我今天有点累，不太想说很多。",
      "系统：嗯，我在。你可以慢慢说，也可以先休息一下。",
      "用户：就是感觉事情有点多。",
      "系统短反馈：嗯，辛苦了。",
    ],
    turn: "模型不是急着解决问题，而是先建立陪伴感。",
    materials: ["用户音频", "系统音频", "数字人版本视频", "输出文本"],
    score: "★★★★☆",
    usage: "数字人展示 / 宣传视频",
    tech: "情感反馈 / 声学通道 / 控制通道",
  },
  {
    tab: "具身陪伴",
    scene: "烹饪手忙场景",
    title: "用户双手忙碌时，边说边改指令",
    summary: "用户在做饭时不断补充限制，模型实时调整步骤并保持简短。",
    ability: ["免手交互", "实时改口", "简短反馈"],
    value: "适合智能家居、机器人助手和多任务场景。",
    script: [
      "用户：帮我看一下这个菜下一步做什么。",
      "系统：先把火调小，然后加入调料……",
      "用户打断：等等，我没有生抽。",
      "系统：没关系，可以先用少量盐和蚝油替代，味道会更柔和。",
    ],
    turn: "用户处于手忙场景，交互必须短、快、可打断。",
    materials: ["用户音频", "系统音频", "机器人录屏", "输出文本"],
    score: "★★★★☆",
    usage: "机器人展示 / 应用案例",
    tech: "具身交互 / 实时控制",
  },
  {
    tab: "具身陪伴",
    scene: "数字人展示",
    title: "语音驱动表情，让交互能力具身化",
    summary: "Lychee-FD 生成语音、文本与控制信号，下游模块驱动数字人表情和口型。",
    ability: ["表情驱动", "口型同步", "自然等待", "短反馈"],
    value: "把模型能力从语音层面扩展到可见、可感知的具身交互体验。",
    script: [
      "Lychee-FD：负责听、说、理解和控制。",
      "Audio2Face：负责表情生成。",
      "数字人渲染：负责把反馈可视化。",
    ],
    turn: "展示重点从语音模型转向具身交互系统。",
    materials: ["系统音频", "数字人视频", "封面截图", "时间轴标注"],
    score: "★★★★☆",
    usage: "官网应用区 / 宣传视频",
    tech: "Lychee-FD / Audio2Face / 数字人渲染",
  },
];

const META_FIELDS = [
  "case 编号", "采集日期", "采集人", "对话场景", "一句话简介", "展示能力", "用户价值", "对话脚本",
  "关键转折点", "复现稳定性评分", "音频质量", "录屏质量", "是否适合官网", "是否适合宣传视频", "是否适合数字人", "推荐展示位置", "对应技术点", "备注",
];

function $(selector) {
  return document.querySelector(selector);
}

function createMetricCard(metric) {
  const card = document.createElement("article");
  card.className = "metric-card";
  card.innerHTML = `<strong data-counter="${metric.value}">${metric.value}</strong><span>${metric.label}</span>`;
  return card;
}

function renderMetrics() {
  const heroMetrics = $("#heroMetrics");
  const resultMetrics = $("#resultMetrics");
  METRICS.forEach((metric) => {
    heroMetrics.appendChild(createMetricCard(metric));
    resultMetrics.appendChild(createMetricCard(metric));
  });
}

function renderHighlights() {
  const grid = $("#highlightGrid");
  HIGHLIGHTS.forEach((item) => {
    const card = document.createElement("article");
    card.className = "highlight-card";
    card.innerHTML = `
      <div class="highlight-icon">${item.icon}</div>
      <h3>${item.title}</h3>
      <p>${item.desc}</p>
    `;
    grid.appendChild(card);
  });
}

function renderTabs() {
  const tabs = $("#demoTabs");
  DEMO_TABS.forEach((tab, index) => {
    const button = document.createElement("button");
    button.className = "tab-button";
    button.type = "button";
    button.role = "tab";
    button.setAttribute("aria-selected", index === 0 ? "true" : "false");
    button.textContent = tab;
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab-button").forEach((btn) => btn.setAttribute("aria-selected", "false"));
      button.setAttribute("aria-selected", "true");
      renderDemoCards(tab);
    });
    tabs.appendChild(button);
  });
}

function materialButton(label) {
  const iconMap = {
    用户音频: "🎙",
    系统音频: "🔊",
    "web 录屏": "▶",
    输出文本: "文",
    对照录屏: "▶",
    说明字幕: "字",
    时间轴标注: "轴",
    四类音频: "🎧",
    字幕文本: "字",
    分类说明: "注",
    结构化摘要: "摘",
    数字人版本视频: "人",
    机器人录屏: "机",
    数字人视频: "人",
    封面截图: "图",
    系统音频: "🔊",
  };
  return `<button class="material-button" type="button" title="占位：后续替换真实素材">${iconMap[label] || "□"} ${label}</button>`;
}

function renderDemoCards(activeTab = DEMO_TABS[0]) {
  const grid = $("#demoGrid");
  grid.innerHTML = "";
  DEMO_CASES.filter((item) => item.tab === activeTab).forEach((item) => {
    const card = document.createElement("article");
    card.className = "demo-card";
    card.innerHTML = `
      <span class="demo-scene">场景：${item.scene}</span>
      <h3>${item.title}</h3>
      <p>${item.summary}</p>
      <div class="demo-ability">${item.ability.map((ability) => `<span>${ability}</span>`).join("")}</div>
      <div class="demo-script">${item.script.map((line) => `<p>${line}</p>`).join("")}</div>
      <div class="demo-materials">${item.materials.map(materialButton).join("")}</div>
      <div class="demo-meta">
        <span><b>用户价值：</b>${item.value}</span>
        <span><b>关键转折点：</b>${item.turn}</span>
        <span><b>稳定性评分：</b>${item.score}</span>
        <span><b>推荐用途：</b>${item.usage}</span>
        <span><b>对应技术点：</b>${item.tech}</span>
      </div>
    `;
    grid.appendChild(card);
  });
}

function renderMetaFields() {
  const grid = $("#metaGrid");
  META_FIELDS.forEach((field) => {
    const pill = document.createElement("span");
    pill.textContent = field;
    grid.appendChild(pill);
  });
}

function renderAudioTokens() {
  const row = $("#audioTokens");
  for (let i = 0; i < 34; i += 1) {
    const token = document.createElement("span");
    token.style.opacity = String(0.45 + (i % 7) * 0.07);
    row.appendChild(token);
  }

  document.querySelectorAll(".semantic-energy em").forEach((item, index) => {
    item.style.setProperty("--i", index);
  });
}

function enableRevealAnimation() {
  const items = document.querySelectorAll(".reveal");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  items.forEach((item) => observer.observe(item));
}

function animateMetricCounters() {
  const counters = document.querySelectorAll("[data-counter]");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const el = entry.target;
        const target = el.dataset.counter;
        const numeric = parseFloat(target.replace(/[^0-9.]/g, ""));
        const suffix = target.replace(/[0-9.]/g, "");
        const duration = 900;
        const start = performance.now();

        function step(now) {
          const progress = Math.min((now - start) / duration, 1);
          const eased = 1 - Math.pow(1 - progress, 3);
          const value = numeric * eased;
          const fixed = numeric < 10 && !suffix.includes("ms") ? value.toFixed(2) : value.toFixed(1);
          el.textContent = `${fixed.replace(/\.0$/, "")}${suffix}`;
          if (progress < 1) requestAnimationFrame(step);
          else el.textContent = target;
        }

        requestAnimationFrame(step);
        observer.unobserve(el);
      });
    },
    { threshold: 0.6 }
  );

  counters.forEach((counter) => observer.observe(counter));
}

function attachPlaceholderEvents() {
  document.addEventListener("click", (event) => {
    const target = event.target;
    if (target.matches(".material-button, .play-button, .audio-button")) {
      const original = target.textContent;
      target.textContent = "素材占位，待替换";
      setTimeout(() => {
        target.textContent = original;
      }, 900);
    }
  });
}

function init() {
  renderMetrics();
  renderHighlights();
  renderTabs();
  renderDemoCards();
  renderMetaFields();
  renderAudioTokens();
  enableRevealAnimation();
  animateMetricCounters();
  attachPlaceholderEvents();
}

init();
