# Demo Case 新增指南

这份文档说明如何新增一个 Demo case，并绑定图片、音频、视频和输出文本。

---

## 1. Demo 数据在哪里

打开：

```text
js/config.js
```

找到：

```js
demoCases: [ ... ]
```

所有 Demo case 都写在这里。

---

## 2. Demo 分类在哪里

同一个文件中找到：

```js
demoTabs: ["自然打断", "附和反馈", "长语义理解", "具身陪伴"]
```

每个 case 的 `category` 必须和这里的某个名称完全一致，才会出现在对应 tab。

---

## 3. 新增一个 case 的模板

可以复制下面这个模板，放到 `demoCases` 数组中：

```js
{
  id: "case8",
  category: "自然打断",
  title: "用户中途改需求，系统立即调整计划",
  scene: "旅行规划",
  summary: "系统正在回答时，用户补充新条件，模型立即停止并更新计划。",
  ability: ["用户可打断", "停止并改口", "上下文更新"],
  value: "用户不必听完冗长回答，可以像真人沟通一样随时修正需求。",
  script: [
    { role: "用户", text: "帮我规划一个周末杭州行程，预算不要太高，最好轻松一点。" },
    { role: "系统", text: "可以，第一天上午可以先去西湖，下午去灵隐寺……" },
    { role: "用户打断", text: "等等，我上午可能到不了，下午才到。" },
    { role: "系统", text: "明白，那我把行程改成下午抵达后从西湖周边开始，第一天不安排太满。" }
  ],
  keyMoment: "系统开始回答后，用户中途打断并补充新条件。",
  imageAsset: "heroDemo",
  userAudio: "assets/audio/case8_user.wav",
  systemAudio: "assets/audio/case8_system.wav",
  video: "assets/videos/case8_recording.mp4",
  transcript: "assets/docs/case8_transcript.txt",
  stability: 4,
  usage: ["首页展示", "宣传视频", "web demo"],
  tech: ["控制通道", "语义对齐通道", "全双工打断处理"]
}
```

---

## 4. 字段解释

```text
id            case 唯一编号，不要重复
category      出现在哪个 tab，必须和 demoTabs 一致
title         卡片标题
scene         场景，例如旅行规划、会议助理、情感陪伴
summary       一句话简介
ability       展示能力标签
value         用户价值
script        对话脚本，每句包含 role 和 text
keyMoment     关键转折点
imageAsset    绑定 imageAssets 里的图片 id
userAudio     用户音频路径，可为空
systemAudio   系统音频路径，可为空
video         web demo 录屏路径，可为空
transcript    输出文本路径，可为空
stability     稳定性评分，1 到 5
usage         推荐用途
tech          对应技术点
```

---

## 5. 如何绑定图片

推荐先在 `imageAssets` 里新增图片配置：

```js
case8Cover: {
  id: "case8Cover",
  title: "旅行规划打断 Demo",
  src: "assets/images/demo/case8_cover.png",
  type: "image",
  section: "Demo 展示",
  tags: ["自然打断"],
  enabled: true,
  description: "用户中途改口，系统立即调整计划。"
}
```

然后在 case 中写：

```js
imageAsset: "case8Cover"
```

如果 `imageAsset` 为空，页面会显示 Demo 封面占位。

---

## 6. 如何绑定音频、视频、文本

把文件放到：

```text
assets/audio/case8_user.wav
assets/audio/case8_system.wav
assets/videos/case8_recording.mp4
assets/docs/case8_transcript.txt
```

然后在 case 中填写：

```js
userAudio: "assets/audio/case8_user.wav",
systemAudio: "assets/audio/case8_system.wav",
video: "assets/videos/case8_recording.mp4",
transcript: "assets/docs/case8_transcript.txt",
```

如果字段为空，页面会显示“占位”按钮，不会报错。

---

## 7. 稳定性评分怎么写

`stability` 使用 1 到 5 的数字：

```text
5：非常稳定，适合首页或宣传视频
4：比较稳定，适合 web demo 或技术展示
3：可展示，但需要筛选素材
2：不建议公开展示，只做内部分析
1：问题较多，暂不展示
```

页面会自动显示为星级。

---

## 8. 如何让 case 出现在对应 tab

只需要保证：

```js
category: "附和反馈"
```

和：

```js
demoTabs: ["自然打断", "附和反馈", "长语义理解", "具身陪伴"]
```

完全一致即可。
