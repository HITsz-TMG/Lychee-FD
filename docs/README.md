# Lychee-FD Project Page v33

本版本基于当前 v32 页面做局部优化：只调整 Demo 视频封面播放按钮，并清理确定已经无入口的旧交互残留。页面主线、章节内容、导航、首页宣传视频、技术介绍、评测效果和数字人区域均保持原样。

## 本轮改动

- Demo 视频封面统一使用一个共享播放按钮组件：`video-play-button demo-video-load`。
- 播放按钮固定居中，默认直径 `160px`，通过 `--play-button-size` 控制。
- 按钮使用轻量 CSS 绘制：圆形蓝白玻璃拟态外观、深蓝播放三角形、克制外发光与阴影。
- 删除旧版 Demo 播放按钮的文字胶囊样式和多处重复覆盖，Demo 按钮样式集中在 `css/styles.css` 的 v33 区块。
- 保留 Demo 视频懒加载：初始只加载 poster，点击按钮后才创建 `<video>` 并挂载真实视频资源。
- 保留播放新 Demo 时释放其他 Demo 视频节点的逻辑，避免多个视频同时占用解码资源。
- 清理 `main.js` 中已经没有入口的旧版指标卡片弹窗属性，以及未使用的评分星级函数。

## 当前 Demo 视频

当前 Demo 配置只使用以下 3 个视频：

- `assets/videos/demo-ai-product-comment.mp4`
- `assets/videos/demo-user-strategy.mp4`
- `assets/videos/demo-cooking-beef.mp4`

首页宣传视频：

- `assets/videos/xuanchuan.mp4`

## 性能注意

代码层继续避免一次性给多个 Demo 视频挂载 `src`，但原视频文件仍然较大。后续如果继续卡顿，优先处理视频素材本身：压缩码率、统一 H.264、将 60fps 降到 30fps，并使用 `ffmpeg -movflags +faststart` 把 `moov atom` 移到文件开头。
