# Lychee-FD Static Page Draft

这是一个可直接放到 GitHub Pages 的静态页面初稿，参考现代 AI 产品/论文博客页面的结构，内容围绕 Lychee-FD 论文展开。

## 目录结构

```text
lychee_fd_static_page/
├── index.html              # 页面结构，主要 section 都在这里
├── styles.css              # 视觉风格、响应式布局、动画、颜色变量
├── script.js               # Demo 数据渲染、滚动进度条、进入动画、导航高亮
└── assets/
    ├── audio/              # 当前是占位音频，后续替换真实 demo
    └── img/                # 后续放论文图、架构图、视频封面
```

## 本地预览

```bash
cd lychee_fd_static_page
python -m http.server 8080
```

浏览器打开：`http://127.0.0.1:8080`

## 挂到 GitHub Pages

方案 A：把这些文件放到仓库根目录，然后在 GitHub Pages 里选择 `main / root`。

方案 B：把这些文件放到仓库 `docs/` 目录，然后在 GitHub Pages 里选择 `main / docs`。

## 后续最建议先改的地方

1. `index.html`：替换 Hero 区标题、副标题、作者、按钮链接。
2. `script.js`：替换 `demos` 数组中的音频路径和 transcript。
3. `styles.css`：在 `:root` 改颜色变量，比如 `--green`、`--bg`，可以快速换整体风格。
4. `assets/img/`：加入论文 Figure 1–5、架构图、真实 demo 封面。

## 设计思路

- Hero：像产品发布页一样先打出大标题、关键指标、CTA。
- Insights：用两个核心发现解释“为什么要这么设计”。
- Architecture：对比 Thinker-Talker、Native End-to-End、Lychee-FD。
- Demos：先做占位音频和文字，后续直接换真实样例。
- Results：保留关键指标卡片和表格，后续可以换成图表。
- Roadmap：提醒后续美化方向。
