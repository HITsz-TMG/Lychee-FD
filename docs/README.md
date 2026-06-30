# Lychee-FD 中文技术展示页

这是 Lychee-FD 项目的中文静态技术展示页，定位是：**AI 技术官网 + 科研项目展示 + Demo 素材整理页 + 数字人展示入口**。

页面主题：

> 从轮流说话，到自然共话。  
> Lychee-FD 让 AI 同时听、说、思考、停顿与接话。

当前版本已经完成：

- 浅色科技官网风视觉：珍珠银白空间科技风为主，冰蓝数据科技风用于 Demo 与指标，少量极光薰衣草点缀。
- 统一配置管理：导航、指标、技术亮点、Demo case、图片资源、素材规范字段都集中在 `js/config.js`。
- 真实素材接入：图片、视频、音频均预留路径；素材不存在时显示优雅占位，不会破版。
- 页面交互：平滑滚动、Demo tab 切换、图片点击放大、卡片 hover、指标数字递增、移动端基本可读。

---

## 1. 本地预览

### 方式一：直接打开

双击项目根目录下的：

```text
index.html
```

这种方式最简单，但某些浏览器可能会限制本地音频、视频、SVG 或跨文件加载。

### 方式二：启动本地静态服务，推荐

在项目根目录运行：

```bash
python -m http.server 8080
```

然后浏览器打开：

```text
http://127.0.0.1:8080
```

如果图片、音频、视频在双击打开时加载异常，请优先使用这种方式。

---

## 2. 项目目录说明

```text
lychee-fd-showcase/
├── index.html                 # 页面骨架，只放结构和挂载点
├── README.md                  # 项目说明
├── docs/                      # 中文操作文档
│   ├── 使用说明.md
│   ├── 素材替换指南.md
│   ├── Demo_Case_新增指南.md
│   └── 页面结构说明.md
├── assets/                    # 所有素材
│   ├── images/
│   │   ├── paper/             # 论文图、技术根因图、挑战机遇图
│   │   ├── architecture/      # 架构图、三种基础架构对比图
│   │   ├── demo/              # 自然打断、用户附和、模型附和等 Demo 图
│   │   ├── avatar/            # 数字人、机器人、Shennie 图
│   │   ├── charts/            # 雷达图、UTMOS 图、实验结果图
│   │   └── misc/              # 其他图片
│   ├── videos/                # mp4/webm 视频
│   ├── audio/                 # wav/mp3 音频
│   └── docs/                  # PDF 原图或补充材料
├── css/
│   └── styles.css             # 全站样式，颜色变量也在这里
├── js/
│   ├── config.js              # 全站数据配置，后续主要改这里
│   └── main.js                # 渲染逻辑和页面交互
└── legacy/                    # 上一版旧文件备份，正常不用改
```

---

## 3. 后续最常改哪里

### 修改页面数据

主要改：

```text
js/config.js
```

这里集中维护：

- 导航项 `navigation`
- 首页指标 `metrics`
- 技术亮点 `highlights`
- 图片资源 `imageAssets`
- 图片分组 `assetGroups`
- Demo 分类 `demoTabs`
- Demo case 数据 `demoCases`
- 素材规范字段 `packageSpec`

### 修改视觉样式

主要改：

```text
css/styles.css
```

颜色变量在文件最上方：

```css
:root {
  --page-bg: #f5f8ff;
  --ice: #1d8eff;
  --lavender: #9c7cff;
  --text: #102033;
}
```

---

## 4. GitHub Pages 部署

1. 把整个 `lychee-fd-showcase` 项目提交到 GitHub 仓库。
2. 进入仓库页面，点击 **Settings**。
3. 找到 **Pages**。
4. Source 选择 `Deploy from a branch`。
5. Branch 选择 `main`。
6. 目录选择 `/root`。
7. 保存后等待 GitHub 生成访问链接。

如果你以后把页面放到仓库的 `docs/` 目录下，则 Pages 目录选择 `/docs`。当前项目的推荐方式是部署根目录 `/root`。

---

## 5. 常见问题

### 1）图片没有显示，只有“待替换真实素材”

检查两件事：

1. 图片文件是否真的放到了 `assets/images/...` 对应目录。
2. `js/config.js` 里的 `src` 路径是否写对。

例如：

```js
src: "assets/images/charts/result_radar.png"
```

就要求真实文件存在于：

```text
assets/images/charts/result_radar.png
```

### 2）PDF 能不能直接放网页里？

不推荐。PDF 可以放在：

```text
assets/docs/
```

但官网展示建议先导出成 PNG 或 SVG，再放到 `assets/images/...`。这样显示更稳定，也更适合 GitHub Pages。

### 3）新增 Demo 后没有出现在页面上

检查 `demoCases` 里的 `category` 是否和 `demoTabs` 里的分类名称完全一致。例如：

```js
category: "自然打断"
```

必须对应：

```js
demoTabs: ["自然打断", "附和反馈", "长语义理解", "具身陪伴"]
```

### 4）音频、视频按钮为什么显示占位？

因为当前 case 的 `userAudio`、`systemAudio`、`video`、`transcript` 还是空字符串。填入路径后，按钮会变成可点击链接。

---

## 6. 后续维护建议

1. 新增图片时，先放到 `assets/images` 对应分类目录。
2. 再到 `js/config.js` 添加或修改配置。
3. 不建议直接在 `index.html` 里硬编码新图片。
4. PDF 图建议先导出为 PNG 或 SVG。
5. 视频建议压缩后再放入 `assets/videos`。
6. 音频建议统一命名，和 case id 对齐，例如 `case1_user.wav`、`case1_system.wav`。
7. 每个 case 最好配一张封面图、一段系统输出文本、一段 web demo 录屏。
8. 页面正式上线前检查所有空路径和占位素材。
