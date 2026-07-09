(() => {
  const CONFIG = window.LYCHEE_CONFIG || {};
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function escapeHTML(value = "") {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function getAsset(id) {
    return CONFIG.imageAssets?.[id] || null;
  }


  function tagHTML(tags = []) {
    return tags.map((tag) => `<span>${escapeHTML(tag)}</span>`).join("");
  }

  function placeholderHTML(title = "内容暂不可用", desc = "当前内容暂时无法显示，请稍后查看。") {
    return `
      <div class="media-placeholder" role="img" aria-label="${escapeHTML(title)}">
        <div class="placeholder-orbit"><span></span><span></span><span></span></div>
        <strong>${escapeHTML(title)}</strong>
        <p>${escapeHTML(desc)}</p>
      </div>
    `;
  }

  function renderInlineSegments(segments = []) {
    if (!Array.isArray(segments) || !segments.length) return "";
    return segments.map((segment) => {
      const text = escapeHTML(segment.text || "");
      if (segment.strong) return `<strong>${text}</strong>`;
      if (segment.accent) return `<span class="text-accent">${text}</span>`;
      return text;
    }).join("");
  }

  function renderTextBlock(block = {}, fallbackClass = "") {
    const className = block.className || fallbackClass;
    const classAttr = className ? ` class="${escapeHTML(className)}"` : "";
    const segments = Array.isArray(block.segments) && block.segments.length
      ? block.segments
      : [{ text: block.text || "" }];
    return `<p${classAttr}>${renderInlineSegments(segments)}</p>`;
  }

  function renderTextBlocks(blocks = [], fallbackClass = "") {
    if (!Array.isArray(blocks)) return "";
    return blocks.map((block) => renderTextBlock(block, fallbackClass)).join("");
  }

  function createAssetCard(assetId, options = {}) {
    const asset = typeof assetId === "string" ? getAsset(assetId) : assetId;
    const variant = options.variant || "standard";
    const card = document.createElement("article");
    card.className = `asset-card asset-${variant}`;

    if (!asset) {
      card.innerHTML = placeholderHTML("内容暂不可用", "当前内容暂时无法显示，请稍后查看。")
        + `<div class="asset-caption"><h3>内容暂不可用</h3><p>当前图片暂时无法显示。</p></div>`;
      return card;
    }

    const enabled = asset.enabled !== false;
    const hasSrc = Boolean(asset.src && asset.src.trim());
    const tags = tagHTML(asset.tags || []);
    const caption = `
      <div class="asset-caption">
        ${tags ? `<div class="asset-tags">${tags}</div>` : ""}
        <h3>${escapeHTML(asset.title)}</h3>
        <p>${escapeHTML(asset.description || "")}</p>
      </div>
    `;

    if (asset.type === "reservedVideo") {
      card.classList.add("asset-reserved-video");
      card.innerHTML = `
        <div class="media-frame reserved-demo-media-frame">
          ${renderVideoPlayer(asset, "reserved-demo-player avatar-demo-player")}
        </div>
        ${caption}
      `;
      return card;
    }

    if (!enabled || !hasSrc) {
      card.innerHTML = placeholderHTML(
        enabled ? "内容暂不可用" : "内容暂不可用",
        "当前内容暂时无法显示，请稍后查看。"
      ) + caption;
      return card;
    }

    if (asset.type === "video") {
      card.innerHTML = `
        <div class="media-frame">
          <video src="${escapeHTML(asset.src)}" controls preload="none"></video>
        </div>
        ${caption}
      `;
      return card;
    }

    const dimensionAttrs = asset.width && asset.height
      ? ` width="${escapeHTML(asset.width)}" height="${escapeHTML(asset.height)}"`
      : "";

    card.innerHTML = `
      <div class="media-frame">
        <img src="${escapeHTML(asset.src)}" alt="${escapeHTML(asset.title)}"${dimensionAttrs} loading="lazy" decoding="async" fetchpriority="low" data-lightbox-src="${escapeHTML(asset.src)}" data-lightbox-title="${escapeHTML(asset.title)}" />
      </div>
      ${caption}
    `;

    const img = $("img", card);
    img.addEventListener("error", () => {
      const frame = $(".media-frame", card);
      frame.innerHTML = placeholderHTML("内容暂不可用", "当前图片暂时无法显示。");
    });

    return card;
  }

  function mountAsset(target) {
    const assetId = target.dataset.assetId;
    const variant = target.dataset.assetVariant || "standard";
    target.innerHTML = "";
    target.appendChild(createAssetCard(assetId, { variant }));
  }

  function mountAssetList(target) {
    const groupName = target.dataset.assetList;
    const variant = target.dataset.assetVariant || "standard";
    const ids = CONFIG.assetGroups?.[groupName] || [];
    target.innerHTML = "";
    if (!ids.length) {
      target.appendChild(createAssetCard(null, { variant }));
      return;
    }
    ids.forEach((id) => target.appendChild(createAssetCard(id, { variant })));
  }


  function renderProblemFigure(figure = {}) {
    const title = figure.title || "路线对比图";
    const src = (figure.src || "").trim();
    if (!src) {
      return placeholderHTML(title, figure.placeholder || "请在 config.js 中配置这张核心对比图的 src。");
    }
    const dimensionAttrs = figure.width && figure.height
      ? ` width="${escapeHTML(figure.width)}" height="${escapeHTML(figure.height)}"`
      : "";
    const alt = figure.alt || title;
    return `
      <figure class="problem-main-figure has-image" aria-label="${escapeHTML(title)}">
        <img src="${escapeHTML(src)}" alt="${escapeHTML(alt)}"${dimensionAttrs} loading="lazy" decoding="async" fetchpriority="low" data-lightbox-src="${escapeHTML(src)}" data-lightbox-title="${escapeHTML(title)}" />
      </figure>
    `;
  }

  function renderProblemRouteCard(card = {}, index = 0) {
    return `
      <article>
        <span>${escapeHTML(card.label || String(index + 1).padStart(2, "0"))}</span>
        <h3>${escapeHTML(card.title || "")}</h3>
        <p>${escapeHTML(card.text || "")}</p>
      </article>
    `;
  }

  function renderProblemSection() {
    const mount = $("[data-problem-section]");
    if (!mount) return;

    const section = CONFIG.problemSection || {};
    const routeCards = Array.isArray(section.routeCards) ? section.routeCards.map(renderProblemRouteCard).join("") : "";
    const conclusion = section.conclusion || {};

    mount.innerHTML = `
      <div class="section-head reveal problem-section-head">
        <p class="section-index">${escapeHTML(section.sectionIndex || "技术介绍 02 / 科学问题")}</p>
        <h2>${escapeHTML(section.title || "")}</h2>
        <div class="problem-flow-card">
          <div class="problem-flow-intro">
            ${renderTextBlocks(section.intro || [], "problem-lead")}
          </div>
          ${renderProblemFigure(section.figure || {})}
          ${routeCards ? `<div class="problem-route-grid" aria-label="语音交互路线对比">${routeCards}</div>` : ""}
          <div class="problem-route-summary">
            ${conclusion.title ? `<h3>${escapeHTML(conclusion.title)}</h3>` : ""}
            ${renderTextBlocks(conclusion.paragraphs || [])}
          </div>
          ${section.callout ? `<div class="problem-final-callout">${renderTextBlock(section.callout)}</div>` : ""}
        </div>
      </div>
    `;
  }

  function renderResearchFigure(figure = {}) {
    const title = figure.title || "原图展示位";
    const src = (figure.src || "").trim();
    if (!src) {
      return `
        <div class="research-image-slot" aria-label="${escapeHTML(title)}预留位">
          <div class="research-slot-inner">
            <span>${escapeHTML(figure.slotLabel || "Original Figure Slot")}</span>
            <strong>${escapeHTML(title)}</strong>
            <p>${escapeHTML(figure.placeholder || "后续在 js/config.js 中配置 src 后自动显示原图。")}</p>
          </div>
        </div>
      `;
    }

    const dimensionAttrs = figure.width && figure.height
      ? ` width="${escapeHTML(figure.width)}" height="${escapeHTML(figure.height)}"`
      : "";
    const alt = figure.alt || title;
    return `
      <div class="research-image-slot has-image" aria-label="${escapeHTML(title)}">
        <img src="${escapeHTML(src)}" alt="${escapeHTML(alt)}"${dimensionAttrs} loading="lazy" decoding="async" fetchpriority="low" data-lightbox-src="${escapeHTML(src)}" data-lightbox-title="${escapeHTML(title)}" />
      </div>
    `;
  }

  function renderResearchModule(item = {}, index = 0, extraClass = "") {
    const delayClass = item.delayClass || `delay-${Math.min(index + 1, 3)}`;
    const className = extraClass.trim() ? ` ${escapeHTML(extraClass.trim())}` : "";
    return `
      <div class="research-module${className} reveal ${escapeHTML(delayClass)}">
        <div class="research-copy">
          <span class="research-kicker">${escapeHTML(item.kicker || "")}</span>
          <h3>${escapeHTML(item.title || "")}</h3>
          <p>${escapeHTML(item.description || item.lead || "")}</p>
        </div>
        ${renderResearchFigure(item.figure || {})}
      </div>
    `;
  }

  function renderResearchPoint(point = {}) {
    return `
      <article class="research-solution-point">
        <span>${escapeHTML(point.label || "")}</span>
        <h4>${escapeHTML(point.title || "")}</h4>
        <p>${escapeHTML(point.text || "")}</p>
      </article>
    `;
  }

  function renderResearchDetail(detail = {}) {
    if (!detail || (!detail.title && !Array.isArray(detail.paragraphs))) return "";
    return `
      <div class="research-solution-detail">
        ${detail.title ? `<h4>${escapeHTML(detail.title)}</h4>` : ""}
        ${renderTextBlocks(detail.paragraphs || [])}
      </div>
    `;
  }

  function renderResearchSolution(group = {}, index = 0) {
    const delayClass = group.delayClass || `delay-${Math.min(index + 1, 3)}`;
    const points = Array.isArray(group.points) ? group.points.map(renderResearchPoint).join("") : "";
    const leadBlocks = Array.isArray(group.leadParagraphs) && group.leadParagraphs.length
      ? group.leadParagraphs
      : [{ text: group.lead || group.description || "" }];
    const detailHTML = renderResearchDetail(group.detail || {});
    const noteHTML = group.note ? `<p class="research-solution-note">${escapeHTML(group.note)}</p>` : "";
    const figureHTML = renderResearchFigure(group.figure || {});

    return `
      <div class="research-solution-card research-solution-card-wide reveal ${escapeHTML(delayClass)}">
        <div class="research-solution-copy">
          <span class="research-kicker">${escapeHTML(group.kicker || "")}</span>
          <h3>${escapeHTML(group.title || "")}</h3>
          <div class="research-solution-lead-list">${renderTextBlocks(leadBlocks, "research-solution-lead")}</div>
          ${points ? `<div class="research-solution-points">${points}</div>` : ""}
        </div>
        <div class="research-solution-figure-row">
          ${figureHTML}
        </div>
        ${(detailHTML || noteHTML) ? `<div class="research-solution-bottom">${detailHTML}${noteHTML}</div>` : ""}
      </div>
    `;
  }

  function renderResearchGroup(group = {}, index = 0) {
    if (group.type === "figurePair") {
      const modules = Array.isArray(group.modules) ? group.modules : [];
      return `
        <div class="research-figure-pair reveal ${escapeHTML(group.delayClass || `delay-${Math.min(index + 1, 3)}`)}" data-research-group="${escapeHTML(group.id || "figurePair")}">
          ${modules.map((item, itemIndex) => renderResearchModule(item, itemIndex, "research-science-module research-figure-card")).join("")}
        </div>
      `;
    }

    if (group.type === "solutionPlaceholder") {
      return renderResearchSolution(group, index);
    }

    return renderResearchModule(group, index);
  }

  function renderResearchPath() {
    const mount = $("[data-research-path]");
    if (!mount) return;

    const path = CONFIG.researchPath || {};
    const groups = Array.isArray(path.groups) && path.groups.length
      ? path.groups
      : (Array.isArray(path.modules) ? path.modules : []);
    const groupHTML = groups.map((group, index) => renderResearchGroup(group, index)).join("");

    mount.innerHTML = `
      <div class="section-head reveal">
        <p class="section-index">${escapeHTML(path.sectionIndex || "技术介绍 03")}</p>
        <h2>${escapeHTML(path.title || "")}</h2>
        <p>${escapeHTML(path.description || "")}</p>
      </div>
      ${groupHTML}
      ${path.summary ? `<p class="root-summary reveal delay-3">${escapeHTML(path.summary)}</p>` : ""}
    `;
  }

  function renderSiteText() {
    const site = CONFIG.site || {};
    $$("[data-config-text]").forEach((node) => {
      const key = node.dataset.configText;
      if (site[key]) node.textContent = site[key];
    });
    const title = $("title");
    if (title && site.title && site.subtitle) title.textContent = `${site.title}｜${site.subtitle}`;
  }

  function renderNavigation() {
    const mounts = $$('[data-nav-mount]');
    const legacyMount = $("#navMount");
    if (!mounts.length && legacyMount) mounts.push(legacyMount);
    if (!mounts.length) return;

    const html = (CONFIG.navigation || [])
      .map((item) => `<a href="#${escapeHTML(item.target)}">${escapeHTML(item.label)}</a>`)
      .join("");
    mounts.forEach((nav) => { nav.innerHTML = html; });
  }

  function renderHeroPills() {
    const mount = $("#heroPills");
    if (!mount) return;
    mount.innerHTML = (CONFIG.heroPills || []).map((pill) => `<span>${escapeHTML(pill)}</span>`).join("");
  }

  function renderHeroPromoVideo() {
    const mount = $("[data-hero-promo]");
    if (!mount) return;

    const promo = CONFIG.heroPromoVideo || {};
    const title = promo.title || "Lychee-FD 交互演示";
    const videoSrc = promo.video || promo.src || "";
    const poster = promo.poster || "";
    const preload = promo.preload || "none";
    const screenState = videoSrc ? "has-video-source is-poster-only" : "is-placeholder no-video-source";
    const posterHTML = poster
      ? `<img class="hero-promo-poster" src="${escapeHTML(poster)}" alt="${escapeHTML(title)} 封面" width="960" height="540" loading="eager" decoding="async" fetchpriority="high" />`
      : "";

    mount.innerHTML = `
      <article class="hero-promo-card hero-promo-card-static hero-promo-card-clean" data-hero-promo-card aria-label="Lychee-FD 交互演示">
        <div class="hero-promo-screen ${screenState}" data-hero-promo-player data-src="${escapeHTML(videoSrc)}" data-poster="${escapeHTML(poster)}" data-title="${escapeHTML(title)}" data-preload="${escapeHTML(preload)}" aria-label="Lychee-FD 交互演示播放区域">
          ${posterHTML}
          <div class="hero-promo-placeholder" aria-hidden="true"></div>
          <button class="hero-promo-main-play" type="button" data-hero-promo-play aria-label="播放交互演示"><span></span></button>
          <button class="hero-promo-fullscreen" type="button" data-hero-fullscreen aria-label="全屏播放交互演示">全屏</button>
        </div>
      </article>
    `;
  }

  function metricCard(metric) {
    return `
      <article class="metric-card">
        <span class="metric-note">${escapeHTML(metric.note || "")}</span>
        <strong data-counter="${escapeHTML(metric.value)}">0</strong>
        <p>${escapeHTML(metric.label)}</p>
      </article>
    `;
  }

  function renderMetrics() {
    const metrics = CONFIG.metrics || [];
    $$("[data-metrics]").forEach((mount) => {
      mount.innerHTML = metrics.map((metric) => metricCard(metric)).join("");
    });
  }

  function getDemoVideoSources(item = {}) {
    const sources = [];
    const add = (value) => {
      if (!value || typeof value !== "string") return;
      const trimmed = value.trim();
      if (trimmed && !sources.includes(trimmed)) sources.push(trimmed);
    };

    if (Array.isArray(item.videoSources)) item.videoSources.forEach(add);
    add(item.videoSrc || item.video || item.videoPath || "");
    return sources;
  }

  function getPositiveNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? number : 0;
  }

  function getDemoMediaDimensions(item) {
    const videoWidth = getPositiveNumber(item.videoWidth || item.width);
    const videoHeight = getPositiveNumber(item.videoHeight || item.height);
    const posterWidth = getPositiveNumber(item.posterWidth) || videoWidth || 960;
    const posterHeight = getPositiveNumber(item.posterHeight) || videoHeight || 540;
    return { videoWidth, videoHeight, posterWidth, posterHeight };
  }

  function getDemoMediaFit(width, height) {
    const ratio = width && height ? width / height : 16 / 9;
    if (!Number.isFinite(ratio) || ratio <= 0) return "cover";
    return ratio < 1.2 || ratio > 2.05 ? "contain" : "cover";
  }

  function getDemoMediaStyle(dimensions) {
    const width = dimensions.videoWidth || dimensions.posterWidth;
    const height = dimensions.videoHeight || dimensions.posterHeight;
    const fit = getDemoMediaFit(width, height);
    if (!width || !height) return `style="--demo-video-fit: ${fit};"`;
    return `style="--demo-video-ratio: ${width} / ${height}; --demo-video-fit: ${fit};"`;
  }

  function applyDemoVideoRatio(player, width, height) {
    if (!player) return;
    const ratioWidth = getPositiveNumber(width);
    const ratioHeight = getPositiveNumber(height);
    if (!ratioWidth || !ratioHeight) return;
    player.style.setProperty("--demo-video-ratio", `${ratioWidth} / ${ratioHeight}`);
    player.style.setProperty("--demo-video-fit", getDemoMediaFit(ratioWidth, ratioHeight));
    player.dataset.videoWidth = String(ratioWidth);
    player.dataset.videoHeight = String(ratioHeight);
  }

  function renderVideoPlayer(item, extraClass = "") {
    const sources = getDemoVideoSources(item);
    const id = item.videoId || item.id || "demo-video-placeholder";
    const hasVideo = sources.length > 0;
    const sourceAttr = sources.join("|");
    const poster = item.poster || "";
    const posterMode = item.posterMode || (poster ? "image" : "none");
    const fallbackText = hasVideo
      ? "当前视频暂时无法播放，请稍后查看。"
      : "当前视频暂时无法播放，请稍后查看。";

    if (!hasVideo) {
      return `
        <div class="lychee-video-player is-placeholder" data-video-id="${escapeHTML(id)}">
          <div class="video-fallback is-visible">
            <strong>视频暂不可用</strong>
            <p>当前视频暂时无法播放，请稍后查看。</p>
          </div>
        </div>
      `;
    }

    const dimensions = getDemoMediaDimensions(item);
    const styleAttr = getDemoMediaStyle(dimensions);
    const firstFrameHTML = posterMode === "firstFrame" && sources[0]
      ? `<video class="video-first-frame" data-preview-src="${escapeHTML(sources[0])}" preload="${escapeHTML(item.previewPreload || "metadata")}" muted playsinline aria-hidden="true"></video>`
      : "";
    const posterHTML = posterMode !== "firstFrame" && poster
      ? `<img class="video-poster" src="${escapeHTML(poster)}" alt="${escapeHTML(item.title)} 封面" width="${dimensions.posterWidth}" height="${dimensions.posterHeight}" loading="lazy" decoding="async" fetchpriority="low" />`
      : firstFrameHTML;

    const extraClassName = extraClass.trim() ? ` ${escapeHTML(extraClass.trim())}` : "";

    return `
      <div class="lychee-video-player${extraClassName} has-video is-poster-only" data-video-id="${escapeHTML(id)}" data-sources="${escapeHTML(sourceAttr)}" data-poster="${escapeHTML(poster)}" data-poster-mode="${escapeHTML(posterMode)}" data-title="${escapeHTML(item.title)}" data-video-width="${dimensions.videoWidth || ""}" data-video-height="${dimensions.videoHeight || ""}" ${styleAttr}>
        ${posterHTML}
        <button class="video-play-button demo-video-load" type="button" data-demo-video-load aria-label="播放 ${escapeHTML(item.title)}">
          <span class="video-play-icon" aria-hidden="true"></span>
        </button>
        <div class="video-fallback" data-video-fallback>
          <strong>视频文件暂未接入或路径不可用</strong>
          <p>${escapeHTML(fallbackText)}</p>
        </div>
      </div>
    `;
  }

  function renderDemoCard(item, index = 0) {
    const tags = tagHTML((item.tags || item.ability || []).slice(0, 3));
    const displayIndex = item.displayIndex || String(index + 1).padStart(2, "0");
    return `
      <article class="demo-video-card type-${escapeHTML(item.type || "standard")}" data-case-id="${escapeHTML(item.id)}" data-demo-video-card>
        <div class="demo-card-head">
          <span class="demo-index">Demo ${escapeHTML(displayIndex)}</span>
          <span class="demo-scene">${escapeHTML(item.scene || item.category || "Demo")}</span>
        </div>
        <h3>${escapeHTML(item.title)}</h3>
        <p class="demo-subtitle">${escapeHTML(item.subtitle || item.description || item.summary || "")}</p>
        <div class="demo-video-slot">
          ${renderVideoPlayer(item)}
        </div>
        <div class="demo-ability demo-tags">${tags}</div>
        <p class="demo-highlight">${escapeHTML(item.highlight || item.description || "")}</p>
      </article>
    `;
  }

  function getVideoSources(video) {
    const raw = video?.dataset.sources || video?.dataset.src || "";
    return raw.split("|").map((item) => item.trim()).filter(Boolean);
  }

  function hydrateVideo(video) {
    if (!video) return false;
    if (video.dataset.hydrated === "true") return true;
    const sources = getVideoSources(video);
    const firstSource = sources[0];
    if (!firstSource) return false;
    video.dataset.sourceIndex = "0";
    video.src = firstSource;
    video.preload = "metadata";
    video.dataset.hydrated = "true";
    video.load();
    return true;
  }

  function playHydratedVideo(video) {
    if (!hydrateVideo(video)) return;
    const player = video.closest(".lychee-video-player");
    player?.classList.add("is-loading");
    const attempt = video.play();
    if (attempt && typeof attempt.catch === "function") {
      attempt.catch(() => {
        player?.classList.remove("is-loading", "is-playing");
        player?.classList.add("is-paused");
      });
    }
  }

  function showVideoFallback(video, player) {
    player?.classList.add("is-unavailable");
    player?.classList.remove("is-loading", "is-ready", "is-playing");
    const fallback = $("[data-video-fallback]", player || document);
    if (fallback) fallback.classList.add("is-visible");
  }

  function tryNextVideoSource(video, player) {
    const sources = getVideoSources(video);
    const currentIndex = Number(video.dataset.sourceIndex || "0");
    const nextIndex = currentIndex + 1;

    if (nextIndex < sources.length) {
      video.dataset.sourceIndex = String(nextIndex);
      video.src = sources[nextIndex];
      video.load();
      return;
    }

    showVideoFallback(video, player);
  }

  function hydrateFirstFramePreview(player) {
    const preview = $(".video-first-frame[data-preview-src]", player);
    if (!preview || preview.dataset.previewHydrated === "true") return;
    preview.src = preview.dataset.previewSrc;
    preview.dataset.previewHydrated = "true";
    try { preview.load(); } catch (_) {}
  }

  function createDemoVideo(player) {
    const existing = $("video[data-demo-video]", player);
    if (existing) return existing;

    const sources = getVideoSources(player);
    const firstSource = sources[0];
    if (!firstSource) return null;

    const video = document.createElement("video");
    video.dataset.demoVideo = "true";
    video.dataset.sources = sources.join("|");
    video.dataset.sourceIndex = "0";
    video.preload = "none";
    video.controls = true;
    video.playsInline = true;
    video.setAttribute("aria-label", `${player.dataset.title || "Demo"} 视频 Demo`);
    if (player.dataset.poster) video.poster = player.dataset.poster;

    const fallback = $("[data-video-fallback]", player);
    applyDemoVideoRatio(player, player.dataset.videoWidth, player.dataset.videoHeight);
    player.insertBefore(video, fallback || null);
    bindDemoVideoEvents(video, player);
    return video;
  }

  function releaseDemoVideo(video) {
    if (!video) return;
    const player = video.closest(".lychee-video-player");
    try { video.pause(); } catch (_) {}
    video.removeAttribute("src");
    video.load();
    video.remove();
    player?.classList.remove("is-loading", "is-ready", "is-playing", "is-paused");
    player?.classList.add("is-poster-only");
  }

  function pauseOtherDemoVideos(activeVideo) {
    $$("video[data-demo-video]").forEach((other) => {
      if (other === activeVideo) return;
      if (!other.paused) other.pause();
      releaseDemoVideo(other);
    });
  }

  function bindDemoVideoEvents(video, player) {
    if (!video || video.dataset.bound === "true") return;
    video.dataset.bound = "true";

    video.addEventListener("play", () => {
      hydrateVideo(video);
      pauseOtherDemoVideos(video);
      player?.classList.add("is-playing");
      player?.classList.remove("is-paused", "is-poster-only");
    });

    video.addEventListener("pause", () => {
      player?.classList.remove("is-playing");
      player?.classList.add("is-paused");
    });

    video.addEventListener("loadedmetadata", () => {
      applyDemoVideoRatio(player, video.videoWidth, video.videoHeight);
      player?.classList.add("is-ready");
      player?.classList.remove("is-loading", "is-poster-only");
    });

    video.addEventListener("error", () => {
      tryNextVideoSource(video, player);
    });
  }

  function initLazyDemoVideos(root) {
    const players = $$(".lychee-video-player.has-video", root);
    if (!players.length) return;

    players.forEach((player) => {
      const loadButton = $("[data-demo-video-load]", player);
      if (!loadButton || loadButton.dataset.bound === "true") return;
      loadButton.dataset.bound = "true";
      loadButton.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        hydrateFirstFramePreview(player);
        const video = createDemoVideo(player);
        if (video) playHydratedVideo(video);
      });
    });

    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            hydrateFirstFramePreview(entry.target);
            return;
          }
          const video = $("video[data-demo-video]", entry.target);
          if (video && !video.paused) video.pause();
        });
      }, { rootMargin: "420px 0px", threshold: 0.01 });

      players.forEach((player) => observer.observe(player));
    } else {
      players.forEach(hydrateFirstFramePreview);
    }

    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) return;
      $$("video[data-demo-video]").forEach((video) => { if (!video.paused) video.pause(); });
    });
  }

  function renderDemoList() {
    const grid = $("#demoGrid");
    if (!grid) return;
    const cases = CONFIG.demoCases || [];
    grid.innerHTML = cases.length
      ? cases.map((item, index) => renderDemoCard(item, index)).join("")
      : placeholderHTML("当前暂未配置 Demo", "在 js/config.js 的 demoCases 中新增 case 即可。新增视频时只需要改配置。"
        );
    initLazyDemoVideos(grid);
  }

  function attachImageErrorHandlers(root = document) {
    $$('img[data-lightbox-src]', root).forEach((img) => {
      if (img.dataset.errorBound === "true") return;
      img.dataset.errorBound = "true";
      img.addEventListener("error", () => {
        const frame = img.closest(".media-frame");
        if (frame) frame.innerHTML = placeholderHTML("内容暂不可用", "当前图片暂时无法显示。");
      }, { once: true });
    });
  }

  function renderAssets() {
    $$('[data-asset-id]').forEach(mountAsset);
    $$('[data-asset-list]').forEach(mountAssetList);
    attachImageErrorHandlers();
    const embodiedSection = $("#embodied");
    if (embodiedSection) initLazyDemoVideos(embodiedSection);
  }

  function enableRevealAnimation() {
    const items = $$(".reveal");
    if (!('IntersectionObserver' in window)) {
      items.forEach((item) => item.classList.add("visible"));
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });
    items.forEach((item) => observer.observe(item));
  }

  function animateMetricCounters() {
    const counters = $$('[data-counter]');
    if (!('IntersectionObserver' in window)) {
      counters.forEach((el) => { el.textContent = el.dataset.counter; });
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const el = entry.target;
        const target = el.dataset.counter || "0";
        const numeric = parseFloat(target.replace(/[^0-9.]/g, ""));
        const suffix = target.replace(/[0-9.]/g, "");
        const duration = 900;
        const start = performance.now();
        const decimals = numeric < 10 && !suffix.includes("ms") && !suffix.includes("%") ? 2 : 1;
        function step(now) {
          const progress = Math.min((now - start) / duration, 1);
          const eased = 1 - Math.pow(1 - progress, 3);
          const value = numeric * eased;
          el.textContent = `${value.toFixed(decimals).replace(/\.0$/, "")}${suffix}`;
          if (progress < 1) requestAnimationFrame(step);
          else el.textContent = target;
        }
        requestAnimationFrame(step);
        observer.unobserve(el);
      });
    }, { threshold: 0.65 });
    counters.forEach((counter) => observer.observe(counter));
  }

  function enableActiveNav() {
    const links = $$('[data-nav-mount] a');
    if (!links.length) return;

    const items = links
      .map((link) => {
        const href = link.getAttribute('href');
        const section = href && href.startsWith('#') ? $(href) : null;
        return section ? { link, href, section } : null;
      })
      .filter(Boolean);

    if (!items.length) return;

    const setActive = (href) => {
      items.forEach(({ link, href: itemHref }) => {
        link.classList.toggle('active', itemHref === href);
      });
    };

    if (!('IntersectionObserver' in window)) {
      setActive(items[0].href);
      return;
    }

    const visibility = new Map();
    const chooseActive = () => {
      let active = items[0];
      let bestScore = -Infinity;
      items.forEach((item) => {
        const score = visibility.get(item.href) || 0;
        if (score > bestScore) {
          bestScore = score;
          active = item;
        }
      });
      setActive(active.href);
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        const href = `#${entry.target.id}`;
        visibility.set(href, entry.isIntersecting ? entry.intersectionRatio : 0);
      });
      chooseActive();
    }, {
      root: null,
      rootMargin: '-28% 0px -58% 0px',
      threshold: [0, 0.1, 0.25, 0.5, 0.75, 1],
    });

    items.forEach((item) => {
      visibility.set(item.href, item === items[0] ? 1 : 0);
      observer.observe(item.section);
    });
    setActive(items[0].href);
  }

  function enableLightbox() {
    const lightbox = $("#lightbox");
    const img = $("#lightboxImage");
    const title = $("#lightboxTitle");
    if (!lightbox || !img || !title) return;

    document.addEventListener("click", (event) => {
      const target = event.target.closest('img[data-lightbox-src]');
      if (!target) return;
      img.src = target.dataset.lightboxSrc;
      img.alt = target.dataset.lightboxTitle || "素材预览";
      title.textContent = target.dataset.lightboxTitle || "素材预览";
      lightbox.classList.add("open");
      lightbox.setAttribute("aria-hidden", "false");
    });

    lightbox.addEventListener("click", (event) => {
      if (event.target.matches("[data-close-lightbox], .lightbox-backdrop")) {
        lightbox.classList.remove("open");
        lightbox.setAttribute("aria-hidden", "true");
        img.removeAttribute("src");
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && lightbox.classList.contains("open")) {
        lightbox.classList.remove("open");
        lightbox.setAttribute("aria-hidden", "true");
        img.removeAttribute("src");
      }
    });
  }

  function enableSmoothAnchors() {
    document.addEventListener("click", (event) => {
      const link = event.target.closest('a[href^="#"]');
      if (!link) return;
      const href = link.getAttribute("href");
      if (!href || href === "#") return;
      const target = $(href);
      if (!target && href !== "#top") return;
      event.preventDefault();
      if (href === "#top") {
        window.scrollTo({ top: 0, behavior: "smooth" });
        return;
      }
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  function enableHeroPromoVideo() {
    const player = $("[data-hero-promo-player]");
    if (!player) return;

    const hero = $("#hero");
    const playButton = $('[data-hero-promo-play]', player);
    const fullscreenButton = $('[data-hero-fullscreen]', player);

    const getHeroVideo = () => $("video[data-hero-video]", player);

    const createHeroVideo = () => {
      const existing = getHeroVideo();
      if (existing) return existing;
      const src = player.dataset.src || "";
      if (!src) return null;

      const video = document.createElement("video");
      video.dataset.heroVideo = "true";
      video.dataset.src = src;
      video.preload = "none";
      video.controls = true;
      video.playsInline = true;
      video.autoplay = false;
      video.loop = false;
      video.muted = false;
      video.setAttribute("aria-label", player.dataset.title || "Lychee-FD 交互演示");
      if (player.dataset.poster) video.poster = player.dataset.poster;

      const firstButton = playButton || fullscreenButton;
      player.insertBefore(video, firstButton || null);

      video.addEventListener("loadedmetadata", () => {
        player.classList.remove("is-placeholder", "is-unavailable", "is-loading", "is-poster-only");
        player.classList.add("is-ready");
        syncPlayingState();
      }, { once: true });
      video.addEventListener("playing", syncPlayingState);
      video.addEventListener("pause", syncPlayingState);
      video.addEventListener("ended", syncPlayingState);
      video.addEventListener("error", () => {
        player.classList.remove("is-ready", "is-playing", "is-loading");
        player.classList.add("is-placeholder", "is-unavailable");
        if (playButton) playButton.disabled = true;
        if (fullscreenButton) fullscreenButton.disabled = true;
      }, { once: true });
      return video;
    };

    const hydrateHeroVideo = () => {
      const video = createHeroVideo();
      if (!video) return null;
      if (video.dataset.hydrated !== "true") {
        video.src = video.dataset.src || player.dataset.src || "";
        video.preload = player.dataset.preload || "metadata";
        video.dataset.hydrated = "true";
        video.load();
        player.classList.remove("is-poster-only");
        player.classList.add("is-loading");
      }
      return video;
    };

    const syncPlayingState = () => {
      const video = getHeroVideo();
      const isPlaying = Boolean(video && !video.paused && !video.ended);
      player.classList.toggle("is-playing", isPlaying);
      player.classList.toggle("is-paused", Boolean(video && !isPlaying));
      if (isPlaying) player.classList.remove("is-loading");
    };

    const safePlay = () => {
      const video = hydrateHeroVideo();
      if (!video) return;
      const attempt = video.play();
      if (attempt && typeof attempt.catch === "function") {
        attempt.catch(() => {
          player.classList.remove("is-loading");
          syncPlayingState();
        });
      }
    };

    if (!player.dataset.src) {
      player.classList.add("is-placeholder", "is-unavailable");
      if (playButton) playButton.disabled = true;
      if (fullscreenButton) fullscreenButton.disabled = true;
      return;
    }

    if (playButton) {
      playButton.addEventListener("click", (event) => {
        event.stopPropagation();
        const video = getHeroVideo();
        if (!video || video.paused) safePlay();
        else video.pause();
      });
    }

    if (fullscreenButton) {
      const syncFullscreenState = () => {
        const isFull = document.fullscreenElement === player || document.webkitFullscreenElement === player;
        player.classList.toggle("is-fullscreen", Boolean(isFull));
        fullscreenButton.textContent = isFull ? "退出" : "全屏";
      };
      fullscreenButton.addEventListener("click", (event) => {
        event.stopPropagation();
        const video = hydrateHeroVideo();
        if (video && video.paused) safePlay();
        if (document.fullscreenElement || document.webkitFullscreenElement) {
          const exit = document.exitFullscreen || document.webkitExitFullscreen;
          if (exit) exit.call(document);
        } else {
          const request = player.requestFullscreen || player.webkitRequestFullscreen || player.msRequestFullscreen;
          if (request) request.call(player);
        }
      });
      document.addEventListener("fullscreenchange", syncFullscreenState);
      document.addEventListener("webkitfullscreenchange", syncFullscreenState);
    }

    if ("IntersectionObserver" in window && hero) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          const video = getHeroVideo();
          if (!entry.isIntersecting && video && !video.paused) video.pause();
        });
      }, { threshold: 0.08 });
      observer.observe(hero);
    }

    document.addEventListener("visibilitychange", () => {
      const video = getHeroVideo();
      if (document.hidden && video && !video.paused) video.pause();
    });

    syncPlayingState();
  }
  function init() {
    renderSiteText();
    renderNavigation();
    renderHeroPills();
    renderHeroPromoVideo();
    renderMetrics();
    renderProblemSection();
    renderResearchPath();
    renderAssets();
    renderDemoList();
    enableRevealAnimation();
    animateMetricCounters();
    enableActiveNav();
    enableSmoothAnchors();
    enableLightbox();
    enableHeroPromoVideo();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
