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

  function stars(score = 0) {
    const full = Math.max(0, Math.min(5, Number(score) || 0));
    return "★".repeat(full) + "☆".repeat(5 - full);
  }

  function tagHTML(tags = []) {
    return tags.map((tag) => `<span>${escapeHTML(tag)}</span>`).join("");
  }

  function placeholderHTML(title = "待替换真实素材", desc = "把图片、视频或音频放入 assets 对应目录后，在 js/config.js 中填写路径。") {
    return `
      <div class="media-placeholder" role="img" aria-label="${escapeHTML(title)}">
        <div class="placeholder-orbit"><span></span><span></span><span></span></div>
        <strong>${escapeHTML(title)}</strong>
        <p>${escapeHTML(desc)}</p>
      </div>
    `;
  }

  function createAssetCard(assetId, options = {}) {
    const asset = typeof assetId === "string" ? getAsset(assetId) : assetId;
    const variant = options.variant || "standard";
    const card = document.createElement("article");
    card.className = `asset-card asset-${variant}`;

    if (!asset) {
      card.innerHTML = placeholderHTML("未找到素材配置", "请检查 data-asset-id 或 js/config.js 中的 imageAssets。")
        + `<div class="asset-caption"><h3>素材配置缺失</h3><p>当前挂载点没有匹配到图片配置。</p></div>`;
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

    if (!enabled || !hasSrc) {
      card.innerHTML = placeholderHTML(
        enabled ? "待替换真实素材" : "素材暂未启用",
        enabled ? `建议路径：${asset.src || "请在 config.js 填写 src"}` : "如需显示，请在 js/config.js 中将 enabled 改为 true。"
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

    card.innerHTML = `
      <div class="media-frame">
        <img src="${escapeHTML(asset.src)}" alt="${escapeHTML(asset.title)}" loading="lazy" decoding="async" fetchpriority="low" data-lightbox-src="${escapeHTML(asset.src)}" data-lightbox-title="${escapeHTML(asset.title)}" />
      </div>
      ${caption}
    `;

    const img = $("img", card);
    img.addEventListener("error", () => {
      const frame = $(".media-frame", card);
      frame.innerHTML = placeholderHTML("待替换真实素材", `当前路径未找到：${asset.src}`);
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
    const title = promo.title || "宣传视频：从轮流说话到自然共话";
    const description = promo.description || "展示 Lychee-FD 在自然打断、附和反馈、低延迟响应和数字人驱动中的核心能力。";
    const eyebrow = promo.eyebrow || "宣传视频 / Lychee-FD 交互演示";
    const tags = tagHTML(promo.tags || ["自然打断", "附和反馈", "数字人驱动"]);
    const videoSrc = promo.video || promo.src || "";
    const poster = promo.poster || "";
    const preload = promo.preload || "metadata";
    const videoAttrs = videoSrc
      ? ` src="${escapeHTML(videoSrc)}"${poster ? ` poster="${escapeHTML(poster)}"` : ""} autoplay muted loop playsinline preload="${escapeHTML(preload)}"`
      : "";
    const screenState = videoSrc ? "has-video-source" : "is-placeholder no-video-source";

    mount.innerHTML = `
      <article class="hero-promo-card hero-promo-card-static" data-hero-promo-card aria-label="Lychee-FD 宣传视频">
        <div class="hero-promo-topline">
          <span class="hero-record-dot" aria-hidden="true"></span>
          <span>${escapeHTML(eyebrow)}</span>
          <em>Hero Video</em>
        </div>
        <div class="hero-promo-screen ${screenState}" data-hero-promo-player aria-label="Lychee-FD 宣传视频自动播放区域">
          ${videoSrc ? `<video${videoAttrs} aria-label="${escapeHTML(title)}"></video>` : ""}
          <div class="hero-promo-placeholder" aria-hidden="true">
            <span class="hero-promo-gridmark"></span>
            <strong>宣传视频占位</strong>
            <p>将真实视频放入 assets/videos 后，会在这里静音自动播放。</p>
          </div>
          <button class="hero-promo-main-play" type="button" data-hero-promo-play aria-label="播放宣传视频"><span></span></button>
          <button class="hero-promo-fullscreen" type="button" data-hero-fullscreen aria-label="全屏播放宣传视频">全屏</button>
        </div>
        <div class="hero-promo-copy">
          <h3>${escapeHTML(title)}</h3>
          <p>${escapeHTML(description)}</p>
          <div class="hero-promo-tags">${tags}</div>
        </div>
      </article>
    `;
  }

  function metricCard(metric, index = 0, interactive = false) {
    const actionAttrs = interactive
      ? ` role="button" tabindex="0" data-open-demo-panel="true" aria-label="打开 Demo 视频弹窗：${escapeHTML(metric.label || `指标 ${index + 1}`)}"`
      : "";
    const actionClass = interactive ? " metric-card-action" : "";
    return `
      <article class="metric-card${actionClass}"${actionAttrs}>
        <span class="metric-note">${escapeHTML(metric.note || "")}</span>
        <strong data-counter="${escapeHTML(metric.value)}">0</strong>
        <p>${escapeHTML(metric.label)}</p>
      </article>
    `;
  }

  function renderMetrics() {
    const metrics = CONFIG.metrics || [];
    $$("[data-metrics]").forEach((mount) => {
      const interactive = mount.classList.contains("hero-metrics");
      mount.innerHTML = metrics.map((metric, index) => metricCard(metric, index, interactive)).join("");
    });
  }

  function renderHighlights() {
    const mount = $("#highlightGrid");
    if (!mount) return;
    mount.innerHTML = (CONFIG.highlights || []).map((item) => `
      <article class="highlight-card">
        <div class="highlight-icon">${escapeHTML(item.icon)}</div>
        <h3>${escapeHTML(item.title)}</h3>
        <p>${escapeHTML(item.text)}</p>
      </article>
    `).join("");
  }

  function demoTypeLabel(type = "standard") {
    const labels = {
      interrupt: "Interrupt Demo",
      backchannel: "Backchannel Demo",
      wait: "Wait Demo",
      scene: "Scene Demo",
      companion: "Companion Demo",
      handsfree: "Hands-free Demo",
      embodied: "Embodied Demo",
    };
    return labels[type] || "Video Demo";
  }

  function renderVideoPlayer(item) {
    const videoSrc = item.videoSrc || item.video || item.videoPath || "";
    const poster = item.poster || "";
    const id = item.videoId || item.id || "demo-video-placeholder";
    const scene = item.scene || item.category || "Demo";
    const hasVideo = Boolean(videoSrc);

    return `
      <div class="lychee-video-player ${hasVideo ? "has-video" : "is-placeholder"} type-${escapeHTML(item.type || "standard")}" data-video-id="${escapeHTML(id)}" aria-label="${escapeHTML(item.title)} 视频 Demo">
        <div class="video-frame-surface">
          ${hasVideo ? `<video data-demo-video data-src="${escapeHTML(videoSrc)}" preload="none" playsinline${poster ? ` poster="${escapeHTML(poster)}"` : ""}></video>` : ""}
          ${poster && !hasVideo ? `<img class="video-poster" src="${escapeHTML(poster)}" alt="${escapeHTML(item.title)} 封面" loading="lazy" decoding="async" />` : ""}
          <div class="video-glow" aria-hidden="true"></div>
          <div class="video-topline">
            <span class="video-status-dot"></span>
            <span>${escapeHTML(item.status || (hasVideo ? "点击播放" : "视频占位"))}</span>
            <em>${escapeHTML(scene)}</em>
          </div>
          <div class="video-center">
            <button class="video-play-button" type="button" data-demo-play aria-label="播放或暂停 ${escapeHTML(item.title)}"></button>
            <strong>${escapeHTML(id)}</strong>
            <p>${hasVideo ? "点击后加载并播放当前 Demo" : "这里将接入真实宣传视频 / Web Demo 录屏"}</p>
          </div>
        </div>
      </div>
    `;
  }

  function renderDemoCard(item) {
    const tags = tagHTML(item.tags || item.ability || []);
    return `
      <article class="demo-card demo-video-card type-${escapeHTML(item.type || "standard")}" data-case-id="${escapeHTML(item.id)}" data-demo-video-card>
        <div class="demo-card-head">
          <span class="demo-scene">${escapeHTML(item.category || "Demo")}</span>
          <span class="demo-type">${escapeHTML(demoTypeLabel(item.type))}</span>
        </div>
        <h3>${escapeHTML(item.title)}</h3>
        <p class="demo-subtitle">${escapeHTML(item.subtitle || item.description || item.summary || "")}</p>
        <div class="demo-ability demo-tags">${tags}</div>
        <div class="demo-video-slot">
          ${renderVideoPlayer(item)}
        </div>
        <p class="demo-highlight">${escapeHTML(item.highlight || item.description || "")}</p>
      </article>
    `;
  }

  function renderDemoTabs() {
    const tabMount = $("#demoTabs");
    const grid = $("#demoGrid");
    if (!tabMount || !grid) return;
    const cases = CONFIG.demoCases || [];
    const tabs = (CONFIG.demoTabs && CONFIG.demoTabs.length)
      ? CONFIG.demoTabs
      : Array.from(new Set(cases.map((item) => item.category).filter(Boolean)));
    let currentPlaying = null;
    let currentTab = tabs[0] || "";

    tabMount.innerHTML = tabs.map((tab, index) => `
      <button type="button" class="${index === 0 ? "active" : ""}" data-tab="${escapeHTML(tab)}" aria-selected="${index === 0 ? "true" : "false"}">${escapeHTML(tab)}</button>
    `).join("");

    const stopCurrentVideo = () => {
      if (!currentPlaying) return;
      const { card, video, player } = currentPlaying;
      if (video) {
        video.pause();
        try { video.currentTime = 0; } catch (error) {}
      }
      if (card) card.classList.remove("is-playing");
      if (player) player.classList.remove("is-playing", "is-loading", "is-paused-placeholder");
      currentPlaying = null;
    };

    const prepareDemoVideo = (video) => {
      const src = video.dataset.src;
      if (!src) return false;
      video.preload = "auto";
      if (video.dataset.loadedSrc !== src || video.getAttribute("src") !== src) {
        video.setAttribute("src", src);
        video.dataset.loadedSrc = src;
        video.load();
      }
      return true;
    };

    const render = (tab) => {
      stopCurrentVideo();
      currentTab = tab;
      const items = cases.filter((item) => item.category === tab);
      grid.innerHTML = items.length
        ? `
          <div class="demo-rail-shell" data-demo-rail-shell>
            <button class="demo-rail-arrow rail-prev" type="button" data-rail-prev aria-label="向左浏览 Demo">‹</button>
            <div class="demo-rail" data-demo-rail tabindex="0" aria-label="${escapeHTML(tab)} 视频 Demo 横向轨道">
              ${items.map(renderDemoCard).join("")}
            </div>
            <button class="demo-rail-arrow rail-next" type="button" data-rail-next aria-label="向右浏览 Demo">›</button>
          </div>
        `
        : placeholderHTML("当前分类暂未配置 Demo", "在 js/config.js 的 demoCases 中新增对应 category 的 case。");
      initRailDrag(grid.querySelector("[data-demo-rail]"));
    };

    const startCard = (card) => {
      if (!card) return;
      const player = $(".lychee-video-player", card);
      const video = $("[data-demo-video]", card);

      if (currentPlaying && currentPlaying.card === card) {
        stopCurrentVideo();
        return;
      }

      stopCurrentVideo();
      card.classList.add("is-playing");
      if (player) player.classList.add("is-playing");

      if (!video) {
        if (player) player.classList.add("is-paused-placeholder");
        currentPlaying = { card, video: null, player };
        return;
      }

      if (!prepareDemoVideo(video)) {
        if (player) player.classList.add("is-paused-placeholder");
        currentPlaying = { card, video: null, player };
        return;
      }

      currentPlaying = { card, video, player };
      if (player) player.classList.add("is-loading");

      video.addEventListener("playing", () => {
        if (player) player.classList.remove("is-loading");
      }, { once: true });

      video.addEventListener("error", () => {
        if (player) player.classList.remove("is-loading");
      }, { once: true });

      const attempt = video.play();
      if (attempt && typeof attempt.catch === "function") {
        attempt.catch(() => {
          if (currentPlaying && currentPlaying.card === card) {
            if (player) player.classList.remove("is-playing", "is-loading");
            card.classList.remove("is-playing");
            currentPlaying = null;
          }
        });
      }
    };

    const initRailDrag = (rail) => {
      if (!rail || rail.dataset.dragReady === "true") return;
      rail.dataset.dragReady = "true";
      let isDown = false;
      let startX = 0;
      let startLeft = 0;
      let moved = false;

      rail.addEventListener("pointerdown", (event) => {
        if (event.button !== undefined && event.button !== 0) return;
        if (event.target.closest("[data-demo-play], .lychee-video-player")) return;
        isDown = true;
        moved = false;
        startX = event.clientX;
        startLeft = rail.scrollLeft;
        rail.classList.add("is-dragging");
        rail.setPointerCapture?.(event.pointerId);
      });

      rail.addEventListener("pointermove", (event) => {
        if (!isDown) return;
        const dx = event.clientX - startX;
        if (Math.abs(dx) > 3) moved = true;
        rail.scrollLeft = startLeft - dx;
      });

      const stopDrag = (event) => {
        if (!isDown) return;
        isDown = false;
        rail.classList.remove("is-dragging");
        rail.dataset.justDragged = moved ? "true" : "false";
        setTimeout(() => { rail.dataset.justDragged = "false"; }, 80);
        try { rail.releasePointerCapture?.(event.pointerId); } catch (error) {}
      };

      rail.addEventListener("pointerup", stopDrag);
      rail.addEventListener("pointercancel", stopDrag);
      rail.addEventListener("mouseleave", stopDrag);
    };

    tabMount.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-tab]");
      if (!button) return;
      $$('button[data-tab]', tabMount).forEach((item) => {
        item.classList.toggle("active", item === button);
        item.setAttribute("aria-selected", item === button ? "true" : "false");
      });
      render(button.dataset.tab);
    });

    grid.addEventListener("click", (event) => {
      const arrow = event.target.closest("[data-rail-prev], [data-rail-next]");
      if (arrow) {
        const rail = grid.querySelector("[data-demo-rail]");
        if (rail) {
          const direction = arrow.matches("[data-rail-prev]") ? -1 : 1;
          rail.scrollBy({ left: direction * Math.round(rail.clientWidth * 0.82), behavior: "smooth" });
        }
        return;
      }

      const rail = event.target.closest("[data-demo-rail]");
      if (rail && rail.dataset.justDragged === "true") return;
      const trigger = event.target.closest("[data-demo-play], .lychee-video-player");
      if (!trigger) return;
      const card = event.target.closest("[data-demo-video-card]");
      startCard(card);
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") stopCurrentVideo();
    });

    if (tabs.length) render(currentTab);
  }

  function renderPackageSpec() {
    const spec = CONFIG.packageSpec || {};
    const folderRule = $("#folderRule");
    const sampleNames = $("#sampleNames");
    const packageTree = $("#packageTree");
    const metaGrid = $("#metaGrid");
    if (folderRule) folderRule.textContent = spec.folderRule || "";
    if (sampleNames) sampleNames.innerHTML = (spec.examples || []).map((name) => `<li>${escapeHTML(name)}</li>`).join("");
    if (packageTree) packageTree.innerHTML = `<div class="tree-root">case_package</div>` + (spec.files || []).map((file) => `<div class="tree-item">${escapeHTML(file)}</div>`).join("");
    if (metaGrid) metaGrid.innerHTML = (spec.metaFields || []).map((field) => `<span>${escapeHTML(field)}</span>`).join("");
  }

  function attachImageErrorHandlers(root = document) {
    $$('img[data-lightbox-src]', root).forEach((img) => {
      if (img.dataset.errorBound === "true") return;
      img.dataset.errorBound = "true";
      img.addEventListener("error", () => {
        const frame = img.closest(".media-frame");
        if (frame) frame.innerHTML = placeholderHTML("待替换真实素材", `当前路径未找到：${img.getAttribute("src")}`);
      }, { once: true });
    });
  }

  function renderAssets() {
    $$('[data-asset-id]').forEach(mountAsset);
    $$('[data-asset-list]').forEach(mountAssetList);
    attachImageErrorHandlers();
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
    const links = $$("[data-nav-mount] a");
    if (!links.length) return;

    const items = links
      .map((link) => {
        const href = link.getAttribute("href");
        const section = href && href.startsWith("#") ? $(href) : null;
        return section ? { link, href, section } : null;
      })
      .filter(Boolean);

    if (!items.length) return;

    let ticking = false;
    let activeHref = "";

    const setActive = (href) => {
      if (!href || href === activeHref) return;
      activeHref = href;
      items.forEach(({ link, href: itemHref }) => {
        link.classList.toggle("active", itemHref === href);
      });
    };

    const update = () => {
      ticking = false;
      const marker = window.innerHeight * 0.38;
      let current = items[0];

      for (const item of items) {
        const rect = item.section.getBoundingClientRect();
        if (rect.top <= marker && rect.bottom > marker) {
          current = item;
          break;
        }
        if (rect.top <= marker) current = item;
      }

      setActive(current.href);
    };

    const requestUpdate = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(update);
    };

    window.addEventListener("scroll", requestUpdate, { passive: true });
    window.addEventListener("resize", requestUpdate);
    update();
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

  function enableHeroScrollTransition() {
    const hero = $("#hero");
    const heroVideo = $("#heroPromoMount");
    const heroIntro = hero ? $("[data-hero-intro]", hero) : null;
    const heroContent = hero ? $("[data-hero-content]", hero) : null;
    const reduceMotion = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
    let ticking = false;

    const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, value));
    const smoothstep = (value) => {
      const t = clamp(value);
      return t * t * (3 - 2 * t);
    };

    const canAnimateHero = () => Boolean(
      hero &&
      heroVideo &&
      heroIntro &&
      heroContent &&
      window.innerWidth > 980 &&
      !(reduceMotion && reduceMotion.matches)
    );

    const resetHeroTransition = () => {
      if (heroVideo) heroVideo.style.transform = "";
      if (heroIntro) {
        heroIntro.style.opacity = "";
        heroIntro.style.transform = "";
      }
      if (heroContent) {
        heroContent.style.opacity = "";
        heroContent.style.transform = "";
        heroContent.style.pointerEvents = "";
      }
    };

    const updateHeroTransition = () => {
      if (!canAnimateHero()) {
        resetHeroTransition();
        return;
      }

      const rect = hero.getBoundingClientRect();
      const travel = Math.max(1, hero.offsetHeight - window.innerHeight);
      const rawProgress = clamp(-rect.top / travel);

      if (rawProgress <= 0.001) {
        resetHeroTransition();
        return;
      }

      const focusStart = 0.15;
      const shrinkStart = 0.45;
      const shrinkEnd = 0.90;

      const focusProgress = smoothstep((rawProgress - focusStart) / (shrinkStart - focusStart));
      const shrinkProgress = smoothstep((rawProgress - shrinkStart) / (shrinkEnd - shrinkStart));

      /*
        Stage 1: 0 ~ 0.15, keep the title and video in normal hero layout.
        Stage 2: 0.15 ~ 0.45, gently move the video toward the visual center without shrinking.
        Stage 3: 0.45 ~ 0.90, slowly shrink and move the video to the left-top area.
        Only transform/opacity are touched during scroll.
      */
      const centerLiftY = -Math.round(focusProgress * Math.min(42, window.innerHeight * 0.052));
      const maxShiftX = Math.min(135, window.innerWidth * 0.092);
      const maxShiftY = Math.min(40, window.innerHeight * 0.046);
      const scale = 1 - shrinkProgress * 0.14;
      const shiftX = -Math.round(shrinkProgress * maxShiftX);
      const shiftY = centerLiftY - Math.round(shrinkProgress * maxShiftY);

      const introFade = smoothstep((rawProgress - 0.34) / 0.42);
      const introOpacity = 1 - introFade * 0.42;
      const introY = -Math.round((focusProgress * 12) + (shrinkProgress * 20));

      const contentOpacity = smoothstep((rawProgress - 0.66) / 0.28);
      const contentShift = Math.round((1 - contentOpacity) * 30);

      heroVideo.style.transform = `translate3d(${shiftX}px, ${shiftY}px, 0) scale(${scale.toFixed(3)})`;
      heroIntro.style.opacity = introOpacity.toFixed(3);
      heroIntro.style.transform = `translate3d(0, ${introY}px, 0)`;
      heroContent.style.opacity = contentOpacity.toFixed(3);
      heroContent.style.transform = `translate3d(${contentShift}px, -50%, 0)`;
      heroContent.style.pointerEvents = contentOpacity > 0.45 ? "auto" : "none";
    };

    const requestUpdate = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        ticking = false;
        updateHeroTransition();
      });
    };

    window.addEventListener("scroll", requestUpdate, { passive: true });
    window.addEventListener("resize", requestUpdate);
    if (reduceMotion && typeof reduceMotion.addEventListener === "function") {
      reduceMotion.addEventListener("change", requestUpdate);
    }
    updateHeroTransition();
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

  function enableDemoModal() {
    const modal = $("#demoModal");
    const pageShell = $("#pageShell");
    const closeButton = $(".demo-modal-close", modal);
    if (!modal) return;
    let lastFocus = null;

    const openModal = () => {
      lastFocus = document.activeElement;
      modal.classList.add("open");
      modal.setAttribute("aria-hidden", "false");
      document.body.classList.add("no-scroll");
      if (pageShell) pageShell.classList.add("is-blurred");
      if (closeButton) closeButton.focus({ preventScroll: true });
    };

    const closeModal = () => {
      modal.classList.remove("open");
      modal.setAttribute("aria-hidden", "true");
      document.body.classList.remove("no-scroll");
      if (pageShell) pageShell.classList.remove("is-blurred");
      if (lastFocus && typeof lastFocus.focus === "function") lastFocus.focus({ preventScroll: true });
      lastFocus = null;
    };

    document.addEventListener("click", (event) => {
      if (event.target.closest("[data-open-demo-panel]")) {
        openModal();
        return;
      }
      if (event.target.matches("[data-close-demo], .demo-modal-backdrop")) {
        closeModal();
      }
    });

    document.addEventListener("keydown", (event) => {
      const trigger = event.target.closest?.("[data-open-demo-panel]");
      if (trigger && (event.key === "Enter" || event.key === " ")) {
        event.preventDefault();
        openModal();
        return;
      }
      if (event.key === "Escape" && modal.classList.contains("open")) closeModal();
    });
  }

  function enableHeroPromoVideo() {
    const player = $("[data-hero-promo-player]");
    if (!player) return;

    const video = $("video", player);
    const hero = $("#hero");
    const playButton = $('[data-hero-promo-play]', player);
    const fullscreenButton = $('[data-hero-fullscreen]', player);
    let heroInView = true;

    if (!video) {
      player.classList.add("is-placeholder", "is-unavailable");
      if (playButton) playButton.disabled = true;
      if (fullscreenButton) fullscreenButton.disabled = true;
      return;
    }

    const syncPlayingState = () => {
      player.classList.toggle("is-playing", !video.paused && !video.ended);
      player.classList.toggle("is-paused", video.paused || video.ended);
    };

    const safePlay = () => {
      if (!video || !heroInView || player.classList.contains("is-unavailable")) return;
      const attempt = video.play();
      if (attempt && typeof attempt.catch === "function") {
        attempt.catch(() => {
          player.classList.add("is-autoplay-blocked");
          syncPlayingState();
        });
      }
    };

    const requestFullscreen = () => {
      const target = player;
      const request = target.requestFullscreen || target.webkitRequestFullscreen || target.msRequestFullscreen;
      if (!request) return;
      player.classList.add("is-fullscreen-requested");
      request.call(target);
      safePlay();
    };

    const syncFullscreenState = () => {
      const isFull = document.fullscreenElement === player || document.webkitFullscreenElement === player;
      player.classList.toggle("is-fullscreen", Boolean(isFull));
      if (fullscreenButton) fullscreenButton.textContent = isFull ? "退出" : "全屏";
    };

    video.muted = true;
    video.autoplay = true;
    video.loop = true;
    video.playsInline = true;

    video.addEventListener("loadedmetadata", () => {
      player.classList.remove("is-placeholder", "is-unavailable");
      player.classList.add("is-ready");
      safePlay();
    }, { once: true });

    video.addEventListener("playing", () => {
      player.classList.remove("is-autoplay-blocked");
      syncPlayingState();
    });
    video.addEventListener("pause", syncPlayingState);
    video.addEventListener("error", () => {
      player.classList.remove("is-ready", "is-playing", "is-autoplay-blocked");
      player.classList.add("is-placeholder", "is-unavailable");
      if (playButton) playButton.disabled = true;
      if (fullscreenButton) fullscreenButton.disabled = true;
    }, { once: true });

    if (playButton) {
      playButton.addEventListener("click", (event) => {
        event.stopPropagation();
        player.classList.remove("is-autoplay-blocked");
        safePlay();
      });
    }

    if (fullscreenButton) {
      fullscreenButton.addEventListener("click", (event) => {
        event.stopPropagation();
        if (document.fullscreenElement || document.webkitFullscreenElement) {
          const exit = document.exitFullscreen || document.webkitExitFullscreen;
          if (exit) exit.call(document);
        } else {
          requestFullscreen();
        }
      });
      document.addEventListener("fullscreenchange", syncFullscreenState);
      document.addEventListener("webkitfullscreenchange", syncFullscreenState);
    }

    if ("IntersectionObserver" in window && hero) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          heroInView = entry.isIntersecting;
          if (!heroInView) {
            video.pause();
            return;
          }
          safePlay();
        });
      }, { threshold: 0.12 });
      observer.observe(hero);
    } else {
      safePlay();
    }
  }

  function attachPlaceholderEvents() {
    document.addEventListener("click", (event) => {
      const button = event.target.closest('[data-placeholder="true"]');
      if (!button) return;
      const old = button.textContent;
      button.textContent = "素材待接入";
      button.disabled = true;
      setTimeout(() => {
        button.textContent = old;
        button.disabled = false;
      }, 900);
    });
  }

  function init() {
    renderSiteText();
    renderNavigation();
    renderHeroPills();
    renderHeroPromoVideo();
    renderMetrics();
    renderHighlights();
    renderAssets();
    renderDemoTabs();
    renderPackageSpec();
    enableRevealAnimation();
    animateMetricCounters();
    enableActiveNav();
    enableSmoothAnchors();
    enableLightbox();
    enableDemoModal();
    enableHeroPromoVideo();
    attachPlaceholderEvents();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
