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
          <video src="${escapeHTML(asset.src)}" controls preload="metadata"></video>
        </div>
        ${caption}
      `;
      return card;
    }

    card.innerHTML = `
      <div class="media-frame">
        <img src="${escapeHTML(asset.src)}" alt="${escapeHTML(asset.title)}" loading="lazy" data-lightbox-src="${escapeHTML(asset.src)}" data-lightbox-title="${escapeHTML(asset.title)}" />
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
    const nav = $("#navMount");
    if (!nav) return;
    nav.innerHTML = (CONFIG.navigation || [])
      .map((item) => `<a href="#${escapeHTML(item.target)}">${escapeHTML(item.label)}</a>`)
      .join("");
  }

  function renderHeroPills() {
    const mount = $("#heroPills");
    if (!mount) return;
    mount.innerHTML = (CONFIG.heroPills || []).map((pill) => `<span>${escapeHTML(pill)}</span>`).join("");
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
    const html = (CONFIG.metrics || []).map(metricCard).join("");
    $$("[data-metrics]").forEach((mount) => { mount.innerHTML = html; });
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

  function materialButton(label, kind, src) {
    if (src) {
      return `<a class="material-button is-ready" href="${escapeHTML(src)}" target="_blank" rel="noreferrer">${escapeHTML(label)}</a>`;
    }
    return `<button class="material-button" type="button" data-placeholder="true">${escapeHTML(label)}占位</button>`;
  }

  function renderDemoCard(item) {
    const asset = item.imageAsset ? createAssetCard(item.imageAsset, { variant: "demo" }).outerHTML : placeholderHTML("Demo 封面占位", "在当前 case 的 imageAsset 或 image 字段中绑定图片。") ;
    const script = (item.script || []).map((line) => `
      <p class="dialogue-line ${line.role.includes("打断") ? "is-interrupt" : ""}">
        <b>${escapeHTML(line.role)}：</b>${escapeHTML(line.text)}
      </p>
    `).join("");
    return `
      <article class="demo-card" data-case-id="${escapeHTML(item.id)}">
        <div class="demo-media">${asset}</div>
        <div class="demo-body">
          <span class="demo-scene">场景：${escapeHTML(item.scene)}</span>
          <h3>${escapeHTML(item.title)}</h3>
          <p>${escapeHTML(item.summary)}</p>
          <div class="demo-ability">${tagHTML(item.ability || [])}</div>
          <div class="demo-script">${script}</div>
          <div class="demo-materials">
            ${materialButton("用户音频", "audio", item.userAudio)}
            ${materialButton("系统音频", "audio", item.systemAudio)}
            ${materialButton("演示视频", "video", item.video)}
            ${materialButton("输出文本", "text", item.transcript)}
          </div>
          <div class="demo-meta">
            <span><b>用户价值：</b>${escapeHTML(item.value)}</span>
            <span><b>关键转折点：</b>${escapeHTML(item.keyMoment)}</span>
            <span><b>稳定性评分：</b>${stars(item.stability)}</span>
            <span><b>推荐用途：</b>${escapeHTML((item.usage || []).join(" / "))}</span>
            <span><b>对应技术点：</b>${escapeHTML((item.tech || []).join(" / "))}</span>
          </div>
        </div>
      </article>
    `;
  }

  function renderDemoTabs() {
    const tabMount = $("#demoTabs");
    const grid = $("#demoGrid");
    if (!tabMount || !grid) return;
    const tabs = CONFIG.demoTabs || [];
    tabMount.innerHTML = tabs.map((tab, index) => `
      <button type="button" class="${index === 0 ? "active" : ""}" data-tab="${escapeHTML(tab)}">${escapeHTML(tab)}</button>
    `).join("");

    const render = (tab) => {
      const items = (CONFIG.demoCases || []).filter((item) => item.category === tab);
      grid.innerHTML = items.length ? items.map(renderDemoCard).join("") : placeholderHTML("当前分类暂未配置 Demo", "在 js/config.js 的 demoCases 中新增对应 category 的 case。") ;
      attachImageErrorHandlers(grid);
    };

    tabMount.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-tab]");
      if (!button) return;
      $$('button[data-tab]', tabMount).forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      render(button.dataset.tab);
    });

    render(tabs[0]);
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
    const links = $$('#navMount a');
    const sections = links.map((link) => $(link.getAttribute("href"))).filter(Boolean);
    if (!sections.length || !('IntersectionObserver' in window)) return;
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        links.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${entry.target.id}`));
      });
    }, { rootMargin: "-42% 0px -54% 0px", threshold: 0.01 });
    sections.forEach((section) => observer.observe(section));
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
    renderMetrics();
    renderHighlights();
    renderAssets();
    renderDemoTabs();
    renderPackageSpec();
    enableRevealAnimation();
    animateMetricCounters();
    enableActiveNav();
    enableLightbox();
    attachPlaceholderEvents();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
