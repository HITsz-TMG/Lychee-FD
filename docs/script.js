const demos = [
  {
    tag: "Interruption",
    title: "User barge-in handling",
    description: "Placeholder sample: the model should stop speaking and respond to a new user intent.",
    audio: "./assets/audio/demo-interruption.wav",
    transcript: [
      ["User", "Can you explain the difference between half-duplex and full-duplex?"],
      ["Assistant", "Sure. Half-duplex means only one side speaks at a time..."],
      ["User interrupts", "Wait, give me an example from voice assistants."],
      ["Assistant", "Good point. A full-duplex assistant keeps listening while speaking and can react immediately."]
    ]
  },
  {
    tag: "AI Backchannel",
    title: "Natural short response",
    description: "Placeholder sample: the assistant injects short backchannels without taking over the conversation.",
    audio: "./assets/audio/demo-ai-backchannel.wav",
    transcript: [
      ["User", "I started with ASR, then tried to build a full-duplex pipeline..."],
      ["Assistant", "Mm-hmm."],
      ["User", "...but the hard part is deciding when the model should speak."],
      ["Assistant", "Exactly — that is a turn-taking and intent problem."]
    ]
  },
  {
    tag: "User Pause",
    title: "Pause vs. interruption",
    description: "Placeholder sample: the model distinguishes hesitation from real interruption intent.",
    audio: "./assets/audio/demo-user-pause.wav",
    transcript: [
      ["User", "I think the architecture should maybe... hmm..."],
      ["Assistant", "[keeps listening]"],
      ["User", "...split the semantic and acoustic heads in deep layers."],
      ["Assistant", "Yes, that matches the optimization-dynamics finding."]
    ]
  }
];

function renderDemos() {
  const grid = document.querySelector("#demoGrid");
  if (!grid) return;
  grid.innerHTML = demos.map((demo, index) => `
    <article class="demo-card reveal ${index === 1 ? 'delay-1' : index === 2 ? 'delay-2' : ''}">
      <span class="demo-tag">${demo.tag}</span>
      <h3>${demo.title}</h3>
      <p>${demo.description}</p>
      <audio controls preload="metadata" src="${demo.audio}"></audio>
      <div class="transcript">
        ${demo.transcript.map(([speaker, text]) => `
          <div class="turn"><b>${speaker}</b><span>${text}</span></div>
        `).join("")}
      </div>
    </article>
  `).join("");
}

function setupScrollProgress() {
  const bar = document.querySelector("#scrollProgress");
  const update = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const pct = max > 0 ? (window.scrollY / max) * 100 : 0;
    bar.style.width = `${pct}%`;
  };
  update();
  window.addEventListener("scroll", update, { passive: true });
}

function setupReveal() {
  const items = document.querySelectorAll(".reveal");
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

function setupActiveNav() {
  const sections = [...document.querySelectorAll("main section[id]")];
  const navLinks = [...document.querySelectorAll(".nav-links a")];
  const byId = new Map(navLinks.map((link) => [link.getAttribute("href")?.slice(1), link]));
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const link = byId.get(entry.target.id);
      if (!link) return;
      if (entry.isIntersecting) {
        navLinks.forEach((item) => item.classList.remove("active"));
        link.classList.add("active");
      }
    });
  }, { rootMargin: "-40% 0px -55% 0px" });
  sections.forEach((section) => observer.observe(section));
}

renderDemos();
setupScrollProgress();
setupReveal();
setupActiveNav();
