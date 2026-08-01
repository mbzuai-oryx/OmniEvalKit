const RESULTS = [
  {
    table: "Table 1",
    title: "AV-Omni Understanding",
    subtitle: "Speech-centric audio-visual benchmarks",
    labels: ["Qwen2.5-3B", "Qwen2.5-7B", "MiniCPM-9B", "VILA-9B", "Qwen3-30B"],
    native: [43.2, 45.8, 54.6, 46.7, 54.9],
    tfo: [46.2, 48.0, 52.4, 47.1, 54.1],
  },
  {
    table: "Table 2",
    title: "Audio-Only Understanding",
    subtitle: "9 audio-only benchmarks",
    labels: ["Qwen2.5-3B", "Qwen2.5-7B", "MiniCPM-9B", "VILA-9B", "Qwen3-30B"],
    native: [59.5, 60.5, 60.6, 50.3, 71.7],
    tfo: [61.0, 61.9, 64.6, 63.8, 73.1],
  },
  {
    table: "Table 3",
    title: "Multilingual Speech",
    subtitle: "CoVoST2 · 21 languages",
    labels: ["Qwen2.5-3B", "Qwen2.5-7B", "MiniCPM-9B", "VILA-9B", "Qwen3-30B"],
    native: [53.0, 49.7, 45.6, 44.6, 61.6],
    tfo: [60.9, 63.2, 64.0, 55.7, 68.4],
  },
  {
    table: "Table 4",
    title: "Image + Video Understanding",
    subtitle: "Mean of image and video averages",
    labels: ["Qwen2.5-3B", "Qwen2.5-7B", "MiniCPM-9B", "VILA-9B", "Qwen3-30B"],
    native: [66.7, 70.55, 74.05, 65.95, 75.65],
    tfo: [69.45, 72.95, 74.25, 67.25, 78.3],
  },
  {
    table: "Table 5",
    title: "Coding + Math Reasoning",
    subtitle: "7 benchmarks · MiniCPM not reported",
    labels: ["Qwen2.5-3B", "Qwen2.5-7B", "VILA-9B", "Qwen3-30B"],
    native: [50.8, 58.9, 46.4, 73.1],
    tfo: [55.1, 61.8, 43.6, 75.6],
  },
  {
    table: "Table 6",
    title: "Medical Question Answering",
    subtitle: "Domain-specific knowledge + reasoning",
    labels: ["Qwen2.5-3B", "Qwen2.5-7B", "MiniCPM-9B", "VILA-9B", "Qwen3-30B"],
    native: [41.6, 44.5, 49.7, 46.1, 54.9],
    tfo: [43.3, 45.7, 51.3, 46.6, 56.4],
  },
];

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

document.querySelectorAll('a[aria-disabled="true"]').forEach((link) => {
  link.addEventListener("click", (event) => event.preventDefault());
});

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function scoreLabel(value) {
  const roundedValue = Math.round((value + Number.EPSILON) * 10) / 10;
  return roundedValue.toFixed(1);
}

function createResults() {
  const grid = document.querySelector("#results-grid");
  if (!grid) return;

  RESULTS.forEach((result, panelIndex) => {
    const panel = document.createElement("article");
    panel.className = "result-panel";
    panel.dataset.panel = String(panelIndex);

    const combined = [...result.native, ...result.tfo];
    const maxValue = Math.max(...combined) * 1.16;
    const delta = average(result.tfo) - average(result.native);
    const deltaSign = delta >= 0 ? "+" : "";

    panel.innerHTML = `
      <div class="result-panel-head">
        <div class="result-table-number">${result.table}</div>
        <h3>${result.title}</h3>
        <div class="result-subtitle">${result.subtitle}</div>
      </div>
      <div class="result-delta ${delta < 0 ? "negative" : ""}" data-delta>
        ${deltaSign}${delta.toFixed(1)} avg
      </div>
      <div class="chart" aria-label="${result.title} comparison chart"></div>
      <div class="chart-labels"></div>
    `;

    const chart = panel.querySelector(".chart");
    const labelRow = panel.querySelector(".chart-labels");

    result.labels.forEach((label, index) => {
      const group = document.createElement("div");
      group.className = "bar-group";
      const nativeHeight = ((result.native[index] / maxValue) * 100).toFixed(2);
      const tfoHeight = ((result.tfo[index] / maxValue) * 100).toFixed(2);

      group.innerHTML = `
        <div
          class="result-bar native"
          data-height="${nativeHeight}"
          tabindex="0"
          role="img"
          aria-label="${label}, Native Omni: ${scoreLabel(result.native[index])}"
        >
          <span class="bar-value">${scoreLabel(result.native[index])}</span>
        </div>
        <div
          class="result-bar tfo"
          data-height="${tfoHeight}"
          tabindex="0"
          role="img"
          aria-label="${label}, TFO: ${scoreLabel(result.tfo[index])}"
        >
          <span class="bar-value">${scoreLabel(result.tfo[index])}</span>
        </div>
      `;
      chart.appendChild(group);

      const labelElement = document.createElement("span");
      labelElement.className = "chart-label";
      labelElement.textContent = label;
      labelRow.appendChild(labelElement);
    });

    grid.appendChild(panel);
  });
}

function resetResults() {
  document.querySelectorAll(".result-bar").forEach((bar) => {
    bar.style.height = "0";
    bar.querySelector(".bar-value")?.classList.remove("visible");
  });
  document.querySelectorAll(".result-delta").forEach((delta) => {
    delta.classList.remove("visible");
  });
}

function playResults() {
  resetResults();

  document.querySelectorAll(".result-panel").forEach((panel, panelIndex) => {
    const delay = reduceMotion ? 0 : 120 + panelIndex * 100;
    window.setTimeout(() => {
      panel.querySelectorAll(".result-bar").forEach((bar, barIndex) => {
        window.setTimeout(() => {
          bar.style.height = `${bar.dataset.height}%`;
          bar.querySelector(".bar-value")?.classList.add("visible");
        }, reduceMotion ? 0 : barIndex * 35);
      });
      window.setTimeout(() => {
        panel.querySelector(".result-delta")?.classList.add("visible");
      }, reduceMotion ? 0 : 550);
    }, delay);
  });
}

function setupResultsObserver() {
  const resultsSection = document.querySelector("#results");
  if (!resultsSection) return;

  let hasPlayed = false;
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && !hasPlayed) {
          hasPlayed = true;
          playResults();
        }
      });
    },
    { threshold: 0.16 }
  );

  observer.observe(resultsSection);
  document.querySelector("#replay-results")?.addEventListener("click", playResults);
}

function setupReveal() {
  const revealItems = document.querySelectorAll(".reveal");
  if (reduceMotion) {
    revealItems.forEach((item) => item.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.08 }
  );

  revealItems.forEach((item) => observer.observe(item));
}

function setupNavigation() {
  const header = document.querySelector(".site-header");
  const toggle = document.querySelector(".nav-toggle");
  const links = document.querySelector(".nav-links");
  const navAnchors = document.querySelectorAll(".nav-links a");

  const updateHeader = () => {
    header?.classList.toggle("scrolled", window.scrollY > 16);
  };

  const closeMenu = () => {
    toggle?.setAttribute("aria-expanded", "false");
    links?.classList.remove("open");
    document.body.classList.remove("nav-open");
  };

  toggle?.addEventListener("click", () => {
    const isOpen = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", String(!isOpen));
    links?.classList.toggle("open", !isOpen);
    document.body.classList.toggle("nav-open", !isOpen);
  });

  navAnchors.forEach((anchor) => anchor.addEventListener("click", closeMenu));
  window.addEventListener("scroll", updateHeader, { passive: true });
  updateHeader();

  const sections = [...navAnchors]
    .map((anchor) => document.querySelector(anchor.getAttribute("href")))
    .filter(Boolean);

  const sectionObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          navAnchors.forEach((anchor) => {
            anchor.classList.toggle(
              "active",
              anchor.getAttribute("href") === `#${entry.target.id}`
            );
          });
        }
      });
    },
    { rootMargin: "-30% 0px -60% 0px", threshold: 0 }
  );

  sections.forEach((section) => sectionObserver.observe(section));
}

createResults();
setupNavigation();
setupReveal();
setupResultsObserver();
