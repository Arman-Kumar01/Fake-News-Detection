document.addEventListener("DOMContentLoaded", () => {
  const newsInput = document.getElementById("newsInput");
  const charCount = document.getElementById("charCount");
  const clearBtn = document.getElementById("clearBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const samplesList = document.getElementById("samplesList");

  const emptyState = document.getElementById("emptyState");
  const resultsView = document.getElementById("resultsView");
  const statusIndicator = document.getElementById("statusIndicator");

  const verdictBanner = document.getElementById("verdictBanner");
  const verdictTitle = document.getElementById("verdictTitle");
  const confidenceValue = document.getElementById("confidenceValue");
  const agreementRatio = document.getElementById("agreementRatio");
  const votesBreakdown = document.getElementById("votesBreakdown");
  const meterProgress = document.getElementById("meterProgress");
  const modelsGrid = document.getElementById("modelsGrid");
  const cleanSnippet = document.getElementById("cleanSnippet");

  let isAnalyzing = false;

  // Textarea metrics update
  function updateCounters() {
    const text = newsInput.value.trim();
    const chars = newsInput.value.length;
    const words = text ? text.split(/\s+/).length : 0;
    charCount.textContent = `${chars} characters | ${words} words`;
  }

  newsInput.addEventListener("input", updateCounters);

  clearBtn.addEventListener("click", () => {
    newsInput.value = "";
    updateCounters();
    emptyState.classList.remove("hidden");
    resultsView.classList.add("hidden");
    statusIndicator.textContent = "Ready";
  });

  // Fetch quick testing samples
  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      const samples = await res.json();
      samplesList.innerHTML = "";

      samples.forEach((sample) => {
        const chip = document.createElement("button");
        chip.className = `chip ${sample.category === "Fake News" ? "fake-tag" : "real-tag"}`;
        chip.textContent = sample.title;
        chip.title = `Click to test: ${sample.title}`;

        chip.addEventListener("click", () => {
          newsInput.value = sample.text;
          updateCounters();
          runAnalysis();
        });

        samplesList.appendChild(chip);
      });
    } catch (err) {
      console.error("Failed to load samples:", err);
    }
  }

  // Fetch benchmark model stats
  async function loadStats() {
    try {
      const res = await fetch("/api/stats");
      const data = await res.json();
      if (data.models && data.models.length) {
        const grid = document.getElementById("benchmarksGrid");
        grid.innerHTML = "";
        data.models.forEach((m) => {
          const card = document.createElement("div");
          card.className = "benchmark-card";
          card.innerHTML = `
            <div class="bm-title">${m.name}</div>
            <div class="bm-val">${m.accuracy}%</div>
            <div class="bm-sub">Train time: ${m.train_time}s</div>
          `;
          grid.appendChild(card);
        });
      }
    } catch (err) {
      console.error("Failed to load stats:", err);
    }
  }

  // Submit and analyze
  async function runAnalysis() {
    const text = newsInput.value.trim();
    if (!text) {
      alert("Please enter or select a news article to analyze.");
      newsInput.focus();
      return;
    }

    if (isAnalyzing) return;
    setLoading(true);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || "Analysis failed");
      }

      const data = await response.json();
      renderResults(data);
    } catch (err) {
      alert("Error: " + err.message);
      statusIndicator.textContent = "Error";
    } finally {
      setLoading(false);
    }
  }

  function setLoading(loading) {
    isAnalyzing = loading;
    analyzeBtn.disabled = loading;
    const textMain = analyzeBtn.querySelector(".btn-text-main");
    const loadingState = analyzeBtn.querySelector(".btn-loading");

    if (loading) {
      textMain.classList.add("hidden");
      loadingState.classList.remove("hidden");
      statusIndicator.textContent = "Analyzing...";
    } else {
      textMain.classList.remove("hidden");
      loadingState.classList.add("hidden");
      statusIndicator.textContent = "Complete";
    }
  }

  function renderResults(data) {
    emptyState.classList.add("hidden");
    resultsView.classList.remove("hidden");

    const isReal = data.consensus === "Not A Fake News";
    verdictBanner.className = `verdict-banner ${isReal ? "real-news" : "fake-news"}`;
    verdictTitle.textContent = isReal ? "Authentic / Real News" : "Fake News Detected";
    confidenceValue.textContent = `${data.confidence_percent}%`;

    agreementRatio.textContent = `${data.confidence_ratio} Models Agree`;
    votesBreakdown.textContent = `Fake: ${data.votes.fake} | Real: ${data.votes.true}`;

    meterProgress.style.width = `${data.confidence_percent}%`;
    meterProgress.style.background = isReal
      ? "linear-gradient(90deg, #10b981, #34d399)"
      : "linear-gradient(90deg, #ef4444, #f87171)";

    cleanSnippet.textContent = data.cleaned_text_preview;

    // Render individual model cards
    modelsGrid.innerHTML = "";
    Object.keys(data.models).forEach((key) => {
      const m = data.models[key];
      const modelIsReal = m.label === "Not A Fake News";

      let probStr = "";
      if (m.probabilities) {
        const pFake = Math.round(m.probabilities.fake * 100);
        const pTrue = Math.round(m.probabilities.true * 100);
        probStr = `Fake: ${pFake}% | Real: ${pTrue}%`;
      }

      const card = document.createElement("div");
      card.className = "model-card";
      card.innerHTML = `
        <div class="model-header">
          <span class="model-name">${m.name}</span>
          <span class="model-status-tag ${modelIsReal ? "real" : "fake"}">
            ${modelIsReal ? "Real" : "Fake"}
          </span>
        </div>
        <div class="model-prob">${probStr || "Deterministic Tree Split"}</div>
      `;
      modelsGrid.appendChild(card);
    });
  }

  analyzeBtn.addEventListener("click", runAnalysis);

  // Initialize
  updateCounters();
  loadSamples();
  loadStats();
});
