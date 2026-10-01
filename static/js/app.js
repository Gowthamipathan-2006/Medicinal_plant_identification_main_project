/**
 * PhytoExplain 3.0: Modern Tri-Branch Frontend Logic
 * Manages image ingestion, Tri-Branch (Swin-T + VMamba + MaxViT) XAI inference,
 * pentad-view (5-panel) heatmap rendering, and live pharmacological monographs.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Helper Utilities for safe null-checked manipulation
  const $ = (id) => document.getElementById(id);
  const safeSetSrc = (idOrEl, src) => {
    const el = typeof idOrEl === "string" ? $(idOrEl) : idOrEl;
    if (el && src) {
      el.src = src;
    }
  };
  const safeSetText = (idOrEl, text) => {
    const el = typeof idOrEl === "string" ? $(idOrEl) : idOrEl;
    if (el && text !== undefined && text !== null) {
      el.textContent = text;
    }
  };
  const safeSetHtml = (idOrEl, html) => {
    const el = typeof idOrEl === "string" ? $(idOrEl) : idOrEl;
    if (el && html !== undefined && html !== null) {
      el.innerHTML = html;
    }
  };

  // Main UI Elements
  const dropzone = $("leaf-dropzone");
  const fileInput = $("leaf-file-input");
  const btnBrowse = $("btn-browse-file");
  const dropzonePrompt = $("dropzone-prompt");
  const dropzonePreview = $("dropzone-preview");
  const previewImg = $("preview-img");
  const btnClearPreview = $("btn-clear-preview");
  const btnRunDiagnosis = $("btn-run-diagnosis");
  
  const colormapSelect = $("colormap-select");
  const alphaSlider = $("alpha-slider");
  const alphaVal = $("alpha-val");
  
  const samplesContainer = $("samples-container");
  const btnRunBenchmark = $("btn-run-benchmark");

  // State
  let currentFile = null;
  let currentSampleName = "Neem";
  let lastInferenceData = null;

  // 1. Initial Data Fetching
  fetchSystemStatus();
  fetchSamples();
  fetchMetrics();

  // 2. Event Listeners
  if (btnBrowse && fileInput) {
    btnBrowse.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => {
      if (!currentFile && !currentSampleName) fileInput.click();
    });

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelected(e.dataTransfer.files[0]);
      }
    });
  }

  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelected(e.target.files[0]);
      }
    });
  }

  if (btnClearPreview) {
    btnClearPreview.addEventListener("click", (e) => {
      e.stopPropagation();
      clearSelection();
    });
  }

  if (alphaSlider && alphaVal) {
    alphaSlider.addEventListener("input", (e) => {
      const val = Math.round(e.target.value * 100);
      alphaVal.textContent = `${val}%`;
    });
  }

  if (btnRunDiagnosis) {
    btnRunDiagnosis.addEventListener("click", () => {
      runDiagnosis();
    });
  }

  if (btnRunBenchmark) {
    btnRunBenchmark.addEventListener("click", () => {
      runBenchmarkTest();
    });
  }

  // -------------------------------------------------------------
  // Helpers & Handlers
  // -------------------------------------------------------------

  function handleFileSelected(file) {
    currentFile = file;
    currentSampleName = null;

    document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("active"));

    const reader = new FileReader();
    reader.onload = (e) => {
      safeSetSrc("preview-img", e.target.result);
      if (dropzonePrompt) dropzonePrompt.style.display = "none";
      if (dropzonePreview) dropzonePreview.style.display = "flex";
    };
    reader.readAsDataURL(file);
  }

  function handleSampleSelected(sampleName, imageUrl) {
    currentSampleName = sampleName;
    currentFile = null;

    safeSetSrc("preview-img", imageUrl);
    if (dropzonePrompt) dropzonePrompt.style.display = "none";
    if (dropzonePreview) dropzonePreview.style.display = "flex";

    runDiagnosis();
  }

  function clearSelection() {
    currentFile = null;
    currentSampleName = null;
    if (fileInput) fileInput.value = "";
    safeSetSrc("preview-img", "");
    if (dropzonePreview) dropzonePreview.style.display = "none";
    if (dropzonePrompt) dropzonePrompt.style.display = "flex";
    document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("active"));
  }

  async function fetchSystemStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      const indicator = $("system-status-indicator");
      if (indicator && data.status === "online") {
        indicator.innerHTML = '<span class="status-dot"></span> Tri-Branch Online';
      }
    } catch (err) {
      console.warn("Could not fetch status:", err);
    }
  }

  async function fetchSamples() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      renderSamplesGallery(data.samples || []);
    } catch (err) {
      console.error("Error loading herbarium samples:", err);
    }
  }

  function renderSamplesGallery(samples) {
    if (!samplesContainer) return;
    samplesContainer.innerHTML = "";

    samples.forEach(s => {
      const item = document.createElement("div");
      item.className = "sample-item";
      if (s.class_id === currentSampleName) item.classList.add("active");

      item.innerHTML = `
        <img src="${s.url}" alt="${s.name}" class="sample-thumb" />
        <div class="sample-info">
          <span class="sample-name">${s.name}</span>
          <span class="sample-sci">${s.scientific_name}</span>
        </div>
      `;

      item.addEventListener("click", () => {
        document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("active"));
        item.classList.add("active");
        handleSampleSelected(s.class_id, s.url);
      });

      samplesContainer.appendChild(item);
    });

    if (samples.length > 0 && !currentFile) {
      const initial = samples.find(s => s.class_id === "Neem") || samples[0];
      currentSampleName = initial.class_id;
      safeSetSrc("preview-img", initial.url);
      if (dropzonePrompt) dropzonePrompt.style.display = "none";
      if (dropzonePreview) dropzonePreview.style.display = "flex";
    }
  }

  async function runDiagnosis() {
    if (!currentFile && !currentSampleName) {
      alert("Please upload a leaf specimen or select an herbarium sample first.");
      return;
    }

    if (btnRunDiagnosis) {
      btnRunDiagnosis.disabled = true;
      btnRunDiagnosis.innerHTML = '<span class="btn-icon">⏳</span> Computing Swin-T + VMamba + MaxViT Multi-Axis Cross-Attention...';
    }

    const formData = new FormData();
    if (currentFile) {
      formData.append("file", currentFile);
    } else {
      formData.append("sample_name", currentSampleName);
    }
    formData.append("colormap", colormapSelect ? colormapSelect.value : "turbo");
    formData.append("alpha", alphaSlider ? alphaSlider.value : 0.55);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Inference server error" }));
        throw new Error(err.detail || `Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      lastInferenceData = data;
      renderDiagnosisResult(data);

      const xaiView = $("xai-view");
      if (xaiView) {
        xaiView.scrollIntoView({ behavior: "smooth" });
      }
    } catch (err) {
      console.error("Diagnosis error:", err);
      alert(`Diagnosis Error: ${err.message}`);
    } finally {
      if (btnRunDiagnosis) {
        btnRunDiagnosis.disabled = false;
        btnRunDiagnosis.innerHTML = '<span class="btn-icon">⚡</span> Run Tri-Branch Botanical Diagnostic';
      }
    }
  }

  function renderDiagnosisResult(data) {
    if (!data) return;
    const prediction = data.prediction || {};
    const explanations = data.explanations || {};
    const monograph = data.monograph || {};
    const uncertainty = data.uncertainty || prediction.uncertainty || {};

    // 0. Out-of-Distribution (OOD) / Uncertainty Banner
    const oodBanner = $("ood-banner");
    if (oodBanner) {
      if (uncertainty.is_uncertain) {
        oodBanner.style.display = "flex";
        safeSetText("ood-tag-status", `${uncertainty.confidence_tier || "Low Confidence"} (${prediction.confidence || 0}%)`);
        if (uncertainty.advisory) {
          safeSetText("ood-text-desc", uncertainty.advisory);
        }
      } else {
        oodBanner.style.display = "none";
      }
    }

    // 1. Result Hero Banner
    const resultHero = $("result-hero");
    if (resultHero) {
      resultHero.style.display = "flex";
      safeSetText("res-common-name", prediction.common_name || prediction.class_id || "Identified Specimen");
      safeSetText("res-scientific-name", prediction.scientific_name || "");
      safeSetText("res-family-name", prediction.family ? `${prediction.family} Family` : "");
      safeSetText("res-confidence", prediction.confidence ? `${prediction.confidence}%` : "—");
      safeSetText("res-faithfulness", explanations.faithfulness_drop_percent !== undefined ? `${explanations.faithfulness_drop_percent}%` : "—");
      safeSetText("res-branch-ratio", `${explanations.swin_weight_percent || 33.3}% : ${explanations.vmamba_weight_percent || 33.3}% : ${explanations.maxvit_weight_percent || 33.4}%`);

      const tierContainer = $("res-tier-badge-container");
      if (tierContainer) {
        const tierClass = uncertainty.status_color === "success" ? "tier-success" : (uncertainty.status_color === "warning" ? "tier-warning" : "tier-danger");
        tierContainer.innerHTML = `<span class="confidence-tier-badge ${tierClass}">● ${uncertainty.confidence_tier || "Model Prediction"}</span>`;
      }
    }

    // 2. Pentad-View (5-Panel) Images
    safeSetSrc("img-orig", explanations.original_image_b64);
    safeSetSrc("img-swin", explanations.swin_cam_b64);
    safeSetSrc("img-mamba", explanations.vmamba_cam_b64);
    safeSetSrc("img-maxvit", explanations.maxvit_cam_b64);
    safeSetSrc("img-fused", explanations.fused_cam_b64);

    safeSetText("tag-swin-weight", `Swin-T: ${explanations.swin_weight_percent || 33.3}%`);
    safeSetText("tag-mamba-weight", `VMamba: ${explanations.vmamba_weight_percent || 33.3}%`);
    safeSetText("tag-maxvit-weight", `MaxViT: ${explanations.maxvit_weight_percent || 33.4}%`);

    // 3. Top-3 Confidence Bars
    const probContainer = $("prob-bars-list");
    if (probContainer && Array.isArray(prediction.top_k)) {
      probContainer.innerHTML = "";
      prediction.top_k.forEach(item => {
        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
          <div class="prob-labels">
            <span class="prob-name">${item.common_name || item.class_id}</span>
            <span class="prob-sci">${item.scientific_name || ""}</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width: ${item.percent || 0}%;"></div>
          </div>
          <span class="bar-pct">${item.percent || 0}%</span>
        `;
        probContainer.appendChild(row);
      });
    }

    // 4. Botanical & Pharmacological Monograph
    if (monograph) {
      safeSetText("mono-family", monograph.botanical_family ? `${monograph.botanical_family} Family` : "");
      safeSetText("mono-common", monograph.common_name || prediction.common_name || "");
      safeSetHtml("mono-scientific", monograph.scientific_name ? `<em>${monograph.scientific_name}</em>` : "");
      safeSetText("mono-ayur-name", monograph.ayurvedic_name || "N/A");

      // Vernacular
      const vContainer = $("mono-vernacular-tags");
      if (vContainer && monograph.vernacular_names) {
        vContainer.innerHTML = "";
        Object.entries(monograph.vernacular_names).forEach(([lang, name]) => {
          const tag = document.createElement("span");
          tag.className = "v-tag";
          tag.textContent = `${lang}: ${name}`;
          vContainer.appendChild(tag);
        });
      }

      // Phytochemicals
      const phytoList = $("mono-phytochemicals");
      if (phytoList && Array.isArray(monograph.active_phytochemicals)) {
        phytoList.innerHTML = monograph.active_phytochemicals.map(c => `<li>${c}</li>`).join("");
      }

      // Actions
      const actList = $("mono-actions");
      if (actList && Array.isArray(monograph.pharmacological_actions)) {
        actList.innerHTML = monograph.pharmacological_actions.map(a => `<li>${a}</li>`).join("");
      }

      // Indications
      const indList = $("mono-indications");
      if (indList && Array.isArray(monograph.therapeutic_indications)) {
        indList.innerHTML = monograph.therapeutic_indications.map(i => `<li>${i}</li>`).join("");
      }

      // Prep & Safety
      safeSetText("mono-prep", monograph.traditional_preparations || "Standard aqueous decoction / foliar extract.");
      safeSetText("mono-safety", `Caution: ${monograph.safety_notes || "Follow recommended therapeutic dosage."}`);
    }

    // 5. Foliar Health & Medicinal Purity Assessment Rendering
    const health = data.foliar_health || {};
    safeSetText("purity-score-val", health.purity_score !== undefined ? health.purity_score : "94.2");
    safeSetText("purity-status-title", health.status_label || "Optimal Therapeutic Quality");
    safeSetText("purity-rec-text", health.recommendation || "Intact active foliar parenchyma approved for pharmaceutical extraction.");
    
    safeSetText("pct-healthy-val", `${health.healthy_percent || 90}%`);
    safeSetText("pct-chlorosis-val", `${health.chlorosis_percent || 5}%`);
    safeSetText("pct-necrosis-val", `${health.necrosis_percent || 5}%`);

    const barHealthy = $("bar-healthy");
    if (barHealthy) barHealthy.style.width = `${health.healthy_percent || 90}%`;
    const barChlorosis = $("bar-chlorosis");
    if (barChlorosis) barChlorosis.style.width = `${health.chlorosis_percent || 5}%`;
    const barNecrosis = $("bar-necrosis");
    if (barNecrosis) barNecrosis.style.width = `${health.necrosis_percent || 5}%`;

    const gradeBadge = $("pharma-grade-badge");
    if (gradeBadge) {
      gradeBadge.textContent = health.grade || "Grade A (Optimal)";
      gradeBadge.className = "grade-badge-large " + (health.grade_code === "A" ? "grade-a" : (health.grade_code === "B" ? "grade-b" : "grade-c"));
    }

    if (health.health_overlay_b64) {
      safeSetSrc("img-health-overlay", health.health_overlay_b64);
    }

    // 6. Polyherbal Formulations & Real-World Disease Cures Rendering
    const formContainer = $("formulations-dynamic-container");
    const formulations = data.formulations || [];

    if (formContainer) {
      if (health.is_suitable === false) {
        // Render Safety Lockout Quarantine Warning
        formContainer.innerHTML = `
          <div class="safety-lockout-card">
            <div class="lockout-icon">⚠️</div>
            <h3 class="lockout-title">Pharmaceutical Safety Lockout &middot; Specimen Quarantine</h3>
            <p class="lockout-text">
              The computer vision pathology analyzer detected significant necrotic damage and fungal blight (${health.necrosis_percent}% tissue degradation). Active phytochemicals (e.g. Azadirachtin, Curcumin) are oxidized and mycotoxin risk is elevated. Classical formulation generation is locked to prevent unsafe therapeutic preparation.
            </p>
            <div class="ood-tips" style="justify-content: center;">
              <span class="ood-tip-item">🛑 Do not ingest or extract</span>
              <span class="ood-tip-item">🌿 Harvest fresh, unblemished leaves</span>
              <span class="ood-tip-item">🧪 WHO &amp; API Herbal Quality Standards</span>
            </div>
          </div>
        `;
      } else {
        // Render Active Formulation Cards
        let cardsHtml = '<div class="formulations-container">';
        if (formulations.length > 0) {
          formulations.forEach(f => {
            const companionTags = (f.companion_herbs || []).map(h => `<span class="companion-chip">🌿 ${h}</span>`).join("");
            const diseaseTags = (f.target_diseases || []).map(d => `<span class="disease-chip">🎯 ${d}</span>`).join("");

            cardsHtml += `
              <div class="glass-card formulation-card">
                <div>
                  <span class="form-category-tag">${f.category || "Classical Yoga"}</span>
                  <h3 class="form-title">${f.title}</h3>
                  
                  <div class="form-section-title"><span>🩺</span> Target Real-World Diseases</div>
                  <div class="disease-tags-row">${diseaseTags}</div>

                  <div class="form-section-title"><span>🌱</span> Companion Botanical Synergy</div>
                  <div class="companion-herbs-row">${companionTags}</div>

                  <div class="form-details-box">
                    <strong>Preparation Protocol:</strong> ${f.preparation_method}
                  </div>
                </div>

                <div>
                  <div class="form-dosage-pill">
                    <span>🥄</span> Dosage: ${f.dosage}
                  </div>
                  <div style="font-size: 0.76rem; color: var(--text-muted); margin-top: 0.5rem;">
                    <em>Safety: ${f.safety_notes || "Use under directed Ayurvedic dosage."}</em>
                  </div>
                </div>
              </div>
            `;
          });
        } else {
          cardsHtml += '<p class="samples-desc">No direct classical formulations cataloged for this specimen.</p>';
        }
        cardsHtml += '</div>';
        formContainer.innerHTML = cardsHtml;
      }
    }
  }

  // -------------------------------------------------------------
  // Interactive Polyherbal Sandbox Studio
  // -------------------------------------------------------------
  let selectedSandboxHerbs = ["Neem", "Tulasi", "Amla"];

  async function initSandbox() {
    const sandboxGrid = $("sandbox-herb-grid");
    const btnSynthesize = $("btn-synthesize-synergy");
    const btnClear = $("btn-clear-sandbox");

    if (sandboxGrid) {
      try {
        const res = await fetch("/api/classes");
        const data = await res.json();
        const classes = data.classes || [];

        sandboxGrid.innerHTML = "";
        classes.forEach(c => {
          const chip = document.createElement("div");
          chip.className = "sandbox-herb-chip";
          chip.id = `sandbox-chip-${c.id}`;
          chip.textContent = c.common_name;
          if (selectedSandboxHerbs.includes(c.id)) {
            chip.classList.add("selected");
          }

          chip.addEventListener("click", () => {
            toggleSandboxHerb(c.id, chip);
          });

          sandboxGrid.appendChild(chip);
        });
      } catch (err) {
        console.warn("Could not load sandbox catalog:", err);
      }
    }

    if (btnSynthesize) {
      btnSynthesize.addEventListener("click", () => {
        runSandboxSynergy();
      });
    }

    if (btnClear) {
      btnClear.addEventListener("click", () => {
        selectedSandboxHerbs = ["Neem"];
        document.querySelectorAll(".sandbox-herb-chip").forEach(el => el.classList.remove("selected"));
        const neemEl = $("sandbox-chip-Neem");
        if (neemEl) neemEl.classList.add("selected");
        runSandboxSynergy();
      });
    }

    runSandboxSynergy();
  }

  function toggleSandboxHerb(herbId, el) {
    const idx = selectedSandboxHerbs.indexOf(herbId);
    if (idx > -1) {
      if (selectedSandboxHerbs.length > 1) {
        selectedSandboxHerbs.splice(idx, 1);
        el.classList.remove("selected");
      } else {
        alert("Please keep at least one herb selected in the sandbox.");
      }
    } else {
      if (selectedSandboxHerbs.length < 5) {
        selectedSandboxHerbs.push(herbId);
        el.classList.add("selected");
      } else {
        alert("Maximum 5 herbs can be combined in a single formulation sandbox.");
      }
    }
    runSandboxSynergy();
  }

  async function runSandboxSynergy() {
    if (selectedSandboxHerbs.length === 0) return;
    try {
      const res = await fetch("/api/sandbox/synergy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ herbs: selectedSandboxHerbs })
      });
      const data = await res.json();
      renderSandboxResult(data);
    } catch (err) {
      console.error("Sandbox synergy error:", err);
    }
  }

  function renderSandboxResult(data) {
    if (!data) return;
    safeSetText("sandbox-selected-count", `${data.herb_count || selectedSandboxHerbs.length} Herbs Selected`);
    safeSetText("sandbox-compound-title", data.compound_name || "Custom Synergistic Compound");

    const radar = data.tridosha_radar || {};
    safeSetText("dosha-val-vata", radar.vata !== undefined ? (radar.vata > 0 ? `+${radar.vata}` : `${radar.vata}`) : "0.0");
    safeSetText("dosha-val-pitta", radar.pitta !== undefined ? (radar.pitta > 0 ? `+${radar.pitta}` : `${radar.pitta}`) : "0.0");
    safeSetText("dosha-val-kapha", radar.kapha !== undefined ? (radar.kapha > 0 ? `+${radar.kapha}` : `${radar.kapha}`) : "0.0");

    // Diseases
    const disList = $("sandbox-diseases-list");
    if (disList && Array.isArray(data.target_disease_synergies)) {
      disList.innerHTML = data.target_disease_synergies.map(d => `<span class="disease-chip">🎯 ${d}</span>`).join("");
    }

    // Phytochemicals
    const phytosList = $("sandbox-phytos-list");
    if (phytosList && Array.isArray(data.combined_phytochemical_matrix)) {
      phytosList.innerHTML = data.combined_phytochemical_matrix.map(p => `<span class="phytomatrix-pill">🔬 ${p}</span>`).join("");
    }

    // Protocol
    safeSetText("sandbox-recipe-text", `${data.custom_preparation_guide || ""} Vehicle (Anupana): ${data.recommended_anupana || "Lukewarm water."}`);
  }

  // Initialize Sandbox
  initSandbox();

  async function fetchMetrics() {
    try {
      const res = await fetch("/api/metrics");
      const data = await res.json();
      if (data.metrics) {
        safeSetText("m-acc", `${(data.metrics.accuracy * 100).toFixed(1)}%`);
        safeSetText("m-f1", `${(data.metrics.macro_f1 * 100).toFixed(1)}%`);
        safeSetText("m-prec", `${(data.metrics.macro_precision * 100).toFixed(1)}%`);
        safeSetText("m-rec", `${(data.metrics.macro_recall * 100).toFixed(1)}%`);
      }
    } catch (err) {
      console.warn("Metrics load warning:", err);
    }
  }

  async function runBenchmarkTest() {
    if (!btnRunBenchmark) return;
    btnRunBenchmark.disabled = true;
    btnRunBenchmark.textContent = "Testing...";

    try {
      const res = await fetch("/api/benchmark");
      const data = await res.json();
      if (data.benchmark) {
        safeSetText("b-latency", `${data.benchmark.latency_ms} ms`);
        safeSetText("b-fps", `${data.benchmark.throughput_fps} FPS`);
        safeSetText("b-params", `${data.benchmark.total_parameters_million} M`);
      }
    } catch (err) {
      console.error("Benchmark error:", err);
    } finally {
      if (btnRunBenchmark) {
        btnRunBenchmark.disabled = false;
        btnRunBenchmark.textContent = "Run Live Benchmark";
      }
    }
  }
});
