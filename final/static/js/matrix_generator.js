// matrix_generator.js — full, consolidated file
// -----------------------------------------------------------------------------
// 2025‑05‑14
//    • Handles folder picker, matrix generation, feature table.
//    • Show / Hide toggles for Features, Spy‑Plot, Heat‑map, Graph (side‑by‑side).
//    • Reuses existing `showVisualBtn`, `showHeatmapBtn`, `showGraphBtn`.
// -----------------------------------------------------------------------------

(function () {
    "use strict";
  
    /* ────────────────── Modal helper ────────────────── */
    function showModal(msg) {
      const modalEl = document.getElementById("messageModal");
      if (!modalEl) { alert(msg); return; }
      document.getElementById("messageModalBody").textContent = msg;
      (bootstrap.Modal.getInstance(modalEl) || new bootstrap.Modal(modalEl)).show();
    }
  
    /* ────────────────── DOM refs used often ────────────────── */
    const folderInput   = document.getElementById("matrix_folder");
    const dropdownGroup = document.getElementById("matrixFileDropdownGroup");
    const fileDropdown  = document.getElementById("matrix_file_select");
    const matrixForm    = document.getElementById("matrixForm");
    const loadingEl     = document.getElementById("loading");
    const resultsDiv    = document.getElementById("results");
    const featuresTbl   = document.getElementById("featuresTable");
  
    let folderFiles = [];
  
    /* ────────────────── Folder‑picker logic ────────────────── */
    if (folderInput) {
      folderInput.addEventListener("change", e => {
        folderFiles = Array.from(e.target.files);
        const mtx = folderFiles.filter(f => f.name.endsWith(".mtx"));
        fileDropdown.innerHTML = "";
        dropdownGroup.style.display = mtx.length ? "block" : "none";
        mtx.forEach(f => {
          const opt = document.createElement("option");
          opt.value = opt.textContent = f.name;
          fileDropdown.appendChild(opt);
        });
      });
    }
  
    /* ────────────────── Form submit ────────────────── */
    if (matrixForm) {
      matrixForm.addEventListener("submit", async evt => {
        evt.preventDefault();
  
        /* reset UI */
        resultsDiv.innerHTML = "";
        
  
        const selName = fileDropdown.value;
        const selFile = folderFiles.find(f => f.name === selName);
        if (!selFile) { showModal("Please select a .mtx file from the dropdown."); return; }
  
        loadingEl.style.display = "block";
  
        try {
          const fd = new FormData(matrixForm);
          fd.delete("matrix_folder");
          fd.set("matrix_file", selFile, selFile.name);
  
          const resp = await fetch("/generate-matrix", { method:"POST", body:fd });
          const data = await resp.json();
          if (data.error) { throw new Error(data.error); }
  
          /* ─── Build results card ─── */
          const m = data.matrix_info;
          resultsDiv.innerHTML = `
            <div class="card mb-4"><div class="card-body">
              <h5 class="card-title">Matrix Generated Successfully</h5>
              <p>Shape: ${m.shape[0]} × ${m.shape[1]}</p>
              <p>Non‑zero elements: ${m.nnz}</p>
              <p>Density: ${(m.density).toFixed(4)}%</p>
              <p>Creation time: ${m.creation_time_sec.toFixed(3)} seconds</p>
              <p>Property loss: ${m.property_loss.toFixed(2)}</p>
              <p>Cosine similarity (values): ${m.cosine_value.toFixed(2)}</p>
              <p>Cosine similarity (pattern): ${m.cosine_pattern.toFixed(2)}</p>
              <button onclick="saveMatrix()" class="btn btn-success mt-3">
                                        Save Matrix to Folder
                                    </button>
              <button id="showFeaturesBtn" class="btn btn-success mt-3 ms-2">Show Features</button>
              <button id="showVisualBtn"   class="btn btn-success mt-3 ms-2">Show Visual</button>
              <button id="showHeatmapBtn" class="btn btn-success mt-3 ms-2">Show Heatmap</button>
              <button id="showGraphBtn"   class="btn btn-success mt-3 ms-2">Show Graph</button>
            </div></div>`;
  
          wireFeatureToggle();
          wireSpyAndHeatmap(data);   // spy plot & heatmap via /visualize-matrix route
          wireGraphView(data);       // side‑by‑side graphs
          if (data.input_features && data.generated_features) {
            await populateFeatures(data.input_features, data.generated_features);
          }
        } catch (err) {
          console.error(err);
          resultsDiv.innerHTML = `<div class='alert alert-danger'>${err.message}</div>`;
        } finally {
          loadingEl.style.display = "none";
        }
      });
    }
  
    /* ────────────────── Helpers ────────────────── */
    /* Feature table toggle */
    function wireFeatureToggle() {
      const btn = document.getElementById("showFeaturesBtn");
      if (!btn || btn.dataset.wired) return;
      btn.dataset.wired = "1";
      btn.addEventListener("click", () => {
        const tableShown = featuresTbl.style.display !== "none";
        featuresTbl.style.display = tableShown ? "none" : "table";
        btn.textContent = tableShown ? "Show Features" : "Hide Features";
        // If showing, scroll to bottom slowly
        if (!tableShown) {
          setTimeout(() => {
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
            toggleBackToTopBtn();
          }, 1000);
        } else {
          toggleBackToTopBtn();
        }
      });
    }
    
    // Show the button only when features table is visible
    function toggleBackToTopBtn() {
      const btn = document.getElementById("backToTopBtn");
      const featuresTbl = document.getElementById("featuresTable");
      if (!btn || !featuresTbl) return;
      btn.style.display = (featuresTbl.style.display !== "none") ? "inline-block" : "none";
    }
  
    // Attach scroll-to-top logic and toggle on features show/hide
    document.addEventListener("DOMContentLoaded", function() {
      const btn = document.getElementById("backToTopBtn");
      if (btn) {
        btn.addEventListener("click", function() {
          window.scrollTo({ top: 0, behavior: "smooth" });
        });
      }
    });
  
    /* Spy plot + heatmap buttons (existing backend route) */
    function wireSpyAndHeatmap(data) {
      const spyBtn  = document.getElementById("showVisualBtn");
      const heatBtn = document.getElementById("showHeatmapBtn");
      const vizCont = document.getElementById("visualizationContainer");
      const vizImg  = document.getElementById("visualizationImage");
  
      // Helper to show/hide and set button text
      async function handleVizToggle(type, btn, otherBtn) {
        const isShown = vizCont.style.display !== "none" && btn.textContent.startsWith("Hide");
        if (isShown) {
          vizCont.style.display = "none";
          btn.textContent = btn.textContent.replace("Hide", "Show");
        } else {
          // Always fetch new image when showing
          const r = await fetch(`/visualize-matrix?type=${type}`);
          const d = await r.json();
          if (d.error) { showModal(d.error); return; }
          vizImg.src = d.image_path;
          vizCont.style.display = "block";
          btn.textContent = btn.textContent.replace("Show", "Hide");
          // Reset the other button
          if (otherBtn) otherBtn.textContent = otherBtn.textContent.replace("Hide", "Show");
          // Add 0.5s delay then scroll to bottom
          setTimeout(() => {
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
          }, 500);
        }
      }
  
      if (spyBtn && !spyBtn.dataset.wired) {
        spyBtn.dataset.wired = "1";
        spyBtn.addEventListener("click", async () => {
          await handleVizToggle("spy", spyBtn, heatBtn);
        });
      }
      if (heatBtn && !heatBtn.dataset.wired) {
        heatBtn.dataset.wired = "1";
        heatBtn.addEventListener("click", async () => {
          await handleVizToggle("heatmap", heatBtn, spyBtn);
        });
      }
    }
  
    /* Graph view toggle (side‑by‑side, uses static PNGs) */
    function wireGraphView(data) {
      const graphBtn = document.getElementById("showGraphBtn");
      const graphSec = document.getElementById("graphSection");
      const origImg  = document.getElementById("graphOriginalImg");
      const genImg   = document.getElementById("graphGeneratedImg");
  
      if (!graphBtn || !graphSec) return;
  
      graphBtn.style.display = "inline-block";
      graphSec.style.display = "none";
      graphBtn.textContent = "Show Graph";
  
      if (!graphBtn.dataset.wired) {
        graphBtn.dataset.wired = "1";
        graphBtn.addEventListener("click", async () => {
          const shown = graphSec.style.display !== "none";
          if (!shown) {
            // Show loading state
            graphBtn.disabled = true;
            graphBtn.textContent = "Loading Graphs...";
            
            try {
              // Check if graphs already exist
              const checkOrig = await fetch('/static/graph_original.png');
              const checkGen = await fetch('/static/graph_generated.png');
              
              if (!checkOrig.ok || !checkGen.ok) {
                // If graphs don't exist, generate them
                graphBtn.textContent = "Generating Graphs...";
                const response = await fetch('/generate-graphs');
                const result = await response.json();
                
                if (!result.success) {
                  throw new Error(result.error);
                }
              }
              
              // Update images with timestamp to prevent caching
              origImg.src = "/static/graph_original.png?t=" + Date.now();
              genImg.src = "/static/graph_generated.png?t=" + Date.now();
              
              // Wait for images to load
              await Promise.all([
                new Promise((resolve, reject) => {
                  origImg.onload = resolve;
                  origImg.onerror = reject;
                }),
                new Promise((resolve, reject) => {
                  genImg.onload = resolve;
                  genImg.onerror = reject;
                })
              ]);
              
              graphSec.style.display = "block";
              graphBtn.textContent = "Hide Graph";
              
              // Scroll to the graphs
              setTimeout(() => {
                window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
              }, 500);
            } catch (error) {
              alert("Error loading/generating graphs: " + error);
              graphBtn.textContent = "Show Graph";
              // Clear the image sources if there was an error
              origImg.src = "";
              genImg.src = "";
            } finally {
              graphBtn.disabled = false;
            }
          } else {
            graphSec.style.display = "none";
            graphBtn.textContent = "Show Graph";
          }
        });
      }
    }
  
    /* Populate feature table */
    const FEATURE_LABELS = {
      num_rows: "Number of rows",
      num_cols: "Number of columns",
      num_nonzeros: "Number of nonzero elements",
      density_percent: "Density (%)",
      pattern_symmetry: "Pattern symmetry",
      numerical_symmetry: "Numerical symmetry",
      nonzeros_per_row_avg: "Avg. nonzeros per row",
      nonzeros_per_col_avg: "Avg. nonzeros per column",
      value_min: "Minimum value",
      value_max: "Maximum value",
      value_avg: "Average value",
      value_std: "Value standard deviation",
      bandwidth: "Matrix bandwidth",
      avg_distance_to_diagonal: "Avg. distance to diagonal",
      norm_1: "1-norm",
      norm_inf: "Infinity norm",
      frobenius_norm: "Frobenius norm",
      estimated_condition_number: "Estimated condition number"
    };
  
    async function populateFeatures(inp, gen) {
      const tbody = featuresTbl.querySelector("tbody");
      tbody.innerHTML = "";
      // Only show features in FEATURE_LABELS
      Object.keys(FEATURE_LABELS).forEach(k => {
        const row = tbody.insertRow();
        row.insertCell().textContent = FEATURE_LABELS[k];
        row.insertCell().textContent = fmt(inp[k]);
        row.insertCell().textContent = fmt(gen[k]);
      });
      featuresTbl.style.display = "none";   // start hidden
      function fmt(v){ return typeof v==="number"? v.toFixed(4): (v??"N/A"); }
    }
  
    /* ────────────────── Algorithm‑specific UI ────────────────── */
    const algorithmSelect = document.getElementById("algorithm");
    const waveletGroup = document.getElementById("waveletTypeGroup");
    const imageGroup = document.getElementById("imageResamplingGroup");
    const kernelGroup = document.getElementById("kernelSizeGroup");
    const kroneckerGroup = document.getElementById("kroneckerGroup");

    if (algorithmSelect) {
        algorithmSelect.addEventListener("change", function() {
            const selectedAlgorithm = this.value;
            
            // Hide all parameter groups first
            waveletGroup.style.display = "none";
            imageGroup.style.display = "none";
            kernelGroup.style.display = "none";
            kroneckerGroup.style.display = "none";
            
            // Show relevant parameter group
            switch(selectedAlgorithm) {
                case "wavelet":
                    waveletGroup.style.display = "block";
                    break;
                case "image":
                    imageGroup.style.display = "block";
                    break;
                case "lanczos":
                    kernelGroup.style.display = "block";
                    break;
                case "kronecker":
                    kroneckerGroup.style.display = "block";
                    break;
            }
        });
    }
  
})();

window.saveMatrix = async function() {
    const loading = document.getElementById('loading');
    const loadingOriginalText = 'Generating matrix...';
    const loadingSavingText = 'Saving matrix...';
    const loadingParagraph = loading ? loading.querySelector('p') : null;
  
    try {
      if (loading) loading.style.display = 'block';
      if (loadingParagraph) loadingParagraph.textContent = loadingSavingText;
  
      const response = await fetch('/save-matrix', { method: 'POST' });
      const data = await response.json();
  
      if (data.error) throw new Error(data.error);
      // Show a simple message box with the backend message
      alert(data.message || 'Matrix saved successfully.');
  
    } catch (error) {
      console.error('Save matrix error:', error);
      alert(`Error saving matrix: ${error.message}`);
    } finally {
      if (loading) loading.style.display = 'none';
      if (loadingParagraph) loadingParagraph.textContent = loadingOriginalText;
    }
  };

function updateMatrixVisualization(data) {
    const timestamp = new Date().getTime();
    const spyPlot = document.getElementById('spyPlot');
    const heatmapPlot = document.getElementById('heatmapPlot');
    const graphPlot = document.getElementById('graphPlot');
    
    if (data.image_path) {
        // Add timestamp to prevent caching
        const imagePath = `${data.image_path}?t=${timestamp}`;
        
        if (spyPlot) {
            spyPlot.src = imagePath;
            spyPlot.style.display = 'block';
        }
        if (heatmapPlot) {
            heatmapPlot.src = imagePath;
            heatmapPlot.style.display = 'block';
        }
        if (graphPlot) {
            graphPlot.src = imagePath;
            graphPlot.style.display = 'block';
        }
    }
  }

function wireGraphView(data) {
    const graphSection = document.getElementById('graphSection');
    const showGraphBtn = document.getElementById('showGraphBtn');
    const originalGraph = document.getElementById('originalGraph');
    const generatedGraph = document.getElementById('generatedGraph');
    
    // Always show the button
    showGraphBtn.style.display = 'block';
    
    showGraphBtn.onclick = async function() {
        try {
            showGraphBtn.textContent = 'Generating Graphs...';
            showGraphBtn.disabled = true;
            
            const response = await fetch('/generate-graphs');
            const result = await response.json();
            
            if (result.success) {
                const timestamp = new Date().getTime();
                originalGraph.src = `/static/graph_original.png?t=${timestamp}`;
                generatedGraph.src = `/static/graph_generated.png?t=${timestamp}`;
                graphSection.style.display = 'block';
            } else {
                alert('Failed to generate graphs: ' + result.error);
            }
        } catch (error) {
            alert('Error generating graphs: ' + error);
        } finally {
            showGraphBtn.textContent = 'Show Graph';
            showGraphBtn.disabled = false;
        }
    };
  }