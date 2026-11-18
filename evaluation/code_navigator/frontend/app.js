/**
 * DeepTensor++ Research Navigator — Frontend Engine
 * Powered by Monaco Editor, KaTeX, and FastAPI Backend
 */

let editor = null;
let currentDecorations = [];
let topics = [];
let experiments = {};
let currentTopicId = 'DT-01';
let isPresentationMode = false;

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  initMonaco();
  fetchInitialData();
  setupEventListeners();
  checkUrlParams();
});

// 1. Initialize Monaco Editor
function initMonaco() {
  require(['vs/editor/editor.main'], function () {
    const container = document.getElementById('monaco-container');
    editor = monaco.editor.create(container, {
      value: '# Loading DeepTensor++ Codebase...',
      language: 'python',
      theme: 'vs-dark',
      readOnly: true,
      lineNumbers: 'on',
      minimap: { enabled: true },
      scrollBeyondLastLine: false,
      automaticLayout: true,
      fontSize: 13,
      fontFamily: "'Fira Code', Consolas, monospace",
      fontLigatures: true,
      renderLineHighlight: 'all',
      glyphMargin: true
    });

    // Resize listener
    window.addEventListener('resize', () => {
      editor.layout();
    });

    // If topic is already loaded, jump
    if (currentTopicId && topics.length > 0) {
      loadTopic(currentTopicId);
    }
  });
}

// 2. Fetch Initial Data from Backend
async function fetchInitialData() {
  try {
    const [resTopics, resExperiments] = await Promise.all([
      fetch('/api/topics'),
      fetch('/api/experiments')
    ]);

    topics = await resTopics.json();
    experiments = await resExperiments.json();

    renderTopicsSidebar();
    renderExperimentsSidebar();
    renderFileTreeSidebar();

    // Check if initial URL parameter specified topic
    const params = new URLSearchParams(window.location.search);
    const topicParam = params.get('topic');
    if (topicParam) {
      loadTopic(topicParam.toUpperCase());
    } else {
      loadTopic('DT-01');
    }
  } catch (err) {
    console.error('Failed to load initial data:', err);
    document.getElementById('api-status-badge').innerHTML = `
      <span class="status-indicator" style="background-color: var(--color-danger)"></span>
      <span class="status-text">Backend Offline</span>
    `;
  }
}

// 3. Render Topics in Sidebar
function renderTopicsSidebar() {
  const container = document.getElementById('topics-accordion');
  container.innerHTML = '';

  // Group topics by section
  const sections = {};
  topics.forEach(t => {
    if (!sections[t.section_id]) {
      sections[t.section_id] = {
        id: t.section_id,
        title: t.section_title,
        topics: []
      };
    }
    sections[t.section_id].topics.push(t);
  });

  Object.values(sections).forEach(sec => {
    const secDiv = document.createElement('div');
    secDiv.className = 'section-group';

    const header = document.createElement('div');
    header.className = 'section-header';
    header.innerHTML = `
      <span>${sec.id}: ${sec.title}</span>
      <i class="fa-solid fa-chevron-down" style="font-size: 0.65rem;"></i>
    `;
    secDiv.appendChild(header);

    const listDiv = document.createElement('div');
    listDiv.className = 'section-topic-list';

    sec.topics.forEach(t => {
      const item = document.createElement('div');
      item.className = `topic-item ${t.id === currentTopicId ? 'active' : ''}`;
      item.id = `sidebar-topic-${t.id}`;
      item.onclick = () => loadTopic(t.id);

      item.innerHTML = `
        <div class="topic-item-top">
          <span class="topic-item-id">${t.id}</span>
          <span class="badge ${t.verification_status.includes('Verified') ? 'badge-success' : 'badge-warning'}">
            ${t.verification_status.includes('Verified') ? 'VERIFIED' : 'AUDIT'}
          </span>
        </div>
        <div class="topic-item-title">${t.title}</div>
      `;
      listDiv.appendChild(item);
    });

    secDiv.appendChild(listDiv);
    container.appendChild(secDiv);
  });
}

// 4. Render Experiments in Sidebar
function renderExperimentsSidebar() {
  const container = document.getElementById('experiments-list');
  container.innerHTML = '';

  Object.values(experiments).forEach(exp => {
    const card = document.createElement('div');
    card.className = 'card';
    card.style.margin = '10px';
    card.style.cursor = 'pointer';
    card.onclick = () => openExperimentModal(exp.id);

    const winnerModel = exp.models[exp.winning_model];

    card.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <span class="badge badge-primary">${exp.id.toUpperCase()}</span>
        <span class="badge badge-success">Winner: ${exp.winning_model}</span>
      </div>
      <div style="font-size: 0.85rem; font-weight: 600; margin-bottom: 4px;">${exp.title}</div>
      <div style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono);">
        Best PSNR: <strong class="text-success">${winnerModel ? winnerModel.psnr.toFixed(2) : ''} dB</strong>
      </div>
      <div style="font-size: 0.72rem; color: var(--text-dim); margin-top: 4px;">
        ${exp.application} (${exp.dimensions})
      </div>
    `;
    container.appendChild(card);
  });
}

// 5. Render File Tree in Sidebar
function renderFileTreeSidebar() {
  const container = document.getElementById('file-tree');
  container.innerHTML = `
    <div style="padding: 12px; font-size: 0.8rem; font-family: var(--font-mono); display: flex; flex-direction: column; gap: 8px;">
      <div style="font-weight: 700; color: var(--text-muted);">CORE MODELS (src/deeptensor/models)</div>
      <div class="file-link" onclick="loadFile('src/deeptensor/models/deeptensor.py', 232, 286)">• deeptensor.py (DeepTensor, CP)</div>
      <div class="file-link" onclick="loadFile('src/deeptensor/models/factory.py', 145, 230)">• factory.py (5 Comparative Models)</div>
      <div class="file-link" onclick="loadFile('src/deeptensor/models/attention/spatial_spectral.py', 14, 230)">• spatial_spectral.py (Attention & Controls)</div>
      <div class="file-link" onclick="loadFile('src/deeptensor/models/attention/spectral_transformer.py', 44, 111)">• spectral_transformer.py (Pure Transformer)</div>
      <div class="file-link" onclick="loadFile('src/deeptensor/models/backbones/skip.py', 5, 197)">• skip.py (1D/2D U-Net Backbones)</div>

      <div style="font-weight: 700; color: var(--text-muted); margin-top: 12px;">PHYSICS OPERATORS</div>
      <div class="file-link" onclick="loadFile('src/deeptensor/physics/operators.py', 28, 97)">• operators.py (Radon & CACTI Video CS)</div>

      <div style="font-weight: 700; color: var(--text-muted); margin-top: 12px;">EXPERIMENT RUNNERS</div>
      <div class="file-link" onclick="loadFile('experiments/reproduction/figure09_hyperspectral/run_figure9.py', 1, 150)">• run_figure9.py (ICVL HSI)</div>
      <div class="file-link" onclick="loadFile('experiments/reproduction/figure10_yale_denoising/run_figure10_yale_tensor.py', 1, 150)">• run_figure10_yale_tensor.py (Yale Face B)</div>
      <div class="file-link" onclick="loadFile('experiments/hybrid_prototypes/run_hybrid_figure11_video.py', 1, 150)">• run_hybrid_figure11_video.py (Video CS)</div>
      <div class="file-link" onclick="loadFile('experiments/hybrid_prototypes/run_hybrid_figure12_pet.py', 1, 150)">• run_hybrid_figure12_pet.py (PET/CT)</div>

      <div style="font-weight: 700; color: var(--text-muted); margin-top: 12px;">IMPROVEMENTS & EXTENSIONS</div>
      <div class="file-link" onclick="loadFile('experiments/extensions/adaptive_rank/run_figure9_adaptive.py', 64, 140)">• run_figure9_adaptive.py (Adaptive Rank)</div>
      <div class="file-link" onclick="loadFile('experiments/extensions/group_sparsity/run_figure9_sparsity.py', 83, 120)">• run_figure9_sparsity.py (Factor Group-Sparsity)</div>
    </div>
  `;
}

// 6. Load Topic and Synchronize UI & Code
async function loadTopic(topicId) {
  const topic = topics.find(t => t.id === topicId);
  if (!topic) return;

  currentTopicId = topicId;

  // Update active item in sidebar
  document.querySelectorAll('.topic-item').forEach(el => el.classList.remove('active'));
  const activeEl = document.getElementById(`sidebar-topic-${topicId}`);
  if (activeEl) {
    activeEl.classList.add('active');
    activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Update Right Detail Panel
  document.getElementById('topic-id-badge').textContent = topic.id;
  document.getElementById('topic-title').textContent = topic.title;
  document.getElementById('topic-section-title').textContent = `${topic.section_id}: ${topic.section_title}`;
  document.getElementById('topic-source-badge').textContent = topic.source_of_truth;
  document.getElementById('topic-status-badge').innerHTML = `<i class="fa-solid fa-check"></i> ${topic.verification_status}`;
  
  document.getElementById('topic-concept').textContent = topic.concept;
  document.getElementById('topic-context').textContent = topic.research_context;
  document.getElementById('topic-math-explanation').textContent = topic.math_explanation;

  // Render LaTeX Formula
  const mathContainer = document.getElementById('topic-math-container');
  if (window.katex && topic.math_formula) {
    try {
      katex.render(topic.math_formula, mathContainer, { displayMode: true, throwOnError: false });
    } catch (e) {
      mathContainer.textContent = topic.math_formula;
    }
  } else {
    mathContainer.textContent = topic.math_formula;
  }

  // Code Symbol Info
  const codeRef = topic.code_reference;
  document.getElementById('topic-code-file').textContent = codeRef.file_path;
  document.getElementById('topic-code-lines').textContent = `L${codeRef.line_start}–L${codeRef.line_end}`;
  document.getElementById('topic-code-symbol').textContent = codeRef.symbol_name;
  document.getElementById('topic-code-summary').textContent = codeRef.summary;

  document.getElementById('btn-jump-code').onclick = () => {
    loadFile(codeRef.file_path, codeRef.line_start, codeRef.line_end);
  };

  // Faculty Viva Q&A Info
  document.getElementById('topic-qa-question').textContent = topic.faculty_qa.question;
  document.getElementById('topic-qa-defense').textContent = topic.faculty_qa.defense;

  // Update Presentation Toolbar
  if (isPresentationMode) {
    updatePresentationBar();
  }

  // Synchronize Browser URL
  const url = new URL(window.location);
  url.searchParams.set('topic', topic.id);
  window.history.pushState({}, '', url);

  // Load File in Monaco Editor
  await loadFile(codeRef.file_path, codeRef.line_start, codeRef.line_end);
}

// 7. Load Source File and Apply Monaco Line Highlighting
async function loadFile(filePath, lineStart, lineEnd) {
  if (!editor) return;

  try {
    const res = await fetch(`/api/code?file_path=${encodeURIComponent(filePath)}&line_start=${lineStart || ''}&line_end=${lineEnd || ''}`);
    if (!res.ok) throw new Error('File not found');
    const data = await res.json();

    // Set model
    const currentModel = editor.getModel();
    if (!currentModel || currentModel.uri.path !== filePath) {
      editor.setValue(data.content);
    }

    // Update Toolbar Breadcrumb
    document.getElementById('current-file-path').textContent = filePath;
    document.getElementById('current-line-badge').textContent = lineStart && lineEnd ? `Lines ${lineStart}–${lineEnd}` : 'Full File';

    // Highlight target lines
    if (lineStart && lineEnd) {
      currentDecorations = editor.deltaDecorations(currentDecorations, [
        {
          range: new monaco.Range(lineStart, 1, lineEnd, 1),
          options: {
            isWholeLine: true,
            className: 'monaco-highlight-line',
            glyphMarginClassName: 'monaco-highlight-glyph',
            overviewRuler: {
              color: '#f59e0b',
              position: monaco.editor.OverviewRulerLane.Full
            }
          }
        }
      ]);

      // Reveal in Center
      editor.revealLineInCenter(lineStart);
    } else {
      currentDecorations = editor.deltaDecorations(currentDecorations, []);
    }
  } catch (err) {
    console.error('Failed to load file:', err);
    editor.setValue(`# Error: Could not load file: ${filePath}\n# ${err.message}`);
  }
}

// 8. Presentation Mode Logic
function togglePresentationMode() {
  isPresentationMode = !isPresentationMode;
  const bar = document.getElementById('presentation-toolbar');
  const btn = document.getElementById('btn-presentation-mode');

  if (isPresentationMode) {
    bar.classList.remove('hidden');
    btn.classList.add('btn-primary');
    btn.classList.remove('btn-outline');
    updatePresentationBar();
  } else {
    bar.classList.add('hidden');
    btn.classList.remove('btn-primary');
    btn.classList.add('btn-outline');
  }
}

function updatePresentationBar() {
  const currentIndex = topics.findIndex(t => t.id === currentTopicId);
  const currentTopic = topics[currentIndex];

  document.getElementById('pres-slide-num').textContent = `Slide ${currentIndex + 1} / ${topics.length}`;
  document.getElementById('pres-slide-title').textContent = `[${currentTopic.id}] ${currentTopic.title}`;
}

function nextSlide() {
  const currentIndex = topics.findIndex(t => t.id === currentTopicId);
  if (currentIndex < topics.length - 1) {
    loadTopic(topics[currentIndex + 1].id);
  }
}

function prevSlide() {
  const currentIndex = topics.findIndex(t => t.id === currentTopicId);
  if (currentIndex > 0) {
    loadTopic(topics[currentIndex - 1].id);
  }
}

// 9. Experiment Modal Logic
function openExperimentModal(expId = 'figure09') {
  const modal = document.getElementById('modal-experiments');
  modal.classList.remove('hidden');
  renderExperimentModalContent(expId);
}

function closeExperimentModal() {
  document.getElementById('modal-experiments').classList.add('hidden');
}

function renderExperimentModalContent(expId) {
  const exp = experiments[expId];
  if (!exp) return;

  // Update tabs
  document.querySelectorAll('.exp-modal-tab').forEach(tab => {
    tab.classList.toggle('active', tab.getAttribute('data-exp') === expId);
  });

  const content = document.getElementById('exp-modal-content');
  
  // Format comparative table
  let tableRows = '';
  const modelKeys = ['baseline', 'cnn_control', 'pointwise_control', 'hybrid', 'transformer'];
  const modelLabels = {
    'baseline': 'Baseline CNN',
    'cnn_control': 'CNN Control',
    'pointwise_control': 'Pointwise Control',
    'hybrid': 'Hybrid CNN-Transformer',
    'transformer': 'Pure Transformer'
  };

  modelKeys.forEach(mKey => {
    const m = exp.models[mKey];
    if (!m) return;
    const isWinner = m.winner;
    tableRows += `
      <tr class="${isWinner ? 'winner-row' : ''}">
        <td><strong>${modelLabels[mKey] || mKey}</strong> ${isWinner ? '🏆 (WINNER)' : ''}</td>
        <td>${m.psnr.toFixed(4)} dB</td>
        <td>${m.ssim.toFixed(4)}</td>
        <td>${m.rmse.toFixed(5)}</td>
        <td>${m.mae.toFixed(5)}</td>
        <td>${m.params.toLocaleString()}</td>
        <td>${m.runtime_sec.toFixed(1)}s</td>
      </tr>
    `;
  });

  content.innerHTML = `
    <div style="margin-bottom: 16px;">
      <h3 style="font-size: 1.2rem; font-weight: 700; margin-bottom: 4px;">${exp.title}</h3>
      <p style="font-size: 0.85rem; color: var(--text-muted);">
        Dataset: <strong>${exp.dataset}</strong> | Dimensions: <strong>${exp.dimensions}</strong> | Rank: <strong>R = ${exp.rank}</strong> | Budget: <strong>${exp.epochs} epochs</strong>
      </p>
    </div>

    <div class="card" style="margin-bottom: 16px;">
      <div class="card-header">
        <i class="fa-solid fa-trophy card-icon text-accent"></i>
        <h3>Empirical Winner & Inductive Bias Analysis</h3>
      </div>
      <p style="font-size: 0.9rem; margin-bottom: 8px;">
        <strong>Winning Architecture:</strong> <span class="badge badge-success">${modelLabels[exp.winning_model]}</span>
        <span style="margin-left: 8px; color: var(--color-success); font-weight: 600;">(${exp.winning_margin})</span>
      </p>
      <p style="font-size: 0.85rem; line-height: 1.5; color: #cbd5e1;">${exp.inductive_bias_rationale}</p>
    </div>

    <div>
      <h4 style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px;">
        Controlled 5-Model Comparative Matrix
      </h4>
      <table class="exp-table">
        <thead>
          <tr>
            <th>Architecture Model</th>
            <th>PSNR (dB)</th>
            <th>SSIM</th>
            <th>RMSE</th>
            <th>MAE</th>
            <th>Parameters</th>
            <th>Runtime</th>
          </tr>
        </thead>
        <tbody>
          ${tableRows}
        </tbody>
      </table>
    </div>

    <div style="margin-top: 20px; display: flex; gap: 12px;">
      <button class="btn btn-primary" onclick="closeExperimentModal(); loadFile('${exp.runner_script}', 1, 150);">
        <i class="fa-solid fa-code"></i> View Experiment Runner Code
      </button>
      <button class="btn btn-outline" onclick="closeExperimentModal(); loadTopic('${exp.topic_id}');">
        <i class="fa-solid fa-graduation-cap"></i> View Research Topic [${exp.topic_id}]
      </button>
    </div>
  `;
}

// 10. Search Functionality
function setupSearch() {
  const input = document.getElementById('global-search');
  const dropdown = document.getElementById('search-dropdown');
  let debounceTimer = null;

  input.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    const q = input.value.trim();
    if (q.length < 2) {
      dropdown.classList.add('hidden');
      return;
    }

    debounceTimer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
        const data = await res.json();

        dropdown.innerHTML = '';
        if (data.matched_topics.length === 0 && data.matched_symbols.length === 0) {
          dropdown.innerHTML = `<div style="padding: 12px; font-size: 0.8rem; color: var(--text-dim);">No matches found for "${q}"</div>`;
          dropdown.classList.remove('hidden');
          return;
        }

        // Matched Topics
        data.matched_topics.forEach(t => {
          const div = document.createElement('div');
          div.className = 'search-result-item';
          div.onclick = () => {
            loadTopic(t.id);
            dropdown.classList.add('hidden');
            input.value = '';
          };
          div.innerHTML = `
            <div class="search-result-header">
              <span class="search-result-title">[${t.id}] ${t.title}</span>
              <span class="badge badge-primary">TOPIC</span>
            </div>
            <span class="search-result-sub">${t.section} • ${t.file_path}</span>
          `;
          dropdown.appendChild(div);
        });

        // Matched Symbols
        data.matched_symbols.forEach(s => {
          const div = document.createElement('div');
          div.className = 'search-result-item';
          div.onclick = () => {
            loadFile(s.file_path, s.line, s.line + 20);
            dropdown.classList.add('hidden');
            input.value = '';
          };
          div.innerHTML = `
            <div class="search-result-header">
              <span class="search-result-title">${s.name}</span>
              <span class="badge badge-neutral">CODE</span>
            </div>
            <span class="search-result-sub">${s.file_path}:L${s.line}</span>
          `;
          dropdown.appendChild(div);
        });

        dropdown.classList.remove('hidden');
      } catch (err) {
        console.error('Search error:', err);
      }
    }, 250);
  });

  // Close search when clicking outside
  document.addEventListener('click', (e) => {
    if (!input.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.classList.add('hidden');
    }
  });
}

// 11. Event Listeners & Hotkeys
function setupEventListeners() {
  // Sidebar Tabs
  document.querySelectorAll('.sidebar-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const paneId = tab.getAttribute('data-tab');
      document.getElementById(paneId).classList.add('active');
    });
  });

  // Presentation Mode Toggle
  document.getElementById('btn-presentation-mode').onclick = togglePresentationMode;
  document.getElementById('btn-pres-next').onclick = nextSlide;
  document.getElementById('btn-pres-prev').onclick = prevSlide;
  document.getElementById('btn-pres-exit').onclick = togglePresentationMode;

  // Experiments Modal
  document.getElementById('btn-experiments-view').onclick = () => openExperimentModal('figure09');
  document.getElementById('btn-close-modal').onclick = closeExperimentModal;

  document.querySelectorAll('.exp-modal-tab').forEach(tab => {
    tab.onclick = () => {
      renderExperimentModalContent(tab.getAttribute('data-exp'));
    };
  });

  // Copy Path Button
  document.getElementById('btn-copy-path').onclick = () => {
    const pathText = document.getElementById('current-file-path').textContent;
    navigator.clipboard.writeText(pathText);
    const btn = document.getElementById('btn-copy-path');
    btn.innerHTML = `<i class="fa-solid fa-check text-success"></i> Copied!`;
    setTimeout(() => {
      btn.innerHTML = `<i class="fa-regular fa-copy"></i> Copy Path`;
    }, 1500);
  };

  // Focus Lines Button
  document.getElementById('btn-reset-view').onclick = () => {
    const topic = topics.find(t => t.id === currentTopicId);
    if (topic && editor) {
      editor.revealLineInCenter(topic.code_reference.line_start);
    }
  };

  // Keyboard Hotkeys
  document.addEventListener('keydown', (e) => {
    // Left/Right Arrow for Presentation Mode
    if (isPresentationMode) {
      if (e.key === 'ArrowRight' || e.key === 'PageDown') {
        nextSlide();
      } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
        prevSlide();
      } else if (e.key === 'Escape') {
        togglePresentationMode();
      }
    } else {
      if (e.key === 'p' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        togglePresentationMode();
      }
    }

    if (e.key === 'Escape') {
      closeExperimentModal();
      document.getElementById('search-dropdown').classList.add('hidden');
    }
  });

  setupSearch();
}

function checkUrlParams() {
  const params = new URLSearchParams(window.location.search);
  const expParam = params.get('exp');
  if (expParam) {
    openExperimentModal(expParam);
  }
}

