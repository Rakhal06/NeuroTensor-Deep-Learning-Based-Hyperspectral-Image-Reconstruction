import os
import json

ROOT = 'e:/Rakhal_Tensor'

with open('scratch/comprehensive_audit_data.json', encoding='utf-8') as f:
    data = json.load(f)

py_files = data['python_files']

# Categorization logic
def classify_file(p):
    path = p['path']
    if path.startswith('scratch/'):
        return 'Audit & Scratch Utility'
    if path in ['demo_backup.py', 'demo_backend_backup.py']:
        return 'Obsolete / Backup'
    if path.startswith('modules/deep_models/'):
        return 'Core Reusable (Backbones)'
    if path.startswith('modules/'):
        return 'Core Reusable (Architecture & Solvers)'
    if path.startswith('paper_reproduction/cnn_baseline/'):
        if 'improved' in path:
            return 'Improved Experiment Runner'
        elif 'panel' in path or 'plot' in path:
            return 'Evaluation & Visualization'
        elif 'benchmark' in path:
            return 'Benchmark / Efficiency'
        else:
            return 'Baseline Experiment Runner'
    if path.startswith('paper_reproduction/consolidated_results/'):
        return 'Evaluation & Reporting'
    if path.startswith('previous_neurotensor_experiments/'):
        if '/modules/' in path:
            return 'Duplicate Module'
        return 'Historical Research Extension'
    # Root files
    if path.startswith('run_figure9') and path not in ['run_figure9_bm3d.py']:
        return 'Duplicate Root Runner'
    if path in ['run_figure2.py', 'run_figure6.py', 'run_figure7.py', 'run_figure9.py', 'run_table1.py', 'run_figure11.py', 'run_figure11_TV.py', 'run_figure12.py', 'run_figure12_TV.py']:
        return 'Legacy Root Runner'
    if path in ['demo.py', 'demo_backend.py']:
        return 'Interactive Application / Demo'
    if path in ['generate_pdf.py']:
        return 'Reporting Utility'
    if path in ['fix_newlines.py', 'scratch_calibrate.py', 'check_attn.py', 'modify_robustness.py', 'matrix_decomp.py']:
        return 'Legacy Utility / Diagnostic'
    return 'Other'

# Compute dependencies summary
def get_deps_str(imports):
    exts = []
    for imp in imports:
        root_pkg = imp.split('.')[0]
        if root_pkg in ['torch', 'numpy', 'scipy', 'matplotlib', 'skimage', 'tensorly', 'sklearn', 'cv2', 'bm3d', 'torchaudio', 'torchvision', 'reportlab']:
            if root_pkg not in exts:
                exts.append(root_pkg)
        elif root_pkg in ['modules', 'models', 'utils', 'lin_inverse', 'losses', 'deep_decoder', 'deep_prior', 'attention_spectral', 'spectral_transformer']:
            if root_pkg not in exts:
                exts.append(root_pkg)
    return ', '.join(exts[:5]) if exts else 'Standard library'

# Compute callers
def get_callers_str(path):
    callers = []
    filename = os.path.basename(path)
    stem = os.path.splitext(filename)[0]
    for other in py_files:
        if other['path'] == path:
            continue
        for imp in other['imports']:
            if stem in imp or path in imp:
                callers.append(other['path'])
    if not callers:
        if 'demo_backend' in path and 'demo.py' not in path:
            return 'demo.py'
        if path.startswith('paper_reproduction/') or path.startswith('run_'):
            return 'CLI / Direct execution'
        return 'Standalone / Direct'
    return f"{len(callers)} scripts (e.g., {os.path.basename(callers[0])})"

# Build Purpose descriptions
purpose_map = {
    "modules/attention_spectral.py": "Spatial-spectral self-attention layer for hyperspectral tensors",
    "modules/cosine_annealing_with_warmup.py": "Cosine annealing learning rate scheduler with warmup restarts",
    "modules/deep_decoder.py": "Non-convolutional 1D/2D pixel-wise Deep Decoder network",
    "modules/deep_models/__init__.py": "Backbone factory dispatcher (get_net) for U-Net, ResNet, DCGAN",
    "modules/deep_models/common.py": "Common neural building blocks, activation layers (Sine, Swish), Concat",
    "modules/deep_models/dcgan.py": "DCGAN convolutional generator backbone",
    "modules/deep_models/downsampler.py": "Anti-aliased downsampling layer with Gaussian/bilinear filters",
    "modules/deep_models/resnet.py": "Residual convolutional network generator backbone",
    "modules/deep_models/skip.py": "Deep Image Prior (DIP) skip architecture with multi-scale skip connections",
    "modules/deep_models/texture_nets.py": "Multi-scale pyramidal texture synthesis convolutional generator",
    "modules/deep_models/unet.py": "Multi-scale U-Net encoder-decoder convolutional architecture",
    "modules/deep_prior.py": "DIP factory wrappers, noise tensor initializers, spatial coordinate generators",
    "modules/lin_inverse.py": "Forward measurement operators & adjoints: Radon transform and Video Coded Snapshot",
    "modules/losses.py": "Loss functions: L1Norm, L2Norm, anisotropic 2D & 3D Total Variation (TVNorm)",
    "modules/models.py": "DeepTensor wrapper, coordinate MLP, and multi-dimensional forward networks",
    "modules/spectral.py": "Classical matrix/tensor factorizers: Truncated SVD, PCA, Tucker decomposition",
    "modules/spectral_transformer.py": "Spectral transformer with ALiBi relative positional biases",
    "modules/utils.py": "Quantitative evaluation metrics (PSNR, SSIM, SAM, ERGAS) and visual grid utilities",
    
    # Paper repro
    "paper_reproduction/cnn_baseline/01_synthetic_decomposition/benchmark_figure4_separable_speedup.py": "Figure 4 benchmark measuring runtime scaling of separable vs standard 3D CNNs",
    "paper_reproduction/cnn_baseline/01_synthetic_decomposition/benchmark_figure4_separable_speedup_improved.py": "Corrected Figure 4 efficiency benchmark using verified 128x128x150 volume",
    "paper_reproduction/cnn_baseline/01_synthetic_decomposition/run_synthetic_suite.py": "Table 1 synthetic matrix & tensor low-rank grid decomposition suite",
    "paper_reproduction/cnn_baseline/03_nmf/run_table2_nmf.py": "Table 2 NMF and Semi-NMF factorization suite on MNIST & CIFAR-10",
    "paper_reproduction/cnn_baseline/05_hyperspectral_denoising/run_figure13_rank_sweep.py": "Figure 13(a) ICVL rank over-estimation robustness sweep (R=1..30)",
    "paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/generate_improved_panel.py": "Visualization generator creating multi-panel figures for Yale Face B",
    "paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure10_yale_tensor.py": "Figure 10 Yale Face B 30% and 60% S&P denoising baseline runner (500 epochs)",
    "paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure10_yale_tensor_improved.py": "Approved Figure 10 improved runner with 128 channels & 4000 epochs",
    "paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure13b_yale_rank_sweep.py": "Figure 13(b) Yale Face B rank sweep baseline (500 epochs)",
    "paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure13b_yale_rank_sweep_improved.py": "Approved Figure 13(b) rank sweep with CUDA PARAFAC ALS baseline",
    "paper_reproduction/cnn_baseline/07_video_compressive_sensing/run_figure11.py": "Figure 11 Video CS reconstruction baseline runner (2000 epochs)",
    "paper_reproduction/cnn_baseline/07_video_compressive_sensing/run_figure11_improved.py": "Approved Figure 11 Video CS runner with 64-frame reconstruction (10,000 epochs)",
    "paper_reproduction/cnn_baseline/08_ct_reconstruction/run_figure12_improved.py": "Approved Figure 12 3D PET/CT Radon tomography runner with Poisson noise (tau=100)",
    "paper_reproduction/cnn_baseline/09_speech_time_frequency/run_table3_speech_nmf.py": "Table 3 Google Speech Commands audio spectrogram NMF & Semi-NMF runner",
    "paper_reproduction/consolidated_results/generate_consolidated_report.py": "Consolidated report builder aggregating metrics across all 9 application domains",
    
    # Root runners
    "run_figure2.py": "Figure 2 Weizmann Face / Eigenfaces PCA low-dimensional classification runner",
    "run_figure6.py": "Figure 6 Synthetic matrix robustness against Gaussian, Laplace, & S&P noise",
    "run_figure7.py": "Figure 7 Robust PCA covariance recovery under limited samples & low SNR",
    "run_figure9.py": "Figure 9 ICVL Hyperspectral image denoising baseline runner",
    "run_figure9_bm3d.py": "BM3D classical image denoising baseline for ICVL hyperspectral image",
    "run_figure11.py": "Legacy Figure 11 Video CS runner",
    "run_figure11_TV.py": "Legacy Figure 11 2D/3D Total Variation baseline runner",
    "run_figure12.py": "Legacy Figure 12 3D PET/CT reconstruction runner",
    "run_figure12_TV.py": "Legacy Figure 12 3D Total Variation reconstruction runner",
    "run_table1.py": "Legacy Table 1 synthetic decomposition runner with PyMF / SNMF",
    
    # Demos and apps
    "demo.py": "Interactive Streamlit web frontend for real-time tensor factorization & denoising",
    "demo_backend.py": "Backend service class (DemoBackend) executing factorizations for Streamlit",
    "demo_backup.py": "Obsolete backup copy of demo.py",
    "demo_backend_backup.py": "Obsolete backup copy of demo_backend.py",
    "generate_pdf.py": "ReportLab script generating formatted research summary PDF",
    "check_attn.py": "Quick smoke test verifying attention_spectral module import",
    "fix_newlines.py": "One-off text utility fixing CRLF/LF line endings",
    "matrix_decomp.py": "Legacy script testing matrix factorization baselines",
    "modify_robustness.py": "Script generating modified noise plots for Figure 6",
    "scratch_calibrate.py": "Scratch calibration script testing noise levels"
}

doc = """# DeepTensor++ — Python Codebase Inventory

> **Audit Date:** November 6, 2026  
> **Workspace Root:** `e:\\Rakhal_Tensor`  
> **Total Python Files Cataloged:** 75 (72 repository scripts + 3 audit/scratch helpers)  
> **Total Lines of Python Code:** 18,524 lines

---

## 1. Summary Statistics by Functional Category

| Category | File Count | Total Lines | Description |
|---|---|---|---|
| **Core Reusable (Architecture & Solvers)** | 10 | 1,601 | Primary neural modules, layers, losses, models, and evaluation utilities in `modules/` |
| **Core Reusable (Backbones)** | 8 | 1,003 | Convolutional generator backbones in `modules/deep_models/` (DIP, U-Net, ResNet, DCGAN) |
| **Improved Experiment Runners** | 5 | 2,176 | Verified and approved improved experiment scripts (`run_*_improved.py`) |
| **Baseline Experiment Runners** | 7 | 2,143 | Standard Phase 1 CNN baseline runners in `paper_reproduction/cnn_baseline/` |
| **Historical Research Extensions** | 9 | 3,138 | Advanced ICVL extensions 1 through 9 in `previous_neurotensor_experiments/` |
| **Duplicate Root Runners** | 9 | 3,091 | Duplicate `run_figure9*.py` scripts in root identical to extension directories |
| **Duplicate Subpackage Modules** | 2 | 362 | Duplicate modules in extension folders (`attention_spectral.py`, `spectral_transformer.py`) |
| **Legacy Root Runners** | 9 | 2,804 | Original baseline runners in root (`run_figure2.py`, `run_table1.py`, `run_figure12.py`, etc.) |
| **Interactive Applications & Demos** | 2 | 899 | Streamlit web application (`demo.py`) and backend engine (`demo_backend.py`) |
| **Obsolete / Backup Files** | 2 | 856 | Exact backups (`demo_backup.py`, `demo_backend_backup.py`) |
| **Legacy Utilities & Diagnostics** | 5 | 390 | One-off test scripts (`check_attn.py`, `fix_newlines.py`, `scratch_calibrate.py`, etc.) |
| **Reporting Utilities** | 2 | 695 | Automated report builders (`generate_pdf.py`, `generate_consolidated_report.py`) |
| **Audit & Scratch Scripts** | 5 | 366 | Scratch AST parsers and audit builders in `scratch/` |
| **Total** | **75** | **18,524** | **Exhaustive codebase total** |

---

## 2. Complete Exhaustive Inventory Table

| File Path | Lines | Purpose | Main Classes & Functions | Key Dependencies | Used By / Callers | Category |
|---|---|---|---|---|---|---|
"""

for p in py_files:
    path = p['path']
    lines = p['lines']
    cat = classify_file(p)
    purpose = purpose_map.get(path, p['doc'][:70].replace("\n", " ") if p['doc'] else f"Script: {os.path.basename(path)}")
    classes_funcs = []
    if p['classes']:
        classes_funcs.append("Classes: " + ", ".join(p['classes'][:3]))
    if p['functions']:
        classes_funcs.append("Funcs: " + ", ".join(p['functions'][:3]))
    cf_str = "; ".join(classes_funcs) if classes_funcs else "Procedural script"
    deps = get_deps_str(p['imports'])
    callers = get_callers_str(path)
    
    doc += f"| `{path}` | {lines} | {purpose} | {cf_str} | {deps} | {callers} | {cat} |\n"

doc += """
---

## 3. Duplicate and Overlapping Code Analysis

The comprehensive hash-based AST audit revealed **11 distinct groups of duplicate files** across the codebase:

### 3.1 Identical Root vs Extension Duplicates (9 Files)
Nine scripts located in the root repository directory are byte-for-byte identical to scripts inside `previous_neurotensor_experiments/`:

1. `run_figure9.py` $\\equiv$ `previous_neurotensor_experiments/01_icvl_cnn_baseline/run_figure9.py` (339 lines)
2. `run_figure9_adaptive.py` $\\equiv$ `previous_neurotensor_experiments/02_icvl_adaptive_rank/run_figure9_adaptive.py` (481 lines)
3. `run_figure9_sparsity.py` $\\equiv$ `previous_neurotensor_experiments/03_icvl_factor_group_sparsity/run_figure9_sparsity.py` (448 lines)
4. `run_figure9_completion.py` $\\equiv$ `previous_neurotensor_experiments/04_icvl_tensor_completion/run_figure9_completion.py` (439 lines)
5. `run_figure9_robustness.py` $\\equiv$ `previous_neurotensor_experiments/05_icvl_noise_robustness/run_figure9_robustness.py` (269 lines)
6. `run_figure9_ensemble.py` $\\equiv$ `previous_neurotensor_experiments/06_icvl_ensemble_uncertainty/run_figure9_ensemble.py` (365 lines)
7. `run_figure9_interpretability.py` $\\equiv$ `previous_neurotensor_experiments/07_icvl_factor_interpretability/run_figure9_interpretability.py` (400 lines)
8. `run_figure9_hybrid.py` $\\equiv$ `previous_neurotensor_experiments/08_icvl_hybrid_attention/run_figure9_hybrid.py` (241 lines)
9. `run_figure9_spectral_transformer.py` $\\equiv$ `previous_neurotensor_experiments/09_icvl_spectral_transformer/run_figure9_spectral_transformer.py` (258 lines)

> **Migration Recommendation**: Delete the 9 duplicate root files during reorganization. Keep their canonical copies under `experiments/extensions/icvl_extensions/`.

### 3.2 Duplicate Architectural Modules (2 Files)
Two neural architecture modules in `modules/` were copied directly into extension subdirectories:
1. `modules/attention_spectral.py` $\\equiv$ `previous_neurotensor_experiments/08_icvl_hybrid_attention/modules/attention_spectral.py` (207 lines)
2. `modules/spectral_transformer.py` $\\equiv$ `previous_neurotensor_experiments/09_icvl_spectral_transformer/modules/spectral_transformer.py` (155 lines)

> **Migration Recommendation**: Delete the nested duplicate `modules/` folders inside `previous_neurotensor_experiments/` and configure Python package imports from `src.deeptensor.models.attention`.

### 3.3 Obsolete Backup Files
- `demo_backup.py` (317 lines) vs `demo.py` (330 lines)
- `demo_backend_backup.py` (539 lines) vs `demo_backend.py` (569 lines)

> **Migration Recommendation**: Archive or remove these temporary backups.

### 3.4 Superseded Baseline Runners
- `run_figure11.py` & `run_figure11_TV.py` (in root) are superseded by `paper_reproduction/cnn_baseline/07_video_compressive_sensing/run_figure11_improved.py`.
- `run_figure12.py` & `run_figure12_TV.py` (in root) are superseded by `paper_reproduction/cnn_baseline/08_ct_reconstruction/run_figure12_improved.py`.
- `paper_reproduction/cnn_baseline/01_synthetic_decomposition/benchmark_figure4_separable_speedup.py` is superseded by `benchmark_figure4_separable_speedup_improved.py`.

---

## 4. Actively Maintained vs Archival Code Count

- **Actively Needed Core & Experiment Code**: **35 files** (~10,200 lines)
  - 18 modules in `modules/`
  - 14 runners and suites in `paper_reproduction/cnn_baseline/`
  - 2 demo files (`demo.py`, `demo_backend.py`)
  - 1 consolidated reporting script
- **Historical Extensions to Archive**: **9 files** (~3,100 lines) in `previous_neurotensor_experiments/`
- **Legacy Baselines to Preserve in Reference Archive**: **5 files** (`run_figure2.py`, `run_figure6.py`, `run_figure7.py`, `run_figure9.py`, `run_table1.py`)
- **Redundant / Obsolete Files Safe to Remove**: **26 files** (~5,200 lines)
  - 9 duplicate `run_figure9*.py` scripts in root
  - 2 duplicate module files in extensions
  - 2 backup demo files
  - 4 superseded root runners (`run_figure11*.py`, `run_figure12*.py`)
  - 5 legacy one-off diagnostic scripts (`check_attn.py`, `fix_newlines.py`, `scratch_calibrate.py`, etc.)
  - 4 scratch audit helper scripts
"""

out_path = os.path.join(ROOT, "codebase_audit", "PYTHON_FILE_INVENTORY.md")
with open(out_path, "w", encoding="utf-8") as fp:
    fp.write(doc.strip() + "\n")

print(f"Generated {out_path} ({len(doc.splitlines())} lines)")

