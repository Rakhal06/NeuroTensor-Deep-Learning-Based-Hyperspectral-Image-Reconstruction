# DeepTensor++ — Comprehensive Codebase Audit & Architectural Assessment Executive Summary

> **Audit Date:** November 6, 2026  
> **Workspace Root:** `e:\Rakhal_Tensor`  
> **Audit Status:** Rigorous Read-Only Verification Complete  
> **Deliverables Generated:** 7 Audit Documents in `codebase_audit/`  
> **Immediate Action:** Strictly no modifications made to existing project files; awaiting user review and approval.

---

## 1. Executive Summary & Core Answers to the 10 Audit Questions

### Q1: How many total Python files exist, and how many are actively needed?
- **Total Existing Python Files**: **75 files** (72 repository scripts + 3 audit/scratch helpers) totaling **18,524 lines of code**.
- **Actively Needed Core & Experiment Code**: **35 files** (~10,200 lines):
  - 18 core modules in `modules/` and `modules/deep_models/`
  - 14 reproduction runner scripts & suites in `paper_reproduction/cnn_baseline/`
  - 2 interactive demo files (`demo.py`, `demo_backend.py`)
  - 1 consolidated reporting generator
- **Historical Extensions to Archive**: **9 files** (~3,100 lines) in `previous_neurotensor_experiments/`
- **Legacy Baseline Runners to Preserve in Reference Archive**: **5 files** (`run_figure2.py`, `run_figure6.py`, `run_figure7.py`, `run_figure9.py`, `run_table1.py`)
- **Redundant / Duplicate / Obsolete Files**: **26 files** (~5,200 lines) that can be removed or archived without any loss of functionality.

---

### Q2 & Q3: What is the status of each of the 13 paper experiments? Which are completed/frozen, improved, or pending?

| Experiment | Description | Status | Baseline Metric | Improved Metric | Paper Metric | Outcome / Notes |
|---|---|---|---|---|---|---|
| **Figure 2** | Eigenfaces PCA Low-Dim Classification | **Frozen** | **88.67%** Acc | N/A | ~88.0% Acc | Exceeds paper accuracy |
| **Figure 4** | 3D Separable ConvNet Speedup | **Pending Approval** | Flawed (0.84s vs 6.02s)| Script Authored | ~100x Speedup | Corrected benchmark prepared |
| **Figure 6** | Synthetic Noise Robustness | **Frozen** | Replicated | N/A | Robust over SVD | Matches paper curve |
| **Figure 7** | Robust PCA Covariance Recovery | **Frozen** | Replicated | N/A | Lower error vs PCA | Matches paper curve |
| **Figure 9** | ICVL Hyperspectral Denoising | **Frozen** | **43.43 dB** | N/A | 43.40 dB | Exceeds paper target |
| **Figure 10** | Yale Face B Denoising (30% & 60% S&P) | **Approved** | 28.52 / 28.18 dB | **30.86 / 30.46 dB**| 28.9 / 29.7 dB | Exceeds paper targets |
| **Figure 11** | Video Compressive Sensing (64 frames) | **Audited** | 21.05 dB | **25.32 dB** (+1.83 TV)| ~27.5 dB (HQ) | Limited by author's cat_video.mat |
| **Figure 12** | 3D PET/CT Radon Tomography | **Audited** | 28.84 dB | **30.55 dB** (0.8302)| ~30.5 dB | Exceeds 2D TV & Algorithmic |
| **Figure 13(a)**| HSI Rank Invariance Sweep (R=1..30) | **Frozen** | Plateau $\ge 20$ | N/A | Plateau $\ge 20$ | Matches paper invariance |
| **Figure 13(b)**| Yale Face B Rank Sweep (R=10..4000) | **Audited** | 23.22–23.38 dB | **28.50–28.97 dB** | ~26.0 dB | Flat plateau; +2.78 dB over ALS |
| **Table 1** | Synthetic Grid Low-Rank Decomposition | **Frozen** | **0.00 $\pm$ 0.00** | N/A | 0.00 $\pm$ 0.00 | Matches paper mean & std |
| **Table 2** | NMF on MNIST & CIFAR-10 | **Frozen** | **7.9–9.5 dB** | N/A | 7.3–8.8 dB | Aligns with paper bounds |
| **Table 3** | Speech Commands Spectrogram NMF | **Frozen** | Matches paper | N/A | Matches paper | Replicates 10 audio classes |

---

### Q4: Where are all checkpoints, metrics, and plots stored?
1. **Checkpoints**:
   - Total of **31 checkpoints (`.pt`)** occupying **440.2 MB**.
   - **Improved High-Performance Baselines**: stored in `paper_reproduction/cnn_baseline/*/improved_run/checkpoint/` (Yale 30%, Yale 60%, Video CS, CT Recon, Yale Rank 4000).
   - **Verified Frozen Baselines**: stored in `RESULTS-BASELINE/figure9/checkpoint/` and `RESULTS-BASELINE/figure12/checkpoint/`.
   - **Historical Extension Weights**: stored in `results/baseline/`, `results/adaptive/`, and `results/sparsity/`.
2. **Metrics**:
   - **36 JSON files** (`metrics.json`, `figure12_improved_metrics.json`, `synthetic_suite_metrics.json`, etc.).
   - **21 CSV trace logs** tracking epoch-by-epoch loss, PSNR, SSIM, and runtime.
3. **Plots & Visual Panels**:
   - **107 high-resolution PNG images** stored in `paper_reproduction/cnn_baseline/**/plots/`, `images/`, and centralized gallery `PAPER_RESULTS/`.

---

### Q5: What datasets do we have, where are they, and are any missing?
- **No datasets are missing.** All 9 benchmark datasets required for the 13 experiments are present and physically verified in `data/`:
  - `data/weizzman.mat` (5.37 MB) — Figure 2
  - Synthetic volumes generated in-memory — Figures 3, 4, 6, 7; Table 1
  - `data/MNIST/` (11.5 MB) — Table 2
  - `data/cifar-10-batches-py/` & `cifar-10-python.tar.gz` (170.5 MB) — Table 2
  - `data/icvl.mat` (11.36 MB) — Figures 9, 13(a)
  - `data/yale_b.mat` (4.93 MB) & `data/CroppedYale/` (20.1 MB) — Figures 10, 13(b)
  - `data/cat_video.mat` (0.78 MB) — Figure 11
  - `data/pet.mat` (1.29 MB) — Figure 12
  - `data/SpeechCommands/` & `speech_commands_v0.02.tar.gz` (2,428.9 MB) — Table 3

---

### Q6: What are the major code duplications and technical debts in the codebase?
1. **Duplicate Root Runners**: 9 `run_figure9*.py` scripts in the root directory are exact duplicates of files in `previous_neurotensor_experiments/`.
2. **Duplicate Neural Modules**: `attention_spectral.py` and `spectral_transformer.py` in `modules/` were copied directly into extension folders.
3. **Backup Files**: `demo_backup.py` and `demo_backend_backup.py` linger in root.
4. **Fragile `sys.path.append('modules')`**: Ubiquitous across runner scripts instead of a standard `pip install -e .` package layout.
5. **Hardcoded Device Calls**: Scattered `cuda:0` and `.cuda()` calls across scripts rather than centralized config-driven device management.
6. **Duplicated Outer Training Loops**: Boilerplate closure, Adam optimizer, loss logging, and PSNR evaluation copied across ~15 scripts.
7. **Scattered Output Locations**: Multiple concurrent result folders (`RESULTS-BASELINE/`, `PAPER_RESULTS/`, `results/`, `paper_reproduction/`).

---

### Q7: Which files are safe to delete or archive immediately?
- **Safe to Delete Immediately**:
  - 9 duplicate root files: `run_figure9.py`, `run_figure9_adaptive.py`, `run_figure9_sparsity.py`, `run_figure9_completion.py`, `run_figure9_robustness.py`, `run_figure9_ensemble.py`, `run_figure9_interpretability.py`, `run_figure9_hybrid.py`, `run_figure9_spectral_transformer.py`.
  - 2 duplicate module files: `previous_neurotensor_experiments/*/modules/*.py`.
  - 2 backup demo files: `demo_backup.py`, `demo_backend_backup.py`.
  - Temporary smoke test checkpoints: `results_ensemble_smoke/` (231.3 MB) and empty result directories.
  - Raw text logs: `results_lambda_*.txt` in root.
- **Move to Reference Archive (Do Not Delete)**:
  - 9 research extensions in `previous_neurotensor_experiments/` $\to$ `experiments/historical_extensions/`.
  - Legacy root runners (`run_figure2.py`, `run_figure6.py`, `run_figure7.py`, `run_table1.py`) $\to$ `experiments/legacy_baselines/`.

---

### Q8: How should the core modules (`modules/`) be restructured into a clean package?
Restructure into a standard Python package under `src/deeptensor/`:
- `src/deeptensor/models/`: `deeptensor.py`, `deep_decoder.py`, `deep_prior.py`
- `src/deeptensor/models/backbones/`: `skip.py`, `unet.py`, `resnet.py`, `dcgan.py`
- `src/deeptensor/models/attention/`: `spatial_spectral.py`, `spectral_transformer.py`
- `src/deeptensor/physics/`: `radon.py` (CT), `video_cs.py` (Video CS)
- `src/deeptensor/losses/`: `objectives.py` ($L_1$, $L_2$, TVNorm)
- `src/deeptensor/baselines/`: `parafac_als.py` (CUDA ALS), `svd.py`, `tucker.py`
- `src/deeptensor/training/`: `trainer.py`, `schedulers/`
- `src/deeptensor/utils/`: `metrics.py` (PSNR, SSIM, SAM), `visualization.py`
- Installable via root `pyproject.toml` using `pip install -e .`.

---

### Q9: How should experiment scripts be organized so baseline and improved runs are clearly separated?
Organize under `experiments/reproduction/<domain>/` with dedicated subdirectories and configs:
```text
experiments/reproduction/06_yale_face_denoising/
├── configs/
│   ├── baseline.yaml                           # 500 epochs, 64 channels
│   └── improved.yaml                           # 4000 epochs, 128 channels
├── run_baseline.py                             # Standard baseline runner
├── run_improved.py                             # High-performance improved runner
└── visualize.py                               # Multi-panel reconstruction generator
```
Outputs route to:
- `results/baseline/06_yale_face_denoising/`
- `results/improved/06_yale_face_denoising/`
Each maintaining independent checkpoints, metrics JSONs, CSV traces, and visual panels.

---

### Q10: What is the recommended roadmap to execute the reorganization without losing any research progress?
1. **Phase 0 — Integrity Checksum**: Compute and write a master SHA-256 manifest of all 31 `.pt` checkpoints, metrics JSONs, and `.mat` datasets. Commit/tag current git state.
2. **Phase 1 — Package Scaffolding**: Create `DeepTensor-Plus-Plus/pyproject.toml` and port `modules/` into `src/deeptensor/` with clean relative imports. Install in editable mode (`pip install -e .`).
3. **Phase 2 — Experiment Porting**: Relocate `paper_reproduction/cnn_baseline/` into `experiments/reproduction/`, updating imports to use `from deeptensor.models import DeepTensor`.
4. **Phase 3 — Results Migration & Verification**: Relocate `RESULTS-BASELINE/` $\to$ `results/baseline/` and `paper_reproduction/**/improved_run/` $\to$ `results/improved/`. Verify SHA-256 hashes against Phase 0 manifest.
5. **Phase 4 — Redundancy Pruning**: Delete confirmed duplicate root files and obsolete backups.
6. **Phase 5 — Automated Smoke Verification**: Run 10-step verification test passes on all 13 experiment targets to ensure end-to-end functionality.

---

## 2. Inventory of Audit Deliverables in `codebase_audit/`

| Document Name | File Path | Focus & Purpose |
|---|---|---|
| **1. Structure Map** | [`CURRENT_CODEBASE_STRUCTURE.md`](file:///e:/Rakhal_Tensor/codebase_audit/CURRENT_CODEBASE_STRUCTURE.md) | Full 4-level annotated directory tree, folder responsibilities, and configuration locations. |
| **2. Python Inventory** | [`PYTHON_FILE_INVENTORY.md`](file:///e:/Rakhal_Tensor/codebase_audit/PYTHON_FILE_INVENTORY.md) | Exhaustive 75-file catalog (lines, classes, functions, dependencies, callers, categories, duplicates). |
| **3. Experiment Mapping** | [`RESEARCH_EXPERIMENT_MAPPING.md`](file:///e:/Rakhal_Tensor/codebase_audit/RESEARCH_EXPERIMENT_MAPPING.md) | Detailed mapping across all 13 paper experiments (scripts, configs, datasets, checkpoints, metrics, plots). |
| **4. Data & Results** | [`DATA_AND_RESULTS_INVENTORY.md`](file:///e:/Rakhal_Tensor/codebase_audit/DATA_AND_RESULTS_INVENTORY.md) | Complete catalog of 9 datasets, 31 checkpoints, 36 JSON metrics, 21 CSV traces, and 107 plots. |
| **5. Dependency Map** | [`DEPENDENCY_AND_IMPORT_MAP.md`](file:///e:/Rakhal_Tensor/codebase_audit/DEPENDENCY_AND_IMPORT_MAP.md) | Mermaid dependency diagrams, module linkage matrix, external imports, and technical debt. |
| **6. Migration Map** | [`PROPOSED_ARCHITECTURE_AND_MIGRATION_MAP.md`](file:///e:/Rakhal_Tensor/codebase_audit/PROPOSED_ARCHITECTURE_AND_MIGRATION_MAP.md) | Target modular structure, file-by-file `CURRENT -> PROPOSED` map, and phased migration plan. |
| **7. Executive Summary** | [`CODEBASE_AUDIT_SUMMARY.md`](file:///e:/Rakhal_Tensor/codebase_audit/CODEBASE_AUDIT_SUMMARY.md) | This master document synthesizing all findings and answering the 10 audit questions. |

---

## 3. Immediate Recommendation & Next Steps

> **IMPORTANT**: In accordance with user instructions, this audit has been executed strictly in **read-only mode**. No project files, checkpoints, datasets, or metrics have been modified, deleted, moved, or overwritten.

The codebase is in an excellent, mature state to undergo reorganization into `DeepTensor-Plus-Plus/`. All required experimental results, high-performance checkpoints, and datasets are confirmed present and intact on disk.

**Awaiting user instructions:**
- Review the 7 audit documents in `codebase_audit/`.
- Provide approval or modifications for the proposed `DeepTensor-Plus-Plus/` architecture and migration plan.
