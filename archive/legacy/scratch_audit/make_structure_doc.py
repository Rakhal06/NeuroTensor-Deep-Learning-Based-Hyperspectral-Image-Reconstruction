import os
import json

ROOT = 'e:/Rakhal_Tensor'

with open('scratch/comprehensive_audit_data.json', encoding='utf-8') as f:
    data = json.load(f)

doc = """# DeepTensor++ — Current Codebase Structure & Directory Map

> **Audit Date:** November 6, 2026  
> **Workspace Root:** `e:\\Rakhal_Tensor`  
> **Audit Status:** Read-Only Verification Complete  
> **Scope:** Comprehensive survey of all directories, subdirectories, configurations, datasets, checkpoints, logs, and artifacts.

---

## 1. Repository Overview & High-Level Architecture

The **DeepTensor++** workspace is an extensive scientific research repository reproducing and extending the foundational paper:
*DeepTensor: Deep Matrix and Tensor Factorization* (NeurIPS / IEEE).

Currently, the codebase contains:
- **75 Python scripts** (72 repository scripts + 3 audit/scratch helpers) totaling ~18,500 lines of code.
- **9 Core Application Domains** covering 13 distinct experimental targets (Figures 2, 4, 6, 7, 9, 10, 11, 12, 13a, 13b, and Tables 1, 2, 3).
- **31 PyTorch checkpoints (`.pt`)** representing baseline models, improved runs, and rank sweeps (~440 MB total checkpoint footprint).
- **5 Physical `.mat` datasets** plus 4 standard vision/audio benchmarks (CroppedYale, MNIST, CIFAR-10, SpeechCommands v0.02) totaling ~5.83 GB.
- **107 High-resolution figures and panels (`.png`)** documenting convergence, reconstructions, and comparisons.
- **71 Detailed research reports and markdown summaries (`.md`)**.

The workspace is organized into several distinct operational areas:
1. **Core Mathematical & Neural Architecture Modules (`modules/`)**: PyTorch layers, Deep Image Prior (DIP) backbones, Deep Decoders, linear inverse operators, regularizers, and spatial/spectral transformers.
2. **Standardized Paper Reproduction (`paper_reproduction/cnn_baseline/`)**: The organized Phase 1 reproduction benchmark suites containing modular experiment scripts, configs, baseline outputs, and improved runs.
3. **Historical Research Extensions (`previous_neurotensor_experiments/`)**: Nine sequential extensions developed on the ICVL hyperspectral task (Adaptive Rank, Group Sparsity, Matrix Completion, Ensemble Uncertainty, Spatial/Spectral Transformers).
4. **Legacy Root Runners & Scratch Utilities (`./`)**: 28 root-level scripts consisting of original baseline runners, interactive Streamlit demo applications, ReportLab PDF generators, and historical one-off scripts.
5. **Data Assets (`data/`)**: Raw matrices, tensors, compressed archives, and extracted vision/audio benchmarks.
6. **Results & Baseline Registries (`RESULTS-BASELINE/`, `PAPER_RESULTS/`, `results/`, etc.)**: Serialized checkpoints, metrics JSONs, CSV traces, and visual panels from prior benchmark runs.

---

## 2. Complete Annotated Directory Tree (Level 1–4)

```text
e:\Rakhal_Tensor/
│
├── .venv/                                      # Virtual environment (Python 3.11, PyTorch CUDA)
├── codebase_audit/                             # Comprehensive audit deliverables & migration maps
├── data/                                       # Raw datasets, archives, and extracted benchmarks
│   ├── cat_video.mat                           # Video CS dataset (64 frames, 8 snapshots, 819 KB)
│   ├── cifar-10-batches-py/                    # Extracted CIFAR-10 python batches (170 MB)
│   ├── cifar-10-python.tar.gz                  # Compressed CIFAR-10 archive
│   ├── CroppedYale/                            # Extracted Extended Yale Face B (38 subjects)
│   ├── CroppedYale.zip                         # Compressed Yale Face B archive (20 MB)
│   ├── icvl.mat                                # ICVL Hyperspectral Image (31 channels, 11.9 MB)
│   ├── MNIST/                                  # PyTorch MNIST raw & processed IDX data
│   ├── pet.mat                                 # 3D PET/CT volume (128x128x150 slices, 1.35 MB)
│   ├── README.txt                              # Dataset documentation and provenance notes
│   ├── SpeechCommands/                         # Extracted Google Speech Commands v0.02 (10 classes)
│   ├── speech_commands_v0.02.tar.gz            # Compressed Speech Commands archive (2.42 GB)
│   ├── weizzman.mat                            # Weizmann Face / Eigenfaces dataset (5.63 MB)
│   └── yale_b.mat                              # Yale Face B 3-way tensor (192x168x64, 5.16 MB)
│
├── documentation/                              # Theoretical derivations, roadmaps, and guidelines
│   ├── architecture_designs/                   # High-level design specs for extensions
│   ├── mathematical_derivations/               # Derivations for tensor ALS, TV, and DIP priors
│   └── protocol_specifications/                # Paper protocol sheets and experimental guidelines
│
├── modules/                                    # Reusable neural & mathematical building blocks
│   ├── __init__.py                             # Subpackage marker
│   ├── attention_spectral.py                   # Hybrid spectral self-attention layer (207 lines)
│   ├── cosine_annealing_with_warmup.py         # Custom LR scheduler with warmup (92 lines)
│   ├── deep_decoder.py                         # Non-convolutional Deep Decoder architecture (257 lines)
│   ├── deep_prior.py                           # Deep Image Prior generator & utility methods (266 lines)
│   ├── lin_inverse.py                          # Forward & adjoint operators: Radon & Video CS (97 lines)
│   ├── losses.py                               # L1, L2, and anisotropic Total Variation norms (35 lines)
│   ├── models.py                               # DeepTensor wrapper & coordinate forward nets (230 lines)
│   ├── spectral.py                             # Truncated SVD and Tucker decomposition baselines (61 lines)
│   ├── spectral_transformer.py                 # Transformer with ALiBi relative positional bias (155 lines)
│   ├── utils.py                                # Evaluation metrics (PSNR, SSIM, SAM, ERGAS) & montages (431 lines)
│   └── deep_models/                            # Classic DIP backbone architectures (Ulyanov et al.)
│       ├── __init__.py                         # Model factory (`get_net`)
│       ├── common.py                           # Concat, Concat1d, Swish, Sine activations (197 lines)
│       ├── dcgan.py                            # DCGAN generator backbone (34 lines)
│       ├── downsampler.py                      # Anti-aliased downsampling layer (168 lines)
│       ├── resnet.py                           # Residual sequential network backbone (96 lines)
│       ├── skip.py                             # Skip-connected DIP architecture (197 lines)
│       ├── texture_nets.py                     # Multi-scale texture generator (79 lines)
│       └── unet.py                             # Multi-scale U-Net architecture (192 lines)
│
├── paper_reproduction/                         # Standardized Phase 1 paper reproduction suites
│   ├── cnn_baseline/                           # CNN & DIP baseline benchmarks across applications
│   │   ├── 01_synthetic_decomposition/         # Table 1 (Grid) & Figure 4 (Separable Speedup)
│   │   │   ├── benchmark_figure4_separable_speedup.py
│   │   │   ├── benchmark_figure4_separable_speedup_improved.py
│   │   │   ├── run_synthetic_suite.py
│   │   │   ├── config.yaml
│   │   │   ├── figure4_benchmark_metrics.json
│   │   │   ├── synthetic_suite_metrics.json
│   │   │   └── plots/
│   │   ├── 03_nmf/                             # Table 2: NMF on MNIST & CIFAR-10
│   │   │   ├── run_table2_nmf.py
│   │   │   ├── config.yaml
│   │   │   ├── metrics.json
│   │   │   └── metrics.txt
│   │   ├── 05_hyperspectral_denoising/         # Figure 9 & Figure 13(a): ICVL Rank Sweep
│   │   │   ├── run_figure13_rank_sweep.py
│   │   │   ├── config.yaml
│   │   │   ├── figure13a_metrics.json
│   │   │   ├── checkpoint/                     # Saved baseline checkpoint (11.65 MB)
│   │   │   └── plots/
│   │   ├── 06_yale_face_tensor_denoising/      # Figure 10 (Denoising) & Figure 13(b) (Rank Sweep)
│   │   │   ├── run_figure10_yale_tensor.py
│   │   │   ├── run_figure10_yale_tensor_improved.py
│   │   │   ├── run_figure13b_yale_rank_sweep.py
│   │   │   ├── run_figure13b_yale_rank_sweep_improved.py
│   │   │   ├── generate_improved_panel.py
│   │   │   ├── config.yaml
│   │   │   ├── FIGURE10_IMPROVEMENT_REPORT.md
│   │   │   ├── FIGURE13B_IMPROVEMENT_REPORT.md
│   │   │   ├── original_baseline/              # 500-epoch baseline metrics, checkpoints, plots
│   │   │   ├── improved_run/                   # Approved 4000-epoch runs (30.86 dB & 30.46 dB)
│   │   │   ├── improved_run_fig13b/            # Approved CUDA ALS & DeepTensor sweep checkpoints
│   │   │   ├── checkpoint/                     # Shared checkpoints
│   │   │   ├── images/                         # Side-by-side reconstruction panels
│   │   │   └── plots/                          # Convergence & rank sweep curves
│   │   ├── 07_video_compressive_sensing/       # Figure 11: Video Compressive Sensing (8 snapshots)
│   │   │   ├── run_figure11.py
│   │   │   ├── run_figure11_improved.py
│   │   │   ├── config.yaml
│   │   │   ├── FIGURE11_IMPROVEMENT_REPORT.md
│   │   │   ├── original_baseline/              # Original baseline outputs
│   │   │   ├── improved_run/                   # Approved 10,000-epoch run (Peak 25.32 dB)
│   │   │   │   ├── checkpoint/                 # Saved model weights (15.98 MB)
│   │   │   │   ├── images/                     # 9-slice montage & frame reconstructions
│   │   │   │   └── plots/                      # PSNR & loss convergence plots
│   │   │   └── checkpoint/
│   │   ├── 08_ct_reconstruction/               # Figure 12: 3D PET/CT Radon Tomography
│   │   │   ├── run_figure12_improved.py
│   │   │   ├── configuration.json
│   │   │   ├── FIGURE12_IMPROVEMENT_REPORT.md
│   │   │   ├── improved_run/                   # Approved run (30.55 dB PSNR / 0.8302 SSIM)
│   │   │   │   ├── checkpoint/                 # Saved checkpoint (15.24 MB)
│   │   │   │   ├── images/                     # Axial, coronal, sagittal comparison panels
│   │   │   │   └── plots/                      # PSNR/Loss progression curves
│   │   │   └── checkpoint/
│   │   └── 09_speech_time_frequency/           # Table 3: Audio Spectrogram NMF & Semi-NMF
│   │       ├── run_table3_speech_nmf.py
│   │       ├── config.yaml
│   │       ├── metrics.json
│   │       ├── metrics.txt
│   │       └── images/                         # Time-frequency factor panels
│   └── consolidated_results/                   # Aggregated baseline reporting
│       ├── generate_consolidated_report.py
│       └── CONSOLIDATED_CNN_BASELINE_REPORT.md
│
├── previous_neurotensor_experiments/           # Historical ICVL Extensions 1 through 9
│   ├── 01_icvl_cnn_baseline/                   # Extension 1: Standard CNN Baseline
│   ├── 02_icvl_adaptive_rank/                  # Extension 2: Adaptive Rank Selection
│   ├── 03_icvl_factor_group_sparsity/          # Extension 3: Group Sparsity Regularization
│   ├── 04_icvl_tensor_completion/              # Extension 4: Missing Slice Completion
│   ├── 05_icvl_noise_robustness/               # Extension 5: Non-Gaussian Noise Sensitivity
│   ├── 06_icvl_ensemble_uncertainty/           # Extension 6: Deep Ensemble Epistemic Uncertainty
│   ├── 07_icvl_factor_interpretability/        # Extension 7: Factor Orthogonality & Unmixing
│   ├── 08_icvl_hybrid_attention/               # Extension 8: Spatial-Spectral Self-Attention
│   │   ├── modules/attention_spectral.py       # (Duplicate of modules/attention_spectral.py)
│   │   └── run_figure9_hybrid.py
│   └── 09_icvl_spectral_transformer/           # Extension 9: Transformer with ALiBi
│       ├── modules/spectral_transformer.py     # (Duplicate of modules/spectral_transformer.py)
│       └── run_figure9_spectral_transformer.py
│
├── RESULTS-BASELINE/                           # Original baseline run results (Figures 2, 6, 7, 9, 12)
│   ├── final_report.txt                        # Master text report of baseline run
│   ├── figure2/                                # Eigenfaces classification results, plots, metrics
│   ├── figure6/                                # Synthetic noise robustness results
│   ├── figure7/                                # Robust PCA error curves
│   ├── figure9/                                # ICVL HSI denoising checkpoint (11.65 MB), plots
│   └── figure12/                               # Legacy PET/CT run checkpoint (15.23 MB)
│
├── PAPER_RESULTS/                              # Central visual plot archive (107 PNG figures)
├── results/                                    # Historical validation checkpoints & logs (254.8 MB)
├── results_ensemble/                           # Deep ensemble experiment checkpoints (99.4 MB)
├── results_ensemble_smoke/                     # Ensemble smoke test checkpoints (231.3 MB)
├── results_sparsity*/                          # Group sparsity hyperparameter sweep outputs
├── results_interpretability*/                  # Factor interpretability loadings & figures
├── scratch/                                    # AST scanners, data builders, and audit scripts
│
└── [Root Python Scripts & Legacy Files]        # 28 Root scripts (runners, demos, utilities)
    ├── run_figure2.py, run_figure6.py, run_figure7.py, run_figure9.py, run_table1.py
    ├── run_figure11.py, run_figure11_TV.py, run_figure12.py, run_figure12_TV.py
    ├── run_figure9_adaptive.py, run_figure9_sparsity.py, run_figure9_completion.py ...
    ├── demo.py, demo_backend.py, demo_backup.py, demo_backend_backup.py
    ├── generate_pdf.py, matrix_decomp.py, modify_robustness.py, scratch_calibrate.py
    └── results_lambda_*.txt                    # Large raw convergence text logs (~10 MB)
```

---

## 3. Purpose & Responsibilities of Major Folders

### 3.1 `modules/` & `modules/deep_models/`
- **Purpose**: Holds the core mathematical implementations, neural network definitions, and algorithmic operators.
- **Key Responsibilities**:
  - `models.py`: Encapsulates `DeepTensor`, `SimpleForward2D`, and coordinate generation for factor reconstruction.
  - `deep_prior.py`: Provides the foundational Deep Image Prior (DIP) skip-network builders (`skip()`, `get_net()`).
  - `deep_decoder.py`: Implements pixel-wise convolutional decoders for compressive representation without spatial convolutions.
  - `lin_inverse.py`: Houses forward physics operators (`radon`, `video2codedvideo`) and adjoint transformations.
  - `losses.py`: Defines numerical objectives: $L_1$, $L_2$, and 2D/3D anisotropic Total Variation (`TVNorm`).
  - `utils.py`: Standardized evaluation pipeline computing Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM), Spectral Angle Mapper (SAM), and montage visualizations.
  - `deep_models/`: Legacy library of convolutional generators (U-Net, ResNet, DCGAN) adapted from Ulyanov et al.

### 3.2 `paper_reproduction/cnn_baseline/`
- **Purpose**: Serves as the primary standardized Phase 1 reproduction benchmark across all nine paper applications.
- **Key Responsibilities**:
  - Contains modular subfolders (`01_synthetic_decomposition` to `09_speech_time_frequency`).
  - Stores self-contained configuration files (`config.yaml`), runner scripts, and segregated `original_baseline/` vs `improved_run/` subfolders.
  - Stores all newly audited and approved high-performance runs (Figures 10, 11, 12, 13b).

### 3.3 `previous_neurotensor_experiments/`
- **Purpose**: Archives the sequential research extensions (Extensions 1 to 9) developed on top of the ICVL hyperspectral denoising problem.
- **Key Responsibilities**:
  - Houses experimental algorithms: adaptive rank regularization, group sparsity, tensor completion, ensemble epistemic uncertainty, factor orthogonality, and attention/transformer backbones.
  - Contains standalone subfolders that mirror root scripts.

### 3.4 `RESULTS-BASELINE/` & `PAPER_RESULTS/`
- **Purpose**: Persistent historical registries for the verified and frozen original baseline experiments.
- **Key Responsibilities**:
  - `RESULTS-BASELINE/` preserves the original checkpoints, logs, and numerical results for Figures 2, 6, 7, 9, and 12.
  - `PAPER_RESULTS/` acts as an image gallery consolidating publication-ready PNG figures.

### 3.5 `data/`
- **Purpose**: Centralized storage for all experimental tensors, matrices, audio clips, and image datasets.
- **Key Responsibilities**:
  - Preserves `.mat` tensor files (`icvl.mat`, `pet.mat`, `yale_b.mat`, `cat_video.mat`, `weizzman.mat`).
  - Stores extracted benchmark directories (`CroppedYale/`, `MNIST/`, `cifar-10-batches-py/`, `SpeechCommands/`).
  - Total disk footprint: 5.83 GB (dominated by SpeechCommands at 2.42 GB).

### 3.6 Root Directory (`.`)
- **Purpose**: Historical workspace root containing top-level runner scripts, interactive demos, and diagnostic logs.
- **Key Responsibilities**:
  - Currently contains 28 Python files, many of which are historical entry points (`run_figure*.py`) or exact duplicates of files in `previous_neurotensor_experiments/`.
  - Holds interactive Streamlit demo scripts (`demo.py`, `demo_backend.py`) and ReportLab PDF generators (`generate_pdf.py`).
  - Holds raw text execution logs (`results_lambda_*.txt`, ~10 MB).

---

## 4. Configuration & Dependency Files

| File Path | Format | Scope / Responsibility |
|---|---|---|
| `paper_reproduction/cnn_baseline/01_synthetic_decomposition/config.yaml` | YAML | Synthetic tensor grid & separable speedup parameters |
| `paper_reproduction/cnn_baseline/03_nmf/config.yaml` | YAML | MNIST & CIFAR-10 NMF ranks, iterations, seeds |
| `paper_reproduction/cnn_baseline/05_hyperspectral_denoising/config.yaml` | YAML | ICVL HSI denoising & rank-sweep hyperparameters |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/config.yaml` | YAML | Yale Face B 30%/60% noise & Figure 13(b) sweep setup |
| `paper_reproduction/cnn_baseline/07_video_compressive_sensing/config.yaml` | YAML | Video CS snapshots, 64-frame tensor, DIP architecture |
| `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/configuration.json` | JSON | 40-angle Radon CT setup, Poisson noise ($\\\\tau=100$), cosine LR |
| `paper_reproduction/cnn_baseline/09_speech_time_frequency/config.yaml` | YAML | Speech Commands STFT, NMF & Semi-NMF parameters |
| `RESULTS-BASELINE/figure2/configuration.json` | JSON | Weizmann Eigenfaces PCA baseline setup |
| `RESULTS-BASELINE/figure6/configuration.json` | JSON | Synthetic noise distributions (Gaussian, Laplace, S&P) |
| `RESULTS-BASELINE/figure7/configuration.json` | JSON | Robust PCA sample size vs SNR configuration |
| `RESULTS-BASELINE/figure9/configuration.json` | JSON | ICVL Rank-20 DIP baseline parameters |

---

## 5. Dataset Locations & Storage Structure

| Dataset Name | Format | Storage Path | Dimensions / Samples | Disk Size | Verified Status |
|---|---|---|---|---|---|
| **Weizmann Faces** | `.mat` | `data/weizzman.mat` | $(100 \\times 100 \\times 100)$ | 5.37 MB | Verified (Figure 2) |
| **Synthetic Volumes** | Generated | In-Memory (NumPy/Torch) | $(128 \\times 128 \\times 150)$ | Transient | Verified (Figs 3, 4, 6, 7; Tab 1) |
| **MNIST** | IDX/PyTorch | `data/MNIST/` | $28 \\times 28$ digits (60k train) | 11.5 MB | Verified (Table 2) |
| **CIFAR-10** | Python batch | `data/cifar-10-batches-py/` | $32 \\times 32 \\times 3$ (50k train) | 170.5 MB | Verified (Table 2) |
| **ICVL Hyperspectral** | `.mat` | `data/icvl.mat` | $31 \\times 1024 \\times 1024$ (Crop $512^2$) | 11.36 MB | Verified (Figs 9, 13a) |
| **Extended Yale Face B** | `.mat` | `data/yale_b.mat` | $192 \\times 168 \\times 64$ (3-way) | 4.93 MB | Verified (Figs 10, 13b) |
| **CroppedYale Source** | PGM / Zip | `data/CroppedYale/` | 38 subjects, 64 illumination conditions | 20.1 MB | Verified raw archive |
| **Cat Video** | `.mat` | `data/cat_video.mat` | $64 \\times 128 \\times 128$ (64 frames) | 0.78 MB | Verified (Figure 11) |
| **3D PET/CT Head** | `.mat` | `data/pet.mat` | $128 \\times 128 \\times 150$ voxels | 1.29 MB | Verified (Figure 12) |
| **Speech Commands v2** | WAV / Archive| `data/SpeechCommands/` | 105,829 audio clips (16 kHz, 1 sec) | 2,428 MB | Verified (Table 3) |

---

## 6. Checkpoint Storage & Organization

The repository contains **31 PyTorch checkpoints (`.pt`)** totaling **440.2 MB**.

### Key Checkpoints:
1. **Improved High-Performance Baselines (Permanent Retention)**:
   - `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run/checkpoint/deeptensor_noise_30_improved_checkpoint.pt` (21.16 MB)
   - `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run/checkpoint/deeptensor_noise_60_improved_checkpoint.pt` (21.16 MB)
   - `paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run/checkpoint/figure11_deeptensor_improved_checkpoint.pt` (15.98 MB)
   - `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/checkpoint/figure12_deeptensor_improved_checkpoint.pt` (15.24 MB)
   - `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_4000_checkpoint.pt` (13.98 MB)
2. **Verified Frozen Baselines (Permanent Retention)**:
   - `RESULTS-BASELINE/figure9/checkpoint/figure9_deeptensor_checkpoint.pt` (11.65 MB)
   - `RESULTS-BASELINE/figure12/checkpoint/figure12_deeptensor_checkpoint.pt` (15.23 MB)
3. **Historical / Extension Checkpoints**:
   - `results/baseline/checkpoint/checkpoint.pt` (42.15 MB)
   - `results/adaptive/checkpoint/checkpoint.pt` (42.16 MB)
   - `results/sparsity/checkpoint/checkpoint_l0.001.pt` (13.86 MB)

---

## 7. Results, Metrics, Logs, and Visualization Folders

- **Metrics Files**:
  - **36 JSON files** (`metrics.json`, `figure12_improved_metrics.json`, `figure13b_improved_metrics.json`, `synthetic_suite_metrics.json`).
  - **21 CSV trace files** (recording per-epoch loss, PSNR, SSIM, and time progression).
  - **Text Reports**: Detailed summary logs in `RESULTS-BASELINE/` and per-experiment markdown audits (`FIGURE10_IMPROVEMENT_REPORT.md`, `FIGURE11_IMPROVEMENT_REPORT.md`, `FIGURE12_IMPROVEMENT_REPORT.md`, `FIGURE13B_IMPROVEMENT_REPORT.md`).
- **Visual Plots & Images**:
  - **107 PNG images** across `plots/` and `images/` subdirectories.
  - Side-by-side reconstruction panels showing ground truth, corrupted measurements, baselines, and DeepTensor reconstructions.

---

## 8. Temporary Files, Caches, and Obsolete Artifacts

1. **Duplicate Python Scripts in Root (`.`)**:
   - 9 scripts (`run_figure9*.py`) are exact byte-for-byte duplicates of scripts in `previous_neurotensor_experiments/`.
2. **Obsolete Backup Files**:
   - `demo_backup.py` (317 lines)
   - `demo_backend_backup.py` (539 lines)
3. **Empty Results Directories**:
   - `results_completion/`, `results_completion_debug/`, `results_completion_sanity/`, `results_completion_smoke/`, `results_robustness/` (all 0 bytes).
4. **Heavy Smoke Test Checkpoints**:
   - `results_ensemble_smoke/` contains 231.3 MB of temporary diagnostic weights that are not part of final paper figures.
5. **Raw Text Logs in Root**:
   - Multiple `results_lambda_*.txt` files (~10 MB) created during historical lambda parameter sweeps.
"""

out_path = os.path.join(ROOT, "codebase_audit", "CURRENT_CODEBASE_STRUCTURE.md")
with open(out_path, "w", encoding="utf-8") as fp:
    fp.write(doc.strip() + "\n")

print(f"Generated {out_path} ({len(doc.splitlines())} lines)")

