# DeepTensor++ — Python Codebase Inventory

> **Audit Date:** November 6, 2026  
> **Workspace Root:** `e:\Rakhal_Tensor`  
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
| `check_attn.py` | 9 | Quick smoke test verifying attention_spectral module import | Procedural script | numpy | Standalone / Direct | Legacy Utility / Diagnostic |
| `demo.py` | 330 | Interactive Streamlit web frontend for real-time tensor factorization & denoising | Funcs: display_factors, display_metrics, display_network | matplotlib, numpy, torch | 1 scripts (e.g., demo_backup.py) | Interactive Application / Demo |
| `demo_backend.py` | 569 | Backend service class (DemoBackend) executing factorizations for Streamlit | Classes: DemoBackend; Funcs: __init__, _compute_metrics, _init_networks | deep_prior, losses, matplotlib, models, numpy | 2 scripts (e.g., demo.py) | Interactive Application / Demo |
| `demo_backend_backup.py` | 539 | Obsolete backup copy of demo_backend.py | Classes: DemoBackend; Funcs: __init__, _compute_metrics, _init_networks | deep_prior, losses, matplotlib, models, numpy | demo.py | Obsolete / Backup |
| `demo_backup.py` | 317 | Obsolete backup copy of demo.py | Funcs: display_factors, display_metrics, display_network | matplotlib, numpy, torch | Standalone / Direct | Obsolete / Backup |
| `fix_newlines.py` | 19 | One-off text utility fixing CRLF/LF line endings | Procedural script | Standard library | Standalone / Direct | Legacy Utility / Diagnostic |
| `generate_pdf.py` | 576 | ReportLab script generating formatted research summary PDF | Classes: NumberedCanvas; Funcs: __init__, build_pdf, draw_page_decorations | reportlab | Standalone / Direct | Reporting Utility |
| `matrix_decomp.py` | 156 | Legacy script testing matrix factorization baselines | Procedural script | cv2, deep_decoder, deep_prior, losses, matplotlib | Standalone / Direct | Legacy Utility / Diagnostic |
| `modify_robustness.py` | 157 | Script generating modified noise plots for Figure 6 | Procedural script | Standard library | Standalone / Direct | Legacy Utility / Diagnostic |
| `modules/attention_spectral.py` | 207 | Spatial-spectral self-attention layer for hyperspectral tensors | Classes: HybridSpectralNet, SpectralPointwiseControl, SpectralSelfAttention; Funcs: __init__, count_parameters, estimate_flops | torch | 6 scripts (e.g., run_figure9_hybrid.py) | Core Reusable (Architecture & Solvers) |
| `modules/cosine_annealing_with_warmup.py` | 92 | Cosine annealing learning rate scheduler with warmup restarts | Classes: CosineAnnealingWarmupRestarts; Funcs: __init__, get_lr, init_lr | torch | 1 scripts (e.g., utils.py) | Core Reusable (Architecture & Solvers) |
| `modules/deep_decoder.py` | 257 | Non-convolutional 1D/2D pixel-wise Deep Decoder network | Classes: ResidualBlock; Funcs: __init__, add_module, conv | torch | 17 scripts (e.g., matrix_decomp.py) | Core Reusable (Architecture & Solvers) |
| `modules/deep_models/__init__.py` | 32 | Backbone factory dispatcher (get_net) for U-Net, ResNet, DCGAN | Funcs: get_net | torch | Standalone / Direct | Core Reusable (Backbones) |
| `modules/deep_models/common.py` | 197 | Common neural building blocks, activation layers (Sine, Swish), Concat | Classes: Concat, Concat1d, GenNoise; Funcs: __init__, __len__, act | numpy, torch | 4 scripts (e.g., resnet.py) | Core Reusable (Backbones) |
| `modules/deep_models/dcgan.py` | 34 | DCGAN convolutional generator backbone | Funcs: dcgan | torch | Standalone / Direct | Core Reusable (Backbones) |
| `modules/deep_models/downsampler.py` | 168 | Anti-aliased downsampling layer with Gaussian/bilinear filters | Classes: Downsampler; Funcs: __init__, forward, get_kernel | numpy, torch | 1 scripts (e.g., common.py) | Core Reusable (Backbones) |
| `modules/deep_models/resnet.py` | 96 | Residual convolutional network generator backbone | Classes: ResNet, ResidualSequential; Funcs: __init__, eval, forward | numpy, torch | 2 scripts (e.g., __init__.py) | Core Reusable (Backbones) |
| `modules/deep_models/skip.py` | 197 | Deep Image Prior (DIP) skip architecture with multi-scale skip connections | Funcs: skip, skip1d | torch | 3 scripts (e.g., __init__.py) | Core Reusable (Backbones) |
| `modules/deep_models/texture_nets.py` | 79 | Multi-scale pyramidal texture synthesis convolutional generator | Funcs: conv, get_texture_nets | torch | 2 scripts (e.g., __init__.py) | Core Reusable (Backbones) |
| `modules/deep_models/unet.py` | 192 | Multi-scale U-Net encoder-decoder convolutional architecture | Classes: ListModule, UNet, unetConv2; Funcs: __getitem__, __init__, __iter__ | torch | 2 scripts (e.g., __init__.py) | Core Reusable (Backbones) |
| `modules/deep_prior.py` | 266 | DIP factory wrappers, noise tensor initializers, spatial coordinate generators | Classes: Downsampler; Funcs: __init__, fill_noise, forward | cv2, numpy, torch, torchvision | 40 scripts (e.g., demo_backend.py) | Core Reusable (Architecture & Solvers) |
| `modules/lin_inverse.py` | 97 | Forward measurement operators & adjoints: Radon transform and Video Coded Snapshot | Funcs: get_video_coding_frames, radon, video2codedvideo | cv2, matplotlib, numpy, scipy, torch | 7 scripts (e.g., run_figure11.py) | Core Reusable (Architecture & Solvers) |
| `modules/losses.py` | 35 | Loss functions: L1Norm, L2Norm, anisotropic 2D & 3D Total Variation (TVNorm) | Classes: L1Norm, L2Norm, TVNorm; Funcs: __call__, __init__ | torch | 40 scripts (e.g., demo_backend.py) | Core Reusable (Architecture & Solvers) |
| `modules/models.py` | 230 | DeepTensor wrapper, coordinate MLP, and multi-dimensional forward networks | Classes: SimpleForward1D, SimpleForward2D, SimpleForwardMLP; Funcs: __init__, forward | torch | 36 scripts (e.g., demo_backend.py) | Core Reusable (Architecture & Solvers) |
| `modules/spectral.py` | 61 | Classical matrix/tensor factorizers: Truncated SVD, PCA, Tucker decomposition | Funcs: lr_decompose, tucker_decompose | numpy, scipy, tensorly, torch | 32 scripts (e.g., demo_backend.py) | Core Reusable (Architecture & Solvers) |
| `modules/spectral_transformer.py` | 155 | Spectral transformer with ALiBi relative positional biases | Classes: ControlSpectralTransformer, SpectralTransformer; Funcs: __init__, count_parameters, forward | torch | 6 scripts (e.g., run_figure9_spectral_transformer.py) | Core Reusable (Architecture & Solvers) |
| `modules/utils.py` | 431 | Quantitative evaluation metrics (PSNR, SSIM, SAM, ERGAS) and visual grid utilities | Funcs: build_montage, get_1d_posencode_inp, get_2d_posencode_inp | cv2, matplotlib, numpy, scipy, torch | 47 scripts (e.g., demo_backend.py) | Core Reusable (Architecture & Solvers) |
| `paper_reproduction/cnn_baseline/01_synthetic_decomposition/benchmark_figure4_separable_speedup.py` | 200 | Figure 4 benchmark measuring runtime scaling of separable vs standard 3D CNNs | Classes: Conv3D_DIP; Funcs: __init__, forward, run_benchmark | deep_prior, matplotlib, numpy, torch, utils | CLI / Direct execution | Benchmark / Efficiency |
| `paper_reproduction/cnn_baseline/01_synthetic_decomposition/benchmark_figure4_separable_speedup_improved.py` | 307 | Corrected Figure 4 efficiency benchmark using verified 128x128x150 volume | Classes: Conv3D_DIP; Funcs: __init__, forward, generate_mri_phantom_volume | deep_prior, matplotlib, numpy, torch, utils | CLI / Direct execution | Improved Experiment Runner |
| `paper_reproduction/cnn_baseline/01_synthetic_decomposition/run_synthetic_suite.py` | 377 | Table 1 synthetic matrix & tensor low-rank grid decomposition suite | Funcs: main, run_figure3, run_figure4 | deep_decoder, deep_prior, losses, matplotlib, models | CLI / Direct execution | Baseline Experiment Runner |
| `paper_reproduction/cnn_baseline/03_nmf/run_table2_nmf.py` | 264 | Table 2 NMF and Semi-NMF factorization suite on MNIST & CIFAR-10 | Funcs: get_data, main, run_deeptensor_nmf | deep_decoder, deep_prior, losses, matplotlib, models | CLI / Direct execution | Baseline Experiment Runner |
| `paper_reproduction/cnn_baseline/05_hyperspectral_denoising/run_figure13_rank_sweep.py` | 175 | Figure 13(a) ICVL rank over-estimation robustness sweep (R=1..30) | Funcs: main, run_deeptensor_rank | deep_decoder, deep_prior, losses, matplotlib, models | CLI / Direct execution | Baseline Experiment Runner |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/generate_improved_panel.py` | 191 | Visualization generator creating multi-panel figures for Yale Face B | Funcs: main | matplotlib, numpy, scipy, skimage, utils | CLI / Direct execution | Improved Experiment Runner |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure10_yale_tensor.py` | 365 | Figure 10 Yale Face B 30% and 60% S&P denoising baseline runner (500 epochs) | Funcs: add_salt_and_pepper_noise, main, run_parafac_deeptensor | deep_decoder, deep_prior, losses, matplotlib, models | CLI / Direct execution | Baseline Experiment Runner |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure10_yale_tensor_improved.py` | 434 | Approved Figure 10 improved runner with 128 channels & 4000 epochs | Funcs: generate_salt_and_pepper_noise, main, run_parafac_deeptensor_improved | deep_prior, losses, matplotlib, numpy, scipy | CLI / Direct execution | Improved Experiment Runner |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure13b_yale_rank_sweep.py` | 179 | Figure 13(b) Yale Face B rank sweep baseline (500 epochs) | Funcs: main, run_deeptensor_parafac_rank | deep_decoder, deep_prior, losses, matplotlib, models | CLI / Direct execution | Baseline Experiment Runner |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure13b_yale_rank_sweep_improved.py` | 388 | Approved Figure 13(b) rank sweep with CUDA PARAFAC ALS baseline | Funcs: calculate_ssim, main, run_classical_parafac_als | deep_prior, losses, matplotlib, numpy, scipy | CLI / Direct execution | Improved Experiment Runner |
| `paper_reproduction/cnn_baseline/07_video_compressive_sensing/run_figure11.py` | 415 | Figure 11 Video CS reconstruction baseline runner (2000 epochs) | Funcs: calculate_ssim, main, run_deeptensor | deep_decoder, deep_prior, lin_inverse, losses, matplotlib | CLI / Direct execution | Baseline Experiment Runner |
| `paper_reproduction/cnn_baseline/07_video_compressive_sensing/run_figure11_improved.py` | 497 | Approved Figure 11 Video CS runner with 64-frame reconstruction (10,000 epochs) | Funcs: calculate_ssim, main, run_deeptensor_improved | deep_prior, lin_inverse, losses, matplotlib, numpy | CLI / Direct execution | Improved Experiment Runner |
| `paper_reproduction/cnn_baseline/08_ct_reconstruction/run_figure12_improved.py` | 463 | Approved Figure 12 3D PET/CT Radon tomography runner with Poisson noise (tau=100) | Funcs: compute_metrics, main | deep_prior, lin_inverse, losses, matplotlib, models | CLI / Direct execution | Improved Experiment Runner |
| `paper_reproduction/cnn_baseline/09_speech_time_frequency/run_table3_speech_nmf.py` | 344 | Table 3 Google Speech Commands audio spectrogram NMF & Semi-NMF runner | Funcs: get_speech_spectrograms, main, run_deeptensor_speech | deep_decoder, deep_prior, losses, matplotlib, models | CLI / Direct execution | Baseline Experiment Runner |
| `paper_reproduction/consolidated_results/generate_consolidated_report.py` | 119 | Consolidated report builder aggregating metrics across all 9 application domains | Funcs: load_json_safe, main | numpy | CLI / Direct execution | Evaluation & Reporting |
| `previous_neurotensor_experiments/01_icvl_cnn_baseline/run_figure9.py` | 339 | This script replicates the DeepTensor, SVD, and BM3D results in Figure | Funcs: compute_metrics | bm3d, deep_prior, losses, matplotlib, models | Standalone / Direct | Historical Research Extension |
| `previous_neurotensor_experiments/02_icvl_adaptive_rank/run_figure9_adaptive.py` | 481 | Research Extension 1: Adaptive Rank Selection — Plateau-Based  Based o | Procedural script | deep_prior, losses, matplotlib, models, numpy | Standalone / Direct | Historical Research Extension |
| `previous_neurotensor_experiments/03_icvl_factor_group_sparsity/run_figure9_sparsity.py` | 448 | Research Extension 2: Factor Group-Sparsity (Group Lasso) Regularizati | Funcs: compute_effective_ranks, compute_group_regularizer, parse_args | deep_prior, losses, matplotlib, models, numpy | Standalone / Direct | Historical Research Extension |
| `previous_neurotensor_experiments/04_icvl_tensor_completion/run_figure9_completion.py` | 439 | Extension 3: Missing-Data / Tensor Completion Capability for DeepTenso | Funcs: compute_subset_metrics, generate_completion_masks, parse_args | deep_prior, losses, matplotlib, models, numpy | Standalone / Direct | Historical Research Extension |
| `previous_neurotensor_experiments/05_icvl_noise_robustness/run_figure9_robustness.py` | 269 | This script replicates the DeepTensor results in Figure 6. The accurac | Procedural script | cv2, deep_decoder, deep_prior, losses, matplotlib | Standalone / Direct | Historical Research Extension |
| `previous_neurotensor_experiments/06_icvl_ensemble_uncertainty/run_figure9_ensemble.py` | 365 | Extension 5: Ensemble Disagreement as a Proxy for Reconstruction Uncer | Funcs: compute_edge_magnitude, main, set_seed | deep_decoder, deep_prior, losses, models, numpy | Standalone / Direct | Historical Research Extension |
| `previous_neurotensor_experiments/07_icvl_factor_interpretability/run_figure9_interpretability.py` | 400 | Extension 7: Post-hoc Factor Analysis and Descriptive Visualization of | Funcs: run_experiment | deep_decoder, deep_prior, losses, matplotlib, models | Standalone / Direct | Historical Research Extension |
| `previous_neurotensor_experiments/08_icvl_hybrid_attention/modules/attention_spectral.py` | 207 | Script: attention_spectral.py | Classes: HybridSpectralNet, SpectralPointwiseControl, SpectralSelfAttention; Funcs: __init__, count_parameters, estimate_flops | torch | 6 scripts (e.g., run_figure9_hybrid.py) | Duplicate Module |
| `previous_neurotensor_experiments/08_icvl_hybrid_attention/run_figure9_hybrid.py` | 241 | Script: run_figure9_hybrid.py | Funcs: main, set_seed | attention_spectral, deep_prior, losses, numpy, scipy | Standalone / Direct | Historical Research Extension |
| `previous_neurotensor_experiments/09_icvl_spectral_transformer/modules/spectral_transformer.py` | 155 | Script: spectral_transformer.py | Classes: ControlSpectralTransformer, SpectralTransformer; Funcs: __init__, count_parameters, forward | torch | 6 scripts (e.g., run_figure9_spectral_transformer.py) | Duplicate Module |
| `previous_neurotensor_experiments/09_icvl_spectral_transformer/run_figure9_spectral_transformer.py` | 258 | Script: run_figure9_spectral_transformer.py | Funcs: main, set_seed | deep_prior, losses, numpy, scipy, skimage | Standalone / Direct | Historical Research Extension |
| `run_figure11.py` | 205 | Legacy Figure 11 Video CS runner | Procedural script | cv2, deep_decoder, deep_prior, lin_inverse, losses | CLI / Direct execution | Legacy Root Runner |
| `run_figure11_TV.py` | 136 | Legacy Figure 11 2D/3D Total Variation baseline runner | Procedural script | cv2, lin_inverse, losses, matplotlib, numpy | CLI / Direct execution | Legacy Root Runner |
| `run_figure12.py` | 371 | Legacy Figure 12 3D PET/CT reconstruction runner | Funcs: compute_metrics | deep_prior, lin_inverse, losses, matplotlib, models | CLI / Direct execution | Legacy Root Runner |
| `run_figure12_TV.py` | 131 | Legacy Figure 12 3D Total Variation reconstruction runner | Procedural script | cv2, lin_inverse, losses, matplotlib, numpy | CLI / Direct execution | Legacy Root Runner |
| `run_figure2.py` | 409 | Figure 2 Weizmann Face / Eigenfaces PCA low-dimensional classification runner | Funcs: average_precision, do_ica_and_svm, do_pca_and_svm | cv2, deep_decoder, deep_prior, losses, matplotlib | CLI / Direct execution | Legacy Root Runner |
| `run_figure6.py` | 299 | Figure 6 Synthetic matrix robustness against Gaussian, Laplace, & S&P noise | Funcs: train_deeptensor_matrix | deep_prior, losses, matplotlib, models, numpy | CLI / Direct execution | Legacy Root Runner |
| `run_figure7.py` | 314 | Figure 7 Robust PCA covariance recovery under limited samples & low SNR | Funcs: train_deeptensor_pca | deep_prior, losses, matplotlib, models, numpy | CLI / Direct execution | Legacy Root Runner |
| `run_figure9.py` | 339 | Figure 9 ICVL Hyperspectral image denoising baseline runner | Funcs: compute_metrics | bm3d, deep_prior, losses, matplotlib, models | CLI / Direct execution | Duplicate Root Runner |
| `run_figure9_adaptive.py` | 481 | Research Extension 1: Adaptive Rank Selection — Plateau-Based  Based o | Procedural script | deep_prior, losses, matplotlib, models, numpy | CLI / Direct execution | Duplicate Root Runner |
| `run_figure9_bm3d.py` | 70 | BM3D classical image denoising baseline for ICVL hyperspectral image | Procedural script | bm3d, matplotlib, numpy, scipy, skimage | CLI / Direct execution | Other |
| `run_figure9_completion.py` | 439 | Extension 3: Missing-Data / Tensor Completion Capability for DeepTenso | Funcs: compute_subset_metrics, generate_completion_masks, parse_args | deep_prior, losses, matplotlib, models, numpy | CLI / Direct execution | Duplicate Root Runner |
| `run_figure9_ensemble.py` | 365 | Extension 5: Ensemble Disagreement as a Proxy for Reconstruction Uncer | Funcs: compute_edge_magnitude, main, set_seed | deep_decoder, deep_prior, losses, models, numpy | CLI / Direct execution | Duplicate Root Runner |
| `run_figure9_hybrid.py` | 241 | Script: run_figure9_hybrid.py | Funcs: main, set_seed | attention_spectral, deep_prior, losses, numpy, scipy | CLI / Direct execution | Duplicate Root Runner |
| `run_figure9_interpretability.py` | 400 | Extension 7: Post-hoc Factor Analysis and Descriptive Visualization of | Funcs: run_experiment | deep_decoder, deep_prior, losses, matplotlib, models | CLI / Direct execution | Duplicate Root Runner |
| `run_figure9_robustness.py` | 269 | This script replicates the DeepTensor results in Figure 6. The accurac | Procedural script | cv2, deep_decoder, deep_prior, losses, matplotlib | CLI / Direct execution | Duplicate Root Runner |
| `run_figure9_sparsity.py` | 448 | Research Extension 2: Factor Group-Sparsity (Group Lasso) Regularizati | Funcs: compute_effective_ranks, compute_group_regularizer, parse_args | deep_prior, losses, matplotlib, models, numpy | CLI / Direct execution | Duplicate Root Runner |
| `run_figure9_spectral_transformer.py` | 258 | Script: run_figure9_spectral_transformer.py | Funcs: main, set_seed | deep_prior, losses, numpy, scipy, skimage | CLI / Direct execution | Duplicate Root Runner |
| `run_table1.py` | 610 | Legacy Table 1 synthetic decomposition runner with PyMF / SNMF | Classes: PyMFBase, SNMF; Funcs: __init__, _converged, _init_h | deep_decoder, deep_prior, losses, matplotlib, models | CLI / Direct execution | Legacy Root Runner |
| `scratch/audit_experiments.py` | 257 | Script: audit_experiments.py | Procedural script | scipy | Standalone / Direct | Audit & Scratch Utility |
| `scratch/audit_helper.py` | 17 | Script: audit_helper.py | Procedural script | Standard library | Standalone / Direct | Audit & Scratch Utility |
| `scratch/build_audit_data.py` | 117 | Script: build_audit_data.py | Funcs: analyze_python_file, get_hash | Standard library | Standalone / Direct | Audit & Scratch Utility |
| `scratch_calibrate.py` | 49 | Scratch calibration script testing noise levels | Funcs: calibrate_gaussian, calibrate_poisson | numpy, scipy, utils | Standalone / Direct | Legacy Utility / Diagnostic |

---

## 3. Duplicate and Overlapping Code Analysis

The comprehensive hash-based AST audit revealed **11 distinct groups of duplicate files** across the codebase:

### 3.1 Identical Root vs Extension Duplicates (9 Files)
Nine scripts located in the root repository directory are byte-for-byte identical to scripts inside `previous_neurotensor_experiments/`:

1. `run_figure9.py` $\equiv$ `previous_neurotensor_experiments/01_icvl_cnn_baseline/run_figure9.py` (339 lines)
2. `run_figure9_adaptive.py` $\equiv$ `previous_neurotensor_experiments/02_icvl_adaptive_rank/run_figure9_adaptive.py` (481 lines)
3. `run_figure9_sparsity.py` $\equiv$ `previous_neurotensor_experiments/03_icvl_factor_group_sparsity/run_figure9_sparsity.py` (448 lines)
4. `run_figure9_completion.py` $\equiv$ `previous_neurotensor_experiments/04_icvl_tensor_completion/run_figure9_completion.py` (439 lines)
5. `run_figure9_robustness.py` $\equiv$ `previous_neurotensor_experiments/05_icvl_noise_robustness/run_figure9_robustness.py` (269 lines)
6. `run_figure9_ensemble.py` $\equiv$ `previous_neurotensor_experiments/06_icvl_ensemble_uncertainty/run_figure9_ensemble.py` (365 lines)
7. `run_figure9_interpretability.py` $\equiv$ `previous_neurotensor_experiments/07_icvl_factor_interpretability/run_figure9_interpretability.py` (400 lines)
8. `run_figure9_hybrid.py` $\equiv$ `previous_neurotensor_experiments/08_icvl_hybrid_attention/run_figure9_hybrid.py` (241 lines)
9. `run_figure9_spectral_transformer.py` $\equiv$ `previous_neurotensor_experiments/09_icvl_spectral_transformer/run_figure9_spectral_transformer.py` (258 lines)

> **Migration Recommendation**: Delete the 9 duplicate root files during reorganization. Keep their canonical copies under `experiments/extensions/icvl_extensions/`.

### 3.2 Duplicate Architectural Modules (2 Files)
Two neural architecture modules in `modules/` were copied directly into extension subdirectories:
1. `modules/attention_spectral.py` $\equiv$ `previous_neurotensor_experiments/08_icvl_hybrid_attention/modules/attention_spectral.py` (207 lines)
2. `modules/spectral_transformer.py` $\equiv$ `previous_neurotensor_experiments/09_icvl_spectral_transformer/modules/spectral_transformer.py` (155 lines)

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
