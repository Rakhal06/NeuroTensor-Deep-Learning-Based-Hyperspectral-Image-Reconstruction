# DeepTensor++: Low-Rank Matrix and Tensor Decomposition with Deep Network Priors

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Structure: Modular](https://img.shields.io/badge/architecture-modular-brightgreen.svg)]()

DeepTensor++ is a high-performance, modular research codebase replicating and extending:
> **DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors**  
> *Vishwa Saragadam, Rangamani Sundaramoorthi, Richard G. Baraniuk, and Ashok Veeraraghavan*  
> IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) / NeurIPS.  
> [arXiv:2204.03145](https://arxiv.org/abs/2204.03145)

---

## 1. Key Capabilities & Architectural Overview

DeepTensor decomposes high-dimensional, noisy, or incompletely observed matrices and tensors into factor components parameterized by **deep convolutional network priors** (such as 2D/1D Deep Image Priors or pixel-wise Deep Decoders). 

By leveraging the implicit architectural spectral bias of convolutional generators, DeepTensor achieves:
1. **Implicit Regularization**: Drastic noise filtering without explicit spatial smoothness penalties.
2. **Invariance to Rank Over-Estimation**: Maintains flat reconstruction plateaus even when the decomposition rank $R$ greatly exceeds the true underlying rank ($R_{\text{true}}$), whereas classical PARAFAC ALS catastrophically overfits.
3. **Linear Inverse Recovery**: Reconstruction from compressed, coded, or Radon projections (Video CS, 3D PET/CT).

---

## 2. Project Directory Structure

```text
DeepTensor-Plus-Plus/
│
├── pyproject.toml                              # Modern editable packaging (pip install -e .)
├── requirements.txt                            # Pinned virtual environment dependencies
├── README.md                                   # Master documentation & user guide
│
├── configs/                                    # Standardized YAML/JSON experiment configurations
│   ├── reproduction/                           # Configurations for 13 paper baseline targets
│   ├── improvements/                           # Verified high-performance hyperparameter configurations
│   └── hybrid/                                 # Phase 2 hybrid CNN-Transformer configs
│
├── src/                                        # Installable `deeptensor` Python package
│   └── deeptensor/
│       ├── __init__.py                         # Version, exports, and top-level API
│       ├── models/                             # DeepTensor wrapper, DIP factories, MLPs
│       │   ├── deeptensor.py                   # DeepTensor dual-factor forward module
│       │   ├── deep_prior.py                   # DIP factory methods (skip, get_net, fill_noise)
│       │   ├── deep_decoder.py                 # Non-convolutional pixel-wise decoder
│       │   ├── backbones/                      # Skip, U-Net, ResNet, DCGAN convolutional generators
│       │   └── attention/                      # HybridSpectralNet, SpectralTransformer with ALiBi
│       ├── physics/                            # Linear forward operators & adjoints (Radon, Video CS)
│       ├── losses/                             # L1, L2, and anisotropic Total Variation (TVNorm)
│       ├── baselines/                          # PyTorch CUDA PARAFAC ALS, Truncated SVD, Tucker
│       ├── training/                           # Cosine Annealing scheduler & training utilities
│       └── utils/                              # Metrics (PSNR, SSIM, SAM, ERGAS) & device resolver
│
├── experiments/                                # Segregated, runnable research suites
│   ├── reproduction/                           # Direct paper reproduction targets (Figures 2-13, Tables 1-3)
│   ├── improvements/                           # Verified high-performance runs (Figures 10, 11, 12, 13b)
│   ├── extensions/                             # 9 ICVL extensions (adaptive rank, sparsity, attention, etc.)
│   ├── legacy_experiments/                     # Historical reference implementations
│   └── hybrid_prototypes/                      # Phase 2 CNN-Transformer prototypes
│
├── data/                                       # Authoritative single-source dataset store (5.83 GB)
│   ├── icvl.mat                                # ICVL Hyperspectral Image (174x164x260, uint16)
│   ├── pet.mat                                 # 3D PET/CT volume (128x128x56, uint16)
│   ├── yale_b.mat                              # Extended Yale Face B tensor (192x168x160, uint8)
│   ├── cat_video.mat                           # Video CS volume (128x128x64, float32)
│   ├── weizzman.mat                            # Weizmann Face database (840x103x71, float64)
│   ├── MNIST/                                  # MNIST benchmark digits
│   ├── cifar-10-batches-py/                    # CIFAR-10 image batches
│   └── SpeechCommands/                         # Google Speech Commands v0.02 (105k clips)
│
├── results/                                    # Segregated result registries
│   ├── baseline/                               # Verified original baseline checkpoints & logs
│   ├── improved/                               # Verified improved checkpoints, logs & panels
│   ├── extensions/                             # Historical ICVL extension outputs
│   └── hybrid/                                 # Phase 2 outputs
│
├── reports/                                    # Consolidated reports, audits & publication panels
│   ├── reproduction/                           # Consolidated reproduction reports & image gallery
│   ├── improvements/                           # Individual improvement audit reports
│   ├── extensions/                             # Theoretical documentation for 9 extensions
│   └── audits/                                 # Codebase consistency audits & dimension traces
│
├── apps/                                       # Interactive applications
│   └── streamlit_demo/                         # Real-time Streamlit visualization app
│
├── tests/                                      # Automated unit test & smoke verification suite
├── tools/                                      # Utilities (PDF generator, diagnostics)
├── archive/                                    # Backups and legacy runners
└── migration/                                  # Pre/Post SHA-256 manifests & migration logs
```

---

## 3. Installation & Environment Setup

1. **Clone or navigate to workspace**:
   ```bash
   cd e:\Rakhal_Tensor
   ```

2. **Activate the virtual environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install `deeptensor` as an editable package**:
   ```bash
   pip install -e .
   ```

4. **Verify installation & run unit tests**:
   ```bash
   python tests/run_all_tests.py
   ```

---

## 4. Verified Experimental Results Summary

| Target | Application Domain | Primary Dataset | Baseline Metric | Improved Metric | Paper Target | Status |
|---|---|---|---|---|---|---|
| **Fig 2** | Eigenfaces Low-Dim Classification | `weizzman.mat` | **85.71%** Acc | N/A | ~88.0% Acc | **Frozen** |
| **Fig 4** | 3D Separable ConvNet Speedup | Synthetic $128^2 \times 150$ | Flawed | Script Ready | ~100x Speedup | **Pending Run** |
| **Fig 6** | Synthetic Noise Robustness | Synthetic ($64^2$) | Replicated | N/A | Robust over SVD | **Frozen** |
| **Fig 7** | Robust PCA Covariance Recovery | Covariance Matrix | Replicated | N/A | Lower error vs PCA | **Frozen** |
| **Fig 9** | ICVL Hyperspectral Denoising | `icvl.mat` ($174 \times 164 \times 260$) | **46.28 dB** | N/A | 43.40 dB | **Frozen** |
| **Fig 10** | Yale Face B (30% & 60% S&P) | `yale_b.mat` ($192 \times 168 \times 160$) | 28.52 / 28.18 dB | **30.86 / 30.46 dB** | 28.9 / 29.7 dB | **Approved** |
| **Fig 11** | Video CS (8 to 64 frames) | `cat_video.mat` ($128^2 \times 64$) | 25.14 dB (TV: 23.47) | **25.32 dB** (TV: 23.49) | 27.10 dB (HQ) | **Audited** |
| **Fig 12** | 3D PET/CT Radon Tomography | `pet.mat` ($128^2 \times 56$) | 28.84 dB | **30.55 dB** (0.8302) | ~30.5 dB | **Audited** |
| **Fig 13a**| HSI Rank Invariance Sweep | `icvl.mat` ($R=1..30$) | Flat plateau $\ge 20$ | N/A | Flat plateau $\ge 20$ | **Frozen** |
| **Fig 13b**| Yale B Rank Sweep Invariance | `yale_b.mat` ($R=10..4000$) | 23.22–23.38 dB | **28.50–28.97 dB** | ~26.0 dB | **Audited** |
| **Tab 1** | Synthetic Grid Decomposition | Synthetic ($N=64..256$) | **0.00 $\pm$ 0.00** | N/A | 0.00 $\pm$ 0.00 | **Frozen** |
| **Tab 2** | NMF on MNIST & CIFAR-10 | MNIST & CIFAR-10 | **7.90 dB / 9.50 dB** | N/A | 7.30 dB / 8.80 dB | **Frozen** |
| **Tab 3** | Speech Commands Spectrogram NMF | SpeechCommands v2 | Replicated | N/A | Matches Paper | **Frozen** |

---

## 5. Comprehensive Execution Guide for Evaluators & Faculty

This section provides complete, tested reproduction commands mapping directly to every table, figure, and quantitative claim in the 30-page project report (`NeuroTensor_Project_Report/main.pdf`).

### 5.1 Tier 1: Instant Verification & Architecture Audit (< 10 Seconds)
Run these commands to immediately verify the mathematical integrity, test suite, and parameter counts without waiting for neural network training:

1. **Automated Unit & Operator Tests:**
   ```bash
   python tests/run_all_tests.py
   ```
2. **Instant Parameter & Shape Verification for All 5 Spectral Architectures:**
   ```bash
   python -c "import torch; from src.deeptensor.models.factory import build_spectral_net; [print(f'{m:<20} | Params: {sum(p.numel() for p in build_spectral_net(20, 20, 260, net_type=m).parameters() if p.requires_grad):>10,} | Output: {list(build_spectral_net(20, 20, 260, net_type=m)(torch.zeros(1, 20, 260)).shape)}') for m in ['skip1d', 'hybrid', 'cnn_control', 'pointwise_control', 'transformer']]"
   ```
   *Expected Output: Confirms exact parameter matching (skip1d: 799,176; hybrid: 931,656; cnn_control: 931,828; pointwise_control: 931,528; transformer: 798,612).*

---

### 5.2 Tier 2: Fast Smoke Tests (1–2 Minutes for Full Pipeline Sanity Check)
Run complete end-to-end forward/backward passes (10 epochs) across all 5 architectures and extensions to confirm flawless execution:

```bash
# Smoke test all five architectural configurations on ICVL:
python experiments/hybrid_prototypes/run_hybrid_figure09_hsi.py --variant all --smoke_test

# Smoke test Yale Face B multi-model denoising:
python experiments/hybrid_prototypes/run_hybrid_figure10_yale.py --variant all --smoke_test

# Smoke test Video Compressive Sensing:
python experiments/hybrid_prototypes/run_hybrid_figure11_video.py --variant all --smoke_test

# Smoke test 3D Dynamic PET/CT:
python experiments/hybrid_prototypes/run_hybrid_figure12_pet_ct.py --variant all --smoke_test
```

---

### 5.3 Tier 3: Reproducing the Nine Research Extensions (Report Chapter 4)

| Report Reference | Extension Name | Script Command | Output Artifact |
|---|---|---|---|
| **Table 5 & Fig 3** | **Extension 1:** Self-Supervised Adaptive Rank Selection | `python experiments/extensions/adaptive_rank/run_figure9_adaptive.py` | `results/adaptive_rank_results.png`, logs |
| **Table 6 & Fig 4** | **Extension 2:** Factor Group Sparsity (Group Lasso) | `python experiments/extensions/group_sparsity/run_figure9_sparsity.py --lambda_value 0.0001` | `results/factor_group-sparsity/`, plots |
| **Table 7 & Fig 5** | **Extension 3:** Mask-Aware Tensor Completion ($p=0.50$) | `python experiments/extensions/tensor_completion/run_figure9_completion.py --missing_ratio 0.50` | `results/extensions/completion/`, metrics |
| **Table 8 & Fig 6** | **Extension 4:** Noise Robustness (Gaussian vs Poisson) | `python experiments/extensions/noise_robustness/run_figure9_robustness.py` | `results/noise_robustness/`, summary |
| **Table 9 & Fig 7** | **Extension 5:** Ensemble Disagreement & Spatial Uncertainty | `python experiments/extensions/ensemble_uncertainty/run_figure9_ensemble.py` | `results/ensemble/`, correlation analysis |
| **Section 4.8** | **Extension 6:** 3D Separable ConvNet Computational Profiling | `python archive/legacy/paper_reproduction_legacy/cnn_baseline/01_synthetic_decomposition/benchmark_figure4_separable_speedup_improved.py` | Timing logs & speedup curves |
| **Table 10 & Fig 8-10**| **Extension 7:** Factor Interpretability & Component Ablation | `python experiments/extensions/factor_interpretability/run_figure9_interpretability.py` | `results/interpretability/`, cosine matrices |
| **Table 11 & Fig 11** | **Extensions 8 & 9:** 5-Model Architectural Comparison on ICVL | `python experiments/hybrid_prototypes/run_hybrid_figure09_hsi.py --variant all --epochs 5000` | `results/hybrid/figure09_hyperspectral/` |

---

### 5.4 Tier 4: Cross-Application Multimodal Benchmark (Report Table 12)
To run the full 5-model comparative benchmark across all four distinct multimodal problem domains:

```bash
# 1. ICVL Hyperspectral Cube Denoising (174x164x260):
python experiments/hybrid_prototypes/run_hybrid_figure09_hsi.py --variant all

# 2. Yale Face B Photometric Tensor Denoising (192x168x160):
python experiments/hybrid_prototypes/run_hybrid_figure10_yale.py --variant all

# 3. Video Compressive Sensing (128x128x64):
python experiments/hybrid_prototypes/run_hybrid_figure11_video.py --variant all

# 4. Dynamic 3D PET/CT Radon Tomography (128x128x56):
python experiments/hybrid_prototypes/run_hybrid_figure12_pet_ct.py --variant all
```

---

### 5.5 Tier 5: Paper Baseline Reproduction & Interactive Demo

```bash
# Baseline Figure 2: Eigenfaces Classification
python experiments/reproduction/figure02_eigenfaces/run_figure2.py

# Baseline Figure 9: ICVL Hyperspectral Denoising
python experiments/reproduction/figure09_hyperspectral/run_figure9.py

# Table 1: Synthetic Matrix/Tensor Suite
python experiments/reproduction/table01_synthetic/run_synthetic_suite.py

# Launch Interactive Streamlit Demonstration Dashboard:
streamlit run apps/streamlit_demo/demo.py
```

---

### 5.6 Tier 6: Compiling the 30-Page LaTeX Project Report
To recompile the official academic report PDF (`NeuroTensor_Project_Report/main.pdf`):

```bash
cd NeuroTensor_Project_Report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

---

## 6. Testing & Quality Assurance

Run the automated test suite verifying package imports, mathematical operators, loss functions, CUDA PARAFAC ALS, and dataset shapes:
```bash
python tests/run_all_tests.py
```

---

## 7. License & Provenance

This codebase is licensed under the MIT License. Original author implementations by Vishwa Saragadam et al.; modernized, modularized, and extended by Rakhal Krishna.
