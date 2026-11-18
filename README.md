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

## 5. How to Run Experiments

### Running Paper Reproduction Baselines:
```bash
# Figure 2: Eigenfaces Classification
python experiments/reproduction/figure02_eigenfaces/run_figure2.py

# Figure 9: ICVL Hyperspectral Denoising
python experiments/reproduction/figure09_hyperspectral/run_figure9.py

# Table 1: Synthetic Matrix/Tensor Suite
python experiments/reproduction/table01_synthetic/run_synthetic_suite.py
```

### Running High-Performance Improved Models:
```bash
# Figure 10: Yale Face B Denoising (128 channels, 4000 epochs)
python experiments/improvements/figure10_yale/run_figure10_yale_tensor_improved.py --noise_ratio 0.3 --epochs 4000

# Figure 11: Video Compressive Sensing (64 frames, 10000 epochs)
python experiments/improvements/figure11_video/run_figure11_improved.py --epochs 10000

# Figure 12: 3D PET/CT Radon Tomography (Poisson tau=100, Cosine LR)
python experiments/improvements/figure12_pet_ct/run_figure12_improved.py --epochs 4000

# Figure 13(b): Yale Face B Rank Sweep with CUDA PARAFAC ALS
python experiments/improvements/figure13b_yale_rank/run_figure13b_yale_rank_sweep_improved.py --epochs 1500
```

### Launching the Interactive Demo:
```bash
streamlit run apps/streamlit_demo/demo.py
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
