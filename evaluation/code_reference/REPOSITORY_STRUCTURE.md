# DeepTensor — Repository Structure Guide

**Document ID:** EVAL-STRUCT-01  
**Project Root:** `E:\Rakhal_Tensor`  
**Package:** `deeptensor` (installed in editable mode via `pyproject.toml`)

---

## 1. High-Level Directory Overview

```text
E:\Rakhal_Tensor\
├── src/deeptensor/                 # Core Python package source code
│   ├── baselines/                  # Traditional mathematical baselines (SVD, Tucker, TV)
│   ├── losses/                     # Data fidelity & regularization losses (L1, MSE, Group Lasso)
│   ├── models/                     # DeepTensor neural architectures & factor generators
│   │   ├── attention/              # Self-attention, Pure Transformer, & Control modules
│   │   ├── backbones/              # 1D/2D Skip U-Net generator architectures
│   │   └── deeptensor.py           # DeepTensor matrix & 3-way CP core classes
│   ├── physics/                    # Physical forward sensing operators (Radon, CACTI)
│   ├── training/                   # Optimization loops & Adaptive Rank Selection
│   └── utils/                      # Metrics, tensor contractions, & legacy utilities
├── experiments/                    # Runnable experiment suites & drivers
│   ├── reproduction/               # Original Heckel & Soltanolkotabi (2020) reproduction
│   ├── hybrid_prototypes/          # Five-architecture comparative benchmark runners
│   ├── extensions/                 # Novel research extensions (Adaptive Rank, Group Sparsity, etc.)
│   └── improvements/               # Extended convergence runs (Figure 10 Yale Face B 5k epochs)
├── results/                        # Generated experimental artifacts & metrics
│   ├── baseline/                   # Reproduction metrics, plots, and checkpoints
│   ├── hybrid/                     # Five-model comparative results across all 4 applications
│   ├── improved/                   # 5000-epoch Yale Face B rerun artifacts (30.86 & 30.46 dB)
│   ├── adaptive_rank_selection/    # Active rank trajectory logs & evolution plots
│   └── factor_group-sparsity/      # Group Lasso metrics, error maps, & network summaries
├── data/                           # Evaluation tensors & datasets (icvl.mat, yale_b.mat, etc.)
├── evaluation/                     # Faculty defense deliverables & verification tooling
│   ├── code_navigator/             # Interactive local web code navigator (FastAPI + React)
│   ├── code_reference/             # Seven formal audited reference markdown documents
│   ├── docs/                       # Research blueprints & comprehensive audit reports
│   └── presentation/               # Rendered slides & presentation slide assets
├── reports/                        # Extensive technical reports & experiment logs
│   ├── reproduction/               # Literature reproduction documentation & registers
│   ├── extensions/                 # Architectural extension reports (AHD-Net v1/v2, etc.)
│   └── audits/                     # Codebase audits & dimension trace records
└── configs/                        # YAML/JSON experiment configuration files
```

---

## 2. Granular Module Responsibilities

### `src/deeptensor/` — Core Neural & Mathematical Library
- **`src/deeptensor/models/deeptensor.py`**: Contains `DeepTensor` (matrix factorization $\hat{X} = UV^T$) and `DeepTensorCP` (3-way CP/PARAFAC factorization $\hat{\mathcal{X}} = \sum_r \lambda_r u_r \circ v_r \circ w_r$). Implements PyTorch multilinear einsum contractions.
- **`src/deeptensor/models/backbones/skip.py`**: The underlying convolutional generative engine. Provides `skip(...)` (2D U-Net) and `skip1d(...)` (1D U-Net) with reflection padding, LeakyReLU activations, and skip connections.
- **`src/deeptensor/models/attention/`**:
  - `spatial_spectral.py`: Implements `SpectralSelfAttention` (MHSA branch), `SpectralConvControl` (conv control), and `SpectralPointwiseControl` ($1 \times 1$ pointwise conv control).
  - `spectral_transformer.py`: Implements `SpectralTransformer` (pure non-convolutional Transformer with Pre-LN and ALiBi relative position biases).
- **`src/deeptensor/models/factory.py`**: Central model dispatcher. Ensures strict parameter-matching across architectures via `count_parameters`.
- **`src/deeptensor/physics/operators.py`**: Differentiable physical forward measurement operators:
  - `radon(...)`: Forward parallel-beam Radon transform for CT/PET imaging.
  - `video2codedvideo(...)`: Shifting binary coded aperture operator for CACTI compressive video sensing.
- **`src/deeptensor/baselines/spectral.py`**: Classical matrix/tensor algorithms:
  - `svd_decompose(...)`: Truncated SVD.
  - `tucker_decompose(...)`: TensorLy Higher-Order SVD / Tucker decomposition.
- **`src/deeptensor/training/adaptive_rank.py`**: `AdaptiveRankManager` class monitoring EMA loss, identifying training plateaus, and triggering rank expansion.
- **`src/deeptensor/losses/`**: `objectives.py` ($\mathcal{L}_1$, $\mathcal{L}_2$ fidelity) and `group_sparsity.py` (Group Lasso penalty).
- **`src/deeptensor/utils/`**:
  - `tensor_ops.py`: Tensor mode unfoldings and CP contractions.
  - `metrics.py`: PSNR, SSIM, RMSE, and MAE implementations.
  - `utils_legacy.py`: Random noise latent tensor initialization (`get_inp`).

---

### `experiments/` — Experiment Runners
- **`experiments/reproduction/`**: Scripts reproducing original paper figures:
  - `figure09_hyperspectral/run_figure9.py`: ICVL HSI baseline.
  - `figure10_yale_denoising/run_figure10_yale_tensor.py`: Yale Face B 1200-epoch baseline.
  - `figure11_video_cs/run_figure11.py`: Video CS baseline.
  - `figure12_pet_ct/run_figure12.py`: PET/CT baseline.
- **`experiments/hybrid_prototypes/`**: The core 5-architecture comparative benchmark:
  - `run_hybrid_figure09_hyperspectral.py`: HSI 5-model test (5000 epochs).
  - `run_hybrid_figure10_yale.py`: Yale B 5-model test (1200 epochs).
  - `run_hybrid_figure11_video.py`: Video CS 5-model test (3000 epochs).
  - `run_hybrid_figure12_pet.py`: PET/CT 5-model test (4000 epochs).
- **`experiments/extensions/`**:
  - `adaptive_rank/run_figure9_adaptive.py`: Dynamic bottom-up rank growth.
  - `group_sparsity/run_figure9_sparsity.py`: Top-down continuous Group Lasso rank pruning.
- **`experiments/improvements/`**:
  - `figure10_yale/run_figure10_yale_tensor_improved.py`: Extended 5000-epoch rerun reaching 30.86 dB and 30.46 dB.

---

### `results/` — Numerical & Visual Data Repositories
- **`results/baseline/`**: Reproduction JSON metrics and comparison figures for all four applications.
- **`results/hybrid/`**: Master JSON comparisons (`figure09_icvl_hsi_comparison.json`, etc.) containing metrics for all 5 models, along with high-resolution visual panels (`*_comparison_panel_annotated.png`).
- **`results/improved/`**: Progression CSVs (`progression_noise_30.csv`) and checkpoints from extended runs.
- **`results/adaptive_rank_selection/`**: Rank evolution plots and summary metrics for Adaptive Rank Selection.
- **`results/factor_group-sparsity/`**: Canonical metrics JSON and visual comparisons for Factor Group Sparsity.

---

### `evaluation/` — Defense, Presentation & Interactive Navigator
- **`evaluation/code_navigator/`**: Interactive web interface allowing real-time navigation from presentation concepts to exact source code lines.
  - `backend/server.py`: FastAPI server serving topics and files.
  - `frontend/`: Interactive React/Tailwind frontend.
  - `data/`: Topic maps, experiment maps, and source mappings.
- **`evaluation/code_reference/`**: The seven audited reference documents connecting every slide to code.
- **`evaluation/presentation/`**: 28 rendered slide images (`rendered_slides/slide_01.png` to `slide_28.png`).

