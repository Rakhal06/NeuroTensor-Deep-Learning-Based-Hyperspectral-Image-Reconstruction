# DeepTensor Implementation Status & Codebase Audit

This document provides an engineering audit of all modules, generator architectures, utility functions, and runner scripts in NeuroTensor.

---

## 1. Module Inventory & Reusability Assessment

| Module File | Component Name | Role / Functionality | Reusability | Required Corrections / Action |
|---|---|---|---|---|
| `modules/deep_prior.py` | `skip()`, `get_net()` | 1D & 2D U-Net Deep Image Prior generator networks | **FULLY REUSABLE** | None. Standard architecture matching paper Figure 5. |
| `modules/deep_decoder.py` | `decodernw1d()`, `decodernw2d()` | Under-parameterized Deep Decoder architecture | **FULLY REUSABLE** | Reusable for Figure 8 stopping criterion comparisons. |
| `modules/models.py` | `UNetND()`, `MLP()`, `Conv1d_Block()` | Generic multi-dimensional U-Net and MLP implementations | **FULLY REUSABLE** | Reusable for baseline comparisons. |
| `modules/losses.py` | `L1Norm()`, `L2Norm()`, `TV2D()` | Loss functions ($\ell_1$, $\ell_2$, Total Variation) | **FULLY REUSABLE** | Reusable across all applications. |
| `modules/lin_inverse.py` | `radon()`, `iradon()`, `cs_video()` | Forward linear models (Radon transform, Video CS) | **REUSABLE WITH FIX** | `radon()` already patched for modern Kornia rotate; add `cs_video()` forward model abstraction. |
| `modules/spectral.py` | `lr_decompose()`, `tucker_decompose()` | Truncated SVD and Tucker decomposition baselines | **FULLY REUSABLE** | Device conversion to CPU/CUDA already patched. |
| `modules/utils.py` | `psnr()`, `get_matrix()`, `get_scheduler()` | Metrics, synthetic data generation, learning rate schedulers | **FULLY REUSABLE** | Device support (`torch.device`) already patched. |
| `modules/attention_spectral.py` | `HybridSpectralAttention()` | Hybrid CNN–Transformer spectral attention module | **REUSABLE FOR HYBRID** | Reusable for Extension 8 and Phase 2 hybrid experiments. |
| `modules/spectral_transformer.py` | `SpectralTransformerGenerator()` | Transformer-dominant factor generator with ALiBi | **REUSABLE FOR HYBRID** | Reusable for Extension 9 and Phase 2 comparisons. |
| `modules/cosine_annealing_with_warmup.py` | `CosineAnnealingWarmupRestarts` | Learning rate scheduler | **REUSABLE** | Required for Table 1 learning rate scheduler comparisons. |

---

## 2. Runner Scripts Audit

| Script | Targeted Paper Experiment | Code Exists? | Executed? | Saved Artifacts? | Required Modifications |
|---|---|---|---|---|---|
| `run_figure2.py` | Figure 2 (Eigenfaces) | Yes | **YES (CUDA)** | Yes (`RESULTS-BASELINE/figure2/`) | Fully working; port into `paper_reproduction/cnn_baseline/04_eigenfaces_classification/`. |
| `run_figure6.py` | Figure 6 (Synthetic robustness) | Yes | **YES (CUDA)** | Yes (`RESULTS-BASELINE/figure6/`) | Fully working; port into `paper_reproduction/cnn_baseline/01_synthetic_decomposition/`. |
| `run_figure7.py` | Figure 7 (Robust PCA) | Yes | **YES (CUDA)** | Yes (`RESULTS-BASELINE/figure7/`) | Fully working; port into `paper_reproduction/cnn_baseline/02_pca/`. |
| `run_figure9.py` | Figure 9 (Hyperspectral) | Yes | **YES (CUDA)** | Yes (`RESULTS-BASELINE/figure9/`) | Fully working; port into `paper_reproduction/cnn_baseline/05_hyperspectral_denoising/`. |
| `run_figure9_bm3d.py` | Figure 9 BM3D baseline | Yes | **YES (CUDA)** | Yes (metrics in `RESULTS-BASELINE/figure9/`) | Standalone BM3D benchmark. |
| `run_figure11.py` | Figure 11 (Video CS DeepTensor) | Yes | **NO (UNRUN)** | None | **BUG FIX:** Change `sys.path.append('../modules')` to `sys.path.append('modules')`. |
| `run_figure11_TV.py` | Figure 11 (Video CS 2D TV) | Yes | **NO (UNRUN)** | None | Ready for execution. |
| `run_figure12.py` | Figure 12 (3D CT Reconstruction) | Yes | **YES (CUDA)** | Yes (`RESULTS-BASELINE/figure12/`) | Fully working; port into `paper_reproduction/cnn_baseline/08_ct_reconstruction/`. |
| `run_figure12_TV.py` | Figure 12 2D TV baseline | Yes | **YES (CUDA)** | Integrated in `run_figure12.py` | Verified. |
| `run_table1.py` | Table 2 (NMF) & Table 3 (Speech) | Yes | **NO (UNRUN)** | None | Script exists but author named it `run_table1.py`. Needs modular separation into App 03 and App 09 runners. |
| `matrix_decomp.py` | Figure 8 / Table 1 | Yes | **NO (UNRUN)** | None | Script contains prototype code; needs structuring into dedicated Figure 3, Figure 8, and Table 1 runners. |

---

## 3. Missing Modules Requiring New Implementation

1. **Application 01 Dedicated Runners:**
   - `run_figure3_inductive_bias.py`: Measures MSE convergence of signal vs. noise.
   - `run_figure4_separable_speedup.py`: Compares FLOPs and runtime of 2D+1D DIP vs. 3D DIP.
   - `run_figure8_stopping_criterion.py`: Evaluates over- vs. under-parameterized DIP stopping curves.
   - `run_table1_lr_schedulers.py`: Executes the $4 \times 4$ scheduler grid.

2. **Application 03 & 09 Dedicated Runners:**
   - `run_table2_nmf_mnist_cifar.py`: Separates MNIST & CIFAR-10 NMF from monolithic `run_table1.py`.
   - `run_table3_speech_nmf.py`: Separates Speech Commands spectrogram factorization from monolithic `run_table1.py`.

3. **Application 06 Full Implementation (BLOCKED on Data):**
   - `run_figure10_yale_tensor_denoising.py`: 3-way PARAFAC DIP model under 30% and 60% Salt-and-Pepper noise.
   - `run_figure13_rank_sweep.py`: Sweeps rank on truncated ICVL (1 to 30) and truncated Yale-B (10 to 4000).

4. **Hybrid CNN–Transformer Architecture Suite:**
   - Application-specific hybrid models for Applications 01, 02, 03, 04, 07, 08, 09 (Application 05 already implemented).

