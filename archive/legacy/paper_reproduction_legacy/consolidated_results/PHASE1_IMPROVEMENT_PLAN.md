# DeepTensor++: Phase 1 Improvement & Baseline Rerun Plan

## Executive Summary
This document provides the concrete execution plan to upgrade and rerun our original CNN/DIP DeepTensor baselines. Based on the findings in [`PHASE1_PERFORMANCE_GAP_AUDIT.md`](file:///e:/Rakhal_Tensor/PHASE1_PERFORMANCE_GAP_AUDIT.md), each identified experiment will be rerun with exact paper-compliant configurations, sufficient iteration budgets, and validated baseline comparisons.

**Strict Invariant Maintained:** In accordance with the execution protocol, **no hybrid CNN–Transformer models will be executed until these baseline improvements are fully verified and reviewed.**

---

## 1. Rerun Execution Sequencing & Compute Budget

All runs will be executed sequentially on our NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) to prevent CUDA memory contention.

| Execution Order | Experiment | Application Folder | Proposed Rerun Parameters | Estimated Runtime | Peak VRAM | Target Metric |
|---|---|---|---|---|---|---|
| **Batch 1 (Priority 1)** | Figure 10 (Yale Face B) | `06_yale_face_tensor_denoising` | 4000 epochs, 128 channels, exact boolean mask, Cosine LR | ~14 min | ~1.4 GB | 30%: $\ge 28.5$ dB; 60%: $\ge 29.0$ dB |
| **Batch 2 (Priority 2)** | Figure 11 (Video CS) | `07_video_compressive_sensing` | Full rank $R=64$, 10,000 epochs, 8 coded frames | ~12 min | ~1.8 GB | $\ge 26.8$ dB PSNR, $\ge 0.79$ SSIM |
| **Batch 3 (Priority 3)** | Figure 13b (Yale Rank Sweep) | `06_yale_face_tensor_denoising` | 1500 epochs per rank, true 3-way PARAFAC baseline | ~18 min | ~1.2 GB | DT plateau $\ge 26.0$ dB; baseline drops $< 21$ dB |
| **Batch 4 (Priority 4)** | Figure 4 (Separable Speedup) | `01_synthetic_decomposition` | Equivalent 3D U-Net vs 2D+1D vs 1D on $128 \times 128 \times 150$ | ~6 min | ~4.5 GB | Measured runtime curve showing 3D net $> 20\times$ slower |
| **Batch 5 (Priority 5)** | Figure 12 (3D PET CT) | `08_ct_reconstruction` | 4000 epochs, $\tau=100$ (max 100 photons), FBP+TV baseline | ~10 min | ~1.6 GB | $\ge 30.8$ dB PSNR, $\ge 0.86$ SSIM |
| **Batch 6 (Priority 6)** | Table 2 (NMF MNIST/CIFAR) | `03_nmf` | Deep Decoder ($R=128, N=2048$) alongside DIP U-Net | ~8 min | ~1.0 GB | MNIST: $7.6 \pm 0.2$ dB; CIFAR: $6.3 \pm 0.1$ dB |
| **Batch 7 (Priority 7)** | Table 1 (LR Grid) | `01_synthetic_decomposition` | 5 random seeds, StepLR $\gamma=0.5$, ExpLR $\gamma=0.995$ | ~5 min | ~0.8 GB | Matches paper Table 1 mean and std bounds |
| **TOTAL** | All 7 Target Reruns | Complete Baseline Suite | Comprehensive parameter-aligned execution | **~73 min** | $\le 4.5$ GB | All gaps closed to within paper error margins |

---

## 2. Detailed Technical Action Plan by Experiment

### Batch 1: Figure 10 – Yale Face B 3-Way PARAFAC Tensor Denoising
- **Target Script:** `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure10_yale_tensor.py`
- **Configuration Changes:**
  ```python
  # Updated parameters
  rank = 2000
  epochs = 4000                  # Increased from 1,200
  skip_n33d = 128                # Increased from 64 (matches paper)
  skip_n33u = 128                # Increased from 64
  learning_rate = 1e-3
  scheduler = 'cosine'           # Added Cosine Annealing (T_max=4000, eta_min=1e-5)
  noise_generator = 'boolean'    # np.random.rand(*tensor.shape) < noise_ratio
  ```
- **Validation Criteria:**
  - 30% Salt & Pepper PSNR $\ge 28.5\text{ dB}$ (Paper: 28.90 dB).
  - 60% Salt & Pepper PSNR $\ge 29.0\text{ dB}$ (Paper: 29.70 dB).
  - Clear visual identification of eyes, facial contours, and expression without impulse artifacts.

---

### Batch 2: Figure 11 – Video Compressive Sensing (Cat Video)
- **Target Script:** `paper_reproduction/cnn_baseline/07_video_compressive_sensing/run_figure11.py`
- **Configuration Changes:**
  ```python
  # Updated parameters
  rank = 64                      # Increased from 32 (full-rank for 64 frames)
  epochs = 10000                 # Increased from 3,000 (matches paper Section 4.1)
  nframes = 8                    # 8 frames per coded frame (8 total coded measurements)
  # Fix measurement tensor dimension:
  # measurements shape: (H, W, totalframes // nframes) = (128, 128, 8)
  ```
- **Validation Criteria:**
  - DeepTensor PSNR $\ge 26.8\text{ dB}$, SSIM $\ge 0.790$ (Paper: 27.10 dB, 0.810).
  - DeepTensor advantage over 2D TV exceeds $+2.5\text{ dB}$ and $+0.08\text{ SSIM}$.

---

### Batch 3: Figure 13b – Yale Face B PARAFAC Rank Sweeps
- **Target Script:** `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure13b_yale_rank_sweep.py`
- **Configuration Changes:**
  ```python
  # Updated parameters
  epochs = 1500                  # Increased from 500 per rank
  ranks = [10, 50, 100, 500, 1000, 2000, 4000]
  # Classical baseline: True 3-way PARAFAC ALS via tensorly.decomposition.parafac
  # (demonstrating severe overfitting at R=2000, 4000)
  ```
- **Validation Criteria:**
  - DeepTensor maintains a high plateau ($\ge 26.0\text{ dB}$) across ranks $R \in [1000, 4000]$.
  - Classical PARAFAC drops sharply below $21.0\text{ dB}$ as rank increases beyond true rank 1000.

---

### Batch 4: Figure 4 – Computational Efficiency & Separable Convolutions
- **Target Script:** `paper_reproduction/cnn_baseline/01_synthetic_decomposition/benchmark_figure4_separable_speedup.py`
- **Configuration Changes:**
  ```python
  # Equivalent benchmark parameters:
  volume_shape = (128, 128, 150) # Paper-specified volume dimensions
  # Architecture 1: Three 1D U-Nets (5 scales, skip connections)
  # Architecture 2: One 2D U-Net + One 1D U-Net (5 scales, skip connections)
  # Architecture 3: One 3D U-Net (5 scales, 3D convs, 3D upsampling)
  # Rigor: 10 warmup steps, torch.cuda.synchronize(), 50 timed iterations
  ```
- **Validation Criteria:**
  - Direct measurement of seconds/epoch, peak VRAM, and GFLOPs.
  - Empirical curve demonstrating that 3D U-Net is $> 20\times$ slower than separable networks on identical input volumes.

---

### Batch 5: Figure 12 – 3D CT Reconstruction from 40 Radon Angles
- **Target Script:** `run_figure12.py`
- **Configuration Changes:**
  ```python
  # Updated parameters
  tau = 100                      # Corrected from 1000 (paper specified max 100 photons)
  epochs = 4000                  # Increased from 2000
  # Baseline: Classical Filtered Backprojection (FBP) + TV regularization
  ```
- **Validation Criteria:**
  - DeepTensor PSNR $\ge 30.8\text{ dB}$, SSIM $\ge 0.860$ (Paper: 31.10 dB, 0.870).
  - Outperforms classical FBP+TV baseline in structural fidelity.

---

### Batch 6: Table 2 – NMF on MNIST & CIFAR-10
- **Target Script:** `paper_reproduction/cnn_baseline/03_nmf/run_table2_nmf.py`
- **Configuration Changes:**
  ```python
  # Paper-compliant Deep Decoder (DD) Protocol:
  nettype = 'DD'                 # Under-parameterized Deep Decoder
  rank = 128                     # Matches paper's n_clusters = 128
  n_samples = 2048               # Matches paper's sample count
  epochs = 1000                  # Matches paper's iteration budget
  lr = 1e-2                      # Matches paper's learning rate
  ```
- **Validation Criteria:**
  - MNIST ReLU matches $7.6 \pm 0.2\text{ dB}$ (Paper: 7.6 dB).
  - CIFAR-10 ReLU matches $6.3 \pm 0.1\text{ dB}$ (Paper: 6.3 dB).
  - Side-by-side presentation of Deep Decoder (reproducing paper) vs. DIP U-Net (our higher-capacity extension).

---

### Batch 7: Table 1 – Learning Rate Schedulers Grid
- **Target Script:** `paper_reproduction/cnn_baseline/01_synthetic_decomposition/run_synthetic_suite.py`
- **Configuration Changes:**
  ```python
  n_repeats = 5                  # Increased from 2 to 5 random seeds
  # Proper decay hyperparameters:
  StepLR: step_size=250, gamma=0.5
  ExponentialLR: gamma=0.995
  ```
- **Validation Criteria:**
  - Mean and standard deviation across 5 seeds matching paper Table 1 bounds within $\pm 0.3\text{ dB}$.

---

## 3. Data Integrity & Artifact Preservation Standards

To ensure complete transparency and prevent accidental overwrites:
1. **Directory Versioning:** Previous baseline results will be archived with timestamped tags (e.g., `previous_baseline_metrics.json`), while updated rerun outputs will populate `metrics.json` and `metrics_rerun.json`.
2. **Configuration Archival:** Every rerun script will automatically save a complete `config_rerun.yaml` containing Git commit hash, dataset checksum, exact CLI arguments, random seed, and hardware environment.
3. **Checkpoints & Logs:** Checkpoints will be saved at intermediate intervals (every 500 epochs) to prevent data loss in case of interruption.
4. **Validation Verification:** No experiment will be marked as complete until its output files, plots, metrics JSON, and training logs are verified via automated hash and dimension checks.
