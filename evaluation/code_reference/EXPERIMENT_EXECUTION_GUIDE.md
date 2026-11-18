# DeepTensor — Experiment Execution Guide

**Document ID:** EVAL-GUIDE-01  
**Scope:** Exact Execution Commands, Model Imports, Datasets, Parameters, and Output Locations for Every Presentation Experiment

---

## 1. Overview & General Execution Protocol

All experiments are executed from the repository root: `E:\Rakhal_Tensor`.  
The virtual environment should be activated prior to execution:
```powershell
.venv\Scripts\Activate.ps1
```
Ensure the Python path includes the workspace root:
```powershell
$env:PYTHONPATH = "E:\Rakhal_Tensor;E:\Rakhal_Tensor\src"
```

---

## 2. Experiment-by-Experiment Reference

### Experiment 1: Figure 09 — Hyperspectral Image Denoising (Baseline Reproduction)
- **Supporting Slides:** Slide 8, Slide 10, Slide 12
- **Runner Script:** `experiments/reproduction/figure09_hyperspectral/run_figure9.py`
- **Model Files Imported:** 
  - `src/deeptensor/models/deeptensor.py` (`DeepTensor`)
  - `src/deeptensor/models/backbones/skip.py` (`skip`)
  - `src/deeptensor/baselines/spectral.py` (`svd_decompose`)
- **Dataset / Inputs Required:** `data/icvl.mat` ($174 \times 164 \times 260$ spatial-spectral cube)
- **Configuration Parameters:**
  - Rank: $R = 20$
  - Epochs: 5000
  - Optimizer: Adam, Learning Rate $= 10^{-3}$
  - Measurement Noise: Poisson ($\tau = 100$) + Additive Gaussian ($\text{SNR} = 2.0$)
  - Objective: Measurement $\mathcal{L}_1$ loss
  - Input Perturbation: Uniform noise $Z \sim \mathcal{U}(-0.1, 0.1)$, $\sigma_\text{noise} = 0.03$
- **Output Directory:** `results/baseline/figure09_hyperspectral/`
  - Output files: `metrics.json`, `figure9_reconstruction.png`, `training_loss.csv`, `checkpoint.pt`
- **Execution Command:**
  ```powershell
  python experiments/reproduction/figure09_hyperspectral/run_figure9.py
  ```

---

### Experiment 2: Figure 10 — Yale Face B Denoising (Baseline & Improved Rerun)
- **Supporting Slides:** Slide 8, Slide 10, Slide 12, Slide 14
- **Runner Scripts:**
  - Standard Baseline: `experiments/reproduction/figure10_yale_denoising/run_figure10_yale_tensor.py`
  - Extended Improved Rerun: `experiments/improvements/figure10_yale/run_figure10_yale_tensor_improved.py`
- **Model Files Imported:**
  - `src/deeptensor/models/deeptensor.py` (`DeepTensorCP`)
  - `src/deeptensor/models/backbones/skip.py` (`skip1d`)
- **Dataset / Inputs Required:** `data/yale_b.mat` (Extended Yale Face Database B, $192 \times 168 \times 160$)
- **Configuration Parameters:**
  - Canonical CP Rank: $R = 2000$
  - Standard Baseline Budget: 1200 epochs (64 channels per skip scale)
  - Improved Extended Budget: 5000 epochs (128 channels per skip scale, 5.47M params)
  - Noise: 30% and 60% Salt & Pepper impulse noise
  - Objective: Measurement $\mathcal{L}_1$ loss
- **Output Directories:**
  - Standard Baseline: `results/baseline/figure10_yale_denoising/`
  - Improved Rerun: `results/improved/figure10_yale_denoising/`
    - Output files: `metrics_noise_30.json`, `metrics_noise_60.json`, `figure10_consolidated_comparison.json`, `progression_noise_30.csv`, `progression_noise_60.csv`, `images/figure10_yale_tensor_denoising_improved_panel.png`
- **Execution Commands:**
  ```powershell
  # 30% Noise Extended Rerun (30.86 dB)
  python experiments/improvements/figure10_yale/run_figure10_yale_tensor_improved.py --noise_level 0.3 --epochs 5000
  
  # 60% Noise Extended Rerun (30.46 dB)
  python experiments/improvements/figure10_yale/run_figure10_yale_tensor_improved.py --noise_level 0.6 --epochs 5000
  ```

---

### Experiment 3: Figure 11 — Video Compressive Sensing (CACTI Reconstruction)
- **Supporting Slides:** Slide 9, Slide 11, Slide 12, Slide 17, Slide 18
- **Runner Script:** `experiments/hybrid_prototypes/run_hybrid_figure11_video.py`
- **Model Files Imported:**
  - `src/deeptensor/models/factory.py` (`build_deeptensor_cp_model`)
  - `src/deeptensor/physics/operators.py` (`get_video_coding_frames`, `video2codedvideo`)
  - `src/deeptensor/models/attention/spatial_spectral.py`
  - `src/deeptensor/models/attention/spectral_transformer.py`
- **Dataset / Inputs Required:** `data/cat_video.mat` ($128 \times 128 \times 64$ grayscale video tensor)
- **Configuration Parameters:**
  - CP Decomposition Rank: $R = 32$
  - CACTI Compression: 8-frame temporal multiplexing ($8\times$ compression ratio)
  - Optical Mask: Pseudorandom binary shifting mask $M \in \{0, 1\}^{128 \times 128 \times 8}$
  - Measurement Noise: Poisson ($\tau = 100$) + Gaussian ($\text{SNR} = 2.0$)
  - Optimization Budget: 3000 epochs
  - Objective: Measurement $\mathcal{L}_2$ (Frobenius) loss
- **Output Directory:** `results/hybrid/figure11_video_cs/`
  - Output files: `figure11_video_cs_comparison.json`, `figure11_video_cs_comparison_panel_annotated.png`, `recon_*.npy`
- **Execution Command:**
  ```powershell
  python experiments/hybrid_prototypes/run_hybrid_figure11_video.py --epochs 3000
  ```

---

### Experiment 4: Figure 12 — 3D PET/CT Radon Compressive Sensing
- **Supporting Slides:** Slide 9, Slide 11, Slide 12, Slide 20, Slide 21
- **Runner Script:** `experiments/hybrid_prototypes/run_hybrid_figure12_pet.py`
- **Model Files Imported:**
  - `src/deeptensor/models/factory.py` (`build_deeptensor_cp_model`)
  - `src/deeptensor/physics/operators.py` (`radon`)
  - `src/deeptensor/models/attention/spatial_spectral.py`
- **Dataset / Inputs Required:** `data/pet.mat` ($128 \times 128 \times 56$ 3D CT volume)
- **Configuration Parameters:**
  - CP Decomposition Rank: $R = 1000$
  - Forward Projection: Parallel-beam Radon transform across 40 equi-spaced angles in $[0, \pi)$
  - Sinogram Dimension: $40 \times 128 \times 56$
  - Measurement Noise: Poisson ($\tau = 100$) + Gaussian ($\text{SNR} = 2.0$)
  - Optimization Budget: 4000 epochs
  - Objective: Measurement $\mathcal{L}_2$ (Frobenius) loss
- **Output Directory:** `results/hybrid/figure12_pet_ct/`
  - Output files: `figure12_pet_ct_comparison.json`, `figure12_pet_ct_comparison_panel_annotated.png`, `recon_*.npy`
- **Execution Command:**
  ```powershell
  python experiments/hybrid_prototypes/run_hybrid_figure12_pet.py --epochs 4000
  ```

---

### Experiment 5: Five-Architecture Controlled Benchmark (All 4 Applications)
- **Supporting Slides:** Slide 13, Slide 14, Slide 15, Slide 17, Slide 19, Slide 20, Slide 22
- **Runner Scripts:**
  - HSI Suite: `experiments/hybrid_prototypes/run_hybrid_figure09_hyperspectral.py`
  - Yale Suite: `experiments/hybrid_prototypes/run_hybrid_figure10_yale.py`
  - Video Suite: `experiments/hybrid_prototypes/run_hybrid_figure11_video.py`
  - PET/CT Suite: `experiments/hybrid_prototypes/run_hybrid_figure12_pet.py`
- **Architectures Evaluated:**
  1. `baseline` (Baseline CNN)
  2. `hybrid` (Hybrid CNN–Transformer)
  3. `cnn_control` (Parameter-Matched CNN Control)
  4. `pointwise_control` (Pointwise Control)
  5. `transformer` (Pure Spectral Transformer)
- **Controlled Invariants:**
  - Identical parameter capacity ($\pm 0.05\%$ matching via `count_parameters`)
  - Identical random seeds (`seed=42`)
  - Identical measurement operators and noise realisations
  - Identical training epoch budgets
- **Execution Commands:**
  ```powershell
  python experiments/hybrid_prototypes/run_hybrid_figure09_hyperspectral.py --epochs 5000
  python experiments/hybrid_prototypes/run_hybrid_figure10_yale.py --epochs 1200
  python experiments/hybrid_prototypes/run_hybrid_figure11_video.py --epochs 3000
  python experiments/hybrid_prototypes/run_hybrid_figure12_pet.py --epochs 4000
  ```

---

### Experiment 6: Adaptive Rank Selection
- **Supporting Slides:** Slide 23, Slide 24
- **Runner Script:** `experiments/extensions/adaptive_rank/run_figure9_adaptive.py`
- **Modules Imported:**
  - `src/deeptensor/training/adaptive_rank.py` (`AdaptiveRankManager`)
  - `src/deeptensor/models/deeptensor.py` (`DeepTensor`)
- **Dataset / Inputs Required:** `data/icvl.mat`
- **Configuration Parameters:**
  - Initial Rank: $R_\text{start} = 4$
  - Maximum Rank: $R_\text{max} = 30$
  - Rank Increment: $R_\text{step} = 2$
  - EMA Smoothing Alpha: $\alpha = 0.99$
  - Monitoring Window: 200 epochs
  - Plateau Threshold: $\epsilon = 0.001$
- **Output Directory:** `results/adaptive_rank_selection/`
  - Output files: `active_rank_evolution.png`, `summary.txt`, `adaptive_rank_metrics.json`
- **Execution Command:**
  ```powershell
  python experiments/extensions/adaptive_rank/run_figure9_adaptive.py --r_start 4 --r_max 30 --r_step 2 --epochs 5000
  ```

---

### Experiment 7: Factor Group-Sparsity (Group Lasso Regularization)
- **Supporting Slides:** Slide 25, Slide 26
- **Runner Script:** `experiments/extensions/group_sparsity/run_figure9_sparsity.py`
- **Modules Imported:**
  - `experiments/extensions/group_sparsity/run_figure9_sparsity.py` (`compute_group_regularizer`)
  - `src/deeptensor/models/deeptensor.py`
- **Dataset / Inputs Required:** `data/icvl.mat`
- **Configuration Parameters:**
  - Maximum Capacity: $R_\text{max} = 30$
  - Regularization Weights Sweep: $\lambda \in \{0.0, 10^{-4}, 10^{-3}, 10^{-2}\}$
  - Validation: 5% held-out blind-spot mask (`seed=42`)
  - Optimization Budget: 5000 epochs
  - Total Loss: $\mathcal{L}_\text{total} = \mathcal{L}_\text{recon} + \lambda \sum_{r=1}^{R_\text{max}} \sqrt{\|U_{:,r}\|_2^2 + \|V_{r,:}\|_2^2}$
- **Output Directories:**
  - Grid Sweep: `results/extensions/sparsity/standard/`
  - Canonical Slide 26 Snapshot: `results/factor_group-sparsity/`
    - Output files: `metrics.json`, `metrics.csv`, `reconstruction_comparison.png`, `error_map.png`, `network_summary.txt`
- **Execution Command:**
  ```powershell
  python experiments/extensions/group_sparsity/run_figure9_sparsity.py --lambda_value 0.0001 --r_max 30 --epochs 5000
  ```

