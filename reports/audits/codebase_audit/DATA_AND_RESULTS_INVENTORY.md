# DeepTensor++ — Data Assets & Experimental Results Inventory

> **Audit Date:** November 6, 2026  
> **Workspace Root:** `e:\Rakhal_Tensor`  
> **Total Repository Footprint:** ~7,175.13 MB (~7.01 GB, excluding `.git` and `.venv`)  
> **Permanent Retention Requirement:** All verified baseline and improved results must be preserved intact.

---

## 1. Storage Footprint Breakdown by Major Category

| Category | Storage Footprint | Percentage | Retention Priority |
|---|---|---|---|
| **Raw Datasets & Benchmarks (`data/`)** | 5,975.58 MB (5.83 GB) | 83.3% | **Permanent** |
| **Standardized Reproduction (`paper_reproduction/`)** | 360.33 MB | 5.0% | **Permanent (Core)** |
| **Historical Validation Checkpoints (`results/`)** | 254.82 MB | 3.5% | Secondary / Archive |
| **Ensemble Diagnostic Weights (`results_ensemble*/`)**| 330.72 MB | 4.6% | Candidate for Pruning |
| **Frozen Baselines (`RESULTS-BASELINE/`)** | 214.59 MB | 3.0% | **Permanent (Core)** |
| **Root Diagnostic Logs & PDFs** | 12.81 MB | 0.2% | Candidate for Cleanup |
| **Publication Visual Gallery (`PAPER_RESULTS/`)** | 9.70 MB | 0.1% | **Permanent** |
| **Historical Extensions (`previous_neurotensor...`)**| 4.19 MB | 0.1% | Reference Archive |
| **Intermediate Sparsity & Transformer Dumps** | 7.39 MB | 0.1% | Candidate for Pruning |
| **Core Source Code (`modules/`)** | 0.35 MB | <0.1% | **Permanent (Core)** |
| **Total** | **7,175.13 MB (~7.01 GB)** | **100.0%** | |

---

## 2. Complete Dataset Inventory

| Dataset Name | Source / Benchmark | File Format | File Path | Tensor Dimensions | Disk Size | Verified Status |
|---|---|---|---|---|---|---|
| **Weizmann Faces** | Weizmann Institute | `.mat` | `data/weizzman.mat` | $(100 \times 100 \times 100)$ | 5.37 MB | Verified (Figure 2) |
| **ICVL Hyperspectral** | Arad & Ben-Shahar | `.mat` | `data/icvl.mat` | $(31 \times 1024 \times 1024)$ | 11.36 MB | Verified (Figs 9, 13a) |
| **Yale Face B 3-Way** | Georghiades et al. | `.mat` | `data/yale_b.mat` | $(192 \times 168 \times 64)$ | 4.93 MB | Verified (Figs 10, 13b) |
| **CroppedYale Archive** | Extended Yale B | `.zip` & PGM | `data/CroppedYale.zip` & `data/CroppedYale/` | 38 subjects, 64 illumination | 20.15 MB (zip) | Raw archive source |
| **Cat Video** | Video CS Benchmark | `.mat` | `data/cat_video.mat` | $(64 \times 128 \times 128)$ | 0.78 MB | Verified (Figure 11) |
| **3D PET/CT Head Volume**| Public Medical Phant.| `.mat` | `data/pet.mat` | $(128 \times 128 \times 150)$ | 1.29 MB | Verified (Figure 12) |
| **MNIST Benchmark** | LeCun et al. | IDX / PyTorch | `data/MNIST/` | $28 \times 28$ digits (60k samples) | 11.50 MB | Verified (Table 2) |
| **CIFAR-10 Benchmark** | Krizhevsky et al. | Python batch | `data/cifar-10-batches-py/` & `data/cifar-10-python.tar.gz` | $32 \times 32 \times 3$ (50k train) | 170.50 MB | Verified (Table 2) |
| **Google Speech Commands**| Warden et al. | WAV / Archive | `data/SpeechCommands/` & `data/speech_commands_v0.02.tar.gz` | 105,829 1-sec WAV files | 2,428.9 MB | Verified (Table 3) |
| **Synthetic Low-Rank Data**| In-Memory Generator | NumPy/Torch | Generated dynamically | $N=64..256, R=2..30$ | In-Memory | Verified (Figs 3, 4, 6, 7; Tab 1) |

---

## 3. Exhaustive PyTorch Checkpoint Inventory (31 Files)

| Checkpoint Path | Model Architecture | Experiment Domain | File Size (MB) | Epoch / Step | Retention Status |
|---|---|---|---|---|---|
| `results/adaptive/checkpoint/checkpoint.pt` | DeepTensor + Adaptive Rank | checkpoint.pt | 42.16 MB | Final Checkpoint | **Archive (Historical)** |
| `results/adaptive_fixed/checkpoint/checkpoint.pt` | DeepTensor + Adaptive Rank | checkpoint.pt | 42.16 MB | Final Checkpoint | **Archive (Historical)** |
| `results/baseline/checkpoint/checkpoint.pt` | DeepTensor (DIP Conv2D/3D) | checkpoint.pt | 42.15 MB | Final Checkpoint | **Archive (Historical)** |
| `results/baseline_fixed/checkpoint/checkpoint.pt` | DeepTensor (DIP Conv2D/3D) | checkpoint.pt | 42.15 MB | Final Checkpoint | **Archive (Historical)** |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run/checkpoint/deeptensor_noise_30_improved_checkpoint.pt` | DeepTensor Yale Face B DIP | deeptensor_noise_30_improved_checkpoint.pt | 21.16 MB | Final Checkpoint | **Permanent Keep (Improved)** |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run/checkpoint/deeptensor_noise_60_improved_checkpoint.pt` | DeepTensor Yale Face B DIP | deeptensor_noise_60_improved_checkpoint.pt | 21.16 MB | Final Checkpoint | **Permanent Keep (Improved)** |
| `paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run/checkpoint/figure11_deeptensor_improved_checkpoint.pt` | DeepTensor Separable Video DIP | figure11_deeptensor_improved_checkpoint.pt | 15.98 MB | Final Checkpoint | **Permanent Keep (Improved)** |
| `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/checkpoint/figure12_deeptensor_improved_checkpoint.pt` | DeepTensor 3D Radon DIP | figure12_deeptensor_improved_checkpoint.pt | 15.24 MB | Final Checkpoint | **Permanent Keep (Improved)** |
| `paper_reproduction/cnn_baseline/08_ct_reconstruction/checkpoint/figure12_deeptensor_checkpoint.pt` | DeepTensor 3D Radon DIP | figure12_deeptensor_checkpoint.pt | 15.23 MB | Final Checkpoint | **Permanent Keep (Core)** |
| `RESULTS-BASELINE/figure12/checkpoint/figure12_deeptensor_checkpoint.pt` | DeepTensor 3D Radon DIP | figure12_deeptensor_checkpoint.pt | 15.23 MB | Final Checkpoint | **Permanent Keep (Baseline)** |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_4000_checkpoint.pt` | DeepTensor Yale Face B DIP | deeptensor_rank_4000_checkpoint.pt | 13.98 MB | Final Checkpoint | **Permanent Keep (Improved)** |
| `results/_validation_10_epoch/sparsity/checkpoint_l0.001.pt` | DeepTensor + Group Sparsity | checkpoint_l0.001.pt | 13.86 MB | Final Checkpoint | **Prunable (Temporary)** |
| `results/sparsity/checkpoint/checkpoint_l0.001.pt` | DeepTensor + Group Sparsity | checkpoint_l0.001.pt | 13.86 MB | Final Checkpoint | **Archive (Historical)** |
| `results/_archived_diagnostic_l0.001/checkpoint_l0.001.pt` | DeepTensor (DIP Conv2D/3D) | checkpoint_l0.001.pt | 13.86 MB | Final Checkpoint | **Prunable (Temporary)** |
| `results/sparsity/checkpoint/checkpoint.pt` | DeepTensor + Group Sparsity | checkpoint.pt | 13.85 MB | Final Checkpoint | **Archive (Historical)** |
| `results/_validation_10_epoch/adaptive/checkpoint.pt` | DeepTensor + Adaptive Rank | checkpoint.pt | 13.85 MB | Final Checkpoint | **Prunable (Temporary)** |
| `results/_validation_10_epoch/baseline/checkpoint.pt` | DeepTensor (DIP Conv2D/3D) | checkpoint.pt | 13.84 MB | Final Checkpoint | **Prunable (Temporary)** |
| `paper_reproduction/cnn_baseline/05_hyperspectral_denoising/checkpoint/figure9_deeptensor_checkpoint.pt` | DeepTensor ICVL HSI DIP | figure9_deeptensor_checkpoint.pt | 11.65 MB | Final Checkpoint | **Permanent Keep (Core)** |
| `RESULTS-BASELINE/figure9/checkpoint/figure9_deeptensor_checkpoint.pt` | DeepTensor ICVL HSI DIP | figure9_deeptensor_checkpoint.pt | 11.65 MB | Final Checkpoint | **Permanent Keep (Baseline)** |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/checkpoint/deeptensor_noise_30_checkpoint.pt` | DeepTensor Yale Face B DIP | deeptensor_noise_30_checkpoint.pt | 8.54 MB | Final Checkpoint | **Permanent Keep (Core)** |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/checkpoint/deeptensor_noise_60_checkpoint.pt` | DeepTensor Yale Face B DIP | deeptensor_noise_60_checkpoint.pt | 8.54 MB | Final Checkpoint | **Permanent Keep (Core)** |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/original_baseline/checkpoint/deeptensor_noise_30_checkpoint.pt` | DeepTensor Yale Face B DIP | deeptensor_noise_30_checkpoint.pt | 8.54 MB | Final Checkpoint | **Permanent Keep (Baseline)** |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/original_baseline/checkpoint/deeptensor_noise_60_checkpoint.pt` | DeepTensor Yale Face B DIP | deeptensor_noise_60_checkpoint.pt | 8.54 MB | Final Checkpoint | **Permanent Keep (Baseline)** |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_2000_checkpoint.pt` | DeepTensor Yale Face B DIP | deeptensor_rank_2000_checkpoint.pt | 8.00 MB | Final Checkpoint | **Permanent Keep (Improved)** |
| `paper_reproduction/cnn_baseline/07_video_compressive_sensing/checkpoint/figure11_deeptensor_checkpoint.pt` | DeepTensor Separable Video DIP | figure11_deeptensor_checkpoint.pt | 5.17 MB | Final Checkpoint | **Permanent Keep (Core)** |
| `paper_reproduction/cnn_baseline/07_video_compressive_sensing/original_baseline/checkpoint/figure11_deeptensor_checkpoint.pt` | DeepTensor Separable Video DIP | figure11_deeptensor_checkpoint.pt | 5.17 MB | Final Checkpoint | **Permanent Keep (Baseline)** |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_1000_checkpoint.pt` | DeepTensor Yale Face B DIP | deeptensor_rank_1000_checkpoint.pt | 5.01 MB | Final Checkpoint | **Permanent Keep (Improved)** |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_500_checkpoint.pt` | DeepTensor Yale Face B DIP | deeptensor_rank_500_checkpoint.pt | 3.51 MB | Final Checkpoint | **Permanent Keep (Improved)** |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_100_checkpoint.pt` | DeepTensor Yale Face B DIP | deeptensor_rank_100_checkpoint.pt | 2.31 MB | Final Checkpoint | **Permanent Keep (Improved)** |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_50_checkpoint.pt` | DeepTensor Yale Face B DIP | deeptensor_rank_50_checkpoint.pt | 2.16 MB | Final Checkpoint | **Permanent Keep (Improved)** |
| `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_10_checkpoint.pt` | DeepTensor Yale Face B DIP | deeptensor_rank_10_checkpoint.pt | 2.04 MB | Final Checkpoint | **Permanent Keep (Improved)** |

---

## 4. Metrics & Evaluation Catalogs

### 4.1 Primary Metrics JSON Files (Key Verified Results)
1. `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run/metrics_improved.json`:
   - 30% Noise: PSNR **30.86 dB**, SSIM **0.9148**
   - 60% Noise: PSNR **30.46 dB**, SSIM **0.9045**
2. `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/figure13b_improved_metrics.json`:
   - CUDA PARAFAC ALS: 26.19 dB (R=1000) $\to$ 26.19 dB (R=4000)
   - Improved DeepTensor: **28.50 dB** (R=1000) $\to$ **28.97 dB** (R=4000) (+2.78 dB over ALS)
3. `paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run/metrics.json`:
   - DeepTensor (64 frames): **25.32 dB** PSNR, **0.8037** SSIM (+1.83 dB over 2D TV 23.49 dB)
4. `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/figure12_improved_metrics.json`:
   - Filtered Backprojection: 11.23 dB
   - Algorithmic (Gradient Descent): 26.97 dB
   - 2D TV: 30.31 dB
   - 2D TV + CP: 27.06 dB
   - DeepTensor Improved: **30.55 dB** PSNR, **0.8302** SSIM
5. `paper_reproduction/cnn_baseline/01_synthetic_decomposition/synthetic_suite_metrics.json`:
   - Table 1 synthetic grid decomposition error across 16 matrix & tensor configurations.
6. `paper_reproduction/cnn_baseline/03_nmf/metrics.json`:
   - Table 2 NMF and Semi-NMF factorization error on MNIST and CIFAR-10 (7.9–9.5 dB).
7. `paper_reproduction/cnn_baseline/09_speech_time_frequency/metrics.json`:
   - Table 3 Google Speech Commands audio spectrogram NMF across 10 audio classes.

### 4.2 Per-Epoch CSV Traces
The repository maintains **21 CSV trace files** recording per-epoch loss, training time, PSNR, and SSIM progressions:
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run/history_noise_30.csv` (4000 epochs)
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run/history_noise_60.csv` (4000 epochs)
- `paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run/history_figure11_improved.csv` (10000 epochs)
- `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/history_figure12_improved.csv` (4000 epochs)

---

## 5. Visual Figures & Publications Panels (107 PNGs)

Key publication panels:
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/images/figure10_yale_tensor_denoising_improved_panel.png`: 4-column side-by-side comparison of 30% and 60% S&P noise denoising.
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/images/figure13b_reconstruction_panel.png`: Rank 10 to 4000 visual reconstruction panel.
- `paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run/images/figure11_video_cs_improved_panel.png`: 9-slice montage of 64 reconstructed video frames.
- `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/images/figure12_reconstruction_panel.png`: Axial, coronal, and sagittal orthogonal slices comparing Ground Truth, Radon measurements, FBP, Algorithmic, 2D TV, and DeepTensor.
- `PAPER_RESULTS/`: Central gallery preserving 107 publication-ready figures.

---

## 6. Research Documentation Inventory (71 Markdown Files)

Notable research documentation documents:
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/FIGURE10_IMPROVEMENT_REPORT.md`
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/FIGURE13B_IMPROVEMENT_REPORT.md`
- `paper_reproduction/cnn_baseline/07_video_compressive_sensing/FIGURE11_IMPROVEMENT_REPORT.md`
- `paper_reproduction/cnn_baseline/08_ct_reconstruction/FIGURE12_IMPROVEMENT_REPORT.md`
- `paper_reproduction/consolidated_results/CONSOLIDATED_CNN_BASELINE_REPORT.md`
- `PHASE1_VALIDATION_GATE_AUDIT.md`
- `PHASE1_PERFORMANCE_GAP_AUDIT.md`
- `PAPER_EXPERIMENT_REGISTER.md`

---

## 7. Permanent Retention vs Safe Cleanup Matrix

### Permanent Retention (NEVER Delete or Overwrite):
1. **Core Source Code**: `modules/` and `paper_reproduction/cnn_baseline/`
2. **Raw Datasets**: All files in `data/` (especially `.mat` files and archive tarballs)
3. **Approved Improved Checkpoints**: All 5 `.pt` files in `improved_run/` subfolders (Yale 30%, Yale 60%, Video CS, PET/CT, Yale Rank 4000)
4. **Verified Baseline Checkpoints**: `RESULTS-BASELINE/figure9/checkpoint/` and `RESULTS-BASELINE/figure12/checkpoint/`
5. **All Metrics JSONs & Per-Epoch CSVs**: `paper_reproduction/cnn_baseline/**/metrics*.json` and `history*.csv`
6. **Publication Panels & Plots**: `paper_reproduction/cnn_baseline/**/images/` and `plots/`
7. **Audit & Improvement Reports**: All `.md` reports in `paper_reproduction/` and root.

### Safe to Clean Up or Prune During Reorganization:
1. **Duplicate Root Scripts**: 9 `run_figure9*.py` files in root (identical copies exist in extensions)
2. **Duplicate Subpackage Modules**: `previous_neurotensor_experiments/*/modules/*.py`
3. **Backup Demo Files**: `demo_backup.py` and `demo_backend_backup.py`
4. **Diagnostic Smoke Test Checkpoints**: `results_ensemble_smoke/` (231.3 MB) and `results/_validation_10_epoch/` (41.5 MB)
5. **Raw Text Logs in Root**: `results_lambda_*.txt` (~10 MB)
6. **Empty Result Directories**: `results_completion*/`, `results_robustness/` (0 bytes)
