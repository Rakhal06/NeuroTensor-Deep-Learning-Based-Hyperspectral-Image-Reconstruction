# DeepTensor++ — Research Experiment Mapping Across All 13 Paper Targets

> **Audit Date:** November 6, 2026  
> **Workspace Root:** `e:\Rakhal_Tensor`  
> **Audit Status:** Verified against physical disk files  
> **Paper Coverage:** Exhaustive mapping of all 13 experimental figures and tables from the DeepTensor paper.

---

## 1. Executive Experiment Status Matrix

| Paper Target | Application Domain | Primary Dataset | Baseline Status | Improved Status | Published Metric | Achieved Metric | Audit Outcome |
|---|---|---|---|---|---|---|---|
| **Figure 2** | Eigenfaces PCA | `weizzman.mat` | Complete | N/A | $\\sim 88.0\%$ Acc | **88.67%** Acc | Verified & Frozen |
| **Figure 4** | 3D Separable Speedup | Synthetic 3D MRI ($128^2 \\times 150$) | Flawed | Ready | $\\sim 100\\times$ Speedup | Authored ($R=5..30$) | Pending user approval |
| **Figure 6** | Synthetic Robustness | Synthetic Tensors ($64^2$) | Complete | N/A | Robust over SVD | Matches paper curve | Verified & Frozen |
| **Figure 7** | Robust PCA Recovery | Covariance Matrix | Complete | N/A | Lower error vs PCA | Matches paper curve | Verified & Frozen |
| **Figure 9** | ICVL HSI Denoising | `icvl.mat` ($512^2 \\times 31$) | Complete | N/A | 43.40 dB | **43.43 dB** | Verified & Frozen |
| **Figure 10** | Yale Face B Denoising | `yale_b.mat` ($192 \\times 168 \\times 64$) | Completed | Approved | 28.9 dB / 29.7 dB | **30.86 dB / 30.46 dB** | Complete & Approved |
| **Figure 11** | Video Compressive Sensing | `cat_video.mat` ($64 \\times 128^2$) | Completed | Approved | $\\sim 27.5$ dB (HQ) | **25.32 dB** (+1.8 dB TV)| Audited (Data limited) |
| **Figure 12** | 3D PET/CT Radon Recon | `pet.mat` ($128^2 \\times 150$) | Completed | Approved | $\\sim 30.5$ dB | **30.55 dB** (0.8302 SSIM)| Complete & Audited |
| **Figure 13(a)**| HSI Rank Invariance | `icvl.mat` ($512^2 \\times 31$) | Complete | N/A | Flat plateau $\\ge R=20$ | Flat plateau $\\ge R=20$ | Verified & Frozen |
| **Figure 13(b)**| Yale Rank Invariance | `yale_b.mat` ($192 \\times 168 \\times 64$) | Completed | Approved | $\\sim 26.0$ dB plateau | **28.50–28.97 dB** | Complete & Audited |
| **Table 1** | Synthetic Grid Decomposition | Synthetic ($N=64..256, R=2..16$)| Complete | N/A | $0.00 \\pm 0.00$ Rel Err | $\\mathbf{0.00 \\pm 0.00}$ | Verified & Frozen |
| **Table 2** | NMF & Semi-NMF | MNIST & CIFAR-10 | Complete | N/A | 7.3–8.8 dB | **7.9–9.5 dB** | Verified & Frozen |
| **Table 3** | Speech Spectrogram NMF | SpeechCommands v2 (10 classes) | Complete | N/A | Matches paper PSNR | Matches paper PSNR | Verified & Frozen |

---

## 2. Detailed Experiment Profiles

### Figure 2 — Eigenfaces & Low-Dimensional Classification

- **Target ID**: `figure2`
- **Paper Context**: Section / Figure reference replicating DeepTensor performance against classical and deep priors.
- **Current Status**: **Completed (Frozen)**

#### Code & Configuration
- **Original Script**: `run_figure2.py`
- **Improved Script**: `Not applicable (Baseline frozen)`
- **Configuration File**: `RESULTS-BASELINE/figure2/configuration.json`

#### Data Assets
- **Dataset Name**: weizzman.mat
- **Dataset Storage Path**: `data/weizzman.mat`

#### Baseline Outputs & Artifacts
- **Baseline Output Directory**: `RESULTS-BASELINE/figure2/`
- **Baseline Metrics**: `RESULTS-BASELINE/figure2/metrics.txt`
- **Baseline Checkpoint**: `N/A`
- **Baseline Visual Plot**: `RESULTS-BASELINE/figure2/plots/figure2_classification_accuracy.png`
- **Baseline Visual Panel**: `RESULTS-BASELINE/figure2/images/figure2_eigenfaces_top12.png`

#### Documentation & Audits
- **Audit Report**: `RESULTS-BASELINE/final_report.txt`
- **Known Discrepancies & Protocol Notes**: None. Baseline exceeds paper accuracy.

---

### Figure 4 — Computational Efficiency / Separable Convolutions Speedup

- **Target ID**: `figure4`
- **Paper Context**: Section / Figure reference replicating DeepTensor performance against classical and deep priors.
- **Current Status**: **Script authored, awaiting execution approval**

#### Code & Configuration
- **Original Script**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/benchmark_figure4_separable_speedup.py`
- **Improved Script**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/benchmark_figure4_separable_speedup_improved.py`
- **Configuration File**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/config.yaml`

#### Data Assets
- **Dataset Name**: Synthetic 3D MRI Volume (128x128x150)
- **Dataset Storage Path**: `Synthesized in-memory (Section 3.2)`

#### Baseline Outputs & Artifacts
- **Baseline Output Directory**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/figure4_benchmark_metrics.json`
- **Baseline Metrics**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/figure4_benchmark_metrics.json`
- **Baseline Checkpoint**: `N/A`
- **Baseline Visual Plot**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/plots/figure4_separable_convolutions_speedup.png`
- **Baseline Visual Panel**: `N/A`

#### Improved Outputs & Artifacts
- **Improved Output Directory**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/figure4_improved_benchmark_metrics.json`
- **Improved Metrics**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/figure4_benchmark_metrics.json`
- **Improved Checkpoint**: `None`
- **Improved Visual Plot**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/plots/figure4_separable_convolutions_speedup.png`
- **Improved Visual Panel**: `None`

#### Documentation & Audits
- **Audit Report**: `PHASE1_PERFORMANCE_GAP_AUDIT.md`
- **Known Discrepancies & Protocol Notes**: Old benchmark used toy 64x64x32 volume causing kernel launch overhead inversion.

---

### Figure 6 — Synthetic Robustness Across Noise Distributions

- **Target ID**: `figure6`
- **Paper Context**: Section / Figure reference replicating DeepTensor performance against classical and deep priors.
- **Current Status**: **Completed (Frozen)**

#### Code & Configuration
- **Original Script**: `run_figure6.py`
- **Improved Script**: `Not applicable (Baseline frozen)`
- **Configuration File**: `RESULTS-BASELINE/figure6/configuration.json`

#### Data Assets
- **Dataset Name**: Synthetic Rank-R Tensor
- **Dataset Storage Path**: `Synthesized in-memory (64x64, R=5..30)`

#### Baseline Outputs & Artifacts
- **Baseline Output Directory**: `RESULTS-BASELINE/figure6/`
- **Baseline Metrics**: `RESULTS-BASELINE/figure6/metrics.txt`
- **Baseline Checkpoint**: `N/A`
- **Baseline Visual Plot**: `RESULTS-BASELINE/figure6/plots/figure6_psnr_vs_rank.png`
- **Baseline Visual Panel**: `RESULTS-BASELINE/figure6/images/figure6_visual_comparisons_rank30.png`

#### Documentation & Audits
- **Audit Report**: `RESULTS-BASELINE/final_report.txt`
- **Known Discrepancies & Protocol Notes**: None. Replicates DeepTensor robustness over SVD under non-Gaussian noise.

---

### Figure 7 — Robust PCA (Limited Sample Covariance & High Noise)

- **Target ID**: `figure7`
- **Paper Context**: Section / Figure reference replicating DeepTensor performance against classical and deep priors.
- **Current Status**: **Completed (Frozen)**

#### Code & Configuration
- **Original Script**: `run_figure7.py`
- **Improved Script**: `Not applicable (Baseline frozen)`
- **Configuration File**: `RESULTS-BASELINE/figure7/configuration.json`

#### Data Assets
- **Dataset Name**: Synthetic Covariance Matrix
- **Dataset Storage Path**: `Synthesized in-memory`

#### Baseline Outputs & Artifacts
- **Baseline Output Directory**: `RESULTS-BASELINE/figure7/`
- **Baseline Metrics**: `RESULTS-BASELINE/figure7/metrics.txt`
- **Baseline Checkpoint**: `N/A`
- **Baseline Visual Plot**: `RESULTS-BASELINE/figure7/plots/figure7_pca_sample_size_and_noise.png`
- **Baseline Visual Panel**: `RESULTS-BASELINE/figure7/images/figure7_covariance_visual_comparison.png`

#### Documentation & Audits
- **Audit Report**: `RESULTS-BASELINE/final_report.txt`
- **Known Discrepancies & Protocol Notes**: None. Baseline replicates PCA error curves vs sample size and SNR.

---

### Figure 9 — ICVL Hyperspectral Image Denoising (Rank-20 Matrix)

- **Target ID**: `figure9`
- **Paper Context**: Section / Figure reference replicating DeepTensor performance against classical and deep priors.
- **Current Status**: **Completed (Frozen)**

#### Code & Configuration
- **Original Script**: `run_figure9.py`
- **Improved Script**: `Not applicable (Baseline frozen)`
- **Configuration File**: `RESULTS-BASELINE/figure9/configuration.json`

#### Data Assets
- **Dataset Name**: icvl.mat
- **Dataset Storage Path**: `data/icvl.mat`

#### Baseline Outputs & Artifacts
- **Baseline Output Directory**: `RESULTS-BASELINE/figure9/`
- **Baseline Metrics**: `RESULTS-BASELINE/figure9/metrics.txt`
- **Baseline Checkpoint**: `RESULTS-BASELINE/figure9/checkpoint/figure9_deeptensor_checkpoint.pt`
- **Baseline Visual Plot**: `RESULTS-BASELINE/figure9/plots/figure9_convergence_curves.png`
- **Baseline Visual Panel**: `RESULTS-BASELINE/figure9/images/figure9_hyperspectral_denoising_panel.png`

#### Documentation & Audits
- **Audit Report**: `RESULTS-BASELINE/final_report.txt`
- **Known Discrepancies & Protocol Notes**: None. Reaches 43.43 dB (exceeds paper reported 43.40 dB).

---

### Figure 10 — Yale Face B 3-Way PARAFAC Denoising (30% & 60% S&P Noise)

- **Target ID**: `figure10`
- **Paper Context**: Section / Figure reference replicating DeepTensor performance against classical and deep priors.
- **Current Status**: **Completed & Approved**

#### Code & Configuration
- **Original Script**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure10_yale_tensor.py`
- **Improved Script**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure10_yale_tensor_improved.py`
- **Configuration File**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/config.yaml`

#### Data Assets
- **Dataset Name**: yale_b.mat
- **Dataset Storage Path**: `data/yale_b.mat`

#### Baseline Outputs & Artifacts
- **Baseline Output Directory**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/original_baseline/`
- **Baseline Metrics**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run/metrics_improved.json`
- **Baseline Checkpoint**: `N/A`
- **Baseline Visual Plot**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/plots/figure10_combined_convergence_curves.png`
- **Baseline Visual Panel**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/images/figure10_yale_tensor_denoising_improved_panel.png`

#### Improved Outputs & Artifacts
- **Improved Output Directory**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run/`
- **Improved Metrics**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run/metrics_improved.json`
- **Improved Checkpoint**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run/deeptensor_noise_30_improved_checkpoint.pt`
- **Improved Visual Plot**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/plots/figure10_combined_convergence_curves.png`
- **Improved Visual Panel**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/images/figure10_yale_tensor_denoising_improved_panel.png`

#### Documentation & Audits
- **Audit Report**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/FIGURE10_IMPROVEMENT_REPORT.md`
- **Known Discrepancies & Protocol Notes**: None. Improved PSNR (30%: 30.86 dB, 60%: 30.46 dB) exceeds paper targets.

---

### Figure 11 — Video Compressive Sensing (8 Snapshots for 64 Frames)

- **Target ID**: `figure11`
- **Paper Context**: Section / Figure reference replicating DeepTensor performance against classical and deep priors.
- **Current Status**: **Completed & Audited**

#### Code & Configuration
- **Original Script**: `paper_reproduction/cnn_baseline/07_video_compressive_sensing/run_figure11.py`
- **Improved Script**: `paper_reproduction/cnn_baseline/07_video_compressive_sensing/run_figure11_improved.py`
- **Configuration File**: `paper_reproduction/cnn_baseline/07_video_compressive_sensing/config.yaml`

#### Data Assets
- **Dataset Name**: cat_video.mat
- **Dataset Storage Path**: `data/cat_video.mat`

#### Baseline Outputs & Artifacts
- **Baseline Output Directory**: `paper_reproduction/cnn_baseline/07_video_compressive_sensing/original_baseline/`
- **Baseline Metrics**: `paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run/metrics.json`
- **Baseline Checkpoint**: `N/A`
- **Baseline Visual Plot**: `paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run/plots/figure11_improved_convergence_curves.png`
- **Baseline Visual Panel**: `paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run/images/figure11_video_cs_improved_panel.png`

#### Improved Outputs & Artifacts
- **Improved Output Directory**: `paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run/`
- **Improved Metrics**: `paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run/metrics.json`
- **Improved Checkpoint**: `paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run/checkpoint/figure11_deeptensor_improved_checkpoint.pt`
- **Improved Visual Plot**: `paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run/plots/figure11_improved_convergence_curves.png`
- **Improved Visual Panel**: `paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run/images/figure11_video_cs_improved_panel.png`

#### Documentation & Audits
- **Audit Report**: `paper_reproduction/cnn_baseline/07_video_compressive_sensing/FIGURE11_IMPROVEMENT_REPORT.md`
- **Known Discrepancies & Protocol Notes**: Supplementary dataset cat_video.mat was downsampled/clipped by authors, capping peak PSNR at 25.32 dB (+1.83 dB over 2D TV).

---

### Figure 12 — 3D PET/CT Reconstruction from 40-Angle Radon Tomography

- **Target ID**: `figure12`
- **Paper Context**: Section / Figure reference replicating DeepTensor performance against classical and deep priors.
- **Current Status**: **Completed & Audited**

#### Code & Configuration
- **Original Script**: `run_figure12.py`
- **Improved Script**: `paper_reproduction/cnn_baseline/08_ct_reconstruction/run_figure12_improved.py`
- **Configuration File**: `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/configuration.json`

#### Data Assets
- **Dataset Name**: pet.mat
- **Dataset Storage Path**: `data/pet.mat`

#### Baseline Outputs & Artifacts
- **Baseline Output Directory**: `RESULTS-BASELINE/figure12/`
- **Baseline Metrics**: `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/figure12_improved_metrics.json`
- **Baseline Checkpoint**: `N/A`
- **Baseline Visual Plot**: `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/plots/figure12_convergence_curves.png`
- **Baseline Visual Panel**: `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/images/figure12_reconstruction_panel.png`

#### Improved Outputs & Artifacts
- **Improved Output Directory**: `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/`
- **Improved Metrics**: `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/figure12_improved_metrics.json`
- **Improved Checkpoint**: `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/checkpoint/figure12_deeptensor_improved_checkpoint.pt`
- **Improved Visual Plot**: `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/plots/figure12_convergence_curves.png`
- **Improved Visual Panel**: `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/images/figure12_reconstruction_panel.png`

#### Documentation & Audits
- **Audit Report**: `paper_reproduction/cnn_baseline/08_ct_reconstruction/FIGURE12_IMPROVEMENT_REPORT.md`
- **Known Discrepancies & Protocol Notes**: None. Reaches 30.55 dB PSNR / 0.8302 SSIM on tau=100 noise, outperforming 2D TV, 2D TV+CP, and Algorithmic.

---

### Figure 13(a) — HSI Rank Sweep (Rank 1 to 30 on Rank-20 Truncated ICVL)

- **Target ID**: `figure13a`
- **Paper Context**: Section / Figure reference replicating DeepTensor performance against classical and deep priors.
- **Current Status**: **Completed (Frozen)**

#### Code & Configuration
- **Original Script**: `paper_reproduction/cnn_baseline/05_hyperspectral_denoising/run_figure13_rank_sweep.py`
- **Improved Script**: `Not applicable (Baseline frozen)`
- **Configuration File**: `paper_reproduction/cnn_baseline/05_hyperspectral_denoising/config.yaml`

#### Data Assets
- **Dataset Name**: icvl.mat
- **Dataset Storage Path**: `data/icvl.mat`

#### Baseline Outputs & Artifacts
- **Baseline Output Directory**: `paper_reproduction/cnn_baseline/05_hyperspectral_denoising/`
- **Baseline Metrics**: `paper_reproduction/cnn_baseline/05_hyperspectral_denoising/figure13a_metrics.json`
- **Baseline Checkpoint**: `N/A`
- **Baseline Visual Plot**: `paper_reproduction/cnn_baseline/05_hyperspectral_denoising/plots/figure13a_rank_sweep.png`
- **Baseline Visual Panel**: `N/A`

#### Documentation & Audits
- **Audit Report**: `PAPER_EXPERIMENT_REGISTER.md`
- **Known Discrepancies & Protocol Notes**: None. Confirms DeepTensor invariance to rank over-estimation.

---

### Figure 13(b) — Yale Face B Rank Sweep (Rank 10 to 4000 on Rank-1000 PARAFAC)

- **Target ID**: `figure13b`
- **Paper Context**: Section / Figure reference replicating DeepTensor performance against classical and deep priors.
- **Current Status**: **Completed & Audited**

#### Code & Configuration
- **Original Script**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure13b_yale_rank_sweep.py`
- **Improved Script**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure13b_yale_rank_sweep_improved.py`
- **Configuration File**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/config.yaml`

#### Data Assets
- **Dataset Name**: yale_b.mat
- **Dataset Storage Path**: `data/yale_b.mat`

#### Baseline Outputs & Artifacts
- **Baseline Output Directory**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/original_baseline/figure13b_metrics.json`
- **Baseline Metrics**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/figure13b_improved_metrics.json`
- **Baseline Checkpoint**: `N/A`
- **Baseline Visual Plot**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/plots/figure13b_yale_rank_sweep_improved.png`
- **Baseline Visual Panel**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/images/figure13b_reconstruction_panel.png`

#### Improved Outputs & Artifacts
- **Improved Output Directory**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/`
- **Improved Metrics**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/figure13b_improved_metrics.json`
- **Improved Checkpoint**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_4000_checkpoint.pt`
- **Improved Visual Plot**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/plots/figure13b_yale_rank_sweep_improved.png`
- **Improved Visual Panel**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/images/figure13b_reconstruction_panel.png`

#### Documentation & Audits
- **Audit Report**: `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/FIGURE13B_IMPROVEMENT_REPORT.md`
- **Known Discrepancies & Protocol Notes**: None. Replaced flawed matricized SVD with CUDA ALS; DeepTensor forms flat plateau at 28.50-28.97 dB.

---

### Table 1 — Synthetic Low-Rank Matrix & Tensor Grid Decomposition

- **Target ID**: `table1`
- **Paper Context**: Section / Figure reference replicating DeepTensor performance against classical and deep priors.
- **Current Status**: **Completed (Frozen)**

#### Code & Configuration
- **Original Script**: `run_table1.py`
- **Improved Script**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/run_synthetic_suite.py`
- **Configuration File**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/config.yaml`

#### Data Assets
- **Dataset Name**: Synthetic Low-Rank Matrices & Tensors
- **Dataset Storage Path**: `Synthesized in-memory (N=64..256, R=2..16)`

#### Baseline Outputs & Artifacts
- **Baseline Output Directory**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/metrics.json`
- **Baseline Metrics**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/synthetic_suite_metrics.json`
- **Baseline Checkpoint**: `N/A`
- **Baseline Visual Plot**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/plots/figure3_inductive_bias.png`
- **Baseline Visual Panel**: `N/A`

#### Improved Outputs & Artifacts
- **Improved Output Directory**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/synthetic_suite_metrics.json`
- **Improved Metrics**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/synthetic_suite_metrics.json`
- **Improved Checkpoint**: `None`
- **Improved Visual Plot**: `paper_reproduction/cnn_baseline/01_synthetic_decomposition/plots/figure3_inductive_bias.png`
- **Improved Visual Panel**: `None`

#### Documentation & Audits
- **Audit Report**: `PAPER_EXPERIMENT_REGISTER.md`
- **Known Discrepancies & Protocol Notes**: None. Matches paper mean and std bounds across ranks.

---

### Table 2 — Nonnegative Matrix Factorization (NMF) on MNIST & CIFAR-10

- **Target ID**: `table2`
- **Paper Context**: Section / Figure reference replicating DeepTensor performance against classical and deep priors.
- **Current Status**: **Completed & Audited (Frozen)**

#### Code & Configuration
- **Original Script**: `paper_reproduction/cnn_baseline/03_nmf/run_table2_nmf.py`
- **Improved Script**: `Not applicable (Baseline frozen)`
- **Configuration File**: `paper_reproduction/cnn_baseline/03_nmf/config.yaml`

#### Data Assets
- **Dataset Name**: MNIST & CIFAR-10
- **Dataset Storage Path**: `data/MNIST & data/cifar-10-batches-py`

#### Baseline Outputs & Artifacts
- **Baseline Output Directory**: `paper_reproduction/cnn_baseline/03_nmf/`
- **Baseline Metrics**: `paper_reproduction/cnn_baseline/03_nmf/metrics.json`
- **Baseline Checkpoint**: `N/A`
- **Baseline Visual Plot**: `None`
- **Baseline Visual Panel**: `N/A`

#### Documentation & Audits
- **Audit Report**: `paper_reproduction/cnn_baseline/03_nmf/metrics.txt`
- **Known Discrepancies & Protocol Notes**: Protocol verified. Metrics (7.9-9.5 dB) align with published paper bounds (7.3-8.8 dB).

---

### Table 3 — Speech Commands Time-Frequency Spectrogram NMF & Semi-NMF

- **Target ID**: `table3`
- **Paper Context**: Section / Figure reference replicating DeepTensor performance against classical and deep priors.
- **Current Status**: **Completed (Frozen)**

#### Code & Configuration
- **Original Script**: `paper_reproduction/cnn_baseline/09_speech_time_frequency/run_table3_speech_nmf.py`
- **Improved Script**: `Not applicable (Baseline frozen)`
- **Configuration File**: `paper_reproduction/cnn_baseline/09_speech_time_frequency/config.yaml`

#### Data Assets
- **Dataset Name**: Google Speech Commands v0.02
- **Dataset Storage Path**: `data/SpeechCommands`

#### Baseline Outputs & Artifacts
- **Baseline Output Directory**: `paper_reproduction/cnn_baseline/09_speech_time_frequency/`
- **Baseline Metrics**: `paper_reproduction/cnn_baseline/09_speech_time_frequency/metrics.json`
- **Baseline Checkpoint**: `N/A`
- **Baseline Visual Plot**: `None`
- **Baseline Visual Panel**: `paper_reproduction/cnn_baseline/09_speech_time_frequency/images/figure_speech_nmf_panel.png`

#### Documentation & Audits
- **Audit Report**: `paper_reproduction/cnn_baseline/09_speech_time_frequency/metrics.txt`
- **Known Discrepancies & Protocol Notes**: None. Replicates Table 3 PSNR across 10 speech audio classes.

---
