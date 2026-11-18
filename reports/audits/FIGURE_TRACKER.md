# FIGURE_TRACKER.md

Master reproduction tracker for all figures and tables from **DeepTensor** (Saragadam et al., IEEE TPAMI / arXiv:2204.03145).

> Full detailed tracker with exact hyperparameter mapping is maintained at:  
> [`paper_reproduction/original_paper/FIGURE_TRACKER.md`](file:///e:/Rakhal_Tensor/paper_reproduction/original_paper/FIGURE_TRACKER.md)

---

## Final Phase 1 Baseline Status Summary

| Item | Title | Application | Validation Status | Empirical Metric vs Paper | Output Artifacts |
|---|---|---|---|---|---|
| **Fig. 1** | Conceptual SVD vs DeepTensor | Concept | Frozen Reference | N/A (Schematic diagram) | Architecture documentation |
| **Fig. 2** | DeepTensor Eigenfaces & SVM Accuracy | `04_eigenfaces_classification` | **1. Faithfully reproduced** | DT: **85.71%** (Paper: ~85%), PCA: **66.83%** (Paper: ~67%) | [`paper_reproduction/cnn_baseline/04_eigenfaces_classification/`](file:///e:/Rakhal_Tensor/paper_reproduction/cnn_baseline/04_eigenfaces_classification/) |
| **Fig. 3** | Implicit Bias Noise Rejection | `01_synthetic_decomposition` | **1. Faithfully reproduced** | Signal MSE: **$1.11 \times 10^{-4}$**, Noise MSE: **$0.239$** | [`plots/figure3_inductive_bias.png`](file:///e:/Rakhal_Tensor/paper_reproduction/cnn_baseline/01_synthetic_decomposition/plots/figure3_inductive_bias.png) |
| **Fig. 4** | Separable Convolutions Speedup | `01_synthetic_decomposition` | **2. Reproduced w/ deviations** | Theoretical $454\times$ FLOP speedup; empirical 1D vs 3D benchmark | [`plots/figure4_separable_convolutions_speedup.png`](file:///e:/Rakhal_Tensor/paper_reproduction/cnn_baseline/01_synthetic_decomposition/plots/figure4_separable_convolutions_speedup.png) |
| **Fig. 5** | DeepTensor Architectures | Model Architecture | **1. Faithfully reproduced** | Exact 2D and 1D U-Net topologies | [`modules/deep_prior.py`](file:///e:/Rakhal_Tensor/modules/deep_prior.py) |
| **Fig. 6** | Signal & Noise Robustness | `01_synthetic_decomposition` | **1. Faithfully reproduced** | Gaussian: DT $\approx$ SVD; Poisson/Rician: DT +1 to +6 dB | [`paper_reproduction/cnn_baseline/01_synthetic_decomposition/`](file:///e:/Rakhal_Tensor/paper_reproduction/cnn_baseline/01_synthetic_decomposition/) |
| **Fig. 7** | Robust PCA (Sample Size & SNR) | `02_pca` | **1. Faithfully reproduced** | Small $M \le 256$: DT +0.80 dB; $\sigma=0.2$: DT +1.63 dB | [`paper_reproduction/cnn_baseline/02_pca/`](file:///e:/Rakhal_Tensor/paper_reproduction/cnn_baseline/02_pca/) |
| **Fig. 8** | Stopping Criterion (DIP vs DD) | `01_synthetic_decomposition` | **1. Faithfully reproduced** | DIP reaches lower min MSE; DD maintains flat plateau | [`plots/figure8_stopping_criterion.png`](file:///e:/Rakhal_Tensor/paper_reproduction/cnn_baseline/01_synthetic_decomposition/plots/figure8_stopping_criterion.png) |
| **Fig. 9** | Hyperspectral Denoising | `05_hyperspectral_denoising` | **1. Faithfully reproduced** | DT: **46.28 dB** (Paper: 46.4 dB), BM3D: **40.50 dB** (Paper: 40.5 dB) | [`paper_reproduction/cnn_baseline/05_hyperspectral_denoising/`](file:///e:/Rakhal_Tensor/paper_reproduction/cnn_baseline/05_hyperspectral_denoising/) |
| **Fig. 10** | Yale Face B Tensor Denoising | `06_yale_face_tensor_denoising` | **2. Reproduced w/ deviations** | 30%: **26.54 dB** (Paper: 28.9 dB), 60%: **26.46 dB** (Paper: 29.7 dB) | [`paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/`](file:///e:/Rakhal_Tensor/paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/) |
| **Fig. 11** | Video Compressive Sensing | `07_video_compressive_sensing` | **2. Reproduced w/ deviations** | DT: **25.14 dB** (Paper: 27.1 dB), TV: **23.47 dB** (Paper: 24.1 dB) | [`paper_reproduction/cnn_baseline/07_video_compressive_sensing/`](file:///e:/Rakhal_Tensor/paper_reproduction/cnn_baseline/07_video_compressive_sensing/) |
| **Fig. 12** | 3D PET CT Reconstruction | `08_ct_reconstruction` | **2. Reproduced w/ deviations** | DT: **29.96 dB** (Paper: 31.1 dB), TV: **31.51 dB** (Paper: 28.6 dB) | [`paper_reproduction/cnn_baseline/08_ct_reconstruction/`](file:///e:/Rakhal_Tensor/paper_reproduction/cnn_baseline/08_ct_reconstruction/) |
| **Fig. 13a**| Rank Sweeping: Matrix (ICVL) | `05_hyperspectral_denoising` | **1. Faithfully reproduced** | SVD drops 38.31 $\to$ 36.54 dB; DT maintains **41.04 $\to$ 41.60 dB** | [`plots/figure13a_rank_sweep.png`](file:///e:/Rakhal_Tensor/paper_reproduction/cnn_baseline/05_hyperspectral_denoising/plots/figure13a_rank_sweep.png) |
| **Fig. 13b**| Rank Sweeping: Tensor (Yale-B) | `06_yale_face_tensor_denoising` | **2. Reproduced w/ deviations** | Baseline drops 24.60 $\to$ 23.76 dB; DT maintains **23.22 $\to$ 23.38 dB** | [`plots/figure13b_yale_rank_sweep.png`](file:///e:/Rakhal_Tensor/paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/plots/figure13b_yale_rank_sweep.png) |
| **Table 1** | LR Schedulers on Toy Matrix | `01_synthetic_decomposition` | **1. Faithfully reproduced** | Cosine Annealing LR $10^{-2}$: **18.10 dB** (Paper: 18.30 dB) | [`synthetic_suite_metrics.json`](file:///e:/Rakhal_Tensor/paper_reproduction/cnn_baseline/01_synthetic_decomposition/synthetic_suite_metrics.json) |
| **Table 2** | NMF on MNIST & CIFAR-10 | `03_nmf` | **2. Reproduced w/ deviations** | MNIST ReLU: **8.40 dB** (Paper: 7.6 dB); CIFAR ReLU: **9.32 dB** (Paper: 6.3 dB) | [`paper_reproduction/cnn_baseline/03_nmf/metrics.json`](file:///e:/Rakhal_Tensor/paper_reproduction/cnn_baseline/03_nmf/metrics.json) |
| **Table 3** | Speech Spectrogram NMF | `09_speech_time_frequency` | **1. Faithfully reproduced** | NMF Softplus: **8.94 dB** (Paper: 9.0 dB), ReLU: **8.90 dB** (Paper: 8.7 dB) | [`paper_reproduction/cnn_baseline/09_speech_time_frequency/`](file:///e:/Rakhal_Tensor/paper_reproduction/cnn_baseline/09_speech_time_frequency/) |
