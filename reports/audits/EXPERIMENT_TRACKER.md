# DeepTensor Application & Architecture Experiment Tracker

This document tracks the execution progress across all 9 research applications for both the **CNN Baseline** and the **Hybrid CNN–Transformer Architecture**, along with statistical and visual comparisons.

---

## Application Tracking Matrix

| ID | Application Name | Key Paper Target | Dataset Status | CNN Baseline Status | Hybrid Arch Status | Comparison Status | Notes & Validation Status |
|---|---|---|---|---|---|---|---|
| **01** | Synthetic Decomposition | Fig. 3, 4, 6, 8, Tab. 1 | Verified (Synthetic) | **100% COMPLETE & VALIDATED** | Awaiting Phase 2 | Pending | Fig 3, 6, 8, Tab 1 faithfully reproduced; Fig 4 speedup documented |
| **02** | Robust PCA | Fig. 7 (Sample size & SNR) | Verified (Synthetic) | **100% COMPLETE & VALIDATED** | Awaiting Phase 2 | Pending | Small sample & high noise advantages faithfully reproduced |
| **03** | NMF & Positivity Constraints | Table 2 (MNIST & CIFAR-10) | Verified (torchvision) | **100% COMPLETE & VALIDATED** | Awaiting Phase 2 | Pending | Softplus, Abs, ReLU nonnegativity constraints evaluated to convergence |
| **04** | Eigenfaces & Classification | Fig. 2 (Weizmann SVM) | Verified (`weizzman.mat`) | **100% COMPLETE & VALIDATED** | Awaiting Phase 2 | Pending | 85.71% accuracy faithfully matches paper (~85%) vs PCA (66.83%) |
| **05** | Hyperspectral Denoising | Fig. 9, 13a (ICVL Poisson) | Verified (`icvl.mat`) | **100% COMPLETE & VALIDATED** | Awaiting Phase 2 | Pending | 46.28 dB matches paper (46.4 dB); rank over-estimation plateau confirmed |
| **06** | Yale Face Tensor Denoising | Fig. 10, 13b (PARAFAC S&P) | Verified (`yale_b.mat`) | **100% COMPLETE & VALIDATED** | Awaiting Phase 2 | Pending | 3-Way PARAFAC rank-2000 L1 denoising; 26.54 dB (30%) & 26.46 dB (60%) |
| **07** | Video Compressive Sensing | Fig. 11 (Coded aperture) | Verified (`cat_video.mat`) | **100% COMPLETE & VALIDATED** | Awaiting Phase 2 | Pending | 25.14 dB (+1.66 dB over 2D TV 23.47 dB) confirms paper advantage |
| **08** | 3D CT Reconstruction | Fig. 12 (PET Radon 40 angles)| Verified (`pet.mat`) | **100% COMPLETE & VALIDATED** | Awaiting Phase 2 | Pending | 29.96 dB PARAFAC reconstruction with 0.8226 SSIM |
| **09** | Speech Spectrogram NMF | Table 3 (Speech Commands) | Verified (SpeechCommands) | **100% COMPLETE & VALIDATED** | Awaiting Phase 2 | Pending | 8.94 dB (NMF) & 8.92 dB (Semi-NMF) faithfully reproduce paper values |

---

## Validation Summary

Every CNN baseline has been executed on CUDA under exact dataset dimensions, full sample counts, and verified noise models.
The formal Phase 1 Validation Gate report is located at:
[`paper_reproduction/consolidated_results/PHASE1_VALIDATION_REPORT.md`](file:///e:/Rakhal_Tensor/paper_reproduction/consolidated_results/PHASE1_VALIDATION_REPORT.md)
