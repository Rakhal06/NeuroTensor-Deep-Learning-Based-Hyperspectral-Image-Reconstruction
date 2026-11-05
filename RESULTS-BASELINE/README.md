# DEEPTENSOR: BASELINE EXPERIMENT SUITE REPRODUCTION

This directory (`RESULTS-BASELINE/`) contains the complete, verified, reproducible execution suite for the five baseline experiments from the research paper:
> **"DeepTensor: Low-Rank Tensor Decomposition With Deep Network Priors"**
> *Vishwanath Saragadam, Randall Balestriero, Ashok Veeraraghavan, Richard G. Baraniuk* (IEEE / arXiv:2204.03145)

All experiments were executed on an NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.1, PyTorch 2.5.1+cu121).

---

## Directory Organization

```
RESULTS-BASELINE/
│
├── README.md                          <- This guide
├── experiment_summary.txt             <- Status table of all five experiments
├── final_report.txt                   <- Comprehensive scientific analysis and verification report
│
├── figure2/                           <- Figure 2: DeepTensor Eigenfaces on Weizmann Dataset
│   ├── configuration.json             <- Full hyperparameter configuration
│   ├── experiment.log                 <- Complete execution trace log
│   ├── metrics.txt                    <- Accuracy, mAP, and classification comparison
│   ├── figure2_results.npz            <- Numerical factor and loss checkpoint arrays
│   ├── images/
│   │   ├── figure2_eigenfaces_top12.png
│   │   └── deeptensor_eigenfaces_montage.png
│   └── plots/
│       ├── figure2_classification_accuracy.png
│       └── figure2_loss_curve.png
│
├── figure6/                           <- Figure 6: Robustness across Signal & Noise Distributions
│   ├── configuration.json
│   ├── experiment.log
│   ├── metrics.txt                    <- PSNR vs Rank across 3 regimes
│   ├── figure6_results.npz
│   ├── images/
│   │   └── figure6_visual_comparisons_rank30.png
│   └── plots/
│       └── figure6_psnr_vs_rank.png
│
├── figure7/                           <- Figure 7: PCA under Low SNR & Small Sample Size
│   ├── configuration.json
│   ├── experiment.log
│   ├── metrics.txt                    <- PSNR vs Sample Size & Noise Std
│   ├── figure7_results.npz
│   ├── images/
│   │   └── figure7_covariance_visual_comparison.png
│   └── plots/
│       └── figure7_pca_sample_size_and_noise.png
│
├── figure9/                           <- Figure 9: Hyperspectral Image Denoising on ICVL
│   ├── configuration.json
│   ├── experiment.log
│   ├── metrics.txt                    <- PSNR, SSIM, RMSE, and runtime comparisons
│   ├── figure9_results.npz
│   ├── checkpoint/
│   │   └── figure9_deeptensor_checkpoint.pt
│   ├── images/
│   │   └── figure9_hyperspectral_denoising_panel.png
│   └── plots/
│       └── figure9_convergence_curves.png
│
└── figure12/                          <- Figure 12: 3D PET CT Reconstruction from Limited Measurements
    ├── configuration.json
    ├── experiment.log
    ├── metrics.txt                    <- PSNR, SSIM, RMSE vs 2D TV & Algorithmic
    ├── figure12_results.npz
    ├── checkpoint/
    │   └── figure12_deeptensor_checkpoint.pt
    ├── images/
    │   └── figure12_reconstruction_panel.png
    └── plots/
        └── figure12_convergence_curves.png
```

---

## How to Re-Run From the Terminal

Using the project's Python environment:
```powershell
python run_figure2.py
python run_figure6.py
python run_figure7.py
python run_figure9.py
python run_figure12.py
```
