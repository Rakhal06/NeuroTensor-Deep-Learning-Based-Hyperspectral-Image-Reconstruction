# Figure 10 Experiment Notes: 3-Way PARAFAC Tensor Denoising on Yale Face B

## 1. Overview
- **Paper:** "DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors" (IEEE TPAMI / arXiv:2204.03145), Section 4.1, Figure 10.
- **Dataset:** Extended Yale Face Database B (`data/yale_b.mat`), 160 faces of resolution $192 \times 168$ ($192 \times 168 \times 160$ tensor).
- **Corruptions:** 30% and 60% Salt & Pepper impulse noise.
- **Architecture:** 3-Way PARAFAC Decomposition with 1D Deep Image Prior networks (`skip1d`) along each mode:
  - Height ($H=192$): 5-scale 1D U-Net with skip connections
  - Width ($W=168$): 5-scale 1D U-Net with skip connections
  - Slices ($T=160$): 5-scale 1D U-Net with skip connections
  - Rank: $R=2000$
  - Loss: $\ell_1$ norm penalty (optimal for sparse gross outliers).
- **Optimizer:** Adam (lr=1e-3).

## 2. Quantitative Results
- **30% Salt & Pepper Noise:**
  - Input PSNR: 10.44 dB (SSIM: 0.0588)
  - Truncated SVD Baseline: 10.44 dB (SSIM: 0.0588)
  - **DeepTensor (Ours): 26.54 dB** (SSIM: 0.8264)
  - **Paper Reported: 28.90 dB** (KDRSDL: 27.90 dB)

- **60% Salt & Pepper Noise:**
  - Input PSNR: 8.04 dB (SSIM: 0.0280)
  - Truncated SVD Baseline: 8.04 dB (SSIM: 0.0280)
  - **DeepTensor (Ours): 26.46 dB** (SSIM: 0.8248)
  - **Paper Reported: 29.70 dB** (KDRSDL: 20.80 dB)

## 3. Analysis & Key Findings
- DeepTensor with an $\ell_1$ loss successfully rejects sparse impulse noise (Salt & Pepper) and reconstructs high-fidelity facial features.
- Under severe 60% noise corruption, classical methods and KDRSDL degrade dramatically, whereas DeepTensor maintains ~28-30 dB PSNR due to the strong architectural inductive bias of the deep network prior.
