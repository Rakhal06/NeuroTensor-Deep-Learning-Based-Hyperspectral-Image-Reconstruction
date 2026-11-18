# DeepTensor Original Paper Reported Metrics vs Reproduced CNN Baseline

This document indexes all numerical values, accuracy scores, and metrics reported in the original paper:
**"DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors"** (Saragadam et al., IEEE TPAMI / arXiv:2204.03145) alongside our **empirically reproduced CNN/DIP baseline results**.

---

## 1. Matrix & Tensor Decomposition Accuracy

### Figure 2: Face Classification on Weizmann FaceBase (Page 2 & 8)
- Input Noisy PSNR: $\approx 15$ dB (Poisson $\tau=20$ + Gaussian $\text{SNR}=2$).
- Classification Accuracy:
  - **DeepTensor:** Paper $\approx 85\%$ | **Reproduced: 85.71%** (mAP: 0.7561)
  - **Clean PCA Oracle:** Paper $\approx 85\%$ | **Reproduced: 85.40%**
  - **Noisy PCA:** Paper $\approx 67\%$ | **Reproduced: 66.83%** (mAP: 0.5516)
  - **Noisy ICA:** Paper $\approx 64\%$ | **Reproduced: 64.76%** (mAP: 0.5342)

### Figure 6: Synthetic Matrix Robustness (Page 5)
- Evaluated across Ranks $R \in [10, 20, 30, 40, 50, 60]$ on $64 \times 64$ matrices.
- **Gaussian Signal + Gaussian Noise ($\sigma = 0.1$):** DeepTensor matches Truncated SVD ($\approx 20$ dB).
- **Piecewise Constant + Poisson Noise ($\tau = 1000$):** DeepTensor achieves $+0.6$ dB to $+6$ dB improvement over SVD across lower ranks.
- **Piecewise Constant + Rician Noise ($\sigma = 0.02$):** DeepTensor significantly outperforms SVD across all ranks (+1.2 to +4.8 dB).

### Figure 7: Robust PCA Covariance Reconstruction (Page 5)
- Dimension $D = 64$, Intrinsic Rank $R = 10$.
- **Sample Size Sweep ($M \in [32, 1024]$, $\sigma = 0.5$):** DeepTensor achieves higher covariance PSNR than SVD for small samples ($M \le 256$, up to $+0.80$ dB gain).
- **Noise Sweep ($\sigma \in [0.05, 1.0]$, $M = 32$):** DeepTensor achieves $+1.63$ dB gain over SVD at $\sigma = 0.2$.

### Figure 9: Hyperspectral Image Denoising on ICVL (Page 8)
- Input Noisy PSNR: $28.43$ dB (Poisson $\tau=100$ + Gaussian $\text{SNR}=2$).
- Reconstruction PSNR / SSIM:
  - **DeepTensor:** Paper $\approx 46.4$ dB | **Reproduced: 46.28 dB** (SSIM: 0.9872, RMSE: 0.0049)
  - **BM3D:** Paper $\approx 40.5$ dB | **Reproduced: 40.50 dB** (SSIM: 0.9499)
  - **Truncated SVD:** Paper $\approx 38.3$ dB | **Reproduced: 38.31 dB** (SSIM: 0.9176)

### Figure 10: Yale Face Database B 3-Way PARAFAC Tensor Denoising (Page 8)
- Model: Rank-2000 PARAFAC tensor decomposition ($192 \times 168 \times 160$).
- **30% Salt & Pepper Noise:**
  - **DeepTensor:** Paper **28.9 dB** | **Reproduced: 26.54 dB** (SSIM: 0.8264)
  - **KDRSDL / SVD:** Paper 27.9 dB | **SVD Baseline: 10.44 dB**
- **60% Salt & Pepper Noise:**
  - **DeepTensor:** Paper **29.7 dB** | **Reproduced: 26.46 dB** (SSIM: 0.8248)
  - **KDRSDL / SVD:** Paper 20.8 dB | **SVD Baseline: 8.04 dB**

### Figure 11: Video Compressive Sensing on Cat Video (Page 9)
- Sensed from 8 frames multiplexed into 1 coded measurement:
  - **DeepTensor:** Paper **27.1 dB**, SSIM **0.81** | **Reproduced: 25.14 dB**, SSIM **0.7239**
  - **2D TV:** Paper **24.1 dB**, SSIM **0.69** | **Reproduced: 23.47 dB**, SSIM **0.6873**
  - **DeepTensor vs TV Advantage:** Paper +3.0 dB, +0.12 SSIM | **Reproduced: +1.66 dB, +0.0366 SSIM**

### Figure 12: 3D CT Reconstruction on PET Volume (Page 9)
- Reconstructed from 40 Radon projection angles per slice:
  - **DeepTensor:** Paper **31.1 dB**, SSIM **0.87** | **Reproduced: 29.96 dB**, SSIM **0.8226**
  - **2D TV:** Paper **28.6 dB**, SSIM **0.71** | **Reproduced: 31.51 dB**, SSIM **0.8075**
  - **Algorithmic Inversion:** Paper **26.3 dB**, SSIM **0.48** | **Reproduced: 26.86 dB**, SSIM **0.5841**

### Figure 13(a): Effect of Rank Choice on HSI Matrix Decomposition (Page 10)
- Truncated ground-truth rank 20, swept $R \in [1, 30]$:
  - **SVD Baseline:** Degrades monotonically above true rank (from 38.31 dB down to 36.54 dB).
  - **DeepTensor:** Maintains flat plateau above true rank (41.04 dB to 41.60 dB).

### Figure 13(b): Effect of Rank Choice on PARAFAC Tensor (Page 10)
- Yale Face B, swept $R \in [10, 4000]$:
  - **Classical Baseline:** Degrades from 24.60 dB down to 23.76 dB at $R=4000$.
  - **DeepTensor:** Maintains stable plateau (23.22 to 23.38 dB) across all over-parameterized ranks.

---

## 2. Tabular Performance Benchmarks

### Table 1: Learning Rate Schedule Grid on Toy Matrix (Page 6)
- Synthetic matrix $64 \times 64$, Rank $R = 32$. Baseline SVD: Paper **14.4 dB** | **Reproduced: 14.51 dB**.
- **Max LR $10^{-2}$:**
  - Fixed: Paper $17.6 \pm 0.7$ | **Reproduced: 17.3 dB**
  - Step: Paper $17.1 \pm 0.4$ | **Reproduced: 17.3 dB**
  - Exp: Paper $16.6 \pm 0.4$ | **Reproduced: 17.3 dB**
  - Cosine: Paper $16.5 \pm 0.3$ | **Reproduced: 18.1 dB**
- **Max LR $10^{-3}$:**
  - Fixed: Paper $17.5 \pm 0.6$ | **Reproduced: 16.5 dB**
  - Step: Paper $17.1 \pm 0.5$ | **Reproduced: 16.5 dB**
  - Exp: Paper $16.6 \pm 0.4$ | **Reproduced: 16.5 dB**
  - Cosine: Paper $16.6 \pm 0.3$ | **Reproduced: 17.4 dB**
- **Max LR $10^{-4}$:**
  - Fixed: Paper $17.0 \pm 0.5$ | **Reproduced: 17.1 dB**
  - Step: Paper $17.0 \pm 0.5$ | **Reproduced: 16.9 dB**
  - Exp: Paper $17.0 \pm 0.5$ | **Reproduced: 16.9 dB**
  - Cosine: Paper $16.4 \pm 0.2$ | **Reproduced: 16.1 dB**

### Table 2: NMF on MNIST & CIFAR-10 (Page 7)
- Rician noise added, input PSNR = 4.8 dB. Average PSNR (dB):
  - **MNIST:**
    - DeepTensor Softplus: Paper $7.4 \pm 0.03$ dB | **Reproduced: 7.95 dB**
    - DeepTensor Abs. value: Paper $7.3 \pm 0.01$ dB | **Reproduced: 8.55 dB**
    - DeepTensor ReLU: Paper $7.6 \pm 0.15$ dB | **Reproduced: 8.40 dB**
    - Standard NMF (Lee & Seung): Paper 7.5 dB | **Reproduced: 8.69 dB**
  - **CIFAR-10:**
    - DeepTensor Softplus: Paper $5.7 \pm 0.03$ dB | **Reproduced: 8.79 dB**
    - DeepTensor Abs. value: Paper $5.7 \pm 0.02$ dB | **Reproduced: 9.34 dB**
    - DeepTensor ReLU: Paper $6.3 \pm 0.04$ dB | **Reproduced: 9.32 dB**
    - Standard NMF: Paper 5.6 dB | **Reproduced: 9.54 dB**

### Table 3: NMF on Speech Spectrograms (Google Speech Commands v0.02) (Page 10)
- STFT Spectrograms $500 \times 500$, Rician noise, input PSNR = 4.7 dB (Paper: 4.6 dB). Average PSNR (dB):
  - **NMF Mode (Nonnegativity on $U, V$):**
    - DeepTensor Softplus: Paper **$9.0 \pm 0.03$ dB** | **Reproduced: $8.94 \pm 0.01$ dB**
    - DeepTensor Abs. value: Paper **$8.7 \pm 0.01$ dB** | **Reproduced: $8.92 \pm 0.01$ dB**
    - DeepTensor ReLU: Paper **$8.7 \pm 0.03$ dB** | **Reproduced: $8.90 \pm 0.00$ dB**
    - Baseline NMF: Paper 4.6 dB | **Reproduced: 7.39 dB**
  - **Semi-NMF Mode (Nonnegativity on $U$ only):**
    - DeepTensor Softplus: Paper **$8.7 \pm 0.09$ dB** | **Reproduced: $8.84 \pm 0.00$ dB**
    - DeepTensor Abs. value: Paper **$8.8 \pm 0.02$ dB** | **Reproduced: $8.92 \pm 0.01$ dB**
    - DeepTensor ReLU: Paper **$8.8 \pm 0.02$ dB** | **Reproduced: $8.90 \pm 0.00$ dB**
    - Baseline Semi-NMF: Paper 4.7 dB | **Reproduced: 7.38 dB**
