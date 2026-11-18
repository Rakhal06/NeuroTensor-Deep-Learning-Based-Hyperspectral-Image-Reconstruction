# DeepTensor Master Experiment Register

This register details every experiment, figure, and table from the original paper:
**"DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors"**  
*Vishwanath Saragadam, Randall Balestriero, Ashok Veeraraghavan, Richard G. Baraniuk* (IEEE TPAMI / arXiv:2204.03145).

---

## Master Experiment Catalog

### Exp 1: Conceptual SVD vs. DeepTensor Factorization (Figure 1)
- **Application:** Conceptual formulation.
- **Dataset / Dimensions:** Synthetic toy schema ($X = U V$, $U \in \mathbb{R}^{M \times R}, V \in \mathbb{R}^{R \times N}$).
- **Network Architecture:** 1D / 2D Deep Image Prior (DIP) skip generators: $U = f_U(z_U), V = f_V(z_V)$.
- **Decomposition Method:** Self-supervised low-rank matrix decomposition minimizing $\|X - f_U(z_U) f_V(z_V)\|_2^2$.
- **Rank:** $R$ (arbitrary latent dimension).
- **Reported Value:** N/A (conceptual paradigm contrast with SVD).
- **Implementation Script:** Architectural paradigm implemented in `modules/deep_prior.py` and `modules/models.py`.
- **Status:** **FROZEN / CONCEPTUAL**.

---

### Exp 2: DeepTensor Eigenfaces & Face Classification under Poisson Noise (Figure 2)
- **Application:** 04_eigenfaces_classification
- **Dataset / Source:** Weizmann FaceBase (`data/weizzman.mat`). 840 face images across 28 subjects (30 images/subject under varying illumination).
- **Original Dimensions:** $840 \times 103 \times 71$ (flattened matrix $X \in \mathbb{R}^{840 \times 7313}$).
- **Decomposition Method:** Low-rank matrix decomposition: sample covariance / data matrix factored into $U \in \mathbb{R}^{840 \times R}$ and $V \in \mathbb{R}^{R \times 7313}$.
- **Rank:** $R = 84$ components (3 per subject) or $R = 20$.
- **Noise / Measurement Settings:** Poisson noise ($\tau = 20$) + Gaussian readout ($\text{SNR} = 2.0$), yielding input noisy PSNR $\approx 14.59$ dB.
- **Loss Function:** MSE / $\ell_2$ data fidelity loss on data matrix or sample covariance.
- **Optimizer & LR:** Adam optimizer, $\text{LR} = 10^{-3}$ or $10^{-4}$.
- **Iterations / Stopping:** 100 epochs.
- **Evaluation Metrics:** Top-12 learned eigenface basis visualization; Classification accuracy and mean Average Precision (mAP) using Linear Support Vector Machine (SVM) with full 5-fold GridSearchCV across $C \in [10^{-3}, 10^3]$.
- **Paper Reported Values:** Top-12 bases visual clarity; Classification accuracy @ 15 dB: DeepTensor $\approx 85\%$, PCA $\approx 67\%$, ICA $\approx 64\%$.
- **Implementation Script:** `run_figure2.py`
- **Execution & Reproduction Status:** **EXECUTED ON CUDA** (`RESULTS-BASELINE/figure2/`).
  - Reproduced Accuracy: DeepTensor **85.71%**, Clean PCA Oracle **85.40%**, Noisy PCA **66.83%**, Noisy ICA **64.76%** (+18.89% gain over noisy PCA).

---

### Exp 3: Inductive Bias of DeepTensor Rejects Noise (Figure 3)
- **Application:** 01_synthetic_decomposition / 05_hyperspectral_denoising
- **Dataset / Source:** Hyperspectral image slice (`data/icvl.mat`) or synthetic matrix ($64 \times 64$).
- **Noise Settings:** iid Gaussian noise with standard deviation $\sigma = 0.01$.
- **Decomposition Method:** Rank-20 separable DeepTensor ($U = f_U(z_U), V = f_V(z_V)$).
- **Rank:** $R = 20$.
- **Loss Function:** $\ell_2$ loss.
- **Optimizer & LR:** Adam, $\text{LR} = 10^{-3}$.
- **Iterations / Stopping:** 5000 epochs.
- **Evaluation Metrics:** Mean Squared Error (MSE) loss on signal component vs. noise component across epochs.
- **Paper Reported Values:** Signal loss rapidly descends while noise loss descends at a much slower rate (implicit regularization).
- **Implementation Script:** Sub-functionality inside `matrix_decomp.py`.
- **Status:** **PARTIALLY VALIDATED VIA BASELINE CONVERGENCE LOGS**; needs isolated dedicated plotting script.

---

### Exp 4: Separable Convolutions Compute Speedup (Figure 4)
- **Application:** 01_synthetic_decomposition / 08_ct_reconstruction
- **Dataset / Source:** 3D tensor volume (e.g., $64 \times 64 \times 64$ or PET $128 \times 128 \times 56$).
- **Decomposition Method:** Separable 2D spatial DIP + 1D temporal DIP vs. monolithic 3D DIP.
- **Rank:** $R = 32$ or $R = 20$.
- **Evaluation Metrics:** Training MSE vs. wall-clock time / training epochs; floating point operations (FLOPs).
- **Paper Reported Values:** Separable approach achieves $\approx 60\times$ to $100\times$ speedup (two orders of magnitude) over full 3D convolutions with matching reconstruction quality.
- **Implementation Script:** Built into core DeepTensor design; isolated benchmark script pending.
- **Status:** **ARCHITECTURALLY VERIFIED**; standalone benchmark script to be formalized in Phase 1.

---

### Exp 5: DeepTensor Generator Architectures (Figure 5)
- **Application:** Architectural reference.
- **Topology:** 2D U-Net (DIP `skip2d`) with reflection padding, LeakyReLU, bilinear upsampling; 1D U-Net (DIP `skip1d`) with linear upsampling.
- **Implementation Script:** Fully implemented in `modules/deep_prior.py` (`get_net()`, `skip()`).
- **Status:** **CORE IMPLEMENTATION VERIFIED**.

---

### Exp 6: Low-Rank Matrix Robustness across Signal & Noise Distributions (Figure 6)
- **Application:** 01_synthetic_decomposition
- **Dataset / Source:** Synthetic low-rank matrices ($64 \times 64$) generated via ground-truth factor multiplication $X = U V$.
- **Dimensions:** $N_{\text{rows}} = 64, N_{\text{cols}} = 64$.
- **Evaluated Ranks:** Swept across $R \in [10, 20, 30, 40, 50, 60]$.
- **Signal & Noise Regimes:**
  1. Gaussian signal + Gaussian noise ($\sigma = 0.1$, $\tau = 1000$).
  2. Piecewise constant signal + Poisson noise ($\tau = 1000$, signal in $[0, 1000]$).
  3. Piecewise constant signal + Rician noise ($\sigma = 0.02$, $\tau = 1000$).
- **Decomposition Method:** 1D DIP U-Net for $U$, 1D DIP U-Net for $V$.
- **Loss Function:** $\ell_2$ reconstruction loss.
- **Optimizer & LR:** Adam, $\text{LR} = 10^{-4}$, 1000 epochs.
- **Evaluation Metrics:** Reconstruction PSNR (dB) against clean ground-truth matrix for DeepTensor vs. Truncated SVD.
- **Paper Reported Values:** Gaussian: DeepTensor matches SVD ($\approx 20$ dB); Poisson: DeepTensor (+1 to +6 dB over SVD); Rician: DeepTensor significantly outperforms SVD across all ranks.
- **Implementation Script:** `run_figure6.py`
- **Execution & Reproduction Status:** **EXECUTED ON CUDA** (`RESULTS-BASELINE/figure6/`).
  - Verified: Gaussian matches SVD (within ~1.5 dB); Poisson Rank-10: DeepTensor **36.59 dB** vs. SVD **35.95 dB**; Rician Rank-10: **36.85 dB**.

---

### Exp 7: Robust PCA under Low SNR & Limited Sample Sizes (Figure 7)
- **Application:** 02_pca
- **Dataset / Source:** Synthetic data drawn from zero-mean Gaussian distribution with low-rank covariance.
- **Dimensions:** Feature dimension $D = 64$, Intrinsic rank $R_{\text{true}} = 10$.
- **Evaluated Conditions:**
  - Part 1: Sample size sweep $M \in [32, 64, 128, 256, 512, 1024]$ at fixed noise standard deviation $\sigma = 0.5$.
  - Part 2: Noise standard deviation sweep $\sigma \in [0.05, 0.1, 0.2, 0.5, 1.0]$ at fixed sample size $M = 32$.
- **Decomposition Method:** DeepTensor decomposition of empirical sample covariance matrix $\hat{\Sigma} = \frac{1}{M} X X^T$ into $U V^T$ with rank 10.
- **Loss Function:** $\ell_2$ Frobenius norm loss.
- **Optimizer & LR:** Adam, $\text{LR} = 10^{-4}$, 1000 epochs.
- **Evaluation Metrics:** Covariance Reconstruction PSNR (dB) against clean true covariance matrix $\Sigma_{\text{gt}}$.
- **Paper Reported Values:** DeepTensor outperforms sample covariance SVD at small sample sizes ($M \le 128$) and high noise levels ($\sigma \ge 0.2$), showing up to +1.6 dB advantage.
- **Implementation Script:** `run_figure7.py`
- **Execution & Reproduction Status:** **EXECUTED ON CUDA** (`RESULTS-BASELINE/figure7/`).
  - Verified: DT outperforming SVD across sample sizes ($M=32$: +0.22 dB; $M=128$: +0.72 dB; $M=256$: +0.80 dB) and high noise ($\sigma=0.2$: DT 15.76 dB vs SVD 14.13 dB, +1.63 dB gain).

---

### Exp 8: Stopping Criterion Analysis: DIP vs. Deep Decoder (Figure 8)
- **Application:** 01_synthetic_decomposition
- **Dataset / Dimensions:** Synthetic low-rank matrix ($64 \times 64$, $R=32$).
- **Network Architectures Compared:**
  - Over-parameterized DIP (`skip1d`, $\approx 1.2\text{M}$ params)
  - Under-parameterized Deep Decoder (`decodernw1d`, $\approx 25\text{K}$ params)
- **Noise Levels:** Gaussian noise with standard deviation $\sigma \in [0.05, 0.5]$.
- **Paper Reported Values:**
  - Fig 8(a): Optimal stopping epoch decreases monotonically with increasing noise standard deviation.
  - Fig 8(b): Over-parameterized DIP reaches lower absolute error at optimal epoch (~100), but overfits rapidly thereafter; Deep Decoder achieves slightly higher minimum error, but exhibits a stable, flat plateau.
- **Implementation Script:** Model options exist in `matrix_decomp.py` (`nettype='dip'` vs `'dd'`); isolated automated script required.
- **Status:** **NEEDS DEDICATED RUNNER SCRIPT**.

---

### Exp 9: Hyperspectral Image Denoising (Figure 9)
- **Application:** 05_hyperspectral_denoising
- **Dataset / Source:** ICVL Hyperspectral Dataset (`data/icvl.mat`).
- **Dimensions:** $174 \times 164$ spatial pixels, 260 spectral bands ($31$ wavelength channels in original publication; paper downsampled cube is $174 \times 164 \times 260$).
- **Decomposition Method:** Separable PARAFAC / Tucker-like matrix decomposition: Spatial factor $U \in \mathbb{R}^{28536 \times 20}$ generated by 2D DIP; Spectral factor $V \in \mathbb{R}^{20 \times 260}$ generated by 1D DIP.
- **Rank:** Fixed rank $R = 20$.
- **Noise / Measurement Settings:** Poisson photon noise ($\tau = 100$) + Gaussian readout noise ($\text{SNR} = 2.0$), yielding input noisy PSNR $\approx 28.43$ dB.
- **Loss Function:** $\ell_1$ norm $\|X_{\text{noisy}} - U V\|_1$.
- **Optimizer & LR:** Adam optimizer, $\text{LR} = 10^{-3}$, 5000 epochs.
- **Evaluation Metrics:** PSNR (dB), SSIM, RMSE, spectral signature fidelity.
- **Paper Reported Values:** DeepTensor: $\approx 46.4$ dB PSNR, SSIM $\approx 0.987$; BM3D baseline: $\approx 40.5$ dB PSNR; Truncated SVD: $\approx 38.3$ dB PSNR.
- **Implementation Script:** `run_figure9.py`
- **Execution & Reproduction Status:** **EXECUTED ON CUDA** (`RESULTS-BASELINE/figure9/`).
  - Reproduced: DeepTensor **46.28 dB** PSNR, **0.9872** SSIM, **0.0049** RMSE; BM3D **40.50 dB** PSNR; SVD **38.31 dB** PSNR.

---

### Exp 10: 3-Way PARAFAC Tensor Denoising on Yale Face Database B (Figure 10)
- **Application:** 06_yale_face_tensor_denoising
- **Dataset / Source:** Extended Yale Face Database B (Georghiades et al., PAMI 2001).
- **Dimensions:** Subset of 160 face images, each $192 \times 168$ pixels, stacked into a 3-way tensor $X \in \mathbb{R}^{192 \times 168 \times 160}$.
- **Noise Settings:** Spatially sparse Salt-and-Pepper noise at 30% and 60% corruption levels.
- **Decomposition Method:** 3-way PARAFAC decomposition: $X \approx \sum_{r=1}^R u_r \circ v_r \circ w_r$, where factors are generated via deep network priors.
- **Rank:** $R = 2000$.
- **Loss Function:** $\ell_1$ penalty (reported more robust than $\ell_2$ for spatially sparse impulse noise).
- **Baselines:** Kronecker Decomposition-based Robust Dictionary Learning (KDRSDL; Saragadam & Sankaranarayanan, 2019).
- **Paper Reported Values:**
  - 30% noise: DeepTensor **28.9 dB** vs. KDRSDL **27.9 dB**
  - 60% noise: DeepTensor **20.8 dB** vs. KDRSDL **29.7 dB** (paper caption lists these values; to be verified during execution)
- **Implementation Script:** **MISSING** (not provided in author release).
- **Execution Status:** **BLOCKED** on acquiring Extended Yale Face B and downloading 160 images at $192 \times 168$.

---

### Exp 11: Video Compressive Sensing (Figure 11)
- **Application:** 07_video_compressive_sensing
- **Dataset / Source:** Cat video (`data/cat_video.mat`, original dimensions $128 \times 128 \times 64$).
- **Discrepancy Resolution:**
  - Paper caption states: "combine 32 frames into a single coded image frame".
  - Paper text (Section 4.3) states: "We combined 8 frames into one single coded image".
  - Figure 11 visual panel shows: 8 reconstructed frames (Frame 1 to Frame 8).
  - Code (`run_figure11.py`) specifies: `nframes = 8`.
  - Resolution: 8 frames sliced from the 64-frame volume are spatially multiplexed into 1 coded measurement.
- **Measurement Model:** Spatially multiplexed coded aperture (Hitomi et al., ICCV 2011 setup): each pixel is sampled at an arbitrary time frame across $T=8$ frames: $y(x, y) = \sum_{t=1}^T C(x, y, t) X(x, y, t)$.
- **Decomposition Method:** Full-rank DeepTensor recovery ($R = 8$ or $R = 32$).
- **Loss Function:** Forward measurement $\ell_2$ data fidelity: $\|y - \mathcal{A}(f(z))\|_2^2$.
- **Optimizer & LR:** Adam, $\text{LR} = 10^{-3}$, 10,000 epochs.
- **Baselines:** 2D Total Variation (TV) regularized inversion (`run_figure11_TV.py`).
- **Paper Reported Values:** DeepTensor: **27.1 dB** PSNR, **0.81** SSIM vs. 2D TV: **24.1 dB** PSNR, **0.69** SSIM.
- **Implementation Script:** `run_figure11.py` and `run_figure11_TV.py` (Script has broken path `sys.path.append('../modules')` that requires fix).
- **Execution Status:** **UNRUN** (ready for execution once path is fixed).

---

### Exp 12: 3D PET CT Reconstruction from Limited Tomographic Angles (Figure 12)
- **Application:** 08_ct_reconstruction
- **Dataset / Source:** 3D Positron Emission Tomography (PET) volume (`data/pet.mat`, Cancer Imaging Archive).
- **Dimensions:** $128 \times 128 \times 56$ (56 axial slices).
- **Forward Measurement Model:** Radon transform along 40 tomographic projection angles per axial slice.
- **Noise Settings:** Poisson noise ($\tau = 1000$) + Gaussian readout noise ($\text{SNR} = 2.0$).
- **Decomposition Method:** Rank-1000 PARAFAC decomposition: volume represented as 3 factor matrices generated by DIP networks.
- **Rank:** $R = 1000$.
- **Loss Function:** Radon projection domain $\ell_2$ loss: $\|\mathcal{R}(X_{\text{recon}}) - Y_{\text{sinogram}}\|_2^2$.
- **Optimizer & LR:** Adam, $\text{LR} = 10^{-3}$, 2000 epochs.
- **Baselines:** Algorithmic reconstruction (Linear inversion without prior), 2D TV regularization, FBPConvNet (trained Jin et al.), Bacca et al., Gong et al.
- **Paper Reported Values:** DeepTensor: **31.1 dB** PSNR, **0.87** SSIM vs. 2D TV: **28.6 dB** PSNR, **0.71** SSIM vs. Algorithmic: **26.3 dB** PSNR, **0.48** SSIM.
- **Implementation Script:** `run_figure12.py`
- **Execution & Reproduction Status:** **EXECUTED ON CUDA** (`RESULTS-BASELINE/figure12/`).
  - Reproduced: DeepTensor **29.96 dB** PSNR, **0.8226** SSIM; 2D TV **31.51 dB** PSNR, **0.8075** SSIM; Algorithmic **26.86 dB** PSNR, **0.5841** SSIM.

---

### Exp 13: Effect of Rank Choice: Matrix vs. Tensor (Figure 13)
- **Application:** 05_hyperspectral_denoising / 06_yale_face_tensor_denoising
- **Evaluated Protocols:**
  - Fig 13(a) Matrix: ICVL hyperspectral image truncated to rank-20; rank swept from $R=1$ to $R=30$.
  - Fig 13(b) PARAFAC Tensor: Yale-B face tensor truncated to rank-1000; rank swept from $R=10$ to $R=4000$.
- **Noise Settings:** Poisson ($\lambda_{\text{max}} = 100$) + Gaussian ($\sigma = 2.0$), yielding $\approx 25$ dB input PSNR.
- **Paper Reported Values:**
  - Under-estimated rank: DeepTensor and SVD/PARAFAC both exhibit degraded reconstruction.
  - Over-estimated rank: SVD/PARAFAC degrades drastically as rank increases (fitting noise); DeepTensor remains flat and stable (implicit rank regularization).
- **Implementation Script:** Needs isolated runner script.
- **Status:** **NEEDS DEDICATED RUNNER SCRIPT**.

---

### Exp 14: Learning Rate Schedulers and Maximum LR on Toy Matrix (Table 1)
- **Application:** 01_synthetic_decomposition
- **Dataset / Dimensions:** Synthetic low-rank matrix ($64 \times 64$, $R=32$).
- **Conditions Tested:**
  - Schedulers: Fixed, Step (multiply by 0.99 every 2000 epochs), Exponential (decay 0.9999), Cosine Annealing.
  - Maximum LR: $10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}$.
  - Repeated 5 independent runs per cell.
- **Baseline:** SVD accuracy: **14.4 dB**.
- **Paper Reported Values (PSNR dB):**
  - LR $10^{-2}$: Fixed $17.6 \pm 0.7$, Step $17.1 \pm 0.4$, Exp $16.6 \pm 0.4$, Cosine $16.5 \pm 0.3$.
  - LR $10^{-3}$: Fixed $17.5 \pm 0.6$, Step $17.1 \pm 0.5$, Exp $16.6 \pm 0.4$, Cosine $16.6 \pm 0.3$.
  - LR $10^{-4}$: Fixed $17.0 \pm 0.5$.
  - LR $10^{-5}$: Degraded ($< 15$ dB).
- **Implementation Script:** Prototype logic in `matrix_decomp.py`; requires dedicated runner script.
- **Execution Status:** **UNRUN** (to be executed in Phase 1).

---

### Exp 15: NMF and Positivity Constraints on MNIST & CIFAR-10 (Table 2)
- **Application:** 03_nmf
- **Datasets:** MNIST (2048 samples, $28 \times 28 \to 784$) and CIFAR-10 (2048 samples converted to grayscale, $32 \times 32 \to 1024$).
- **Noise Settings:** Rician noise $n = 0.3(0.3 z_1 + z_2^2)$ where $z_1, z_2 \sim \mathcal{N}(0, 1)$, yielding input PSNR $\approx 4.8$ dB.
- **Nonnegativity Enforcements:** Softplus, Absolute Value, ReLU.
- **Loss Function:** $\ell_2$ data fidelity + $\ell_1$ penalty on factor matrices.
- **Baselines:** Standard NMF (Lee & Seung, 2001 multiplicative update).
- **Paper Reported Values (PSNR dB over 10 runs):**
  - MNIST: DeepTensor (Softplus $7.4 \pm 0.03$, Abs $7.3 \pm 0.01$, ReLU **$7.6 \pm 0.15$**) vs. Standard NMF: **7.5 dB**.
  - CIFAR-10: DeepTensor (Softplus $5.7 \pm 0.03$, Abs $5.7 \pm 0.02$, ReLU **$6.3 \pm 0.04$**) vs. Standard NMF: **5.6 dB**.
- **Implementation Script:** `run_table1.py` (author release name for Table 2/3).
- **Execution Status:** **UNRUN** (script present but unexecuted).

---

### Exp 16: NMF and Semi-NMF on Speech Spectrograms (Table 3)
- **Application:** 09_speech_time_frequency
- **Dataset / Source:** Google Speech Commands dataset (`torchaudio.datasets.SPEECHCOMMANDS`).
- **Spectrogram Parameters:** STFT with window size 1024, hop size 32, yielding $512 \times 512$ time-frequency images (40 utterances stacked).
- **Noise Settings:** Rician noise, yielding input SNR $\approx 4.6$ dB.
- **Decomposition Modes:**
  - NMF: Nonnegativity on both $U$ and $V$.
  - Semi-NMF: Nonnegativity on $U$ only.
- **Activation Functions:** Softplus, Absolute Value, ReLU.
- **Baselines:** Standard NMF (4.6 dB) and Semi-NMF (4.7 dB).
- **Paper Reported Values (PSNR dB over 10 runs):**
  - NMF: Softplus **$9.0 \pm 0.03$**, Abs **$8.7 \pm 0.01$**, ReLU **$8.7 \pm 0.03$** vs. Baseline NMF: **4.6 dB**.
  - Semi-NMF: Softplus **$8.7 \pm 0.09$**, Abs **$8.8 \pm 0.02$**, ReLU **$8.8 \pm 0.02$** vs. Baseline Semi-NMF: **4.7 dB**.
- **Implementation Script:** `run_table1.py`.
- **Execution Status:** **UNRUN** (script present but unexecuted).

