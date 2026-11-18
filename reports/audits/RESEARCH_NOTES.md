# NeuroTensor Research Notes & Scientific Observations

This document captures architectural insights, empirical findings, and scientific observations made during the DeepTensor reproduction and Hybrid CNN–Transformer comparison project.

---

## 1. Inductive Bias in Low-Rank Tensor Recovery: CNN vs. Transformer

### The Implicit Regularization of Deep Image Prior (DIP)
- In classical low-rank factorization, low rank alone ($X = U V$) is often insufficient to prevent overfitting to non-Gaussian noise (Poisson, Rician, sparse impulse noise) or ill-conditioned linear measurements (Radon transform, compressive sensing).
- DeepTensor injects an architectural prior by parametrizing factor matrices as outputs of deep convolutional networks:
  $$U = f_U(z_U; \theta_U), \quad V = f_V(z_V; \theta_V)$$
- Convolutional networks with skip connections exhibit a pronounced spectral bias: they fit low-frequency, spatially/spectrally smooth signals rapidly while fitting high-frequency stochastic noise exponentially slower (Figure 3 in paper).
- This explains why early-stopped or moderately trained DeepTensor networks consistently outperform truncated SVD and classical algorithms on non-Gaussian noise regimes (Poisson, Rician, Salt-and-Pepper).

### What We Learned from ICVL Extensions 8 & 9 (Hybrid vs. CNN Baseline)
- In Extension 8, we augmented the 1D spectral DIP network with multi-head self-attention (`modules/attention_spectral.py`).
  - Empirical result: Hybrid PSNR was **45.39 dB** vs. matched control **45.72 dB** vs. CNN baseline **46.28 dB** (-0.89 dB delta).
- In Extension 9, we replaced the convolutional spectral generator with a Transformer-dominant backbone with ALiBi relative positional biases (`modules/spectral_transformer.py`).
  - Empirical result: Transformer PSNR was **44.08 dB** vs. matched control **42.67 dB** vs. CNN baseline **46.28 dB** (-2.20 dB delta).
- **Core Scientific Finding:**
  - Standard self-attention has a permutation-equivariant inductive bias, lacking the localized translation-equivariance and high-frequency damping inherent to convolutional filter banks.
  - For smooth continuous 1D spectra (such as ICVL reflectance curves across 260 bands), localized 1D convolutions provide a superior prior than unconstrained global token mixing.
  - **Hypothesis for Cross-Application Testing:** Transformers are expected to show stronger relative advantages in applications where long-range non-local dependencies or discrete semantic groupings dominate (e.g., Speech spectrogram harmonics in App 09, multi-subject Eigenface clustering in App 04, or long-range temporal motions in Video CS App 07), whereas spatially/spectrally smooth volumes (App 05, App 08) strongly favor convolutional priors.

---

## 2. Resolving Discrepancies in the Original Paper

### Discrepancy 1: Video Compressive Sensing Frame Count (Figure 11)
- **Paper Caption (Page 9):** *"We simulated video CS following the setup by [54] to combine 32 frames into a single coded image frame. We then recovered with DeepTensor at full rank, and with 2D TV."*
- **Paper Body Text (Section 4.3, Page 7):** *"We combined 8 frames into one single coded image and then used DeepTensor to solve the linear inverse problem."*
- **Figure 11 Visual Display:** Illustrates exactly 8 frames (labeled `Frame 1` through `Frame 8`).
- **Original Source Code (`run_figure11.py`):** Explicitly hard-codes `nframes = 8` and loads `cat_video.mat` (total 64 frames of size $128 \times 128$).
- **Resolution:** The caption contains an editorial typo ("32 frames" from an earlier draft or different experiment variant). The actual experimental protocol, visual figure, and verified author code uniformly operate on **$T = 8$ frames**.

### Discrepancy 2: Table Numbering in Author Scripts
- The author repository named the NMF script `run_table1.py`, but its docstring and contents implement MNIST, CIFAR-10, and Google Speech Commands NMF.
- In the published paper, Table 1 is the *Learning Rate Schedule Grid*, Table 2 is *MNIST/CIFAR NMF*, and Table 3 is *Speech Commands NMF*.
- **Resolution:** In `paper_reproduction/`, we separate these into dedicated, accurately labeled runners (`run_table1_lr_schedulers.py`, `run_table2_nmf_mnist_cifar.py`, and `run_table3_speech_nmf.py`).

---

## 3. Critical Implementation Discipline

1. **BatchNorm Evaluation Mode:**
   - Deep Image Prior generator networks contain multiple BatchNorm layers.
   - When evaluating final reconstruction metrics or saving outputs, DIP networks **must remain in `.train()` mode** (`model.train()`) to prevent running-statistic collapse caused by static noise inputs.
2. **Ground-Truth Data Isolation:**
   - Ground truth matrices must strictly be used for post-hoc metric computation (PSNR, SSIM, RMSE).
   - Never pass ground truth to optimizers, loss functions, early stopping logic, or rank selectors.
3. **Reproducibility Over Speed:**
   - Always log random seeds, PyTorch CUDA deterministic flags, execution wall-clock time, and peak GPU memory (`torch.cuda.max_memory_allocated()`).

