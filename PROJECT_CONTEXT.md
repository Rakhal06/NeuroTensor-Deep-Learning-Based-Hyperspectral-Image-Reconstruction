# DeepTensor Research Project: Master Context Document

## 1. Project Identity
- **Project Name:** DeepTensor-Extended (DeepTensor-Low-Rank-Tensor-Decomposition)
- **Faculty:** Not explicitly specified in the current context
- **Students:** Rakhal
- **Research Topic:** DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors
- **Main Objective:** Extend the baseline DeepTensor framework to support Adaptive Rank Selection and Sparsity-Regularized Rank Discovery, removing the need for manual rank specification, with future directions toward tensor completion, robustness, and Transformer-based extensions.

## 2. Research Problem
Classical low-rank tensor decomposition and the original DeepTensor approach require a fixed, manually specified rank hyperparameter (e.g., $R=20$). Selecting this rank is often arbitrary or requires computationally expensive trial-and-error grid searches. The core research problem is how to dynamically and adaptively determine the optimal effective rank during self-supervised optimization without relying on clean ground truth data.

## 3. Completed Extensions & Baseline Verification

### A. Original Frozen Baseline (Full-Data)
- **Commit:** `a146cc7` ("Baseline: reproduce DeepTensor Figure 9 on RTX 4060") on `main`
- **Script:** `run_figure9.py` (Frozen)
- **Dataset:** ICVL Hyperspectral (`data/icvl.mat`), shape `(174, 164, 260)`, `noise_snr=2`, `tau=100`
- **Method:** Fixed Rank-20, Full-Data 100% loss computation, 5000 epochs, Adam (`lr=1e-3`)
- **Verified Results:**
  - Input Noisy PSNR: 28.43 dB
  - SVD Rank-20 PSNR: ~38.31 dB | SSIM: ~0.9169
  - DeepTensor Rank-20 PSNR: 46.45 dB | SSIM: 0.9872
  - Runtime: ~159.57 sec
  - Peak GPU VRAM: 640.37 MB

### B. Extension 1: Adaptive Rank Selection — COMPLETED AND FROZEN
- **Commit:** `85ff484` ("Fix adaptive rank reporting for incomplete cooldown") on `extension/adaptive-rank`
- **Script:** `run_figure9_adaptive.py` (Frozen)
- **Terminology:** **"Self-Supervised Online Model Selection for Effective Rank"**
  *(Not theoretical tensor-rank discovery or mathematical rank recovery)*
- **Framework:**
  - Initial active rank $R_\text{start}=4$, max capacity $R_\text{max}=30$, step $R_\text{step}=2$.
  - Fixed 5% held-out binary mask for self-supervised validation (95% train / 5% val).
  - Train & Val losses calculated strictly independently without ground truth leaks.
  - Directional plateau criterion (`0 <= rel_train_improvement < 1e-3`) triggers rank growth if validation EMA improved over previous plateau.
  - PyTorch Adam handles newly unmasked parameter rows naturally (custom moment resetting removed).
  - Unlocked end-of-run fallback reports `selected_rank = best_rank_at_best` at `best_epoch`.

- **Adaptive Run #1 Results (5000 Epochs):**
  - Runtime: 274.61 sec | Peak VRAM: 1006.66 MB
  - Selected Rank: **Rank 8** (Best Val Epoch: 4841, Val L1: $2.991574 \times 10^{-2}$)
  - Active Rank Trajectory: $4 \to 6$ (ep 2358) $\to 8$ (ep 3923) $\to 10$ (ep 4851, mid-cooldown)
  - Reconstruction Quality: **PSNR 46.37 dB | SSIM 0.9872**

- **Adaptive Run #2 Results (Reproducibility Run, 5000 Epochs):**
  - Runtime: 273.97 sec | Peak VRAM: 1006.66 MB
  - Selected Rank: **Rank 8** (Best Val Epoch: 4930, Val L1: $2.999883 \times 10^{-2}$)
  - Active Rank Trajectory: $4 \to 6$ (ep 2090) $\to 8$ (ep 3735) $\to 10$ (ep 4938, mid-cooldown)
  - Reconstruction Quality: **PSNR 45.95 dB | SSIM 0.9856**

### C. Matched-Mask Fixed Rank-20 Control — COMPLETED
- **Purpose:** Isolate the effect of the 5% blind-spot mask from adaptive rank selection.
- **Setup:** Temporary run of `run_figure9_adaptive.py` with `R_START=20, R_MAX=20` (restored immediately afterward via `git checkout`).
- **Control Results (5000 Epochs):**
  - Runtime: 274.06 sec | Peak VRAM: 1007.31 MB
  - Selected Rank: **Rank 20** (Best Val Epoch: 4915, Val L1: $3.001085 \times 10^{-2}$)
  - Reconstruction Quality: **PSNR 46.14 dB | SSIM 0.9872 | RMSE ~0.00493**
  - Code restored clean: `git diff -- run_figure9_adaptive.py` is completely empty.

### D. Scientific Conclusion for Extension 1
The self-supervised online model-selection mechanism consistently selected **Rank 8** across independent stochastic runs, achieving reconstruction quality matching the matched fixed Rank-20 control (46.37 dB vs. 46.14 dB). This proves that self-supervised validation can discover a compact effective rank without manually tuning Rank 20 or relying on clean ground truth.

---

## 4. Extension 2: Factor Group-Sparsity (Group Lasso) Regularization — APPROVED DESIGN

### Status
**APPROVED — READY FOR IMPLEMENTATION.**
This extension is an independent alternative to Adaptive Rank Selection. **Do NOT combine the two methods.**

### Research Objective
Investigate whether continuous joint factor group-sparsity regularization can suppress unnecessary rank components in an over-parameterized DeepTensor decomposition ($R_\text{max}=30$), allowing a compact effective rank to emerge without manually specifying the final rank.

### Core Formulation & Architecture
Uses the **ORIGINAL UNNORMALIZED DeepTensor architecture**:
$$\hat{Y} = U V^T$$
where:
- $U \in \mathbb{R}^{1 \times HW \times R_\text{max}}$
- $V \in \mathbb{R}^{1 \times R_\text{max} \times B}$
- $R_\text{max} = 30$

For each rank component $r \in \{1, \dots, R_\text{max}\}$, the joint factor norm is:
$$g_r = \sqrt{\|U_r\|_2^2 + \|V_r\|_2^2}$$

The regularization term is defined as:
$$\mathcal{R}(U,V) = \sum_{r=1}^{R_\text{max}} g_r$$

Training objective:
$$\mathcal{L}_\text{total} = \frac{1}{|\Omega_\text{train}|} \left\| \mathcal{P}_{\Omega_\text{train}} \left( Y - \hat{Y} \right) \right\|_1 + \lambda \, \mathcal{R}(U,V)$$

### Abandonment of Previous Formulation (Record of Failure)
The previously considered normalized-gamma formulation ($\hat{Y} = \sum_r \gamma_r \hat{U}_r \hat{V}_r^T$ with unit-normalized factors and explicit $\gamma_r$ parameterization) was **abandoned**.
- **Reason for Abandonment:** Unit-normalization combined with explicit $\gamma$ parameterization caused severe scale and optimization problems under Adam ($lr=1\text{e-}3$). Specifically, it caused a catastrophic failure of the mandatory $\lambda=0$ control run (achieving only ~25.1 dB PSNR vs. the baseline ~46.4 dB PSNR) because scale dynamics and optimization bounds were severely degraded.

### Component Importance & Effective Rank Metrics
- **Component Power:**
  $$\text{power}_r = \|U_r\|_2 \cdot \|V_r\|_2$$
- **Primary Effective Rank:**
  $$R_\text{eff, primary} = \text{count of components satisfying } \text{power}_r > 0.01 \times \max_j(\text{power}_j)$$
- **Secondary Effective Rank:**
  Smallest number of components accounting for $\ge 99\%$ of total component power ($\frac{\sum_{i=1}^k \text{power}_{(i)}}{\sum_{j=1}^{R_\text{max}} \text{power}_{(j)}} \ge 0.99$).

### Hyperparameter Selection ($\lambda$)
- **Selection Criterion:** Selected **ONLY** by the lowest held-out 5% blind-spot validation loss ($\lambda^* = \arg\min_\lambda \mathcal{L}_\text{val}$).
- **Rule:** Ground-truth PSNR, SSIM, or RMSE must **NEVER** be used to select $\lambda$.
- **Predefined Grid:** $\lambda \in \{0, 10^{-4}, 10^{-3}, 10^{-2}\}$.
  - $\lambda=0$ serves as the mandatory architecture-only control.

### Fixed Settings
- $R_\text{max} = 30$
- $n_\text{inputs} = 20$
- Optimizer: Adam ($\text{learning rate} = 1\text{e-}3$)
- Epochs: 5000
- Dataset / Noise: Same ICVL preprocessing and noise configuration (`noise_snr=2`, `tau=100`)
- Validation Framework: Same 5% blind-spot validation framework with a fixed mask across all $\lambda$-grid runs (`seed=42`).

### Pipeline Audit: BatchNorm Evaluation Bug & Corrected Control
During Extension 2 evaluation, a critical pipeline discrepancy was identified and resolved:

1. **Pipeline Bug Discovery**:
   - `run_figure9_sparsity.py` previously executed `im_net.eval()` and `spec_net.eval()` prior to final reconstruction on the best validation checkpoint.
   - The DIP skip networks contain ~60 `BatchNorm` layers (`BatchNorm2d` / `BatchNorm1d`).
   - In a single-image unsupervised DIP setting with perturbed inputs, accumulated running statistics are unreliable, creating a severe train-vs-eval mode performance discrepancy.

2. **Controlled BatchNorm Diagnostic Proof**:
   - A controlled 500-epoch test evaluated the **EXACT SAME** checkpoint weights (`best_im_net_state`, `best_spec_net_state`) and **EXACT SAME** input tensors (`im_inp`, `spec_inp`):
     - **TRAIN Mode (`im_net.train()`)**: PSNR **37.6465 dB** | SSIM **0.926632** | RMSE **0.013112**
     - **EVAL Mode (`im_net.eval()`)**: PSNR **34.7731 dB** | SSIM **0.913607** | RMSE **0.018253**
     - **Performance Gap**: **~2.87 dB** drop in eval mode on identical weights/inputs.
     - **Tensor Difference**: Max absolute difference = `0.093265`, Mean absolute difference = `0.008068`.

3. **Evaluation Pipeline Fix**:
   - The evaluation section in `run_figure9_sparsity.py` now uses `im_net.train()` and `spec_net.train()`, matching the baseline (`run_figure9.py`) and adaptive (`run_figure9_adaptive.py`) pipeline behavior.
   - All temporary diagnostic code was removed.

4. **Corrected Definitive Control Experiment (`lambda=0`, `R_MAX=20`)**:
   - Run configuration: `R_MAX=20`, `lambda=0`, `seed=42`, `epochs=5000` (`results_sparsity_diagnostic`).
   - **Runtime**: 320.58 seconds
   - **Best Validation Epoch**: 4705 (Best Val L1: `0.03005810`)
   - **PSNR**: **46.4172 dB**
   - **SSIM**: **0.989254**
   - **RMSE**: **0.004774**
   - **Primary Effective Rank**: 20
   - **Secondary 99% Effective Rank**: 18
   - **Peak VRAM**: 966.56 MB
   - **Errors / Warnings**: None.

5. **Invalidated / Superseded Historical Runs**:
   - **SUPERSEDED**: The uncorrected `R_MAX=20, lambda=0` diagnostic run (**38.60 dB**) is marked **INVALID** (generated prior to the BatchNorm evaluation fix).
   - **SUPERSEDED**: All previous `R_MAX=30` sparsity-grid runs ($\lambda \in \{0, 1\text{e-}4, 1\text{e-}3, 1\text{e-}2\}$) are marked **INVALID** for final scientific comparison (generated prior to the BatchNorm evaluation fix).

### Final Verified Results (Corrected Definitive Grid)
- **Dataset:** ICVL
- **R_MAX:** 30
- **Seed:** 42
- **Epochs:** 5000
- **n_inputs:** 20
- **Mask:** Fixed 5% blind-spot mask
- **Optimizer:** Adam, lr=1e-3
- **Evaluation:** Corrected final evaluation uses `im_net.train()` and `spec_net.train()`

| Parameter ($\lambda$) | Best Val Epoch | Best Val L1 Loss | PSNR (dB) | SSIM | RMSE | Primary $R_\text{eff}$ | Secondary 99% $R_\text{eff}$ | Runtime (s) | Peak VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0** (Control) | 4977 | 0.03003683 | 46.8587 | 0.989582 | 0.004538 | 27 | 27 | 315.42 | 968.18 |
| **1e-4** | 4974 | 0.03024783 | 42.4859 | 0.983794 | 0.007511 | 3 | 3 | 315.41 | 968.18 |
| **1e-3** | 4832 | 0.03381762 | 33.0366 | 0.927514 | 0.022295 | 1 | 1 | 313.17 | 967.54 |
| **1e-2** | 24 | 0.1022802 | 18.8475 | 0.029707 | 0.114113 | 30 | 30 | 310.52 | 967.32 |

### Final Scientific Conclusion and Framing
Extension 2 is a valid negative/analytical result. A well-motivated hypothesis about continuous rank regularization was experimentally tested with controlled self-supervised validation, yielding a negative but informative result: factor group-sparsity achieves mechanical rank sparsification but incurs a disproportionately large reconstruction-quality cost under the tested DeepTensor setting.

The group-lasso factor penalty mechanically produces strong component sparsification (from effective rank 27 at $\lambda=0$ to effective rank 3 at $\lambda=10^{-4}$ and rank 1 at $\lambda=10^{-3}$). However, this sparsification causes a steep reconstruction-quality loss ($46.86\text{ dB} \to 42.49\text{ dB} \to 33.04\text{ dB}$). No favorable quality-compression tradeoff was found in the tested lambda grid. Self-supervised 5% held-out validation correctly selects $\lambda=0$, meaning the best tested regularization strength is no explicit regularization.

**Contrast with Extension 1:**
Extension 1 uses bottom-up adaptive rank growth and achieved a selected rank of 8 with a PSNR of 46.37 dB (SSIM 0.9872). Extension 2 uses top-down factor sparsification and produced rank 3 at 42.49 dB and rank 1 at 33.04 dB. Under the tested DeepTensor architecture and ICVL experimental setting, top-down factor suppression produced a substantially worse quality–sparsity tradeoff than bottom-up adaptive rank growth.

### Research Terminology
- Official Name: **Factor Group-Sparsity (Group Lasso) Regularization**
- *Note:* Do NOT refer to this method as a nuclear-norm surrogate.

---

## 5. Decision Log

1. **[Verified]** Environment: RTX 4060 Laptop GPU, Windows 11, PyTorch 2.5.1+cu121.
2. **[Verified]** Baseline Reproduction: Fixed Rank-20 full-data baseline reproduced and frozen (`main` branch, commit `a146cc7`).
3. **[Verified]** Extension 1 Completed: Adaptive Rank Selection implemented, smoke-tested, evaluated across 2 independent runs ($R=8$ selected), matched control run ($R=20$), and frozen (`extension/adaptive-rank` branch, commit `85ff484`).
4. **[Abandoned]** Normalized Gamma Formulation: Abandoned explicit $\gamma$ parameterization with unit-normalized factors after $\lambda=0$ control failed (~25.1 dB PSNR) due to severe scale/optimization issues under Adam.
5. **[Approved]** Extension 2 Final Design: Factor Group-Sparsity (Group Lasso) Regularization using original unnormalized DeepTensor architecture ($\hat{Y} = U V^T$), $R_\text{max}=30$, $g_r = \sqrt{\|U_r\|_2^2 + \|V_r\|_2^2}$, $\text{power}_r = \|U_r\|_2 \|V_r\|_2$, $\lambda \in \{0, 10^{-4}, 10^{-3}, 10^{-2}\}$ evaluated strictly via 5% blind-spot validation loss.
6. **[Resolved]** BatchNorm Evaluation Bug: Confirmed that `im_net.eval()` and `spec_net.eval()` caused a ~2.87 dB degradation due to DIP BatchNorm running statistics. Fixed by setting networks to `train()` mode during final evaluation.
7. **[Verified]** Corrected Control Benchmark: `lambda=0, R_MAX=20` run achieved **46.42 dB PSNR** (matching the 46.14 dB baseline adaptive control). Previous buggy runs (38.60 dB diagnostic and pre-fix $R_\text{max}=30$ grid) are marked superseded.
8. **[Completed]** Extension 2 Final Rerun: Completed the full corrected grid. Result is a negative but informative finding: top-down factor suppression yields a substantially worse quality-sparsity tradeoff than bottom-up adaptive rank growth. Extension 2 is now FROZEN.

---

## 6. Current Status & Next Immediate Action

| Task | Status |
|---|---|
| Baseline reproduction & verification | COMPLETE (Commit `a146cc7`) |
| Adaptive-rank implementation & testing | COMPLETE (Commit `85ff484`) |
| Adaptive-rank 5000-epoch runs & control | COMPLETE ($R=8$ selected) |
| Adaptive-rank scientific review | COMPLETE & FROZEN |
| Extension 2 design & failure review | COMPLETE (Normalized-gamma abandoned, Group Lasso approved) |
| Extension 2 BatchNorm bug identification & fix | COMPLETE (`im_net.train()` restored, 46.42 dB control verified) |
| Extension 2 full grid rerun & analysis | COMPLETE (Negative/analytical result documented) |

**Immediate Next Step:** All planned extensions are completed and frozen. Await further project directions.

---

## 7. Important Files Registry
| File / Directory | Purpose | Status |
|---|---|---|
| `run_figure9.py` | Baseline DeepTensor script (Fixed Rank-20). | **FROZEN** (Commit `a146cc7`) |
| `run_figure9_adaptive.py` | Extension 1: Adaptive Rank Selection. | **FROZEN** (Commit `85ff484`) |
| `run_figure9_sparsity.py` | Extension 2: Factor Group-Sparsity (Group Lasso). | **FROZEN** |
| `modules/` | Core network definitions & utilities. | **FROZEN** |
| `data/icvl.mat` | ICVL Hyperspectral Dataset. | **FROZEN** |
| `PROJECT_CONTEXT.md` | Master handoff and research context document. | **UPDATED** |

---

## 8. Research Integrity Rules
- Clean ground truth must **NEVER** be used for training gradients, rank selection, stopping, or hyperparameter tuning.
- Ground truth is reserved strictly for post-hoc evaluation (PSNR / SSIM / RMSE).
- Do not claim theoretical tensor-rank recovery; distinguish mathematical rank from effective rank.
- Always run matched-mask controls when introducing architectural or loss modifications.
- Experimental evidence supersedes hypotheses.
