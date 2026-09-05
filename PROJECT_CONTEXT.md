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

## 4. Extension 2: Sparsity-Regularized Rank Discovery — APPROVED DESIGN (NOT YET IMPLEMENTED)

### Status
**APPROVED WITH MODIFICATIONS — NOT YET IMPLEMENTED.**
This extension is an independent alternative to Adaptive Rank Selection. **Do NOT combine the two methods.**

### Research Objective
Investigate whether continuous sparsity regularization can suppress unnecessary rank components in an over-parameterized DeepTensor decomposition ($R_\text{max}=30$), allowing a compact effective rank to emerge without manually specifying the final rank.

### Critical Mathematical Decision: Scale Identifiability Flaw & Fix
- **Rejected Formulation:** The original proposal used $\hat{Y} = \sum_{r} \gamma_r (U_r \otimes V_r)$ with free $\gamma_r \ge 0$ and $L_1$ penalty $\lambda \sum_r \gamma_r$.
  *Flaw:* For any $c > 0$, scaling $\gamma_r \to \gamma_r / c$ and $U_r \to c U_r$ leaves the reconstruction unchanged while artificially shrinking the $L_1$ penalty. Free $\gamma$ cannot reliably represent component importance because $U$ and $V$ can absorb the scale.
- **Final Approved Formulation:** Unit-normalize spatial and spectral factor components prior to scaling:
  $$\hat{Y} = \sum_{r=1}^{R_\text{max}} \gamma_r \left( \hat{U}_{:,r} \otimes \hat{V}_{r,:} \right) \quad \text{where} \quad \|\hat{U}_{:,r}\|_2 = 1, \quad \|\hat{V}_{r,:}\|_2 = 1, \quad \gamma_r \ge 0$$
  $$\mathcal{L}_\text{total} = \frac{1}{|\Omega_\text{train}|} \left\| \mathcal{P}_{\Omega_\text{train}} \left( Y - \hat{Y} \right) \right\|_1 + \lambda \sum_{r=1}^{R_\text{max}} \gamma_r$$
  - Uses $\epsilon$-stabilized normalization ($\frac{U_{:,r}}{\|U_{:,r}\|_2 + \epsilon}$) to prevent zero-division.
  - Honest interpretation: An interpretable reparameterization closely equivalent to Group Lasso.

### Effective Rank Definition
- **Primary Measure (Fixed Pre-experiment):**
  $$R_\text{eff} = \text{count of components satisfying } \gamma_r > 0.01 \times \max_j(\gamma_j)$$
- **Secondary Measure:** Smallest $k$ reaching $99\%$ cumulative $\gamma$ energy ($\frac{\sum_{i=1}^k \gamma_{(i)}}{\sum_{j=1}^{R_\text{max}} \gamma_{(j)}} \ge 0.99$).
- *Rule:* Thresholds are fixed in advance and must NOT be adjusted after observing results.

### Hyperparameter Selection ($\lambda$)
- Select $\lambda^*$ using **ONLY** the 5% blind-spot self-supervised validation framework ($\lambda^* = \arg\min_\lambda \mathcal{L}_\text{val}$).
- **Predefined Grid:** $\lambda \in \{0, 10^{-4}, 10^{-3}, 10^{-2}\}$. ($\lambda=0$ is a mandatory control with normalized factors but no sparsity penalty).
- Ground-truth PSNR/SSIM/RMSE must **NEVER** be used to select $\lambda$.

### Execution & Script Isolation
- Implementation script: `run_figure9_sparsity.py` (**Standalone script, NOT created yet**).
- Candidate capacity $R_\text{max} = 30$, $n_\text{inputs} = 20$, Adam ($\text{lr}=1\text{e-}3$), 5000 epochs, RTX 4060 GPU.
- Frozen baseline (`run_figure9.py`) and frozen adaptive (`run_figure9_adaptive.py`) remain completely untouched.

### Planned Experimental Procedure
1. Create and smoke-test `run_figure9_sparsity.py` (verify normalization, $\gamma$ non-negativity, gradients, GPU).
2. Run single passes for each grid setting ($\lambda \in \{0, 10^{-4}, 10^{-3}, 10^{-2}\}$).
3. Select $\lambda^*$ strictly via validation loss.
4. Repeat the $\lambda^*$ configuration once with a second random seed for reproducibility.
5. Compare against frozen baselines (Full-Data Rank 20, Matched-Mask Rank 20, Adaptive Rank 8).

### Scientific Framing & Expected Interpretation
- **Scientific Distinctness:** Adaptive Rank (discrete temporal growth) and Sparsity (continuous spectral pruning) are mechanistically distinct alternatives for model selection. However, because both rely on the same 5% blind-spot validation framework, finding a similar rank should be framed as **"convergent evidence from two mechanistically distinct effective-rank selection strategies"**, NOT as completely independent validations.
- **Allowed Claims:** Automatic selection of a compact effective rank matching fixed Rank 20 quality; interpretable component-scale signal via normalized $\gamma$.
- **Disallowed Claims:** True mathematical tensor-rank recovery; theoretical rank discovery; superiority over Adaptive Rank; statistical significance from small sample size.

---

## 5. Decision Log

1. **[Verified]** Environment: RTX 4060 Laptop GPU, Windows 11, PyTorch 2.5.1+cu121.
2. **[Verified]** Baseline Reproduction: Fixed Rank-20 full-data baseline reproduced and frozen (`main` branch, commit `a146cc7`).
3. **[Verified]** Extension 1 Completed: Adaptive Rank Selection implemented, smoke-tested, evaluated across 2 independent runs ($R=8$ selected), matched control run ($R=20$), and frozen (`extension/adaptive-rank` branch, commit `85ff484`).
4. **[Approved]** Extension 2 Design: Multiplicative scaling $\boldsymbol{\gamma}$ with unit-normalized factors $\hat{U}, \hat{V}$ approved to eliminate scale identifiability. Free $\gamma$ formulation rejected.
5. **[Approved]** Extension 2 Parameters: $R_\text{max}=30$, primary $R_\text{eff}$ threshold at $1\%$ of $\max(\gamma)$, $\lambda \in \{0, 10^{-4}, 10^{-3}, 10^{-2}\}$ selected via 5% blind-spot validation.

---

## 6. Current Status & Next Immediate Action

| Task | Status |
|---|---|
| Baseline reproduction & verification | COMPLETE (Commit `a146cc7`) |
| Adaptive-rank implementation & testing | COMPLETE (Commit `85ff484`) |
| Adaptive-rank 5000-epoch runs & control | COMPLETE ($R=8$ selected) |
| Adaptive-rank scientific review | COMPLETE & FROZEN |
| Sparsity extension design & review | COMPLETE (Approved with modifications) |
| **Sparsity extension implementation** | **NEXT STEP (Implementation Design & Smoke Test)** |

**Immediate Next Step:** Prepare implementation plan for standalone script `run_figure9_sparsity.py` strictly following the unit-normalized $\boldsymbol{\gamma}$ formulation.

---

## 7. Important Files Registry
| File / Directory | Purpose | Status |
|---|---|---|
| `run_figure9.py` | Baseline DeepTensor script (Fixed Rank-20). | **FROZEN** (Commit `a146cc7`) |
| `run_figure9_adaptive.py` | Extension 1: Adaptive Rank Selection. | **FROZEN** (Commit `85ff484`) |
| `run_figure9_sparsity.py` | Extension 2: Sparsity-Regularized Rank Discovery. | **NOT YET CREATED** |
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
