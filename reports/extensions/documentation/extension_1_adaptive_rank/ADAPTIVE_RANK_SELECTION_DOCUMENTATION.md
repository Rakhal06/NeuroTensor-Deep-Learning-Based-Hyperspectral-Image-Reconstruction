# Extension 1 — Adaptive Rank Selection

## 1. Executive Summary

- **Objective:** Dynamically determine an optimal effective tensor rank during self-supervised training, replacing DeepTensor's manually specified fixed rank (e.g., $R=20$).
- **Motivation:** Avoid computationally expensive trial-and-error grid searches or arbitrary rank selection while maintaining high reconstruction quality without clean ground-truth access.
- **Final Implementation:** Implemented in `run_figure9_adaptive.py`. The algorithm initializes networks with over-parameterized capacity ($R_\text{max}=30$) but reconstructs using only an active slice ($R_\text{start}=4$). It monitors an exponential moving average (EMA) of self-supervised training reconstruction loss over a sliding window (`PLATEAU_WINDOW=200`). When training loss improvement plateaus (`0 <= rel_train_improvement < 1e-3`), active rank increases by $R_\text{step}=2$, provided the 5% held-out validation EMA loss has improved since the last rank growth event.
- **Key Result:** Across two independent stochastic 5000-epoch runs on the ICVL hyperspectral dataset, the self-supervised online model-selection mechanism consistently selected **Rank 8** (PSNR 46.37 dB | SSIM 0.9872 in Run 1; PSNR 45.95 dB | SSIM 0.9856 in Run 2). This matched the performance of a matched-mask fixed Rank-20 control (PSNR 46.14 dB | SSIM 0.9872).
- **Final Status:** **COMPLETE / FROZEN**.

---

## 2. DeepTensor Limitation Being Addressed

Original baseline DeepTensor (and classical low-rank tensor decomposition methods) relies on a fixed, manually specified rank hyperparameter $R$ (e.g., $R=20$). 

This introduces two key limitations:
1. **Manual Tuning Burden:** Finding the optimal rank requires running multiple full training passes across a grid of rank candidates.
2. **Suboptimal Capacity / Risk of Overfitting:** Setting $R$ too small restricts representation capacity, whereas setting $R$ too large wastes memory and risks fitting sensor noise.

DeepTensor baseline provides deep image priors (DIP) for spatial and spectral factor matrices ($\hat{Y} = U V^T$), but provides no mechanism to infer how many rank-1 factors are necessary to reconstruct a given hyperspectral tensor.

---

## 3. Proposed Extension in the Project Proposal

**Project Proposal Specification:**
> "Implements adaptive rank selection using a plateau-based mechanism. Instead of fixing rank=20, the effective rank starts small and grows when the reconstruction loss plateaus."

- **Intended Objective:** Automate rank selection during self-supervised optimization.
- **Expected Role:** Core required Extension 1 of the DeepTensor research proposal.
- **Scope:** Introduce online rank expansion driven by reconstruction dynamics without altering the underlying DIP generator architectures.

---

## 4. Our Initial Research Idea

- **Initial Concept:** Start optimization with a small active rank $R_\text{start}=4$ within over-parameterized networks ($R_\text{max}=30$). When the 100%-data training reconstruction loss flatlined over a fixed window of epochs, increment active rank by $R_\text{step}=2$.
- **Initial Assumptions:**
  - Lower ranks learn global, low-frequency spatial-spectral structures first.
  - As lower-rank capacity saturates, training loss plateaus, signaling that additional rank components are needed for finer details.
- **Perceived Advantage:** Simple, intuitive, and computationally lightweight without requiring network architecture modifications.

---

## 5. Design Evolution

During research and development, several critical methodological and algorithmic flaws were identified in the initial concept, forcing a multi-stage design evolution:

1. **Fairness Issue Regarding $n_\text{inputs}$:**
   - *Problem:* In DeepTensor, network input channels $n_\text{inputs}$ match the rank ($n_\text{inputs}=R$). Dynamically changing $n_\text{inputs}$ during training would require rebuilding internal convolutional layers or zero-padding inputs mid-training.
   - *Revision:* Fixed $n_\text{inputs}=20$ across all runs (matching baseline) while decoupling input channel dimension from the active factor slicing rank.

2. **Adam Optimizer State Management at Rank Growth:**
   - *Problem:* Initially, we considered resetting or copying Adam running moments (`exp_avg`, `exp_avg_sq`) for newly unmasked output layer channels.
   - *Analysis:* PyTorch Adam naturally tracks state per-tensor. Newly activated output channels received zero gradient while masked, meaning their moments were exactly zero. When activated at a large global step, they experience a known transient effective learning rate multiplier of ~3.16x, which is perfectly stable at $lr=1\text{e-}3$.
   - *Revision:* Removed custom moment-resetting logic; allowed standard PyTorch Adam to manage parameters naturally.

3. **Training-Loss-Only Growth vs. Genuine Model Selection:**
   - *Problem:* Growing rank based purely on training loss plateauing causes the model to unconditionally expand rank to $R_\text{max}=30$, because higher rank almost always lowers training loss on noisy data (overfitting).
   - *Revision:* Introduced a **5% held-out blind-spot validation mask**. Rank expansion occurs ONLY if training loss plateaus AND validation loss EMA is lower than at the previous plateau. If validation loss stalls or degrades, rank growth is permanently locked.

4. **Terminology Refinement:**
   - *Problem:* Claiming "theoretical rank discovery" or "mathematical rank recovery" is unscientific for deep network priors.
   - *Revision:* Formally titled the approach: **"Self-Supervised Online Model Selection for Effective Rank"**.

---

## 6. Final Research Question / Hypothesis

**Research Question:** Can a self-supervised online model-selection mechanism, operating strictly on noisy observations via a held-out blind-spot validation loss, dynamically expand the active factor rank from $R=4$ to discover a compact effective rank that matches the reconstruction quality of a fixed Rank-20 baseline?

---

## 7. Final Methodology

- **Dataset:** ICVL Hyperspectral (`data/icvl.mat`), shape $(174, 164, 260)$, downscaled by `scaling=1`, normalized by `cube / cube.max()`.
- **Noise Model:** `noise_snr=2`, `tau=100` (Poisson photon noise + Gaussian readout noise).
- **Architecture:** Standard DeepTensor DIP generators (`deep_prior.get_net`). $n_\text{inputs}=20$, $R_\text{max}=30$.
- **Active Rank Slicing:** Factor matrices $U \in \mathbb{R}^{1 \times HW \times R_\text{max}}$ and $V \in \mathbb{R}^{1 \times R_\text{max} \times B}$ are sliced to active channels $R_\text{current}$:
  $$\hat{Y} = \mathbf{U}[:, :, :R_\text{current}] \mathbf{V}[:, :R_\text{current}, :]$$
- **Masking:** Fixed 5% random binary validation mask (`mask_val`), 95% training mask (`mask_train`), generated with `seed=42`.
- **Loss Function:** L1 loss over unmasked training pixels:
  $$\mathcal{L}_\text{train} = \frac{1}{|\Omega_\text{train}|} \sum_{(i,j) \in \Omega_\text{train}} |\hat{Y}_{ij} - Y_{ij}^\text{noisy}|$$
- **Plateau Detection & Growth Rule:**
  - Exponential Moving Average (EMA) decay: $\alpha = 0.99$.
  - Cooldown period: `MIN_EPOCHS_BEFORE_GROWTH = 300` epochs.
  - Evaluation window: `PLATEAU_WINDOW = 200` epochs.
  - Relative training improvement:
    $$\text{rel\_improvement} = \frac{\text{EMA}_\text{train}(t - 200) - \text{EMA}_\text{train}(t)}{\text{EMA}_\text{train}(t - 200)}$$
  - Growth Trigger: If $0 \le \text{rel\_improvement} < 10^{-3}$ AND $\text{EMA}_\text{val}(t) < \text{EMA}_\text{val}(\text{last\_growth})$, increment $R_\text{current} \leftarrow \min(R_\text{current} + 2, R_\text{max})$. Else if validation degrades, lock rank permanently.
- **Model Selection & Evaluation:** Best checkpoint tracked by minimum validation L1 loss. Final evaluation uses `im_net.train()` and `spec_net.train()` to maintain BatchNorm stability.

---

## 8. Implementation Details

- **Script:** [`run_figure9_adaptive.py`](file:///e:/Rakhal_Tensor/run_figure9_adaptive.py)
- **Key Parameters:**
  - `R_START = 4`, `R_MAX = 30`, `R_STEP = 2`
  - `PLATEAU_WINDOW = 200`, `PLATEAU_THRESHOLD = 1e-3`, `MIN_EPOCHS_BEFORE_GROWTH = 300`
  - `LOSS_EMA_ALPHA = 0.99`, `TOTAL_EPOCHS = 5000`
- **Output Artifacts:** `results/adaptive_rank_results.png`

---

## 9. Debugging and Methodological Issues

1. **Unconditional Rank Growth in Early Drafts:**
   - *Issue:* Early code grew rank based on raw epoch thresholds or unvalidated training loss, always reaching $R=30$.
   - *Fix:* Added validation EMA check (`val_ema_at_last_plateau`), enabling permanent rank locking when validation loss stutters.
2. **Unlocked End-of-Run Rank Selection:**
   - *Issue:* If training ended mid-cooldown after a rank increase before locking occurred, reporting the final active rank was misleading.
   - *Fix:* Added an explicit fallback: if unlocked at epoch 5000, report `selected_rank = best_rank_at_best` corresponding to the epoch of minimum validation loss.

---

## 10. Experimental Protocol

1. **Adaptive Run #1 (Primary):** 5000 epochs, default initialization, $R_\text{start}=4 \to R_\text{max}=30$.
2. **Adaptive Run #2 (Reproducibility):** 5000 epochs, independent random seed run.
3. **Matched-Mask Fixed Rank-20 Control:** 5000 epochs, $R_\text{start}=20, R_\text{max}=20$, identical 5% blind-spot mask.

---

## 11. Experimental Results

### Verified Definitive Results (5000 Epochs)

| Run / Model | Selected Rank | Best Val Epoch | Best Val L1 Loss | PSNR (dB) | SSIM | RMSE | Runtime (s) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adaptive Run #1** | **Rank 8** | 4841 | $2.9916 \times 10^{-2}$ | **46.37** | **0.9872** | ~0.00480 | 274.61 | 1006.66 |
| **Adaptive Run #2** | **Rank 8** | 4930 | $2.9999 \times 10^{-2}$ | **45.95** | **0.9856** | ~0.00504 | 273.97 | 1006.66 |
| **Fixed R=20 Control** | **Rank 20** | 4915 | $3.0011 \times 10^{-2}$ | **46.14** | **0.9872** | ~0.00493 | 274.06 | 1007.31 |

### Active Rank Trajectory (Run #1):
- Epoch 0–2357: Active Rank 4
- Epoch 2358–3922: Active Rank 6
- Epoch 3923–4850: Active Rank 8 *(Best Val Epoch: 4841)*
- Epoch 4851–5000: Active Rank 10 (Mid-cooldown at run completion; selected rank falls back to Rank 8)

---

## 12. Results Interpretation

- **Equivalence to Fixed Baseline:** Selected Rank 8 achieves 46.37 dB PSNR, matching the 46.14 dB PSNR of the matched-mask fixed Rank-20 control and 46.45 dB of the original full-data baseline.
- **60% Rank Reduction:** DeepTensor represents the dataset effectively with 8 rank components instead of 20, demonstrating significant over-parameterization in the manual baseline setting.

---

## 13. Comparison With Baseline / Other Extensions

- **Vs Baseline ($R=20$):** Matches PSNR (46.37 dB vs 46.45 dB) while reducing rank components by 60%.
- **Vs Extension 2 (Group Lasso):** Extension 1 (bottom-up growth) significantly outperforms Extension 2 (top-down sparsification), which suffered severe PSNR degradation when regularized (42.49 dB at rank 3, 33.04 dB at rank 1).

---

## 14. Limitations

1. **Dataset Limitation:** Evaluated on ICVL hyperspectral data only.
2. **Stochastic Cooldown Boundary:** Run completion at 5000 epochs can interrupt a cooldown period, requiring fallback logic to `best_rank_at_best`.
3. **Claims We MUST NOT Make:**
   - We MUST NOT claim theoretical tensor-rank recovery.
   - We MUST NOT claim exact mathematical rank identification.

---

## 15. Scientific Conclusion

Self-supervised online model selection via plateau detection and held-out validation successfully automates rank selection in DeepTensor. It discovers a compact effective rank ($R=8$) that achieves reconstruction accuracy on par with a fixed Rank-20 model, eliminating manual rank tuning.

---

## 16. Contribution to the Overall Project

Extension 1 resolves the core usability limitation of DeepTensor (manual rank specification) through a bottom-up rank growth mechanism, serving as a primary benchmark for adaptive model selection.

---

## 17. Final Status

**COMPLETE / FROZEN** — Fully implemented, validated across independent runs, compared against matched controls, and documented.

---

## 18. Reproducibility Reference

- **Script:** [`run_figure9_adaptive.py`](file:///e:/Rakhal_Tensor/run_figure9_adaptive.py)
- **Branch / Commit:** `extension/adaptive-rank` (Commit `85ff484`)
- **Master Documentation:** [`PROJECT_CONTEXT.md`](file:///e:/Rakhal_Tensor/PROJECT_CONTEXT.md)
