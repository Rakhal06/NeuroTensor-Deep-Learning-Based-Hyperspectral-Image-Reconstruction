# Extension 2 — Factor Group-Sparsity (Group Lasso) Regularization

## 1. Executive Summary

- **Objective:** Investigate whether continuous joint factor group-sparsity regularization can suppress unnecessary rank components in an over-parameterized DeepTensor decomposition ($R_\text{max}=30$), allowing a compact effective rank to emerge automatically.
- **Motivation:** Explore a top-down continuous regularization alternative to bottom-up discrete rank growth (Extension 1), using a Group Lasso penalty to drive unused factor matrices toward zero.
- **Final Implementation:** Implemented in `run_figure9_sparsity.py`. Uses the unnormalized DeepTensor factorized representation $\hat{Y} = U V^T$ with $R_\text{max}=30$. Regularizes the joint $L_2$ norm of spatial and spectral factor vectors for each component $r$:
  $$g_r = \sqrt{\|U_r\|_2^2 + \|V_r\|_2^2}, \quad \mathcal{R}(U,V) = \sum_{r=1}^{R_\text{max}} g_r$$
  Hyperparameter $\lambda \in \{0, 10^{-4}, 10^{-3}, 10^{-2}\}$ is evaluated using a 5% held-out blind-spot validation loss.
- **Key Result:** Factor Group-Sparsity achieves mechanical rank sparsification (reducing primary effective rank from 27 at $\lambda=0$ to 3 at $\lambda=10^{-4}$ and 1 at $\lambda=10^{-3}$), but incurs a severe reconstruction quality penalty (PSNR drops from 46.86 dB to 42.49 dB to 33.04 dB). Held-out validation correctly selects $\lambda=0$ (unregularized control).
- **Final Status:** **COMPLETE / FROZEN** (Valid negative/analytical research finding).

---

## 2. DeepTensor Limitation Being Addressed

Baseline DeepTensor requires hard-coding the factor rank $R$. While Extension 1 addresses this via discrete, bottom-up rank growth, an alternative paradigm is continuous regularization: over-parameterizing the network to $R_\text{max}=30$ and letting an optimization penalty zero out redundant factors.

Without explicit regularization, deep neural networks over-parameterized with $R_\text{max}=30$ retain arbitrary non-zero weights across all 30 channels, preventing automatic rank pruning.

---

## 3. Proposed Extension in the Project Proposal

**Project Proposal Specification:**
> "Implements factor group-sparsity (Group Lasso) regularization on the spatial and spectral factor matrices. Promotes joint column-wise sparsity across factor pairs to automatically prune redundant rank components during training."

- **Intended Objective:** Continuous rank discovery through group-sparse penalization.
- **Expected Role:** Core required Extension 2 of the DeepTensor research proposal, evaluated independently of Extension 1.

---

## 4. Our Initial Research Idea (Abandoned Normalized-Gamma Formulation)

- **Initial Concept:** Parameterize the decomposition explicitly with normalized factor vectors and explicit scalar weights $\gamma_r$:
  $$\hat{Y} = \sum_{r=1}^{R_\text{max}} \gamma_r \hat{U}_r \hat{V}_r^T, \quad \text{where } \|\hat{U}_r\|_2 = 1, \|\hat{V}_r\|_2 = 1$$
  Apply an $L_1$ penalty to vector $\boldsymbol{\gamma} = [\gamma_1, \dots, \gamma_R]^T$.
- **Why It Failed:** Explicit unit-normalization combined with $\gamma_r$ scaling severely disrupted gradient scaling under Adam ($lr=1\text{e-}3$). The mandatory $\lambda=0$ architecture control run collapsed to **~25.1 dB PSNR** (vs baseline ~46.4 dB).
- **Outcome:** The normalized-gamma formulation was **ABANDONED**.

---

## 5. Design Evolution

1. **Abandonment of Normalized-Gamma Formulation:**
   - *Problem:* Severe scale ambiguity and optimization collapse (~25.1 dB PSNR at $\lambda=0$).
   - *Decision:* Reverted to original unnormalized DeepTensor factorized architecture $\hat{Y} = U V^T$.
2. **Rejection of SVD Factor Initialization:**
   - *Idea:* Initialize factor networks using SVD components of the noisy input.
   - *Decision:* Rejected to preserve self-supervised deep image prior (DIP) random initialization.
3. **Formulation of Unnormalized Group Lasso:**
   - Define joint column group norm $g_r = \sqrt{\|U_r\|_2^2 + \|V_r\|_2^2}$ on unnormalized factors $U, V$.
4. **Identification and Resolution of BatchNorm Evaluation Bug:**
   - *Problem:* Early Extension 2 evaluation scripts invoked `im_net.eval()` and `spec_net.eval()` before final reconstruction, causing a severe ~2.87 dB drop due to DIP BatchNorm running statistics.
   - *Fix:* Restored `im_net.train()` and `spec_net.train()` for final reconstruction evaluation.

---

## 6. Final Research Question / Hypothesis

**Research Question:** Can joint Factor Group-Sparsity (Group Lasso) regularization on unnormalized factor matrices $U$ and $V$ automatically suppress redundant rank components in an over-parameterized model ($R_\text{max}=30$) while maintaining baseline reconstruction quality, and will self-supervised validation select a non-zero regularization strength $\lambda > 0$?

---

## 7. Final Methodology

- **Architecture:** Unnormalized DIP generators ($\hat{Y} = U V^T$), $R_\text{max}=30$, $n_\text{inputs}=20$.
- **Group Lasso Penalty:**
  $$\mathcal{R}(U,V) = \sum_{r=1}^{R_\text{max}} \sqrt{\|U_r\|_2^2 + \|V_r\|_2^2}$$
- **Total Training Loss:**
  $$\mathcal{L}_\text{total} = \frac{1}{|\Omega_\text{train}|} \sum_{(i,j) \in \Omega_\text{train}} |\hat{Y}_{ij} - Y_{ij}^\text{noisy}| + \lambda \, \mathcal{R}(U,V)$$
- **Component Power:** $\text{power}_r = \|U_r\|_2 \cdot \|V_r\|_2$
- **Effective Rank Metrics:**
  - *Primary Effective Rank:* Count of components with $\text{power}_r > 0.01 \times \max_j(\text{power}_j)$.
  - *Secondary 99% Effective Rank:* Smallest number of sorted components accounting for $\ge 99\%$ total power.
- **Validation Framework:** 5% blind-spot validation mask (`seed=42`). Best model selected at minimum validation L1 loss. Final evaluation executed in `train()` mode.

---

## 8. Implementation Details

- **Script:** [`run_figure9_sparsity.py`](file:///e:/Rakhal_Tensor/run_figure9_sparsity.py)
- **Key Command-Line Arguments:** `--lambda_value`, `--seed`, `--epochs`, `--r_max`, `--output_dir`
- **Output Artifacts:** `results_sparsity_corrected/summary_lambda_{val}_seed_42.txt`

---

## 9. Debugging and Methodological Issues: The BatchNorm Evaluation Bug

During Extension 2 evaluation, a critical pipeline discrepancy was discovered and resolved:

### The Bug
`run_figure9_sparsity.py` previously executed `im_net.eval()` and `spec_net.eval()` prior to final checkpoint reconstruction. Because DIP skip networks contain ~60 BatchNorm layers, using population running statistics accumulated during single-image input-perturbed training caused severe distribution shift.

### Controlled Diagnostic Proof (500 Epochs, Identical Weights & Inputs)
- **TRAIN Mode (`im_net.train()`):** PSNR **37.6465 dB** | SSIM **0.926632** | RMSE **0.013112**
- **EVAL Mode (`im_net.eval()`):** PSNR **34.7731 dB** | SSIM **0.913607** | RMSE **0.018253**
- **Discrepancy:** **~2.87 dB** degradation in eval mode. Max absolute pixel difference: `0.093265`.

### Fix & Impact
Set `im_net.train()` and `spec_net.train()` during final evaluation. All pre-fix $R_\text{max}=30$ runs (which reported lower PSNRs) were marked **INVALID / SUPERSEDED** and re-run.

---

## 10. Experimental Protocol

1. **Controlled Baseline Check (`R_MAX=20, lambda=0`):** Verify corrected evaluation pipeline.
2. **Definitive Lambda Grid (`R_MAX=30`):** Evaluate $\lambda \in \{0, 10^{-4}, 10^{-3}, 10^{-2}\}$ under seed 42 for 5000 epochs.

---

## 11. Experimental Results

### Verified Definitive Results (Corrected Grid, 5000 Epochs, `seed=42`)

| Parameter ($\lambda$) | Best Val Epoch | Best Val L1 Loss | PSNR (dB) | SSIM | RMSE | Primary $R_\text{eff}$ | Secondary 99% $R_\text{eff}$ | Runtime (s) | Peak VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0** (Control) | 4977 | $3.0037 \times 10^{-2}$ | **46.8587** | **0.989582** | **0.004538** | 27 | 27 | 315.42 | 968.18 |
| **1e-4** | 4974 | $3.0248 \times 10^{-2}$ | **42.4859** | **0.983794** | **0.007511** | 3 | 3 | 315.41 | 968.18 |
| **1e-3** | 4832 | $3.3818 \times 10^{-2}$ | **33.0366** | **0.927514** | **0.022295** | 1 | 1 | 313.17 | 967.54 |
| **1e-2** | 24 | $1.0228 \times 10^{-1}$ | **18.8475** | **0.029707** | **0.114113** | 30 | 30 | 310.52 | 967.32 |

### Corrected R_MAX=20 Control (`lambda=0`)
- **PSNR:** **46.4172 dB** | **SSIM:** **0.989254** | **RMSE:** **0.004774** | Primary Rank: 20 | Secondary 99% Rank: 18

---

### Historical / Invalid / Superseded Results

The following historical runs generated prior to the BatchNorm evaluation fix are **INVALID**:
- *Old pre-fix R_MAX=20 lambda=0 diagnostic:* 38.60 dB (Invalidated by eval mode bug).
- *Old pre-fix R_MAX=30 grid runs:* Reported lower PSNRs due to BatchNorm eval mode degradation.

---

## 12. Results Interpretation

1. **Mechanical Sparsification Succeeds:** Increasing $\lambda$ successfully forces component norms to zero, reducing effective rank from 27 ($\lambda=0$) to 3 ($\lambda=10^{-4}$) and 1 ($\lambda=10^{-3}$).
2. **Quality-Sparsity Tradeoff is Unfavorable:** Sparsification causes immediate, severe PSNR loss ($46.86\text{ dB} \to 42.49\text{ dB} \to 33.04\text{ dB}$).
3. **Self-Supervised Selection Picks $\lambda=0$:** Validation loss is lowest at $\lambda=0$ ($0.030037$), meaning self-supervised model selection correctly rejects explicit group-lasso regularization.

---

## 13. Comparison With Baseline / Other Extensions

- **Vs Extension 1 (Adaptive Rank):** Extension 1 achieves Rank 8 with **46.37 dB PSNR**. Extension 2 at $\lambda=10^{-4}$ achieves Rank 3 but drops to **42.49 dB PSNR** (~3.88 dB worse). Discrete bottom-up rank growth provides a vastly superior quality-compression tradeoff.

---

## 14. Limitations

1. **No Quality-Compression Benefit:** Group Lasso fails to find a sweet spot where rank decreases without degrading quality.
2. **Claims We MUST NOT Make:**
   - MUST NOT claim Group Lasso is a nuclear-norm surrogate.
   - MUST NOT claim true rank discovery.
   - MUST NOT claim competitiveness with Extension 1.

---

## 15. Scientific Conclusion

Extension 2 produced a **valid negative/analytical finding**. While Factor Group-Sparsity mechanically prunes factor matrices, the continuous penalty conflicts with DIP optimization dynamics, causing severe reconstruction degradation. Held-out validation correctly selects $\lambda=0$.

---

## 16. Contribution to the Overall Project

Provides a rigorous negative result demonstrating that continuous top-down factor penalization is inferior to discrete bottom-up adaptive rank growth in DeepTensor.

---

## 17. Final Status

**COMPLETE / FROZEN** (Analytical negative result fully documented).

---

## 18. Reproducibility Reference

- **Script:** [`run_figure9_sparsity.py`](file:///e:/Rakhal_Tensor/run_figure9_sparsity.py)
- **Results Directory:** `results_sparsity_corrected/`
- **Master Documentation:** [`PROJECT_CONTEXT.md`](file:///e:/Rakhal_Tensor/PROJECT_CONTEXT.md)
