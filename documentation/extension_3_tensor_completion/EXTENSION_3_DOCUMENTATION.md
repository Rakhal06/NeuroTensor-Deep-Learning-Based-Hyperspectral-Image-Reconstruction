# Extension 3 — Missing-Data / Tensor Completion Capability

## 1. Executive Summary

- **Objective:** Extend DeepTensor to evaluate on missing-data tensor completion (unobserved entries) in addition to image denoising, using a mask-aware reconstruction loss.
- **Motivation:** Demonstrate that DeepTensor's low-rank matrix factorization architecture ($\hat{Y} = U V^T$) driven by deep image priors can impute unobserved tensor entries without introducing any new network components or regularization terms.
- **Final Implementation:** Implemented in `run_figure9_completion.py`. Preserves the fixed-rank ($R=20$) baseline architecture, DIP generators, and latent-only inputs (`im_inp`, `spec_inp`). Uses a scoped NumPy RNG (`np.random.default_rng(seed)`) to generate nested missing-data masks ($p_{\text{missing}} \in \{0.20, 0.50, 0.80\}$) and splits remaining observed entries into 95% training and 5% self-supervised validation. Missing entries contribute ZERO gradient to training or validation loss.
- **Key Result:**
  - **0% Sanity Check:** Reproduces the baseline (Full PSNR 46.24 dB | SSIM 0.9892).
  - **20% Noisy Missingness:** Missing Entries PSNR **46.11 dB** | Full PSNR **46.02 dB** (SSIM 0.9877).
  - **50% Noisy Missingness:** Missing Entries PSNR **45.17 dB** | Full PSNR **45.18 dB** (SSIM 0.9834).
  - **80% Noisy Missingness:** Missing Entries PSNR **42.51 dB** | Full PSNR **42.50 dB** (SSIM 0.9709).
  - **50% Clean Control:** Missing Entries PSNR **50.50 dB** | Full PSNR **50.47 dB** (SSIM 0.9975), demonstrating a **+5.33 dB** improvement over noisy completion on identical masks.
- **Final Status:** **COMPLETE / FROZEN**.

---

## 2. DeepTensor Limitation Being Addressed

Original DeepTensor assumes 100% of tensor entries are physically observed. Loss computation evaluates across the entire $H \times W \times B$ voxel grid:
$$\mathcal{L} = \frac{1}{M \cdot B} \|\hat{Y} - Y^\text{noisy}\|_1$$

When entries are missing/unobserved (a common challenge in hyperspectral remote sensing, medical imaging, and corrupted sensor arrays), applying standard full-tensor loss either forces arbitrary zero-infilling artifacts into the loss or fails to compute gradients. DeepTensor required a formal mask-aware reconstruction objective to enable genuine matrix/tensor completion.

---

## 3. Proposed Extension in the Project Proposal

**Project Proposal Specification:**
> "Adds a mask-aware term to the existing reconstruction loss so DeepTensor can be evaluated on tensor completion (missing entries), not only denoising — directly addresses tensor completion and requires only a loss-function change plus a masking utility, no new network architecture."

- **Intended Objective:** Direct tensor completion capability via loss function modification.
- **Expected Role:** Core required Extension 3 of the DeepTensor research proposal.

---

## 4. Our Initial Research Idea

- **Initial Concept:** Distinguish between a *synthetic blind-spot validation mask* (used for early stopping) and an *actual missing-data mask* (unobserved entries).
- **Network Input Safety:** DeepTensor networks receive latent noise grids (`im_inp`, `spec_inp`) as inputs, NOT the target tensor $Y$. Therefore, missing entries in the target matrix do not contaminate network convolutional feature maps. Setting missing entries in the target array to any placeholder (e.g. `0.0`) is harmless as long as the loss function strictly indexes observed elements.

---

## 5. Preprocessing & Data Leakage Audit

Prior to implementation, a thorough audit of the codebase was conducted to ensure no data leakage occurs during preprocessing:

1. **Global Normalization (`cube / cube.max()`):**
   - Executed on raw MAT data upon loading to define the canonical physical signal scale $Y_\text{GT} \in [0, 1]^{H \times W \times B}$. Occurs before missingness mask simulation. Valid dataset scaling; not data leakage.
2. **Noise Generation (`utils.measure`):**
   - Independent elementwise Poisson and Gaussian sampling. No cross-entry aggregation.
3. **Latent Inputs (`im_inp`, `spec_inp`):**
   - Independent uniform random noise in $[0, 0.1]$. Completely decoupled from image data and missingness masks.
4. **Audit Conclusion:** **Zero Data Leakage.**

---

## 6. Final Research Question / Hypothesis

**Research Question:** Can DeepTensor's fixed-rank ($R=20$) factorized network architecture, trained exclusively on a subset of observed entries (76% train / 4% val) using a mask-aware L1 loss, accurately complete up to 80% unobserved missing entries while preserving high spatial-spectral reconstruction fidelity?

---

## 7. Final Methodology

- **Dataset:** ICVL Hyperspectral (`data/icvl.mat`), shape $(174, 164, 260)$.
- **Architecture:** Fixed-rank DeepTensor ($R=20$), $n_\text{inputs}=20$, `nettype='dip'`.
- **Mask Hierarchy Design (Scoped RNG):**
  - Generator: `rng = np.random.default_rng(seed)`.
  - Draw 1: `u = rng.random(cube_shape)`. Missing entries: `mask_missing = u < missing_ratio`.
    *Nesting Property:* The 80% missing mask is a strict superset of the 50% mask, which is a strict superset of the 20% mask.
  - Draw 2: `val_draw = rng.random(cube_shape)` from the SAME `rng`.
    `mask_val = mask_obs & (val_draw < 0.05)`
    `mask_train = mask_obs & ~(val_draw < 0.05)`
- **Runtime Disjointness Assertions:**
  ```python
  assert (mask_train_np & mask_missing_np).sum() == 0
  assert (mask_val_np & mask_missing_np).sum() == 0
  assert (mask_train_np & mask_val_np).sum() == 0
  assert (mask_train_np | mask_val_np | mask_missing_np).sum() == mask_train_np.size
  ```
- **Mask-Aware Objective:**
  $$\mathcal{L}_\text{train} = \frac{1}{|\Omega_\text{train}|} \sum_{(i,j) \in \Omega_\text{train}} |\hat{Y}_{ij} - Y_{ij}^\text{target}|$$
  $$\mathcal{L}_\text{val} = \frac{1}{|\Omega_\text{val}|} \sum_{(i,j) \in \Omega_\text{val}} |\hat{Y}_{ij} - Y_{ij}^\text{target}|$$
  Missing entries $\Omega_\text{missing}$ receive **EXACTLY ZERO** gradient.
- **Evaluation:** Best checkpoint selected at minimum $\mathcal{L}_\text{val}$. Evaluated in `train()` mode for BatchNorm stability. Metrics evaluated against clean ground truth $Y_\text{GT}$ post-hoc.

---

## 8. Implementation Details

- **Script:** [`run_figure9_completion.py`](file:///e:/Rakhal_Tensor/run_figure9_completion.py)
- **Key Arguments:** `--missing_ratio`, `--seed`, `--clean`, `--epochs`, `--output_dir`
- **Output Files:** `results_completion/summary_p_{ratio}_cond_{noisy|clean}_seed_42.txt`

---

## 9. Debugging and Methodological Issues

1. **Spatial Contiguity & SSIM Metrics:**
   - *Issue:* Calculating SSIM on scattered boolean subsets ($\Omega_\text{missing}$) is mathematically invalid because SSIM requires 2D spatial sliding windows.
   - *Fix:* Full-tensor SSIM (`SSIM_full`) is reported for the 3D cube. Missing-entry performance is evaluated strictly using `PSNR_missing` and `RMSE_missing`.
2. **Clean vs. Noisy Mask Determinism:**
   - *Issue:* Ensuring the 50% clean control uses the exact same mask as the 50% noisy experiment.
   - *Fix:* Mask generation depends solely on `cube_shape`, `missing_ratio`, and `seed=42`, guaranteeing deterministic mask reproduction regardless of `--clean` flag.

---

## 10. Experimental Protocol

1. **0% Missing Sanity Check:** Verify reproduction of baseline pipeline.
2. **Noisy Completion Grid:** $p_\text{missing} \in \{0.20, 0.50, 0.80\}$, $\text{SNR}=2, \tau=100$, 5000 epochs, `seed=42`.
3. **Clean 50% Control:** $p_\text{missing} = 0.50$, `--clean`, 5000 epochs, `seed=42`.

---

## 11. Experimental Results

### Verified Definitive Results (5000 Epochs, $R=20$, `seed=42`)

| Experiment | Condition | Missing Ratio ($p_\text{missing}$) | Best Val Epoch | Best Val L1 | Missing PSNR (dB) | Missing RMSE | Observed PSNR (dB) | Observed RMSE | Full PSNR (dB) | Full SSIM | Full RMSE | Runtime (s) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sanity Check** | Noisy ($\text{SNR}=2$) | 0.00 | 4831 | $0.030015$ | N/A | N/A | 46.24 | 0.004878 | **46.24** | **0.989216** | **0.004878** | 299.31 | 964.62 |
| **Noisy 20%** | Noisy ($\text{SNR}=2$) | 0.20 | 4770 | $0.030037$ | **46.11** | **0.004952** | 45.99 | 0.005015 | **46.02** | **0.987744** | **0.005002** | 268.19 | 878.10 |
| **Noisy 50%** | Noisy ($\text{SNR}=2$) | 0.50 | 3996 | $0.030095$ | **45.17** | **0.005512** | 45.19 | 0.005501 | **45.18** | **0.983365** | **0.005506** | 223.16 | 746.42 |
| **Noisy 80%** | Noisy ($\text{SNR}=2$) | 0.80 | 2786 | $0.030361$ | **42.51** | **0.007492** | 42.49 | 0.007511 | **42.50** | **0.970913** | **0.007495** | 168.81 | 624.35 |
| **Clean 50%** | Clean (No noise) | 0.50 | 4962 | $0.001631$ | **50.50** | **0.002984** | 50.44 | 0.003005 | **50.47** | **0.997485** | **0.002995** | 198.65 | 633.46 |

### Element Counts by Missing Ratio

| Missing Ratio | Total Voxels | Observed Voxels ($\Omega_\text{obs}$) | Training Voxels ($\Omega_\text{train}$) | Validation Voxels ($\Omega_\text{val}$) | Missing Voxels ($\Omega_\text{missing}$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | 7,419,360 | 7,419,360 (100.00%) | 7,048,650 (95.00%) | 370,710 (5.00%) | 0 (0.00%) |
| **0.20** | 7,419,360 | 5,935,793 (80.00%) | 5,639,450 (76.01%) | 296,343 (3.99%) | 1,483,567 (20.00%) |
| **0.50** | 7,419,360 | 3,707,981 (49.98%) | 3,522,973 (47.48%) | 185,008 (2.49%) | 3,711,379 (50.02%) |
| **0.80** | 7,419,360 | 1,483,736 (20.00%) | 1,409,622 (19.00%) | 74,114 (1.00%) | 5,935,624 (80.00%) |

---

## 12. Results Interpretation

- **Exceptional Resilience to Missing Data:** DeepTensor retains high completion quality even under extreme missingness:
  - 20% Missing: Missing PSNR **46.11 dB** (Only 0.13 dB drop from 0% baseline).
  - 50% Missing: Missing PSNR **45.17 dB** (Only 1.07 dB drop from 0% baseline).
  - 80% Missing: Missing PSNR **42.51 dB** (Remains >42 dB despite losing 80% of entries).
- **Clean vs. Noisy Impact:** Removing measurement noise increases 50% completion performance from 45.17 dB to **50.50 dB PSNR** (+5.33 dB improvement), demonstrating that observation noise, rather than missing entries, is the primary performance limiter.

---

## 13. Comparison With Baseline / Other Extensions

- **Vs Baseline (0% Missing):** The mask-aware loss seamlessly handles missing data while matching baseline performance when $p_\text{missing}=0$.
- **Independence:** Evaluated independently of Extensions 1 & 2 using baseline fixed $R=20$.

---

## 14. Limitations

1. **Uniform Random Entrywise Missingness Only:** Does not evaluate structured missingness (e.g. dead spatial columns or dropped spectral bands).
2. **Single Seed & Dataset:** Evaluated on ICVL under `seed=42`.
3. **Claims We MUST NOT Make:**
   - MUST NOT claim state-of-the-art tensor completion against classical methods (e.g. HaLRTC).
   - MUST NOT claim generalization to structured fiber/slice missingness.

---

## 15. Scientific Conclusion

Extension 3 successfully endows DeepTensor with robust tensor-completion capabilities through a mask-aware reconstruction loss. The deep image prior architecture effectively interpolates up to 80% missing data, achieving >42 dB PSNR under noisy conditions and >50 dB under clean conditions.

---

## 16. Contribution to the Overall Project

Expands DeepTensor's applicability from pure image denoising to missing-data tensor completion, establishing a mask-aware foundation for future extensions.

---

## 17. Final Status

**COMPLETE / FROZEN** — Implemented, verified against baseline, and fully evaluated across noisy and clean completion regimes.

---

## 18. Reproducibility Reference

- **Script:** [`run_figure9_completion.py`](file:///e:/Rakhal_Tensor/run_figure9_completion.py)
- **Results Directory:** `results_completion/`
- **Master Documentation:** [`PROJECT_CONTEXT.md`](file:///e:/Rakhal_Tensor/PROJECT_CONTEXT.md)
