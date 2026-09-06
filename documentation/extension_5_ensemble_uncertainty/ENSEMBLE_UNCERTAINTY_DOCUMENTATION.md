# Extension 5: Ensemble Disagreement as a Proxy for Reconstruction Uncertainty

**Status:** COMPLETE / FROZEN PENDING GIT CHECKPOINT

---

## 1. Title & Executive Summary

- **Extension Name:** Ensemble Disagreement as a Proxy for Reconstruction Uncertainty
- **Status:** Complete / Frozen
- **Primary Scientific Terminology:** "Empirical ensemble disagreement" (or "empirical ensemble disagreement as a proxy for reconstruction uncertainty").
- **Primary Finding:** Per-voxel disagreement across $N=5$ independently initialized DeepTensor reconstructions trained on a single fixed noisy observation exhibits a positive correlation with true post-hoc reconstruction error (Raw Pearson $r = 0.342246$, Raw Spearman $r_s = 0.347170$). Controlling for spatial image structure (clean ground-truth edge magnitude) substantially reduces this relationship (Partial Pearson $r_{ue \cdot g} = 0.223452$, Partial Spearman $r_{s, ue \cdot g} = 0.152689$). Thus, image structure explains a substantial portion of disagreement-error correlation, but a modest positive residual association remains.
- **Secondary Finding:** The ensemble mean reconstruction achieves $48.5286\text{ dB}$ PSNR, representing a $+2.3656\text{ dB}$ improvement over individual members ($46.1630 \pm 0.2377\text{ dB}$), at approximately $5\times$ training computation cost.

---

## 2. Research Question

Does per-voxel disagreement among $N=5$ independently initialized DeepTensor reconstructions of one fixed noisy observation correlate with true reconstruction error, beyond what is already explained by image structure (edge/high-frequency content) alone?

---

## 3. Scientific Definition & Qualified Terminology

- **Empirical Optimization-Sensitivity Proxy:** Disagreement across ensemble members reflects sensitivity to random parameter initialization and optimization trajectory within DeepTensor's non-convex loss landscape.
- **Qualified Terminology:** Disagreement may be qualified as an "epistemic-style optimization-sensitivity proxy" only with explicit qualification.
- **Excluded Claims & Prohibited Terms:**
  - Do **NOT** describe as formal Bayesian uncertainty or posterior variance.
  - Do **NOT** claim calibrated predictive uncertainty.
  - Do **NOT** claim aleatoric uncertainty (input noise draw variance).

---

## 4. Methodological Rationale: Fixed Observation Protocol

All $N=5$ ensemble members receive the **EXACT SAME noisy observation tensor**, generated deterministically once using master seed `42` (`noise_snr=2`, `tau=100`, mixed Poisson + Gaussian noise).

- **Why a Fixed Observation is Used:** Generating a single observation isolates *initialization sensitivity* (the non-convex optimization trajectory variance under a fixed observation). Generating separate noise realizations for each member would conflate observation noise variance (aleatoric corruption) with model initialization sensitivity.

---

## 5. Experimental Protocol & Hyperparameters

- **Dataset:** ICVL hyperspectral benchmark cube (`icvl.mat`).
- **Observation Generation:** Master seed `42`, mixed Poisson + Gaussian noise, `noise_snr=2`, `tau=100`, resize scaling `1.0`. Fixed observation PSNR = `28.4321 dB`.
- **Ensemble Members ($N=5$):**
  - Member 1 Seed: `43`
  - Member 2 Seed: `44`
  - Member 3 Seed: `45`
  - Member 4 Seed: `46`
  - Member 5 Seed: `47`
- **RNG Seeding Policy:** Immediately prior to building each member's networks (`im_net`, `spec_net`) and latent inputs (`im_inp`, `spec_inp`), all random number generators (`random`, `np.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`) are explicitly seeded.
- **Architecture & Optimization:**
  - Original frozen baseline unnormalized DeepTensor DIP architecture (Rank $R=20$).
  - Optimizer: Adam with learning rate $\eta = 10^{-3}$.
  - Budget: Exactly $5000$ epochs per member.
  - Controls: No early stopping, no validation split, no adaptive rank, no sparsity regularization.
- **Evaluation Mode:** Networks remain in `train()` mode for final inference, adhering to the DIP BatchNorm dynamics established in the project baseline.

---

## 6. Uncertainty, Error & Edge Formulation

Let $Y_1, Y_2, Y_3, Y_4, Y_5 \in \mathbb{R}^{H \times W \times B}$ be the 5 member reconstructions ($H=174, W=164, B=260$).

1. **Ensemble Mean Reconstruction ($\bar{Y}$):**
   $$\bar{Y}_{h,w,b} = \frac{1}{5} \sum_{m=1}^{5} Y_{m, h,w,b}$$

2. **Empirical Ensemble Disagreement (Uncertainty Tensor $\sigma$):**
   $$\sigma_{h,w,b} = \sqrt{ \frac{1}{5-1} \sum_{m=1}^{5} \left(Y_{m, h,w,b} - \bar{Y}_{h,w,b}\right)^2 }$$

3. **True Absolute Reconstruction Error ($E$):**
   $$E_{h,w,b} = \left| \bar{Y}_{h,w,b} - Y_{\text{GT}, h,w,b} \right|$$

4. **Clean Ground-Truth Edge Magnitude Control ($G$):**
   $$G_{h,w,b} = \sqrt{ (\nabla_x Y_{\text{GT}, h,w,b})^2 + (\nabla_y Y_{\text{GT}, h,w,b})^2 }$$
   Computed via spatial finite differences (`np.gradient(cube, axis=(0, 1))`) on clean ground truth $Y_{\text{GT}}$ ONLY.

---

## 7. Data Leakage & Integrity Controls

- Clean ground truth ($Y_{\text{GT}}$) was used **STRICTLY post-hoc** for evaluating absolute error, GT edge-magnitude, and final PSNR/SSIM/RMSE metrics.
- Ground truth was **NEVER** used during training, member checkpoint selection, member weighting, stopping decisions, or hyperparameter tuning.

---

## 8. Statistical Caveats & Methodology

- **No p-values:** Tensor entries across spatial $(H, W)$ and spectral $(B)$ dimensions exhibit heavy autocorrelation. Standard parametric i.i.d. hypothesis testing (p-values) is methodologically invalid for spatial-spectral tensor fields and is strictly omitted.
- **Sample Size:** $N=5$ produces an empirical sample disagreement estimate.

---

## 9. Numerical Results

### Hardware & Runtime
- **Total Runtime:** `722.78 seconds` (~12.0 minutes)
- **Peak GPU VRAM:** `640.64 MB`
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU

### Individual Member Metrics
- **Member 1 (Seed 43):** PSNR = `46.0345 dB`, SSIM = `0.987208`, RMSE = `0.004992`, Runtime = `145.20 s`
- **Member 2 (Seed 44):** PSNR = `46.4523 dB`, SSIM = `0.987806`, RMSE = `0.004758`, Runtime = `144.00 s`
- **Member 3 (Seed 45):** PSNR = `45.9779 dB`, SSIM = `0.986405`, RMSE = `0.005025`, Runtime = `144.06 s`
- **Member 4 (Seed 46):** PSNR = `46.3886 dB`, SSIM = `0.987676`, RMSE = `0.004793`, Runtime = `143.58 s`
- **Member 5 (Seed 47):** PSNR = `45.9616 dB`, SSIM = `0.987466`, RMSE = `0.005034`, Runtime = `144.38 s`
- **Individual Mean ± SD:**
  - **PSNR:** $46.1630 \pm 0.2377\text{ dB}$
  - **SSIM:** $0.987312 \pm 0.000555$
  - **RMSE:** $0.004920 \pm 0.000134$

### Secondary Analysis: Ensemble Mean Reconstruction
- **Ensemble Mean PSNR:** `48.5286 dB`
- **Ensemble Mean SSIM:** `0.992547`
- **Ensemble Mean RMSE:** `0.003746`
- **PSNR Gain vs Mean Individual:** `+2.3656 dB` (requires $\approx 5\times$ training computation)

### Primary Uncertainty & Edge Control Analysis (Full $H \times W \times B$ Tensor)
- **Global Mean Uncertainty ($\bar{\sigma}$):** `0.002989174` (Std Dev)
- **Raw Correlation (Uncertainty $\sigma$ vs Absolute Error $E$):**
  - Pearson $r$: `0.342246`
  - Spearman $r_s$: `0.347170`
- **Edge Confound Correlations:**
  - GT Edge Mag vs Absolute Error ($G$ vs $E$): Pearson $r = 0.311521$, Spearman $r_s = 0.365347$
  - GT Edge Mag vs Uncertainty ($G$ vs $\sigma$): Pearson $r = 0.513929$, Spearman $r_s = 0.656932$
- **Post-Hoc Partial Correlation Controlling for Clean GT Edge Magnitude:**
  - **Partial Pearson ($r_{\sigma E \cdot G}$):** `0.223452` (reduced from `0.342246`, $-34.7\%$)
  - **Partial Spearman ($r_{s, \sigma E \cdot G}$):** `0.152689` (reduced from `0.347170`, $-56.0\%$)

### Quantile Analysis
- **Top 10% Most Uncertain Entries MAE:** `0.004362`
- **Bottom 10% Most Certain Entries MAE:** `0.001433`
- **Quantile Ratio (Top 10% / Bottom 10% MAE):** `3.0445×`

---

## 10. Summary Table

| Metric / Parameter | Value |
|---|---|
| **Observation Master Seed** | 42 |
| **Member Seeds** | 43, 44, 45, 46, 47 |
| **Observation PSNR** | 28.4321 dB |
| **Individual Member PSNR (Mean ± SD)** | 46.1630 ± 0.2377 dB |
| **Individual Member SSIM (Mean ± SD)** | 0.987312 ± 0.000555 |
| **Individual Member RMSE (Mean ± SD)** | 0.004920 ± 0.000134 |
| **Ensemble Mean PSNR** | 48.5286 dB |
| **Ensemble Mean SSIM** | 0.992547 |
| **Ensemble Mean RMSE** | 0.003746 |
| **PSNR Gain vs Mean Individual** | +2.3656 dB (cost ~5x compute) |
| **Global Mean Uncertainty** | 0.002989174 |
| **Raw Pearson (Uncertainty vs Abs Error)** | 0.342246 |
| **Raw Spearman (Uncertainty vs Abs Error)** | 0.347170 |
| **Pearson (GT Edge Mag vs Abs Error)** | 0.311521 |
| **Spearman (GT Edge Mag vs Abs Error)** | 0.365347 |
| **Pearson (GT Edge Mag vs Uncertainty)** | 0.513929 |
| **Spearman (GT Edge Mag vs Uncertainty)** | 0.656932 |
| **Partial Pearson (Uncertainty vs Abs Error \| GT Edge)** | 0.223452 |
| **Partial Spearman (Uncertainty vs Abs Error \| GT Edge)** | 0.152689 |
| **Top 10% Most Uncertain MAE** | 0.004362 |
| **Bottom 10% Most Certain MAE** | 0.001433 |
| **Quantile Ratio (Top 10% / Bottom 10% MAE)** | 3.0445× |
| **Total Runtime** | 722.78 s (~12.0 min) |
| **Peak GPU VRAM** | 640.64 MB |

---

## 11. Scientific Interpretation & Safe Claims

### Safe Claims
1. **Raw Association:** Independently initialized DeepTensor models exhibit measurable variability (disagreement) under a fixed observation, and this disagreement raw-correlates positively with true reconstruction error ($r = 0.342246$, $r_s = 0.347170$).
2. **Structural Confound:** Image spatial structure (edge magnitude) is strongly correlated with disagreement ($r = 0.513929$, $r_s = 0.656932$) and reconstruction error ($r = 0.311521$, $r_s = 0.365347$).
3. **Partial Correlation Effect:** Controlling for GT edge magnitude substantially reduces the correlation between disagreement and error (Pearson drops by 34.7% to `0.223452`, Spearman drops by 56.0% to `0.152689`).
4. **Residual Utility:** A modest positive residual association remains ($r_{\sigma E \cdot G} = 0.223452$), indicating that ensemble disagreement is not entirely redundant with simple spatial gradient magnitude.
5. **Descriptive Separation:** Top 10% most uncertain voxels have a $3.0445\times$ higher MAE than bottom 10% most certain voxels.
6. **Ensemble Reconstruction Gain:** Averaging $N=5$ reconstructions improves PSNR by $+2.3656\text{ dB}$ over mean individual members, at $5\times$ compute cost.

### Explicitly Avoided Claims
1. **Do NOT claim general error prediction:** Disagreement does not independently predict error in a fully general, structure-agnostic sense.
2. **Do NOT claim Bayesian posterior variance:** Disagreement reflects non-convex optimization sensitivity, not calibrated Bayesian uncertainty.
3. **Do NOT claim aleatoric uncertainty:** The observation was fixed; aleatoric noise draw variance was not measured.
4. **Do NOT generalize across datasets/noises:** Results apply strictly to the ICVL benchmark under the tested mixed Poisson + Gaussian noise condition.

---

## 12. Artifacts & Output Files

- **Script:** [`run_figure9_ensemble.py`](file:///e:/Rakhal_Tensor/run_figure9_ensemble.py)
- **Summary Text:** [`results_ensemble/summary.txt`](file:///e:/Rakhal_Tensor/results_ensemble/summary.txt)
- **Compressed Arrays:** `results_ensemble/ensemble_results.npz` (contains `reconstructions`, `ensemble_mean`, `uncertainty`, `absolute_error`, `edge_magnitude`)
- **Documentation:** [`documentation/extension_5_ensemble_uncertainty/ENSEMBLE_UNCERTAINTY_DOCUMENTATION.md`](file:///e:/Rakhal_Tensor/documentation/extension_5_ensemble_uncertainty/ENSEMBLE_UNCERTAINTY_DOCUMENTATION.md)

---

## 13. Final Conclusion

The experiment demonstrates measurable reconstruction disagreement among $N=5$ independently initialized DeepTensor models on a fixed noisy observation. The raw disagreement-error relationship is positive (Pearson `0.342246`, Spearman `0.347170`). However, image edge structure is strongly associated with disagreement and moderately associated with error. After controlling for GT edge magnitude, the relationship is substantially reduced to a partial Pearson of `0.223452` and partial Spearman of `0.152689`. This indicates that image structure explains a substantial portion of the raw relationship, while a modest positive residual association remains. This supports treating empirical ensemble disagreement as an optimization-sensitivity proxy with a modest relationship to reconstruction error in this tested setting, without claiming calibrated or Bayesian uncertainty.
