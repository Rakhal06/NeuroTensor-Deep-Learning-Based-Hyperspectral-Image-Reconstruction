# DeepTensor Research Project: Master Context Document

## 1. Project Identity
- **Project Name:** DeepTensor-Extended (DeepTensor-Low-Rank-Tensor-Decomposition)
- **Faculty:** Dr. Sibasankar Padhy
- **Students:** B. Rakhal Krishna (24BDS009), Dargupally Ashish (24BDS015)
- **Research Topic:** DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors
- **Main Objective:** Extend the baseline DeepTensor framework across multiple dimensions—including Adaptive Rank Selection, Sparsity-Regularized Rank Discovery, Tensor Completion under missing data, Noise-Type Robustness at matched input corruption, and Ensemble-Based Uncertainty Estimation—while keeping the baseline framework reproducible and scientifically audited.

---

## 2. Research Problem & Project Overview
Classical low-rank tensor decomposition and the original DeepTensor approach require a fixed, manually specified rank hyperparameter (e.g., $R=20$). Selecting this rank is often arbitrary or requires computationally expensive trial-and-error grid searches. Beyond rank selection, evaluating DeepTensor's capabilities under incomplete observations (tensor completion), diverse noise statistical structures (Gaussian vs. Poisson vs. Mixed), and empirical reconstruction uncertainty via ensemble disagreement is essential to understanding the scope and limits of deep network priors in hyperspectral tensor recovery.

---

## 3. Completed Extensions & Baseline Verification Summary

| Stage / Extension | Scientific Focus / Name | Status | Primary Outcome / Finding |
|---|---|---|---|
| **Baseline** | Figure 9 Reproduction | COMPLETE / FROZEN | Reproduced Rank-20 baseline on ICVL (`46.45 dB` PSNR, Commit `a146cc7`). |
| **Extension 1** | Self-Supervised Online Model Selection for Effective Rank | COMPLETE / FROZEN | Selected **Rank 8** dynamically (`46.37 dB` PSNR matching Rank-20 control, Commit `85ff484`). |
| **Extension 2** | Factor Group-Sparsity (Group Lasso) Regularization | COMPLETE / FROZEN | **Negative/analytical result**: Top-down factor suppression produced severe quality degradation (`46.86 dB` $\to$ `42.49 dB` $\to$ `33.04 dB`). |
| **Extension 3** | Missing-Data / Tensor Completion Capability | COMPLETE / FROZEN | Mask-aware loss imputed unobserved tensor entries cleanly (Missing PSNR `46.11 dB` at 20% missing, `45.17 dB` at 50%, `42.51 dB` at 80%). |
| **Extension 4** | Noise-Type Robustness at Matched Input Corruption | COMPLETE / FROZEN | Fixed input PSNR `28.43 dB`: Gaussian `46.67 dB` PSNR vs Poisson `45.13 dB` PSNR vs Mixed `46.20 dB` PSNR (Commit `4961e52`). |
| **Extension 5** | Ensemble Disagreement as a Proxy for Reconstruction Uncertainty | COMPLETE / FROZEN PENDING GIT CHECKPOINT | $N=5$ ensemble on fixed observation: Raw correlation ($r=0.342$, $r_s=0.347$) drops to partial correlation ($r=0.223$, $r_s=0.153$) when controlling for GT edge magnitude. |
| **Extension 6** | Computational Efficiency Analysis | NOT STARTED | Future work. |
| **Extension 7** | Factor / Component Interpretability | NOT STARTED | Future work. |

---

## 4. Mandatory Design Review & Research Workflow

Every new extension design in this project strictly adheres to the following 11-step process:

```
1. Gemini Pro High (Antigravity): Independent codebase inspection + research/design review
2. Claude Web 1: Skeptical independent critical review
3. Claude Web 2: Final synthesis / design gate
4. Only after approval: Implementation in Antigravity
5. Smoke / sanity tests (e.g., reproducing baseline condition)
6. Full experiments
7. Scientific audit (empirical verification of logs, metrics, code integrity)
8. Detailed extension documentation (created in documentation/extension_X_.../)
9. Update PROJECT_CONTEXT.md
10. Git checkpoint + push to origin/main
11. Freeze extension
```

---

## 5. Detailed Extension Descriptions

### A. Original Frozen Baseline (Full-Data)
- **Commit:** `a146cc7` ("Baseline: reproduce DeepTensor Figure 9 on RTX 4060") on `main`
- **Script:** `run_figure9.py` (Frozen)
- **Dataset:** ICVL Hyperspectral (`data/icvl.mat`), shape `(174, 164, 260)`, `noise_snr=2`, `tau=100`
- **Method:** Fixed Rank-20, Full-Data 100% loss computation, 5000 epochs, Adam (`lr=1e-3`)
- **Verified Results:**
  - Input Noisy PSNR: 28.43 dB
  - SVD Rank-20 PSNR: ~38.31 dB | SSIM: ~0.9169
  - DeepTensor Rank-20 PSNR: 46.45 dB | SSIM: 0.9872
  - Runtime: ~159.57 sec | Peak GPU VRAM: 640.37 MB

---

### B. Extension 1: Adaptive Rank Selection — COMPLETED AND FROZEN
- **Commit:** `85ff484` ("Fix adaptive rank reporting for incomplete cooldown") on `extension/adaptive-rank` (Merged into `main` at `8c84a0d`)
- **Script:** `run_figure9_adaptive.py` (Frozen)
- **Scientific Name:** **Self-Supervised Online Model Selection for Effective Rank**
- **Framework:**
  - Initial active rank $R_\text{start}=4$, max capacity $R_\text{max}=30$, step $R_\text{step}=2$.
  - Fixed 5% held-out binary mask for self-supervised validation (95% train / 5% val).
  - Directional plateau criterion (`0 <= rel_train_improvement < 1e-3`) triggers rank growth if validation EMA improved over previous plateau.
- **Adaptive Run Results (5000 Epochs):**
  - **Run #1:** Selected **Rank 8** (PSNR **46.37 dB** | SSIM **0.9872**)
  - **Run #2:** Selected **Rank 8** (PSNR **45.95 dB** | SSIM **0.9856**)
- **Matched-Mask Fixed Rank-20 Control:**
  - PSNR **46.14 dB** | SSIM **0.9872**
- **Scientific Conclusion:** Bottom-up self-supervised online model-selection consistently discovers a compact effective rank ($R=8$) that matches fixed Rank-20 performance without ground-truth tuning.

---

### C. Extension 2: Factor Group-Sparsity (Group Lasso) Regularization — COMPLETED AND FROZEN
- **Script:** `run_figure9_sparsity.py` (Frozen)
- **Scientific Name:** **Factor Group-Sparsity (Group Lasso) Regularization**
- **Formulation:** Joint factor norm $g_r = \sqrt{\|U_r\|_2^2 + \|V_r\|_2^2}$ for $R_\text{max}=30$, penalized with $\lambda \, \sum_r g_r$.
- **Key Diagnostic Finding:** Resolved a ~2.87 dB evaluation discrepancy by fixing `im_net.train()` mode during evaluation (BatchNorm running statistics in single-image DIP settings are inaccurate in `eval()` mode).
- **Final Corrected Grid Results (5000 Epochs, Seed 42, ICVL):**
  - $\lambda=0.0$ (Control): PSNR **46.86 dB** | SSIM **0.9896** | $R_\text{eff}=27$
  - $\lambda=1\text{e-}4$: PSNR **42.49 dB** | SSIM **0.9838** | $R_\text{eff}=3$
  - $\lambda=1\text{e-}3$: PSNR **33.04 dB** | SSIM **0.9275** | $R_\text{eff}=1$
  - $\lambda=1\text{e-}2$: PSNR **18.85 dB** | SSIM **0.0297** | $R_\text{eff}=30$
- **Scientific Conclusion (Negative/Analytical Result):** Top-down group-lasso factor suppression mechanically prunes components but incurs a severe reconstruction quality drop. Self-supervised validation correctly selects $\lambda=0$, proving top-down factor suppression is inferior to Extension 1's bottom-up rank growth.

---

### D. Extension 3: Missing-Data / Tensor Completion Capability — COMPLETED AND FROZEN
- **Script:** `run_figure9_completion.py` (Frozen)
- **Documentation:** `documentation/extension_3_tensor_completion/EXTENSION_4_DOCUMENTATION.md`
- **Formulation:** Mask-aware L1 reconstruction loss over observed entries $\Omega_\text{obs}$:
  $$\mathcal{L} = \frac{1}{|\Omega_\text{train}|} \left\| \mathcal{P}_{\Omega_\text{train}} \left( Y - \hat{Y} \right) \right\|_1$$
- **Verified Results (5000 Epochs, Fixed Rank-20, Seed 42):**
  - **0% Missing (Sanity Check):** Missing PSNR N/A | Full PSNR **46.24 dB** | SSIM **0.9892**
  - **20% Noisy Missingness:** Missing PSNR **46.11 dB** | Full PSNR **46.02 dB** | SSIM **0.9877**
  - **50% Noisy Missingness:** Missing PSNR **45.17 dB** | Full PSNR **45.18 dB** | SSIM **0.9834**
  - **80% Noisy Missingness:** Missing PSNR **42.51 dB** | Full PSNR **42.50 dB** | SSIM **0.9709**
  - **50% Clean Control:** Missing PSNR **50.50 dB** | Full PSNR **50.47 dB** | SSIM **0.9975** (+5.33 dB over noisy missingness)
- **Scientific Conclusion:** DeepTensor's DIP generators impute unobserved entries cleanly without architectural changes or explicit regularization.

---

### E. Extension 4: Noise-Type Robustness at Matched Input Corruption — COMPLETED AND FROZEN
- **Script:** `run_figure9_robustness.py` (Frozen)
- **Documentation:** `documentation/extension_4_noise_type_robustness/EXTENSION_4_DOCUMENTATION.md`
- **Git Commit:** `4961e52` (Full SHA: `4961e5287e07caaaef8cdd3073998f8045ee33d3`) on `main`
- **Research Question:** Does DeepTensor's reconstruction quality, at a fixed and matched input-corruption severity, transfer across noise families with different statistical structure (signal-independent Gaussian vs. signal-dependent Poisson) as well as it performs under the mixed noise model the baseline was originally evaluated on?

#### Experimental Results (5000 Epochs)

| Metric / Condition | Mixed Poisson + Gaussian | Pure Gaussian | Pure Poisson |
|---|---|---|---|
| **Noise Configuration** | `noise_snr=2, tau=100` | `noise_snr=0.037875, tau=inf` | `noise_snr=0, tau=72.124592` |
| **Input PSNR (dB)** | 28.4321 | 28.4321 | 28.4294 |
| **Reconstruction PSNR (dB)** | **46.2035** | **46.6665** | **45.1343** |
| **Reconstruction SSIM** | 0.987190 | 0.987765 | 0.986597 |
| **Reconstruction RMSE** | 0.004896 | 0.004642 | 0.005537 |
| **PSNR Gain (dB)** | +17.7714 | **+18.2344** | +16.7049 |
| **Noise-Intensity Corr.** | 0.211393 | 0.000394 | 0.284420 |
| **Runtime (sec)** | 168.57 | 163.59 | 168.57 |
| **Peak VRAM (MB)** | 668.67 | 668.67 | 668.67 |

#### Scientific Interpretation & Key Findings
- **Gaussian Performance:** Gaussian noise produced the highest reconstruction quality (**46.67 dB** PSNR, +18.23 dB gain).
- **Poisson Sensitivity:** Poisson noise produced lower reconstruction quality (**45.13 dB** PSNR, +16.70 dB gain), demonstrating a 1.53 dB drop relative to Gaussian.
- **Statistical Structure Impact:** DeepTensor is sensitive to noise statistical structure. Heteroscedastic Poisson noise is harder for the unweighted L1 loss and CNN prior to denoise than homoscedastic Gaussian noise.

---

### F. Extension 5: Ensemble Disagreement as a Proxy for Reconstruction Uncertainty — COMPLETED AND FROZEN PENDING GIT CHECKPOINT
- **Script:** `run_figure9_ensemble.py`
- **Documentation:** `documentation/extension_5_ensemble_uncertainty/EXTENSION_5_DOCUMENTATION.md`
- **Results Summary:** `results_ensemble/summary.txt`
- **Saved Arrays:** `results_ensemble/ensemble_results.npz` (`reconstructions`, `ensemble_mean`, `uncertainty`, `absolute_error`, `edge_magnitude`)
- **Scientific Name:** **Ensemble Disagreement as a Proxy for Reconstruction Uncertainty**
- **Primary Scientific Terminology:** "Empirical ensemble disagreement" (an empirical optimization/initialization-sensitivity proxy). "Epistemic-style" used only with explicit qualification.
- **Excluded Terminology:** Formal Bayesian uncertainty, calibrated predictive uncertainty, posterior variance, aleatoric uncertainty.

#### Experimental Protocol
- **Fixed Observation:** ICVL benchmark cube, master seed `42`, mixed Poisson + Gaussian noise (`noise_snr=2`, `tau=100`), input PSNR `28.4321 dB`. Frozen across all $N=5$ ensemble members to isolate initialization sensitivity from observation noise draw variance.
- **Ensemble Setup:** $N=5$ members, seeds `43, 44, 45, 46, 47`. Explicit RNG seeding (`random`, `np.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`) before network construction.
- **Architecture & Budget:** Unmodified frozen DIP baseline ($R=20$), Adam $\eta=10^{-3}$, 5000 epochs per member, `train()` evaluation mode, no early stopping, no validation split.
- **Formulation:**
  - Empirical uncertainty tensor $\sigma$: sample standard deviation across 5 reconstructions.
  - Absolute reconstruction error $E = |\bar{Y} - Y_{\text{GT}}|$ (GT used strictly post-hoc).
  - Spatial gradient magnitude $G = \sqrt{(\nabla_x Y_{\text{GT}})^2 + (\nabla_y Y_{\text{GT}})^2}$ computed from clean GT ONLY.

#### Verified Experimental Results ($5 \times 5000$ Epochs)

| Metric / Parameter | Value |
|---|---|
| **Observation Master Seed** | 42 |
| **Member Seeds** | 43, 44, 45, 46, 47 |
| **Observation PSNR** | 28.4321 dB |
| **Individual Member PSNR (Mean ± SD)** | 46.1630 ± 0.2377 dB |
| **Individual Member SSIM (Mean ± SD)** | 0.987312 ± 0.000555 |
| **Individual Member RMSE (Mean ± SD)** | 0.004920 ± 0.000134 |
| **Member Runtimes** | M1: 145.20s \| M2: 144.00s \| M3: 144.06s \| M4: 143.58s \| M5: 144.38s |
| **Ensemble Mean PSNR (Secondary)** | 48.5286 dB |
| **Ensemble Mean SSIM (Secondary)** | 0.992547 |
| **Ensemble Mean RMSE (Secondary)** | 0.003746 |
| **PSNR Improvement vs Mean Individual** | +2.3656 dB (at ~5x compute cost) |
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

#### Scientific Interpretation & Safe Claims
1. **Raw Association:** Independently initialized DeepTensor models exhibit measurable optimization sensitivity under a fixed observation, raw-correlating positively with true error ($r = 0.342246$, $r_s = 0.347170$).
2. **Structural Confounding:** Clean GT edge magnitude strongly correlates with uncertainty ($r = 0.513929$, $r_s = 0.656932$) and error ($r = 0.311521$, $r_s = 0.365347$). Controlling for edge magnitude substantially reduces the correlation (Pearson drops to `0.223452`, Spearman drops to `0.152689`).
3. **Residual Utility:** Image structure explains a substantial portion of the raw relationship, but a modest positive residual association remains ($r_{\sigma E \cdot G} = 0.223452$), indicating disagreement is not entirely redundant with spatial gradient magnitude.
4. **Quantile Separation:** Top 10% most uncertain voxels have a $3.0445\times$ higher MAE than bottom 10% most certain voxels.
5. **Ensemble Reconstruction Gain:** Averaging $N=5$ member reconstructions improves PSNR by $+2.3656\text{ dB}$ over mean individual members, as a secondary result at $\approx 5\times$ compute cost.
6. **No p-values reported:** Parametric i.i.d. hypothesis testing is omitted due to spatial and spectral autocorrelation across tensor entries.
7. **Scientific Limitations:** $N=5$ produces a noisy per-voxel estimate. Evaluated on one dataset, one fixed observation, and one noise condition; results do not establish universal generalization, calibrated uncertainty, or aleatoric noise estimation.

---

## 6. Current Status & Next Immediate Action

| Extension / Task | Status | Output / Location |
|---|---|---|
| Baseline Reproduction | COMPLETE / FROZEN | `run_figure9.py` (Commit `a146cc7`) |
| Extension 1: Adaptive Rank Selection | COMPLETE / FROZEN | `run_figure9_adaptive.py` (Commit `85ff484`) |
| Extension 2: Factor Group-Sparsity | COMPLETE / FROZEN | `run_figure9_sparsity.py` |
| Extension 3: Missing-Data Tensor Completion | COMPLETE / FROZEN | `run_figure9_completion.py` |
| Extension 4: Noise-Type Robustness | COMPLETE / FROZEN | `run_figure9_robustness.py` (Commit `4961e52`) |
| Extension 5: Ensemble Uncertainty Estimation | **COMPLETE / FROZEN PENDING GIT CHECKPOINT** | `run_figure9_ensemble.py`, `documentation/extension_5_ensemble_uncertainty/` |
| Extension 6: Computational Efficiency | NOT STARTED | Future work |
| Extension 7: Factor Interpretability | NOT STARTED | Future work |

**Immediate Next Step:**
Perform Git Checkpoint commit and push for Extension 5 following project protocols.

---

## 7. Important Files Registry

| File / Directory | Description / Role | Status |
|---|---|---|
| `run_figure9.py` | Baseline DeepTensor script (Fixed Rank-20). | **FROZEN** (Commit `a146cc7`) |
| `run_figure9_adaptive.py` | Extension 1: Adaptive Rank Selection. | **FROZEN** (Commit `85ff484`) |
| `run_figure9_sparsity.py` | Extension 2: Factor Group-Sparsity (Group Lasso). | **FROZEN** |
| `run_figure9_completion.py` | Extension 3: Tensor Completion under missing data. | **FROZEN** |
| `run_figure9_robustness.py` | Extension 4: Noise-Type Robustness at matched input PSNR. | **FROZEN** (Commit `4961e52`) |
| `run_figure9_ensemble.py` | Extension 5: Ensemble Disagreement Uncertainty Estimation. | **COMPLETE / PENDING CHECKPOINT** |
| `documentation/extension_3_tensor_completion/` | Detailed documentation for Extension 3. | **FROZEN** |
| `documentation/extension_4_noise_type_robustness/` | Detailed documentation for Extension 4. | **FROZEN** |
| `documentation/extension_5_ensemble_uncertainty/` | Detailed documentation for Extension 5. | **COMPLETE / PENDING CHECKPOINT** |
| `results_ensemble/` | Summary text and compressed `.npz` archive for Extension 5. | **COMPLETE / PENDING CHECKPOINT** |
| `modules/` | Core network definitions, DIP generators, & `utils.py`. | **FROZEN** |
| `data/icvl.mat` | ICVL Hyperspectral Dataset. | **FROZEN** |
| `PROJECT_CONTEXT.md` | Master handoff and research context document. | **UPDATED** |

---

## 8. Research Integrity & Hardware Environment Rules

1. **Hardware / Environment:** RTX 4060 Laptop GPU (8GB VRAM), Windows 11, PyTorch 2.5.1+cu121, CUDA 12.1.
2. **Ground-Truth Data Usage:** Clean ground-truth target matrices must **NEVER** be used for gradient computation, model selection, early stopping, member weighting, or hyperparameter tuning. Ground truth is reserved strictly for post-hoc metric computation (PSNR / SSIM / RMSE / Absolute Error / Edge Magnitude control).
3. **Evaluation Mode for DIP Networks:** DIP generator networks (`im_net`, `spec_net`) contain ~60 BatchNorm layers. Always keep networks in `.train()` mode during final inference/evaluation to avoid running-statistic degradation.
4. **Matched-Mask Controls:** Always compare adaptive, missing-data, or ensemble mechanisms against baseline matched controls to isolate structural effects.
5. **No Claims Beyond Empirical Evidence:** Clearly distinguish mathematical rank from effective rank, avoid framing bounded noise tests as universal robustness guarantees, and treat ensemble disagreement strictly as an empirical optimization-sensitivity proxy rather than formal Bayesian uncertainty.
