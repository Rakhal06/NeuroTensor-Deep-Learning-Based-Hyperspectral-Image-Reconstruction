# DeepTensor++ Figure 10 (Yale Face B) Hybrid Readiness Report

**Date:** 2026-11-08  
**Experiment:** Figure 10 — 3-Way PARAFAC Tensor Denoising (Extended Yale Face Database B)  
**Task Status:** Readiness Assessment & Protocol Verification (Phase 4 & 5 Complete — Awaiting User Approval to Launch)  
**Hardware Target:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MB VRAM), CUDA 12.1  
**Software Environment:** Python 3.11.9 (`.venv`), PyTorch 2.5.1+cu121  

---

## 1. Verified Dataset & Preprocessing Protocol

### Dataset Specification
- **Source File:** `data/yale_b.mat`
- **Stored Arrays:**
  - `'faces'`: Shape **$192 \times 168 \times 160$** (`uint8`, range $[0, 255]$).
  - `'filenames'`: Shape **$(160,)$** (`<U24`, string labels detailing subject, pose, and lighting angles).
- **Physical Tensor Dimensions:**
  - Mode 1 ($H = 192$): Vertical facial pixel grid.
  - Mode 2 ($W = 168$): Horizontal facial pixel grid.
  - Mode 3 ($T = 160$): Illumination conditions across discrete spherical coordinates (Azimuth $A \in [-130^\circ, +130^\circ]$, Elevation $E \in [-40^\circ, +90^\circ]$) spanning 3 subjects (`yaleB01`, `yaleB02`, `yaleB03`).
- **Data Normalization:** Scaled to $[0.0, 1.0]$ via `tensor = faces.astype(np.float32) / 255.0`.
- **Degradation Model:** Salt-and-Pepper impulse noise applied to $30\%$ of voxels (and optionally $60\%$):
  - Random coordinates corrupted to either $0.0$ (pepper) or $1.0$ (salt) with equal probability.
  - Fixed master seed (`set_seed(42)`) to ensure identical corruption across all 5 variants.

---

## 2. Architecture Compatibility: Continuous Spectra vs. Discrete Illumination

### Why Self-Attention is Conceptually Justified on Figure 10
In Figure 09 (ICVL), local 1D convolutions outperformed self-attention because adjacent wavelength channels represent a continuous, smooth 1D physical spectrum where local smoothness is an optimal low-pass prior.

In Figure 10 (Yale Face B), the physical structure of Mode 3 is fundamentally different:
1. **Discrete Non-Local Lighting Coordinates:**
   - The 160 images along Mode 3 are ordered by filename (e.g., `yaleB01_P00A+000E+90.pgm` is followed by `yaleB01_P00A+000E-20.pgm`).
   - Adjacent indices in the tensor represent lighting angles that differ by $110^\circ$ in elevation, while images that are physically close on the illumination sphere are separated by dozens of indices in the array.
2. **Failure of Convolutional Inductive Bias:**
   - 1D convolutions apply a local sliding kernel (e.g., width 3 or 5) along Mode 3, assuming that index $i$ is locally correlated with indices $i-1$ and $i+1$. For discrete illumination sequences, this enforces an arbitrary and physically incorrect locality bias.
3. **Natural Advantage of Multi-Head Self-Attention:**
   - Pre-LN Multi-Head Self-Attention calculates pairwise dot-product affinity matrices:
     $$A_{ij} = \text{Softmax}\left(\frac{\mathbf{q}_i \mathbf{k}_j^T}{\sqrt{d}}\right)$$
   - Self-attention is fundamentally **permutation-equivariant** (modulated by positional encodings). It enables the network to discover that image $i$ ($A+000, E+00$) correlates with image $j$ ($A+005, E+00$) across arbitrary sequence positions.
   - Consequently, **Mode 3 of Yale Face B is the exact regime where self-attention is structurally and theoretically expected to outperform local convolutions.**

---

## 3. Mathematical Factorization Model (3-Way PARAFAC)

The reconstruction model factorizes the 3D tensor $\mathbf{X} \in \mathbb{R}^{192 \times 168 \times 160}$ into three 1D factor generator networks and a learnable core weight vector:
$$\hat{\mathbf{X}}_{ijk} = \sum_{r=1}^R \frac{\lambda_r}{R} \mathbf{U}_{ir} \mathbf{V}_{jr} \mathbf{W}_{kr}$$

- **Mode 1 Network ($\mathbf{U}_{\text{net}}$):** Generates spatial height factor $\mathbf{U} \in \mathbb{R}^{192 \times R}$.
- **Mode 2 Network ($\mathbf{V}_{\text{net}}$):** Generates spatial width factor $\mathbf{V} \in \mathbb{R}^{168 \times R}$.
- **Mode 3 Network ($\mathbf{W}_{\text{net}}$):** Generates illumination factor $\mathbf{W} \in \mathbb{R}^{160 \times R}$.
- **Core Vector ($\mathbf{\lambda}$):** Trainable scale vector $\mathbf{\lambda} \in \mathbb{R}^R$, initialized to ones.
- **Decomposition Rank:** $R = 2000$ (as specified by the original paper).
- **Positional Input Encoding:** 1D Fourier positional encodings for each mode:
  $$\mathbf{z}_U \in \mathbb{R}^{1 \times 2000 \times 192}, \quad \mathbf{z}_V \in \mathbb{R}^{1 \times 2000 \times 168}, \quad \mathbf{z}_W \in \mathbb{R}^{1 \times 2000 \times 160}$$

---

## 4. Evaluated Model Variants and Parameter Matching Audit

All models have been instantiated and audited on device to verify exact parameter matching:

| Model Variant | Mode 1 ($\mathbf{U}$) | Mode 2 ($\mathbf{V}$) | Mode 3 ($\mathbf{W}$) | Instantiated Parameters | $\Delta$ Params vs. Hybrid | Status |
|:---|:---|:---|:---|---:|---:|:---:|
| **`baseline`** | 1D Skip CNN | 1D Skip CNN | 1D Skip CNN (`skip1d`) | 5,470,508 | -132,480 (-2.36%) | Reference |
| **`hybrid`** | 1D Skip CNN | 1D Skip CNN | 1D Skip + Pre-LN MHSA + FFN | 5,602,988 | 0 (Reference) | **Target** |
| **`cnn_control`** | 1D Skip CNN | 1D Skip CNN | 1D Skip + 1D Conv Residual Block | 5,603,160 | **+172 (+0.0031%)** | **PASS (< 0.01%)** |
| **`pointwise_control`** | 1D Skip CNN | 1D Skip CNN | 1D Skip + Pointwise Linear MLP | 5,602,860 | **-128 (-0.0023%)** | **PASS (< 0.01%)** |
| **`transformer`** | 1D Skip CNN | 1D Skip CNN | Pre-LN SpecViT with ALiBi | 4,955,144 | -647,844 (-11.56%) | Standalone |

### Control Block Formulations
- **`cnn_control`:** Mode 3 incorporates a 1D convolutional residual block with intermediate channel width $d_{\text{mid}} = 172$, matching the MHSA+FFN block within $+172$ parameters (+0.0031%).
- **`pointwise_control`:** Mode 3 incorporates a two-layer pointwise linear MLP with intermediate width $d_{\text{mid}} = 514$, matching the MHSA+FFN block within $-128$ parameters (-0.0023%).

---

## 5. Experimental Fairness and Integrity Controls

1. **Shared Degraded Observation:** Generated once under master seed 42 and shared identically across all 5 models.
2. **Latent Code Preservation:** Learning from the Figure 09 audit, the runner script will explicitly save `{'model_state_dict': ..., 'u_inp': u_inp.cpu(), 'v_inp': v_inp.cpu(), 'w_inp': w_inp.cpu(), 'core': core.cpu()}` in the `.pth` checkpoint to guarantee 100% offline metric reproducibility.
3. **Strict Checkpoint Rule:** Checkpoint selection and best reconstruction capture will use **strictly unsupervised $L_1$ measurement loss**:
   $$\mathcal{L}(\mathbf{Y}, \hat{\mathbf{X}}) = \|\mathbf{Y} - \hat{\mathbf{X}}\|_1$$
   Zero ground-truth PSNR or SSIM leakage during training.
4. **Optimization Protocol:**
   - Optimizer: Adam ($\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$)
   - Learning rate: $\eta = 10^{-3}$
   - Training budget: $1,200$ epochs per variant (verified paper protocol for Figure 10).
5. **Evaluation Metric Standards:**
   - Both full-volume 3D tensor metrics ($\text{data\_range} = 1.0$) and slice-based metrics will be reported clearly, preventing any title-scale confusion.

---

## 6. Resource Requirements and Execution Estimates

- **Device:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MB VRAM).
- **Per-Variant Runtime:** ~140 seconds (~2.3 minutes).
- **Total Suite Runtime (5 variants):** ~11 to 12 minutes.
- **Estimated Peak VRAM:** ~895 MB (< 11% of available VRAM).
- **Output Destination:** `results/hybrid/figure10_yale_denoising/`.

---

## 7. Deliverables Plan

Upon execution, the runner will produce:
1. `results/hybrid/figure10_yale_denoising/baseline_checkpoint.pth`
2. `results/hybrid/figure10_yale_denoising/hybrid_checkpoint.pth`
3. `results/hybrid/figure10_yale_denoising/cnn_control_checkpoint.pth`
4. `results/hybrid/figure10_yale_denoising/pointwise_control_checkpoint.pth`
5. `results/hybrid/figure10_yale_denoising/transformer_checkpoint.pth`
6. `figure10_yale_b_comparison.csv` and `.json`
7. `figure10_yale_b_comparison_panel.png` (with dual metric labeling)
8. `figure10_yale_b_diff_map.png` (Hybrid vs CNN Control residual map)
9. Final research report: `reports/extensions/YALE_HYBRID_EXPERIMENT_REPORT.md`

---

## 8. Go / No-Go Verdict

- **Readiness Verdict:** **READY FOR FULL-SCALE TRAINING.**
- **Holding Action:** Standing by for explicit user approval before launching the 1,200-epoch execution.

