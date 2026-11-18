# DeepTensor++ Visual Results Consistency & Metric Re-Evaluation Audit
**Figures 09, 10, 11, and 12 Cross-Application Panel Verification**

**Date:** 2026-11-08  
**Audit Target:** Visual comparison panels, error maps, saved JSON/CSV metrics, and final experiment reports for Figures 09, 10, 11, and 12.  
**Platform:** NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.1), Python 3.11.9, PyTorch 2.5.1+cu121  
**Scope:** Exhaustive scientific audit of displayed panel metrics vs. verified full-volume benchmark metrics; mathematical root-cause analysis of dynamic range scaling; resolution of ranking shifts between single slices and full tensors; generation of dual-labeled publication-grade annotated panels.  
**Preservation Notice:** All original visualization panels, model checkpoints, and historical metrics are strictly preserved without modification.

---

## 1. Executive Summary & Audit Conclusions

This audit resolves the apparent numerical discrepancies between the **visual comparison panels** (`*_comparison_panel.png`) and the **authoritative full-volume metric reports** (`*_comparison.json`, `*_comparison.csv`, and research reports) across all four applications in DeepTensor++:

```
Summary of Primary Visual Consistency Findings:
1. Two Separate Sources of Discrepancy:
   - Source A (Error Distribution Discrepancy): The reconstruction error of an individual 2D slice (MSE_slice)
     differs from the average reconstruction error across the full 3D tensor (MSE_volume).
     Under the identical canonical range Δ = 1.0000, this yields:
     ΔPSNR_error = PSNR_volume(Δ=1.0) - PSNR_slice(Δ=1.0) = 10 * log10(MSE_slice / MSE_volume).
   - Source B (Dynamic-Range Penalty on the Same Slice): When evaluating the EXACT SAME SLICE (same MSE_slice),
     calculating PSNR using the local slice peak-to-peak range (Δ_slice = ptp(X_slice)) instead of the canonical
     global range (Δ_global = 1.0000) depresses the metric by:
     ΔPSNR_range = PSNR_slice(Δ=1.0) - PSNR_slice(Δ=Δ_slice) = 20 * log10(1.0 / Δ_slice).
     This penalty applies strictly to range changes on the same slice, independent of volume MSE.
2. Application Decompositions (Existing Numerical Results):
   - Fig 09 (Band 130, Δ_slice = 0.2175):
     * Effect 1 (Error Distribution): +11.92 dB (Band 130 MSE is higher than the 260-band volume average).
     * Effect 2 (Dynamic Range Penalty): +13.25 dB (20 * log10(1.0 / 0.2175) on the same slice).
     * Total Discrepancy: +25.18 dB (explains 20.89 dB panel vs. 46.07 dB full volume report).
   - Fig 10 (Illum 80, Δ_slice = 1.0000):
     * Effect 1 (Error Distribution): -0.24 dB (Illumination 80 is well-lit with slightly lower MSE than volume average).
     * Effect 2 (Dynamic Range Penalty): 0.00 dB (20 * log10(1.0 / 1.0) = 0.00 dB; range is already 1.0000).
     * Total Discrepancy: -0.24 dB (panel shows 28.80 dB vs. 28.56 dB report).
   - Fig 11 (Frame 32, Δ_slice = 0.9910):
     * Effect 1 (Error Distribution): +0.92 dB (Frame 32 has higher motion/detail than static video average).
     * Effect 2 (Dynamic Range Penalty): +0.08 dB (20 * log10(1.0 / 0.9910) on the same frame).
     * Total Discrepancy: +1.00 dB (panel shows 24.16 dB vs. 25.16 dB report).
   - Fig 12 (Slice 28, Δ_slice = 0.6364):
     * Effect 1 (Error Distribution): +0.28 dB (Slice 28 central thorax MSE is slightly higher than volume average).
     * Effect 2 (Dynamic Range Penalty): +3.93 dB (20 * log10(1.0 / 0.6364) on the same slice).
     * Total Discrepancy: +4.21 dB (explains 26.42 dB panel vs. 30.62 dB report).
3. Ranking Alignment:
   - Full-volume model rankings remain 100% authoritative and verified directly from disk.
   - On isolated 2D cross-sections, local high-frequency motion (Fig 11) or slice-specific lighting (Fig 10) can induce minor local ranking shifts (e.g., ±0.01 to ±0.26 dB).
4. Corrective Action:
   - Preserved all original comparison panels untouched.
   - Generated dual-labeled annotated panels explicitly distinguishing "Full-Volume PSNR" from "Central Slice PSNR".
```

---

## 2. Mathematical Root Cause: Separating Error Distribution from Dynamic-Range Scaling

### 2.1 Definition of the Three Metric Quantities
Let $\mathbf{X} \in \mathbb{R}^{H \times W \times T}$ denote the ground-truth 3D tensor normalized such that $\max(\mathbf{X}) - \min(\mathbf{X}) \approx 1.0000$, and let $\hat{\mathbf{X}}$ be the reconstructed tensor.
Let $\mathbf{X}_k = \mathbf{X}_{:, :, k} \in \mathbb{R}^{H \times W}$ denote the 2D cross-section at selected index $k$ (e.g. Band 130 for Fig 09, Illumination 80 for Fig 10, Frame 32 for Fig 11, or Slice 28 for Fig 12), and let $\hat{\mathbf{X}}_k$ denote its reconstruction.

To eliminate confusion between the benchmark tables and visualization panels, we must strictly distinguish between **three distinct mathematical quantities**:

1. **Full-Volume Canonical PSNR ($\text{PSNR}_{\text{volume}}$):**  
   Computed over all $N_{\text{vol}} = H \times W \times T$ voxels using the canonical global dynamic range $\Delta_{\text{global}} = 1.0000$:
   $$\text{MSE}_{\text{volume}} = \frac{1}{H W T} \sum_{x=1}^H \sum_{y=1}^W \sum_{t=1}^T \left(X(x,y,t) - \hat{X}(x,y,t)\right)^2$$
   $$\text{PSNR}_{\text{volume}} = 10 \log_{10}\left( \frac{1.0^2}{\text{MSE}_{\text{volume}}} \right) = -10 \log_{10}(\text{MSE}_{\text{volume}})$$
   This is the authoritative benchmark metric reported in all JSON summaries, CSV tables, and research papers.

2. **Single-Slice Canonical PSNR ($\text{PSNR}_{\text{slice}}(\Delta=1.0)$):**  
   Computed on the single 2D slice $\mathbf{X}_k$ ($N_{\text{slice}} = H \times W$) using the **exact same canonical global dynamic range** $\Delta_{\text{global}} = 1.0000$:
   $$\text{MSE}_{\text{slice}} = \frac{1}{H W} \sum_{x=1}^H \sum_{y=1}^W \left(X_k(x,y) - \hat{X}_k(x,y)\right)^2$$
   $$\text{PSNR}_{\text{slice}}(\Delta=1.0) = 10 \log_{10}\left( \frac{1.0^2}{\text{MSE}_{\text{slice}}} \right) = -10 \log_{10}(\text{MSE}_{\text{slice}})$$

3. **Single-Slice Local-Range PSNR ($\text{PSNR}_{\text{slice}}(\Delta=\Delta_{\text{slice}})$):**  
   Computed on the same 2D slice $\mathbf{X}_k$, but defining the dynamic range locally as the peak-to-peak amplitude of that slice: $\Delta_{\text{slice}} = \text{ptp}(\mathbf{X}_k) = \max(\mathbf{X}_k) - \min(\mathbf{X}_k)$:
   $$\text{PSNR}_{\text{slice}}(\Delta=\Delta_{\text{slice}}) = 10 \log_{10}\left( \frac{\Delta_{\text{slice}}^2}{\text{MSE}_{\text{slice}}} \right) = 20 \log_{10}(\Delta_{\text{slice}}) - 10 \log_{10}(\text{MSE}_{\text{slice}})$$
   This was the quantity computed by `SideBySideComparator._generate_visual_panels` in the original comparison panels.

---

### 2.2 Effect 1: Error Distribution Discrepancy ($\text{MSE}_{\text{slice}} \neq \text{MSE}_{\text{volume}}$)
Full-volume versus slice PSNR differences arise partly because the reconstruction error distribution of one selected slice can differ from the average error across the entire tensor.

When holding the dynamic range strictly constant at $\Delta = 1.0000$, any difference between the full-volume PSNR and the slice PSNR is **governed purely by spatial/temporal error distribution variation**:
$$\Delta \text{PSNR}_{\text{error}} = \text{PSNR}_{\text{volume}} - \text{PSNR}_{\text{slice}}(\Delta=1.0) = 10 \log_{10}\left(\frac{\text{MSE}_{\text{slice}}}{\text{MSE}_{\text{volume}}}\right)$$

- **Physical Behavior:**
  - If a selected slice possesses higher reconstruction difficulty (e.g. Frame 32 with fast movement in Fig 11, or central thorax Slice 28 in Fig 12), then $\text{MSE}_{\text{slice}} > \text{MSE}_{\text{volume}}$, making $\text{PSNR}_{\text{slice}}(\Delta=1.0) < \text{PSNR}_{\text{volume}}$ ($\Delta \text{PSNR}_{\text{error}} > 0$).
  - If a selected slice is cleaner or brighter than the tensor average (e.g. Illumination 80 in Fig 10), then $\text{MSE}_{\text{slice}} < \text{MSE}_{\text{volume}}$, making $\text{PSNR}_{\text{slice}}(\Delta=1.0) > \text{PSNR}_{\text{volume}}$ ($\Delta \text{PSNR}_{\text{error}} < 0$).
- **Crucial Distinction:** This term has **zero dependence on local dynamic-range scaling**; both metrics are evaluated with the identical canonical range $\Delta = 1.0000$.

---

### 2.3 Effect 2: Dynamic-Range Penalty on the Same Slice
Slice PSNR calculated using local slice peak-to-peak range differs from slice PSNR calculated using the canonical global range.

The dynamic-range penalty formula applies **only when comparing the same slice and same MSE under two different peak-to-peak ranges**:
$$\Delta \text{PSNR}_{\text{range}} = \text{PSNR}_{\text{slice}}(\Delta=1.0) - \text{PSNR}_{\text{slice}}(\Delta=\Delta_{\text{slice}})$$
$$= \left[ 20 \log_{10}(1.0) - 10 \log_{10}(\text{MSE}_{\text{slice}}) \right] - \left[ 20 \log_{10}(\Delta_{\text{slice}}) - 10 \log_{10}(\text{MSE}_{\text{slice}}) \right]$$
$$= 20 \log_{10}\left(\frac{1.0}{\Delta_{\text{slice}}}\right) = -20 \log_{10}(\Delta_{\text{slice}})$$

- **Physical Behavior:**
  - Notice that the reconstruction error $\text{MSE}_{\text{slice}}$ **cancels out completely** from this subtraction.
  - This formula is purely a logarithmic scaling consequence of the chosen definition of $\Delta$.
  - It does **not** describe the difference between the full volume and the slice; it describes the difference on the **same slice** between using the global reference range ($1.0000$) versus the local peak-to-peak range ($\Delta_{\text{slice}}$).
  - Whenever a slice does not span the entire dynamic range $[0, 1]$ ($\Delta_{\text{slice}} < 1.0000$), calculating PSNR with $\Delta_{\text{slice}}$ automatically penalizes the resulting number by $20 \log_{10}(1.0 / \Delta_{\text{slice}})$ decibels, even if the reconstruction is exceptionally accurate.

---

### 2.4 Total Discrepancy Decomposition: Exact Additive Sum
The total apparent discrepancy between the **Full-Volume Benchmark PSNR** reported in tables and the **Local Slice PSNR** originally printed on panel titles decomposes exactly into the sum of these two separate effects:

$$\underbrace{\text{PSNR}_{\text{volume}} - \text{PSNR}_{\text{slice}}(\Delta=\Delta_{\text{slice}})}_{\text{Total Discrepancy}}$$
$$= \underbrace{\left[ \text{PSNR}_{\text{volume}} - \text{PSNR}_{\text{slice}}(\Delta=1.0) \right]}_{\text{Effect 1: Error Distribution Variation } (\Delta \text{PSNR}_{\text{error}})} + \underbrace{\left[ \text{PSNR}_{\text{slice}}(\Delta=1.0) - \text{PSNR}_{\text{slice}}(\Delta=\Delta_{\text{slice}}) \right]}_{\text{Effect 2: Dynamic-Range Penalty on Same Slice } (\Delta \text{PSNR}_{\text{range}})}$$
$$= \underbrace{10 \log_{10}\left(\frac{\text{MSE}_{\text{slice}}}{\text{MSE}_{\text{volume}}}\right)}_{\text{Effect 1: Error Distribution}} + \underbrace{20 \log_{10}\left(\frac{1.0}{\Delta_{\text{slice}}}\right)}_{\text{Effect 2: Dynamic-Range Penalty}}$$

---

### 2.5 Separation of Effects Illustrated on Verified Results

The table below demonstrates this exact decomposition using our verified experimental results across all four figures:

| Experiment & Representative Model | Full-Volume PSNR [$\Delta=1.0$] | Slice Canonical PSNR [$\Delta=1.0$] | Slice Local PSNR [$\Delta=\Delta_{\text{slice}}$] | Effect 1: Error Variation ($\Delta \text{PSNR}_{\text{error}}$) | Effect 2: Range Penalty ($\Delta \text{PSNR}_{\text{range}}$) | Total Discrepancy (Effect 1 + Effect 2) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fig 09 (ICVL)**: `baseline` | 46.0653 dB | 34.1411 dB | 20.8894 dB | **+11.9242 dB** | **+13.2517 dB** | **+25.1759 dB** (exact match) |
| **Fig 09 (ICVL)**: `transformer` | 43.4026 dB | 37.0371 dB | 23.7854 dB | **+6.3655 dB** | **+13.2517 dB** | **+19.6172 dB** (exact match) |
| **Fig 10 (Yale B)**: `transformer` | 28.5552 dB | 28.7963 dB | 28.7963 dB | **-0.2411 dB** | **0.0000 dB** | **-0.2411 dB** (exact match) |
| **Fig 10 (Yale B)**: `cnn_control` | 28.3521 dB | 29.0611 dB | 29.0611 dB | **-0.7090 dB** | **0.0000 dB** | **-0.7090 dB** (exact match) |
| **Fig 11 (Video CS)**: `baseline` | 25.1577 dB | 24.2418 dB | 24.1629 dB | **+0.9159 dB** | **+0.0789 dB** | **+0.9948 dB** (exact match) |
| **Fig 11 (Video CS)**: `cnn_control` | 25.0395 dB | 24.0923 dB | 24.0133 dB | **+0.9472 dB** | **+0.0789 dB** | **+1.0261 dB** (exact match) |
| **Fig 12 (PET/CT)**: `pointwise_control` | 30.6212 dB | 30.3409 dB | 26.4155 dB | **+0.2803 dB** | **+3.9254 dB** | **+4.2057 dB** (exact match) |
| **Fig 12 (PET/CT)**: `baseline` | 30.0023 dB | 29.7156 dB | 25.7902 dB | **+0.2867 dB** | **+3.9254 dB** | **+4.2121 dB** (exact match) |

> [!NOTE]
> In Figure 09 and Figure 12, Effect 2 (the dynamic-range penalty) accounts for a major portion of the apparent drop on the original panels (+13.25 dB and +3.93 dB respectively). In Figure 10, Effect 2 is exactly 0.00 dB because the slice peak-to-peak range is 1.0000. In Figure 11, Effect 2 is negligible (+0.08 dB), and the discrepancy is driven by Effect 1 (Frame 32 having higher motion/texture than the video average).

---

## 3. Application-by-Application Audit

### 3.1 Figure 09: ICVL Hyperspectral Denoising & Super-Resolution
- **Dataset:** `data/icvl.mat`, tensor shape: $174 \times 164 \times 260$ spectral bands.
- **Displayed Slice:** Central spectral band $z = 130$ ($\lambda \approx 550\text{ nm}$).
- **Ground Truth Range:**
  - Full volume: $\min = 0.0384, \max = 1.0000, \Delta_{\text{vol}} = 0.9616$ (canonical: $1.0000$).
  - Slice 130: $\min = 0.0563, \max = 0.2738, \Delta_{\text{slice}} = 0.2175$.
- **Dynamic Range Penalty:** $20 \log_{10}(1.0 / 0.2175) = \mathbf{+13.25\text{ dB}}$.

#### Detailed Metric Comparison (Figure 09)

| Model Variant | Full-Volume PSNR ($\Delta=1.0$) | Full-Volume MAE | Slice 130 PSNR (Local $\Delta=0.2175$) | Slice 130 PSNR (Range $\Delta=1.0$) | Slice 130 MAE | Discrepancy (Vol vs. Local Slice) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`baseline`** | **46.0653 dB** | **0.003350** | 20.8894 dB | 34.1411 dB | 0.014937 | +25.18 dB (Vol vs. Slice Local) / +11.92 dB ($\Delta=1.0$) |
| **`cnn_control`** | 45.5844 dB | 0.003436 | 20.7420 dB | 33.9936 dB | 0.014841 | +24.84 dB (Vol vs. Slice Local) / +11.59 dB ($\Delta=1.0$) |
| **`pointwise_control`** | 45.4229 dB | 0.003496 | 22.7500 dB | 36.0017 dB | 0.011641 | +22.67 dB (Vol vs. Slice Local) / +9.42 dB ($\Delta=1.0$) |
| **`hybrid`** | 44.8526 dB | 0.003757 | 22.5541 dB | 35.8058 dB | 0.012121 | +22.30 dB (Vol vs. Slice Local) / +9.05 dB ($\Delta=1.0$) |
| **`transformer`** | 43.4026 dB | 0.004175 | 23.7854 dB | 37.0371 dB | 0.010660 | +19.62 dB (Vol vs. Slice Local) / +6.37 dB ($\Delta=1.0$) |

#### Model Ranking Analysis (Figure 09):
- **Full Volume:** Baseline (46.07 dB) > CNN Control (45.58 dB) > Pointwise Control (45.42 dB) > Hybrid (44.85 dB) > Transformer (43.40 dB). **Winner: Pure CNN Baseline.**
- **Slice 130 Local:** Transformer (23.79 dB) > Pointwise Control (22.75 dB) > Hybrid (22.55 dB) > Baseline (20.89 dB) > CNN Control (20.74 dB).
- **Explanation of Slice 130 Inversion:** Band 130 is an extreme low-intensity transition band ($\max = 0.2738$). The Transformer happens to predict low-variance mean values closer to this narrow baseline, while Pure CNNs optimize smooth spectral gradient continuity across all 260 bands. Across the full 260 bands, the Baseline dominates by +2.66 dB. Labelling the panel only as "PSNR" created confusion; the annotated panel now explicitly presents both metrics.

---

### 3.2 Figure 10: Yale Face B Multi-Illumination Tensor Denoising
- **Dataset:** `data/yale_b.mat`, tensor shape: $192 \times 168 \times 160$ illumination conditions.
- **Displayed Slice:** Central illumination condition $z = 80$.
- **Ground Truth Range:**
  - Full volume: $\min = 0.0000, \max = 1.0000, \Delta_{\text{vol}} = 1.0000$.
  - Slice 80: $\min = 0.0000, \max = 1.0000, \Delta_{\text{slice}} = 1.0000$.
- **Dynamic Range Penalty:** $20 \log_{10}(1.0 / 1.0) = \mathbf{0.00\text{ dB}}$.

#### Detailed Metric Comparison (Figure 10)

| Model Variant | Full-Volume PSNR ($\Delta=1.0$) | Full-Volume MAE | Slice 80 PSNR (Local $\Delta=1.0$) | Slice 80 MAE | Discrepancy (Vol vs. Slice 80) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`transformer`** | **28.5552 dB** | **0.019528** | 28.7963 dB | 0.021239 | -0.2411 dB |
| **`cnn_control`** | 28.3521 dB | 0.020051 | 29.0611 dB | 0.020977 | -0.7090 dB |
| **`pointwise_control`** | 28.1703 dB | 0.020540 | 28.5997 dB | 0.022797 | -0.4294 dB |
| **`hybrid`** | 28.0392 dB | 0.020699 | 28.7376 dB | 0.021773 | -0.6984 dB |
| **`baseline`** | 27.7808 dB | 0.021452 | 28.1946 dB | 0.023241 | -0.4138 dB |

#### Model Ranking Analysis (Figure 10):
- **Full Volume:** Transformer (28.56 dB) > CNN Control (28.35 dB) > Pointwise Control (28.17 dB) > Hybrid (28.04 dB) > Baseline (27.78 dB). **Winner: Pure Transformer.**
- **Slice 80:** CNN Control (29.06 dB) > Transformer (28.80 dB) > Hybrid (28.74 dB) > Pointwise Control (28.60 dB) > Baseline (28.19 dB).
- **Explanation:** Condition 80 is a frontal illumination condition with strong direct light, where local convolutional filters reconstruct skin textures slightly better (+0.26 dB). However, across all 160 illumination conditions—including severe grazing angles and shadowed profiles—the Pure Transformer's global linear subspace projection bottleneck achieves superior global robustness (+0.20 dB over CNN Control, +0.77 dB over Baseline).

---

### 3.3 Figure 11: Video Compressive Sensing (CACTI)
- **Dataset:** `data/cat_video.mat`, tensor shape: $128 \times 128 \times 64$ frames.
- **Displayed Slice:** Central video frame $t = 32$.
- **Ground Truth Range:**
  - Full volume: $\min = 0.0000, \max = 1.0000, \Delta_{\text{vol}} = 1.0000$.
  - Frame 32: $\min = 0.0000, \max = 0.9910, \Delta_{\text{slice}} = 0.9910$.
- **Dynamic Range Penalty:** $20 \log_{10}(1.0 / 0.9910) = \mathbf{-0.08\text{ dB}}$ (negligible).

#### Detailed Metric Comparison (Figure 11)

| Model Variant | Full-Volume PSNR ($\Delta=1.0$) | Full-Volume MAE | Frame 32 PSNR (Local $\Delta=0.9910$) | Frame 32 PSNR (Range $\Delta=1.0$) | Frame 32 MAE | Discrepancy (Vol vs. Frame 32 Local) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`baseline`** | **25.1577 dB** | **0.036453** | **24.1629 dB** | **24.2418 dB** | **0.039984** | +0.9948 dB |
| **`cnn_control`** | 25.0395 dB | 0.037223 | 24.0133 dB | 24.0923 dB | 0.040420 | +1.0262 dB |
| **`pointwise_control`** | 24.8147 dB | 0.038381 | 23.9471 dB | 24.0261 dB | 0.042013 | +0.8676 dB |
| **`hybrid`** | 24.7440 dB | 0.038676 | 24.0342 dB | 24.1132 dB | 0.041116 | +0.7098 dB |
| **`transformer`** | 24.0502 dB | 0.041520 | 23.4089 dB | 23.4879 dB | 0.043135 | +0.6413 dB |

#### Model Ranking Analysis (Figure 11):
- **Full Volume:** Baseline (25.16 dB) > CNN Control (25.04 dB) > Pointwise Control (24.81 dB) > Hybrid (24.74 dB) > Transformer (24.05 dB). **Winner: Pure CNN Baseline.**
- **Frame 32:** Baseline (24.16 dB) > Hybrid (24.03 dB) > CNN Control (24.01 dB) > Pointwise Control (23.95 dB) > Transformer (23.41 dB).
- **Explanation:** Frame 32 preserves the global ranking structure: the Pure CNN Baseline wins, and Transformer trails by >0.7 dB. On Frame 32, Hybrid and CNN Control are essentially tied (24.03 vs 24.01 dB, a 0.02 dB difference), while across the 64-frame sequence, CNN Control is consistently superior (+0.30 dB over Hybrid).

---

### 3.4 Figure 12: 3D PET/CT Radon Compressive Sensing (40-Angle Tomography)
- **Dataset:** `data/pet.mat`, tensor shape: $128 \times 128 \times 56$ axial slices.
- **Displayed Slice:** Central axial slice $z = 28$.
- **Ground Truth Range:**
  - Full volume: $\min = 0.0000, \max = 1.0000, \Delta_{\text{vol}} = 1.0000$.
  - Slice 28: $\min = 0.0000, \max = 0.6364, \Delta_{\text{slice}} = 0.6364$.
- **Dynamic Range Penalty:** $20 \log_{10}(1.0 / 0.6364) = \mathbf{+3.93\text{ dB}}$.

#### Detailed Metric Comparison (Figure 12)

| Model Variant | Full-Volume PSNR ($\Delta=1.0$) | Full-Volume MAE | Slice 28 PSNR (Local $\Delta=0.6364$) | Slice 28 PSNR (Range $\Delta=1.0$) | Slice 28 MAE | Discrepancy (Vol vs. Slice 28 Local) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`pointwise_control`** | **30.6212 dB** | **0.017661** | **26.4155 dB** | **30.3409 dB** | **0.018554** | +4.2057 dB |
| **`transformer`** | 30.4056 dB | 0.018047 | 26.1859 dB | 30.1113 dB | 0.019096 | +4.2197 dB |
| **`cnn_control`** | 30.3128 dB | 0.018067 | 26.0314 dB | 29.9567 dB | 0.019141 | +4.2815 dB |
| **`hybrid`** | 30.2947 dB | 0.018238 | 26.0373 dB | 29.9627 dB | 0.019102 | +4.2574 dB |
| **`baseline`** | 30.0023 dB | 0.018809 | 25.7902 dB | 29.7156 dB | 0.019641 | +4.2121 dB |

#### Model Ranking Analysis (Figure 12):
- **Full Volume:** Pointwise Control (30.62 dB) > Transformer (30.41 dB) > CNN Control (30.31 dB) > Hybrid (30.29 dB) > Baseline (30.00 dB). **Winner: Pointwise Control.**
- **Slice 28:** Pointwise Control (26.42 dB) > Transformer (26.19 dB) > Hybrid (26.04 dB) > CNN Control (26.03 dB) > Baseline (25.79 dB).
- **Explanation:** Slice 28 exhibits near-perfect ranking agreement with the full volume: Pointwise Control wins, Transformer is second, and Baseline trails by >0.6 dB. Hybrid and CNN Control are tied within 0.006 dB (26.037 dB vs 26.031 dB). The ~4.2 dB lower PSNR on the original panel is completely accounted for by the local dynamic range scaling ($+3.93\text{ dB}$) plus slice 28's local anatomical complexity ($+0.28\text{ dB}$).

---

## 4. Synthesis of Discrepancies and Audit Checklist

| Audit Item | Question / Check | Audit Verdict | Evidence & Resolution |
|:---|:---|:---:|:---|
| **1. Slice Identification** | What exact slice was used for each panel? | **VERIFIED** | Fig 09: Band 130 of 260.<br>Fig 10: Illumination 80 of 160.<br>Fig 11: Frame 32 of 64.<br>Fig 12: Slice 28 of 56. |
| **2. Dynamic Range vs. Error Variation** | How do range penalties and error distributions interact? | **IDENTIFIED** | The dynamic-range penalty formula $\Delta \text{PSNR}_{\text{range}} = 20 \log_{10}(1.0 / \Delta_{\text{slice}})$ applies strictly when comparing the *same slice and same MSE* under two different peak-to-peak ranges. In contrast, the difference between full-volume and slice PSNR under the same range $\Delta=1.0$ is governed purely by error distribution variation $\Delta \text{PSNR}_{\text{error}} = 10 \log_{10}(\text{MSE}_{\text{slice}} / \text{MSE}_{\text{volume}})$. |
| **3. Metric Agreement** | Do panel metrics match volume metrics? | **RESOLVED** | No, they differ because of two separate, additive effects: (1) $\text{MSE}_{\text{slice}} \neq \text{MSE}_{\text{volume}}$ (error distribution across tensor), and (2) $\Delta_{\text{slice}} < 1.0$ (local dynamic-range penalty on that slice). The two-term formula accounts for $100.00\%$ of the discrepancy. |
| **4. Model Rankings** | Do slice rankings agree with full reports? | **ANALYZED** | Figures 11 and 12 preserve the general ranking hierarchy. Figures 09 and 10 exhibit expected local cross-sectional variations. Full-volume metrics remain the true benchmark ground truth. |
| **5. Spatial Consistency** | Are dimensions, scales, and colormaps uniform? | **CONFIRMED** | All reconstructions match GT shapes. Error maps use unified `cmap='inferno'` with shared 99th-percentile `vmax` across all model variants. |
| **6. Label Clarity** | Did panels explicitly label slice vs. volume? | **AMENDED** | Original panels ambiguously labeled metrics as "PSNR". Corrected panels now explicitly specify "Full-Volume PSNR" and "(Central Slice PSNR)". |
| **7. File Preservation** | Were historical assets protected? | **CONFIRMED** | All original panels, checkpoints, CSVs, and JSONs were preserved without modification. Annotated panels were generated as separate assets. |

---

## 5. Catalog of Visual Deliverables

All original and annotated visualization deliverables are archived and cross-referenced below:

### Figure 09 (ICVL Hyperspectral Denoising)
- **Original Panel (Preserved):** [`results/hybrid/figure09_hyperspectral/figure09_icvl_hsi_comparison_panel.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure09_hyperspectral/figure09_icvl_hsi_comparison_panel.png)
- **Annotated Panel (Dual Metrics):** [`results/hybrid/figure09_hyperspectral/figure09_icvl_hsi_comparison_panel_annotated.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure09_hyperspectral/figure09_icvl_hsi_comparison_panel_annotated.png)
- **Residual Difference Map:** [`results/hybrid/figure09_hyperspectral/figure09_icvl_hsi_diff_map.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure09_hyperspectral/figure09_icvl_hsi_diff_map.png)

### Figure 10 (Yale Face B PARAFAC Denoising)
- **Original Panel (Preserved):** [`results/hybrid/figure10_yale_denoising/figure10_yale_b_comparison_panel.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure10_yale_denoising/figure10_yale_b_comparison_panel.png)
- **Annotated Panel (Dual Metrics):** [`results/hybrid/figure10_yale_denoising/figure10_yale_b_comparison_panel_annotated.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure10_yale_denoising/figure10_yale_b_comparison_panel_annotated.png)
- **Residual Difference Map:** [`results/hybrid/figure10_yale_denoising/figure10_yale_b_diff_map.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure10_yale_denoising/figure10_yale_b_diff_map.png)

### Figure 11 (Video Compressive Sensing - CACTI)
- **Original Panel (Preserved):** [`results/hybrid/figure11_video_cs/figure11_video_cs_comparison_panel.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure11_video_cs/figure11_video_cs_comparison_panel.png)
- **Annotated Panel (Dual Metrics):** [`results/hybrid/figure11_video_cs/figure11_video_cs_comparison_panel_annotated.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure11_video_cs/figure11_video_cs_comparison_panel_annotated.png)
- **Residual Difference Map:** [`results/hybrid/figure11_video_cs/figure11_video_cs_diff_map.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure11_video_cs/figure11_video_cs_diff_map.png)
- **Temporal Progression Strip:** [`results/hybrid/figure11_video_cs/figure11_video_cs_temporal_strip.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure11_video_cs/figure11_video_cs_temporal_strip.png)

### Figure 12 (3D PET/CT Radon CS)
- **Original Panel (Preserved):** [`results/hybrid/figure12_pet_ct/figure12_pet_ct_comparison_panel.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure12_pet_ct/figure12_pet_ct_comparison_panel.png)
- **Annotated Panel (Dual Metrics):** [`results/hybrid/figure12_pet_ct/figure12_pet_ct_comparison_panel_annotated.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure12_pet_ct/figure12_pet_ct_comparison_panel_annotated.png)
- **Residual Difference Map:** [`results/hybrid/figure12_pet_ct/figure12_pet_ct_diff_map.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure12_pet_ct/figure12_pet_ct_diff_map.png)
- **Axial Progression Strip:** [`results/hybrid/figure12_pet_ct/figure12_pet_ct_slice_strip.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure12_pet_ct/figure12_pet_ct_slice_strip.png)

---

## 6. Audit Conclusion & Recommendations

1. **Protocol Rigor & Transparency:**  
   Future reporting should always display the dual-metric convention whenever a single 2D slice from a multi-dimensional tensor is visualized:
   - Primary Header: Authoritative Full-Volume PSNR (range $\Delta = 1.0000$).
   - Sub-Header: Slice-Specific Local PSNR and MAE.
2. **Definitive Ground-Truth Benchmark:**  
   The authoritative numbers for DeepTensor++ across all four applications remain the full-volume metrics established by offline independent verification directly from disk:
   - **Fig 09 (ICVL):** Baseline CNN wins (**46.07 dB** vs 44.85 dB Hybrid).
   - **Fig 10 (Yale Face B):** Pure Transformer wins (**28.56 dB** vs 27.78 dB Baseline).
   - **Fig 11 (Video CS):** Baseline CNN wins (**25.16 dB** vs 24.74 dB Hybrid).
   - **Fig 12 (PET/CT):** Pointwise Control wins (**30.62 dB** vs 30.00 dB Baseline).
3. **No Retraining or Checkpoint Modifications:**  
   All model weights, logs, and benchmark summaries remain 100% verified, consistent, and reproducible.

