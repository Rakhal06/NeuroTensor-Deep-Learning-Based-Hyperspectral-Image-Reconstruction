# DeepTensor++ Research Report: Figure 12 3D PET/CT Radon CS Readiness & Protocol Verification

**Date:** 2026-11-08  
**Experiment Target:** Figure 12 — 3D PET/CT Radon Compressive Sensing (40-Angle Tomography)  
**Dataset:** 3D PET/CT Scan (`data/pet.mat`), shape: $128 \times 128 \times 56$  
**Hardware Platform:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MB VRAM), CUDA 12.1  
**Software Environment:** Python 3.11.9 (`.venv`), PyTorch 2.5.1+cu121  
**Output Directory:** [`results/hybrid/figure12_pet_ct/`](file:///e:/Rakhal_Tensor/results/hybrid/figure12_pet_ct/)  
**Configuration File:** [`configs/hybrid/figure12_pet_ct.yaml`](file:///e:/Rakhal_Tensor/configs/hybrid/figure12_pet_ct.yaml)  
**Experiment Runner:** [`experiments/hybrid_prototypes/run_hybrid_figure12_pet.py`](file:///e:/Rakhal_Tensor/experiments/hybrid_prototypes/run_hybrid_figure12_pet.py)  
**Independent Verification Tool:** [`tools/verify_figure12_metrics.py`](file:///e:/Rakhal_Tensor/tools/verify_figure12_metrics.py)  
**Target Budget:** 4,000 epochs per variant (verified paper improvement budget)  

---

## 1. Executive Summary & Readiness Verdict

We have completed the pre-training protocol verification, physical sensing audit, cross-experiment metadata reconciliation, parameter-matching calibration, and 1-epoch smoke test for **Figure 12: 3D PET/CT Radon Compressive Sensing** across all five model variants:
1. **`baseline`**: Rank-1000 3-Way CP: $U(128)$ 1D Skip + $V(128)$ 1D Skip + $W(56)$ 1D Skip ($3,918,508$ params).
2. **`hybrid`**: Rank-1000 3-Way CP: $U(128)$ 1D Skip + $V(128)$ 1D Skip + $W(56)$ Hybrid Attention ($4,050,988$ params).
3. **`cnn_control`**: Parameter-matched pure 1D CNN with Conv1d residual block ($4,051,160$ params, **+0.0042% match** to Hybrid).
4. **`pointwise_control`**: Parameter-matched pointwise linear MLP mixing + FFN ($4,050,860$ params, **-0.0032% match** to Hybrid).
5. **`transformer`**: Rank-1000 3-Way CP: $U(128)$ 1D Skip + $V(128)$ 1D Skip + $W(56)$ Pure Spectral Transformer ($3,663,144$ params, **-9.57%**).

### Audit Verdict: **PASSED (100% READY FOR FULL TRAINING)**
- **Preservation Confirmed:** All Figure 09, Figure 10, and Figure 11 checkpoints, logs, and verification files are 100% intact and preserved.
- **Cross-Experiment Discrepancy Reconciled:** The Figure 09 configuration is verified as **Rank $R=20$** with mixed Poisson ($\tau=100$) and Gaussian readout noise ($\text{SNR}=2.0$). The inadvertent typo in the cross-experiment summary table of the Figure 11 artifact has been reconciled and corrected.
- **Physical Sensing Operator:** 40-angle slice-wise 2D Radon forward projection $\mathbf{Y} = \mathcal{R}_{40}(\mathbf{X})$ and detector noise model verified against the peer-reviewed baseline specification.
- **Continuous Axial Geometry:** Mode 3 represents the continuous cranio-caudal axial slice dimension $z \in [0, 55]$ of the human patient (not projection angles). 1D sinusoidal positional encoding correctly maps vertical anatomical progression.
- **Strict Parameter Matching:** `cnn_control` (+0.0042%) and `pointwise_control` (-0.0032%) match `hybrid` within $\pm 0.005\%$.
- **Unsupervised Model Selection:** Zero ground-truth leakage. Model tracking and checkpoint selection rely strictly on unsupervised sinogram measurement $\ell_2$ loss.
- **Independent Verification Tool:** [`tools/verify_figure12_metrics.py`](file:///e:/Rakhal_Tensor/tools/verify_figure12_metrics.py) verified all smoke test checkpoints with **0.0000 dB discrepancy**.
- **Resource Projections:** Peak VRAM is **~577 MB** (<7.1% of GPU capacity). Total expected runtime for all 5 models at 4,000 epochs is **~33.3 minutes**.

---

## 2. Cross-Experiment Metadata Reconciliation

### 2.1 Reconciling Figure 09 Metadata
- **Discrepancy Noticed:** An earlier summary cell in the Figure 11 report inadvertently listed Figure 09 with "Rank $R=64$, Additive Gaussian $\sigma=0.1$".
- **Files Inspected:**
  1. `configs/reproduction/figure09_hyperspectral.json`: `rank: 20`, `noise_snr: 2`, `tau: 100`.
  2. `configs/hybrid/figure09_hyperspectral.yaml`: `rank: 20`, `noise_snr: 2.0`, `tau: 100`.
  3. `results/hybrid/figure09_hyperspectral/experiment_metadata.json`: `rank: 20`, `snr_gaussian: 2.0`, `tau_poisson: 100.0`.
  4. `reports/extensions/FIGURE09_METRIC_VERIFICATION_REPORT.md`: `Rank R=20`, `Poisson tau=100`, `Gaussian SNR=2.0`.
- **Finding:** The actual, authoritative Figure 09 experiment was executed with **Rank $R=20$** and mixed Poisson ($\tau=100$) + Gaussian readout noise ($\text{SNR}=2.0$). The text "Rank $R=64$, $\sigma=0.1$" was an inadvertent documentation typo in the cross-experiment summary table of the Figure 11 artifact, which has now been updated and corrected. Historical reports remain unmodified.

### 2.2 Verification of Figure 10 and Figure 11 Protocols
- **Figure 10:** Confirmed Rank $R=2000$, 3-Way CP ($U \otimes V \otimes W$), 30% Salt & Pepper noise, 1,200 epochs, $L_1$ unsupervised measurement loss.
- **Figure 11:** Confirmed Rank $R=32$, Bilinear Matrix ($U \times V$), 8-frame CACTI temporal multiplexing, 3,000 epochs, $L_2$ unsupervised measurement loss.

---

## 3. Figure 12 Physical Forward Model & Tomographic Geometry

### 3.1 Dataset Specifications
- **File:** `data/pet.mat` (or `data/PET.mat`)
- **Key:** `'hypercube'`
- **Tensor Shape:** $H=128, W=128, T=56$ axial slices.
- **Data Type & Range:** Originally `uint16` in $[0, 53980]$. Normalized to `float32` in $[0.0, 1.0]$ via $\mathbf{X}_{\text{norm}} = \mathbf{X} / \max(\mathbf{X})$. Dynamic range is $\Delta = 1.0000$.
- **Origin:** Real 3D PET/CT scan from Clark et al. (The Cancer Imaging Archive / TCIA), downsampled by the paper authors from original $256 \times 256 \times 56$ to comply with supplementary material distribution limits.

### 3.2 Decomposition Architecture
Figure 12 employs a **Rank-1000 3-Way CP (PARAFAC) tensor decomposition**:
$$\hat{\mathbf{X}}(x, y, z) = \sum_{r=1}^{1000} \lambda_r \, U_r(x) \, V_r(y) \, W_r(z)$$
- **Factor $U(x)$:** Mode 1 (height $H=128$), generated by 1D Deep Image Prior Skip network (`skip1d`, 128 channels, 5 scales).
- **Factor $V(y)$:** Mode 2 (width $W=128$), generated by 1D Deep Image Prior Skip network (`skip1d`, 128 channels, 5 scales).
- **Factor $W(z)$:** Mode 3 (axial slice $T=56$), generated by the evaluated architecture (`baseline`, `hybrid`, `cnn_control`, `pointwise_control`, `transformer`).
- **Core Weights $\lambda$:** Learnable diagonal core vector $\lambda \in \mathbb{R}^{1000}$ initialized to $1/R = 1/1000$.
- **Positional Encoding:** All three factor networks receive fixed 1D sinusoidal positional encoding inputs:
  $$\mathbf{z}_U \in \mathbb{R}^{1 \times 500 \times 128}, \quad \mathbf{z}_V \in \mathbb{R}^{1 \times 500 \times 128}, \quad \mathbf{z}_W \in \mathbb{R}^{1 \times 500 \times 56}$$

### 3.3 Forward Radon Projection Operator
- **Tomographic Angles:** 40 angles uniformly spaced in $[0^\circ, 180^\circ]$:
  $$\theta_k = \frac{180^\circ \cdot k}{40}, \quad k = 0, \dots, 39$$
- **Slice-Wise Parallelism:** The 2D Radon transform is applied slice-by-slice across all 56 axial slices:
  $$\mathbf{Y}(k, z, s) = \mathcal{R}_{\theta_k}\left[\mathbf{X}(:, :, z)\right](s), \quad \mathbf{Y} \in \mathbb{R}^{40 \times 56 \times 128}$$
- **Detector Noise Model:** In physical computed tomography, electronic readout noise and photon Poisson shot noise corrupt the **detector sinogram**, not the unobserved patient volume:
  $$\mathbf{Y}_{\text{meas}} = \text{measure}(\mathcal{R}_{40}(\mathbf{X}), \text{noise\_snr}=2.0, \tau=100.0)$$
- **Unsupervised Optimization Objective:**
  $$\mathcal{L}_{\text{meas}}(\hat{\mathbf{X}}) = \frac{1}{2} \|\mathbf{Y}_{\text{meas}} - \mathcal{R}_{40}(\hat{\mathbf{X}})\|_2^2 = \text{losses.L2Norm}(\mathbf{Y}_{\text{meas}} - \hat{\mathbf{Y}})$$

---

## 4. Geometric & Architectural Suitability of Mode 3 ($W(z)$)

| Dimension Property | Figure 10 (Yale Face B) | Figure 11 (Video CS) | Figure 12 (PET/CT Radon CS) |
|:---|:---|:---|:---|
| **Mode 3 Dimension** | 160 discrete illumination angles | 64 continuous time frames | **56 continuous vertical axial slices ($z$-axis)** |
| **Ordering Property** | Lexicographical filenames (non-monotonic angles) | Strictly chronological ($t = 0 \dots 63$) | **Strictly continuous cranio-caudal depth ($z = 0 \dots 55$)** |
| **Local Continuity** | Discontinuous (elevation jumps up to $110^\circ$) | Smooth temporal motion | **Smooth anatomical cross-sections (2.5–5 mm spacing)** |
| **Non-Local Structure** | Consistent lighting across subjects | Cyclical motion / recurring dynamics | **Vertical anatomical organ continuity (spine, aorta, ribcage)** |
| **1D Positional Encoding** | Imposed artificial sequence-distance prior | Encoded physical time progression | **Physically faithful:** encodes vertical patient depth |
| **Attention Hypothesis** | Penalized matching lighting across subjects | Caused temporal cross-talk during demultiplexing | **Enables non-local anatomical feature communication across axial slices** |

**Scientific Hypothesis for Figure 12:** In PET/CT tomography, 40-angle projection undersampling causes severe streak artifacts that corrupt individual 2D slices. Because anatomical structures (such as the spine, aorta, and chest wall) extend continuously across dozens of vertical slices, global slice-to-slice self-attention along $W(z)$ is structurally positioned to aggregate consistent boundary evidence across slices to suppress 2D streak artifacts. Evaluating the parameter-matched controls will reveal whether global attention or local convolutions provide better cross-slice regularization.

---

## 5. Parameter-Matching Audit

All five variants share identical factor generators $U(x)$ ($1,878,740$ params) and $V(y)$ ($1,878,740$ params), plus the learnable rank core ($1,000$ params), totaling $3,758,480$ common parameters. Architectural variations occur exclusively in $W(z)$ for the axial slice dimension ($T=56$):

| Model Variant | Common $U, V, \lambda$ Params | Mode 3 ($W$) Params | Total Trainable Params | $\Delta$ vs. Hybrid | Relative Match | Match Classification |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`hybrid`** | 3,758,480 | 292,508 | **4,050,988** | 0 | 0.0000% | **Exact Reference** |
| **`cnn_control`** | 3,758,480 | 292,680 | **4,051,160** | +172 | **+0.0042%** | **Parameter-Matched ($\pm 0.005\%$)** |
| **`pointwise_control`** | 3,758,480 | 292,380 | **4,050,860** | -128 | **-0.0032%** | **Parameter-Matched ($\pm 0.005\%$)** |
| **`baseline`** | 3,758,480 | 160,028 | **3,918,508** | -132,480 | -3.2703% | Original Paper Baseline |
| **`transformer`** | 3,758,480 | 104,664 | **3,663,144** | -387,844 | -9.5741% | Pure Pre-LN Transformer with ALiBi |

Both `cnn_control` and `pointwise_control` are matched to the Hybrid model within **$\pm 0.005\%$**, isolating local convolutions vs pointwise mixing vs global self-attention.

---

## 6. One-Epoch Smoke Test & Verification

A 1-epoch dry run was executed on CUDA via [`experiments/hybrid_prototypes/run_hybrid_figure12_pet.py`](file:///e:/Rakhal_Tensor/experiments/hybrid_prototypes/run_hybrid_figure12_pet.py):

```
=== Figure 12 3D PET/CT Radon CS Suite on cuda (Epochs: 1) ===
[baseline          ]: Params=3,918,508 | Meas L2 Loss=3.11099e+02 | Time=0.54s | VRAM=553.6MB
[hybrid            ]: Params=4,050,988 | Meas L2 Loss=3.10507e+02 | Time=0.22s | VRAM=562.6MB
[cnn_control       ]: Params=4,051,160 | Meas L2 Loss=3.11018e+02 | Time=0.19s | VRAM=562.4MB
[pointwise_control ]: Params=4,050,860 | Meas L2 Loss=3.11009e+02 | Time=0.20s | VRAM=562.6MB
[transformer       ]: Params=3,663,144 | Meas L2 Loss=3.07035e+02 | Time=0.16s | VRAM=561.3MB
```

### Smoke Test Output Verification:
- **Checkpoints saved:** [`results/hybrid/figure12_pet_ct/`](file:///e:/Rakhal_Tensor/results/hybrid/figure12_pet_ct/) (`baseline`, `hybrid`, `cnn_control`, `pointwise_control`, `transformer`).
- **Telemetry files:** [`figure12_pet_ct_comparison.csv`](file:///e:/Rakhal_Tensor/results/hybrid/figure12_pet_ct/figure12_pet_ct_comparison.csv), [`figure12_pet_ct_comparison.json`](file:///e:/Rakhal_Tensor/results/hybrid/figure12_pet_ct/figure12_pet_ct_comparison.json), [`experiment_metadata.json`](file:///e:/Rakhal_Tensor/results/hybrid/figure12_pet_ct/experiment_metadata.json).
- **Visuals generated:**
  - Reconstruction Comparison Panel (Slice 28): [`figure12_pet_ct_comparison_panel.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure12_pet_ct/figure12_pet_ct_comparison_panel.png)
  - Error Difference Map: [`figure12_pet_ct_diff_map.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure12_pet_ct/figure12_pet_ct_diff_map.png)
  - Axial Slice Progression Strip (Slices 7, 14, 21, 28, 35, 42, 49): [`figure12_pet_ct_slice_strip.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure12_pet_ct/figure12_pet_ct_slice_strip.png)
- **Independent Verification:** [`tools/verify_figure12_metrics.py`](file:///e:/Rakhal_Tensor/tools/verify_figure12_metrics.py) successfully loaded all five checkpoints from disk and verified metrics with **0.0000 dB discrepancy**.

---

## 7. Expected Runtime & Resource Plan (4,000 Epochs)

Empirical throughput measured over 30 iterations on CUDA:

| Model Variant | Throughput (ms/epoch) | Estimated Runtime (4,000 Epochs) | Peak VRAM Footprint | VRAM Margin (of 8,188 MB) |
|:---|:---:|:---:|:---:|:---:|
| **`transformer`** | 83.62 ms | 334.5 s (~5.6 min) | 577.4 MB | 92.9% free |
| **`pointwise_control`** | 102.48 ms | 409.9 s (~6.8 min) | 577.4 MB | 92.9% free |
| **`hybrid`** | 103.55 ms | 414.2 s (~6.9 min) | 577.2 MB | 92.9% free |
| **`cnn_control`** | 105.06 ms | 420.2 s (~7.0 min) | 577.2 MB | 92.9% free |
| **`baseline`** | 105.32 ms | 421.3 s (~7.0 min) | 575.8 MB | 93.0% free |
| **Total Suite** | — | **2,000.1 s (~33.3 min)** | **~577 MB** | **Safe execution** |

---

## 8. Proposed Execution Command

Upon user approval, the full experiment will be launched with:

```powershell
.\.venv\Scripts\python.exe experiments/hybrid_prototypes/run_hybrid_figure12_pet.py --variant all --epochs 4000
```
Followed immediately by:
```powershell
.\.venv\Scripts\python.exe tools/verify_figure12_metrics.py
```
And comprehensive deliverable report in `reports/extensions/PET_CT_HYBRID_EXPERIMENT_REPORT.md`.

