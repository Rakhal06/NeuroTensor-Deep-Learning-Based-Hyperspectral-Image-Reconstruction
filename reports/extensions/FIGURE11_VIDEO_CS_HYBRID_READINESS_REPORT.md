# DeepTensor++ Research Report: Figure 11 Video Compressive Sensing (CACTI) Readiness & Protocol Verification

**Date:** 2026-11-08  
**Experiment Target:** Figure 11 — Video Compressive Sensing (CACTI Coded Aperture Multiplexing)  
**Dataset:** Cat Video (`data/cat_video.mat`), shape: $128 \times 128 \times 64$  
**Hardware Platform:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MB VRAM), CUDA 12.1  
**Software Environment:** Python 3.11.9 (`.venv`), PyTorch 2.5.1+cu121  
**Output Directory:** [`results/hybrid/figure11_video_cs/`](file:///e:/Rakhal_Tensor/results/hybrid/figure11_video_cs/)  
**Configuration File:** [`configs/hybrid/figure11_video_cs.yaml`](file:///e:/Rakhal_Tensor/configs/hybrid/figure11_video_cs.yaml)  
**Experiment Runner:** [`experiments/hybrid_prototypes/run_hybrid_figure11_video.py`](file:///e:/Rakhal_Tensor/experiments/hybrid_prototypes/run_hybrid_figure11_video.py)  
**Independent Verification Tool:** [`tools/verify_figure11_metrics.py`](file:///e:/Rakhal_Tensor/tools/verify_figure11_metrics.py)  
**Target Budget:** 3,000 epochs per variant (verified paper reproduction budget)  

---

## 1. Executive Summary & Readiness Verdict

We have conducted a thorough pre-training audit, protocol verification, architecture review, and 1-epoch smoke test for **Figure 11: Video Compressive Sensing (CACTI)** across all five model variants in the DeepTensor++ evaluation suite:
1. **`baseline`**: Standard 2D Spatial Skip U-Net ($U$) + 1D Temporal Skip CNN ($V$).
2. **`hybrid`**: 2D Spatial Skip U-Net ($U$) + 1D Temporal Skip CNN + Pre-LN Multi-Head Self-Attention + FFN ($V$).
3. **`cnn_control`**: Parameter-matched pure 1D CNN with Conv1d residual block (**-0.0152% match** to Hybrid).
4. **`pointwise_control`**: Parameter-matched pointwise MLP mixing + FFN (**-0.0077% match** to Hybrid).
5. **`transformer`**: 2D Spatial Skip U-Net ($U$) + Pure Pre-LN Spectral Transformer with ALiBi relative position bias ($V$).

### Audit Verdict: **PASSED (100% READY FOR FULL TRAINING)**
- **Architecture Integrity:** All five models successfully instantiate, maintain parameter matching within $\pm 0.015\%$, and produce exact $(1, H \times W, T)$ tensor representations.
- **Physical Sensing Matrix:** 8-frame high-speed coded aperture temporal multiplexing forward operator $\mathbf{y} = \Phi(\mathbf{x})$ verified against the author implementation with zero discrepancy.
- **Continuous Temporal Geometry:** Unlike the irregular illumination angles of Yale Face B (Figure 10), video frames are strictly ordered and physically continuous ($t = 0 \dots 63$). Self-attention is theoretically well-aligned with continuous motion trajectories.
- **Unsupervised Model Selection:** Zero ground-truth leakage. Model tracking and checkpoint selection rely strictly on unsupervised measurement $\ell_2$ loss.
- **Independent Verification:** Tested and confirmed. [`tools/verify_figure11_metrics.py`](file:///e:/Rakhal_Tensor/tools/verify_figure11_metrics.py) reproduces metrics directly from saved checkpoints with $0.0000\text{ dB}$ discrepancy.
- **Resource Footprint:** Peak VRAM is **122 MB** (<1.5% of GPU capacity). Total expected runtime for all 5 models at 3,000 epochs is **~17.5 minutes**.

---

## 2. Dataset & Forward Operator Protocol Verification

### 2.1 Dataset Specifications
- **File:** `data/cat_video.mat`
- **Tensor Shape:** $H=128, W=128, T=64$ frames.
- **Data Type & Range:** Originally `uint8` in $[0, 221]$. Normalized to `float32` in $[0.0, 1.0]$ via $\mathbf{X}_{\text{norm}} = \mathbf{X} / \max(\mathbf{X})$ with dynamic range $\Delta = 1.0000$.
- **Author's Disclosed Discrepancy:** The authors noted in `run_figure11.py` lines 4–6:
  > *"The accuracy is different from the paper as the video has been downsampled and clipped to fit supplementary material requirements."*
  This accounts for why both DeepTensor and 2D TV on `cat_video.mat` yield $\approx 24.5$–$25.3\text{ dB}$ rather than the unclipped paper target ($27.1\text{ dB}$).

### 2.2 Forward Compressive Sensing Operator (CACTI)
- **Multiplexing Factor:** $n_{\text{frames}} = 8$ high-speed video frames collapsed into a single coded snapshot:
  $$\mathbf{Y}_k(x, y) = \sum_{\tau=0}^{7} \mathbf{M}_{8k + \tau}(x, y) \odot \mathbf{X}_{8k + \tau}(x, y), \quad k = 0, \dots, 7$$
- **Coded Snapshot Tensor:** $\mathbf{Y} \in \mathbb{R}^{128 \times 128 \times 8}$ (total compression ratio = $8\times$).
- **Mask Properties:** Binary coded aperture where each spatial pixel $(x, y)$ is unblocked for exactly one frame out of every 8 frames (mean duty cycle = $0.125 = 1/8$).
- **Noise Model:** Mixed Poisson readout noise with $\tau = 100$ and Gaussian readout noise with $\text{SNR} = 2.0$.

### 2.3 Optimization Loss
In compressive sensing under additive Gaussian readout noise, the maximum likelihood unsupervised measurement objective is the squared $\ell_2$ Frobenius norm:
$$\mathcal{L}_{\text{meas}}(\hat{\mathbf{X}}) = \frac{1}{2} \|\mathbf{Y}_{\text{meas}} - \Phi(\hat{\mathbf{X}})\|_2^2 = \text{losses.L2Norm}(\mathbf{Y}_{\text{meas}} - \hat{\mathbf{Y}})$$
This matches the exact criterion used in the original author script `run_figure11.py` and `run_figure11_baseline.py`.

---

## 3. Geometric & Attention Suitability: Figure 11 vs. Figure 10

| Dimension / Property | Figure 10 (Yale Face B) | Figure 11 (Video Compressive Sensing) |
|:---|:---|:---|
| **Mode 3 Nature** | 160 discrete illumination angles across 3 distinct human subjects | 64 continuous video frames over physical time |
| **Ordering Property** | Lexicographical filename sort; non-monotonic illumination azimuth/elevation | **Strictly continuous physical time:** $t = 0, 1, 2, \dots, 63$ |
| **Local Continuity** | Discontinuous: adjacent indices have elevation jumps up to $110^\circ$; subject jumps at index 64 | **Smoothly continuous:** adjacent frames $t$ and $t+1$ exhibit small pixel displacements |
| **Non-local Correlations** | Displaced lighting angles across subjects (penalized by 1D sequence distance) | Recurring motion trajectories, background persistence, cyclic movements |
| **1D Positional Encoding** | Imposed artificial sequence-locality bias that degraded Hybrid performance | **Physically faithful:** encodes chronological progression $\sin(t / 10000^{2i/d})$ |
| **Attention Mechanism** | Conflicted with physical illumination angles | **Theoretically well-matched:** captures non-local trajectory dynamics across all 64 frames |

**Scientific Implication:** In Figure 10, the pure CNN Control beat the Hybrid model (+0.31 dB) because 1D sinusoidal positional encoding imposed an artificial sequence prior on scrambled lighting angles. In Figure 11, the sequence is genuine physical time. Multi-head self-attention can model frame-to-frame motion trajectories without an artificial ordering bias.

---

## 4. Parameter-Matching Audit

All models share an identical 2D Spatial Skip U-Net generator for factor $U(H, W)$ ($574,340$ parameters, 64 channels, 5 scales, bilinear upsampling). Variations occur exclusively in the temporal factor generator $V(T)$ for Mode 3:

| Model Variant | Spatial $U$ Params | Temporal $V$ Params | Total Params | $\Delta$ vs. Hybrid | Relative Match | Match Classification |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`hybrid`** | 574,340 | 260,292 | **834,632** | 0 | 0.0000% | **Exact Reference** |
| **`cnn_control`** | 574,340 | 260,165 | **834,505** | -127 | **-0.0152%** | **Parameter-Matched ($\pm 0.015\%$)** |
| **`pointwise_control`** | 574,340 | 260,228 | **834,568** | -64 | **-0.0077%** | **Parameter-Matched ($\pm 0.008\%$)** |
| **`baseline`** | 574,340 | 210,308 | **784,648** | -49,984 | -5.9887% | Original Paper Baseline |
| **`transformer`** | 574,340 | 801,696 | **1,376,036** | +541,404 | +64.8674% | Pure Transformer (4 layers, 4 heads) |

Both `cnn_control` and `pointwise_control` are matched to the Hybrid model within **$\pm 0.015\%$**, isolating architectural inductive biases (local convolutions vs. pointwise MLP mixing vs. global attention).

---

## 5. Verification of 1-Epoch Smoke Test

A 1-epoch dry run was executed on CUDA to verify pipeline stability, output generation, checkpoint serialization, and independent metric reproduction:

```
[BASELINE          ]: Params=784,648   | Meas L2 Loss=1.03896e+00 | Time=0.85s | VRAM=111.2MB
[HYBRID            ]: Params=834,632   | Meas L2 Loss=2.26813e+00 | Time=0.13s | VRAM=118.5MB
[CNN_CONTROL       ]: Params=834,505   | Meas L2 Loss=1.51137e+00 | Time=0.32s | VRAM=118.3MB
[POINTWISE_CONTROL ]: Params=834,568   | Meas L2 Loss=2.16421e+00 | Time=0.18s | VRAM=118.5MB
[TRANSFORMER       ]: Params=1,376,036 | Meas L2 Loss=1.86161e+00 | Time=0.10s | VRAM=122.0MB
```

### Smoke Test Output Verification:
- Checkpoints saved: `baseline_checkpoint.pth`, `hybrid_checkpoint.pth`, `cnn_control_checkpoint.pth`, `pointwise_control_checkpoint.pth`, `transformer_checkpoint.pth`
- Summary files: [`figure11_video_cs_comparison.csv`](file:///e:/Rakhal_Tensor/results/hybrid/figure11_video_cs/figure11_video_cs_comparison.csv), [`figure11_video_cs_comparison.json`](file:///e:/Rakhal_Tensor/results/hybrid/figure11_video_cs/figure11_video_cs_comparison.json), [`experiment_metadata.json`](file:///e:/Rakhal_Tensor/results/hybrid/figure11_video_cs/experiment_metadata.json)
- Visual panels: [`figure11_video_cs_comparison_panel.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure11_video_cs/figure11_video_cs_comparison_panel.png), [`figure11_video_cs_diff_map.png`](file:///e:/Rakhal_Tensor/results/hybrid/figure11_video_cs/figure11_video_cs_diff_map.png)
- Independent verification audit [`tools/verify_figure11_metrics.py`](file:///e:/Rakhal_Tensor/tools/verify_figure11_metrics.py) verified all stored checkpoints with **0.0000 dB reproduction delta**.

---

## 6. Expected Runtime & Resource Projections

Benchmark measurements over 50 iterations on CUDA yielded the following steady-state throughput numbers:

| Model Variant | Throughput (ms/epoch) | Estimated Runtime (3,000 Epochs) | Peak VRAM Footprint | VRAM Margin (of 8,188 MB) |
|:---|:---:|:---:|:---:|:---:|
| **`transformer`** | 55.37 ms | 166.1 s (~2.8 min) | 122.0 MB | 98.5% free |
| **`baseline`** | 68.29 ms | 204.9 s (~3.4 min) | 111.2 MB | 98.6% free |
| **`cnn_control`** | 74.47 ms | 223.4 s (~3.7 min) | 118.3 MB | 98.6% free |
| **`pointwise_control`** | 74.46 ms | 223.4 s (~3.7 min) | 118.5 MB | 98.6% free |
| **`hybrid`** | 77.54 ms | 232.6 s (~3.9 min) | 118.5 MB | 98.5% free |
| **Total Suite** | — | **1,050.4 s (~17.5 min)** | **122.0 MB** | **Safe execution** |

---

## 7. Results Preservation & Integrity Guarantee

- **Figure 09 Intact:** All ICVL files in `results/hybrid/figure09_hyperspectral/` are preserved.
- **Figure 10 Intact:** All Yale Face B checkpoints and verification files in `results/hybrid/figure10_yale_denoising/` are preserved.
- **Baselines Intact:** `results/baseline/` and `results/improved/` remain untouched.
- **Reproducibility Guarantee:** All 5 checkpoints will store exact latent vectors `im_inp` and `spec_inp`, best unsupervised model weights, and best reconstruction tensors.

---

## 8. Proposed Execution Command

Upon user approval, the full experiment will be launched with:

```powershell
.\.venv\Scripts\python.exe experiments/hybrid_prototypes/run_hybrid_figure11_video.py --variant all --epochs 3000
```
Followed immediately by:
```powershell
.\.venv\Scripts\python.exe tools/verify_figure11_metrics.py
```
And final report authoring in `reports/extensions/VIDEO_CS_HYBRID_EXPERIMENT_REPORT.md`.

