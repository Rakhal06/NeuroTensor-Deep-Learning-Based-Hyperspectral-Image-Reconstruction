# DeepTensor++ Adaptive Hybrid DeepTensor v1 Smoke Test Report
**Four-Application Protocol Smoke Testing & Pipeline Integration Audit**

**Date:** 2026-11-08  
**Scope:** One-Epoch End-to-End Pipeline Validation across All Four Modalities  
**Target Architecture:** Adaptive Hybrid DeepTensor v1 (AHD-Net)  
**Execution Script:** `experiments/smoke_tests/run_ahd_net_smoke_tests.py`  
**Output Directory:** `results/extensions/adaptive_hybrid_v1_smoke/`  
**Summary Data:** `results/extensions/adaptive_hybrid_v1_smoke/smoke_test_summary.json`  
**Target Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.1, PyTorch 2.5.1+cu121)  
**Status:** Smoke Testing & Independent Verification Complete (Awaiting User Review)  

---

## 1. Executive Summary

In accordance with user authorization, **Phase 3 (Four-Application Protocol Smoke Testing)** has been executed. The Adaptive Hybrid DeepTensor v1 architecture (AHD-Net) was embedded directly into the production reconstruction pipelines for Figures 09, 10, 11, and 12 under their original physics forward operators, noise models, decomposition ranks, and observation seeds.

### Key Verification Milestones:
1. **Pipeline Integration (4/4 Applications):** Model construction, forward sensing operator execution, measurement loss calculation, backpropagation, and latent optimization executed without errors across all 4 applications.
2. **Exact Parameter Parity ($0\text{ Parameter Delta}$):**
   - Figure 09 (ICVL HSI): **$3,137,808$ params** (Target: $3,137,808$, Delta: **$0$**, **$0.0000\%$**).
   - Figure 10 (Yale Face B): **$5,602,988$ params** (Target: $5,602,988$, Delta: **$0$**, **$0.0000\%$**).
   - Figure 11 (Video CS): **$834,632$ params** (Target: $834,632$, Delta: **$0$**, **$0.0000\%$**).
   - Figure 12 (PET/CT Radon): **$4,050,988$ params** (Target: $4,050,988$, Delta: **$0$**, **$0.0000\%$**).
3. **Router & Gradient Health:**
   - **Zero Branch Starvation:** All three branches (CNN, Pointwise, Attention) and the router received non-zero, finite gradients across all four applications.
   - **Zero Router Collapse:** Gating entropy remained near maximal exploration ($H \approx \ln 3 \approx 1.0986$), with routing weights smoothly balanced around $1/3$ and individual channels demonstrating distinct specialization spreads ($\text{std} \approx 0.04 - 0.06$).
4. **Computational Efficiency:** Total single-epoch runtime was under $1.2\text{s}$ per application, with peak VRAM remaining below $850\text{MB}$ ($118\text{MB} - 844\text{MB}$), confirming feasibility for full-scale execution.
5. **Checkpoint Integrity:** All four checkpoints were serialized to disk and independently verified through fresh instantiation and evaluation forward passes with zero numerical discrepancy.

---

## 2. Experimental Protocol & Preservation Safeguards

### 2.1 Preserved Experimental Constants
All application protocols were preserved identically from historical DeepTensor++ benchmarks:

```
Application Protocol Verification Matrix:
┌───────────┬───────────────────────┬──────────────┬──────────────┬─────────────────────────────┬───────────────────────────┐
│ App       │ Modality              │ Tensor Shape │ Rank         │ Physical Sensing / Noise    │ Unsupervised Loss         │
├───────────┼───────────────────────┼──────────────┼──────────────┼─────────────────────────────┼───────────────────────────┤
│ Figure 09 │ ICVL Hyperspectral    │ 128x128x260  │ R=20         │ Poisson (t=100) + SNR=2     │ L1 Measurement Norm       │
│ Figure 10 │ Yale Face B           │ 192x168x160  │ R=2000 (CP)  │ 30% Salt & Pepper Impulse   │ L1 Measurement Norm       │
│ Figure 11 │ Video CS (CACTI)      │ 128x128x64   │ R=32         │ 8-Frame Binary Mask Multip. │ L2 Measurement Sinogram   │
│ Figure 12 │ 3D PET/CT Radon CS    │ 128x128x56   │ R=1000 (CP)  │ 40-Angle Radon + Noise      │ L2 Measurement Projection │
└───────────┴───────────────────────┴──────────────┴──────────────┴─────────────────────────────┴───────────────────────────┘
```

### 2.2 Preservation of Historical Assets
- Existing baseline checkpoints (`results/baseline/`), improved checkpoints (`results/improved/`), and historical hybrid checkpoints (`results/hybrid/`) were left completely untouched.
- All smoke test artifacts (checkpoints, metrics, logs) were strictly isolated in `results/extensions/adaptive_hybrid_v1_smoke/`.

---

## 3. Application-by-Application Smoke Test Results

### 3.1 Figure 09 — ICVL Hyperspectral Denoising
- **Pipeline Architecture:** 2-Factor DeepTensor ($U(128, 128) \times V(260)$), Rank $R=20$.
- **Spectral Generator:** Skip1D body + Adaptive Tri-Branch Block ($d_{\text{model}}=128$, $c_{\text{cnn}}=143$, $c_{\text{pw}}=135$, $d_{\text{attn}}=64$) + Conv1D projection ($128 \to 20$).
- **Latents & Seeds:** `im_inp` $[1, 20, 128, 128]$ + `spec_inp` $[1, 20, 260]$ (Seed 200); Model init (Seed 100); Observation noise (Seed 42).

```
Figure 09 Diagnostics:
• Trainable Parameters : 3,137,808 (Historical Hybrid Target: 3,137,808, Delta: 0)
• 1-Epoch L1 Loss      : 0.529080 (Finite: True, NaN/Inf: None)
• Timing               : Forward: 568.1 ms | Backward: 384.8 ms | Total Epoch: 1.15 s
• Peak CUDA VRAM       : 409.0 MB
• Routing Weights      : CNN: 0.3339 | Pointwise: 0.3295 | Attention: 0.3365 (Sums to 1.00000)
• Channel Spread (std) : CNN: 0.0460 | Pointwise: 0.0413 | Attention: 0.0468
• Router Entropy       : 1.098575 (Theoretical Uniform Max: ln(3) ≈ 1.098612)
• Branch Gradient Norms:
    - ||∇_CNN L||_2     : 0.102352
    - ||∇_PW L||_2      : 0.095384
    - ||∇_Attn L||_2    : 0.227387
    - ||∇_Router L||_2  : 0.010206
• Checkpoint Saved     : results/extensions/adaptive_hybrid_v1_smoke/fig09_ahd_net_checkpoint.pth (43.06 MB)
• Independent Reload   : PASS (100% state_dict restoration, finite evaluation forward pass)
```

---

### 3.2 Figure 10 — Yale Face B Denoising
- **Pipeline Architecture:** 3-Way CP DeepTensor ($U(192) \otimes V(168) \otimes W(160)$), Rank $R=2000$.
- **Mode 3 Generator (W):** Skip1D body + Adaptive Tri-Branch Block ($d_{\text{model}}=128$, $c_{\text{cnn}}=143$, $c_{\text{pw}}=135$, $d_{\text{attn}}=64$) + Conv1D projection ($128 \to 2000$).
- **Latents & Seeds:** 1D positional latents `u_inp` $[1, 1000, 192]$, `v_inp` $[1, 1000, 168]$, `w_inp` $[1, 1000, 160]$ (Seed 200); Model init (Seed 100); 30% Salt & Pepper noise (Seed 42).

```
Figure 10 Diagnostics:
• Trainable Parameters : 5,602,988 (Historical Hybrid Target: 5,602,988, Delta: 0)
• 1-Epoch L1 Loss      : 0.364592 (Finite: True, NaN/Inf: None)
• Timing               : Forward: 182.6 ms | Backward: 181.5 ms | Total Epoch: 0.40 s
• Peak CUDA VRAM       : 844.2 MB
• Routing Weights      : CNN: 0.3345 | Pointwise: 0.3332 | Attention: 0.3323 (Sums to 1.00000)
• Channel Spread (std) : CNN: 0.0423 | Pointwise: 0.0435 | Attention: 0.0460
• Router Entropy       : 1.098609 (Theoretical Uniform Max: ln(3) ≈ 1.098612)
• Branch Gradient Norms:
    - ||∇_CNN L||_2     : 0.000456
    - ||∇_PW L||_2      : 0.000249
    - ||∇_Attn L||_2    : 0.000903
    - ||∇_Router L||_2  : 0.000048
• Checkpoint Saved     : results/extensions/adaptive_hybrid_v1_smoke/fig10_ahd_net_checkpoint.pth (76.72 MB)
• Independent Reload   : PASS (100% state_dict restoration, finite evaluation forward pass)
```

---

### 3.3 Figure 11 — Video Compressive Sensing (CACTI)
- **Pipeline Architecture:** 2-Factor DeepTensor ($U(128, 128) \times V(64)$), Rank $R=32$.
- **Spectral Generator:** Skip1D body ($c=64$) + Adaptive Tri-Branch Block ($d_{\text{model}}=64$, $c_{\text{cnn}}=113$, $c_{\text{pw}}=113$, $d_{\text{attn}}=40$) + Conv1D projection ($64 \to 32$).
- **Forward Operator:** 8-frame temporal multiplexing with random binary mask ($1, 8, 128, 128$).
- **Latents & Seeds:** `im_inp` $[1, 32, 128, 128]$ + `spec_inp` $[1, 32, 64]$ (Seed 200); Model init (Seed 100); Observation noise & masks (Seed 42).

```
Figure 11 Diagnostics:
• Trainable Parameters : 834,632 (Historical Hybrid Target: 834,632, Delta: 0)
• 1-Epoch L2 Loss      : 1.175987 (Finite: True, NaN/Inf: None)
• Timing               : Forward: 160.3 ms | Backward: 123.7 ms | Total Epoch: 0.34 s
• Peak CUDA VRAM       : 118.0 MB
• Routing Weights      : CNN: 0.3532 | Pointwise: 0.3240 | Attention: 0.3228 (Sums to 1.00000)
• Channel Spread (std) : CNN: 0.0613 | Pointwise: 0.0583 | Attention: 0.0485
• Router Entropy       : 1.097732 (Theoretical Uniform Max: ln(3) ≈ 1.098612)
• Branch Gradient Norms:
    - ||∇_CNN L||_2     : 0.864599
    - ||∇_PW L||_2      : 0.468848
    - ||∇_Attn L||_2    : 1.090545
    - ||∇_Router L||_2  : 0.091489
• Checkpoint Saved     : results/extensions/adaptive_hybrid_v1_smoke/fig11_ahd_net_checkpoint.pth (16.00 MB)
• Independent Reload   : PASS (100% state_dict restoration, finite evaluation forward pass)
```

---

### 3.4 Figure 12 — 3D PET/CT Radon Compressive Sensing
- **Pipeline Architecture:** 3-Way CP DeepTensor ($U(128) \otimes V(128) \otimes W(56)$), Rank $R=1000$.
- **Mode 3 Generator (W):** Skip1D body ($c=128$) + Adaptive Tri-Branch Block ($d_{\text{model}}=128$, $c_{\text{cnn}}=143$, $c_{\text{pw}}=135$, $d_{\text{attn}}=64$) + Conv1D projection ($128 \to 1000$).
- **Forward Operator:** 40-angle parallel-beam Radon transform (`lin_inverse.radon`).
- **Latents & Seeds:** 1D positional latents `u_inp` $[1, 500, 128]$, `v_inp` $[1, 500, 128]$, `w_inp` $[1, 500, 56]$ (Seed 200); Model init (Seed 100); Poisson/Gaussian detector noise (Seed 42).

```
Figure 12 Diagnostics:
• Trainable Parameters : 4,050,988 (Historical Hybrid Target: 4,050,988, Delta: 0)
• 1-Epoch L2 Loss      : 310.750488 (Finite: True, NaN/Inf: None)
• Timing               : Forward: 146.7 ms | Backward: 133.2 ms | Total Epoch: 0.31 s
• Peak CUDA VRAM       : 547.4 MB
• Routing Weights      : CNN: 0.3403 | Pointwise: 0.3286 | Attention: 0.3311 (Sums to 1.00000)
• Channel Spread (std) : CNN: 0.0429 | Pointwise: 0.0410 | Attention: 0.0426
• Router Entropy       : 1.098500 (Theoretical Uniform Max: ln(3) ≈ 1.098612)
• Branch Gradient Norms:
    - ||∇_CNN L||_2     : 2.925123
    - ||∇_PW L||_2      : 1.601585
    - ||∇_Attn L||_2    : 6.752200
    - ||∇_Router L||_2  : 0.212853
• Checkpoint Saved     : results/extensions/adaptive_hybrid_v1_smoke/fig12_ahd_net_checkpoint.pth (50.58 MB)
• Independent Reload   : PASS (100% state_dict restoration, finite evaluation forward pass)
```

---

## 4. Cross-Application Synthesis & Comparative Resource Audit

```
Cross-Application Resource & Diagnostic Summary Table:
┌───────────┬──────────────┬──────────────┬──────────┬───────────┬───────────┬───────────┬──────────┬───────────┬───────────┐
│ App       │ Target Model │ Trainable    │ Delta    │ 1-Epoch   │ Fwd Time  │ Bwd Time  │ Peak     │ Router    │ Starvation│
│           │ Parameters   │ Parameters   │ (Params) │ Loss      │ (ms)      │ (ms)      │ VRAM     │ Entropy   │ Check     │
├───────────┼──────────────┼──────────────┼──────────┼───────────┼───────────┼───────────┼──────────┼───────────┼───────────┤
│ Figure 09 │  3,137,808   │  3,137,808   │    0     │ 0.529080  │ 568.1 ms  │ 384.8 ms  │ 409.0 MB │ 1.098575  │ NONE (OK) │
│ Figure 10 │  5,602,988   │  5,602,988   │    0     │ 0.364592  │ 182.6 ms  │ 181.5 ms  │ 844.2 MB │ 1.098609  │ NONE (OK) │
│ Figure 11 │    834,632   │    834,632   │    0     │ 1.175987  │ 160.3 ms  │ 123.7 ms  │ 118.0 MB │ 1.097732  │ NONE (OK) │
│ Figure 12 │  4,050,988   │  4,050,988   │    0     │ 310.7505  │ 146.7 ms  │ 133.2 ms  │ 547.4 MB │ 1.098500  │ NONE (OK) │
└───────────┴──────────────┴──────────────┴──────────┴───────────┴───────────┴───────────┴──────────┴───────────┴───────────┘
```

```
Branch Gradient Norms Across Applications:
┌───────────┬───────────────────┬───────────────────┬───────────────────┬───────────────────┐
│ App       │ CNN Branch Norm   │ PW Branch Norm    │ Attn Branch Norm  │ Router Norm       │
├───────────┼───────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ Figure 09 │ 0.102352          │ 0.095384          │ 0.227387          │ 0.010206          │
│ Figure 10 │ 0.000456          │ 0.000249          │ 0.000903          │ 0.000048          │
│ Figure 11 │ 0.864599          │ 0.468848          │ 1.090545          │ 0.091489          │
│ Figure 12 │ 2.925123          │ 1.601585          │ 6.752200          │ 0.212853          │
└───────────┴───────────────────┴───────────────────┴───────────────────┴───────────────────┘
```

> [!NOTE]
> In all four modalities, the Attention branch displays a slightly higher gradient norm than CNN or PW, reflecting the sensitivity of multi-head softmax scaling. However, all three branches remain within 1 order of magnitude of each other, confirming that the cosine temperature starting value ($\tau=1.0$ or $2.0$) provides balanced backpropagation without starving any sub-module.

---

## 5. Artifacts and Generated Deliverables

The following files have been saved to disk and independently verified:
- **Smoke Test Summary JSON:** [`results/extensions/adaptive_hybrid_v1_smoke/smoke_test_summary.json`](file:///e:/Rakhal_Tensor/results/extensions/adaptive_hybrid_v1_smoke/smoke_test_summary.json)
- **Figure 09 Checkpoint:** [`results/extensions/adaptive_hybrid_v1_smoke/fig09_ahd_net_checkpoint.pth`](file:///e:/Rakhal_Tensor/results/extensions/adaptive_hybrid_v1_smoke/fig09_ahd_net_checkpoint.pth) ($43.06\text{MB}$)
- **Figure 10 Checkpoint:** [`results/extensions/adaptive_hybrid_v1_smoke/fig10_ahd_net_checkpoint.pth`](file:///e:/Rakhal_Tensor/results/extensions/adaptive_hybrid_v1_smoke/fig10_ahd_net_checkpoint.pth) ($76.72\text{MB}$)
- **Figure 11 Checkpoint:** [`results/extensions/adaptive_hybrid_v1_smoke/fig11_ahd_net_checkpoint.pth`](file:///e:/Rakhal_Tensor/results/extensions/adaptive_hybrid_v1_smoke/fig11_ahd_net_checkpoint.pth) ($16.00\text{MB}$)
- **Figure 12 Checkpoint:** [`results/extensions/adaptive_hybrid_v1_smoke/fig12_ahd_net_checkpoint.pth`](file:///e:/Rakhal_Tensor/results/extensions/adaptive_hybrid_v1_smoke/fig12_ahd_net_checkpoint.pth) ($50.58\text{MB}$)
- **Runner Script:** [`experiments/smoke_tests/run_ahd_net_smoke_tests.py`](file:///e:/Rakhal_Tensor/experiments/smoke_tests/run_ahd_net_smoke_tests.py)

---

## 6. Adherence to Strict Stop Rule

In compliance with the governing instructions:
1. **No full-scale training was launched.** Only 1 epoch per application was executed for pipeline verification.
2. **No hyperparameter tuning was performed.** Learning rates, ranks, noise stds, and sensing parameters were kept strictly identical to historical runs.
3. **No performance improvement is claimed.** One-epoch metrics demonstrate only numerical stability and gradient health; scientific evaluation awaits full-budget execution.

**STOPPED.** Awaiting user review and formal authorization to proceed to full-scale training or systematic ablations.

