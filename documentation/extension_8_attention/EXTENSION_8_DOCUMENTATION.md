# Extension 8: Hybrid CNN–Transformer / Spectral Self-Attention

## Objective
The goal of this extension is to insert a lightweight spectral self-attention block into the DeepTensor spectral branch to model global long-range dependencies across the spectral dimension, which the standard 1D CNN baseline might struggle to capture fully.

## Architecture Discovery
The DeepTensor architecture was inspected to determine the exact structure of the `spec_net` output and data flow.
- The `spec_net` is created via `deep_prior.get_net('skip1d', ...)`.
- It consists of a 1D encoder-decoder (U-Net) with skip connections.
- The actual returned module is a PyTorch `Sequential`.
- The final layer is a projection: `Conv1d(128, 20, kernel_size=(1,), stride=(1,))`.
- The shape of the tensor immediately prior to this final projection is `[1, 128, 260]`.

### Branch A Confirmation
Because the tensor immediately prior to the final 1×1 projection is `[1, 128, 260]` (batch=1, channels=128, sequence/bands=260), this represents a genuine per-band representation. **Branch A** is confirmed and utilized.

### Insertion Point
The attention mechanism is inserted directly between the main U-Net body and the final 1×1 `Conv1d` projection. The computational graph is decomposed safely without modifying frozen baseline files. 

## Attention Configuration
- `seq_len`: 260
- `d_model`: 128
- `heads`: 2
- `layers`: 1
- `positional encoding`: Fixed sinusoidal, registered as a non-trainable buffer.
- `normalization`: Pre-norm (`LayerNorm`).
- `FFN`: 128 → 256 → 128 with GELU activation.
- `residual connections`: Standard Post-Attention and Post-FFN additions.

## Control Configuration
The control network uses the identical outer structure (Pre-norm, Residuals, FFN) but replaces the sequence-mixing Multi-Head Self-Attention with a pointwise MLP mapping 128 → 256 → 128. This isolates the effect of increased parameter count/depth from actual cross-band communication.

## Parameter Counts
Parameter matching was enforced as a hard requirement (tolerance ≤ 10%).

- `im_net` (all variants): 2,206,152
- `spec_net` baseline: 799,176
- `spec_net` hybrid: 931,656
- `spec_net` control: 931,528

Total Hybrid Params: 3,137,808  
Total Control Params: 3,137,680  
**Absolute Difference:** 128 parameters  
**Percentage Difference:** 0.004%  
*(Requirement of ≤ 10% successfully met).*

## Computation
- **Hybrid extra FLOPs estimated:** 52,422,240
- **Control extra FLOPs estimated:** 34,977,280
- **Smoke-Test Max VRAM:** 643.93 MB (Well within the safe 6GB target budget).

## RNG Protocol
To ensure fairness, deterministic initialization was enforced across all variants despite differences in module parameters:
1. Global seed 42 set for initial data loading and noisy observation creation.
2. `im_net` seed set explicitly to 100 before module instantiation.
3. `spec_net` seed set explicitly to 200 before module instantiation.

This guarantees that `im_net` weights and the exact noise realization are completely identical for baseline, hybrid, and control experiments.

## Validation and Smoke Tests

### Baseline Observation Check
- Target PSNR: ≈ 28.43 dB
- Actual Input PSNR: **28.4321 dB**
*(Observation generation successfully matched the frozen baseline).*

### Smoke Test Status
- **Shape Checks:** Passed. Attention correctly handles `[1, 128, 260]` and returns `[1, 128, 260]`, preserving downstream shapes.
- **Finite-Loss Checks:** Passed. Initial losses were finite (e.g., `0.1456`). No NaN/Inf detected.
- **Attention-Weight Checks:** Passed. Extracted weights during hybrid forward pass had shape `(1, 260, 260)` with valid, finite values and no NaN/Inf.
- **BatchNorm Train-Mode Check:** Passed. Verified recursively that all `BatchNorm1d` layers in the wrapped `HybridSpectralNet` properly inherit `.train()` mode during the training loop.

## Full Experiment Results (Phase B)

The full 5000-epoch experiments were run using strict RNG fairness, preserving the noisy observation (`input PSNR ≈ 28.43 dB`) and initial image network weights exactly.

### Comparison Table

| Metric | Baseline (Frozen) | Hybrid (Attention) | Control (Pointwise MLP) |
|---|---|---|---|
| **PSNR (dB)** | 46.6277 | 45.3893 | 45.7180 |
| **SSIM** | 0.9875 | 0.9841 | 0.9851 |
| **RMSE** | N/A (Sanity Check) | 0.005377 | 0.005177 |
| **Runtime (s)** | ~150-160 | 154.64 | 152.89 |
| **Peak VRAM (MB)** | ~640.37 | 643.67 | 643.67 |
| **Total Params** | 3,005,328 | 3,137,808 | 3,137,680 |

### Result Differentials
- **Hybrid − Baseline PSNR:** -1.2384 dB
- **Hybrid − Control PSNR:** -0.3287 dB
- **Control − Baseline PSNR:** -0.9097 dB

### Post-Training Attention Diagnostic
A dedicated post-training forward pass extracted the attention weights from the Hybrid network:
- **Shape:** `(1, 260, 260)`
- **Finite Check:** Passed (no NaN or Inf).
- **Qualitative Observation:** The attention weights are highly diffuse and nearly uniform. The trace ratio (sum of diagonal elements divided by the sum of all elements) is `0.003969`, which is practically identical to the uniform distribution expectation (`1/260 = 0.003846`). The maximum attention weight across any token pair is `0.022`. The network did not learn a sparse, local, or highly structured attention pattern, but rather averages broadly across the spectral dimension.

### Scientific Interpretation
The results show that introducing self-attention into the spectral network actually **degrades performance** relative to the 1D CNN baseline (-1.23 dB). Furthermore, it degrades performance slightly more than the parameter-matched pointwise control (-0.32 dB). 

In the Deep Image Prior / untrained network framework, the inductive bias of the architecture is what guides the reconstruction. The diffuse, uniform attention weights indicate that the self-attention layer failed to provide a useful structural prior for the spectral low-rank factors. The additional parameters introduced by the Hybrid and Control models degraded the reconstruction quality compared to the strict constraint of the baseline U-Net, and the specific sequence-mixing mechanism of self-attention did not offer a compensatory beneficial inductive bias for this task.
