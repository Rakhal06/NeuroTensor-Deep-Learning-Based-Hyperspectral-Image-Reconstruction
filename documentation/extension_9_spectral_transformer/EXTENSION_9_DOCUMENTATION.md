# Extension 9: Transformer-Dominant Spectral Generator

## Motivation and Research Question
Test whether a Transformer-dominant spectral generator can outperform the frozen DeepTensor CNN spectral generator under a strict parameter-matched, latent-identical comparison. The goal is to determine if global sequence-modeling with ALiBi provides a stronger structural prior than local convolutions for hyperspectral factor generation.

## Implementation Details
- **Architecture**: 4-layer Transformer Encoder, `d_model=128`, 4 heads, Pre-LN, GELU, no dropout. Fixed sinusoidal PE and fixed non-trainable ALiBi position bias.
- **Control**: Parameter-matched network where MHSA and ALiBi are replaced with an independent per-token linear operation (`Linear(128, 256)` -> `Linear(256, 128)` with no activation).
- **RNG Fairness**: `spec_inp` tensor generation is strictly placed before the spectral model construction to avoid different initialization parameter amounts consuming the RNG sequence and altering the latent `spec_inp`.
- **Latent-identical Fairness**: The explicit RNG ordering verifies that all variants (Baseline, Transformer, Control) consume the exact identical literal `spec_inp` tensor sum of 258.771820.
- **Training Protocol**: ICVL, Rank-20, mixed Poisson + Gaussian noise (snr=2, tau=100), Adam (lr=1e-3), 5000 epochs, `.train()` mode. No early stopping or warmup.

## Exact Parameter Counts
- **Original CNN Baseline**: 799,176
- **Transformer + ALiBi**: 798,612
- **Pointwise Control**: 798,100

## Smoke Test Verification
- Completed successfully. All variants verified for finite outputs, exact parameter counts (within ±10% difference), intact RNG order parity, shape matches (`V=[1, 20, 260]`), and completely non-trainable ALiBi biases. Attention weights proven to sum to exactly 1.0. 

## Full Experimental Results (5000 Epochs)

- **CNN baseline**:
  - PSNR: 46.6277 dB
  - SSIM: ~0.9875

- **Transformer + ALiBi**:
  - PSNR: 44.0766 dB
  - SSIM: 0.9793
  - RMSE: 0.006254
  - Runtime: 248.08 s
  - Peak VRAM: 681.51 MB

- **Pointwise control**:
  - PSNR: 42.6737 dB
  - SSIM: 0.9710
  - RMSE: 0.007350
  - Runtime: 244.83 s
  - Peak VRAM: 673.00 MB

## Derived Comparisons
- Transformer − CNN Baseline = **−2.5511 dB**
- Transformer − Control = **+1.4029 dB**

## Attention Diagnostics
Attention weights were extracted at epoch 5 during the smoke test and verified for correct shape `(1, 260, 260)` and normalization (rows sum to exactly 1). Attention weights were also extracted at the end of the full training phase.

## Limitations and Claim Boundaries
- The comparisons are strictly limited to the rank-20 ICVL mixed-noise test bed with standard DeepTensor architectures.
- We do not claim Transformers generally perform worse than CNNs, or that CNNs are inherently superior across all domains.
- We do not assert causal superiority of one architectural component or make claims of state-of-the-art performance.
- We do not claim this generalization extends beyond this specific dataset or configuration.

## Scientific Conclusion

The Transformer outperformed the parameter-matched pointwise control, indicating that cross-token sequence modeling contributed useful modeling capacity under the tested configuration. 

However, the Transformer-dominant spectral generator substantially outperformed the parameter-matched pointwise control but remained 2.5511 dB below the frozen CNN DeepTensor baseline under the tested ICVL, rank-20, mixed-noise configuration. This indicates that cross-token sequence modeling was useful relative to the control, but did not provide a stronger reconstruction prior than the original CNN spectral generator in this setting.
