# Extension 4: Noise-Type Robustness at Matched Input Corruption

## 1. Executive Summary
This extension evaluates the robustness of the DeepTensor architecture across different noise distributions—specifically signal-independent Gaussian noise versus signal-dependent Poisson noise—while controlling for the overall severity of the corruption (measured by global input PSNR). We demonstrate that despite maintaining identical total input corruption, DeepTensor is markedly more sensitive to the spatially structured, signal-dependent Poisson noise compared to the signal-independent Gaussian noise.

## 2. Extension Definition
- **Final Name**: Noise-Type Robustness at Matched Input Corruption
- **What the extension means**: It systematically isolates the behavior of DeepTensor under distinct noise families with identical input severity (PSNR) to reveal vulnerabilities tied to the statistical structure of the noise itself.
- **What it does NOT mean**: It does not claim universal noise robustness, nor does it test arbitrarily large noise severities or arbitrary architectural modifications. It is solely an evaluation under the specified noise distributions.

## 3. Problem Addressed
- **Limitation of the original baseline**: The original baseline evaluates DeepTensor on a mixed Poisson-Gaussian noise model. By averaging multiple noise distributions together, this condition masks the architecture's specific responses to underlying homoscedastic (Gaussian) versus heteroscedastic (Poisson) noise.
- **Why noise distribution matters**: Algorithms that rely on deep image priors can exhibit vastly different capabilities depending on whether the variance of the corruption depends on the true signal intensity (Poisson) or is invariant to it (Gaussian).

## 4. Original Proposal Requirement
- **Preserve intended scope**: The proposal sought to determine if the denoising efficacy demonstrated by DeepTensor was fundamentally tied to the particular noise mix used in Figure 9, or if it translated to other corruption types.
- **Why this satisfies it**: By explicitly isolating the component distributions from the originally proposed mixed noise condition and carefully matching severity, we provide a mathematically fair and rigorous stress test of the robustness profile.

## 5. Initial Design Idea
The initial design proposed evaluating three conditions:
- **Mixed baseline**: The original condition containing both Gaussian and Poisson noise.
- **Pure Gaussian**: Isolating the signal-independent noise.
- **Pure Poisson**: Isolating the signal-dependent noise.
- **Reuse of `utils.measure()`**: The design intended to utilize the existing data generation pipeline from the baseline implementation to ensure perfect structural consistency.

## 6. Design Review Process
The experimental design was independently and rigorously reviewed through multiple stages to ensure scientific validity:
- **Gemini Pro High** (Initial experimental design proposal)
- **Claude Web 1** (Critical methodology review)
- **Claude Web 2** (Final verification of the constraints and calibration approach)

## 7. Critical Issues Found During Review
The review process identified several key considerations:
- **Verification of `utils.measure()` semantics**: Auditing the function confirmed that `tau=inf` purely removes Poisson sampling, and `noise_snr=0` purely removes Gaussian sampling.
- **Ambiguity of `noise_snr`**: Acknowledged that `noise_snr` in the codebase corresponds to standard deviation in photon-count space, not a decibel SNR metric.
- **PSNR matching limits**: Identified that matching global input PSNR equalizes total noise energy, but cannot eliminate structural and local statistical differences (which is precisely the variable being tested).
- **Stochastic calibration**: Raised the concern that variable iterations of random parameter search (calibration) could alter the PyTorch or NumPy RNG state prior to the final noise sampling, destroying reproducibility.
- **New-script sanity check**: Emphasized the necessity of a mixed-noise sanity check in the new script to ensure no deviations occurred during adaptation.

## 8. Final Design Evolution
To resolve the identified issues:
- Calibration logic was restricted to NumPy's RNG, preserving PyTorch's deterministic initialization.
- The NumPy RNG is explicitly re-seeded immediately prior to the final calibrated noise generation.
- A new script, `run_figure9_robustness.py`, was created and passed a full 5000-epoch mixed-noise sanity check before execution of the isolated noise runs.

## 9. Verified Noise Model
The existing `utils.measure(x, noise_snr, tau)` semantics were confirmed:
- Gaussian component: `noise = np.random.randn(...) * noise_snr`
- For `tau != Inf`: `x_meas = (np.random.poisson(x * tau) + noise) / tau`
- For `tau == Inf`: `x_meas = x + noise`

Thus:
- **Pure Gaussian**: `tau = inf`, calibrate `noise_snr`.
- **Pure Poisson**: `noise_snr = 0`, calibrate `tau`.
- **Mixed Baseline**: `noise_snr = 2`, `tau = 100`.

## 10. Experimental Methodology
- **Dataset**: ICVL (`icvl.mat`)
- **Normalization**: `/ cube.max()`
- **Preprocessing**: `scaling = 1`
- **Architecture**: DIP skip-net with 5 scales, reflection padding, bilinear/linear upsampling
- **Rank**: 20
- **Optimizer**: Adam
- **Learning Rate**: 1e-3
- **Epochs**: 5000
- **Seed**: 42
- **Evaluation Mode**: Network models kept in `.train()` to preserve original batch-norm dynamics.
- **Calibration Method**: Binary search used to identify the precise `tau` or `noise_snr` value needed.
- **PSNR Matching**: Target input PSNR of exactly 28.4321 dB (with an acceptable tolerance of $\pm$ 0.05 dB).

## 11. Fairness Controls
To ensure scientific fairness, all hyperparameters (learning rate, epochs, optimizer, architecture, rank, dataset, preprocessing) were frozen entirely across the three runs. Only the calibrated parameter associated with the specific noise type was altered. Total initial noise energy (global input PSNR) was tightly controlled.

## 12. Data-Leakage Boundary
- **GT may be used to construct/calibrate the controlled corruption**: The clean ground truth tensor was utilized by the calibration algorithm strictly to measure the intermediate input PSNR.
- **GT is NOT used in training**: The `criterion_l1` strictly operates on the noisy tensor.
- **GT is NOT used for checkpoint selection**: The "best epoch" is determined solely by the lowest training loss on the noisy tensor.
- **GT is NOT used for optimization**: Gradients are calculated against the noisy tensor.
- **GT is only used for final evaluation metrics**: Reconstruction PSNR, SSIM, and RMSE are evaluated on the final selected state post-hoc.

## 13. Reproducibility Controls
- **Seed**: Fixed at `42` globally.
- **Deterministic calibration**: The binary search bounds and iterations were statically defined.
- **Reseeding**: The NumPy RNG was re-seeded directly before the final tensor generation step, ensuring the random state was entirely decoupled from calibration steps.
- **Exact Calibrated Parameters**: 
  - Mixed: `tau=100`, `noise_snr=2`
  - Gaussian: `tau=inf`, `noise_snr=0.037875`
  - Poisson: `tau=72.124592`, `noise_snr=0`

## 14. Sanity Check
A mixed-noise run was successfully completed using the new `run_figure9_robustness.py` script. The reconstruction PSNR achieved was 46.2035 dB, perfectly mirroring the established baseline range (46.23 dB from previous validations) and verifying no unintended deviations were introduced.

## 15. Results

| Condition      | Noise Parameters          | Input PSNR | Recon PSNR | SSIM     | RMSE     | PSNR Gain | Noise-Intensity Correlation | Runtime  | Peak VRAM |
|----------------|---------------------------|------------|------------|----------|----------|-----------|-----------------------------|----------|-----------|
| **Mixed**      | `tau=100`, `noise_snr=2`      | 28.4321 dB | 46.2035 dB | 0.987190 | 0.004896 | 17.7714 dB| 0.211393                    | 168.57 s | 668.67 MB |
| **Gaussian**   | `tau=inf`, `noise_snr=0.037875` | 28.4321 dB | 46.6665 dB | 0.987765 | 0.004642 | 18.2344 dB| 0.000394                    | 163.59 s | 668.67 MB |
| **Poisson**    | `tau=72.124592`, `noise_snr=0`  | 28.4294 dB | 45.1343 dB | 0.986597 | 0.005537 | 16.7049 dB| 0.284420                    | 168.57 s | 668.67 MB |

## 16. Derived Comparisons
- **Gaussian minus Mixed PSNR**: +0.4630 dB
- **Poisson minus Mixed PSNR**: -1.0692 dB
- **Gaussian minus Poisson PSNR**: +1.5322 dB
- **RMSE Comparisons**: 
  - Poisson RMSE (0.005537) is approximately 19% higher than Gaussian RMSE (0.004642).
- **PSNR Gains**: 
  - Gaussian achieved an 18.23 dB gain.
  - Poisson achieved a 16.70 dB gain.

## 17. Noise-Structure Diagnostic
To mathematically characterize the specific statistical behaviors of the added noise, the correlation coefficient between the *clean signal intensity* and the *absolute magnitude of the noise* was computed across the flattened tensors.

Values:
- Mixed: 0.211393
- Gaussian: 0.000394
- Poisson: 0.284420

These diagnostic correlations are purely descriptive. They structurally confirm that the Gaussian noise acts independently of the signal, whereas the Poisson noise exhibits heteroscedasticity, generating larger variance where the signal intensity is higher.

## 18. Scientific Interpretation
- Gaussian noise produced the highest reconstruction PSNR among the three tested conditions.
- Poisson noise produced the lowest reconstruction PSNR.
- At matched global input corruption, the tested Poisson condition produced lower reconstruction quality than the tested Gaussian condition.
- The results indicate that DeepTensor's reconstruction capability exhibits a structural sensitivity to the statistical distribution of the noise. The mixed condition provides the original reference condition, and the pure-component experiments reveal differing architectural behavior under isolated noise distributions.

## 19. Limitations
- Evaluated on a single dataset (`icvl.mat`).
- Evaluated at a single global corruption severity (~28.43 dB Input PSNR).
- Run on a single random seed (`42`).
- Restricted specifically to the three tested noise conditions.
- Global PSNR matching does not enforce equivalent local or statistical severity; differing behavior stems entirely from those differing statistical properties.
- The correlation diagnostic is descriptive of the noise arrays drawn, not a comprehensive mathematical proof of theoretical distributions.
- No claim is made regarding universal algorithm generalization.

## 20. Conclusion
DeepTensor successfully denoises both Gaussian and Poisson noise conditions, offering significant PSNR gains in all cases. However, at matched global input severity, the architecture reconstructs Gaussian-corrupted data with noticeably higher fidelity than Poisson-corrupted data. This confirms that DeepTensor exhibits varied robustness influenced by the signal dependence and statistical structure of the target noise.

## 21. Contribution
Extension 4 advances the empirical rigor of the project by replacing a singular "mixed noise" assumption with a controlled ablation of noise families. It demonstrates that DeepTensor is not a uniform denoising panacea, but rather an algorithm whose performance boundary maps cleanly to the specific statistical profile of the corruption, providing crucial context for real-world sensor applications.

## 22. Reproducibility Information
- **Script**: `run_figure9_robustness.py`
- **Dataset**: `icvl.mat`
- **Seed**: `42`
- **Parameters**: `tau=inf, noise_snr=0.037875` (Gaussian); `tau=72.124592, noise_snr=0` (Poisson)
- **Target PSNR**: 28.4321 dB
- **Result Files**: `results_robustness/summary_mixed.txt`, `results_robustness/summary_gaussian.txt`, `results_robustness/summary_poisson.txt`

## 23. Status
Extension 4 — COMPLETE / FROZEN
