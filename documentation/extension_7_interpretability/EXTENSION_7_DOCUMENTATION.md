# Extension 7: Post-hoc Factor Analysis and Descriptive Visualization of DeepTensor Components

**Status:** COMPLETE / FROZEN PENDING GIT CHECKPOINT

---

## 1. Title & Executive Summary

- **Extension Name:** Post-hoc Factor Analysis and Descriptive Visualization of DeepTensor Components
- **Primary Scientific Scope:** Post-hoc descriptive analysis of unconstrained spatial and spectral tensor factors ($U \in \mathbb{R}^{HW \times R}$, $V \in \mathbb{R}^{R \times N_\lambda}$) from a converged Rank-20 DeepTensor model on the ICVL benchmark dataset under mixed Poisson + Gaussian noise.
- **Core Research Question:** How are reconstruction energy and structural representation distributed across the $R=20$ learned latent components in an unconstrained DeepTensor model, and how orthogonal/correlated are the resulting factors?
- **Key Findings:**
  1. **Fresh Factor Extraction & Reconstruction Verification:** The original frozen baseline artifact did not contain saved final $U/V$ factors. Therefore Extension 7 used a fresh extraction run with the exact frozen baseline configuration ($R=20$, seed=42, mixed Poisson+Gaussian noise, `noise_snr=2`, `tau=100`, same architecture/training configuration, and `.train()` mode). The resulting full reconstruction PSNR was **46.1629 dB**, which falls within the project's previously observed single-run stochastic range (~45.95–46.45 dB). The extracted factors were independently verified to reproduce the model reconstruction, with maximum reconstruction difference 0 and top-$k=20$ difference $9.164796 \times 10^{-8}$.
  2. **Energy Concentration:** Component energy $E_r = \|U_r\|_F \cdot \|V_r\|_F$ shows substantial decay across post-hoc sorted components (Top 2 components $E_{11}=98.30, E_{12}=97.02$ down to $E_{19}=1.27$).
  3. **PSNR vs SSIM Metric Trajectories:** SSIM rises rapidly with the first few components, reaching 0.961697 at $k=5$ compared with 0.986746 for the full $k=20$ reconstruction (an explicit difference of $0.986746 - 0.961697 = 0.025049$), while PSNR reaches 34.2985 dB at $k=5$ compared with 46.1629 dB at $k=20$. Thus structural similarity improves substantially with early components, but neither metric should be described as fully saturated at $k=5$.
  4. **Descriptive Energy-Concentration Point ($k^*$):** Under the pre-registered 0.1 dB threshold ($\text{PSNR}(k) \ge \text{PSNR}(20) - 0.1 \text{ dB} = 46.0629 \text{ dB}$), the descriptive energy-concentration point is **$k^* = 18$**, where the top 18 components achieve **46.1186 dB** PSNR (compared to 46.1629 dB for all 20 components). This indicates that many lower-energy components continue to contribute to fine pixel-level fidelity. $k^*=18$ is a post-hoc descriptive threshold for this converged solution, not a true or intrinsic rank.
  5. **Pairwise Component Correlation Structure:** Spatial factors exhibit a mean absolute off-diagonal cosine similarity of **0.345825**, while spectral factors exhibit **0.366159**, quantifying the unconstrained pairwise correlation present without explicit orthogonality penalties.

---

## 2. Approved Scientific Definitions & Scope Boundaries

- **Scientific Terminology:** "Post-hoc energy-norm component ordering", "descriptive energy-concentration point", "pairwise component correlation structure".
- **Explicitly Excluded Claims:**
  - DO NOT call $k^*$ true rank, intrinsic rank, rank recovery, or model selection.
  - DO NOT claim high or low cosine similarity represents proven redundancy or entanglement.
  - DO NOT assign physical, semantic, or material spectral library identities to individual components.
  - DO NOT claim Extension 7 validates or confirms Extension 1's rank selection result.

---

## 3. Experimental Protocol & Leakage Controls

### Fresh Factor Extraction Protocol
The original frozen baseline artifact did not contain saved final $U/V$ factors. Therefore Extension 7 used a fresh extraction run with the exact frozen baseline configuration ($R=20$, seed=42, mixed Poisson+Gaussian noise, `noise_snr=2`, `tau=100`, same architecture/training configuration, and `.train()` mode). The resulting full reconstruction PSNR was 46.1629 dB, which falls within the project's previously observed single-run stochastic range (~45.95–46.45 dB). The extracted factors were independently verified to reproduce the model reconstruction, with maximum reconstruction difference 0 and top-$k=20$ difference $9.164796 \times 10^{-8}$.

### Experimental Configuration
- **Dataset:** ICVL Hyperspectral cube (`data/icvl.mat`), shape `(174, 164, 260)`.
- **Observation:** Master seed `42`, mixed Poisson + Gaussian noise (`noise_snr=2`, `tau=100`), input PSNR `28.4321 dB`.
- **Model Architecture:** Unmodified frozen DIP baseline ($R=20$, $N_\text{inputs}=20$, `skip` spatial DIP generator, `skip1d` spectral DIP generator).
- **Optimization:** Adam ($\eta=10^{-3}$), 5000 epochs, `.train()` evaluation mode (BatchNorm running statistics preserved).

### Strict Leakage Controls
- Ground Truth (GT) clean tensor was **NEVER** used during factor extraction, component norm calculation ($E_r$), component sorting, or cosine similarity matrix computation.
- GT was used strictly post-hoc for calculating progressive reconstruction metrics ($\text{PSNR}(k)$, $\text{SSIM}(k)$, $\text{RMSE}(k)$) and full-model sanity verification.

---

## 4. Mathematical Formulations

### Factor Extraction & Tensor Reconstruction
The DeepTensor spatial DIP generator produces $U_\text{img} \in \mathbb{R}^{1 \times R \times H \times W}$ and the spectral DIP generator produces $V_\text{raw} \in \mathbb{R}^{1 \times R \times N_\lambda}$.
Preserving row-major spatial flattening, the spatial factor $U \in \mathbb{R}^{HW \times R}$ and spectral factor $V \in \mathbb{R}^{R \times N_\lambda}$ yield full rank-$R$ tensor reconstruction:
$$\hat{Y} = U V = \sum_{r=1}^R U_{:, r} \otimes V_{r, :}$$

### Post-hoc Component Importance Metric ($E_r$)
DeepTensor has no intrinsic component ordering. $E_r$ sorting is strictly post-hoc for visualization and ablation analysis. For each component index $r \in \{1, \dots, R\}$, the scale-invariant post-hoc energy norm is defined as:
$$E_r = \|U_{:, r}\|_F \cdot \|V_{r, :}\|_F = \sqrt{\sum_{i=1}^{HW} U_{i, r}^2} \cdot \sqrt{\sum_{j=1}^{N_\lambda} V_{r, j}^2}$$
Components are sorted in descending order of $E_r$, yielding permutation indices $\pi$.

### Progressive Top-$k$ Ablation & Pre-registered $k^*$ Point
The top-$k$ partial reconstruction is given by:
$$\hat{Y}_k = \sum_{m=1}^k U_{:, \pi(m)} \otimes V_{\pi(m), :}$$
The pre-registered descriptive energy-concentration point $k^*$ is defined as:
$$k^* = \min \left\{ k \in \{1, \dots, R\} \;\middle|\; \text{PSNR}(\hat{Y}_k, Y_\text{GT}) \ge \text{PSNR}(\hat{Y}_R, Y_\text{GT}) - 0.1 \text{ dB} \right\}$$
$k^*$ is a descriptive energy-concentration threshold for this specific solution, NOT a true, effective, or intrinsic rank.

### Pairwise Cosine Similarity Matrices
For normalized spatial columns $\hat{u}_r = U_{:, r} / \|U_{:, r}\|_2$ and normalized spectral rows $\hat{v}_r = V_{r, :} / \|V_{r, :}\|_2$:
$$C_\text{spatial}[r, s] = \hat{u}_r^T \hat{u}_s \in [-1, 1]$$
$$C_\text{spectral}[r, s] = \hat{v}_r^T \hat{v}_s \in [-1, 1]$$
These matrices quantify the pairwise component correlation structure. High or low cosine similarity must not be described as proven redundancy or entanglement.

---

## 5. Non-Identifiability & Mathematical Limitations

1. **Transformational Invariance:** For any invertible matrix $G \in \mathbb{R}^{R \times R}$, transformed factors $U' = U G$ and $V' = V G^{-T}$ yield an identical tensor product $U' V' = U G G^{-T} V = U V = \hat{Y}$. Thus, individual neural factors are non-identifiable and not canonically unique.
2. **Reciprocal Scale Ambiguity:** For any scalars $\alpha_r \ne 0$, scaling $U_{:, r} \leftarrow \alpha_r U_{:, r}$ and $V_{r, :} \leftarrow \frac{1}{\alpha_r} V_{r, :}$ leaves $U_{:, r} \otimes V_{r, :}$ and $E_r$ completely invariant.
3. **Joint Sign Ambiguity:** Flipping signs $(U_{:, r}, V_{r, :}) \leftarrow (-U_{:, r}, -V_{r, :})$ leaves the reconstruction unchanged.
4. **No Physical/Semantic Identity:** Individual components do not represent physically fundamental or material-specific spectral signatures; no semantic spectral library matching is established.
5. **Single-Seed & Single-Solution Scope:** All findings describe one specific converged optimization point under seed 42.

---

## 6. Comparison with Extension 1 (Adaptive Rank Selection)

Extension 7 and Extension 1 produce numerically different component/rank-selection outcomes: Extension 7's descriptive PSNR-based energy-concentration point is $k^*=18$ of 20, whereas Extension 1 selected rank 8 during self-supervised online model selection from a maximum rank of 30. These results should not be interpreted as estimates of the same quantity. Extension 1 performs validation-driven rank growth during training, while Extension 7 performs post-hoc component truncation of an already-converged $R=20$ solution. The divergence is therefore an important methodological observation rather than evidence for a common effective rank.

---

## 7. Numerical Experimental Results

| Metric / Parameter | Value |
|---|---|
| **Dataset / Observation** | ICVL benchmark, Seed 42, Mixed Noise (`noise_snr=2`, `tau=100`) |
| **Full Model PSNR** | **46.1629 dB** |
| **Full Model SSIM** | **0.986746** |
| **Full Model RMSE** | **0.004919** |
| **Runtime** | **149.54 s** |
| **Peak VRAM** | **642.86 MB** |
| **Top-1 Component $E_r$ (Comp 11)** | **98.301598** |
| **Top-2 Component $E_r$ (Comp 12)** | **97.020035** |
| **Top-3 Component $E_r$ (Comp 15)** | **44.734650** |
| **Top-4 Component $E_r$ (Comp 04)** | **38.442600** |
| **Top-5 Component $E_r$ (Comp 06)** | **35.302658** |
| **$k=1$ Ablation PSNR** | **21.4585 dB** (SSIM: 0.527911) |
| **$k=5$ Ablation PSNR** | **34.2985 dB** (SSIM: 0.961697, SSIM diff vs full: 0.025049) |
| **$k=10$ Ablation PSNR** | **42.3128 dB** (SSIM: 0.981748) |
| **$k=15$ Ablation PSNR** | **45.3169 dB** (SSIM: 0.985909) |
| **$k=18$ Ablation PSNR** | **46.1186 dB** (SSIM: 0.986663) |
| **$k=20$ Ablation PSNR** | **46.1629 dB** (SSIM: 0.986746) |
| **Descriptive Energy-Concentration Point $k^*$** | **18** (Threshold: $\text{PSNR}(20) - 0.1 \text{ dB} = 46.0629 \text{ dB}$) |
| **Mean Abs Off-Diagonal Spatial Cosine Sim.** | **0.345825** |
| **Mean Abs Off-Diagonal Spectral Cosine Sim.** | **0.366159** |

---

## 8. Artifacts & Outputs

- **Script:** [`run_figure9_interpretability.py`](file:///e:/Rakhal_Tensor/run_figure9_interpretability.py)
- **Results Summary:** [`results_interpretability/summary.txt`](file:///e:/Rakhal_Tensor/results_interpretability/summary.txt)
- **Data Archive:** [`results_interpretability/interpretability_results.npz`](file:///e:/Rakhal_Tensor/results_interpretability/interpretability_results.npz)
- **Visualizations:**
  - [`results_interpretability/top5_components.png`](file:///e:/Rakhal_Tensor/results_interpretability/top5_components.png)
  - [`results_interpretability/progressive_topk_metrics.png`](file:///e:/Rakhal_Tensor/results_interpretability/progressive_topk_metrics.png)
  - [`results_interpretability/spatial_cosine_similarity.png`](file:///e:/Rakhal_Tensor/results_interpretability/spatial_cosine_similarity.png)
  - [`results_interpretability/spectral_cosine_similarity.png`](file:///e:/Rakhal_Tensor/results_interpretability/spectral_cosine_similarity.png)

