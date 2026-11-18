# DeepTensor++ — Phase 2 Presentation Blueprint
**Document ID:** EVAL-PPT-BLUEPRINT-01  
**Evaluation Phase:** Phase 2 Faculty Evaluation  
**Auditor / Presenter:** DeepTensor++ Research Team  
**Scope:** Complete Slide-by-Slide Academic Blueprint covering Topics DT-01 through DT-24  
**Integrated Navigator Port:** `http://localhost:8000`

---

## Structure Overview

This presentation blueprint organizes the DeepTensor++ research journey into six coherent sections containing 24 academic topics. Each slide profile specifies the exact slide title, topic ID badge, mathematical formulation, visual diagram specification, verified source code reference, empirical data, and faculty viva defense prompts.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2 RESEARCH PRESENTATION BLUEPRINT                   │
├──────────────────────────────────────────────────────────────────────────────┤
│ SECTION 1: Research Background & Original Paper (DT-01 to DT-04)             │
│   • Inverse Problems • Traditional Methods • Deep Image Prior • DeepTensor   │
├──────────────────────────────────────────────────────────────────────────────┤
│ SECTION 2: Tensor Mathematics & U, V, W (DT-05 to DT-10)                     │
│   • Tensor Algebra • Factor Generation • Latent Init • CP • Tucker • Splitting│
├──────────────────────────────────────────────────────────────────────────────┤
│ SECTION 3: DeepTensor Architecture & Workflow (DT-11 to DT-13)               │
│   • U-Net Skip Backbone • End-to-End Workflow • Autograd Training Loop       │
├──────────────────────────────────────────────────────────────────────────────┤
│ SECTION 4: Baselines & Comparative Architectures (DT-14 to DT-16)            │
│   • Traditional Baselines • 5 Comparative Models • Fair Evaluation Protocol  │
├──────────────────────────────────────────────────────────────────────────────┤
│ SECTION 5: Four Application-Specific Experiments (DT-17 to DT-21)            │
│   • ICVL HSI • Yale Face B • Video CS (CACTI) • PET/CT • Cross-App Synthesis │
├──────────────────────────────────────────────────────────────────────────────┤
│ SECTION 6: Additional Novelty & Improvements (DT-22 to DT-24)                │
│   • Adaptive Rank Selection • Factor Group-Sparsity • Limitations & Roadmap  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

# SECTION 1 — Research Background and Original Paper

---

### Slide 1 | Topic DT-01: Research Problem and Inverse Problems
- **Topic ID Badge:** `[DT-01] Inverse Problems & Forward Models`
- **Slide Title:** High-Dimensional Inverse Problems in Computational Imaging
- **Research Context & Problem Statement:**
  Reconstructing high-fidelity multidimensional physical signals (hyperspectral cubes, dynamic video sequences, 3D tomographic volumes) from indirect, undersampled, and noise-corrupted measurements is a fundamental challenge in computational imaging.
- **Mathematical Formulation:**
  The physical observation process is governed by the linear inverse formulation:
  $$y = \mathcal{A}(X) + n$$
  where:
  - $X \in \mathbb{R}^{I \times J \times K}$ is the underlying multidimensional ground-truth tensor.
  - $\mathcal{A}: \mathbb{R}^{I \times J \times K} \to \mathbb{R}^M$ is the physics-informed forward measurement operator ($M \ll I \cdot J \cdot K$).
  - $n \sim \mathcal{P}(\tau) + \mathcal{N}(0, \sigma^2)$ is Poisson-Gaussian or impulse noise.
  - $y \in \mathbb{R}^M$ is the observed measurement vector/sinogram/coded frame.
  Because $M \ll I \cdot J \cdot K$, the nullspace $\text{null}(\mathcal{A}) \neq \{0\}$, making naive inversion severely ill-posed and non-unique.
- **Visual Diagram Specification:**
  Flow diagram: Physical Object Tensor $X \xrightarrow{\text{Physics Operator } \mathcal{A}} \text{Ideal Projection } \xrightarrow{+ \text{ Noise } n} \text{Degraded Observation } y \xrightarrow{\text{Deep Prior Inversion}} \text{Reconstructed } \hat{X}$.
- **Verified Code Reference:**
  - File: [`src/deeptensor/physics/operators.py`](file:///e:/Rakhal_Tensor/src/deeptensor/physics/operators.py#L28-L97)
  - Key functions: `radon` (L28–L47), `video2codedvideo` (L72–L97).
- **Faculty Viva Q&A Defense:**
  - *Q:* Why are multidimensional representations necessary instead of slicing into 2D images?
  - *A:* Treating hyperspectral bands, video frames, or CT slices as isolated 2D images discards inter-band spectral correlation, temporal motion trajectories, and through-plane anatomical continuity. Tensor formulations explicitly preserve multilinear rank constraints.
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-01`

---

### Slide 2 | Topic DT-02: Traditional Reconstruction Methods
- **Topic ID Badge:** `[DT-02] Classical Reconstruction Baselines`
- **Slide Title:** Traditional Methods: Total Variation, Wavelets, and Tensor Decompositions
- **Research Context & Traditional Approaches:**
  Before deep learning, inverse problems relied on handcrafted analytical priors:
  1. *Total Variation (TV):* Promotes gradient sparsity ($\min_X \|\nabla X\|_1$ s.t. $\|y - \mathcal{A}(X)\|_2 \le \epsilon$). Advantage: preserves sharp boundaries. Limitation: produces artificial staircasing artifacts and erases fine textures.
  2. *Wavelet Sparsity:* Exploits multiscale transform sparsity ($X = W^T \alpha$). Limitation: struggles with complex non-stationary geometric patterns.
  3. *Truncated SVD / Low-Rank Matrix Approximation:* Nuclear norm minimization $\|X\|_*$. Limitation: requires 2D matrix unrolling, destroying tensor multiway geometry.
  4. *Classical Tensor Factorization (CP-ALS, Tucker):* Alternating Least Squares on $y$. Limitation: non-convex optimization prone to local minima, highly sensitive to noise, and computationally slow for large ranks.
- **Mathematical Formulation:**
  $$\hat{X}_{\text{TV}} = \arg\min_X \frac{1}{2} \|y - \mathcal{A}(X)\|_2^2 + \lambda_{\text{TV}} \sum_{i,j,k} \sqrt{(\Delta_x X)^2 + (\Delta_y X)^2 + (\Delta_z X)^2}$$
- **Verified Code Reference:**
  - File: [`src/deeptensor/baselines/spectral.py`](file:///e:/Rakhal_Tensor/src/deeptensor/baselines/spectral.py#L12-L60)
  - Key functions: `svd_decompose` (L12–L36), `tucker_decompose` (L39–L60).
  - TV Runner: [`experiments/reproduction/figure11_video_cs/run_figure11_TV.py`](file:///e:/Rakhal_Tensor/experiments/reproduction/figure11_video_cs/run_figure11_TV.py#L30-L120)
- **Faculty Viva Q&A Defense:**
  - *Q:* Did you implement these traditional baselines or just cite them?
  - *A:* Truncated SVD, TensorLy Tucker decomposition, and 3D Total Variation were fully implemented and executed on the identical datasets to establish controlled baselines.
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-02`

---

### Slide 3 | Topic DT-03: Deep Image Prior
- **Topic ID Badge:** `[DT-03] Generative Neural Priors (DIP)`
- **Slide Title:** Deep Image Prior: Untrained CNN Architectures as Structural Regularizers
- **Research Context & Principles of DIP:**
  Discovered by Ulyanov et al. (2018), the architecture of a convolutional neural network imposes a profound inductive bias favoring natural images over high-frequency noise.
  - *No Pre-Training Dataset:* Network weights $\theta$ are randomly initialized and optimized directly on the single corrupted observation $y$.
  - *Spectral Bias:* Gradient descent on CNN parameters fits low-frequency spatial patterns and natural textures orders of magnitude faster than it fits random noise.
  - *Early Stopping:* Halting optimization before convergence yields restored images without requiring an explicit regularizer.
- **Mathematical Formulation:**
  $$\theta^* = \arg\min_\theta \mathcal{L}\left(\mathcal{A}(f_\theta(z)), y\right), \quad \hat{X} = f_{\theta^*}(z)$$
  where $z \sim \mathcal{U}(0, 0.1)$ is a fixed random latent code, and $\theta$ are CNN parameters.
- **Visual Diagram Specification:**
  Convergence dynamics plot: Image Signal Fitting (rapid error drop in early iterations) vs Noise Fitting (slow, delayed error drop), illustrating the optimal early-stopping window.
- **Verified Code Reference:**
  - File: [`src/deeptensor/models/deep_prior.py`](file:///e:/Rakhal_Tensor/src/deeptensor/models/deep_prior.py#L28-L95)
  - Key functions: `get_net` (L75–L150), `Downsampler` (L28–L70).
- **Faculty Viva Q&A Defense:**
  - *Q:* How does DIP avoid simply memorizing the noisy pixels?
  - *A:* The convolution operator's weight sharing and multiscale receptive fields enforce spatial coherence. High-frequency noise lacks spatial correlation, requiring significantly more optimization steps to fit than natural structural priors.
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-03`

---

### Slide 4 | Topic DT-04: Original DeepTensor Paper
- **Topic ID Badge:** `[DT-04] DeepTensor Literature & Foundations`
- **Slide Title:** The DeepTensor Breakthrough: Marrying DIP with Tensor Decompositions
- **Research Context & Paper Summary:**
  Heckel & Soltanolkotabi (*"DeepTensor: Low-Rank Tensor Decomposition With Deep Network Priors"*, 2022) identified a critical limitation in monolithic 3D DIP: applying 3D convolutions directly to massive volumes requires astronomical GPU memory and fails to capture low-rank multilinear structure.
  - *DeepTensor Formulation:* Factorize the target tensor into low-rank factor matrices, and generate each factor matrix using an independent 1D or 2D deep prior network.
  - *Benefits:* Dramatically reduces parameter footprint, eliminates the need for 3D convolutions, and combines the structural image prior of CNNs with the rank constraints of tensor algebra.
- **Mathematical Formulation:**
  $$\hat{X} = \text{Contract}\left(f_{\theta_1}(z_1), f_{\theta_2}(z_2), \dots, f_{\theta_N}(z_N)\right)$$
  $$\min_{\{\theta_n\}} \mathcal{L}\left(\mathcal{A}\left(\text{Contract}(f_{\theta_1}(z_1), \dots)\right), y\right)$$
- **Verified Code Reference:**
  - File: [`src/deeptensor/models/deeptensor.py`](file:///e:/Rakhal_Tensor/src/deeptensor/models/deeptensor.py#L232-L286)
  - Classes: `DeepTensor` (L232–L253), `DeepTensorCP` (L254–L286).
- **Faculty Viva Q&A Defense:**
  - *Q:* What was the original contribution of DeepTensor vs our DeepTensor++ project?
  - *A:* The original paper introduced the 2D+1D and CP factor generator concept using pure CNNs. Our project engineered the standardized modular framework, introduced the 5 comparative architectural controls (Hybrid, Pointwise, Transformer), developed Adaptive Rank Selection, and proved how inductive biases vary across imaging modalities.
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-04`

---

# SECTION 2 — Tensor Mathematics and U, V, W

---

### Slide 5 | Topic DT-05: Tensor Fundamentals
- **Topic ID Badge:** `[DT-05] Multilinear Tensor Algebra`
- **Slide Title:** Tensor Fundamentals: Orders, Fibers, Slices, and Multilinear Rank
- **Mathematical Foundations:**
  - *Order & Dimension:* A tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times I_3}$ is a 3rd-order multilinear array.
  - *Fibers & Slices:* Mode-1 fibers $\mathcal{X}_{:, j, k}$ (columns), mode-2 fibers $\mathcal{X}_{i, :, k}$ (rows), mode-3 fibers $\mathcal{X}_{i, j, :}$ (tubes). Slices are 2D subarrays formed by fixing one index.
  - *Outer Product ($\circ$):* For vectors $\mathbf{u} \in \mathbb{R}^I, \mathbf{v} \in \mathbb{R}^J, \mathbf{w} \in \mathbb{R}^K$, their rank-1 outer product is:
    $$(\mathbf{u} \circ \mathbf{v} \circ \mathbf{w})_{ijk} = u_i v_j w_k$$
  - *Tensor Rank:* The minimum integer $R$ such that $\mathcal{X}$ can be expressed as the sum of $R$ rank-1 tensors. Unlike matrix rank, tensor rank computation is NP-hard.
- **Visual Diagram Specification:**
  3D cube diagram displaying tensor fibers (mode-1 column, mode-2 row, mode-3 tube) and planar slices (horizontal, lateral, frontal).
- **Verified Code Reference:**
  - File: [`src/deeptensor/models/deeptensor.py`](file:///e:/Rakhal_Tensor/src/deeptensor/models/deeptensor.py#L244-L286)
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-05`

---

### Slide 6 | Topic DT-06: U, V and W Factor Generation
- **Topic ID Badge:** `[DT-06] Factor Matrix Generators`
- **Slide Title:** Neural Factor Generation: Producing Factor Matrices U, V, and W
- **Architectural Mechanics:**
  Instead of directly optimizing tensor elements or factor matrix entries via ALS, DeepTensor generates factor matrices through neural network forward passes:
  1. *Spatial Factor $\mathbf{U} \in \mathbb{R}^{HW \times R}$:* Generated by a 2D U-Net `spatial_net` taking latent grid $z_U \in \mathbb{R}^{1 \times R \times H \times W} \to \text{output } (1, R, H, W) \xrightarrow{\text{reshape}} (1, HW, R)$.
  2. *Spectral Factor $\mathbf{V} \in \mathbb{R}^{R \times C}$:* Generated by a 1D U-Net `spectral_net` taking latent vector $z_V \in \mathbb{R}^{1 \times R \times C} \to \text{output } (1, R, C)$.
  3. *Reconstruction via Bilinear Contraction:*
     $$\hat{\mathbf{X}} = \mathbf{U} \mathbf{V} \in \mathbb{R}^{HW \times C} \xrightarrow{\text{reshape}} \mathbb{R}^{H \times W \times C}$$
- **Visual Diagram Specification:**
  Generator schematic: Latent $z_U \to \text{2D CNN} \to \mathbf{U}$; Latent $z_V \to \text{1D CNN} \to \mathbf{V}$; $\mathbf{U} \times \mathbf{V} \xrightarrow{\text{Batch MatMul}} \hat{\mathbf{X}}$.
- **Verified Code Reference:**
  - File: [`src/deeptensor/models/deeptensor.py`](file:///e:/Rakhal_Tensor/src/deeptensor/models/deeptensor.py#L244-L253)
  - Code:
    ```python
    U_img = self.spatial_net(spatial_inp)  # (B, rank, H, W)
    V = self.spectral_net(spectral_inp)    # (B, rank, C)
    U = U_img.reshape(B, R, H * W).permute(0, 2, 1)  # (B, H*W, R)
    reconstruction = torch.bmm(U, V)       # (B, H*W, C)
    ```
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-06`

---

### Slide 7 | Topic DT-07: Random Latent Initialization
- **Topic ID Badge:** `[DT-07] Latent Inputs & Positional Priors`
- **Slide Title:** Latent Initialization: Uniform Scaling and Fourier Feature Positional Encodings
- **Mechanisms of Initialization:**
  - *Uniform Scaled Noise:* Latent inputs are initialized as uniform random variables scaled by 0.1 (`torch.rand(tensize)/10.0`). Setting small amplitude prevents early saturation of activation functions.
  - *Positional Frequency Encodings:* In coordinate-based and inpainting testbeds, random Gaussian frequency projections mapping coordinates $(x, y)$ or $t$ to sinusoidal features ($\sin(2\pi B x), \cos(2\pi B x)$) bypass the spectral bias barrier, enabling rapid synthesis of high-frequency details.
  - *Perturbation Regularization:* During optimization, latent vectors are perturbed at every step ($z_t = z + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma^2)$) to prevent overfitting to degenerate local minima.
- **Verified Code Reference:**
  - File: [`src/deeptensor/utils/utils_legacy.py`](file:///e:/Rakhal_Tensor/src/deeptensor/utils/utils_legacy.py#L330-L390)
  - Functions: `get_inp` (L330–L339), `get_1d_posencode_inp` (L367–L390).
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-07`

---

### Slide 8 | Topic DT-08: CP/PARAFAC Decomposition
- **Topic ID Badge:** `[DT-08] CANDECOMP/PARAFAC (CP)`
- **Slide Title:** 3-Way CP/PARAFAC Decomposition with Deep Network Priors
- **Mathematical Formulation:**
  $$\hat{\mathcal{X}} = \sum_{r=1}^R \lambda_r (\mathbf{u}_r \circ \mathbf{v}_r \circ \mathbf{w}_r)$$
  where $\mathbf{u}_r \in \mathbb{R}^I, \mathbf{v}_r \in \mathbb{R}^J, \mathbf{w}_r \in \mathbb{R}^K$, and $\lambda \in \mathbb{R}^R$ is a learnable core scale vector.
  - *Elementwise Form:*
    $$\hat{\mathcal{X}}_{ijk} = \sum_{r=1}^R \frac{\text{core}_r}{R} u_{ir} v_{jr} w_{kr}$$
  - *Extreme Parameter Efficiency:* Total representation requires $(I + J + K) \cdot R$ entries compared to $I \cdot J \cdot K$ entries. For Yale Face B ($192 \times 168 \times 160$, $R=2000$), parameters drop from 5.16 million tensor entries to $(192+168+160) \times 2000 \approx 1.04$ million factor entries.
- **Verified Code Reference:**
  - File: [`src/deeptensor/models/deeptensor.py`](file:///e:/Rakhal_Tensor/src/deeptensor/models/deeptensor.py#L254-L286)
  - Method: `DeepTensorCP.forward`:
    ```python
    scale = (self.core / self.rank) if self.normalize_core else self.core
    recon = torch.einsum('r, ir, jr, kr -> ijk', scale, u_fac, v_fac, w_fac)
    ```
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-08`

---

### Slide 9 | Topic DT-09: Tucker Decomposition Audit
- **Topic ID Badge:** `[DT-09] Tucker Decomposition Audit`
- **Slide Title:** Scientific Audit: Classical Tucker Baseline vs Generative Deep Prior
- **Formal Audit Findings:**
  1. *Classical Truncated Tucker IS Implemented:* Found in `src/deeptensor/baselines/spectral.py` (L39–L60). Computes classical HOSVD using TensorLy (`decomposition.tucker`), providing a non-deep low-rank matrix/tensor baseline.
  2. *Neural Generative Tucker Prior is NOT Implemented:* A deep generative prior learning a dense 3D neural core tensor $\mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times R_3}$ does not exist in the codebase.
- **Scientific Justification:**
  $$\hat{\mathcal{X}}_{\text{Tucker}} = \mathcal{G} \times_1 \mathbf{U} \times_2 \mathbf{V} \times_3 \mathbf{W}$$
  While Tucker provides mode-specific multilinear ranks $(R_1, R_2, R_3)$, the core tensor $\mathcal{G}$ scales cubically $O(R^3)$. For large rank applications such as PET/CT ($R=1000$), a dense core would require $10^9$ parameters—far exceeding GPU memory limits. The project intentionally adopted the CP formulation to maintain $O(R)$ linear scaling.
- **Verified Code Reference:**
  - File: [`src/deeptensor/baselines/spectral.py`](file:///e:/Rakhal_Tensor/src/deeptensor/baselines/spectral.py#L39-L60)
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-09`

---

### Slide 10 | Topic DT-10: Dimensionality Splitting Approaches
- **Topic ID Badge:** `[DT-10] Dimensionality Splitting Taxonomy`
- **Slide Title:** Dimensionality Splitting: Monolithic 3D vs 2D+1D vs Three 1D Generators
- **Comparative Taxonomy:**
  | Dimension Strategy | Architectural Realization | Factor Outputs | Computational Complexity | Primary Inductive Bias |
  | :--- | :--- | :--- | :--- | :--- |
  | **Approach 1: Monolithic 3D** | `UNetND(ndim=3)` | Full volume $\hat{X}$ | $O(I \cdot J \cdot K)$ (Cubic memory) | Volumetric spatial smoothness |
  | **Approach 2: 2D + 1D Split** | `DeepTensor` | Spatial $U(HW \times R)$, Spec $V(R \times C)$ | $O(HW \cdot R + C \cdot R)$ | 2D image texture + 1D spectral curve |
  | **Approach 3: Three 1D Split** | `DeepTensorCP` | $u(I \times R), v(J \times R), w(K \times R)$ | $O((I + J + K) \cdot R)$ (Linear) | Separable rank-1 lines/fibers |
- **Architectural Conclusion:**
  Approach 2 (2D+1D) is optimal when data possesses 2D planar structure (hyperspectral imaging, video frames). Approach 3 (CP) is optimal for high-dimensional separable volumes with large ranks (tomography, photometric illumination tensors).
- **Verified Code Reference:**
  - File: [`src/deeptensor/models/deeptensor.py`](file:///e:/Rakhal_Tensor/src/deeptensor/models/deeptensor.py#L112-L286)
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-10`

---

# SECTION 3 — DeepTensor Architecture and Workflow

---

### Slide 11 | Topic DT-11: Original DeepTensor CNN Architecture
- **Topic ID Badge:** `[DT-11] U-Net Backbone Architecture`
- **Slide Title:** Convolutional Backbone: Multiscale U-Net with Skip Connections
- **Architectural Details:**
  The core backbone for factor generation is a symmetric encoder-decoder U-Net:
  - *Encoder:* 5 downsampling stages with stride-2 convolutions or Lanczos filters. Expands channels ($128 \to 128$).
  - *Skip Connections:* $1 \times 1$ convolutions connecting corresponding encoder and decoder levels to bypass the bottleneck, directly preserving high-frequency spatial/spectral gradients.
  - *Decoder:* 5 upsampling stages with bilinear/linear interpolation and $3 \times 3$ (or $1 \times 3$) convolutional refinement.
  - *Activation & Norm:* LeakyReLU ($\alpha = 0.2$) and BatchNorm1d/2d.
- **Verified Code Reference:**
  - File: [`src/deeptensor/models/backbones/skip.py`](file:///e:/Rakhal_Tensor/src/deeptensor/models/backbones/skip.py#L5-L197)
  - Functions: `skip` (L5–L101), `skip1d` (L102–L197).
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-11`

---

### Slide 12 | Topic DT-12: Complete Reconstruction Workflow
- **Topic ID Badge:** `[DT-12] Reconstruction Workflow`
- **Slide Title:** End-to-End Reconstruction Workflow: From Latent Codes to Clean Inference
- **Execution Pipeline Steps:**
  1. **Dataset Ingestion:** Load clean ground-truth tensor $X$ (used exclusively for post-hoc PSNR/SSIM evaluation).
  2. **Physics Simulation:** Apply forward operator $\mathcal{A}$ and noise model to generate degraded observation $y$.
  3. **Latent Code Sampling:** Sample fixed random tensors $z_U, z_V$ (or $z_u, z_v, z_w$).
  4. **Neural Factor Forward Pass:** Pass perturbed latents through generative networks to obtain factor matrices.
  5. **Tensor Contraction:** Contract factors via matrix multiplication or CP einsum into candidate tensor $\hat{X}$.
  6. **Measurement Forward Projection:** Pass $\hat{X}$ through physics operator $\mathcal{A}(\hat{X})$.
  7. **Loss Evaluation:** Compute measurement loss $\mathcal{L}(\mathcal{A}(\hat{X}), y)$.
  8. **Clean Inference:** Periodically evaluate clean, unperturbed latents without gradient calculation.
- **Verified Code Reference:**
  - File: [`experiments/reproduction/figure09_hyperspectral/run_figure9.py`](file:///e:/Rakhal_Tensor/experiments/reproduction/figure09_hyperspectral/run_figure9.py#L130-L240)
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-12`

---

### Slide 13 | Topic DT-13: Training Loop and Backpropagation
- **Topic ID Badge:** `[DT-13] Optimization & Autograd Engine`
- **Slide Title:** Optimization Engine: Autograd Backpropagation and Adam Update Dynamics
- **Optimization Formulation:**
  - *Optimizer:* Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$). Learning rate $\eta = 10^{-3}$.
  - *Autograd Backpropagation:* Gradients flow backward from the measurement loss through the forward physical operator $\mathcal{A}$, through the multilinear tensor contraction, and into the neural network parameter weights $\theta$:
    $$\nabla_\theta \mathcal{L} = \left(\frac{\partial \hat{X}}{\partial \theta}\right)^T \left(\frac{\partial \mathcal{A}}{\partial \hat{X}}\right)^T \nabla_{\hat{y}} \mathcal{L}$$
  - *Exponential Moving Average (EMA):* An exponential running average of reconstructed tensors is maintained ($\alpha = 0.99$) to stabilize high-frequency oscillations.
- **Verified Code Reference:**
  - File: [`experiments/reproduction/figure09_hyperspectral/run_figure9.py`](file:///e:/Rakhal_Tensor/experiments/reproduction/figure09_hyperspectral/run_figure9.py#L170-L240)
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-13`

---

# SECTION 4 — Baselines and Comparative Architectures

---

### Slide 14 | Topic DT-14: Traditional Methods and Baseline Comparisons
- **Topic ID Badge:** `[DT-14] Baseline Taxonomy & Purpose`
- **Slide Title:** Baseline Taxonomy: Isolating the Benefit of Deep Generative Priors
- **Baseline Framework:**
  To scientifically isolate where performance gains originate, the research establishes two distinct baseline tiers:
  1. *Non-Deep Numerical Baselines:* Truncated SVD, TensorLy Tucker decomposition, and 3D Total Variation.
     - *Purpose:* Quantifies what can be reconstructed using purely linear or convex piecewise-smooth assumptions.
  2. *Neural Control Baselines:* Baseline CNN, CNN Control, Pointwise Control.
     - *Purpose:* Isolates whether performance differences in neural architectures stem from parameter capacity, convolutional locality, or attention mechanisms.
- **Verified Code Reference:**
  - File: [`src/deeptensor/baselines/spectral.py`](file:///e:/Rakhal_Tensor/src/deeptensor/baselines/spectral.py#L12-L60)
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-14`

---

### Slide 15 | Topic DT-15: Five Model Architectures
- **Topic ID Badge:** `[DT-15] Five Comparative Models`
- **Slide Title:** The Five Comparative Architectures: From Local Convolutions to Pure Attention
- **Architectural Suite Specifications:**
  1. **Baseline CNN (`skip1d`):** Dmitry Ulyanov's 1D convolutional skip network. Receptive field: strictly local.
  2. **Hybrid CNN-Transformer (`hybrid`):** U-Net body + `SpectralSelfAttention` (MHA with 2 heads, $d_{\text{model}}=128$, Sinusoidal PE) + FFN. Combines multiscale local features with global token interactions.
  3. **CNN Control (`cnn_control`):** U-Net body + 1D Convolutional Residual Block (`Conv1d(k=3) -> GELU -> Conv1d(k=3)`). Matched to MHA within $\pm 0.05\%$ parameters.
  4. **Pointwise Control (`pointwise_control`):** U-Net body + Pointwise MLP (`Linear(128, 256) -> GELU -> Linear(256, 128)`). Token-isolated channel mixing with zero cross-token interaction.
  5. **Pure Transformer (`transformer`):** 4-layer Pre-LN Transformer with bidirectional ALiBi distance bias. Zero convolutional layers; entirely attention-driven.
- **Verified Code Reference:**
  - File: [`src/deeptensor/models/factory.py`](file:///e:/Rakhal_Tensor/src/deeptensor/models/factory.py#L145-L230)
  - Components: [`src/deeptensor/models/attention/spatial_spectral.py`](file:///e:/Rakhal_Tensor/src/deeptensor/models/attention/spatial_spectral.py#L14-L230)
  - Transformer: [`src/deeptensor/models/attention/spectral_transformer.py`](file:///e:/Rakhal_Tensor/src/deeptensor/models/attention/spectral_transformer.py#L44-L111)
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-15`

---

### Slide 16 | Topic DT-16: Fair Comparative Evaluation
- **Topic ID Badge:** `[DT-16] Fair Benchmarking Protocol`
- **Slide Title:** Methodological Rigor: Parameter Matching, Latent Controls, and Consistent Inference
- **Fair Benchmarking Safeguards:**
  - *Strict Parameter Matching:* Pointwise Control and CNN Control are mathematically calibrated to match the Hybrid Attention model within $\pm 0.05\%$ parameters.
  - *Identical Latent Seeding:* All 5 models in any given testbed share the exact same latent seed ($z$) and observation seed ($y$).
  - *Equal Epoch Budgets:* Training runs for identical epoch counts (ICVL: 5000, Yale: 1200, Video: 3000, PET: 4000).
  - *Identical DIP Inference Convention:* Metrics are calculated using clean-latent inference in `torch.no_grad()` mode. Ground-truth data is never exposed during training.
- **Verified Code Reference:**
  - File: [`src/deeptensor/models/factory.py`](file:///e:/Rakhal_Tensor/src/deeptensor/models/factory.py#L29-L32) (`count_parameters`).
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-16`

---

# SECTION 5 — Four Application-Specific Experiments

---

### Slide 17 | Topic DT-17: Figure 09 — ICVL Hyperspectral Image Denoising
- **Topic ID Badge:** `[DT-17] Experiment: Figure 09 (ICVL)`
- **Slide Title:** Figure 09: ICVL Hyperspectral Denoising — Convolutional Locality Dominates
- **Experimental Protocol:**
  - *Dataset:* `data/icvl.mat` ($174 \times 164 \times 260$ spectral bands).
  - *Model:* DeepTensor $(U \times V)$, Rank $R=20$. Loss: $\mathcal{L}_1$. Budget: 5000 epochs.
  - *Noise:* Poisson ($\tau = 100$) + Gaussian readout ($\text{SNR} = 2.0$) $\to$ Input PSNR: 28.43 dB.
- **Measured Results:**
  - **Baseline CNN: 46.0653 dB | SSIM: 0.9874 | 3.005M params (WINNER)**
  - CNN Control: 45.5844 dB | SSIM: 0.9857 | 3.138M params (-0.48 dB)
  - Pointwise Control: 45.4229 dB | SSIM: 0.9852 | 3.138M params (-0.64 dB)
  - Hybrid Attention: 44.8526 dB | SSIM: 0.9842 | 3.138M params (-1.21 dB)
  - Pure Transformer: 43.4026 dB | SSIM: 0.9766 | 3.005M params (-2.66 dB)
- **Scientific Conclusion:**
  Continuous spectral bands exhibit strong physical correlation between neighboring wavelengths. 1D convolutions enforce local spectral smoothness. Transformer attention adds unnecessary degrees of freedom, overfitting to noise and degrading PSNR by -2.66 dB.
- **Verified Code Reference:**
  - File: [`experiments/reproduction/figure09_hyperspectral/run_figure9.py`](file:///e:/Rakhal_Tensor/experiments/reproduction/figure09_hyperspectral/run_figure9.py#L1-L341)
  - Results: `results/hybrid/figure09_hyperspectral/figure09_icvl_hsi_comparison.json`
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-17`

---

### Slide 18 | Topic DT-18: Figure 10 — Extended Yale Face B
- **Topic ID Badge:** `[DT-18] Experiment: Figure 10 (Yale B)`
- **Slide Title:** Figure 10: Yale Face B — Pure Transformer Dominates Photometric Tensors
- **Experimental Protocol:**
  - *Dataset:* `data/yale_b.mat` ($192 \times 168 \times 160$ illuminations).
  - *Model:* DeepTensorCP ($u \otimes v \otimes w$), Rank $R=2000$. Loss: $\mathcal{L}_1$. Budget: 1200 epochs.
  - *Corruption:* 30% Salt & Pepper impulse noise.
- **Measured Results:**
  - **Pure Transformer: 28.5552 dB | SSIM: 0.8655 | 3.923M params (WINNER)**
  - CNN Control: 28.3521 dB | SSIM: 0.8612 | 5.603M params (+0.57 dB over baseline)
  - Pointwise Control: 28.1703 dB | SSIM: 0.8601 | 5.603M params (+0.39 dB over baseline)
  - Hybrid Attention: 28.0392 dB | SSIM: 0.8592 | 5.603M params (+0.26 dB over baseline)
  - Baseline CNN: 27.7808 dB | SSIM: 0.8528 | 5.471M params (Reference)
- **Scientific Conclusion:**
  The 160 illumination conditions do not possess metric translational smoothness; lighting changes produce global, non-local shadow and highlight couplings. Pure Transformer attention captures global photometric relationships across light angles, outperforming Baseline CNN by +0.77 dB with 1.5 million fewer parameters.
- **Verified Code Reference:**
  - File: [`experiments/reproduction/figure10_yale_denoising/run_figure10_yale_tensor.py`](file:///e:/Rakhal_Tensor/experiments/reproduction/figure10_yale_denoising/run_figure10_yale_tensor.py#L1-L363)
  - Results: `results/hybrid/figure10_yale_denoising/figure10_yale_b_comparison.json`
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-18`

---

### Slide 19 | Topic DT-19: Figure 11 — Video Compressive Sensing
- **Topic ID Badge:** `[DT-19] Experiment: Figure 11 (Video CS)`
- **Slide Title:** Figure 11: Video CS (CACTI) — Temporal Convolutions Outperform Attention
- **Experimental Protocol:**
  - *Dataset:* `data/cat_video.mat` ($128 \times 128 \times 64$ frames).
  - *Application:* CACTI 8-frame temporal multiplexing ($8\times$ compressive ratio).
  - *Model:* DeepTensor $(U \times V)$, Rank $R=32$. Loss: $\mathcal{L}_2$. Budget: 3000 epochs.
- **Measured Results:**
  - **Baseline CNN: 25.1577 dB | SSIM: 0.7223 | 785k params (WINNER)**
  - CNN Control: 25.0395 dB | SSIM: 0.7101 | 834k params (-0.12 dB)
  - Pointwise Control: 24.8147 dB | SSIM: 0.6885 | 835k params (-0.34 dB)
  - Hybrid Attention: 24.7440 dB | SSIM: 0.6865 | 835k params (-0.41 dB)
  - Pure Transformer: 24.0502 dB | SSIM: 0.6531 | 1.376M params (-1.11 dB)
- **Scientific Conclusion:**
  Dynamic video sequences possess extreme temporal locality governed by physical object velocities. 1D temporal convolutions provide the exact inductive bias needed for motion continuity. Attention allows distant frames to cross-talk, introducing temporal blur under $8\times$ compressive multiplexing.
- **Verified Code Reference:**
  - File: [`experiments/hybrid_prototypes/run_hybrid_figure11_video.py`](file:///e:/Rakhal_Tensor/experiments/hybrid_prototypes/run_hybrid_figure11_video.py#L1-L398)
  - Results: `results/hybrid/figure11_video_cs/figure11_video_cs_comparison.json`
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-19`

---

### Slide 20 | Topic DT-20: Figure 12 — PET/CT Radon Compressive Sensing
- **Topic ID Badge:** `[DT-20] Experiment: Figure 12 (PET/CT)`
- **Slide Title:** Figure 12: PET/CT Radon Tomography — Pointwise Channel Mixing Wins
- **Experimental Protocol:**
  - *Dataset:* `data/pet.mat` ($128 \times 128 \times 56$ slices).
  - *Application:* 40-angle Radon projection tomographic compressive sensing.
  - *Model:* DeepTensorCP ($u \otimes v \otimes w$), Rank $R=1000$. Loss: $\mathcal{L}_2$. Budget: 4000 epochs.
- **Measured Results:**
  - **Pointwise Control: 30.6212 dB | SSIM: 0.8621 | 4.051M params (WINNER)**
  - Pure Transformer: 30.4056 dB | SSIM: 0.8585 | 3.663M params (+0.40 dB over baseline)
  - CNN Control: 30.3128 dB | SSIM: 0.8579 | 4.051M params (+0.31 dB over baseline)
  - Hybrid Attention: 30.2947 dB | SSIM: 0.8566 | 4.051M params (+0.29 dB over baseline)
  - Baseline CNN: 30.0023 dB | SSIM: 0.8472 | 3.919M params (Reference)
- **Scientific Conclusion:**
  In 3D tomographic CT/PET z-axis slices (56 slices), each anatomical level has distinct structural features. Enforcing 1D translational convolutional invariance across slices is physically incorrect. Pointwise Control provides non-linear channel capacity without enforcing spatial coupling across organs, setting the state of the art at 30.62 dB.
- **Verified Code Reference:**
  - File: [`experiments/hybrid_prototypes/run_hybrid_figure12_pet.py`](file:///e:/Rakhal_Tensor/experiments/hybrid_prototypes/run_hybrid_figure12_pet.py#L1-L416)
  - Results: `results/hybrid/figure12_pet_ct/figure12_pet_ct_comparison.json`
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-20`

---

### Slide 21 | Topic DT-21: Cross-Application Comparative Results
- **Topic ID Badge:** `[DT-21] Cross-Application Synthesis`
- **Slide Title:** Cross-Application Master Results: Application-Dependent Architectural Performance
- **Master Results Comparison Table:**
  | Testbed | Physical Operator & Domain | Baseline CNN | Hybrid Attn | CNN Control | Pointwise Control | Pure Transformer | Domain Winner |
  | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
  | **Fig 09: ICVL** | Hyperspectral Denoising ($174 \times 164 \times 260$) | **46.07 dB** | 44.85 dB | 45.58 dB | 45.42 dB | 43.40 dB | **Baseline CNN (+0.48 dB)** |
  | **Fig 10: Yale B**| 30% S&P Face Denoising ($192 \times 168 \times 160$) | 27.78 dB | 28.04 dB | 28.35 dB | 28.17 dB | **28.56 dB** | **Pure Transformer (+0.77 dB)** |
  | **Fig 11: Video** | CACTI 8x Video CS ($128 \times 128 \times 64$) | **25.16 dB** | 24.74 dB | 25.04 dB | 24.81 dB | 24.05 dB | **Baseline CNN (+0.12 dB)** |
  | **Fig 12: PET/CT**| 40-Angle Radon Tomography ($128 \times 128 \times 56$) | 30.00 dB | 30.29 dB | 30.31 dB | **30.62 dB** | 30.41 dB | **Pointwise Control (+0.62 dB)** |
- **Core Scientific Insight:**
  Architectural performance in untrained neural generative priors is fundamentally application-dependent. Rather than a universally optimal architecture, empirical evidence supports hypotheses aligned with coordinate physical properties:
  - *Smooth continuous metric dimensions* (wavelengths, time steps) $\implies$ **Convolutional Locality wins.**
  - *Disordered, global photometric dimensions* (illuminations) $\implies$ **Self-Attention wins.**
  - *Decoupled, anatomically distinct slices* (CT z-axis) $\implies$ **Pointwise Channel Mixing wins.**
- **Verified Code Reference:**
  - Files: `results/hybrid/*/*_comparison.json`
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-21`

---

# SECTION 6 — Additional Novelty and Improvements

---

### Slide 22 | Topic DT-22: Adaptive Rank Selection
- **Topic ID Badge:** `[DT-22] Extension: Adaptive Rank Selection`
- **Slide Title:** Adaptive Rank Selection: Overcoming Fixed-Rank Limits via Loss Plateaus
- **Motivation & Implementation:**
  Manual rank tuning requires expensive trial-and-error. Setting rank too low truncates signal; setting it too high causes noise overfitting.
  - *Plateau-Based Expansion:* Networks initialize with $R_{\text{max}} = 30$ capacity but start with active rank $R_{\text{start}} = 4$.
  - *EMA Tracking:* Measurement loss $\mathcal{L}_{\text{L1}}$ is smoothed using Exponential Moving Average ($\alpha = 0.99$).
  - *Rank Growth Trigger:* When relative loss reduction over a 200-epoch window drops below $\tau_{\text{plateau}} = 10^{-3}$, rank grows by $R_{\text{step}} = 2$ after a 300-epoch stabilization cooldown.
  - *Adam State Reset:* When new channels are activated, Adam moments (`exp_avg`, `exp_avg_sq`) for those specific parameters are reset to zero to prevent stale gradient bias.
- **Empirical Validation:**
  - Achieves **46.1571 dB PSNR** on ICVL (outperforming the fixed Rank-20 baseline of 46.0653 dB by **+0.0918 dB**) without requiring prior knowledge of the optimal rank!
- **Verified Code Reference:**
  - File: [`experiments/extensions/adaptive_rank/run_figure9_adaptive.py`](file:///e:/Rakhal_Tensor/experiments/extensions/adaptive_rank/run_figure9_adaptive.py#L64-L140)
  - Metrics: `results/adaptive_rank_selection/metrics.json`
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-22`

---

### Slide 23 | Topic DT-23: Factor Group-Sparsity
- **Topic ID Badge:** `[DT-23] Extension: Factor Group-Sparsity`
- **Slide Title:** Factor Group-Sparsity: Automated Component Pruning via Group Lasso
- **Mathematical Formulation:**
  To automatically prune unnecessary rank components from an over-parameterized network ($R_{\text{max}} = 30$), we introduce a joint Factor Group-Sparsity regularizer:
  $$g_r = \sqrt{\|U_r\|_2^2 + \|V_r\|_2^2 + 10^{-12}}$$
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{measurement}}(UV^T, y) + \lambda \sum_{r=1}^{R_{\text{max}}} g_r$$
- **Empirical Regularization Sweep:**
  | Penalty $\lambda$ | Reconstructed PSNR (dB) | SSIM | Primary Effective Rank | Cumulative Energy Rank | Observation |
  | :--- | :---: | :---: | :---: | :---: | :--- |
  | $\lambda = 0.0$ | 19.79 dB | 0.2905 | 18 | 17 | Unregularized short baseline |
  | $\lambda = 10^{-4}$ | **38.15 dB** | **0.9774** | **3** | **3** | **Optimal sparse discovery (3 dominant factors)** |
  | $\lambda = 10^{-3}$ | 31.37 dB | 0.9179 | 1 | 1 | Over-penalized (excessive shrinkage) |
  | $\lambda = 10^{-2}$ | 18.65 dB | -0.027 | 30 | 28 | Complete collapse of factor representation |
- **Scientific Finding:**
  Group Lasso successfully shrinks inactive rank components into near-zero norms, demonstrating that over-parameterized deep priors can automatically discover low-rank structures when $\lambda$ is carefully calibrated.
- **Verified Code Reference:**
  - File: [`experiments/extensions/group_sparsity/run_figure9_sparsity.py`](file:///e:/Rakhal_Tensor/experiments/extensions/group_sparsity/run_figure9_sparsity.py#L83-L120)
  - Results: `results/extensions/sparsity/standard/summary_lambda_0.0001_seed_42.txt`
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-23`

---

### Slide 24 | Topic DT-24: Limitations and Future Work
- **Topic ID Badge:** `[DT-24] Research Roadmap & Future Work`
- **Slide Title:** Limitations, Methodological Reflections, and Future Research Horizons
- **Critical Methodological Limitations:**
  1. *Instance Optimization Latency:* DeepTensor solves an optimization problem per image/volume (300 to 500 seconds on RTX 4060 GPU). Unlike supervised feed-forward models, it is not real-time.
  2. *Inductive Bias Inconsistency:* As proven in Topic DT-21, static architectural choices cannot simultaneously succeed across all modalities.
- **Future Research Horizons (AHD-Net Preview):**
  - *Dynamic Context-Aware Routing:* Rather than forcing a static architecture, future work can incorporate an automated router (such as the CAGC routing explored in AHD-Net) that dynamically allocates weighting across CNN, Pointwise, and Attention branches based on the physics of the input problem.
  - *Physics-Informed Preconditioning:* Incorporating approximate inverse operators (e.g., FBP for Radon, Pseudo-inverse for CACTI) as initializers to accelerate convergence by $10\times$.
- **Verified Code Reference:**
  - Document: [`reports/extensions/ADAPTIVE_HYBRID_V1_FINAL_VALIDATION.md`](file:///e:/Rakhal_Tensor/reports/extensions/ADAPTIVE_HYBRID_V1_FINAL_VALIDATION.md#L1-L150)
- **Interactive Navigator Link:** `http://localhost:8000/?topic=DT-24`

