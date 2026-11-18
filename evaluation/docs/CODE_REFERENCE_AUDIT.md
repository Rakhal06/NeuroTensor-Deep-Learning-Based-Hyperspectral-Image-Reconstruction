# DeepTensor++ — Code Reference & Topic Audit
**Document ID:** EVAL-REF-01  
**Evaluation Phase:** Phase 2 Faculty Evaluation  
**Auditor:** DeepTensor++ Research & Verification Team  
**Scope:** Verified Mapping of Research Topics (DT-01 to DT-24) to Repository Source Code

---

## 1. Topic-to-Code Master Index

This document establishes the precise, audited source code locations for all 24 research topics. Every entry has been verified against the physical file system in `E:\Rakhal_Tensor\`.

| Topic ID | Topic Name | Primary File Path | Line Range | Primary Symbol / Function | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DT-01** | Research Problem & Inverse Problems | `src/deeptensor/physics/operators.py` | L28–L97 | `radon`, `video2codedvideo` | **VERIFIED** |
| **DT-02** | Traditional Reconstruction Methods | `src/deeptensor/baselines/spectral.py` | L12–L60 | `svd_decompose`, `tucker_decompose` | **VERIFIED** |
| **DT-03** | Deep Image Prior (DIP) | `src/deeptensor/models/deep_prior.py` | L28–L95 | `Downsampler`, `get_net` | **VERIFIED** |
| **DT-04** | Original DeepTensor Paper | `src/deeptensor/models/deeptensor.py` | L232–L286 | `DeepTensor`, `DeepTensorCP` | **VERIFIED** |
| **DT-05** | Tensor Fundamentals | `src/deeptensor/models/deeptensor.py` | L244–L286 | `forward` (matrix & CP contraction) | **VERIFIED** |
| **DT-06** | U, V, and W Factor Generation | `src/deeptensor/models/deeptensor.py` | L244–L286 | `DeepTensor.forward`, `DeepTensorCP.forward` | **VERIFIED** |
| **DT-07** | Random Latent Initialization | `src/deeptensor/utils/utils_legacy.py` | L330–L390 | `get_inp`, `get_1d_posencode_inp` | **VERIFIED** |
| **DT-08** | CP/PARAFAC Decomposition | `src/deeptensor/models/deeptensor.py` | L254–L286 | `DeepTensorCP` | **VERIFIED** |
| **DT-09** | Tucker Decomposition Audit | `src/deeptensor/baselines/spectral.py` | L39–L60 | `tucker_decompose` | **VERIFIED (Classical Only)** |
| **DT-10** | Dimensionality Splitting Approaches | `src/deeptensor/models/deeptensor.py` | L112–L286 | `UNetND`, `DeepTensor`, `DeepTensorCP` | **VERIFIED** |
| **DT-11** | Original DeepTensor CNN Architecture | `src/deeptensor/models/backbones/skip.py`| L5–L197 | `skip`, `skip1d` | **VERIFIED** |
| **DT-12** | Complete Reconstruction Workflow | `experiments/reproduction/figure09_hyperspectral/run_figure9.py` | L130–L240 | Training & Reconstruction Workflow | **VERIFIED** |
| **DT-13** | Training Loop & Backpropagation | `experiments/reproduction/figure09_hyperspectral/run_figure9.py` | L170–L240 | Autograd Optimization Loop | **VERIFIED** |
| **DT-14** | Traditional Baselines vs Neural Controls| `src/deeptensor/baselines/spectral.py` | L12–L60 | SVD and Truncated Tucker Baselines | **VERIFIED** |
| **DT-15** | Five Comparative Model Architectures | `src/deeptensor/models/factory.py` | L145–L230 | `build_spectral_net` | **VERIFIED** |
| **DT-16** | Fair Comparative Evaluation Protocol | `src/deeptensor/models/factory.py` | L29–L32 | `count_parameters` | **VERIFIED** |
| **DT-17** | Figure 09 — ICVL Hyperspectral Denoising| `experiments/reproduction/figure09_hyperspectral/run_figure9.py` | L1–L341 | Full ICVL Experiment Pipeline | **VERIFIED** |
| **DT-18** | Figure 10 — Extended Yale Face B | `experiments/reproduction/figure10_yale_denoising/run_figure10_yale_tensor.py` | L1–L363 | Yale B 30% S&P Experiment | **VERIFIED** |
| **DT-19** | Figure 11 — Video Compressive Sensing | `experiments/hybrid_prototypes/run_hybrid_figure11_video.py` | L1–L398 | CACTI 8x Compression Pipeline | **VERIFIED** |
| **DT-20** | Figure 12 — PET/CT Radon Tomography | `experiments/hybrid_prototypes/run_hybrid_figure12_pet.py` | L1–L416 | 40-Angle Radon Projection Pipeline | **VERIFIED** |
| **DT-21** | Cross-Application Comparative Results | `results/hybrid/figure09_hyperspectral/figure09_icvl_hsi_comparison.json` | L1–L100 | Cross-Application Metrics Matrix | **VERIFIED** |
| **DT-22** | Adaptive Rank Selection Improvement | `experiments/extensions/adaptive_rank/run_figure9_adaptive.py` | L64–L140 | Plateau-Driven Rank Growth Engine | **VERIFIED** |
| **DT-23** | Factor Group-Sparsity Regularization | `experiments/extensions/group_sparsity/run_figure9_sparsity.py` | L83–L120 | `compute_group_regularizer` | **VERIFIED** |
| **DT-24** | Limitations, Trade-offs & Future Work | `reports/extensions/ADAPTIVE_HYBRID_V1_FINAL_VALIDATION.md` | L1–L150 | Inductive Bias Analysis & Roadmap | **VERIFIED** |

---

## 2. Granular Topic Verification Profiles

### [DT-01] Research Problem and Inverse Problems
- **Conceptual Scope:** Physical measurement model $y = \mathcal{A}(X) + n$, ill-posedness, forward vs inverse operators, undersampling, noise corruption, and multidimensional tensor representations.
- **Primary Source Code:** `src/deeptensor/physics/operators.py` (Lines 28–97)
  - `radon(imten, angles)` (L28–L47): Forward Radon tomographic projection operator for CT/PET imaging.
  - `get_video_coding_frames(video_size, nframes)` (L49–L70): Generates pseudorandom binary shutter masks for CACTI compressive video sensing.
  - `video2codedvideo(video_ten, masks_ten, nframes)` (L72–L97): Implements the forward optical multiplexing operator compressing 8 video frames into 1 snapshot.
- **Loss Functions:** `src/deeptensor/losses/objectives.py` (Lines 1–50) defining measurement-space data fidelity penalties $\mathcal{L}(y, \hat{y})$.

### [DT-02] Traditional Reconstruction Methods
- **Conceptual Scope:** Analytical and iterative classical baselines (Total Variation, Wavelet thresholding, Truncated SVD, Classical Tucker decomposition).
- **Primary Source Code:** `src/deeptensor/baselines/spectral.py` (Lines 12–60)
  - `svd_decompose(cube, rank)` (L12–L36): Truncated Singular Value Decomposition for 2D spatial-spectral matrices.
  - `tucker_decompose(cube, rank, max_iters=1000)` (L39–L60): Higher-Order SVD / Tucker decomposition using TensorLy.
- **TV Regularization:** `experiments/reproduction/figure11_video_cs/run_figure11_TV.py` (Lines 30–120) implementing 3D anisotropic Total Variation minimization via gradient projection.

### [DT-03] Deep Image Prior (DIP)
- **Conceptual Scope:** Untrained convolutional networks as generative image priors; implicit regularization of natural frequencies; optimization over network parameter weights without training datasets; early stopping dynamics.
- **Primary Source Code:** `src/deeptensor/models/deep_prior.py` (Lines 28–95)
  - `Downsampler` (L28–L70): Anti-aliased Lanczos and Gaussian downsampling modules.
  - `get_net(...)` (L75–L150): Dynamic factory instantiating U-Net skip architectures, setting padding modes, skip channels, and activation functions.
- **Backbone Definition:** `src/deeptensor/models/backbones/skip.py` (Lines 5–101) detailing encoder, decoder, and skip connections.

### [DT-04] Original DeepTensor Paper
- **Conceptual Scope:** Motivation of the DeepTensor framework; coupling Deep Image Prior with low-rank tensor factorization; parameter efficiency over monolithic 3D DIP; original formulation by Heckel & Soltanolkotabi.
- **Primary Source Code:** `src/deeptensor/models/deeptensor.py` (Lines 232–286)
  - `DeepTensor` (L232–L253): Factorizes a 3D tensor into a 2D spatial factor matrix $U$ and 1D spectral factor matrix $V$.
  - `DeepTensorCP` (L254–L286): Factorizes a 3D tensor into three 1D factor generator networks $u, v, w$ combined with a learnable rank core scale.

### [DT-05] Tensor Fundamentals
- **Conceptual Scope:** Multilinear algebra foundations: orders, fibers, slices, matricization, outer product $\circ$, Khatri-Rao product $\odot$, rank-one components, and tensor rank vs matrix rank.
- **Primary Source Code:** `src/deeptensor/models/deeptensor.py` (Lines 244–286)
  - Demonstrates batch matrix contraction `torch.bmm(U, V)` for 2-way matrix product ($HW \times R \times C$).
  - Demonstrates multilinear contraction `torch.einsum('r, ir, jr, kr -> ijk', ...)` constructing rank-1 outer product sums $\sum_r \lambda_r u_r \otimes v_r \otimes w_r$.

### [DT-06] U, V and W Factor Generation
- **Conceptual Scope:** How neural factor generators produce column factor matrices; tracking latent input dimensions through conv blocks; structural interpretation of spatial factor $U(H \cdot W \times R)$, spectral factor $V(R \times C)$, and temporal/slice factor $W(K \times R)$.
- **Primary Source Code:** `src/deeptensor/models/deeptensor.py` (Lines 244–252 & Lines 274–286)
  - `U_img = self.spatial_net(spatial_inp)`: Downsamples and upsamples 2D latent grid to shape $(B, R, H, W)$.
  - `V = self.spectral_net(spectral_inp)`: Generates 1D spectral curve of shape $(B, R, C)$.
  - `U = U_img.reshape(B, R, H * W).permute(0, 2, 1)`: Flattens spatial coordinates into factor matrix $(B, HW, R)$.
  - `reconstruction = torch.bmm(U, V)`: Bilinear contraction producing the full spatial-spectral cube $(B, HW, C)$.

### [DT-07] Random Latent Initialization
- **Conceptual Scope:** Seed control, uniform latent tensor scaling, Fourier feature positional encodings, parameter initialization, and latent perturbation during training ($z + \mathcal{N}(0, \sigma^2)$).
- **Primary Source Code:** `src/deeptensor/utils/utils_legacy.py` (Lines 330–390)
  - `get_inp(tensize, const=10.0, device=None)` (L330–L339): Generates uniform noise scaled by $0.1$ as a leaf `torch.nn.Parameter`.
  - `get_2d_posencode_inp(H, W, n_inputs, device=None)` (L341–L365): Generates random Fourier frequency projections for 2D spatial coordinates.
  - `get_1d_posencode_inp(H, n_inputs, device=None)` (L367–L390): Generates 1D sinusoidal/cosinusoidal Fourier positional features.

### [DT-08] CP/PARAFAC Decomposition
- **Conceptual Scope:** CANDECOMP/PARAFAC canonical polyadic formulation:
  $$\mathcal{X} \approx \sum_{r=1}^R \lambda_r (u_r \circ v_r \circ w_r)$$
  Parameter complexity reduction from $O(I \cdot J \cdot K)$ to $O((I + J + K) \cdot R)$; unicity of CP representation.
- **Primary Source Code:** `src/deeptensor/models/deeptensor.py` (Lines 254–286)
  - `class DeepTensorCP(torch.nn.Module)` (L254–L273): Holds three 1D generators `u_net`, `v_net`, `w_net` and a learnable core vector $\lambda \in \mathbb{R}^R$.
  - `forward(...)` (L274–L286): Executes outer-product contraction using PyTorch einsum: `torch.einsum('r, ir, jr, kr -> ijk', scale, u_fac, v_fac, w_fac)`.

### [DT-09] Tucker Decomposition Audit
- **Conceptual Scope:** Theoretical formulation of Tucker decomposition:
  $$\mathcal{X} = \mathcal{G} \times_1 \mathbf{U} \times_2 \mathbf{V} \times_3 \mathbf{W}$$
  Investigation of whether neural generative Tucker was implemented in the deep learning pipeline.
- **Primary Source Code:** `src/deeptensor/baselines/spectral.py` (Lines 39–60)
  - `tucker_decompose(...)`: Uses TensorLy for classical truncated Tucker decomposition.
- **Audit Verification Result:** Neural generative Tucker is **NOT implemented** in DeepTensor's deep neural networks. The CP/PARAFAC model was specifically chosen to avoid the cubic $O(R^3)$ parameter explosion of a dense neural core tensor $\mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times R_3}$.

### [DT-10] Dimensionality Splitting Approaches
- **Conceptual Scope:** Comparative architectural analysis of three approaches to high-dimensional generative priors:
  1. Monolithic 3D Generator: $f_\theta(z) \to \mathbb{R}^{I \times J \times K}$
  2. 2D Spatial + 1D Spectral: $f_{\theta_1}(z_1) \otimes f_{\theta_2}(z_2)$
  3. Three 1D CP Generators: $f_{\theta_1}(z_1) \otimes f_{\theta_2}(z_2) \otimes f_{\theta_3}(z_3)$
- **Primary Source Code:** `src/deeptensor/models/deeptensor.py`
  - Approach 1: `UNetND(ndim=3)` (Lines 112–231).
  - Approach 2: `DeepTensor` (Lines 232–253).
  - Approach 3: `DeepTensorCP` (Lines 254–286).

### [DT-11] Original DeepTensor CNN Architecture
- **Conceptual Scope:** Dmitry Ulyanov's Skip U-Net backbone adapted for 1D and 2D tensor factor generation; reflection padding, LeakyReLU activations, BatchNorm1d/2d, bilinear/linear upsampling, and skip-concatenations.
- **Primary Source Code:** `src/deeptensor/models/backbones/skip.py`
  - `skip(...)` (Lines 5–101): 2D spatial skip network generator.
  - `skip1d(...)` (Lines 102–197): 1D spectral/temporal skip network generator.

### [DT-12] Complete Reconstruction Workflow
- **Conceptual Scope:** End-to-end execution pipeline:
  Observation loading $\to$ Latent generation $\to$ Factor network evaluation $\to$ Tensor contraction $\to$ Physics forward projection $\to$ Measurement loss computation $\to$ Gradient backpropagation $\to$ Clean evaluation inference.
- **Primary Source Code:** `experiments/reproduction/figure09_hyperspectral/run_figure9.py` (Lines 130–240)
  - Data loading from `data/icvl.mat`.
  - Noise simulation: Poisson ($\tau = 100$) + Gaussian ($\text{SNR} = 2.0$).
  - Dual-network forward pass producing $U$ and $V$.
  - Matrix contraction and L1 measurement loss against noisy observation.

### [DT-13] Training Loop and Backpropagation
- **Conceptual Scope:** PyTorch autograd graph computation, Adam optimizer state updates, input perturbation ($z + \epsilon$), exponential running average logging, checkpoint selection based on measurement loss.
- **Primary Source Code:** `experiments/reproduction/figure09_hyperspectral/run_figure9.py` (Lines 170–240)
  - `optimizer.zero_grad()` $\to$ `loss.backward()` $\to$ `optimizer.step()`.
  - Input perturbation: `spatial_inp_pert = spatial_inp + torch.randn_like(...) * reg_noise_std`.

### [DT-14] Traditional Methods and Baseline Comparisons
- **Conceptual Scope:** Experimental role of non-deep baselines: Truncated SVD, Classical Tucker, and 3D Total Variation; isolating the benefit of deep neural priors over purely linear or piecewise-smooth models.
- **Primary Source Code:** `src/deeptensor/baselines/spectral.py` (Lines 12–60)
  - Evaluated under identical measurement conditions to verify whether deep prior networks provide a statistically significant PSNR advantage.

### [DT-15] Five Model Architectures
- **Conceptual Scope:** Controlled comparative architecture framework:
  1. Baseline CNN (`skip1d`)
  2. Hybrid CNN-Transformer (`HybridSpectralNet` with `SpectralSelfAttention`)
  3. CNN Control (`HybridSpectralNet` with `SpectralConvControl`)
  4. Pointwise Control (`HybridSpectralNet` with `SpectralPointwiseControl`)
  5. Pure Transformer (`SpectralTransformer` with Pre-LN and ALiBi)
- **Primary Source Code:**
  - `src/deeptensor/models/factory.py` (Lines 145–230): Instantiation dispatcher.
  - `src/deeptensor/models/attention/spatial_spectral.py` (Lines 14–230): Attention, Pointwise, and CNN Control blocks.
  - `src/deeptensor/models/attention/spectral_transformer.py` (Lines 44–111): Pure Spectral Transformer.

### [DT-16] Fair Comparative Evaluation
- **Conceptual Scope:** Methodological rigor: parameter matching ($\pm 0.05\%$), identical latent seeds, frozen measurement observations, identical training epoch budgets, and consistent DIP evaluation conventions.
- **Primary Source Code:**
  - `src/deeptensor/models/factory.py` (Lines 29–32): `count_parameters` guaranteeing identical capacity.
  - `src/deeptensor/utils/comparison.py` (Lines 1–80): Cross-architecture metric verification engine.

### [DT-17] Figure 09 — ICVL Hyperspectral Image Denoising
- **Conceptual Scope:** ICVL dataset ($174 \times 164 \times 260$), Poisson + Gaussian noise, Rank $R=20$, 5000 epochs, L1 loss.
- **Historical Result:** Baseline CNN wins (**46.07 dB**) over Hybrid (44.85 dB) and Transformer (43.40 dB).
- **Primary Source Code:** `experiments/reproduction/figure09_hyperspectral/run_figure9.py` (Lines 1–341).
- **Verified Result JSON:** `results/hybrid/figure09_hyperspectral/figure09_icvl_hsi_comparison.json`.

### [DT-18] Figure 10 — Extended Yale Face B
- **Conceptual Scope:** Yale Face B dataset ($192 \times 168 \times 160$), 30% Salt & Pepper noise, CP Rank $R=2000$, 1200 epochs, L1 loss.
- **Historical Result:** Pure Transformer wins (**28.56 dB**) over Baseline CNN (27.78 dB).
- **Primary Source Code:** `experiments/reproduction/figure10_yale_denoising/run_figure10_yale_tensor.py` (Lines 1–363).
- **Verified Result JSON:** `results/hybrid/figure10_yale_denoising/figure10_yale_b_comparison.json`.

### [DT-19] Figure 11 — Video Compressive Sensing
- **Conceptual Scope:** CACTI video compressive sensing ($128 \times 128 \times 64$), 8-frame multiplexing, Rank $R=32$, 3000 epochs, L2 loss.
- **Historical Result:** Baseline CNN wins (**25.16 dB**) over Hybrid (24.74 dB) and Transformer (24.05 dB).
- **Primary Source Code:** `experiments/hybrid_prototypes/run_hybrid_figure11_video.py` (Lines 1–398).
- **Verified Result JSON:** `results/hybrid/figure11_video_cs/figure11_video_cs_comparison.json`.

### [DT-20] Figure 12 — PET/CT Radon Compressive Sensing
- **Conceptual Scope:** 3D PET/CT volume ($128 \times 128 \times 56$), 40-angle Radon transform, CP Rank $R=1000$, 4000 epochs, L2 loss.
- **Historical Result:** Pointwise Control wins (**30.62 dB**) over Transformer (30.41 dB) and Baseline CNN (30.00 dB).
- **Primary Source Code:** `experiments/hybrid_prototypes/run_hybrid_figure12_pet.py` (Lines 1–416).
- **Verified Result JSON:** `results/hybrid/figure12_pet_ct/figure12_pet_ct_comparison.json`.

### [DT-21] Cross-Application Comparative Results
- **Conceptual Scope:** Synthesis of all 4 testbeds across all 5 models; analysis of why no single architecture dominates; matching inductive biases to physical signal characteristics.
- **Primary Source Code:** `results/hybrid/` comparison files and `reports/reproduction/REPORTED_METRICS.md`.

### [DT-22] Adaptive Rank Selection
- **Conceptual Scope:** Dynamic rank expansion driven by reconstruction loss plateaus; $R_{\text{start}} = 4$, $R_{\text{max}} = 30$, $R_{\text{step}} = 2$, EMA smoothing $\alpha = 0.99$; Adam state reset.
- **Primary Source Code:** `experiments/extensions/adaptive_rank/run_figure9_adaptive.py` (Lines 64–140).
- **Verified Result:** **46.1571 dB PSNR** on ICVL (+0.09 dB over fixed Rank-20).

### [DT-23] Factor Group-Sparsity
- **Conceptual Scope:** Group Lasso regularization across joint spatial-spectral factor norms: $\sum_r \sqrt{\|U_r\|_2^2 + \|V_r\|_2^2}$; effective rank reduction.
- **Primary Source Code:** `experiments/extensions/group_sparsity/run_figure9_sparsity.py` (Lines 83–120).
- **Verified Sweep:** $\lambda \in \{0.0, 10^{-4}, 10^{-3}, 10^{-2}\}$ in `results/extensions/sparsity/standard/`.

### [DT-24] Limitations and Future Work
- **Conceptual Scope:** Algorithmic trade-offs, computation times, single-instance optimization cost, lack of generalization across different applications, and roadmap for adaptive gating (AHD-Net v1/v2).
- **Primary Source Document:** `reports/extensions/ADAPTIVE_HYBRID_V1_FINAL_VALIDATION.md` (Lines 1–150).

