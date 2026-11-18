# DeepTensor — Faculty Viva Defense Code Guide

**Document ID:** EVAL-VIVA-01  
**Target Audience:** Viva Examination Defense & Faculty Reviewers  
**Purpose:** Quick-Reference Cheat Sheet for Viva Defense with Direct File Paths, Line Numbers, and Clear Verbal Answers

---

## 1. Top 10 Most Critical Repository Files

| Rank | File Path | Primary Function / Class | Why You Must Know It |
| :---: | :--- | :--- | :--- |
| **1** | `src/deeptensor/models/deeptensor.py` | `DeepTensor`, `DeepTensorCP` | The core mathematical model implementing matrix & 3-way CP tensor contraction. |
| **2** | `src/deeptensor/models/backbones/skip.py` | `skip`, `skip1d` | The convolutional Deep Image Prior generator network (Ulyanov U-Net). |
| **3** | `src/deeptensor/models/factory.py` | `build_spectral_net`, `count_parameters` | The factory building and parameter-matching all five comparative models. |
| **4** | `src/deeptensor/models/attention/spatial_spectral.py` | `SpectralSelfAttention`, `SpectralPointwiseControl` | Contains the MHSA attention block and $1 \times 1$ pointwise convolution control. |
| **5** | `src/deeptensor/models/attention/spectral_transformer.py` | `SpectralTransformer` | The pure non-convolutional Transformer backbone using Pre-LN and ALiBi. |
| **6** | `src/deeptensor/physics/operators.py` | `radon`, `video2codedvideo` | Differentiable physical forward measurement operators (CT Radon & CACTI). |
| **7** | `experiments/extensions/adaptive_rank/run_figure9_adaptive.py` | `update_rank` | Implements plateau-driven dynamic bottom-up rank growth ($4 \to 6 \to 8$). |
| **8** | `experiments/extensions/group_sparsity/run_figure9_sparsity.py` | `compute_group_regularizer` | Implements Group Lasso regularizer $\sum_r \sqrt{\|U_r\|_2^2 + \|V_r\|_2^2}$ on factors. |
| **9** | `src/deeptensor/utils/metrics.py` | `psnr`, `ssim`, `rmse`, `mae` | Formal metric evaluation functions ensuring consistent calculation. |
| **10** | `experiments/hybrid_prototypes/run_hybrid_figure10_yale.py` | Main script | Driver for the 5-architecture benchmark on Yale Face B (Transformer winning at 28.56 dB). |

---

## 2. Likely Faculty Viva Questions & Model Answers

### Question 1: "Where is CP/PARAFAC tensor decomposition actually implemented?"
- **File:** `src/deeptensor/models/deeptensor.py`
- **Lines:** Lines 254–286 (`class DeepTensorCP`)
- **Key Code Line (L282–L285):**
  ```python
  recon = torch.einsum('r, ir, jr, kr -> ijk', scale, u_fac, v_fac, w_fac)
  ```
- **Verbal Answer:**
  > *"CP decomposition is implemented in `DeepTensorCP` inside `src/deeptensor/models/deeptensor.py`. The class instantiates three separate 1D generator networks—`u_net`, `v_net`, and `w_net`—together with a learnable rank core scale vector. In the `forward()` method, each network outputs a factor matrix: $U \in \mathbb{R}^{I \times R}$, $V \in \mathbb{R}^{J \times R}$, and $W \in \mathbb{R}^{K \times R}$. We then perform multilinear rank-1 outer product summation across rank index $r$ using PyTorch's `torch.einsum`."*

---

### Question 2: "How are the factors generated from random noise?"
- **File:** `src/deeptensor/utils/utils_legacy.py` and `src/deeptensor/models/deeptensor.py`
- **Lines:** `utils_legacy.py` (L330–L339), `deeptensor.py` (L244–L252)
- **Verbal Answer:**
  > *"Fixed random uniform noise tensors $Z_U, Z_V, Z_W \sim \mathcal{U}(-0.1, 0.1)$ are instantiated as non-trainable leaf parameters. During each forward pass, these latent inputs are fed into the respective generator networks ($f_U, f_V, f_W$). The networks use convolutional downsampling and upsampling with skip connections to map the noise latents into structured, smooth low-rank factor vectors. The network weights are updated via self-supervised backpropagation directly from the corrupted observation."*

---

### Question 3: "Where is the convolutional generator defined?"
- **File:** `src/deeptensor/models/backbones/skip.py`
- **Lines:** Lines 5–101 (`skip` 2D) and Lines 102–197 (`skip1d` 1D)
- **Verbal Answer:**
  > *"The convolutional generator uses Dmitry Ulyanov's Deep Image Prior Skip U-Net architecture, located in `src/deeptensor/models/backbones/skip.py`. It features an encoder-decoder hierarchy with reflection padding, $3 \times 3$ (or 1D kernel 3) convolutions, LeakyReLU non-linearities, BatchNorm, and linear upsampling, coupled with lateral skip connections that concatenate encoder feature maps directly to decoder layers."*

---

### Question 4: "Where is Transformer self-attention implemented, and how does it compare to the CNN?"
- **Files:** `src/deeptensor/models/attention/spatial_spectral.py` and `src/deeptensor/models/attention/spectral_transformer.py`
- **Lines:** `spatial_spectral.py` (L30–L95: `SpectralSelfAttention`), `spectral_transformer.py` (L44–L111: `SpectralTransformer`)
- **Verbal Answer:**
  > *"Multi-Head Self-Attention is implemented in `SpectralSelfAttention` using scaled dot-product attention across factor tokens. For our Pure Transformer model, `SpectralTransformer` implements a non-convolutional stack using Pre-LayerNorm and ALiBi (Attention with Linear Biases) to encode relative token distances. Unlike CNNs which rely on local receptive fields, self-attention computes pairwise token affinities globally across all coordinates simultaneously."*

---

### Question 5: "Where is the forward physics operator for CT Radon projection implemented?"
- **File:** `src/deeptensor/physics/operators.py`
- **Lines:** Lines 28–47 (`def radon`)
- **Verbal Answer:**
  > *"The forward Radon projection operator is in `src/deeptensor/physics/operators.py`. It projects a 2D image slice across 40 equi-spaced angles between $0$ and $\pi$ using differentiable bilinear grid sampling and rotation matrices. In the PET/CT experiment, this operator is placed in the forward loop so that the measurement $\mathcal{L}_2$ loss is computed directly in sinogram projection space."*

---

### Question 6: "Where is Adaptive Rank Selection implemented, and how does it decide when to grow the rank?"
- **Files:** `experiments/extensions/adaptive_rank/run_figure9_adaptive.py` (L64–L140) and `src/deeptensor/training/adaptive_rank.py` (L15–L85)
- **Verbal Answer:**
  > *"Adaptive Rank Selection starts training with a small rank ($R=4$) and tracks reconstruction loss using an Exponential Moving Average with smoothing parameter $\alpha = 0.99$. Every 200 epochs, the `AdaptiveRankManager` checks whether the rate of loss improvement has plateaued ($\Delta \mathcal{L} < 0.001$). If a plateau is detected, it dynamically appends 2 new rank components to the factor generators, resets the optimizer states, and resumes training until reaching the optimal rank ($R=8$ on ICVL, achieving 46.1571 dB)."*

---

### Question 7: "Where is Factor Group-Sparsity (Group Lasso) implemented?"
- **File:** `experiments/extensions/group_sparsity/run_figure9_sparsity.py`
- **Lines:** Lines 83–120 (`compute_group_regularizer`)
- **Verbal Answer:**
  > *"Group Lasso is implemented in `run_figure9_sparsity.py`. For an over-parameterized rank $R_\text{max}=30$, it computes the Euclidean norm across the $r$-th spatial column $U_{:,r}$ and spectral row $V_{r,:}$: $g_r = \sqrt{\|U_{:,r}\|_2^2 + \|V_{r,:}\|_2^2}$, and sums these norms across all $r$ to form the regularization penalty $\mathcal{R}(U,V)$. This drives uninformative factor pairs toward zero during backpropagation."*

---

### Question 8: "Where are PSNR and SSIM calculated?"
- **File:** `src/deeptensor/utils/metrics.py` (and `comparison.py`)
- **Lines:** `metrics.py` Lines 15–70 (`psnr`, `ssim`, `rmse`, `mae`)
- **Verbal Answer:**
  > *"PSNR, SSIM, RMSE, and MAE are implemented in `src/deeptensor/utils/metrics.py`. PSNR is calculated using $10 \log_{10}(\text{MAX}^2 / \text{MSE})$ with dynamic range $\text{MAX}=1.0$. SSIM is calculated using Scikit-Image's `structural_similarity` on 2D slices with identical window parameters to prevent evaluation discrepancies."*

---

### Question 9: "Initially we achieved 30.86 dB on Yale Face B, but in the 5-model comparison it was 27.78 dB. Why is there a mismatch?"
- **Files:** `results/improved/figure10_yale_denoising/progression_noise_30.csv` and `results/hybrid/figure10_yale_denoising/figure10_yale_b_comparison.json`
- **Verbal Answer:**
  > *"There is no mismatch or model regression. The difference is purely the training budget: 5,000 epochs versus 1,200 epochs. The 5-model comparison was run for the standard paper benchmark budget of 1,200 epochs to keep computation fair across all 5 models. In the 5,000-epoch progression log, at epoch 1,200, the PSNR was exactly 27.74 dB—matching the 1,200-epoch Baseline CNN (27.78 dB) within 0.04 dB. Between epochs 1,200 and 5,000, DeepTensor climbs continuously from 27.74 dB to 30.86 dB."*

---

### Question 10: "Why did Pointwise Control win on PET/CT while CNN won on Hyperspectral and Video?"
- **Files:** `results/hybrid/figure12_pet_ct/figure12_pet_ct_comparison.json` and `results/hybrid/figure09_hyperspectral/figure09_icvl_hsi_comparison.json`
- **Verbal Answer:**
  > *"Because inductive bias must match the physical characteristics of the degradation. Hyperspectral images and natural videos have smooth spatial and temporal manifolds, so CNN local convolutional kernels excel. In PET/CT, however, reconstructing from 40 sparse Radon angles introduces high-frequency streak artifacts across the volume. Spatial convolutions blur and spread these streaks across neighboring pixels, whereas Pointwise Control ($1 \times 1$ channel-mixing) operates along the fiber dimension without spatial neighborhood blurring, resulting in the highest PSNR (30.62 dB) and lowest MAE (0.0177)."*

