# DeepTensor — Master Code Reference Table

**Document ID:** EVAL-REF-02  
**Scope:** Complete Index of Files, Classes, Functions, and Line Numbers Mapped to Presentation Slides

---

| Slide | Topic | File Path | Class / Function | Purpose | Important Lines |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **01** | Package Architecture | `pyproject.toml` | Build Configuration | Defines `deeptensor` package dependencies and layout | L1–L40 |
| **01** | Package Entry Point | `src/deeptensor/__init__.py` | Package Exports | Exposes core models, baselines, and utilities | L1–L25 |
| **02** | Deep Prior Network Factory | `src/deeptensor/models/deep_prior.py` | `get_net(...)` | Instantiates untrained convolutional DIP generators | L75–L150 |
| **02** | U-Net Generator Backbone | `src/deeptensor/models/backbones/skip.py` | `skip(...)` | Dmitry Ulyanov 2D Skip U-Net generator | L5–L101 |
| **02** | DeepTensor Matrix Formulation | `src/deeptensor/models/deeptensor.py` | `DeepTensor.forward` | Generates factors U, V and computes `torch.bmm(U, V)` | L232–L253 |
| **02** | Latent Random Noise | `src/deeptensor/utils/utils_legacy.py` | `get_inp(...)` | Generates uniform noise leaf tensor $Z$ | L330–L339 |
| **03** | Truncated SVD Baseline | `src/deeptensor/baselines/spectral.py` | `svd_decompose` | Linear low-rank SVD decomposition on matricized cubes | L12–L36 |
| **03** | Noise Robustness Benchmark | `experiments/reproduction/figure06_synthetic_robustness/run_figure6.py` | `add_noise` | Injects Gaussian, Poisson, and Rician noise distributions | L45–L80 |
| **04** | Truncated Tucker Baseline | `src/deeptensor/baselines/spectral.py` | `tucker_decompose` | Classical TensorLy Tucker decomposition baseline | L39–L60 |
| **04** | BM3D Denoising Baseline | `experiments/reproduction/figure09_hyperspectral/run_figure9_bm3d.py` | `bm3d.bm3d` | Band-by-band BM3D baseline execution | L45–L95 |
| **04** | 3D Total Variation Baseline | `experiments/reproduction/figure11_video_cs/run_figure11_TV.py` | `total_variation_3d` | Anisotropic 3D TV gradient projection | L30–L115 |
| **04** | 3D Monolithic CNN | `src/deeptensor/models/deeptensor.py` | `UNetND(ndim=3)` | Direct 3D volumetric U-Net baseline | L112–L231 |
| **05** | Training Loop & Optimization | `experiments/reproduction/figure09_hyperspectral/run_figure9.py` | Training Loop | Adam optimizer loop, L1 loss, backpropagation | L170–L240 |
| **05** | Loss Objectives | `src/deeptensor/losses/objectives.py` | `L1Norm`, `MSELoss` | Data fidelity objectives in measurement space | L10–L45 |
| **06** | CP Factor Generator | `src/deeptensor/models/deeptensor.py` | `DeepTensorCP` | Instantiates 3 factor nets and executes CP contraction | L254–L286 |
| **06** | 1D Skip Factor Network | `src/deeptensor/models/backbones/skip.py` | `skip1d(...)` | Generates 1D factor curves for CP decomposition | L102–L197 |
| **07** | CP Multilinear Contraction | `src/deeptensor/models/deeptensor.py` | `DeepTensorCP.forward` | `torch.einsum('r, ir, jr, kr -> ijk', ...)` | L274–L286 |
| **07** | Tensor Operations | `src/deeptensor/utils/tensor_ops.py` | `cp_to_tensor` | Tensor contraction, mode matricization | L15–L40 |
| **08** | HSI Reproduction Pipeline | `experiments/reproduction/figure09_hyperspectral/run_figure9.py` | Main script | ICVL reproduction under Poisson + Gaussian noise | L50–L320 |
| **08** | Yale Face B Reproduction | `experiments/reproduction/figure10_yale_denoising/run_figure10_yale_tensor.py` | Main script | Yale Face B CP decomposition under 30% S&P noise | L40–L350 |
| **09** | CACTI Optical Operator | `src/deeptensor/physics/operators.py` | `video2codedvideo` | Simulates 8-frame shifting optical mask compression | L72–L97 |
| **09** | Forward Radon Operator | `src/deeptensor/physics/operators.py` | `radon` | Parallel-beam tomographic projection across angles | L28–L47 |
| **10** | Improved Yale Face B Rerun | `experiments/improvements/figure10_yale/run_figure10_yale_tensor_improved.py` | Main script | 5000-epoch extended rerun (30.86 dB & 30.46 dB) | L115–L240 |
| **11** | Video CS Reconstruction Plot | `experiments/reproduction/figure11_video_cs/run_figure11.py` | `plot_reconstructions` | Visualizes 8 video frames vs 2D TV | L220–L280 |
| **11** | PET/CT Reconstruction Plot | `experiments/reproduction/figure12_pet_ct/run_figure12.py` | `plot_ct_slices` | Visualizes Slice 28 tomographic reconstruction | L240–L310 |
| **12** | Master Baseline Metrics | `reports/reproduction/REPORTED_METRICS.md` | Summary Table | Canonical table comparing paper vs our baselines | L15–L65 |
| **13** | Multi-Model Factory | `src/deeptensor/models/factory.py` | `build_spectral_net` | Instantiates and parameter-matches all five models | L145–L230 |
| **13** | Parameter Matching Engine | `src/deeptensor/models/factory.py` | `count_parameters` | Validates parameter count matching within $\pm 0.05\%$ | L29–L32 |
| **13** | Hybrid Attention Block | `src/deeptensor/models/attention/spatial_spectral.py` | `SpectralSelfAttention` | Multi-Head Self-Attention coupled with CNN backbone | L30–L95 |
| **13** | Parameter-Matched CNN Control| `src/deeptensor/models/attention/spatial_spectral.py` | `SpectralConvControl` | Deep convolutional control matched to Hybrid params | L140–L190 |
| **14** | Pointwise Control Block | `src/deeptensor/models/attention/spatial_spectral.py` | `SpectralPointwiseControl` | $1 \times 1$ pointwise channel-mixing convolutions | L192–L230 |
| **14** | Pure Transformer Backbone | `src/deeptensor/models/attention/spectral_transformer.py` | `SpectralTransformer` | Pure Transformer with Pre-LN and ALiBi | L44–L111 |
| **15** | HSI 5-Model Experiment | `experiments/hybrid_prototypes/run_hybrid_figure09_hyperspectral.py` | Main script | Runs 5 models on ICVL HSI (Baseline CNN wins 46.07 dB) | L1–L320 |
| **16** | HSI Visual Comparison Panel | `tools/generate_annotated_panels.py` | `generate_panel` | Generates annotated comparison panel and error maps | L40–L125 |
| **17** | Video CS 5-Model Experiment | `experiments/hybrid_prototypes/run_hybrid_figure11_video.py` | Main script | Runs 5 models on Video CS (Baseline CNN wins 25.16 dB) | L1–L398 |
| **18** | Video CS Visual Panel | `tools/generate_annotated_panels.py` | `generate_panel` | Generates 5-model Frame 32 comparison strip | L130–L210 |
| **19** | Yale Face B 5-Model Experiment| `experiments/hybrid_prototypes/run_hybrid_figure10_yale.py` | Main script | Runs 5 models on Yale B (Transformer wins 28.56 dB) | L1–L221 |
| **20** | PET/CT 5-Model Experiment | `experiments/hybrid_prototypes/run_hybrid_figure12_pet.py` | Main script | Runs 5 models on PET/CT (Pointwise wins 30.62 dB) | L1–L416 |
| **21** | PET/CT Visual Panel | `tools/generate_annotated_panels.py` | `generate_panel` | Generates 5-model Slice 28 comparison strip | L220–L310 |
| **22** | Cross-Application Matrix | `evaluation/code_navigator/data/experiment_map.json` | Cross-App Benchmark | Compiles winning models and metrics for all 4 tasks | L1–L120 |
| **23** | Adaptive Rank Selection Engine | `experiments/extensions/adaptive_rank/run_figure9_adaptive.py` | `update_rank` | Dynamically monitors EMA loss and grows active rank | L64–L140 |
| **23** | Adaptive Rank Manager Module | `src/deeptensor/training/adaptive_rank.py` | `AdaptiveRankManager` | Plateau detection and optimizer state re-allocation | L15–L85 |
| **24** | Adaptive Rank Evolution Plot | `results/adaptive_rank_selection/active_rank_evolution.png` | Artifact Plot | Trajectory plot showing rank growth $4 \to 6 \to 8$ | Binary |
| **24** | Adaptive Rank Results Summary | `results/adaptive_rank_selection/summary.txt` | Result Log | Logs final rank 10, best rank 8, PSNR 46.1571 dB | L1–L35 |
| **25** | Group Lasso Regularizer | `experiments/extensions/group_sparsity/run_figure9_sparsity.py` | `compute_group_regularizer` | Computes $\sum_r \sqrt{\|U_{:,r}\|_2^2 + \|V_{r,:}\|_2^2}$ | L83–L120 |
| **26** | Group Sparsity Metric Source | `results/factor_group-sparsity/metrics.json` | JSON Metrics File | Verified source for 40.39088 dB PSNR, 0.98616 SSIM | L1–L5 |
| **27** | Adaptive Hybrid Router Design | `reports/extensions/ADAPTIVE_HYBRID_V1_RESEARCH_DESIGN.md` | Specification Doc | Mathematical design for future adaptive routing | L1–L150 |
| **28** | Interactive Code Navigator | `evaluation/code_navigator/backend/server.py` | FastAPI Server | Powers interactive Code Navigator web interface | L1–L180 |

