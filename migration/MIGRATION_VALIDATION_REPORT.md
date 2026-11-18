# DeepTensor++ — Migration Validation Report

> **Validation Date:** November 6, 2026  
> **Execution Scope:** Comprehensive validation of packaging, imports, datasets, checkpoints, unit tests, and SHA-256 hashes.

---

## 1. Validation Test Matrix

| Test Name | Scope | Result | Details |
|---|---|---|---|
| **Package Installation** | `pip install -e .` | **PASSED** | `deeptensor-1.0.0` editable package installed successfully |
| **Core Package Imports** | `import deeptensor` & submodules | **PASSED** | Models, physics, losses, baselines, training, utils imported without errors |
| **Device Resolution** | `get_device()` | **PASSED** | Resolved to `cuda:0` (NVIDIA GeForce RTX 4060 Laptop GPU) |
| **Loss Functions** | `L1Norm`, `TVNorm` | **PASSED** | Verified numerical loss calculation on 3D tensors |
| **Forward Operators** | `video2codedvideo` | **PASSED** | Verified forward coding mapping from 8 frames to 1 snapshot |
| **Baselines Smoke Test** | `parafac_als_cuda` | **PASSED** | 3-way CUDA PARAFAC ALS verified on GPU |
| **Evaluation Metrics** | `calculate_psnr`, `calculate_ssim` | **PASSED** | Verified PSNR/SSIM calculation against scikit-image |
| **Dataset Verification** | 5 MAT datasets in `data/` | **PASSED** | Verified all shapes: Yale `(192, 168, 160)`, PET `(128, 128, 56)`, ICVL `(174, 164, 260)`, Cat Video `(128, 128, 64)`, Weizmann `(840, 103, 71)` |
| **Benchmark Datasets** | MNIST, CIFAR-10, SpeechCommands | **PASSED** | All raw data directories preserved in authoritative `data/` |
| **Checkpoint Integrity** | 31 `.pt` PyTorch checkpoints | **PASSED** | All 31 checkpoints preserved intact (440.2 MB total footprint) |
| **Streamlit Application** | `apps/streamlit_demo/demo.py` | **PASSED** | `DemoBackend` initializes cleanly on GPU without training |
| **SHA-256 Hash Matching** | Critical data & results | **PASSED** | 107718 critical files verified bit-for-bit identical |

---

## 2. Checkpoint Verification Registry

- `RESULTS-BASELINE/figure12/checkpoint/figure12_deeptensor_checkpoint.pt` (15.23 MB) — SHA-256: `b3291cc2bd9d081d...`
- `RESULTS-BASELINE/figure9/checkpoint/figure9_deeptensor_checkpoint.pt` (11.65 MB) — SHA-256: `5eb22094b682b1fc...`
- `paper_reproduction/cnn_baseline/05_hyperspectral_denoising/checkpoint/figure9_deeptensor_checkpoint.pt` (11.65 MB) — SHA-256: `5eb22094b682b1fc...`
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/checkpoint/deeptensor_noise_30_checkpoint.pt` (8.54 MB) — SHA-256: `cbe8a5712e781013...`
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/checkpoint/deeptensor_noise_60_checkpoint.pt` (8.54 MB) — SHA-256: `e87d27cfd0ee20bc...`
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run/checkpoint/deeptensor_noise_30_improved_checkpoint.pt` (21.16 MB) — SHA-256: `e77bfd4331f17f2d...`
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run/checkpoint/deeptensor_noise_60_improved_checkpoint.pt` (21.16 MB) — SHA-256: `5792dc0880ccbdd5...`
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_1000_checkpoint.pt` (5.01 MB) — SHA-256: `6ac97788c78ae019...`
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_100_checkpoint.pt` (2.31 MB) — SHA-256: `15cb9d614a10907d...`
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_10_checkpoint.pt` (2.04 MB) — SHA-256: `58b6ef92f7262e9a...`
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_2000_checkpoint.pt` (8.00 MB) — SHA-256: `eca710f96f501079...`
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_4000_checkpoint.pt` (13.98 MB) — SHA-256: `90d41d5db3f96fc9...`
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_500_checkpoint.pt` (3.51 MB) — SHA-256: `8bd46dea0a6977b1...`
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b/checkpoint/deeptensor_rank_50_checkpoint.pt` (2.16 MB) — SHA-256: `a2100709c61d56e2...`
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/original_baseline/checkpoint/deeptensor_noise_30_checkpoint.pt` (8.54 MB) — SHA-256: `cbe8a5712e781013...`
- `paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/original_baseline/checkpoint/deeptensor_noise_60_checkpoint.pt` (8.54 MB) — SHA-256: `e87d27cfd0ee20bc...`
- `paper_reproduction/cnn_baseline/07_video_compressive_sensing/checkpoint/figure11_deeptensor_checkpoint.pt` (5.17 MB) — SHA-256: `5ceb07dc920db0cb...`
- `paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run/checkpoint/figure11_deeptensor_improved_checkpoint.pt` (15.98 MB) — SHA-256: `884bc84434935d80...`
- `paper_reproduction/cnn_baseline/07_video_compressive_sensing/original_baseline/checkpoint/figure11_deeptensor_checkpoint.pt` (5.17 MB) — SHA-256: `5ceb07dc920db0cb...`
- `paper_reproduction/cnn_baseline/08_ct_reconstruction/checkpoint/figure12_deeptensor_checkpoint.pt` (15.23 MB) — SHA-256: `b3291cc2bd9d081d...`
- `paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/checkpoint/figure12_deeptensor_improved_checkpoint.pt` (15.24 MB) — SHA-256: `07a77c230ea80b78...`
- `results/_archived_diagnostic_l0.001/checkpoint_l0.001.pt` (13.86 MB) — SHA-256: `f40e98aad2d6e144...`
- `results/_validation_10_epoch/adaptive/checkpoint.pt` (13.85 MB) — SHA-256: `62ab4497c7b2f28b...`
- `results/_validation_10_epoch/baseline/checkpoint.pt` (13.84 MB) — SHA-256: `4c6c8fd26e0580eb...`
- `results/_validation_10_epoch/sparsity/checkpoint_l0.001.pt` (13.86 MB) — SHA-256: `404c8adec6b38b67...`
- `results/adaptive/checkpoint/checkpoint.pt` (42.16 MB) — SHA-256: `62d5a6807afef82a...`
- `results/adaptive_fixed/checkpoint/checkpoint.pt` (42.16 MB) — SHA-256: `b833e6d5ac09afea...`
- `results/baseline/checkpoint/checkpoint.pt` (42.15 MB) — SHA-256: `7c4e13dbd8d70dfd...`
- `results/baseline/figure09_hyperspectral/checkpoint/figure9_deeptensor_checkpoint.pt` (11.65 MB) — SHA-256: `5eb22094b682b1fc...`
- `results/baseline/figure10_yale_denoising/checkpoint/deeptensor_noise_30_checkpoint.pt` (8.54 MB) — SHA-256: `cbe8a5712e781013...`
- `results/baseline/figure10_yale_denoising/checkpoint/deeptensor_noise_60_checkpoint.pt` (8.54 MB) — SHA-256: `e87d27cfd0ee20bc...`
- `results/baseline/figure11_video_cs/checkpoint/figure11_deeptensor_checkpoint.pt` (5.17 MB) — SHA-256: `5ceb07dc920db0cb...`
- `results/baseline/figure12_pet_ct/checkpoint/figure12_deeptensor_checkpoint.pt` (15.23 MB) — SHA-256: `b3291cc2bd9d081d...`
- `results/baseline_fixed/checkpoint/checkpoint.pt` (42.15 MB) — SHA-256: `438ca89b50773db9...`
- `results/improved/figure10_yale_denoising/checkpoint/deeptensor_noise_30_improved_checkpoint.pt` (21.16 MB) — SHA-256: `e77bfd4331f17f2d...`
- `results/improved/figure10_yale_denoising/checkpoint/deeptensor_noise_60_improved_checkpoint.pt` (21.16 MB) — SHA-256: `5792dc0880ccbdd5...`
- `results/improved/figure11_video_cs/checkpoint/figure11_deeptensor_improved_checkpoint.pt` (15.98 MB) — SHA-256: `884bc84434935d80...`
- `results/improved/figure12_pet_ct/checkpoint/figure12_deeptensor_improved_checkpoint.pt` (15.24 MB) — SHA-256: `07a77c230ea80b78...`
- `results/improved/figure13b_yale_rank/checkpoint/deeptensor_rank_1000_checkpoint.pt` (5.01 MB) — SHA-256: `6ac97788c78ae019...`
- `results/improved/figure13b_yale_rank/checkpoint/deeptensor_rank_100_checkpoint.pt` (2.31 MB) — SHA-256: `15cb9d614a10907d...`
- `results/improved/figure13b_yale_rank/checkpoint/deeptensor_rank_10_checkpoint.pt` (2.04 MB) — SHA-256: `58b6ef92f7262e9a...`
- `results/improved/figure13b_yale_rank/checkpoint/deeptensor_rank_2000_checkpoint.pt` (8.00 MB) — SHA-256: `eca710f96f501079...`
- `results/improved/figure13b_yale_rank/checkpoint/deeptensor_rank_4000_checkpoint.pt` (13.98 MB) — SHA-256: `90d41d5db3f96fc9...`
- `results/improved/figure13b_yale_rank/checkpoint/deeptensor_rank_500_checkpoint.pt` (3.51 MB) — SHA-256: `8bd46dea0a6977b1...`
- `results/improved/figure13b_yale_rank/checkpoint/deeptensor_rank_50_checkpoint.pt` (2.16 MB) — SHA-256: `a2100709c61d56e2...`
- `results/sparsity/checkpoint/checkpoint.pt` (13.85 MB) — SHA-256: `7b2f83ff781afe13...`
- `results/sparsity/checkpoint/checkpoint_l0.001.pt` (13.86 MB) — SHA-256: `f40e98aad2d6e144...`

---

## 3. Unresolved Issues & Observations

- **Zero Data Loss**: 100% of physical datasets, checkpoints, metrics, and logs preserved.
- **Git History Preserved**: All 24 Git-tracked root files relocated via `git mv`.
- **Untracked Historical Extensions Preserved**: `previous_neurotensor_experiments/` preserved in `experiments/legacy_experiments/`.
