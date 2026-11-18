# DeepTensor Figure 9 Official Baseline Execution Report

## 1. Purpose of Experiment
The goal of this experiment is to reproduce the official baseline implementation of **DeepTensor** on Figure 9 from the paper:
*"DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors"* (arXiv:2204.03145).

The target task is **Hyperspectral Image Denoising** using low-rank matrix/tensor factorization with Deep Image Prior (DIP) factor networks.

---

## 2. Experimental Environment

* **Hardware**: HP Omen Laptop
* **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
* **RAM**: 16 GB
* **OS**: Windows
* **Python**: 3.11.9 (Virtual environment: `.venv`)
* **PyTorch**: 2.5.1+cu121
* **CUDA**: 12.1

---

## 3. Experimental Configuration

* **Target Script**: `run_figure9.py`
* **Dataset**: ICVL Hyperspectral Dataset (`data/icvl.mat`)
* **Tensor / Matrix Rank**: `rank = 20`
* **Spatial Factor Network**: 2D U-Net / Deep Image Prior (`deep_prior.get_net`, 5 scales, 128 skip channels)
* **Spectral Factor Network**: 1D U-Net / Deep Image Prior (`deep_prior.get_net`, 5 scales, 128 skip channels)
* **Noise Type & Parameters**: Mixed Poisson photon noise + Additive Gaussian readout noise (`noise_snr = 2`, `tau = 100`)
* **Optimizer**: Adam (`learning_rate = 1e-3`)
* **Loss Function**: L1 Loss ($||Y - \hat{X}||_1$)
* **Training Epochs**: `epochs = 5000`
* **Compatibility Notes**: `run_figure9.py` contains only the minimal compatibility update for `scikit-image >= 0.24.0` (`channel_axis=-1`, `data_range=cube.max() - cube.min()`).

---

## 4. Quantitative Results & Comparison with SVD

| Method / Baseline | Input Noisy | SVD Baseline (Rank 20) | **DeepTensor (5000 Epochs)** | Baseline Improvement ($\Delta$) |
|---|---|---|---|---|
| **PSNR (dB)** | 28.43 dB | 38.31 dB | **46.45 dB** | **+8.14 dB vs. SVD** |
| **SSIM** | - | 0.9169 | **0.9872** | **+0.0703 vs. SVD** |

---

## 5. Hardware Performance & Footprint

* **Total Execution Time**: `159.57 seconds` (~2.66 minutes)
* **Peak GPU VRAM Allocated**: `640.37 MB` (out of 8,192 MB available; ~7.8% utilization)
* **Training Speed**: ~31.3 iterations/sec

---

## 6. Independent Verification Summary

Two independent 5000-epoch runs were conducted on the same hardware:
* **Run 1**: DeepTensor PSNR = `46.37 dB`, SSIM = `0.9868`, Runtime = `159.42 s`
* **Run 2**: DeepTensor PSNR = `46.45 dB`, SSIM = `0.9872`, Runtime = `159.57 s`

**Verdict**: **PASS — results independently reproduced with zero errors.**
