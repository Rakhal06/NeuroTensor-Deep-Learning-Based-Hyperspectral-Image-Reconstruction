# DeepTensor Experimental Settings & Hyperparameters

This document specifies the exact hyperparameters, network layers, forward models, and loss functions for every experiment in the paper.

---

## 1. Network Architectures

### 2D Spatial Deep Image Prior (DIP)
- **Module:** `modules/deep_prior.py` (`get_net(skip2d)`)
- **Topology:** 5-scale U-Net with skip connections.
- **Layers:** Conv2d with $3 \times 3$ kernel, reflection padding, BatchNorm2d, LeakyReLU ($\alpha = 0.2$), Bilinear upsampling ($2\times$).
- **Channels:** 128 channels per skip layer (`skip_n33d=128`, `skip_n33u=128`).
- **Input:** Static 2D uniform / Gaussian noise tensor $z \in \mathbb{R}^{1 \times C_{\text{in}} \times H \times W}$ divided by constant 10.0.
- **Total Parameters:** $\approx 2.2\text{M}$ trainable parameters.

### 1D Spectral / Factor Deep Image Prior (DIP)
- **Module:** `modules/deep_prior.py` (`get_net(skip1d)`)
- **Topology:** 5-scale 1D U-Net with skip connections.
- **Layers:** Conv1d with kernel size 3, reflection padding, BatchNorm1d, LeakyReLU, Linear upsampling ($2\times$).
- **Channels:** 128 channels per skip layer.
- **Input:** Static 1D uniform / Gaussian noise tensor $z \in \mathbb{R}^{1 \times C_{\text{in}} \times L}$ divided by constant 10.0.
- **Total Parameters:** $\approx 0.8\text{M}$ trainable parameters.

### Under-Parameterized Deep Decoder
- **Module:** `modules/deep_decoder.py` (`decodernw1d`, `decodernw2d`)
- **Topology:** Non-convolutional or $1 \times 1$ conv layer stack with fixed non-learned bilinear upsampling, no skip connections.
- **Parameters:** $\approx 25\text{K}$ to $50\text{K}$ parameters.

---

## 2. Optimization Settings by Application

| Application | Optimizer | Learning Rate | Schedule | Epochs | Loss Function | Batch Norm Mode |
|---|---|---|---|---|---|---|
| **01: Synthetic Robustness (Fig 6)** | Adam | $10^{-4}$ | None (Fixed) | 1000 | $\ell_2$ MSE | `train()` |
| **01: LR Schedule (Tab 1)** | Adam | $10^{-2} \dots 10^{-5}$ | Fixed, Step, Exp, Cosine | 1000 | $\ell_2$ MSE | `train()` |
| **02: Robust PCA (Fig 7)** | Adam | $10^{-4}$ | None (Fixed) | 1000 | $\ell_2$ Frobenius | `train()` |
| **03: NMF (Tab 2)** | Adam | $10^{-3}$ | None (Fixed) | 2000 | $\ell_2 + \lambda \ell_1$ | `train()` |
| **04: Eigenfaces (Fig 2)** | Adam | $10^{-3}$ | None (Fixed) | 100 | $\ell_2$ Data Fidelity | `train()` |
| **05: Hyperspectral (Fig 9)** | Adam | $10^{-3}$ | None (Fixed) | 5000 | $\ell_1$ Norm | `train()` |
| **06: Yale Tensor (Fig 10)** | Adam | $10^{-3}$ | None (Fixed) | 2000 | $\ell_1$ Penalty | `train()` |
| **07: Video CS (Fig 11)** | Adam | $10^{-3}$ | None (Fixed) | 10,000 | Measurement $\ell_2$ | `train()` |
| **08: CT Recon (Fig 12)** | Adam | $10^{-3}$ | None (Fixed) | 2000 | Radon Domain $\ell_2$ | `train()` |
| **09: Speech NMF (Tab 3)** | Adam | $10^{-3}$ | None (Fixed) | 2000 | $\ell_2 + \lambda \ell_1$ | `train()` |

