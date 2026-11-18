import os
import json

ROOT = 'e:/Rakhal_Tensor'

with open('scratch/comprehensive_audit_data.json', encoding='utf-8') as f:
    data = json.load(f)

doc = """# DeepTensor++ — Dependency Graph & Import Map

> **Audit Date:** November 6, 2026  
> **Workspace Root:** `e:\\Rakhal_Tensor`  
> **Audit Status:** Verified against AST code inspection  
> **Scope:** Internal module linkages, third-party libraries, architectural instantiations, and technical debt.

---

## 1. High-Level Architectural Dependency Graph

```mermaid
graph TD
    subgraph ThirdParty["External Dependencies"]
        PyTorch["PyTorch (torch, torchaudio, torchvision)"]
        SciPy["SciPy (scipy.io, scipy.ndimage)"]
        NumPy["NumPy"]
        Matplotlib["Matplotlib / Seaborn"]
        SKImage["scikit-image (PSNR, SSIM)"]
        TensorLy["TensorLy (PARAFAC, Tensor Ops)"]
        CV2["OpenCV (cv2)"]
    end

    subgraph CoreModules["Core Modules (modules/)"]
        Utils["utils.py<br/>(Metrics, Grids, Encodings)"]
        Losses["losses.py<br/>(L1, L2, TVNorm)"]
        DeepPrior["deep_prior.py<br/>(Skip & DIP Initializers)"]
        DeepDecoder["deep_decoder.py<br/>(1D/2D Decoders)"]
        Models["models.py<br/>(DeepTensor, Coordinate MLPs)"]
        LinInverse["lin_inverse.py<br/>(Radon, Video CS)"]
        Spectral["spectral.py<br/>(SVD, Tucker ALS)"]
        Sched["cosine_annealing_with_warmup.py"]
        AttnSpec["attention_spectral.py"]
        TransSpec["spectral_transformer.py"]
        DeepModels["deep_models/<br/>(U-Net, ResNet, DCGAN, Skip)"]
    end

    subgraph ReproductionSuite["Paper Reproduction Suite (paper_reproduction/cnn_baseline/)"]
        SynDecomp["01_synthetic_decomposition/"]
        NMF["03_nmf/"]
        HSIDenoise["05_hyperspectral_denoising/"]
        YaleDenoise["06_yale_face_tensor_denoising/"]
        VideoCS["07_video_compressive_sensing/"]
        CTRecon["08_ct_reconstruction/"]
        SpeechNMF["09_speech_time_frequency/"]
    end

    subgraph Historical["Extensions (previous_neurotensor_experiments/)"]
        Ext1to9["Extensions 01 - 09"]
    end

    %% Internal Module Relationships
    DeepPrior --> DeepModels
    Models --> DeepPrior
    Models --> DeepDecoder
    
    %% Reproduction Usage
    SynDecomp --> Models
    SynDecomp --> Utils
    NMF --> Models
    NMF --> Utils
    HSIDenoise --> DeepPrior
    HSIDenoise --> Utils
    YaleDenoise --> DeepPrior
    YaleDenoise --> Losses
    YaleDenoise --> Utils
    VideoCS --> DeepPrior
    VideoCS --> LinInverse
    VideoCS --> Losses
    VideoCS --> Utils
    CTRecon --> DeepPrior
    CTRecon --> LinInverse
    CTRecon --> Losses
    CTRecon --> Sched
    SpeechNMF --> Models
    SpeechNMF --> Utils

    %% Extensions Usage
    Ext1to9 --> DeepPrior
    Ext1to9 --> Losses
    Ext1to9 --> AttnSpec
    Ext1to9 --> TransSpec
```

---

## 2. Internal Module Linkage & Utilization Matrix

| Core Module | File Path | Total Callers | Primary Importers |
|---|---|---|---|
| `utils` | `modules/utils.py` | **45 scripts** | Imported across almost all runners for metric calculation (`psnr`, `ssim`), noise addition, and image grids. |
| `deep_prior` | `modules/deep_prior.py` | **40 scripts** | Standard DIP model creation (`skip()`, `get_net()`, `fill_noise()`) used in Yale, Video CS, CT, and HSI. |
| `losses` | `modules/losses.py` | **40 scripts** | Used for $L_1$, $L_2$, and anisotropic Total Variation (`TVNorm`) regularization. |
| `models` | `modules/models.py` | **31 scripts** | Encapsulates `DeepTensor` class, forward MLP coordinates, and factor aggregation. |
| `deep_decoder` | `modules/deep_decoder.py` | **17 scripts** | Used by `models.py` and synthetic decomposition benchmarks for non-convolutional factorization. |
| `lin_inverse` | `modules/lin_inverse.py` | **7 scripts** | Forward measurement operators: `radon` (CT) and `video2codedvideo` (Video CS). |
| `spectral` | `modules/spectral.py` | **4 scripts** | Low-rank SVD and Tucker decomposition baselines. |
| `cosine_annealing...` | `modules/cosine_annealing_with_warmup.py`| **2 scripts** | Cosine Annealing LR scheduler with warmup used in Figure 12 improved run. |
| `attention_spectral` | `modules/attention_spectral.py` | **2 scripts** | Spatial-spectral self-attention mechanism in ICVL Extension 8. |
| `spectral_transformer`| `modules/spectral_transformer.py` | **2 scripts** | ALiBi transformer backbone in ICVL Extension 9. |
| `deep_models/skip` | `modules/deep_models/skip.py` | **40 scripts** | Core Skip architecture called via `deep_prior.skip()`. |

---

## 3. Third-Party Dependencies & Version Compatibility

| External Library | Import Count | Purpose in DeepTensor++ | Criticality |
|---|---|---|---|
| `torch` | **64 files** | Core deep learning framework, autograd, GPU CUDA tensors | Mandatory |
| `numpy` | **56 files** | Matrix operations, synthetic tensor generation, data conversion | Mandatory |
| `scipy` | **43 files** | Loading `.mat` files (`scipy.io.loadmat`), spline filtering, TV baselines | Mandatory |
| `matplotlib` | **42 files** | Plotting convergence curves, rank sweeps, and generating figures | Mandatory |
| `skimage` | **34 files** | Computing reference image metrics (`metrics.peak_signal_noise_ratio`, `ssim`) | Mandatory |
| `tqdm` | **17 files** | Progress bars during long training loops (up to 10,000 epochs) | Optional / UI |
| `cv2` (OpenCV) | **10 files** | Image resizing, color space conversions, and video frame manipulation | Mandatory |
| `tensorly` | **4 files** | Classical tensor algorithms: PARAFAC decomposition, tensor unfolding | Mandatory |
| `sklearn` | **4 files** | Linear regression, PCA baselines, classification accuracy (Eigenfaces) | Mandatory |
| `bm3d` | **3 files** | Classical BM3D image denoising benchmark | Optional Baseline |
| `torchaudio` | **2 files** | Speech Commands dataset loading and STFT spectrogram extraction | Mandatory |
| `torchvision` | **3 files** | MNIST and CIFAR-10 torchvision dataset loaders | Mandatory |
| `reportlab` | **1 file** | Automated PDF research report generator (`generate_pdf.py`) | Optional Tool |

---

## 4. Architectural Model Instantiation Map

| Model Architecture | Defining Module | Instantiated In | Hardware / Memory Profile |
|---|---|---|---|
| **DeepTensor (2D DIP Prior)** | `modules/deep_prior.py` (`skip`) | Figure 9 (HSI), Figure 10 (Yale), Figure 13(a, b) | GPU: $\\\\sim 1.2$ GB VRAM; 128 channels, 5 levels |
| **DeepTensor (Separable 2D+1D DIP)**| `modules/lin_inverse.py` & `deep_prior.py` | Figure 11 (Video Compressive Sensing) | GPU: $\\\\sim 2.1$ GB VRAM; 64 frame temporal factor |
| **DeepTensor (3D Radon DIP)** | `modules/deep_prior.py` & `lin_inverse.py` | Figure 12 (3D PET/CT Radon Tomography) | GPU: $\\\\sim 2.8$ GB VRAM; 40-angle projection adjoint |
| **Deep Decoder (Pixel-wise)** | `modules/deep_decoder.py` | Table 1 (Synthetic Grid), Table 2 (NMF) | GPU / CPU: Lightweight ($\\\\sim 150$k params) |
| **Total Variation (2D / 3D TV)** | `modules/losses.py` (`TVNorm`) | Classical baselines in Video CS & CT | CPU / GPU: Convex optimization ($1000$ iters) |
| **CUDA PARAFAC ALS** | Native Torch (`run_figure13b...improved.py`)| Figure 13(b) Classical PARAFAC Baseline | GPU: Alternating Least Squares ($R=10..4000$) |
| **Spatial-Spectral Attention Net**| `modules/attention_spectral.py` | Extension 8 (`run_figure9_hybrid.py`) | GPU: $\\\\sim 3.2$ GB VRAM; multi-head self-attention |
| **Spectral Transformer with ALiBi**| `modules/spectral_transformer.py` | Extension 9 (`run_figure9_spectral_transformer.py`)| GPU: $\\\\sim 3.5$ GB VRAM; relative positional bias |

---

## 5. Technical Debt & Code Duplication Analysis

### 5.1 Fragile `sys.path.append('modules')` Pattern
Across almost every script in `paper_reproduction/` and root, scripts use the fragile pattern:
```python
import sys, os
sys.path.append(os.path.abspath('modules'))
sys.path.append(os.path.abspath('.'))
```
- **Risk**: Breaks if run from any other directory; causes namespace collisions with external packages; prevents standard packaging (`pip install -e .`).
- **Remediation**: Structure `DeepTensor-Plus-Plus` with an installable `src/deeptensor/` layout.

### 5.2 Hardcoded CUDA Device Assumptions
Multiple legacy scripts contain hardcoded strings:
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Or direct:
tensor.cuda()
```
- **Remediation**: Standardize on a centralized device resolver (`deeptensor.utils.device.get_device(config)`).

### 5.3 Repeated Training Loops & Boilerplate
- The outer training loop (Adam optimizer, closure, MSE loss computation, metric evaluation every 100 epochs, CSV logging) is duplicated across ~15 runner scripts with minor variations.
- **Remediation**: Implement a unified `Trainer` class in `src/deeptensor/training/trainer.py` supporting hooks and callbacks.

### 5.4 Repeated Loss Implementations
- Total Variation (`TVNorm`) and custom $L_1$ penalties are defined in `modules/losses.py`, but several scripts re-implement 2D and 3D TV inline using `torch.diff`.
- **Remediation**: Centralize all loss objectives strictly in `src/deeptensor/losses/`.

### 5.5 Missing Standard `pyproject.toml`
- The current repository lacks a formal `pyproject.toml` or `setup.py`, relying entirely on loose folder structures and ad-hoc virtual environment packages.
- **Remediation**: Add a standard `pyproject.toml` with pinned dependency specifications and entry points.
"""

out_path = os.path.join(ROOT, "codebase_audit", "DEPENDENCY_AND_IMPORT_MAP.md")
with open(out_path, "w", encoding="utf-8") as fp:
    fp.write(doc.strip() + "\n")

print(f"Generated {out_path} ({len(doc.splitlines())} lines)")

