# NeuroTensor Reproducibility Protocol & Environment Audit

This document defines the computational environment, dependency specifications, dataset integrity standards, and command-line execution protocols required to reproduce every experiment in this repository.

---

## 1. Hardware & System Specifications

- **Primary Compute Platform:** HP Omen Gaming Laptop
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
- **VRAM:** 8,188 MB (8 GB GDDR6)
- **Host CPU:** Intel Core / AMD x86_64
- **Host Operating System:** Microsoft Windows 11 Home / Pro (64-bit)
- **Local Timezone:** Asia/Kolkata (UTC+05:30)
- **Workspace Directory:** `E:\Rakhal_Tensor\`

---

## 2. Software & Toolchain Configuration

- **Python Runtime:** Python 3.11.9 (`E:\Rakhal_Tensor\.venv\Scripts\python.exe`)
- **PyTorch:** `2.5.1+cu121`
- **CUDA Runtime:** CUDA 12.1 (Compute Capability `sm_89`, Ada Lovelace)
- **CuDNN:** PyTorch bundled `cudnn 9.1`
- **Core Scientific Libraries:**
  - `numpy`: 1.26.4
  - `scipy`: 1.14.1
  - `scikit-learn`: 1.5.2
  - `scikit-image`: 0.24.0
  - `tensorly`: 0.8.1
  - `kornia`: 0.8.3
  - `bm3d`: 4.0.3
  - `opencv-python`: 4.10.0
  - `torchaudio`: 2.5.1+cu121
  - `torchvision`: 0.20.1+cu121
  - `matplotlib`: 3.9.2

---

## 3. Dataset Integrity Checksums (SHA-256)

| Dataset File | File Size (Bytes) | SHA-256 Checksum | Dimensions / Resolution | Verification Method |
|---|---|---|---|---|
| `data/icvl.mat` | 11,914,846 | `43fe3ef247e69c57d7fe89f76fbdb601f7c1979d95f87b3a9cbca977119f96b2` | $174 \times 164 \times 260$ | `scipy.io.loadmat` |
| `data/weizzman.mat` | 5,633,264 | `b99e0df2dc36c0a0c6a59b66bbef1310ad4592fe528340d046c3fb8c80d5d95e` | $103 \times 71 \times 840$ | `scipy.io.loadmat` |
| `data/pet.mat` | 1,353,604 | `f6a45741639d6ad7a0487440266da59d2a453f6834b6b1fc923ffc5d8aa1d48c` | $128 \times 128 \times 56$ | `scipy.io.loadmat` |
| `data/cat_video.mat` | 823,260 | `6c12d46e3a0937c569ff45fbc5b3fe8db3ecae663a8beeb5c5539d06b63795ba` | $128 \times 128 \times 8$ | `scipy.io.loadmat` |
| `data/yale_b.mat` | 5,165,072 | `7d9ddabf2caebc95412b0ad72a6580eb076066c06b89fa8b3ec083b6934cd233` | $192 \times 168 \times 160$ | `scipy.io.loadmat` |
| Google Speech Commands | ~2.3 GB (Local Cache) | v0.02 Release | 40 utterances $\times$ 16kHz | `scipy.io.wavfile` |
| MNIST | ~11 MB (torchvision) | 60,000 train / 10,000 test | $28 \times 28$ grayscale | `torchvision.datasets` |
| CIFAR-10 | ~170 MB (torchvision) | 50,000 train / 10,000 test | $32 \times 32 \times 3$ RGB | `torchvision.datasets` |

---

## 4. Phase 1 Execution Validation Gate
All 9 applications have been executed and validated under Phase 1.
Full validation findings are documented at:
[`paper_reproduction/consolidated_results/PHASE1_VALIDATION_REPORT.md`](file:///e:/Rakhal_Tensor/paper_reproduction/consolidated_results/PHASE1_VALIDATION_REPORT.md)
