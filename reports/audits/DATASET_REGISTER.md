# DeepTensor Master Dataset Register

This document registers and audits every dataset required to replicate all experiments from:
**"DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors"** (Saragadam et al., IEEE TPAMI / arXiv:2204.03145).

---

## Dataset Audit & Verification Matrix

| Application | Dataset Name | File / Resource | Target Dimensions | Data Type | Source URL / Provenance | Verification Status | Storage Path | SHA-256 (first 12 chars) |
|---|---|---|---|---|---|---|---|---|
| **01_synthetic_decomposition** | Synthetic Low-Rank Matrix | On-the-fly generated | $64 \times 64$ | `float32` | Synthetic (`utils.get_matrix`) | **VERIFIED** | Code generator | N/A |
| **02_pca** | Synthetic Low-Rank Covariance | On-the-fly generated | $D=64, M \in [32, 1024]$ | `float32` | Synthetic Gaussian | **VERIFIED** | Code generator | N/A |
| **03_nmf** | MNIST Digits | `torchvision.datasets.MNIST` | 2048 samples $\times 28 \times 28$ | `float32` | PyTorch torchvision | **ACCESSIBLE VIA API** | Local cache / download | API download |
| **03_nmf** | CIFAR-10 Images | `torchvision.datasets.CIFAR10` | 2048 samples $\times 32 \times 32$ | `float32` | PyTorch torchvision | **ACCESSIBLE VIA API** | Local cache / download | API download |
| **04_eigenfaces_classification** | Weizmann Face Database | `data/weizzman.mat` | 840 faces $\times 103 \times 71$ | `uint8` | Weizmann FaceBase | **LOCAL & VERIFIED** | `data/weizzman.mat` | `b99e0df2dc36` |
| **05_hyperspectral_denoising** | ICVL Hyperspectral Image | `data/icvl.mat` | $174 \times 164 \times 260$ | `uint16` | ICVL Hyperspectral Database | **LOCAL & VERIFIED** | `data/icvl.mat` | `43fe3ef247e6` |
| **06_yale_face_tensor_denoising** | Extended Yale Face Database B | 160 images $\times 192 \times 168$ | $192 \times 168 \times 160$ | `uint8` / `float32` | Yale Face B / UCSD Vision | **MISSING / BLOCKED** | Needs acquisition | Pending download |
| **07_video_compressive_sensing** | Cat Video | `data/cat_video.mat` | $128 \times 128 \times 64$ | `uint8` | Google Drive (author release) | **LOCAL & VERIFIED** | `data/cat_video.mat` | `6c12d46e3a09` |
| **08_ct_reconstruction** | 3D PET CT Volume | `data/pet.mat` | $128 \times 128 \times 56$ | `uint16` | TCIA Cancer Imaging Archive | **LOCAL & VERIFIED** | `data/pet.mat` | `f6a45741639d` |
| **09_speech_time_frequency** | Google Speech Commands | `torchaudio.datasets.SPEECHCOMMANDS` | 40 utterances $\to 512 \times 512$ | `float32` | PyTorch torchaudio | **ACCESSIBLE VIA API** | Local cache / download | API download |

---

## Detailed Dataset Profiles

### 1. ICVL Hyperspectral Dataset (`data/icvl.mat`)
- **Status:** Local, verified, frozen.
- **Original Source:** Interdisciplinary Computational Vision Laboratory (ICVL), Ben-Gurion University of the Negev.
- **File Contents:**
  - `hypercube`: shape `(174, 164, 260)`, dtype `uint16`, min = 0, max = 4095 (12-bit).
  - `wavelengths`: shape `(1, 260)`, dtype `float64`, range [400nm, 1000nm].
- **Preprocessing:** Normalized by maximum count `cube / cube.max()`.
- **Integrity:** SHA-256 verified against author release.

### 2. Weizmann FaceBase (`data/weizzman.mat`)
- **Status:** Local, verified, frozen.
- **Original Source:** Weizmann Institute of Science FaceBase.
- **File Contents:**
  - `faces`: shape `(840, 103, 71)`, dtype `uint8`.
  - `labels`: shape `(1, 840)`, dtype `uint8` (28 subjects, 30 illumination conditions each).
  - `names`: shape `(1, 28)`, subject identifiers.
- **Integrity:** Used in `RESULTS-BASELINE/figure2/` with verified SVM 5-fold cross-validation.

### 3. 3D PET CT Volume (`data/pet.mat`)
- **Status:** Local, verified, frozen.
- **Original Source:** The Cancer Imaging Archive (TCIA).
- **File Contents:**
  - `hypercube`: shape `(128, 128, 56)`, dtype `uint16`.
- **Integrity:** Used in `RESULTS-BASELINE/figure12/` with 40-angle Radon transform simulation.

### 4. Cat Video (`data/cat_video.mat`)
- **Status:** Local, verified, frozen.
- **Original Source:** Author Google Drive repository.
- **File Contents:**
  - `hypercube`: shape `(128, 128, 64)`, dtype `uint8`.
- **Usage:** Sliced along temporal dimension: $T=8$ frames used for video compressive sensing.

### 5. Extended Yale Face Database B (Application 6 - BLOCKED)
- **Status:** **NOT FOUND LOCALLY**.
- **Requirement:** 160 face images under diverse lighting, cropped and aligned to $192 \times 168$ pixels, stacked into a 3D tensor $192 \times 168 \times 160$.
- **Action Plan:** Obtain Extended Yale B images from standard academic repositories (e.g., UCSD or Kaggle), pre-process into `data/yale_b.mat`, verify dimensions and checksums.

### 6. MNIST & CIFAR-10 (Application 3)
- **Status:** API accessible via `torchvision.datasets`.
- **Requirement:** 2048 samples per dataset, standardized and normalized to $[0, 1]$.

### 7. Google Speech Commands (Application 9)
- **Status:** API accessible via `torchaudio.datasets.SPEECHCOMMANDS`.
- **Requirement:** 40 audio files converted to STFT spectrograms ($N_{\text{fft}}=1024, \text{hop}=32$), cropped to $512 \times 512$.

