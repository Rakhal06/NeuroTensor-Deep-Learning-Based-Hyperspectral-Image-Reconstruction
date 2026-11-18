# Paper Reproduction Dataset Audit & Provenance

This directory stores provenance records, verification scripts, checksums, and dataset metadata for all 9 target applications.

---

## Dataset Directory Layout

```
paper_reproduction/datasets/
├── DATASET_AUDIT.md
├── 01_synthetic/
├── 02_pca/
├── 03_nmf/
├── 04_weizmann/
├── 05_icvl/
├── 06_yale_b/             (BLOCKED - pending acquisition)
├── 07_cat_video/
├── 08_pet_ct/
└── 09_speech_commands/
```

---

## Dataset Inventory and Verification Status

1. **ICVL Hyperspectral Image:**
   - Location: `data/icvl.mat`
   - Target Application: `05_hyperspectral_denoising`
   - Verified Dimensions: $174 \times 164 \times 260$, `uint16`
   - Status: **LOCAL & VERIFIED**

2. **Weizmann FaceBase:**
   - Location: `data/weizzman.mat`
   - Target Application: `04_eigenfaces_classification`
   - Verified Dimensions: $840 \times 103 \times 71$, `uint8`
   - Status: **LOCAL & VERIFIED**

3. **3D PET CT Volume:**
   - Location: `data/pet.mat`
   - Target Application: `08_ct_reconstruction`
   - Verified Dimensions: $128 \times 128 \times 56$, `uint16`
   - Status: **LOCAL & VERIFIED**

4. **Cat Video:**
   - Location: `data/cat_video.mat`
   - Target Application: `07_video_compressive_sensing`
   - Verified Dimensions: $128 \times 128 \times 64$, `uint8` (8 frames utilized)
   - Status: **LOCAL & VERIFIED**

5. **MNIST & CIFAR-10:**
   - Source: `torchvision.datasets`
   - Target Application: `03_nmf`
   - Dimensions: 2048 samples, $28 \times 28$ and $32 \times 32$
   - Status: **ACCESSIBLE VIA API**

6. **Google Speech Commands:**
   - Source: `torchaudio.datasets.SPEECHCOMMANDS`
   - Target Application: `09_speech_time_frequency`
   - Dimensions: 40 audio files $\to 512 \times 512$ spectrograms
   - Status: **ACCESSIBLE VIA API**

7. **Extended Yale Face Database B:**
   - Target Application: `06_yale_face_tensor_denoising`
   - Target Dimensions: 160 face images, $192 \times 168$
   - Status: **MISSING / BLOCKED**
   - Action Required: Download and format into `data/yale_b.mat`.

