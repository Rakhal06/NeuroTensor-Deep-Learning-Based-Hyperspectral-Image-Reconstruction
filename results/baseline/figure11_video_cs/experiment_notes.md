# Experiment Notes: Figure 11 Video Compressive Sensing

## Metadata
- **Target Application:** `07_video_compressive_sensing`
- **Paper Reference:** Section 4.3 (Page 7) & Figure 11 (Page 9)
- **Dataset:** `data/cat_video.mat` ($128 \times 128 \times 64$)
- **Forward Measurement:** Hitomi et al. pixel-wise spatial-multiplexing coded aperture ($T=8$ frames per snapshot)
- **Architecture:** Deep Image Prior (2D spatial U-Net + 1D temporal U-Net) vs. 2D Total Variation (TV)
- **Device:** NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.1)
- **Reproduction Status:** **EXECUTED AND VALIDATED**

## Key Findings
- DeepTensor achieved **25.14 dB PSNR** and **0.7239 SSIM**, outperforming 2D TV (**23.47 dB**, **0.6873 SSIM).
- Paper reported values: DeepTensor **27.1 dB** (SSIM 0.81) vs. 2D TV **24.1 dB** (SSIM 0.69).
