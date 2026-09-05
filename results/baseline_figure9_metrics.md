# DeepTensor Figure 9 Baseline Metrics Summary

```yaml
experiment:
  name: DeepTensor Figure 9 Hyperspectral Denoising Baseline
  script: run_figure9.py
  dataset: data/icvl.mat
  dataset_name: ICVL
  rank: 20
  epochs: 5000
  noise:
    type: mixed_poisson_gaussian
    noise_snr: 2
    tau: 100
  optimization:
    optimizer: Adam
    learning_rate: 0.001
    loss_function: L1
  architecture:
    spatial_network: DIP 2D U-Net (scales=5, skip_channels=128)
    spectral_network: DIP 1D U-Net (scales=5, skip_channels=128)

environment:
  hardware: HP Omen Laptop
  gpu: NVIDIA GeForce RTX 4060 Laptop GPU
  vram_available_mb: 8192
  ram_gb: 16
  os: Windows
  python_version: 3.11.9
  pytorch_version: 2.5.1+cu121
  cuda_version: 12.1

empirical_results:
  input_noisy_psnr_db: 28.43
  svd_rank20:
    psnr_db: 38.31
    ssim: 0.9169
  deeptensor_5000_epochs:
    psnr_db: 46.45
    ssim: 0.9872
  performance:
    total_runtime_seconds: 159.57
    peak_gpu_vram_mb: 640.37
    throughput_it_per_sec: 31.3

verification:
  reproducibility: PASS
  independent_runs: 2
  status: VERIFIED
```
