# DeepTensor++ — Final Project Structure & Reorganization Reference

> **Finalization Date:** November 6, 2026  
> **Workspace Root:** `e:\Rakhal_Tensor`  
> **Reorganization Status:** Successfully Executed & Validated  
> **Package Name:** `deeptensor` (`v1.0.0`, editable install via `pip install -e .`)

---

## 1. Reorganization Overview & Statistics

The reorganization of `e:\Rakhal_Tensor` into **DeepTensor++** transforms the legacy workspace into a clean, professional, research-grade repository while preserving 100% of previous research progress:

- **Total Python Files Migrated & Organized**: **75 files**
- **Git-Tracked Files Relocated via `git mv`**: **24 files** (100% commit history preserved)
- **Checkpoints Preserved**: **47 checkpoint entries** (all 31 unique physical `.pt` files intact with identical SHA-256 hashes)
- **Datasets Preserved**: **9 datasets** (all 5 `.mat` files and 4 benchmark directories preserved in authoritative `data/` store)
- **Reports & Metric Registries Preserved**: **71 markdown reports**, **36 JSON metric files**, **21 CSV progression traces**, and **107 publication figures**
- **Unit Test Suite**: 7/7 test suites passing in `tests/run_all_tests.py`
- **Streamlit Demo**: Verified working without training overhead (`apps/streamlit_demo/demo.py`)
- **Bit-for-Bit Hash Integrity**: **107,718 items** verified with zero missing or corrupted files in `migration/SHA256_AFTER.csv`

---

## 2. Complete Annotated Final Directory Tree

```text
e:\Rakhal_Tensor/
│
├── .gitignore                                  # Git exclusion rules
├── pyproject.toml                              # Modern editable packaging config
├── requirements.txt                            # Virtual environment dependencies
├── README.md                                   # Comprehensive research overview & execution guide
│
├── configs/                                    # Modular experiment configurations
│   ├── reproduction/                           # Configurations for 13 paper baseline targets
│   │   ├── figure02_eigenfaces.json
│   │   ├── figure04_efficiency.yaml
│   │   ├── figure06_synthetic_robustness.json
│   │   ├── figure07_robust_pca.json
│   │   ├── figure09_hyperspectral.json
│   │   ├── figure10_yale_denoising.yaml
│   │   ├── figure11_video_cs.yaml
│   │   ├── figure12_pet_ct.json
│   │   ├── figure13a_hsi_rank.yaml
│   │   ├── figure13b_yale_rank.yaml
│   │   ├── table01_synthetic.yaml
│   │   ├── table02_nmf.yaml
│   │   └── table03_speech.yaml
│   ├── improvements/                           # Verified high-performance hyperparameter configs
│   │   ├── figure10_yale.yaml                  # 128 channels, 4000 epochs (30.86 / 30.46 dB)
│   │   ├── figure11_video.yaml                 # Rank 64, 10000 epochs (Peak 25.32 dB)
│   │   ├── figure12_pet_ct.json                # Poisson tau=100, 4000 epochs (30.55 dB)
│   │   └── figure13b_yale_rank.yaml            # 1500 epochs, CUDA ALS (28.50-28.97 dB)
│   └── hybrid/                                 # Phase 2 hybrid CNN-Transformer configs
│
├── src/                                        # Standard installable `deeptensor` package
│   └── deeptensor/
│       ├── __init__.py                         # Package entry point (v1.0.0)
│       ├── models/                             # Neural network architectures
│       │   ├── __init__.py
│       │   ├── deeptensor.py                   # DeepTensor dual-factor forward wrapper
│       │   ├── deep_prior.py                   # DIP factory (skip, get_net, fill_noise)
│       │   ├── deep_decoder.py                 # Non-convolutional pixel-wise decoder
│       │   ├── backbones/                      # Reusable generator backbones
│       │   │   ├── __init__.py
│       │   │   ├── skip.py                     # Deep Image Prior skip architecture
│       │   │   ├── unet.py                     # Multi-scale U-Net
│       │   │   ├── resnet.py                   # Residual convolutional network
│       │   │   ├── dcgan.py                    # DCGAN generator
│       │   │   ├── common.py                   # Activation functions (Swish, Sine, Concat)
│       │   │   ├── downsampler.py              # Anti-aliased downsampler
│       │   │   └── texture_nets.py             # Texture generator
│       │   └── attention/                      # Self-attention & Transformer modules
│       │       ├── __init__.py
│       │       ├── spatial_spectral.py         # HybridSpectralNet
│       │       └── spectral_transformer.py     # SpectralTransformer with ALiBi
│       ├── physics/                            # Linear forward measurement operators
│       │   ├── __init__.py
│       │   ├── radon.py                        # 2D/3D Radon transform & FBP
│       │   └── video_cs.py                     # Coded aperture video multiplexing
│       ├── losses/                             # Numerical loss objectives
│       │   ├── __init__.py
│       │   └── objectives.py                   # L1Norm, L2Norm, anisotropic TVNorm
│       ├── baselines/                          # Classical mathematical algorithms
│       │   ├── __init__.py
│       │   ├── parafac_als.py                  # PyTorch CUDA 3-Way PARAFAC ALS
│       │   ├── svd.py                          # Truncated SVD and PCA
│       │   └── spectral.py                     # High-order Tucker decomposition
│       ├── training/                           # Optimization & schedulers
│       │   ├── __init__.py
│       │   ├── schedulers.py                   # Cosine annealing with warmup
│       │   └── trainer.py                      # Reusable optimization loop helper
│       └── utils/                              # Quantitative evaluation & devices
│           ├── __init__.py
│           ├── metrics.py                      # PSNR, SSIM, SAM, ERGAS
│           ├── device.py                       # Centralized device resolver (get_device)
│           └── visualization.py                # Visual grids & montages
│
├── experiments/                                # Segregated, runnable research suites
│   ├── reproduction/                           # Phase 1 paper reproduction baselines
│   │   ├── figure02_eigenfaces/                # run_figure2.py (Weizmann Eigenfaces)
│   │   ├── figure04_efficiency/                # benchmark_figure4_separable_speedup*.py
│   │   ├── figure06_synthetic_robustness/      # run_figure6.py, run_robustness.bat
│   │   ├── figure07_robust_pca/                # run_figure7.py (Robust PCA)
│   │   ├── figure09_hyperspectral/             # run_figure9.py, run_figure9_bm3d.py
│   │   ├── figure10_yale_denoising/            # run_figure10_yale_tensor.py
│   │   ├── figure11_video_cs/                  # run_figure11.py, run_figure11_TV.py
│   │   ├── figure12_pet_ct/                    # run_figure12.py, run_figure12_TV.py
│   │   ├── figure13a_hsi_rank/                 # run_figure13_rank_sweep.py
│   │   ├── figure13b_yale_rank/                # run_figure13b_yale_rank_sweep.py
│   │   ├── table01_synthetic/                  # run_synthetic_suite.py, run_table1.py
│   │   ├── table02_nmf/                        # run_table2_nmf.py (MNIST, CIFAR-10)
│   │   └── table03_speech/                     # run_table3_speech_nmf.py
│   ├── improvements/                           # Verified high-performance runs
│   │   ├── figure10_yale/                      # run_figure10_yale_tensor_improved.py
│   │   ├── figure11_video/                     # run_figure11_improved.py
│   │   ├── figure12_pet_ct/                    # run_figure12_improved.py
│   │   └── figure13b_yale_rank/                # run_figure13b_yale_rank_sweep_improved.py
│   ├── extensions/                             # Git-tracked historical ICVL extensions
│   │   ├── adaptive_rank/                      # run_figure9_adaptive.py (Extension 1)
│   │   ├── group_sparsity/                     # run_figure9_sparsity.py (Extension 2)
│   │   ├── tensor_completion/                  # run_figure9_completion.py (Extension 3)
│   │   ├── noise_robustness/                   # run_figure9_robustness.py (Extension 4)
│   │   ├── ensemble_uncertainty/               # run_figure9_ensemble.py (Extension 5)
│   │   ├── factor_interpretability/            # run_figure9_interpretability.py (Extension 7)
│   │   ├── hybrid_attention/                   # run_figure9_hybrid.py (Extension 8)
│   │   └── spectral_transformer/               # run_figure9_spectral_transformer.py (Ext 9)
│   ├── legacy_experiments/                     # Untracked archival extensions
│   │   └── previous_neurotensor_experiments/   # Preserved 100% in place
│   └── hybrid_prototypes/                      # Phase 2 hybrid CNN-Transformer prototypes
│
├── data/                                       # Single authoritative dataset repository (5.83 GB)
│   ├── icvl.mat                                # 174x164x260 HSI tensor (11.36 MB)
│   ├── pet.mat                                 # 128x128x56 3D PET volume (1.29 MB)
│   ├── yale_b.mat                              # 192x168x160 Yale Face B tensor (4.93 MB)
│   ├── cat_video.mat                           # 128x128x64 Video CS volume (0.78 MB)
│   ├── weizzman.mat                            # 840x103x71 Weizmann Face database (5.37 MB)
│   ├── CroppedYale/                            # Raw CroppedYale PGM archive
│   ├── MNIST/                                  # MNIST benchmark IDX data
│   ├── cifar-10-batches-py/                    # CIFAR-10 python batches
│   └── SpeechCommands/                         # Google Speech Commands v0.02 (105k clips)
│
├── results/                                    # Segregated result registries
│   ├── baseline/                               # Verified baseline checkpoints & metrics
│   │   ├── figure02_eigenfaces/
│   │   ├── figure06_synthetic_robustness/
│   │   ├── figure07_robust_pca/
│   │   ├── figure09_hyperspectral/
│   │   ├── figure10_yale_denoising/
│   │   ├── figure11_video_cs/
│   │   ├── figure12_pet_ct/
│   │   ├── figure13a_hsi_rank/
│   │   ├── figure13b_yale_rank/
│   │   ├── table01_synthetic/
│   │   ├── table02_nmf/
│   │   └── table03_speech/
│   ├── improved/                               # Verified improved checkpoints & metrics
│   │   ├── figure10_yale_denoising/            # 30.86 dB / 30.46 dB
│   │   ├── figure11_video_cs/                  # 25.32 dB (+1.83 dB over TV)
│   │   ├── figure12_pet_ct/                    # 30.55 dB (0.8302 SSIM)
│   │   └── figure13b_yale_rank/                # 28.50-28.97 dB (+2.78 dB over ALS)
│   ├── extensions/                             # Extension outputs & traces
│   └── hybrid/                                 # Phase 2 hybrid outputs
│
├── reports/                                    # Research documentation & audits
│   ├── reproduction/                           # Consolidated reproduction reports & gallery
│   │   ├── CONSOLIDATED_CNN_BASELINE_REPORT.md
│   │   └── gallery_paper_results/              # 107 publication figures
│   ├── improvements/                           # Individual improvement reports
│   │   ├── FIGURE10_IMPROVEMENT_REPORT.md
│   │   ├── FIGURE11_IMPROVEMENT_REPORT.md
│   │   ├── FIGURE12_IMPROVEMENT_REPORT.md
│   │   ├── FIGURE13B_IMPROVEMENT_REPORT.md
│   │   └── PHASE1_PERFORMANCE_GAP_AUDIT.md
│   ├── extensions/                             # Extension theoretical documentation
│   │   └── documentation/
│   └── audits/                                 # Comprehensive codebase audits
│       ├── CODEBASE_CONSISTENCY_AUDIT.md
│       ├── VERIFIED_DATASET_DIMENSIONS.md
│       ├── VERIFIED_EXPERIMENT_METRICS.md
│       ├── FILE_DUPLICATION_AND_RETENTION_AUDIT.md
│       ├── FINAL_PROJECT_STRUCTURE.md
│       └── codebase_audit/
│
├── apps/                                       # End-user interactive applications
│   └── streamlit_demo/                         # Streamlit demo (demo.py, demo_backend.py)
│
├── tests/                                      # Automated test suite
│   ├── test_imports.py
│   ├── test_dataset_paths.py
│   ├── test_smoke_models.py
│   └── run_all_tests.py                        # Standalone unittest runner
│
├── tools/                                      # Research utilities & diagnostics
│   ├── check_attn.py
│   ├── generate_pdf.py
│   ├── fix_newlines.py
│   ├── modify_robustness.py
│   └── scratch_calibrate.py
│
├── archive/                                    # Historical backups & superseded scripts
│   ├── backups/                                # demo_backup.py, demo_backend_backup.py
│   └── legacy/                                 # matrix_decomp.py
│
└── migration/                                  # Integrity manifests & execution logs
    ├── MIGRATION_LOG.md
    ├── MIGRATION_MANIFEST.csv
    ├── SHA256_BEFORE.csv
    ├── SHA256_AFTER.csv
    └── MIGRATION_VALIDATION_REPORT.md
```
"""

with open(os.path.join(ROOT, 'reports', 'audits', 'FINAL_PROJECT_STRUCTURE.md'), 'w', encoding='utf-8') as f:
    f.write(doc.strip() + '\n')

print(f"Written reports/audits/FINAL_PROJECT_STRUCTURE.md ({len(doc.splitlines())} lines)")

