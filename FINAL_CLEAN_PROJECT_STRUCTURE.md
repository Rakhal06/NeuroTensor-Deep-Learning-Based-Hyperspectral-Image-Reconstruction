# DeepTensor++ — Final Consolidated Project Architecture

**Document Version:** 1.0.0  
**Date:** November 8, 2026  
**Status:** Completed & Validated  
**Workspace:** `e:\Rakhal_Tensor`  

---

## 1. Executive Summary

The DeepTensor++ workspace has undergone a comprehensive, zero-loss root cleanup and structural consolidation. All legacy root folders (`modules/`, `paper_reproduction/`, `PAPER_RESULTS/`, `RESULTS-BASELINE/`, `previous_neurotensor_experiments/`, `codebase_audit/`, `scratch/`, and 14 individual `results_*` directories) and 35 loose root files have been consolidated into an authoritative, single-source-of-truth research repository architecture.

Every single file, script, checkpoint, dataset, log, and report has been preserved with bit-for-bit SHA-256 verification. Git commit history has been preserved for all tracked files using Git-aware relocations.

---

## 2. Final Root Directory Layout

```text
DeepTensor-Plus-Plus/
├── README.md                                   # Comprehensive project documentation & guide
├── LICENSE                                     # MIT License
├── .gitignore                                  # Standardized Git exclusion rules
├── pyproject.toml                              # PEP 517/621 editable package definition (deeptensor)
├── requirements.txt                            # Pinned Python dependencies
├── PROJECT_STRUCTURE.txt                       # Clean recursive directory tree
│
├── configs/                                    # Canonical YAML/JSON experiment configurations
│   ├── reproduction/                           # Configurations for 13 original paper targets
│   ├── improvements/                           # Configurations for improved performance runs
│   └── hybrid/                                 # Phase 2 hybrid CNN-Transformer configs
│
├── src/                                        # Authoritative installable Python library
│   └── deeptensor/                             # `pip install -e .` package root
│       ├── __init__.py                         # Version, exports, and top-level API
│       ├── models/                             # DeepTensor wrapper, DIP factories, MLPs
│       │   ├── deeptensor.py                   # DeepTensor unified dual-factor forward module
│       │   ├── deep_prior.py                   # DIP factory methods (skip, get_net, fill_noise)
│       │   ├── deep_decoder.py                 # Non-convolutional pixel-wise decoder
│       │   ├── backbones/                      # Skip, U-Net, ResNet, DCGAN convolutional generators
│       │   └── attention/                      # HybridSpectralNet, SpectralTransformer with ALiBi
│       ├── physics/                            # Linear forward operators & adjoints (Radon, Video CS)
│       ├── losses/                             # L1, L2, and anisotropic Total Variation (TVNorm)
│       ├── baselines/                          # PyTorch CUDA PARAFAC ALS, Truncated SVD, Tucker
│       ├── training/                           # Cosine Annealing scheduler & training utilities
│       └── utils/                              # Metrics (PSNR, SSIM, SAM, ERGAS) & device resolver
│
├── data/                                       # Single-source dataset repository (5.83 GB)
│   ├── icvl.mat                                # ICVL Hyperspectral Image (174x164x260, uint16)
│   ├── pet.mat                                 # 3D PET/CT volume (128x128x56, uint16)
│   ├── yale_b.mat                              # Extended Yale Face B tensor (192x168x160, uint8)
│   ├── cat_video.mat                           # Video CS volume (128x128x64, float32)
│   ├── weizzman.mat                            # Weizmann Face database (840x103x71, float64)
│   ├── MNIST/                                  # MNIST benchmark digits
│   ├── cifar-10-batches-py/                    # CIFAR-10 image batches
│   └── SpeechCommands/                         # Google Speech Commands v0.02 (105k audio clips)
│
├── experiments/                                # Segregated, runnable research suites
│   ├── reproduction/                           # Direct paper reproduction targets (Figures 2-13, Tables 1-3)
│   ├── improvements/                           # Verified high-performance runs (Figures 10, 11, 12, 13b)
│   ├── extensions/                             # 9 ICVL extensions (adaptive rank, sparsity, attention, etc.)
│   ├── hybrid_prototypes/                      # Phase 2 CNN-Transformer prototypes
│   └── legacy_experiments/                     # Historical reference implementations (previous_neurotensor)
│
├── results/                                    # Segregated result registries (Zero loose files at root)
│   ├── baseline/                               # Verified original baseline checkpoints, logs & metrics
│   ├── improved/                               # Verified improved checkpoints, logs & panels (Fig 10, 11, 12, 13b)
│   ├── extensions/                             # ICVL extension outputs (completion, ensemble, sparsity, etc.)
│   └── hybrid/                                 # Phase 2 hybrid outputs (attention, spectral transformer)
│
├── reports/                                    # Consolidated reports, audits & publication panels
│   ├── reproduction/                           # Consolidated reproduction reports & image gallery
│   ├── improvements/                           # Individual improvement audit reports
│   ├── extensions/                             # Theoretical documentation for 9 extensions
│   └── audits/                                 # Codebase consistency audits & dimension traces
│
├── apps/                                       # Interactive applications
│   └── streamlit_demo/                         # Real-time Streamlit visualization app (demo.py, demo_backend.py)
│
├── tests/                                      # Automated test suite
│   ├── run_all_tests.py                        # 7-suite comprehensive regression test runner
│   └── test_smoke_models.py                    # Fast module & CUDA smoke tests
│
├── tools/                                      # Maintenance and utility scripts
│   ├── check_attn.py                           # Attention weights verification
│   ├── generate_pdf.py                         # PDF document compiler
│   ├── fix_newlines.py                         # File line ending normalizer
│   └── execute_final_cleanup.py                # Automated root cleanup engine
│
├── archive/                                    # Backups and legacy runners
│   ├── backups/                                # Historical backup scripts (demo backups)
│   └── legacy/                                 # Legacy scripts, audit scratch, and modules_legacy
│
└── migration/                                  # Migration tracking and audit manifests
    ├── CLEANUP_MANIFEST.csv                    # 502-item exhaustive relocation & SHA-256 manifest
    ├── MIGRATION_LOG.md                        # Historical migration journey
    ├── SHA256_BEFORE.csv                       # Pre-migration baseline hashes
    └── SHA256_AFTER.csv                        # Post-migration baseline hashes
```

---

## 3. Root Directory Consolidation Mapping

| Old Root Directory | Destination | Action & Rationale |
|---|---|---|
| `modules/` | `archive/legacy/modules_legacy/` | Git-moved to preserve historical commit history. Modern authoritative code lives in `src/deeptensor/`. |
| `paper_reproduction/` | `archive/legacy/paper_reproduction_legacy/` | Full tree archived; key reports copied to `reports/reproduction/` and `reports/audits/`. |
| `PAPER_RESULTS/` | `reports/reproduction/gallery_paper_results/` | Verified identical bit-for-bit SHA-256; duplicate root folder safely pruned. |
| `RESULTS-BASELINE/` | `results/baseline/` | Git-moved into canonical baseline figure folders (`figure02_eigenfaces/`, etc.) with full Git tracking. |
| `previous_neurotensor_experiments/` | `experiments/legacy_experiments/previous_neurotensor_experiments/` | Verified identical bit-for-bit SHA-256; duplicate root folder safely pruned. |
| `codebase_audit/` | `reports/audits/codebase_audit/` | Verified identical bit-for-bit SHA-256; duplicate root folder safely pruned. |
| `scratch/` | `archive/legacy/scratch_audit/` | 45 temporary audit and helper scripts preserved intact under archive. |
| `documentation/` | `reports/extensions/documentation/` | Git-moved all 8 extension documentation markdown files to preserve commit history. |
| `results_completion*` (4 folders) | `results/extensions/completion/` | Segregated into `standard/`, `debug/`, `sanity/`, and `smoke/`. |
| `results_ensemble*` (2 folders) | `results/extensions/ensemble/` | Segregated into `standard/` and `smoke/`. |
| `results_interpretability*` (2 folders) | `results/extensions/interpretability/` | Segregated into `standard/` and `smoke/`. |
| `results_robustness/` | `results/extensions/robustness/standard/` | Canonical noise robustness metrics and summaries preserved. |
| `results_sparsity*` (3 folders) | `results/extensions/sparsity/` | Segregated into `standard/`, `corrected/`, and `diagnostic/`. |
| `results_hybrid/` | `results/hybrid/hybrid_attention/` | Hybrid attention weights and loss trajectories preserved. |
| `results_spectral_transformer/` | `results/hybrid/spectral_transformer/` | Spectral transformer weights, summaries, and loss curves preserved. |

---

## 4. Root File Consolidation Mapping

| Old Root File | New Canonical Destination | Rationale |
|---|---|---|
| `DATA_DIMENSION_TRACE.md` | `reports/audits/DATA_DIMENSION_TRACE.md` | Git-moved audit document |
| `DEMO_IMPLEMENTATION_PLAN.md` | `reports/audits/DEMO_IMPLEMENTATION_PLAN.md` | Git-moved audit document |
| `IMPLEMENTATION_DOCUMENTATION.md` | `reports/audits/IMPLEMENTATION_DOCUMENTATION.md` | Git-moved audit document |
| `PROJECT_CONTEXT.md` | `reports/audits/PROJECT_CONTEXT.md` | Git-moved audit document |
| `DeepTensor_Implementation_Documentation.pdf` | `reports/reproduction/papers_and_slides/` | Reference PDF |
| `DeepTensor_Paper_2204.03145.pdf` | `reports/reproduction/papers_and_slides/` | Original paper PDF |
| `Enhanced_DeepTensor_Phase1_Detailed_Presentation-1.pptx` | `reports/reproduction/papers_and_slides/` | Reference presentation |
| `DATASET_REGISTER.md` | `reports/audits/DATASET_REGISTER.md` | Dataset provenance register |
| `EXPERIMENT_TRACKER.md` | `reports/audits/EXPERIMENT_TRACKER.md` | Experiment status tracker |
| `FIGURE_TRACKER.md` | `reports/audits/FIGURE_TRACKER.md` | Figure status tracker |
| `IMPLEMENTATION_STATUS.md` | `reports/audits/IMPLEMENTATION_STATUS.md` | Implementation matrix |
| `PAPER_EXPERIMENT_REGISTER.md` | `reports/reproduction/PAPER_EXPERIMENT_REGISTER.md` | Paper reproduction register |
| `PHASE1_EXPERIMENT_CONFIG_AUDIT.csv` | `reports/audits/PHASE1_EXPERIMENT_CONFIG_AUDIT.csv` | Configuration audit table |
| `PHASE1_IMPROVEMENT_PLAN.md` | `reports/improvements/PHASE1_IMPROVEMENT_PLAN.md` | Phase 1 improvement roadmap |
| `PHASE1_PERFORMANCE_GAP_AUDIT.md` | `reports/improvements/PHASE1_PERFORMANCE_GAP_AUDIT.md` | Performance gap audit report |
| `PHASE1_RERUN_PRIORITY.csv` | `reports/improvements/PHASE1_RERUN_PRIORITY.csv` | Improvement priority matrix |
| `PROJECT_ROADMAP.md` | `reports/audits/PROJECT_ROADMAP.md` | Research roadmap |
| `REPRODUCIBILITY.md` | `reports/reproduction/REPRODUCIBILITY.md` | Reproducibility guidelines |
| `RESEARCH_NOTES.md` | `reports/audits/RESEARCH_NOTES.md` | Research logs |
| `demo.py.bak` ... `demo_backup.py` (4 files) | `archive/backups/` | Historical backups |
| `modify_robustness.py`, `scratch_calibrate.py` | `archive/legacy/` | Legacy scratch scripts |
| `scratch_exp_mapping.json`, `scratch_py_inventory.json` | `archive/legacy/` | Temporary audit artifacts |
| `paper_extracted_text.txt`, `user_prompt.txt` | `archive/legacy/` | Legacy reference text |
| `fix_newlines.py` | `tools/fix_newlines.py` | Code formatting utility |
| `results_lambda_*.txt` (4 raw log files) | `results/extensions/sparsity/logs/` | Group sparsity experiment raw logs |

---

## 5. Cleanliness & Single-Source-of-Truth Principles

1. **Root Cleanliness:** The root directory now contains strictly 6 authoritative items: `.gitignore`, `LICENSE`, `PROJECT_STRUCTURE.txt`, `README.md`, `pyproject.toml`, and `requirements.txt`.
2. **Zero Redundancy:** There are no duplicate categories (no redundant `PAPER_RESULTS`, `RESULTS-BASELINE`, `modules`, or `results_*` directories).
3. **Reproducibility Guarantee:** Any researcher cloning the repository can install `deeptensor` with `pip install -e .` and immediately run reproduction experiments or the interactive demo without broken path references.

