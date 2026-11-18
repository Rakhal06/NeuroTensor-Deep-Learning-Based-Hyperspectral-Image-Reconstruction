import os
import json

ROOT = 'e:/Rakhal_Tensor'

with open('scratch/comprehensive_audit_data.json', encoding='utf-8') as f:
    data = json.load(f)

py_files = data['python_files']

# Migration path generator
def get_proposed_path(p):
    path = p['path']
    if path.startswith('scratch/'):
        return 'ARCHIVE / SCRATCH', 'scratch/' + os.path.basename(path), 'Scratch file'
    if path in ['demo_backup.py', 'demo_backend_backup.py']:
        return 'DELETE / OBSOLETE', 'None (Obsolete backup)', 'Redundant backup'
    if path.startswith('modules/deep_models/'):
        fname = os.path.basename(path)
        return 'MIGRATE / REFACTOR', f"src/deeptensor/models/backbones/{fname}", 'Keep modular backbone'
    if path.startswith('modules/'):
        fname = os.path.basename(path)
        if fname in ['attention_spectral.py', 'spectral_transformer.py']:
            return 'MIGRATE / REFACTOR', f"src/deeptensor/models/attention/{fname}", 'Attention / Transformer module'
        elif fname in ['losses.py']:
            return 'MIGRATE / REFACTOR', f"src/deeptensor/losses/{fname}", 'Loss functions'
        elif fname in ['utils.py']:
            return 'MIGRATE / REFACTOR', f"src/deeptensor/utils/{fname}", 'Evaluation utilities'
        elif fname in ['lin_inverse.py']:
            return 'MIGRATE / REFACTOR', f"src/deeptensor/physics/{fname}", 'Linear forward operators'
        elif fname in ['cosine_annealing_with_warmup.py']:
            return 'MIGRATE / REFACTOR', f"src/deeptensor/training/schedulers/{fname}", 'LR Scheduler'
        elif fname in ['spectral.py']:
            return 'MIGRATE / REFACTOR', f"src/deeptensor/baselines/{fname}", 'Classical baseline solvers'
        else:
            return 'MIGRATE / REFACTOR', f"src/deeptensor/models/{fname}", 'Core model wrapper'
    if path.startswith('paper_reproduction/cnn_baseline/'):
        subpath = path[len('paper_reproduction/cnn_baseline/'):]
        return 'MIGRATE / UNCHANGED', f"experiments/reproduction/{subpath}", 'Core reproduction suite'
    if path.startswith('paper_reproduction/consolidated_results/'):
        fname = os.path.basename(path)
        return 'MIGRATE / UNCHANGED', f"reports/generators/{fname}", 'Reporting tool'
    if path.startswith('previous_neurotensor_experiments/'):
        if '/modules/' in path:
            return 'DELETE / DUPLICATE', 'None (Duplicate of modules/)', 'Duplicate module'
        subpath = path[len('previous_neurotensor_experiments/'):]
        return 'ARCHIVE / EXTENSION', f"experiments/historical_extensions/{subpath}", 'ICVL research extension'
    if path.startswith('run_figure9') and path != 'run_figure9_bm3d.py':
        return 'DELETE / DUPLICATE', 'None (Duplicate of extension files)', 'Duplicate root runner'
    if path in ['run_figure11.py', 'run_figure11_TV.py']:
        return 'ARCHIVE / SUPERSEDED', f"experiments/reproduction/07_video_compressive_sensing/legacy/{os.path.basename(path)}", 'Superseded by improved runner'
    if path in ['run_figure12.py', 'run_figure12_TV.py']:
        return 'ARCHIVE / SUPERSEDED', f"experiments/reproduction/08_ct_reconstruction/legacy/{os.path.basename(path)}", 'Superseded by improved runner'
    if path in ['run_figure2.py', 'run_figure6.py', 'run_figure7.py', 'run_figure9.py', 'run_table1.py', 'run_figure9_bm3d.py']:
        return 'MIGRATE / LEGACY_BASELINE', f"experiments/legacy_baselines/{os.path.basename(path)}", 'Original baseline runner'
    if path in ['demo.py', 'demo_backend.py']:
        return 'MIGRATE / APP', f"apps/streamlit_demo/{os.path.basename(path)}", 'Interactive demo'
    if path in ['generate_pdf.py']:
        return 'MIGRATE / UTILITY', f"tools/pdf_generator/{os.path.basename(path)}", 'PDF generator'
    if path in ['check_attn.py', 'fix_newlines.py', 'matrix_decomp.py', 'modify_robustness.py', 'scratch_calibrate.py']:
        return 'ARCHIVE / DIAGNOSTIC', f"tools/diagnostics/{os.path.basename(path)}", 'Legacy diagnostic'
    return 'OTHER', f"misc/{os.path.basename(path)}", 'Miscellaneous'

doc = """# DeepTensor++ — Proposed Modular Architecture & Migration Map

> **Audit Date:** November 6, 2026  
> **Workspace Root:** `e:\\Rakhal_Tensor`  
> **Target Reorganization Root:** `DeepTensor-Plus-Plus/`  
> **Strategy:** Zero-Risk Phased Migration (No loss of checkpoints, metrics, or research provenance).

---

## 1. Clean Modular Target Architecture (`DeepTensor-Plus-Plus/`)

```text
DeepTensor-Plus-Plus/
├── pyproject.toml                              # Modern packaging & pinned dependency management
├── README.md                                   # Comprehensive research overview & reproduction guide
├── configs/                                    # Centralized, modular YAML experiment configs
│   ├── baseline/                               # Paper baseline hyperparameter configs
│   ├── improved/                               # High-performance improved hyperparameter configs
│   └── hybrid/                                 # Phase 2 hybrid CNN-Transformer configs
│
├── src/                                        # Installable Python package (`pip install -e .`)
│   └── deeptensor/
│       ├── __init__.py
│       ├── models/                             # Neural network architectures
│       │   ├── __init__.py
│       │   ├── deeptensor.py                   # DeepTensor core wrapper
│       │   ├── deep_decoder.py                 # Non-convolutional decoders
│       │   ├── deep_prior.py                   # DIP factory & noise tensor generators
│       │   ├── attention/                      # Self-attention & Transformer modules
│       │   │   ├── __init__.py
│       │   │   ├── spatial_spectral.py         # HybridSpectralNet
│       │   │   └── spectral_transformer.py     # SpectralTransformer with ALiBi
│       │   └── backbones/                      # Reusable convolutional generator backbones
│       │       ├── __init__.py
│       │       ├── skip.py                     # DIP Skip architecture
│       │       ├── unet.py                     # Multi-scale U-Net
│       │       ├── resnet.py                   # Residual ConvNet
│       │       └── dcgan.py                    # DCGAN generator
│       ├── physics/                            # Forward measurement operators & adjoints
│       │   ├── __init__.py
│       │   ├── radon.py                        # 2D/3D Radon transform & FBP
│       │   └── video_cs.py                     # Video coded snapshot multiplexer
│       ├── losses/                             # Numerical objectives & regularizers
│       │   ├── __init__.py
│       │   └── objectives.py                   # L1, L2, Anisotropic 2D/3D Total Variation
│       ├── baselines/                          # Classical mathematical algorithms
│       │   ├── __init__.py
│       │   ├── parafac_als.py                  # PyTorch CUDA Alternating Least Squares
│       │   ├── svd.py                          # Truncated SVD & PCA
│       │   └── tucker.py                       # High-order SVD / Tucker
│       ├── training/                           # Optimization & training infrastructure
│       │   ├── __init__.py
│       │   ├── trainer.py                      # Unified Trainer loop with closure handling
│       │   └── schedulers/                     # Cosine annealing with warmup
│       └── utils/                              # Quantitative evaluation & visualization
│           ├── __init__.py
│           ├── metrics.py                      # PSNR, SSIM, SAM, ERGAS metrics
│           └── visualization.py                # Visual grids & montages
│
├── experiments/                                # Segregated, runnable experiment suites
│   ├── reproduction/                           # Standardized Phase 1 paper reproduction
│   │   ├── 01_synthetic_decomposition/
│   │   ├── 02_computational_efficiency/        # Figure 4 Separable speedup benchmark
│   │   ├── 03_nmf_mnist_cifar/
│   │   ├── 04_eigenfaces_pca/                  # Figure 2 Weizmann Eigenfaces
│   │   ├── 05_hyperspectral_denoising/         # Figure 9 & Figure 13(a)
│   │   ├── 06_yale_face_denoising/             # Figure 10 & Figure 13(b)
│   │   ├── 07_video_compressive_sensing/       # Figure 11
│   │   ├── 08_ct_reconstruction/               # Figure 12
│   │   └── 09_speech_time_frequency/           # Table 3
│   ├── historical_extensions/                  # Archived ICVL Extensions 1 through 9
│   └── hybrid_prototypes/                      # Phase 2 CNN-Transformer implementations
│
├── data/                                       # Centralized datasets (read-only symlink or path)
├── results/                                    # Standardized outputs, segregated by run type
│   ├── baseline/                               # Verified original baseline checkpoints & logs
│   ├── improved/                               # Verified improved checkpoints, logs, & panels
│   └── hybrid/                                 # Future Phase 2 experimental outputs
│
├── reports/                                    # Research reports, publication panels & PDFs
│   ├── audits/                                 # Phase 1 audit & validation reports
│   ├── panels/                                 # Multi-slice high-resolution publication panels
│   └── generators/                             # Consolidated report & PDF builders
│
├── apps/                                       # End-user interactive applications
│   └── streamlit_demo/                         # Clean demo.py & demo_backend.py
│
├── tools/                                      # Diagnostic & conversion utilities
└── tests/                                      # PyTest suite for mathematical operators & models
```

---

## 2. Complete File-by-File Migration Mapping Table

| Current Path | Target Migration Action | Proposed Target Path | Rationale & Responsibility |
|---|---|---|---|
"""

for p in py_files:
    action, target, desc = get_proposed_path(p)
    doc += f"| `{p['path']}` | **{action}** | `{target}` | {desc} |\n"

doc += """
---

## 3. Classification of Action Items

### 3.1 Files to Keep Unchanged (or Relocate Without Logic Modification)
- **All Core Algorithms in `modules/`**: Refactor into standard package structure under `src/deeptensor/`.
- **All Improved Runners in `paper_reproduction/cnn_baseline/`**: Move to `experiments/reproduction/` preserving arguments and configs.
- **All Saved Checkpoints in `improved_run/` and `RESULTS-BASELINE/`**: Move to `results/improved/` and `results/baseline/` with hash verification.
- **All Datasets in `data/`**: Keep strictly preserved in `data/`.

### 3.2 Files to Consolidate / Merge
- **Duplicate Loss Functions**: Merge inline `torch.diff` TV routines into `src/deeptensor/losses/objectives.py`.
- **Duplicate Metrics Logic**: Ensure all scripts call `src.deeptensor.utils.metrics.calculate_psnr` and `calculate_ssim`.
- **Duplicate Outer Loops**: Replace repeated Adam training loops with a clean `src.deeptensor.training.trainer.Trainer`.

### 3.3 Files to Archive in Dedicated Historical Directories
- `previous_neurotensor_experiments/01` through `09`: Relocate to `experiments/historical_extensions/`.
- `run_figure2.py`, `run_figure6.py`, `run_figure7.py`, `run_figure9.py`, `run_table1.py`: Move to `experiments/reproduction/` or `experiments/legacy_baselines/`.

### 3.4 Files Safe to Delete Immediately (Redundant / Obsolete)
- **9 Duplicate Root Scripts**: `run_figure9*.py` in root (identical copies exist in extensions).
- **2 Duplicate Subpackage Modules**: `previous_neurotensor_experiments/*/modules/*.py`.
- **2 Obsolete Backup Files**: `demo_backup.py`, `demo_backend_backup.py`.
- **Superseded Diagnostic Dumps**: `results_ensemble_smoke/` (231.3 MB), `results_lambda_*.txt` logs.
- **Empty Directories**: `results_completion*/`, `results_robustness/`.

---

## 4. Step-by-Step Phased Migration Roadmap

```mermaid
flowchart TD
    Phase0["Phase 0: Pre-Migration Backup & Safety Gate<br/>(Git commit tag, checkpoint SHA-256 manifest)"] --> Phase1
    Phase1["Phase 1: Package Scaffolding & Core Refactor<br/>(Create pyproject.toml, src/deeptensor structure)"] --> Phase2
    Phase2["Phase 2: Experiment Suite Standardization<br/>(Migrate experiments/reproduction, remove sys.path hacks)"] --> Phase3
    Phase3["Phase 3: Results & Checkpoint Reorganization<br/>(Migrate results/baseline and results/improved, verify hashes)"] --> Phase4
    Phase4["Phase 4: Cleanup & Archival<br/>(Delete confirmed duplicates, archive legacy diagnostics)"] --> Phase5
    Phase5["Phase 5: Verification Smoke Suite<br/>(Run 10-step smoke tests across all 13 reproduction targets)"]
```

### Detailed Roadmap:
1. **Phase 0: Integrity Checksum & Snapshot**:
   - Compute and save a SHA-256 manifest of all 31 `.pt` checkpoints, all metrics JSONs, and all `.mat` datasets.
   - Tag current state in Git.
2. **Phase 1: Package Scaffolding**:
   - Establish `DeepTensor-Plus-Plus/pyproject.toml`.
   - Port `modules/` into `src/deeptensor/` with clean relative imports.
   - Test package installation in virtual environment: `pip install -e .`.
3. **Phase 2: Experiment Suite Standardization**:
   - Port all runners from `paper_reproduction/cnn_baseline/` to `experiments/reproduction/`.
   - Remove `sys.path.append('modules')` from all scripts; replace with standard package imports (`from deeptensor.models import DeepTensor`).
4. **Phase 3: Results & Checkpoint Reorganization**:
   - Relocate verified checkpoints, metrics, and plots into `results/baseline/` and `results/improved/`.
   - Verify SHA-256 checksums to guarantee zero data loss or byte corruption.
5. **Phase 4: Safe Redundancy Pruning**:
   - Remove confirmed duplicate root scripts, backup files, and temporary smoke checkpoints.
6. **Phase 5: Automated Verification Suite**:
   - Execute a 1-epoch / 10-step verification smoke test on each experiment target to confirm that all refactored scripts execute cleanly without errors.
"""

out_path = os.path.join(ROOT, "codebase_audit", "PROPOSED_ARCHITECTURE_AND_MIGRATION_MAP.md")
with open(out_path, "w", encoding="utf-8") as fp:
    fp.write(doc.strip() + "\n")

print(f"Generated {out_path} ({len(doc.splitlines())} lines)")

