# DeepTensor++ — Post-Cleanup Independent Verification Audit Report

**Audit Date:** November 8, 2026  
**Auditor:** Independent Automated Research Codebase Auditor  
**Workspace:** `E:\Rakhal_Tensor`  
**Execution Environment:** Windows, Python 3.11.9, PyTorch 2.5.1+cu121, CUDA 12.1, NVIDIA GeForce RTX 4060 Laptop GPU  
**Audit Scope:** Read-only inspection of filesystem, Git working tree, code modifications, dataset integrity, checkpoints, import validity, experiment separation, and lightweight validation.  

---

## 1. Overall Audit Verdict

```text
========================================================================================
                                 OVERALL AUDIT VERDICT
                    CONDITIONALLY VERIFIED / STRUCTURALLY CONSOLIDATED
                               WITH OPERATIONAL WARNINGS
========================================================================================
```

### Executive Summary of Verdict:
- **Filesystem Restructuring:** **VERIFIED (100% Complete)**. All 21 obsolete root-level directories (`modules/`, `paper_reproduction/`, `PAPER_RESULTS/`, `RESULTS-BASELINE/`, `previous_neurotensor_experiments/`, `codebase_audit/`, `scratch/`, and all 14 `results_*` folders) and 35 loose root files have been successfully consolidated.
- **Data & Checkpoint Preservation:** **VERIFIED (Zero Data Loss)**. All 108,148 files from the pre-migration baseline (`SHA256_BEFORE.csv`) are 100% accounted for (108,124 unchanged in place, 24 relocated via `git mv`). All 502 entries in `CLEANUP_MANIFEST.csv` match their on-disk SHA-256 hashes bit-for-bit (0 missing, 0 mismatched).
- **Core Library & Test Suite:** **VERIFIED**. `src/deeptensor` is properly installed in editable mode (`deeptensor-1.0.0`). The 7 comprehensive unit tests in `tests/run_all_tests.py` and 4 smoke tests in `tests/test_smoke_models.py` pass cleanly on CUDA:0.
- **OPERATIONAL WARNING / INCOMPLETE MIGRATION:** **FAILED (Stale References in Experiment Scripts)**. While the root is structurally clean, **30 experiment runner scripts** in `experiments/` still contain legacy `sys.path.append('modules')` and direct `import models` calls. When executed from their new directory locations, they throw `ModuleNotFoundError: No module named 'models'`. Additionally, many scripts and YAML configs retain default output paths pointing to old directories (e.g., `RESULTS-BASELINE/`, `paper_reproduction/`, `results_sparsity/`), which will re-pollute the root if executed.
- **Publication-Ready Status:** **NOT YET FULLY PUBLICATION-READY**. A second pass is required to modernize runner imports and update output directories before the repository can be claimed plug-and-play runnable for third-party researchers.

---

## 2. Directory Structure Verification

| Target / Assertion | Expected State | Actual State on Disk | Audit Finding |
|---|---|---|---|
| **Root Cleanliness** | No obsolete folders (`modules`, `paper_reproduction`, `PAPER_RESULTS`, `RESULTS-BASELINE`, `results_*`, `scratch`, `codebase_audit`) | All 21 obsolete directories return `False` on `os.path.exists()` | **VERIFIED** |
| **Top-Level Directories** | Only canonical directories exist: `apps`, `archive`, `configs`, `data`, `experiments`, `migration`, `reports`, `results`, `src`, `tests`, `tools` (plus `.git`, `.venv`) | Exactly 11 canonical directories + `.git` and `.venv` | **VERIFIED** |
| **Root Loose Files** | Only essential root files present | 8 files: `.gitignore`, `LICENSE`, `README.md`, `pyproject.toml`, `requirements.txt`, `PROJECT_STRUCTURE.txt`, `FINAL_CLEAN_PROJECT_STRUCTURE.md`, `FINAL_CLEANUP_VALIDATION.md` | **VERIFIED** |
| **Results Segregation** | `results/` strictly partitioned into `baseline`, `improved`, `extensions`, `hybrid` | Exactly 4 directories; 0 loose files directly at `results/` level | **VERIFIED** |
| **Reports Segregation** | `reports/` partitioned into `audits`, `extensions`, `improvements`, `reproduction` | Exactly 4 directories; all 16 paper gallery figures and audit documents organized | **VERIFIED** |
| **Exclusion Compliance** | `.venv`, `.git`, `__pycache__` excluded from tree dump | Verified excluded in `PROJECT_STRUCTURE.txt` (1,043 lines) | **VERIFIED** |
| **Empty Directories** | No unexpected empty folders | 2 empty directories: `configs/hybrid/` and `experiments/hybrid_prototypes/` (Intentional Phase 2 placeholders) | **WARNING (Documented)** |

---

## 3. Data & Results Preservation Audit

### 3.1 Core MAT Datasets Physical Verification
Direct inspection via `scipy.io.loadmat` on `data/`:

| Dataset File | Physical Size | Internal Variables | Verified Array Shape | Data Type | Audit Status |
|---|---|---|---|---|---|
| `data/icvl.mat` | 11.36 MB | `hypercube`, `wavelengths` | `(174, 164, 260)`, `(1, 260)` | `uint16`, `float64` | **VERIFIED** |
| `data/pet.mat` | 1.29 MB | `hypercube` | `(128, 128, 56)` | `uint16` | **VERIFIED** |
| `data/yale_b.mat` | 4.93 MB | `faces`, `filenames` | `(192, 168, 160)`, `(160,)` | `uint8`, `<U24` | **VERIFIED** |
| `data/cat_video.mat` | 0.78 MB | `hypercube` | `(128, 128, 64)` | `uint8` | **VERIFIED** |
| `data/weizzman.mat` | 5.37 MB | `faces`, `labels`, `names` | `(840, 103, 71)`, `(1, 840)` | `uint8`, `uint8` | **VERIFIED** |

### 3.2 Large Benchmark Datasets
- `data/MNIST/`: 8 raw index/image files, 63.46 MB — **VERIFIED**
- `data/cifar-10-batches-py/`: 8 files (5 train batches, 1 test batch, metadata), 177.59 MB — **VERIFIED**
- `data/SpeechCommands/`: 105,841 files across 36 command subdirectories, 3,185.26 MB — **VERIFIED**
- `data/CroppedYale/`: 1,760 image files, 27.32 MB — **VERIFIED**

### 3.3 Model Weights & Checkpoints Inventory
A total of **61 model checkpoint (`.pt`) and result (`.npz`) files** were discovered across the active repository. All key checkpoints are physically intact:

| Target | Canonical Path | Size | Epochs / Status | Audit Status |
|---|---|---|---|---|
| **Figure 9 Baseline** | `results/baseline/figure09_hyperspectral/checkpoint/figure9_deeptensor_checkpoint.pt` | 11.65 MB | 4,000 iters | **VERIFIED** |
| **Figure 9 NPZ** | `results/baseline/figure09_hyperspectral/figure9_results.npz` | 159.30 MB | Full volume | **VERIFIED** |
| **Figure 12 Baseline** | `results/baseline/figure12_pet_ct/checkpoint/figure12_deeptensor_checkpoint.pt` | 15.23 MB | 4,000 iters | **VERIFIED** |
| **Figure 10 Improved (30%)**| `results/improved/figure10_yale_denoising/checkpoint/deeptensor_noise_30_improved_checkpoint.pt` | 21.16 MB | 4,000 epochs | **VERIFIED** |
| **Figure 10 Improved (60%)**| `results/improved/figure10_yale_denoising/checkpoint/deeptensor_noise_60_improved_checkpoint.pt` | 21.16 MB | 4,000 epochs | **VERIFIED** |
| **Figure 11 Improved** | `results/improved/figure11_video_cs/checkpoint/figure11_deeptensor_improved_checkpoint.pt` | 15.98 MB | 10,000 epochs | **VERIFIED** |
| **Figure 12 Improved** | `results/improved/figure12_pet_ct/checkpoint/figure12_deeptensor_improved_checkpoint.pt` | 15.24 MB | 4,000 epochs | **VERIFIED** |
| **Figure 13b Improved (R=4k)**| `results/improved/figure13b_yale_rank/checkpoint/deeptensor_rank_4000_checkpoint.pt` | 13.98 MB | 1,500 epochs | **VERIFIED** |
| **Figure 13b Rank Sweeps** | `results/improved/figure13b_yale_rank/checkpoint/deeptensor_rank_{10,50,100,500,1000,2000}_checkpoint.pt` | 2.04–8.00 MB | Full sweep | **VERIFIED** |

### 3.4 SHA-256 Manifest Comparison
1. **`CLEANUP_MANIFEST.csv` Audit:**
   - Manifest entries checked: 502
   - Files verified on disk: 502
   - Missing files: **0**
   - SHA-256 hash mismatches: **0**
   - Finding: **VERIFIED (100% bit-for-bit integrity)**
2. **`SHA256_BEFORE.csv` Cross-Verification:**
   - Pre-migration baseline records: 108,148
   - Directly verified in place: 108,124
   - Relocated via `git mv` (tracked in `MIGRATION_MANIFEST.csv`): 24
   - Total accounted for: 108,148 / 108,148
   - Unaccounted / lost files: **0**
   - Finding: **VERIFIED (Zero Data Loss Evidence-Backed)**

---

## 4. Git History & Working Tree Audit

### 4.1 Git Status & Commit State
- **Branch:** `main` (Ahead of `origin/main` by 1 commit: `b37e7d8`)
- **Working Tree Cleanliness:** **UNCOMMITTED CHANGES PRESENT**
- **Changes Staged for Commit:** **138 files** (All staged as `renamed: old -> new` via `git mv`)
- **Changes Not Staged:** 4 modified files (`.gitignore`, `README.md`, `apps/streamlit_demo/demo.py`, `apps/streamlit_demo/demo_backend.py`)
- **Untracked Directories:** `src/`, `configs/`, `reports/`, `results/improved/`, `migration/`, `tests/test_smoke_models.py`, `tools/execute_final_cleanup.py`, `FINAL_CLEAN_PROJECT_STRUCTURE.md`, `FINAL_CLEANUP_VALIDATION.md`, `PROJECT_STRUCTURE.txt`.
- **Finding:** **WARNING**. The structural cleanup is completely staged/present in the working tree, but has NOT been committed to the Git repository.

### 4.2 Relocated Tracked Files History Tracing
Running `git log -5 -- <original_file>` confirms historical provenance is intact:
- `demo.py`: Tracked to commit `f36de09` (Nov 5, 2026)
- `modules/models.py`: Tracked to initial commit
- `RESULTS-BASELINE/figure2/configuration.json`: Tracked to commit `f36de09`
- `documentation/extension_1_adaptive_rank/ADAPTIVE_RANK_SELECTION_DOCUMENTATION.md`: Tracked to commit `918a96c`
- **Finding:** **VERIFIED**. Full Git commit history is preserved and will be carried forward upon commit.

---

## 5. Review of Code Modifications

An independent inspection of `git diff` on all modified files reveals:

### 5.1 `.gitignore`
- **Diff:**
  ```diff
  -# Large binaries exceeding GitHub file size limit (100MB)
  -RESULTS-BASELINE/figure9/figure9_results.npz
  +results/baseline/figure09_hyperspectral/figure9_results.npz
  ```
- **Rationale:** Updates path to the 159 MB hyperspectral results binary following the relocation of `RESULTS-BASELINE/`.
- **Necessity:** Mandatory to prevent GitHub push rejections.
- **Algorithm Risk:** 0% (Git metadata only).
- **Finding:** **VERIFIED**.

### 5.2 `README.md`
- **Diff:** Complete rewrite of legacy notes to a publication-grade README with architecture diagram, editable package installation instructions, verified results table (Figures 2–13, Tables 1–3), and experiment execution commands.
- **Rationale:** Provides modern onboarding and documentation.
- **Necessity:** Necessary for usability.
- **Algorithm Risk:** 0% (Documentation only).
- **Finding:** **VERIFIED**.

### 5.3 `apps/streamlit_demo/demo.py`
- **Diff:**
  ```python
  +sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
   from demo_backend import DemoBackend
  ```
- **Rationale:** Ensures `demo_backend.py` is discoverable on `sys.path` when running from workspace root.
- **Necessity:** Strictly necessary to prevent `ModuleNotFoundError: No module named 'demo_backend'`.
- **Algorithm Risk:** 0% (Application import path only).
- **Finding:** **VERIFIED**.

### 5.4 `apps/streamlit_demo/demo_backend.py`
- **Diff:**
  ```python
  -sys.path.append('modules')
  -import models
  -import utils
  -import losses
  -import spectral
  -import deep_prior
  +# Add legacy modules and deeptensor package to path
  +cur_dir = os.path.dirname(os.path.abspath(__file__))
  +root_dir = os.path.abspath(os.path.join(cur_dir, '..', '..'))
  +legacy_dir = os.path.join(root_dir, 'archive', 'legacy', 'modules_legacy')
  +for p in [root_dir, legacy_dir, os.path.join(root_dir, 'src')]:
  +    if p not in sys.path:
  +        sys.path.insert(0, p)
  +
  +try:
  +    import deeptensor.models as models
  +    import deeptensor.utils as utils
  +    import deeptensor.losses as losses
  +    import deeptensor.models.spectral as spectral
  +    import deeptensor.models.deep_prior as deep_prior
  +except ImportError:
  +    import models
  +    import utils
  +    import losses
  +    import spectral
  +    import deep_prior
  ```
- **Rationale:** Resolves imports from the new `deeptensor` library with graceful fallback to `archive/legacy/modules_legacy`.
- **Necessity:** Strictly necessary to prevent `ModuleNotFoundError: No module named 'models'` on demo startup.
- **Algorithm Risk:** 0% (Backend data loading and algorithms untouched).
- **Finding:** **VERIFIED**.

### 5.5 `tests/test_smoke_models.py`
- **Diff:** Replaced `pytest` test runner dependency with standard library `unittest.TestCase`.
- **Rationale:** Virtual environment `.venv` lacks `pytest`.
- **Necessity:** Allows running unit smoke tests in vanilla Python 3.11 environment.
- **Algorithm Risk:** 0% (Test harness syntax only).
- **Finding:** **VERIFIED**.

---

## 6. Audit of Broken Paths and Stale References

An automated static code analysis scanning 55 Python scripts across `experiments/`, `apps/`, `tools/`, and `tests/` identified **30 scripts with broken imports or stale paths**:

```text
========================================================================================
                          BROKEN IMPORT REFERENCES AUDIT
========================================================================================
```

### 6.1 Critical Broken Imports (Throws `ModuleNotFoundError: No module named 'models'`)
When these scripts are executed directly (e.g., `python experiments/reproduction/figure02_eigenfaces/run_figure2.py`), they fail on line 37 because `sys.path.append('modules')` looks for `./modules` relative to current working directory, which no longer exists:

1. `experiments/reproduction/figure02_eigenfaces/run_figure2.py`: Line 35 (`sys.path.append('modules')`), Lines 37–40 (`import models, utils, losses, spectral`) — **FAILED**
2. `experiments/reproduction/figure06_synthetic_robustness/run_figure6.py`: Line 30 (`sys.path.append('modules')`), Lines 31–34 (`import models, utils, losses, spectral`) — **FAILED**
3. `experiments/reproduction/figure07_robust_pca/run_figure7.py`: Line 32 (`sys.path.append('modules')`), Lines 33–36 (`import models, utils, losses, spectral`) — **FAILED**
4. `experiments/reproduction/figure09_hyperspectral/run_figure9.py`: Line 35 (`sys.path.append('modules')`), Lines 36–40 (`import models, utils, losses, spectral, deep_prior`) — **FAILED**
5. `experiments/reproduction/figure09_hyperspectral/run_figure9_bm3d.py`: Line 21 (`sys.path.append('modules')`), Line 22 (`import utils`) — **FAILED**
6. `experiments/reproduction/figure11_video_cs/run_figure11.py`: Line 25 (`sys.path.append('../modules')`), Lines 27–31 (`import models, utils, losses, lin_inverse, deep_prior`) — **FAILED**
7. `experiments/reproduction/figure11_video_cs/run_figure11_TV.py`: Line 25 (`sys.path.append('modules')`), Lines 27–29 (`import utils, losses, lin_inverse`) — **FAILED**
8. `experiments/reproduction/figure12_pet_ct/run_figure12.py`: Line 39 (`sys.path.append('modules')`), Lines 40–44 (`import models, utils, losses, lin_inverse, deep_prior`) — **FAILED**
9. `experiments/reproduction/figure12_pet_ct/run_figure12_TV.py`: Line 23 (`sys.path.append('modules')`), Lines 25–27 (`import utils, losses, lin_inverse`) — **FAILED**
10. `experiments/reproduction/table01_synthetic/run_table1.py`: Line 21 (`sys.path.append("modules")`), Lines 25–28 (`import models, utils, losses, spectral`) — **FAILED**
11. `experiments/extensions/adaptive_rank/run_figure9_adaptive.py`: Line 55 (`sys.path.append('modules')`), Lines 57–61 (`import models, utils, losses, spectral, deep_prior`) — **FAILED**
12. `experiments/extensions/ensemble_uncertainty/run_figure9_ensemble.py`: Line 23 (`sys.path.append('modules')`), Lines 26–30 (`import deep_prior, losses, models, spectral, utils`) — **FAILED**
13. `experiments/extensions/factor_interpretability/run_figure9_interpretability.py`: Line 26 (`sys.path.append('modules')`), Lines 28–32 (`import models, utils, losses, spectral, deep_prior`) — **FAILED**
14. `experiments/extensions/group_sparsity/run_figure9_sparsity.py`: Line 37 (`sys.path.append('modules')`), Lines 39–43 (`import models, utils, losses, spectral, deep_prior`) — **FAILED**
15. `experiments/extensions/hybrid_attention/run_figure9_hybrid.py`: Line 16 (`sys.path.append('modules')`), Lines 17–20 (`import utils, losses, deep_prior, spectral`) — **FAILED**
16. `experiments/extensions/noise_robustness/run_figure9_robustness.py`: Line 28 (`sys.path.append('modules')`), Lines 30–34 (`import models, utils, losses, spectral, deep_prior`) — **FAILED**
17. `experiments/extensions/spectral_transformer/run_figure9_spectral_transformer.py`: Line 16 (`sys.path.append('modules')`), Lines 17–21 (`import utils, losses, deep_prior, spectral, spectral_transformer`) — **FAILED**
18. `experiments/extensions/tensor_completion/run_figure9_completion.py`: Line 39 (`sys.path.append('modules')`), Lines 41–45 (`import models, utils, losses, spectral, deep_prior`) — **FAILED**

### 6.2 Hardcoded Output Paths to Obsolete Root Folders
Executing these scripts without `--output_dir` will recreate deleted legacy root directories:
- `experiments/reproduction/figure02_eigenfaces/run_figure2.py`: Line 88 (`out_dir = 'RESULTS-BASELINE/figure2'`) — **FAILED**
- `experiments/reproduction/figure06_synthetic_robustness/run_figure6.py`: Line 91 (`out_dir = 'RESULTS-BASELINE/figure6'`) — **FAILED**
- `experiments/reproduction/figure07_robust_pca/run_figure7.py`: Line 83 (`out_dir = 'RESULTS-BASELINE/figure7'`) — **FAILED**
- `experiments/reproduction/figure09_hyperspectral/run_figure9.py`: Line 57 (`out_dir = 'RESULTS-BASELINE/figure9'`) — **FAILED**
- `experiments/reproduction/figure12_pet_ct/run_figure12.py`: Line 62 (`out_dir = 'RESULTS-BASELINE/figure12'`) — **FAILED**
- `experiments/reproduction/figure10_yale_denoising/run_figure10_yale_tensor.py`: Line 147 (`default='paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising'`) — **FAILED**
- `experiments/reproduction/figure11_video_cs/run_figure11_baseline.py`: Line 224 (`default='paper_reproduction/cnn_baseline/07_video_compressive_sensing'`) — **FAILED**
- `experiments/reproduction/figure13a_hsi_rank/run_figure13_rank_sweep.py`: Line 94 (`default='paper_reproduction/cnn_baseline/05_hyperspectral_denoising'`) — **FAILED**
- `experiments/reproduction/figure13b_yale_rank/run_figure13b_yale_rank_sweep.py`: Line 95 (`default='paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising'`) — **FAILED**
- `experiments/reproduction/table01_synthetic/run_synthetic_suite.py`: Line 352 (`default='paper_reproduction/cnn_baseline/01_synthetic_decomposition'`) — **FAILED**
- `experiments/reproduction/table02_nmf/run_table2_nmf.py`: Line 187 (`default='paper_reproduction/cnn_baseline/03_nmf'`) — **FAILED**
- `experiments/reproduction/table03_speech/run_table3_speech_nmf.py`: Line 224 (`default='paper_reproduction/cnn_baseline/09_speech_time_frequency'`) — **FAILED**
- `experiments/extensions/ensemble_uncertainty/run_figure9_ensemble.py`: Line 72 (`out_dir = 'results_ensemble_smoke' if args.smoke_test else 'results_ensemble'`) — **FAILED**
- `experiments/extensions/factor_interpretability/run_figure9_interpretability.py`: Line 260 (`out_dir = "results_interpretability_smoke" if smoke_test else "results_interpretability"`) — **FAILED**
- `experiments/extensions/group_sparsity/run_figure9_sparsity.py`: Line 77 (`default='results_sparsity'`) — **FAILED**
- `experiments/extensions/hybrid_attention/run_figure9_hybrid.py`: Line 36 (`os.makedirs('results_hybrid', exist_ok=True)`) — **FAILED**
- `experiments/extensions/noise_robustness/run_figure9_robustness.py`: Line 251 (`res_dir = "results_robustness"`) — **FAILED**
- `experiments/extensions/spectral_transformer/run_figure9_spectral_transformer.py`: Line 36 (`os.makedirs('results_spectral_transformer', exist_ok=True)`) — **FAILED**
- `experiments/extensions/tensor_completion/run_figure9_completion.py`: Line 79 (`default='results_completion'`) — **FAILED**

### 6.3 Stale YAML Configuration Paths
- `configs/improvements/figure11_video.yaml`: Line 11 (`output_dir: paper_reproduction/cnn_baseline/07_video_compressive_sensing`) — **FAILED**
- `configs/reproduction/figure11_video_cs.yaml`: Line 11 (`output_dir: paper_reproduction/cnn_baseline/07_video_compressive_sensing`) — **FAILED**

---

## 7. Experiment Separation Verification

| Check | Requirement | Verification Finding | Status |
|---|---|---|---|
| **Reproduction Suites** | Original paper reproduction targets available | Available in `experiments/reproduction/` (13 figure/table suites) | **VERIFIED** |
| **Improvement Suites** | High-performance improved runs segregated | Segregated in `experiments/improvements/` (Figures 10, 11, 12, 13b) | **VERIFIED** |
| **Extension Suites** | 9 ICVL extensions separated from reproduction baseline | Segregated in `experiments/extensions/` (8 suites) and `results/extensions/` | **VERIFIED** |
| **Hybrid Prototypes** | Future CNN-Transformer decoupled from completed experiments | Decoupled in `experiments/hybrid_prototypes/` (placeholder) | **VERIFIED** |
| **Historical Experiments**| Previous NeuroTensor runs preserved | 62 files preserved under `experiments/legacy_experiments/previous_neurotensor_experiments/` | **VERIFIED** |
| **Baseline Overwrite Protection** | No baseline result overwritten by improved result | `results/baseline/` and `results/improved/` maintain distinct checkpoints, logs, and metrics | **VERIFIED** |

---

## 8. Test Execution Results

Actual validation executed in project virtual environment (`.\.venv\Scripts\python.exe`):

### Test 1: Comprehensive Unit Test Suite
- **Command:** `.\.venv\Scripts\python.exe tests/run_all_tests.py`
- **Exit Code:** `0`
- **Tests Run:** 7
- **Failures:** 0
- **Duration:** 5.773s
- **Output Summary:**
  ```text
  test_01_version_and_imports (TestDeepTensorPackage) ... ok
  test_02_metrics (TestDeepTensorPackage) ............. ok
  test_03_simple_forward_2d (TestDeepTensorPackage) ... ok
  test_04_losses (TestDeepTensorPackage) .............. ok
  test_05_physics_video2coded (TestDeepTensorPackage) . ok
  test_06_datasets_exist (TestDeepTensorPackage) ...... ok
  test_07_parafac_als_cuda_smoke (TestDeepTensorPackage) ok
  Ran 7 tests in 5.773s - OK
  ```
- **Finding:** **VERIFIED**

### Test 2: Smoke Model Tests
- **Command:** `.\.venv\Scripts\python.exe tests/test_smoke_models.py`
- **Exit Code:** `0`
- **Tests Run:** 4
- **Failures:** 0
- **Duration:** 0.884s
- **Output Summary:** `Ran 4 tests in 0.884s - OK`
- **Finding:** **VERIFIED**

### Test 3: Dataset Path Verification
- **Command:** `.\.venv\Scripts\python.exe -c "import tests.test_dataset_paths as tdp; tdp.test_mat_datasets_exist(); tdp.test_benchmark_dirs_exist(); print('Dataset paths: ALL PASS')"`
- **Exit Code:** `0`
- **Output Summary:** `Dataset paths: ALL PASS`
- **Finding:** **VERIFIED**

### Test 4: Package Import Unit Test File
- **Command:** `.\.venv\Scripts\python.exe tests/test_imports.py`
- **Exit Code:** `1`
- **Error:** `ModuleNotFoundError: No module named 'pytest'`
- **Finding:** **FAILED / WARNING**. `tests/test_imports.py` requires `pytest` which is not installed in `.venv`.

### Test 5: Interactive Streamlit Demo Backend
- **Command:** `.\.venv\Scripts\python.exe -c "from apps.streamlit_demo.demo_backend import DemoBackend; b = DemoBackend('icvl'); print('Demo Backend OK on', b.device)"`
- **Exit Code:** `0`
- **Output Summary:** `Demo Backend OK on cuda` (Loaded `icvl.mat` $174 \times 164 \times 260$ successfully).
- **Finding:** **VERIFIED**

---

## 9. Outstanding Risks & Recommended Actions

| Priority | Issue / Risk Description | Severity | Recommended Remediation |
|---|---|---|---|
| **HIGH** | **30 experiment runner scripts fail on import** due to `sys.path.append('modules')` and `import models`. | Functional Blocker | Modernize imports across all runner scripts to import from `deeptensor` (e.g. `from deeptensor.models import DeepTensor`) or inject `archive/legacy/modules_legacy` dynamically into `sys.path`. |
| **HIGH** | **Hardcoded legacy output directories** in runner scripts and YAML configs (`RESULTS-BASELINE/`, `paper_reproduction/`, `results_sparsity/`) will recreate deleted root folders upon execution. | Integrity Risk | Refactor default `--output_dir` arguments in all experiment runners and YAML files to point to canonical paths in `results/baseline/` and `results/extensions/`. |
| **MEDIUM** | **Working tree has uncommitted structural changes** (138 staged renames, 4 modified files, untracked new packages). | Git Hygiene Risk | Create a formal Git commit (e.g., `refactor: structural consolidation and root cleanup`) to lock in the directory history. |
| **LOW** | **`tests/test_imports.py` crashes** due to uninstalled `pytest` dependency. | Usability Risk | Refactor `tests/test_imports.py` to use Python standard library `unittest` (consistent with `run_all_tests.py`), eliminating external test runner dependencies. |

---

## 10. Audit Summary Table

| Audit Dimension | Target Requirement | Independent Finding | Classification |
|---|---|---|---|
| **1. Root Directory Cleanliness** | No obsolete folders at root | 21/21 obsolete directories eliminated | **VERIFIED** |
| **2. Canonical Directory Structure** | Canonical architecture populated | All 11 canonical directories populated | **VERIFIED** |
| **3. Dataset Preservation** | 5.83 GB single-source store | All 5 MAT + 4 benchmark datasets intact | **VERIFIED** |
| **4. Checkpoint Preservation** | All baseline and improved weights intact | 61/61 checkpoint & NPZ files preserved | **VERIFIED** |
| **5. SHA-256 Manifest Integrity** | 100% bit-for-bit file match | 502/502 manifest files verified on disk | **VERIFIED** |
| **6. Git History Preservation** | Tracked file relocations preserve history | 138 tracked files staged via `git mv` | **VERIFIED** |
| **7. Working Tree Status** | Status reported accurately | Cleanup remains uncommitted in working tree | **WARNING** |
| **8. Code Modifications Review** | Modified files inspected and justified | 4 modified files verified necessary & safe | **VERIFIED** |
| **9. Script Import Consistency** | Runner scripts resolve modules cleanly | 30 scripts throw `No module named 'models'` | **FAILED** |
| **10. Script Output Directories** | Runners output to canonical results | Hardcoded paths to obsolete folders remain | **FAILED** |
| **11. Experiment Separation** | Baseline vs Improved vs Extensions isolated | Strict subfolder isolation verified | **VERIFIED** |
| **12. Regression & Smoke Tests** | Automated tests pass on CUDA | 7 unit tests + 4 smoke tests pass (100%) | **VERIFIED** |

