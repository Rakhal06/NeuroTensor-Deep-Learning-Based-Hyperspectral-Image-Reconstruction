# DeepTensor++ — File Duplication, Git Tracking & Retention Audit

> **Audit Date:** November 6, 2026  
> **Workspace Root:** `e:\Rakhal_Tensor`  
> **Audit Status:** Comprehensive SHA-256 Hash Matching, Git History Forensics, and AST Dependency Analysis.  
> **Strict Retention Constraint:** Strictly Read-Only (Zero files deleted, moved, or renamed).

---

## 1. Executive Summary & Taxonomy

To ensure zero risk of data loss, loss of Git commit history, or loss of research provenance during the planned reorganization into `DeepTensor-Plus-Plus/`, this audit performs a forensic examination of every duplicate, backup, and historical file in the workspace.

Files are classified into five rigorous categories:
1. **Exact Duplicate**: Byte-for-byte identical (identical SHA-256 hash).
2. **Modified Copy**: Shares common ancestry or design but has divergent line counts, bug fixes, or enhancements.
3. **Historical Implementation**: Earlier prototype or baseline superseded by an improved runner.
4. **Backup**: Explicit snapshot or backup copy created prior to modifications.
5. **Unknown**: Origin or purpose cannot be completely resolved from code or Git history alone.

---

## 2. Complete Inventory of SHA-256 Exact Duplicates (11 Groups)

| SHA-256 Hash Prefix | File Path 1 (Canonical / Tracked) | File Path 2 (Duplicate Copy) | Git Tracking Status | Execution / Import Dependencies | Migration Action & Safety Rule |
|---|---|---|---|---|---|
| `eef933a3` | `run_figure9.py` (339 lines) | `previous_neurotensor_experiments/01_icvl_cnn_baseline/run_figure9.py` | File 1 is **Git-Tracked** (`f36de09`); File 2 is **Untracked** | Standalone runner; imports `modules/utils.py`, `deep_prior.py` | **DO NOT DELETE FILE 1**. Move File 1 via `git mv` to target archive to preserve Git commit history. |
| `da1287c6` | `run_figure9_adaptive.py` (481 lines) | `previous_neurotensor_experiments/02_icvl_adaptive_rank/run_figure9_adaptive.py` | File 1 is **Git-Tracked** (`85ff484`); File 2 is **Untracked** | Extension 1 runner; imports `modules/` | Move File 1 via `git mv` to target archive. |
| `a274c154` | `run_figure9_sparsity.py` (448 lines) | `previous_neurotensor_experiments/03_icvl_factor_group_sparsity/run_figure9_sparsity.py` | File 1 is **Git-Tracked** (`600a5b2`); File 2 is **Untracked** | Extension 2 runner; imports `modules/` | Move File 1 via `git mv` to target archive. |
| `346cb299` | `run_figure9_completion.py` (439 lines) | `previous_neurotensor_experiments/04_icvl_tensor_completion/run_figure9_completion.py` | File 1 is **Git-Tracked** (`600a5b2`); File 2 is **Untracked** | Extension 3 runner; imports `modules/` | Move File 1 via `git mv` to target archive. |
| `09ac1388` | `run_figure9_robustness.py` (269 lines) | `previous_neurotensor_experiments/05_icvl_noise_robustness/run_figure9_robustness.py` | File 1 is **Git-Tracked** (`4961e52`); File 2 is **Untracked** | Extension 4 runner; imports `modules/` | Move File 1 via `git mv` to target archive. |
| `0e343a35` | `run_figure9_ensemble.py` (365 lines) | `previous_neurotensor_experiments/06_icvl_ensemble_uncertainty/run_figure9_ensemble.py` | File 1 is **Git-Tracked** (`e271eb3`); File 2 is **Untracked** | Extension 5 runner; imports `modules/` | Move File 1 via `git mv` to target archive. |
| `91faab6c` | `run_figure9_interpretability.py` (400 lines) | `previous_neurotensor_experiments/07_icvl_factor_interpretability/run_figure9_interpretability.py` | File 1 is **Git-Tracked** (`fede681`); File 2 is **Untracked** | Extension 7 runner; imports `modules/` | Move File 1 via `git mv` to target archive. |
| `a2959c7a` | `run_figure9_hybrid.py` (241 lines) | `previous_neurotensor_experiments/08_icvl_hybrid_attention/run_figure9_hybrid.py` | File 1 is **Git-Tracked** (`6dae2c6`); File 2 is **Untracked** | Extension 8 runner; imports `attention_spectral` | Move File 1 via `git mv` to target archive. |
| `283768c6` | `run_figure9_spectral_transformer.py` (258 lines) | `previous_neurotensor_experiments/09_icvl_spectral_transformer/run_figure9_spectral_transformer.py` | File 1 is **Git-Tracked** (`e4b9ebf`); File 2 is **Untracked** | Extension 9 runner; imports `spectral_transformer` | Move File 1 via `git mv` to target archive. |
| `417eb058` | `modules/attention_spectral.py` (207 lines) | `previous_neurotensor_experiments/08_icvl_hybrid_attention/modules/attention_spectral.py` | File 1 is **Git-Tracked** (`6dae2c6`); File 2 is **Untracked** | Core architectural layer; imported by Extension 8 | Canonical home is `modules/` (future `src/deeptensor/models/attention/`). |
| `39a81075` | `modules/spectral_transformer.py` (155 lines) | `previous_neurotensor_experiments/09_icvl_spectral_transformer/modules/spectral_transformer.py` | File 1 is **Git-Tracked** (`e4b9ebf`); File 2 is **Untracked** | Core architectural layer; imported by Extension 9 | Canonical home is `modules/` (future `src/deeptensor/models/attention/`). |

---

## 3. Git Tracking Forensics & Ancestry Analysis

### 3.1 The Git Tracking Paradox
A naive file inspection suggested that the 9 `run_figure9*.py` scripts in the root directory were "temporary duplicates" that should simply be deleted in favor of `previous_neurotensor_experiments/`.

**Our forensic Git audit revealed that this would destroy repository commit history:**
- **Git Fact 1**: The root files (`run_figure9_adaptive.py`, `run_figure9_sparsity.py`, etc.) were committed by `rakhal06` **9 weeks ago** with explicit commit messages:
  - Commit `85ff484`: *"Fix adaptive rank reporting for incomplete cooldown"*
  - Commit `600a5b2`: *"Complete core extensions and research documentation"*
  - Commit `4961e52`: *"Extension 4: noise-type robustness at matched input corruption"*
  - Commit `e271eb3`: *"Extension 5: ensemble disagreement uncertainty analysis"*
  - Commit `fede681`: *"Extension 7: post-hoc factor analysis and component visualization"*
  - Commit `6dae2c6`: *"Extension 8: hybrid CNN-transformer spectral attention"*
  - Commit `e4b9ebf`: *"Extension 9: Transformer-dominant spectral generator"*
  - Commit `f36de09`: *"Add baseline experiment suite, interactive demo, documentation, and research artifacts"*
- **Git Fact 2**: The directory `previous_neurotensor_experiments/` is completely **untracked** (`?? previous_neurotensor_experiments/` in `git status`). It was created locally as an archival experiment without committing.

### 3.2 Safe Relocation Strategy
Instead of deleting root files:
- Use `git mv run_figure9*.py experiments/historical_extensions/` to preserve 9 weeks of commit authoring history and blamability.
- The untracked files in `previous_neurotensor_experiments/` can then be safely cleared or merged.

---

## 4. Modified Copies & Historical Backups

### 4.1 Interactive Demo Files
1. **`demo.py` (330 lines, 12,326 bytes) vs `demo_backup.py` (317 lines, 11,765 bytes)**:
   - **Classification**: **Backup (Historical Snapshot)**.
   - **Diff Analysis**: `demo.py` contains CPU/CUDA dynamic device selection (`self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`), whereas `demo_backup.py` has hardcoded `.cuda()`.
   - **Dependency**: Only `demo.py` is called by users or terminal launchers. `demo_backup.py` is uncalled.
   - **Retention**: Archive `demo_backup.py` to `archive/backups/`.
2. **`demo_backend.py` (569 lines, 26,062 bytes) vs `demo_backend_backup.py` (539 lines, 24,499 bytes)**:
   - **Classification**: **Backup (Historical Snapshot)**.
   - **Diff Analysis**: Same evolution: `demo_backend.py` contains proper device fallbacks and updated rank sweeps.
   - **Dependency**: `demo.py` imports `DemoBackend` from `demo_backend.py`.
   - **Retention**: Archive `demo_backend_backup.py` to `archive/backups/`.

---

### 4.2 Runner Prototypes vs Standardized Suites
1. **`run_figure11.py` in root (205 lines) vs `paper_reproduction/.../run_figure11.py` (415 lines) vs `run_figure11_improved.py` (497 lines)**:
   - **Classification**:
     - Root `run_figure11.py`: **Historical Implementation (Prototype)**. Contains hardcoded `sys.path.append('../modules')` and no JSON metric dumping.
     - `paper_reproduction/.../run_figure11.py`: **Baseline Experiment Runner**. Standard Phase 1 runner ($R=32$, 3000 ep, saves `metrics.json`).
     - `paper_reproduction/.../run_figure11_improved.py`: **Improved Experiment Runner**. $R=64$, 128 channels, 10,000 epochs, progression CSV logging.
   - **Retention**: Keep `run_figure11_improved.py` as primary. Move root `run_figure11.py` to `archive/legacy_runners/`.
2. **`run_figure12.py` in root (371 lines) vs `paper_reproduction/.../run_figure12_improved.py` (463 lines)**:
   - **Classification**:
     - Root `run_figure12.py`: **Historical Implementation**. Lacks Poisson $\tau=100$ noise calibration and cosine LR scheduling.
     - `paper_reproduction/.../run_figure12_improved.py`: **Approved Improved Runner**. Implements exact paper noise, 40-angle projection adjoint, and saves checkpoints.
   - **Retention**: Keep `run_figure12_improved.py` as primary. Move root `run_figure12.py` to `archive/legacy_runners/`.

---

### 4.3 Miscellaneous Diagnostic Scripts in Root
1. **`check_attn.py` (9 lines)**:
   - **Classification**: **Diagnostic / Smoke Test**.
   - **Purpose**: Quick 9-line script verifying that `from modules.attention_spectral import HybridSpectralNet` imports without syntax errors.
   - **Retention**: Move to `tools/diagnostics/`.
2. **`fix_newlines.py` (19 lines)**:
   - **Classification**: **Utility / One-Off**.
   - **Purpose**: Strips CRLF Windows line endings from text files.
   - **Retention**: Move to `tools/utilities/`.
3. **`scratch_calibrate.py` (49 lines)**:
   - **Classification**: **Diagnostic / Calibration**.
   - **Purpose**: Diagnostic calculating PSNR scaling across noise parameters.
   - **Retention**: Move to `tools/diagnostics/`.
4. **`modify_robustness.py` (157 lines)**:
   - **Classification**: **Utility / Visualization**.
   - **Purpose**: Regerating Figure 6 noise robustness plots with custom styling.
   - **Retention**: Move to `tools/utilities/`.
5. **`matrix_decomp.py` (156 lines)**:
   - **Classification**: **Historical Implementation**.
   - **Purpose**: Early synthetic matrix factorization experiments using PyMF.
   - **Retention**: Move to `archive/legacy_runners/`.

---

## 5. Master Retention Matrix & Action Plan

| Category | File Paths | Current Tracking | Target Reorganization Action | Safety Justification |
|---|---|---|---|---|
| **Git-Tracked Canonical Extensions** | `run_figure9_adaptive.py`<br>`run_figure9_sparsity.py`<br>`run_figure9_completion.py`<br>`run_figure9_robustness.py`<br>`run_figure9_ensemble.py`<br>`run_figure9_interpretability.py`<br>`run_figure9_hybrid.py`<br>`run_figure9_spectral_transformer.py`<br>`run_figure9.py` | Git-Tracked (`rakhal06`, 9 weeks ago) | **`git mv`** to `experiments/historical_extensions/` | Preserves all Git commit history, timestamps, and blamability. |
| **Untracked Duplicate Folders** | `previous_neurotensor_experiments/` | Untracked (`??`) | Consolidate unique documentation and plots; remove duplicate `.py` files | Resolves redundancy without losing tracked commits. |
| **Duplicate Architectural Modules** | `previous_neurotensor_experiments/*/modules/*.py` | Untracked (`??`) | Remove redundant copies | Canonical modules reside in `modules/` (future `src/deeptensor/models/attention/`). |
| **Historical Demo Backups** | `demo_backup.py`<br>`demo_backend_backup.py` | Tracked in Git | Move to `archive/backups/` | Clean workspace while preserving evolutionary history. |
| **Superseded Root Runners** | `run_figure11.py`<br>`run_figure11_TV.py`<br>`run_figure12.py`<br>`run_figure12_TV.py` | Tracked in Git | Move to `experiments/reproduction/<domain>/legacy/` | Prevents confusion with audited improved runners. |
| **Diagnostics & Utilities** | `check_attn.py`<br>`fix_newlines.py`<br>`scratch_calibrate.py`<br>`modify_robustness.py`<br>`matrix_decomp.py` | Tracked in Git | Move to `tools/diagnostics/` or `tools/utilities/` | Keeps workspace root clean and professional. |
