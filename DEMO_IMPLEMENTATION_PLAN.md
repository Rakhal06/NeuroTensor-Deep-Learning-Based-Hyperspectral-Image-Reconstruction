# DEMO_IMPLEMENTATION_PLAN

## 1. Current Repository Structure
- **Root Directory**: Contains multiple experiment scripts (`run_figure9.py`, `run_figure9_adaptive.py`, `run_figure9_sparsity.py`, etc.).
- **Modules (`modules/`)**: Contains core network components (`deep_prior.py`, `deep_decoder.py`, `models.py`), utilities (`utils.py`, `spectral.py`), and loss functions (`losses.py`).
- **Data (`data/`)**: Contains MAT files for experiments (`icvl.mat`, `pet.mat`, `cat_video.mat`, `weizzman.mat`).
- **Results (`results/`, `results_sparsity/`, etc.)**: Contains `.png` plots and text summaries (`.txt`, `.md`) from previous runs.

## 2. Existing Implementations
- **Baseline**: `run_figure9.py` [IMPLEMENTED — VERIFIED]
- **Adaptive Rank Selection**: `run_figure9_adaptive.py` [IMPLEMENTED — VERIFIED] (Uses plateau-based rank expansion).
- **Factor Group-Sparsity**: `run_figure9_sparsity.py` [IMPLEMENTED — VERIFIED] (Uses Group Lasso regularizer on unnormalized factors U and V).

## 3. Existing Checkpoints
- **Checkpoints**: [NOT AVAILABLE] / [DESIGN ONLY — NOT IMPLEMENTED]
  *Inspection shows that while final plots and text summaries are saved, actual model checkpoints (`.pt`/`.pth`), optimized input tensors, and optimizer states are NOT saved by the current scripts.*

## 4. Existing Result Files
- **Plots**: Yes (e.g., `adaptive_rank_results.png`, `plot_lambda_0.001_seed_42.png`).
- **Text Summaries**: Yes (e.g., `summary_lambda_0.001_seed_42.txt`, `baseline_figure9_report.md`).

## 5. Existing Metrics
- **Metrics**: [IMPLEMENTED — VERIFIED] 
  *Scripts currently calculate and log PSNR, SSIM, RMSE, L1 Loss, Rank history, Group norms (`g_r`), and Component Powers.*

## 6. Existing Visualization Capability
- **Visualization**: [IMPLEMENTED — VERIFIED]
  *Matplotlib is used to plot reconstruction comparisons, diff images, loss curves, and component power distributions.*

## 7. What Can Be Reused
- The core logic from `run_figure9.py`, `run_figure9_adaptive.py`, and `run_figure9_sparsity.py`.
- Network architectures, data loaders, and metric computations from `modules/`.

## 8. What Needs To Be Implemented
1. **Checkpointing System**: Must implement saving and loading of model weights (`state_dict`), optimized input tensors (`im_inp`, `spec_inp`), and configuration parameters (rank, lambda, etc.) for fast `DEMO MODE`.
2. **Master `demo.py` Script**: An interactive terminal application with menus, modes (`demo`, `full`, `compare`), and clean console output.
3. **Forward Reconstruction Pipeline**: Code to load checkpoints and perform actual forward passes without retraining.
4. **Three-Way Comparison**: Logic to run all three methods on the same input sample and aggregate metrics into a single table and plot.

## 9. What Must Remain Untouched
- The mathematical formulations and network architectures in `modules/`.
- Existing frozen result files from past baseline/extension validations.
- The core algorithm of Adaptive Rank and Group Sparsity.

## 10. Exact Plan for the Demonstration System
1. **Step 1: Core wrapper**: Write `demo_backend.py` to modularize the `full` training and `demo` evaluation (checkpoint loading + forward pass) for Baseline, Adaptive Rank, and Group Sparsity.
2. **Step 2: Checkpoint Support**: Within `demo_backend.py`, add explicit save routines for `.pt` files that bundle `im_net`, `spec_net`, `im_inp`, `spec_inp`, `current_rank`, `lambda_val`.
3. **Step 3: Train Checkpoints**: Run `demo_backend.py` with 10 epochs (or copy weights if we can train it later) to generate checkpoints so the demo doesn't fail.
4. **Step 4: Three-Way Comparison**: Create `compare` function inside `demo_backend.py` to evaluate all three on `icvl.mat`.
5. **Step 5: Master CLI**: Write `demo.py` interactive console to satisfy user requirements.
