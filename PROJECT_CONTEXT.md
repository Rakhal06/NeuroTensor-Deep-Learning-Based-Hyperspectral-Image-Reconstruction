# DeepTensor Research Project: Master Context Document

## 1. Project Identity
- **Project Name:** DeepTensor-Extended (DeepTensor-Low-Rank-Tensor-Decomposition)
- **Faculty:** Not explicitly specified in the current context
- **Students:** Rakhal
- **Research Topic:** DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors
- **Main Objective:** Extend the baseline DeepTensor framework to support Adaptive Rank Selection, removing the need for manual rank specification, with future directions toward sparsity-based rank discovery, tensor completion, robustness, and Transformer-based extensions.

## 2. Research Problem
Classical low-rank tensor decomposition and the original DeepTensor approach require a fixed, manually specified rank hyperparameter (e.g., $R=20$). Selecting this rank is often arbitrary or requires computationally expensive trial-and-error grid searches. The core research problem is how to dynamically and adaptively determine the optimal effective rank during self-supervised optimization without relying on clean ground truth data.

## 3. Proposed Approach
**DeepTensor Baseline:** Instead of using classical linear factors (like CP or Tucker decomposition), DeepTensor parameterizes the spatial and spectral factors of a hyperspectral image using untrained Deep Image Prior (DIP) skip networks. The networks are optimized solely on the noisy observation using self-supervised L1 loss.
**Adaptive Rank Extension:** We introduce a **Plateau-Based Adaptive Rank Selection**. The DIP networks are initialized with a maximum capacity ($R_{max}=30$). During training, only the first $r$ components are used for reconstruction (Effective Rank Masking). When the self-supervised reconstruction loss plateaus, the active rank $r$ is increased. Inactive internal layers continue to learn, while inactive output layers receive zero gradients until unmasked.

## 4. Dataset
- **Primary Dataset:** ICVL Hyperspectral Dataset
- **Secondary Dataset:** (To be determined/applicable in later stages)
- **Relevant Files:** `data/icvl.mat`
- **Data Dimensions/Preprocessing:** Shape `(H, W, nwvl)` normalized. In the reproduced experiment: Gaussian noise added (`noise_snr=2`), `tau=100`.

## 5. Baselines
- **Baseline Methods:** Truncated SVD (Rank 20) and Original DeepTensor (Rank 20).
- **DeepTensor Baseline Configuration:** 5000 epochs, L1 Loss, Adam Optimizer (lr=1e-3), Rank 20, ICVL dataset.

## 6. Baseline Reproduction — COMPLETED
These are **ACTUAL EXPERIMENTAL RESULTS**, empirically verified from independent runs.

- **Hardware:** HP Omen, Windows
- **GPU:** NVIDIA RTX 4060 Laptop GPU (8GB VRAM)
- **Python Version:** 3.11.9
- **PyTorch Version:** 2.5.1+cu121
- **CUDA Setup:** CUDA 12.1
- **Configuration:** 5000 epochs, L1 Loss, Adam (1e-3)
- **Noise Configuration:** `noise_snr=2`, `tau=100`
- **Rank:** 20
- **Runtime:** ~159.57 seconds (Run 1: 159.42s, Run 2: 159.57s)
- **Peak VRAM:** 640.37 MB
- **Dataset:** ICVL (`data/icvl.mat`)
- **Input Noisy PSNR:** 28.43 dB
- **SVD Rank-20 PSNR:** 38.31 dB
- **SVD Rank-20 SSIM:** 0.9169
- **DeepTensor Rank-20 PSNR:** 46.45 dB (Run 1: 46.37 dB)
- **DeepTensor Rank-20 SSIM:** 0.9872 (Run 1: 0.9868)
- **Reproducibility Verdict:** Highly reproducible. Baseline verified and frozen.

## 7. Repository / Git State
- **Original Upstream Repository:** https://github.com/vishwa91/DeepTensor
- **Our GitHub Repository:** https://github.com/Rakhal06/DeepTensor-Low-Rank-Tensor-Decomposition.git
- **Current Branch:** `extension/adaptive-rank`
- **Important Commits:** 
  - `a146cc7` (Baseline: reproduce DeepTensor Figure 9 on RTX 4060) - Frozen state on `main`.
- **Changes in `run_figure9.py`:** Updated `ssim_func` kwargs (`channel_axis=-1`, `data_range=cube.max() - cube.min()`) for `scikit-image` 0.24.0 compatibility.
- **Files Added:** `run_figure9_adaptive.py` (Currently untracked/unstaged)
- **Current Known Git Status:** `run_figure9_adaptive.py` is untracked; working tree otherwise clean relative to `a146cc7`.

## 8. Environment
- **Working Environment:** `E:\Rakhal_Tensor\`
- **Virtual Environment Setup:** `.venv` created with Python 3.11.9.
- **Compatibility Decisions:** The original `requirements.txt` was not strictly adhered to due to package age. Modern PyTorch (2.5.1+cu121) was installed for RTX 4060 support. `scikit-image` required an API fix for `ssim` calculation in `run_figure9.py`.

## 9. Adaptive Rank Selection Extension
- **Motivation:** Eliminate manual rank search.
- **Research Question:** Can a self-supervised neural network adaptively grow its tensor factorization rank by observing reconstruction loss plateaus?
- **Current Implementation:** Over-parameterized DIP networks with dynamic tensor slicing in `run_figure9_adaptive.py`.
- **A. Decisions Established/Approved:**
  - Rank initialization: `R_START = 4`
  - Maximum rank: `R_MAX = 30`
  - Rank step: `R_STEP = 2`
  - Plateau detection: Relative improvement over `PLATEAU_WINDOW` (200 epochs).
  - EMA: `LOSS_EMA_ALPHA = 0.99` smoothing on L1 loss.
  - Optimizer handling: Adam state (`exp_avg`, `exp_avg_sq`) for newly activated final-layer components is explicitly reset to 0.
- **B. Implementation Details Currently in Code:**
  - Cooldown: `MIN_EPOCHS_BEFORE_GROWTH = 300` (Reduced from 500 to ensure `R_MAX` is theoretically reachable within 5000 epochs).
  - Ground-truth usage: MSE is calculated for monitoring/plotting ONLY. It does not contribute to gradients, rank-growth, or early stopping.
  - Gradient behavior: Final output-layer parameters for inactive channels receive zero gradient. Internal/shared layers still receive gradients from active channels.
  - Stochasticity: Unseeded, matching the baseline protocol.
- **C. Ideas/Questions Still Open:**
  - How does adaptive rank compare to fixed rank in varying noise regimes?
  - Will sparsity-based rank discovery be superior to plateau-based growth?

## 10. Research Integrity Rules
- Clean ground truth must **NOT** be used for training gradients.
- Clean ground truth must **NOT** be used for rank-growth decisions.
- Clean ground truth must **NOT** be used for stopping decisions.
- Ground truth may be used for evaluation and monitoring post-hoc.
- Do not tune methods using test/ground-truth information in a way that creates data leakage.
- Experimental claims must be supported by actual measured results, not hypotheses.

## 11. AI Collaboration Workflow
- **Roles:**
  - ChatGPT / Claude / Gemini / GPT-OSS: Serving as varied reasoning, coding, and review engines depending on task complexity.
  - Antigravity IDE: The agentic orchestration layer managing context, execution, and git operations.
- **Preferred Workflow:** Research Objective → Independent AI Reasoning → Evaluation → Approval → Implementation → Experiments → Analysis.
- **IMPORTANT:** For research tasks, do not prescribe an exact algorithm prematurely. Let the research model independently reason about possible approaches first, then evaluate them.

## 12. Model Selection Guidance
- **Gemini 3.6 / 3.7 / 3.8 Flash (Low/Medium):** Fast, efficient coding, file edits, simple scripting, and terminal execution.
- **Gemini 3.1 Pro (Low):** Broad orchestrator, handling context-heavy reasoning and general task planning.
- **Claude Sonnet 4.6 / Opus 4.6 (Thinking):** Deep research reasoning, critical code reviews, complex architectural decisions, and rigorous mathematical evaluation.
- **GPT-OSS 120B Medium:** Alternative reasoning perspective when facing stubborn bugs or needing diverse design inputs.

## 13. Current Project Status

| Task | Status |
|------|--------|
| Environment setup | COMPLETE |
| Official DeepTensor repository | COMPLETE |
| Baseline reproduction | COMPLETE |
| Baseline verification | COMPLETE |
| GitHub repository setup | COMPLETE |
| Adaptive-rank implementation | COMPLETE |
| **Adaptive-rank smoke test** | **IN PROGRESS** |
| Full adaptive-rank experiment | NOT STARTED |
| Sparsity-based rank discovery | OPTIONAL |
| Missing-data/tensor completion | NOT STARTED |
| Robustness experiments | NOT STARTED |
| Optional Transformer extension | OPTIONAL |
| Final experiments | NOT STARTED |
| Analysis | NOT STARTED |
| Report/viva preparation | NOT STARTED |

## 14. EXACT NEXT STEP
**Current Known State:**
- Branch: `extension/adaptive-rank`
- `run_figure9_adaptive.py` exists, implementation has passed `py_compile`.
- Code review complete; Adam state reset and documentation fixes applied.
- Baseline frozen and verified on `main`.
- Environment verified.

**Immediate Next Action: 10-Epoch Smoke Test**
- We must run a 10-epoch smoke test on `run_figure9_adaptive.py` to verify the execution pipeline.
- **Constraints:**
  - Do NOT modify files.
  - Do NOT run the 5000-epoch experiment.
  - Do NOT commit or push.
- **Verification Checklist for Smoke Test:**
  - [ ] Networks initialize at `R_MAX=30`.
  - [ ] Forward pass, masking, and loss computation execute without CUDA/shape errors.
  - [ ] Backpropagation succeeds.
  - [ ] L1 loss decreases over the 10 epochs.
  - [ ] Rank remains unchanged at `R_START=4` (cooldown prevents growth in 10 epochs).
  - [ ] Research integrity conditions hold (no ground-truth leaks).

## 15. Future Workflow
Instructions for new Antigravity IDE agents:
1. Read this `PROJECT_CONTEXT.md` completely.
2. Inspect Git status (`git status`).
3. Inspect relevant source files.
4. Compare actual repository state against this document.
5. If there is a conflict, trust actual files/Git state over assumptions in the document and report the discrepancy.
6. Do not make research changes without first reasoning about them.
7. Do not overwrite the frozen baseline without explicit approval.
8. Keep experiments reproducible and logged.
9. Commit logical milestones separately.

## 16. Important Files
| File / Directory | Purpose |
|------------------|---------|
| `run_figure9.py` | Verified, frozen baseline script for DeepTensor (Rank 20). |
| `run_figure9_adaptive.py` | New research extension script for adaptive rank selection. |
| `modules/` | Core DeepTensor architecture (DIP networks, losses, utils). |
| `data/icvl.mat` | Primary hyperspectral dataset. |
| `PROJECT_CONTEXT.md` | This master handoff and context document. |
| `.venv/` | Python virtual environment. |

## 17. Known Constraints / Warnings
- **Dependency:** `scikit-image` API change required modifying the `ssim_func` kwargs.
- **GPU:** RTX 4060 has 8GB VRAM. `R_MAX=30` fits comfortably within memory limits (peak VRAM observed ~640MB for Rank 20).
- **Adam Optimizer:** `Adam` momentum requires state resets for newly unmasked network channels to prevent sudden learning rate spikes.
- **Research Integrity:** MSE computation inside the training loop uses `hsten` (clean ground truth). This is explicitly detached via `.item()` and used ONLY for tracking/monitoring, not for training logic.

## 18. Decision Log
1. **[Verified]** Use RTX 4060 laptop environment with PyTorch 2.5.1+cu121.
2. **[Verified]** Apply `scikit-image` compatibility fix to baseline SSIM evaluation.
3. **[Verified]** Freeze verified baseline to GitHub (`main` branch, commit `a146cc7`).
4. **[Verified]** Develop extensions in separate scripts (e.g., `run_figure9_adaptive.py`) instead of modifying the baseline script.
5. **[Verified]** Use Effective Rank Masking (over-parameterization) to allow dynamic rank growth without breaking optimizer state.
6. **[Verified]** Use L1 loss EMA plateau detection for triggering rank increases.
7. **[Verified]** Reset Adam state specifically for newly unmasked final output layers upon rank growth.
8. **[Verified]** Reduce cooldown from 500 to 300 epochs to make `R_MAX=30` reachable within 5000 epochs.

## 19. Open Research Questions
- Does plateau-based adaptive rank achieve equal or better PSNR/SSIM compared to manual fixed-rank tuning?
- How much runtime overhead does the over-parameterized `R_MAX=30` forward pass introduce compared to a strict `R=20` baseline?
- Is plateau-detection robust across varying noise levels (e.g., different `noise_snr` values)?
- Can we introduce a sparsity penalty (e.g., L1 on singular values or factor channels) to prune the rank dynamically instead of just growing it?

## 20. Instructions to Future AI Agents
- **Context is King:** This is a research project. Experimental evidence supersedes hypotheses.
- **Reproducibility:** The baseline (`run_figure9.py`) must remain untouched and reproducible.
- **No Premature Claims:** Do not claim an algorithm is "better" until experimental metrics (PSNR/SSIM) confirm it.
- **Data Leakage:** Vigorously protect against data leakage. Ground truth is for evaluation only.
- **Transparency:** Clearly distinguish between implementation facts (what the code currently does) and hypotheses (what we think it will do).
- **Approval:** Ask and seek user approval before making major changes to the research direction or executing long-running experiments (like 5000-epoch runs).
- **Modification Protocol:** Modify specific extension scripts. Leave core `modules/` untouched unless architecturally necessary and approved.
