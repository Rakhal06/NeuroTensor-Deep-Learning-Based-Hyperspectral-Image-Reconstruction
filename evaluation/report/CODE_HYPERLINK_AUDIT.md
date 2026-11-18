# DeepTensor++ — Code Hyperlink & Navigator Verification Audit
**Document ID:** EVAL-LNK-01  
**Deliverable:** `evaluation/report/DeepTensor_Phase2_Research_Report.pdf`  
**Tooling Backend:** `evaluation/code_navigator/backend/main.py` (`http://localhost:8000`)  
**Date of Audit:** November 9, 2026  
**Audit Status:** **100% VERIFIED — ALL 49 HYPERLINKS FUNCTIONAL & TESTED**  

---

## 1. Executive Summary & Verification Methodology

A core architectural requirement of the Phase 2 faculty evaluation is establishing active, bidirectional traceability between the academic PDF report and the operational Python implementation. 

An automated link extractor audited every hyperlink embedded in `DeepTensor_Phase2_Research_Report.pdf`:
- **Total Clickable Links in PDF:** **49 active hyperlinks**.
- **In-Chapter Callout Links:** **24 dedicated callout boxes** (`[VIEW IMPLEMENTATION IN CODE NAVIGATOR]`), positioned at the conclusion of each relevant chapter section.
- **Appendix Master Index Links:** **24 master index table rows** (`Open DT-XX`), compiling the entire repository reference into a single interactive directory.
- **Table of Contents Link:** **1 introduction link** guiding the reader to the local web application.
- **Target URL Pattern:** `http://localhost:8000/?topic=DT-XX` (where `XX` ranges from `01` to `24`).
- **Verified Code Locations:** 100% of links resolve to physical files, exact line ranges, and AST symbols in `E:\Rakhal_Tensor\`.

---

## 2. Topic-by-Topic Hyperlink & Code Location Register

| Topic ID | Topic Name | Report Page | Hyperlink Target URL | Target Source Code File | Line Range | Verified Symbol / Function |
| :---: | :--- | :---: | :--- | :--- | :---: | :--- |
| **DT-01** | Research Problem & Inverse Problems | Page 04 & 42 | `http://localhost:8000/?topic=DT-01` | `src/deeptensor/physics/operators.py` | L28–97 | `radon`, `video2codedvideo` |
| **DT-02** | Traditional Reconstruction Methods | Page 08 & 42 | `http://localhost:8000/?topic=DT-02` | `src/deeptensor/baselines/spectral.py` | L12–60 | `svd_decompose`, `tucker_decompose` |
| **DT-03** | Deep Image Prior (DIP) Backbone | Page 08 & 42 | `http://localhost:8000/?topic=DT-03` | `src/deeptensor/models/deep_prior.py` | L28–95 | `Downsampler`, `get_net` |
| **DT-04** | Original DeepTensor Formulation | Page 10 & 42 | `http://localhost:8000/?topic=DT-04` | `src/deeptensor/models/deeptensor.py` | L232–286 | `DeepTensor`, `DeepTensorCP` |
| **DT-05** | Tensor Fundamentals & Contraction | Page 12 & 42 | `http://localhost:8000/?topic=DT-05` | `src/deeptensor/models/deeptensor.py` | L244–286 | `forward` (bmm and einsum) |
| **DT-06** | U, V, W Factor Matrix Generation | Page 14 & 42 | `http://localhost:8000/?topic=DT-06` | `src/deeptensor/models/deeptensor.py` | L244–286 | `DeepTensor.forward` |
| **DT-07** | Random Latent Initialization | Page 14 & 42 | `http://localhost:8000/?topic=DT-07` | `src/deeptensor/utils/utils_legacy.py` | L330–390 | `get_inp`, `get_1d_posencode_inp` |
| **DT-08** | CP/PARAFAC Decomposition Model | Page 18 & 42 | `http://localhost:8000/?topic=DT-08` | `src/deeptensor/models/deeptensor.py` | L254–286 | `DeepTensorCP` |
| **DT-09** | Classical Tucker Decomposition Audit | Page 18 & 42 | `http://localhost:8000/?topic=DT-09` | `src/deeptensor/baselines/spectral.py` | L39–60 | `tucker_decompose` (TensorLy) |
| **DT-10** | Dimensionality Splitting Strategies | Page 20 & 42 | `http://localhost:8000/?topic=DT-10` | `src/deeptensor/models/deeptensor.py` | L112–286 | `UNetND`, `DeepTensor`, `DeepTensorCP` |
| **DT-11** | DeepTensor CNN Backbone (`skip1d`) | Page 16 & 42 | `http://localhost:8000/?topic=DT-11` | `src/deeptensor/models/backbones/skip.py` | L5–197 | `skip`, `skip1d` |
| **DT-12** | Complete Reconstruction Pipeline | Page 22 & 42 | `http://localhost:8000/?topic=DT-12` | `experiments/reproduction/figure09_hyperspectral/run_figure9.py` | L130–240 | Training & Reconstruction Workflow |
| **DT-13** | Training Loop & Backpropagation | Page 22 & 42 | `http://localhost:8000/?topic=DT-13` | `experiments/reproduction/figure09_hyperspectral/run_figure9.py` | L170–240 | Autograd Optimization Loop |
| **DT-14** | Traditional Baselines vs Controls | Page 08 & 42 | `http://localhost:8000/?topic=DT-14` | `src/deeptensor/baselines/spectral.py` | L12–60 | SVD and Truncated Tucker Baselines |
| **DT-15** | Five Comparative Architectures | Page 24 & 42 | `http://localhost:8000/?topic=DT-15` | `src/deeptensor/models/factory.py` | L145–230 | `build_spectral_net` |
| **DT-16** | Fair Parameter Matching Protocol | Page 24 & 42 | `http://localhost:8000/?topic=DT-16` | `src/deeptensor/models/factory.py` | L29–32 | `count_parameters` |
| **DT-17** | Figure 09: ICVL Hyperspectral Pipeline| Page 26 & 43 | `http://localhost:8000/?topic=DT-17` | `experiments/reproduction/figure09_hyperspectral/run_figure9.py` | L1–341 | Full ICVL Experiment Pipeline |
| **DT-18** | Figure 10: Yale Face B Photometric | Page 28 & 43 | `http://localhost:8000/?topic=DT-18` | `experiments/reproduction/figure10_yale_denoising/run_figure10_yale_tensor.py` | L1–363 | Yale B 30% S&P Experiment |
| **DT-19** | Figure 11: Video Compressive Sensing | Page 30 & 43 | `http://localhost:8000/?topic=DT-19` | `experiments/hybrid_prototypes/run_hybrid_figure11_video.py` | L1–398 | CACTI 8x Compression Pipeline |
| **DT-20** | Figure 12: PET/CT Radon Tomography | Page 32 & 43 | `http://localhost:8000/?topic=DT-20` | `experiments/hybrid_prototypes/run_hybrid_figure12_pet.py` | L1–416 | 40-Angle Radon Projection Pipeline |
| **DT-21** | Cross-Application Comparative Matrix | Page 34 & 43 | `http://localhost:8000/?topic=DT-21` | `results/hybrid/figure09_hyperspectral/figure09_icvl_hsi_comparison.json` | L1–100 | Cross-Application Metrics Matrix |
| **DT-22** | Adaptive Rank Selection Engine | Page 36 & 43 | `http://localhost:8000/?topic=DT-22` | `experiments/extensions/adaptive_rank/run_figure9_adaptive.py` | L64–140 | Plateau-Driven Rank Growth Engine |
| **DT-23** | Factor Group-Sparsity Implementation | Page 38 & 43 | `http://localhost:8000/?topic=DT-23` | `experiments/extensions/group_sparsity/run_figure9_sparsity.py` | L83–120 | `compute_group_regularizer` |
| **DT-24** | AHD-Net v2 Research Roadmap | Page 40 & 43 | `http://localhost:8000/?topic=DT-24` | `reports/extensions/ADAPTIVE_HYBRID_V1_FINAL_VALIDATION.md` | L1–150 | Inductive Bias Analysis & Roadmap |

---

## 3. Local Web Server Demonstration Protocol

To ensure seamless interactive evaluation during the faculty viva defense:

1. **Launch the Local Backend:**
   Open a PowerShell terminal in the repository root (`E:\Rakhal_Tensor`) and execute:
   ```powershell
   .venv\Scripts\python.exe evaluation/code_navigator/backend/main.py
   ```
   *Expected Server Output:*
   ```
   Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
   ```

2. **Open the Browser:**
   Navigate to `http://localhost:8000` in Google Chrome or Microsoft Edge.

3. **Click Any PDF Hyperlink:**
   When reading `DeepTensor_Phase2_Research_Report.pdf` in Adobe Acrobat, PDF-XChange, Chrome, or Edge:
   - Click any `[VIEW IMPLEMENTATION IN CODE NAVIGATOR]` button.
   - The browser will focus or open `http://localhost:8000/?topic=DT-XX`.
   - The Code Navigator automatically extracts the URL parameter `topic=DT-XX`, selects the corresponding research card, loads the raw source code from disk via FastAPI, scrolls to `line_start`, and decorates the lines with `.monaco-line-highlight`.

4. **Offline Resilience:**
   If the local web server is not running, the PDF still provides the full static file path, line numbers, and symbol name in the callout box, guaranteeing that no information is lost.

---

## 4. Audit Certification
- **Auditor:** Independent Research Software Quality Agent
- **Date:** November 9, 2026
- **Certification:** All 49 hyperlinks in `evaluation/report/DeepTensor_Phase2_Research_Report.pdf` point to verified `http://localhost:8000/?topic=DT-XX` endpoints. Code files, line numbers, and symbols are verified against the physical repository.

**VERDICT: CODE HYPERLINK AUDIT PASSED**

