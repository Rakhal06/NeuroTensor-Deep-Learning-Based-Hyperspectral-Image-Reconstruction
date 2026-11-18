# DeepTensor++ — Phase 2 Academic Research Report Deliverables

**Primary Evaluation Deliverable:** Academic Research Report in PDF format  
**Target File:** `evaluation/report/DeepTensor_Phase2_Research_Report.pdf` (43 Pages, A4 Portrait)  
**Interactive Code Navigator:** `http://localhost:8000` (FastAPI + Monaco Editor)  
**Evaluation Phase:** Phase 2 Comprehensive Faculty Evaluation & Defense  
**Repository State:** Frozen & Verified (Preservation Mode)  

---

## 1. Overview of Deliverables

In accordance with Phase 2 faculty evaluation guidelines, the primary deliverable for DeepTensor++ is a publication-grade academic research report in PDF format. This report replaces the previous slide-based presentation, presenting a detailed, mathematically rigorous, and empirically grounded defense of the research journey.

### Key Highlights
- **Document Length:** **43 pages** (A4 Portrait, clean academic typography).
- **Structure:** 20 comprehensive chapters, executive front matter, bibliography, and a consolidated code index appendix.
- **Embedded Visual Evidence:** **24 high-resolution figures**, including authentic reconstructed experimental panels, spatial error maps, temporal strips, and publication-quality diagrams. Zero fabricated or synthetic illustrations.
- **Interactive Code Hyperlinks:** **49 active hyperlinks** connecting theoretical explanations and empirical benchmarks directly to verified source code lines in the local **DeepTensor++ Research Navigator** (`http://localhost:8000/?topic=DT-XX`).
- **Preservation of Core Assets:** Zero modifications were made to research source code (`src/`), raw datasets (`data/`), experiment configs (`configs/`), checkpoints, or historical logs (`results/`). The previous PowerPoint presentation has been safely archived in `evaluation/archive/presentation/`.

---

## 2. Directory Structure

```
evaluation/report/
├── DeepTensor_Phase2_Research_Report.pdf   # THE PRIMARY DELIVERABLE (43 Pages, A4)
├── README.md                               # This documentation and execution guide
├── REPORT_CONTENT_AUDIT.md                 # Chapter-by-chapter content, structure, and text audit
├── FIGURE_AND_METRIC_AUDIT.md              # Reconciled metrics and figure resolution audit
├── CODE_HYPERLINK_AUDIT.md                 # 49 hyperlinks verification and Code Navigator mapping
├── rendered_pages/                         # High-resolution (150 DPI) PNG renders of all 43 pages
│   ├── page_01.png                         # Cover Page
│   ├── page_02.png                         # Table of Contents
│   └── ... page_43.png                     # Master Appendix (DT-01 to DT-24)
└── source/                                 # Complete editable source and generation scripts
    ├── build_pdf_report.py                 # Master ReportLab PDF compilation script
    ├── report_content.py                   # Full academic text, equations, tables, and callouts
    ├── report_styles.py                    # Typography styles, NumberedCanvas, and custom flowables
    ├── generate_report_charts.py           # Matplotlib script generating all 12 publication charts
    ├── audit_pdf_quality.py                # PyMuPDF audit and page rasterization script
    └── figures/                            # Publication-quality vector/high-res PNG charts
        ├── fig_inverse_problem.png
        ├── fig_tensor_cp_contraction.png
        ├── fig_splitting_strategies.png
        ├── fig_five_architectures.png
        ├── fig_training_workflow.png
        ├── fig_cross_app_matrix.png
        ├── fig_icvl_bar_chart.png
        ├── fig_yale_bar_chart.png
        ├── fig_video_bar_chart.png
        ├── fig_pet_bar_chart.png
        ├── fig_sparsity_tradeoff.png
        └── fig_parameter_efficiency.png
```

---

## 3. How to Launch and Use the Interactive System

### Step 1: Start the Local Code Navigator Server
The interactive links in the PDF report connect directly to the local FastAPI backend. Open a terminal in the repository root (`E:\Rakhal_Tensor`) and run:
```powershell
.venv\Scripts\python.exe evaluation/code_navigator/backend/main.py
```
*Expected Server Output:*
```
Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 2: Open the PDF Report
Open `evaluation/report/DeepTensor_Phase2_Research_Report.pdf` in Adobe Acrobat Reader, Google Chrome, Microsoft Edge, or any modern PDF viewer.

### Step 3: Interactive Navigation During Faculty Viva
- While reading any chapter, click the prominent blue button:
  `[CLICK TO INSPECT DT-XX IN CODE NAVIGATOR]`
- The PDF will launch your default web browser to:
  `http://localhost:8000/?topic=DT-XX`
- The **DeepTensor++ Research Navigator** will automatically:
  1. Highlight the selected research topic in the left navigation sidebar.
  2. Display the mathematical formulation, research significance, and faculty viva Q&A defense.
  3. Load the exact physical Python file from the repository using FastAPI.
  4. Scroll the embedded **Monaco Editor** directly to the verified start line and highlight the exact code range in blue.

---

## 4. How to Recompile and Re-Audit the Report

All generation scripts are fully automated and reproducible:

```powershell
# 1. Regenerate all publication charts and diagrams (optional):
.venv\Scripts\python.exe evaluation/report/source/generate_report_charts.py

# 2. Compile the master PDF report (takes ~14 seconds):
.venv\Scripts\python.exe evaluation/report/source/build_pdf_report.py

# 3. Rasterize all 43 pages to PNG and run automated quality control:
.venv\Scripts\python.exe evaluation/report/source/audit_pdf_quality.py
```

---

## 5. Master Experimental Results Summary

All values match authoritative JSON files in `results/hybrid/` to 4 decimal places. Evaluated under clean-latent DIP inference (`torch.no_grad()`) on full 3D volumes normalized to $[0, 1]$.

| Application & Experiment | Dimensions & Rank | Baseline CNN | Hybrid Attn | CNN Control | Pointwise Ctrl | Pure Transformer | Winning Model & Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fig 09: ICVL Hyperspectral**<br>Mixed Poisson + Gauss Noise | $174 \times 164 \times 260$<br>$R=20$ | **46.0653 dB** 🏆<br>(3,005,328) | 44.8526 dB<br>(3,005,584) | 45.5844 dB<br>(3,005,584) | 45.4229 dB<br>(3,005,296) | 43.4026 dB<br>(3,004,764) | **Baseline CNN (+0.48 dB)**<br>Smooth continuous spectra |
| **Fig 10: Yale Face B**<br>30% Salt-and-Pepper Noise | $192 \times 168 \times 160$<br>$R=2000$ | 27.7808 dB<br>(5,470,508) | 28.0392 dB<br>(5,603,632) | 28.3521 dB<br>(5,603,632) | 28.1703 dB<br>(5,603,048) | **28.5552 dB** 🏆<br>(4,955,144) | **Pure Transformer (+0.77 dB)**<br>Global photometric lighting |
| **Fig 11: Video CS (CACTI)**<br>8x Temporal Multiplexing | $128 \times 128 \times 64$<br>$R=32$ | **25.1577 dB** 🏆<br>(784,648) | 24.7440 dB<br>(834,632) | 25.0395 dB<br>(834,505) | 24.8147 dB<br>(834,568) | 24.0502 dB<br>(1,376,036) | **Baseline CNN (+0.12 dB)**<br>Continuous velocity motion |
| **Fig 12: PET/CT Radon Tomo**<br>40 Projection Angles | $128 \times 128 \times 56$<br>$R=1000$ | 30.0023 dB<br>(3,918,508) | 30.2947 dB<br>(4,050,988) | 30.3128 dB<br>(4,051,160) | **30.6212 dB** 🏆<br>(4,050,860) | 30.4056 dB<br>(3,663,144) | **Pointwise Control (+0.62 dB)**<br>Decoupled anatomical z-slices |

### Independent Research Extensions
- **Adaptive Rank Selection (DT-22):** **46.1571 dB** on ICVL at epoch 5000 (+0.0918 dB gain over fixed rank-20 Baseline CNN 46.0653 dB) via self-supervised plateau-growth.
- **Factor Group-Sparsity (DT-23):** **38.15 dB** at $\lambda = 10^{-4}$ under a 95% visible / 5% held-out validation mask framework (seed 42), successfully pruning rank down to 3 dominant factors.

---

## 6. Archive Notice
The initial PowerPoint presentation has been archived at:
- `evaluation/archive/presentation/DeepTensor_Phase2_Evaluation.pptx`
- It is superseded by `evaluation/report/DeepTensor_Phase2_Research_Report.pdf` as the primary evaluation document.

