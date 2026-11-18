# NeuroTensor Project Report
**Course:** Tensor Methods for Multimodal Data Analysis (CS376)  
**Institution:** Department of Computer Science and Engineering, Indian Institute of Information Technology Dharwad (IIIT Dharwad)  
**Faculty Guide:** Dr. Sibasankar Padhy  
**Students:**  
- B. Rakhal Krishna (24BDS009)  
- Dargupally Ashish (24BDS015)  
- Soham Nalage (24BDS059)  

---

## 1. Directory Structure

```
NeuroTensor_Project_Report/
│
├── main.tex                       # Primary LaTeX document driver
├── references.bib                 # Verified academic BibTeX bibliography
├── README.md                      # Compilation and reproduction guide
├── main.pdf                       # Compiled academic research report (PDF)
│
├── chapters/                      # Main report sections matching template
│   ├── 00_frontmatter.tex         # Title page, Certificate, Abstract, TOC, LOF, LOT
│   ├── 01_introduction.tex        # Section 1: Introduction, Problem Statement, Objectives
│   ├── 02_related_work.tex        # Section 2: Related Work & Multilinear Foundations
│   ├── 03_data_methods.tex        # Section 3: Data, DeepTensor Formulation, 5 Models, Extensions 1-9
│   ├── 04_results_discussion.tex  # Section 4: Quantitative Results, 12 Tables, Discussion
│   └── 05_conclusion.tex          # Section 5: Conclusion, Inductive Biases, Future Scope
│
├── appendices/                    # Detailed technical appendices
│   ├── appendix_a_architectures.tex # Detailed Layer-by-Layer Specifications for 5 Models
│   ├── appendix_b_derivations.tex   # Multilinear Contraction & Group Lasso Gradients, ALiBi
│   └── appendix_c_extended_tables.tex # Complete Configurations, Hyperparameters & RNG Seeds
│
├── figures/                       # High-resolution experimental figures and institutional logo
│   ├── iiIT-Dharwad-Logo.png
│   ├── baseline_reconstruction_comparison.png
│   ├── baseline_error_map.png
│   ├── figure9_hyperspectral_denoising_panel.png
│   ├── figure9_convergence_curves.png
│   ├── adaptive_rank_history.png
│   ├── adaptive_rank_summary.png
│   ├── sparsity_lambda_0.0001.png
│   ├── sparsity_rank_results.png
│   ├── extension3_completion_curve.png
│   ├── extension4_noise_robustness_bars.png
│   ├── extension5_ensemble_uncertainty.png
│   ├── progressive_topk_metrics.png
│   ├── spatial_cosine_similarity.png
│   ├── spectral_cosine_similarity.png
│   ├── top5_components.png
│   ├── extension8_9_loss_convergence.png
│   ├── figure09_hsi_5models_panel.png
│   ├── figure10_yale_b_comparison_panel.png
│   ├── figure11_video_cs_comparison_panel.png
│   ├── figure12_pet_ct_comparison_panel.png
│   └── yale_ablation_routing_trajectories.png
│
└── tables/                        # Auxiliary table structures
```

---

## 2. Compilation Instructions

The report requires standard MiKTeX or TeXLive distribution with `pdflatex` and `bibtex`.

To compile the complete report from PowerShell:

```powershell
cd E:\Rakhal_Tensor\NeuroTensor_Project_Report
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

This multi-pass sequence resolves:
1. First pass: generates section references and `.aux` entries.
2. Second pass (`bibtex`): compiles bibliography from `references.bib` into `.bbl`.
3. Third pass: incorporates citation references into text.
4. Fourth pass: finalizes Table of Contents, List of Figures, and List of Tables page numbers.

