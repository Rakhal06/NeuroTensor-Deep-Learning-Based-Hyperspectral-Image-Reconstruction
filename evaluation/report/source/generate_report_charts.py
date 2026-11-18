"""
Generate Publication-Quality Academic Figures and Diagrams for DeepTensor++ Research Report
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, Arrow, FancyArrowPatch

OUT_DIR = "evaluation/report/source/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Set global style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Color palette (Academic Navy & Modern Accents)
C_NAVY = "#0F2537"
C_BLUE = "#1E3A8A"
C_STEEL = "#2563EB"
C_LIGHT_BLUE = "#DBEAFE"
C_AMBER = "#D97706"
C_GREEN = "#15803D"
C_GRAY = "#64748B"
C_LIGHT_GRAY = "#F1F5F9"
C_BORDER = "#CBD5E1"
C_WHITE = "#FFFFFF"

# ==============================================================================
# DIAGRAM 1: Forward and Inverse Problem Formulation
# ==============================================================================
def make_inverse_problem_diagram():
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Ground Truth Tensor X
    box_x = FancyBboxPatch((0.4, 0.8), 2.2, 2.4, boxstyle="round,pad=0.1",
                           fc=C_LIGHT_BLUE, ec=C_BLUE, lw=1.5)
    ax.add_patch(box_x)
    ax.text(1.5, 2.5, "Unknown Ground Truth\nTensor $\\mathcal{X}$", ha='center', va='center',
            weight='bold', color=C_NAVY, fontsize=9.5)
    ax.text(1.5, 1.4, "Hyperspectral (H×W×C)\nDynamic Video (H×W×T)\n3D CT/PET (H×W×Z)\nDimension: $\\mathbb{R}^{I \\times J \\times K}$",
            ha='center', va='center', color="#334155", fontsize=8)

    # Forward Arrow
    arr1 = FancyArrowPatch((2.8, 2.0), (4.4, 2.0), arrowstyle='-|>',
                           mutation_scale=18, color=C_BLUE, lw=2)
    ax.add_patch(arr1)
    ax.text(3.6, 2.3, "Forward Physics $\\mathcal{A}(\\cdot)$", ha='center', va='bottom',
            weight='bold', color=C_BLUE, fontsize=8.5)
    ax.text(3.6, 1.6, "+ Noise $\\mathbf{n}$", ha='center', va='top',
            color="#DC2626", fontsize=8)

    # Observed Measurement y
    box_y = FancyBboxPatch((4.6, 0.8), 2.2, 2.4, boxstyle="round,pad=0.1",
                           fc="#FEF3C7", ec=C_AMBER, lw=1.5)
    ax.add_patch(box_y)
    ax.text(5.7, 2.5, "Corrupted Observation\n$\\mathbf{y} = \\mathcal{A}(\\mathcal{X}) + \\mathbf{n}$",
            ha='center', va='center', weight='bold', color="#92400E", fontsize=9.5)
    ax.text(5.7, 1.4, "• Radon Projections (40 angles)\n• CACTI Coded Mux (8× CS)\n• Impulse S&P Noise (30%)\n• Poisson + Readout Noise",
            ha='center', va='center', color="#78350F", fontsize=8)

    # Inverse Arrow
    arr2 = FancyArrowPatch((7.0, 2.0), (8.6, 2.0), arrowstyle='-|>',
                           mutation_scale=18, color=C_GREEN, lw=2)
    ax.add_patch(arr2)
    ax.text(7.8, 2.3, "DeepTensor++\nInverse Prior", ha='center', va='bottom',
            weight='bold', color=C_GREEN, fontsize=8.5)
    ax.text(7.8, 1.6, "Self-Supervised", ha='center', va='top',
            color=C_GREEN, fontsize=8)

    # Reconstructed Tensor
    box_xhat = FancyBboxPatch((8.8, 0.8), 2.2, 2.4, boxstyle="round,pad=0.1",
                              fc="#DCFCE7", ec=C_GREEN, lw=1.5)
    ax.add_patch(box_xhat)
    ax.text(9.9, 2.5, "Reconstructed\nTensor $\\hat{\\mathcal{X}}$", ha='center', va='center',
            weight='bold', color="#166534", fontsize=9.5)
    ax.text(9.9, 1.4, "$\\hat{\\mathcal{X}} = \\sum_{r=1}^R \\lambda_r u_r \\otimes v_r \\otimes w_r$\nNeural Generators\n$f_{\\theta_U}, f_{\\theta_V}, f_{\\theta_W}$",
            ha='center', va='center', color="#14532D", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_inverse_problem.png"), bbox_inches='tight')
    plt.close()
    print("Generated fig_inverse_problem.png")

# ==============================================================================
# DIAGRAM 2: Tensor CP Contraction
# ==============================================================================
def make_tensor_cp_diagram():
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Title label
    ax.text(5.0, 3.6, "CANDECOMP/PARAFAC (CP) Tensor Factorization: $\\hat{\\mathcal{X}} = \\sum_{r=1}^R \\lambda_r (u_r \\circ v_r \\circ w_r)$",
            ha='center', va='center', weight='bold', color=C_NAVY, fontsize=10)

    # Component vectors for rank r
    # u_r vector
    rect_u = Rectangle((0.8, 0.8), 0.5, 2.2, fc="#E0E7FF", ec=C_BLUE, lw=1.2)
    ax.add_patch(rect_u)
    ax.text(1.05, 1.9, "$\\mathbf{u}_r$\nMode-1\n$(I \\times 1)$", ha='center', va='center', fontsize=8, color=C_NAVY)

    ax.text(1.6, 1.9, "$\\circ$", ha='center', va='center', fontsize=16, color=C_GRAY)

    # v_r vector
    rect_v = Rectangle((2.0, 1.6), 1.8, 0.5, fc="#FEF3C7", ec=C_AMBER, lw=1.2)
    ax.add_patch(rect_v)
    ax.text(2.9, 1.85, "$\\mathbf{v}_r$ Mode-2 $(1 \\times J)$", ha='center', va='center', fontsize=8, color="#92400E")

    ax.text(4.1, 1.9, "$\\circ$", ha='center', va='center', fontsize=16, color=C_GRAY)

    # w_r vector
    rect_w = Rectangle((4.5, 1.0), 0.6, 1.8, fc="#DCFCE7", ec=C_GREEN, lw=1.2)
    ax.add_patch(rect_w)
    ax.text(4.8, 1.9, "$\\mathbf{w}_r$\nMode-3\n$(K \\times 1)$", ha='center', va='center', fontsize=8, color="#166534")

    # Equals
    ax.text(5.5, 1.9, "$=$", ha='center', va='center', fontsize=18, color=C_NAVY)

    # Rank-1 outer product box
    rect_cube = FancyBboxPatch((5.9, 0.8), 1.8, 2.2, boxstyle="round,pad=0.05",
                               fc="#F1F5F9", ec=C_GRAY, lw=1.2)
    ax.add_patch(rect_cube)
    ax.text(6.8, 1.9, "Rank-1 Component\n$\\mathcal{X}_r = \\mathbf{u}_r \\circ \\mathbf{v}_r \\circ \\mathbf{w}_r$\nShape: $(I \\times J \\times K)$",
            ha='center', va='center', fontsize=8, color="#334155")

    # Summation arrow
    arr = FancyArrowPatch((7.9, 1.9), (8.6, 1.9), arrowstyle='-|>',
                          mutation_scale=15, color=C_BLUE, lw=1.8)
    ax.add_patch(arr)
    ax.text(8.25, 2.2, "$\\sum_{r=1}^R \\lambda_r$", ha='center', va='bottom',
            weight='bold', color=C_BLUE, fontsize=9)

    # Full Tensor
    rect_full = FancyBboxPatch((8.8, 0.7), 2.2, 2.4, boxstyle="round,pad=0.08",
                               fc=C_LIGHT_BLUE, ec=C_BLUE, lw=1.8)
    ax.add_patch(rect_full)
    ax.text(9.9, 2.4, "Full Reconstructed\nTensor $\\hat{\\mathcal{X}}$", ha='center', va='center',
            weight='bold', color=C_NAVY, fontsize=9)
    ax.text(9.9, 1.3, "Storage: $O((I+J+K)R)$\nvs $O(I \\cdot J \\cdot K)$\nLinear in Rank $R$",
            ha='center', va='center', color="#334155", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_tensor_cp_contraction.png"), bbox_inches='tight')
    plt.close()
    print("Generated fig_tensor_cp_contraction.png")

# ==============================================================================
# DIAGRAM 3: Dimensionality Splitting Taxonomy
# ==============================================================================
def make_splitting_strategies_diagram():
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Card 1: Monolithic 3D
    c1 = FancyBboxPatch((0.2, 0.4), 2.9, 3.2, boxstyle="round,pad=0.1",
                        fc="#F8FAFC", ec=C_BORDER, lw=1.2)
    ax.add_patch(c1)
    ax.text(1.65, 3.2, "Approach 1: Monolithic 3D", ha='center', va='center',
            weight='bold', color=C_NAVY, fontsize=9)
    ax.text(1.65, 2.8, "One 3D Generator (UNetND)", ha='center', va='center',
            color=C_BLUE, fontsize=8)
    ax.text(1.65, 1.6, "• Latent: $\\mathbf{z} \\in \\mathbb{R}^{C \\times H \\times W \\times D}$\n• Direct 3D convs throughout\n• Memory: $O(I \\cdot J \\cdot K)$ (prohibitive)\n• Parameters: Very high\n• Status: Implemented in codebase\n  (`UNetND`), but omitted from\n  main benchmarks due to VRAM.",
            ha='center', va='center', color="#334155", fontsize=7.5)

    # Card 2: 2D Spatial + 1D Spectral
    c2 = FancyBboxPatch((3.5, 0.4), 3.0, 3.2, boxstyle="round,pad=0.1",
                        fc="#EFF6FF", ec=C_BLUE, lw=1.5)
    ax.add_patch(c2)
    ax.text(5.0, 3.2, "Approach 2: Spatial + Spectral", ha='center', va='center',
            weight='bold', color=C_BLUE, fontsize=9)
    ax.text(5.0, 2.8, "One 2D Net + One 1D Net", ha='center', va='center',
            color=C_NAVY, fontsize=8)
    ax.text(5.0, 1.6, "• Spatial Net: $U \\in \\mathbb{R}^{HW \\times R}$\n• Spectral Net: $V \\in \\mathbb{R}^{R \\times C}$\n• Contraction: $\\hat{X} = \\mathbf{U} \\mathbf{V}$ (bmm)\n• Exploits 2D spatial texture\n• Evaluated in:\n  - Figure 09 (ICVL HSI)\n  - Figure 11 (Video CACTI)\n• Parameters: ~3.0M (balanced)",
            ha='center', va='center', color="#1E293B", fontsize=7.5)

    # Card 3: Three 1D CP Generators
    c3 = FancyBboxPatch((6.9, 0.4), 2.9, 3.2, boxstyle="round,pad=0.1",
                        fc="#ECFDF5", ec=C_GREEN, lw=1.5)
    ax.add_patch(c3)
    ax.text(8.35, 3.2, "Approach 3: Three 1D CP", ha='center', va='center',
            weight='bold', color=C_GREEN, fontsize=9)
    ax.text(8.35, 2.8, "DeepTensorCP (u, v, w nets)", ha='center', va='center',
            color="#065F46", fontsize=8)
    ax.text(8.35, 1.6, "• Nets: $u(I \\times R), v(J \\times R), w(K \\times R)$\n• Contraction: `torch.einsum`\n• Core scale vector: $\\lambda \\in \\mathbb{R}^R$\n• Extreme parameter efficiency\n• Evaluated in:\n  - Figure 10 (Yale Face B)\n  - Figure 12 (PET/CT Radon)\n• Scales to Rank $R=2000$",
            ha='center', va='center', color="#064E3B", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_splitting_strategies.png"), bbox_inches='tight')
    plt.close()
    print("Generated fig_splitting_strategies.png")

# ==============================================================================
# DIAGRAM 4: Five Comparative Model Architectures
# ==============================================================================
def make_five_architectures_diagram():
    fig, axes = plt.subplots(1, 5, figsize=(10, 3.5), dpi=300)

    models = [
        ("1. Baseline CNN", "Pure Conv1D\nSkip U-Net\n(skip1d)\n\nLocality Bias\nTranslation\nEquivariance", C_BLUE, "#DBEAFE"),
        ("2. Hybrid Attn", "Conv1D U-Net\n+ Multi-Head\nSelf-Attention\n+ FFN Bottleneck\n\nGlobal Context\nCross-token mixing", "#7C3AED", "#EDE9FE"),
        ("3. CNN Control", "Conv1D U-Net\n+ Residual Conv\nBlock\n(Matches params\nwithin ±0.05%)\n\nTests Depth vs Attn", "#0284C7", "#E0F2FE"),
        ("4. Pointwise Ctrl", "Conv1D U-Net\n+ 1x1 Linear MLP\n(Matches params\nwithin ±0.05%)\n\nToken-Isolated\nChannel Mixing", C_AMBER, "#FEF3C7"),
        ("5. Pure Transformer", "4-Layer Pre-LN\nTransformer\n+ ALiBi Pos Bias\n+ Linear Embed\n\nPure Attention\nNo Conv Inductive\nSmoothness", C_GREEN, "#DCFCE7"),
    ]

    for ax, (title, desc, color, bg) in zip(axes, models):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.05",
                             fc=bg, ec=color, lw=1.5)
        ax.add_patch(box)
        ax.text(0.5, 0.85, title, ha='center', va='center', weight='bold', color=color, fontsize=8.5)
        ax.text(0.5, 0.45, desc, ha='center', va='center', color="#334155", fontsize=7.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_five_architectures.png"), bbox_inches='tight')
    plt.close()
    print("Generated fig_five_architectures.png")

# ==============================================================================
# DIAGRAM 5: Reconstruction and Training Workflow
# ==============================================================================
def make_training_workflow_diagram():
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis('off')

    stages = [
        (0.4, "1. Latents $\\mathbf{z}$", "Uniform noise / 10\nFixed leaf tensor", C_BLUE, "#DBEAFE"),
        (2.3, "2. Generators", "Spatial $f_{\\theta_U}$, Spec $f_{\\theta_V}$\nor 1D CP $u, v, w$", "#7C3AED", "#EDE9FE"),
        (4.2, "3. Factors", "Matrices $U, V$\nor vectors $u, v, w$", "#0284C7", "#E0F2FE"),
        (6.1, "4. Contraction", "$\\hat{X} = UV$ or\n$\\sum \\lambda_r u_r v_r w_r$", C_NAVY, "#F1F5F9"),
        (8.0, "5. Forward $\\mathcal{A}$", "Radon, CACTI,\nor Identity", C_AMBER, "#FEF3C7"),
        (9.8, "6. Loss $\\mathcal{L}(y, \\hat{y})$", "$\\|\\mathcal{A}(\\hat{X}) - y\\|_p$\nL1 or L2", "#DC2626", "#FEE2E2")
    ]

    for x, title, sub, color, bg in stages:
        box = FancyBboxPatch((x, 1.4), 1.5, 1.8, boxstyle="round,pad=0.05",
                             fc=bg, ec=color, lw=1.2)
        ax.add_patch(box)
        ax.text(x + 0.75, 2.7, title, ha='center', va='center', weight='bold', color=color, fontsize=8)
        ax.text(x + 0.75, 1.9, sub, ha='center', va='center', color="#334155", fontsize=6.8)

    # Arrows forward
    for x in [1.9, 3.8, 5.7, 7.6, 9.5]:
        arr = FancyArrowPatch((x, 2.3), (x + 0.4, 2.3), arrowstyle='-|>',
                              mutation_scale=12, color=C_BLUE, lw=1.5)
        ax.add_patch(arr)

    # Backpropagation Arrow
    back_arr = FancyArrowPatch((10.5, 1.2), (3.0, 1.2), arrowstyle='-|>',
                               mutation_scale=15, color="#DC2626", lw=2,
                               connectionstyle="arc3,rad=-0.15")
    ax.add_patch(back_arr)
    ax.text(6.8, 0.4, "PyTorch Autograd Backpropagation: $\\nabla_\\theta \\mathcal{L} \\rightarrow$ Adam Optimizer Update",
            ha='center', va='center', weight='bold', color="#DC2626", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_training_workflow.png"), bbox_inches='tight')
    plt.close()
    print("Generated fig_training_workflow.png")

# ==============================================================================
# DIAGRAM 6: Cross-Application Master Results Grouped Bar Chart
# ==============================================================================
def make_cross_application_bar_chart():
    fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=300)

    apps = [
        "Fig 09: ICVL HSI\n(174x164x260, R=20)",
        "Fig 10: Yale Face B\n(192x168x160, R=2000)",
        "Fig 11: Video CS\n(128x128x64, R=32)",
        "Fig 12: PET/CT\n(128x128x56, R=1000)"
    ]

    baseline = [46.0653, 27.7808, 25.1577, 30.0023]
    hybrid = [44.8526, 28.0392, 24.7440, 30.2947]
    cnn_ctrl = [45.5844, 28.3521, 25.0395, 30.3128]
    pt_ctrl = [45.4229, 28.1703, 24.8147, 30.6212]
    transformer = [43.4026, 28.5552, 24.0502, 30.4056]

    x = np.arange(len(apps))
    w = 0.15

    r1 = ax.bar(x - 2*w, baseline, w, label='Baseline CNN', color='#1E3A8A', edgecolor='none')
    r2 = ax.bar(x - w, hybrid, w, label='Hybrid CNN-Attn', color='#7C3AED', edgecolor='none')
    r3 = ax.bar(x, cnn_ctrl, w, label='CNN Control', color='#0284C7', edgecolor='none')
    r4 = ax.bar(x + w, pt_ctrl, w, label='Pointwise Control', color='#D97706', edgecolor='none')
    r5 = ax.bar(x + 2*w, transformer, w, label='Pure Transformer', color='#15803D', edgecolor='none')

    ax.set_ylabel('Reconstruction PSNR (dB)', fontweight='bold', color=C_NAVY)
    ax.set_title('Cross-Application Comparative Architecture Benchmark (Ground-Truth PSNR)',
                 fontweight='bold', color=C_NAVY, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(apps, fontweight='bold', color=C_NAVY)
    ax.legend(loc='upper right', frameon=True, facecolor='#F8FAFC', edgecolor=C_BORDER)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Highlight winners with small star/trophy marker
    winners = [(0, 46.0653, -2*w), (1, 28.5552, 2*w), (2, 25.1577, -2*w), (3, 30.6212, w)]
    for idx, psnr, offset in winners:
        ax.plot(idx + offset, psnr + 0.8, marker='*', color='#EAB308', markersize=10)
        ax.text(idx + offset, psnr + 1.6, f"{psnr:.2f}", ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=C_NAVY)

    ax.set_ylim(20, 52)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_cross_app_matrix.png"), bbox_inches='tight')
    plt.close()
    print("Generated fig_cross_app_matrix.png")

# ==============================================================================
# DIAGRAMS 7-10: Application Specific Bar Charts
# ==============================================================================
def make_app_bar_chart(title, models_dict, filename, winner_name):
    fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=300)
    names = list(models_dict.keys())
    psnrs = list(models_dict.values())

    colors = ['#15803D' if n == winner_name else '#1E3A8A' for n in names]

    bars = ax.barh(names, psnrs, color=colors, height=0.55, edgecolor=C_BORDER)
    ax.set_xlabel('Reconstruction PSNR (dB)', fontweight='bold', color=C_NAVY)
    ax.set_title(title, fontweight='bold', color=C_NAVY, pad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Annotate values
    min_val = min(psnrs) - 1.5
    max_val = max(psnrs) + 2.0
    ax.set_xlim(min_val, max_val)

    for bar, val in zip(bars, psnrs):
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(w + 0.2, y, f"{val:.4f} dB", va='center', fontsize=8, fontweight='bold',
                color='#15803D' if bar.get_facecolor()[:3] == (0.082, 0.502, 0.239) else C_NAVY)

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), bbox_inches='tight')
    plt.close()
    print(f"Generated {filename}")

def generate_all_app_bar_charts():
    # Fig 09
    make_app_bar_chart(
        "Figure 09: ICVL Hyperspectral Denoising (R=20)",
        {"Baseline CNN": 46.0653, "CNN Control": 45.5844, "Pointwise Control": 45.4229,
         "Hybrid CNN-Attn": 44.8526, "Pure Transformer": 43.4026},
        "fig_icvl_bar_chart.png", "Baseline CNN"
    )

    # Fig 10
    make_app_bar_chart(
        "Figure 10: Yale Face B Photometric Denoising (R=2000)",
        {"Pure Transformer": 28.5552, "CNN Control": 28.3521, "Pointwise Control": 28.1703,
         "Hybrid CNN-Attn": 28.0392, "Baseline CNN": 27.7808},
        "fig_yale_bar_chart.png", "Pure Transformer"
    )

    # Fig 11
    make_app_bar_chart(
        "Figure 11: Video Compressive Sensing CACTI (R=32)",
        {"Baseline CNN": 25.1577, "CNN Control": 25.0395, "Pointwise Control": 24.8147,
         "Hybrid CNN-Attn": 24.7440, "Pure Transformer": 24.0502},
        "fig_video_bar_chart.png", "Baseline CNN"
    )

    # Fig 12
    make_app_bar_chart(
        "Figure 12: PET/CT Radon Compressive Tomography (R=1000)",
        {"Pointwise Control": 30.6212, "Pure Transformer": 30.4056, "CNN Control": 30.3128,
         "Hybrid CNN-Attn": 30.2947, "Baseline CNN": 30.0023},
        "fig_pet_bar_chart.png", "Pointwise Control"
    )

# ==============================================================================
# DIAGRAM 11: Sparsity Trade-off
# ==============================================================================
def make_sparsity_tradeoff_chart():
    fig, ax1 = plt.subplots(figsize=(6.5, 3.2), dpi=300)

    lambdas = ["0.0 (Baseline)", "1e-5", "1e-4", "1e-3", "1e-2"]
    psnrs = [38.22, 38.20, 38.15, 36.85, 31.40]
    effective_ranks = [20, 19, 3, 2, 1]

    color1 = '#1E3A8A'
    color2 = '#D97706'

    x = np.arange(len(lambdas))

    ax1.set_xlabel('Group Lasso Regularization Weight $\\lambda$', fontweight='bold', color=C_NAVY)
    ax1.set_ylabel('Validation PSNR (dB, 95% mask)', color=color1, fontweight='bold')
    line1 = ax1.plot(x, psnrs, marker='o', color=color1, lw=2, label='Val PSNR (dB)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lambdas)
    ax1.grid(linestyle='--', alpha=0.4)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Effective Rank Components ($>1\\%$ energy)', color=color2, fontweight='bold')
    line2 = ax2.plot(x, effective_ranks, marker='s', color=color2, lw=2, linestyle='--', label='Effective Rank')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Factor Group-Sparsity: Regularization Weight vs Quality & Effective Rank',
              fontweight='bold', color=C_NAVY, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_sparsity_tradeoff.png"), bbox_inches='tight')
    plt.close()
    print("Generated fig_sparsity_tradeoff.png")

# ==============================================================================
# DIAGRAM 12: Parameter vs Performance
# ==============================================================================
def make_param_efficiency_chart():
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=300)

    # Plotting parameters (in Millions) vs PSNR for Figure 09 and Figure 10
    models = ['Baseline CNN', 'Hybrid Attn', 'CNN Control', 'Pointwise Ctrl', 'Pure Transformer']
    params_fig9 = [3.005, 3.006, 3.006, 3.005, 3.005]
    psnr_fig9 = [46.0653, 44.8526, 45.5844, 45.4229, 43.4026]

    colors = ['#1E3A8A', '#7C3AED', '#0284C7', '#D97706', '#15803D']

    for name, p, psnr, c in zip(models, params_fig9, psnr_fig9, colors):
        ax.scatter(p, psnr, color=c, s=120, zorder=5, label=name)
        ax.annotate(name, (p, psnr), textcoords="offset points", xytext=(8, -2), fontsize=7.5)

    ax.set_xlabel('Parameter Count (Millions)', fontweight='bold', color=C_NAVY)
    ax.set_ylabel('ICVL Denoising PSNR (dB)', fontweight='bold', color=C_NAVY)
    ax.set_title('Strict Parameter Matching vs ICVL Denoising Performance (Capacity Invariant)',
                 fontweight='bold', color=C_NAVY, pad=10)
    ax.grid(linestyle='--', alpha=0.5)
    ax.set_xlim(2.99, 3.04)
    ax.set_ylim(42.5, 47.0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_parameter_efficiency.png"), bbox_inches='tight')
    plt.close()
    print("Generated fig_parameter_efficiency.png")

if __name__ == "__main__":
    make_inverse_problem_diagram()
    make_tensor_cp_diagram()
    make_splitting_strategies_diagram()
    make_five_architectures_diagram()
    make_training_workflow_diagram()
    make_cross_application_bar_chart()
    generate_all_app_bar_charts()
    make_sparsity_tradeoff_chart()
    make_param_efficiency_chart()
    print("All report charts and diagrams generated successfully!")

