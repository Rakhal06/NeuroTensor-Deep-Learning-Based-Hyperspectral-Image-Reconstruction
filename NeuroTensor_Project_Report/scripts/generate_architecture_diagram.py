import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle
import numpy as np

def create_architecture_diagram():
    fig = plt.figure(figsize=(16, 11), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Background color
    fig.patch.set_facecolor('#ffffff')

    # Color Palette - Professional Academic IEEE Style
    c_blue = '#1f4e79'
    c_light_blue = '#d9e1f2'
    c_teal = '#176b6e'
    c_light_teal = '#d0ece7'
    c_coral = '#c00000'
    c_light_coral = '#f8d7da'
    c_purple = '#4a235a'
    c_light_purple = '#ebdef0'
    c_amber = '#b9770e'
    c_light_amber = '#fef9e7'
    c_gray_box = '#f2f4f4'
    c_border = '#34495e'
    c_dark = '#1a1a1a'

    # =========================================================================
    # PART A: GLOBAL DEEPTENSOR MULTILINEAR CONTRACTION WORKFLOW (Top Section)
    # =========================================================================
    # Main outer bounding box for Part A
    rect_a = FancyBboxPatch((0.4, 6.2), 15.2, 4.4, boxstyle="round,pad=0.2",
                            edgecolor='#7f8c8d', facecolor='#fafbfc', linewidth=1.5, linestyle='--')
    ax.add_patch(rect_a)
    ax.text(0.7, 10.3, "PART A: Global DeepTensor Multilinear Contraction & Optimization Pipeline",
            fontsize=12, fontweight='bold', color='#1a252f')

    # 1. Spatial Latent & Generator
    box_zu = FancyBboxPatch((0.8, 8.4), 2.2, 1.4, boxstyle="round,pad=0.1",
                            edgecolor=c_blue, facecolor=c_light_blue, linewidth=1.2)
    ax.add_patch(box_zu)
    ax.text(1.9, 9.3, "Spatial Noise Code", fontsize=10, fontweight='bold', ha='center', color=c_blue)
    ax.text(1.9, 8.9, r"$\mathbf{z}_U \sim \mathcal{U}(0, 0.1)$", fontsize=9, ha='center', color=c_dark)
    ax.text(1.9, 8.6, r"$[20 \times 174 \times 164]$", fontsize=8.5, ha='center', color='#555')

    box_fu = FancyBboxPatch((3.5, 8.4), 2.6, 1.4, boxstyle="round,pad=0.1",
                            edgecolor=c_blue, facecolor='#ffffff', linewidth=1.5)
    ax.add_patch(box_fu)
    ax.text(4.8, 9.3, "Spatial Generator", fontsize=10, fontweight='bold', ha='center', color=c_blue)
    ax.text(4.8, 8.9, r"$f_U(\mathbf{z}_U; \boldsymbol{\theta}_U)$", fontsize=9.5, fontweight='bold', ha='center', color=c_dark)
    ax.text(4.8, 8.6, "5-Scale 2D Skip/U-Net", fontsize=8.5, ha='center', color='#555')
    ax.text(4.8, 8.35, "2,206,152 Params", fontsize=8, fontweight='bold', ha='center', color='#27ae60')

    # 2. Spectral Latent & Generator
    box_zv = FancyBboxPatch((0.8, 6.6), 2.2, 1.4, boxstyle="round,pad=0.1",
                            edgecolor=c_teal, facecolor=c_light_teal, linewidth=1.2)
    ax.add_patch(box_zv)
    ax.text(1.9, 7.5, "Spectral Noise Code", fontsize=10, fontweight='bold', ha='center', color=c_teal)
    ax.text(1.9, 7.1, r"$\mathbf{z}_V \sim \mathcal{U}(0, 0.1)$", fontsize=9, ha='center', color=c_dark)
    ax.text(1.9, 6.8, r"$[20 \times 260]$", fontsize=8.5, ha='center', color='#555')

    box_fv = FancyBboxPatch((3.5, 6.6), 2.6, 1.4, boxstyle="round,pad=0.1",
                            edgecolor=c_teal, facecolor='#ffffff', linewidth=1.5)
    ax.add_patch(box_fv)
    ax.text(4.8, 7.5, "Spectral Generator", fontsize=10, fontweight='bold', ha='center', color=c_teal)
    ax.text(4.8, 7.1, r"$f_V(\mathbf{z}_V; \boldsymbol{\theta}_V)$", fontsize=9.5, fontweight='bold', ha='center', color=c_dark)
    ax.text(4.8, 6.8, "(One of 5 Investigated)", fontsize=8.5, fontstyle='italic', ha='center', color='#555')
    ax.text(4.8, 6.55, "799k - 932k Params", fontsize=8, fontweight='bold', ha='center', color='#27ae60')

    # 3. Factor Matrices U and V
    box_u = FancyBboxPatch((6.6, 8.4), 2.1, 1.4, boxstyle="round,pad=0.1",
                           edgecolor=c_blue, facecolor=c_light_blue, linewidth=1.2)
    ax.add_patch(box_u)
    ax.text(7.65, 9.3, "Spatial Factors", fontsize=9.5, fontweight='bold', ha='center', color=c_blue)
    ax.text(7.65, 8.9, r"$\mathbf{U} \in \mathbb{R}^{HW \times R}$", fontsize=9.5, ha='center', color=c_dark)
    ax.text(7.65, 8.6, r"$[28,536 \times 20]$", fontsize=8.5, ha='center', color='#555')

    box_v = FancyBboxPatch((6.6, 6.6), 2.1, 1.4, boxstyle="round,pad=0.1",
                           edgecolor=c_teal, facecolor=c_light_teal, linewidth=1.2)
    ax.add_patch(box_v)
    ax.text(7.65, 7.5, "Spectral Factors", fontsize=9.5, fontweight='bold', ha='center', color=c_teal)
    ax.text(7.65, 7.1, r"$\mathbf{V} \in \mathbb{R}^{B \times R}$", fontsize=9.5, ha='center', color=c_dark)
    ax.text(7.65, 6.8, r"$[260 \times 20]$", fontsize=8.5, ha='center', color='#555')

    # 4. Multilinear Product & Output Tensor
    box_prod = FancyBboxPatch((9.2, 7.4), 2.3, 1.7, boxstyle="round,pad=0.1",
                              edgecolor=c_purple, facecolor=c_light_purple, linewidth=1.5)
    ax.add_patch(box_prod)
    ax.text(10.35, 8.7, "Bilinear Contraction", fontsize=9.5, fontweight='bold', ha='center', color=c_purple)
    ax.text(10.35, 8.25, r"$\hat{Y}_{(1)} = \mathbf{U} \mathbf{V}^T$", fontsize=10.5, fontweight='bold', ha='center', color=c_dark)
    ax.text(10.35, 7.85, r"Reshape: $\hat{\mathcal{Y}} \in \mathbb{R}^{H \times W \times B}$", fontsize=8.5, ha='center', color=c_dark)
    ax.text(10.35, 7.55, r"$[174 \times 164 \times 260]$", fontsize=8, ha='center', color='#555')

    # 5. Loss & Optimization
    box_loss = FancyBboxPatch((12.1, 7.4), 3.2, 1.7, boxstyle="round,pad=0.1",
                              edgecolor=c_coral, facecolor=c_light_coral, linewidth=1.5)
    ax.add_patch(box_loss)
    ax.text(13.7, 8.7, "Unsupervised Loss & Training", fontsize=9.5, fontweight='bold', ha='center', color=c_coral)
    ax.text(13.7, 8.25, r"$\min_{\boldsymbol{\theta}} \frac{1}{|\Omega|} \sum |\hat{Y}_{i,j} - Y^\mathrm{noisy}_{i,j}|$",
            fontsize=9.5, fontweight='bold', ha='center', color=c_dark)
    ax.text(13.7, 7.85, "Adam Optimizer: lr=1e-3, 5000 Epochs", fontsize=8, ha='center', color='#333')
    ax.text(13.7, 7.55, r"$\nabla_{\mathbf{U}} \mathcal{L} \propto \mathbf{V}, \quad \nabla_{\mathbf{V}} \mathcal{L} \propto \mathbf{U}$",
            fontsize=8.5, fontweight='bold', ha='center', color='#c0392b')

    # Arrows for Part A
    arrow_kw = dict(arrowstyle="->", lw=1.5, color='#2c3e50')
    ax.annotate("", xy=(3.5, 9.1), xytext=(3.0, 9.1), arrowprops=arrow_kw)
    ax.annotate("", xy=(6.6, 9.1), xytext=(6.1, 9.1), arrowprops=arrow_kw)
    ax.annotate("", xy=(9.2, 8.5), xytext=(8.7, 9.1), arrowprops=arrow_kw)

    ax.annotate("", xy=(3.5, 7.3), xytext=(3.0, 7.3), arrowprops=arrow_kw)
    ax.annotate("", xy=(6.6, 7.3), xytext=(6.1, 7.3), arrowprops=arrow_kw)
    ax.annotate("", xy=(9.2, 7.9), xytext=(8.7, 7.3), arrowprops=arrow_kw)

    ax.annotate("", xy=(12.1, 8.25), xytext=(11.5, 8.25), arrowprops=arrow_kw)

    # Feedback backprop arrow
    ax.annotate("", xy=(4.8, 6.6), xytext=(12.1, 7.6),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=c_coral, linestyle="--",
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(8.5, 6.35, "Backprop Gradient Flow (Self-Supervised Reconstruction)",
            fontsize=8, fontweight='bold', ha='center', color=c_coral)

    # =========================================================================
    # PART B: THE FIVE INVESTIGATED SPECTRAL GENERATOR PARADIGMS (Bottom Section)
    # =========================================================================
    rect_b = FancyBboxPatch((0.4, 0.4), 15.2, 5.5, boxstyle="round,pad=0.2",
                            edgecolor='#7f8c8d', facecolor='#fafbfc', linewidth=1.5, linestyle='--')
    ax.add_patch(rect_b)
    ax.text(0.7, 5.6, "PART B: The Five Investigated Factor Generator Architectures",
            fontsize=12, fontweight='bold', color='#1a252f')

    # Column coordinates for 5 models
    xs = [0.65, 3.7, 6.75, 9.8, 12.85]
    w = 2.85

    # -------------------------------------------------------------
    # 1. Baseline CNN
    # -------------------------------------------------------------
    b1 = FancyBboxPatch((xs[0], 0.65), w, 4.75, boxstyle="round,pad=0.08",
                        edgecolor='#2980b9', facecolor='#ffffff', linewidth=1.4)
    ax.add_patch(b1)
    ax.text(xs[0]+w/2, 5.15, "1. Baseline CNN", fontsize=10, fontweight='bold', ha='center', color='#2980b9')
    ax.text(xs[0]+w/2, 4.9, "1D Skip / U-Net", fontsize=8.5, fontstyle='italic', ha='center', color='#555')

    y_cur = 4.5
    layers1 = [
        ("Input Conv1D (k=3, s=1)", "20 -> 128 channels"),
        ("Encoder Scale 1 (k=3, s=2)", "260 -> 130 tokens"),
        ("Encoder Scale 2 (k=3, s=2)", "130 -> 65 tokens"),
        ("Encoder Scale 3 (k=3, s=2)", "65 -> 33 tokens"),
        ("Encoder Scale 4 (k=3, s=2)", "33 -> 17 tokens"),
        ("Conv Bottleneck (17 tok)", "Direct Residual Skip"),
        ("Decoder Up 4 (Interp+Cat)", "17 -> 33 tokens"),
        ("Decoder Up 3 (Interp+Cat)", "33 -> 65 tokens"),
        ("Decoder Up 2 (Interp+Cat)", "65 -> 130 tokens"),
        ("Decoder Up 1 (Interp+Cat)", "130 -> 260 tokens"),
        ("Output Conv1D (1x1)", "128 -> 20 channels")
    ]
    for name, sub in layers1:
        box_l = FancyBboxPatch((xs[0]+0.15, y_cur-0.28), w-0.3, 0.3, boxstyle="square,pad=0.02",
                               edgecolor='#bdc3c7', facecolor='#ebf5fb', linewidth=0.8)
        ax.add_patch(box_l)
        ax.text(xs[0]+w/2, y_cur-0.12, name, fontsize=7.5, fontweight='bold', ha='center', color='#1b4f72')
        ax.text(xs[0]+w/2, y_cur-0.24, sub, fontsize=6.8, ha='center', color='#555')
        y_cur -= 0.33

    ax.text(xs[0]+w/2, 0.95, "Params: 3,005,328 (Base)", fontsize=8, fontweight='bold', ha='center', color='#27ae60')
    ax.text(xs[0]+w/2, 0.75, "Bias: Local Spatial/Spectral", fontsize=7.5, ha='center', color='#333')

    # -------------------------------------------------------------
    # 2. Hybrid CNN-Attention
    # -------------------------------------------------------------
    b2 = FancyBboxPatch((xs[1], 0.65), w, 4.75, boxstyle="round,pad=0.08",
                        edgecolor='#8e44ad', facecolor='#ffffff', linewidth=1.4)
    ax.add_patch(b2)
    ax.text(xs[1]+w/2, 5.15, "2. Hybrid Attention", fontsize=10, fontweight='bold', ha='center', color='#8e44ad')
    ax.text(xs[1]+w/2, 4.9, "CNN Body + MHSA Bottleneck", fontsize=8.5, fontstyle='italic', ha='center', color='#555')

    y_cur = 4.5
    layers2 = [
        ("Input Conv1D (k=3, s=1)", "20 -> 128 channels"),
        ("Encoder Scales 1-4", "260 -> 17 tokens"),
        ("Bottleneck Transpose", "Shape: [1, 17, 128]"),
        ("Fixed Sinusoidal PE", "Added to 17 tokens"),
        ("Pre-LayerNorm(128)", "Channel Normalization"),
        ("4-Head MHSA (d_k=32)", "Q, K, V Projections (128)"),
        ("Residual Add (+)", "x = x + MHSA(LN(x))"),
        ("Pre-LayerNorm + FFN", "Linear(128->512->128)"),
        ("Residual Add (+)", "x = x + FFN(LN(x))"),
        ("Decoder Scales 4-1", "Linear Interp + Skip Cat"),
        ("Output Conv1D (1x1)", "128 -> 20 channels")
    ]
    for name, sub in layers2:
        is_attn = "MHSA" in name or "FFN" in name or "PE" in name
        bg_col = '#f5eef8' if is_attn else '#f4f6f6'
        fg_col = '#512e5f' if is_attn else '#2c3e50'
        box_l = FancyBboxPatch((xs[1]+0.15, y_cur-0.28), w-0.3, 0.3, boxstyle="square,pad=0.02",
                               edgecolor='#a569bd' if is_attn else '#bdc3c7', facecolor=bg_col, linewidth=1.0 if is_attn else 0.8)
        ax.add_patch(box_l)
        ax.text(xs[1]+w/2, y_cur-0.12, name, fontsize=7.5, fontweight='bold', ha='center', color=fg_col)
        ax.text(xs[1]+w/2, y_cur-0.24, sub, fontsize=6.8, ha='center', color='#555')
        y_cur -= 0.33

    ax.text(xs[1]+w/2, 0.95, "Params: 3,137,808 (+4.4%)", fontsize=8, fontweight='bold', ha='center', color='#8e44ad')
    ax.text(xs[1]+w/2, 0.75, "Bias: Local + Global Routing", fontsize=7.5, ha='center', color='#333')

    # -------------------------------------------------------------
    # 3. Conv Control
    # -------------------------------------------------------------
    b3 = FancyBboxPatch((xs[2], 0.65), w, 4.75, boxstyle="round,pad=0.08",
                        edgecolor='#d35400', facecolor='#ffffff', linewidth=1.4)
    ax.add_patch(b3)
    ax.text(xs[2]+w/2, 5.15, "3. Conv Control", fontsize=10, fontweight='bold', ha='center', color='#d35400')
    ax.text(xs[2]+w/2, 4.9, "Parameter-Matched Conv", fontsize=8.5, fontstyle='italic', ha='center', color='#555')

    y_cur = 4.5
    layers3 = [
        ("Input Conv1D (k=3, s=1)", "20 -> 128 channels"),
        ("Encoder Scales 1-4", "260 -> 17 tokens"),
        ("Bottleneck Transpose", "Shape: [1, 17, 128]"),
        ("Fixed Sinusoidal PE", "Added to 17 tokens"),
        ("Conv1D Block 1 (k=3)", "128 -> 172 channels"),
        ("GELU Non-Linearity", "Nonlinear Activation"),
        ("Conv1D Block 2 (k=3)", "172 -> 128 channels"),
        ("LayerNorm(128)", "Channel Normalization"),
        ("Residual Add (+)", "x = x + Conv(x)"),
        ("Decoder Scales 4-1", "Linear Interp + Skip Cat"),
        ("Output Conv1D (1x1)", "128 -> 20 channels")
    ]
    for name, sub in layers3:
        is_c = "Conv1D Block" in name or "172" in sub
        bg_col = '#fbeee6' if is_c else '#f4f6f6'
        fg_col = '#7e5109' if is_c else '#2c3e50'
        box_l = FancyBboxPatch((xs[2]+0.15, y_cur-0.28), w-0.3, 0.3, boxstyle="square,pad=0.02",
                               edgecolor='#e59866' if is_c else '#bdc3c7', facecolor=bg_col, linewidth=1.0 if is_c else 0.8)
        ax.add_patch(box_l)
        ax.text(xs[2]+w/2, y_cur-0.12, name, fontsize=7.5, fontweight='bold', ha='center', color=fg_col)
        ax.text(xs[2]+w/2, y_cur-0.24, sub, fontsize=6.8, ha='center', color='#555')
        y_cur -= 0.33

    ax.text(xs[2]+w/2, 0.95, "Params: 3,137,980 (+172)", fontsize=8, fontweight='bold', ha='center', color='#d35400')
    ax.text(xs[2]+w/2, 0.75, "Bias: Deep Local Convolution", fontsize=7.5, ha='center', color='#333')

    # -------------------------------------------------------------
    # 4. Pointwise Control
    # -------------------------------------------------------------
    b4 = FancyBboxPatch((xs[3], 0.65), w, 4.75, boxstyle="round,pad=0.08",
                        edgecolor='#16a085', facecolor='#ffffff', linewidth=1.4)
    ax.add_patch(b4)
    ax.text(xs[3]+w/2, 5.15, "4. Pointwise Control", fontsize=10, fontweight='bold', ha='center', color='#16a085')
    ax.text(xs[3]+w/2, 4.9, "1x1 Linear Channels", fontsize=8.5, fontstyle='italic', ha='center', color='#555')

    y_cur = 4.5
    layers4 = [
        ("Input Conv1D (k=3, s=1)", "20 -> 128 channels"),
        ("Encoder Scales 1-4", "260 -> 17 tokens"),
        ("Bottleneck Transpose", "Shape: [1, 17, 128]"),
        ("Fixed Sinusoidal PE", "Added to 17 tokens"),
        ("Pre-LayerNorm(128)", "Channel Normalization"),
        ("Linear Pointwise 1", "128 -> 256 (GELU)"),
        ("Linear Pointwise 2", "256 -> 128 (No spatial)"),
        ("Residual Add (+)", "x = x + Linear(LN(x))"),
        ("Pre-LayerNorm + FFN", "Linear(128->256->128)"),
        ("Decoder Scales 4-1", "Linear Interp + Skip Cat"),
        ("Output Conv1D (1x1)", "128 -> 20 channels")
    ]
    for name, sub in layers4:
        is_pw = "Pointwise" in name or "Linear" in name
        bg_col = '#e8f8f5' if is_pw else '#f4f6f6'
        fg_col = '#0e6251' if is_pw else '#2c3e50'
        box_l = FancyBboxPatch((xs[3]+0.15, y_cur-0.28), w-0.3, 0.3, boxstyle="square,pad=0.02",
                               edgecolor='#76d7c4' if is_pw else '#bdc3c7', facecolor=bg_col, linewidth=1.0 if is_pw else 0.8)
        ax.add_patch(box_l)
        ax.text(xs[3]+w/2, y_cur-0.12, name, fontsize=7.5, fontweight='bold', ha='center', color=fg_col)
        ax.text(xs[3]+w/2, y_cur-0.24, sub, fontsize=6.8, ha='center', color='#555')
        y_cur -= 0.33

    ax.text(xs[3]+w/2, 0.95, "Params: 3,137,680 (-128)", fontsize=8, fontweight='bold', ha='center', color='#16a085')
    ax.text(xs[3]+w/2, 0.75, "Bias: Zero Window (k=1)", fontsize=7.5, ha='center', color='#333')

    # -------------------------------------------------------------
    # 5. Pure Spectral Transformer with ALiBi
    # -------------------------------------------------------------
    b5 = FancyBboxPatch((xs[4], 0.65), w, 4.75, boxstyle="round,pad=0.08",
                        edgecolor='#c0392b', facecolor='#ffffff', linewidth=1.4)
    ax.add_patch(b5)
    ax.text(xs[4]+w/2, 5.15, "5. Pure Transformer", fontsize=10, fontweight='bold', ha='center', color='#c0392b')
    ax.text(xs[4]+w/2, 4.9, "Pre-LN + ALiBi Biases", fontsize=8.5, fontstyle='italic', ha='center', color='#555')

    y_cur = 4.5
    layers5 = [
        ("Token Permute & Linear", "20 -> 128 (L=260 tokens)"),
        ("ALiBi Relative Slopes", "m_h = 2^(-8h/4) = [1/4..1/256]"),
        ("Transformer Block 1", "Pre-LN + 4-Head MHSA + ALiBi"),
        ("FeedForward Block 1", "Linear(128->512->128)"),
        ("Transformer Block 2", "Pre-LN + 4-Head MHSA + ALiBi"),
        ("FeedForward Block 2", "Linear(128->512->128)"),
        ("Transformer Block 3", "Pre-LN + 4-Head MHSA + ALiBi"),
        ("FeedForward Block 3", "Linear(128->512->128)"),
        ("Transformer Block 4", "Pre-LN + 4-Head MHSA + ALiBi"),
        ("FeedForward Block 4", "Linear(128->512->128)"),
        ("Final LN + Output Proj", "Linear(128 -> 20 channels)")
    ]
    for name, sub in layers5:
        is_alibi = "ALiBi" in name or "Transformer" in name
        bg_col = '#fcedec' if is_alibi else '#f4f6f6'
        fg_col = '#78281f' if is_alibi else '#2c3e50'
        box_l = FancyBboxPatch((xs[4]+0.15, y_cur-0.28), w-0.3, 0.3, boxstyle="square,pad=0.02",
                               edgecolor='#e6b0aa' if is_alibi else '#bdc3c7', facecolor=bg_col, linewidth=1.0 if is_alibi else 0.8)
        ax.add_patch(box_l)
        ax.text(xs[4]+w/2, y_cur-0.12, name, fontsize=7.5, fontweight='bold', ha='center', color=fg_col)
        ax.text(xs[4]+w/2, y_cur-0.24, sub, fontsize=6.8, ha='center', color='#555')
        y_cur -= 0.33

    ax.text(xs[4]+w/2, 0.95, "Params: 3,004,764 (-564)", fontsize=8, fontweight='bold', ha='center', color='#c0392b')
    ax.text(xs[4]+w/2, 0.75, "Bias: Distance-Decayed Global", fontsize=7.5, ha='center', color='#333')

    plt.tight_layout()
    output_path = r'E:\Rakhal_Tensor\NeuroTensor_Project_Report\figures\architecture_5models_pipeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated architecture diagram at: {output_path}")

if __name__ == '__main__':
    create_architecture_diagram()

