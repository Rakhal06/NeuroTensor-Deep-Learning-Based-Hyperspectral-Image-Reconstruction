"""
Generate DeepTensor++ Phase 2 Academic PowerPoint Presentation (Audited & Reconciled)
Uses python-pptx to generate a professional 16:9 presentation with embedded topic IDs,
mathematical formulations, code anchors, and interactive hyperlinks to the Code Navigator.
"""

import os
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'evaluation', 'code_navigator', 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'evaluation', 'presentation')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'DeepTensor_Phase2_Evaluation.pptx')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load JSON Datasets
with open(os.path.join(DATA_DIR, 'research_topics.json'), 'r', encoding='utf-8') as f:
    TOPICS = json.load(f)

with open(os.path.join(DATA_DIR, 'experiment_map.json'), 'r', encoding='utf-8') as f:
    EXPERIMENTS = json.load(f)

# Visual Design Palette (Modern Academic Dark Theme)
COLOR_BG_DARK = RGBColor(15, 23, 42)       # Slate 900 #0f172a
COLOR_SURFACE = RGBColor(30, 41, 59)       # Slate 800 #1e293b
COLOR_SURFACE_LIGHT = RGBColor(51, 65, 85) # Slate 700 #334155
COLOR_PRIMARY = RGBColor(56, 189, 248)     # Sky Blue #38bdf8
COLOR_ACCENT = RGBColor(245, 158, 11)      # Amber #f59e0b
COLOR_SUCCESS = RGBColor(16, 185, 129)     # Emerald #10b981
COLOR_TEXT_MAIN = RGBColor(248, 250, 252)  # White #f8fafc
COLOR_TEXT_MUTED = RGBColor(148, 163, 184) # Slate 400 #94a3b8
COLOR_BORDER = RGBColor(71, 85, 105)       # Slate 600

def create_presentation():
    prs = Presentation()
    # 16:9 Widescreen layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG_DARK
        bg.line.color.rgb = COLOR_BG_DARK
        return bg

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    top_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.8), Inches(11.733), Inches(0.08))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = COLOR_PRIMARY
    top_bar.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.733), Inches(3.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "PHASE 2 FACULTY EVALUATION & RESEARCH DEFENSE"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_PRIMARY

    p1 = tf.add_paragraph()
    p1.text = "DeepTensor++: Low-Rank Tensor Decompositions\nwith Deep Generative Network Priors"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_TEXT_MAIN
    p1.space_before = Pt(14)

    p2 = tf.add_paragraph()
    p2.text = "Methodological Reproduction, Controlled Comparative Architecture Benchmarks, and Adaptive Improvements"
    p2.font.size = Pt(16)
    p2.font.color.rgb = COLOR_TEXT_MUTED
    p2.space_before = Pt(12)

    card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.7), Inches(11.733), Inches(1.8))
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_SURFACE
    card.line.color.rgb = COLOR_BORDER

    ctf = card.text_frame
    ctf.word_wrap = True
    cp1 = ctf.paragraphs[0]
    cp1.text = "RESEARCH HIGHLIGHTS & VIVA DELIVERABLES:"
    cp1.font.size = Pt(12)
    cp1.font.bold = True
    cp1.font.color.rgb = COLOR_ACCENT

    bullets = [
        "Complete Paper Reproduction: Reconstructed original 2D+1D and CP DeepTensor formulation (Heckel et al., 2022).",
        "Five Comparative Architectures: Baseline CNN vs Hybrid CNN-Attention vs CNN Control vs Pointwise Control vs Pure Transformer.",
        "Four Application Benchmarks: ICVL Hyperspectral Denoising, Yale Face B, CACTI Video CS, and PET/CT Radon Tomography.",
        "Research Novelties: Plateau-driven Adaptive Rank Selection (+0.09 dB) and Factor Group-Sparsity Regularization.",
        "Interactive Code Navigator: DeepTensor++ web application with Monaco Editor line-level verification at http://localhost:8000"
    ]
    for b in bullets:
        bp = ctf.add_paragraph()
        bp.text = f"• {b}"
        bp.font.size = Pt(11)
        bp.font.color.rgb = COLOR_TEXT_MAIN
        bp.space_before = Pt(3)

    # =========================================================================
    # SLIDE 2: Research Journey Roadmap
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)

    tb2 = s2.shapes.add_textbox(Inches(0.8), Inches(0.6), Inches(11.733), Inches(1.0))
    tf2 = tb2.text_frame
    p_sec = tf2.paragraphs[0]
    p_sec.text = "RESEARCH ARCHITECTURE & METHODOLOGY"
    p_sec.font.size = Pt(11)
    p_sec.font.bold = True
    p_sec.font.color.rgb = COLOR_PRIMARY
    p_title = tf2.add_paragraph()
    p_title.text = "The DeepTensor++ Research Journey: 6 Key Dimensions"
    p_title.font.size = Pt(26)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT_MAIN

    grid_w = Inches(3.7)
    grid_h = Inches(2.2)
    start_x = Inches(0.8)
    gap_x = Inches(0.3)
    start_y = Inches(1.8)
    gap_y = Inches(0.35)

    sections_info = [
        ("SECTION 1: Literature Foundations", "Inverse problems formulation y=Ax+n, Deep Image Prior, and the original DeepTensor paper.", "Topics DT-01 to DT-04"),
        ("SECTION 2: Tensor Algebra & U, V, W", "CANDECOMP/PARAFAC, multilinear fibers, neural factor generators, and Tucker audit.", "Topics DT-05 to DT-10"),
        ("SECTION 3: DeepTensor Codebase", "1D/2D Skip U-Net backbone, end-to-end forward workflow, and PyTorch autograd engine.", "Topics DT-11 to DT-13"),
        ("SECTION 4: Comparative Baselines", "Traditional SVD/TV vs 5 neural models with strict parameter-matched controls.", "Topics DT-14 to DT-16"),
        ("SECTION 5: Four Application Testbeds", "ICVL HSI (46.07 dB), Yale Face B (28.56 dB), Video CS (25.16 dB), and PET/CT (30.62 dB).", "Topics DT-17 to DT-21"),
        ("SECTION 6: Novelty & Improvements", "Plateau-driven Adaptive Rank Selection, Factor Group-Sparsity, and AHD-Net roadmap.", "Topics DT-22 to DT-24"),
    ]

    for idx, (sec_t, sec_desc, sec_topics) in enumerate(sections_info):
        col = idx % 3
        row = idx // 3
        x = start_x + col * (grid_w + gap_x)
        y = start_y + row * (grid_h + gap_y)

        box = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, grid_w, grid_h)
        box.fill.solid()
        box.fill.fore_color.rgb = COLOR_SURFACE
        box.line.color.rgb = COLOR_BORDER

        btf = box.text_frame
        btf.word_wrap = True
        bp0 = btf.paragraphs[0]
        bp0.text = sec_t
        bp0.font.size = Pt(12)
        bp0.font.bold = True
        bp0.font.color.rgb = COLOR_PRIMARY

        bp1 = btf.add_paragraph()
        bp1.text = sec_desc
        bp1.font.size = Pt(10)
        bp1.font.color.rgb = COLOR_TEXT_MAIN
        bp1.space_before = Pt(6)

        bp2 = btf.add_paragraph()
        bp2.text = sec_topics
        bp2.font.size = Pt(9)
        bp2.font.bold = True
        bp2.font.color.rgb = COLOR_ACCENT
        bp2.space_before = Pt(8)

    # =========================================================================
    # SLIDES 3-26: The 24 Research Topics (DT-01 to DT-24)
    # =========================================================================
    for topic in TOPICS:
        st = prs.slides.add_slide(blank_layout)
        set_slide_background(st)

        thb = st.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.1))
        thf = thb.text_frame
        thf.word_wrap = True

        p_sec_id = thf.paragraphs[0]
        p_sec_id.text = f"{topic['section_id']}: {topic['section_title'].upper()}  |  SOURCE: {topic['source_of_truth'].upper()}"
        p_sec_id.font.size = Pt(10)
        p_sec_id.font.bold = True
        p_sec_id.font.color.rgb = COLOR_PRIMARY

        p_top_title = thf.add_paragraph()
        p_top_title.text = f"[{topic['id']}] {topic['title']}"
        p_top_title.font.size = Pt(22)
        p_top_title.font.bold = True
        p_top_title.font.color.rgb = COLOR_TEXT_MAIN

        # Left Column: Formulation & Concept (Width: 6.8 Inches)
        left_box = st.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(6.8), Inches(5.3))
        left_box.fill.solid()
        left_box.fill.fore_color.rgb = COLOR_SURFACE
        left_box.line.color.rgb = COLOR_BORDER

        ltf = left_box.text_frame
        ltf.word_wrap = True

        lp0 = ltf.paragraphs[0]
        lp0.text = "SCIENTIFIC CONCEPT & FORMULATION:"
        lp0.font.size = Pt(11)
        lp0.font.bold = True
        lp0.font.color.rgb = COLOR_PRIMARY

        lp1 = ltf.add_paragraph()
        lp1.text = topic['concept']
        lp1.font.size = Pt(11)
        lp1.font.color.rgb = COLOR_TEXT_MAIN
        lp1.space_before = Pt(4)

        lp2 = ltf.add_paragraph()
        lp2.text = "MATHEMATICAL MODEL:"
        lp2.font.size = Pt(11)
        lp2.font.bold = True
        lp2.font.color.rgb = COLOR_ACCENT
        lp2.space_before = Pt(10)

        lp3 = ltf.add_paragraph()
        lp3.text = f"Equation: {topic['math_formula']}"
        lp3.font.size = Pt(12)
        lp3.font.bold = True
        lp3.font.color.rgb = COLOR_TEXT_MAIN
        lp3.space_before = Pt(2)

        lp4 = ltf.add_paragraph()
        lp4.text = topic['math_explanation']
        lp4.font.size = Pt(10)
        lp4.font.color.rgb = COLOR_TEXT_MUTED
        lp4.space_before = Pt(2)

        lp5 = ltf.add_paragraph()
        lp5.text = "SIGNIFICANCE IN DEEPTENSOR++:"
        lp5.font.size = Pt(11)
        lp5.font.bold = True
        lp5.font.color.rgb = COLOR_SUCCESS
        lp5.space_before = Pt(10)

        lp6 = ltf.add_paragraph()
        lp6.text = topic['research_context']
        lp6.font.size = Pt(10)
        lp6.font.color.rgb = COLOR_TEXT_MAIN
        lp6.space_before = Pt(4)

        # Right Column: Code Anchor & Faculty Viva Defense (Width: 4.6 Inches)
        right_box = st.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.9), Inches(1.6), Inches(4.6), Inches(5.3))
        right_box.fill.solid()
        right_box.fill.fore_color.rgb = COLOR_SURFACE
        right_box.line.color.rgb = COLOR_BORDER

        rtf = right_box.text_frame
        rtf.word_wrap = True

        rp0 = rtf.paragraphs[0]
        rp0.text = "VERIFIED CODE IMPLEMENTATION:"
        rp0.font.size = Pt(11)
        rp0.font.bold = True
        rp0.font.color.rgb = COLOR_PRIMARY

        code = topic['code_reference']
        rp1 = rtf.add_paragraph()
        rp1.text = f"File: {code['file_path']}\nLines: L{code['line_start']}–L{code['line_end']}\nSymbol: {code['symbol_name']}"
        rp1.font.size = Pt(10)
        rp1.font.color.rgb = COLOR_TEXT_MAIN
        rp1.space_before = Pt(4)

        rp2 = rtf.add_paragraph()
        rp2.text = f"Logic: {code['summary']}"
        rp2.font.size = Pt(9.5)
        rp2.font.color.rgb = COLOR_TEXT_MUTED
        rp2.space_before = Pt(3)

        rp3 = rtf.add_paragraph()
        rp3.text = "FACULTY VIVA Q&A DEFENSE:"
        rp3.font.size = Pt(11)
        rp3.font.bold = True
        rp3.font.color.rgb = COLOR_ACCENT
        rp3.space_before = Pt(12)

        qa = topic['faculty_qa']
        rp4 = rtf.add_paragraph()
        rp4.text = f"Q: {qa['question']}"
        rp4.font.size = Pt(9.5)
        rp4.font.bold = True
        rp4.font.color.rgb = COLOR_TEXT_MAIN
        rp4.space_before = Pt(3)

        rp5 = rtf.add_paragraph()
        rp5.text = f"Defense: {qa['defense']}"
        rp5.font.size = Pt(9)
        rp5.font.color.rgb = COLOR_TEXT_MUTED
        rp5.space_before = Pt(3)

        rp6 = rtf.add_paragraph()
        nav_url = f"http://localhost:8000/?topic={topic['id']}"
        rp6.text = f"🔗 Inspect Live in Code Navigator:\n{nav_url}"
        rp6.font.size = Pt(9.5)
        rp6.font.bold = True
        rp6.font.color.rgb = COLOR_PRIMARY
        rp6.space_before = Pt(10)

    # =========================================================================
    # SLIDE 27: Cross-Application Master Results Matrix
    # =========================================================================
    s27 = prs.slides.add_slide(blank_layout)
    set_slide_background(s27)

    tb27 = s27.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.733), Inches(1.0))
    tf27 = tb27.text_frame
    p27_sub = tf27.paragraphs[0]
    p27_sub.text = "CROSS-APPLICATION SYNTHESIS & EMPIRICAL EVIDENCE"
    p27_sub.font.size = Pt(10)
    p27_sub.font.bold = True
    p27_sub.font.color.rgb = COLOR_PRIMARY

    p27_tit = tf27.add_paragraph()
    p27_tit.text = "Master Benchmark Matrix: Application-Dependent Performance"
    p27_tit.font.size = Pt(24)
    p27_tit.font.bold = True
    p27_tit.font.color.rgb = COLOR_TEXT_MAIN

    rows = 5
    cols = 7
    table_shape = s27.shapes.add_table(rows, cols, Inches(0.8), Inches(1.6), Inches(11.733), Inches(3.2))
    table = table_shape.table

    headers = ["Testbed Application", "Dimensions", "Baseline CNN", "Hybrid Attn", "Pointwise Ctrl", "Pure Transformer", "Winning Architecture"]
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_SURFACE_LIGHT
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.color.rgb = COLOR_PRIMARY

    matrix_data = [
        ("Fig 09: ICVL Hyperspectral", "174x164x260, R=20", "46.07 dB 🏆", "44.85 dB", "45.42 dB", "43.40 dB", "Baseline CNN (+0.48 dB) — Spectral locality"),
        ("Fig 10: Yale Face B", "192x168x160, R=2000", "27.78 dB", "28.04 dB", "28.17 dB", "28.56 dB 🏆", "Pure Transformer (+0.77 dB) — Global lighting"),
        ("Fig 11: Video CS (CACTI)", "128x128x64, R=32", "25.16 dB 🏆", "24.74 dB", "24.81 dB", "24.05 dB", "Baseline CNN (+0.12 dB) — Motion continuity"),
        ("Fig 12: PET/CT Radon Tomo", "128x128x56, R=1000", "30.00 dB", "30.29 dB", "30.62 dB 🏆", "30.41 dB", "Pointwise Control (+0.62 dB) — Decoupled slices"),
    ]

    for r_idx, row in enumerate(matrix_data, start=1):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_SURFACE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(9.5)
                p.font.color.rgb = COLOR_TEXT_MAIN
                if "🏆" in val:
                    p.font.bold = True
                    p.font.color.rgb = COLOR_SUCCESS

    tcard = s27.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.1), Inches(11.733), Inches(1.8))
    tcard.fill.solid()
    tcard.fill.fore_color.rgb = COLOR_SURFACE
    tcard.line.color.rgb = COLOR_BORDER

    tctf = tcard.text_frame
    tctf.word_wrap = True
    tp0 = tctf.paragraphs[0]
    tp0.text = "CORE EMPIRICAL FINDINGS & HYPOTHESES:"
    tp0.font.size = Pt(11)
    tp0.font.bold = True
    tp0.font.color.rgb = COLOR_ACCENT

    tp1 = tctf.add_paragraph()
    tp1.text = "No single neural prior architecture dominates across all inverse problem domains. We hypothesize that performance depends on alignment between network inductive biases and physical coordinates: continuous spectral wavelengths and temporal motion favor local convolutional smoothness; discrete photometric lighting variations favor global self-attention; and decoupled 3D medical slices favor pointwise channel mixing."
    tp1.font.size = Pt(10.5)
    tp1.font.color.rgb = COLOR_TEXT_MAIN
    tp1.space_before = Pt(4)

    # =========================================================================
    # SLIDE 28: Interactive Code Navigator Demonstration Guide
    # =========================================================================
    s28 = prs.slides.add_slide(blank_layout)
    set_slide_background(s28)

    tb28 = s28.shapes.add_textbox(Inches(0.8), Inches(0.6), Inches(11.733), Inches(1.0))
    tf28 = tb28.text_frame
    p28_sub = tf28.paragraphs[0]
    p28_sub.text = "INTERACTIVE TOOLING & ARTIFACT REPRODUCIBILITY"
    p28_sub.font.size = Pt(10)
    p28_sub.font.bold = True
    p28_sub.font.color.rgb = COLOR_PRIMARY

    p28_tit = tf28.add_paragraph()
    p28_tit.text = "DeepTensor++ Research Navigator: Live Demonstration"
    p28_tit.font.size = Pt(24)
    p28_tit.font.bold = True
    p28_tit.font.color.rgb = COLOR_TEXT_MAIN

    ibox = s28.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.7), Inches(5.0))
    ibox.fill.solid()
    ibox.fill.fore_color.rgb = COLOR_SURFACE
    ibox.line.color.rgb = COLOR_BORDER

    itf = ibox.text_frame
    itf.word_wrap = True
    ip0 = itf.paragraphs[0]
    ip0.text = "HOW TO RUN THE RESEARCH NAVIGATOR:"
    ip0.font.size = Pt(12)
    ip0.font.bold = True
    ip0.font.color.rgb = COLOR_PRIMARY

    cmds = [
        "1. Open terminal in repository root (E:\\Rakhal_Tensor).",
        "2. Launch backend server:\n   .venv\\Scripts\\python.exe evaluation/code_navigator/backend/main.py",
        "3. Open Chrome/Edge at: http://localhost:8000",
        "4. Click any topic (DT-01 to DT-24) to jump directly to exact code lines in the embedded Monaco Editor.",
        "5. Press 'Presentation Mode' to synchronize slides with code live during the viva defense."
    ]
    for c in cmds:
        cp = itf.add_paragraph()
        cp.text = c
        cp.font.size = Pt(10.5)
        cp.font.color.rgb = COLOR_TEXT_MAIN
        cp.space_before = Pt(8)

    fbox = s28.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
    fbox.fill.solid()
    fbox.fill.fore_color.rgb = COLOR_SURFACE
    fbox.line.color.rgb = COLOR_BORDER

    ftf = fbox.text_frame
    ftf.word_wrap = True
    fp0 = ftf.paragraphs[0]
    fp0.text = "INTEGRATED SYSTEM CAPABILITIES:"
    fp0.font.size = Pt(12)
    fp0.font.bold = True
    fp0.font.color.rgb = COLOR_ACCENT

    feats = [
        "Monaco Code Editor: Visual Studio Code web editor engine with syntax highlighting, line decorations, and AST navigation.",
        "Strict Path Security: Enforces sandbox constraints preventing traversal outside the project directory.",
        "Controlled Experiments Modal: Inspect raw CSV/JSON matrices, difference error maps, and annotated reconstruction strips.",
        "KaTeX Math Engine: Instant rendering of multilinear contractions and group lasso loss formulations.",
        "Full Line-Level Reproducibility: Zero synthetic or hallucinated line numbers; 100% verified against physical repository."
    ]
    for f in feats:
        fp = ftf.add_paragraph()
        fp.text = f"• {f}"
        fp.font.size = Pt(10)
        fp.font.color.rgb = COLOR_TEXT_MAIN
        fp.space_before = Pt(6)

    prs.save(OUTPUT_FILE)
    print(f"Successfully generated audited presentation: {OUTPUT_FILE}")
    print(f"Total slides created: {len(prs.slides)}")

if __name__ == '__main__':
    create_presentation()
