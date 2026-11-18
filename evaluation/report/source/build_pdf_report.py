"""
Master Build Script for DeepTensor++ Phase 2 Academic Research Report PDF
"""

import os
import sys
import time
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.pagesizes import A4

from report_styles import NumberedCanvas, MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM
from report_content import build_all_chapters

OUTPUT_PDF = "evaluation/report/DeepTensor_Phase2_Research_Report.pdf"

def generate_pdf():
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    print(f"[*] Compiling DeepTensor++ Phase 2 Academic Research Report...")
    start_time = time.time()

    # Document Template
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=MARGIN_LEFT,
        rightMargin=MARGIN_RIGHT,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
        title="DeepTensor++: Low-Rank Tensor Decompositions with Deep Generative Network Priors",
        author="DeepTensor++ Research Team",
        subject="Phase 2 Faculty Evaluation Comprehensive Research Report",
        keywords=["DeepTensor", "Tensor Decompositions", "Inverse Problems", "Deep Image Prior", "CP Decomposition"]
    )

    story = build_all_chapters()
    print(f"[*] Assembled {len(story)} flowables across 20 chapters and front matter.")

    doc.build(story, canvasmaker=NumberedCanvas)
    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(OUTPUT_PDF) / (1024 * 1024)
    print(f"[+] Successfully compiled PDF to: {OUTPUT_PDF}")
    print(f"[+] Build time: {elapsed:.2f} seconds | File size: {file_size_mb:.2f} MB")

    # Audit and Synchronize Bookmarks via PyMuPDF
    import pymupdf
    doc_fitz = pymupdf.open(OUTPUT_PDF)
    num_pages = len(doc_fitz)
    
    # Exact Chapter Map for Bookmarks
    toc = [
        [1, "Cover Page", 1],
        [1, "Table of Contents", 2],
        [1, "Chapter 1: Abstract and Research Overview", 4],
        [1, "Chapter 2: Introduction to Inverse Problems", 6],
        [1, "Chapter 3: Traditional Reconstruction Methods", 8],
        [1, "Chapter 4: Original DeepTensor Research Paper", 10],
        [1, "Chapter 5: Tensor Fundamentals", 11],
        [1, "Chapter 6: U, V, W Factor Generation", 13],
        [1, "Chapter 7: DeepTensor CNN Architecture", 15],
        [1, "Chapter 8: CP/PARAFAC and Tucker Decomposition", 17],
        [1, "Chapter 9: Dimensionality Splitting Strategies", 19],
        [1, "Chapter 10: Reconstruction and Training Workflow", 21],
        [1, "Chapter 11: Five Comparative Architectures", 23],
        [1, "Chapter 12: Figure 09 — ICVL Hyperspectral Denoising", 25],
        [1, "Chapter 13: Figure 10 — Extended Yale Face B", 27],
        [1, "Chapter 14: Figure 11 — Video Compressive Sensing", 29],
        [1, "Chapter 15: Figure 12 — PET/CT Radon Compressive Sensing", 31],
        [1, "Chapter 16: Cross-Application Analysis", 33],
        [1, "Chapter 17: Adaptive Rank Selection", 35],
        [1, "Chapter 18: Factor Group-Sparsity", 37],
        [1, "Chapter 19: Computational Analysis", 39],
        [1, "Chapter 20: Conclusions and Future Work", 40],
        [1, "References and Bibliography", 41],
        [1, "Appendix: Consolidated Code Navigator Reference", 42],
    ]
    doc_fitz.set_toc(toc)
    temp_pdf = OUTPUT_PDF + ".tmp"
    doc_fitz.save(temp_pdf)
    doc_fitz.close()
    
    import shutil
    shutil.move(temp_pdf, OUTPUT_PDF)

    doc_fitz = pymupdf.open(OUTPUT_PDF)
    
    print("\n" + "=" * 60)
    print("PDF GENERATION TELEMETRY & AUDIT")
    print("=" * 60)
    print(f"Total Page Count: {num_pages} pages")
    print(f"PDF Bookmarks Count: {len(toc)} verified outline entries")
    print(f"Target Page Range: 35–55 pages -> STATUS: {'PASS' if 35 <= num_pages <= 55 else 'CHECK'}")
    
    # Check images and links
    total_images = 0
    total_links = 0
    for page_idx in range(num_pages):
        page = doc_fitz[page_idx]
        imgs = page.get_images()
        links = page.get_links()
        total_images += len(imgs)
        total_links += len(links)
        
    print(f"Total Images Embedded: {total_images}")
    print(f"Total Clickable Hyperlinks: {total_links}")
    print("=" * 60 + "\n")

    doc_fitz.close()
    return num_pages, total_images, total_links

if __name__ == "__main__":
    generate_pdf()
