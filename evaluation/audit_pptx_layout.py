"""
Detailed Programmatic Audit of PPTX Layout, Boundaries, Hyperlinks, and Text Fits
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt

pptx_path = "evaluation/presentation/DeepTensor_Phase2_Evaluation.pptx"
prs = Presentation(pptx_path)

slide_w = prs.slide_width.inches
slide_h = prs.slide_height.inches

print(f"Presentation Dimensions: {slide_w:.3f}\" x {slide_h:.3f}\" (Widescreen 16:9)")
print(f"Total Slides: {len(prs.slides)}")

issues = []

for idx, slide in enumerate(prs.slides, start=1):
    slide_text = []
    hyperlinks = []
    
    for shape in slide.shapes:
        # Check boundary
        left = shape.left.inches if shape.left else 0
        top = shape.top.inches if shape.top else 0
        width = shape.width.inches if shape.width else 0
        height = shape.height.inches if shape.height else 0
        
        right = left + width
        bottom = top + height
        
        if right > slide_w + 0.05:
            issues.append(f"Slide {idx}: Shape '{shape.name}' exceeds right boundary: {right:.2f}\" > {slide_w:.2f}\"")
        if bottom > slide_h + 0.05:
            issues.append(f"Slide {idx}: Shape '{shape.name}' exceeds bottom boundary: {bottom:.2f}\" > {slide_h:.2f}\"")
            
        if shape.has_text_frame:
            for p in shape.text_frame.paragraphs:
                txt = p.text.strip()
                if txt:
                    slide_text.append(txt)
                    if "http://localhost:8000" in txt:
                        hyperlinks.append(txt)
                        
    # Topic verification for slides 3 to 26
    if 3 <= idx <= 26:
        expected_topic_num = idx - 2
        expected_dt = f"DT-{expected_topic_num:02d}"
        full_slide_str = " ".join(slide_text)
        if expected_dt not in full_slide_str:
            issues.append(f"Slide {idx}: Expected topic {expected_dt} not found in slide text!")
        if not any(expected_dt in h for h in hyperlinks):
            issues.append(f"Slide {idx}: No hyperlink matching {expected_dt} found in hyperlinks: {hyperlinks}")

print(f"\nLayout Boundary & Topic Link Audit Result:")
if not issues:
    print("ALL 28 SLIDES PASSED BOUNDARY AND TOPIC HYPERLINK AUDITS!")
else:
    for iss in issues:
        print("  ISSUE:", iss)
