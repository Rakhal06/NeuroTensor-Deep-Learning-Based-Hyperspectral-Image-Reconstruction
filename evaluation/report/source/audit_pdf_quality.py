"""
Comprehensive Quality Control and Page Rendering Script for DeepTensor++ Research Report PDF
"""

import os
import pymupdf  # PyMuPDF

PDF_PATH = "evaluation/report/DeepTensor_Phase2_Research_Report.pdf"
RENDER_DIR = "evaluation/report/rendered_pages"
os.makedirs(RENDER_DIR, exist_ok=True)

def audit_and_render_pdf():
    print(f"[*] Opening PDF: {PDF_PATH}")
    doc = pymupdf.open(PDF_PATH)
    total_pages = len(doc)
    print(f"[*] Total pages: {total_pages}")

    audit_results = {
        "total_pages": total_pages,
        "blank_pages": [],
        "pages_with_images": 0,
        "total_images": 0,
        "total_links": 0,
        "dt_hyperlinks": {},
        "bookmarks": [],
        "chapter_start_pages": {}
    }

    # Extract Bookmarks
    toc = doc.get_toc()
    for item in toc:
        lvl, title, pnum = item
        audit_results["bookmarks"].append({"level": lvl, "title": title, "page": pnum})
        if "Chapter" in title or "Appendix" in title:
            audit_results["chapter_start_pages"][title] = pnum

    print("\n--- CHAPTER & SECTION START PAGES ---")
    for b in audit_results["bookmarks"]:
        indent = "  " * (b["level"] - 1)
        print(f"{indent}• {b['title']} -> Page {b['page']}")

    print("\n--- RENDERING AND AUDITING PAGES ---")
    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]

        # Render page to PNG (150 DPI)
        pix = page.get_pixmap(dpi=150)
        img_path = os.path.join(RENDER_DIR, f"page_{page_num:02d}.png")
        pix.save(img_path)

        # Text check
        text = page.get_text().strip()
        word_count = len(text.split())
        
        # Image check
        images = page.get_images()
        if len(images) > 0:
            audit_results["pages_with_images"] += 1
            audit_results["total_images"] += len(images)

        # Links check
        links = page.get_links()
        audit_results["total_links"] += len(links)
        for link in links:
            uri = link.get("uri", "")
            if "localhost:8000" in uri:
                # Extract DT topic
                topic = uri.split("topic=")[-1] if "topic=" in uri else uri
                audit_results["dt_hyperlinks"][topic] = audit_results["dt_hyperlinks"].get(topic, 0) + 1

        # Check for blank page
        if word_count == 0 and len(images) == 0:
            audit_results["blank_pages"].append(page_num)
            print(f"[!] Page {page_num}: BLANK PAGE DETECTED!")
        else:
            img_info = f", {len(images)} images" if images else ""
            link_info = f", {len(links)} links" if links else ""
            print(f"Page {page_num:02d}: {word_count} words{img_info}{link_info} -> Saved {os.path.basename(img_path)}")

    print("\n" + "=" * 60)
    print("PDF AUDIT SUMMARY REPORT")
    print("=" * 60)
    print(f"Total Pages Rendered: {total_pages}")
    print(f"Blank Pages Found: {len(audit_results['blank_pages'])} ({audit_results['blank_pages']})")
    print(f"Total Images Extracted: {audit_results['total_images']} across {audit_results['pages_with_images']} pages")
    print(f"Total Clickable Links: {audit_results['total_links']}")
    print(f"Verified DT Topics in Links: {len(audit_results['dt_hyperlinks'])} topics ({sorted(audit_results['dt_hyperlinks'].keys())})")
    print("=" * 60)

    doc.close()
    return audit_results

if __name__ == "__main__":
    audit_and_render_pdf()

