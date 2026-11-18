"""
Academic ReportLab Styles, Canvas Customization, and Flowables for DeepTensor++ Research Report
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Flowable, Paragraph, Spacer, Table, TableStyle, KeepTogether, Image, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from PIL import Image as PILImage

# Page geometry
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_LEFT = 54
MARGIN_RIGHT = 54
MARGIN_TOP = 54
MARGIN_BOTTOM = 54
USABLE_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
USABLE_HEIGHT = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

# Color Palette (Academic Navy, Slate, and Professional Accents)
COLOR_NAVY_DARK = colors.HexColor("#0B192C")      # Deep cover navy
COLOR_NAVY = colors.HexColor("#1E3A8A")           # Primary heading navy
COLOR_SLATE_DARK = colors.HexColor("#1E293B")     # Body text dark slate
COLOR_SLATE = colors.HexColor("#334155")          # Secondary body
COLOR_SLATE_LIGHT = colors.HexColor("#64748B")    # Captions & running headers
COLOR_BLUE_ACCENT = colors.HexColor("#2563EB")    # Hyperlinks & primary badges
COLOR_ICE_BLUE = colors.HexColor("#EFF6FF")       # Callout backgrounds
COLOR_AMBER = colors.HexColor("#D97706")          # Viva defense warning
COLOR_AMBER_BG = colors.HexColor("#FFFBEB")       # Viva defense background
COLOR_GREEN = colors.HexColor("#15803D")          # Winner badge & success
COLOR_GREEN_BG = colors.HexColor("#DCFCE7")       # Winner background
COLOR_BORDER = colors.HexColor("#CBD5E1")         # Table & card borders
COLOR_CARD_BG = colors.HexColor("#F8FAFC")        # Card background
COLOR_WHITE = colors.HexColor("#FFFFFF")

# ==============================================================================
# NUMBERED CANVAS WITH RUNNING HEADERS, FOOTERS & PDF BOOKMARKS
# ==============================================================================
class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute total page count and render
    professional running headers, footers, and bookmarks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []
        self.page_headings = {}  # page_num -> (chapter_title, section_title)

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        page_num = self._pageNumber
        if page_num == 1:
            # Cover page has its own full-bleed graphic styling
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(COLOR_SLATE_LIGHT)

        # Running Header (pages 3 and onward, excluding front matter cover)
        if page_num >= 3:
            # Header line
            self.setStrokeColor(COLOR_BORDER)
            self.setLineWidth(0.6)
            self.line(MARGIN_LEFT, PAGE_HEIGHT - 40, PAGE_WIDTH - MARGIN_RIGHT, PAGE_HEIGHT - 40)
            
            # Left Header: Project title
            self.drawString(MARGIN_LEFT, PAGE_HEIGHT - 35, "DeepTensor++: Phase 2 Academic Research Report")
            
            # Right Header: Section / Chapter marker
            heading_text = self.page_headings.get(page_num, "Low-Rank Tensor Decompositions with Deep Generative Priors")
            self.drawRightString(PAGE_WIDTH - MARGIN_RIGHT, PAGE_HEIGHT - 35, heading_text[:55])

        # Running Footer (pages 2 and onward)
        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.6)
        self.line(MARGIN_LEFT, 45, PAGE_WIDTH - MARGIN_RIGHT, 45)

        # Left Footer: Academic defense context
        self.drawString(MARGIN_LEFT, 32, "Faculty Evaluation — Autonomous Inverse Problems & Neural Generative Priors")

        # Right Footer: Page X of Y
        page_str = f"Page {page_num} of {total_pages}"
        self.drawRightString(PAGE_WIDTH - MARGIN_RIGHT, 32, page_str)

        self.restoreState()


# ==============================================================================
# BOOKMARK & OUTLINE FLOWABLE
# ==============================================================================
class BookmarkFlowable(Flowable):
    """
    Injects an internal PDF destination and an outline entry into the PDF.
    """
    def __init__(self, key, title, level=0):
        super().__init__()
        self.key = key
        self.title = title
        self.level = level

    def wrap(self, availWidth, availHeight):
        return (0, 0)

    def draw(self):
        self.canv.bookmarkPage(self.key)
        self.canv.addOutlineEntry(self.title, self.key, level=self.level, closed=False)
        # Register heading for running header
        current_page = self.canv._pageNumber
        if hasattr(self.canv, 'page_headings'):
            self.canv.page_headings[current_page] = self.title


# ==============================================================================
# TYPOGRAPHY & PARAGRAPH STYLES
# ==============================================================================
styles = getSampleStyleSheet()

# Modify Normal
normal = styles['Normal']
normal.fontName = 'Helvetica'
normal.fontSize = 9.5
normal.leading = 14
normal.textColor = COLOR_SLATE_DARK

# Headings
style_cover_title = ParagraphStyle(
    'CoverTitle',
    parent=normal,
    fontName='Helvetica-Bold',
    fontSize=26,
    leading=32,
    textColor=COLOR_NAVY_DARK,
    alignment=TA_LEFT,
    spaceAfter=12
)

style_cover_subtitle = ParagraphStyle(
    'CoverSubtitle',
    parent=normal,
    fontName='Helvetica',
    fontSize=13,
    leading=18,
    textColor=COLOR_BLUE_ACCENT,
    alignment=TA_LEFT,
    spaceAfter=20
)

style_cover_meta = ParagraphStyle(
    'CoverMeta',
    parent=normal,
    fontName='Helvetica',
    fontSize=9,
    leading=14,
    textColor=COLOR_SLATE,
    alignment=TA_LEFT
)

style_chapter_num = ParagraphStyle(
    'ChapterNum',
    parent=normal,
    fontName='Helvetica-Bold',
    fontSize=11,
    leading=14,
    textColor=COLOR_BLUE_ACCENT,
    textTransform='uppercase',
    spaceBefore=14,
    spaceAfter=4,
    keepWithNext=True
)

style_chapter_title = ParagraphStyle(
    'ChapterTitle',
    parent=normal,
    fontName='Helvetica-Bold',
    fontSize=18,
    leading=22,
    textColor=COLOR_NAVY,
    spaceAfter=12,
    keepWithNext=True
)

style_section_title = ParagraphStyle(
    'SectionTitle',
    parent=normal,
    fontName='Helvetica-Bold',
    fontSize=12.5,
    leading=16,
    textColor=COLOR_NAVY,
    spaceBefore=12,
    spaceAfter=6,
    keepWithNext=True
)

style_subsection_title = ParagraphStyle(
    'SubsectionTitle',
    parent=normal,
    fontName='Helvetica-Bold',
    fontSize=10.5,
    leading=14,
    textColor=COLOR_SLATE_DARK,
    spaceBefore=8,
    spaceAfter=4,
    keepWithNext=True
)

style_body = ParagraphStyle(
    'BodyJustified',
    parent=normal,
    fontName='Helvetica',
    fontSize=9.2,
    leading=13.5,
    textColor=COLOR_SLATE_DARK,
    alignment=TA_JUSTIFY,
    spaceAfter=6
)

style_body_lead = ParagraphStyle(
    'BodyLead',
    parent=normal,
    fontName='Helvetica',
    fontSize=9.8,
    leading=14.5,
    textColor=COLOR_SLATE_DARK,
    alignment=TA_JUSTIFY,
    spaceAfter=8
)

style_bullet = ParagraphStyle(
    'BulletItem',
    parent=normal,
    fontName='Helvetica',
    fontSize=9.0,
    leading=13.0,
    textColor=COLOR_SLATE_DARK,
    leftIndent=15,
    firstLineIndent=-10,
    spaceAfter=3
)

style_caption = ParagraphStyle(
    'Caption',
    parent=normal,
    fontName='Helvetica',
    fontSize=8.2,
    leading=11.5,
    textColor=COLOR_SLATE,
    alignment=TA_CENTER,
    spaceBefore=4,
    spaceAfter=10
)

style_equation = ParagraphStyle(
    'EquationText',
    parent=normal,
    fontName='Times-Italic',
    fontSize=9.8,
    leading=14,
    textColor=COLOR_NAVY_DARK,
    alignment=TA_CENTER
)

style_eq_num = ParagraphStyle(
    'EquationNum',
    parent=normal,
    fontName='Helvetica-Bold',
    fontSize=8.5,
    leading=14,
    textColor=COLOR_SLATE_LIGHT,
    alignment=TA_RIGHT
)

style_table_header = ParagraphStyle(
    'TableHeader',
    parent=normal,
    fontName='Helvetica-Bold',
    fontSize=8.2,
    leading=11,
    textColor=COLOR_WHITE,
    alignment=TA_CENTER
)

style_table_cell = ParagraphStyle(
    'TableCell',
    parent=normal,
    fontName='Helvetica',
    fontSize=8.0,
    leading=11,
    textColor=COLOR_SLATE_DARK,
    alignment=TA_LEFT
)

style_table_cell_center = ParagraphStyle(
    'TableCellCenter',
    parent=normal,
    fontName='Helvetica',
    fontSize=8.0,
    leading=11,
    textColor=COLOR_SLATE_DARK,
    alignment=TA_CENTER
)

style_table_cell_bold = ParagraphStyle(
    'TableCellBold',
    parent=normal,
    fontName='Helvetica-Bold',
    fontSize=8.0,
    leading=11,
    textColor=COLOR_NAVY,
    alignment=TA_CENTER
)

# Navigator & Callout Styles
style_code_header = ParagraphStyle(
    'CodeHeader',
    parent=normal,
    fontName='Helvetica-Bold',
    fontSize=8.5,
    leading=11,
    textColor=COLOR_BLUE_ACCENT,
    textTransform='uppercase'
)

style_code_link = ParagraphStyle(
    'CodeLink',
    parent=normal,
    fontName='Helvetica-Bold',
    fontSize=9.2,
    leading=13,
    textColor=COLOR_BLUE_ACCENT
)

style_code_meta = ParagraphStyle(
    'CodeMeta',
    parent=normal,
    fontName='Helvetica',
    fontSize=8.0,
    leading=11,
    textColor=COLOR_SLATE
)

style_viva_header = ParagraphStyle(
    'VivaHeader',
    parent=normal,
    fontName='Helvetica-Bold',
    fontSize=8.5,
    leading=11,
    textColor=COLOR_AMBER,
    textTransform='uppercase'
)

style_viva_q = ParagraphStyle(
    'VivaQ',
    parent=normal,
    fontName='Helvetica-Bold',
    fontSize=8.8,
    leading=12,
    textColor=COLOR_SLATE_DARK,
    spaceBefore=2
)

style_viva_a = ParagraphStyle(
    'VivaA',
    parent=normal,
    fontName='Helvetica',
    fontSize=8.4,
    leading=12,
    textColor=COLOR_SLATE,
    alignment=TA_JUSTIFY,
    spaceBefore=2
)

# ==============================================================================
# HELPER FLOWABLE FACTORIES
# ==============================================================================
def make_code_navigator_box(topic_id, title, file_path, lines, symbol):
    """
    Creates a styled academic callout card with a direct clickable URL
    linking to http://localhost:8000/?topic=DT-XX.
    """
    url = f"http://localhost:8000/?topic={topic_id}"
    content = [
        Paragraph(f"<b>INTERACTIVE CODE NAVIGATOR INTEGRATION:</b>", style_code_header),
        Spacer(1, 2),
        Paragraph(
            f'<a href="{url}" color="#1D4ED8"><b>&rarr; [CLICK TO INSPECT {topic_id}: {title.upper()} IN CODE NAVIGATOR]</b></a>',
            style_code_link
        ),
        Spacer(1, 2),
        Paragraph(
            f"<b>Source File:</b> <code>{file_path}</code> &nbsp;|&nbsp; "
            f"<b>Lines:</b> {lines} &nbsp;|&nbsp; <b>Symbol:</b> <code>{symbol}</code>",
            style_code_meta
        )
    ]
    t = Table([[content]], colWidths=[USABLE_WIDTH])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_ICE_BLUE),
        ('BOX', (0,0), (-1,-1), 1.0, COLOR_BLUE_ACCENT),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    return t


def make_viva_defense_box(question, defense):
    """
    Creates a faculty viva defense callout card with academic QA format.
    """
    content = [
        Paragraph(f"<b>FACULTY VIVA Q&amp;A DEFENSE:</b>", style_viva_header),
        Paragraph(f"<b>Q: {question}</b>", style_viva_q),
        Paragraph(f"<b>Defense:</b> {defense}", style_viva_a),
    ]
    t = Table([[content]], colWidths=[USABLE_WIDTH])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_AMBER_BG),
        ('BOX', (0,0), (-1,-1), 1.0, COLOR_AMBER),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    return t


def make_equation_box(eq_text, eq_num):
    """
    Creates an academic equation box with a light background and right-aligned equation number.
    """
    t = Table([
        [Paragraph(f"<i>{eq_text}</i>", style_equation),
         Paragraph(f"<b>({eq_num})</b>", style_eq_num)]
    ], colWidths=[USABLE_WIDTH - 50, 50])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_CARD_BG),
        ('BOX', (0,0), (-1,-1), 0.6, COLOR_BORDER),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    return t


def make_figure_box(image_path, caption_text, target_width=USABLE_WIDTH, target_height=None, max_height=240):
    """
    Loads an image from image_path, scales it preserving aspect ratio,
    and returns a KeepTogether flowable pairing the image with its caption.
    """
    if not os.path.exists(image_path):
        return Paragraph(f"<b>[Image not found: {image_path}]</b>", style_caption)

    try:
        im = PILImage.open(image_path)
        w, h = im.size
        aspect = h / float(w)

        actual_w = target_width
        actual_h = actual_w * aspect

        if max_height and actual_h > max_height:
            actual_h = max_height
            actual_w = actual_h / aspect

        img_flowable = Image(image_path, width=actual_w, height=actual_h)
        caption_flowable = Paragraph(caption_text, style_caption)

        return KeepTogether([
            Spacer(1, 4),
            img_flowable,
            Spacer(1, 2),
            caption_flowable,
            Spacer(1, 4)
        ])
    except Exception as e:
        return Paragraph(f"<b>[Error rendering image {image_path}: {e}]</b>", style_caption)


def make_academic_table(headers, data, col_widths=None, winner_row=None):
    """
    Formats an academic table with dark blue header and alternating row fills.
    """
    table_rows = []
    
    # Headers
    h_row = [Paragraph(h, style_table_header) for h in headers]
    table_rows.append(h_row)

    # Data
    for r_idx, row in enumerate(data):
        row_cells = []
        is_winner = (winner_row is not None and r_idx == winner_row)
        for c_idx, cell in enumerate(row):
            txt = str(cell)
            if c_idx == 0:
                p = Paragraph(f"<b>{txt}</b>", style_table_cell)
            elif is_winner and ("dB" in txt or "🏆" in txt or c_idx == len(row)-1):
                p = Paragraph(f"<b>{txt}</b>", style_table_cell_bold)
            else:
                p = Paragraph(txt, style_table_cell_center)
            row_cells.append(p)
        table_rows.append(row_cells)

    if col_widths is None:
        col_w = USABLE_WIDTH / len(headers)
        col_widths = [col_w] * len(headers)

    t = Table(table_rows, colWidths=col_widths)
    
    t_style = [
        ('BACKGROUND', (0,0), (-1,0), COLOR_NAVY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
    ]

    # Row zebra striping
    for i in range(1, len(table_rows)):
        if winner_row is not None and (i - 1) == winner_row:
            t_style.append(('BACKGROUND', (0, i), (-1, i), COLOR_GREEN_BG))
        elif i % 2 == 0:
            t_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#F1F5F9")))
        else:
            t_style.append(('BACKGROUND', (0, i), (-1, i), COLOR_WHITE))

    t.setStyle(TableStyle(t_style))
    return t

