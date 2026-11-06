#!/usr/bin/env python3.14
"""Assemble the manuscript figures as true-vector composites (PyMuPDF show_pdf_page) from the panels
written by ms_fig2/3/4/supp, with the panel layout of the current manuscript figures.
Output: native_pipeline/figures/composites/Figure 2.pdf, Figure 3.pdf, Figure 4.pdf,
        WhatRes_all_figures.pdf (Figure 1 is the author's own file, taken unchanged from the manuscript
        folder; supplementary figures one per page) and PNG renders in composites/_renders/.
"""
import os, fitz
from ms_common import OUT, MS
C = os.path.join(OUT, "composites"); R = os.path.join(C, "_renders"); os.makedirs(R, exist_ok=True)
W, MARGIN, GAP, VGAP, LAB = 612.0, 28.0, 12.0, 22.0, 15

def src(fig, name):
    for ext in (".pdf", ".png"):
        p = os.path.join(OUT, fig, name + ext)
        if os.path.exists(p): return p
    raise FileNotFoundError(f"{fig}/{name}")

def aspect(p):
    if p.endswith(".pdf"):
        r = fitz.open(p)[0].rect; return r.width / r.height
    pm = fitz.Pixmap(p); return pm.width / pm.height

def compose(fig, rows, out):
    """rows: list of rows; a row is a list of (label, path, weight) - weight = share of the row width.
    A row whose weights sum to < 1 leaves the remainder empty (left-aligned)."""
    usable = W - 2 * MARGIN; plan = []; y = MARGIN
    for row in rows:
        tot = max(1.0, sum(w for _, _, w in row)); gaps = GAP * (len(row) - 1)
        boxes = [((usable - gaps) * w / tot) for _, _, w in row]
        h = min(bw / aspect(p) for (_, p, _), bw in zip(row, boxes))
        x = MARGIN; cells = []
        for (lab, p, _), bw in zip(row, boxes):
            pw = h * aspect(p); cells.append((lab, p, fitz.Rect(x + (bw - pw) / 2, y + LAB, x + (bw - pw) / 2 + pw, y + LAB + h), x)); x += bw + GAP
        plan.append(cells); y += LAB + h + VGAP
    doc = fitz.open(); page = doc.new_page(width=W, height=y - VGAP + MARGIN)
    for cells in plan:
        for lab, p, rect, x0 in cells:
            if p.endswith(".pdf"):
                s = fitz.open(p); page.show_pdf_page(rect, s, 0)
            else:
                page.insert_image(rect, filename=p)
            if lab: page.insert_text((x0, rect.y0 - 3), lab, fontsize=12, fontname="hebo")
    doc.save(out); doc.close()
    fitz.open(out)[0].get_pixmap(dpi=150).save(os.path.join(R, os.path.basename(out).replace(".pdf", ".png")))
    print("[compose]", out)

F2, F3, F4, SU = "Figure2", "Figure3", "Figure4", "Supplementary"
compose(F2, [[("a", src(F2, "Fig2a_LDA_projection"), 1), ("b", src(F2, "Fig2b_confusion_matrix"), 1)],
             [("", src(F2, "Fig2cf_overregularisation"), 1)]], os.path.join(C, "Figure 2.pdf"))
compose(F3, [[("b", src(F3, "Fig3b_descriptors_vs_resolution"), 1)],
             [("a", src(F3, "Fig3a_scaling_descriptors_ranked"), 1), ("c", src(F3, "Fig3c_training_correlation"), 1)],
             [("d", src(F3, "Fig3d_independent_set"), 1), ("e", src(F3, "Fig3e_by_method"), 1)],
             [("f", src(F3, "Fig3f_pLDDT_vs_apparent_resolution"), 0.5)]], os.path.join(C, "Figure 3.pdf"))
compose(F4, [[("a", src(F4, "Fig4a_resolution_invariant"), 1)],
             [("b", src(F4, "Fig4b_residualisation_fit"), 1)],
             [("c", src(F4, "Fig4c_top_contributors"), 1), ("d", src(F4, "Fig4d_realism_by_method"), 1)],
             [("e", src(F4, "Fig4e_orthogonality"), 1), ("f", src(F4, "Fig4f_pLDDT_vs_realism"), 2)],
             [("g", src(F4, "Fig4g_7dg7_overlay"), 1), ("h", src(F4, "Fig4h_7dg7_rotamers"), 1)]], os.path.join(C, "Figure 4.pdf"))

book = fitz.open()
book.insert_pdf(fitz.open(os.path.join(MS, "Figure 1.pdf")))
for f in ["Figure 2.pdf", "Figure 3.pdf", "Figure 4.pdf"]: book.insert_pdf(fitz.open(os.path.join(C, f)))
for s in ["SuppFig1_realism_vs_resolution", "SuppFig2_ai_vs_experimental", "SuppFig3_shoulder_helix_sheet", "SuppFig4a_threshold_sweep",
          "SuppFig4b_random_control", "SuppFig5_era_before_after_1995", "SuppFig6_size_reliability"]:
    p = src(SU, s)
    if p.endswith(".pdf"): book.insert_pdf(fitz.open(p))
book.save(os.path.join(C, "WhatRes_all_figures.pdf")); print("[compose] WhatRes_all_figures.pdf:", book.page_count, "pages")
