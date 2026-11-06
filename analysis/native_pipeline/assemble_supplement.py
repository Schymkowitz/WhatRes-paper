#!/usr/bin/env python3
"""Assemble the WhatRes Supplementary Figures PDF (figures + legends) for bioRxiv / journal upload.

usage: assemble_supplement.py <manuscript.docx> <out.pdf>
  legend_overrides/S<N>.txt, if present, replaces the docx legend of Supplementary Figure N
  (plain text; used while the docx is open for editing). Each override is reported when used.
Figures: native_pipeline/figures/Supplementary/SuppFig*.pdf (as made by ms_supp.py).
Legends: taken verbatim from the 'Supplementary Figure N.' paragraphs of the docx
(bold / italic / superscript kept), so the PDF always carries the manuscript's current text.
Every figure needs exactly one legend and every legend one figure; otherwise it stops."""
import glob, html, os, re, sys
import docx, pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures", "Supplementary")
A4, MG = (595.3, 841.9), 50            # A4 portrait (pt), margin

class _Run:   # plain-text stand-in for a docx run (overrides)
    def __init__(s, t):
        s.text, s.italic = t, None
        s.font = type("F", (), {"superscript": None, "subscript": None})()

def legend_html(p):
    """Runs -> HTML. Bold is normalised to 'Supplementary Figure N. <title sentence>.' for every
    legend (the docx bolds more than the title in some legends); sup/sub/italic are kept."""
    text = "".join(r.text for r in p.runs)
    m = re.match(r"\s*Supplementary Figure \d+\.\s+.*?\.(?=\s)", text)
    cut, pos, out = (m.end() if m else 0), 0, []
    for r in p.runs:
        for seg_bold, seg in ((True, r.text[:max(0, cut - pos)]), (False, r.text[max(0, cut - pos):])):
            if not seg: continue
            t = html.escape(seg)
            if r.font.superscript: t = f"<sup>{t}</sup>"
            if r.font.subscript: t = f"<sub>{t}</sub>"
            if r.italic: t = f"<i>{t}</i>"
            out.append(f"<b>{t}</b>" if seg_bold else t)
        pos += len(r.text)
    return "".join(out).strip()

def main(ms, out):
    d = docx.Document(ms)
    legends = {}
    for p in d.paragraphs:
        m = re.match(r"Supplementary Figure (\d+)\.", p.text.strip())
        if m:
            n = int(m.group(1))
            if n in legends: sys.exit(f"STOP: two legends for Supplementary Figure {n}")
            legends[n] = legend_html(p)
    for n in list(legends):
        o = os.path.join(HERE, "legend_overrides", f"S{n}.txt")
        if os.path.exists(o):
            legends[n] = legend_html(type("P", (), {"runs": [_Run(open(o).read().strip())]})())
            print(f"  NOTE: Supplementary Figure {n} legend from {os.path.relpath(o, HERE)}, not the docx")
    title = next((p.text.strip() for p in d.paragraphs if p.text.strip()), "")

    figs = {}
    for f in sorted(glob.glob(os.path.join(FIG, "SuppFig*.pdf"))):
        m = re.match(r"SuppFig(\d+)([a-z]?)_", os.path.basename(f))
        figs.setdefault(int(m.group(1)), []).append((m.group(2), f))
    if set(figs) != set(legends):
        sys.exit(f"STOP: figures {sorted(figs)} vs legends {sorted(legends)}")

    css = "* {font-family: sans-serif; font-size: 9pt; line-height: 1.3;} sup {font-size: 6pt;}"
    doc = pymupdf.open()
    for n in sorted(figs):
        parts = figs[n]
        srcs = [pymupdf.open(f)[0].rect for _, f in parts]
        gap = 14 if len(parts) > 1 else 0
        nat_w = sum(r.width for r in srcs) + gap * (len(parts) - 1)        # natural width at 100%
        W, H = (A4 if nat_w <= A4[0] - 2 * MG else A4[::-1])              # landscape for wide figures
        CW = W - 2 * MG
        k = min(1.0, CW / nat_w)                                            # never enlarge
        while True:
            pg = doc.new_page(width=W, height=H); y = MG
            if n == min(figs):
                hdr = (f"<p style='font-size:11pt'><b>Supplementary Information</b></p>"
                       f"<p style='font-size:10pt'>{html.escape(title)}</p>")
                spare, _ = pg.insert_htmlbox(pymupdf.Rect(MG, y, W - MG, y + 120), hdr, css=css)
                y += 120 - spare + 18
            x = MG + (CW - k * nat_w) / 2
            for (letter, f), r in zip(parts, srcs):
                w, h = k * r.width, k * r.height
                pg.show_pdf_page(pymupdf.Rect(x, y, x + w, y + h), pymupdf.open(f), 0)
                if letter: pg.insert_text((x, y + 10 * k), letter, fontname="hebo", fontsize=10 * k)
                x += w + gap * k
            y += k * max(r.height for r in srcs) + 14
            spare, _ = pg.insert_htmlbox(pymupdf.Rect(MG, y, W - MG, H - MG), f"<p>{legends[n]}</p>", css=css)
            if spare >= 0: break
            doc.delete_page(-1); k *= 0.92
        pg.insert_text((W / 2 - 3, H - 25), str(doc.page_count), fontname="helv", fontsize=8)
        print(f"  S{n}: {'portrait' if W < H else 'landscape'}, figure at {100 * k:.0f}%")
    doc.set_metadata({"title": f"Supplementary Information: {title}", "author": "van der Kant et al."})
    doc.save(out, garbage=4, deflate=True)
    print(f"[supp] {len(figs)} figures, {doc.page_count} pages -> {out}")

if __name__ == "__main__":
    main(*sys.argv[1:3])
